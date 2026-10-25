"""S4.4 approved queue: R6 (seeds 1-3), R5-gamma0 (seed 1), R4 (seeds 1-3), R3 (seed 1),
one run at a time, in that order. Starts the next run only after the previous run AND its
scoring are complete. Validation split only -- training uses training.skip_test_eval: true
(config/s4_4/{r6,r5_gamma0,r4,r3}.yaml) so the test split is never loaded by runner.run(),
and scoring (score_s4_4_gate4_run.score_run) only ever builds a validation DataLoader.

Launched only after R8 (scripts/run_r8.py) finishes training AND scoring -- the author's
2026-10-24 instruction for this queue is conditional on that (PC-STGNN/S4.4_remaining_arms_plan.md
section 11, D8 lineage). Does not start R1 or R2 (loss code written but not trained this round)
or PCRG (not written). If this script is invoked before R8 is done, it refuses to run.

Generalizes scripts/run_s4_4_gate4.py's safety pattern (resumable, heartbeat, config-match
resume guard, stop-the-queue-on-failure, GPU release) from a fixed 6-run (r0n, r5) queue to an
arbitrary ordered list of (arm, seed_label, seed) triples.

Resumable: a pair's training is skipped only if checkpoint_best.pt AND metrics.json already
exist for it, AND the run's own saved config.yaml matches the frozen config/s4_4/<arm>.yaml
exactly (with that pair's seed substituted in). If the saved config differs, the queue stops
immediately and reports the diff -- it never silently retrains or silently treats a mismatched
run as valid. A pair's scoring is skipped only if scoring_full_grid.json already exists for it.

Stop-the-queue behavior (do not restart automatically):
  - If a run's metrics.json reports collapse_abort.triggered == true, the queue stops after
    writing a QUEUE_STOPPED.json marker with reason "collapsed". R5-gamma0 has no
    training.collapse_abort key (exempt, D2), so this never fires for it in practice.
  - If training or scoring raises an exception, the queue stops after writing a
    QUEUE_STOPPED.json marker with reason "exception" and the traceback.
  - If a resumed run's saved config does not match the frozen settings, the queue stops with
    reason "config_mismatch" and the diff.
  - On any stop, this process exits with a non-zero status.

GPU memory is released (gc.collect + torch.cuda.empty_cache) after every training call and
every scoring call, in addition to score_run's own internal release.

Progress is appended to results/s4_4/remaining_queue_progress.json after every step.

Heartbeat: a background thread watches each run's own metrics_by_epoch.csv for new rows while
training.run() is in progress, and appends {"epoch": N, "utc": ...} to that run's
heartbeat.log every time a new epoch row appears.

Usage: uv run python scripts/run_s4_4_remaining_queue.py
"""
from __future__ import annotations

import csv
import gc
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from fault_localization.experiments.config import load_config, validate_config
from fault_localization.experiments.runner import run as runner_run
from score_s4_4_gate4_run import score_run

QUEUE = [
    ("r6", 1, 42), ("r6", 2, 43), ("r6", 3, 44),
    ("r5_gamma0", 1, 42),
    ("r4", 1, 42), ("r4", 2, 43), ("r4", 3, 44),
    ("r3", 1, 42),
]
R8_PROGRESS_PATH = Path("results/s4_4/r8_progress.json")
PROGRESS_PATH = Path("results/s4_4/remaining_queue_progress.json")
STOP_PATH = Path("results/s4_4/remaining_queue_STOPPED.json")


def _r8_complete() -> bool:
    if not R8_PROGRESS_PATH.exists():
        return False
    history = json.loads(R8_PROGRESS_PATH.read_text())
    return any(entry.get("stage") == "complete" for entry in history)


def _append_progress(entry: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(PROGRESS_PATH.read_text()) if PROGRESS_PATH.exists() else []
    history.append(entry)
    PROGRESS_PATH.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")


def _stop_queue(reason: str, detail: str, position: int) -> None:
    STOP_PATH.parent.mkdir(parents=True, exist_ok=True)
    STOP_PATH.write_text(json.dumps({
        "reason": reason, "detail": detail, "stopped_at_queue_position": position,
        "stopped_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")


def _release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _heartbeat_loop(run_dir: Path, stop_event: threading.Event, poll_seconds: float = 5.0) -> None:
    csv_path = run_dir / "metrics_by_epoch.csv"
    heartbeat_path = run_dir / "heartbeat.log"
    n_seen = 0
    while not stop_event.is_set():
        if csv_path.exists():
            try:
                with csv_path.open() as f:
                    rows = list(csv.DictReader(f))
            except (OSError, csv.Error):
                rows = []
            if len(rows) > n_seen:
                run_dir.mkdir(parents=True, exist_ok=True)
                with heartbeat_path.open("a") as hb:
                    for row in rows[n_seen:]:
                        hb.write(json.dumps({
                            "epoch": int(row["epoch"]),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }) + "\n")
                n_seen = len(rows)
        stop_event.wait(poll_seconds)


def _run_with_heartbeat(run_dir: Path, fn, *args, **kwargs):
    stop_event = threading.Event()
    thread = threading.Thread(target=_heartbeat_loop, args=(run_dir, stop_event), daemon=True)
    thread.start()
    try:
        return fn(*args, **kwargs)
    finally:
        stop_event.set()
        thread.join(timeout=10)


def _frozen_config(arm: str, seed: int) -> dict:
    config = validate_config(load_config(f"config/s4_4/{arm}.yaml"))
    config["seed"] = seed
    return config


def _dict_diff(expected: dict, actual: dict, prefix: str = "") -> list[str]:
    diffs = []
    keys = set(expected) | set(actual)
    for k in sorted(keys):
        ek, ak = expected.get(k, "<missing>"), actual.get(k, "<missing>")
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(ek, dict) and isinstance(ak, dict):
            diffs.extend(_dict_diff(ek, ak, path))
        elif ek != ak:
            diffs.append(f"{path}: frozen={ek!r} saved={ak!r}")
    return diffs


def _run_dir(arm: str, seed: int) -> Path:
    return Path("results/s4_4") / arm / "pcstgnn" / f"seed_{seed:05d}"


def main() -> int:
    # SUPERSEDED 2026-10-25: this queue stopped on the R6 epoch-3 collapse guard (2026-10-24T16:58Z). Its
    # helper functions are still imported by scripts/run_s4_4_queue.py, but it must not run again, so that
    # only one queue can start.
    print("QUEUE REFUSED: run_s4_4_remaining_queue.py is superseded by scripts/run_s4_4_queue.py.", file=sys.stderr)
    return 1
    if not _r8_complete():
        print("QUEUE REFUSED: R8 (scripts/run_r8.py) has not finished training and scoring yet "
              "-- this queue is approved to start only after R8 completes "
              "(PC-STGNN/S4.4_remaining_arms_plan.md section 11, D8).", file=sys.stderr)
        return 1

    for position, (arm, seed_label, seed) in enumerate(QUEUE, start=1):
        run_dir = _run_dir(arm, seed)
        ckpt_path = run_dir / "checkpoint_best.pt"
        metrics_path = run_dir / "metrics.json"
        already_trained = ckpt_path.exists() and metrics_path.exists()

        if already_trained:
            saved_config = yaml.safe_load((run_dir / "config.yaml").read_text())
            expected_config = _frozen_config(arm, seed)
            diffs = _dict_diff(expected_config, saved_config)
            if diffs:
                _stop_queue("config_mismatch", "\n".join(diffs), position)
                _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                                  "seed": seed, "stage": "config_mismatch", "diffs": diffs})
                print(f"QUEUE STOPPED at position {position} ({arm} seed{seed_label}): "
                      f"saved config does not match the frozen settings.\n" + "\n".join(diffs),
                      file=sys.stderr)
                return 1
            _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                              "seed": seed, "stage": "training_skipped_already_done"})
            metrics = json.loads(metrics_path.read_text())
            train_s = None
        else:
            t0 = time.perf_counter()
            _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                              "seed": seed, "stage": "training_started",
                              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            try:
                config = _frozen_config(arm, seed)
                run_dir = _run_with_heartbeat(run_dir, runner_run, config)
            except Exception:
                tb = traceback.format_exc()
                _stop_queue("exception", tb, position)
                _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                                  "seed": seed, "stage": "training_failed", "error": tb})
                print(f"QUEUE STOPPED at position {position} ({arm} seed{seed_label}): "
                      f"training raised an exception.\n{tb}", file=sys.stderr)
                return 1
            finally:
                _release_gpu()

            metrics = json.loads((run_dir / "metrics.json").read_text())
            train_s = time.perf_counter() - t0

        collapsed = bool(metrics.get("collapse_abort", {}).get("triggered", False))
        _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                          "seed": seed, "stage": "training_done",
                          "train_seconds": train_s, "collapsed": collapsed,
                          "best_validation_graph_f1": metrics.get("best_validation_graph_f1")})

        if collapsed:
            _stop_queue("collapsed", json.dumps(metrics.get("collapse_abort")), position)
            print(f"QUEUE STOPPED at position {position} ({arm} seed{seed_label}): "
                  f"run collapsed at epoch 3. {metrics.get('collapse_abort')}", file=sys.stderr)
            return 1

        scoring_path = run_dir / "scoring_full_grid.json"
        if scoring_path.exists():
            _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                              "seed": seed, "stage": "scoring_skipped_already_done"})
        else:
            _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                              "seed": seed, "stage": "scoring_started",
                              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            try:
                t1 = time.perf_counter()
                score_run(arm, seed)
                score_s = time.perf_counter() - t1
            except Exception:
                tb = traceback.format_exc()
                _stop_queue("exception", tb, position)
                _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                                  "seed": seed, "stage": "scoring_failed", "error": tb})
                print(f"QUEUE STOPPED at position {position} ({arm} seed{seed_label}): "
                      f"scoring raised an exception.\n{tb}", file=sys.stderr)
                return 1
            finally:
                _release_gpu()
            _append_progress({"position": position, "arm": arm, "seed_label": seed_label,
                              "seed": seed, "stage": "scoring_done", "scoring_seconds": score_s})

        print(f"[{position}/{len(QUEUE)}] {arm} seed{seed_label} (seed={seed}) done.", flush=True)

    _append_progress({"stage": "queue_complete",
                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("Remaining-arms queue complete: R6 x3, R5-gamma0, R4 x3, R3 all trained and scored.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
