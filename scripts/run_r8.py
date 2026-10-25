"""R8 (R5 trained noise-free -- the ablation of the new default, S4.4 table):
single run, 1 seed, 30 epochs, validation split only.

Launched 2026-10-23 after: S5.1's hard gate passed (AUROC 0.997-0.999 at bias +-2%, threshold
0.75) and the Gate 4 sign-consistency rule applied to H1 (10 ohm, k=0) passed (R5 > R0^n in all
3 seeds, mean +0.099, raw p=0.014) -- see PC-STGNN/S4.4_remaining_arms_plan.md Part 4 / D8 and
the author's standing authorization to launch R8 without a further prompt once both checks clear.

R8 is exempt from the collapse guard by construction (config/s4_4/r8.yaml has no
training.collapse_abort key): it is expected to degenerate under S2.0^a consequence 4 (noise-free
training lets beta trust the physics completely), and that degeneracy is the result it exists to
produce, not a failure to catch early.

Mirrors scripts/run_s4_4_gate4.py's safety pattern (progress log, heartbeat, config match check,
exception handling, GPU release) for a single run instead of a 6-run queue. Writes
results/s4_4/r8_progress.json and results/s4_4/r8/pcstgnn/seed_00042/heartbeat.log.

Usage: uv run python scripts/run_r8.py
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

ARM = "r8"
SEED = 42
PROGRESS_PATH = Path("results/s4_4/r8_progress.json")
STOP_PATH = Path("results/s4_4/r8_STOPPED.json")


def _append_progress(entry: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(PROGRESS_PATH.read_text()) if PROGRESS_PATH.exists() else []
    history.append(entry)
    PROGRESS_PATH.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")


def _stop(reason: str, detail: str) -> None:
    STOP_PATH.write_text(json.dumps({
        "reason": reason, "detail": detail,
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
                        hb.write(json.dumps({"epoch": int(row["epoch"]),
                                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
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


def _run_dir() -> Path:
    return Path("results/s4_4") / ARM / "pcstgnn" / f"seed_{SEED:05d}"


def main() -> int:
    run_dir = _run_dir()
    ckpt_path = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"

    if ckpt_path.exists() and metrics_path.exists():
        _append_progress({"stage": "training_skipped_already_done"})
        metrics = json.loads(metrics_path.read_text())
        train_s = None
    else:
        t0 = time.perf_counter()
        _append_progress({"stage": "training_started",
                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            config = validate_config(load_config(f"config/s4_4/{ARM}.yaml"))
            config["seed"] = SEED
            run_dir = _run_with_heartbeat(run_dir, runner_run, config)
        except Exception:
            tb = traceback.format_exc()
            _stop("exception", tb)
            print(f"R8 STOPPED: training raised an exception.\n{tb}", file=sys.stderr)
            return 1
        finally:
            _release_gpu()
        metrics = json.loads((run_dir / "metrics.json").read_text())
        train_s = time.perf_counter() - t0

    collapsed = bool(metrics.get("collapse_abort", {}).get("triggered", False))
    _append_progress({"stage": "training_done", "train_seconds": train_s, "collapsed": collapsed,
                      "best_validation_graph_f1": metrics.get("best_validation_graph_f1"),
                      "beta_final": metrics.get("beta_final")})
    # No stop-on-collapse here: R8 is an oracle condition, exempt by design (no
    # training.collapse_abort key in r8.yaml) -- there is no collapse_abort key to check in
    # practice, this line is defensive only in case that ever changes.
    if collapsed:
        print("R8: collapse_abort reported true despite no collapse_abort key in r8.yaml -- "
              "unexpected, check the saved config.yaml.", file=sys.stderr)

    scoring_path = run_dir / "scoring_full_grid.json"
    if scoring_path.exists():
        _append_progress({"stage": "scoring_skipped_already_done"})
    else:
        _append_progress({"stage": "scoring_started",
                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            t1 = time.perf_counter()
            score_run(ARM, SEED)
            score_s = time.perf_counter() - t1
        except Exception:
            tb = traceback.format_exc()
            _stop("exception", tb)
            print(f"R8 STOPPED: scoring raised an exception.\n{tb}", file=sys.stderr)
            return 1
        finally:
            _release_gpu()
        _append_progress({"stage": "scoring_done", "scoring_seconds": score_s})

    _append_progress({"stage": "complete", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("R8 complete: trained and scored.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
