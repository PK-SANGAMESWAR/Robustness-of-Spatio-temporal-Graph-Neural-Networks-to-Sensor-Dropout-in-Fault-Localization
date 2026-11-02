"""S4.4 merged queue -- the only queue. Remaining work after the R6 collapses (restarted 2026-11-02), in this order:

  1. R6^i scoring, seeds 42, 43, 44   inference only: the three trained R5 checkpoints scored with the shuffled table
                                      (shuffle_seed 6) on the full S4.4 grid; pre-registered in ``06_STAGE_E_pcstgnn.md``
  2. R1                               seed 42
  3. R2                               seeds 42, 43, 44

Done earlier by this queue (see ``results/s4_4/queue_progress.json``): G-TF core, R5-gamma0, R4 x3, R3. R6 training was stopped after two
collapses (seeds 43 and 44 are not run). The G-TF2 watcher (``scripts/run_gtf2_after_queue.py``) starts G-TF2 when this queue writes
``queue_complete``. A lock file lets only one copy run at a time.

Safety (same pattern as the queues it replaces):
  - **Resume guard.** A run's training is skipped only if ``checkpoint_best.pt`` and ``metrics.json`` exist and
    the run's saved ``config.yaml`` equals the frozen ``config/s4_4/<arm>.yaml`` (seed substituted); a mismatch
    stops the queue. Scoring is skipped only if ``scoring_full_grid.json`` exists.
  - **Heartbeat.** A thread appends ``{"epoch", "utc"}`` to the run's ``heartbeat.log`` per finished epoch.
  - **Stop, never restart.** A collapse (``collapse_abort.triggered``), an exception, a config mismatch, or a failed
    G-TF step writes ``results/s4_4/queue_STOPPED.json`` and exits non-zero. Nothing restarts automatically.
  - **G-TF** starts only if ``run_gtf.check_ready('core')`` is true (pre-registration committed, detector decision
    recorded, inputs present). If not ready the step is skipped and logged and the queue goes on. The full scope
    is never run here.
  - Validation split only. The test split and the sealed OOD set are never touched.

``--dry-run`` hides CUDA before torch is imported, validates every config, builds each model on the CPU, reports the
resume state and G-TF readiness, and creates no CUDA context.

Progress: ``results/s4_4/queue_progress.json``. Usage: ``uv run python scripts/run_s4_4_queue.py [--dry-run]``
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

if "--dry-run" in sys.argv:          # must happen before torch is imported anywhere
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import yaml

QUEUE = [
    ("r6i", 1, 42), ("r6i", 2, 43), ("r6i", 3, 44),
    ("r1", 1, 42),
    ("r2", 1, 42), ("r2", 2, 43), ("r2", 3, 44),
]
R6I_SHUFFLE_SEED = 6                  # same permutation as the R6 training arm
R6I_CHECKPOINT_ARM = "r5"
PROGRESS_PATH = REPO / "results/s4_4/queue_progress.json"
STOP_PATH = REPO / "results/s4_4/queue_STOPPED.json"
LOCK_PATH = REPO / "results/s4_4/queue.lock"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append(entry: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(PROGRESS_PATH.read_text()) if PROGRESS_PATH.exists() else []
    hist.append(entry)
    PROGRESS_PATH.write_text(json.dumps(hist, indent=2, sort_keys=True) + "\n")


def _stop(reason: str, detail: str, position: int) -> None:
    STOP_PATH.write_text(json.dumps({"reason": reason, "detail": detail, "stopped_at_queue_position": position,
                                     "stopped_at_utc": utc()}, indent=2) + "\n")


def acquire_lock() -> bool:
    """Only one queue at a time: refuse if the lock names a live process."""
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)
            return False                       # that process is alive
        except (ValueError, ProcessLookupError):
            pass                               # stale lock
        except PermissionError:
            return False
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()))
    atexit.register(lambda: LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()) and LOCK_PATH.unlink())
    return True


def run_gtf_core(progress) -> dict:
    import run_gtf as G
    ready, problems = G.check_ready("core")
    if not ready:
        return {"stage": "gtf_skipped_not_ready", "problems": problems}
    out_root = G.OUT_ROOT / "core"
    data_root = REPO / "dataset_fullscale_50"
    t0 = time.perf_counter()
    G.stage_split(out_root, data_root)
    progress({"stage": "gtf_split_done", "utc": utc()})
    for sg in G.SIGMA_CORE:
        if not (out_root / f"thresholds_sigma_{sg:g}.npz").exists():
            G.stage_thresholds(sg, out_root, data_root, threads=8, max_batches=None)
        progress({"stage": "gtf_thresholds_done", "sigma0": sg, "utc": utc()})
    for seed in G.SEEDS:
        G.stage_score(seed, "core", out_root, data_root, threads=4)
        progress({"stage": "gtf_seed_done", "seed": seed, "utc": utc()})
    summary = G.stage_summarize("core", out_root)
    return {"stage": "gtf_core_done", "seconds": time.perf_counter() - t0,
            "decision_rule": summary.get("decision_rule_bias2pct_sigma1e-2")}


def run_r6i(seed: int, position: int, score_run) -> None:
    """R6^i: score the trained R5 checkpoint with the shuffled table. Scoring only; nothing is trained."""
    out = REPO / f"results/s4_4/r6i/pcstgnn/seed_{seed:05d}/scoring_full_grid.json"
    if out.exists():
        _append({"position": position, "arm": "r6i", "seed": seed, "stage": "scoring_skipped_already_done"})
        return
    _append({"position": position, "arm": "r6i", "seed": seed, "stage": "scoring_started", "utc": utc()})
    t1 = time.perf_counter()
    summary = score_run(R6I_CHECKPOINT_ARM, seed, shuffle_seed=R6I_SHUFFLE_SEED, out_arm="r6i")
    _append({"position": position, "arm": "r6i", "seed": seed, "stage": "scoring_done",
             "scoring_seconds": time.perf_counter() - t1, "table_check": summary.get("table_check"), "utc": utc()})


def dry_run() -> int:
    import numpy as np
    import torch
    import run_s4_4_remaining_queue as rq          # helper functions only (its main() refuses to run)
    from fault_localization.experiments.runner import _model
    import run_gtf as G

    print(f"[dry-run] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}; torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[dry-run] queue lock present: {LOCK_PATH.exists()}; queue stop marker present: {STOP_PATH.exists()}")
    edge_index = np.load(REPO / "dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
    ok = True
    for position, (arm, seed_label, seed) in enumerate(QUEUE, start=1):
        if arm == "gtf_core":
            ready, problems = G.check_ready("core")
            print(f"[dry-run] {position:2d}. gtf_core: " + ("READY -> would run" if ready else "NOT READY -> would be skipped"))
            for p in problems:
                print(f"            - {p}")
            ok = ok and ready
            continue
        if arm == "r6i":
            ck = REPO / f"results/s4_4/{R6I_CHECKPOINT_ARM}/pcstgnn/seed_{seed:05d}/checkpoint_best.pt"
            cfgs = yaml.safe_load((ck.parent / "config.yaml").read_text())
            cfgs.setdefault("model", {})["shuffle_seed"] = R6I_SHUFFLE_SEED
            m = _model(cfgs, edge_index)
            perm = json.loads((REPO / "config/s4_4/r6_shuffle_permutation.json").read_text())["permutation_26"]
            state = torch.load(ck, map_location="cpu", weights_only=True)["model"]
            out_exists = (REPO / f"results/s4_4/r6i/pcstgnn/seed_{seed:05d}/scoring_full_grid.json").exists()
            good = (ck.exists() and m.shuffle_map.tolist() == perm and not torch.equal(m.phys_table, state["phys_table"])
                    and torch.equal(m.phys_table, state["phys_table"][m.shuffle_map]))
            print(f"[dry-run] {position:2d}. r6i seed {seed}: scoring only; R5 checkpoint present: {ck.exists()}, shuffled table equals the "
                  f"R6 permutation and differs from the true table: {good}, already scored: {out_exists}")
            ok = ok and good
            continue
        cfg = rq._frozen_config(arm, seed)
        n_params = sum(p.numel() for p in _model(cfg, edge_index).parameters())
        run_dir = rq._run_dir(arm, seed)
        trained = (run_dir / "checkpoint_best.pt").exists() and (run_dir / "metrics.json").exists()
        diffs = rq._dict_diff(cfg, yaml.safe_load((run_dir / "config.yaml").read_text())) if trained else []
        guard = cfg.get("training", {}).get("collapse_abort")
        out_ok = (cfg["benchmark_version"] == arm)
        print(f"[dry-run] {position:2d}. {arm} seed {seed}: config OK, {n_params:,} parameters, output dir name matches arm: {out_ok}, "
              f"collapse guard: {'on' if guard else 'off (exempt)'}, already trained: {trained}, saved-config diffs: {len(diffs)}")
        ok = ok and out_ok and not diffs
    print(f"[dry-run] CUDA context created: {torch.cuda.is_initialized()}")
    ok = ok and not torch.cuda.is_initialized()
    print("[dry-run] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run()
    if STOP_PATH.exists():
        print(f"QUEUE REFUSED: {STOP_PATH} exists. A stopped queue is not restarted without the author's permission.", file=sys.stderr)
        return 1
    if not acquire_lock():
        print("QUEUE REFUSED: another queue process holds the lock.", file=sys.stderr)
        return 1

    import run_s4_4_remaining_queue as rq
    from fault_localization.experiments.runner import run as runner_run
    from score_s4_4_gate4_run import score_run

    _append({"stage": "queue_started", "pid": os.getpid(), "utc": utc(), "steps": [f"{a}:{s}" for a, _, s in QUEUE]})
    for position, (arm, seed_label, seed) in enumerate(QUEUE, start=1):
        if arm == "gtf_core":
            _append({"position": position, "arm": arm, "stage": "gtf_started", "utc": utc()})
            try:
                entry = run_gtf_core(lambda e: _append({"position": position, "arm": arm, **e}))
            except Exception:
                tb = traceback.format_exc()
                rq._release_gpu()
                _stop("exception", tb, position)
                _append({"position": position, "arm": arm, "stage": "gtf_failed", "error": tb})
                print(f"QUEUE STOPPED at position {position} (G-TF core): exception.\n{tb}", file=sys.stderr)
                return 1
            rq._release_gpu()
            _append({"position": position, "arm": arm, **entry, "utc": utc()})
            continue

        if arm == "r6i":
            try:
                run_r6i(seed, position, score_run)
            except Exception:
                tb = traceback.format_exc()
                rq._release_gpu()
                _stop("exception", tb, position)
                _append({"position": position, "arm": arm, "seed": seed, "stage": "scoring_failed", "error": tb})
                print(f"QUEUE STOPPED at position {position} (r6i seed {seed}): scoring raised an exception.\n{tb}", file=sys.stderr)
                return 1
            rq._release_gpu()
            print(f"[{position}/{len(QUEUE)}] r6i seed (seed={seed}) done.", flush=True)
            continue

        run_dir = rq._run_dir(arm, seed)
        trained = (run_dir / "checkpoint_best.pt").exists() and (run_dir / "metrics.json").exists()
        if trained:
            diffs = rq._dict_diff(rq._frozen_config(arm, seed), yaml.safe_load((run_dir / "config.yaml").read_text()))
            if diffs:
                _stop("config_mismatch", "\n".join(diffs), position)
                _append({"position": position, "arm": arm, "seed": seed, "stage": "config_mismatch", "diffs": diffs})
                print(f"QUEUE STOPPED at position {position} ({arm}): saved config differs from the frozen one.\n" + "\n".join(diffs), file=sys.stderr)
                return 1
            _append({"position": position, "arm": arm, "seed": seed, "stage": "training_skipped_already_done"})
            metrics = json.loads((run_dir / "metrics.json").read_text()); train_s = None
        else:
            t0 = time.perf_counter()
            _append({"position": position, "arm": arm, "seed_label": seed_label, "seed": seed,
                     "stage": "training_started", "utc": utc()})
            try:
                run_dir = rq._run_with_heartbeat(run_dir, runner_run, rq._frozen_config(arm, seed))
            except Exception:
                tb = traceback.format_exc()
                _stop("exception", tb, position)
                _append({"position": position, "arm": arm, "seed": seed, "stage": "training_failed", "error": tb})
                print(f"QUEUE STOPPED at position {position} ({arm} seed {seed}): training raised an exception.\n{tb}", file=sys.stderr)
                return 1
            finally:
                rq._release_gpu()
            metrics = json.loads((run_dir / "metrics.json").read_text()); train_s = time.perf_counter() - t0
        collapsed = bool(metrics.get("collapse_abort", {}).get("triggered", False))
        _append({"position": position, "arm": arm, "seed_label": seed_label, "seed": seed, "stage": "training_done",
                 "train_seconds": train_s, "collapsed": collapsed,
                 "best_validation_graph_f1": metrics.get("best_validation_graph_f1")})
        if collapsed:
            _stop("collapsed", json.dumps(metrics.get("collapse_abort")), position)
            print(f"QUEUE STOPPED at position {position} ({arm} seed {seed}): collapsed at epoch 3. {metrics.get('collapse_abort')}", file=sys.stderr)
            return 1
        if (run_dir / "scoring_full_grid.json").exists():
            _append({"position": position, "arm": arm, "seed": seed, "stage": "scoring_skipped_already_done"})
        else:
            _append({"position": position, "arm": arm, "seed": seed, "stage": "scoring_started", "utc": utc()})
            try:
                t1 = time.perf_counter(); score_run(arm, seed); score_s = time.perf_counter() - t1
            except Exception:
                tb = traceback.format_exc()
                _stop("exception", tb, position)
                _append({"position": position, "arm": arm, "seed": seed, "stage": "scoring_failed", "error": tb})
                print(f"QUEUE STOPPED at position {position} ({arm} seed {seed}): scoring raised an exception.\n{tb}", file=sys.stderr)
                return 1
            finally:
                rq._release_gpu()
            _append({"position": position, "arm": arm, "seed": seed, "stage": "scoring_done", "scoring_seconds": score_s})
        print(f"[{position}/{len(QUEUE)}] {arm} seed{seed_label} (seed={seed}) done.", flush=True)

    _append({"stage": "queue_complete", "utc": utc()})
    print("Queue complete: R6^i x3, R1, R2 x3.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
