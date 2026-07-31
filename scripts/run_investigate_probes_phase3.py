"""Phase 3: resolve the depth-1-vs-paper discrepancy, and test whether
gradient clipping / LR warmup stabilizes depth-4's training-instability spike
(val-F1 crash at epochs 25-26 in diag_combined_depth4_epochs30).

Run AFTER diag_depth1_epochs30 (50-run, launched separately/already running)
completes -- this script does not touch that run, only skips it if done.

  1. diag_depth1_epochs30           -- (skip-if-done) paper-faithful arch
                                        (depth=1) at 30 epochs, 50-run data.
  2. diag_depth1_epochs30_100runs   -- same architecture/epochs, 100-run data
                                        (dataset_fullscale_100). Answers the
                                        gating question: does depth-1 climb
                                        toward the paper's range with more
                                        data, or stay stuck ~0.44?
  3. diag_depth4_epochs30_stable    -- depth=4, 30 epochs, 50-run data, WITH
                                        grad_clip_norm=1.0 and warmup_epochs=3.
                                        Compares against the unstable
                                        diag_combined_depth4_epochs30 (0.834,
                                        crashed at epoch 25-26) to see if
                                        clipping/warmup prevents the spike
                                        (important before trusting 3 seeds).

Usage:
    uv run python scripts/run_investigate_probes_phase3.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

SEED = 42
MODEL_CONFIG = "config/rgsage_max.yaml"

RUNS = [
    {
        "tag": "diag_depth1_epochs30",
        "data_root": "dataset_fullscale_50",
        "extra": ["--epochs", "30"],
    },
    {
        "tag": "diag_depth1_epochs30_100runs",
        "data_root": "dataset_fullscale_100",
        "extra": ["--epochs", "30"],
    },
    {
        "tag": "diag_depth4_epochs30_stable",
        "data_root": "dataset_fullscale_50",
        "extra": ["--gnn-layers", "4", "--epochs", "30", "--grad-clip-norm", "1.0", "--warmup-epochs", "3"],
    },
    # DEADLINE MODE (locked decision): diag_depth1_epochs30_100runs_stable is
    # permanently CUT from scope -- reproduction is accepted at depth-4=0.834,
    # and a 1-layer model cannot improve the result. Do not re-add without
    # an explicit decision to revisit.
]


def run_one(tag: str, data_root: str, extra: list[str]) -> tuple[bool, float]:
    run_dir = Path("results") / tag / "rgsage_max" / f"seed_{SEED:05d}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {tag} — already complete", flush=True)
        return True, 0.0

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", MODEL_CONFIG,
        "--seed", str(SEED),
        "--data-root", data_root,
        "--benchmark-version", tag,
        *extra,
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (rc={result.returncode})"
    print(f"[{status}] {tag} — {elapsed/60:.1f} min", flush=True)
    return ok, elapsed


def main() -> None:
    failures: list[str] = []
    total_start = time.time()

    for run in RUNS:
        ok, _ = run_one(run["tag"], run["data_root"], run["extra"])
        if not ok:
            failures.append(run["tag"])

    total_elapsed = time.time() - total_start
    print(f"\nPhase 3 done in {total_elapsed/60:.1f} min. {len(failures)} failure(s).", flush=True)
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
