"""Full-scale (100-run / 3.30M-window) seed-42 sweep — the user-requested
larger-corpus counterpart to run_fullscale_final_sweep.py (which uses the
50-run / 1.65M corpus).

Same FROZEN config as the 50-run sweep — GNN depth 4, 30 epochs, inverse-freq
weighted CE, grad-clip 1.0 + warmup 3 — the ONLY change is the dataset
(dataset_fullscale_100) and the output benchmark_version (fullscale_final_100).

All FIVE models train fresh: unlike the 50-run sweep, there is no rgsage_max to
reuse (the reused checkpoint was trained on the 50-run corpus), so rgsage_max is
included and trained from scratch on 100-run data.

rgatv2 is scheduled first: at 50-run it showed a late-epoch (ep29-30) instability
spike that best-checkpoint selection absorbed cleanly (test 0.610 from the ep26
peak); scheduling it first surfaces any repeat early rather than after ~4 days.

Each (model, seed) is a crash-isolated subprocess, skipped if already complete
(resumable). Expect ~25 h/model (~2x the 50-run time) → ~4-5 days total.

Usage:
    uv run python scripts/run_fullscale100_seed42_sweep.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

MODELS = ["rgatv2", "rgcn", "gru", "rgsage_mean", "rgsage_max"]
SEEDS = [42]
DATA_ROOT = "dataset_fullscale_100"
BENCHMARK_VERSION = "fullscale_final_100"
EPOCHS = 30
GNN_DEPTH = 4
GRAD_CLIP_NORM = "1.0"
WARMUP_EPOCHS = "3"


def run_one(model: str, seed: int) -> tuple[bool, float]:
    run_dir = Path("results") / BENCHMARK_VERSION / model / f"seed_{seed:05d}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {model} seed={seed} — already complete", flush=True)
        return True, 0.0

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", f"config/{model}.yaml",
        "--seed", str(seed),
        "--data-root", DATA_ROOT,
        "--benchmark-version", BENCHMARK_VERSION,
        "--epochs", str(EPOCHS),
        "--grad-clip-norm", GRAD_CLIP_NORM,
        "--warmup-epochs", WARMUP_EPOCHS,
    ]
    if model != "gru":
        cmd += ["--gnn-layers", str(GNN_DEPTH)]

    print(f"[run] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (rc={result.returncode})"
    print(f"[{status}] {model} seed={seed} — {elapsed/60:.1f} min", flush=True)
    return ok, elapsed


def main() -> None:
    failures: list[str] = []
    total_start = time.time()
    for model in MODELS:
        for seed in SEEDS:
            ok, _ = run_one(model, seed)
            if not ok:
                failures.append(f"{model} seed={seed}")
    total = time.time() - total_start
    print(f"\n100-run sweep done in {total/60:.1f} min. {len(failures)} failure(s).", flush=True)
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
