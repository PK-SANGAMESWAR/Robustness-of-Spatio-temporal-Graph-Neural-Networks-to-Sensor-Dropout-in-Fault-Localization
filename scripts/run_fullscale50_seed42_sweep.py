"""Definitive seed-42-only 5-model sweep on the 50-run/scenario dataset
(dataset_fullscale_50), per 50-dataset-training.md Step 1.

Deliberately seed 42 ONLY (no 43/44) -- this is a de-risking probe before
committing to the full 3-seed sweep. Model configs (config/<model>.yaml) are
reused unmodified -- only --data-root/--benchmark-version/--seed are
overridden -- so no architecture, hyperparameter, or loss change slips in.

Each model runs as its own subprocess so one crash does not abort the sweep;
completed runs (metrics.json already present) are skipped, so this script is
safely resumable/rerunnable.

Usage:
    uv run python scripts/run_fullscale50_seed42_sweep.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

MODELS = ["gru", "rgcn", "rgsage_mean", "rgsage_max", "rgatv2"]
SEED = 42
DATA_ROOT = "dataset_fullscale_50"
BENCHMARK_VERSION = "fullscale_50_seed42"


def run_one(model: str) -> tuple[bool, float]:
    run_dir = Path("results") / BENCHMARK_VERSION / model / f"seed_{SEED:05d}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {model} — already complete", flush=True)
        return True, 0.0

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", f"config/{model}.yaml",
        "--seed", str(SEED),
        "--data-root", DATA_ROOT,
        "--benchmark-version", BENCHMARK_VERSION,
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (rc={result.returncode})"
    print(f"[{status}] {model} — {elapsed/60:.1f} min", flush=True)
    return ok, elapsed


def main() -> None:
    failures: list[str] = []
    total_start = time.time()

    for model in MODELS:
        ok, _ = run_one(model)
        if not ok:
            failures.append(model)

    total_elapsed = time.time() - total_start
    print(f"\nSweep done in {total_elapsed/60:.1f} min. {len(failures)} failure(s).", flush=True)
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
