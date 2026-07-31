"""Train all baselines on a scaled trend-test dataset (docs/MEDIUM-DATASET.md).

Runs each of {gru, rgcn, rgsage_mean, rgsage_max, rgatv2} at seeds 42/43/44
against dataset_<tag>/, writing to results/trend_<tag>/. Model configs are
reused unmodified (--data-root/--benchmark-version overrides only) so no
architecture, hyperparameter, or loss change slips into the trend test.

Each run is a separate subprocess so one crash (e.g. an OOM) does not abort
the sweep; failures are collected and reported at the end. Completed runs
(metrics.json already present) are skipped automatically — the sweep is
resumable.

Usage:
    uv run python scripts/run_trend_sweep.py --tag medium_A
    uv run python scripts/run_trend_sweep.py --tag medium_A --models gru rgatv2 --seeds 42
    uv run python scripts/run_trend_sweep.py --tag medium_A --ablation-model rgsage_max --ablation-seed 42
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

MODELS = ["gru", "rgcn", "rgsage_mean", "rgsage_max", "rgatv2"]
SEEDS = [42, 43, 44]


def run_one(model: str, seed: int, tag: str, disable_edges: bool = False, benchmark_prefix: str = "trend") -> tuple[bool, float]:
    benchmark_version = f"{benchmark_prefix}_{tag}" + ("_noedges" if disable_edges else "")
    run_dir = Path("results") / benchmark_version / model / f"seed_{seed:05d}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {model} seed={seed} tag={tag} disable_edges={disable_edges} — already complete")
        return True, 0.0

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", f"config/{model}.yaml",
        "--seed", str(seed),
        "--data-root", f"dataset_{tag}",
        "--benchmark-version", benchmark_version,
    ]
    if disable_edges:
        cmd.append("--disable-edges")

    print(f"[run] {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (rc={result.returncode})"
    print(f"[{status}] {model} seed={seed} tag={tag} disable_edges={disable_edges} — {elapsed/60:.1f} min")
    return ok, elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--ablation-model", default=None, choices=MODELS, help="Also run one edges-off ablation for this model")
    p.add_argument("--ablation-seed", type=int, default=42)
    p.add_argument("--benchmark-prefix", default="trend", help="results/<prefix>_<tag>/ output namespace (default 'trend')")
    args = p.parse_args()

    failures: list[str] = []
    total_start = time.time()

    for model in args.models:
        for seed in args.seeds:
            ok, _ = run_one(model, seed, args.tag, benchmark_prefix=args.benchmark_prefix)
            if not ok:
                failures.append(f"{model} seed={seed}")

    if args.ablation_model:
        ok, _ = run_one(args.ablation_model, args.ablation_seed, args.tag, disable_edges=True, benchmark_prefix=args.benchmark_prefix)
        if not ok:
            failures.append(f"{args.ablation_model} seed={args.ablation_seed} (no-edges ablation)")

    total_elapsed = time.time() - total_start
    print(f"\nSweep for tag={args.tag} done in {total_elapsed/60:.1f} min. "
          f"{len(failures)} failure(s).")
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
