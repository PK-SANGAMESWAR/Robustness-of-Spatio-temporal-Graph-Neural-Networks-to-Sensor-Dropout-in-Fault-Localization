"""Train FaultQuery-STGNN — the physics-biased query-decoder architecture.

FaultQuery-STGNN (fault_localization/models/fault_query.py) keeps the GRU
temporal encoder and the ungated RGSAGE-Max spatial block of the strongest
baseline and replaces per-node classification + soft voting with a
cross-attention decoder over 26 learned fault-bus queries, biased by hop
distance over the full 123-bus topology, gated by sensor availability, and
weighted by a learned per-sensor reliability.

Frozen recipe — identical to run_fullscale_final_sweep.py and run_sdr_stgnn.py
(depth 4, 30 epochs, grad-clip 1.0, warmup 3) — so the decoder is the only thing
that differs from the RGSAGE-Max baseline it is compared against.

Two arms, selected by --dropout-aware:
    standard        config/fault_query.yaml               -> results/fault_query/
    dropout-aware   config/dropout_aware/fault_query.yaml -> results/fault_query_aware/

Resumable: any (arm, seed) with an existing metrics.json is skipped.

Usage:
    uv run python scripts/run_fault_query.py --seed 42
    uv run python scripts/run_fault_query.py --seed 42 --dropout-aware
    uv run python scripts/run_fault_query.py --seeds 42 43 44 --dropout-aware

Smoke check (1 epoch on the real corpus — confirms the loss moves and the
[B,1,26] contract survives the full pipeline):
    uv run python scripts/run_fault_query.py --seed 42 --epochs 1 \
        --benchmark-version fault_query_smoke
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "fault_query"


def run_one(config_path: str, seed: int, args, benchmark_version: str,
            model: str = MODEL) -> bool:
    run_dir = (Path(args.output_root) / benchmark_version / model
               / f"seed_{seed:05d}")
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {benchmark_version} seed={seed} — already complete", flush=True)
        return True

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", config_path,
        "--seed", str(seed),
        "--data-root", args.data_root,
        "--benchmark-version", benchmark_version,
        "--epochs", str(args.epochs),
        "--gnn-layers", str(args.gnn_layers),
        "--grad-clip-norm", str(args.grad_clip_norm),
        "--warmup-epochs", str(args.warmup_epochs),
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = (time.time() - start) / 60
    ok = result.returncode == 0
    print(f"[{'ok' if ok else f'FAILED rc={result.returncode}'}] "
          f"{benchmark_version} seed={seed} — {elapsed:.1f} min", flush=True)
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=None, help="single seed")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="several seeds, run in order (resumable)")
    p.add_argument("--dropout-aware", action="store_true",
                   help="train with sensor-dropout augmentation (k in 0,1,2,5)")
    p.add_argument("--output-root", default="results")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--benchmark-version", default=None,
                   help="defaults to fault_query / fault_query_aware")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--gnn-layers", type=int, default=4)
    p.add_argument("--grad-clip-norm", default="1.0")
    p.add_argument("--warmup-epochs", default="3")
    args = p.parse_args()

    seeds = args.seeds if args.seeds else [args.seed if args.seed is not None else 42]
    config_path = (f"config/dropout_aware/{MODEL}.yaml" if args.dropout_aware
                   else f"config/{MODEL}.yaml")
    benchmark_version = args.benchmark_version or (
        "fault_query_aware" if args.dropout_aware else "fault_query")

    print(f"Queue: {len(seeds)} run(s) — {benchmark_version} "
          f"({'dropout-aware' if args.dropout_aware else 'standard'})", flush=True)

    failures = 0
    for seed in seeds:
        if not run_one(config_path, seed, args, benchmark_version):
            failures += 1
    if failures:
        print(f"[done] {failures} of {len(seeds)} run(s) failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
