"""Train ZoneRefine-STGNN — physics-partitioned hierarchical fault localization.

ZoneRefine (fault_localization/models/zone_refine.py) scores the 16 electrical
zones by a masked least-squares fit of the observed sag pattern against
impedance-derived templates, then resolves the bus within a zone with a learned
head, and multiplies the two into a 26-class distribution.

Frozen recipe — identical to every other sweep (depth 4, 30 epochs, grad-clip
1.0, warmup 3) — so the decoder/physics stage is the only difference from the
RGSAGE-Max baseline it is compared against.

Two arms, selected by --dropout-aware:
    standard        config/zone_refine.yaml               -> results/zone_refine/
    dropout-aware   config/dropout_aware/zone_refine.yaml -> results/zone_refine_aware/

Resumable: any (arm, seed) with an existing metrics.json is skipped.

Usage:
    uv run python scripts/run_zone_refine.py --seed 42
    uv run python scripts/run_zone_refine.py --seeds 42 43 44 --dropout-aware
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same launcher as the FaultQuery arm — one implementation, two models.
from run_fault_query import run_one

MODEL = "zone_refine"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--dropout-aware", action="store_true")
    p.add_argument("--output-root", default="results")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--benchmark-version", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--gnn-layers", type=int, default=4)
    p.add_argument("--grad-clip-norm", default="1.0")
    p.add_argument("--warmup-epochs", default="3")
    args = p.parse_args()

    seeds = args.seeds if args.seeds else [args.seed if args.seed is not None else 42]
    config_path = (f"config/dropout_aware/{MODEL}.yaml" if args.dropout_aware
                   else f"config/{MODEL}.yaml")
    benchmark_version = args.benchmark_version or (
        "zone_refine_aware" if args.dropout_aware else "zone_refine")

    print(f"Queue: {len(seeds)} run(s) — {benchmark_version} "
          f"({'dropout-aware' if args.dropout_aware else 'standard'})", flush=True)

    failures = sum(0 if run_one(config_path, s, args, benchmark_version, model=MODEL)
                   else 1 for s in seeds)
    if failures:
        print(f"[done] {failures} of {len(seeds)} run(s) failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
