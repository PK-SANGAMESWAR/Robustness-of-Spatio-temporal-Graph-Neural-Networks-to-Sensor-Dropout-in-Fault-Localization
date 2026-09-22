"""PIGNN selection screen — which *kind* of physics injection helps?

"PIGNN" is not one method. Physics enters a graph model at different places,
and those places have different data requirements, different costs, and — this
is the open question — possibly very different value. This screen answers it by
holding the backbone fixed and varying only the injection point.

    P3  Type A, physics in the LOSS   CE + lambda*L_phys, sag-pattern residual
    P1  Type C, physics in the GRAPH  impedance-weighted messages
    P2  Type D, physics in the PRIOR  sensitivity bias on the class logits

Every arm is the same GRU-128 + 4-hop RGSAGE-Max that produced
``results/fullscale_final/rgsage_max/``. P1 and P2 add exactly **one scalar
parameter** each (95,258 -> 95,259); P3 adds **none**. There is therefore no
capacity confound anywhere in the screen, and with all switches off the model is
byte-for-byte RGSAGE (asserted in ``tests/test_pignn.py``).

Reference, free of charge
-------------------------
The screen is run at a reduced 15-epoch budget, so it is compared against the
baseline's **own epoch-15 validation F1 = 0.816**, read from
``results/fullscale_final/rgsage_max/seed_00042/metrics_by_epoch.csv``. No
baseline re-run is needed, and the comparison is budget-matched.

Known limitation, stated up front: a reduced-epoch screen is biased against
slow-converging models. ZoneRefine is the local precedent — 0.700 at e15 but
0.811 at e30, against RGSAGE-Max's 0.784/0.844 — so an arm that trails at e15 by
a small margin has NOT been shown to be worse at e30. Only the winner is taken
to full budget; a close second should be too.

Order is deliberate: P3 runs first because it exercises the most new code (the
physics term inside the training loop), so a pipeline failure surfaces in the
first ~15 minutes instead of after hours.

Resumable: any arm with an existing metrics.json is skipped.

Usage:
    uv run python scripts/run_pignn_screen.py                  # the screen
    uv run python scripts/run_pignn_screen.py --epochs 1 \
        --suffix _smoke                                        # pipeline check
    uv run python scripts/run_pignn_screen.py --arms p1        # one arm
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fault_query import run_one

MODEL = "pignn"
ARMS = ["p3", "p1", "p2"]
BASELINE_EPOCH_CSV = "results/fullscale_final/rgsage_max/seed_00042/metrics_by_epoch.csv"


def baseline_reference(epoch: int) -> float | None:
    """RGSAGE-Max's own validation F1 at the screen's budget."""
    try:
        rows = list(csv.DictReader(open(BASELINE_EPOCH_CSV)))
    except OSError:
        return None
    for r in rows:
        if int(r["epoch"]) == epoch:
            return float(r["val_graph_f1"])
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default="results")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--gnn-layers", type=int, default=4)
    p.add_argument("--grad-clip-norm", default="1.0")
    p.add_argument("--warmup-epochs", default="3")
    p.add_argument("--suffix", default="",
                   help="appended to the benchmark_version, e.g. _smoke")
    args = p.parse_args()

    ref = baseline_reference(args.epochs)
    print(f"PIGNN screen — arms {args.arms}, seed {args.seed}, {args.epochs} epochs",
          flush=True)
    if ref is not None:
        print(f"Reference: RGSAGE-Max val_graph_f1 @ epoch {args.epochs} = {ref:.3f} "
              f"(same budget, no re-run needed)", flush=True)
    else:
        print(f"Reference: NOT FOUND at {BASELINE_EPOCH_CSV} — screen still valid, "
              f"but compare manually", flush=True)

    failures = []
    for arm in args.arms:
        version = f"pignn_{arm}{args.suffix}"
        if not run_one(f"config/pignn_{arm}.yaml", args.seed, args, version, model=MODEL):
            failures.append(arm)

    print("\n=== screen complete ===", flush=True)
    for arm in args.arms:
        mj = (Path(args.output_root) / f"pignn_{arm}{args.suffix}" / MODEL
              / f"seed_{args.seed:05d}" / "metrics.json")
        if mj.exists():
            import json
            d = json.loads(mj.read_text())
            print(f"  {arm}: test_graph_f1={d['test_graph']['f1']:.4f}  "
                  f"best_val={d['best_validation_graph_f1']:.4f}", flush=True)
        else:
            print(f"  {arm}: NO RESULT", flush=True)
    if ref is not None:
        print(f"  baseline RGSAGE-Max @ e{args.epochs}: val={ref:.4f}", flush=True)
    if failures:
        print(f"[done] failed arms: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
