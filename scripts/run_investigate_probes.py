"""INVESTIGATE-branch diagnostics for the seed-42 sweep (all 4 GNNs clustered
at 0.42-0.44, well below the paper's 0.86-0.95 and below the 0.55 EXTEND
floor). Training curves show val_graph_f1 still rising at the final epoch
for rgsage_max/rgatv2 and train_loss still falling for 3/4 models -> pointing
to underfitting and/or the 1-hop GNN depth, not an architecture bug.

Three cheap, single-model (rgsage_max), single-seed (42) probes on the
EXISTING dataset_fullscale_50 -- no regeneration, no other model touched:

  1. diag_epochs45   -- epochs 15 -> 45 (3x), everything else unchanged.
                        Tests: is 15 epochs just not enough (underfitting)?
  2. diag_depth2     -- gnn_layers 1 -> 2, epochs unchanged (15).
  3. diag_depth3     -- gnn_layers 1 -> 3, epochs unchanged (15).
                        depth2 and depth3 together test whether 1-hop depth
                        is the ceiling (faults >1 hop from a mu-PMU are
                        architecturally unreachable) WITHOUT conflating that
                        with over-smoothing: GNNs commonly degrade at 3
                        layers even when 2 layers helps, so a depth3-only
                        probe risks a false "depth doesn't help" reading
                        when depth2 was actually the fix. Comparing 1 vs 2
                        vs 3 distinguishes "depth doesn't help" from
                        "3 is too deep, 2 is the sweet spot."
  4. diag_noedges50  -- edges-off ablation (existing --disable-edges flag),
                        epochs/depth unchanged (15/1), at the full 50-run
                        scale (medium-A's ablation was only at 20 runs).
                        Tests: does the graph mechanism (edges genuinely
                        help) still hold at this scale, or has the gap
                        closed?

Each run is its own subprocess (crash-isolated) and skipped if already
complete (resumable), same pattern as the other sweep scripts.

Usage:
    uv run python scripts/run_investigate_probes.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

SEED = 42
DATA_ROOT = "dataset_fullscale_50"
MODEL_CONFIG = "config/rgsage_max.yaml"

PROBES = [
    # Cheapest/most-informative first: the depth probes test the likely real
    # ceiling (1-hop receptive field). If depth-2 lifts F1, a 45-epoch 1-layer
    # run would have been partly wasted -- so it runs last, after we already
    # know whether depth matters.
    {"tag": "diag_depth2", "extra": ["--gnn-layers", "2"]},
    {"tag": "diag_depth3", "extra": ["--gnn-layers", "3"]},
    {"tag": "diag_noedges50", "extra": ["--disable-edges"]},
    {"tag": "diag_epochs45", "extra": ["--epochs", "45"]},
]


def run_one(tag: str, extra: list[str]) -> tuple[bool, float]:
    run_dir = Path("results") / tag / "rgsage_max" / f"seed_{SEED:05d}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {tag} — already complete", flush=True)
        return True, 0.0

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", MODEL_CONFIG,
        "--seed", str(SEED),
        "--data-root", DATA_ROOT,
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

    for probe in PROBES:
        ok, _ = run_one(probe["tag"], probe["extra"])
        if not ok:
            failures.append(probe["tag"])

    total_elapsed = time.time() - total_start
    print(f"\nInvestigate-probes done in {total_elapsed/60:.1f} min. {len(failures)} failure(s).", flush=True)
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
