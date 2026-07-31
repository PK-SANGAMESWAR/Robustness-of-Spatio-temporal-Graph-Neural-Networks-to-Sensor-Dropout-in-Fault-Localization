"""DEADLINE MODE — the critical-path definitive sweep (50-dataset-training.md
follow-up, locked decisions from the deadline-mode instruction).

Reproduction is ACCEPTED at depth-4 rgsage_max = 0.834 (vs. paper's 0.86-0.95).
No further score-chasing: this sweep trains all 5 models at the frozen config
below, 3 seeds each, and that is the result reported for the baseline table.

Frozen config (do not change without an explicit new decision):
  - models: gru, rgcn, rgsage_mean, rgsage_max, rgatv2
  - GNN depth = 4 for every GNN variant (rgcn/rgsage_mean/rgsage_max/rgatv2);
    GRU has no spatial layer, so --gnn-layers is not passed to it.
  - epochs = 30 for every model (not the original paper's 11/15 split --
    this is the new locked recipe, chosen because 30 epochs is what the
    depth-4 diagnostics were run and validated at).
  - seeds = 42, 43, 44
  - 50 runs/scenario (dataset_fullscale_50), default feeder, inverse-frequency
    weighted CE (frozen, unchanged from every prior run in this project).
  - stabilization (grad_clip_norm=1.0, warmup_epochs=3) applied to ALL runs,
    per the Step-1 stability gate (diag_depth4_epochs30_stable).

Each (model, seed) is its own subprocess (crash-isolated) and skipped if
already complete (metrics.json present) -- resumable, same pattern as every
other sweep script in this project.

Usage:
    uv run python scripts/run_fullscale_final_sweep.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

# NEXT-STEPS.md Step 2 refinement: seed 42 ONLY first (not 3 seeds) for a
# complete baseline table in ~1.5 days instead of ~4-5; seeds 43/44 are
# Step 5 (backfilled later, not a blocker). rgatv2 scheduled early (not
# last) since rgatv2/rgcn at depth-4 have only been smoke-tested at 5
# epochs, not the full 30 -- if depth-4 attention instability shows up at
# full length, we want to know within hours, not after 30h.
MODELS = ["rgatv2", "rgcn", "gru", "rgsage_mean", "rgsage_max"]
SEEDS = [42]
DATA_ROOT = "dataset_fullscale_50"
BENCHMARK_VERSION = "fullscale_final"
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

    total_elapsed = time.time() - total_start
    print(f"\nFull sweep done in {total_elapsed/60:.1f} min. {len(failures)} failure(s).", flush=True)
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
