"""Phase 2 of the INVESTIGATE branch, run after diag_depth2/diag_depth3
established that 1-hop GNN depth was the real ceiling (0.439 -> 0.571 ->
0.751) and diag_epochs45 was resolved (killed or let finish as informational,
since it tests the now-obsolete 1-layer architecture).

  1. diag_depth4     -- rgsage_max, 4 GNN layers, 15 epochs, seed 42.
                        Decides the depth sweet spot: if depth-4 beats
                        depth-3, deeper still helps; if it ties/loses,
                        depth-3 is locked (over-smoothing/plateau).
  2. diag_combined   -- rgsage_max, LOCKED depth (3 or 4, chosen from step 1),
                        30 epochs. Depth and epochs were only tested
                        SEPARATELY so far; this is the first run that
                        combines both, and is the real gate toward the
                        paper's 0.86-0.95 range.

Locked depth is chosen automatically from the diag_depth3/diag_depth4
graph_macro_f1 comparison (see _metrics.row_for_run) -- no manual step.

Usage:
    uv run python scripts/run_investigate_probes_phase2.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

from _metrics import row_for_run

SEED = 42
DATA_ROOT = "dataset_fullscale_50"
MODEL_CONFIG = "config/rgsage_max.yaml"
COMBINED_EPOCHS = 30


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


def graph_f1(benchmark_dir: str) -> float | None:
    seed_dir = Path("results") / benchmark_dir / "rgsage_max" / f"seed_{SEED:05d}"
    row = row_for_run(benchmark_dir, 50, "rgsage_max", seed_dir)
    return row["graph_macro_f1"] if row else None


def main() -> None:
    failures: list[str] = []
    total_start = time.time()

    ok, _ = run_one("diag_depth4", ["--gnn-layers", "4"])
    if not ok:
        failures.append("diag_depth4")
        print("\ndiag_depth4 FAILED — cannot decide locked depth, stopping before the combined run.", flush=True)
        sys.exit(1)

    f1_depth3 = graph_f1("diag_depth3")
    f1_depth4 = graph_f1("diag_depth4")
    print(f"\ndepth3 graph_macro_f1={f1_depth3:.4f}, depth4 graph_macro_f1={f1_depth4:.4f}", flush=True)

    locked_depth = 4 if f1_depth4 > f1_depth3 else 3
    print(f"LOCKED depth = {locked_depth} "
          f"({'depth-4 beat depth-3' if locked_depth == 4 else 'depth-4 did not beat depth-3 (plateau/over-smoothing)'})",
          flush=True)

    combined_tag = f"diag_combined_depth{locked_depth}_epochs{COMBINED_EPOCHS}"
    ok, _ = run_one(combined_tag, ["--gnn-layers", str(locked_depth), "--epochs", str(COMBINED_EPOCHS)])
    if not ok:
        failures.append(combined_tag)

    total_elapsed = time.time() - total_start
    print(f"\nPhase 2 done in {total_elapsed/60:.1f} min. {len(failures)} failure(s).", flush=True)
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
