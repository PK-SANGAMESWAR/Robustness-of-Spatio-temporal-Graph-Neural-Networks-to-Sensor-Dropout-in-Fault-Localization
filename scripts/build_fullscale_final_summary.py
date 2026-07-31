"""Aggregate the deadline-mode definitive sweep (results/fullscale_final/):
5 models x 3 seeds, GNN depth=4, epochs=30, stabilized (grad clip + warmup).

Writes:
  results/fullscale_final/benchmark_summary.csv    — per model, mean±std over seeds
  results/fullscale_final/benchmark_comparison.csv — one row per (model, seed)

No gate/verdict logic here — per the deadline-mode decision, reproduction is
already accepted at depth-4 rgsage_max=0.834; this script just reports the
table.

Usage:
    uv run python scripts/build_fullscale_final_summary.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _metrics import collect_rows

RESULTS = Path("results")
BENCHMARK_DIR = "fullscale_final"
SCALES = {"fullscale_final": (BENCHMARK_DIR, 50)}

METRIC_COLUMNS = [
    "graph_macro_f1", "perwindow_macro_f1",
    "f1_r0p1", "f1_r1", "f1_r10",
    "nofault_precision", "nofault_recall",
]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        row: dict[str, float | str | int] = {"model": model, "n_seeds": len(g)}
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def main() -> None:
    df = collect_rows(SCALES, RESULTS)
    out_dir = RESULTS / BENCHMARK_DIR
    if df.empty:
        print(f"No completed runs found yet under {out_dir}/.")
        return

    df.to_csv(out_dir / "benchmark_comparison.csv", index=False)
    summary = build_summary(df)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)

    print(f"Wrote {out_dir / 'benchmark_comparison.csv'} ({len(df)} rows) and "
          f"{out_dir / 'benchmark_summary.csv'} ({len(summary)} models)")
    print("\n--- Full results table (mean ± std over seeds) ---")
    display_cols = ["model", "n_seeds"] + [f"{c}_mean" for c in METRIC_COLUMNS] + ["nofault_recall_std"]
    print(summary[display_cols].to_string(index=False))

    n_models_expected = 5
    n_seeds_expected = 3
    incomplete = summary[summary["n_seeds"] < n_seeds_expected]
    if len(summary) < n_models_expected or not incomplete.empty:
        print(f"\n(Note: sweep still in progress — {len(summary)}/{n_models_expected} models present, "
              f"{'all' if incomplete.empty else 'not all'} at {n_seeds_expected} seeds.)")

    min_recall = summary["nofault_recall_mean"].min()
    print(f"\nDetection-preserved check: min no-fault recall across models = {min_recall:.4f}")


if __name__ == "__main__":
    main()
