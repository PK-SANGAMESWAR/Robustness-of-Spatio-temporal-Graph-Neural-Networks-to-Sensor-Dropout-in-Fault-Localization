"""Aggregate the seed-42-only 5-model sweep (50-dataset-training.md Steps 2-3)
and apply the STOP-AT-50 / EXTEND-TO-100 / INVESTIGATE gate.

Reads results/fullscale_50_seed42/*/seed_00042/{metrics.json,predictions_test.csv},
writes:

  results/fullscale_50_seed42/benchmark_summary.csv    — per model (n=1 seed)
  results/fullscale_50_seed42/benchmark_comparison.csv — one row per model

and prints the full table plus the gate verdict, using medium-A's 0.343
(rgsage_max, edges-on, single seed) as the prior trend point.

Usage:
    uv run python scripts/build_seed42_summary.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _metrics import collect_rows

RESULTS = Path("results")
BENCHMARK_DIR = "fullscale_50_seed42"
MEDIUM_A_PRIOR = 0.3426210973075208  # rgsage_max, edges-on, medium-A (20 runs/scenario), seed 42

SCALES = {"fullscale_50_seed42": (BENCHMARK_DIR, 50)}

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


def resistance_shortfall_note(summary: pd.DataFrame, best_model: str) -> str:
    row = summary[summary["model"] == best_model].iloc[0]
    r01, r1, r10 = row["f1_r0p1_mean"], row["f1_r1_mean"], row["f1_r10_mean"]
    spread = max(r01, r1, r10) - min(r01, r1, r10)
    if r10 < min(r01, r1) - 0.15:
        return (f"shortfall concentrated at 10Ω (f1_r0p1={r01:.3f}, f1_r1={r1:.3f}, "
                f"f1_r10={r10:.3f}) — consistent with a PHYSICAL limit (weak fault signal), not a bug.")
    if spread < 0.10:
        return (f"shortfall spread evenly across resistances (f1_r0p1={r01:.3f}, f1_r1={r1:.3f}, "
                f"f1_r10={r10:.3f}) — NOT resistance-specific, more consistent with a BUG than physics.")
    return f"f1_r0p1={r01:.3f}, f1_r1={r1:.3f}, f1_r10={r10:.3f} — mixed pattern, inspect manually."


def print_gate(summary: pd.DataFrame) -> None:
    best_row = summary.loc[summary[summary["model"] != "gru"]["graph_macro_f1_mean"].idxmax()] \
        if (summary["model"] != "gru").any() else summary.loc[summary["graph_macro_f1_mean"].idxmax()]
    best_model, best_f1 = best_row["model"], best_row["graph_macro_f1_mean"]

    print("\n--- GATE: STOP-AT-50 / EXTEND-TO-100 / INVESTIGATE (50-dataset-training.md Step 3) ---")
    print(f"  best GNN model: {best_model}, graph_macro_f1 = {best_f1:.4f} (seed 42 only)")
    print(f"  medium-A (20 runs) prior point: {MEDIUM_A_PRIOR:.4f}")
    rising = best_f1 >= MEDIUM_A_PRIOR
    print(f"  20->50 trend: {'rising' if rising else 'flat/falling'} ({MEDIUM_A_PRIOR:.4f} -> {best_f1:.4f})")

    nofault_recall = best_row["nofault_recall_mean"]
    print(f"  no-fault recall (detection preserved check): {nofault_recall:.4f}")

    if best_f1 >= 0.80 or (best_f1 >= 0.75 and rising):
        print(f"VERDICT: STOP-AT-50 — best-GNN {best_f1:.4f} is within striking distance of the "
              f"paper's 0.86-0.95 range. Next step is seeds 43/44 for error bars.")
    elif 0.60 <= best_f1 < 0.80 and rising:
        print(f"VERDICT: EXTEND-TO-100 — best-GNN {best_f1:.4f} is promising and still rising from "
              f"{MEDIUM_A_PRIOR:.4f}. Append runs 51-100, rebuild, re-run the sweep at 100.")
    else:
        print(f"VERDICT: INVESTIGATE — best-GNN {best_f1:.4f} is well below expectation (<~0.55) "
              f"or the 20->50 trend is flat/weak. Do NOT spend compute extending to 100 or adding seeds.")
        print(f"  Per-resistance breakdown for {best_model}: {resistance_shortfall_note(summary, best_model)}")
        print("  Also check: hyperparameters vs. paper (LR/epochs/hidden dims/GAT heads); "
              "graph correctness at this scale (edges-on/off gap still present); "
              "label alignment / normalization on the correct split.")


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
    print("\n--- Full results table (seed 42 only) ---")
    display_cols = ["model", "n_seeds"] + [f"{c}_mean" for c in METRIC_COLUMNS]
    print(summary[display_cols].to_string(index=False))

    n_models_expected = 5
    if len(summary) < n_models_expected:
        print(f"\n(Note: sweep still in progress — {len(summary)}/{n_models_expected} models present. "
              f"Gate below is computed on partial data.)")

    print_gate(summary)


if __name__ == "__main__":
    main()
