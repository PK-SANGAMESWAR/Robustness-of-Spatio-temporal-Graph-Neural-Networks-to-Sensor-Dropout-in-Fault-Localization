"""Aggregate the MEDIUM_DATASET trend-test results and apply the decision rubric.

Reads whichever of {tiny, medium_A, medium_B} have completed runs under
results/, computes the metrics docs/MEDIUM-DATASET.md asks for from each
run's metrics.json + predictions_test.csv, writes:

  results/trend_test/trend_summary.csv   — one row per (scale, model, seed)
  results/trend_test/pivot_graph_f1.csv  — model x scale, mean±std graph macro-F1

and prints the pivot table, the edges-on/off ablation result, and the
GO / INCONCLUSIVE / DEBUG verdict.

Metric mapping notes (this codebase's existing evaluate_loader/reporting
naming, reused unchanged):
  - graph_macro_f1    = metrics.json["test_graph"]["f1"]            (soft-voted, one prediction per window)
  - perwindow_macro_f1 = metrics.json["test_node_diagnostic"]["f1"]  (per-node diagnostic, existing "secondary" metric)
  - f1_r0p1/f1_r1/f1_r10, nofault_precision/recall are computed here from predictions_test.csv.

Usage:
    uv run python scripts/build_trend_summary.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _metrics import collect_rows as _collect_rows

RESULTS = Path("results")
OUT_DIR = RESULTS / "trend_test"

# scale -> (benchmark_version dir under results/, runs_per_scenario)
SCALES = {
    "tiny": ("benchmark_v1.1_samplerfix", 2),
    "medium_A": ("trend_medium_A", 20),
    "medium_A_noedges": ("trend_medium_A_noedges", 20),
    "medium_B": ("trend_medium_B", 40),
}
GNN_MODELS = ["rgcn", "rgsage_mean", "rgsage_max", "rgatv2"]


def collect_rows() -> pd.DataFrame:
    return _collect_rows(SCALES, RESULTS)


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["edges_enabled"]]
    scale_order = [s for s in SCALES if s in main["scale"].unique()]
    pivot = {}
    for model, g in main.groupby("model"):
        row = {}
        for scale in scale_order:
            vals = g[g["scale"] == scale]["graph_macro_f1"]
            if vals.empty:
                row[scale] = ""
            elif len(vals) == 1:
                row[scale] = f"{vals.mean():.4f} (n=1)"
            else:
                row[scale] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f} (n={len(vals)})"
        pivot[model] = row
    return pd.DataFrame(pivot).T[scale_order]


def print_ablation(df: pd.DataFrame) -> None:
    on = df[(df["model"] == "rgsage_max") & (df["scale"] == "medium_A") & (df["edges_enabled"])]
    off = df[(df["model"] == "rgsage_max") & (df["scale"] == "medium_A_noedges") & (~df["edges_enabled"])]
    print("\n--- Edges-on vs edges-off ablation: rgsage_max @ medium_A ---")
    if on.empty or off.empty:
        print("  (not yet available — run the medium_A sweep and the --ablation-model rgsage_max probe)")
        return
    on_f1 = on["graph_macro_f1"].mean()
    off_f1 = off["graph_macro_f1"].mean()
    print(f"  edges ON  graph_macro_f1: {on_f1:.4f} (n={len(on)})")
    print(f"  edges OFF graph_macro_f1: {off_f1:.4f} (n={len(off)})")
    verdict = "spatial propagation IS contributing" if on_f1 > off_f1 else "graph is NOT contributing — structural bug likely"
    print(f"  -> {verdict}")


def print_verdict(df: pd.DataFrame) -> None:
    main = df[df["edges_enabled"]]
    means = main.groupby(["scale", "model"])["graph_macro_f1"].mean().unstack("model")

    def best_gnn(scale: str) -> float | None:
        if scale not in means.index:
            return None
        available = [m for m in GNN_MODELS if m in means.columns]
        vals = means.loc[scale, available].dropna()
        return float(vals.max()) if not vals.empty else None

    def gru_floor(scale: str) -> float | None:
        if scale not in means.index or "gru" not in means.columns:
            return None
        v = means.loc[scale, "gru"]
        return float(v) if pd.notna(v) else None

    a_best, b_best = best_gnn("medium_A"), best_gnn("medium_B")
    a_gru, b_gru = gru_floor("medium_A"), gru_floor("medium_B")

    print("\n--- Decision rubric (docs/MEDIUM-DATASET.md) ---")
    print(f"  medium_A best-GNN graph_macro_f1: {a_best}")
    print(f"  medium_B best-GNN graph_macro_f1: {b_best}")

    if a_best is None:
        print("VERDICT: not enough data yet — medium_A has not completed.")
        return

    gap_a = (a_best - a_gru) if (a_best is not None and a_gru is not None) else None
    gap_b = (b_best - b_gru) if (b_best is not None and b_gru is not None) else None
    gap_widens = gap_a is not None and gap_b is not None and gap_b > gap_a

    if b_best is None:
        # Two-point trend only (tiny/medium_A implicit via a_best vs. chance floor).
        if a_best >= 0.15:
            print("VERDICT: INCONCLUSIVE-leaning-GO — medium_A clears the 0.15 bar, but medium_B "
                  "is needed to confirm the trend keeps rising before committing to full scale.")
        elif a_best >= 0.08:
            print("VERDICT: INCONCLUSIVE — some movement off the chance floor at medium_A "
                  "(0.08-0.15). Run medium_B (or an 80-run third point) before deciding.")
        else:
            print("VERDICT: DEBUG-leaning — medium_A best-GNN is still near chance (~0.05-0.08) "
                  "at 10x the tiny data. Run medium_B to confirm before concluding structural bug.")
        return

    if a_best >= 0.15 and b_best >= a_best and gap_widens:
        print("VERDICT: GO — mechanism confirmed (starvation was the cause). "
              "Proceed to full-scale dataset generation.")
    elif (0.08 <= a_best < 0.15) and b_best >= a_best:
        print("VERDICT: INCONCLUSIVE — rising but weak. Run one larger scale "
              "(e.g. 80 runs/scenario) before committing to full scale.")
    else:
        print("VERDICT: DEBUG — best-GNN macro-F1 stuck near chance (~0.05-0.08) despite "
              "10-20x more data, with no widening GNN-GRU gap. Do NOT scale up; "
              "work through the DEBUG CHECKLIST in docs/MEDIUM-DATASET.md.")


def main() -> None:
    df = collect_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("No completed trend-test runs found yet under results/.")
        return

    df.to_csv(OUT_DIR / "trend_summary.csv", index=False)
    pivot = build_pivot(df)
    pivot.to_csv(OUT_DIR / "pivot_graph_f1.csv")

    print(f"Wrote {OUT_DIR / 'trend_summary.csv'} ({len(df)} rows)")
    print("\n--- Pivot: graph_macro_f1 mean ± std (n seeds) ---")
    print(pivot.to_string())

    print_ablation(df)
    print_verdict(df)


if __name__ == "__main__":
    main()
