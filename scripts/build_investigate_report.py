"""Compare the three INVESTIGATE-branch probes against the rgsage_max
baseline from the seed-42 sweep, and report pass/fail on whether either
lifts graph_macro_f1 toward the 0.6-0.8 band.

Usage:
    uv run python scripts/build_investigate_report.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _metrics import row_for_run

SEED = 42
RUNS = [
    ("baseline (epochs=15, depth=1, edges=on)", "fullscale_50_seed42"),
    ("diag_depth2 (epochs=15, depth=2, edges=on)", "diag_depth2"),
    ("diag_depth3 (epochs=15, depth=3, edges=on)", "diag_depth3"),
    ("diag_depth4 (epochs=15, depth=4, edges=on)", "diag_depth4"),
    ("diag_noedges50 (epochs=15, depth=1, edges=off)", "diag_noedges50"),
    ("diag_combined_depth4_epochs30 (epochs=30, depth=4, edges=on)", "diag_combined_depth4_epochs30"),
]

TARGET_BAND = (0.60, 0.80)


def main() -> None:
    rows = []
    for label, benchmark_dir in RUNS:
        seed_dir = Path("results") / benchmark_dir / "rgsage_max" / f"seed_{SEED:05d}"
        row = row_for_run(benchmark_dir, 50, "rgsage_max", seed_dir)
        if row is None:
            print(f"[pending] {label} — not complete yet")
            continue
        row["label"] = label
        rows.append(row)

    if not rows:
        print("No probes complete yet.")
        return

    df = pd.DataFrame(rows)
    out_dir = Path("results") / "investigate_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False)

    print("\n--- INVESTIGATE-branch probe comparison (rgsage_max, seed 42) ---")
    print(df[["label", "graph_macro_f1", "f1_r0p1", "f1_r1", "f1_r10",
              "nofault_precision", "nofault_recall"]].to_string(index=False))

    baseline_f1 = df.loc[df["label"].str.startswith("baseline"), "graph_macro_f1"]
    if baseline_f1.empty:
        print("\nBaseline not complete — cannot compute deltas yet.")
        return
    baseline_f1 = baseline_f1.iloc[0]

    print(f"\nBaseline graph_macro_f1 = {baseline_f1:.4f}; target band {TARGET_BAND}")
    for _, row in df.iterrows():
        if row["label"].startswith("baseline"):
            continue
        delta = row["graph_macro_f1"] - baseline_f1
        in_band = TARGET_BAND[0] <= row["graph_macro_f1"] <= TARGET_BAND[1]
        above_band = row["graph_macro_f1"] > TARGET_BAND[1]
        verdict = "PASS (ceiling lifted into/past target band)" if (in_band or above_band) \
            else ("moved but still short of target band" if delta > 0.02 else "no meaningful change")
        print(f"  {row['label']}: {row['graph_macro_f1']:.4f} (delta {delta:+.4f}) -> {verdict}")


if __name__ == "__main__":
    main()
