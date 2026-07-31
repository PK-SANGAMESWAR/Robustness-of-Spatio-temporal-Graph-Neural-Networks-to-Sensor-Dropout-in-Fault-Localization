"""NEXT-STEPS.md Step 4b report — standard vs dropout-aware degradation overlay.

Merges two degradation_curves.csv files produced by run_dropout_study.py:
  - standard-trained models     (results/dropout_study/degradation_curves.csv)
  - dropout-aware models        (results/dropout_study_aware/degradation_curves.csv)

Writes:
  <output-dir>/dropout_aware_comparison.csv  — per (model, k): F1 standard vs
      dropout-aware + recovery (Δ = dropout_aware − standard).
  <output-dir>/dropout_aware_comparison.png  — overlaid degradation curves,
      standard (dashed) vs dropout-aware (solid), one color per model.

Usage:
    uv run python scripts/build_dropout_comparison.py \
        --standard results/dropout_study/degradation_curves.csv \
        --dropout-aware results/dropout_study_aware/degradation_curves.csv \
        --output-dir results/dropout_study
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

MODELS_ORDER = ["gru", "rgcn", "rgsage_mean", "rgsage_max", "rgatv2"]
MODEL_LABELS = {"gru": "GRU", "rgcn": "RGCN", "rgsage_mean": "RGSAGE-Mean",
                "rgsage_max": "RGSAGE-Max", "rgatv2": "RGATv2"}
MODEL_COLOR = {"gru": "#2a78d6", "rgcn": "#eb6834", "rgsage_mean": "#1baf7a",
               "rgsage_max": "#eda100", "rgatv2": "#e87ba4"}
INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"


def build_comparison(std: pd.DataFrame, aware: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "k", "f1_mean", "f1_std"]
    merged = std[cols].merge(
        aware[cols], on=["model", "k"], suffixes=("_standard", "_dropout_aware"),
    )
    merged["recovery"] = merged["f1_mean_dropout_aware"] - merged["f1_mean_standard"]
    merged = merged.sort_values(
        ["model", "k"],
        key=lambda s: s.map({m: i for i, m in enumerate(MODELS_ORDER)}) if s.name == "model" else s,
    ).reset_index(drop=True)
    return merged


def build_figure(std: pd.DataFrame, aware: pd.DataFrame, out_path: Path,
                 k_levels: "list[int]") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED); ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0)

    models = [m for m in MODELS_ORDER if m in std["model"].values]
    for m in models:
        c = MODEL_COLOR.get(m, INK_SECONDARY)
        gs = std[std["model"] == m].sort_values("k")
        ga = aware[aware["model"] == m].sort_values("k")
        ax.errorbar(gs["k"], gs["f1_mean"], yerr=gs["f1_std"], marker="o",
                    linestyle="--", linewidth=1.8, markersize=5, capsize=3,
                    color=c, alpha=0.75, zorder=3)
        ax.errorbar(ga["k"], ga["f1_mean"], yerr=ga["f1_std"], marker="o",
                    linestyle="-", linewidth=2.2, markersize=6, capsize=3,
                    color=c, zorder=4)

    # Two legends: color = model, style = training regime.
    model_handles = [Line2D([0], [0], color=MODEL_COLOR[m], lw=2,
                            label=MODEL_LABELS[m]) for m in models]
    style_handles = [
        Line2D([0], [0], color=INK_SECONDARY, lw=2, linestyle="--", label="standard"),
        Line2D([0], [0], color=INK_SECONDARY, lw=2, linestyle="-", label="dropout-aware"),
    ]
    leg1 = ax.legend(handles=model_handles, frameon=False, loc="upper right",
                     title="model")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, frameon=False, loc="lower left", title="training")

    ax.set_xticks(k_levels)
    ax.set_xlabel("μPMUs dropped at inference (k of 25)")
    ax.set_ylabel("Soft-voted graph macro-F1 (test), mean ± std over mask seeds")
    ax.set_title("Standard vs dropout-aware training under sensor dropout")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--standard", default="results/dropout_study/degradation_curves.csv")
    p.add_argument("--dropout-aware", default="results/dropout_study_aware/degradation_curves.csv")
    p.add_argument("--output-dir", default="results/dropout_study")
    args = p.parse_args()

    std_path, aware_path = Path(args.standard), Path(args.dropout_aware)
    if not std_path.exists() or not aware_path.exists():
        print(f"Missing input(s): standard={std_path.exists()} "
              f"dropout_aware={aware_path.exists()}")
        return 1

    std = pd.read_csv(std_path)
    aware = pd.read_csv(aware_path)
    merged = build_comparison(std, aware)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dropout_aware_comparison.csv"
    merged.to_csv(csv_path, index=False)

    k_levels = sorted(std["k"].unique().tolist())
    fig_path = out_dir / "dropout_aware_comparison.png"
    build_figure(std, aware, fig_path, k_levels)

    print(f"Wrote {csv_path} ({len(merged)} rows) and {fig_path}")
    print("\n--- recovery (dropout_aware − standard) macro-F1 ---")
    pivot = merged.pivot(index="model", columns="k", values="recovery")
    print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
