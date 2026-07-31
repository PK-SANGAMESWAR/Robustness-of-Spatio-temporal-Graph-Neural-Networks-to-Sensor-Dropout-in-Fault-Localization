"""Figure for the INVESTIGATE-branch depth/epoch diagnostics
(50-dataset-training.md follow-up): the ceiling-lifting progression from the
depth-1 baseline through the combined depth-4 + 30-epoch confirmation run.

Usage:
    uv run python scripts/build_investigate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
PAPER_RANGE = (0.86, 0.95)

# Sequential blue ramp -- these are ordered steps along one dimension (more
# depth / more training), not independent categories, so one hue light->dark
# rather than the categorical palette.
STEPS = [
    ("baseline\n(depth=1, 15ep)", "fullscale_50_seed42", "#cde2fb"),
    ("depth=2\n(15ep)", "diag_depth2", "#86b6ef"),
    ("depth=3\n(15ep)", "diag_depth3", "#3987e5"),
    ("depth=4\n(15ep)", "diag_depth4", "#1c5cab"),
    ("depth=4 + 30ep\n(combined)", "diag_combined_depth4_epochs30", "#0d366b"),
]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "font.family": "sans-serif",
})


def main() -> None:
    labels, values, colors = [], [], []
    for label, benchmark_dir, color in STEPS:
        metrics_path = RESULTS / benchmark_dir / "rgsage_max" / "seed_00042" / "metrics.json"
        if not metrics_path.exists():
            continue
        import json
        d = json.loads(metrics_path.read_text())
        labels.append(label)
        values.append(d["test_graph"]["f1"])
        colors.append(color)

    if not values:
        print("No completed investigate-branch runs found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    ax.axhspan(*PAPER_RANGE, color="#e1e0d9", alpha=0.6, zorder=0)
    ax.text(len(labels) - 0.5, PAPER_RANGE[1] + 0.01, "paper range 0.86–0.95",
            ha="right", va="bottom", color=INK_MUTED, fontsize=9)

    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Soft-voted graph macro-F1 (test), rgsage_max seed 42")
    ax.set_title("INVESTIGATE branch: lifting the ceiling via GNN depth")
    fig.tight_layout()

    out_dir = RESULTS / "investigate_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "depth_progression.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
