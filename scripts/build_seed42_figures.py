"""Figures for the seed-42-only 5-model sweep (50-dataset-training.md).

Reads the same per-run metrics.json/predictions_test.csv/metrics_by_epoch.csv
that build_seed42_summary.py reads, and renders PNGs to
results/fullscale_50_seed42/figures/. Skips any model whose run hasn't
completed yet (resumable-friendly, same as the summary script).

Colors: validated categorical order from the dataviz skill's reference
palette (references/palette.md) for model identity; a single-hue sequential
blue ramp for fault resistance, since resistance is an ordered magnitude
(0.1 < 1 < 10 Ω), not a category — light step = easiest condition (low R),
dark step = hardest (high R).

Usage:
    uv run python scripts/build_seed42_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _metrics import nofault_precision_recall, resistance_f1

RESULTS = Path("results")
BENCHMARK_DIR = "fullscale_50_seed42"
SEED = 42
MEDIUM_A_PRIOR = 0.3426210973075208
PAPER_RANGE = (0.86, 0.95)

MODELS = ["gru", "rgcn", "rgsage_mean", "rgsage_max", "rgatv2"]
MODEL_LABELS = {"gru": "GRU", "rgcn": "RGCN", "rgsage_mean": "RGSAGE-Mean",
                "rgsage_max": "RGSAGE-Max", "rgatv2": "RGATv2"}
MODEL_COLOR = {
    "gru": "#2a78d6", "rgcn": "#eb6834", "rgsage_mean": "#1baf7a",
    "rgsage_max": "#eda100", "rgatv2": "#e87ba4",
}
RESISTANCE_COLOR = {0.1: "#6da7ec", 1.0: "#2a78d6", 10.0: "#184f95"}

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


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load_runs() -> dict[str, dict]:
    runs = {}
    for model in MODELS:
        run_dir = RESULTS / BENCHMARK_DIR / model / f"seed_{SEED:05d}"
        metrics_path = run_dir / "metrics.json"
        preds_path = run_dir / "predictions_test.csv"
        epoch_path = run_dir / "metrics_by_epoch.csv"
        if not metrics_path.exists() or not preds_path.exists():
            continue
        runs[model] = {
            "metrics": json.loads(metrics_path.read_text()),
            "preds": pd.read_csv(preds_path),
            "epochs": pd.read_csv(epoch_path) if epoch_path.exists() else None,
            "confusion": np.loadtxt(run_dir / "confusion_matrix.csv", delimiter=",")
                if (run_dir / "confusion_matrix.csv").exists() else None,
        }
    return runs


def fig_graph_macro_f1(runs: dict, out_dir: Path) -> None:
    models = [m for m in MODELS if m in runs]
    values = [runs[m]["metrics"]["test_graph"]["f1"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_axes(ax)
    ax.axhspan(*PAPER_RANGE, color="#e1e0d9", alpha=0.6, zorder=0)
    ax.text(len(models) - 0.5, PAPER_RANGE[1] + 0.01, "paper range 0.86–0.95",
            ha="right", va="bottom", color=INK_MUTED, fontsize=9)
    ax.axhline(MEDIUM_A_PRIOR, color=INK_MUTED, linestyle="--", linewidth=1.5, zorder=1)
    ax.text(-0.4, MEDIUM_A_PRIOR + 0.015, f"medium-A (20 runs) = {MEDIUM_A_PRIOR:.3f}",
            color=INK_MUTED, fontsize=9, va="bottom")

    bars = ax.bar(models, values, color=[MODEL_COLOR[m] for m in models],
                   width=0.6, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Soft-voted graph macro-F1 (test)")
    ax.set_title("Seed-42 sweep: soft-voted graph macro-F1 by model")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    fig.tight_layout()
    fig.savefig(out_dir / "graph_macro_f1_by_model.png", dpi=150)
    plt.close(fig)


def fig_per_resistance(runs: dict, out_dir: Path) -> None:
    models = [m for m in MODELS if m in runs]
    resistances = [0.1, 1.0, 10.0]

    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax)
    n_r = len(resistances)
    width = 0.8 / n_r
    x = np.arange(len(models))
    for i, r in enumerate(resistances):
        vals = [resistance_f1(runs[m]["preds"], r) for m in models]
        offset = (i - (n_r - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9, color=RESISTANCE_COLOR[r],
                       label=f"{r:g} Ω", zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", color=INK_SECONDARY, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title("Per-resistance macro-F1 by model (0.1 / 1 / 10 Ω)")
    ax.legend(title="Fault resistance", frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "per_resistance_f1.png", dpi=150)
    plt.close(fig)


def fig_nofault_precision_recall(runs: dict, out_dir: Path) -> None:
    models = [m for m in MODELS if m in runs]
    precisions, recalls = [], []
    for m in models:
        p, r = nofault_precision_recall(runs[m]["preds"])
        precisions.append(p)
        recalls.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_axes(ax)
    width = 0.35
    x = np.arange(len(models))
    b1 = ax.bar(x - width / 2, precisions, width, color="#2a78d6", label="Precision", zorder=3)
    b2 = ax.bar(x + width / 2, recalls, width, color="#eb6834", label="Recall", zorder=3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", color=INK_SECONDARY, fontsize=8)

    ax.axhline(0.99, color=INK_MUTED, linestyle="--", linewidth=1.5, zorder=1)
    ax.text(len(models) - 0.5, 0.985, "0.99 detection-preserved floor",
            ha="right", va="top", color=INK_MUTED, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("No-fault class (0) score")
    ax.set_title("No-fault detection: precision & recall by model")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "nofault_precision_recall.png", dpi=150)
    plt.close(fig)


def fig_training_curves(runs: dict, out_dir: Path) -> None:
    models = [m for m in MODELS if m in runs and runs[m]["epochs"] is not None]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_axes(ax)
    for m in models:
        ep = runs[m]["epochs"]
        ax.plot(ep["epoch"], ep["val_graph_f1"], color=MODEL_COLOR[m],
                 linewidth=2, marker="o", markersize=4, label=MODEL_LABELS[m], zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation graph macro-F1")
    ax.set_title("Training curves: validation soft-voted graph macro-F1")
    ax.legend(frameon=False, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves_val_f1.png", dpi=150)
    plt.close(fig)


def fig_confusion_matrix(runs: dict, out_dir: Path) -> None:
    completed = {m: runs[m] for m in MODELS if m in runs and runs[m]["confusion"] is not None}
    if not completed:
        return
    best_model = max(completed, key=lambda m: completed[m]["metrics"]["test_graph"]["f1"])
    cm = completed[best_model]["confusion"]
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Confusion matrix (row-normalized): best model = {MODEL_LABELS[best_model]}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks(range(0, 26, 5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true-class row")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{best_model}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = RESULTS / BENCHMARK_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    if not runs:
        print(f"No completed runs found yet under results/{BENCHMARK_DIR}/.")
        return

    fig_graph_macro_f1(runs, out_dir)
    fig_per_resistance(runs, out_dir)
    fig_nofault_precision_recall(runs, out_dir)
    fig_training_curves(runs, out_dir)
    fig_confusion_matrix(runs, out_dir)

    print(f"Wrote figures for {len(runs)}/{len(MODELS)} completed models to {out_dir}/")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
