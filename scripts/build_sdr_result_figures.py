"""All result figures for the SDR-STGNN sensor-dropout contribution.

Consumes degradation_curves.csv files produced by run_dropout_study.py (columns
model,k,f1_mean,f1_std,...) for three groups:

  --baselines  the 5 standard-trained baselines (gru/rgcn/rgsage_*/rgatv2)
  --aware      the dropout-aware RGSAGE-Max (the mask-aware / mask-concat baseline:
               receives the availability mask + trained under dropout, NOT gated)
  --sdr        the SDR-STGNN curve (availability-gated aggregation + dropout-aware)

Emits five conference figures (+ a summary CSV) to --out-dir:

  sdr_degradation_all.png   F1 vs k, all 5 baselines + SDR-STGNN (bold).
  sdr_ablation_backbone.png the 3-way ablation on the RGSAGE-Max backbone —
                            standard vs mask-aware+dropout-aware vs SDR-STGNN.
                            Isolates the structural gate (mask-aware → SDR).
  sdr_retention.png         retention = F1(k)/F1(0) per backbone variant (robustness
                            normalised so the flatter = more robust curve is obvious).
  sdr_audc_bar.png          normalised area-under-retention-curve — one robustness
                            scalar per variant (higher = degrades more gracefully).
  sdr_heatmap.png           model × k grid of macro-F1 (all variants), annotated.
  sdr_robustness_summary.csv per-variant F1@k, retention@k, and AUDC.

Usage (after the contribution run):
    uv run python scripts/build_sdr_result_figures.py \
        --baselines results/dropout_study/degradation_curves.csv \
        --aware     results/dropout_study_aware/degradation_curves.csv \
        --sdr       results/dropout_study_sdr/degradation_curves.csv \
        --out-dir   paper/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Shared palette (matches every other figure in the paper).
MODELS_ORDER = ["gru", "rgcn", "rgsage_mean", "rgsage_max", "rgatv2"]
MODEL_LABELS = {"gru": "GRU", "rgcn": "RGCN", "rgsage_mean": "RGSAGE-Mean",
                "rgsage_max": "RGSAGE-Max", "rgatv2": "RGATv2",
                "sdr_stgnn": "SDR-STGNN (ours)"}
MODEL_COLOR = {"gru": "#2a78d6", "rgcn": "#eb6834", "rgsage_mean": "#1baf7a",
               "rgsage_max": "#eda100", "rgatv2": "#e87ba4", "sdr_stgnn": "#111111"}
INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"

# Backbone-ablation variant styling.
V_STANDARD = "#eda100"    # standard RGSAGE-Max
V_AWARE = "#1b7a9b"       # mask-aware + dropout-aware (ungated)
V_SDR = "#111111"         # SDR-STGNN (gated)

_RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "font.family": "sans-serif",
}


def _style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED); ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _load(path: str | None, keep_model: str | None = None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warn] not found, skipping: {p}")
        return None
    df = pd.read_csv(p)
    if keep_model:
        df = df[df["model"] == keep_model].copy()
    return df


# numpy 2.x renamed trapz → trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _audc(k: np.ndarray, ret: np.ndarray) -> float:
    """Normalised area under the retention curve over the observed k-range (0..1)."""
    order = np.argsort(k)
    k, ret = k[order], ret[order]
    return float(_trapezoid(ret, k) / (k.max() - k.min())) if k.max() > k.min() else float(ret.mean())


# ---------------------------------------------------------------------------
def fig_degradation_all(baselines, sdr, out_path, k_levels):
    plt.rcParams.update(_RC)
    # Short canvas: the paper places this at ~0.85\linewidth in a single-column
    # layout, so a low aspect ratio saves page height without shrinking the text.
    fig, ax = plt.subplots(figsize=(6.5, 3.1)); _style(ax)
    for m in [x for x in MODELS_ORDER if baselines is not None and x in baselines["model"].values]:
        g = baselines[baselines["model"] == m].sort_values("k")
        ax.errorbar(g["k"], g["f1_mean"], yerr=g["f1_std"], marker="o", capsize=4,
                    linewidth=2, markersize=6, color=MODEL_COLOR[m],
                    label=MODEL_LABELS[m], zorder=3)
    if sdr is not None and not sdr.empty:
        g = sdr[sdr["model"] == "sdr_stgnn"].sort_values("k")
        ax.errorbar(g["k"], g["f1_mean"], yerr=g["f1_std"], marker="D", capsize=4,
                    linewidth=3, markersize=8, color=MODEL_COLOR["sdr_stgnn"],
                    label=MODEL_LABELS["sdr_stgnn"], zorder=5)
    ax.set_xticks(k_levels)
    ax.set_xlabel("μPMUs dropped at inference (k of 25)")
    ax.set_ylabel("Soft-voted graph macro-F1 (test)")
    # No in-figure title: the LaTeX caption carries it (and saves vertical space).
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=400); plt.close(fig)


def _backbone_variants(std_bb, aware_bb, sdr):
    """Return ordered list of (label, color, df) for the RGSAGE-Max ablation."""
    out = []
    if std_bb is not None and not std_bb.empty:
        out.append(("RGSAGE-Max (standard)", V_STANDARD, std_bb.sort_values("k")))
    if aware_bb is not None and not aware_bb.empty:
        out.append(("+ mask-aware & dropout-aware (ungated)", V_AWARE, aware_bb.sort_values("k")))
    if sdr is not None and not sdr.empty:
        out.append(("SDR-STGNN (gated + dropout-aware, ours)", V_SDR,
                    sdr[sdr["model"] == "sdr_stgnn"].sort_values("k")))
    return out


def fig_ablation_backbone(variants, out_path, k_levels):
    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(6.5, 4.0)); _style(ax)
    styles = ["--", "-", "-"]; widths = [2.0, 2.2, 3.2]; markers = ["o", "s", "D"]
    for (label, color, g), ls, lw, mk in zip(variants, styles, widths, markers):
        ax.errorbar(g["k"], g["f1_mean"], yerr=g["f1_std"], marker=mk, linestyle=ls,
                    linewidth=lw, markersize=7, capsize=4, color=color, label=label,
                    zorder=4)
    ax.set_xticks(k_levels)
    ax.set_xlabel("μPMUs dropped at inference (k of 25)")
    ax.set_ylabel("Soft-voted graph macro-F1 (test), mean ± std")
    ax.set_title("Ablation on the RGSAGE-Max backbone: dropout-aware training drives robustness")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=400); plt.close(fig)


def fig_retention(variants, out_path, k_levels):
    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(6.5, 4.0)); _style(ax)
    styles = ["--", "-", "-"]; widths = [2.0, 2.2, 3.2]; markers = ["o", "s", "D"]
    for (label, color, g), ls, lw, mk in zip(variants, styles, widths, markers):
        f0 = g[g["k"] == 0]["f1_mean"].iloc[0]
        ret = 100.0 * g["f1_mean"] / f0
        ax.plot(g["k"], ret, marker=mk, linestyle=ls, linewidth=lw, markersize=7,
                color=color, label=label, zorder=4)
    ax.axhline(100, color=INK_MUTED, lw=1, linestyle=":", zorder=1)
    ax.set_xticks(k_levels)
    ax.set_xlabel("μPMUs dropped at inference (k of 25)")
    ax.set_ylabel("Macro-F1 retained vs k=0  (%)")
    ax.set_title("Accuracy retention under sensor dropout (higher = more robust)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=400); plt.close(fig)


def fig_audc_bar(variants, out_path):
    plt.rcParams.update(_RC)
    labels, colors, vals = [], [], []
    for label, color, g in variants:
        f0 = g[g["k"] == 0]["f1_mean"].iloc[0]
        ret = (g["f1_mean"] / f0).to_numpy()
        vals.append(_audc(g["k"].to_numpy(), ret))
        labels.append(label.replace(" (", "\n(")); colors.append(color)
    fig, ax = plt.subplots(figsize=(6.5, 4.0)); _style(ax)
    bars = ax.bar(range(len(vals)), vals, color=colors, zorder=3, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center",
                va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Normalised AUDC (area under retention curve)")
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    ax.set_title("Robustness score: area under the retention curve")
    fig.tight_layout(); fig.savefig(out_path, dpi=400); plt.close(fig)


def fig_heatmap(baselines, sdr, out_path, k_levels):
    plt.rcParams.update(_RC)
    rows, labels = [], []
    for m in [x for x in MODELS_ORDER if baselines is not None and x in baselines["model"].values]:
        g = baselines[baselines["model"] == m].set_index("k")["f1_mean"]
        rows.append([g.get(k, np.nan) for k in k_levels]); labels.append(MODEL_LABELS[m])
    if sdr is not None and not sdr.empty:
        g = sdr[sdr["model"] == "sdr_stgnn"].set_index("k")["f1_mean"]
        rows.append([g.get(k, np.nan) for k in k_levels]); labels.append(MODEL_LABELS["sdr_stgnn"])
    mat = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(rows) + 1.6))
    im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(k_levels))); ax.set_xticklabels([f"k={k}" for k in k_levels])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, color=INK if mat[i, j] > 0.4 else "#555")
    ax.set_title("Macro-F1 under sensor dropout")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="macro-F1")
    fig.tight_layout(); fig.savefig(out_path, dpi=400); plt.close(fig)


def write_summary(variants, out_path, k_levels):
    recs = []
    for label, _, g in variants:
        f0 = g[g["k"] == 0]["f1_mean"].iloc[0]
        ret = (g["f1_mean"] / f0).to_numpy()
        audc = _audc(g["k"].to_numpy(), ret)
        row = {"variant": label, "audc": round(audc, 4)}
        for k in k_levels:
            sel = g[g["k"] == k]
            row[f"f1@k={k}"] = round(float(sel["f1_mean"].iloc[0]), 4) if len(sel) else np.nan
            row[f"ret@k={k}"] = round(float(sel["f1_mean"].iloc[0] / f0), 4) if len(sel) else np.nan
        recs.append(row)
    pd.DataFrame(recs).to_csv(out_path, index=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines", default="results/dropout_study/degradation_curves.csv")
    p.add_argument("--aware", default="results/dropout_study_aware/degradation_curves.csv")
    p.add_argument("--sdr", default="results/dropout_study_sdr/degradation_curves.csv")
    p.add_argument("--out-dir", default="paper/figures")
    args = p.parse_args()

    baselines = _load(args.baselines)
    aware_bb = _load(args.aware, keep_model="rgsage_max")
    sdr = _load(args.sdr, keep_model="sdr_stgnn")
    std_bb = baselines[baselines["model"] == "rgsage_max"].copy() if baselines is not None else None

    if baselines is None and sdr is None:
        print("No input CSVs found — run the dropout studies first."); return 1

    # k-levels from whatever we have
    src = baselines if baselines is not None else sdr
    k_levels = sorted(src["k"].unique().tolist())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    variants = _backbone_variants(std_bb, aware_bb, sdr)

    if baselines is not None or sdr is not None:
        fig_degradation_all(baselines, sdr, out / "sdr_degradation_all.png", k_levels)
    if len(variants) >= 2:
        fig_ablation_backbone(variants, out / "sdr_ablation_backbone.png", k_levels)
        fig_retention(variants, out / "sdr_retention.png", k_levels)
        fig_audc_bar(variants, out / "sdr_audc_bar.png")
        write_summary(variants, out / "sdr_robustness_summary.csv", k_levels)
    fig_heatmap(baselines, sdr, out / "sdr_heatmap.png", k_levels)

    print(f"Wrote SDR result figures to {out}/ (k-levels={k_levels}, "
          f"{len(variants)} backbone variants).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
