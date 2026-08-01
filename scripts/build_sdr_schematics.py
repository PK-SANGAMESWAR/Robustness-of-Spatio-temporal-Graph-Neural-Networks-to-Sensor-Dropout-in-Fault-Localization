"""Schematic (no-data) figures for the SDR-STGNN contribution.

Produces three conference figures that do NOT depend on experimental results, so
they can be built any time (CPU only):

  paper/figures/sdr_architecture.png   — SDR-STGNN forward pipeline, gating block
                                          highlighted (the novel component).
  paper/figures/sdr_mechanism.png      — standard vs availability-gated message
                                          passing on a small graph (why a dead
                                          sensor no longer corrupts neighbours).
  paper/figures/sdr_protocol.png       — the sensor-dropout degradation protocol
                                          (k μPMUs dropped, paired masks, sweep).

Palette matches scripts/build_fullscale_final_figures.py so every figure in the
paper shares one look.

Usage:
    uv run python scripts/build_sdr_schematics.py
    uv run python scripts/build_sdr_schematics.py --out-dir paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
ACCENT = "#c62828"       # the novel gated component
LIVE = "#1baf7a"         # available sensor
DEAD = "#e53935"         # dropped sensor
NEUTRAL = "#d9d7cf"      # standard block fill

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 11, "font.family": "sans-serif", "text.color": INK,
})


def _box(ax, x, y, w, h, text, fc, ec, tc=INK, lw=1.6, fs=10, fw="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=fw, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=INK_SECONDARY, lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, zorder=1,
        connectionstyle="arc3,rad=0"))


# ---------------------------------------------------------------------------
# Figure 1 — architecture pipeline
# ---------------------------------------------------------------------------
def build_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.5); ax.axis("off")

    y, h = 1.4, 1.15
    blocks = [
        (0.15, 1.9, "Input window\n[N=25, S=20, F=6]\n3 voltage + 3 avail. mask", NEUTRAL, INK_MUTED, "normal"),
        (2.25, 1.9, "Shared GRU\n(TemporalEncoder)\nper node → h ∈ ℝ¹²⁸", NEUTRAL, INK_MUTED, "normal"),
        (4.35, 2.1, "Availability-Gated\nMessage Passing × 4\n(masked-max aggregation)", "#fde8e6", ACCENT, "bold"),
        (6.65, 1.9, "Node classifier\nBN→ReLU→Drop→Lin-26", NEUTRAL, INK_MUTED, "normal"),
        (8.75, 1.9, "Soft-vote\nΣ node softmax", NEUTRAL, INK_MUTED, "normal"),
        (10.85, 1.9, "Fault location\n1 of 26 classes", "#e8f3ee", LIVE, "bold"),
    ]
    centers = []
    for x, w, text, fc, ec, fw in blocks:
        _box(ax, x, y, w, h, text, fc, ec, fs=9.5, fw=fw)
        centers.append((x, x + w))

    for i in range(len(centers) - 1):
        _arrow(ax, centers[i][1], y + h / 2, centers[i + 1][0], y + h / 2)

    # Availability side-branch feeding the gated block (centred under it).
    gate_cx = 4.35 + 2.1 / 2      # centre of the gated block
    ab_x, ab_w, ab_y, ab_h = gate_cx - 1.55, 3.1, 0.15, 0.78
    ax.add_patch(FancyBboxPatch(
        (ab_x, ab_y), ab_w, ab_h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=ACCENT, facecolor=SURFACE, zorder=2))
    ax.text(gate_cx, ab_y + ab_h / 2,
            "availability  a ∈ {0,1}ᴺ\n(1 if sensor's mask present)",
            ha="center", va="center", fontsize=8.5, color=ACCENT, fontweight="bold")
    _arrow(ax, gate_cx, ab_y + ab_h, gate_cx, y, color=ACCENT, lw=1.8)
    # mask tap from input, routed along the bottom to the box's left edge.
    tap_y = ab_y + ab_h / 2
    _arrow(ax, 1.1, y, 1.1, tap_y, color=ACCENT, lw=1.4, style="-")
    _arrow(ax, 1.1, tap_y, ab_x, tap_y, color=ACCENT, lw=1.4)

    ax.text(6.5, 3.28, "SDR-STGNN: availability-gated spatio-temporal GNN",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(gate_cx, 1.05, "dead sensors emit no messages",
            ha="center", va="center", fontsize=8, style="italic", color=ACCENT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — mechanism: standard vs gated aggregation
# ---------------------------------------------------------------------------
def _star(ax, cx, cy, gated: bool):
    r = 0.95
    neigh = [(cx - r, cy + 0.55, LIVE, False),
             (cx + r, cy + 0.55, LIVE, False),
             (cx - r, cy - 0.55, LIVE, False),
             (cx + r, cy - 0.55, DEAD, True)]   # bottom-right neighbour is dropped
    # edges / messages first (under nodes)
    for nx, ny, col, dead in neigh:
        if dead:
            style = dict(color=DEAD, lw=1.6,
                         linestyle=(0, (3, 3)) if gated else "solid")
            ax.add_patch(FancyArrowPatch((nx, ny), (cx, cy), arrowstyle="-|>",
                         mutation_scale=13, **style, zorder=1, alpha=0.5 if gated else 1.0))
        else:
            ax.add_patch(FancyArrowPatch((nx, ny), (cx, cy), arrowstyle="-|>",
                         mutation_scale=13, color=LIVE, lw=1.8, zorder=1))
    # nodes
    for nx, ny, col, dead in neigh:
        ax.add_patch(Circle((nx, ny), 0.22, facecolor=col, edgecolor=INK,
                            lw=1.2, zorder=3, alpha=0.35 if (dead and gated) else 1.0))
        if dead:
            ax.text(nx, ny, "✕", ha="center", va="center", fontsize=12,
                    color="white", fontweight="bold", zorder=4)
    # centre node j
    ax.add_patch(Circle((cx, cy), 0.28, facecolor="#2a78d6", edgecolor=INK,
                        lw=1.5, zorder=3))
    ax.text(cx, cy, "j", ha="center", va="center", fontsize=12, color="white",
            fontweight="bold", zorder=4)


def build_mechanism(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    titles = ["Standard aggregation\n(baseline: dead sensor corrupts node j)",
              "Availability-gated (SDR-STGNN)\n(dead sensor's message is blocked)"]
    for ax, title, gated in zip(axes, titles, (False, True)):
        ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.axis("off")
        _star(ax, 2.0, 2.0, gated=gated)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=8)

    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LIVE,
               markeredgecolor=INK, markersize=11, label="available μPMU"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DEAD,
               markeredgecolor=INK, markersize=11, label="dropped μPMU (✕)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2a78d6",
               markeredgecolor=INK, markersize=11, label="node j (aggregating)"),
        Line2D([0], [0], color=DEAD, lw=1.6, linestyle=(0, (3, 3)),
               label="blocked message"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — sensor-dropout evaluation protocol
# ---------------------------------------------------------------------------
def build_protocol(out_path: Path) -> None:
    import numpy as np
    rng = np.random.default_rng(7)
    ks = [0, 1, 2, 5, 10]
    n = 25
    # fixed random layout for a stylised 25-node measured-only graph
    pos = rng.uniform(0.15, 0.85, size=(n, 2))

    fig, axes = plt.subplots(1, len(ks), figsize=(13, 3.0))
    for ax, k in zip(axes, ks):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        dropped = set(rng.choice(n, size=k, replace=False).tolist()) if k else set()
        # light edges: connect each node to its nearest neighbour (stylised tree)
        order = np.argsort(pos[:, 0])
        for a, b in zip(order[:-1], order[1:]):
            ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                    color=GRID, lw=1.0, zorder=1)
        for i in range(n):
            dead = i in dropped
            ax.add_patch(Circle((pos[i, 0], pos[i, 1]), 0.035,
                                facecolor=DEAD if dead else LIVE,
                                edgecolor=INK, lw=0.6, zorder=2,
                                alpha=0.9 if not dead else 0.85))
        ax.set_title(f"k = {k}", fontsize=12, fontweight="bold", color=INK)

    fig.suptitle("Sensor-dropout protocol: k of 25 μPMUs dropped at inference "
                 "(paired masks, swept k ∈ {0,1,2,5,10}; F1 reported mean ± std over mask seeds)",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="paper/figures")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    build_architecture(out / "sdr_architecture.png")
    build_mechanism(out / "sdr_mechanism.png")
    build_protocol(out / "sdr_protocol.png")
    print(f"Wrote 3 schematic figures to {out}/:")
    print("  sdr_architecture.png  sdr_mechanism.png  sdr_protocol.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
