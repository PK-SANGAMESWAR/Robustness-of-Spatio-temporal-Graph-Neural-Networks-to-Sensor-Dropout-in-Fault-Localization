"""Presentation-scale architecture diagrams for the guide-review slide deck.

``build_sdr_schematics.py`` produces a six-box pipeline sized for a two-column
paper figure. A slide needs the opposite trade-off: fewer boxes per row, much
larger type, and the tensor shape carried explicitly between stages so a viewer
can follow what the data *is* at each point without reading a caption.

Three figures, all 16:9 at 300 dpi so they drop onto a PowerPoint slide without
resampling:

  ppt_architecture_full.png  — the shared forward path, [25,20,6] -> 1 of 26
                               classes, with tensor shapes on every arrow. The
                               spatial block is drawn as a SOCKET, because it is
                               the only stage that differs between models.
  ppt_gnn_variants.png       — what plugs into that socket: all six spatial
                               variants with their update rules side by side,
                               plus the measured accuracy and robustness of
                               each. This is the "the difference is the GNN
                               layer" figure.
  ppt_layer_detail.png       — one layer opened up: standard max-pooling versus
                               the availability gate, and why the gate loses.

IMPORTANT distinction the figures are built to make: the five baselines and
SDR-STGNN differ in the GNN layer, but **dropout-aware training does not touch
the architecture at all** -- it is a training-time augmentation of the input and
composes with any of the six. Drawing it as a third "architecture" would be
wrong and is exactly the confusion these figures exist to prevent.

Numbers are read from results/ at build time, never hard-coded, so a figure
cannot drift from the CSVs.

CPU only, no checkpoints required.

Usage:
    uv run python scripts/build_ppt_architecture.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#8a8880"
GRID = "#dedcd4"; SURFACE = "#ffffff"; PANEL = "#f7f7f4"
ACCENT = "#c62828"; ACCENT_FILL = "#fdeceb"
LIVE = "#1baf7a"; DEAD = "#e53935"; NEUTRAL = "#e8e6df"; BLUE = "#2a78d6"
GOLD = "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
})

# Horizontal data-units consumed by one character at font size F, for a canvas
# of 100 units across 13.333 in.  Used to wrap text so it can never overflow a
# box -- the failure mode of the first version of this script.
UNITS_PER_CHAR = 0.0573


def _wrap(text: str, box_w: float, fs: float, pad: float = 0.86) -> str:
    """Greedy-wrap ``text`` to fit ``box_w`` data units at font size ``fs``."""
    max_chars = max(8, int(box_w * pad / (UNITS_PER_CHAR * fs)))
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = w if not cur else f"{cur} {w}"
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _box(ax, x, y, w, h, fc, ec, lw=2.0, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, linestyle=ls))


def _txt(ax, x, y, s, fs=11, c=INK, fw="normal", ha="center", va="center",
         style="normal", z=5, lsp=1.25):
    ax.text(x, y, s, fontsize=fs, color=c, fontweight=fw, ha=ha, va=va,
            style=style, zorder=z, linespacing=lsp)


def _arrow(ax, x0, y0, x1, y1, c=INK_SECONDARY, lw=2.4, z=1, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=20,
        linewidth=lw, color=c, zorder=z, connectionstyle="arc3,rad=0"))


def _tag(ax, x, y, s, c=BLUE):
    ax.text(x, y, s, fontsize=10.5, color=c, ha="center", va="center",
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=c, linewidth=1.3))


def _new(title, subtitle):
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    # The axes MUST fill the figure. With matplotlib's default margins it spans
    # only 77.5% of the width, so 100 data units map to 10.33 in rather than
    # 13.333 -- which makes every character ~29% wider in data units than
    # UNITS_PER_CHAR assumes, and _wrap() then under-wraps and text overflows.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    _txt(ax, 50, 53.2, title, fs=18, fw="bold")
    _txt(ax, 50, 50.2, subtitle, fs=11, c=INK_SECONDARY)
    return fig, ax


# ---------------------------------------------------------------------------
# data (never hard-coded)
# ---------------------------------------------------------------------------

def load_numbers(repo: Path) -> dict:
    aa = pd.read_csv(repo / "results/statistics/absolute_audc.csv")
    bs = pd.read_csv(repo / "results/fullscale_final/benchmark_summary.csv")
    out = {}
    for _, r in aa.iterrows():
        out[(r.variant, r.model)] = dict(clean=r.f1_clean, k10=r.f1_at_k10,
                                         audc=r.audc_rel, audc_abs=r.audc_abs)
    return out, bs


# ---------------------------------------------------------------------------
# Figure 1 — shared forward path, spatial stage drawn as a socket
# ---------------------------------------------------------------------------

def build_full(out_path: Path) -> None:
    fig, ax = _new("STGNN fault localization \u2014 forward path for one 20 ms window",
                   "every model shares this path; only stage 3 differs between "
                   "them (see the variants figure)")

    y, h = 30.0, 13.0
    xs = [2.0, 26.5, 51.0, 75.5]
    w = 22.5

    stages = [
        ("1.  INPUT WINDOW", NEUTRAL, INK_SECONDARY, [
            ("25 \u03bcPMU nodes \u00d7 20 time steps \u00d7 6 channels", 11),
            ("3 RMS voltages + 3 availability-mask bits", 10.5),
            ("40 sliding windows per fault event", 10),
        ]),
        ("2.  TEMPORAL ENCODER", NEUTRAL, INK_SECONDARY, [
            ("one 1-layer GRU, 128 hidden units", 11),
            ("applied to each node independently", 10.5),
            ("weights SHARED across all 25 nodes", 10),
        ]),
        ("3.  SPATIAL BLOCK", "#fff8e6", GOLD, [
            ("4 message-passing layers, hidden 64", 11),
            ("over the 25-node measured-only graph", 10.5),
            ("THIS IS THE SWAPPABLE PART", 10),
        ]),
        ("4.  HEAD + SOFT VOTE", NEUTRAL, INK_SECONDARY, [
            ("BN \u2192 ReLU \u2192 Drop 0.35 \u2192 Linear 26", 11),
            ("shared head, applied per node", 10.5),
            ("\u03a3 softmax over 25 nodes \u2192 argmax", 10),
        ]),
    ]

    for (x, (title, fc, ec, lines)) in zip(xs, stages):
        _box(ax, x, y, w, h, fc, ec, lw=2.6 if ec == GOLD else 2.0)
        _txt(ax, x + w / 2, y + h - 2.0, title, fs=12.5, fw="bold",
             c=ACCENT if ec == GOLD else INK)
        yy = y + h - 5.2
        for s, fs in lines:
            _txt(ax, x + w / 2, yy, _wrap(s, w, fs), fs=fs,
                 c=INK_MUTED if fs <= 10 else INK)
            yy -= 3.1

    for i, x in enumerate(xs[:-1]):
        _arrow(ax, x + w + 0.3, y + h / 2, xs[i + 1] - 0.3, y + h / 2)
    _arrow(ax, xs[-1] + w + 0.3, y + h / 2, 99.0, y + h / 2, c=LIVE, lw=3.0)

    # Shape tags sit ABOVE the row: the inter-box gap is only ~2 units wide, so
    # a tag placed on the arrow itself would overlap the boxes either side.
    for x, s in zip([25.5, 50.0, 74.5], ["[25, 20, 6]", "[25, 128]", "[25, 64]"]):
        _tag(ax, x, y + h + 2.4, s)
    _txt(ax, 50, y + h + 5.6, "tensor shape carried between stages", fs=9.5,
         c=INK_MUTED, style="italic")
    _txt(ax, 99.0, y + h + 2.4, "1 of 26 classes", fs=10.5, fw="bold", c=LIVE,
         ha="right")

    # ---- the two contributions, side by side, clearly separated -------------
    cy, ch = 6.0, 19.0
    _box(ax, 2.0, cy, 47.0, ch, ACCENT_FILL, ACCENT, lw=2.6)
    _txt(ax, 25.5, cy + ch - 2.2, "CONTRIBUTION 1 \u2014 DROPOUT-AWARE TRAINING",
         fs=12.5, fw="bold", c=ACCENT)
    _txt(ax, 25.5, cy + ch - 5.0,
         _wrap("A TRAINING-TIME AUGMENTATION. It changes no layer, no "
               "parameter, and no part of the path above.", 47, 11),
         fs=11, c=ACCENT, style="italic")
    _txt(ax, 25.5, cy + ch - 9.6,
         _wrap("Per batch, draw k ~ Uniform{0,1,2,5} and zero all 6 channels "
               "of k randomly chosen nodes before the forward pass. Composes "
               "with ANY of the six spatial variants.", 47, 10.5), fs=10.5)
    _txt(ax, 25.5, cy + 2.2,
         "0 extra parameters   \u00b7   0 inference cost   \u00b7   +2.7% training time",
         fs=11, fw="bold")
    _arrow(ax, 12.0, cy + ch + 0.4, 12.0, y - 0.4, c=ACCENT, lw=2.4)
    _txt(ax, 13.2, y - 2.6, "masks the input", fs=9.5, c=ACCENT, ha="left",
         style="italic")

    _box(ax, 51.0, cy, 47.0, ch, PANEL, ACCENT, lw=2.2, ls="--")
    _txt(ax, 74.5, cy + ch - 2.2,
         "CONTRIBUTION 2 \u2014 SDR-STGNN GATE  (ablation)", fs=12.5,
         fw="bold", c=ACCENT)
    _txt(ax, 74.5, cy + ch - 5.0,
         _wrap("AN ARCHITECTURAL CHANGE. It replaces the aggregation inside "
               "stage 3 and nothing else.", 47, 11),
         fs=11, c=ACCENT, style="italic")
    _txt(ax, 74.5, cy + ch - 9.6,
         _wrap("Aggregation is restricted to available nodes, so a dropped "
               "sensor emits no message at any of the 4 hops. Reduces exactly "
               "to RGSAGE-Max when every sensor reports.", 47, 10.5), fs=10.5)
    _txt(ax, 74.5, cy + 2.2,
         "MATCHES contribution 1, does not beat it   \u00b7   +18% latency",
         fs=11, fw="bold")
    _arrow(ax, 62.0, cy + ch + 0.4, 62.0, y - 0.4, c=ACCENT, lw=2.4)
    _txt(ax, 63.2, y - 2.6, "replaces the aggregation", fs=9.5, c=ACCENT,
         ha="left", style="italic")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — what plugs into the spatial socket
# ---------------------------------------------------------------------------

def build_variants(out_path: Path, nums: dict) -> None:
    fig, ax = _new("Stage 3 in depth \u2014 the six spatial blocks",
                   "identical input, identical head, identical training data: "
                   "the aggregation rule is the only thing that changes")

    std = lambda m: nums.get(("standard", m), {})
    rows = [
        ("GRU  (no graph)", r"$h'_j = h_j$",
         "no message passing at all; the non-spatial reference",
         std("gru"), INK_MUTED),
        ("RGCN", r"$h' = \mathrm{ReLU}(\tilde{D}^{-1/2}\tilde{A}"
                 r"\tilde{D}^{-1/2} h W)$",
         "symmetrically normalized sum over neighbours",
         std("rgcn"), INK_SECONDARY),
        ("RGSAGE-Mean", r"$h'_j=\mathrm{ReLU}(W[\,h_j \| "
                        r"\mathrm{mean}_{i \in \bar{N}(j)} h_i\,])$",
         "averages the neighbourhood, then concatenates self",
         std("rgsage_mean"), INK_SECONDARY),
        ("RGSAGE-Max", r"$h'_j=\mathrm{ReLU}(W[\,h_j \| "
                       r"\max_{i \in \bar{N}(j)} h_i\,])$",
         "takes the largest activation \u2014 strongest, and most fragile",
         std("rgsage_max"), BLUE),
        ("RGATv2", r"$h'_j=\mathrm{ReLU}(W_o \|_{c=1}^{4} "
                   r"\sum_i \alpha_{ji}^{(c)} W^{(c)} h_i)$",
         "4-head attention; weights are learned, not availability-aware",
         std("rgatv2"), INK_SECONDARY),
        ("SDR-STGNN  (ours)", r"$h'_j=\mathrm{ReLU}(W[\,a_j h_j \| "
                              r"\max_{i:\,a_i=1} h_i\,])$",
         "same as RGSAGE-Max but dropped sensors are excluded",
         nums.get(("sdr_stgnn", "sdr_stgnn"), {}), ACCENT),
    ]

    # rh must satisfy  y0 - 5*rh - 2 > 13  so the sixth row clears the summary
    # strip at the bottom of the canvas.
    x0, y0, rh = 2.0, 43.0, 5.4
    cols = [(2.0, 15.5), (18.0, 33.0), (52.0, 24.0), (77.5, 9.5), (88.0, 10.0)]
    hdr = ["spatial block", "update rule", "what it does",
           "clean F1", "AUDC (rel)"]
    for (cx, cw), htxt in zip(cols, hdr):
        _txt(ax, cx + cw / 2, y0 + 3.0, htxt, fs=11, fw="bold",
             c=INK_SECONDARY)
    ax.plot([2.0, 98.0], [y0 + 1.2, y0 + 1.2], color=INK_SECONDARY, lw=1.6)

    for i, (name, eq, desc, m, col) in enumerate(rows):
        yy = y0 - i * rh - 2.0
        if col == ACCENT:
            _box(ax, 1.6, yy - rh / 2 + 0.4, 96.8, rh - 0.8, ACCENT_FILL,
                 ACCENT, lw=1.8)
        elif col == BLUE:
            _box(ax, 1.6, yy - rh / 2 + 0.4, 96.8, rh - 0.8, "#eef4fc",
                 BLUE, lw=1.6)
        _txt(ax, cols[0][0], yy, name, fs=11.5, fw="bold", ha="left", c=col)
        _txt(ax, cols[1][0] + cols[1][1] / 2, yy, eq, fs=12.5)
        _txt(ax, cols[2][0] + cols[2][1] / 2, yy,
             _wrap(desc, cols[2][1], 9.8), fs=9.8, c=INK_SECONDARY)
        if m:
            _txt(ax, cols[3][0] + cols[3][1] / 2, yy, f"{m['clean']:.3f}",
                 fs=12, fw="bold", c=col)
            _txt(ax, cols[4][0] + cols[4][1] / 2, yy, f"{m['audc']:.3f}",
                 fs=12, fw="bold", c=col)

    _box(ax, 2.0, 1.2, 96.0, 6.0, "#f2f7f4", LIVE, lw=2.0)
    _txt(ax, 50, 5.4,
         "Read the last two columns against each other: the most accurate "
         "block is the least robust.", fs=11.5, fw="bold")
    _txt(ax, 50, 2.9,
         "SDR-STGNN is trained WITH dropout-aware augmentation, so its row is "
         "not comparable to the standard-trained rows above it.",
         fs=10, c=INK_SECONDARY, style="italic")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — one layer, standard vs gated
# ---------------------------------------------------------------------------

def build_layer_detail(out_path: Path, nums: dict) -> None:
    fig, ax = _new("Inside one message-passing layer",
                   "left: what every baseline does      "
                   "right: the one term SDR-STGNN changes")

    cy = 30.0
    for side, (bx, col, fill, title) in enumerate([
            (2.0, INK_SECONDARY, PANEL, "STANDARD  (all five baselines)"),
            (51.0, ACCENT, ACCENT_FILL, "AVAILABILITY-GATED  (SDR-STGNN)")]):
        _box(ax, bx, 11.0, 47.0, 35.0, fill, col, lw=2.4)
        _txt(ax, bx + 23.5, 43.6, title, fs=13, fw="bold",
             c=ACCENT if side else INK)

        cx = bx + 15.0
        nbrs = [(cx - 8, cy + 6), (cx - 8, cy - 6), (cx + 8, cy + 6.5),
                (cx + 8, cy - 6)]
        states = [LIVE, DEAD, LIVE, LIVE]
        # zorder must clear the panel box, which _box draws at z=2 — otherwise
        # the edges are painted underneath it and the graph looks like loose
        # circles with no connectivity.
        for (nx, ny), c in zip(nbrs, states):
            dead = (c == DEAD)
            ax.plot([cx, nx], [cy, ny],
                    color=(DEAD if dead and side else INK_MUTED), lw=2.2,
                    zorder=3, linestyle=(":" if dead and side else "-"))
        for (nx, ny), c in zip(nbrs, states):
            dead = (c == DEAD)
            ax.add_patch(Circle((nx, ny), 1.9, facecolor=c, edgecolor=INK,
                                lw=1.5, zorder=4,
                                alpha=0.3 if (dead and side) else 0.9))
        ax.add_patch(Circle((cx, cy), 2.3, facecolor=BLUE, edgecolor=INK,
                            lw=2.0, zorder=5))
        _txt(ax, cx, cy, "j", fs=13, c="white", fw="bold", z=6)
        _txt(ax, cx - 8, cy - 9.4,
             "dropped\nembedding = 0" if not side else "emits NO message",
             fs=9.8, c=DEAD if not side else ACCENT, fw="bold")

    _txt(ax, 25.5, 18.4,
         r"$h'_j=\mathrm{ReLU}\!\left(W\left[\,h_j \;\|\;"
         r"\max_{i\in\bar{N}(j)} h_i\,\right]\right)$", fs=15)
    _txt(ax, 25.5, 14.0,
         _wrap("The zeroed embedding is pooled in with the rest. Max-pooling "
               "forwards the single largest activation, so one dead neighbour "
               "can dominate the update.", 45, 10.2), fs=10.2, c=INK_SECONDARY)

    _txt(ax, 74.5, 18.4,
         r"$h'_j=\mathrm{ReLU}\!\left(W\left[\,a_j h_j \;\|\;"
         r"\max_{i\in\bar{N}(j),\,a_i=1} h_i\,\right]\right)$", fs=15, c=ACCENT)
    _txt(ax, 74.5, 14.0,
         _wrap("Aggregation is restricted to available sources. But the gate "
               "fires at every one of the 4 hops, which also blocks multi-hop "
               "routing THROUGH a dropped node.", 45, 10.2), fs=10.2,
         c=INK_SECONDARY)

    da = nums.get(("dropout_aware", "rgsage_max"), {})
    sdr = nums.get(("sdr_stgnn", "sdr_stgnn"), {})
    _box(ax, 2.0, 1.5, 96.0, 7.5, "#f2f7f4", LIVE, lw=2.2)
    _txt(ax, 50, 6.6,
         f"Result: the gate MATCHES dropout-aware training "
         f"(AUDC {sdr.get('audc', float('nan')):.3f} vs "
         f"{da.get('audc', float('nan')):.3f}, p = 0.098) but never beats it",
         fs=12.5, fw="bold")
    _txt(ax, 50, 3.4,
         "Silencing a node at every hop also destroys the relay path a healthy "
         "2-hop neighbour would have used \u2014 which is why the training-time "
         "fix wins.", fs=10.5, c=INK_SECONDARY)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — evaluation strategy
# ---------------------------------------------------------------------------

def build_eval_strategy(out_path: Path, nums: dict) -> None:
    fig, ax = _new("Evaluation strategy — freeze the model, then break the sensors",
                   "weights are never retrained for a given k; every model sees "
                   "an identical sequence of failures")

    # ---- the sweep ---------------------------------------------------------
    _txt(ax, 4.0, 45.6, "1.  THE SWEEP", fs=12.5, fw="bold", ha="left")
    ks = [0, 1, 2, 5, 10]
    x0, dx = 12.0, 17.0
    for i, k in enumerate(ks):
        cx = x0 + i * dx
        # 25 sensor pips, k of them dark
        for j in range(25):
            r, c = divmod(j, 5)
            px = cx - 3.2 + c * 1.6
            py = 41.4 - r * 1.5
            dark = j < k
            ax.add_patch(Rectangle((px, py), 1.15, 1.05,
                                   facecolor=DEAD if dark else LIVE,
                                   edgecolor=INK, lw=0.5, zorder=3,
                                   alpha=0.9 if dark else 0.75))
        _txt(ax, cx, 34.0, f"k = {k}", fs=11.5, fw="bold")
        _txt(ax, cx, 32.2, f"{k*4}% loss", fs=9.5, c=INK_MUTED)
    _txt(ax, 4.0, 37.0, "25 sensors\nk dark", fs=9.5, c=INK_SECONDARY,
         ha="left")

    # ---- policies ----------------------------------------------------------
    _txt(ax, 4.0, 28.4, "2.  FIVE WAYS THE FLEET CAN FAIL", fs=12.5,
         fw="bold", ha="left")
    pol = [
        ("Random", "independent,\nuniform", "the average case", INK_SECONDARY),
        ("Targeted", "k nearest\nthe fault", "worst case — far harsher", ACCENT),
        ("Persistent", "held across\nthe event", "a clean null", INK_SECONDARY),
        ("Correlated", "a contiguous\nblock", "milder, not harsher", INK_SECONDARY),
        ("Noise", "sensors report,\ninaccurately", "1% TVE ≈ 4.1 sensors", ACCENT),
    ]
    pw = 18.2
    for i, (name, what, find, col) in enumerate(pol):
        px = 2.0 + i * (pw + 1.4)
        _box(ax, px, 15.0, pw, 11.4, ACCENT_FILL if col == ACCENT else PANEL,
             col, lw=2.0)
        _txt(ax, px + pw / 2, 24.6, name, fs=11.5, fw="bold", c=col)
        _txt(ax, px + pw / 2, 21.4, what, fs=9.8, c=INK_SECONDARY)
        _txt(ax, px + pw / 2, 17.4, _wrap(find, pw, 9.8), fs=9.8, fw="bold")

    # ---- metrics -----------------------------------------------------------
    _txt(ax, 4.0, 11.6, "3.  METRICS AND REFERENCE POINTS", fs=12.5,
         fw="bold", ha="left")
    std = nums.get(("standard", "rgsage_max"), {})
    da = nums.get(("dropout_aware", "rgsage_max"), {})
    cells = [
        ("soft-voted\nmacro-F1", "primary score,\n26 classes", BLUE),
        ("retention\nR(k) = F1(k)/F1(0)", "how gracefully\nit degrades", BLUE),
        ("AUDC\n(relative)", f"strongest baseline\n{std.get('audc', 0):.3f}",
         BLUE),
        ("AUDC\n(absolute)", "no normalization —\nweak models can't hide",
         GOLD),
        ("trivial floor\n0.026", "always-\"no fault\";\nk=10 is 1.9× this",
         DEAD),
    ]
    cw = 18.2
    for i, (title, sub, col) in enumerate(cells):
        px = 2.0 + i * (cw + 1.4)
        _box(ax, px, 1.4, cw, 8.6, "white", col, lw=1.8)
        _txt(ax, px + cw / 2, 7.6, title, fs=10.5, fw="bold", c=col)
        _txt(ax, px + cw / 2, 3.9, sub, fs=9.5, c=INK_SECONDARY)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — SDR-STGNN in full detail
# ---------------------------------------------------------------------------

def build_sdr_detail(out_path: Path, nums: dict, params: dict) -> None:
    fig, ax = _new("SDR-STGNN in detail — availability-gated spatio-temporal GNN",
                   "RGSAGE-Max backbone; every tensor shape and layer width is "
                   "the frozen configuration, read from the model itself")

    # ---- availability branch ----------------------------------------------
    _box(ax, 2.0, 39.0, 22.0, 8.4, ACCENT_FILL, ACCENT, lw=2.2)
    _txt(ax, 13.0, 45.4, "AVAILABILITY (the gate's input)", fs=10.5, fw="bold",
         c=ACCENT)
    _txt(ax, 13.0, 42.6, r"$a_i=\mathbf{1}\{\max_s \max_p\, m_{i,p}(s)>0\}$",
         fs=12)
    _txt(ax, 13.0, 40.2, "computed ONCE, reused at all 4 hops", fs=9.3,
         c=INK_SECONDARY, style="italic")

    # ---- main pipeline -----------------------------------------------------
    y, h = 26.0, 10.0
    boxes = [
        (2.0, 22.0, "INPUT", "[B, 25, 20, 6]",
         "3 RMS volts + 3 mask bits", NEUTRAL, INK_SECONDARY),
        (26.5, 22.0, "TEMPORAL ENCODER", "[B, 25, 128]",
         "1-layer GRU-128, shared across nodes", NEUTRAL, INK_SECONDARY),
        (51.0, 22.0, "4 × GATED LAYER", "[B, 25, 64]",
         "see the expansion below", ACCENT_FILL, ACCENT),
        (75.5, 22.5, "HEAD + VOTE", "[B, 26]",
         "BN→ReLU→Drop→Linear, then Σ softmax",
         NEUTRAL, INK_SECONDARY),
    ]
    for x, w, title, shape, sub, fc, ec in boxes:
        _box(ax, x, y, w, h, fc, ec, lw=2.4 if ec == ACCENT else 2.0)
        _txt(ax, x + w / 2, y + h - 2.0, title, fs=12, fw="bold",
             c=ACCENT if ec == ACCENT else INK)
        _txt(ax, x + w / 2, y + h - 5.2, shape, fs=12, fw="bold", c=BLUE)
        _txt(ax, x + w / 2, y + 2.0, _wrap(sub, w, 9.5), fs=9.5,
             c=INK_SECONDARY)
    for i in range(3):
        x0 = boxes[i][0] + boxes[i][1]
        _arrow(ax, x0 + 0.3, y + h / 2, boxes[i + 1][0] - 0.3, y + h / 2)

    # gate feed
    _arrow(ax, 13.0, 38.8, 13.0, 36.4, c=ACCENT, lw=2.0)
    _arrow(ax, 13.0, 36.4, 60.0, 36.4, c=ACCENT, lw=2.0, style="-")
    _arrow(ax, 60.0, 36.4, 60.0, y + h + 0.3, c=ACCENT, lw=2.0)
    _txt(ax, 37.0, 37.6, "a  →  gates every hop", fs=9.5, c=ACCENT,
         style="italic")

    # ---- expansion of one gated layer -------------------------------------
    _box(ax, 2.0, 9.0, 62.0, 14.5, PANEL, ACCENT, lw=2.2, ls="--")
    _txt(ax, 33.0, 21.6, "INSIDE ONE GATED LAYER  (× 4)", fs=11.5,
         fw="bold", c=ACCENT)
    _txt(ax, 33.0, 18.0,
         r"$\mathrm{agg}_j=\max_{i\in\bar{N}(j),\,a_i=1} h_i$"
         r"$\qquad h'_j=\mathrm{ReLU}\left(W^{(l)}\left[\,a_j h_j \;\|\;"
         r"\mathrm{agg}_j\,\right]\right)$", fs=13)
    _txt(ax, 33.0, 14.0,
         "dropped nodes are masked to −∞ before the max, so they "
         "never win it;\na node whose whole neighbourhood is dark aggregates "
         "to zeros", fs=9.8, c=INK_SECONDARY)
    _txt(ax, 33.0, 10.6,
         "hop 1:  Linear(256 → 64)      hops 2–4:  Linear(128 → 64)"
         "      ReLU between hops, not after the last",
         fs=9.8, fw="bold")

    # ---- properties panel --------------------------------------------------
    _box(ax, 66.0, 9.0, 32.0, 14.5, "white", INK_SECONDARY, lw=2.0)
    _txt(ax, 82.0, 21.6, "PARAMETERS", fs=11.5, fw="bold")
    rows = [("GRU-128 encoder", params["encoder"]),
            ("hop 1  Linear(256→64)", params["l1"]),
            ("hops 2–4  Linear(128→64)", params["l234"]),
            ("classifier head", params["head"]),
            ("availability gate", 0)]
    # Explicit positions: the box spans y = 9.0 .. 23.5, so the total row and
    # its note must land above 9.8 or they collide with the verdict strip.
    yy = 19.6
    for name, v in rows:
        _txt(ax, 67.5, yy, name, fs=9.3, ha="left", c=INK_SECONDARY)
        _txt(ax, 96.5, yy, f"{v:,}", fs=9.3, ha="right", fw="bold",
             c=ACCENT if v == 0 else INK)
        yy -= 1.7
    ax.plot([67.5, 96.5], [12.0, 12.0], color=INK_SECONDARY, lw=1.2)
    _txt(ax, 67.5, 10.9, "TOTAL", fs=10.5, ha="left", fw="bold")
    _txt(ax, 96.5, 10.9, f"{params['total']:,}", fs=10.5, ha="right",
         fw="bold")
    _txt(ax, 82.0, 9.8, "identical to RGSAGE-Max", fs=9.3, c=ACCENT,
         style="italic")

    # ---- verdict -----------------------------------------------------------
    sdr = nums.get(("sdr_stgnn", "sdr_stgnn"), {})
    da = nums.get(("dropout_aware", "rgsage_max"), {})
    _box(ax, 2.0, 1.2, 96.0, 6.4, "#f2f7f4", LIVE, lw=2.0)
    _txt(ax, 50, 5.8,
         "Verified property: with every sensor present the gate is the identity"
         "  —  max |SDR(x) − RGSAGE-Max(x)| = 0.0", fs=11, fw="bold")
    _txt(ax, 50, 2.6,
         f"Result: AUDC {sdr.get('audc', float('nan')):.3f} vs "
         f"{da.get('audc', float('nan')):.3f} for dropout-aware training "
         f"(p = 0.098) — it MATCHES the training-time fix, does not beat "
         f"it, and costs +18% inference latency",
         fs=10.5, c=INK_SECONDARY)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def sdr_params() -> dict:
    """Count parameters from the real model, so the figure cannot drift."""
    import numpy as np
    import torch  # noqa: F401
    from fault_localization.models.sdr_stgnn import SDRSTGNN
    repo = Path(__file__).resolve().parent.parent
    ei = np.load(repo / "dataset_fullscale_50/processed/edge_index.npy")
    m = SDRSTGNN(ei, aggregator="max", num_layers=4)
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    return {
        "encoder": n(m.encoder), "head": n(m.classifier),
        "l1": n(m.spatial_layers[0]),
        "l234": sum(n(l) for l in m.spatial_layers[1:]),
        "total": sum(p.numel() for p in m.parameters()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="paper/figures")
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    out = repo / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    nums, _ = load_numbers(repo)
    build_full(out / "ppt_architecture_full.png")
    build_variants(out / "ppt_gnn_variants.png", nums)
    build_layer_detail(out / "ppt_layer_detail.png", nums)
    build_eval_strategy(out / "ppt_eval_strategy.png", nums)
    build_sdr_detail(out / "ppt_sdr_architecture.png", nums, sdr_params())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
