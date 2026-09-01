"""Slide figure: the built STGNN pipeline, and the three points where feeder
physics would enter it (STGNN + PIGNN).

This is the "direction we have started exploring" slide. It has to do two jobs at
once and must not blur them:

  BUILT      -- the forward path that produced every number in fault-loc.tex.
                Drawn solid. Carries its measured accuracy.
  EXPLORING  -- physics-informed additions. Drawn dashed, in a separate band,
                with NO accuracy attached to any of them, because none has been
                trained. A panel will ask "what does it score?" and the honest
                answer is "nothing yet, that is the work"; the figure is built so
                it can never imply otherwise.

The three injection points are not speculative hand-waving -- each one is backed
by a physics artifact that is already computed and cached in this repo:

  P1  impedance-weighted messages   <- graph/edge_attributes.py  (R, X, |Z| per edge)
  P2  sensitivity + geometry prior  <- graph/sensitivity.py      (sag signatures)
                                       graph/fault_distance.py   (hop distances)
  P3  physics residual loss         <- the same first-order sag model as P2

The left-hand panel carries the constraint that makes the whole direction
worth arguing for: the sag signatures are collinear in groups, so the 25 fault
classes collapse into a smaller number of electrical zones that NO
spatial-amplitude method can separate. That is a ceiling derived from physics,
not from any model, and it says exactly where the learned part has to do its
work.

Every count in the figure (zone count, matrix shapes, accuracies) is read from
the cached artifacts and CSVs at build time, so the figure cannot drift from
what is on disk.

Output: paper/figures/ppt_stgnn_pignn.png -- 16:9 at 300 dpi for the deck.

Usage:
    uv run python scripts/build_ppt_pignn_figure.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the deck's drawing vocabulary so this slide matches the others.
from build_ppt_architecture import (  # noqa: E402
    ACCENT,
    BLUE,
    GOLD,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PANEL,
    SURFACE,
    _arrow,
    _box,
    _new,
    _txt,
    _wrap,
    plt,
)

REPO = Path(__file__).resolve().parent.parent

GOLD_FILL = "#fdf4e0"
BLUE_FILL = "#eaf2fd"


def load_facts() -> dict:
    """Read every number the figure quotes from the artifacts on disk."""
    cfg = REPO / "fault_localization/data/ieee123/config"
    sens = np.load(cfg / "fault_sensitivity.npz")
    hop = np.load(cfg / "fault_hop_distance.npy")
    edge = np.load(REPO / "dataset_fullscale_50/processed/edge_attr.npy")
    summ = pd.read_csv(REPO / "paper/figures/sdr_robustness_summary.csv")

    std = summ[summ.variant.str.contains("standard")].iloc[0]
    aware = summ[summ.variant.str.contains("dropout-aware")].iloc[0]
    return dict(
        n_zones=int(sens["n_zones"]),
        n_classes=int(sens["zone_of_class"].shape[0]),
        sens_shape=tuple(sens["sensitivity"].shape),
        hop_shape=tuple(hop.shape),
        hop_max=int(hop.max()),
        edge_shape=tuple(edge.shape),
        f1_clean=float(std["f1@k=0"]),
        ret10_std=float(std["ret@k=10"]),
        ret10_aware=float(aware["ret@k=10"]),
    )


def build(out_path: Path, f: dict) -> None:
    fig, ax = _new(
        "STGNN + physics-informed GNN — the direction now being explored",
        "solid = built and measured   │   dashed = proposed, not yet trained, "
        "no accuracy claimed",
    )

    # ---------------------------------------------------------------- band A
    ax.add_patch(plt.Rectangle((1.5, 31.5), 97, 16.0, facecolor=PANEL,
                               edgecolor=GRID, lw=1.2, zorder=0))
    _txt(ax, 3.0, 45.6, "BUILT — the STGNN of fault-loc.tex", fs=11.5,
         c=BLUE, fw="bold", ha="left")
    _txt(ax, 97.0, 45.6,
         f"macro-F1 {f['f1_clean']:.3f}   │   retains {f['ret10_aware']:.0%} "
         f"at 40% sensor loss (dropout-aware)",
         fs=10, c=INK_SECONDARY, ha="right")

    stages = [
        ("μPMU window",
         "3 RMS phase voltages\n+ 3 availability channels"),
        ("GRU-128\ntemporal encoder",
         "shared across all\n25 sensor nodes"),
        ("4 × RGSAGE-Max\nspatial block",
         "measured-only graph\n25 nodes, 24 edges"),
        ("Node classifier",
         "BN → ReLU → drop 0.35\n→ Linear-26"),
        ("Soft vote",
         "sum the 25 node\nsoftmaxes → argmax"),
    ]
    xs = [2.5, 21.2, 39.9, 58.6, 77.3]
    w, h, y0 = 15.0, 9.6, 32.6
    centers = [x + w / 2 for x in xs]

    for (title, body), x in zip(stages, xs):
        _box(ax, x, y0, w, h, BLUE_FILL, BLUE, lw=2.0, z=2)
        _txt(ax, x + w / 2, y0 + h - 2.7, title, fs=10.6, fw="bold", c=INK)
        _txt(ax, x + w / 2, y0 + 2.9, body, fs=8.2, c=INK_SECONDARY, lsp=1.35)

    # Tensor shapes sit ABOVE the box row, not in the 3.7-unit gaps -- at any
    # readable size the label is wider than the gap and would sit on the boxes.
    shapes = ["[25, 20, 6]", "[25, 128]", "[25, 128]", "[25, 26]"]
    for i, s in enumerate(shapes):
        xa, xb = xs[i] + w, xs[i + 1]
        _arrow(ax, xa + 0.2, y0 + h / 2, xb - 0.2, y0 + h / 2, c=INK_SECONDARY, lw=2.2)
        _txt(ax, (xa + xb) / 2, y0 + h + 1.5, s, fs=7.6, c=INK_MUTED)

    _arrow(ax, xs[-1] + w + 0.2, y0 + h / 2, 96.6, y0 + h / 2, c=INK_SECONDARY, lw=2.2)
    _txt(ax, 94.6, y0 + h + 1.5, "1 of 26", fs=7.6, c=INK_MUTED)

    # ---------------------------------------------------------------- band B
    ax.add_patch(plt.Rectangle((31.0, 2.6), 67.5, 25.4, facecolor=GOLD_FILL,
                               edgecolor=GOLD, lw=1.8, ls=(0, (6, 4)), zorder=0))
    _txt(ax, 32.5, 26.3, "EXPLORING — physics-informed GNN (PIGNN)", fs=11.5,
         c="#a86c00", fw="bold", ha="left")
    _txt(ax, 64.7, 4.3, "every prior above is already computed and cached in this repo",
         fs=9.0, c=INK_SECONDARY, style="italic")

    pw, ph, py = 19.5, 14.4, 6.4
    physics = [
        (33.0, centers[2],
         "P1  Impedance-weighted\nmessages",
         None,
         "Scale every message by the line it crosses, using the per-edge "
         "R, X and |Z| already computed for this feeder, so a message over a "
         "high-impedance line is attenuated the way the sag physically is.",
         f"edge_attr {list(f['edge_shape'])}"),
        (55.5, centers[3],
         "P2  Sensitivity +\ngeometry prior",
         None,
         "Initialize the fault-bus templates from the first-order sag "
         "signature, and bias the decoder by hop distance from each candidate "
         "fault to each sensor over the full 123-bus feeder.",
         f"S {list(f['sens_shape'])}  ·  D {list(f['hop_shape'])}"),
        (78.0, centers[4],
         "P3  Physics residual\nloss",
         r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda\,\mathcal{L}_{phys}$",
         "Penalize a prediction whose implied sag pattern violates the "
         "Ohm/KCL consistency the feeder must obey.",
         "training-time only · 0 inference cost"),
    ]

    for x, target_cx, title, eqn, body, tag in physics:
        _box(ax, x, py, pw, ph, SURFACE, GOLD, lw=2.0, z=2, ls=(0, (5, 3)))
        _txt(ax, x + pw / 2, py + ph - 2.6, title, fs=10.2, fw="bold", c=INK, lsp=1.3)
        by = py + 6.2
        if eqn:
            _txt(ax, x + pw / 2, py + ph - 6.0, eqn, fs=11.0, c=INK)
            by = py + 4.6
        _txt(ax, x + pw / 2, by, _wrap(body, pw, 8.0), fs=8.0,
             c=INK_SECONDARY, lsp=1.42)
        _txt(ax, x + pw / 2, py + 1.3, tag, fs=7.6, c="#a86c00", fw="bold")
        # arrow up into the stage this prior modifies
        _arrow(ax, x + pw / 2, py + ph + 0.3, target_cx, y0 - 0.4,
               c=GOLD, lw=2.4)

    # ------------------------------------------------- the physics-derived ceiling
    cw, ch, cx, cy = 27.0, 21.4, 2.5, 6.4
    _box(ax, cx, cy, cw, ch, SURFACE, ACCENT, lw=2.0, z=2)
    _txt(ax, cx + cw / 2, cy + ch - 2.8,
         "The physics also sets a ceiling", fs=10.6, fw="bold", c=ACCENT)
    _txt(ax, cx + cw / 2, cy + ch - 8.2,
         _wrap(
             "Sensitivity signatures are collinear in groups, so the "
             f"{f['n_classes'] - 1} fault classes collapse into "
             f"{f['n_zones']} electrical zones that no method reading "
             "sag amplitude alone can separate.",
             cw, 8.2),
         fs=8.2, c=INK_SECONDARY, lsp=1.42)
    _txt(ax, cx + cw / 2, cy + 6.6,
         _wrap(
             "Within a zone the answer must come from temporal shape and "
             "per-phase asymmetry — that is where the learned part earns "
             "its keep.", cw, 8.2),
         fs=8.2, c=INK, lsp=1.42)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"  {f['n_classes'] - 1} fault classes → {f['n_zones']} electrical zones; "
          f"S{list(f['sens_shape'])}, D{list(f['hop_shape'])} (max {f['hop_max']} hops), "
          f"edge_attr{list(f['edge_shape'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/figures/ppt_stgnn_pignn.png")
    a = ap.parse_args()
    build(REPO / a.out, load_facts())


if __name__ == "__main__":
    main()
