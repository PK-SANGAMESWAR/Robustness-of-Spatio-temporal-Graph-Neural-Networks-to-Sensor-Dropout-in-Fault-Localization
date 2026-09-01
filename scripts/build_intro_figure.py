"""Build the Introduction figure: the problem this paper attacks, in one row.

Three panels over the SAME real measured-only graph (25 nodes / 24 edges, read
from the frozen corpus), differing only in which micro-PMUs are reporting and
how the model was trained:

  (a) full observability            -> macro-F1 0.846
  (b) 10 units dark, standard       -> macro-F1 0.048  (1.9x the trivial floor)
  (c) 10 units dark, dropout-aware  -> macro-F1 0.674

Nothing here is drawn by hand.  The topology comes from
``dataset_fullscale_50/processed/edge_index.npy``; the dropped set is the mask
the evaluator itself produces at k=10 (``random.Random(10000*k + offset)``,
matching ``fault_localization/dropout/masking.py``); every macro-F1 is read from
the degradation-curve CSVs at run time.

Usage:
    uv run python scripts/build_intro_figure.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Shared project palette (see scripts/build_dataset_assets.py).
INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
LIVE = "#2a78d6"      # reporting muPMU
DARK = "#d8d6cf"      # dropped muPMU
GOOD = "#1baf7a"      # mitigated
BAD = "#d1495b"       # collapsed

K_DROP = 10
# A degenerate "always no-fault" classifier: F1 2/3 on class 0, zero elsewhere.
TRIVIAL = (2 * 0.5 * 1.0 / 1.5) / 26

REPO = Path(__file__).resolve().parent.parent


def load_graph(data_root: Path) -> tuple[nx.Graph, dict]:
    edge_index = np.load(data_root / "processed" / "edge_index.npy")
    node_mapping = json.loads((data_root / "processed" / "node_mapping.json").read_text())
    idx_to_bus = {v: int(k) for k, v in node_mapping.items()}
    G = nx.Graph()
    G.add_nodes_from(range(len(node_mapping)))
    for s, d in zip(edge_index[0], edge_index[1]):
        if s != d:
            G.add_edge(int(s), int(d))
    return G, idx_to_bus


def f1_at(csv_rel: str, model: str, k: int) -> float:
    df = pd.read_csv(REPO / csv_rel)
    row = df[(df.model == model) & (df.k == k)]
    if row.empty:
        raise SystemExit(f"no row for {model} k={k} in {csv_rel}")
    return float(row.iloc[0]["f1_mean"])


def draw_panel(ax, G, pos, dropped: set[int], title: str, subtitle: str) -> None:
    live_edges = [(u, v) for u, v in G.edges if u not in dropped and v not in dropped]
    dead_edges = [(u, v) for u, v in G.edges if u in dropped or v in dropped]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=live_edges,
                           edge_color=INK_MUTED, width=1.3)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dead_edges,
                           edge_color=GRID, width=1.3, style="dashed")

    live = [n for n in G.nodes if n not in dropped]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=live, node_size=118,
                           node_color=LIVE, edgecolors=INK, linewidths=0.8)
    if dropped:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(dropped), node_size=118,
                               node_color=DARK, edgecolors=INK_MUTED,
                               linewidths=0.8, node_shape="o")
        # an explicit cross so the panel survives greyscale printing
        ax.scatter([pos[n][0] for n in dropped], [pos[n][1] for n in dropped],
                   marker="x", s=42, c=INK_SECONDARY, linewidths=1.4, zorder=4)

    ax.set_title(title, fontsize=10.5, color=INK, pad=6, fontweight="bold",
                 linespacing=1.35)
    ax.text(0.5, -0.005, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=INK_SECONDARY)
    ax.set_axis_off()
    ax.margins(0.06)


def draw_bar(ax, value: float, color: str, label: str, show_floor: bool) -> None:
    ax.barh([0], [value], height=0.52, color=color, edgecolor=INK, linewidth=0.7)
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-0.6, 0.6)
    ax.text(value + 0.02, 0, label, va="center", ha="left",
            fontsize=10, color=INK, fontweight="bold")
    if show_floor:
        ax.axvline(TRIVIAL, color=BAD, linewidth=1.1, linestyle=":")
        ax.text(TRIVIAL + 0.015, -0.52, f"always-\"no fault\" floor {TRIVIAL:.3f}",
                fontsize=7.4, color=BAD, va="bottom", ha="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_yticks([])
    ax.set_xticks([0, 0.4, 0.8])
    ax.tick_params(axis="x", labelsize=7.5, colors=INK_MUTED, length=2)


def main() -> int:
    data_root = REPO / "dataset_fullscale_50"
    G, _ = load_graph(data_root)
    n = G.number_of_nodes()
    pos = nx.kamada_kawai_layout(G)

    # The evaluator's own mask at k=10, first offset -- not an illustrative guess.
    dropped = set(random.Random(10000 * K_DROP + 0).sample(range(n), K_DROP))

    std_clean = f1_at("results/dropout_study/degradation_curves.csv", "rgsage_max", 0)
    std_drop = f1_at("results/dropout_study/degradation_curves.csv", "rgsage_max", K_DROP)
    aware_drop = f1_at("results/dropout_study_aware/degradation_curves.csv", "rgsage_max", K_DROP)

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "font.family": "sans-serif", "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 4.3),
                             gridspec_kw={"height_ratios": [3.4, 1.0],
                                          "hspace": 0.30, "wspace": 0.14})

    panels = [
        (set(), "(a) All 25 $\\mu$PMUs reporting\nstandard training",
         "the setting prior work evaluates",
         std_clean, LIVE, f"{std_clean:.3f}", False),
        (dropped, f"(b) {K_DROP} $\\mu$PMUs dark\nstandard training",
         "40% sensor loss at inference",
         std_drop, BAD, f"{std_drop:.3f}", True),
        (dropped, f"(c) {K_DROP} $\\mu$PMUs dark\ndropout-aware training",
         "identical failure, identical network",
         aware_drop, GOOD, f"{aware_drop:.3f}", False),
    ]
    for i, (drop, title, sub, val, color, label, floor) in enumerate(panels):
        draw_panel(axes[0, i], G, pos, drop, title, sub)
        draw_bar(axes[1, i], val, color, label, floor)

    axes[1, 0].set_xlabel("soft-voted macro-F1 (26 classes)", fontsize=8,
                          color=INK_SECONDARY, labelpad=2)

    fig.subplots_adjust(left=0.025, right=0.98, top=0.865, bottom=0.115)
    out = REPO / "paper" / "figures" / "intro_problem.png"
    fig.savefig(out, dpi=400)
    plt.close(fig)

    print(f"dropped set (k={K_DROP}): {sorted(dropped)}")
    print(f"(a) {std_clean:.3f}   (b) {std_drop:.3f}   (c) {aware_drop:.3f}")
    print(f"trivial floor {TRIVIAL:.4f};  (b) is {std_drop / TRIVIAL:.1f}x it")
    print(f"retention (c) = {aware_drop / f1_at('results/dropout_study_aware/degradation_curves.csv','rgsage_max',0):.3f}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
