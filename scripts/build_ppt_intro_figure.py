"""Build the PPT topic-introduction diagram.

One 16:9 slide that states what fault localization on a distribution feeder is,
before any of this project's own contributions are introduced:

  left   the real IEEE 123-bus feeder, showing how little of it is instrumented
  right  how we pose the task -- measurements in, 26-way answer out
  bottom detect / classify / localize, with the hard one marked

The feeder topology is the real one (compiled from the IEEE-123 DSS source via
``build_full_topology_graph``), cached to a small edgelist so the slide can be
rebuilt without OpenDSS.  Bus and sensor counts come from
``fault_localization.config.constants``.

Usage:
    uv run python scripts/build_ppt_intro_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fault_localization.config import constants as C

INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"; PANEL = "#f3f2ee"
BLUE = "#2a78d6"; DEEP = "#184f95"; GREEN = "#1baf7a"; RED = "#d1495b"
AMBER = "#eda100"

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "paper" / "figures" / "feeder123.edgelist"


def feeder_graph() -> nx.Graph:
    """The real 123-bus topology, from cache or freshly compiled."""
    if CACHE.exists():
        return nx.read_edgelist(CACHE, nodetype=int)
    from fault_localization.simulation.feeder import compile_feeder
    from fault_localization.graph.measured_only import build_full_topology_graph
    compile_feeder()
    G = build_full_topology_graph()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    nx.write_edgelist(G, CACHE, data=False)
    return G


def box(ax, x, y, w, h, text, *, fc, ec, tc=INK, fs=13, weight="normal", pad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, weight=weight, linespacing=1.5)


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=17, linewidth=1.8,
                                 color=INK_SECONDARY, zorder=3))


def main() -> int:
    G = feeder_graph()
    pmu = set(C.PMU_BUSES)
    faults = set(C.FAULT_BUSES)
    measured_faults = sorted(pmu & faults)

    # A radial feeder reads best drawn as a tree, so lay it out from the
    # substation end (the lowest-numbered bus present) outward.
    root = min(G.nodes)
    try:
        pos = nx.bfs_layout(G, root)
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    plt.rcParams.update({"figure.facecolor": SURFACE, "font.family": "sans-serif",
                         "text.color": INK})
    fig = plt.figure(figsize=(13.333, 7.5))

    # ---- title -----------------------------------------------------------
    ft = fig.add_axes([0, 0.90, 1, 0.10]); ft.set_axis_off()
    ft.text(0.5, 0.60, "Fault Localization on a Power Distribution Feeder",
            ha="center", va="center", fontsize=24, weight="bold", color=INK)
    ft.text(0.5, 0.08, "a fault is detected in milliseconds — finding WHERE it is "
                       "is what keeps customers off supply",
            ha="center", va="center", fontsize=13, color=INK_SECONDARY, style="italic")

    # ---- left: the feeder ------------------------------------------------
    gx = fig.add_axes([0.015, 0.20, 0.55, 0.69]); gx.set_axis_off()
    gx.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                transform=gx.transAxes, facecolor=PANEL,
                                edgecolor=GRID, linewidth=1.4, zorder=0))
    nx.draw_networkx_edges(G, pos, ax=gx, edge_color=INK_MUTED, width=1.4)
    plain = [n for n in G.nodes if n not in pmu]
    nx.draw_networkx_nodes(G, pos, ax=gx, nodelist=plain, node_size=42,
                           node_color="#ffffff", edgecolors=INK_MUTED, linewidths=1.0)
    nx.draw_networkx_nodes(G, pos, ax=gx, nodelist=sorted(pmu), node_size=115,
                           node_color=BLUE, edgecolors=INK, linewidths=1.0)
    # the substation end of the feeder
    gx.scatter([pos[root][0]], [pos[root][1]], marker="s", s=210, c=DEEP,
               edgecolors=INK, linewidths=1.2, zorder=5)

    # one fault, deliberately at a bus that carries no sensor
    fault_bus = sorted(faults - pmu)[len(sorted(faults - pmu)) // 2]
    fx, fy = pos[fault_bus]
    gx.scatter([fx], [fy], marker="*", s=620, c=RED, edgecolors=INK,
               linewidths=1.0, zorder=6)

    gx.text(0.5, 0.965, "IEEE 123-bus test feeder  —  the real topology",
            transform=gx.transAxes, ha="center", va="top",
            fontsize=13.5, weight="bold", color=INK)

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=11, mfc=DEEP, mec=INK, label="substation"),
        plt.Line2D([], [], marker="o", ls="", ms=6, mfc="#ffffff", mec=INK_MUTED,
                   label="bus with no measurement"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=BLUE, mec=INK,
                   label=f"$\\mu$PMU  ({len(pmu)} of 123 buses)"),
        plt.Line2D([], [], marker="*", ls="", ms=16, mfc=RED, mec=INK, label="fault"),
    ]
    gx.legend(handles=handles, loc="lower left", frameon=True, fontsize=11,
              facecolor="#ffffff", edgecolor=GRID, borderpad=0.7,
              handletextpad=0.7, labelspacing=0.55,
              bbox_to_anchor=(0.005, 0.005))

    gx.text(0.985, 0.055,
            f"{len(faults)} candidate fault buses\n"
            f"only {len(measured_faults)} carry a $\\mu$PMU\n"
            f"$\\Rightarrow$ {len(faults) - len(measured_faults)} are never directly measured",
            transform=gx.transAxes, ha="right", va="bottom", fontsize=11.5,
            color=INK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                      edgecolor=AMBER, linewidth=1.6))

    # ---- right: how the task is posed ------------------------------------
    px = fig.add_axes([0.585, 0.20, 0.40, 0.69])
    px.set_xlim(0, 1); px.set_ylim(0, 1); px.set_axis_off()
    px.text(0.5, 0.985, "How the task is posed", ha="center", va="top",
            fontsize=13.5, weight="bold", color=INK)

    box(px, 0.06, 0.735, 0.88, 0.155,
        f"$\\mu$PMU measurements\n"
        f"3-phase RMS voltage + availability mask\n"
        f"a {C.WINDOW_LENGTH} ms window  →  [{len(pmu)} × {C.WINDOW_LENGTH} × 6]",
        fc="#ffffff", ec=BLUE, fs=12)
    arrow(px, 0.5, 0.725, 0.635)

    box(px, 0.06, 0.455, 0.88, 0.175,
        "Spatio-temporal GNN\n"
        "GRU along time  +  message passing\nover the feeder topology",
        fc="#eaf2fc", ec=DEEP, fs=12.5, weight="bold")
    px.text(0.5, 0.437, "topology lets an unmeasured bus be inferred\n"
                        "from its measured neighbours",
            ha="center", va="top", fontsize=10.5, color=INK_SECONDARY,
            style="italic", linespacing=1.45)
    arrow(px, 0.5, 0.365, 0.275)

    box(px, 0.06, 0.095, 0.88, 0.175,
        f"one of {C.NUM_CLASSES} classes\n"
        f"{len(faults)} candidate fault buses  +  “no fault”",
        fc="#e9f7f1", ec=GREEN, fs=12.5, weight="bold")
    px.text(0.5, 0.055, "answer: the faulted bus, not just “a fault occurred”",
            ha="center", va="center", fontsize=11, color=INK_SECONDARY)

    # ---- bottom: the three questions -------------------------------------
    bx = fig.add_axes([0.015, 0.015, 0.97, 0.17])
    bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.set_axis_off()
    steps = [
        ("1.  DETECT", "Did a fault occur?", "solved — protection relays, milliseconds", GRID, INK_SECONDARY),
        ("2.  CLASSIFY", "What kind of fault?", "line-to-ground / line-to-line / 3-phase", GRID, INK_SECONDARY),
        ("3.  LOCALIZE", "WHERE on the feeder?", "hours of patrolling — this is the problem", AMBER, INK),
    ]
    w, gap = 0.305, 0.038
    for i, (head, q, note, ec, tc) in enumerate(steps):
        x = i * (w + gap)
        lw = 2.6 if i == 2 else 1.4
        fc = "#fff7e6" if i == 2 else "#ffffff"
        bx.add_patch(FancyBboxPatch((x, 0.10), w, 0.80,
                                    boxstyle="round,pad=0.012,rounding_size=0.03",
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        bx.text(x + 0.028, 0.66, head, fontsize=12.5, weight="bold", color=tc, va="center")
        bx.text(x + 0.028, 0.42, q, fontsize=12, color=INK, va="center")
        bx.text(x + 0.028, 0.21, note, fontsize=10.5, color=INK_SECONDARY,
                va="center", style="italic")
        if i < 2:
            bx.annotate("", xy=(x + w + gap - 0.006, 0.50), xytext=(x + w + 0.006, 0.50),
                        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=2))

    out = REPO / "paper" / "figures" / "ppt_intro_concept.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"feeder: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges, root bus {root}")
    print(f"fault bus drawn: {fault_bus} (unmeasured); measured fault buses: {measured_faults}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
