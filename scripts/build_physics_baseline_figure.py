"""Degradation figure: the training-free physics baseline against the learned models.

The point of the figure is a crossing. The physics matched filter
(scripts/run_physics_matched_filter.py) has zero learned parameters and is
5x weaker than RGSAGE-Max on a fully instrumented feeder -- and it is
*completely flat* under sensor dropout, because a masked cosine drops terms
from a sum rather than feeding a network zeros it never saw in training. So
past a few sensors it overtakes the trained model it started far behind.

Everything is read from results/ at build time; nothing is recomputed here.

Output: paper/figures/physics_baseline_degradation.png

Usage:
    uv run python scripts/build_physics_baseline_figure.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRIVIAL = 0.0256          # always-"no fault"; scripts/build_absolute_audc.py
KS = [0, 1, 2, 5, 10]


def curve(path, model):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["model"] == model:
            out[int(r["k"])] = (float(r["f1_mean"]), float(r["f1_std"]))
    return out


def main() -> int:
    series = []
    std = curve("results/dropout_study/degradation_curves.csv", "rgsage_max")
    aware = curve("results/dropout_study_aware/degradation_curves.csv", "rgsage_max")
    sdr = curve("results/dropout_study_sdr/degradation_curves.csv", "sdr_stgnn")
    series += [("RGSAGE-Max (standard)", std, "#b91c1c", "-", "o"),
               ("RGSAGE-Max + dropout-aware training", aware, "#15803d", "-", "s"),
               ("SDR-STGNN", sdr, "#7c3aed", "-", "^")]

    imp = json.load(open("results/dropout_study_impute/imputation_summary.json"))
    imp_c = {int(k): (v, 0.0) for k, v in zip(imp["k_levels"], imp["f1_mean"])} \
        if isinstance(imp.get("f1_mean"), list) else {}
    if imp_c:
        series.append(("+ neighbour-mean imputation", imp_c, "#0891b2", "-", "d"))

    phys = json.load(open("results/physics_baseline/matched_filter.json"))
    phys_c = {int(k): (v["f1_mean"], v["f1_std"]) for k, v in phys["curve"].items()}
    series.append(("Physics matched filter (0 learned params)", phys_c,
                   "#c2410c", "--", "*"))

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=300)
    for label, c, colour, ls, mk in series:
        ks = [k for k in KS if k in c]
        if not ks:
            continue
        m = np.array([c[k][0] for k in ks])
        s = np.array([c[k][1] for k in ks])
        ax.plot(ks, m, ls, color=colour, marker=mk, lw=2.1,
                ms=7 if mk == "*" else 5, label=label)
        ax.fill_between(ks, m - s, m + s, color=colour, alpha=.15, lw=0)

    ax.axhline(TRIVIAL, color="0.45", lw=1.1, ls=":", zorder=0)
    ax.annotate('trivial "no fault" classifier (0.026)', (10, TRIVIAL),
                xytext=(-6, 6), textcoords="offset points",
                ha="right", fontsize=8, color="0.35")

    # Mark the crossing of the flat physics line and the collapsing baseline.
    for k in KS:
        if k in std and k in phys_c and phys_c[k][0] > std[k][0]:
            ax.annotate(f"physics overtakes\nthe trained model (k={k})",
                        (k, phys_c[k][0]), xytext=(12, 34),
                        textcoords="offset points", fontsize=8.5, color="#c2410c",
                        arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.2))
            break

    ax.set_xlabel("k — number of dropped μPMUs (of 25)")
    ax.set_ylabel("macro-F1 (26-class, soft-voted)")
    ax.set_title("Sensor dropout: a training-free physics rule is flat where\n"
                 "a trained network collapses", fontsize=11.5)
    ax.set_xticks(KS)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.4, loc="upper right", framealpha=.95)
    fig.tight_layout()
    out = Path("paper/figures/physics_baseline_degradation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    print(f"\n{'k':>3} {'standard':>10} {'physics':>10}   ratio")
    for k in KS:
        if k in std and k in phys_c:
            print(f"{k:>3} {std[k][0]:10.4f} {phys_c[k][0]:10.4f}   "
                  f"{phys_c[k][0]/std[k][0]:5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
