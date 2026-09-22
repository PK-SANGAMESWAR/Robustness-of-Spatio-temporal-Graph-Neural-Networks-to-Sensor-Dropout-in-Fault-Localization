"""Noise x dropout: where the physics rule wins and why.

Panel A -- macro-F1 against dropout at each noise level. The trained model
(solid) starts far higher and collapses; the training-free physics rule
(dashed) is flat. They cross at every noise level, and the crossing moves
*earlier* as noise rises.

Panel B -- the decomposition that explains the physics rule's own noise loss.
Its 26-class macro-F1 falls 40% at the IEEE C37.118 1% TVE cap, but its
fault-window accuracy -- the localization half -- falls only 12%. The
difference is the detection threshold tau: noise inflates the deviation energy,
pushing no-fault windows over a tau that was fitted at sigma = 0 and then held
fixed. So the loss is threshold calibration, not a failure of the matched
filter, and re-fitting tau per sigma would recover most of it.

All values are read from results/ at build time.

Output: paper/figures/physics_noise_dropout.png
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

KS = [0, 1, 2, 5, 10]
SIGMAS = [0.0, 0.0005, 0.002, 0.01]
TRIVIAL = 0.0256
COLOURS = {0.0: "#1e3a8a", 0.0005: "#0891b2", 0.002: "#ea580c", 0.01: "#b91c1c"}


def main() -> int:
    learned = {}
    for r in csv.DictReader(open("results/dropout_study_noise/noise_curves_standard.csv")):
        if r["model"] == "rgsage_max":
            learned[(float(r["sigma_pu"]), int(r["k"]))] = float(r["f1_mean"])

    d = json.load(open("results/physics_baseline/matched_filter.json"))
    phys = {(v["sigma_pu"], v["k"]): v["f1_mean"] for v in d["curve"].values()}
    acc = {(v["sigma_pu"], v["k"]): v["fault_window_accuracy_mean"]
           for v in d["curve"].values()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2), dpi=300)

    for s in SIGMAS:
        c = COLOURS[s]
        lk = [k for k in KS if (s, k) in learned]
        ax1.plot(lk, [learned[(s, k)] for k in lk], "-", color=c, marker="o",
                 ms=4.5, lw=2.0, label=f"trained, σ={s:g}")
        pk = [k for k in KS if (s, k) in phys]
        ax1.plot(pk, [phys[(s, k)] for k in pk], "--", color=c, marker="*",
                 ms=8, lw=1.8, alpha=.85)
    ax1.axhline(TRIVIAL, color="0.45", lw=1.0, ls=":", zorder=0)
    ax1.annotate('trivial floor 0.026', (10, TRIVIAL), xytext=(-4, 5),
                 textcoords="offset points", ha="right", fontsize=7.5, color="0.35")
    ax1.set_yscale("log")
    ax1.set_xticks(KS)
    ax1.set_xlabel("k — dropped μPMUs (of 25)")
    ax1.set_ylabel("macro-F1 (log scale)")
    ax1.set_title("Solid = trained RGSAGE-Max   ·   Dashed = physics matched filter\n"
                  "(0 learned parameters)", fontsize=10)
    ax1.grid(alpha=.3, which="both")
    ax1.legend(fontsize=7.6, ncol=2, loc="lower left")

    x = np.arange(len(SIGMAS))
    f1 = [phys[(s, 0)] / phys[(0.0, 0)] for s in SIGMAS]
    ac = [acc[(s, 0)] / acc[(0.0, 0)] for s in SIGMAS]
    lr = [learned[(s, 0)] / learned[(0.0, 0)] for s in SIGMAS]
    ax2.bar(x - .27, lr, .26, color="#b91c1c", label="trained model, macro-F1")
    ax2.bar(x, f1, .26, color="#c2410c", label="physics, macro-F1 (26-class)")
    ax2.bar(x + .27, ac, .26, color="#f59e0b",
            label="physics, localization only (fault-window acc.)")
    for xi, v in zip(x + .27, ac):
        ax2.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5)
    ax2.axvline(2.5, color="0.7", lw=1, ls="--")
    ax2.annotate("IEEE C37.118\n1% TVE cap", (3, 1.02), ha="center", fontsize=7.5,
                 color="0.35")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s:g}" for s in SIGMAS])
    ax2.set_xlabel("measurement noise σ (p.u.)")
    ax2.set_ylabel("retained fraction of the σ=0 score")
    ax2.set_title("At k=0: the physics rule's localization is nearly noise-immune;\n"
                  "its macro-F1 loss is the fixed detection threshold", fontsize=10)
    ax2.set_ylim(0, 1.12)
    ax2.grid(alpha=.3, axis="y")
    ax2.legend(fontsize=7.6, loc="lower left")

    fig.tight_layout()
    out = Path("paper/figures/physics_noise_dropout.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
