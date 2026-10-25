"""F-F1, F-F2, F-F3 -- training-free silent-failure detection (``07_STAGE_F_gate.md`` S5.1/S5.2).

F-F1  ROC curves of rho_i^LOO per corruption type (PSM refit, s = 1, 2, 3).
F-F2  Detection AUROC vs corruption severity / number of corrupted sensors, with the 0.75 gate.
F-F3  LOO vs in-sample AUROC per condition (the S5.2 contrast).

All values are read from results/pcstgnn/audit/s5_1_silent_detection/ at build time
(written by scripts/run_silent_detection.py). PSM refit only; the oracle refit is a diagnostic
and is not plotted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path("results/pcstgnn/audit/s5_1_silent_detection")
CORRUPTIONS = ["stuck", "bias_2pct", "bias_5pct", "gauss_0.02", "swap"]
NAMES = {"stuck": "stuck (frozen at window start)", "bias_2pct": "bias ±2 %",
         "bias_5pct": "bias ±5 %", "gauss_0.02": "Gaussian σ = 0.02 p.u.",
         "swap": "swapped phases"}
COLOURS = {"stuck": "#1e3a8a", "bias_2pct": "#b91c1c", "bias_5pct": "#ea580c",
           "gauss_0.02": "#0891b2", "swap": "#6d28d9"}
SENSORS = [1, 2, 3]
GATE = 0.75


def main() -> int:
    res = json.loads((DIR / "silent_detection.json").read_text())
    roc = json.loads((DIR / "silent_detection_roc.json").read_text())
    rows = res["rows"]

    def auroc(cond, score, stratum="all"):
        return next(r["auroc"] for r in rows if r["condition"] == cond and r["refit"] == "psm"
                    and r["score"] == score and r["stratum"] == stratum)

    # F-F1 ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(17, 3.8), dpi=300, sharey=True)
    for ax, c in zip(axes, CORRUPTIONS):
        for s, ls in zip(SENSORS, ["-", "--", ":"]):
            r = next(r for r in roc if r["condition"] == f"{c}_s{s}" and r["refit"] == "psm"
                     and r["score"] == "loo")
            ax.plot(r["fpr"], r["tpr"], ls, color=COLOURS[c], lw=1.8,
                    label=f"s={s}  AUROC {auroc(f'{c}_s{s}', 'loo'):.3f}")
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
        ax.axvline(0.05, color="0.6", lw=0.8, ls="--")
        ax.set_title(NAMES[c], fontsize=9.5)
        ax.set_xlabel("false-positive rate")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("true-positive rate")
    fig.suptitle("F-F1  Detection ROC of ρᵢ^LOO (PSM refit, validation split, k = 0, σ₀ = 0)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(DIR / "F-F1_roc_per_type.png", bbox_inches="tight")
    plt.close(fig)

    # F-F2 ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=300)
    for c in CORRUPTIONS:
        ax.plot(SENSORS, [auroc(f"{c}_s{s}", "loo") for s in SENSORS], "-o", color=COLOURS[c],
                lw=1.8, ms=4.5, label=NAMES[c])
    ax.axhline(GATE, color="0.3", lw=1.0, ls="--")
    ax.annotate("hard gate 0.75 (bias ±2 %)", (3, GATE), xytext=(-4, -12),
                textcoords="offset points", ha="right", fontsize=7.5, color="0.3")
    ax.set_xticks(SENSORS)
    ax.set_xlabel("s — corrupted μPMUs per window (of 25)")
    ax.set_ylabel("AUROC of ρᵢ^LOO")
    ax.set_title("F-F2  Detection AUROC vs severity", fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    fig.savefig(DIR / "F-F2_auroc_vs_severity.png", bbox_inches="tight")
    plt.close(fig)

    # F-F3 ------------------------------------------------------------------
    conds = [f"{c}_s{s}" for c in CORRUPTIONS for s in SENSORS]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=300)
    ax.bar(x - .2, [auroc(c, "loo") for c in conds], .4, color="#1e3a8a", label="LOO (subtract, then re-minimise)")
    ax.bar(x + .2, [auroc(c, "ins") for c in conds], .4, color="#f59e0b", label="in-sample (global argmin)")
    lo = min(min(auroc(c, "loo"), auroc(c, "ins")) for c in conds)
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_s", "\ns=") for c in conds], fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_title("F-F3  LOO vs in-sample detection AUROC (PSM refit)", fontsize=10)
    ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(DIR / "F-F3_loo_vs_insample.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote F-F1, F-F2, F-F3 to {DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
