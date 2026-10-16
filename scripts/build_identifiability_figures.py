"""F-C1 (26x26 D heatmaps at k=0,2,5,10) and F-C2 (fraction indistinguishable vs k, per R),
from S2.1's saved D matrices (``results/pcstgnn/audit/s2_1_identifiability/``).

Usage:
    uv run python scripts/build_identifiability_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

IN_DIR = REPO / "results/pcstgnn/audit/s2_1_identifiability"
OUT_DIR = REPO / "paper/figures"
K_LEVELS = [0, 2, 5, 10]
CORPUS_R = [0.1, 1.0, 10.0]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((IN_DIR / "identifiability_summary.json").read_text())

    # F-C1: 26x26 heatmaps at each k, log-scale (D spans ~1e-3 to ~1e3)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, k in zip(axes, K_LEVELS):
        d = np.load(IN_DIR / f"D_global_k{k}.npy")
        d_plot = np.log10(np.clip(d, 1e-3, None))
        im = ax.imshow(d_plot, cmap="viridis", vmin=-3, vmax=3)
        ax.set_title(f"k={k}  (indist. frac={summary['global'][str(k)]['indistinguishable_fraction']:.3f})")
        ax.set_xlabel("f2"); ax.set_ylabel("f1")
    fig.colorbar(im, ax=axes, label="log10 D(f1,f2;A)", shrink=0.8)
    fig.suptitle("F-C1 -- Identifiability D(f1,f2;A), profiled over full (tau,R,L), by dropout k")
    fig.savefig(OUT_DIR / "fc1_identifiability_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F-C2: fraction indistinguishable vs k, per R (+ global reference)
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = K_LEVELS
    ax.plot(ks, [summary["global"][str(k)]["indistinguishable_fraction"] for k in ks],
            marker="o", label="global (profiled R)", linewidth=2, color="black")
    for r in CORPUS_R:
        vals = [summary["per_r"][str(r)][str(k)]["indistinguishable_fraction"] for k in ks]
        ax.plot(ks, vals, marker="s", label=f"R={r} Ohm")
    ax.set_xlabel("dropout k"); ax.set_ylabel("fraction of pairs with D < 1")
    ax.set_title("F-C2 -- Indistinguishable-pair fraction vs k, per R")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fc2_indistinguishable_fraction_vs_k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT_DIR}/fc1_identifiability_heatmaps.png, fc2_indistinguishable_fraction_vs_k.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
