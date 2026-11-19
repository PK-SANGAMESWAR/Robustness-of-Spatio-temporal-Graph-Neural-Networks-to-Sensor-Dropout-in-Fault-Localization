"""F-H1: test-split errors over the identifiability map (S7.1). Needs ``run_error_attribution.py`` output.

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/build_stage_g_fh1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fault_localization.stage_g import attribution as A  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
COLOR = {"R5": "#2a78d6", "R0ⁿ": "#eb6834", "R2": "#1baf7a"}


def error_matrix(name: str, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    with np.load(REPO / f"results/stage_g/test/{name}/seed_{seed:05d}_predictions.npz", allow_pickle=False) as z:
        keep = A.labelable(z["window_ids"])
        y, p = z["y_true"][keep].astype(int), z[f"pf__k{k}_s0_o0"][keep].astype(int)
    m = np.zeros((26, 26))
    np.add.at(m, (y[y != p], p[y != p]), 1)
    return m, A.load_D(k)


def main() -> int:
    att = json.loads((REPO / "results/stage_g/analysis/error_attribution.json").read_text())
    fig = plt.figure(figsize=(11, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.35)
    data = {}
    for i, k in enumerate((0, 10)):
        ax = fig.add_subplot(gs[0, i])
        m, D = error_matrix("r5", k)
        im = ax.imshow(np.where(m > 0, m, np.nan), cmap="Blues", norm=LogNorm(vmin=1, vmax=max(2, m.max())), origin="upper")
        ii, jj = np.where((D < A.D_THRESHOLD) & ~np.eye(26, dtype=bool))
        ax.scatter(jj, ii, s=60, facecolors="none", edgecolors="#e34948", linewidths=1.2, label="D < 1 pair")
        ax.set_xlabel("predicted class", color=MUTED, fontsize=8)
        ax.set_ylabel("true class" if i == 0 else "", color=MUTED, fontsize=8)
        ax.set_title(f"R5, k = {k}, seed 42: errors per cell", fontsize=8, loc="left", color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        data[f"R5_k{k}_seed42"] = {"errors_total": float(m.sum()), "errors_in_D_lt_1": float(m[(D < A.D_THRESHOLD) & ~np.eye(26, dtype=bool)].sum())}
    ax = fig.add_subplot(gs[0, 2])
    labels, vals, cols = [], [], []
    for model in ("R0ⁿ", "R5", "R2"):
        for k in (0, 10):
            v = att["per_seed_means_over_offsets"][f"{model}|k{k}"]["frac_errors_in_indist_pairs"]
            labels.append(f"{model}\nk={k}")
            vals.append(v)
            cols.append(COLOR[model])
    xs = np.arange(len(vals))
    for x, v, c in zip(xs, vals, cols):
        ax.bar(x, np.mean(v) * 100, color=c, width=0.6)
        ax.plot([x] * len(v), np.array(v) * 100, "o", color=INK, markersize=3, zorder=3)
    ch = att["per_seed_means_over_offsets"]["R5|k0"]["chance_frac"][0] * 100
    ch10 = att["per_seed_means_over_offsets"]["R5|k10"]["chance_frac"][0] * 100
    ax.plot([-0.4, 1.4], [ch, ch], color=MUTED, linestyle="--", linewidth=1)
    ax.plot([-0.4 + 0, 5.4], [ch10, ch10], color=MUTED, linestyle=":", linewidth=1)
    ax.set_xticks(xs, labels, fontsize=7)
    ax.set_ylabel("% of errors in a D < 1 pair", color=MUTED, fontsize=8)
    ax.set_title("Share of errors (dots = seeds); dashed/dotted = chance\n(k=0: 1.5 %, k=10: 1.9 %)", fontsize=8, loc="left", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors=MUTED)
    fo = REPO / "results/stage_g/figures"
    fig.savefig(fo / "F-H1_error_attribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(fo / "F-H1_error_attribution.pdf", bbox_inches="tight")
    (fo / "F-H1_data.json").write_text(json.dumps({"source": "results/stage_g/analysis/error_attribution.json", "heatmaps": data}, indent=1) + "\n")
    print("written F-H1", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
