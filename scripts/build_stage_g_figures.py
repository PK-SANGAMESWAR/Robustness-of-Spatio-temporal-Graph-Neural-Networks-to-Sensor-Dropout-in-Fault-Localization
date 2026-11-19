"""Stage G figures from the non-sealed splits: F-G1, F-G4, F-G5, F-G6 (``PC-STGNN/10_FIGURES_TABLES.md``).

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/build_stage_g_figures.py

Writes PNG (200 dpi) and PDF to ``results/stage_g/figures/`` and ``figures_data.json`` (every plotted value, with
its source). Data come from ``fault_localization/stage_g/tables.py`` readers; no number is typed in here.
F-G2 / F-G3 (sealed-set resistance figures) are built after the one sealed run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fault_localization.stage_g import tables as T  # noqa: E402

OUT = REPO / "results/stage_g/figures"
# one fixed colour per model, every figure (reference categorical palette, slots 1-6 in order)
COLOR = {"R5": "#2a78d6", "R0ⁿ": "#eb6834", "R2": "#1baf7a", "PSM": "#eda100", "B2ᵃ": "#e87ba4", "R0": "#008300"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _line(ax, xs, means, stds, name, label=None, dashed=False, dy=0):
    c = COLOR[name]
    ax.plot(xs, means, color=c, linewidth=2, linestyle="--" if dashed else "-", marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.2, label=label or name, zorder=3)
    if stds is not None and not all(s is None for s in stds):
        s = np.array([0 if v is None else v for v in stds])
        ax.fill_between(xs, np.array(means) - s, np.array(means) + s, color=c, alpha=0.15, linewidth=0, zorder=2)
    ax.annotate(label or name, (xs[-1], means[-1]), xytext=(5, dy), textcoords="offset points", fontsize=8, color=INK,
                va="center")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def curve_stats(per_seed: dict, ks) -> tuple[list, list]:
    m, s = [], []
    for k in ks:
        mm, ss = T.mean_std([per_seed[seed][k] for seed in per_seed])
        m.append(mm)
        s.append(ss)
    return m, s


def fig_g1(data: dict) -> None:
    ks = list(T.K_LEVELS)
    curves = {"R5": T.stage_g_curve("r5"), "R0ⁿ": T.stage_g_curve("r0n"), "R2": T.stage_g_curve("r2"),
              "B2ᵃ": T.stage_g_curve("b2a"),
              "R0": T.paper1_curve(REPO / T.PAPER1_CURVES["dropout-aware"], "rgsage_max")}
    psm = {s: {k: v["psm"] for (k, sg), v in T.load_cells(T.stage_g_path("test", "r5", s)).items() if sg == 0.0}
           for s in (42,)}
    curves["PSM"] = psm
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    d = {}
    for name in ("R5", "R0", "R0ⁿ", "R2", "PSM", "B2ᵃ"):
        m, s = curve_stats(curves[name], ks)
        lab = {"R0": "R0 (Paper 1 eval.)", "B2ᵃ": "B2ᵃ (1 seed)", "PSM": "PSM (no weights)"}.get(name, name)
        _line(ax, ks, m, s, name, lab, dashed=(name in ("R0", "PSM")), dy={"R0ⁿ": 5, "R2": -6}.get(name, 0))
        d[name] = {"k": ks, "mean": m, "std": s, "n_seeds": len(curves[name])}
    _style(ax, "sensors lost, k (of 25)", "macro-F1 (26 classes)")
    ax.set_xticks(ks)
    ax.set_xlim(-0.3, 12.5)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("F-G1  Test split, random sensor loss, no noise", fontsize=9, color=INK, loc="left")
    _save(fig, "F-G1_f1_vs_k")
    data["F-G1"] = {"source": "results/stage_g/test/*/seed_*_scoring.json; results/dropout_study_aware/degradation_curves.csv",
                    "series": d}


def fig_g4(data: dict) -> None:
    sig = list(T.SIGMAS)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    d = {}
    for ax, k in zip(axes, (0, 10)):
        for name, arm in (("R5", "r5"), ("R0ⁿ", "r0n"), ("R2", "r2")):
            per = {s: {sg: v["f1"] for (kk, sg), v in T.load_cells(T.stage_g_path("test", arm, s)).items() if kk == k}
                   for s in T.SEEDS}
            m, sd = curve_stats(per, sig)
            xs = list(range(len(sig)))
            _line(ax, xs, m, sd, name)
            d[f"k{k}_{name}"] = {"sigma": sig, "mean": m, "std": sd}
        _style(ax, "noise σ (p.u.)", "macro-F1" if k == 0 else "")
        ax.set_xticks(range(len(sig)), ["0", "5e-4", "2e-3", "1e-2"])
        ax.set_xlim(-0.2, 3.6)
        ax.set_title(f"k = {k}", fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("F-G4  Test split, measurement noise (3 seeds, band = ±1 std)", fontsize=9, color=INK, x=0.01, y=1.06, ha="left")
    _save(fig, "F-G4_noise")
    data["F-G4"] = {"source": "results/stage_g/test/{r5,r0n,r2}/seed_*_scoring.json", "series": d}


def fig_g5(data: dict) -> None:
    ks = [0, 1, 2, 5, 10]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    d = {}
    for ax, pol in zip(axes, ("random", "persistent", "contiguous", "targeted")):
        for name, base in (("R5", "r5"), ("R0ⁿ", "r0n")):
            per = {}
            for s in T.SEEDS:
                k0 = T.load_cells(T.stage_g_path("test", base, s))[(0, 0.0)]["f1"]       # k = 0 is policy-free
                src = base if pol == "random" else f"{base}_{pol}"
                cells = T.load_cells(T.stage_g_path("test", src, s))
                per[s] = {0: k0, **{k: cells[(k, 0.0)]["f1"] for k in ks[1:]}}
            m, sd = curve_stats(per, ks)
            _line(ax, ks, m, sd, name)
            d[f"{pol}_{name}"] = {"k": ks, "mean": m, "std": sd}
        _style(ax, "k", "macro-F1" if pol == "random" else "")
        ax.set_xticks(ks)
        ax.set_xlim(-0.3, 12.8)
        ax.set_title(pol, fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("F-G5  Test split, sensor-loss policies, σ = 0 (3 seeds)", fontsize=9, color=INK, x=0.01, y=1.06, ha="left")
    _save(fig, "F-G5_policies")
    data["F-G5"] = {"source": "results/stage_g/test/{r5,r0n}{,_persistent,_contiguous,_targeted}/seed_*_scoring.json",
                    "series": d, "note": "k = 0 point is the random-policy k = 0 cell (no sensor is lost)."}


def fig_g6(data: dict) -> None:
    summary = json.loads((REPO / "results/stage_g/analysis/summary.json").read_text())
    cells = ["k10_sigma0.0", "k10_sigma0.0005", "k10_sigma0.002", "k10_sigma0.01", "k0_sigma0.0005", "k0_sigma0.002",
             "k0_sigma0.01"]
    labels = ["k10\nσ0", "k10\n5e-4", "k10\n2e-3", "k10\n1e-2", "k0\n5e-4", "k0\n2e-3", "k0\n1e-2"]
    # tables are a different entity from the models of the other figures: own colours (ink, violet, red, muted) and marker shapes
    tables = [("nominal table", None, INK, "o"), ("impedance ±5 %", "imp5", "#6250d6", "s"),
              ("tap −1", "tap_m1", "#e34948", "^"), ("tap +1", "tap_p1", "#8a8985", "v")]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.axhline(0, color=MUTED, linewidth=1)
    xs = np.arange(len(cells))
    d = {}
    for i, (lab, key, col, mk) in enumerate(tables):
        mm = summary["mismatch_phase1_val"]["imp5"]["per_cell"]
        if key is None:
            ys = [mm[c]["r5_nominal_mean"] - mm[c]["r0n_mean"] for c in cells]
        else:
            pc = summary["mismatch_phase1_val"][key]["per_cell"]
            ys = [pc[c]["r5_on_table_mean"] - pc[c]["r0n_mean"] for c in cells]
        ax.plot(xs + (i - 1.5) * 0.12, ys, color=col, linewidth=0, marker=mk, markersize=6,
                markeredgecolor="white", markeredgewidth=1.0, label=lab, zorder=3)
        d[lab] = {"cells": cells, "delta_vs_r0n_mean": ys}
    _style(ax, "rule cell (k, σ)", "ΔF1 = R5 − R0ⁿ (validation, mean of 3 seeds)")
    ax.set_xticks(xs, labels, fontsize=7)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("F-G6  Feeder-model mismatch at inference (below 0: R5 worse than R0ⁿ)", fontsize=9, color=INK, loc="left")
    _save(fig, "F-G6_mismatch")
    data["F-G6"] = {"source": "results/stage_g/analysis/summary.json (mismatch_phase1_val)", "series": d,
                    "note": "R0ⁿ does not read the table; its values are the same for every table."}


def main() -> int:
    data: dict = {}
    for f in (fig_g1, fig_g4, fig_g5, fig_g6):
        f(data)
    (OUT / "figures_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print("written:", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
