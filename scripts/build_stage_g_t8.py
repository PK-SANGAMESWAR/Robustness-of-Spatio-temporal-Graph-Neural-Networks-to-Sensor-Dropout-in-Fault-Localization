"""T8 and F-G3 from the finished sealed run (CPU). Run after ``run_gate6_analysis.py``.

    nice -n 19 uv run python scripts/build_stage_g_t8.py

Refuses without ``COMPLETED.json`` (the one run) and ``gate6_analysis.json``. Writes ``results/stage_g/tables/T8_ood.{csv,md}``
and ``results/stage_g/figures/F-G3_f1_vs_resistance.{png,pdf}`` with ``F-G3_data.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fault_localization.stage_g import sealed_tables as S  # noqa: E402
from fault_localization.stage_g import tables as T  # noqa: E402

OUT_DIR = REPO / "results/stage_g/ood"
COLOR = {"r5": "#2a78d6", "r0n": "#eb6834", "r0": "#008300", "r2": "#1baf7a", "b2a": "#e87ba4", "r6i": "#8a8985"}
NAME = {"r5": "R5", "r0n": "R0ⁿ", "r0": "R0", "r2": "R2", "b2a": "B2ᵃ (1 seed)", "r6i": "R6ⁱ"}


def main() -> int:
    a = S.check_reportable(OUT_DIR)
    rows = S.build_t8(a)
    cols = ["cell", "role", "subset", "n_events", "R5", "R0ⁿ", "R0", "ΔF1 (R5 − R0ⁿ)", "per-seed Δ", "seeds > 0", "95 % CI (event bootstrap)"]
    T.write_table(rows, cols, REPO / "results/stage_g/tables/T8_ood", "T8 — sealed OOD set (11,000 events, R ∈ {3, 5, 20, 30} Ω)",
                  S.gate6_lines(a) + [
                      "Primary cells (Gate 6, pooled over R): k = 0 at σ = 5e-4, 2e-3, 1e-2. Everything else is descriptive (14, Amendments 3, 4, 11).",
                      "F1: mean ± sample std over seeds 42/43/44, each F1 the mean over 3 mask/noise offsets; Δ and CI as in 14, Amendments 1 and 9.",
                      "Operator grid (08, G8): 30 Ω is on the operator R axis; 3, 5, 20 Ω are between grid points. The test is out of "
                      "distribution for the network at all four R, for PSM only at 3, 5, 20 Ω.",
                      "Grounded / ungrounded strata were registered before the opening (14, Amendment 4); no decision rule."])
    series = S.f1_by_resistance_series(OUT_DIR, REPO / "results/stage_g/test")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    data = {}
    for name in ("r5", "r0n", "r0", "r2", "b2a"):
        if name not in series:
            continue
        pts = {**{r: v for r, v in series[name]["in"].items() if r in (0.1, 1.0, 10.0)}, **series[name]["ood"]}
        rs = sorted(pts)
        m = [T.mean_std(pts[r])[0] for r in rs]
        ax.plot(rs, m, color=COLOR[name], marker="o", markersize=5, markeredgecolor="white", linewidth=2, label=NAME[name])
        data[name] = {"R": rs, "mean": m, "n_seeds": {str(r): len(pts[r]) for r in rs}}
    ax.axvspan(2.5, 35, color="#e4e3df", alpha=0.5, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("fault resistance R (Ω); shaded = sealed OOD values {3, 5, 20, 30}")
    ax.set_ylabel("macro-F1 (26 classes), k = 0, σ = 0")
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.legend(fontsize=7, frameon=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("F-G3  F1 by resistance (R ≤ 10 Ω: test split; R ≥ 3 Ω shaded: sealed set)", fontsize=9, loc="left")
    fo = REPO / "results/stage_g/figures"
    fo.mkdir(parents=True, exist_ok=True)
    fig.savefig(fo / "F-G3_f1_vs_resistance.png", dpi=200, bbox_inches="tight")
    fig.savefig(fo / "F-G3_f1_vs_resistance.pdf", bbox_inches="tight")
    (fo / "F-G3_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print("written T8 and F-G3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
