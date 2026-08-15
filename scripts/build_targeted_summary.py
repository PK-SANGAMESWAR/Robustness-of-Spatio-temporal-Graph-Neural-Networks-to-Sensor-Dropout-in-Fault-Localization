"""Summarize the targeted (worst-case) dropout study and test it against the random sweep.

Consumes the per-seed CSVs written by ``run_targeted_dropout_study.py`` plus the
existing random-dropout curves, and emits every number the paper quotes for the
targeted experiment, so none of them is transcribed by hand.

AUDC uses the same trapezoidal definition as ``run_significance_tests.py`` and
``build_sdr_result_figures.py`` — a mean of per-k retention ratios would NOT
reproduce the paper's published 0.402 / 0.913.

Usage:
    uv run python scripts/build_targeted_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
TARGETED = REPO / "results/dropout_study_targeted"
K_LEVELS = [0, 1, 2, 5, 10]
SEEDS = [42, 43, 44]
MODEL = "rgsage_max"


def audc(f1_by_k: dict[int, float], ks: Sequence[int] = K_LEVELS) -> float:
    """Normalized area under the retention curve R(k)=F1(k)/F1(0) (trapezoidal)."""
    r = [f1_by_k[k] / f1_by_k[ks[0]] for k in ks]
    area = sum((r[i] + r[i + 1]) / 2 * (ks[i + 1] - ks[i]) for i in range(len(ks) - 1))
    return area / (ks[-1] - ks[0])


def targeted_per_seed(variant: str) -> list[dict[int, float]]:
    df = pd.read_csv(TARGETED / f"targeted_per_seed_{variant}.csv")
    df = df[df.model == MODEL]
    return [
        {k: float(df[(df.k == k) & (df.seed == s)].f1_seed.iloc[0]) for k in K_LEVELS}
        for s in SEEDS
    ]


def random_per_seed(curve_csv: Path) -> list[dict[int, float]]:
    df = pd.read_csv(curve_csv)
    df = df[df.model == MODEL]
    by_k = {int(r.k): json.loads(r.f1_per_train_seed) for _, r in df.iterrows()}
    return [{k: by_k[k][i] for k in K_LEVELS} for i in range(len(SEEDS))]


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    t = stats.ttest_rel(a, b)
    d = a - b
    return {
        "a_mean": float(a.mean()), "b_mean": float(b.mean()),
        "mean_diff": float(d.mean()), "t_stat": float(t.statistic),
        "p_ttest": float(t.pvalue),
        "a_per_seed": [round(float(x), 6) for x in a],
        "b_per_seed": [round(float(x), 6) for x in b],
        "signs": "".join("+" if x > 0 else "-" for x in d),
    }


def main() -> int:
    curves = {
        "standard_targeted": targeted_per_seed("standard"),
        "standard_random": random_per_seed(REPO / "results/dropout_study/degradation_curves.csv"),
        "aware_targeted": targeted_per_seed("dropout_aware"),
        "aware_random": random_per_seed(REPO / "results/dropout_study_aware/degradation_curves.csv"),
    }
    f1 = {n: {k: np.array([c[k] for c in v]) for k in K_LEVELS} for n, v in curves.items()}
    au = {n: np.array([audc(c) for c in v]) for n, v in curves.items()}

    out: dict = {
        "model": MODEL, "seeds": SEEDS, "k_levels": K_LEVELS,
        "policy": "targeted_nearest_fault",
        "notes": (
            "Targeted selection is deterministic given the label, so there are no "
            "mask seeds; the only error bar is over training seeds. No-fault "
            "windows have no fault bus and fall back to random selection, so the "
            "reported macro-F1 blends worst-case (fault) and average-case "
            "(no-fault) windows and is a conservative bound on the true worst case."
        ),
        "f1": {n: {str(k): {"mean": float(v[k].mean()),
                            "std": float(v[k].std(ddof=1)),
                            "per_seed": [round(float(x), 6) for x in v[k]]}
                   for k in K_LEVELS} for n, v in f1.items()},
        "audc": {n: {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                     "per_seed": [round(float(x), 6) for x in v]}
                 for n, v in au.items()},
        "tests": {},
    }

    for variant in ("standard", "aware"):
        for k in K_LEVELS[1:]:
            out["tests"][f"{variant}: targeted vs random @k={k}"] = paired(
                f1[f"{variant}_targeted"][k], f1[f"{variant}_random"][k])
        out["tests"][f"{variant}: targeted vs random AUDC"] = paired(
            au[f"{variant}_targeted"], au[f"{variant}_random"])

    for k in K_LEVELS[1:]:
        out["tests"][f"mitigation under targeting @k={k}"] = paired(
            f1["aware_targeted"][k], f1["standard_targeted"][k])
        out["tests"][f"aware/targeted vs standard/random @k={k}"] = paired(
            f1["aware_targeted"][k], f1["standard_random"][k])

    # Guard: the random AUDCs must reproduce the paper's published values, which
    # is what proves this script's AUDC matches the one behind Table 4.
    for name, expected in (("standard_random", 0.4022), ("aware_random", 0.9129)):
        got = au[name].mean()
        if abs(got - expected) > 0.005:
            raise SystemExit(
                f"AUDC mismatch for {name}: {got:.4f} vs published {expected} — "
                "the definition has drifted from the rest of the paper."
            )

    path = TARGETED / "targeted_summary.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"{'k':>3} | {'std/tgt':>13} {'std/rnd':>13} | {'aware/tgt':>13} {'aware/rnd':>13}")
    for k in K_LEVELS:
        cells = [f"{f1[n][k].mean():.3f}±{f1[n][k].std(ddof=1):.3f}"
                 for n in ("standard_targeted", "standard_random",
                           "aware_targeted", "aware_random")]
        print(f"{k:>3} | {cells[0]:>13} {cells[1]:>13} | {cells[2]:>13} {cells[3]:>13}")
    print("\nAUDC:", {n: round(float(v.mean()), 4) for n, v in au.items()})
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
