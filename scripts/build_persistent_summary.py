"""Summarize the persistent (per-event) dropout study against the random sweep.

Consumes the per-seed CSVs written by ``run_persistent_dropout_study.py`` plus the
existing per-window random-dropout curves, and emits every number the paper quotes
for the persistence experiment, so none of them is transcribed by hand.

The contrast this isolates: both sweeps drop k of 25 sensors uniformly at random
and zero them identically; they differ only in *when* the set is resampled — every
window (random) versus once per event and held for all 40 of its windows
(persistent).  Any gap is therefore attributable to mask persistence alone.

AUDC uses the same trapezoidal definition as ``run_significance_tests.py`` and
``build_targeted_summary.py``.

Usage:
    uv run python scripts/build_persistent_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
PERSISTENT = REPO / "results/dropout_study_persistent"
K_LEVELS = [0, 1, 2, 5, 10]
SEEDS = [42, 43, 44]
MODEL = "rgsage_max"


def audc(f1_by_k: dict[int, float], ks: Sequence[int] = K_LEVELS) -> float:
    """Normalized area under the retention curve R(k)=F1(k)/F1(0) (trapezoidal)."""
    r = [f1_by_k[k] / f1_by_k[ks[0]] for k in ks]
    area = sum((r[i] + r[i + 1]) / 2 * (ks[i + 1] - ks[i]) for i in range(len(ks) - 1))
    return area / (ks[-1] - ks[0])


def persistent_per_seed(variant: str) -> list[dict[int, float]]:
    df = pd.read_csv(PERSISTENT / f"persistent_per_seed_{variant}.csv")
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
        "standard_persistent": persistent_per_seed("standard"),
        "standard_random": random_per_seed(
            REPO / "results/dropout_study/degradation_curves.csv"),
        "aware_persistent": persistent_per_seed("dropout_aware"),
        "aware_random": random_per_seed(
            REPO / "results/dropout_study_aware/degradation_curves.csv"),
    }
    f1 = {n: {k: np.array([c[k] for c in v]) for k in K_LEVELS}
          for n, v in curves.items()}
    au = {n: np.array([audc(c) for c in v]) for n, v in curves.items()}

    def retention(name: str, k: int = 10) -> np.ndarray:
        return f1[name][k] / f1[name][0]

    out: dict = {
        "model": MODEL, "seeds": SEEDS, "k_levels": K_LEVELS,
        "policy": "persistent_per_event",
        "notes": (
            "The dropped set is drawn once per event from (event_id, k, mask_seed) "
            "and held across that event's 40 windows, so a failed sensor stays dark "
            "for the whole event as a real one would, instead of being redrawn every "
            "window. Selection is uniform at random, exactly as in the per-window "
            "sweep, and zeroing semantics are unchanged, so the only variable that "
            "moves is the resampling rate. Averaged over 3 mask seeds within each "
            "training seed; the error bar is over training seeds 42/43/44."
        ),
        "f1": {n: {str(k): {"mean": float(v[k].mean()),
                            "std": float(v[k].std(ddof=1)),
                            "per_seed": [round(float(x), 6) for x in v[k]]}
                   for k in K_LEVELS} for n, v in f1.items()},
        "audc": {n: {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                     "per_seed": [round(float(x), 6) for x in v]}
                 for n, v in au.items()},
        "retention_at_10": {n: {"mean": float(retention(n).mean()),
                                "std": float(retention(n).std(ddof=1)),
                                "per_seed": [round(float(x), 6)
                                             for x in retention(n)]}
                            for n in curves},
        "tests": {},
    }

    # Does persistence change the measured degradation at all?
    for variant in ("standard", "aware"):
        for k in K_LEVELS[1:]:
            out["tests"][f"{variant}: persistent vs random @k={k}"] = paired(
                f1[f"{variant}_persistent"][k], f1[f"{variant}_random"][k])
        out["tests"][f"{variant}: persistent vs random AUDC"] = paired(
            au[f"{variant}_persistent"], au[f"{variant}_random"])
        out["tests"][f"{variant}: persistent vs random Ret@10"] = paired(
            retention(f"{variant}_persistent"), retention(f"{variant}_random"))

    # Does the headline mitigation claim survive under the coherent model?
    for k in K_LEVELS[1:]:
        out["tests"][f"mitigation under persistence @k={k}"] = paired(
            f1["aware_persistent"][k], f1["standard_persistent"][k])
    out["tests"]["mitigation under persistence AUDC"] = paired(
        au["aware_persistent"], au["standard_persistent"])
    out["tests"]["mitigation under persistence Ret@10"] = paired(
        retention("aware_persistent"), retention("standard_persistent"))

    # Guard: the random AUDCs must reproduce the paper's published values, which
    # is what proves this script's AUDC matches the one behind Table 4.
    for name, expected in (("standard_random", 0.4022), ("aware_random", 0.9129)):
        got = au[name].mean()
        if abs(got - expected) > 0.005:
            raise SystemExit(
                f"AUDC mismatch for {name}: {got:.4f} vs published {expected} — "
                "the definition has drifted from the rest of the paper."
            )

    PERSISTENT.mkdir(parents=True, exist_ok=True)
    path = PERSISTENT / "persistent_summary.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"{'k':>3} | {'std/pers':>13} {'std/rnd':>13} | "
          f"{'aware/pers':>13} {'aware/rnd':>13}")
    for k in K_LEVELS:
        cells = [f"{f1[n][k].mean():.3f}±{f1[n][k].std(ddof=1):.3f}"
                 for n in ("standard_persistent", "standard_random",
                           "aware_persistent", "aware_random")]
        print(f"{k:>3} | {cells[0]:>13} {cells[1]:>13} | "
              f"{cells[2]:>13} {cells[3]:>13}")
    print("\nAUDC:", {n: round(float(v.mean()), 4) for n, v in au.items()})
    print("Ret@10:", {n: round(float(retention(n).mean()), 4) for n in curves})
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
