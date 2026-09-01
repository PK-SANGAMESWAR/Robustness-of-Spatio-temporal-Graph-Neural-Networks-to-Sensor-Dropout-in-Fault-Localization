"""Paired significance tests for the correlated-dropout and noise studies.

Companion to ``run_significance_tests.py``, which covers the mitigation claims
of the random sweep.  This one covers the two stressors added afterwards, under
the same discipline required throughout this project: three training seeds, the
seed as the paired unit, paired two-sided t-tests (n=3, df=2) with 95 % CIs and
Holm--Bonferroni correction applied per family.  Wilcoxon is not used; its
smallest attainable two-sided p at n=3 is 0.25.

Families tested
---------------
``correlated``  Contiguous outage against the scattered (random) sweep at equal
                k, for both the standard and the dropout-aware checkpoints, plus
                dropout-aware against standard under contiguous loss.
``noise``       Each non-zero noise level against the noise-free arm at equal k,
                and dropout-aware against standard at each (sigma, k).

Every number is read from a CSV already on disk; nothing is recomputed on GPU.

Usage:
    uv run python scripts/run_new_policy_significance.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

MODEL = "rgsage_max"


def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired two-sided t-test on a-b, with a 95 % CI and Cohen's dz."""
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired inputs: {a.shape} vs {b.shape}")
    d = a - b
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return {"n": n, "mean_diff": mean, "t_stat": float("nan"),
                "p_ttest": float("nan"), "ci95_lo": float("nan"),
                "ci95_hi": float("nan"), "cohens_dz": float("nan")}
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, n - 1))
    t_stat, p = stats.ttest_rel(a, b)
    return {
        "n": n, "mean_diff": mean,
        "t_stat": float(t_stat), "p_ttest": float(p),
        "ci95_lo": mean - t_crit * se, "ci95_hi": mean + t_crit * se,
        "cohens_dz": mean / sd if sd > 0 else float("inf"),
        "a_per_seed": [round(float(v), 6) for v in a],
        "b_per_seed": [round(float(v), 6) for v in b],
    }


def holm(tests: list[dict]) -> None:
    """Holm--Bonferroni adjust p_ttest in place, within one family."""
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_ttest"])
    m = len(tests)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * tests[i]["p_ttest"])
        running = max(running, adj)  # enforce monotonicity
        tests[i]["p_holm"] = running


def per_seed_map(path: Path, key_cols: list[str], value_col: str) -> dict:
    """Map key-tuple -> per-training-seed list, from an aggregated curve CSV."""
    df = pd.read_csv(path)
    df = df[df["model"] == MODEL]
    out = {}
    for _, r in df.iterrows():
        key = tuple(r[c] for c in key_cols)
        out[key] = json.loads(r[value_col])
    return out


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    results: dict[str, list[dict]] = {}

    # ---------------- Correlated (contiguous) vs random (scattered) ----------
    corr_dir = repo / "results" / "dropout_study_correlated"
    rand = {
        "standard": repo / "results" / "dropout_study" / "degradation_curves.csv",
        "dropout_aware": repo / "results" / "dropout_study_aware" / "degradation_curves.csv",
    }
    if corr_dir.exists():
        family: list[dict] = []
        corr_by_variant = {}
        for variant in ("standard", "dropout_aware"):
            cpath = corr_dir / f"correlated_curves_{variant}.csv"
            if not cpath.exists() or not rand[variant].exists():
                continue
            corr = per_seed_map(cpath, ["k"], "f1_per_train_seed")
            ref = per_seed_map(rand[variant], ["k"], "f1_per_train_seed")
            corr_by_variant[variant] = corr
            for k in sorted({key[0] for key in corr}):
                if (k,) not in ref or k == 0:
                    continue
                t = paired_ttest(corr[(k,)], ref[(k,)])
                t.update({"family": "correlated", "variant": variant,
                          "comparison": f"contiguous - scattered, k={k}", "k": int(k)})
                family.append(t)

        if "standard" in corr_by_variant and "dropout_aware" in corr_by_variant:
            for k in sorted({key[0] for key in corr_by_variant["standard"]}):
                if k == 0:
                    continue
                t = paired_ttest(corr_by_variant["dropout_aware"][(k,)],
                                 corr_by_variant["standard"][(k,)])
                t.update({"family": "correlated", "variant": "DA vs Std",
                          "comparison": f"DA - Std under contiguous loss, k={k}",
                          "k": int(k)})
                family.append(t)
        if family:
            holm(family)
            results["correlated"] = family

    # ---------------- Measurement noise --------------------------------------
    noise_dir = repo / "results" / "dropout_study_noise"
    if noise_dir.exists():
        family = []
        noise_by_variant = {}
        for variant in ("standard", "dropout_aware"):
            npath = noise_dir / f"noise_curves_{variant}.csv"
            if not npath.exists():
                continue
            grid = per_seed_map(npath, ["sigma_pu", "k"], "f1_per_train_seed")
            noise_by_variant[variant] = grid
            sigmas = sorted({key[0] for key in grid})
            ks = sorted({key[1] for key in grid})
            base = min(sigmas)
            for s, k in product(sigmas, ks):
                if s == base:
                    continue
                t = paired_ttest(grid[(s, k)], grid[(base, k)])
                t.update({"family": "noise", "variant": variant,
                          "comparison": f"sigma={s} - sigma={base}, k={k}",
                          "sigma_pu": float(s), "k": int(k)})
                family.append(t)

        if "standard" in noise_by_variant and "dropout_aware" in noise_by_variant:
            common = set(noise_by_variant["standard"]) & set(noise_by_variant["dropout_aware"])
            for (s, k) in sorted(common):
                t = paired_ttest(noise_by_variant["dropout_aware"][(s, k)],
                                 noise_by_variant["standard"][(s, k)])
                t.update({"family": "noise", "variant": "DA vs Std",
                          "comparison": f"DA - Std at sigma={s}, k={k}",
                          "sigma_pu": float(s), "k": int(k)})
                family.append(t)
        if family:
            holm(family)
            results["noise"] = family

    if not results:
        print("Neither study has produced curves yet — nothing to test.")
        return 1

    out_dir = repo / "results" / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": ("Paired two-sided t-tests over training seeds 42/43/44 "
                 "(n=3, df=2), Holm-Bonferroni per family. Wilcoxon not used "
                 "(p floor 0.25 at n=3)."),
        "model": MODEL,
        "families": results,
    }
    (out_dir / "new_policy_significance.json").write_text(json.dumps(payload, indent=2))

    flat = [t for fam in results.values() for t in fam]
    pd.DataFrame(flat).to_csv(out_dir / "new_policy_significance.csv", index=False)

    for fam, tests in results.items():
        print(f"\n=== family: {fam}")
        for t in sorted(tests, key=lambda x: x["p_holm"]):
            print(f"  {t['variant']:<14} {t['comparison']:<40} "
                  f"d={t['mean_diff']:+.3f} "
                  f"CI[{t['ci95_lo']:+.3f},{t['ci95_hi']:+.3f}] "
                  f"p={t['p_ttest']:.4f} p_holm={t['p_holm']:.4f}")
    print(f"\nWrote {out_dir/'new_policy_significance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
