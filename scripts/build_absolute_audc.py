"""Absolute-scale companion to the AUDC of the degradation curves.

AUDC as the paper defines it normalizes every point by that model's own clean
score, which is the right metric for the question "how gracefully does this
model degrade" but has a reporting hazard: a model with a depressed clean score
can post a high retention while being useless in absolute terms.  RGATv2 is
exactly that case here — the weakest GNN at k=0 and, on the relative metric,
the most robust.

This script recomputes the same trapezoidal integral **without** the
normalization, giving a mean macro-F1 across the sweep (``AUDC_abs``) on the
same 0-1 scale as macro-F1 itself.  Reading the two side by side makes the
relative metric's normalization explicit instead of leaving it to a caption.

It also emits the degenerate reference the degradation curves need to be read
against: a classifier that always answers "no fault" scores 2*0.5*1/(0.5+1) =
0.667 on class 0 and 0 on the 25 fault classes, so its 26-class macro-F1 is
0.667/26 = 0.0256.  A model at that level has no localization ability at all,
which is not visible from a retention percentage.

Reads only existing CSVs; writes results/statistics/absolute_audc.csv.

Usage:
    uv run python scripts/build_absolute_audc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# A degenerate "always predict no-fault" classifier on the balanced corpus:
# precision 0.5, recall 1.0 on class 0 -> F1 = 2/3; zero on the 25 fault classes.
TRIVIAL_NOFAULT_MACRO_F1 = (2 * 0.5 * 1.0 / 1.5) / 26

CURVES = {
    "standard": "results/dropout_study/degradation_curves.csv",
    "dropout_aware": "results/dropout_study_aware/degradation_curves.csv",
    "sdr_stgnn": "results/dropout_study_sdr/degradation_curves.csv",
    # Non-deep, non-graph reference on the identical corpus, splits, and mask
    # seeds.  Single fit rather than three training seeds, so its rows carry no
    # seed error bar; see scripts/run_classical_baseline.py.
    "classical": "results/classical_baseline/classical_curves.csv",
}


def normalized_trapezoid(ks: np.ndarray, values: np.ndarray) -> float:
    """Mean of ``values`` over ``k`` by trapezoidal integration.

    Matches Eq. (8) of the paper: the integral over the sweep divided by
    ``k_max - k_min``.  With normalized values this is AUDC; with raw macro-F1
    it is the absolute-scale counterpart.
    """
    order = np.argsort(ks)
    ks, values = ks[order], values[order]
    return float(np.trapezoid(values, ks) / (ks[-1] - ks[0]))


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    rows: list[dict] = []

    for variant, rel in CURVES.items():
        path = repo / rel
        if not path.exists():
            print(f"[skip] {rel} not found")
            continue
        df = pd.read_csv(path)
        for model, g in df.groupby("model", sort=False):
            g = g.sort_values("k")
            ks = g["k"].to_numpy(dtype=float)
            f1 = g["f1_mean"].to_numpy(dtype=float)

            # Per-training-seed curves, so the absolute metric carries the same
            # kind of error bar as the relative one rather than a point value.
            # The classical baseline is a single fit and instead records
            # per-mask-seed values, which give a spread but not a seed spread.
            col = ("f1_per_train_seed" if "f1_per_train_seed" in g.columns
                   else "f1_per_mask_seed")
            per_seed = [json.loads(s) for s in g[col]]
            n_seeds = min(len(p) for p in per_seed)
            abs_per_seed, rel_per_seed = [], []
            for i in range(n_seeds):
                curve = np.array([p[i] for p in per_seed], dtype=float)
                abs_per_seed.append(normalized_trapezoid(ks, curve))
                rel_per_seed.append(normalized_trapezoid(ks, curve / curve[0]))

            rows.append({
                "variant": variant,
                "model": model,
                "f1_clean": float(f1[0]),
                "f1_at_k10": float(f1[-1]),
                "audc_rel": float(np.mean(rel_per_seed)),
                "audc_rel_std": float(np.std(rel_per_seed, ddof=1)) if n_seeds > 1 else 0.0,
                "audc_abs": float(np.mean(abs_per_seed)),
                "audc_abs_std": float(np.std(abs_per_seed, ddof=1)) if n_seeds > 1 else 0.0,
                "audc_abs_per_seed": json.dumps([round(v, 6) for v in abs_per_seed]),
                "n_train_seeds": n_seeds,
                "trivial_nofault_macro_f1": round(TRIVIAL_NOFAULT_MACRO_F1, 6),
                "f1_at_k10_over_trivial": float(f1[-1] / TRIVIAL_NOFAULT_MACRO_F1),
            })

    if not rows:
        print("Nothing computed — no degradation curves found.")
        return 1

    out = pd.DataFrame(rows).sort_values(["variant", "audc_abs"], ascending=[True, False])
    out_path = repo / "results" / "statistics" / "absolute_audc.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Trivial 'always no-fault' 26-class macro-F1 = "
          f"{TRIVIAL_NOFAULT_MACRO_F1:.4f}\n")
    for variant in out["variant"].unique():
        print(f"--- {variant}")
        sub = out[out["variant"] == variant]
        print(f"  {'model':<13} {'F1(0)':>6} {'F1(10)':>7} {'AUDC_rel':>9} "
              f"{'AUDC_abs':>9} {'x trivial @k=10':>16}")
        for _, r in sub.iterrows():
            print(f"  {r['model']:<13} {r['f1_clean']:>6.3f} {r['f1_at_k10']:>7.3f} "
                  f"{r['audc_rel']:>9.3f} {r['audc_abs']:>9.3f} "
                  f"{r['f1_at_k10_over_trivial']:>15.1f}x")
        print()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
