"""Statistical significance analysis for the paper's headline comparisons.

Reviewer request (CHANGES.md item 1): the paper reports mean ± std over three
training seeds but never states whether the differences are significant.  This
script turns the *already-computed* per-seed numbers on disk into an explicit
test battery.  It runs no model and invents no number: every input is read from

  results/dropout_study/degradation_curves.csv        (standard training)
  results/dropout_study_aware/degradation_curves.csv  (dropout-aware training)
  results/dropout_study_sdr/degradation_curves.csv    (SDR-STGNN)
  results/dropout_study_impute/imputation_summary.json (mean-imputation control)
  results/fullscale_final/<model>/seed_000XX/predictions_test.csv (event level)

Two units of analysis are reported, because they answer different questions.

1. **Training seed (n=3), paired.**  The seed is the unit that the paper's
   ±std already uses, and it is the unit a reviewer cares about: would a
   re-trained model show the same gap?  Every comparison is a paired
   two-sided t-test on the three seed-wise differences (df=2), reported with
   the mean difference and its 95% CI (t_{0.975,2}=4.303).  Holm-Bonferroni
   controls the family-wise error rate across the primary family.
   Wilcoxon signed-rank is *reported but not relied on*: with n=3 its smallest
   attainable two-sided p-value is 0.25, so it cannot reject at any
   conventional level regardless of effect size.  That is a property of the
   sample size, not of the data, and the script states it explicitly.

2. **Test event (n=6175), paired cluster bootstrap.**  The 247,000 test
   windows come from 6,175 simulated events (40 windows each), so windows are
   *not* independent; the bootstrap resamples whole events.  This quantifies
   how much of the observed spread is test-set sampling noise as opposed to
   training-seed noise, and is only available at k=0, where per-window
   predictions were saved.  Its role is to show which of the two sources
   dominates, not to replace the seed-level test.

Outputs
  results/statistics/significance_tests.json  (machine-readable, all fields)
  results/statistics/significance_tests.csv   (one row per test)
  paper/tables/significance.tex               (the compact table for the paper)

Usage
  uv run python scripts/run_significance_tests.py
  uv run python scripts/run_significance_tests.py --bootstrap-reps 10000
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
from sklearn.metrics import f1_score  # type: ignore

K_LEVELS = [0, 1, 2, 5, 10]
SEEDS = [42, 43, 44]
WINDOWS_PER_EVENT = 40

MODEL_LABEL = {
    "gru": "GRU",
    "rgcn": "RGCN",
    "rgsage_mean": "RGSAGE-Mean",
    "rgsage_max": "RGSAGE-Max",
    "rgatv2": "RGATv2",
    "sdr_stgnn": "SDR-STGNN",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_curve(path: Path) -> dict[tuple[str, int], list[float]]:
    """Read a degradation_curves.csv into {(model, k): [f1 per training seed]}.

    ``f1_per_train_seed`` is stored as a stringified Python list, and its order
    matches the ``seeds`` column (42/43/44) written by run_dropout_study.py.
    """
    df = pd.read_csv(path)
    out: dict[tuple[str, int], list[float]] = {}
    for _, row in df.iterrows():
        per_seed = ast.literal_eval(str(row["f1_per_train_seed"]))
        seeds = ast.literal_eval(str(row["seeds"]))
        if list(seeds) != SEEDS:
            raise ValueError(f"{path}: unexpected seed order {seeds}")
        out[(str(row["model"]), int(row["k"]))] = [float(v) for v in per_seed]
    return out


def load_impute(path: Path) -> dict[tuple[str, int], list[float]]:
    """Read imputation_summary.json into the same {(model, k): [f1/seed]} shape."""
    blob = json.loads(path.read_text())
    per_seed = blob["per_seed_f1"]
    model = str(blob["model"])
    out: dict[tuple[str, int], list[float]] = {}
    for k in blob["k_levels"]:
        out[(model, int(k))] = [float(per_seed[str(s)][str(k)]) for s in blob["seeds"]]
    return out


def audc(f1_by_k: dict[int, float], ks: Sequence[int]) -> float:
    """Normalized area under the retention curve R(k)=F1(k)/F1(0).

    Identical to ``audc`` in run_imputation_baseline.py and ``_audc`` in
    build_sdr_result_figures.py, so the values here match the paper's tables.
    """
    r = [f1_by_k[k] / f1_by_k[ks[0]] for k in ks]
    area = sum((r[i] + r[i + 1]) / 2 * (ks[i + 1] - ks[i]) for i in range(len(ks) - 1))
    return area / (ks[-1] - ks[0])


def per_seed_audc(curve: dict[tuple[str, int], list[float]], model: str) -> list[float]:
    """AUDC computed *per seed* (then averaged elsewhere), matching the tables."""
    return [
        audc({k: curve[(model, k)][i] for k in K_LEVELS}, K_LEVELS)
        for i in range(len(SEEDS))
    ]


def per_seed_retention(
    curve: dict[tuple[str, int], list[float]], model: str, k: int = 10
) -> list[float]:
    """Retention F1(k)/F1(0) in percent, per seed."""
    return [
        100.0 * curve[(model, k)][i] / curve[(model, 0)][i] for i in range(len(SEEDS))
    ]


# --------------------------------------------------------------------------
# seed-level paired tests
# --------------------------------------------------------------------------
T_CRIT_95_DF2 = float(stats.t.ppf(0.975, 2))


def paired_seed_test(
    name: str,
    metric: str,
    a_label: str,
    b_label: str,
    a: Sequence[float],
    b: Sequence[float],
    family: str,
) -> dict[str, Any]:
    """Paired two-sided t-test of a-b over training seeds, with 95% CI.

    Also carries the Wilcoxon signed-rank p-value for completeness, flagged as
    uninformative at n=3 (its floor is 0.25 two-sided).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"{name}: shape mismatch {a_arr.shape} vs {b_arr.shape}")
    d = a_arr - b_arr
    n = d.size
    mean_d = float(d.mean())
    # ddof=1: sample std of the differences, the t-test's denominator.
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    t_stat, p_t = stats.ttest_rel(a_arr, b_arr)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    if se > 0:
        ci = (mean_d - tcrit * se, mean_d + tcrit * se)
    else:
        # All three differences identical: the t-test is degenerate.  Report the
        # point estimate as its own interval rather than a fabricated width.
        ci = (mean_d, mean_d)
    try:
        p_w = float(stats.wilcoxon(a_arr, b_arr, zero_method="wilcox").pvalue)
    except ValueError:
        p_w = float("nan")
    # Cohen's d_z for paired designs; an effect-size companion to the p-value.
    dz = mean_d / sd if sd > 0 else float("inf")
    return {
        "test": name,
        "family": family,
        "unit": "training seed",
        "n": int(n),
        "metric": metric,
        "variant_a": a_label,
        "variant_b": b_label,
        "a_per_seed": [float(v) for v in a_arr],
        "b_per_seed": [float(v) for v in b_arr],
        "a_mean": float(a_arr.mean()),
        "b_mean": float(b_arr.mean()),
        "mean_diff": mean_d,
        "sd_diff": sd,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "p_wilcoxon_uninformative": p_w,
        "cohens_dz": float(dz),
    }


def holm(rows: list[dict[str, Any]], family: str) -> None:
    """Annotate a family of tests in place with Holm-Bonferroni adjusted p-values."""
    members = [r for r in rows if r["family"] == family]
    order = sorted(range(len(members)), key=lambda i: members[i]["p_ttest"])
    m = len(members)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * members[idx]["p_ttest"])
        # Holm's adjusted p-values must be monotone non-decreasing in raw p.
        running = max(running, adj)
        members[idx]["p_holm"] = running
        members[idx]["holm_family_size"] = m
        members[idx]["significant_holm_0p05"] = bool(running < 0.05)


# --------------------------------------------------------------------------
# event-level cluster bootstrap (k=0 only)
# --------------------------------------------------------------------------
def event_key(window_ids: pd.Series) -> pd.Series:
    """Strip the trailing ``_wNNNN`` so all windows of one simulated event share a key."""
    return window_ids.str.rsplit("_w", n=1).str[0]


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (truth, prediction, event_index) for one run's test predictions."""
    df = pd.read_csv(path, usecols=["window_id", "truth", "vote_prediction"])
    codes, _ = pd.factorize(event_key(df["window_id"]))
    return (
        df["truth"].to_numpy(np.int16),
        df["vote_prediction"].to_numpy(np.int16),
        codes.astype(np.int32),
    )


def macro_f1_fast(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 via a bincount confusion matrix.

    Equivalent to ``sklearn.metrics.f1_score(average="macro", zero_division=0)``
    but ~2 orders of magnitude cheaper, which is what makes a 247k-row metric
    affordable inside a few-thousand-replicate bootstrap loop.  Verified against
    sklearn on the unresampled data before the loop runs.
    """
    cm = np.bincount(
        y.astype(np.int64) * n_classes + pred.astype(np.int64),
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    true_pos = cm.sum(axis=1).astype(np.float64)
    denom = pred_pos + true_pos
    # zero_division=0: a class with no predictions and no support contributes 0,
    # matching sklearn's convention, and macro averaging divides by all classes.
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def cluster_bootstrap_delta_f1(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    events: np.ndarray,
    reps: int,
    rng: np.random.Generator,
    n_classes: int = 26,
) -> dict[str, Any]:
    """Paired cluster bootstrap on macro-F1(a) - macro-F1(b), resampling events.

    Windows within an event share a fault scenario and are strongly dependent,
    so resampling windows rather than events would understate the interval by
    roughly sqrt(windows per event).
    """
    counts = np.bincount(events)
    n_events = counts.size
    if not np.all(counts == counts[0]):
        raise ValueError(
            f"events have unequal sizes (min {counts.min()}, max {counts.max()}); "
            "the reshape fast path below assumes a fixed windows-per-event"
        )
    per_event = int(counts[0])
    # Predictions are dumped in loader order, which interleaves events in chunks of
    # a few windows, so rows of one event are not contiguous.  A stable sort by
    # event id groups them without disturbing within-event order, and the sorted
    # positions then reshape cleanly into one row per event.  Assert the result
    # rather than trust it: a resample built on the wrong rows would silently
    # produce a plausible but meaningless interval.
    order = np.argsort(events, kind="stable")
    if not np.array_equal(events[order], np.repeat(np.arange(n_events), per_event)):
        raise ValueError("event ids do not group into equal contiguous blocks")
    index_matrix = order.reshape(n_events, per_event)

    sk = f1_score(y, pred_a, average="macro", zero_division=0)
    fast = macro_f1_fast(y, pred_a, n_classes)
    if not np.isclose(sk, fast, atol=1e-12):
        raise AssertionError(f"macro_f1_fast disagrees with sklearn: {fast} vs {sk}")

    point = float(fast - macro_f1_fast(y, pred_b, n_classes))
    deltas = np.empty(reps, dtype=float)
    for r in range(reps):
        pick = rng.integers(0, n_events, size=n_events)
        idx = index_matrix[pick].ravel()
        ys = y[idx]
        deltas[r] = macro_f1_fast(ys, pred_a[idx], n_classes) - macro_f1_fast(
            ys, pred_b[idx], n_classes
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p: proportion of resamples on the far side of zero,
    # doubled, with the standard +1 continuity correction.
    tail = min((deltas <= 0).sum(), (deltas >= 0).sum())
    p = min(1.0, 2.0 * (tail + 1) / (reps + 1))
    return {
        "n_events": n_events,
        "reps": reps,
        "delta_f1": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_se": float(deltas.std(ddof=1)),
        "p_bootstrap": float(p),
    }


# --------------------------------------------------------------------------
# LaTeX table
# --------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<\\!0.001$"
    return f"${p:.3f}$"


def build_latex(rows: list[dict[str, Any]], selection: list[str]) -> str:
    """Compact significance table: one row per selected test, in the given order.

    Kept to a single column (five columns, \\scriptsize, short labels) because the
    paper is at its page cap; the $t$ statistic is omitted as $\\Delta$ and its CI
    already carry it, and is available in the JSON.
    """
    by_name = {r["test"]: r for r in rows}
    lines: list[str] = []
    for name in selection:
        r = by_name[name]
        # Retention differences are percentage points, where three decimals would
        # be spurious precision; F1 and AUDC differences keep three.
        dec = 1 if "retention" in r["metric"] else 3
        lines.append(
            f"    {r['tex_label']} & ${r['mean_diff']:+.{dec}f}$ "
            f"& $[{r['ci95_low']:+.{dec}f},\\,{r['ci95_high']:+.{dec}f}]$ "
            f"& {fmt_p(r['p_ttest'])} & {fmt_p(r['p_holm'])} \\\\"
        )
    body = "\n".join(lines)
    return (
        "% Generated by scripts/run_significance_tests.py -- do not edit by hand.\n"
        "% Source: results/statistics/significance_tests.json\n"
        "\\begin{table}[!tb]\n"
        "  \\centering\n"
        "  \\caption{Paired two-sided $t$-tests over training seeds 42/43/44\n"
        "           ($n=3$, $\\mathrm{df}=2$) on the mitigation claims, all on the\n"
        "           RGSAGE-Max backbone. $\\Delta$ is the mean seed-wise difference in\n"
        "           the stated metric (F1 points, AUDC, or percentage points for\n"
        "           Ret.@10); $p_{\\text{Holm}}$ is Holm-Bonferroni adjusted over all\n"
        "           six tests, the two expected null results included. Std\\ =\\ standard\n"
        "           training, DA\\ =\\ dropout-aware.}\n"
        "  \\label{tab:significance}\n"
        "  \\scriptsize\n"
        "  \\setlength{\\tabcolsep}{3pt}\n"
        "  \\begin{tabular}{@{}lcccc@{}}\n"
        "    \\toprule\n"
        "    Comparison & $\\Delta$ & $95\\%$ CI & $p$ & $p_{\\text{Holm}}$ \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output-dir", default="results/statistics")
    ap.add_argument("--tex-out", default="paper/tables/significance.tex")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Seed-level tests only (skips reading the 247k-row prediction CSVs).",
    )
    args = ap.parse_args()

    root = Path(args.results_root)
    std = load_curve(root / "dropout_study" / "degradation_curves.csv")
    aware = load_curve(root / "dropout_study_aware" / "degradation_curves.csv")
    sdr = load_curve(root / "dropout_study_sdr" / "degradation_curves.csv")
    imp = load_impute(root / "dropout_study_impute" / "imputation_summary.json")

    rows: list[dict[str, Any]] = []

    # ---- Family 1: the mitigation claims (the paper's contribution) --------
    #
    # This family is deliberately confined to the six statements the paper makes
    # about *mitigations* on the shared RGSAGE-Max backbone, including the two it
    # asserts as null results (no clean-accuracy cost; the gate does not beat the
    # augmentation).  The nulls are inside the Holm family rather than exempted
    # from it, so the correction is not applied selectively to the comparisons
    # that happen to reject.  Architecture comparisons live in their own families
    # below: they are context for this work, not the claim being defended, and
    # pooling them here would only dilute the correction.
    fam = "primary"
    primary_specs: list[tuple[str, str, str, str, Sequence[float], Sequence[float], str]] = [
        (
            "aware_vs_std_k0",
            "graph macro-F1 at $k=0$",
            "Dropout-aware",
            "Standard",
            aware[("rgsage_max", 0)],
            std[("rgsage_max", 0)],
            "DA vs.\\ Std, clean F1 ($k{=}0$)",
        ),
        (
            "aware_vs_std_k10",
            "graph macro-F1 at $k=10$",
            "Dropout-aware",
            "Standard",
            aware[("rgsage_max", 10)],
            std[("rgsage_max", 10)],
            "DA vs.\\ Std, F1 at $k{=}10$",
        ),
        (
            "aware_vs_std_ret10",
            "retention at $k=10$ (\\%)",
            "Dropout-aware",
            "Standard",
            per_seed_retention(aware, "rgsage_max"),
            per_seed_retention(std, "rgsage_max"),
            "DA vs.\\ Std, Ret.@10",
        ),
        (
            "aware_vs_std_audc",
            "AUDC",
            "Dropout-aware",
            "Standard",
            per_seed_audc(aware, "rgsage_max"),
            per_seed_audc(std, "rgsage_max"),
            "DA vs.\\ Std, AUDC",
        ),
        (
            "aware_vs_impute_k10",
            "graph macro-F1 at $k=10$",
            "Dropout-aware",
            "Mean-imputation",
            aware[("rgsage_max", 10)],
            imp[("rgsage_max", 10)],
            "DA vs.\\ imputation, F1 at $k{=}10$",
        ),
        (
            "aware_vs_sdr_audc",
            "AUDC",
            "Dropout-aware",
            "SDR-STGNN",
            per_seed_audc(aware, "rgsage_max"),
            per_seed_audc(sdr, "sdr_stgnn"),
            "DA vs.\\ SDR-STGNN, AUDC",
        ),
    ]
    for name, metric, a_label, b_label, a_vals, b_vals, tex_label in primary_specs:
        rows.append(
            paired_seed_test(name, metric, a_label, b_label, a_vals, b_vals, fam)
        )
        rows[-1]["tex_label"] = tex_label
    holm(rows, fam)
    latex_selection = [r["test"] for r in rows if r["family"] == fam]

    # ---- Family 2: dropout-aware gain at every k (RGSAGE-Max) -------------
    fam = "aware_vs_std_by_k"
    for k in K_LEVELS:
        rows.append(
            paired_seed_test(
                f"bykk_aware_vs_std_k{k}",
                f"graph macro-F1 at $k={k}$",
                "Dropout-aware",
                "Standard",
                aware[("rgsage_max", k)],
                std[("rgsage_max", k)],
                fam,
            )
        )
    holm(rows, fam)

    # ---- Family 3: cross-architecture AUDC gain from dropout-aware -------
    fam = "aware_vs_std_by_arch"
    for m in ["rgsage_max", "rgsage_mean", "rgcn", "rgatv2", "gru"]:
        rows.append(
            paired_seed_test(
                f"aware_vs_std_audc_{m}",
                "AUDC",
                f"{MODEL_LABEL[m]} dropout-aware",
                f"{MODEL_LABEL[m]} standard",
                per_seed_audc(aware, m),
                per_seed_audc(std, m),
                fam,
            )
        )
    holm(rows, fam)

    # ---- Family 4: SDR-STGNN vs dropout-aware at every k -----------------
    fam = "sdr_vs_aware_by_k"
    for k in K_LEVELS:
        rows.append(
            paired_seed_test(
                f"sdr_vs_aware_k{k}",
                f"graph macro-F1 at $k={k}$",
                "SDR-STGNN",
                "Dropout-aware",
                sdr[("sdr_stgnn", k)],
                aware[("rgsage_max", k)],
                fam,
            )
        )
    holm(rows, fam)

    # ---- Family 5: mean-imputation vs standard zeroing -------------------
    # k=0 is excluded: with no dropped node there is nothing to impute, so the
    # two variants are the same forward pass by construction and any nonzero
    # difference is only the 6-decimal rounding of f1_per_train_seed on disk.
    fam = "impute_vs_std_by_k"
    for k in [k for k in K_LEVELS if k > 0]:
        rows.append(
            paired_seed_test(
                f"impute_vs_std_k{k}",
                f"graph macro-F1 at $k={k}$",
                "Mean-imputation",
                "Standard",
                imp[("rgsage_max", k)],
                std[("rgsage_max", k)],
                fam,
            )
        )
    holm(rows, fam)

    # ---- Family 5b: the clean-accuracy/robustness inversion --------------
    # The abstract's "the most accurate architecture degrades the fastest" is a
    # claim about RGSAGE-Max relative to the other GNNs, and it can be read two
    # ways: absolute macro-F1 at k=10, or retention (F1(10)/F1(0)).  Both are
    # tested here, against every other GNN, because they do not agree on
    # significance at n=3 and the paper must say which one it is claiming.
    fam = "inversion"
    for m in ["rgcn", "rgatv2", "rgsage_mean"]:
        rows.append(
            paired_seed_test(
                f"inversion_abs_{m}_vs_rgsage_max_k10",
                "graph macro-F1 at $k=10$",
                f"{MODEL_LABEL[m]} (standard)",
                "RGSAGE-Max (standard)",
                std[(m, 10)],
                std[("rgsage_max", 10)],
                fam,
            )
        )
        rows.append(
            paired_seed_test(
                f"inversion_ret_{m}_vs_rgsage_max_k10",
                "retention at $k=10$ (\\%)",
                f"{MODEL_LABEL[m]} (standard)",
                "RGSAGE-Max (standard)",
                per_seed_retention(std, m),
                per_seed_retention(std, "rgsage_max"),
                fam,
            )
        )
    holm(rows, fam)

    # ---- Family 6: architecture ranking at k=0 (adjacent pairs) ----------
    fam = "arch_ranking_k0"
    ranking = ["rgsage_max", "rgsage_mean", "rgcn", "rgatv2", "gru"]
    for hi, lo in zip(ranking, ranking[1:]):
        rows.append(
            paired_seed_test(
                f"arch_{hi}_vs_{lo}_k0",
                "graph macro-F1 at $k=0$",
                MODEL_LABEL[hi],
                MODEL_LABEL[lo],
                std[(hi, 0)],
                std[(lo, 0)],
                fam,
            )
        )
    holm(rows, fam)

    # ---- event-level cluster bootstrap at k=0 ----------------------------
    bootstrap: dict[str, Any] = {}
    if not args.skip_bootstrap:
        rng = np.random.default_rng(args.bootstrap_seed)
        pairs = [
            (
                "aware_vs_std_k0_events",
                root / "dropout_aware" / "rgsage_max" / "seed_00042" / "predictions_test.csv",
                root / "fullscale_final" / "rgsage_max" / "seed_00042" / "predictions_test.csv",
                "Dropout-aware",
                "Standard",
            ),
            (
                "rgsagemax_vs_rgsagemean_k0_events",
                root / "fullscale_final" / "rgsage_max" / "seed_00042" / "predictions_test.csv",
                root / "fullscale_final" / "rgsage_mean" / "seed_00042" / "predictions_test.csv",
                "RGSAGE-Max",
                "RGSAGE-Mean",
            ),
        ]
        for name, path_a, path_b, label_a, label_b in pairs:
            y_a, pa, ev_a = load_predictions(path_a)
            y_b, pb, ev_b = load_predictions(path_b)
            if not np.array_equal(y_a, y_b) or not np.array_equal(ev_a, ev_b):
                raise ValueError(
                    f"{name}: prediction files are not row-aligned; "
                    "the paired bootstrap requires the same test order"
                )
            res = cluster_bootstrap_delta_f1(
                y_a, pa, pb, ev_a, args.bootstrap_reps, rng
            )
            res.update(
                {"seed": 42, "variant_a": label_a, "variant_b": label_b, "k": 0}
            )
            bootstrap[name] = res
            print(
                f"[bootstrap] {name}: dF1 = {res['delta_f1']:+.4f} "
                f"CI95 [{res['ci95_low']:+.4f}, {res['ci95_high']:+.4f}] "
                f"p = {res['p_bootstrap']:.4f} (n_events={res['n_events']})"
            )

    # ---- report ----------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "note": (
            "Paired tests over training seeds 42/43/44. Wilcoxon p-values are "
            "included for completeness but are uninformative at n=3 (floor 0.25 "
            "two-sided); the t-test is the reported test. Cluster bootstrap "
            "resamples whole simulated events (40 windows each)."
        ),
        "seeds": SEEDS,
        "k_levels": K_LEVELS,
        "windows_per_event": WINDOWS_PER_EVENT,
        "t_crit_95_df2": T_CRIT_95_DF2,
        "seed_level_tests": rows,
        "event_level_bootstrap": bootstrap,
    }
    (out_dir / "significance_tests.json").write_text(json.dumps(summary, indent=2))

    flat = pd.DataFrame(
        [
            {k: v for k, v in r.items() if not isinstance(v, list)}
            for r in rows
        ]
    )
    flat.to_csv(out_dir / "significance_tests.csv", index=False)

    tex_path = Path(args.tex_out)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(build_latex(rows, latex_selection))

    print(f"\nWrote {out_dir/'significance_tests.json'}")
    print(f"Wrote {out_dir/'significance_tests.csv'}")
    print(f"Wrote {tex_path}")
    print("\nPrimary family:")
    for r in rows:
        if r["family"] == "primary":
            print(
                f"  {r['test']:38s} d={r['mean_diff']:+.4f} "
                f"CI[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}] "
                f"t={r['t_stat']:+8.2f} p={r['p_ttest']:.5f} "
                f"p_holm={r['p_holm']:.5f}"
            )
    print("\nAll other families:")
    for r in rows:
        if r["family"] != "primary":
            print(
                f"  [{r['family']:22s}] {r['test']:30s} d={r['mean_diff']:+.4f} "
                f"p={r['p_ttest']:.5f} p_holm={r['p_holm']:.5f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
