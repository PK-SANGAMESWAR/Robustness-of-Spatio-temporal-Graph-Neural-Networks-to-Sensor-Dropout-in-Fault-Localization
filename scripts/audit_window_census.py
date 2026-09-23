"""Stage A S0.3 -- window census, the quasi-static check, and the F1 ceiling.

Three things, all cheap, one of them new and consequential.

**Census.**  Window ``w`` covers steps ``[w, w+20)`` and inception is at step 40,
so fault window ``w`` contains ``w - 20`` post-inception samples: exactly one
fault window per event (w = 20) has none, and five (w = 21-25) have 1-5.  The
census is analytic; this confirms it on the corpus.

**Quasi-static check.**  The observation vector of ``01_SPEC.md`` §3 is the
window's *last* timestep, which is exact only if every post-inception sample in a
window sits at the same level.  The fault loop of ``simulation/runner.py`` never
rescales the loads, so it should be -- but the whole operator rests on it, so it
is measured rather than argued.

**Unlabelable-window ceiling (new, ``PC-STGNN-PLAN.md`` §2.6).**  The w = 20
windows carry a fault label while containing zero post-inception samples: they
are physically identical to a healthy window of the same event, and their
content depends only on the load multiplier, which is drawn independently of
where the fault will be.  No model can beat chance on them, so they impose a
hard ceiling on every absolute macro-F1 in both papers -- a ceiling that has
never been quantified.  Two outputs:

* each existing model's accuracy on that subset, which must be at chance.
  Meaningfully above chance would mean **label leakage**, and that is a
  stop-everything finding, not a curiosity;
* the implied 26-class macro-F1 ceiling, computed from the real test labels, for
  a model that is perfect everywhere else and does the best available thing on
  w = 20 (predict no-fault, since that is what those windows look like).

Usage:
    uv run python scripts/audit_window_census.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results/pcstgnn/audit"

FAULT_START_STEP = 40
WINDOW_LENGTH = 20


def post_inception_samples(window_index: int) -> int:
    """How many of window ``w``'s 20 samples are after inception."""
    return max(0, window_index + WINDOW_LENGTH - FAULT_START_STEP)


def census(data_root: Path, split: str) -> dict:
    labels = {r["window_id"]: int(r["label"])
              for r in csv.DictReader((data_root / "processed/labels.csv").open())}
    ids = [x.strip() for x in (data_root / "splits" / f"{split}.txt").read_text().splitlines()
           if x.strip()]
    by_count: Counter[int] = Counter()
    fault_windows = 0
    rows = []
    for wid in ids:
        index = int(wid.rpartition("_w")[2])
        n_post = post_inception_samples(index)
        if labels[wid] != 0:
            fault_windows += 1
            by_count[n_post] += 1
    for n_post in sorted(by_count):
        rows.append({"post_inception_samples": n_post,
                     "n_fault_windows": by_count[n_post],
                     "fraction_of_fault_windows": by_count[n_post] / fault_windows})
    return {
        "split": split,
        "n_windows": len(ids),
        "n_fault_windows": fault_windows,
        "n_nofault_windows": len(ids) - fault_windows,
        "zero_post_inception": by_count.get(0, 0),
        "zero_fraction": by_count.get(0, 0) / fault_windows,
        "one_to_five_fraction": sum(by_count.get(i, 0) for i in range(1, 6)) / fault_windows,
        "rows": rows,
    }


# The post-fault level is reached after about two OpenDSS solves, not one: on the
# worst of 400 events the voltage moves 1.36e-05 p.u. between steps 40 and 41,
# 4.7e-07 between 41 and 42, and is then constant to < 1e-08.  This is *solver
# settling* -- the number of solves the iteration needs after the fault element
# is energised -- and NOT a transient of the feeder.  Never call it a transient
# in the paper: this project does not claim physics-informed temporal dynamics
# (01_SPEC.md 3), and the word invites "so there are dynamics, why not model
# them?".  See PC-STGNN/02_STAGE_A_audit.md S0.3.
SETTLING_BOUND_PU = 2.0e-5
"""Bulk upper bound on the settling, asserted below.

Measured over **all 41,250 events** (2026-09-23): max |V[41] - V[40]| is
1.98e-05 p.u. for 41,247 of them, p99.9 = 1.60e-05, median 1.67e-06.  An earlier
400-event sample reported 1.44e-05 and missed the tail entirely, which is
precisely why this is an assertion and not a recorded number.

It matters because s_ip is the *last* timestep of the window: window w covers
steps [w, w+20), so w = 21 observes step 40 and w = 22 observes step 41 -- the
two unsettled steps.  The bound must therefore stay well below sigma_eps.
"""

SETTLING_OUTLIERS = {
    "55_BC_1p0_0007.npz":   7.09e-3,
    "7_CA_10p0_0002.npz":   6.46e-3,
    "51_ABC_1p0_0000.npz":  6.37e-3,
}
"""The three events -- 0.007 % of the corpus -- that settle 300x slower.

All three are **ungrounded** faults (BC, CA, ABC), and in each the voltage
*recovers* between steps 40 and 41 (e.g. 1.0137 -> 1.0208 p.u. at node 16 of
55_BC_1p0_0007) rather than settling monotonically downward.  That is a control
device -- a regulator tap or a cap bank -- operating in response to the fault
and taking one extra solve, not a numerical settling artefact.

Two consequences, both recorded rather than absorbed:

1. It is almost certainly the same mechanism as the 3.3 % non-monotone sag cells
   seen on the smoke grid (PC-STGNN/03_STAGE_B_operator.md S1.3), so the S1.3
   control-state decision covers both.  Check these three buses there.
2. For these events the w = 21 and w = 22 windows -- 6 windows out of 1.65 M --
   observe a level up to 7e-3 p.u. off the settled one, which is *not* far below
   sigma_eps.  Too rare to matter for training or for any reported metric; listed
   here so a fourth one fails the assertion loudly instead of being averaged in.
"""

SETTLED_START_STEP = FAULT_START_STEP + 2
"""First step at which the post-fault level is flat to < 1e-6 p.u."""


def quasi_static(raw_dir: Path, n_events: int, seed: int = 0,
                 sigma_eps: float | None = None) -> dict:
    """Post-inception flatness, split into solver settling and the settled level.

    Returns the settled spread (steps 42+, the restated < 1e-6 criterion), the
    settling step sizes (40->41 and 41->42), and -- when ``sigma_eps`` is known
    -- the margin between the settling bound and the residual floor.
    """
    files = sorted(raw_dir.glob("*.npz"))
    picks = (files if n_events <= 0
             else Random(seed).sample(files, min(n_events, len(files))))
    spreads, settled, step0, step1 = [], [], [], []
    outliers_seen = {}
    for path in picks:
        with np.load(path, allow_pickle=True) as loaded:
            record = loaded["data"]
        instrumented = record[:, 0, 3:] > 0.5
        post = record[:, FAULT_START_STEP:, :3]
        spreads.append(float((post.max(axis=1) - post.min(axis=1))[instrumented].max()))
        tail = record[:, SETTLED_START_STEP:, :3]
        settled.append(float((tail.max(axis=1) - tail.min(axis=1))[instrumented].max()))
        volts = record[:, :, :3]
        step0.append(float(np.abs(volts[:, FAULT_START_STEP + 1]
                                  - volts[:, FAULT_START_STEP])[instrumented].max()))
        step1.append(float(np.abs(volts[:, FAULT_START_STEP + 2]
                                  - volts[:, FAULT_START_STEP + 1])[instrumented].max()))
        if step0[-1] > SETTLING_BOUND_PU:
            outliers_seen[path.name] = step0[-1]
    arr, settled_arr = np.asarray(spreads), np.asarray(settled)
    step0_arr = np.asarray(step0)
    bulk = step0_arr[step0_arr <= SETTLING_BOUND_PU]
    max_settling = float(max(bulk.max() if bulk.size else 0.0, np.asarray(step1).max()))
    unknown = sorted(set(outliers_seen) - set(SETTLING_OUTLIERS))

    out = {
        "n_events": len(picks),
        "max_spread_pu": float(arr.max()),
        "median_spread_pu": float(np.median(arr)),
        "passes_1e-6": bool(arr.max() < 1e-6),          # the ORIGINAL criterion; expected False
        "settled_start_step": SETTLED_START_STEP,
        "max_settled_spread_pu": float(settled_arr.max()),
        "passes_settled_1e-6": bool(settled_arr.max() < 1e-6),   # the RESTATED criterion
        "max_step_40_to_41_pu": float(np.asarray(step0).max()),
        "max_step_41_to_42_pu": float(np.asarray(step1).max()),
        "max_settling_pu": max_settling,
        "max_settling_pu_including_outliers": float(step0_arr.max()),
        "settling_bound_pu": SETTLING_BOUND_PU,
        "passes_settling_bound": bool(max_settling <= SETTLING_BOUND_PU),
        "known_outliers_seen": {k: float(v) for k, v in sorted(outliers_seen.items())
                                if k in SETTLING_OUTLIERS},
        "unknown_outliers": unknown,
        "passes_no_new_outliers": not unknown,
    }
    if sigma_eps is not None:
        out["sigma_eps"] = float(sigma_eps)
        out["settling_margin_vs_sigma_eps"] = float(sigma_eps) / max_settling
        out["passes_settling_below_sigma_eps"] = bool(max_settling * 10.0 <= float(sigma_eps))

    # Assertions, not prose (PC-STGNN/02_STAGE_A_audit.md S0.3).
    assert out["passes_settled_1e-6"], (
        f"post-fault level is not flat from step {SETTLED_START_STEP}: "
        f"max settled spread {settled_arr.max():.3e} p.u. >= 1e-6")
    assert out["passes_settling_bound"], (
        f"settling {max_settling:.3e} p.u. exceeds the recorded bulk bound "
        f"{SETTLING_BOUND_PU:.3e} -- windows w = 21-22 observe the unsettled level, "
        f"so 01_SPEC.md 3's last-timestep observation must be revisited")
    assert out["passes_no_new_outliers"], (
        f"control-driven settling outliers not in the documented set of 3: {unknown} "
        f"-- these settle up to 300x slower than the corpus bulk (see "
        f"SETTLING_OUTLIERS); investigate the control state before trusting the "
        f"observation vector on their w = 21/22 windows")
    if sigma_eps is not None:
        assert out["passes_settling_below_sigma_eps"], (
            f"solver settling {max_settling:.3e} p.u. is not an order below "
            f"sigma_eps {sigma_eps:.3e} -- the observation vector is no longer exact "
            f"at the residual's own scale")
    return out


def _macro_f1(truth: np.ndarray, pred: np.ndarray, n_classes: int = 26) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wid, truth, pred = [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            wid.append(int(row["window_id"].rpartition("_w")[2]))
            truth.append(int(row["truth"]))
            pred.append(int(row["vote_prediction"]))
    return np.asarray(wid), np.asarray(truth), np.asarray(pred)


def ceiling(truth: np.ndarray, windows: np.ndarray) -> dict:
    """Macro-F1 of a model that is perfect except on the unlabelable windows."""
    unlabelable = windows == FAULT_START_STEP - WINDOW_LENGTH        # w = 20
    oracle = truth.copy()
    oracle[unlabelable] = 0                       # best available: looks healthy
    rng = np.random.default_rng(0)
    uniform = truth.copy()
    uniform[unlabelable] = rng.integers(0, 26, size=int(unlabelable.sum()))
    return {
        "n_unlabelable": int(unlabelable.sum()),
        "fraction_of_test": float(unlabelable.mean()),
        "ceiling_predict_nofault": _macro_f1(truth, oracle),
        "ceiling_uniform_guess": _macro_f1(truth, uniform),
    }


def model_runs(results_root: Path) -> list[tuple[str, Path]]:
    out = []
    for family in ("fullscale_final", "dropout_aware", "sdr_stgnn"):
        base = results_root / family
        if not base.exists():
            continue
        for model_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "figures"):
            for seed_dir in sorted(model_dir.glob("seed_*")):
                pred = seed_dir / "predictions_test.csv"
                if pred.exists():
                    out.append((f"{family}/{model_dir.name}/{seed_dir.name}", pred))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--results-root", default=str(REPO / "results"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-quasi-static", type=int, default=100,
                    help="events to scan for the settling checks; <= 0 scans all "
                         "41,250 (15 s, and the only way to see the 3 outliers)")
    ap.add_argument("--sigma-eps", type=float, default=None,
                    help="residual floor, once S1.2 has set it; asserts the solver "
                         "settling stays an order of magnitude below it")
    ap.add_argument("--max-models", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cen = census(root, args.split)
    qs = quasi_static(root / "raw_sim", args.n_quasi_static, sigma_eps=args.sigma_eps)

    runs = model_runs(Path(args.results_root))
    if args.max_models:
        runs = runs[:args.max_models]
    per_model = []
    ceil = None
    for name, path in runs:
        windows, truth, pred = read_predictions(path)
        if ceil is None:
            ceil = ceiling(truth, windows)
        sub = windows == FAULT_START_STEP - WINDOW_LENGTH
        per_model.append({
            "run": name,
            "n_unlabelable": int(sub.sum()),
            "accuracy_on_unlabelable": float((pred[sub] == truth[sub]).mean()),
            "fraction_predicted_nofault": float((pred[sub] == 0).mean()),
            "overall_macro_f1": _macro_f1(truth, pred),
            "macro_f1_excluding_unlabelable": _macro_f1(truth[~sub], pred[~sub]),
        })

    chance = 1.0 / 26
    worst = max((m["accuracy_on_unlabelable"] for m in per_model), default=0.0)
    leakage = worst > 3 * chance
    passed = (abs(cen["zero_fraction"] - 0.05) < 1e-9
              and abs(cen["one_to_five_fraction"] - 0.25) < 1e-9
              and qs["passes_settled_1e-6"] and qs["passes_settling_bound"]
              and not leakage)

    with (out_dir / "window_census.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["post_inception_samples", "n_fault_windows",
                                           "fraction_of_fault_windows"])
        w.writeheader()
        w.writerows(cen["rows"])
    with (out_dir / "unlabelable_window_f1.csv").open("w", newline="") as fh:
        if per_model:
            w = csv.DictWriter(fh, fieldnames=list(per_model[0]))
            w.writeheader()
            w.writerows(per_model)
    (out_dir / "window_census.json").write_text(json.dumps(
        {"pass": passed, "census": cen, "quasi_static": qs,
         "ceiling": ceil, "per_model": per_model, "chance": chance}, indent=2) + "\n")

    _figure(cen, out_dir)

    print(f"census: {cen['n_fault_windows']} fault windows; "
          f"zero post-inception {cen['zero_fraction']:.4f} (expect 0.05); "
          f"1-5 samples {cen['one_to_five_fraction']:.4f} (expect 0.25)")
    print(f"quasi-static: max within-event post-fault spread "
          f"{qs['max_spread_pu']:.3e} p.u. over {qs['n_events']} events; "
          f"settled (step {qs['settled_start_step']}+) {qs['max_settled_spread_pu']:.3e} "
          f"p.u. (< 1e-6: {qs['passes_settled_1e-6']}); solver settling "
          f"{qs['max_settling_pu']:.3e} p.u. (bound {qs['settling_bound_pu']:.1e}: "
          f"{qs['passes_settling_bound']}); control-driven outliers seen: "
          f"{len(qs['known_outliers_seen'])} known, {len(qs['unknown_outliers'])} new")
    if ceil:
        print(f"unlabelable windows: {ceil['n_unlabelable']} "
              f"({ceil['fraction_of_test']*100:.2f}% of test)")
        print(f"macro-F1 CEILING = {ceil['ceiling_predict_nofault']:.4f} "
              f"(uniform-guess variant {ceil['ceiling_uniform_guess']:.4f})")
    print(f"chance accuracy on that subset = {chance:.4f}; "
          f"worst model = {worst:.4f}  -> leakage: {leakage}")
    for m in per_model:
        print(f"  {m['run']}: acc@w20={m['accuracy_on_unlabelable']:.4f} "
              f"pred_nofault={m['fraction_predicted_nofault']:.3f} "
              f"F1={m['overall_macro_f1']:.4f} -> {m['macro_f1_excluding_unlabelable']:.4f} "
              f"excluding them")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def _figure(cen: dict, out_dir: Path) -> None:
    """F-A1 -- window census histogram (appendix)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    counts = {r["post_inception_samples"]: r["n_fault_windows"] for r in cen["rows"]}
    xs = sorted(counts)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    colors = ["#c0392b" if x == 0 else ("#e67e22" if x <= 5 else "#2c6fbb") for x in xs]
    ax.bar(xs, [counts[x] for x in xs], color=colors)
    ax.set_xlabel("post-inception samples in the window")
    ax.set_ylabel("fault-labelled windows")
    ax.set_title("Window census, test split")
    ax.set_ylim(0, max(counts.values()) * 1.35)
    ax.annotate("unlabelable\n(w = 20)", xy=(0, counts.get(0, 0)),
                xytext=(2.5, max(counts.values()) * 1.05), color="#c0392b", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    fig.tight_layout()
    fig.savefig(out_dir / "F-A1_window_census.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
