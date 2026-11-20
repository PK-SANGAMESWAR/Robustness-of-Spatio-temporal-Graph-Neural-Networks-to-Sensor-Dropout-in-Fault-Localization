"""Build the Stage G tables from the non-sealed splits (CPU only; never opens the sealed set).

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/build_stage_g_tables.py [--n-boot 10000] [--skip-t13]

Writes ``results/stage_g/tables/{T5_main_test,T7_ablation_val,T10_noise_policies_test,T11_mismatch_val,
T13_statistics_test}.{csv,md}``, ``T13_bootstrap_test.json`` (raw bootstrap output) and ``manifest.json``
(git head, time, sha256 of every input). Logic and tests: ``fault_localization/stage_g/tables.py``,
``tests/test_stage_g_tables.py``. The sealed-set table T8 is built by ``scripts/build_stage_g_t8.py`` after the
one sealed run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/stage_g/tables"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--skip-t13", action="store_true")
    ap.add_argument("--reuse-bootstrap", action="store_true",
                    help="rebuild T13 rows from results/stage_g/tables/T13_bootstrap_test.json instead of resampling")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    from fault_localization.stage_g import tables as T

    OUT.mkdir(parents=True, exist_ok=True)
    cons = T.r0n_metric_consistency()
    if max(r["abs_diff"] for r in cons) > 1e-9:
        print("STOP: R0n metric-consistency check failed", cons, file=sys.stderr)
        return 2

    t5 = T.build_t5()
    T.write_table(t5, ["model", "n_seeds", "k=0", "k=1", "k=2", "k=5", "k=10", "Ret@10", "AUDC rel", "AUDC abs", "protocol"],
                  OUT / "T5_main_test", "T5 — main results, test split (247,000 windows), random dropout, σ = 0",
                  ["Soft-voted 26-class macro-F1, mean ± sample std over seeds 42/43/44; 1-seed rows have no std.",
                   "AUDC: Paper 1 Eq. 8, per training seed, then averaged; relative = on F1(k)/F1(0); absolute = on raw F1(k).",
                   "Stage G rows: same mask/noise seeds for every model (3 offsets). Paper-1 rows: Paper-1 evaluator, same split "
                   "and metric; at k = 0 the two evaluators agree (R0ⁿ check: |Δ| < 1e-9 on validation); for k > 0 the mask "
                   "streams differ (same distribution).",
                   "PSM is read from the R5 scorer (physics-only prediction). R6ⁱ was scored at k ∈ {0, 10} only.",
                   "B2ᵃ: the best-validation checkpoint is epoch 1 of 30; validation F1 then varies between 0.46 and 0.67 "
                   "(metrics_by_epoch.csv) with a train loss near 100 against 2.6 for R5. It is the epoch-1 checkpoint of an "
                   "unstable run, so it does not show how well a regressed R̂ can work.",
                   "Ceiling: 0.9744 (unlabelable w = 20 windows, 11_METRICS § 1). Trivial always-no-fault: 0.0256."])

    t7 = T.build_t7()
    cols7 = ["arm", "n_seeds"] + [c[3] for c in T.T7_CELLS]
    T.write_table(t7, cols7, OUT / "T7_ablation_val", "T7 — core ablation, validation split (248,000 windows)",
                  ["Fused soft-voted macro-F1 (Gate-4 scorer), mean ± sample std over the seeds listed; 3 offsets per cell.",
                   "R0: only the training-loop validation F1 at k = 0, σ = 0 exists (R0 was not run through the noise grid).",
                   "R0ⁿ − R0 is what noise augmentation alone buys (12_CHECKLISTS § A, pairing trap).",
                   "R6 (trained) collapsed twice and is not a row; R6ⁱ replaces it (14, Amendment 5). R7 dropped (D6).",
                   "1-seed rows (R1, R3, R5-γ0, R8, B2ᵃ): no significance claim.",
                   "B2ᵃ is the epoch-1 checkpoint of an unstable run (see T5 notes); its row is not evidence for or against a regressed R̂."])

    t6 = T.build_t6()
    T.write_table(t6, ["block", "rung", "model", "n_seeds", "learned parameters", "k=0", "k=10", "AUDC abs", "protocol"],
                  OUT / "T6_baseline_ladder", "T6 — baseline ladder (two blocks: validation and test; numbers are not compared across blocks)",
                  ["Direction-only physics → table physics → profiled physics → non-graph learning → learned representation → both.",
                   "The validation block holds the physics baselines of the audit scripts (B1ᵃ, B1, PSM) next to R0ⁿ and R5 on the same split. The test block "
                   "holds the classical and Paper-1 rows next to R0ⁿ, R5 and PSM on the test split. B1ᵃ and B1 were not scored on the test split.",
                   "PSM's clean k = 0 row is an operator-fidelity check (table and corpus come from the same simulator), not physics beating the network "
                   "(09_STAGE_H_writing, framing 1).",
                   "AUDC abs: raw F1 trapezoid over k ∈ {0, 1, 2, 5, 10}, divided by 10 (Paper 1 Eq. 8 form without normalisation); B1ᵃ and PSM(val) have no 5-level curve in the audit files.",
                   "n_seeds = 1 rows have no seed std; classical rows are one fit."])

    t9d = T.build_t9_detection()
    T.write_table(t9d, ["condition", "AUROC LOO σ0=0", "AUROC in-sample σ0=0", "TPR@FPR0.05 LOO σ0=0", "AUROC LOO σ0=0.01",
                        "AUROC in-sample σ0=0.01", "TPR@FPR0.05 LOO σ0=0.01"], OUT / "T9a_silent_detection_val",
                  "T9a — silent-sensor detection (PSM-refit residual), validation split, k = 0",
                  ["AUROC per sensor: positives = corrupted sensors, negatives = clean sensors of the corrupted windows; all 248,000 validation windows.",
                   "S5.1 hard gate (bias ±2 %, LOO, threshold 0.75) passed at every σ0. Stuck sensors are the weak case (AUROC about 0.6 at σ0 = 0).",
                   "S5.2: in-sample AUROC ≥ LOO at σ0 ≤ 2e-3 (the expected 'LOO clearly better' did not hold); the in-sample residual is used.",
                   "Source: results/pcstgnn/audit/s5_1_silent_detection/silent_detection.json and results/stage_f/silent_detection_noise/sigma_0.01/."])
    t9g = T.build_t9_gate()
    T.write_table(t9g, ["arm", "condition", "role", "no gate", "gate", "oracle gate", "gate − no gate", "seeds > 0", "p (raw t)",
                        "recovered of gap to oracle", "FPR clean sensors"], OUT / "T9b_gate_val",
                  "T9b — G-TF2 training-free gate, validation half B, σ0 = 1e-2, k = 0 (3 seeds)",
                  ["Macro-F1 of the pooled soft vote with no gate, with the gate, and with an oracle gate that knows which sensors are corrupted.",
                   "Primary pre-registered rule: stuck, gate > no gate in 3/3 seeds at each s (met). Gains are small (see the column); do not call them large.",
                   "arm 'resid05' is the residual-only variant: descriptive, not in a decision rule. The test-split run is T9c.",
                   "Source: results/stage_f/gtf2/core/summary.json, results/stage_f/gtf2r/core/summary.json."])

    t9t = T.build_t9_gate_test()
    if t9t:
        T.write_table(t9t, ["arm", "condition", "role", "no gate", "gate", "oracle gate", "gate − no gate", "seeds > 0", "p (raw t)",
                            "recovered of gap to oracle", "FPR clean sensors"], OUT / "T9c_gate_test",
                      "T9c — G-TF2 training-free gate, TEST split (247,000 windows), σ0 = 1e-2, k = 0 (3 seeds)",
                      ["Same design as T9b with the split changed; thresholds frozen from validation half A (sha256 in stage_f_gtf2_test_preregistration.md). "
                       "Registered before any test number existed.",
                       "Primary rule (stuck, 9/9 pairs) and the secondary rules are copied from the validation registration. A pass is a sign result, not a significance claim (n = 3).",
                       "'recovered' is read only where oracle − no gate ≥ 0.01 (— otherwise). Gaussian 0.02 is reported only.",
                       "Compare with T9b (validation half B); the two tables use different splits and are not pooled.",
                       "Source: results/stage_f/gtf2_test/summary.json."])

    t10 = T.build_t10()
    T.write_table(t10, ["condition", "k", "R5", "R0ⁿ", "ΔF1", "seeds > 0", "p (raw)"], OUT / "T10_noise_policies_test",
                  "T10 — measurement noise and sensor-loss policies, test split, R5 vs R0ⁿ",
                  ["Mean over seeds 42/43/44; ΔF1 = R5 − R0ⁿ; p = paired two-sided t over seeds (df = 2), not corrected.",
                   "Policies at σ = 0: random (S4.4 protocol), persistent (per event), contiguous (connected block), targeted "
                   "(nearest the fault; no-fault windows fall back to random).",
                   "n = 3: a sign in 3/3 seeds without p < 0.05 is a trend, not a result."])

    t11 = T.build_t11()
    cols11 = []
    for r in t11:
        cols11 += [c for c in r if c not in cols11 and c != "source"]
    T.write_table(t11, cols11, OUT / "T11_mismatch_val",
                  "T11 — model mismatch (perturbed signature table at inference), validation split",
                  ["Rule (14, Amendment 6): graceful if R5 with the perturbed table > R0ⁿ in all 3 seeds in the 7 H2/H4 cells "
                   "(21 comparisons). R0ⁿ does not read the table.",
                   "The retrain row is 1 seed (42): descriptive only.",
                   "Source: results/stage_g/analysis/summary.json (scripts/summarize_stage_g.py)."])

    if not args.skip_t13:
        t0 = time.perf_counter()
        prev_boot = None
        if args.reuse_bootstrap:
            saved = json.loads((OUT / "T13_bootstrap_test.json").read_text())
            if saved["n_boot"] != args.n_boot:
                print("STOP: saved bootstrap has a different n_boot", file=sys.stderr)
                return 2
            prev_boot = {pair: {"n_events": b["n_events"], "n_windows": b["n_windows"],
                                "cells": {T.parse_cell(k): v for k, v in b["cells"].items()}} for pair, b in saved["pairs"].items()}
        t13, boot = T.build_t13(n_boot=args.n_boot, boot=prev_boot)
        secs = time.perf_counter() - t0 if prev_boot is None else saved["seconds"]
        T.write_table(t13, ["pair", "hypothesis", "cell", "R5", "reference", "ΔF1", "seeds > 0", "p (t, raw)",
                            "p Holm (cells chosen post hoc)", "95 % CI (event bootstrap)", "supported (3/3 and CI excl. 0)"],
                      OUT / "T13_statistics_test", "T13 — statistical tests, test split (6,175 events, 247,000 windows)",
                      [f"Event-level paired cluster bootstrap: {args.n_boot} resamples of the test events, RandomState(20261102), "
                       "statistic = mean over the 3 seeds of ΔF1 (each F1 the mean over 3 offsets); 95 % percentile CI "
                       "(fault_localization/stage_g/gate6.py::analyze_cell).",
                       "Pairs pre-registered (14 § 3, amended): R5 vs R0ⁿ, R5 vs R6ⁱ, R5 vs R2. 'Supported' = sign 3/3 and CI "
                       "excludes zero (14 § 3).",
                       "Holm families {H1–H4} and {A1, A2} are pre-registered; the cells inside each family were chosen after the "
                       "Gate-4 results, so the Holm column is post hoc. H3 is on the sealed set (T8), not here.",
                       "The bootstrap CI covers sampling of test events only; the 3 trained models are fixed, so it does not include "
                       "training-seed variance and is narrower than the seed spread. With n = 3 the t-test over seeds is the "
                       "seed-level evidence: a cell with 3/3 signs and a CI above zero but raw p > 0.05 (H4 and A2 at σ = 1e-2) is a "
                       "trend, not a result.",
                       "H6 is a control: rule |ΔF1| ≤ 1 seed-std of R0ⁿ; it has no Holm family. The rule was written to catch a "
                       "loss of clean accuracy; R5 is higher, which also lies outside the band, so the control is not met as written "
                       "(R5 does not lose clean accuracy).",
                       "R=10 rows use only the test events at R = 10 Ω (n_events in the CSV)."])
        raw = {pair: {"n_events": b["n_events"], "n_windows": b["n_windows"],
                      "cells": {f"k{k}_sigma{s:g}": v for (k, s), v in b["cells"].items()}} for pair, b in boot.items()}
        (OUT / "T13_bootstrap_test.json").write_text(json.dumps({"n_boot": args.n_boot, "seconds": secs, "pairs": raw},
                                                                indent=2) + "\n")

    patterns = ["results/stage_g/test/*/seed_*_scoring.json", "results/stage_g/val/b2a/seed_*_scoring.json",
                "results/s4_4/*/pcstgnn/seed_*/scoring_full_grid.json", "results/stage_g/analysis/summary.json",
                "results/dropout_study/degradation_curves.csv", "results/dropout_study_aware/degradation_curves.csv",
                "results/dropout_aware/rgsage_max/seed_*/metrics.json", "results/s4_4/r0n/pcstgnn/seed_*/metrics.json"]
    if not args.skip_t13:
        patterns += [f"results/stage_g/test/{m}/seed_*_predictions.npz" for m in ("r5", "r0n", "r6i", "r2")]
    files = {str(p.relative_to(REPO)): _sha(p) for pat in patterns for p in sorted(REPO.glob(pat)) if p.is_file()}
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (OUT / "manifest.json").write_text(json.dumps({"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                                   "git_head": head, "r0n_metric_consistency": cons, "inputs_sha256": files},
                                                  indent=2) + "\n")
    print(json.dumps({"written": sorted(p.name for p in OUT.iterdir())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
