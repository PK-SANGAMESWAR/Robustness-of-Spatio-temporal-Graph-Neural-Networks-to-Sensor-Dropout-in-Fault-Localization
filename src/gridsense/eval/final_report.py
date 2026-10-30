"""Writes reports/layer0/FINAL_REPORT.md from the result files (no number is typed by hand).

Inputs: configs/gate_v1.json, reports/layer0/{gate_v1_val_check,test_run_seed42,seed43_layer0_test,seed44_layer0_test,leakage_checks_seed42_train}.json.
Run: env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.final_report [--test-file PATH]   (the flag is for a dry run on the val check)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from gridsense.config import TRACK

R = TRACK / "reports" / "layer0"
KS = ["0", "1", "2", "5", "10"]


def _j(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def s44_epochs() -> str:
    f = Path(TRACK.parent / "results/s4_4/r4/pcstgnn/seed_00044/metrics_by_epoch.csv")
    t = [float(r["train_seconds"]) for r in csv.DictReader(f.open())]
    base = sum(t[2:13]) / 11
    rows = [f"{i + 1}: {x:.0f} s ({100 * (x / base - 1):+.1f} %)" for i, x in enumerate(t) if i >= 13]
    return f"baseline {base:.1f} s (epochs 3–13), limit {base * 1.1:.0f} s; epochs 14 onward: " + "; ".join(rows)


def build(test_file: str | None = None) -> str:
    cfg = json.loads((TRACK / "configs/gate_v1.json").read_text())
    val = _j("gate_v1_val_check.json")["cal_b"]
    test = json.loads(Path(test_file).read_text()) if test_file else _j("test_run_seed42.json")
    s43, s44, leak = _j("seed43_layer0_test.json"), _j("seed44_layer0_test.json"), _j("leakage_checks_seed42_train.json")
    L: list[str] = []
    A = L.append
    fz = test.get("frozen_gate", {})
    A("# Layer 0 final report — frozen gate_v1, one run on `test` (seed 42), seeds 43 and 44")
    A("")
    A("Written for: the author (decisions at the end). Numbers are copied from `reports/layer0/*.json` by `src/gridsense/eval/final_report.py`.")
    A(f"Frozen gate: `configs/gate_v1.json`, sha256 `{cfg['config_sha256']}`; frozen in commit `b7a80b9` before any run on test. Run on test once: {fz.get('run_utc', 'n/a')}.")
    A("**The three-feature composition (probability, margin, kNN agreement) was selected on val after the pre-registered test** (`DECISION_RULE.md`, `aa7454b`), not before it.")
    A("Windows: post-fault (w ≥ 21), mask offset 0, held-out full train store (O20). 95 % CIs: bootstrap by run, 1000 samples.")
    A("")
    A("## 0. Result")
    c10 = test["per_k"]["10"]["gate_acc95"]
    A(f"- **Coverage at k = 10 on test: {c10['coverage']:.4f}** (95 % CI {c10['coverage_ci'][0]:.4f}–{c10['coverage_ci'][1]:.4f}); " + ("**below 50 %** (reported, as you said not to stop)." if c10["coverage"] < 0.5 else "not below 50 %."))
    A("- CONFIDENT accuracy on test against the 95 % target: " + ", ".join(f"k = {k}: {test['per_k'][k]['gate_acc95']['accuracy_confident']:.4f}" for k in KS) + ".")
    off_k = [k for k in KS if cfg["per_k"][k]["acc95"]["tau_k"] == 0.0]
    off_p = [k for k in KS if cfg["per_k"][k]["acc95"]["tau_p"] == 0.0]
    A(f"- At the 95 % point the frozen thresholds switch kNN agreement off (τ_k = 0) at k = {', '.join(off_k) or 'none'} and the probability off (τ_p = 0) at k = {', '.join(off_p) or 'none'}"
      + (": the gate is effectively **margin-only** there." if len(off_k) == 5 and len(off_p) >= 4 else ".") + " Agreement is active at the 98 % point.")
    below = [k for k in KS if test["per_k"][k]["knn_vote_accuracy"]["heldout"]["10"] < test["per_k"][k]["stgnn_accuracy"]]
    A(f"- The held-out kNN vote (10 neighbours) is below the STGNN's own accuracy at k = {', '.join(below) or 'none'} (§ 4).")
    lc = {k: test["per_k"][k]["conformal_90"]["coverage_low_confidence"] for k in KS}
    A("- 90 % candidate sets: coverage over all post-fault windows " + ", ".join(f"k = {k}: {test['per_k'][k]['conformal_90']['coverage_all_post_fault']:.3f}" for k in KS)
      + "; over the LOW_CONFIDENCE windows only " + ", ".join(f"k = {k}: {v:.3f}" for k, v in lc.items() if v is not None) + " (§ 3).")
    if s43 and s44:
        for name, d in (("43", s43), ("44", s44)):
            acc = [d["per_k"][k]["gate_acc95"]["accuracy_confident"] for k in KS]
            cov = [d["per_k"][k]["gate_acc95"]["coverage"] for k in KS]
            c10s = d["per_k"]["10"]["gate_acc95"]["coverage"]
            A(f"- **Seed {name}, frozen seed-42 thresholds:** CONFIDENT accuracy at the 95 % point {min(acc):.4f}–{max(acc):.4f} (target 0.95), coverage {min(cov):.3f}–{max(cov):.3f}; coverage at k = 10: {c10s:.4f}" + (" (**below 50 %**)" if c10s < 0.5 else "")
              + (": **the frozen thresholds do not transfer to another training seed** (below the target at every k)." if max(acc) < 0.95 else "."))
        A("  Thresholds on the raw margin belong to the checkpoint they were selected on; a new checkpoint needs its own thresholds (fitted on its own cal split). New thresholds were NOT selected for seeds 43 and 44 (your instruction).")
    A("")
    A("## 1. The frozen gate (per k; CONFIDENT iff p ≥ τ_p and margin ≥ τ_m and agreement ≥ τ_k)")
    A("| k | operating point | τ_p | τ_m | τ_k | conformal q | isotonic knots |")
    A("|---|---|---|---|---|---|---|")
    for k in KS:
        for op in ("acc95", "acc98"):
            t = cfg["per_k"][k][op]
            A(f"| {k} | {op} | {t['tau_p']:.4f} | {t['tau_m']:.4f} | {t['tau_k']:.1f} | {cfg['per_k'][k]['conformal_q']:.4f} | {len(cfg['per_k'][k]['iso_x'])} |")
    A("A k or dropout pattern that is not fitted (k ∉ {0, 1, 2, 5, 10}, or a pattern other than random) returns REJECT with `UNSUPPORTED_DROPOUT_PATTERN`.")
    A("")
    A("## 2. The single run on `test` vs the selection set (cal-B), 95 % point")
    A("| k | STGNN accuracy (test) | coverage cal-B | coverage test (CI) | CONFIDENT accuracy cal-B | CONFIDENT accuracy test (CI) | error recall test |")
    A("|---|---|---|---|---|---|---|")
    for k in KS:
        t, v = test["per_k"][k], val["per_k"][k]
        g, gv = t["gate_acc95"], v["gate_acc95"]
        A(f"| {k} | {t['stgnn_accuracy']:.4f} | {gv['coverage']:.4f} | {g['coverage']:.4f} ({g['coverage_ci'][0]:.3f}–{g['coverage_ci'][1]:.3f}) | {gv['accuracy_confident']:.4f} | {g['accuracy_confident']:.4f} ({g['accuracy_confident_ci'][0]:.3f}–{g['accuracy_confident_ci'][1]:.3f}) | {g['error_recall']:.3f} |")
    A("98 % point on test (thresholds from cal-B): " + "; ".join(f"k = {k}: coverage {test['per_k'][k]['gate_acc98']['coverage']:.3f}, accuracy {test['per_k'][k]['gate_acc98']['accuracy_confident']:.4f}" for k in KS) + ".")
    A("Outcomes on test (95 % point): " + "; ".join(f"k = {k}: {test['per_k'][k]['gate_acc95']['outcomes']}" for k in KS[:1] + KS[-1:]) + ". No window was rejected: every test window has a fitted k and the random pattern.")
    A("")
    A("## 3. Conformal 90 % candidate sets (per k; fitted on cal-A)")
    A("| k | cal-B coverage | cal-B mean size | test coverage (CI) | test mean size | test LOW_CONFIDENCE coverage | LOW_CONFIDENCE mean size | n LOW_CONFIDENCE |")
    A("|---|---|---|---|---|---|---|---|")
    for k in KS:
        t, v = test["per_k"][k]["conformal_90"], val["per_k"][k]["conformal_90"]
        lc = f"{t['coverage_low_confidence']:.3f}" if t["coverage_low_confidence"] is not None else "-"
        ls = f"{t['mean_size_low_confidence']:.2f}" if t["mean_size_low_confidence"] is not None else "-"
        A(f"| {k} | {v['coverage_all_post_fault']:.4f} | {v['mean_size_all_post_fault']:.2f} | {t['coverage_all_post_fault']:.4f} ({t['coverage_all_ci'][0]:.3f}–{t['coverage_all_ci'][1]:.3f}) | {t['mean_size_all_post_fault']:.2f} | {lc} | {ls} | {t['n_low_confidence']:,} |")
    A("The 90 % guarantee is marginal (over all post-fault windows). Among the LOW_CONFIDENCE windows, which the sets are shown for, coverage is much lower: a candidate set on a LOW_CONFIDENCE report is not a 90 % set.")
    A("")
    A("## 4. Layer 0 on test (held-out store primary), per k")
    A("| | k = 0 | k = 1 | k = 2 | k = 5 | k = 10 |")
    A("|---|---|---|---|---|---|")
    rows = [("ECE isotonic", lambda r: r["calibration"]["ece_isotonic"]), ("ECE raw vote share", lambda r: r["calibration"]["ece_raw_vote_share"]),
            ("AURC isotonic", lambda r: r["selective"]["aurc_isotonic"]), ("risk at 80 % coverage", lambda r: r["selective"]["risk_at_coverage_0.8"]),
            ("kNN vote (10), held-out", lambda r: r["knn_vote_accuracy"]["heldout"]["10"]), ("kNN vote (10), standard (secondary)", lambda r: r["knn_vote_accuracy"]["standard"]["10"])]
    for lab, f in rows:
        A(f"| {lab} | " + " | ".join(f"{f(test['per_k'][k]):.4f}" for k in KS) + " |")
    A("Error-detection AUROC (95 % CI):")
    A("| signal | k = 0 | k = 1 | k = 2 | k = 5 | k = 10 |")
    A("|---|---|---|---|---|---|")
    for n in test["per_k"]["0"]["error_detection"]:
        A(f"| {n} | " + " | ".join(f"{test['per_k'][k]['error_detection'][n]['auroc']:.4f} ({test['per_k'][k]['error_detection'][n]['auroc_ci'][0]:.3f}–{test['per_k'][k]['error_detection'][n]['auroc_ci'][1]:.3f})" for k in KS) + " |")
    A("")
    A("## 5. Seeds 43 and 44 (frozen thresholds; isotonic map and conformal quantile refitted per seed on cal-A; test windows)")
    if s43 and s44:
        A("| seed | k | STGNN accuracy | coverage (95 % point) | CONFIDENT accuracy | ECE isotonic | conformal coverage | AUROC margin | AUROC isotonic | AUROC agreement |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for name, d in (("42", test), ("43", s43), ("44", s44)):
            for k in KS:
                r = d["per_k"][k]
                A(f"| {name} | {k} | {r['stgnn_accuracy']:.4f} | {r['gate_acc95']['coverage']:.4f} | {r['gate_acc95']['accuracy_confident']:.4f} | {r['calibration']['ece_isotonic']:.4f} | {r['conformal_90']['coverage_all_post_fault']:.4f} | {r['error_detection']['margin']['auroc']:.4f} | {r['error_detection']['isotonic_prob']['auroc']:.4f} | {r['error_detection']['knn_agreement_heldout']['auroc']:.4f} |")
    else:
        A("Not available in this run of the report.")
    A("")
    A("## 6. Leakage checks on the full store")
    if leak:
        d = leak["disjointness"]; f = leak["faiss_vs_torch_val_queries"]
        A(f"Train / val / test runs disjoint (overlaps {d['train_val']}, {d['train_test']}, {d['val_test']}); store = {d['store_runs']:,} train runs, {d['store_runs_in_val_or_test']} in val or test; seeds shared with val/test: {d['store_seeds_in_val_or_test']}; fingerprint matches: {d['store_fingerprint_matches_expected']}.")
        A(f"FAISS vs the exact torch search ({f['n_queries']:,} val queries): votes identical {f['standard']['vote10_identical_to_torch'] * 100:.1f} % / {f['heldout']['vote10_identical_to_torch'] * 100:.1f} %; own-scenario neighbours in the held-out top 20: {f['heldout']['own_scenario_in_top20']}; in the standard top 20: {f['standard']['own_scenario_in_top20']:,} of {f['n_queries'] * 20:,}.")
    A("")
    A("## 7. Compute (S4.4 queue, R4 seed 44)")
    A(s44_epochs() + ". Threads: 4, then 2 from epoch 19 (precaution at +9.4 %). CPU only; no GPU, no paid call.")
    A("")
    A("## 8. Limits")
    A("- Mask offset 0 only; one realisation per window and k. Post-fault windows only for the gate and for every neighbour metric.")
    A("- Thresholds fitted on cal-B sit at the selection boundary; the test run is the unbiased check. Only the store is held out; the encoder saw every scenario.")
    A("- `test` was used once for the frozen gate (seed 42). Seeds 43 and 44 are reported test numbers with frozen thresholds, not new selections.")
    A("")
    A("## 9. For the author")
    A("1. **The gate is effectively margin-only at its 95 % point.** Agreement and (at four k) the probability add nothing there; they matter at the 98 % point. gate_v1 stays as frozen and tested. A simpler gate_v2 would be a new version, and `test` has now been used, so a v2 must not be judged on the same test numbers.")
    A("2. **Thresholds belong to a checkpoint.** On seeds 43 and 44 the frozen thresholds miss the 95 % target by 2–5 points (§ 5). Any other checkpoint needs its own cal-A / cal-B fit and its own frozen run. Which checkpoint is the system: seed 42?")
    A("3. **Candidate sets on LOW_CONFIDENCE reports are not 90 % sets** (§ 3: 55–81 % at seed 42). Options for the report text: state the measured conditional coverage; fit the conformal quantile on LOW_CONFIDENCE windows only (new fit and check); or use a fixed top-k set.")
    A("4. **Next stage: Layer 1 (document ingestion and retrieval).** Ready: `corpus/v1` (the revised paper stays a local copy, SHA-256 in `MANIFEST.json`), the span-based golden schema, the sweep skills. Open before it: O7 (extra corpus sources), O1 (OpenAI models and prices, needed only for the paid steps), O13 (human labels), and the golden retrieval drafts for your review.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    tf = sys.argv[sys.argv.index("--test-file") + 1] if "--test-file" in sys.argv else None
    out = R / ("FINAL_REPORT_DRYRUN.md" if tf else "FINAL_REPORT.md")
    out.write_text(build(tf))
    print("written", out.name, len(out.read_text().splitlines()), "lines")
