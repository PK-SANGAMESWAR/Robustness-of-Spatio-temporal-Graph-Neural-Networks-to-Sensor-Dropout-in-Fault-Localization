"""Ablation ladder A0 to A4 (rag-eval.md § 10; EVAL.md § 10). Assembled from existing Layer 0 results; no model run, no new use of the test split.

Rungs on `val` (cal-B, post-fault windows, seed 42; `reports/layer0/layer0_val_seed42.json`), accuracy target 95 %, held-out store (O20):
A0 argmax (accept everything) | A1 + calibrated threshold (probability only) | A2 + kNN agreement (probability and agreement) |
extra rows, not in the plan: probability + margin, and the three-feature composition (probability, margin, agreement) that became `gate_v1` | A3 NOT RUN (the Stage F silent-failure check is not recorded as usable) |
A4 + retrieval and LLM report: the report does not change who is accepted, so its selection columns equal the gate's; its quality is in the S4 tables.
Frozen gate on `test` (one run, `reports/layer0/test_run_seed42.json`): one row, shown apart. A1 and A2 were never run on test.
Caveat: on val the thresholds are chosen on cal-B and measured on cal-B, so coverage there is an optimistic in-sample figure; compare rungs only with their accuracy column and not at differences of a few points.
`python -m gridsense.eval.ablation_ladder`
"""
from __future__ import annotations

import json

from gridsense.config import ROOT, TRACK

L0 = TRACK / "reports" / "layer0"
KS = ("0", "1", "2", "5", "10")
RUNGS = [("A1", "probability only", "p_only"), ("A2", "probability + kNN agreement", "p+agreement_heldout"), ("extra", "probability + margin", "p+margin")]


def _row(c: dict) -> dict:
    n, e, ea, acc = c["n_windows"], c["n_errors"], c["errors_accepted"], c["accuracy_confident"]
    return {"coverage": c["coverage"], "selective_risk": 1 - acc, "false_acceptance": ea / n, "error_recall": c["error_recall"], "accuracy_confident": acc}


def build() -> dict:
    val = json.loads((L0 / "layer0_val_seed42.json").read_text())
    test = json.loads((L0 / "test_run_seed42.json").read_text())
    out: dict = {"val": {}, "test_frozen_gate": {}, "aurc_val": {}}
    for k in KS:
        st = val["per_k"][k]["stgnn"]
        base = val["gates"]["p_only"]["0.95"]["per_k"][k]["cal_b"]
        acc0 = st["acc_cal_b_post"]
        rows = {"A0": {"coverage": 1.0, "selective_risk": 1 - acc0, "false_acceptance": 1 - acc0, "error_recall": 0.0, "accuracy_confident": acc0}}
        for rung, _, key in RUNGS:
            g = val["gates"][key]["0.95"]["per_k"][k]
            rows[rung if rung != "extra" else "p+margin"] = _row(g["cal_b"]) | {"feasible": g["feasible"]}
        g3 = val["posthoc"]["gate3"]["0.95"]["per_k"][k]
        rows["gate3 (p+margin+agreement)"] = _row(g3["cal_b"]) | {"feasible": g3["feasible"]}
        out["val"][k] = rows
        sel = val["per_k"][k]["selective_cal_b_post"]
        out["aurc_val"][k] = {"probability (isotonic)": sel["aurc_iso"], "raw vote share": sel["aurc_raw_share"], "margin": sel["aurc_margin"]}
        t = test["per_k"][k]
        g = t["gate_acc95"]
        out["test_frozen_gate"][k] = {"A0 (argmax) accuracy": t["stgnn_accuracy"], "gate_v1 coverage": g["coverage"], "gate_v1 selective_risk": 1 - g["accuracy_confident"], "gate_v1 error_recall": g["error_recall"],
                                      "gate_v1 false_acceptance": g["errors_accepted"] / g["n_windows"], "aurc_isotonic": t["selective"]["aurc_isotonic"]}
        assert base["n_windows"] > 0
    return out


def markdown(d: dict) -> str:
    f = lambda x: f"{x:.3f}"
    L = ["# Ablation ladder A0 to A4 (Layer 0 data, assembled by `gridsense.eval.ablation_ladder`)", "",
         "Written for the author. Seed 42, post-fault windows, held-out store, accuracy target 95 %. Every number is copied or derived from `reports/layer0/layer0_val_seed42.json` and `test_run_seed42.json`. "
         "No model run; the test split was not used again. Definitions: coverage = share of windows accepted; selective risk = 1 − accuracy of the accepted windows; false acceptance = accepted and wrong, as a share of all windows; error recall = share of wrong predictions that were NOT accepted.", "",
         "## A. `val` (cal-B): the rungs", "",
         "Thresholds are chosen on cal-B and measured on cal-B: coverage is an in-sample figure. Compare rungs with the accuracy column. No confidence intervals are computed for these rows. "
         "A1 overshoots the 95 % target (accuracy about 0.96): the isotonic probability is a step function with ties, so its thresholds are coarse. Its lower coverage and higher error recall are therefore NOT a like-for-like comparison with A2. "
         "A2, probability + margin and the three-feature composition differ by about 0.01 coverage or less; no interval is available, so no ranking among them is claimed.", "",
         "| k | rung | coverage | accuracy of accepted | selective risk | false acceptance | error recall |", "|---|---|---|---|---|---|---|"]
    for k, rows in d["val"].items():
        for name, r in rows.items():
            L.append(f"| {k} | {name} | {f(r['coverage'])} | {f(r['accuracy_confident'])} | {f(r['selective_risk'])} | {f(r['false_acceptance'])} | {f(r['error_recall'])} |")
    L += ["", "A3 (+ silent-failure check): **NOT RUN** (the Stage F check is not recorded as usable; `PC-STGNN/07_*`).",
          "A4 (+ retrieval and LLM report): the report does not change who is accepted, so A4 has the selection columns of the gate. Report quality is measured apart (S4, S6).", "",
          "## B. AURC on `val` (cal-B, post-fault; lower is better)", "", "| k | probability (isotonic) | raw vote share | margin |", "|---|---|---|---|"]
    L += [f"| {k} | {f(a['probability (isotonic)'])} | {f(a['raw vote share'])} | {f(a['margin'])} |" for k, a in d["aurc_val"].items()]
    L += ["", "## C. `test`: the frozen gate, one run (the only rung that was run on test)", "", "| k | A0 accuracy | gate_v1 coverage | selective risk | false acceptance | error recall | AURC (isotonic) |", "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {f(t['A0 (argmax) accuracy'])} | {f(t['gate_v1 coverage'])} | {f(t['gate_v1 selective_risk'])} | {f(t['gate_v1 false_acceptance'])} | {f(t['gate_v1 error_recall'])} | {f(t['aurc_isotonic'])} |" for k, t in d["test_frozen_gate"].items()]
    L += ["", "A1 and A2 were never run on `test` (the test split is used once per frozen configuration). A second test use needs the author's approval."]
    return "\n".join(L) + "\n"


def main() -> None:
    d = build()
    (L0 / "ABLATION_LADDER.json").write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    (L0 / "ABLATION_LADDER.md").write_text(markdown(d), encoding="utf-8")
    print(f"written {(L0 / 'ABLATION_LADDER.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
