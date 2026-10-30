"""Fit and freeze the gate (O21, O22, O23, O24 step 2-3; author 2026-10-30). `val` only; `test` is not read here.

Per k in {0, 1, 2, 5, 10}, on post-fault windows (w >= 21), mask offset 0, neighbours from the full train store (held-out version, O20):
  cal-A (50 % of the val runs, stratified by scenario): isotonic map of the top-1 vote share fitted separately for each k (O22), logistic
        combination of [p, margin, agreement] (reported only, not used for the decision), conformal 90 % quantile (O23).
  cal-B (the other 50 %): thresholds (tau_p, tau_m, tau_k): maximum coverage with accuracy >= 95 % on the CONFIDENT windows ("acc95", the
        operating point), and >= 98 % ("acc98", reported). Composition (probability, margin, agreement) selected on val after the
        pre-registered test (reports/layer0/DECISION_RULE.md).
Writes configs/gate_v1.json with its sha256. Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.gate_fit
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from gridsense.config import ROOT, TRACK
from gridsense.data.splits import cal_split, split_events
from gridsense.eval import metrics as M
from gridsense.eval.knn_features import KS
from gridsense.eval.layer0_final import evaluate, load_split
from gridsense.model.predictor import checkpoint_sha256
from gridsense.verification.gate import N_VOTES, Gate, canonical_hash, vote_features

CONFIG = TRACK / "configs" / "gate_v1.json"
ALPHA = 0.10
TARGETS = {"acc95": 0.95, "acc98": 0.98}
INFEASIBLE = 2.0          # threshold above every possible value: never CONFIDENT
CODE_FILES = ["src/gridsense/verification/gate.py", "src/gridsense/eval/gate_fit.py", "src/gridsense/eval/layer0_final.py",
              "src/gridsense/eval/metrics.py", "src/gridsense/eval/knn_features.py", "src/gridsense/cases/store.py", "src/gridsense/cases/build.py"]


def cal_masks(runs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_runs, b_runs = (set(r) for r in cal_split(split_events("val"), seed=42))
    return np.isin(runs, list(a_runs)), np.isin(runs, list(b_runs))


def conformal_quantile(share_true: np.ndarray, alpha: float = ALPHA) -> float:
    """Split-conformal quantile of the nonconformity 1 - (vote share of the true class), finite-sample corrected."""
    s = np.sort(1.0 - share_true)
    idx = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(idx, len(s) - 1)])


def fit_isotonic(top1: np.ndarray, correct: np.ndarray) -> tuple[list[float], list[float]]:
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(top1, correct)
    assert np.allclose(np.interp(top1, iso.X_thresholds_, iso.y_thresholds_), iso.predict(top1))
    return [float(x) for x in iso.X_thresholds_], [float(y) for y in iso.y_thresholds_]


def fit_k(c, A: np.ndarray, B: np.ndarray) -> dict:
    votes, y = c.votes, c.y
    correct = votes.argmax(1) == y
    top1, margin = vote_features(votes)
    agree = c.knn["heldout_agree_10"]
    iso_x, iso_y = fit_isotonic(top1[A], correct[A])
    p = np.interp(top1, iso_x, iso_y)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(np.c_[p, margin, agree][A], correct[A])
    res: dict = {"iso_x": iso_x, "iso_y": iso_y, "lr": {"coef": [float(x) for x in lr.coef_[0]], "intercept": float(lr.intercept_[0]), "note": "reported only; not used for the decision"},
                 "conformal_q": conformal_quantile(votes[np.arange(len(y)), y][A] / N_VOTES)}
    for name, target in TARGETS.items():
        g = M.best_gate3(p[B], margin[B], agree[B], correct[B], target)
        res[name] = ({"feasible": True, "tau_p": g["tau_p"], "tau_m": g["tau_2"], "tau_k": g["tau_3"], "cal_b_coverage": g["coverage"], "cal_b_accuracy": g["accuracy"]}
                     if g["feasible"] else {"feasible": False, "tau_p": INFEASIBLE, "tau_m": INFEASIBLE, "tau_k": INFEASIBLE})
    return res


def main() -> None:
    val = load_split(42, "val")
    A, B = cal_masks(val.runs)
    per_k = {str(k): fit_k(val.by_k[k], A, B) for k in KS}
    cfg = {
        "gate_version": "gate_v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": 42, "checkpoint_sha256": checkpoint_sha256(42),
        "store": {"version": "full train store, scenario-held-out version (O20)", "fingerprint": json.loads((TRACK / "eval/case_store/seed42_full/store_manifest.json").read_text())["fingerprint"]},
        "features": ["isotonic_probability_per_k", "margin_top1_minus_top2_vote_share", "knn_agreement_heldout_10_neighbours"],
        "rule": "CONFIDENT iff p >= tau_p and margin >= tau_m and agreement >= tau_k; REJECT (UNSUPPORTED_DROPOUT_PATTERN) for k or pattern not fitted; REJECT (INVALID_INPUT) for bad votes; else LOW_CONFIDENCE with a conformal 90 % candidate set",
        "supported_patterns": ["random"], "dropout_policy": "zero_messages", "windows": "post-fault (w >= 21), mask offset 0", "k_nn": 10,
        "selection": {"split": "val", "cal_split": "50/50 by run, stratified by scenario, seed 42", "fit_on_cal_a": ["isotonic map per k", "logistic combination (reported only)", "conformal quantile"],
                      "thresholds_on_cal_b": "maximum coverage with accuracy >= target on CONFIDENT post-fault windows; one set per k",
                      "composition_note": "The three-feature composition was selected on val AFTER the pre-registered test (DECISION_RULE.md, commit aa7454b); the pre-registered test compared probability + agreement only."},
        "operating_point": "acc95", "targets": TARGETS, "conformal": {"alpha": ALPHA, "nonconformity": "1 - vote share of the true class", "set": "classes with vote share >= 1 - q"},
        "per_k": per_k,
        "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES},
        "base_commit": subprocess.run(["git", "rev-parse", "gridsense-eval"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    }
    gate = Gate(cfg)
    check = {"cal_a": evaluate(gate, val, A, boot=False), "cal_b": evaluate(gate, val, B, boot=True)}
    cfg["val_check_summary"] = {k: {"cal_b_coverage_acc95": v["gate_acc95"]["coverage"], "cal_b_accuracy_confident_acc95": v["gate_acc95"]["accuracy_confident"],
                                    "cal_b_conformal_coverage": v["conformal_90"]["coverage_all_post_fault"], "cal_b_conformal_mean_size": v["conformal_90"]["mean_size_all_post_fault"]}
                                for k, v in check["cal_b"]["per_k"].items()}
    cfg["config_sha256"] = canonical_hash(cfg)
    CONFIG.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    (TRACK / "configs" / "gate_v1.sha256").write_text(cfg["config_sha256"] + "  configs/gate_v1.json\n")
    (TRACK / "reports" / "layer0").mkdir(parents=True, exist_ok=True)
    (TRACK / "reports" / "layer0" / "gate_v1_val_check.json").write_text(json.dumps(check, indent=1, default=float) + "\n")
    Gate.load(CONFIG)       # re-read: the hash must verify
    print("gate_v1 written, sha256", cfg["config_sha256"], flush=True)


if __name__ == "__main__":
    main()
