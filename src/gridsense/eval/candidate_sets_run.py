"""Fit, freeze and run (once on test) the LOW_CONFIDENCE candidate sets (O26; author 2026-10-31).
  python -m gridsense.eval.candidate_sets_run fit    # val only: fit on cal-A, check on cal-B, write configs/candidate_sets_v1.json (+ hash)
  python -m gridsense.eval.candidate_sets_run test   # ONE run on test (second use of test, for this component only); refuses a second run
`test` requires the config to be committed on gridsense-eval and gate_v1 to be unchanged.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys

import numpy as np

from gridsense.config import ROOT, TRACK
from gridsense.eval.gate_fit import CONFIG as GATE_CONFIG
from gridsense.eval.gate_fit import cal_masks, conformal_quantile
from gridsense.eval.knn_features import KS
from gridsense.eval.layer0_final import N_BOOT, _ci, _counts, load_split
from gridsense.verification.candidate_sets import CandidateSets
from gridsense.verification.gate import LOW_CONFIDENCE, N_VOTES, Gate, canonical_hash

CONFIG = TRACK / "configs" / "candidate_sets_v1.json"
RESULT = TRACK / "reports" / "layer0" / "candidate_sets_test_run_seed42.json"
ALPHA = 0.10
CODE = ["src/gridsense/verification/candidate_sets.py", "src/gridsense/eval/candidate_sets_run.py", "src/gridsense/eval/gate_fit.py"]


def _low_windows(gate: Gate, c, k: int) -> np.ndarray:
    return gate.decide(k, c.votes, c.knn["heldout_agree_10"], operating_point="acc95")["outcome"] == LOW_CONFIDENCE


def _stats(votes, y, low, q, mask=None) -> dict:
    m = low if mask is None else low & mask
    share = votes[m] / N_VOTES
    covered = share[np.arange(m.sum()), y[m]] >= 1 - q
    size = (share >= 1 - q).sum(1)
    return {"n": int(m.sum()), "coverage": float(covered.mean()), "mean_size": float(size.mean()), "median_size": float(np.median(size))}


def fit() -> None:
    gate = Gate.load(GATE_CONFIG)
    val = load_split(42, "val")
    A, B = cal_masks(val.runs)
    per_k = {}
    for k in KS:
        c = val.by_k[k]
        low = _low_windows(gate, c, k)
        share_true = c.votes[np.arange(len(c.y)), c.y] / N_VOTES
        q = conformal_quantile(share_true[A & low])
        old_q = gate.per_k[k]["conformal_q"]
        per_k[str(k)] = {"q": q, "n_cal_a_low_confidence": int((A & low).sum()),
                         "cal_a": _stats(c.votes, c.y, low, q, A), "cal_b": _stats(c.votes, c.y, low, q, B),
                         "cal_b_old_all_window_sets": _stats(c.votes, c.y, low, old_q, B)}
    cfg = {"component": "candidate_sets_v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seed": 42,
           "gate_v1_config_sha256": gate.cfg["config_sha256"], "alpha": ALPHA, "operating_point": "acc95",
           "population": "windows that gate_v1 labels LOW_CONFIDENCE (95 % point), post-fault, mask offset 0, per k",
           "fit": "cal-A (LOW_CONFIDENCE windows only); checked on cal-B", "nonconformity": "1 - vote share of the true class", "set": "classes with vote share >= 1 - q",
           "per_k": per_k, "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE}}
    cfg["config_sha256"] = canonical_hash(cfg)
    CONFIG.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    CandidateSets.load(CONFIG)
    print("candidate_sets_v1 written, sha256", cfg["config_sha256"])


def test_run() -> None:
    if RESULT.exists():
        sys.exit(f"{RESULT.name} exists: the candidate sets were already run once on test (second use of test); not again")
    committed = subprocess.run(["git", "show", "gridsense-eval:rag-for-dropout/configs/candidate_sets_v1.json"], cwd=ROOT, capture_output=True)
    if committed.returncode != 0 or committed.stdout != CONFIG.read_bytes():
        sys.exit("configs/candidate_sets_v1.json is not committed on gridsense-eval: freeze first")
    gate, cs = Gate.load(GATE_CONFIG), CandidateSets.load(CONFIG)
    if cs.cfg["gate_v1_config_sha256"] != gate.cfg["config_sha256"]:
        sys.exit("gate_v1 differs from the gate this component was fitted for")
    data = load_split(42, "test")
    u, inv = np.unique(data.runs, return_inverse=True)
    counts = _counts(len(u)).astype(float)
    out = {"n_windows": len(data.runs), "n_runs": int(len(u)), "per_k": {}}
    for k in KS:
        c = data.by_k[k]
        low = _low_windows(gate, c, k)
        q = cs.q[k]
        share = c.votes / N_VOTES
        covered = share[np.arange(len(c.y)), c.y] >= 1 - q
        size = (share >= 1 - q).sum(1).astype(float)
        old = gate.per_k[k]["conformal_q"]
        old_cov = share[np.arange(len(c.y)), c.y] >= 1 - old
        r_n, r_l = np.bincount(inv, weights=low.astype(float), minlength=len(u)), None
        cov_run = np.bincount(inv, weights=(covered & low).astype(float), minlength=len(u))
        size_run = np.bincount(inv, weights=size * low, minlength=len(u))
        r = _stats(c.votes, c.y, low, q)
        r["coverage_ci"] = _ci((counts @ cov_run) / np.maximum(counts @ r_n, 1))
        r["mean_size_ci"] = _ci((counts @ size_run) / np.maximum(counts @ r_n, 1))
        r["old_all_window_sets"] = {"coverage": float(old_cov[low].mean()), "mean_size": float((share >= 1 - old).sum(1)[low].mean())}
        r["n_low_confidence_windows"] = int(low.sum())
        r["fraction_of_post_fault_windows"] = float(low.mean())
        out["per_k"][str(k)] = r
    out["component"] = {"config_sha256": cs.cfg["config_sha256"], "gate_v1_config_sha256": gate.cfg["config_sha256"],
                        "run_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "test_use": "second use of the test split, for the candidate-set component only; gate_v1 unchanged"}
    RESULT.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("candidate-set test run written", flush=True)


if __name__ == "__main__":
    {"fit": fit, "test": test_run}[sys.argv[1]]()
