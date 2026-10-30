"""Layer 0 for training seeds 43 and 44 with the FROZEN method (O18; author 2026-10-30). Layer 0 only; `test` windows; no new thresholds.

Frozen: features, store construction (full train store of that seed), post-fault windows, mask offset 0, and the THRESHOLDS (tau_p, tau_m, tau_k per k)
of gate_v1. Re-fitted per seed because the vote scale belongs to the model: the isotonic map per k and the conformal 90 % quantile per k
(cal-A of val, post-fault, the same 50/50 run split). The logistic combination is not refitted. Seed 42's own numbers are in test_run_seed42.json.
Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.seed_runs 43
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np

from gridsense.config import TRACK
from gridsense.eval.gate_fit import CONFIG, cal_masks, conformal_quantile, fit_isotonic
from gridsense.eval.knn_features import KS
from gridsense.eval.layer0_final import evaluate, load_split
from gridsense.verification.gate import N_VOTES, Gate, vote_features


def seed_gate(seed: int) -> tuple[Gate, dict]:
    frozen = Gate.load(CONFIG)
    val = load_split(seed, "val", with_knn=False)
    A, _ = cal_masks(val.runs)
    cfg = copy.deepcopy(frozen.cfg)
    cfg.pop("config_sha256", None)
    cfg["seed"] = seed
    for k in KS:
        c = val.by_k[k]
        correct = c.votes.argmax(1) == c.y
        top1, _ = vote_features(c.votes)
        x, y = fit_isotonic(top1[A], correct[A])
        cfg["per_k"][str(k)].update({"iso_x": x, "iso_y": y, "lr": None,
                                     "conformal_q": conformal_quantile(c.votes[np.arange(len(c.y)), c.y][A] / N_VOTES)})
    return Gate(cfg), {"fitted_on": "val cal-A, post-fault, per k", "thresholds": "frozen from gate_v1 (seed 42)", "seed": seed}


def main(seed: int) -> None:
    gate, note = seed_gate(seed)
    res = evaluate(gate, load_split(seed, "test"), None, boot=True)
    res["frozen_method"] = {**note, "gate_v1_config_sha256": Gate.load(CONFIG).cfg["config_sha256"]}
    out = TRACK / "reports" / "layer0" / f"seed{seed}_layer0_test.json"
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("written", out.name, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
