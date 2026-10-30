"""Layer 0 metric functions against brute force and sklearn (synthetic data only)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridsense.eval.metrics import (TieGroups, apply_gate, aurc, auroc, best_gate, ece_equal_mass, gate_stats,  # noqa: E402
                                     risk_at_coverage)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    score = np.round(rng.random(n), 1)                                   # many ties on purpose
    is_err = (rng.random(n) < 0.2 + 0.5 * score).astype(int)
    return score, is_err


def test_auroc_matches_sklearn_with_ties() -> None:
    s, e = _data()
    assert abs(auroc(s, e) - roc_auc_score(e, s)) < 1e-12


def test_weighted_auroc_equals_replication() -> None:
    s, e = _data()
    w = np.random.default_rng(1).integers(0, 4, len(s)).astype(float)
    rep = np.repeat(np.arange(len(s)), w.astype(int))
    assert abs(TieGroups(s, e).auroc(w) - roc_auc_score(e[rep], s[rep])) < 1e-12


def test_aurc_matches_brute_force_without_ties() -> None:
    rng = np.random.default_rng(2)
    score = rng.permutation(50).astype(float)
    correct = rng.random(50) < 0.7
    order = np.argsort(-score)
    risk = [(~correct[order[: i + 1]]).mean() for i in range(50)]
    assert abs(aurc(score, correct) - np.mean(risk)) < 1e-12


def test_risk_at_coverage() -> None:
    score = np.array([0.9, 0.9, 0.8, 0.5, 0.5, 0.1])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    risk, cov = risk_at_coverage(score, correct, 0.5)
    assert cov == 0.5 and risk == 0.0                                   # top 3 windows are all correct
    risk, cov = risk_at_coverage(score, correct, 0.6)
    assert abs(cov - 5 / 6) < 1e-12 and abs(risk - 1 / 5) < 1e-12      # tie group {0.5, 0.5} enters together


def test_ece_perfect_and_biased() -> None:
    p = np.repeat([0.1, 0.9], 100)
    c = np.concatenate([np.r_[np.ones(10), np.zeros(90)], np.r_[np.ones(90), np.zeros(10)]])
    assert ece_equal_mass(p, c, bins=2) < 1e-12
    assert abs(ece_equal_mass(np.full(10, 0.9), np.zeros(10), bins=1) - 0.9) < 1e-12


def test_best_gate_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    n = 300
    p = np.round(rng.random(n), 2)
    f2 = np.round(rng.random(n), 1)
    correct = rng.random(n) < (0.3 + 0.6 * p)
    for target in (0.8, 0.9):
        got = best_gate(p, f2, correct, target)
        best = (-1.0, None)
        for tp, t2 in itertools.product(np.unique(p), np.unique(f2)):
            acc_mask = (p >= tp) & (f2 >= t2)
            if acc_mask.any() and correct[acc_mask].mean() >= target - 1e-12:
                best = max(best, (acc_mask.mean(), (tp, t2)), key=lambda x: x[0])
        assert got["feasible"] == (best[1] is not None)
        if got["feasible"]:
            assert abs(got["coverage"] - best[0]) < 1e-12
            st = gate_stats(apply_gate(p, f2, got["tau_p"], got["tau_2"]), correct)
            assert abs(st["coverage"] - got["coverage"]) < 1e-12 and st["accuracy_confident"] >= target - 1e-12


def test_best_gate_infeasible_and_one_threshold() -> None:
    p = np.linspace(0, 1, 50)
    assert not best_gate(p, None, np.zeros(50, dtype=bool), 0.95)["feasible"]
    g = best_gate(p, None, p > 0.5, 0.95)      # 25 correct windows on top; accepting ONE wrong window still gives 25/26 = 0.9615 >= 0.95
    assert g["feasible"] and g["tau_2"] is None and g["n_accepted"] == 26 and abs(g["accuracy"] - 25 / 26) < 1e-12


def test_best_gate3_matches_brute_force() -> None:
    from gridsense.eval.metrics import best_gate3
    rng = np.random.default_rng(4)
    n = 250
    p, f2, f3 = np.round(rng.random(n), 1), np.round(rng.random(n), 1), np.round(rng.random(n), 1)
    correct = rng.random(n) < (0.3 + 0.5 * p + 0.2 * f3)
    got = best_gate3(p, f2, f3, correct, 0.85, n_bins=50)
    best = -1.0
    for a, b, c in itertools.product(np.unique(p), np.unique(f2), np.unique(f3)):
        m = (p >= a) & (f2 >= b) & (f3 >= c)
        if m.any() and correct[m].mean() >= 0.85 - 1e-12:
            best = max(best, m.mean())
    assert got["feasible"] == (best >= 0) and (not got["feasible"] or abs(got["coverage"] - best) < 1e-12)
