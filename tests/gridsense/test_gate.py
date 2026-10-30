"""The deterministic gate (O21-O23, 2026-10-30): decisions, reason codes, unsupported dropout patterns, frozen-config hash, conformal sets. Synthetic."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gridsense.eval.gate_fit import ALPHA, conformal_quantile, fit_k  # noqa: E402
from gridsense.verification.gate import (CONFIDENT, LOW_CONFIDENCE, R_INVALID, R_LOW_AGREE, R_LOW_MARGIN, R_LOW_P, R_UNSUPPORTED, REJECT,  # noqa: E402
                                         Gate, canonical_hash, reasons_to_names)


def _cfg() -> dict:
    th = {"feasible": True, "tau_p": 0.8, "tau_m": 0.3, "tau_k": 0.6}
    k = {"iso_x": [0.0, 0.5, 1.0], "iso_y": [0.0, 0.5, 1.0], "acc95": th, "acc98": th, "conformal_q": 0.6, "lr": None}
    cfg = {"supported_patterns": ["random"], "per_k": {"0": copy.deepcopy(k), "10": copy.deepcopy(k)}}
    cfg["config_sha256"] = canonical_hash(cfg)
    return cfg


def _votes(shares: list[float]) -> np.ndarray:
    v = np.zeros(26)
    v[: len(shares)] = np.array(shares) * 25.0
    return v


def _shares(*s: float) -> np.ndarray:
    v = np.zeros((1, 26))
    v[0, : len(s)] = np.array(s) * 25.0
    return v


def test_confident_and_each_failure_reason() -> None:
    g = Gate(_cfg())                                   # identity isotonic map; tau_p 0.8, tau_m 0.3, tau_k 0.6
    d = g.decide(0, _shares(0.95, 0.05), np.array([0.9]))
    assert d["outcome"][0] == CONFIDENT and d["reason"][0] == 0
    d = g.decide(0, _shares(0.75, 0.25), np.array([0.9]))            # p 0.75 < 0.8, margin 0.5 ok
    assert d["outcome"][0] == LOW_CONFIDENCE and d["reason"][0] == R_LOW_P
    d = g.decide(0, _shares(0.95, 0.05), np.array([0.4]))            # agreement 0.4 < 0.6
    assert d["outcome"][0] == LOW_CONFIDENCE and d["reason"][0] == R_LOW_AGREE and d["set"][0].sum() >= 1
    strict = _cfg()
    strict["per_k"]["0"]["acc95"]["tau_m"] = 0.75
    strict["config_sha256"] = canonical_hash(strict)
    d = Gate(strict).decide(0, _shares(0.85, 0.15), np.array([0.9]))  # p 0.85 ok, margin 0.70 < 0.75
    assert d["outcome"][0] == LOW_CONFIDENCE and d["reason"][0] == R_LOW_MARGIN
    d = Gate(strict).decide(0, _shares(0.7, 0.3), np.array([0.1]))     # all three fail
    assert d["reason"][0] == R_LOW_P | R_LOW_MARGIN | R_LOW_AGREE
    assert reasons_to_names(R_LOW_P | R_LOW_AGREE) == ["LOW_CALIBRATED_CONFIDENCE", "LOW_KNN_AGREEMENT"]
    assert reasons_to_names(0) == ["VERIFICATION_PASSED"]


def test_unsupported_k_or_pattern_is_rejected_with_reason_code() -> None:
    g = Gate(_cfg())
    v = _votes([0.95, 0.05])[None, :]
    for k, pattern in ((3, "random"), (25, "random"), (0, "targeted"), (10, "contiguous"), (5, "random")):
        d = g.decide(k, v, np.array([0.9]), pattern=pattern)
        assert d["outcome"][0] == REJECT and d["reason"][0] == R_UNSUPPORTED, (k, pattern)
        assert reasons_to_names(int(d["reason"][0])) == ["UNSUPPORTED_DROPOUT_PATTERN"]
    assert g.decide(10, v, np.array([0.9]))["outcome"][0] == CONFIDENT      # a fitted k with the fitted pattern


def test_invalid_input_is_rejected() -> None:
    g = Gate(_cfg())
    good = _votes([0.95, 0.05])
    for bad_votes, agree in ((np.where(np.arange(26) == 3, np.nan, good), 0.9), (good * 2, 0.9), (good, np.nan)):
        d = g.decide(0, bad_votes[None, :], np.array([agree]))
        assert d["outcome"][0] == REJECT and d["reason"][0] == R_INVALID


def test_frozen_config_hash_detects_any_change() -> None:
    cfg = _cfg()
    Gate(cfg)
    changed = copy.deepcopy(cfg)
    changed["per_k"]["0"]["acc95"]["tau_p"] = 0.79
    with pytest.raises(ValueError):
        Gate(changed)


def test_conformal_quantile_gives_at_least_90_percent_coverage() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    probs = rng.dirichlet(np.full(26, 0.15), size=n)
    y = np.array([rng.choice(26, p=p) for p in probs])
    share_true = probs[np.arange(n), y]
    a, b = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    q = conformal_quantile(share_true[a], ALPHA)
    covered = share_true[b] >= 1 - q
    assert covered.mean() >= 0.895                                       # finite-sample guarantee, small slack for sampling noise
    sets = probs >= 1 - q
    assert sets.sum(1).mean() >= 1.0


def test_fit_k_thresholds_meet_the_target_on_cal_b() -> None:
    rng = np.random.default_rng(1)
    n = 8000
    logits = rng.normal(size=(n, 26)) * 1.5
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(26, p=p) for p in probs])
    votes = probs * 25.0
    agree = np.clip(np.round(probs.max(1) + rng.normal(0, 0.15, n), 1), 0, 1)
    c = SimpleNamespace(votes=votes, y=y, knn={"heldout_agree_10": agree})
    A, B = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    res = fit_k(c, A, B)
    g95 = res["acc95"]
    if g95["feasible"]:
        assert g95["cal_b_accuracy"] >= 0.95 - 1e-12
        gate = Gate({"supported_patterns": ["random"], "per_k": {"0": res}})
        d = gate.decide(0, votes[B], agree[B])
        conf = d["outcome"] == CONFIDENT
        assert abs(conf.mean() - g95["cal_b_coverage"]) < 1e-12
        assert ((votes[B].argmax(1) == y[B])[conf]).mean() >= 0.95 - 1e-12
    assert 0 < res["conformal_q"] <= 1
