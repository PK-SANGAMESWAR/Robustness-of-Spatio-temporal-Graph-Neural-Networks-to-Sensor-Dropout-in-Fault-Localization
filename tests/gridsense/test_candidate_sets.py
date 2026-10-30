"""LOW_CONFIDENCE candidate sets (O26): frozen-config hash, set rule, conditional coverage. Synthetic data."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gridsense.eval.gate_fit import conformal_quantile  # noqa: E402
from gridsense.verification.candidate_sets import CandidateSets  # noqa: E402
from gridsense.verification.gate import canonical_hash  # noqa: E402


def _cfg() -> dict:
    cfg = {"per_k": {"0": {"q": 0.8}, "10": {"q": 0.95}}}
    cfg["config_sha256"] = canonical_hash(cfg)
    return cfg


def test_set_rule_and_unknown_k() -> None:
    cs = CandidateSets(_cfg())
    votes = np.zeros((1, 26))
    votes[0, :3] = np.array([0.5, 0.25, 0.25]) * 25
    assert cs.sets(0, votes)[0].sum() == 3          # shares >= 1 - 0.8 = 0.2: three classes
    assert cs.sets(10, votes)[0].sum() == 3         # share >= 1 - 0.95 = 0.05: still the three classes with nonzero share
    full = _cfg()
    full["per_k"]["0"]["q"] = 1.0
    full["config_sha256"] = canonical_hash({k: v for k, v in full.items() if k != "config_sha256"})
    assert CandidateSets(full).sets(0, votes)[0].sum() == 26   # q = 1: threshold 0, every class is in the set
    with pytest.raises(KeyError):
        cs.sets(3, votes)


def test_frozen_hash_detects_change() -> None:
    cfg = _cfg()
    CandidateSets(cfg)
    bad = copy.deepcopy(cfg)
    bad["per_k"]["0"]["q"] = 0.7
    with pytest.raises(ValueError):
        CandidateSets(bad)


def test_conditional_conformal_covers_90_percent_of_the_selected_subpopulation() -> None:
    rng = np.random.default_rng(0)
    n = 40000
    probs = rng.dirichlet(np.full(26, 0.1), size=n)
    low = probs.max(1) < np.quantile(probs.max(1), 0.3)   # selection by features only (as a gate does): the 30 % least confident windows
    y = np.where(low, rng.integers(0, 26, n), probs.argmax(1))   # confident windows are right, the selected windows are close to guesses
    share_true = probs[np.arange(n), y]
    a, b = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    q = conformal_quantile(share_true[a & low])
    q_all = conformal_quantile(share_true[a])
    cond = (share_true[b & low] >= 1 - q).mean()
    allw = (share_true[b & low] >= 1 - q_all).mean()
    assert cond >= 0.88                               # 90 % target; coverage of one split has about +-0.003 sampling noise per 20,000 windows (more for fewer)
    assert cond - allw > 0.05                         # the all-window quantile clearly under-covers the selected subpopulation
