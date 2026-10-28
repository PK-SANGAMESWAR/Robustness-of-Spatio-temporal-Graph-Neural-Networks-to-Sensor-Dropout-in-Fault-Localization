"""Tolerance rule of the S2 identity gate (author, 2026-10-28): CPU predictions may differ from the paper only where
the vote gap is below 2e-3. Synthetic votes only; no model or data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridsense.eval.identity import GAP_TOL, tolerance_check  # noqa: E402


def _votes(top_two: list[tuple[int, float, int, float]], n_classes: int = 26) -> np.ndarray:
    v = np.full((len(top_two), n_classes), 0.0)
    for i, (c1, v1, c2, v2) in enumerate(top_two):
        v[i, c1], v[i, c2] = v1, v2
    return v


def test_identical_predictions_pass() -> None:
    v = _votes([(3, 20.0, 5, 1.0), (7, 15.0, 2, 14.0)])
    r = tolerance_check(v, v.argmax(1), ["a", "b"])
    assert r["n_diff"] == 0 and r["rule_ok"] and r["max_gap"] == 0.0


def test_near_tie_within_rule_passes_and_reports_gap() -> None:
    v = _votes([(3, 10.0010, 5, 10.0000)])  # CPU says 3, paper said 5, gap 1e-3 < 2e-3
    r = tolerance_check(v, np.array([5]), ["w"])
    assert r["n_diff"] == 1 and r["rule_ok"] and abs(r["max_gap"] - 1e-3) < 1e-9


def test_difference_above_rule_fails() -> None:
    v = _votes([(3, 10.0, 5, 9.99)])  # gap 1e-2 >= 2e-3
    r = tolerance_check(v, np.array([5]), ["w"])
    assert not r["rule_ok"] and r["violations"][0]["window_id"] == "w"


def test_gap_equal_to_tolerance_is_a_violation() -> None:
    v = _votes([(3, 10.0 + GAP_TOL, 5, 10.0)])
    assert not tolerance_check(v, np.array([5]), ["w"])["rule_ok"]  # rule is strictly below 2e-3
