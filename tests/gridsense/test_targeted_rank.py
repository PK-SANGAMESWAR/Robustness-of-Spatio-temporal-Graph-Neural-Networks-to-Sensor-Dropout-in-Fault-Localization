"""Tests for the corrected targeted-dropout rank check (rag-for-dropout/reports/findings/paper1_open_switch.md § 11). No model, no GPU."""
from __future__ import annotations

import numpy as np
import pytest

from gridsense.eval import targeted_rank_check as T


def test_macro_f1_from_confusion_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 26, 4000)
    p = np.where(rng.random(4000) < 0.7, y, rng.integers(0, 26, 4000))
    cm = np.bincount(y * 26 + p, minlength=676).reshape(26, 26).astype(float)
    assert T.macro_f1_from_cm(cm) == pytest.approx(f1_score(y, p, average="macro", zero_division=0), abs=1e-12)


def test_paired_bootstrap_is_zero_for_identical_predictions_and_signs_a_gain():
    rng = np.random.default_rng(1)
    runs = np.repeat(np.arange(50), 40)
    y = rng.integers(0, 26, runs.size)
    good = y.copy()
    bad = np.where(rng.random(y.size) < 0.5, y, (y + 1) % 26)
    same = T.paired_run_bootstrap(runs, y, bad, bad, 200)
    assert same["ci_low"] == 0.0 == same["ci_high"]
    gain = T.paired_run_bootstrap(runs, y, bad, good, 200)
    assert gain["ci_low"] > 0 and gain["n_runs"] == 50


def test_changed_labels_compares_sets_not_order():
    a = np.tile(np.arange(25), (25, 1))
    b = a.copy()
    b[3, :2] = [1, 0]            # same set {0, 1}: order only
    b[4, :3] = [0, 1, 7]         # set {0, 1, 7} differs from {0, 1, 2}
    assert T.changed_labels(a, b, 2) == []
    assert T.changed_labels(a, b, 3) == [5]


def test_operated_rank_changes_only_the_expected_fault_buses():
    """Needs OpenDSS and the frozen corpus files (read only). The counts are those of reports/findings/paper1_open_switch.md § 5."""
    saved = np.load(T.PAPER / "proximity_rank.npy")
    op = T.operated_rank(T.DATA / "processed" / "node_mapping.json")
    assert all(sorted(r.tolist()) == list(range(25)) for r in op)
    assert [len(T.changed_labels(saved, op, k)) for k in (1, 2, 5, 10)] == [1, 2, 7, 11]
    same = [lab for lab in range(1, 26) if lab not in T.changed_labels(saved, op, 10)]
    assert all(set(saved[lab - 1, :10]) == set(op[lab - 1, :10]) for lab in same)
