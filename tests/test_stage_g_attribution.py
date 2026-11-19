"""S7.1 attribution helpers on hand-built data (no result files needed)."""
from __future__ import annotations

import numpy as np
import pytest

from fault_localization.stage_g import attribution as A


def _D():
    D = np.full((26, 26), 50.0)
    np.fill_diagonal(D, 0.0)
    D[3, 4] = D[4, 3] = 0.1                      # one indistinguishable pair (3, 4)
    return D


def test_labelable_drops_only_w20():
    ids = ["e_w0019", "e_w0020", "e_w0021", "e_w0000"]
    assert A.labelable(ids).tolist() == [True, False, True, True]


def test_attribute_counts_and_chance():
    D = _D()
    y = np.array([3, 3, 3, 7, 7, 3])
    pred = np.array([4, 4, 3, 8, 7, 9])
    ids = [f"e{i}_w0030" for i in range(6)]
    r = A.attribute(y, pred, ids, D)
    assert r["n_wrong"] == 4 and r["n_errors_in_indist_pairs"] == 2 and r["frac_errors_in_indist_pairs"] == pytest.approx(0.5)
    assert r["n_indist_pairs"] == 1 and r["chance_frac"] == pytest.approx(2 / (26 * 25))
    assert r["frac_indist_class_windows_wrong"] == pytest.approx(3 / 4)       # windows of class 3: 3 wrong of 4
    assert r["frac_other_class_windows_wrong"] == pytest.approx(1 / 2)


def test_attribute_excludes_w20_errors():
    D = _D()
    y = np.array([3, 3])
    pred = np.array([0, 0])                                                  # a model that says "no fault" everywhere
    r = A.attribute(y, pred, ["e_w0020", "e_w0025"], D)
    assert r["n_windows"] == 1 and r["n_wrong"] == 1


def test_no_errors_gives_nan_fraction():
    r = A.attribute(np.array([1, 2]), np.array([1, 2]), ["a_w0030", "b_w0030"], _D())
    assert r["n_wrong"] == 0 and np.isnan(r["frac_errors_in_indist_pairs"])


def test_top_confusions_sorted_with_D():
    D = _D()
    y = np.array([3, 3, 5, 5, 5])
    pred = np.array([4, 4, 6, 6, 6])
    top = A.top_confusions(y, pred, [f"x{i}_w0030" for i in range(5)], D)
    assert top[0] == {"true": 5, "pred": 6, "count": 3, "D": 50.0} and top[1]["D"] == pytest.approx(0.1)


def test_real_D_matrices_are_valid():
    for k in (0, 10):
        D = A.load_D(k)
        assert (D >= 0).all() and np.diag(D).max() == 0
