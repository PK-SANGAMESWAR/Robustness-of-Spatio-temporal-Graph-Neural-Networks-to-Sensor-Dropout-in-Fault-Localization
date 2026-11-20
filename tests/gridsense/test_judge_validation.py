"""Judge validation (O13): kappa by hand and against sklearn, the pre-registered rules, and the refusal of missing labels. Synthetic labels only; no real label exists yet."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from gridsense.eval import judge_validation as JV


def test_kappa_by_hand_and_against_sklearn():
    a, b = np.array([1, 1, 0, 0, 1, 0, 1, 1]), np.array([1, 0, 0, 0, 1, 1, 1, 1])
    assert JV.kappa(a, b, [0, 1]) == pytest.approx(cohen_kappa_score(a, b)) == pytest.approx(0.4666666, abs=1e-6)
    x, y = np.array([1, 2, 3, 4, 5, 3, 4, 2]), np.array([1, 3, 3, 5, 5, 2, 4, 2])
    assert JV.kappa(x, y, [1, 2, 3, 4, 5], "linear") == pytest.approx(cohen_kappa_score(x, y, weights="linear"))
    assert JV.kappa(x, x, [1, 2, 3, 4, 5], "linear") == 1.0


def test_kappa_is_undefined_not_zero_or_one_without_variance():
    five = np.array([5, 5, 5, 5])
    assert np.isnan(JV.kappa(five, five, [1, 2, 3, 4, 5], "linear"))


def _rows(author_r, judge_r, author_f, judge_f):
    sheet, scores = [], []
    for i in range(len(author_r)):
        lid = f"H-{i + 1:04d}"
        sheet.append({"label_id": lid, "report_ref": f"x|event-{i // 2:03d}|repeat{i % 2}", "faithfulness_0_or_1": str(author_f[i]), "notes": "",
                      **{f"{r}_1_to_5": str(author_r[i]) for r in JV.RUBRICS}})
        scores.append({"label_id": lid, "judge_faithfulness": judge_f[i], **{f"judge_{r}": judge_r[i] for r in JV.RUBRICS}})
    return sheet, scores


def test_validate_marks_good_and_poor_agreement_and_applies_the_bar():
    rng = np.random.default_rng(1)
    ar = rng.integers(1, 6, 60)
    sheet, scores = _rows(ar, ar, [1, 0] * 30, [0.95, 0.2] * 30)                      # perfect agreement
    res = JV.validate(sheet, scores)
    assert res["metrics"]["adherence"]["kappa"] == pytest.approx(1.0) and res["metrics"]["adherence"]["status"] == "usable" and res["metrics"]["faithfulness"]["status"] == "usable"
    noisy = rng.integers(1, 6, 60)
    res = JV.validate(*_rows(ar, noisy, [1, 0] * 30, [0.95, 0.95] * 30))             # unrelated scores; the judge calls everything faithful
    assert res["metrics"]["adherence"]["status"] == "informational" and res["metrics"]["faithfulness"]["status"] == "informational"
    assert res["metrics"]["adherence"]["kappa_ci95_by_event"] is not None


def test_faithfulness_threshold_is_the_preregistered_0_9():
    sheet, scores = _rows([3] * 4, [3] * 4, [1, 1, 0, 0], [0.9, 0.89, 0.1, 0.95])
    m = JV.validate(sheet, scores)["metrics"]["faithfulness"]
    assert m["judge_distribution"] == {"0": 2, "1": 2} and m["agreement_exact"] == 0.5


def test_a_missing_or_invalid_label_stops_the_run_and_never_becomes_a_score():
    sheet, scores = _rows([3, 4], [3, 4], [1, 0], [1.0, 0.0])
    sheet[1]["evidence_use_1_to_5"] = ""
    with pytest.raises(ValueError, match="label missing"):
        JV.validate(sheet, scores)
    sheet[1]["evidence_use_1_to_5"] = "7"
    with pytest.raises(ValueError, match="not in"):
        JV.validate(sheet, scores)


def test_labels_file_rows_follow_the_schema():
    sheet, _ = _rows([3, 4], [3, 4], [1, 0], [1.0, 0.0])
    rows = JV.to_labels(sheet)
    assert rows[0]["labeller"] == "author" and set(rows[0]["rubric_scores"]) == set(JV.RUBRICS) and rows[0]["faithfulness"] == 1 and rows[1]["faithfulness"] == 0
