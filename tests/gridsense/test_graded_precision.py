"""Graded precision@3 and nDCG@3 from labels: hand-calculated values."""
from __future__ import annotations

import math

import pytest

from gridsense.eval import graded_precision as GP


def rows(qid, labels, split="dev", answerable="yes"):
    return [{"query_id": qid, "split": split, "answerable": answerable, "rank": str(i + 1), "label": lab} for i, lab in enumerate(labels)]


def test_graded_and_strict_precision_and_ndcg_by_hand():
    data = rows("A", ["relevant", "not relevant", "partly relevant"]) + rows("B", ["not relevant", "relevant", "relevant"]) + rows("C", ["n", "n", "n"])
    r = GP.compute(data)["dev"]
    assert r["graded_p@3"] == pytest.approx(((1 + 0 + 0.5) / 3 + (0 + 1 + 1) / 3 + 0) / 3)
    assert r["strict_p@3"] == pytest.approx(((1 / 3) + (2 / 3) + 0) / 3)
    ideal_a = 3 / math.log2(2) + 1 / math.log2(3)
    dcg_a = 3 / math.log2(2) + 0 + 1 / math.log2(4)
    ideal_b = 3 / math.log2(2) + 3 / math.log2(3)
    dcg_b = 0 + 3 / math.log2(3) + 3 / math.log2(4)
    assert r["ndcg@3"] == pytest.approx((dcg_a / ideal_a + dcg_b / ideal_b + 0.0) / 3)
    assert r["queries_without_any_relevant_or_partly"] == ["C"]


def test_unanswerable_queries_report_relevant_passages_and_short_labels_work():
    data = rows("A", ["r", "p", "n"]) + rows("U1", ["n", "n", "n"], answerable="no") + rows("U2", ["n", "p", "n"], answerable="no")
    r = GP.compute(data)["dev"]
    assert r["unanswerable_queries"] == 2 and r["unanswerable_passages_labelled_relevant_or_partly"] == 1 and r["unanswerable_queries_with_such_passages"] == ["U2"]


def test_missing_or_unknown_labels_stop_the_run():
    with pytest.raises(SystemExit):
        GP.compute(rows("A", ["relevant", "", "not relevant"]))
    with pytest.raises(ValueError):
        GP.compute(rows("A", ["relevant", "maybe", "not relevant"]))
