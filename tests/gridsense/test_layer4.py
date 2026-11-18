"""Layer 4 (deterministic, with ground truth). Hand-built reports and events; no model, no API call."""
from __future__ import annotations

from gridsense.eval import layer4 as L4

from .test_s4 import _item


def _ev(gate="CONFIDENT", nb=None, pred=55):
    nb = nb if nb is not None else [55] * 10
    return {"event_object": _item(gate_outcome=gate, predicted_bus=pred, knn={"agreement": 1.0, "neighbours": [{"bus": b, "similarity": 0.9, "store_prediction_correct": b == pred} for b in nb]})["event_object"]}


def _it(correct, final="sufficient", loc=55, **rep):
    return {"prediction_correct": correct, "evidence": {"final": final}, "stratum": "x", "event_id": "e", "report": {"predicted_location": loc, "case_evidence_summary": "10 of 10.", "evidence_status_reason": "ok", "limitations": [], **rep}}


def test_L4_location_and_doubt_sources_by_hand():
    r = L4.judge_item(_it(True), _ev())
    assert r["names_predicted_bus"] and not r["doubt_stated"] and not r["doubt_without_cause"] and r["no_signal_against_prediction"]
    assert not L4.judge_item(_it(True, loc=53), _ev())["names_predicted_bus"]


def test_L4_a_wrong_prediction_with_all_neighbours_agreeing_has_no_signal_against_it():
    r = L4.judge_item(_it(False), _ev(nb=[55] * 10))
    assert r["no_signal_against_prediction"] and not r["doubt_stated"] and not r["doubt_cause_in_event"]
    r = L4.judge_item(_it(False), _ev(nb=[55] * 6 + [53] * 4))                                   # neighbours disagree: a cause exists
    assert not r["no_signal_against_prediction"] and r["doubt_cause_in_event"]


def test_L4_doubt_by_gate_by_insufficient_evidence_and_by_text():
    assert L4.judge_item(_it(False), _ev(gate="LOW_CONFIDENCE"))["sources"]["gate_not_confident"]
    r = L4.judge_item(_it(True, final="insufficient"), _ev())
    assert r["sources"]["evidence_insufficient"] and r["doubt_stated"] and r["doubt_without_cause"]        # doubt with no cause in the event
    r = L4.judge_item(_it(False, limitations=["The neighbours disagree with the predicted bus."]), _ev(nb=[55] * 5 + [53] * 5))
    assert r["sources"]["text_doubt"] and r["doubt_stated"] and r["doubt_cause_in_event"] and not r["doubt_without_cause"]


def test_L4_summary_has_rates_with_n_and_empty_cells_are_none():
    rows = [L4.judge_item(_it(False), _ev()) | {"stratum": "high_conf_error", "event_id": "a"}, L4.judge_item(_it(True), _ev()) | {"stratum": "confident_correct", "event_id": "b"}]
    s = L4.summarize(rows)
    assert s["high_conf_error"]["wrong_predictions"]["no_signal_against_prediction"] == {"rate": 1.0, "n": 1}
    assert s["confident_correct"]["wrong_predictions"]["doubt_stated"] == {"rate": None, "n": 0}               # never 0 for an empty cell
    assert s["high_conf_error"]["distinct_events"] == 1 and s["low_confidence"]["reports"] == 0
