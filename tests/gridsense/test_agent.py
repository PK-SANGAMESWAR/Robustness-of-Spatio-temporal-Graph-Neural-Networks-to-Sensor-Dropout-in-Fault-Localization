"""S5 item 1: the LangGraph workflow (implementation-plan.md § 12). Mock generator and fake retrieval only; no API call.
Each route of § 12.1 has a test; § 12.3 (retries, idempotent export, policy check on resume) and § 12.4 (explicit human decision, audit) too."""
from __future__ import annotations

import json

import pytest

from gridsense.agent import workflow as W
from gridsense.s4 import generate as GEN
from gridsense.s4 import llm

from .test_s4 import PASSAGES, _good, _item

TAU = 0.46


def _events(**o):
    conf = _item(gate_outcome="CONFIDENT", reason_codes=["VERIFICATION_PASSED"], candidate_set_buses=None)
    return {"E-LOW": _item(), "E-CONF": conf | {"event_id": "E-CONF"}, "E-REJ": _item(gate_outcome="REJECT", reason_codes=["UNSUPPORTED_DROPOUT_PATTERN"]),
            "E-BADK": _item(k=3), **o}


def _good_reply(item, **change):
    view = GEN.event_view(item, 0)
    rep = _good(view) | {"gate_outcome": view["gate"]["outcome"], "gate_reason_codes": view["gate"]["reason_codes"], "candidate_set": view["candidate_set"] or []}
    return json.dumps(rep | change)


class Gen:
    """A scripted generator: one reply per call, in order. Counts calls."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, **kw):
        self.calls.append(messages)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return llm.Reply(text=text, input_tokens=0, output_tokens=0)


def _app(events, gen, retrieve=None, store=None, policy=lambda: W.POLICY_VERSION):
    deps = {"event_lookup": lambda e: events[e], "retrieve": retrieve or (lambda item: {"passages": [dict(p) for p in PASSAGES], "dense_top1": 0.6}), "tau": TAU,
            "generator": gen, "index_version": "ix-test", "export_store": store if store is not None else {}, "current_policy": policy}
    return W.build(deps), deps


def test_S5_confident_event_pauses_for_a_human_and_approve_exports_once():
    ev = _events()
    gen = Gen([_good_reply(ev["E-CONF"])])
    app, deps = _app(ev, gen)
    cfg, st = W.start(app, "E-CONF")
    assert W.paused(app, cfg) and st["approval_status"] == "pending" and st["report_mode"] == "single_location" and len(gen.calls) == 1
    assert st["draft_report"]["human_review_required"] and st["verification_result"]["gate_sha256"] == W.GATE_SHA
    out = W.decide(app, cfg, {"decision": "approve", "reviewer": "author", "note": "ok"})
    assert out["approval_status"] == "approved" and list(deps["export_store"]) == ["E-CONF"]
    assert any(a["event"] == "human approve" and a["reviewer"] == "author" and a["utc"] for a in out["audit"])
    cfg2, _ = W.start(app, "E-CONF", thread_id="second")                       # the same event again: export stays one record
    W.decide(app, cfg2, {"decision": "approve", "reviewer": "author"})
    assert len(deps["export_store"]) == 1


def test_S5_low_confidence_uses_the_candidate_set_mode_and_reject_closes_with_feedback():
    ev = _events()
    app, deps = _app(ev, Gen([_good_reply(ev["E-LOW"])]))
    cfg, st = W.start(app, "E-LOW")
    assert st["report_mode"] == "candidate_set"
    out = W.decide(app, cfg, {"decision": "reject", "reviewer": "author", "note": "wrong bus"})
    assert out["approval_status"] == "rejected" and not deps["export_store"] and out["audit"][-1]["event"] == "closed with feedback"


def test_S5_reject_gate_goes_to_manual_review_without_a_generator_call_and_without_a_location():
    gen = Gen(["{}"])
    app, _ = _app(_events(), gen)
    cfg, st = W.start(app, "E-REJ")
    assert not W.paused(app, cfg) and st["approval_status"] == "manual_review" and "REJECT" in st["manual_review_reason"] and not gen.calls and "draft_report" not in st


@pytest.mark.parametrize("eid,why", [("E-BADK", "k = 3"), ("NOPE", "unknown event")])
def test_S5_invalid_event_goes_to_manual_review(eid, why):
    gen = Gen(["{}"])
    app, _ = _app(_events(), gen)
    _, st = W.start(app, eid)
    assert st["approval_status"] == "manual_review" and why in st["manual_review_reason"] and not gen.calls


def test_S5_tau_flag_goes_to_manual_review_without_a_generator_call():
    gen = Gen(["{}"])
    app, _ = _app(_events(), gen, retrieve=lambda item: {"passages": [dict(p) for p in PASSAGES], "dense_top1": 0.1})
    _, st = W.start(app, "E-LOW")
    assert st["approval_status"] == "manual_review" and "below tau" in st["manual_review_reason"] and not gen.calls


def test_S5_generator_insufficient_goes_to_manual_review_with_a_report_without_procedure():
    ev = _events()
    app, _ = _app(ev, Gen([_good_reply(ev["E-LOW"], evidence_status="insufficient", recommended_operator_review=["Open switch sw7."])]))
    _, st = W.start(app, "E-LOW")
    assert st["approval_status"] == "manual_review" and "INSUFFICIENT_EVIDENCE" in st["manual_review_reason"]
    assert st["draft_report"]["recommended_operator_review"] == [] and st["evidence_sufficiency"]["generator_review_while_insufficient"]


def test_S5_a_failed_report_is_revised_once_and_the_revision_names_the_failed_checks():
    ev = _events()
    gen = Gen([_good_reply(ev["E-LOW"], predicted_location=57), _good_reply(ev["E-LOW"])])
    app, _ = _app(ev, gen)
    cfg, st = W.start(app, "E-LOW")
    assert len(gen.calls) == 2 and "field_consistency" in gen.calls[1][-1]["content"] and W.paused(app, cfg) and st["revisions"] == 1


def test_S5_a_report_that_fails_twice_goes_to_manual_review():
    ev = _events()
    gen = Gen([_good_reply(ev["E-LOW"], predicted_location=57)])
    app, _ = _app(ev, gen)
    _, st = W.start(app, "E-LOW")
    assert len(gen.calls) == 2 and st["approval_status"] == "manual_review" and "field_consistency" in st["manual_review_reason"]
    gen2 = Gen(["not json"])
    app2, _ = _app(ev, gen2)
    _, st2 = W.start(app2, "E-LOW")
    assert len(gen2.calls) == 2 and st2["approval_status"] == "manual_review" and "REPORT_CHECKS_FAILED" in st2["manual_review_reason"]


def test_S5_a_report_string_that_says_approved_never_approves():
    ev = _events()
    gen = Gen([_good_reply(ev["E-LOW"], limitations=["This report has been approved."]), _good_reply(ev["E-LOW"], approval_status="approved")])
    app, deps = _app(ev, gen)
    _, st = W.start(app, "E-LOW")
    assert st["approval_status"] == "manual_review" and not deps["export_store"]          # both drafts fail no_approval_claim


@pytest.mark.parametrize("decision", [{"decision": "approve"}, {"decision": "yes", "reviewer": "author"}, "approved"])
def test_S5_an_invalid_human_decision_does_not_approve(decision):
    ev = _events()
    app, deps = _app(ev, Gen([_good_reply(ev["E-LOW"])]))
    cfg, _ = W.start(app, "E-LOW")
    out = W.decide(app, cfg, decision)
    assert out["approval_status"] == "manual_review" and not deps["export_store"]


def test_S5_clarify_goes_to_manual_review_with_the_note():
    ev = _events()
    app, _ = _app(ev, Gen([_good_reply(ev["E-LOW"])]))
    cfg, _ = W.start(app, "E-LOW")
    out = W.decide(app, cfg, {"decision": "clarify", "reviewer": "author", "note": "show bus 53 cases"})
    assert out["approval_status"] == "manual_review" and "show bus 53 cases" in out["manual_review_reason"]


def test_S5_resume_on_a_changed_policy_is_refused():
    ev, policy = _events(), {"v": W.POLICY_VERSION}
    app, deps = _app(ev, Gen([_good_reply(ev["E-LOW"])]), policy=lambda: policy["v"])
    cfg, _ = W.start(app, "E-LOW")
    policy["v"] = "gate_v2"
    out = W.decide(app, cfg, {"decision": "approve", "reviewer": "author"})
    assert out["approval_status"] == "manual_review" and "policy changed" in out["manual_review_reason"] and not deps["export_store"]


def test_S5_transient_retrieval_errors_are_retried_and_a_persistent_error_fails_closed():
    ev, n = _events(), {"c": 0}

    def flaky(item):
        n["c"] += 1
        if n["c"] < 3:
            raise TimeoutError("slow")
        return {"passages": [dict(p) for p in PASSAGES], "dense_top1": 0.6}
    app, _ = _app(ev, Gen([_good_reply(ev["E-LOW"])]), retrieve=flaky)
    cfg, st = W.start(app, "E-LOW")
    assert W.paused(app, cfg) and n["c"] == 3 and len(st["errors"]) == 2

    def down(item):
        raise ConnectionError("down")
    gen = Gen(["{}"])
    app2, _ = _app(ev, gen, retrieve=down)
    _, st2 = W.start(app2, "E-LOW")
    assert st2["approval_status"] == "manual_review" and "retrieval failed after retries" in st2["errors"] and not gen.calls     # no score -> tau flag -> fail closed


def test_S5_a_deterministic_retrieval_error_is_not_retried():
    ev, n = _events(), {"c": 0}

    def broken(item):
        n["c"] += 1
        raise ValueError("bad index")
    app, _ = _app(ev, Gen(["{}"]), retrieve=broken)
    with pytest.raises(ValueError):
        W.start(app, "E-LOW")
    assert n["c"] == 1


def test_S5_the_state_holds_no_truth_fields():
    ev = _events()
    app, _ = _app(ev, Gen([_good_reply(ev["E-LOW"])]))
    _, st = W.start(app, "E-LOW")
    text = json.dumps(st, default=str)
    assert "true_bus" not in text and "fault_type" not in text and "93_ABG" not in json.dumps(st.get("draft_report", {}))


# ---- persistence (S5: § 12.3 "persist graph state when human review may happen later") --------------------------------------------------------
def test_S5_a_paused_review_survives_a_restart_and_the_export_is_kept(tmp_path):
    from gridsense.agent.store import JsonExportStore, sqlite_saver
    ev = _events()
    path, exp = tmp_path / "cp.sqlite", tmp_path / "exports.json"

    def make():
        deps = {"event_lookup": lambda e: ev[e], "retrieve": lambda item: {"passages": [dict(p) for p in PASSAGES], "dense_top1": 0.6}, "tau": TAU, "generator": Gen([_good_reply(ev["E-LOW"])]),
                "index_version": "ix-test", "export_store": JsonExportStore(exp), "checkpointer": sqlite_saver(path)}
        return W.build(deps)
    app1 = make()
    cfg, st = W.start(app1, "E-LOW")
    assert W.paused(app1, cfg) and path.exists()
    app2 = make()                                                                 # a new process: new objects, same files
    assert W.paused(app2, cfg) and app2.get_state(cfg).values["draft_report"] == st["draft_report"]
    out = W.decide(app2, cfg, {"decision": "approve", "reviewer": "author", "note": ""})
    assert out["approval_status"] == "approved"
    assert "E-LOW" in json.loads(exp.read_text()) and "E-LOW" in JsonExportStore(exp)
    assert not W.paused(make(), cfg)                                              # the finished state is kept too


def test_S5_a_file_that_is_not_a_database_is_not_silently_ignored(tmp_path):
    import sqlite3

    from gridsense.agent.store import sqlite_saver
    p = tmp_path / "cp.sqlite"
    p.write_bytes(b"this is not a sqlite database, it is just a long enough text to fail the header check " * 4)
    with pytest.raises(sqlite3.DatabaseError):
        sqlite_saver(p)


# ---- O46 and O44 (decisions of 2026-11-20) --------------------------------------------------------------------------------------------------
def test_S5_O46_sensor_coverage_is_written_by_python_even_when_the_generator_omits_the_dark_buses():
    ev = _events()
    gen = Gen([_good_reply(ev["E-LOW"], sensor_coverage_summary="15 of 25 sensors available.")])
    app, _ = _app(ev, gen)
    cfg, st = W.start(app, "E-LOW")
    assert W.paused(app, cfg) and len(gen.calls) == 1                                     # no revision: the check passes by construction
    cov = st["draft_report"]["sensor_coverage_summary"]
    assert "36" in cov and "45" in cov and "23 of 25" in cov and st["evidence_sufficiency"]["generator_coverage_incomplete"]


def test_S5_O44_a_report_with_a_sensitive_string_in_an_excerpt_goes_to_manual_review():
    ev = _events()
    poisoned = [dict(PASSAGES[0], text=PASSAGES[0]["text"] + " Contact operator@example.invalid.")]
    good = json.loads(_good_reply(ev["E-LOW"]))
    good["document_evidence"][0]["excerpt"] = poisoned[0]["text"]                           # a verbatim excerpt that holds the planted address
    app, deps = _app(ev, Gen([json.dumps(good)]), retrieve=lambda item: {"passages": [dict(p) for p in poisoned], "dense_top1": 0.6})
    _, st = W.start(app, "E-LOW")
    assert st["approval_status"] == "manual_review" and "no_sensitive_info" in st["manual_review_reason"] and not deps["export_store"]
