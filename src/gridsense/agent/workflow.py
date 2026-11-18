"""S5 item 1: the LangGraph workflow (implementation-plan.md § 12; spec/impl_08.md § 12.1 to 12.4). It orchestrates; it never replaces a deterministic check.

Graph (§ 12.1, Rev 2):
  validate_event -(invalid)-> manual_review
  run_or_load_prediction -> verify_prediction -(REJECT)-> manual_review
  build_retrieval_query -> retrieve_cases -> compute_evidence_score -> retrieve_evidence -> rerank_if_enabled
  check_evidence_sufficiency -(tau flag)-> manual_review            (no generator call: fail-closed, no cost)
  draft_report -> validate_report_and_citations -(fail)-> revise once -(fail again)-> manual_review
                                                -(generator says insufficient)-> manual_review (assembled report: no procedure, E + A)
  human_approval (interrupt; explicit decision) -(reject)-> closed_with_feedback | -(approve)-> export_approved_report | -(clarify)-> manual_review

Rules carried in code: the gate, the evidence decision and the approval state are Python (claude-rag.md § 4 rule 3); a report string never changes them.
The prediction and the gate decision are LOADED from the event object (frozen gate_v1 decision, Layer 0); the policy version and the gate_v1 sha256 are checked.
Large artifacts (passages, reports) stay in the state only as needed; no telemetry, no secrets. Export is idempotent (one record per event ID).
Difference to the S4 harness: S4 generated a report even when tau flagged (to measure); here a tau flag goes to manual review without a call. Tau never flagged in the 60 S4 reports.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from gridsense.config import TRACK
from gridsense.s4 import checks as CK
from gridsense.s4 import generate as GEN
from gridsense.s4.decision import assemble, final_evidence_status, tau_flag

POLICY_VERSION = "gate_v1"
GATE_SHA = (TRACK / "configs" / "gate_v1.sha256").read_text().split()[0]
K_GRID = (0, 1, 2, 5, 10)
TRANSIENT = (ConnectionError, TimeoutError)
MAX_RETRIES = 2
# Checks that must pass before a human sees the report as a draft for approval (gate roles of the registry plus the safety checks).
BLOCKING = ("schema", "field_consistency", "neighbour_counts", "citation_validity", "no_confirmation", "no_control_command", "no_approval_claim", "no_injection_echo", "evidence_consistency")


class State(TypedDict, total=False):
    """§ 12.2 fields. `approval_status` is set only by Python nodes."""
    event_id: str
    input_reference: dict
    prediction: dict
    verification_result: dict
    report_mode: str
    retrieval_query: str
    retrieved_cases: list
    evidence_score: dict
    retrieved_chunks: list
    reranked_chunks: list
    evidence_sufficiency: dict
    draft_report: dict
    report_raw: dict
    citation_validation: dict
    revisions: int
    approval_status: str
    manual_review_reason: str
    audit: list
    trace_id: str
    errors: list
    model_version: str
    policy_version: str
    index_version: str


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(deps: dict) -> Any:
    """deps: event_lookup(event_id) -> golden-event dict; retrieve(item) -> {passages, dense_top1}; tau; generator (client with .complete);
    index_version; export_store (dict-like, idempotent export); current_policy() -> str (policy at resume time)."""
    def validate_event(s: State) -> dict:
        try:
            ev = deps["event_lookup"](s["event_id"])
        except KeyError:
            return {"manual_review_reason": "INVALID_INPUT: unknown event", "errors": [*s.get("errors", []), "unknown event"]}
        o = ev.get("event_object") or {}
        need = ("k", "predicted_bus", "gate_outcome", "reason_codes", "knn", "calibrated_probability", "margin", "dropped_pmu_buses")
        miss = [k for k in need if k not in o]
        if miss or o.get("k") not in K_GRID:
            return {"manual_review_reason": f"INVALID_INPUT: missing {miss}" if miss else f"INVALID_INPUT: k = {o.get('k')} not in {K_GRID}", "input_reference": {"window_id": ev.get("window_id")}}
        return {"input_reference": {"window_id": ev.get("window_id"), "event_object_hash": hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]},
                "policy_version": POLICY_VERSION, "index_version": deps.get("index_version", ""), "model_version": "dropout-aware RGSAGE-Max seed 42", "revisions": 0,
                "approval_status": "pending", "audit": [*s.get("audit", []), {"utc": _now(), "event": "validated"}]}

    def run_or_load_prediction(s: State) -> dict:
        o = deps["event_lookup"](s["event_id"])["event_object"]
        return {"prediction": {"predicted_bus": o["predicted_bus"], "calibrated_probability": o["calibrated_probability"], "margin": o["margin"], "k": o["k"], "source": "loaded (Layer 0, frozen)"}}

    def verify_prediction(s: State) -> dict:
        o = deps["event_lookup"](s["event_id"])["event_object"]
        res = {"outcome": o["gate_outcome"], "reason_codes": o["reason_codes"], "policy_version": POLICY_VERSION, "gate_sha256": GATE_SHA}
        mode = {"CONFIDENT": "single_location", "LOW_CONFIDENCE": "candidate_set"}.get(o["gate_outcome"])
        out = {"verification_result": res, "report_mode": mode or "none"}
        if mode is None:
            out["manual_review_reason"] = f"{o['gate_outcome']}: {', '.join(o['reason_codes'])} (no location and no candidate set are shown)"
        return out

    def build_retrieval_query(s: State) -> dict:
        return {"retrieval_query": GEN.retrieval_query(deps["event_lookup"](s["event_id"]))}

    def retrieve_cases(s: State) -> dict:
        nb = deps["event_lookup"](s["event_id"])["event_object"]["knn"]["neighbours"]
        return {"retrieved_cases": [{"bus": n["bus"], "similarity": n["similarity"], "store_prediction_correct": n["store_prediction_correct"]} for n in nb]}

    def compute_evidence_score(s: State) -> dict:
        """Python only. Information for the reader; the gate already used kNN agreement. Never a confidence."""
        pred, cases = s["prediction"]["predicted_bus"], s["retrieved_cases"]
        agree = sum(c["bus"] == pred for c in cases)
        return {"evidence_score": {"neighbours_with_predicted_bus": agree, "neighbours": len(cases), "note": "information only; not a confidence"}}

    def retrieve_evidence(s: State) -> dict:
        err = []
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = deps["retrieve"](deps["event_lookup"](s["event_id"]))
                return {"retrieved_chunks": r["passages"], "evidence_sufficiency": {"dense_top1": r["dense_top1"]}, "errors": [*s.get("errors", []), *err]}
            except TRANSIENT as e:                                  # transient: retry a limited number of times
                err.append(f"retrieval attempt {attempt + 1}: {type(e).__name__}")
        return {"retrieved_chunks": [], "evidence_sufficiency": {"dense_top1": None}, "errors": [*s.get("errors", []), *err, "retrieval failed after retries"]}

    def rerank_if_enabled(s: State) -> dict:
        return {"reranked_chunks": list(s["retrieved_chunks"])}           # frozen configuration v2: reranker off

    def check_evidence_sufficiency(s: State) -> dict:
        flag = tau_flag(s["evidence_sufficiency"].get("dense_top1"), deps["tau"])
        out = {"evidence_sufficiency": s["evidence_sufficiency"] | {"tau": deps["tau"], "tau_flag": flag}}
        if flag:
            out["manual_review_reason"] = "INSUFFICIENT_EVIDENCE: retrieval score below tau (no report drafted)"
        return out

    def draft_report(s: State) -> dict:
        item = deps["event_lookup"](s["event_id"])
        msgs = GEN.build_messages(item, deps.get("anon_index", lambda e: 0)(s["event_id"]), s["reranked_chunks"])
        if s.get("revisions", 0) > 0:                                       # revise once: the failed checks are named; nothing else changes
            fails = [f"{k}: {v['detail']}" for k, v in (s.get("citation_validation") or {}).items() if not v["pass"] and not v.get("informational")]
            msgs = msgs + [{"role": "user", "content": "The report failed these checks. Write the whole report again and correct them:\n" + "\n".join(fails)}]
        reply = deps["generator"].complete(msgs, purpose="generator", json_mode=True, max_tokens=4000, salt=f"agent{s.get('revisions', 0)}")
        rep, err = GEN.parse_report(reply.text)
        if err:
            return {"draft_report": {}, "report_raw": {"error": err}, "errors": [*s.get("errors", []), err]}
        return {"report_raw": rep}

    def validate_report_and_citations(s: State) -> dict:
        raw = s.get("report_raw") or {}
        if "error" in raw:
            chk = {"schema": {"pass": False, "detail": raw["error"]}}
            out = {"citation_validation": chk, "draft_report": {}}
            if s.get("revisions", 0) > 0:
                out["manual_review_reason"] = "REPORT_CHECKS_FAILED after one revision: " + raw["error"]
            return out
        item = deps["event_lookup"](s["event_id"])
        final = final_evidence_status(s["evidence_sufficiency"]["tau_flag"], raw.get("evidence_status"))
        rep, info = assemble(raw, final["final"])
        view = GEN.event_view(item, deps.get("anon_index", lambda e: 0)(s["event_id"]))
        chk = CK.run_checks(rep, view, item["event_object"]["knn"]["neighbours"], s["reranked_chunks"], final["final"])
        out = {"draft_report": rep, "citation_validation": chk, "evidence_sufficiency": s["evidence_sufficiency"] | {"final": final["final"], "by": final["by"]} | info}
        passed = all(chk.get(k, {"pass": False})["pass"] for k in BLOCKING)
        if passed and final["final"] == "insufficient":
            out["manual_review_reason"] = "INSUFFICIENT_EVIDENCE: the generator found the passages insufficient (report kept, no procedure)"
        if not passed and s.get("revisions", 0) > 0:
            out["manual_review_reason"] = "REPORT_CHECKS_FAILED after one revision: " + ", ".join(k for k in BLOCKING if not chk.get(k, {"pass": False})["pass"])
        return out

    def route_after_validation(s: State) -> str:
        chk = s.get("citation_validation") or {}
        ok = all(chk.get(k, {"pass": False})["pass"] for k in BLOCKING)
        if not ok:
            return "revise" if s.get("revisions", 0) == 0 else "manual_review"
        return "manual_review" if s.get("manual_review_reason") else "human_approval"

    def revise(s: State) -> dict:
        return {"revisions": s.get("revisions", 0) + 1, "audit": [*s.get("audit", []), {"utc": _now(), "event": "revise once"}]}

    def human_approval(s: State) -> dict:
        decision = interrupt({"event_id": s["event_id"], "report": s["draft_report"], "verification": s["verification_result"], "citations": s["citation_validation"],
                              "status": "draft", "options": ["approve", "reject", "clarify"]})
        if not isinstance(decision, dict) or decision.get("decision") not in ("approve", "reject", "clarify") or not decision.get("reviewer"):
            return {"approval_status": "manual_review", "manual_review_reason": "invalid human decision (needs decision and reviewer)", "audit": [*s.get("audit", []), {"utc": _now(), "event": "invalid decision"}]}
        cur = deps.get("current_policy", lambda: POLICY_VERSION)()
        if cur != s.get("policy_version"):                                 # § 12.3: never resume on an obsolete policy
            return {"approval_status": "manual_review", "manual_review_reason": f"policy changed while paused: {s.get('policy_version')} -> {cur}",
                    "audit": [*s.get("audit", []), {"utc": _now(), "event": "resume refused: policy version"}]}
        rec = {"utc": _now(), "event": f"human {decision['decision']}", "reviewer": decision["reviewer"], "note": decision.get("note", "")}
        status = {"approve": "approved", "reject": "rejected", "clarify": "manual_review"}[decision["decision"]]
        out = {"approval_status": status, "audit": [*s.get("audit", []), rec]}
        if status == "manual_review":
            out["manual_review_reason"] = "reviewer asked for clarification: " + decision.get("note", "")
        return out

    def export_approved_report(s: State) -> dict:
        store = deps["export_store"]
        if s["event_id"] not in store:                                     # idempotent: one record per event ID
            store[s["event_id"]] = {"report": s["draft_report"], "approval": s["audit"][-1], "policy_version": s["policy_version"], "index_version": s.get("index_version")}
        return {"audit": [*s.get("audit", []), {"utc": _now(), "event": "exported"}]}

    def closed_with_feedback(s: State) -> dict:
        return {"audit": [*s.get("audit", []), {"utc": _now(), "event": "closed with feedback"}]}

    def manual_review(s: State) -> dict:
        return {"approval_status": "manual_review", "audit": [*s.get("audit", []), {"utc": _now(), "event": "manual review", "reason": s.get("manual_review_reason", "")}]}

    g = StateGraph(State)
    for name, fn in [("validate_event", validate_event), ("run_or_load_prediction", run_or_load_prediction), ("verify_prediction", verify_prediction),
                     ("build_retrieval_query", build_retrieval_query), ("retrieve_cases", retrieve_cases), ("compute_evidence_score", compute_evidence_score),
                     ("retrieve_evidence", retrieve_evidence), ("rerank_if_enabled", rerank_if_enabled), ("check_evidence_sufficiency", check_evidence_sufficiency),
                     ("draft_report", draft_report), ("validate_report_and_citations", validate_report_and_citations), ("revise", revise), ("human_approval", human_approval),
                     ("export_approved_report", export_approved_report), ("closed_with_feedback", closed_with_feedback), ("manual_review", manual_review)]:
        g.add_node(name, fn)
    g.add_edge(START, "validate_event")
    g.add_conditional_edges("validate_event", lambda s: "manual_review" if s.get("manual_review_reason") else "run_or_load_prediction")
    g.add_edge("run_or_load_prediction", "verify_prediction")
    g.add_conditional_edges("verify_prediction", lambda s: "manual_review" if s.get("manual_review_reason") else "build_retrieval_query")
    for a, b in [("build_retrieval_query", "retrieve_cases"), ("retrieve_cases", "compute_evidence_score"), ("compute_evidence_score", "retrieve_evidence"),
                 ("retrieve_evidence", "rerank_if_enabled"), ("rerank_if_enabled", "check_evidence_sufficiency")]:
        g.add_edge(a, b)
    g.add_conditional_edges("check_evidence_sufficiency", lambda s: "manual_review" if s.get("manual_review_reason") else "draft_report")
    g.add_edge("draft_report", "validate_report_and_citations")
    g.add_conditional_edges("validate_report_and_citations", route_after_validation, {"revise": "revise", "manual_review": "manual_review", "human_approval": "human_approval"})
    g.add_edge("revise", "draft_report")
    g.add_conditional_edges("human_approval", lambda s: {"approved": "export_approved_report", "rejected": "closed_with_feedback"}.get(s["approval_status"], "manual_review"))
    for n in ("export_approved_report", "closed_with_feedback", "manual_review"):
        g.add_edge(n, END)
    return g.compile(checkpointer=deps.get("checkpointer") or InMemorySaver())


def start(app, event_id: str, thread_id: str | None = None) -> tuple[dict, dict]:
    """Run until the end or the human-approval pause. (config, state values)."""
    cfg = {"configurable": {"thread_id": thread_id or f"event-{event_id}"}}
    app.invoke({"event_id": event_id, "trace_id": thread_id or f"event-{event_id}", "errors": [], "audit": []}, cfg)
    return cfg, app.get_state(cfg).values


def decide(app, cfg: dict, decision: dict) -> dict:
    """The explicit human action: resume the paused graph with {decision, reviewer, note}. Returns the final state values."""
    app.invoke(Command(resume=decision), cfg)
    return app.get_state(cfg).values


def paused(app, cfg: dict) -> bool:
    return bool(app.get_state(cfg).next)


