"""Report generation for S4: event -> retrieval query -> top-3 passages (frozen configuration v2) -> generator prompt -> parsed report.

What the generator sees: the structured event fields, the 10 case neighbours (anonymised case numbers, their fault bus and whether the store's prediction was right) and the retrieved passages.
What it never sees: the truth (true bus, fault type, resistance), the run ID and the window ID (they contain the truth), the model and gate hashes. `event_view` builds that and a test checks it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from gridsense.config import TRACK

PROMPTS = TRACK / "prompts"
REPORT_KEYS = ("event_id", "predicted_location", "prediction_status", "gate_outcome", "gate_reason_codes", "candidate_set", "sensor_coverage_summary", "case_evidence_summary",
               "document_evidence", "limitations", "recommended_operator_review", "evidence_status", "evidence_status_reason")
SCHEMA = {"event_id": "str (copy the event_id of the EVENT block)", "predicted_location": "int (the predicted bus, or null if the top class is no fault)", "prediction_status": "candidate_not_confirmed",
          "gate_outcome": "CONFIDENT|LOW_CONFIDENCE|REJECT", "gate_reason_codes": ["str"], "candidate_set": ["int"], "sensor_coverage_summary": "str",
          "case_evidence_summary": "str (state the agreement as N of 10 and the neighbour buses)", "document_evidence": [{"claim": "str", "document_id": "str", "section": "str", "excerpt": "str (copied from the passage)"}],
          "limitations": ["str"], "recommended_operator_review": ["str"], "evidence_status": "sufficient|insufficient", "evidence_status_reason": "str"}


def split_prompt(name: str) -> tuple[str, str, str]:
    """(system, user template, full file text) of prompts/<name>.md."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    body = text.split("\n---\n", 1)[1]
    sys_, user = body.split("## User", 1)
    return sys_.replace("## System", "").strip(), user.strip(), text


def anon_id(index: int) -> str:
    return f"event-{index:03d}"


def event_view(item: dict, index: int) -> dict:
    """The event object without the truth and without identifiers that contain it."""
    o = item["event_object"]
    return {"event_id": anon_id(index), "report_window_post_fault_samples": o["report_window_index"] - 20, "k": o["k"], "dark_uPMU_buses": o["dropped_pmu_buses"],
            "predicted_bus": o["predicted_bus"], "vote_share_top1": o["top1_vote_share"], "calibrated_probability": o["calibrated_probability"], "margin": o["margin"],
            "gate": {"outcome": o["gate_outcome"], "reason_codes": o["reason_codes"], "policy_version": "gate_v1"}, "candidate_set": o["candidate_set_buses"],
            "sensor_coverage": {"available": 25 - o["k"], "total": 25}, "nearest_3_uPMU_buses_of_predicted_bus": o["nearest_pmu_buses_of_predicted_bus"],
            "knn_agreement_of_10": o["knn"]["agreement"]}


def neighbours_table(item: dict) -> str:
    nb = item["event_object"]["knn"]["neighbours"]
    return "case | true bus | store prediction correct | similarity\n" + "\n".join(
        f"c{i + 1:02d} | {n['bus']} | {'yes' if n['store_prediction_correct'] else 'no'} | {n['similarity']:.4f}" for i, n in enumerate(nb))


def retrieval_query(item: dict) -> str:
    """One fixed template, built from event features only (no truth)."""
    o = item["event_object"]
    names = {"VERIFICATION_PASSED": "verification passed", "LOW_CALIBRATED_CONFIDENCE": "low calibrated confidence", "LOW_MARGIN": "low margin", "LOW_KNN_AGREEMENT": "low kNN agreement",
             "UNSUPPORTED_DROPOUT_PATTERN": "unsupported dropout pattern", "INVALID_INPUT": "invalid input"}
    why = ", ".join(names.get(r, r) for r in o["reason_codes"])
    q = f"What does the gate outcome {o['gate_outcome']} ({why}) mean for a fault-location prediction?"
    if o["k"]:
        q += f" The prediction was made with {o['k']} of 25 μPMUs dark."
    if o["candidate_set_buses"] is not None:
        q += " A candidate set of buses is shown."
    return q


def retrieval_queries(item: dict, mode: str = "single") -> list[tuple[str, int]]:
    """[(query, top_k)]. `single` = the plan (one query, top 3). `multi` = gate meaning, dropout (only if k > 0) and neighbours as precedents, top 2 each (at most 6 passages after de-duplication)."""
    if mode == "single":
        return [(retrieval_query(item), 3)]
    qs = [(retrieval_query(item), 2)]
    if item["event_object"]["k"]:
        qs.append(("How does the system handle sensors that are dark or missing, and when is a dark sensor near the predicted bus a concern?", 2))
    qs.append(("How should the most similar simulated cases (nearest neighbours) be used when a prediction is explained?", 2))
    return qs


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['document_id']} › {' › '.join(p['section_path'][-1:])}]\n{p['text']}" for p in passages) or "(no passages)"


def build_messages(item: dict, index: int, passages: list[dict]) -> list[dict]:
    sys_, user, _ = split_prompt("generator_v0")
    body = user.format(event_json=json.dumps(event_view(item, index), indent=1), neighbours_table=neighbours_table(item), passages=format_passages(passages),
                       schema_json=json.dumps(SCHEMA, indent=1))
    return [{"role": "system", "content": sys_}, {"role": "user", "content": body}]


def parse_report(text: str) -> tuple[dict | None, str | None]:
    """(report, error). A reply that is not a JSON object is a failure with a cause, never an empty report."""
    try:
        d = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except Exception:
            d = None
    if not isinstance(d, dict):
        return None, "generator reply is not a JSON object"
    return d, None


def report_text(r: dict) -> str:
    """The report as plain text for the judge (the judge reads text, not JSON)."""
    L = [f"Event {r.get('event_id')}. Predicted location: {r.get('predicted_location')}. Status: {r.get('prediction_status')}.",
         f"Gate outcome: {r.get('gate_outcome')} (reason codes: {', '.join(map(str, r.get('gate_reason_codes') or []))}). Candidate set: {', '.join(map(str, r.get('candidate_set') or [])) or 'none'}.",
         f"Sensor coverage: {r.get('sensor_coverage_summary')}", f"Case evidence: {r.get('case_evidence_summary')}"]
    for d in r.get("document_evidence") or []:
        if isinstance(d, dict):
            L.append(f"Document evidence: {d.get('claim')} ({d.get('document_id')}, \"{d.get('section')}\": \"{d.get('excerpt')}\")")
    L += [f"Limitation: {x}" for x in r.get("limitations") or []] + [f"Recommended operator review: {x}" for x in r.get("recommended_operator_review") or []]
    if r.get("human_review_required"):
        L.append(f"Human review (system-set): {r['human_review_required']}")
    L.append(f"Evidence status: {r.get('evidence_status')} ({r.get('evidence_status_reason')}). Approval status: pending_human_review.")
    return "\n".join(map(str, L))
