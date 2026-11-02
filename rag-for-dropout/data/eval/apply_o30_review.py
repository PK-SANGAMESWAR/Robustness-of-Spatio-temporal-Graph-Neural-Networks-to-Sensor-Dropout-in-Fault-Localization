"""One-off: apply the author's O30 review (2026-11-02) to golden_retrieval.jsonl (v0 -> v1). Run from the repo root:
    PYTHONPATH=src:. .venv/bin/python rag-for-dropout/data/eval/apply_o30_review.py

Author instructions: accept all queries except R-0005, R-0036, R-0034, R-0044, R-0047, R-0048; add R-0055 and R-0056; never change or remove a
query because of its rank. Every span is re-checked against the parsed corpus (must occur exactly once). The version-0 file is in Git (commit 692ef94).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from gridsense.eval.golden import locate, norm, span_hash, validate
from gridsense.ingest.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "rag-for-dropout/data/eval/golden_retrieval.jsonl"
TODAY, AGENT = "2026-11-02", "agent:claude-sonnet-5-5"
SIX_BUSES = ("13", "18", "25", "35", "47", "97")


def ctx(doc, section_path, span, grade):
    return {"document_id": doc, "section_path": section_path, "span": span, "span_hash": span_hash(span), "grade": grade,
            "recheck": {"result": "found_once", "note": ""}}


def main() -> None:
    docs = load_corpus()
    by = {d.document_id: d for d in docs}
    rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 54 and all(r["status"] == "draft" and r.get("golden_version") == 0 for r in rows), "expects the v0 file"
    byid = {r["query_id"]: r for r in rows}
    changes: dict[str, list[str]] = {}

    # R-0005: remove the results_md span (a bare table row: the column labels 0.1 / 1 / 10 ohm are in the header row, not in the span)
    r = byid["R-0005"]
    row_span = next(c for c in r["golden_contexts"] if c["document_id"] == "results_md")
    assert "0.1 Ω" not in row_span["span"] and "10 Ω" not in row_span["span"], "the span would show the resistance labels"
    r["golden_contexts"] = [c for c in r["golden_contexts"] if c["document_id"] != "results_md"]
    changes["R-0005"] = ["removed the results_md golden context: the span is a bare table row without the column labels (0.1 / 1 / 10 ohm), so it does not show accuracy by fault resistance"]

    # R-0034: the span (a table row) stays; the reference answer was the row itself, now a sentence (bus 26 has 2 phases: pmu_phase_metadata.yaml, node 5)
    r = byid["R-0034"]
    assert "| 5 | 26 | 2 | - | 25 |" in r["golden_contexts"][0]["span"]
    r["reference_answer"] = "Two phases: the μPMU at bus 26 (graph node 5) measures two phases."
    changes["R-0034"] = ["edit: reference_answer was the bare table row, now a sentence; span unchanged (the row is the only text that gives the phase count of bus 26)"]

    # R-0036: both spans already contain the six bus IDs
    r = byid["R-0036"]
    for c in r["golden_contexts"]:
        assert all(b in c["span"] for b in SIX_BUSES), c["span"]
    changes["R-0036"] = ["accepted unchanged: both golden spans (feeder_overview, note_08) contain all six bus IDs 13, 18, 25, 35, 47, 97"]

    # R-0044: the reference answer still held the '(ref)' parser artifact of session B
    r = byid["R-0044"]
    r["reference_answer"] = next(c for c in r["golden_contexts"] if c["document_id"] == "paper1_rev1170")["span"] + "."
    changes["R-0044"] = ["edit: reference_answer lost the '(ref)' artifact (now equal to the re-anchored paper span); spans unchanged"]

    # R-0047, R-0048: unanswerable; the corpus must not answer them
    full = " ".join(d.text for d in docs).lower()
    assert "breaker" not in full and "recloser" not in full
    changes["R-0047"] = ["accepted unchanged: 'breaker' does not occur anywhere in the corpus"]
    changes["R-0048"] = ["accepted unchanged: 'recloser' does not occur in the corpus; the nearest text (SOP-ILL-7.1) says the example has no switching or isolation step"]

    # new items (author, O30): query, split, document and answer from the author; the golden span is chosen by the agent
    fb = by["feeder_fault_buses"]
    row51 = "| 51 | bus 47 (0.229 km, 3) | bus 45 (0.366 km, 5) | bus 35 (0.518 km, 7) |"
    assert len(locate(fb, row51)) == 1
    tax = by["feeder_fault_taxonomy"]
    sent = "The letters are the faulted phases; a trailing G means the fault includes ground (ABG, ABCG). ABC is a three-phase fault without ground."
    bullet = "three-phase: ABC, ABCG"
    assert len(locate(tax, sent)) == 1 and len(locate(tax, bullet)) == 1
    new = [
        {"query_id": "R-0055", "query": "Which μPMU is nearest to bus 51 on the operated feeder?", "query_type": "exact_identifier",
         "golden_contexts": [ctx("feeder_fault_buses", list(fb.section_at(locate(fb, row51)[0][0])), row51, 3)],
         "reference_answer": "Bus 47, at 0.229 km (3 hops) from fault bus 51 on the operated feeder.", "answerable": True, "critical_terms": ["bus 47"], "split": "test"},
        {"query_id": "R-0056", "query": "What does the fault type code ABCG mean?", "query_type": "exact_identifier",
         "golden_contexts": [ctx("feeder_fault_taxonomy", list(tax.section_at(locate(tax, sent)[0][0])), sent, 3),
                             ctx("feeder_fault_taxonomy", list(tax.section_at(locate(tax, bullet)[0][0])), bullet, 2)],
         "reference_answer": "ABCG is a three-phase fault on phases A, B and C that includes ground (a trailing G means the fault includes ground).",
         "answerable": True, "critical_terms": ["ABCG"], "split": "dev"},
    ]
    for n in new:
        n.update({"source": "author (O30 review 2026-11-02): query, split, document and answer; golden span chosen by " + AGENT, "created": TODAY,
                  "created_by": "author (query, split, document, answer) + " + AGENT + " (span)", "converted_by": None})
        changes[n["query_id"]] = ["added by the author in the O30 review; the golden span was chosen by the agent from the generated document"]
    rows += new

    for r in rows:
        r["status"], r["reviewed"], r["golden_version"] = "reviewed", TODAY, 1
        r["review"] = {"by": "author", "date": TODAY, "round": "O30", "verdict": "accepted" if r["query_id"] not in
                       ("R-0005", "R-0034", "R-0044") else "accepted with edit", "changes": changes.get(r["query_id"], [])}
    problems = validate(rows, docs)
    assert not problems, problems
    assert sum(r["split"] == "dev" and r["answerable"] for r in rows) == 25 and sum(r["split"] == "test" and r["answerable"] for r in rows) == 23
    PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"items": len(rows), "reviewed": sum(r["status"] == "reviewed" for r in rows), "version": 1,
                      "dev_answerable": 25, "dev_unanswerable": sum(r["split"] == "dev" and not r["answerable"] for r in rows),
                      "test_answerable": 23, "test_unanswerable": sum(r["split"] == "test" and not r["answerable"] for r in rows),
                      "edited": [q for q, c in changes.items() if q in ("R-0005", "R-0034", "R-0044")]}))


if __name__ == "__main__":
    main()
