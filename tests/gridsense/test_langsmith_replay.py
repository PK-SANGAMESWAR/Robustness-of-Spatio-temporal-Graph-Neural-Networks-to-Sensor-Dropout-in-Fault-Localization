"""S8 tracing by replay: the scrub withholds sensitive sources everywhere, the replay generator costs nothing and never calls an API. No network."""
from __future__ import annotations

import json

from gridsense.eval import langsmith_replay as LR


def test_S8_scrub_withholds_sensitive_documents_in_nested_state_and_keeps_the_rest():
    state = {"retrieved_chunks": [{"document_id": "note_24_gate_outcomes", "section_path": ["S"], "text": "open text", "score": 0.5, "chunk_id": "c1"},
                                  {"document_id": "paper1_rev1170", "section_path": ["Secret heading"], "text": "MANUSCRIPT TEXT", "score": 0.4, "chunk_id": "paper1_rev1170:1-2:x"}],
             "draft_report": {"document_evidence": [{"claim": "c", "document_id": "results_md", "section": "Table 3", "excerpt": "0.839"}, {"claim": "ok", "document_id": "note_25", "section": "A", "excerpt": "e"}],
                              "limitations": ["see paper1_rev1170 section 4"]}, "path": "/home/ubuntu/x", "n": 3}
    out = LR.scrub(state)
    text = json.dumps(out)
    for bad in ("MANUSCRIPT TEXT", "Secret heading", "paper1_rev1170", "results_md", "Table 3", "/home/ubuntu"):
        assert bad not in text, bad
    assert out["retrieved_chunks"][0]["text"] == "open text" and out["retrieved_chunks"][1]["score"] == 0.4 and out["draft_report"]["document_evidence"][1]["excerpt"] == "e" and out["n"] == 3
    assert out["draft_report"]["limitations"] == [LR.WITHHELD]


def test_S8_replay_generator_returns_the_stored_report_for_the_event_and_costs_nothing():
    gen = LR.ReplayGenerator({"event-007": {"predicted_location": 55, "evidence_status": "sufficient"}})
    r = gen.complete([{"role": "user", "content": 'EVENT\n{"event_id": "event-007", "k": 0}'}], purpose="generator")
    assert json.loads(r.text)["predicted_location"] == 55 and r.usd == 0.0 and r.input_tokens == 0


def test_S8_replay_is_refused_without_the_authors_approval(monkeypatch):
    import pytest
    monkeypatch.setattr(LR, "load_config", lambda: {"langsmith": {"enabled": False, "upload_approved": False, "project": None, "endpoint": "e"}})
    with pytest.raises(PermissionError):
        LR.run()
