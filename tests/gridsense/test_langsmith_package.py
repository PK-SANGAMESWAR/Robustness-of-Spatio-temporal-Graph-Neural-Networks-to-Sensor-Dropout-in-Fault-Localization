"""S8 offline package: reviewed-only, exclusions, redaction, and the refusal to upload without the author's approval. A fake client; nothing is sent."""
from __future__ import annotations

import pytest

from gridsense.eval import langsmith_package as LP


def _row(qid, status="reviewed", doc="note_24_gate_outcomes", q="What is a gate?"):
    return {"query_id": qid, "query": q, "query_type": "paraphrase", "answerable": True, "reference_answer": "A gate decides.", "split": "dev", "status": status, "golden_version": 4,
            "golden_contexts": [{"document_id": doc, "section_path": ["S"], "span": "SECRET SPAN TEXT", "span_hash": "h1", "grade": 3}]}


class Fake:
    def __init__(self):
        self.calls = []

    def create_dataset(self, **k):
        self.calls.append(("dataset", k))
        return type("D", (), {"id": "ds-1"})()

    def create_examples(self, **k):
        self.calls.append(("examples", k))


def test_S8_only_reviewed_items_without_sensitive_sources_are_in_the_payload_and_no_span_text_goes_up():
    rows = [_row("R1"), _row("R2", status="draft"), _row("R3", doc="paper1_rev1170"), _row("R4", doc="results_md"), _row("R5", doc="sop_illustrative_gridsense_review")]
    p = LP.build_payload(rows)
    assert [e["metadata"]["query_id"] for e in p["examples"]] == ["R1"]
    assert p["excluded"]["draft"] == 1 and [e["query_id"] for e in p["excluded"]["sensitive_source"]] == ["R3", "R4", "R5"]
    assert "SECRET SPAN TEXT" not in str(p["examples"]) and p["examples"][0]["outputs"]["golden_context_refs"][0]["span_hash"] == "h1"


@pytest.mark.parametrize("bad", ["see /home/ubuntu/x", "mail me@example.com", "sk-abcdefghijk", "OPENAI_API_KEY"])
def test_S8_redaction_finds_paths_keys_addresses_and_env_names(bad):
    p = LP.build_payload([_row("R1", q=bad)])
    assert LP.redaction_hits(p)


def test_S8_upload_is_refused_without_approval_and_with_a_wrong_hash(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(LP, "load_rows", lambda: [_row("R1")])
    monkeypatch.setattr(LP, "load_config", lambda: {"langsmith": {"enabled": True, "upload_approved": False, "project": "p", "endpoint": "e"}})
    with pytest.raises(PermissionError, match="needs langsmith.enabled"):
        LP.upload("x", client=fake)                                                              # not approved (the real config is approved since 2026-11-20: this test sets its own)
    assert fake.calls == []
    approved = {"langsmith": {"enabled": True, "upload_approved": True, "project": "p", "endpoint": "e"}}
    monkeypatch.setattr(LP, "load_config", lambda: approved)
    with pytest.raises(PermissionError, match="hash"):
        LP.upload("wrong", client=fake)
    assert fake.calls == []
    out = LP.upload(LP.payload_hash(LP.build_payload([_row("R1")])), client=fake)
    assert out["examples"] == 1 and [c[0] for c in fake.calls] == ["dataset", "examples"]


def test_S8_a_payload_with_a_redaction_hit_is_never_uploaded(monkeypatch):
    fake = Fake()
    rows = [_row("R1", q="my key is sk-abcdefghijk")]
    monkeypatch.setattr(LP, "load_rows", lambda: rows)
    monkeypatch.setattr(LP, "load_config", lambda: {"langsmith": {"enabled": True, "upload_approved": True, "project": "p", "endpoint": "e"}})
    with pytest.raises(ValueError, match="redaction"):
        LP.upload(LP.payload_hash(LP.build_payload(rows)), client=fake)
    assert fake.calls == []


def test_S8_experiment_outputs_never_carry_text_or_ids_of_sensitive_sources():
    from gridsense.eval import langsmith_experiment as LX
    ok = LX.public_ref({"document_id": "note_01", "section_path": ["A", "B"], "chunk_id": "c1", "score": 0.123456, "text": "PASSAGE TEXT"})
    assert ok == {"document_id": "note_01", "section": "B", "chunk_id": "c1", "score": 0.1235} and "PASSAGE TEXT" not in str(ok)
    for doc in ("paper1_rev1170", "results_md", "sop_illustrative_gridsense_review"):
        w = LX.public_ref({"document_id": doc, "section_path": ["Secret heading"], "chunk_id": f"{doc}:1-2:abc", "score": 0.5, "text": "x"})
        assert "Secret" not in str(w) and doc not in str(w) and w["score"] == 0.5
