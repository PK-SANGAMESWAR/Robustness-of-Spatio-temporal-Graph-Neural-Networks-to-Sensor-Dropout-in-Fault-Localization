"""Publishing guard: the sanitizer withholds sensitive documents, the scan finds manuscript text and keys, the commit refuses a leak. Synthetic data; the real manuscript is never read here."""
from __future__ import annotations

import json

import pytest

from gridsense import publish as P


def test_P_jsonl_records_of_the_unpublished_source_lose_their_text_and_others_keep_it():
    lines = [{"document_id": "paper1_rev1170", "section_path": ["Secret"], "text": "MANUSCRIPT SENTENCE", "span_hash": "h", "score": 0.4},
             {"document_id": "note_01", "section_path": ["Open"], "text": "open text", "score": 0.5}]
    data = "\n".join(json.dumps(x) for x in lines).encode() + b"\n"
    out = P.sanitize("rag-for-dropout/eval/items/x.jsonl", data).decode()
    assert "MANUSCRIPT SENTENCE" not in out and "Secret" not in out and "open text" in out and '"span_hash": "h"' in out and out.endswith("\n")


def test_P_a_golden_item_citing_the_manuscript_also_withholds_the_reference_answer():
    item = {"query_id": "R-1", "query": "q", "reference_answer": "PARAPHRASE OF THE MANUSCRIPT", "golden_contexts": [{"document_id": "paper1_rev1170", "span": "S", "span_hash": "h"}]}
    out = json.loads(P.sanitize("rag-for-dropout/data/eval/g.jsonl", json.dumps(item).encode()))
    assert out["reference_answer"] == P.WITHHELD and out["golden_contexts"][0]["span"] == P.WITHHELD and out["golden_contexts"][0]["span_hash"] == "h" and out["query"] == "q"
    clean = {"query_id": "R-2", "reference_answer": "fine", "golden_contexts": [{"document_id": "note_01", "span": "S"}]}
    assert json.loads(P.sanitize("rag-for-dropout/data/eval/g.jsonl", json.dumps(clean).encode()))["reference_answer"] == "fine"


def test_P_csv_rows_of_the_unpublished_source_are_withheld_and_json_without_change_keeps_its_bytes():
    csv_in = b"query_id,document_id,passage_text,label\nR1,paper1_rev1170,SECRET TEXT,\nR2,note_01,open text,\n"
    out = P.sanitize("rag-for-dropout/reports/x.csv", csv_in).decode()
    assert "SECRET TEXT" not in out and "open text" in out and out.count(P.WITHHELD) == 1
    js = b'{"a":   1,"b": [2,3]}\n'
    assert P.sanitize("rag-for-dropout/reports/x.json", js) == js


def test_P_langsmith_org_id_is_replaced_in_text_and_regenerable_files_are_excluded():
    md = b"see https://smith.langchain.com/o/9cb1af9d-3073-4914-908c-22c79b394fc8/projects/p/abc"
    assert b"9cb1af9d" not in P.sanitize("rag-for-dropout/reports/s7/x.md", md)
    for path in ("rag-for-dropout/eval/doc_index/abc/chunks.jsonl", "rag-for-dropout/reports/findings/paper1_open_switch.md"):
        with pytest.raises(ValueError):
            P.sanitize(path, b"x")


def test_P_the_scan_finds_a_ten_word_run_of_the_manuscript_and_key_like_strings_but_not_the_synthetic_test_key():
    ms = "this sentence exists only in the unpublished revision of the paper and nowhere else".split()
    sh = {hash(" ".join(P._tokens(" ".join(ms))[i:i + P.SHINGLE])) for i in range(len(ms) - P.SHINGLE + 1)}
    hit = P.leak_scan_text("a.md", b"Intro. This sentence exists only in the unpublished revision of the paper and nowhere else. End.", sh)
    assert hit["shingle_hits"] >= 1 and P.leak_scan_text("a.md", b"a short unrelated text of a few words", sh)["shingle_hits"] == 0
    fake = "sk" + "-" + "abcdefghijklmnopqrstuvwxyz0123"                                   # built at run time: the source holds no key-shaped literal
    assert P.leak_scan_text("a.py", f'key = "{fake}"'.encode(), sh)["key_hits"] == 1
    assert P.leak_scan_text("a.py", b'fake = "sk-test-0000000000000000"', sh)["key_hits"] == 0
    assert P.leak_scan_text("a.bin", fake.encode(), sh)["key_hits"] == 0                                   # binary suffixes are not scanned


def test_P_a_non_utf8_or_binary_file_passes_through_unchanged():
    data = bytes(range(256))
    assert P.sanitize("rag-for-dropout/eval/x.npy", data) == data


def test_P_commit_refuses_a_leak_on_a_throwaway_branch_and_leaves_the_real_branch_alone(monkeypatch):
    """The guard runs on a temporary branch copy; a manuscript-like file must stop the commit before any ref moves."""
    from gridsense.config import ROOT
    real = P._git("rev-parse", "gridsense-eval").decode().strip()
    P._git("branch", "tmp-publish-test", "gridsense-eval")
    probe = ROOT / "rag-for-dropout" / "eval" / "_tmp_publish_probe.md"
    text = "this probe sentence pretends to be a revision only sentence of the manuscript and must stop the commit"
    probe.write_text(text, encoding="utf-8")
    msg = ROOT / "rag-for-dropout" / "eval" / "_tmp_msg.txt"
    msg.write_text("probe", encoding="utf-8")
    toks = P._tokens(text)
    monkeypatch.setattr(P, "manuscript_shingles", lambda *a, **k: {hash(" ".join(toks[i:i + P.SHINGLE])) for i in range(len(toks) - P.SHINGLE + 1)})
    try:
        tip_before = P._git("rev-parse", "tmp-publish-test").decode().strip()
        with pytest.raises(RuntimeError, match="leak scan"):
            P.commit("tmp-publish-test", str(msg), ["rag-for-dropout/eval/_tmp_publish_probe.md"])
        assert P._git("rev-parse", "tmp-publish-test").decode().strip() == tip_before and P._git("rev-parse", "gridsense-eval").decode().strip() == real
    finally:
        probe.unlink(missing_ok=True)
        msg.unlink(missing_ok=True)
        P._git("branch", "-D", "tmp-publish-test")
