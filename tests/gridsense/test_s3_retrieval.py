"""S3 (Layer 1) tests: parsers, chunkers, golden matching, retrieval metrics, index fingerprint, stores, BM25/RRF, cost ledger,
OpenAI embedder with a mock client, corpus integrity and the golden file. No model download, no API call (rag-eval.md § 10.1)."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gridsense.cases.store import StaleIndexError
from gridsense.cost import CostLimitError, Ledger, MissingPriceError
from gridsense.eval import golden as G
from gridsense.eval import retrieval_metrics as RM
from gridsense.ingest.chunking import STRATEGIES, chunk_document
from gridsense.ingest.parsers import Document, parse_latex, parse_markdown
from gridsense.retrieval.embeddings import Embedder, EmbedderSpec, prefixed
from gridsense.retrieval.lexical import rrf, tokenize
from gridsense.retrieval.retriever import DocIndex, RetrievalConfig, Retriever, index_fingerprint
from gridsense.retrieval.stores import make_store

MD = "# Title\n\nIntro text.\n\n## Part A\nAlpha beta gamma. Delta epsilon.\n\n## Part B\nThe fault at bus 57 is LG.\n"
TEX = (r"\documentclass{x}\begin{document}\begin{abstract}We study $\mu$PMU loss.\end{abstract}"
       r"\section{Intro}Text with \cite{K2026} and \rev{revised words}.% comment" "\n"
       r"\subsection{Setup}$k\sim\mathrm{Uniform}\{0,1,2,5\}$ per batch.\end{document}")


def words(s: str) -> int:
    return len(s.split())


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder: no model download."""
    def __init__(self, dim: int = 64, window: int = 256, prefix: str = ""):
        super().__init__(EmbedderSpec(model=f"fake-{dim}", backend="fake", query_prefix=prefix, max_seq_length=window))

    def count_tokens(self, text: str) -> int:
        return words(text)

    def encode(self, texts, kind):
        out = np.zeros((len(texts), 64), np.float32)
        for i, t in enumerate(prefixed(self.spec, texts, kind)):
            for w in tokenize(t):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def doc(text=MD, did="d1") -> Document:
    t, secs = parse_markdown(text)
    return Document(did, "Doc", "curated_note", "curated_for_research", "0", "draft", t, secs)


def test_parse_markdown_section_paths():
    t, secs = parse_markdown(MD)
    assert [s.path for s in secs] == [("Title",), ("Title", "Part A"), ("Title", "Part B")]
    assert t[secs[2].start:secs[2].end].startswith("## Part B")


def test_parse_latex_cleans_commands_and_keeps_sections():
    t, secs = parse_latex(TEX)
    assert [s.path for s in secs] == [("Abstract",), ("Intro",), ("Intro", "Setup")]
    assert "μPMU" in t and "[K2026]" in t and "revised words" in t and "comment" not in t
    assert "k∼Uniform{0,1,2,5}" in t and "\\" not in t


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunkers_keep_offsets_and_size(strategy):
    d, emb = doc(MD * 3), FakeEmbedder()
    ch = chunk_document(d, strategy, 8, words, embed_fn=lambda x: emb.encode(x, "passage"))
    assert ch, strategy
    for c in ch:
        assert c.text == d.text[c.start:c.end]
        if strategy != "parent_child":
            assert words(c.text) <= 8 or len(c.text.split()) == 1
    covered = "".join(sorted({c.text for c in ch}, key=d.text.find))
    assert G.norm("The fault at bus 57 is LG.") in G.norm(" ".join(c.text for c in ch))


def test_overlap_extends_chunks_left():
    d = doc("w " * 60)
    a, b = chunk_document(d, "recursive", 10, words), chunk_document(d, "recursive", 10, words, overlap=0.2)
    assert len(a) == len(b) and all(y.start <= x.start for x, y in zip(a, b)) and any(y.start < x.start for x, y in zip(a, b))


def test_golden_span_matches_across_chunk_sizes_not_by_id():
    d, span = doc(), "fault at bus 57"
    ctx = {"document_id": "d1", "span": span}
    for size in (6, 50):
        assert any(G.matches(c.document_id, c.text, ctx) for c in chunk_document(d, "recursive", size, words))
    assert not G.matches("other_doc", "The fault at bus 57 is LG.", ctx)
    assert G.span_hash("  Fault at\nBUS 57 ") == G.span_hash("fault at bus 57")
    assert len(G.locate(d, "FAULT  at bus 57")) == 1


def test_query_metrics_by_hand():
    item = {"golden_contexts": [{"document_id": "d", "span": "alpha", "grade": 3}, {"document_id": "d", "span": "beta", "grade": 1}],
            "critical_terms": ["k = 10"]}
    ps = [{"document_id": "d", "text": "nothing"}, {"document_id": "d", "text": "alpha here, k = 10"}, {"document_id": "d", "text": "beta"}]
    m = RM.query_metrics(ps, item, ks=(1, 3))
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["mrr@10"] == 0.5
    assert m["precision@3"] == pytest.approx(2 / 3) and m["context_coverage@3"] == 1.0
    dcg = 7 / np.log2(3) + 1 / np.log2(4)
    idcg = 7 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg@3"] == pytest.approx(dcg / idcg) and m["identifier_recall@3"] == 1.0 and m["identifier_recall@1"] == 0.0


def test_sufficiency_threshold_on_dev():
    r = RM.choose_sufficiency_threshold([0.9, 0.8, 0.3, 0.2], [True, True, False, False])
    assert r["tau"] == 0.8 and r["balanced_accuracy_dev"] == 1.0
    s = RM.sufficiency_metrics([0.9, 0.5, 0.85], [True, True, False], 0.8)
    assert s["unanswerable_flagged_insufficient"] == 0.0 and s["answerable_flagged_insufficient"] == 0.5


def test_index_fingerprint_changes_and_stale_index_refused(tmp_path):
    emb, docs = FakeEmbedder(), [doc()]
    a, b = RetrievalConfig(chunker="recursive", chunk_size=8), RetrievalConfig(chunker="recursive", chunk_size=16)
    assert index_fingerprint(a, emb, docs) != index_fingerprint(b, emb, docs)
    assert index_fingerprint(a, emb, docs) != index_fingerprint(a, FakeEmbedder(prefix="query: "), docs)
    ix = DocIndex.build(a, emb, docs)
    ix.save(tmp_path)
    assert DocIndex.load(tmp_path, a, emb, docs).fingerprint == ix.fingerprint
    with pytest.raises(StaleIndexError):
        DocIndex.load(tmp_path, b, emb, docs)
    with pytest.raises(ValueError, match="window"):
        DocIndex.build(RetrievalConfig(chunk_size=300), emb, docs)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_retriever_modes_and_filters(mode):
    emb = FakeEmbedder()
    d2 = Document("sop", "SOP", "sop", "illustrative", "0", "draft", *parse_markdown("# S\nbus 57 procedure step.\n"))
    ix = DocIndex.build(RetrievalConfig(chunker="recursive", chunk_size=12, mode=mode), emb, [doc(), d2])
    res = Retriever(ix).retrieve("fault at bus 57", top_k=3)
    assert res["passages"][0]["rank"] == 1 and np.isfinite(res["dense_top1"])
    ix.cfg.filters = {"approval_status": ["curated_for_research"]}
    assert all(p["document_id"] == "d1" for p in Retriever(ix).retrieve("bus 57", top_k=5)["passages"])


def test_parent_child_returns_parent_once():
    emb = FakeEmbedder()
    ix = DocIndex.build(RetrievalConfig(chunker="parent_child", chunk_size=4, parent_size=40), emb, [doc()])
    ps = Retriever(ix).retrieve("alpha beta delta", top_k=5)["passages"]
    keys = [(p["document_id"], p["char_start"], p["char_end"]) for p in ps]
    assert len(keys) == len(set(keys)) and all(len(p["text"].split()) > 4 for p in ps[:1])


def test_bm25_tokenizer_keeps_identifiers_and_rrf():
    t = tokenize("RGSAGE-Max at k=10 has 0.839; SOP-ILL-3.2")
    assert {"rgsage-max", "rgsage", "max", "k=10", "10", "0.839", "sop-ill-3.2"} <= set(t)
    assert [d for d, _ in rrf([["a", "b"], ["b", "c"]], 60)][0] == "b"


@pytest.mark.parametrize("store", ["chroma", "qdrant_local"])
def test_exact_stores_match_faiss_ranking(store):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids, pay = [f"c{i}" for i in range(40)], [{"approval_status": "curated_for_research" if i % 2 else "x"} for i in range(40)]
    ref, other = make_store("faiss_flat", 16), make_store(store, 16)
    for s in (ref, other):
        s.add(ids, v, pay)
    q = v[:3] + 0.01
    flt = {"approval_status": ["curated_for_research"]}
    a, b = ref.search(q, 5, flt), other.search(q, 5, flt)
    assert [[i for i, _ in r] for r in a] == [[i for i, _ in r] for r in b]


def test_cost_ledger_price_limit_and_append(tmp_path):
    prices = {"m": {"input_per_1m": 0.02, "date": "2026-10-31"}}
    L = Ledger(tmp_path / "l.jsonl", prices, limit_usd=0.001)
    with pytest.raises(MissingPriceError):
        L.reserve("unknown", 10, 0, "t")
    assert L.reserve("m", 10_000, 0, "t") == pytest.approx(0.0002)
    L.commit("m", 10_000, 0, "t")
    assert L.spent() == pytest.approx(0.0002)
    with pytest.raises(CostLimitError):
        L.reserve("m", 100_000, 0, "t")


def test_openai_embedder_uses_ledger_and_cache(tmp_path, monkeypatch):
    import gridsense.retrieval.embeddings as E
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path / "cache")
    calls = []

    class Client:
        class embeddings:  # noqa: N801
            @staticmethod
            def create(model, input, **kw):
                calls.append(list(input))
                return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, float(len(t))]) for t in input],
                                       usage=SimpleNamespace(prompt_tokens=7))
    L = Ledger(tmp_path / "l.jsonl", {"text-embedding-3-small": {"input_per_1m": 0.02, "date": "2026-10-31"}}, limit_usd=1.0)
    e = Embedder(EmbedderSpec(model="text-embedding-3-small", backend="openai", max_seq_length=512), client=Client(), ledger=L)
    v1 = e.encode(["a", "bb"], "passage")
    v2 = e.encode(["a", "bb"], "passage")
    assert len(calls) == 1 and np.allclose(v1, v2) and np.allclose(np.linalg.norm(v1, axis=1), 1)
    assert json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])["input_tokens"] == 7
    with pytest.raises(RuntimeError, match="ledger"):
        Embedder(EmbedderSpec(model="text-embedding-3-small", backend="openai", max_seq_length=512), client=Client()).encode(["new"], "query")


def test_corpus_loads_with_integrity_and_unique_ids():
    from gridsense.ingest.corpus import load_corpus
    docs = load_corpus()
    ids = [d.document_id for d in docs]
    assert len(ids) == len(set(ids)) and {"paper1_rev1170", "results_md", "feeder_overview", "sop_illustrative_gridsense_review"} <= set(ids)
    notes = [d for d in docs if d.document_type == "technical_note"]
    assert len(notes) == 30
    # O30 and O34 (2026-11-02): all 30 notes accepted by the author; the illustrative SOP stays draft until the author has read it
    assert {d.status for d in notes} == {"reviewed"}
    assert next(d for d in docs if d.document_type == "sop_illustrative").status == "draft"


def test_no_note_calls_0p846_the_dropout_aware_value():
    """Author rule (O30): 0.846 is the STANDARD RGSAGE-Max macro-F1 at k = 0; the dropout-aware value is 0.839."""
    import re
    from gridsense.ingest.corpus import load_corpus
    for d in load_corpus():
        if d.document_type not in ("technical_note", "sop_illustrative"):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", d.text):
            if "0.846" in sent:
                assert "standard" in sent.lower() and "dropout-aware" not in sent.lower(), (d.document_id, sent)


def test_reviewed_note_front_matter_follows_the_o30_rules():
    from gridsense.ingest.corpus import CORPUS
    from gridsense.ingest.parsers import split_frontmatter
    generic = "general background, not checked against a specific page"
    for n in ("01_single_line_to_ground", "02_line_to_line", "03_double_line_to_ground", "04_three_phase_fault", "05_fault_resistance",
              "06_voltage_sag_propagation", "07_micro_pmu", "11_causes_of_sensor_dropout", "19_calibration", "20_margin", "21_selective_prediction", "22_conformal_sets"):
        m, _ = split_frontmatter((CORPUS / "notes_v1" / f"note_{n}.md").read_text(encoding="utf-8"))
        assert m["source_ref"] == generic and m["further_reading"], n
    for n in ("09_ambiguity_set", "25_prediction_is_not_confirmation", "26_gate_is_not_changed_by_text", "28_simulation_limits"):
        m, _ = split_frontmatter((CORPUS / "notes_v1" / f"note_{n}.md").read_text(encoding="utf-8"))
        assert "implementation-plan.md" in m["source_ref"] and "further_reading" not in m
        assert not any(w in m["source_ref"] for w in ("Phadke", "Thorp", "Anderson", "Kersting"))


def test_golden_retrieval_v2_is_valid_and_statuses_follow_the_author():
    from gridsense.config import ROOT, load_config
    from gridsense.ingest.corpus import load_corpus
    rows = G.load(ROOT / load_config()["golden_sets"]["retrieval"]["path"], statuses=("draft", "reviewed"))
    by = {r["query_id"]: r for r in rows}
    assert len(rows) == 68 and all(r["golden_version"] == 2 for r in rows)
    drafts = [r["query_id"] for r in rows if r["status"] == "draft"]
    assert drafts == ["R-0055", "R-0056"] + [f"R-{i:04d}" for i in range(57, 69)]       # the author checks the spans (O34) and the 12 near-miss queries
    assert sum(r["status"] == "reviewed" for r in rows) == 54
    assert G.validate(rows, load_corpus()) == []
    assert [c["document_id"] for c in by["R-0005"]["golden_contexts"]] == ["note_05_fault_resistance"]     # results_md row removed (O30)
    assert by["R-0055"]["split"] == "test" and by["R-0055"]["golden_contexts"][0]["document_id"] == "feeder_fault_buses" and "bus 47" in by["R-0055"]["reference_answer"].lower()
    assert by["R-0056"]["split"] == "dev" and by["R-0056"]["golden_contexts"][0]["document_id"] == "feeder_fault_taxonomy"
    reviewed = [r for r in rows if r["status"] == "reviewed"]
    assert sum(r["split"] == "dev" and r["answerable"] for r in reviewed) == 24 and sum(r["split"] == "test" and r["answerable"] for r in reviewed) == 22


def test_near_miss_unanswerable_queries_stay_unanswerable_by_the_corpus():
    """Author, 2026-11-02 (step 6): 12 near-miss queries (6 dev, 6 test). The topic is in the corpus, the answer is not: the absence patterns must still find nothing."""
    import re
    from gridsense.config import ROOT, load_config
    from gridsense.ingest.corpus import load_corpus
    rows = [r for r in G.load(ROOT / load_config()["golden_sets"]["retrieval"]["path"], statuses=("draft",)) if r["query_type"] == "unanswerable"]
    assert len(rows) == 12 and sum(r["split"] == "dev" for r in rows) == 6 and sum(r["split"] == "test" for r in rows) == 6
    corpus = "\n".join(d.text for d in load_corpus())
    for r in rows:
        assert not r["answerable"] and r["golden_contexts"] == [] and r["near_miss_of"] and r["absence_check"]["patterns_not_found_in_the_corpus"]
        for pat in r["absence_check"]["patterns_not_found_in_the_corpus"]:
            assert not re.search(pat, corpus, flags=re.I), (r["query_id"], pat)


def test_ledger_batch_price_is_half_and_missing_batch_price_is_refused(tmp_path):
    prices = {"m": {"input_per_1m": 1.0, "output_per_1m": 5.0, "batch_input_per_1m": 0.5, "batch_output_per_1m": 2.5, "date": "2026-11-02"}, "n": {"input_per_1m": 1.0, "date": "2026-11-02"}}
    L = Ledger(tmp_path / "l.jsonl", prices, limit_usd=1.0)
    assert L.price("m", 1_000_000, 1_000_000) == pytest.approx(6.0) and L.price("m", 1_000_000, 1_000_000, batch=True) == pytest.approx(3.0)
    with pytest.raises(MissingPriceError):
        L.price("n", 10, 10, batch=True)
    L.commit("m", 100_000, 10_000, "t", batch=True)
    assert json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])["batch"] is True
