"""Tests for the Layer 1 sweep driver: candidate lists, the paired bootstrap, and the selection rule (ties go to the cheaper option)."""
from __future__ import annotations

import numpy as np

from gridsense.eval import sweep as S


def test_candidate_lists_follow_eval_md_and_change_one_variable():
    st = S.base_state()
    assert [c[0] for c in S.candidates("chunker", st)] == S.CHUNKERS
    for name, rc, emb in S.candidates("chunker", st):
        assert {k: v for k, v in rc.items() if k not in ("chunker", "parent_size", "overlap")} == {k: v for k, v in st["retrieval"].items() if k not in ("chunker", "parent_size", "overlap")}
        assert (rc["parent_size"] == 4 * rc["chunk_size"]) == (name == "parent_child")
    assert [c[0] for c in S.candidates("chunk_size", st)] == ["size_100", "size_150", "size_200"]
    assert [c[0] for c in S.candidates("overlap", st)] == ["overlap_0.00", "overlap_0.10", "overlap_0.15"]
    assert [c[2] for c in S.candidates("embedder", st)] == S.EMBEDDERS and all(c[1] == st["retrieval"] for c in S.candidates("embedder", st))
    modes = {n: (rc["mode"], rc["rrf_k"]) for n, rc, _ in S.candidates("retrieval_mode", st)}
    assert modes["hybrid_k20"] == ("hybrid", 20) and modes["hybrid_k100"] == ("hybrid", 100) and modes["bm25"][0] == "bm25"
    assert [c[1]["reranker"] is not None for c in S.candidates("reranker", st)] == [False, True]
    assert S.candidates("overlap", {**st, "retrieval": {**st["retrieval"], "chunker": "parent_child"}}) == []   # not applicable


def test_every_order_list_covers_its_candidates():
    st = S.base_state()
    for axis in S.AXES:
        if axis == "top_k":
            assert S.ORDER[axis] == ["k3", "k5", "k8"]
            continue
        assert sorted(c[0] for c in S.candidates(axis, st)) == sorted(S.ORDER[axis]), axis


def test_paired_bootstrap_ci():
    base = np.random.default_rng(0).random(25)
    lo, hi = S.paired_ci(base + 0.1, base)
    assert abs(lo - 0.1) < 1e-9 and abs(hi - 0.1) < 1e-9


def test_selection_rule_v2_best_quality_then_cheap():
    rng = np.random.default_rng(1)
    base = rng.random(25) * 0.5
    # a clearly better candidate wins even if it is the most expensive (the author: not just cheap)
    per_q = {"fixed_token": base, "recursive": base + 0.2, "sentence": base}
    sel = S.select("chunker", per_q, cost={"fixed_token": 100.0, "recursive": 900.0, "sentence": 300.0})
    assert sel["best"] == "recursive" and sel["winner"] == "recursive" and sel["band"] == ["recursive"]
    # near-equal quality (inside one query on one of three metrics = 0.0133): the cheaper candidate wins
    near = {"recursive": base + 0.005, "sentence": base, "structure_aware": base - 0.30}
    sel = S.select("chunker", near, cost={"recursive": 800.0, "sentence": 500.0, "structure_aware": 100.0})
    assert sel["best"] == "recursive" and sel["winner"] == "sentence" and set(sel["band"]) == {"recursive", "sentence"}
    # a cheaper candidate that is clearly worse never wins (size_150 versus size_200 in the first run)
    sel = S.select("chunk_size", {"size_150": base - 0.05, "size_200": base}, cost={"size_150": 400.0, "size_200": 600.0})
    assert sel["winner"] == "size_200"
    # more split golden spans exclude a candidate from the band even when the quality is equal
    sel = S.select("chunk_size", {"size_150": base, "size_200": base}, cost={"size_150": 400.0, "size_200": 600.0}, splits={"size_150": 3, "size_200": 0})
    assert sel["winner"] == "size_200" and "size_150" not in sel["band"]
    # top_k: one metric, one query = 0.04; the smaller k wins when recall differs by one query
    r8 = np.array([1.0] * 24 + [0.0]); r5 = np.array([1.0] * 23 + [0.0, 0.0])
    sel = S.select("top_k", {"k5": r5, "k8": r8}, cost={"k5": 500.0, "k8": 800.0}, n_metrics=1)
    assert sel["best"] == "k8" and sel["winner"] == "k5" and abs(sel["resolution"] - 0.04) < 1e-12


def test_estimate_and_cap_are_consistent_with_the_approved_estimate():
    assert S.CAP_USD > 0.0013 and S.CAP_USD <= 0.002      # O31: estimate 0.0013 USD; cap with a margin


def test_reachability_detects_a_span_that_a_chunk_boundary_splits():
    from gridsense.eval.layer1 import reachability
    from gridsense.ingest.parsers import Document, parse_markdown
    from gridsense.retrieval.retriever import DocIndex, RetrievalConfig
    import hashlib
    from gridsense.retrieval.embeddings import Embedder, EmbedderSpec, prefixed
    from gridsense.retrieval.lexical import tokenize

    class FakeEmbedder(Embedder):
        def __init__(self):
            super().__init__(EmbedderSpec(model="fake", backend="fake", max_seq_length=256))

        def count_tokens(self, text):
            return len(text.split())

        def encode(self, texts, kind):
            out = np.zeros((len(texts), 32), np.float32)
            for i, t in enumerate(prefixed(self.spec, texts, kind)):
                for w in tokenize(t):
                    out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    text, secs = parse_markdown("# T\n\n" + " ".join(f"w{i}" for i in range(60)) + ". The fault is at bus 57 and it is a ground fault.\n")
    d = Document("d", "D", "curated_note", "curated_for_research", "0", "draft", text, secs)
    rows = [{"query_id": "R-1", "golden_contexts": [{"document_id": "d", "span": "bus 57 and it is a ground fault"}], "answerable": True}]
    big = DocIndex.build(RetrievalConfig(chunker="fixed_token", chunk_size=200), FakeEmbedder(), [d])
    assert reachability(big, rows)["fraction"] == 1.0
    cut = [s for s in range(8, 14) if reachability(DocIndex.build(RetrievalConfig(chunker="fixed_token", chunk_size=s), FakeEmbedder(), [d]), rows)["fraction"] < 1.0]
    assert cut, "a small chunk size must split the span for at least one size"


def test_eval_yaml_matches_the_frozen_sweep_winners():
    """The system configuration must equal the recorded sweep winners (reports/sweeps/WINNERS.json), so that a later stage cannot drift from what was frozen."""
    from gridsense.config import load_config
    cfg, st = load_config(), S.load_winners()["state"]["retrieval"]
    assert (cfg["chunking"]["strategy"], cfg["chunking"]["chunk_size_tokens"], cfg["chunking"]["overlap_fraction"]) == (st["chunker"], st["chunk_size"], st["overlap"])
    assert (cfg["retrieval"]["mode"], cfg["retrieval"]["top_k"], cfg["retrieval"]["rrf_k"], cfg["vector_store"]["backend"]) == (st["mode"], st["top_k"], st["rrf_k"], st["store"])
    assert cfg["chunking"]["chunk_meter"] == st["chunk_meter"] and st["store"] != "faiss_flat"      # author, 2026-11-02: a real vector store, not FAISS
