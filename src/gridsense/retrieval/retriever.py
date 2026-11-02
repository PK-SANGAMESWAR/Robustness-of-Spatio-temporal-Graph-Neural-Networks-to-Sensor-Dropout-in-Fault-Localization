"""Document index and retriever (EVAL.md § 3; implementation-plan.md § 8–9; IMPLEMENTATION.md § 6).

DocIndex = chunks + vectors + one vector store + BM25, with a fingerprint over every setting that changes the index
(corpus content, parser, chunker, size, overlap, embedder + revision + prefixes + window, store). A saved index whose
fingerprint differs from the requested one raises StaleIndexError: rebuild, never reuse.
Retrieval modes: dense, bm25, hybrid (RRF); optional cross-encoder rerank of the fused top candidates. Parent-child returns
the parent text, deduplicated. The sufficiency signal is the top-1 dense cosine of the query over all chunks, whatever the
mode; its threshold is chosen on the dev split (gridsense.eval.retrieval_metrics.choose_sufficiency_threshold).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gridsense.cases.store import StaleIndexError, fingerprint
from gridsense.ingest.chunking import CHUNKER_VERSION, Chunk, chunk_document
from gridsense.ingest.corpus import corpus_fingerprint
from gridsense.ingest.parsers import PARSER_VERSION, Document
from gridsense.retrieval.embeddings import Embedder, get_meter
from gridsense.retrieval.lexical import BM25, Reranker, rrf
from gridsense.retrieval.stores import make_store

MODES = ("dense", "bm25", "hybrid")


@dataclass
class RetrievalConfig:
    chunker: str = "structure_aware"
    chunk_size: int = 200
    overlap: float = 0.0
    parent_size: int | None = None
    store: str = "faiss_flat"
    mode: str = "dense"
    top_k: int = 5
    dense_candidates: int = 20
    lexical_candidates: int = 20
    rrf_k: int = 60
    reranker: str | None = None
    reranker_revision: str | None = None
    rerank_top: int = 20
    filters: dict = field(default_factory=lambda: {"approval_status": ["curated_for_research", "illustrative"]})
    semantic_percentile: float = 20.0
    chunk_meter: str | None = None   # tokenizer that measures chunk size; None = the embedder's own (sweeps pin MiniLM so all embedders see the same chunks)

    def index_part(self) -> dict:
        return {k: getattr(self, k) for k in ("chunker", "chunk_size", "overlap", "parent_size", "store", "semantic_percentile", "chunk_meter")}


def index_fingerprint(cfg: RetrievalConfig, embedder: Embedder, docs: list[Document], chunk_embedder: Embedder | None = None) -> str:
    ce = chunk_embedder.spec.key() if (cfg.chunker == "semantic" and chunk_embedder is not None) else None   # semantic cuts depend on this model
    return fingerprint(corpus=corpus_fingerprint(docs), parser=PARSER_VERSION, chunker_version=CHUNKER_VERSION,
                       index=cfg.index_part(), embedder=embedder.spec.key(), window=embedder.window, chunk_embedder=ce)


@dataclass
class DocIndex:
    cfg: RetrievalConfig
    embedder: Embedder
    chunks: list[Chunk]
    vectors: np.ndarray
    fingerprint: str
    payloads: list[dict]
    build_seconds: float = 0.0
    truncated: int = 0
    max_embed_tokens: int = 0
    store: object = None
    bm25: BM25 | None = None

    @classmethod
    def build(cls, cfg: RetrievalConfig, embedder: Embedder, docs: list[Document], chunk_embedder: Embedder | None = None) -> "DocIndex":
        t0 = time.perf_counter()
        if cfg.chunk_size > embedder.window:
            raise ValueError(f"chunk size {cfg.chunk_size} > embedder window {embedder.window}: the size would only measure truncation")
        ce = chunk_embedder or embedder
        embed_fn = (lambda texts: ce.encode(texts, "passage")) if cfg.chunker == "semantic" else None
        count = get_meter(cfg.chunk_meter).count_tokens if cfg.chunk_meter else embedder.count_tokens
        chunks: list[Chunk] = []
        for d in docs:
            chunks += chunk_document(d, cfg.chunker, cfg.chunk_size, count, cfg.overlap,
                                     parent_size=cfg.parent_size, embed_fn=embed_fn, semantic_percentile=cfg.semantic_percentile)
        meta = {d.document_id: d for d in docs}
        payloads = [{"document_id": c.document_id, "approval_status": meta[c.document_id].approval_status,
                     "document_type": meta[c.document_id].document_type, "status": meta[c.document_id].status} for c in chunks]
        before = embedder.truncated
        vectors = embedder.encode([c.embed_text for c in chunks], "passage")
        idx = cls(cfg, embedder, chunks, vectors, index_fingerprint(cfg, embedder, docs, chunk_embedder), payloads,
                  truncated=embedder.truncated - before,
                  max_embed_tokens=max(embedder.count_tokens(c.embed_text) for c in chunks))
        idx._attach()
        idx.build_seconds = time.perf_counter() - t0
        return idx

    def _attach(self) -> None:
        self.store = make_store(self.cfg.store, self.vectors.shape[1])
        self.store.add([c.chunk_id for c in self.chunks], self.vectors, self.payloads)
        self.bm25 = BM25([c.chunk_id for c in self.chunks], [c.embed_text for c in self.chunks])
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c, p in zip(self.chunks, self.payloads):
                f.write(json.dumps({**asdict(c), "section_path": list(c.section_path), "payload": p}, ensure_ascii=False) + "\n")
        (folder / "index_manifest.json").write_text(json.dumps(
            {"fingerprint": self.fingerprint, "config": asdict(self.cfg), "embedder": self.embedder.spec.key(),
             "window": self.embedder.window, "n_chunks": len(self.chunks), "truncated": self.truncated,
             "max_embed_tokens": self.max_embed_tokens, "parser": PARSER_VERSION, "chunker": CHUNKER_VERSION, "build_seconds": self.build_seconds}, indent=2))

    @classmethod
    def load(cls, folder: Path, cfg: RetrievalConfig, embedder: Embedder, docs: list[Document], chunk_embedder: Embedder | None = None) -> "DocIndex":
        man = json.loads((folder / "index_manifest.json").read_text())
        want = index_fingerprint(cfg, embedder, docs, chunk_embedder)
        if man["fingerprint"] != want:
            raise StaleIndexError(f"{folder}: index fingerprint {man['fingerprint'][:12]} != requested {want[:12]}; rebuild")
        rows = [json.loads(l) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        chunks = [Chunk(**{k: (tuple(v) if k == "section_path" else v) for k, v in r.items() if k != "payload"}) for r in rows]
        idx = cls(cfg, embedder, chunks, np.load(folder / "vectors.npy"), want, [r["payload"] for r in rows],
                  truncated=man["truncated"], max_embed_tokens=man["max_embed_tokens"], build_seconds=man.get("build_seconds", float("nan")))
        idx._attach()
        return idx


class Retriever:
    def __init__(self, index: DocIndex):
        self.ix, self.cfg = index, index.cfg
        self.reranker = Reranker(self.cfg.reranker, self.cfg.reranker_revision) if self.cfg.reranker else None

    def retrieve(self, query: str, top_k: int | None = None) -> dict:
        """{'passages': [...], 'dense_top1': float, 'latency_ms': float}. Passages carry provenance, never a probability."""
        t0, cfg, ix = time.perf_counter(), self.cfg, self.ix
        k = top_k or cfg.top_k
        q = ix.embedder.encode([query], "query")
        allowed = {c.chunk_id for c, p in zip(ix.chunks, ix.payloads)
                   if all(p.get(f) in set(v) for f, v in (cfg.filters or {}).items())}
        dense = ix.store.search(q, max(cfg.dense_candidates, k), cfg.filters)[0]
        dense_top1 = float(np.max(ix.vectors[[ix._by_id[c] for c in allowed]] @ q[0])) if allowed else float("-inf")
        if cfg.mode == "dense":
            ranked = [(cid, s) for cid, s in dense]
        elif cfg.mode == "bm25":
            ranked = ix.bm25.search(query, max(cfg.lexical_candidates, k), allowed)
        elif cfg.mode == "hybrid":
            lex = ix.bm25.search(query, cfg.lexical_candidates, allowed)
            ranked = rrf([[c for c, _ in dense[:cfg.dense_candidates]], [c for c, _ in lex]], cfg.rrf_k)
        else:
            raise ValueError(f"mode {cfg.mode!r}; one of {MODES}")
        if self.reranker is not None:
            cand = [cid for cid, _ in ranked[:cfg.rerank_top]]
            s = self.reranker.scores(query, [ix.chunks[ix._by_id[c]].embed_text for c in cand])
            ranked = [(cand[i], float(s[i])) for i in np.argsort(-s, kind="stable")]
        out, seen = [], set()
        for cid, score in ranked:
            c = ix.chunks[ix._by_id[cid]]
            key = (c.document_id, c.start, c.end)       # parent-child: one passage per parent
            if key in seen:
                continue
            seen.add(key)
            out.append({"rank": len(out) + 1, "chunk_id": cid, "document_id": c.document_id, "section_path": list(c.section_path),
                        "char_start": c.start, "char_end": c.end, "text": c.text, "score": float(score),
                        "approval_status": ix.payloads[ix._by_id[cid]]["approval_status"]})
            if len(out) == k:
                break
        return {"passages": out, "dense_top1": dense_top1, "latency_ms": (time.perf_counter() - t0) * 1000}
