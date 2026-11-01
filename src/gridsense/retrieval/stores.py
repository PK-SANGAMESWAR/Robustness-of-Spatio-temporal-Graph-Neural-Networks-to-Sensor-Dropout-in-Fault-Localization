"""Vector stores with one interface (EVAL.md § 3 step 7): FAISS flat (reference), Chroma (HNSW), Qdrant local mode.

Qdrant local mode (no server) always searches exactly (brute force; qdrant-client warning, 2026-10-31), so it cannot
measure ANN recall; that would need a Qdrant server, which is not run here.

With exact search every store returns the same ranking for the same vectors, so the store is chosen by operational numbers
(parity, ANN recall, latency, build time, RAM, disk, filtering). Payload filters: {field: [allowed values]}.
Cosine on L2-normalized vectors = inner product.
"""
from __future__ import annotations

import tempfile
import uuid

import numpy as np

STORES = ("faiss_flat", "chroma", "qdrant_local")


def _passes(payload: dict, filters: dict | None) -> bool:
    return not filters or all(payload.get(k) in set(v) for k, v in filters.items())


class FaissFlat:
    name = "faiss_flat"

    def __init__(self, dim: int):
        import faiss
        self.index, self.ids, self.payloads = faiss.IndexFlatIP(dim), [], []

    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.ids += ids
        self.payloads += payloads

    def search(self, q: np.ndarray, top_n: int, filters: dict | None = None) -> list[list[tuple[str, float]]]:
        n = self.index.ntotal  # small corpus: search all, then filter (exact)
        D, I = self.index.search(np.ascontiguousarray(q, dtype=np.float32), n)
        return [[(self.ids[i], float(d)) for d, i in zip(dr, ir) if i >= 0 and _passes(self.payloads[i], filters)][:top_n]
                for dr, ir in zip(D, I)]


class ChromaStore:
    name = "chroma"

    def __init__(self, dim: int, path: str | None = None):
        import gridsense.settings  # noqa: F401  (telemetry off before chromadb)
        import chromadb
        self.client = chromadb.PersistentClient(path=path or tempfile.mkdtemp(prefix="gs_chroma_"))
        self.col = self.client.create_collection(f"gs_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "ip"})

    def add(self, ids, vectors, payloads) -> None:
        meta = [{k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in p.items()} for p in payloads]
        for s in range(0, len(ids), 1000):
            self.col.add(ids=ids[s:s + 1000], embeddings=vectors[s:s + 1000].tolist(), metadatas=meta[s:s + 1000])

    def search(self, q, top_n, filters=None):
        where = None
        if filters:
            conds = [{k: {"$in": list(v)}} for k, v in filters.items()]
            where = conds[0] if len(conds) == 1 else {"$and": conds}
        r = self.col.query(query_embeddings=q.tolist(), n_results=top_n, where=where)
        return [[(i, 1.0 - float(d)) for i, d in zip(ids, ds)] for ids, ds in zip(r["ids"], r["distances"])]


class QdrantLocal:
    name = "qdrant_local"

    def __init__(self, dim: int, path: str | None = None):
        from qdrant_client import QdrantClient, models
        self.m = models
        self.client = QdrantClient(path=path or tempfile.mkdtemp(prefix="gs_qdrant_"))
        self.col = f"gs_{uuid.uuid4().hex[:8]}"
        self.client.create_collection(self.col, vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT))
        self.key = {}

    def add(self, ids, vectors, payloads) -> None:
        pts = []
        for i, v, p in zip(ids, vectors, payloads):
            u = str(uuid.uuid5(uuid.NAMESPACE_URL, i))
            self.key[u] = i
            pts.append(self.m.PointStruct(id=u, vector=v.tolist(), payload={**p, "_id": i}))
        for s in range(0, len(pts), 500):
            self.client.upsert(self.col, points=pts[s:s + 500])

    def search(self, q, top_n, filters=None):
        flt = None
        if filters:
            flt = self.m.Filter(must=[self.m.FieldCondition(key=k, match=self.m.MatchAny(any=list(v))) for k, v in filters.items()])
        out = []
        for row in q:
            r = self.client.query_points(self.col, query=row.tolist(), limit=top_n, query_filter=flt)
            out.append([(p.payload["_id"], float(p.score)) for p in r.points])
        return out


def make_store(name: str, dim: int):
    if name == "faiss_flat":
        return FaissFlat(dim)
    if name == "chroma":
        return ChromaStore(dim)
    if name == "qdrant_local":
        return QdrantLocal(dim)
    raise ValueError(f"unknown store {name!r}; one of {STORES}")
