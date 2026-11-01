"""BM25 with an identifier-aware tokenizer, Reciprocal Rank Fusion and an optional cross-encoder (EVAL.md § 3 steps 4–5).

The tokenizer keeps identifiers whole ("k=10", "rgsage-max", "0.839", "sw7", "μpmu") and also emits their parts, so an exact
identifier and its pieces both match (implementation-plan.md § 8.2).
"""
from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"[\wμ]+(?:[-=.:/][\wμ]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    out = []
    for m in _TOKEN.finditer(text.lower()):
        tok = m[0]
        out.append(tok)
        parts = re.split(r"[-=.:/]", tok)
        if len(parts) > 1:
            out += [p for p in parts if p]
    return out


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.ids = ids
        self.bm = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query: str, top_n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        s = self.bm.get_scores(tokenize(query))
        order = np.argsort(-s, kind="stable")
        return [(self.ids[i], float(s[i])) for i in order if allowed is None or self.ids[i] in allowed][:top_n]


def rrf(rankings: list[list[str]], k_rrf: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k_rrf + rank), rank from 1. Ties keep first-seen order."""
    score: dict[str, float] = {}
    for r in rankings:
        for rank, d in enumerate(r, start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(score.items(), key=lambda kv: -kv[1])


class Reranker:
    def __init__(self, model: str, revision: str | None = None):
        self.model_name, self.revision, self._m = model, revision, None

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        if self._m is None:
            from sentence_transformers import CrossEncoder
            self._m = CrossEncoder(self.model_name, revision=self.revision, device="cpu")
        return np.asarray(self._m.predict([(query, t) for t in texts], show_progress_bar=False), dtype=float)
