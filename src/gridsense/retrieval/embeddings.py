"""Embedders for document retrieval (IMPLEMENTATION.md § 2; implementation-plan.md § 4.4–4.6).

Prefixes are applied in ONE function (`prefixed`), so ingestion and query time cannot drift apart (rule R7).
Backends:
  local   sentence-transformers on CPU (MiniLM, bge-small, e5-small); free.
  openai  OpenAI embeddings API (author request 2026-10-31). Paid: every call goes through `gridsense.cost.Ledger`,
          which refuses a call above the 5 USD track limit. Responses are cached on disk by (model, dimensions, text hash),
          so the same text is never paid twice. The price comes from configs/eval.yaml `cost.prices` (the author's, with a
          date); without a price the call is refused. The key comes from gridsense.settings only.
Vectors are L2-normalized float32.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gridsense.config import TRACK

CACHE_DIR = TRACK / "eval" / "embedding_cache"


@dataclass
class EmbedderSpec:
    model: str
    backend: str = "local"               # local | openai
    revision: str | None = None          # HF commit for local models
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int | None = None    # read from the model (local) or the config (openai)
    dimensions: int | None = None        # openai only (optional shortening)
    batch_size: int = 64

    def key(self) -> dict:
        return {k: getattr(self, k) for k in ("model", "backend", "revision", "query_prefix", "passage_prefix", "dimensions")}


def prefixed(spec: EmbedderSpec, texts: list[str], kind: str) -> list[str]:
    if kind not in ("query", "passage"):
        raise ValueError(kind)
    p = spec.query_prefix if kind == "query" else spec.passage_prefix
    return [p + t for t in texts]


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


@dataclass
class Embedder:
    spec: EmbedderSpec
    _model: object = field(default=None, repr=False)
    _tok: object = field(default=None, repr=False)
    client: object = field(default=None, repr=False)      # injected OpenAI-like client (tests use a mock)
    ledger: object = field(default=None, repr=False)      # gridsense.cost.Ledger for the openai backend
    truncated: int = 0                                     # passages longer than the window (logged, never hidden)

    # ---- tokenizer and window --------------------------------------------------------------------------------
    def _load_local(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            from gridsense.config import torch_threads
            torch.set_num_threads(torch_threads())
            self._model = SentenceTransformer(self.spec.model, revision=self.spec.revision, device="cpu")
            self._tok = self._model.tokenizer
            self.spec.max_seq_length = int(self._model.max_seq_length)
        return self._model

    def _tokenizer(self):
        if self._tok is None:
            if self.spec.backend == "local":
                self._load_local()
            else:
                import tiktoken
                self._tok = tiktoken.get_encoding("cl100k_base")
        return self._tok

    def count_tokens(self, text: str) -> int:
        tok = self._tokenizer()
        if self.spec.backend == "local":
            return len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])
        return len(tok.encode(text))

    @property
    def window(self) -> int:
        if self.spec.max_seq_length is None and self.spec.backend == "local":
            self._load_local()
        if self.spec.max_seq_length is None:
            raise ValueError(f"{self.spec.model}: max_seq_length not set (configs/eval.yaml)")
        return self.spec.max_seq_length

    # ---- encoding -------------------------------------------------------------------------------------------
    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        texts = prefixed(self.spec, texts, kind)
        if kind == "passage":
            self.truncated += sum(self.count_tokens(t) > self.window for t in texts)
        if self.spec.backend == "local":
            v = self._load_local().encode(texts, batch_size=self.spec.batch_size, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False)
            return _norm(v)
        if self.spec.backend == "openai":
            return self._encode_openai(texts)
        raise ValueError(self.spec.backend)

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha256(json.dumps([self.spec.model, self.spec.dimensions, text]).encode()).hexdigest()
        return CACHE_DIR / self.spec.model.replace("/", "_") / f"{h[:2]}/{h}.npy"

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            p = self._cache_path(t)
            if p.exists():
                out[i] = np.load(p)
            else:
                todo.append(i)
        for s in range(0, len(todo), self.spec.batch_size):
            idx = todo[s:s + self.spec.batch_size]
            batch = [texts[i] for i in idx]
            n_tok = sum(self.count_tokens(t) for t in batch)
            if self.ledger is None:
                raise RuntimeError("openai embedding needs a cost ledger (gridsense.cost.Ledger)")
            self.ledger.reserve(self.spec.model, input_tokens=n_tok, output_tokens=0, purpose="embedding")
            client = self.client or _openai_client()
            kw = {"dimensions": self.spec.dimensions} if self.spec.dimensions else {}
            resp = client.embeddings.create(model=self.spec.model, input=batch, **kw)
            used = getattr(getattr(resp, "usage", None), "prompt_tokens", n_tok)
            self.ledger.commit(self.spec.model, input_tokens=used, output_tokens=0, purpose="embedding")
            for i, d in zip(idx, resp.data):
                v = np.asarray(d.embedding, dtype=np.float32)
                p = self._cache_path(texts[i])
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, v)
                out[i] = v
        return _norm(np.stack(out)) if out else np.zeros((0, self.spec.dimensions or 1), np.float32)


def _openai_client():
    from openai import OpenAI

    from gridsense.settings import secret
    return OpenAI(api_key=secret("OPENAI_API_KEY"))
