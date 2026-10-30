"""FAISS case store over latent embeddings of TRAIN windows (implementation-plan.md § 8.6; author decisions O4, O15, O20).

Entry = one train window at k = 0: representation (b) mean + max of the per-node embeddings (128-d, L2-normalised, inner product =
cosine), true label, STGNN prediction and whether it was correct, run, scenario. Exact search (IndexFlatIP). A search can exclude
every entry of the query's own scenario (held-out store, O20) or of the query's own run (leave-one-out).
Safety rules enforced in code: no entry from a forbidden (val/test) run; the stored fingerprint must equal the expected one; a query
from another checkpoint is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gridsense.data.splits import event_of, scenario_of


class StaleIndexError(RuntimeError):
    """The store's fingerprint or checkpoint differs from what the caller expects. Rebuild; never use a stale store."""


class LeakageError(RuntimeError):
    """A forbidden (val/test) run or simulation seed is in the store."""


def fingerprint(**parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@dataclass
class CaseStore:
    vectors: np.ndarray            # [n, d] float32, L2-normalised
    window_ids: list[str]
    labels: np.ndarray             # true class (0..25)
    pred: np.ndarray               # STGNN top-1 at k = 0
    manifest: dict                 # includes "fingerprint", "checkpoint_sha256"

    def __post_init__(self) -> None:
        import faiss

        self.runs = np.array([event_of(w) for w in self.window_ids])
        self.scenarios = np.array([scenario_of(r) for r in self.runs])
        self.pred_correct = self.pred == self.labels
        self._run_vocab = {r: i for i, r in enumerate(sorted(set(self.runs)))}
        self._scn_vocab = {s: i for i, s in enumerate(sorted(set(self.scenarios)))}
        self.run_code = np.array([self._run_vocab[r] for r in self.runs])
        self.scn_code = np.array([self._scn_vocab[s] for s in self.scenarios])
        self.max_per_scenario = int(np.bincount(self.scn_code).max())
        self.max_per_run = int(np.bincount(self.run_code).max())
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    # ---- construction and safety ---------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, vectors: np.ndarray, window_ids: Sequence[str], labels: np.ndarray, pred: np.ndarray, *, checkpoint_sha256: str,
              forbidden_runs: set[str] = frozenset(), store_seeds: dict[str, int] | None = None, forbidden_seeds: set[int] = frozenset(),
              **fingerprint_parts) -> "CaseStore":
        runs = {event_of(w) for w in window_ids}
        clash = runs & set(forbidden_runs)
        if clash:
            raise LeakageError(f"{len(clash)} forbidden (val/test) runs in the store, e.g. {sorted(clash)[:3]}")
        if store_seeds is not None:
            bad = {r for r in runs if store_seeds[r] in forbidden_seeds}
            if bad:
                raise LeakageError(f"{len(bad)} store runs share a simulation seed with a forbidden run, e.g. {sorted(bad)[:3]}")
        ids_hash = hashlib.sha256("\n".join(window_ids).encode()).hexdigest()
        manifest = {"checkpoint_sha256": checkpoint_sha256, "n": len(window_ids), "dim": int(vectors.shape[1]), "window_ids_sha256": ids_hash,
                    "fingerprint": fingerprint(checkpoint_sha256=checkpoint_sha256, window_ids_sha256=ids_hash, dim=int(vectors.shape[1]), **fingerprint_parts),
                    "fingerprint_parts": {k: v for k, v in fingerprint_parts.items()}}
        return cls(_normalise(vectors), list(window_ids), np.asarray(labels), np.asarray(pred), manifest)

    def save(self, folder: Path) -> None:
        import faiss

        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        np.savez(folder / "meta.npz", vectors=self.vectors, window_ids=np.array(self.window_ids), labels=self.labels, pred=self.pred)
        (folder / "store_manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    @classmethod
    def load(cls, folder: Path, expected_fingerprint: str) -> "CaseStore":
        manifest = json.loads((folder / "store_manifest.json").read_text())
        if manifest["fingerprint"] != expected_fingerprint:
            raise StaleIndexError(f"store fingerprint {manifest['fingerprint'][:12]} != expected {expected_fingerprint[:12]}: rebuild the store")
        with np.load(folder / "meta.npz") as z:
            return cls(z["vectors"], [str(w) for w in z["window_ids"]], z["labels"], z["pred"], manifest)

    # ---- search ---------------------------------------------------------------------------------------------------------------
    def search(self, queries: np.ndarray, k: int, *, query_checkpoint_sha256: str, exclude_scenarios: Sequence[str] | None = None,
               exclude_runs: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(similarity [n, k], store index [n, k]) of the k nearest entries after removing the query's own scenario and/or run."""
        if query_checkpoint_sha256 != self.manifest["checkpoint_sha256"]:
            raise StaleIndexError("query latents come from another checkpoint than the store: refused")
        q = _normalise(queries)
        extra = (self.max_per_scenario if exclude_scenarios is not None else 0) + (self.max_per_run if exclude_runs is not None else 0)
        kk = min(self.index.ntotal, k + extra)
        sim, idx = self.index.search(q, kk)
        valid = np.ones(idx.shape, dtype=bool)
        if exclude_scenarios is not None:
            qs = np.array([self._scn_vocab.get(s, -1) for s in exclude_scenarios])
            valid &= self.scn_code[idx] != qs[:, None]
        if exclude_runs is not None:
            qr = np.array([self._run_vocab.get(r, -1) for r in exclude_runs])
            valid &= self.run_code[idx] != qr[:, None]
        if (valid.sum(1) < k).any():
            raise RuntimeError("fewer than k entries left after exclusion")
        order = np.argsort(~valid, axis=1, kind="stable")[:, :k]          # valid entries first, original (similarity) order kept
        rows = np.arange(len(q))[:, None]
        return sim[rows, order], idx[rows, order]
