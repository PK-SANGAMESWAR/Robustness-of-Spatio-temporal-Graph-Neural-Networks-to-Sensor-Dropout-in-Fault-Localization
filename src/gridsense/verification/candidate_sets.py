"""Candidate sets for LOW_CONFIDENCE windows (O26, author 2026-10-31). Separate from gate_v1 (which is not changed).

Split-conformal sets whose 90 % target holds CONDITIONAL on the window being LOW_CONFIDENCE under gate_v1 (95 % operating point): the quantile is fitted on
cal-A windows that gate_v1 labels LOW_CONFIDENCE, per k. Nonconformity = 1 - vote share of the true class; set = classes with vote share >= 1 - q.
CONFIDENT windows keep their single location; REJECT windows have no set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gridsense.verification.gate import N_VOTES, canonical_hash


class CandidateSets:
    def __init__(self, cfg: dict):
        if cfg.get("config_sha256") and cfg["config_sha256"] != canonical_hash(cfg):
            raise ValueError("candidate-set config hash mismatch: the frozen config was changed")
        self.cfg = cfg
        self.q = {int(k): float(v["q"]) for k, v in cfg["per_k"].items()}

    @classmethod
    def load(cls, path: Path) -> "CandidateSets":
        return cls(json.loads(Path(path).read_text()))

    def sets(self, k: int, votes: np.ndarray) -> np.ndarray:
        """Boolean [n, 26]. Only meaningful for windows that gate_v1 labels LOW_CONFIDENCE."""
        if k not in self.q:
            raise KeyError(f"no candidate-set quantile fitted for k = {k}")
        return (np.asarray(votes, dtype=float) / N_VOTES) >= 1.0 - self.q[k]
