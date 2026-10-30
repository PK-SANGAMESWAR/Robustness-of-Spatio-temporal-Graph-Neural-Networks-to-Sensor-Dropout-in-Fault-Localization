"""The deterministic gate (implementation-plan.md § 11; author decisions O21, O22, O23, 2026-10-30). Pure numpy; no LLM.

CONFIDENT      : isotonic probability >= tau_p AND margin >= tau_m AND kNN agreement >= tau_k   (per k; thresholds from cal-B)
LOW_CONFIDENCE : valid input, supported dropout pattern, but a condition failed; a conformal candidate set (90 %, per k) is attached
REJECT         : invalid input (INVALID_INPUT) or a dropout level / pattern the gate was not fitted for (UNSUPPORTED_DROPOUT_PATTERN)

The isotonic map (fitted on cal-A per k, post-fault windows), the thresholds (cal-B) and the conformal quantile (cal-A) are in the frozen
config (configs/gate_v1.json). The logistic combination of the three features is NOT used for the decision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CONFIDENT, LOW_CONFIDENCE, REJECT = 0, 1, 2
OUTCOME_NAMES = {CONFIDENT: "CONFIDENT", LOW_CONFIDENCE: "LOW_CONFIDENCE", REJECT: "REJECT"}
R_PASSED, R_LOW_P, R_LOW_MARGIN, R_LOW_AGREE, R_UNSUPPORTED, R_INVALID = 0, 1, 2, 4, 8, 16
REASON_NAMES = {R_PASSED: "VERIFICATION_PASSED", R_LOW_P: "LOW_CALIBRATED_CONFIDENCE", R_LOW_MARGIN: "LOW_MARGIN",
                R_LOW_AGREE: "LOW_KNN_AGREEMENT", R_UNSUPPORTED: "UNSUPPORTED_DROPOUT_PATTERN", R_INVALID: "INVALID_INPUT"}
N_VOTES = 25.0


def canonical_hash(cfg: dict) -> str:
    """sha256 of the canonical JSON of the config without its own hash field."""
    body = {k: v for k, v in cfg.items() if k != "config_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reasons_to_names(code: int) -> list[str]:
    return [REASON_NAMES[b] for b in (R_LOW_P, R_LOW_MARGIN, R_LOW_AGREE, R_UNSUPPORTED, R_INVALID) if code & b] or [REASON_NAMES[R_PASSED]]


def vote_features(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(top-1 vote share, top-1 minus top-2 share). Votes are the soft vote (sum of 25 node softmaxes)."""
    part = np.sort(votes, axis=1)
    return part[:, -1] / N_VOTES, (part[:, -1] - part[:, -2]) / N_VOTES


class Gate:
    def __init__(self, cfg: dict):
        if cfg.get("config_sha256") and cfg["config_sha256"] != canonical_hash(cfg):
            raise ValueError("gate config hash mismatch: the frozen config was changed")
        self.cfg = cfg
        self.per_k = {int(k): v for k, v in cfg["per_k"].items()}
        self.supported_k = sorted(self.per_k)
        self.supported_patterns = list(cfg["supported_patterns"])

    @classmethod
    def load(cls, path: Path) -> "Gate":
        return cls(json.loads(Path(path).read_text()))

    def calibrated(self, k: int, top1_share: np.ndarray) -> np.ndarray:
        p = self.per_k[k]
        return np.interp(top1_share, p["iso_x"], p["iso_y"])

    def candidate_sets(self, k: int, votes: np.ndarray) -> np.ndarray:
        """Boolean [n, 26]: classes whose vote share is >= 1 - q (conformal, fitted on cal-A)."""
        return (votes / N_VOTES) >= 1.0 - self.per_k[k]["conformal_q"]

    def decide(self, k: int, votes: np.ndarray, agreement: np.ndarray, pattern: str = "random", operating_point: str = "acc95") -> dict:
        """Vectorised decision for n windows that share the dropout level k and pattern. Returns arrays (outcome, reason bitmask, p, margin, set)."""
        n = len(votes)
        outcome, reason = np.full(n, LOW_CONFIDENCE), np.zeros(n, dtype=int)
        if k not in self.per_k or pattern not in self.supported_patterns:
            return {"outcome": np.full(n, REJECT), "reason": np.full(n, R_UNSUPPORTED), "p": np.full(n, np.nan), "margin": np.full(n, np.nan), "set": None}
        votes = np.asarray(votes, dtype=float)
        bad = ~np.isfinite(votes).all(1) | (np.abs(votes.sum(1) - N_VOTES) > 1e-3) | ~np.isfinite(agreement)
        top1, margin = vote_features(np.where(bad[:, None], 0.0, votes))
        p = self.calibrated(k, top1)
        th = self.per_k[k][operating_point]
        low_p, low_m, low_a = p < th["tau_p"], margin < th["tau_m"], agreement < th["tau_k"]
        reason |= np.where(low_p, R_LOW_P, 0) | np.where(low_m, R_LOW_MARGIN, 0) | np.where(low_a, R_LOW_AGREE, 0)
        outcome[~(low_p | low_m | low_a)] = CONFIDENT
        outcome[bad], reason[bad] = REJECT, R_INVALID
        return {"outcome": outcome, "reason": reason, "p": p, "margin": margin, "set": self.candidate_sets(k, np.where(bad[:, None], 0.0, votes))}
