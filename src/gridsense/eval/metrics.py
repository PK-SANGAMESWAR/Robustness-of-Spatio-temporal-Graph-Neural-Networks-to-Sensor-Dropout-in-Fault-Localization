"""Pure metric functions for Layer 0 (no I/O). Positive class of error detection = a wrong STGNN prediction.

Ties: AUROC counts ties as 0.5. The risk-coverage curve is evaluated at tie-group boundaries (isotonic probabilities have plateaus):
AURC = sum over score groups of (group share of windows) * (error rate among windows whose score is >= the group's score).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


class TieGroups:
    """Precomputed tie groups of an error score so that weighted AUROC is O(n) per bootstrap resample."""

    def __init__(self, err_score: np.ndarray, is_err: np.ndarray):
        _, self.g = np.unique(err_score, return_inverse=True)
        self.n_groups = int(self.g.max()) + 1
        self.err = is_err.astype(np.float64)

    def auroc(self, weights: np.ndarray | None = None) -> float:
        w = np.ones_like(self.err) if weights is None else weights
        wp = np.bincount(self.g, weights=w * self.err, minlength=self.n_groups)
        wn = np.bincount(self.g, weights=w * (1.0 - self.err), minlength=self.n_groups)
        below = np.cumsum(wn) - wn
        denom = wp.sum() * wn.sum()
        return float((wp * (below + 0.5 * wn)).sum() / denom) if denom > 0 else float("nan")


def auroc(err_score: np.ndarray, is_err: np.ndarray, weights: np.ndarray | None = None) -> float:
    return TieGroups(err_score, is_err).auroc(weights)


def auprc(err_score: np.ndarray, is_err: np.ndarray) -> float:
    return float(average_precision_score(is_err.astype(int), err_score)) if 0 < is_err.sum() < len(is_err) else float("nan")


def ece_equal_mass(p: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    parts = np.array_split(np.argsort(p, kind="stable"), bins)
    return float(sum(len(b) / len(p) * abs(correct[b].mean() - p[b].mean()) for b in parts))


def brier(p: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((p - correct) ** 2))


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per distinct score (descending): (share of windows with score >= it, error rate among them, group share)."""
    u, inv = np.unique(score, return_inverse=True)
    n_g = np.bincount(inv, minlength=len(u)).astype(float)
    err_g = np.bincount(inv, weights=(~correct.astype(bool)).astype(float), minlength=len(u))
    n_cum, err_cum = np.cumsum(n_g[::-1]), np.cumsum(err_g[::-1])
    return n_cum / len(score), err_cum / n_cum, n_g[::-1] / len(score)


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    _, risk, share = risk_coverage(score, correct)
    return float((risk * share).sum())


def risk_at_coverage(score: np.ndarray, correct: np.ndarray, coverage: float) -> tuple[float, float]:
    """(risk, realised coverage) at the first score-group boundary whose coverage is >= the requested one."""
    cov, risk, _ = risk_coverage(score, correct)
    j = int(np.argmax(cov >= coverage - 1e-12))
    return float(risk[j]), float(cov[j])


def _edges(f2: np.ndarray, n_bins: int) -> np.ndarray:
    u = np.unique(f2)
    return u if len(u) <= n_bins else np.unique(np.quantile(f2, np.linspace(0.0, 1.0, n_bins, endpoint=False)))


def best_gate(p: np.ndarray, f2: np.ndarray | None, correct: np.ndarray, target: float, n_bins: int = 200) -> dict:
    """Thresholds (tau_p, tau_2) with maximum coverage subject to accuracy >= target on the accepted windows (accept: p >= tau_p and f2 >= tau_2).

    f2 = None gives a one-threshold gate on p. Exhaustive over distinct p values and over the distinct (or quantile-binned) f2 values.
    Returns {"feasible": False} when no threshold pair reaches the target.
    """
    pu, pi = np.unique(p, return_inverse=True)
    if f2 is None:
        edges, fi = np.array([-np.inf]), np.zeros(len(p), dtype=int)
    else:
        edges = _edges(f2, n_bins)
        fi = np.searchsorted(edges, f2, side="right") - 1
        fi = np.clip(fi, 0, len(edges) - 1)
    tot = np.zeros((len(pu), len(edges)))
    cor = np.zeros_like(tot)
    np.add.at(tot, (pi, fi), 1.0)
    np.add.at(cor, (pi, fi), correct.astype(float))
    st = tot[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]       # accepted count for (p >= pu[i], f2 >= edges[j])
    sc = cor[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(st > 0, sc / st, np.nan)
    ok = (st > 0) & (acc >= target - 1e-12)
    if not ok.any():
        return {"feasible": False, "target": target}
    cov = np.where(ok, st / len(p), -1.0)
    best = np.argwhere(cov == cov.max())
    i, j = best[np.argmax([acc[a, b] for a, b in best])]
    return {"feasible": True, "target": target, "tau_p": float(pu[i]), "tau_2": None if f2 is None else float(edges[j]),
            "coverage": float(st[i, j] / len(p)), "accuracy": float(acc[i, j]), "n_accepted": int(st[i, j])}


def apply_gate(p: np.ndarray, f2: np.ndarray | None, tau_p: float, tau_2: float | None) -> np.ndarray:
    keep = p >= tau_p
    return keep if f2 is None or tau_2 is None else keep & (f2 >= tau_2)


def gate_stats(accept: np.ndarray, correct: np.ndarray) -> dict:
    n_err = int((~correct.astype(bool)).sum())
    err_acc = int((accept & ~correct.astype(bool)).sum())
    return {"coverage": float(accept.mean()), "accuracy_confident": float(correct[accept].mean()) if accept.any() else float("nan"),
            "n_windows": int(len(accept)), "n_errors": n_err, "errors_accepted": err_acc,
            "error_recall": float(1 - err_acc / n_err) if n_err else float("nan")}


def best_gate3(p: np.ndarray, f2: np.ndarray, f3: np.ndarray, correct: np.ndarray, target: float, n_bins: int = 100) -> dict:
    """Like best_gate with two extra features (accept: p >= tau_p, f2 >= tau_2, f3 >= tau_3). Post-hoc diagnostic only."""
    pu, pi = np.unique(p, return_inverse=True)
    e2, e3 = _edges(f2, n_bins), _edges(f3, n_bins)
    i2 = np.clip(np.searchsorted(e2, f2, side="right") - 1, 0, len(e2) - 1)
    i3 = np.clip(np.searchsorted(e3, f3, side="right") - 1, 0, len(e3) - 1)
    tot = np.zeros((len(pu), len(e2), len(e3)))
    cor = np.zeros_like(tot)
    np.add.at(tot, (pi, i2, i3), 1.0)
    np.add.at(cor, (pi, i2, i3), correct.astype(float))

    def rev(a):
        for ax in range(3):
            a = np.flip(np.cumsum(np.flip(a, ax), ax), ax)
        return a

    st, sc = rev(tot), rev(cor)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(st > 0, sc / st, np.nan)
    ok = (st > 0) & (acc >= target - 1e-12)
    if not ok.any():
        return {"feasible": False, "target": target}
    cov = np.where(ok, st / len(p), -1.0)
    best = np.argwhere(cov == cov.max())
    i, j, k = best[np.argmax([acc[tuple(b)] for b in best])]
    return {"feasible": True, "target": target, "tau_p": float(pu[i]), "tau_2": float(e2[j]), "tau_3": float(e3[k]),
            "coverage": float(st[i, j, k] / len(p)), "accuracy": float(acc[i, j, k]), "n_accepted": int(st[i, j, k])}
