"""Layer 1 deterministic retrieval metrics (EVAL.md § 3; rag-eval.md § 4 Layer 1). Free: no LLM, no API.

Per query, with passages ranked 1..n and golden contexts with grades 1-3 (matching: gridsense.eval.golden.matches):
  recall@k            1 if at least one golden context is inside a top-k passage (hit rate), else 0
  context_coverage@k  fraction of the golden contexts found in the top k
  precision@k         fraction of the top-k passages that contain at least one golden context
  mrr@10              1 / rank of the first relevant passage within the top 10 (0 if none)
  ndcg@k              gain 2^grade - 1, each golden context credited once at its first matching rank; ideal = grades sorted
  identifier_recall@k 1 if every critical term (normalized) occurs in the top-k passages (exact-identifier queries)
Sufficiency: "sufficient" iff dense_top1 >= tau. tau is chosen on dev to maximize balanced accuracy (answerable vs not).
Confidence intervals: percentile bootstrap over queries.
"""
from __future__ import annotations

import math

import numpy as np

from gridsense.eval.golden import matches, norm

KS = (1, 3, 5, 8, 10)


def query_metrics(passages: list[dict], item: dict, ks: tuple[int, ...] = KS) -> dict:
    ctxs = item["golden_contexts"]
    rel = [[j for j, c in enumerate(ctxs) if matches(p["document_id"], p["text"], c)] for p in passages]
    out: dict[str, float] = {}
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    out["mrr@10"] = 1.0 / first if first and first <= 10 else 0.0
    for k in ks:
        top = rel[:k]
        found = {j for r in top for j in r}
        out[f"recall@{k}"] = float(bool(found))
        out[f"context_coverage@{k}"] = len(found) / len(ctxs)
        out[f"precision@{k}"] = sum(bool(r) for r in top) / k
        credited, dcg = set(), 0.0
        for i, r in enumerate(top):
            new = [j for j in r if j not in credited]
            credited.update(new)
            dcg += sum(2 ** ctxs[j]["grade"] - 1 for j in new) / math.log2(i + 2)
        ideal = sorted((c["grade"] for c in ctxs), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
        if item.get("critical_terms"):
            blob = norm(" ".join(p["text"] for p in passages[:k]))
            out[f"identifier_recall@{k}"] = float(all(norm(t) in blob for t in item["critical_terms"]))
    return out


def bootstrap_ci(values: list[float], n: int = 2000, level: float = 0.95, seed: int = 42) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def aggregate(rows: list[dict], key: str, n_boot: int = 2000) -> dict:
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return {"n": 0}
    lo, hi = bootstrap_ci(vals, n_boot)
    return {"mean": float(np.mean(vals)), "ci_low": lo, "ci_high": hi, "n": len(vals)}


def choose_sufficiency_threshold(top1: list[float], answerable: list[bool]) -> dict:
    """tau maximizing balanced accuracy of (top1 >= tau) vs answerable; ties: the lowest such tau. Fit on dev only."""
    s, y = np.asarray(top1, float), np.asarray(answerable, bool)
    if y.all() or (~y).all():
        return {"tau": None, "reason": "dev needs answerable and unanswerable queries"}
    best = (-1.0, None)
    for t in np.unique(s):
        pred = s >= t
        ba = 0.5 * (pred[y].mean() + (~pred[~y]).mean())
        if ba > best[0] + 1e-12:
            best = (float(ba), float(t))
    return {"tau": best[1], "balanced_accuracy_dev": best[0]}


def sufficiency_metrics(top1: list[float], answerable: list[bool], tau: float | None) -> dict:
    if tau is None:
        return {"n": 0}
    s, y = np.asarray(top1, float), np.asarray(answerable, bool)
    pred = s >= tau
    return {"unanswerable_flagged_insufficient": float((~pred[~y]).mean()) if (~y).any() else float("nan"),
            "answerable_flagged_insufficient": float((~pred[y]).mean()) if y.any() else float("nan"),
            "n_unanswerable": int((~y).sum()), "n_answerable": int(y.sum())}
