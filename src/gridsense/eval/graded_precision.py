"""Graded precision@3 and nDCG@3 from the author's labels (reports/layer1/labelling_table.csv, column `label`).

    PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.graded_precision [--csv reports/layer1/labelling_table.csv]

Labels: "relevant" (weight 1, gain 2), "partly relevant" (0.5, gain 1), "not relevant" (0, gain 0); the short forms r / p / n are accepted. A row without a label stops the run.
  graded P@3   mean over the answerable queries of (sum of the weights of the 3 passages) / 3
  strict P@3   the same with "relevant" only
  nDCG@3       DCG of the 3 passages in the retrieved order divided by the DCG of the same 3 labels in the best order (gain = 2^grade - 1, grade 2, 1, 0; discount 1/log2(rank+1));
               a query whose 3 labels are all "not relevant" has no ideal order: it counts as 0 and is listed
Unanswerable queries are not scored: the number of their passages labelled relevant or partly relevant is reported (it should be 0; otherwise the corpus answers them in part).
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from gridsense.config import TRACK

ALIAS = {"relevant": 2, "r": 2, "partly relevant": 1, "partly": 1, "p": 1, "not relevant": 0, "not": 0, "n": 0}
WEIGHT = {2: 1.0, 1: 0.5, 0: 0.0}
GAIN = {2: 3.0, 1: 1.0, 0: 0.0}


def grade(label: str) -> int:
    k = label.strip().lower()
    if k not in ALIAS:
        raise ValueError(f"unknown label {label!r}; use relevant, partly relevant or not relevant")
    return ALIAS[k]


def ndcg(grades: list[int]) -> float:
    dcg = lambda g: sum(GAIN[x] / math.log2(i + 2) for i, x in enumerate(g))
    ideal = dcg(sorted(grades, reverse=True))
    return dcg(grades) / ideal if ideal > 0 else 0.0


def compute(rows: list[dict]) -> dict:
    missing = [(r["query_id"], r["rank"]) for r in rows if not r["label"].strip()]
    if missing:
        raise SystemExit(f"{len(missing)} passages have no label (first: {missing[:3]})")
    by: dict[tuple[str, str], dict] = defaultdict(dict)
    for r in rows:
        by[(r["split"], r["query_id"])][int(r["rank"])] = (grade(r["label"]), r["answerable"] == "yes")
    out: dict = {}
    for split in sorted({s for s, _ in by}):
        ans = {q: [g for _, (g, _a) in sorted(v.items())] for (s, q), v in by.items() if s == split and next(iter(v.values()))[1]}
        un = {q: [g for _, (g, _a) in sorted(v.items())] for (s, q), v in by.items() if s == split and not next(iter(v.values()))[1]}
        gp = [sum(WEIGHT[g] for g in gs) / len(gs) for gs in ans.values()]
        sp = [sum(g == 2 for g in gs) / len(gs) for gs in ans.values()]
        nd = [ndcg(gs) for gs in ans.values()]
        out[split] = {"answerable_queries": len(ans), "graded_p@3": float(np.mean(gp)), "strict_p@3": float(np.mean(sp)), "ndcg@3": float(np.mean(nd)),
                      "queries_without_any_relevant_or_partly": sorted(q for q, gs in ans.items() if max(gs) == 0),
                      "unanswerable_queries": len(un), "unanswerable_passages_labelled_relevant_or_partly": int(sum(g > 0 for gs in un.values() for g in gs)),
                      "unanswerable_queries_with_such_passages": sorted(q for q, gs in un.items() if max(gs) > 0)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=str(TRACK / "reports" / "layer1" / "labelling_table.csv")); a = ap.parse_args()
    with open(a.csv, encoding="utf-8", newline="") as f:
        res = compute(list(csv.DictReader(f)))
    for split, r in res.items():
        print(f"{split}: {r['answerable_queries']} answerable queries | graded P@3 {r['graded_p@3']:.3f} | strict P@3 {r['strict_p@3']:.3f} | nDCG@3 {r['ndcg@3']:.3f} | no relevant or partly: {r['queries_without_any_relevant_or_partly']} "
              f"| unanswerable queries with relevant or partly passages: {r['unanswerable_queries_with_such_passages']}")
