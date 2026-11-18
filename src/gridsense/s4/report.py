"""S4 aggregation (EVAL.md § 11, author 2026-11-03): the PRIMARY result is the per-stratum table, CONFIDENT-wrong (`high_conf_error`) first.
A weighted figure is secondary and is always labelled "over the covered strata". An unweighted mean over the 30 events is stored only as `informational_unweighted` and never printed as a score.
Weights: the share of each stratum among the test runs at the same k (`reports/events/stratum_shares.json`), renormalised over the three strata; then the mean over the k levels (equal weight).
A metric that failed for an event is left out of that metric's mean and counted (n shown); it is never scored 0 or 1.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from gridsense.config import TRACK

STRATA = ("high_conf_error", "confident_correct", "low_confidence")      # report order: CONFIDENT-wrong first
CHECK_KEYS = ("schema", "field_consistency", "neighbour_counts", "citation_validity", "no_confirmation", "no_control_command", "no_approval_claim", "no_injection_echo", "evidence_consistency")
JUDGE_KEYS = ("faithfulness", "answer_relevancy", "contextual_relevancy", "geval_prediction_vs_confirmed", "geval_completeness", "geval_adherence", "geval_evidence_use", "geval_no_aggregate_metric_as_confidence")


def shares() -> dict:
    return json.loads((TRACK / "reports" / "events" / "stratum_shares.json").read_text())


def metric_values(item: dict) -> dict[str, float | None]:
    """One number per metric for one item (None = not measured: a failure or a skipped step)."""
    out: dict[str, float | None] = {}
    chk = item.get("checks") or {}
    for k in CHECK_KEYS:
        out[f"check.{k}"] = (1.0 if chk[k]["pass"] else 0.0) if k in chk else None
    j = (item.get("judge") or {}).get("scores") or {}
    for k in JUDGE_KEYS:
        v = j.get(k)
        out[f"judge.{k}"] = None if v is None else (v["raw_1_to_5"] if "raw_1_to_5" in v else v["score"])
    out["evidence.final_insufficient"] = None if "evidence" not in item else (1.0 if item["evidence"]["final"] == "insufficient" else 0.0)
    out["evidence.generator_review_while_insufficient"] = None if "evidence" not in item or "generator_review_while_insufficient" not in item["evidence"] else float(item["evidence"]["generator_review_while_insufficient"])
    return out


def per_stratum(items: list[dict]) -> dict:
    """{stratum: {n, metrics: {name: {mean, n}}}} over the items given (all k together) and per k."""
    res: dict = {"all_k": {}, "per_k": {}}
    for s in STRATA:
        sel = [i for i in items if i["stratum"] == s]
        res["all_k"][s] = _summ(sel)
        for k in sorted({i["k"] for i in items}):
            res["per_k"].setdefault(str(k), {})[s] = _summ([i for i in sel if i["k"] == k])
    return res


def _summ(sel: list[dict]) -> dict:
    vals: dict[str, list[float]] = defaultdict(list)
    for it in sel:
        for k, v in metric_values(it).items():
            if v is not None:
                vals[k].append(v)
    return {"n": len(sel), "metrics": {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in vals.items()}}


def weighted_over_covered(items: list[dict]) -> dict:
    """Secondary figure. Per k: sum over strata of (share / sum of covered shares) x stratum mean; then the mean over the k that have every stratum."""
    sh, out, per_k = shares(), {}, defaultdict(dict)
    ks = sorted({i["k"] for i in items})
    names = sorted({m for i in items for m in metric_values(i)})
    for name in names:
        ks_ok = []
        for k in ks:
            by = {s: [metric_values(i)[name] for i in items if i["k"] == k and i["stratum"] == s and metric_values(i)[name] is not None] for s in STRATA}
            if all(by[s] for s in STRATA):
                w = {s: sh[str(k)][s] for s in STRATA}
                tot = sum(w.values())
                per_k[name][k] = sum(w[s] / tot * float(np.mean(by[s])) for s in STRATA)
                ks_ok.append(k)
        if ks_ok:
            out[name] = {"value": float(np.mean([per_k[name][k] for k in ks_ok])), "k_levels": ks_ok}
    uncovered = {str(k): sh[str(k)]["not_in_any_stratum_confident_wrong_p_below_0.9"] for k in ks}
    return {"label": "over the covered strata", "uncovered_share_per_k": uncovered, "metrics": out}


def informational_unweighted(items: list[dict]) -> dict:
    v: dict[str, list[float]] = defaultdict(list)
    for it in items:
        for k, x in metric_values(it).items():
            if x is not None:
                v[k].append(x)
    return {k: {"mean": float(np.mean(x)), "n": len(x), "note": "INFORMATIONAL ONLY: the 30 events over-sample wrong predictions; not a system score"} for k, x in v.items()}


def markdown(items: list[dict], title: str) -> str:
    ps, w = per_stratum(items), weighted_over_covered(items)
    names = [k for k in (*(f"check.{c}" for c in CHECK_KEYS), *(f"judge.{j}" for j in JUDGE_KEYS), "evidence.final_insufficient", "evidence.generator_review_while_insufficient") if any(k in ps["all_k"][s]["metrics"] for s in STRATA)]
    L = [f"# {title}", "", "Primary result: per stratum, CONFIDENT-wrong (`high_conf_error`) first. Checks: share of events that pass (1 = all pass). Judge: GEval rubrics on a 1 to 5 scale, DeepEval metrics 0 to 1. "
         "n counts the events measured for that metric (a failed metric is left out, never scored 0).", "",
         "| metric | " + " | ".join(f"{s} (n={ps['all_k'][s]['n']})" for s in STRATA) + " |", "|---|" + "---|" * len(STRATA)]
    for n in names:
        cells = []
        for s in STRATA:
            m = ps["all_k"][s]["metrics"].get(n)
            cells.append("–" if m is None else f"{m['mean']:.2f} (n={m['n']})")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += ["", "## Secondary: weighted figure, over the covered strata", "Weights: share of the stratum among the test runs at the same k (`reports/events/stratum_shares.json`), renormalised over the three strata; mean over k. "
          "Share of runs in no stratum (CONFIDENT, wrong, p < 0.9) per k: " + ", ".join(f"k={k} {v:.3f}" for k, v in w["uncovered_share_per_k"].items()) + ".", "",
          "| metric (over the covered strata) | value |", "|---|---|"]
    L += [f"| {n} | {w['metrics'][n]['value']:.2f} |" for n in names if n in w["metrics"]]
    return "\n".join(L) + "\n"
