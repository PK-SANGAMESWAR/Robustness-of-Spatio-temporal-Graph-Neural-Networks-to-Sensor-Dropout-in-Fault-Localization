"""Judge validation against the author's labels (O13; EVAL.md § 5; rag-eval.md). No API call. Run it after the author has filled `reports/s4/O13_labelling_sheet.csv`.

Rules fixed BEFORE any label exists (2026-11-20):
- faithfulness: the author gives 0 or 1 ("every claim of the report is supported by the event facts or the passages"). The judge score (0 to 1) counts as 1 when it is >= 0.9. Statistic: Cohen's kappa and % agreement.
- the five rubrics: author and judge both give integers 1 to 5. Statistic: linearly weighted Cohen's kappa, exact agreement and agreement within one point.
- 95 % interval: percentile bootstrap, 1,000 resamples, by EVENT (repeats of one event stay together), seed 0.
- A metric with kappa < 0.6, or with an undefined kappa (no variance in the labels or the scores), is INFORMATIONAL and the human labels are reported (EVAL.md § 5 item 4). Undefined is never shown as 0 or 1.
- An empty label cell stops the run: a missing label is never a score.
Not done here (costs judge calls): the position and verbosity probe of EVAL.md § 5 item 3.
`python -m gridsense.eval.judge_validation`
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date

import numpy as np

from gridsense.config import ROOT, TRACK

OUT = TRACK / "reports" / "s4"
LABELS = TRACK / "data" / "eval" / "golden_human_labels.jsonl"
RUBRICS = ("prediction_vs_confirmed", "completeness", "adherence", "evidence_use", "no_aggregate_metric_as_confidence")
FAITH_THRESHOLD = 0.9
KAPPA_BAR = 0.6
N_BOOT = 1000


def kappa(a: np.ndarray, b: np.ndarray, categories: list[int], weights: str = "none") -> float:
    """Cohen's kappa (weights none or linear). NaN when it is undefined (expected disagreement 0)."""
    n = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    o = np.zeros((n, n))
    for x, y in zip(a, b):
        o[idx[int(x)], idx[int(y)]] += 1
    o /= o.sum()
    e = np.outer(o.sum(1), o.sum(0))
    w = np.array([[0.0 if i == j else (1.0 if weights == "none" else abs(i - j) / (n - 1)) for j in range(n)] for i in range(n)])
    den = (w * e).sum()
    return float("nan") if den == 0 else float(1 - (w * o).sum() / den)


def _ci(a, b, groups, categories, weights):
    rng = np.random.default_rng(0)
    ug = sorted(set(groups))
    by = {g: np.where(np.array(groups) == g)[0] for g in ug}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        ix = np.concatenate([by[ug[i]] for i in pick])
        k = kappa(a[ix], b[ix], categories, weights)
        if not math.isnan(k):
            vals.append(k)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 50 else None


def validate(sheet_rows: list[dict], score_rows: list[dict]) -> dict:
    sc = {r["label_id"]: r for r in score_rows}
    res: dict = {"n_reports": len(sheet_rows), "rules": __doc__.split("Rules fixed BEFORE any label exists (2026-11-20):")[1].split("Not done here")[0].strip(), "metrics": {}}
    groups = [r["report_ref"].split("|")[1] for r in sheet_rows]
    cols = {"faithfulness": ("faithfulness_0_or_1", "judge_faithfulness", [0, 1], "none")}
    cols |= {r: (f"{r}_1_to_5", f"judge_{r}", [1, 2, 3, 4, 5], "linear") for r in RUBRICS}
    for name, (hc, jc, cats, w) in cols.items():
        h, j = [], []
        for r in sheet_rows:
            if str(r[hc]).strip() == "":
                raise ValueError(f"label missing: {r['label_id']} {hc} (a missing label is never a score)")
            hv = int(float(r[hc]))
            if hv not in cats:
                raise ValueError(f"{r['label_id']} {hc}: {hv} not in {cats}")
            jv = sc[r["label_id"]][jc]
            if jv in (None, ""):
                raise ValueError(f"judge score missing: {r['label_id']} {jc}")
            h.append(hv)
            j.append(int(float(jv) >= FAITH_THRESHOLD) if name == "faithfulness" else int(float(jv)))
        a, b = np.array(h), np.array(j)
        k = kappa(a, b, cats, w)
        status = "informational" if (math.isnan(k) or k < KAPPA_BAR) else "usable"
        res["metrics"][name] = {"kappa": None if math.isnan(k) else k, "kappa_undefined_no_variance": math.isnan(k), "kappa_type": "cohen" if w == "none" else "linear-weighted", "kappa_ci95_by_event": _ci(a, b, groups, cats, w),
                                "agreement_exact": float((a == b).mean()), "agreement_within_1": float((abs(a - b) <= 1).mean()) if w == "linear" else None,
                                "author_distribution": {str(c): int((a == c).sum()) for c in cats}, "judge_distribution": {str(c): int((b == c).sum()) for c in cats}, "status": status,
                                "confusion_author_rows_judge_cols": [[int(((a == r_) & (b == c_)).sum()) for c_ in cats] for r_ in cats]}
    return res


def to_labels(sheet_rows: list[dict]) -> list[dict]:
    today = date.today().isoformat()
    return [{"label_id": r["label_id"], "report_ref": r["report_ref"], "faithfulness": int(float(r["faithfulness_0_or_1"])), "rubric_scores": {k: int(float(r[f"{k}_1_to_5"])) for k in RUBRICS},
             "labeller": "author", "labelled": today, "notes": r.get("notes", "")} for r in sheet_rows]


def markdown(res: dict) -> str:
    L = ["# Judge validation (O13)", "", f"Reports labelled: {res['n_reports']}. Rules (fixed before the labels existed):", "", res["rules"], "",
         "| metric | kappa (95 % CI by event) | exact agreement | within 1 | status |", "|---|---|---|---|---|"]
    for m, v in res["metrics"].items():
        k = "undefined (no variance)" if v["kappa_undefined_no_variance"] else f"{v['kappa']:.2f}" + (f" ({v['kappa_ci95_by_event'][0]:.2f} to {v['kappa_ci95_by_event'][1]:.2f})" if v["kappa_ci95_by_event"] else "")
        within = "–" if v["agreement_within_1"] is None else f"{v['agreement_within_1']:.2f}"
        L.append(f"| {m} | {k} | {v['agreement_exact']:.2f} | {within} | {v['status']} |")
    L += ["", "A metric that is `informational` is reported with the human labels, not as a gate. The position and verbosity probe (EVAL.md § 5 item 3) is not part of this run."]
    return "\n".join(L) + "\n"


def main() -> None:
    sheet = list(csv.DictReader((OUT / "O13_labelling_sheet.csv").open(encoding="utf-8")))
    scores = list(csv.DictReader((OUT / "O13_judge_scores_DO_NOT_OPEN_BEFORE_LABELLING.csv").open(encoding="utf-8")))
    res = validate(sheet, scores)
    LABELS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in to_labels(sheet)), encoding="utf-8")
    (OUT / "JUDGE_VALIDATION.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (OUT / "JUDGE_VALIDATION.md").write_text(markdown(res), encoding="utf-8")
    print(json.dumps({"labels": str(LABELS.relative_to(ROOT)), "report": "rag-for-dropout/reports/s4/JUDGE_VALIDATION.md", "status": {m: v["status"] for m, v in res["metrics"].items()}}, indent=1))


if __name__ == "__main__":
    main()
