"""Identity gate (author, S2): the pipeline uses its own CPU predictions; the paper predictions are only a check.

Tolerance rule (author, 2026-10-28; set AFTER the data was seen, see rag-for-dropout/dropout-rag-intro.md § 6):
a CPU prediction may differ from the paper prediction only where the vote gap is below 2e-3, with
gap = vote(CPU class) - vote(paper class) >= 0 (votes sum to 25). Any other difference stops S2.

k = 0 test: per window against results/.../predictions_test.csv (the only per-window paper predictions on disk), and
macro-F1 against metrics.json test_graph.f1. k = 0 val: macro-F1 against best_validation_graph_f1.
k > 0: no per-window paper predictions exist; macro-F1 per training seed (mean over the 3 mask offsets) is compared
with results/dropout_study_aware/degradation_curves.csv with a tolerance of 1e-3 (author, 2026-10-28).
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from gridsense.config import ROOT, TRACK
from gridsense.model.predictor import seed_dir

LAT = TRACK / "eval" / "latents"
LABELS = list(range(26))
GAP_TOL = 2e-3       # author rule for per-window differences
F1_TOL_K = 1e-3      # author tolerance for k > 0 macro-F1
CURVES = ROOT / "results" / "dropout_study_aware" / "degradation_curves.csv"


def tolerance_check(votes: np.ndarray, paper_pred: np.ndarray, window_ids: list[str], gap_tol: float = GAP_TOL) -> dict:
    """Windows where argmax(votes) != paper_pred, their gaps, and which of them break the rule."""
    pred = votes.argmax(1)
    diff = np.flatnonzero(pred != paper_pred)
    rows = []
    for i in diff:
        gap = float(votes[i, pred[i]] - votes[i, paper_pred[i]])
        rows.append({"window_id": window_ids[i], "cpu_class": int(pred[i]), "paper_class": int(paper_pred[i]),
                     "gap": gap, "within_rule": gap < gap_tol})
    return {"n_windows": int(len(pred)), "n_diff": len(rows), "max_gap": max((r["gap"] for r in rows), default=0.0),
            "violations": [r for r in rows if not r["within_rule"]], "differences": rows,
            "rule_ok": all(r["within_rule"] for r in rows)}


def _load(seed: int, tag: str, k: int = 0, off: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    d = LAT / f"seed{seed}" / tag
    ids = (d / "window_ids.txt").read_text().split()
    votes = np.load(d / f"votes_k{k}_o{off}.npy")
    return ids, votes, np.load(d / "labels.npy").astype(int)


def _f1(y: np.ndarray, votes: np.ndarray) -> float:
    return float(f1_score(y, votes.argmax(1), labels=LABELS, average="macro", zero_division=0))


def check_k0(seed: int = 42) -> dict:
    metrics = json.loads((seed_dir(seed) / "metrics.json").read_text())
    ids, votes, y = _load(seed, "test")
    ref = pd.read_csv(seed_dir(seed) / "predictions_test.csv", usecols=["window_id", "truth", "vote_prediction"])
    ref = ref.set_index("window_id").loc[ids]
    out: dict = {"seed": seed, "rule": f"differ only where vote gap < {GAP_TOL:g}; set after the data was seen"}
    out["test"] = tolerance_check(votes, ref["vote_prediction"].to_numpy(), ids)
    out["test"]["label_mismatch"] = int((y != ref["truth"].to_numpy()).sum())
    out["test"]["f1_cpu"], out["test"]["f1_paper"] = _f1(y, votes), float(metrics["test_graph"]["f1"])
    out["test"]["f1_diff"] = out["test"]["f1_cpu"] - out["test"]["f1_paper"]
    _, votes_v, y_v = _load(seed, "val")
    out["val"] = {"n_windows": int(len(y_v)), "f1_cpu": _f1(y_v, votes_v), "f1_paper": float(metrics["best_validation_graph_f1"])}
    out["val"]["f1_diff"] = out["val"]["f1_cpu"] - out["val"]["f1_paper"]
    out["passes"] = (out["test"]["rule_ok"] and out["test"]["label_mismatch"] == 0 and abs(out["val"]["f1_diff"]) < 1e-4
                     and abs(out["test"]["f1_diff"]) < 1e-4)
    return out


def check_curve(seed: int = 42, tag: str = "test", k_levels=(1, 2, 5, 10), n_offsets: int = 3, tol: float = F1_TOL_K) -> dict:
    rows = {int(r["k"]): json.loads(r["f1_per_train_seed"]) for r in csv.DictReader(CURVES.open()) if r["model"] == "rgsage_max"}
    pos = [42, 43, 44].index(seed)
    out: dict = {"tolerance": tol, "per_k": {}}
    for k in k_levels:
        f1s = [_f1(_load(seed, tag, k, o)[2], _load(seed, tag, k, o)[1]) for o in range(n_offsets)]
        mean = float(np.mean(f1s))
        out["per_k"][k] = {"f1_cpu_mean": mean, "f1_per_offset": f1s, "paper": rows[k][pos], "abs_diff": abs(mean - rows[k][pos])}
        out["per_k"][k]["within_tol"] = out["per_k"][k]["abs_diff"] <= tol
    out["passes"] = all(v["within_tol"] for v in out["per_k"].values())
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "k0"
    (TRACK / "reports" / "s2").mkdir(parents=True, exist_ok=True)
    res = check_k0(42) if what == "k0" else check_curve(42)
    (TRACK / "reports" / "s2" / f"identity_{what}_seed42.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["passes"] else 1)
