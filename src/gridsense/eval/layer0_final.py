"""Layer 0 evaluation of a gate on a set of post-fault windows (O24, O23, O18; author 2026-10-30). Used for the val check (cal-A / cal-B),
the single frozen run on `test`, and seeds 43 / 44. Post-fault windows (w >= 21), mask offset 0, k in {0, 1, 2, 5, 10}; neighbours from the FULL
train store of the same seed; held-out version primary (O20), standard version secondary only. Bootstrap by run: 1000 resamples, seed 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridsense.data.splits import event_of
from gridsense.eval import metrics as M
from gridsense.eval.knn_features import KS, K_NN, knn_path, lat_dir, post_fault_rows
from gridsense.verification.gate import CONFIDENT, LOW_CONFIDENCE, REJECT, Gate, vote_features

N_BOOT, BOOT_SEED, KNN = 1000, 0, 10


@dataclass
class KData:
    votes: np.ndarray               # [n, 26] soft vote of the post-fault windows
    y: np.ndarray                   # true class
    knn: dict                       # arrays from knn_features (held-out / standard)


@dataclass
class SplitData:
    seed: int
    split: str
    window_ids: list[str]
    runs: np.ndarray
    by_k: dict[int, KData]
    macro_f1_all: dict[int, float]  # STGNN-only, all windows without w0020 (needs the split's complete votes)


def load_split(seed: int, split: str, ks=KS, with_knn: bool = True) -> SplitData:
    from sklearn.metrics import f1_score

    from fault_localization.models.common import unlabelable_window_mask

    d = lat_dir(seed) / split
    ids = (d / "window_ids.txt").read_text().split()
    rows = post_fault_rows(ids)
    y_all = np.load(d / "labels.npy").astype(int)
    keep = unlabelable_window_mask(ids).numpy().astype(bool)
    by_k, f1 = {}, {}
    for k in ks:
        votes_all = np.load(d / f"votes_k{k}_o0.npy")
        f1[k] = float(f1_score(y_all[keep], votes_all[keep].argmax(1), labels=list(range(26)), average="macro", zero_division=0))
        knn = {}
        if with_knn:
            z = np.load(knn_path(seed, split, k))
            assert (z["rows"] == rows).all()
            knn = {n: z[n] for n in z.files if n != "rows"}
        by_k[k] = KData(votes_all[rows], y_all[rows], knn)
    return SplitData(seed, split, [ids[r] for r in rows], np.array([event_of(ids[r]) for r in rows]), by_k, f1)


def _counts(n_runs: int) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED)
    c = np.zeros((N_BOOT, n_runs), dtype=np.int16)
    for i in range(N_BOOT):
        c[i] = np.bincount(rng.integers(0, n_runs, n_runs), minlength=n_runs)
    return c


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _lr_error_score(lr: dict | None, p: np.ndarray, margin: np.ndarray, agree: np.ndarray) -> np.ndarray | None:
    if lr is None:
        return None
    return -(np.c_[p, margin, agree] @ np.array(lr["coef"]) + lr["intercept"])     # higher = more likely wrong


def evaluate(gate: Gate, data: SplitData, subset: np.ndarray | None = None, boot: bool = True) -> dict:
    """Metrics per k on the windows where `subset` is True (all windows when None)."""
    n_all = len(data.runs)
    subset = np.ones(n_all, dtype=bool) if subset is None else subset
    runs = data.runs[subset]
    u, inv = np.unique(runs, return_inverse=True)
    counts = _counts(len(u)) if boot else None
    out: dict = {"n_windows": int(subset.sum()), "n_runs": int(len(u)), "per_k": {}}
    for k, c in data.by_k.items():
        votes, y = c.votes[subset], c.y[subset]
        correct = votes.argmax(1) == y
        top1, margin = vote_features(votes)
        res: dict = {"stgnn_accuracy": float(correct.mean()), "macro_f1_all_windows_val_or_test": data.macro_f1_all[k]}
        if k not in gate.per_k:
            res["gate"] = "REJECT: UNSUPPORTED_DROPOUT_PATTERN"
            out["per_k"][str(k)] = res
            continue
        p = gate.calibrated(k, top1)
        agree = c.knn["heldout_agree_10"][subset]
        res["calibration"] = {"ece_isotonic": M.ece_equal_mass(p, correct), "brier_isotonic": M.brier(p, correct), "mean_p": float(p.mean()),
                              "ece_raw_vote_share": M.ece_equal_mass(top1, correct)}
        res["selective"] = {"aurc_isotonic": M.aurc(p, correct), "risk_at_coverage_0.8": M.risk_at_coverage(p, correct, 0.8)[0]}
        res["knn_vote_accuracy"] = {s: {str(kk): float((c.knn[f"{s}_vote_{kk}"][subset] == y).mean()) for kk in K_NN} for s in ("heldout", "standard")}
        lr = gate.per_k[k].get("lr")
        sig = {"raw_vote_share": -top1, "isotonic_prob": -p, "margin": -margin, "knn_agreement_heldout": -agree,
               "neighbour_error_rate_heldout": c.knn[f"heldout_nbrerr_{KNN}"][subset], "knn_agreement_standard (secondary)": -c.knn[f"standard_agree_{KNN}"][subset]}
        lrs = _lr_error_score(lr, p, margin, agree)
        if lrs is not None:
            sig["logistic_combination_p+margin+agreement"] = lrs
        ed = {}
        for name, sc in sig.items():
            tg = M.TieGroups(sc, ~correct)
            ed[name] = {"auroc": tg.auroc(), "auprc": M.auprc(sc, ~correct)}
            if boot:
                ed[name]["auroc_ci"] = _ci(np.array([tg.auroc(counts[i][inv].astype(float)) for i in range(N_BOOT)]))
        res["error_detection"] = ed
        for op in ("acc95", "acc98"):
            dec = gate.decide(k, votes, agree, operating_point=op)
            conf = dec["outcome"] == CONFIDENT
            g = M.gate_stats(conf, correct)
            g["outcomes"] = {n: int((dec["outcome"] == v).sum()) for n, v in (("CONFIDENT", CONFIDENT), ("LOW_CONFIDENCE", LOW_CONFIDENCE), ("REJECT", REJECT))}
            if boot:
                cw = counts.astype(float)
                n_run, c_run, cc_run = (np.bincount(inv, weights=x.astype(float), minlength=len(u)) for x in (np.ones(len(conf)), conf, conf & correct))
                g["coverage_ci"] = _ci((cw @ c_run) / (cw @ n_run))
                g["accuracy_confident_ci"] = _ci((cw @ cc_run) / np.maximum(cw @ c_run, 1))
            res[f"gate_{op}"] = g
            if op == "acc95":
                sets = dec["set"]
                covered, size = sets[np.arange(len(y)), y], sets.sum(1)
                low = dec["outcome"] == LOW_CONFIDENCE
                res["conformal_90"] = {"coverage_all_post_fault": float(covered.mean()), "mean_size_all_post_fault": float(size.mean()),
                                       "coverage_low_confidence": float(covered[low].mean()) if low.any() else None,
                                       "mean_size_low_confidence": float(size[low].mean()) if low.any() else None, "n_low_confidence": int(low.sum())}
                if boot:
                    cov_run = np.bincount(inv, weights=covered.astype(float), minlength=len(u))
                    res["conformal_90"]["coverage_all_ci"] = _ci((counts.astype(float) @ cov_run) / (counts.astype(float) @ np.bincount(inv, minlength=len(u)).astype(float)))
        out["per_k"][str(k)] = res
    return out
