"""S2 analysis at k = 0 on `val` (never `test`): report window (O12), latent method (O4/O15), calibration (O2).

Every metric drops windows with `unlabelable_window_mask` (w0020, 0 post-fault samples). Bootstrap CIs resample whole
runs. Values marked (P) are still proposals (OPEN_ITEMS.md O12r).
Selection rule for the latent method (written before the comparison ran): highest kNN-vote accuracy on `val` at
k_nn = 10 over all unmasked windows; the paired run-bootstrap difference to the runner-up is reported; the
accuracy on post-fault windows (w >= 21) is reported next to it, and a disagreement between the two is flagged.
Run (CPU only): nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.s2_k0 [window|latent|calib|all]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from gridsense.config import TRACK
from gridsense.data.splits import cal_split, event_of, split_events, stratified_run_subsample, window_index
from gridsense.model.latent_repr import METHODS, fit_scaler_pca, represent, save_scaler_pca

LAT = TRACK / "eval" / "latents" / "seed42"
OUT = TRACK / "reports" / "s2"
N_BOOT = 2000          # (P) O12r
ECE_BINS = 15          # (P) O12r, equal-mass
K_NN = (5, 10, 20)
SELECT_K_NN = 10
CHUNK = 1024


def split_arrays(tag: str):
    from fault_localization.models.common import unlabelable_window_mask

    d = LAT / tag
    ids = (d / "window_ids.txt").read_text().split()
    keep = unlabelable_window_mask(ids).numpy().astype(bool)
    votes = np.load(d / "votes_k0_o0.npy")
    y = np.load(d / "labels.npy").astype(int)
    runs = np.array([event_of(w) for w in ids])
    w = np.array([window_index(i) for i in ids])
    return ids, keep, votes, y, runs, w


def _run_sums(values: np.ndarray, runs: np.ndarray):
    u, inv = np.unique(runs, return_inverse=True)
    return np.bincount(inv, weights=values.astype(float), minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)


def boot_ci(values: np.ndarray, runs: np.ndarray, seed: int = 0) -> list[float]:
    """95 % CI of a window-level mean, resampling runs."""
    s, c = _run_sums(values, runs)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(N_BOOT, len(s)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def paired_boot_diff(a: np.ndarray, b: np.ndarray, runs: np.ndarray, seed: int = 0) -> dict:
    """a - b on the same windows, run-bootstrap CI."""
    s, c = _run_sums(a.astype(float) - b.astype(float), runs)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(N_BOOT, len(s)))
    est = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {"diff": float(s.sum() / c.sum()), "ci": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}


def report_window(val) -> dict:
    _, keep, votes, y, runs, w = val
    correct = votes.argmax(1) == y
    acc = {int(i): float(correct[(w == i) & keep].mean()) for i in range(40) if ((w == i) & keep).any()}
    cand = {i: a for i, a in acc.items() if i >= 21}
    best = max(cand.values())
    chosen = min(i for i, a in cand.items() if a >= best - 0.01)
    sel = (w == chosen) & keep
    return {"rule": "earliest w>=21 with val accuracy within 1 pp of the max over w>=21", "accuracy_by_w": acc,
            "max_acc_w_ge_21": best, "argmax_w": max(cand, key=cand.get), "chosen_index": chosen,
            "chosen_acc": acc[chosen], "chosen_acc_ci": boot_ci(correct[sel], runs[sel]), "n_runs_at_index": int(sel.sum())}


def report_window_by_k(val) -> dict:
    """Same rule at every k (accuracy averaged over the 3 mask offsets for k > 0). The index is chosen at k = 0 only."""
    _, keep, _, y, _, w = val
    out: dict = {"rule": "earliest w>=21 within 1 pp of the max over w>=21, per k; informational except k=0"}
    for k in (0, 1, 2, 5, 10):
        offs = [0] if k == 0 else [0, 1, 2]
        correct = [np.load(LAT / "val" / f"votes_k{k}_o{o}.npy").argmax(1) == y for o in offs]
        acc = {i: float(np.mean([c[(w == i) & keep].mean() for c in correct])) for i in range(21, 40)}
        best = max(acc.values())
        out[f"k{k}"] = {"max_acc": best, "argmax_w": max(acc, key=acc.get), "chosen_index": min(i for i, a in acc.items() if a >= best - 0.01),
                        "acc_at_25": acc[25], "accuracy_by_w": acc}
    return out


def _knn_chunk(store_n: torch.Tensor, store_y: torch.Tensor, q: np.ndarray) -> dict[int, np.ndarray]:
    qq = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)), dim=1)
    sim, idx = (qq @ store_n.T).topk(max(K_NN), dim=1)
    lab = store_y[idx]
    out = {}
    for k in K_NN:
        counts = torch.zeros(len(qq), 26).scatter_add_(1, lab[:, :k], torch.ones_like(sim[:, :k]))
        counts += 1e-6 * torch.zeros(len(qq), 26).scatter_add_(1, lab[:, :k], sim[:, :k])  # tie -> larger similarity sum
        out[k] = counts.argmax(1).numpy()
    return out


def latent_comparison(val, run_fraction: float | None = None) -> dict:
    """(a)/(b)/(c) by kNN-vote accuracy on val queries; store = the 10 % train-run subsample at k = 0 (O15)."""
    _, keep, _, y, runs, w = val
    store_h = np.load(LAT / "train_sub10" / "h_k0_o0.npy")
    store_y = torch.from_numpy(np.load(LAT / "train_sub10" / "labels.npy").astype(np.int64))
    t0 = time.time()
    pca = fit_scaler_pca(store_h)                      # fitted on the train subsample only
    save_scaler_pca(pca, LAT / "scaler_pca256_train_sub10.npz")
    print(f"scaler+PCA fitted on {len(store_h)} train-subsample windows in {time.time() - t0:.0f}s; "
          f"explained variance of 256 dims: {pca['explained_variance_ratio'].sum():.3f}", flush=True)
    sel = np.flatnonzero(keep)
    if run_fraction is not None:
        chosen_runs = set(stratified_run_subsample(sorted(set(runs[sel])), run_fraction, seed=42))
        sel = sel[np.isin(runs[sel], list(chosen_runs))]
    q_h = np.load(LAT / "val" / "h_k0_o0.npy", mmap_mode="r")
    correct: dict[str, dict[int, np.ndarray]] = {}
    out: dict = {"store": "train_sub10 (k=0), fitted scaler/PCA on the store only", "store_windows": int(len(store_y)),
                 "queries": int(len(sel)), "val_run_fraction": run_fraction, "pca256_explained_variance": float(pca["explained_variance_ratio"].sum()),
                 "methods": {}}
    for m in METHODS:
        t0 = time.time()
        store_n = torch.nn.functional.normalize(torch.from_numpy(represent(store_h, m, pca)), dim=1)
        preds = {k: np.empty(len(sel), dtype=np.int64) for k in K_NN}
        for i in range(0, len(sel), CHUNK * 8):
            rep = represent(np.asarray(q_h[sel[i:i + CHUNK * 8]]), m, pca)
            for j in range(0, len(rep), CHUNK):
                for k, p in _knn_chunk(store_n, store_y, rep[j:j + CHUNK]).items():
                    preds[k][i + j:i + j + len(p)] = p
            if (i // (CHUNK * 8)) % 5 == 0:
                print(f"  {m} {i + len(rep)}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
        correct[m] = {k: preds[k] == y[sel] for k in K_NN}
        post = w[sel] >= 21
        out["methods"][m] = {"dim": int(store_n.shape[1]), "seconds": round(time.time() - t0, 1), **{
            f"k{k}": {"acc": float(correct[m][k].mean()), "ci": boot_ci(correct[m][k], runs[sel]),
                      "acc_post_fault_w_ge_21": float(correct[m][k][post].mean())} for k in K_NN}}
        print(m, {k: round(out["methods"][m][f"k{k}"]["acc"], 4) for k in K_NN}, flush=True)
        del store_n
    ranked = sorted(METHODS, key=lambda m: -out["methods"][m][f"k{SELECT_K_NN}"]["acc"])
    best, second = ranked[0], ranked[1]
    best_post = max(METHODS, key=lambda m: out["methods"][m][f"k{SELECT_K_NN}"]["acc_post_fault_w_ge_21"])
    out["selection"] = {"rule": f"highest all-window kNN-vote accuracy at k_nn={SELECT_K_NN}", "chosen": best, "runner_up": second,
                        "paired_diff_vs_runner_up": paired_boot_diff(correct[best][SELECT_K_NN], correct[second][SELECT_K_NN], runs[sel]),
                        "best_on_post_fault_windows": best_post, "post_fault_agrees": best_post == best}
    return out


def _ece(p: np.ndarray, c: np.ndarray) -> float:
    bins = np.array_split(np.argsort(p, kind="stable"), ECE_BINS)
    return float(sum(len(b) / len(p) * abs(c[b].mean() - p[b].mean()) for b in bins))


def calibration(val) -> dict:
    """Vote share = vote / 25. Platt and isotonic on the top-1 share fit on cal-A, compared on cal-B (O2). No logit temperature."""
    ids, keep, votes, y, runs, _ = val
    srt = np.sort(votes / 25.0, 1)
    top1, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    correct = (votes.argmax(1) == y).astype(int)
    a_runs, b_runs = (set(r) for r in cal_split(split_events("val"), seed=42))
    A, B = keep & np.isin(runs, list(a_runs)), keep & np.isin(runs, list(b_runs))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(top1[A, None], correct[A])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(top1[A], correct[A])
    probs = {"raw_top1_share": top1, "platt": platt.predict_proba(top1[:, None])[:, 1], "isotonic": iso.predict(top1)}
    res = {m: {"ece": _ece(p[B], correct[B]), "brier": float(np.mean((p[B] - correct[B]) ** 2)),
               "ece_ci": None} for m, p in probs.items()}
    best = min(("platt", "isotonic"), key=lambda m: (res[m]["ece"], res[m]["brier"]))
    post = B & (np.array([window_index(i) for i in ids]) >= 21)   # informational: pre-fault windows are trivial (share ~ 1.0)
    res_post = {m: {"ece": _ece(p[post], correct[post]), "brier": float(np.mean((p[post] - correct[post]) ** 2)),
                    "mean_prob": float(p[post].mean())} for m, p in probs.items()}
    np.savez_compressed(TRACK / "eval" / "calibration_k0_val.npz", top1_share=top1.astype(np.float32), margin=margin.astype(np.float32),
                        platt=probs["platt"].astype(np.float32), isotonic=probs["isotonic"].astype(np.float32),
                        correct=correct.astype(np.int8), keep=keep, part=np.where(A, 0, np.where(B, 1, -1)).astype(np.int8))
    return {"cal_a_windows": int(A.sum()), "cal_b_windows": int(B.sum()), "cal_a_runs": len(a_runs), "cal_b_runs": len(b_runs),
            "ece_bins": f"{ECE_BINS} equal-mass (P)", "results_on_cal_b": res, "results_on_cal_b_post_fault_w_ge_21": res_post, "post_fault_accuracy_cal_b": float(correct[post].mean()),
            "n_post_fault_cal_b": int(post.sum()), "lower_ece_on_cal_b": best,
            "platt_coef_intercept": [float(platt.coef_[0, 0]), float(platt.intercept_[0])],
            "top1_share_mean_cal_b": float(top1[B].mean()), "margin_mean_cal_b": float(margin[B].mean()),
            "margin_median_cal_b": float(np.median(margin[B])), "top1_accuracy_cal_b": float(correct[B].mean()),
            "argmax_unchanged": True, "note": "monotone maps never change the argmax; margin saved in eval/calibration_k0_val.npz"}


def main() -> None:
    torch.set_num_threads(4)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else None
    OUT.mkdir(parents=True, exist_ok=True)
    val = split_arrays("val")
    for name, fn in (("window", lambda: report_window(val)), ("window_k", lambda: report_window_by_k(val)), ("calib", lambda: calibration(val)),
                     ("latent", lambda: latent_comparison(val, frac))):
        if what in (name, "all"):
            res = fn()
            (OUT / f"s2_{name}_k0_val.json").write_text(json.dumps(res, indent=2) + "\n")
            print(name, "written", flush=True)


if __name__ == "__main__":
    main()
