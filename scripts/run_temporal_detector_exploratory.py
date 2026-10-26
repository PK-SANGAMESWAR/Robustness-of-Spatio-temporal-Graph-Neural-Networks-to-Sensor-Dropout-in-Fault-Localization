"""EXPLORATORY, NOT PRE-REGISTERED. Temporal-variation detector vs the residual detector (CPU only).

Detector score per sensor = - (standard deviation of its readings over the 20-step window), averaged over the
sensor's instrumented phases, in p.u. A low variation gives a high score. Evaluated on the S5.1 corrupted windows
at sigma0 = 1e-2 (same noise draw and same corruption draw as the S5.1 run; the regenerated labels are asserted
equal to the stored ones), all 5 corruption types, s = 1, 2, 3, **half B** of the G-TF event split.
AUROC: temporal alone, residual alone (stored in-sample PSM score), and combined = max(rank-normalized residual,
rank-normalized temporal). Rank normalization is within the evaluated population of one condition (every available
sensor of every half-B window), as a percentile in (0, 1].

Nothing here enters a decision rule. Results are written to ``results/stage_f/temporal_detector/``.
Run with a low priority and few threads:  nice -n 19 ionice -c 3 python scripts/run_temporal_detector_exploratory.py --threads 2
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, by construction
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve

import run_gtf as G
import run_silent_detection as S1
from fault_localization.dropout.silent import apply_silent_failure_batch
from fault_localization.physics import observation as OBS

SIGMA = 0.01
OUT = REPO / "results/stage_f/temporal_detector"
LABEL = "EXPLORATORY, NOT PRE-REGISTERED. Not used in any decision rule."


def temporal_score(x: torch.Tensor, mean, std) -> np.ndarray:
    """[B,25] score = - mean over instrumented phases of std_t(voltage in p.u.). x: [B,N,S,6] normalized."""
    mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32)); std_t = torch.as_tensor(np.asarray(std, dtype=np.float32))
    volts = x[..., :3] * std_t + mean_t                         # [B,N,S,3] p.u.
    sd = volts.std(dim=2, unbiased=False)                      # [B,N,3]
    ok = (x[:, :, 0, 3:] > 0.5).float()                        # [B,N,3] static instrumentation mask
    n_ok = ok.sum(-1).clamp(min=1)
    return (-(sd * ok).sum(-1) / n_ok).numpy()


def rank01(v: np.ndarray) -> np.ndarray:
    return rankdata(v, method="average") / v.size


def tpr_at_fpr(y, sc, fpr_t=0.01):
    fpr, tpr, _ = roc_curve(y, sc)
    return float(np.interp(fpr_t, fpr, tpr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    data_root = REPO / "dataset_fullscale_50"
    ds, loader, mean, std, table, a_events = G._setup(data_root, args.threads)
    raw = S1.open_raw(G.raw_dir(SIGMA), 0, "r")
    conds = S1.conditions()
    n_total = len(ds) if args.max_batches is None else sum(len(i) for _, i in zip(range(args.max_batches), loader.batch_sampler))
    acc = {c[0]: {"temp": [], "res_ins": [], "res_loo": [], "lab": [], "avail": []} for c in conds}
    t0, pos = time.time(), 0
    for bi, batch in enumerate(loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        x = batch["x"]; b = x.shape[0]; sl = slice(pos, pos + b)
        rows_b = np.array([G.event_of(w) not in a_events for w in batch["window_id"]])
        if rows_b.any():
            xn = G._noisy(x, SIGMA, std, bi)
            avail = (OBS.observe(xn, mean, std, reduce="last")[1].sum(dim=-1) > 0).numpy()
            for cname, kind, sev, n_s in conds:
                x_c, corrupted = apply_silent_failure_batch(xn, n_s, kind, seed=S1.CORRUPTION_SEED * 10000 + bi,
                                                            norm_stats=(mean, std), severity=sev)
                lab = np.zeros((b, 25), bool)
                for r, nodes in enumerate(corrupted):
                    lab[r, nodes] = True
                if not np.array_equal(lab, np.asarray(raw[f"{cname}__label"][sl]).astype(bool)):
                    raise RuntimeError(f"{cname}: regenerated labels differ from the stored S5.1 labels")
                d = acc[cname]
                d["temp"].append(temporal_score(x_c[torch.as_tensor(rows_b)], mean, std))
                d["res_ins"].append(np.asarray(raw[f"{cname}__psm_ins"][sl])[rows_b])
                d["res_loo"].append(np.asarray(raw[f"{cname}__psm_loo"][sl])[rows_b])
                d["lab"].append(lab[rows_b]); d["avail"].append(avail[rows_b])
        pos += b
        if bi % 100 == 0:
            print(f"  batch {bi} {(time.time()-t0)/60:.1f} min", flush=True)

    results = {"label": LABEL, "sigma0": SIGMA, "half": "B", "split_seed": G.SPLIT_SEED, "partial_run": args.max_batches is not None,
               "temporal_score": "-mean over instrumented phases of std over the 20 window steps, p.u.",
               "combined": "max(rank-normalized residual (in-sample PSM), rank-normalized temporal)",
               "rank_normalization": "percentile within all available sensors of all half-B windows of the condition",
               "conditions": {}}
    for cname, *_ in conds:
        d = acc[cname]
        av = np.concatenate(d["avail"]); lab = np.concatenate(d["lab"])[av].astype(int)
        t = np.concatenate(d["temp"])[av]; ri = np.concatenate(d["res_ins"])[av]; rl = np.concatenate(d["res_loo"])[av]
        comb = np.maximum(rank01(ri), rank01(t))
        row = {"n_sensor_scores": int(av.sum()), "n_positive": int(lab.sum()),
               "auroc": {"temporal": float(roc_auc_score(lab, t)), "residual_insample": float(roc_auc_score(lab, ri)),
                         "residual_loo": float(roc_auc_score(lab, rl)), "combined_max_insample_temporal": float(roc_auc_score(lab, comb))},
               "tpr_at_fpr_1pct_pooled": {"temporal": tpr_at_fpr(lab, t), "residual_insample": tpr_at_fpr(lab, ri),
                                          "combined": tpr_at_fpr(lab, comb)},
               "note": "pooled FPR over the clean sensors of the corrupted windows (not the clean-window threshold of G-TF)"}
        # diagnostic: temporal std of corrupted vs clean sensors (p.u.)
        row["median_temporal_std_pu"] = {"corrupted": float(-np.median(t[lab == 1])), "clean": float(-np.median(t[lab == 0]))}
        row["fraction_corrupted_with_zero_std"] = float(np.mean(t[lab == 1] == 0.0))
        row["fraction_clean_with_zero_std"] = float(np.mean(t[lab == 0] == 0.0))
        results["conditions"][cname] = row
    OUT.mkdir(parents=True, exist_ok=True)
    name = "temporal_detector_PARTIAL.json" if args.max_batches else "temporal_detector.json"
    (OUT / name).write_text(json.dumps(results, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {OUT/name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
