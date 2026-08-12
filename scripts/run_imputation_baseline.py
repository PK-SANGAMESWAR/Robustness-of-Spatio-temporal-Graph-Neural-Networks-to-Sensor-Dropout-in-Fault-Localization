"""Imputation baseline for the sensor-dropout study (reviewer request).

Impute-then-infer control: instead of zeroing a dropped mu-PMU, we reconstruct its
three voltage channels from the *available* graph neighbours (per timestep) and
present the imputed reading to the STANDARD-trained RGSAGE-Max as if observed
(mask channels set to 1). This is the lightweight, graph-based analogue of
Dirichlet/GDIA imputation (Zhang et al. 2025). We then run the same
k in {0,1,2,5,10} dropout sweep and compare against zeroing (standard) and
dropout-aware training.

Fairness: the dropped-node selection reuses the evaluator's mask seeding
(mask_seed = 10000*k + offset, per-sample rng.sample) so the SAME nodes are
dropped as in the zero_messages study; only the fill differs (neighbour-mean
vs. zero). Standard-trained checkpoints only (no retraining).

Usage:
    uv run python scripts/run_imputation_baseline.py \
        --results-dir results/fullscale_final \
        --data-root dataset_fullscale_50 \
        --seeds 42 43 44 \
        --output-dir results/dropout_study_impute
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import f1_score  # type: ignore

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.models.voting import soft_vote
from scripts.run_dropout_study import load_trained_model


def build_adjacency(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Undirected 0/1 adjacency [N, N] with zero diagonal (no self-loops)."""
    a = torch.zeros(n, n)
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        if s != d:
            a[s, d] = 1.0
            a[d, s] = 1.0
    return a


def impute_batch(x: torch.Tensor, k: int, mask_seed: int, adj: torch.Tensor) -> torch.Tensor:
    """Neighbour-mean impute the k dropped nodes' voltage channels.

    x: [B, N, S, 6] on device. Returns a new tensor with dropped nodes' voltage
    channels (0:3) filled by the mean of their AVAILABLE neighbours (per step,
    per phase) and mask channels (3:6) set to 1 (presented as observed). Nodes
    whose whole neighbourhood is dropped fall back to 0.
    """
    if k <= 0:
        return x
    b, n, s, _ = x.shape
    # Per-sample availability (1 = kept, 0 = dropped) — same selection rule as
    # apply_sensor_dropout_batch (fresh rng per batch, per-sample sampling).
    # VERIFIED byte-for-byte identical to the zeroing study's dropped sets for
    # every (k, offset) with batch=32, shuffle=False — so the ONLY difference
    # from the zero_messages run is the fill (neighbour-mean vs zero), which is
    # what makes this a fair paired imputation baseline.
    avail = torch.ones(b, n, device=x.device)
    rng = random.Random(mask_seed)
    for i in range(b):
        dropped = rng.sample(range(n), min(k, n))
        avail[i, dropped] = 0.0

    xv = x[..., 0:3]                              # [B,N,S,3]
    xw = xv * avail[:, :, None, None]             # zero out dropped neighbours' contribution
    num = torch.einsum("nm,bmsc->bnsc", adj, xw)  # sum of available neighbours' voltages
    den = torch.einsum("nm,bm->bn", adj, avail)   # count of available neighbours
    imp = num / den[..., None, None].clamp(min=1.0)

    dropped_mask = (avail == 0)[..., None, None]  # [B,N,1,1]
    x_imp = x.clone()
    x_imp[..., 0:3] = torch.where(dropped_mask, imp, xv)
    x_imp[..., 3:6] = torch.where(dropped_mask, torch.ones_like(x[..., 3:6]), x[..., 3:6])
    return x_imp


@torch.no_grad()
def eval_at_k(model, loader, device, k, mask_seed, adj, f1_average="macro") -> float:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        x_in = impute_batch(x, k, mask_seed, adj) if k > 0 else x
        logits = model(x_in)
        _, graph_pred = soft_vote(logits)
        preds.extend(graph_pred.cpu().tolist())
        labels.extend(y.cpu().tolist())
    return float(f1_score(labels, preds, average=f1_average, zero_division=0))


def audc(f1_by_k: dict[int, float], ks: list[int]) -> float:
    """Normalized area under the retention curve R(k)=F1(k)/F1(0) (trapezoid)."""
    r = [f1_by_k[k] / f1_by_k[ks[0]] for k in ks]
    area = sum((r[i] + r[i + 1]) / 2 * (ks[i + 1] - ks[i]) for i in range(len(ks) - 1))
    return area / (ks[-1] - ks[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/fullscale_final")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/dropout_study_impute")
    p.add_argument("--model", default="rgsage_max")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_ds = WindowGraphDataset(args.data_root, "test")
    loader = make_dataloader(test_ds, args.batch_size, shuffle=False, num_workers=0)
    n_nodes = int(test_ds.edge_index.max()) + 1
    adj = build_adjacency(test_ds.edge_index, n_nodes).to(device)

    # per-seed F1(k) = mean over mask offsets
    per_seed: dict[int, dict[int, float]] = {}
    for seed in args.seeds:
        seed_dir = Path(args.results_dir) / args.model / f"seed_{seed:05d}"
        model, cfg = load_trained_model(seed_dir, test_ds.edge_index, device)
        f1avg = cfg.get("metrics", {}).get("f1_average", "macro")
        f1_by_k: dict[int, float] = {}
        for k in args.k_levels:
            n_off = 1 if k == 0 else args.n_mask_seeds  # k=0 imputation is a no-op
            vals = [eval_at_k(model, loader, device, k, 10000 * k + off, adj, f1avg)
                    for off in range(n_off)]
            f1_by_k[k] = float(np.mean(vals))
            print(f"[impute] {args.model} seed={seed} k={k}: F1={f1_by_k[k]:.4f} "
                  f"(mask offsets={args.n_mask_seeds})", flush=True)
        per_seed[seed] = f1_by_k

    # aggregate over seeds
    ks = args.k_levels
    f1_mean = {k: float(np.mean([per_seed[s][k] for s in args.seeds])) for k in ks}
    f1_std = {k: float(np.std([per_seed[s][k] for s in args.seeds])) for k in ks}
    audc_per_seed = [audc(per_seed[s], ks) for s in args.seeds]
    ret10_per_seed = [per_seed[s][ks[-1]] / per_seed[s][ks[0]] for s in args.seeds]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model, "policy": "neighbour_mean_impute",
        "seeds": args.seeds, "k_levels": ks,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "audc_mean": float(np.mean(audc_per_seed)),
        "audc_std": float(np.std(audc_per_seed)),
        "ret10_mean": float(np.mean(ret10_per_seed)),
        "ret10_std": float(np.std(ret10_per_seed)),
        "per_seed_f1": {str(s): per_seed[s] for s in args.seeds},
    }
    (out / "imputation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Mean-imputation + standard RGSAGE-Max (seeds "
          f"{args.seeds}) ===")
    for k in ks:
        print(f"  k={k:2d}: F1 = {f1_mean[k]:.3f} ± {f1_std[k]:.3f}")
    print(f"  AUDC   = {summary['audc_mean']:.3f} ± {summary['audc_std']:.3f}")
    print(f"  Ret@10 = {100*summary['ret10_mean']:.1f}% ± {100*summary['ret10_std']:.1f}%")
    print(f"\nWrote {out/'imputation_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
