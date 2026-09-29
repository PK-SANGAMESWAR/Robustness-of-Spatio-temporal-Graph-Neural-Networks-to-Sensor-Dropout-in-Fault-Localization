"""Physics-imputation baseline for the sensor-dropout study (``05_STAGE_D_infra.md`` S3.3, row 2).

Impute-then-infer control, physics variant: instead of the graph-mean fill of
``scripts/run_imputation_baseline.py``, reconstruct each dropped node's 3 voltage
channels from the frozen operator table at the **PSM-best hypothesis**
(``argmin_f rho_f(A)`` on the *available* channels, profiled ``(tau,R,L)`` read
off at that class), present it to the model as observed (mask -> 1), and run the
same k in {0,1,2,5,10} sweep with the STANDARD-trained RGSAGE-Max (never
retrained). This closes the "impute then infer" family for a *physically
consistent* fill and answers whether getting the fill right (vs. a graph-mean
heuristic) helps a model that was never trained to distrust either.

Fairness with the existing neighbour-mean baseline (so the two rows are directly
comparable in the same table): the dropped-node selection reuses
``apply_sensor_dropout_batch`` with the SAME per-batch seed derivation
(mask_seed = 10000*k + offset; per-batch seed = mask_seed*10000 + batch_idx) --
byte-identical dropped sets to ``run_imputation_baseline.py`` and to the zeroing
("standard") arm. Only the fill differs.

Physics detail: the table's mu is a *steady-state* per-unit solve, one value per
(location, tau, R, L, node, phase) -- not a time series. The residual itself
already treats the window as quasi-static (``physics/observation.py``: the last
timestep represents the whole post-inception plateau), so the same p.u. fill is
broadcast across all S=20 timesteps of a dropped node's window; this is the same
approximation every physics-based score in this project makes, not a new one.

Usage:
    uv run python scripts/run_physics_imputation_baseline.py \
        --results-dir results/fullscale_final \
        --data-root dataset_fullscale_50 \
        --seeds 42 43 44 \
        --output-dir results/dropout_study_impute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import f1_score  # type: ignore

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.masking import apply_sensor_dropout_batch
from fault_localization.models.voting import soft_vote
from fault_localization.physics import observation as OBS
from fault_localization.physics import residual as R
from fault_localization.physics import table as T
from scripts.run_dropout_study import load_trained_model

CANONICAL_SIGMA_EPS = 0.0007255549105131452
RESIDUAL_METHOD = "gram"  # offline/evaluation path -- float64 matmul form, ~2x
                           # faster than "direct" on the T1000's weak fp64
                           # throughput and exact to float tolerance
                           # (``scripts/run_residual_speed_benchmark.py``); the
                           # 45ms/batch "direct" cost only matters for a
                           # differentiable training loop, not this eval sweep.


def physics_impute(
    x_dropped: torch.Tensor,
    dropped_per_sample: "list[list[int]]",
    table: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Fill ``x_dropped``'s dropped nodes with mu at the PSM-best hypothesis.

    ``x_dropped`` already has the dropped nodes' voltage AND mask channels
    zeroed (``apply_sensor_dropout_batch``'s output) -- that is exactly what
    ``observation.observe`` needs to read off the correct availability A for
    the residual: profiling only ever sees the sensors that survived dropout.
    """
    b, n, s, _ = x_dropped.shape
    device = x_dropped.device
    if not any(dropped_per_sample):
        return x_dropped

    obs_s, obs_a = OBS.observe(x_dropped, mean, std, reduce="last")   # [B,25,3] each, p.u.
    rho_vals, hyp = R.rho(obs_s.double(), obs_a.double(), table, CANONICAL_SIGMA_EPS,
                          sigma0=0.0, method=RESIDUAL_METHOD)
    f_hat = R.psm_predict(rho_vals)                                   # [B]
    h_star = hyp.gather(1, f_hat[:, None]).squeeze(1)                 # [B]
    mu_star = table[f_hat, h_star].to(dtype=x_dropped.dtype)          # [B,75] p.u.
    mu_star = mu_star.reshape(b, 25, 3)

    mean_t = torch.as_tensor(mean, dtype=x_dropped.dtype, device=device)
    std_t = torch.as_tensor(std, dtype=x_dropped.dtype, device=device)
    mu_star_norm = (mu_star - mean_t) / std_t                         # [B,25,3] normalised

    dropped_mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    for i, ds in enumerate(dropped_per_sample):
        if ds:
            dropped_mask[i, ds] = True
    dm = dropped_mask[:, :, None, None]                               # [B,N,1,1]

    fill = mu_star_norm[:, :, None, :].expand(-1, -1, s, -1)          # broadcast over time
    x_imp = x_dropped.clone()
    x_imp[..., 0:3] = torch.where(dm, fill, x_dropped[..., 0:3])
    x_imp[..., 3:6] = torch.where(dm, torch.ones_like(x_dropped[..., 3:6]), x_dropped[..., 3:6])
    return x_imp


@torch.no_grad()
def eval_at_k(model, loader, device, k, mask_seed, table, mean, std,
              f1_average="macro") -> float:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if k > 0:
            seed = mask_seed * 10000 + batch_idx
            x_dropped, _, dropped = apply_sensor_dropout_batch(x, k=k, seed=seed)
            x_in = physics_impute(x_dropped, dropped, table, mean, std)
        else:
            x_in = x
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/fullscale_final")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/dropout_study_impute")
    p.add_argument("--model", default="rgsage_max")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_ds = WindowGraphDataset(args.data_root, "test")
    loader = make_dataloader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    tbl = T.load_table(Path(__file__).resolve().parent.parent / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"]).to(device)
    norm_stats = json.loads(Path(args.data_root, "processed/norm_stats.json").read_text())
    mean = np.asarray(norm_stats["mean"], dtype=np.float32)
    std = np.asarray(norm_stats["std"], dtype=np.float32)
    print(f"[physics-impute] table {tuple(table.shape)} on {device}", flush=True)

    per_seed: dict[int, dict[int, float]] = {}
    for seed in args.seeds:
        seed_dir = Path(args.results_dir) / args.model / f"seed_{seed:05d}"
        model, cfg = load_trained_model(seed_dir, test_ds.edge_index, device)
        f1avg = cfg.get("metrics", {}).get("f1_average", "macro")
        f1_by_k: dict[int, float] = {}
        for k in args.k_levels:
            n_off = 1 if k == 0 else args.n_mask_seeds
            vals = [eval_at_k(model, loader, device, k, 10000 * k + off, table, mean, std, f1avg)
                    for off in range(n_off)]
            f1_by_k[k] = float(np.mean(vals))
            print(f"[physics-impute] {args.model} seed={seed} k={k}: F1={f1_by_k[k]:.4f} "
                  f"(mask offsets={n_off})", flush=True)
        per_seed[seed] = f1_by_k

    ks = args.k_levels
    f1_mean = {k: float(np.mean([per_seed[s][k] for s in args.seeds])) for k in ks}
    f1_std = {k: float(np.std([per_seed[s][k] for s in args.seeds], ddof=1)) for k in ks}
    audc_per_seed = [audc(per_seed[s], ks) for s in args.seeds]
    ret10_per_seed = [per_seed[s][ks[-1]] / per_seed[s][ks[0]] for s in args.seeds]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model, "policy": "physics_impute_psm_best_hypothesis",
        "seeds": args.seeds, "k_levels": ks,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "audc_mean": float(np.mean(audc_per_seed)),
        "audc_std": float(np.std(audc_per_seed, ddof=1)),
        "ret10_mean": float(np.mean(ret10_per_seed)),
        "ret10_std": float(np.std(ret10_per_seed, ddof=1)),
        "err_kind": "train_seed_std",
        "sigma_eps": CANONICAL_SIGMA_EPS,
        "residual_method": RESIDUAL_METHOD,
        "per_seed_f1": {str(s): per_seed[s] for s in args.seeds},
    }
    (out / "physics_imputation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Physics-imputation (PSM-best mu) + standard RGSAGE-Max (seeds "
          f"{args.seeds}) ===")
    for k in ks:
        print(f"  k={k:2d}: F1 = {f1_mean[k]:.3f} ± {f1_std[k]:.3f}")
    print(f"  AUDC   = {summary['audc_mean']:.3f} ± {summary['audc_std']:.3f}")
    print(f"  Ret@10 = {100*summary['ret10_mean']:.1f}% ± {100*summary['ret10_std']:.1f}%")
    print(f"\nWrote {out/'physics_imputation_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
