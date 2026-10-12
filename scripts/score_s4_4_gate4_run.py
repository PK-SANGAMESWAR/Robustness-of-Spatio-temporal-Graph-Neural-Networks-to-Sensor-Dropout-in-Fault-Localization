"""S4.4 Gate 4: score a completed R0^n / R5 training run over the full evaluation grid
(k in {0,1,2,5,10} x sigma in {0,5e-4,2e-3,1e-2}, 3 mask-seed offsets), on the FULL
validation split only (248,000 windows). Never touches the test split.

Fixed 2026-10-12 (see PC-STGNN/06_STAGE_E_pcstgnn.md S4.4 "Tooling incident"): the previous
version cached the entire validation split as GPU tensors (twice -- once clean, once noised)
and OOM'd on the 8GB T1000 after R0^n seed 1's training finished. This version makes a single
streaming pass over the validation DataLoader (the same pattern dropout/evaluator.py's
evaluate_dropout_curve already uses and documents the reason for), holding only the current
32-window batch on the GPU at a time.

Determinism: every window's noise and dropout-mask draw is seeded from (level, mask-seed
offset, GLOBAL window index) -- global index = that window's fixed position in
dataset.records (the split file's own order), looked up by window_id. This is independent of
DataLoader batch order, batch size, or any shared RNG state: each window gets its own fresh
Generator/Random instance for every (level, offset) it's drawn under, so results do not depend
on which other windows happen to share its batch. Noise is varied by mask-seed offset even at
k=0 (per the frozen seed formula), so there is no k=0 dedup across offsets -- a deliberate,
documented behavior change from the first (buggy) version, which held noise fixed across the
3 offsets within a sigma level.

For R5 (use_fusion=True) also computes the fusion decomposition (fused / network-only / PSM)
at every (k, sigma) cell. For R0^n (use_fusion=False) there is no physics table, so only
fused(=network-only) F1 is reported at each cell.

Writes results/s4_4/<arm>/pcstgnn/seed_<seed:05d>/scoring_full_grid.json and returns the
summary dict. Releases GPU memory (model + empty_cache) before returning.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import f1_score

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.masking import apply_sensor_dropout_batch
from fault_localization.dropout.noise import apply_measurement_noise_batch
from fault_localization.models.pc_stgnn import PCSTGNN
from fault_localization.models.voting import soft_vote

DATA_ROOT = "dataset_fullscale_50"
BATCH_SIZE = 32
K_LEVELS = [0, 1, 2, 5, 10]
SIGMA_LEVELS = [0.0, 0.0005, 0.002, 0.01]
N_MASK_SEEDS = 3
PHYSICS_SIGMA_EPS = 0.0007255549105131452

# Seed-space separation: global window index is always < 1,000,000 (val split is 248,000
# windows; even the full 1.65M-window corpus is < 10,000,000), so multiplying the level index
# and the mask-seed offset by disjoint powers of ten cannot collide with the window index term.
_LEVEL_MULT = 10_000_000
_OFFSET_MULT = 1_000_000


def noise_seed(sigma_level_idx: int, offset: int, global_idx: int) -> int:
    return sigma_level_idx * _LEVEL_MULT + offset * _OFFSET_MULT + global_idx


def dropout_seed(k: int, offset: int, global_idx: int) -> int:
    return k * _LEVEL_MULT + offset * _OFFSET_MULT + global_idx


def _build_model(use_fusion: bool, edge_index, device):
    if use_fusion:
        from fault_localization.physics import residual as R, table as T
        from fault_localization.physics.observation import load_norm_stats
        tbl = T.load_table("operator")
        table = R.flatten_table(tbl["mu"], tbl["mu0"])
        ns = load_norm_stats(DATA_ROOT)
    else:
        table, ns = None, None
    model = PCSTGNN(
        edge_index, table, ns,
        use_fusion=use_fusion, sigma_eps=PHYSICS_SIGMA_EPS, sigma0=0.0,
        aggregator="max", gnn_hidden_dim=64, num_layers=4,
        temporal_hidden_dim=128, num_classes=26, dropout=0.35,
    ).to(device)
    return model


def score_run(arm: str, seed: int, max_windows: int | None = None) -> dict:
    """Score a Gate 4 checkpoint. ``max_windows``, if given, restricts scoring to the first
    ``max_windows`` windows by global index (val split order) -- used only by the equivalence
    test; production calls (the CLI / the orchestrator) leave it ``None`` (the full split)."""
    use_fusion = arm == "r5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("results/s4_4") / arm / "pcstgnn" / f"seed_{seed:05d}"
    ckpt_path = run_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} -- train before scoring")

    val_ds = WindowGraphDataset(DATA_ROOT, "val")
    edge_index = val_ds.edge_index
    norm_std = val_ds.std
    window_to_global_idx = {wid: i for i, (wid, *_rest) in enumerate(val_ds.records)}

    model = _build_model(use_fusion, edge_index, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)["model"]
    model.load_state_dict(state)
    model.eval()
    beta = float(model.beta) if use_fusion else None

    loader = make_dataloader(val_ds, BATCH_SIZE, shuffle=False)

    per_offset_preds: dict[tuple, dict[str, list]] = {
        (k, sigma, off): {"pf": [], "pn": [], "pp": []}
        for sigma in SIGMA_LEVELS for k in K_LEVELS for off in range(N_MASK_SEEDS)
    }
    labels: list[int] = []
    n_windows_seen = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if max_windows is not None and n_windows_seen >= max_windows:
                break
            x_full = batch["x"]
            wids_full = batch["window_id"]
            y_full = batch["y"].tolist()

            if max_windows is not None:
                keep = max_windows - n_windows_seen
                if keep < len(wids_full):
                    x_full = x_full[:keep]
                    wids_full = wids_full[:keep]
                    y_full = y_full[:keep]

            x_full = x_full.to(device)
            b = x_full.shape[0]
            gidx = [window_to_global_idx[w] for w in wids_full]
            labels.extend(y_full)
            n_windows_seen += b

            for sigma_idx, sigma in enumerate(SIGMA_LEVELS):
                for off in range(N_MASK_SEEDS):
                    if sigma > 0:
                        x_noisy = x_full.clone()
                        for i in range(b):
                            s = noise_seed(sigma_idx, off, gidx[i])
                            x_noisy[i:i + 1] = apply_measurement_noise_batch(
                                x_full[i:i + 1], sigma, norm_std, seed=s,
                            )
                    else:
                        x_noisy = x_full

                    for k in K_LEVELS:
                        if k > 0:
                            x_eval = x_noisy.clone()
                            for i in range(b):
                                s = dropout_seed(k, off, gidx[i])
                                x_eval[i:i + 1], _, _ = apply_sensor_dropout_batch(
                                    x_noisy[i:i + 1], k=k, seed=s,
                                    policy="zero_messages", edge_index=edge_index,
                                )
                        else:
                            x_eval = x_noisy

                        dest = per_offset_preds[(k, sigma, off)]
                        if use_fusion:
                            z = model.network_logits(x_eval)
                            rho, _, _, _ = model._rho(x_eval, sigma0=sigma)
                            dest["pf"].extend(soft_vote(z - beta * rho[:, None, :])[1].cpu().tolist())
                            dest["pn"].extend(soft_vote(z)[1].cpu().tolist())
                            dest["pp"].extend(rho.argmin(-1).cpu().tolist())
                        else:
                            dest["pf"].extend(soft_vote(model(x_eval))[1].cpu().tolist())
    elapsed_s = time.perf_counter() - t0

    y_true = np.asarray(labels)
    out_cells: dict[str, dict] = {}
    for sigma in SIGMA_LEVELS:
        for k in K_LEVELS:
            f1_per_offset, decomp_per_offset = [], []
            for off in range(N_MASK_SEEDS):
                d = per_offset_preds[(k, sigma, off)]
                pf = np.asarray(d["pf"])
                f1_per_offset.append(
                    float(f1_score(y_true, pf, average="macro",
                                   labels=list(range(26)), zero_division=0))
                )
                if use_fusion:
                    pn = np.asarray(d["pn"])
                    pp = np.asarray(d["pp"])
                    decomp_per_offset.append({
                        "f1_network_only": float(f1_score(y_true, pn, average="macro",
                                                          labels=list(range(26)), zero_division=0)),
                        "f1_psm": float(f1_score(y_true, pp, average="macro",
                                                 labels=list(range(26)), zero_division=0)),
                        "fused_eq_psm": float(np.mean(pf == pp)),
                        "fused_eq_network_only": float(np.mean(pf == pn)),
                    })
            cell = {
                "f1_fused_mean": float(np.mean(f1_per_offset)),
                "f1_fused_std": float(np.std(f1_per_offset)),
                "f1_fused_per_offset": f1_per_offset,
            }
            if use_fusion:
                for key in decomp_per_offset[0]:
                    cell[key + "_mean"] = float(np.mean([d[key] for d in decomp_per_offset]))
            out_cells[f"k{k}_sigma{sigma}"] = cell

    summary = {
        "arm": arm,
        "seed": seed,
        "use_fusion": use_fusion,
        "beta": beta,
        "n_windows": n_windows_seen,
        "k_levels": K_LEVELS,
        "sigma_levels": SIGMA_LEVELS,
        "n_mask_seeds": N_MASK_SEEDS,
        "decomposition_computed": use_fusion,
        "cells": out_cells,
        "scoring_wallclock_seconds": elapsed_s,
        "note": "Validation split only; test split never loaded. Single streaming pass over "
                "the loader (evaluate_dropout_curve's own pattern). Noise/dropout seeded per "
                "window from (level, mask-seed offset, global window index); no k=0 dedup "
                "across offsets because noise now varies by offset there too.",
    }
    if max_windows is None:
        out_path = run_dir / "scoring_full_grid.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["r0n", "r5"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-windows", type=int, default=None)
    args = p.parse_args()
    summary = score_run(args.arm, args.seed, max_windows=args.max_windows)
    print(json.dumps({"arm": summary["arm"], "seed": summary["seed"],
                      "n_windows": summary["n_windows"],
                      "scoring_wallclock_seconds": summary["scoring_wallclock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
