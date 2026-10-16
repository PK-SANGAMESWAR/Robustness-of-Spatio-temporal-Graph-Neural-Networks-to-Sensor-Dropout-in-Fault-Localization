"""S4.4 planning, Part 2: measure scoring time over the full S4.4 evaluation grid
(k in {0,1,2,5,10} x sigma in {0,5e-4,2e-3,1e-2}, 3 mask seeds), on a 2% sample of the
validation split (4,960 of 248,000 windows -- 155 batches at batch_size=32).

Timing only. No F1 value is recorded or reported anywhere in this script's output.

Two configurations, one at a time:
  --config physics      the selected S4.3e checkpoint (lambda1=0, lambda2=0.1), use_fusion=True,
                         real physics table. Decomposition (fused/network-only/PSM) computed at
                         every (k, sigma) cell, mirroring scripts/run_pcstgnn_tuning_aux.py's
                         decompose() but extended off the k=0-clean-only cache to the live rho
                         path at every cell (needed for k>0 and sigma>0).
  --config nonphysics    a freshly-constructed PCSTGNN with use_fusion=False, no table, random
                         (untrained) weights. No rho/PSM exists for this configuration (no
                         physics table), so only the forward-pass + masking + noise cost is
                         timed -- which is the honest cost structure for a non-physics arm.

Writes timing (seconds for the 2% grid pass, and x50 scaled to "full validation split") into
results/s4_4_timing/timing.json under keys "scoring_physics_2pct" / "scoring_nonphysics_2pct".

Usage: uv run python scripts/measure_s4_4_scoring_time.py --config {physics,nonphysics}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.masking import apply_sensor_dropout_batch
from fault_localization.dropout.noise import apply_measurement_noise_batch
from fault_localization.models.pc_stgnn import PCSTGNN
from fault_localization.models.voting import soft_vote

DATA_ROOT = "dataset_fullscale_50"
BATCH_SIZE = 32
N_BATCHES_2PCT = 155  # 155*32=4960 ~= 2% of 248,000 val windows
FULL_VAL_WINDOWS = 248000
SAMPLE_WINDOWS = N_BATCHES_2PCT * BATCH_SIZE
SCALE = FULL_VAL_WINDOWS / SAMPLE_WINDOWS  # 50x

K_LEVELS = [0, 1, 2, 5, 10]
SIGMA_LEVELS = [0.0, 0.0005, 0.002, 0.01]
N_MASK_SEEDS = 3

PHYSICS_CKPT = Path(
    "results/pcstgnn_tuning_aux/lam1_0p0_lam2_0p1/pcstgnn/seed_00042/checkpoint_best.pt"
)
PHYSICS_SIGMA_EPS = 0.0007255549105131452


def build_physics_model(edge_index, norm_std, norm_mean) -> PCSTGNN:
    from fault_localization.physics import residual as R, table as T
    from fault_localization.physics.observation import load_norm_stats

    tbl = T.load_table("operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"])
    ns = load_norm_stats(DATA_ROOT)
    model = PCSTGNN(
        edge_index, table, ns,
        use_fusion=True, sigma_eps=PHYSICS_SIGMA_EPS, sigma0=0.0,
        aggregator="max", gnn_hidden_dim=64, num_layers=4,
        temporal_hidden_dim=128, num_classes=26, dropout=0.35,
    )
    state = torch.load(PHYSICS_CKPT, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state)
    return model


def build_nonphysics_model(edge_index) -> PCSTGNN:
    # Random (untrained) weights -- timing only, per instruction. No table: no rho/PSM path
    # is reachable, mirroring what a real non-physics S4.4 arm (R0^n/R1/R2/R7) actually costs.
    return PCSTGNN(
        edge_index, table=None, norm_stats=None,
        use_fusion=False,
        aggregator="max", gnn_hidden_dim=64, num_layers=4,
        temporal_hidden_dim=128, num_classes=26, dropout=0.35,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=["physics", "nonphysics"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = WindowGraphDataset(DATA_ROOT, "val")
    edge_index = val_ds.edge_index
    norm_std = val_ds.std
    norm_mean = val_ds.mean

    if args.config == "physics":
        model = build_physics_model(edge_index, norm_std, norm_mean)
    else:
        model = build_nonphysics_model(edge_index)
    model = model.to(device)
    model.eval()

    # Cache the 2% sample of clean batches once, reused across every (k, sigma) cell.
    loader = make_dataloader(val_ds, BATCH_SIZE, shuffle=False)
    cached_x = []
    for i, batch in enumerate(loader):
        if i >= N_BATCHES_2PCT:
            break
        cached_x.append(batch["x"].to(device))
    n_batches = len(cached_x)
    n_windows = n_batches * BATCH_SIZE

    use_physics = args.config == "physics"

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for sigma in SIGMA_LEVELS:
            noisy = [
                apply_measurement_noise_batch(x, sigma, norm_std, seed=990000 + bi)
                if sigma > 0 else x
                for bi, x in enumerate(cached_x)
            ]
            for k in K_LEVELS:
                offsets = [0] if k == 0 else range(N_MASK_SEEDS)
                for off in offsets:
                    for bi, x in enumerate(noisy):
                        if k > 0:
                            x_eval, _, _ = apply_sensor_dropout_batch(
                                x, k=k, seed=(10000 * k + off) * 10000 + bi,
                                policy="zero_messages", edge_index=edge_index,
                            )
                        else:
                            x_eval = x
                        if use_physics:
                            z = model.network_logits(x_eval)
                            rho, _, _, _ = model._rho(x_eval, sigma0=sigma)
                            beta = float(model.beta)
                            _ = soft_vote(z)
                            _ = soft_vote(z - beta * rho[:, None, :])
                            _ = rho.argmin(-1)
                        else:
                            _ = soft_vote(model(x_eval))

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_2pct_s = time.perf_counter() - t0

    full_scoring_estimate_s = elapsed_2pct_s * SCALE
    result = {
        "config": args.config,
        "n_windows_sampled": n_windows,
        "sample_fraction_of_val": n_windows / FULL_VAL_WINDOWS,
        "k_levels": K_LEVELS,
        "sigma_levels": SIGMA_LEVELS,
        "n_mask_seeds": N_MASK_SEEDS,
        "decomposition_computed": use_physics,
        "elapsed_2pct_seconds": elapsed_2pct_s,
        "scale_to_full_val": SCALE,
        "full_scoring_estimate_seconds": full_scoring_estimate_s,
        "full_scoring_estimate_hours": full_scoring_estimate_s / 3600.0,
        "device": str(device),
        "note": "Timing only -- no F1/accuracy value computed or recorded by this script.",
    }

    out_path = Path("results/s4_4_timing/timing.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    key = "scoring_physics_2pct" if use_physics else "scoring_nonphysics_2pct"
    existing[key] = result
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
