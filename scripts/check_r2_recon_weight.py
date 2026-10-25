"""R2 reconstruction-loss weight check (PC-STGNN/S4.4_remaining_arms_plan.md section 4.2,
D5 follow-up, 2026-10-24): on one fixed batch at seed 42, at model initialization (no
training steps), compute the main loss (CE) and the reconstruction loss (recon_loss_weight=1.0,
unweighted) from config/s4_4/r2.yaml, and report their ratio -- this is the balance check
before committing a weight, not a training run.

Usage: uv run python scripts/check_r2_recon_weight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from fault_localization.experiments.config import load_config, validate_config
from fault_localization.experiments.runner import _model as build_model_from_config
from fault_localization.experiments.seed import seed_everything
from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.training import PhysicsDropoutLoader
from fault_localization.models.common import pc_loss, unlabelable_window_mask
from fault_localization.physics.observation import observe


def main() -> None:
    config = validate_config(load_config("config/s4_4/r2.yaml"))
    config["seed"] = 42
    seed_everything(42)

    edge_index = torch.tensor(
        __import__("numpy").load(f"{config['data_root']}/processed/edge_index.npy").astype("int64")
    )
    model = build_model_from_config(config, edge_index)
    model.train()  # matches training-time BN/dropout behavior, same as pc_train_epoch

    train_ds = WindowGraphDataset(config["data_root"], "train")
    base_loader = make_dataloader(train_ds, config["training"]["batch_size"], shuffle=True,
                                  seed=config["seed"], num_workers=0)
    ds_cfg = config.get("dropout_study", {})
    noise_cfg = config.get("noise_augmentation", {})
    noise_sigmas = list(noise_cfg.get("sigma_grid", [])) if noise_cfg.get("enabled", False) else None
    loader = PhysicsDropoutLoader(
        base_loader,
        k_range=ds_cfg.get("train_k_range", [0, 1, 2, 5]),
        rng_seed=config["seed"],
        policy=ds_cfg.get("policy", "zero_messages"),
        noise_sigmas=noise_sigmas,
        norm_std=train_ds.std if noise_sigmas else None,
    )
    # The seed-42 stream's first batch may draw k=0 (no synthetic dropout -> drop_mask all
    # zero -> a vacuous 0/0 reconstruction loss, uninformative for a balance check). Take the
    # first batch in the same deterministic stream that actually has a dropped channel --
    # still "one fixed batch at seed 42" (fully determined by the seed, not cherry-picked
    # across runs), just not necessarily index 0.
    batch_index = 0
    for batch_index, batch in enumerate(loader):
        if batch["drop_mask"].sum() > 0:
            break
    print(f"batch index in the seed-42 stream (first with a dropped channel): {batch_index}")

    x = batch["x"]
    x_clean = batch["x_clean"]
    drop_mask = batch["drop_mask"]
    y = batch["y"]
    loss_mask = unlabelable_window_mask(batch["window_id"])
    sigma0 = batch.get("sigma0")

    out = model.forward_pc(x, x_clean, drop_mask, sigma0=sigma0)
    with torch.no_grad():
        recon_target, _ = observe(x_clean, model.norm_mean, model.norm_std, model.obs_reduce)

    parts = pc_loss(out, y, lam1=0.0, lam2=0.0, loss_mask=loss_mask,
                    recon_loss_weight=1.0, recon_target=recon_target, recon_mask=drop_mask)

    main_loss = float(parts["ce"])
    recon_loss = float(parts["recon_mse"])
    ratio = recon_loss / main_loss if main_loss else float("inf")
    n_dropped_channels = int(drop_mask.sum().item())
    print(f"batch size: {x.shape[0]}  dropout_k: {batch.get('dropout_k', 'n/a')}")
    print(f"dropped channels in this batch (recon_mask sum): {n_dropped_channels}")
    print(f"main loss (CE):              {main_loss:.6f}")
    print(f"reconstruction loss (unweighted MSE): {recon_loss:.6f}")
    print(f"ratio recon/main:            {ratio:.6f}")
    print(f"ratio main/recon:            {(main_loss / recon_loss) if recon_loss else float('inf'):.6f}")
    in_range = 0.1 <= ratio <= 10.0
    print(f"0.1 <= ratio <= 10: {in_range}")


if __name__ == "__main__":
    main()
