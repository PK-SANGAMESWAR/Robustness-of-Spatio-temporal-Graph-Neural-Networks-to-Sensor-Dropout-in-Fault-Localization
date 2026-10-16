"""S4.4 planning: measure training speed for the non-physics arms (R0^n, R1, R2, R7).

Not an S4.4 run. Trains 200 batches per arm, one arm per invocation, at seed 42, with no
checkpoints and no results/ output besides the merged timing file this script writes
(results/s4_4_timing/timing.json). Mirrors experiments/runner.py's model construction and
training-loop plumbing exactly (same PCSTGNN class, same PhysicsDropoutLoader, same pc_train_epoch/
pc_loss) so the measured speed reflects the actual planned S4.4 settings, not a simplified stand-in.

R0^n / R1 / R2 have no physics table (use_fusion=False, lambda1=lambda2=aux_net_loss_weight=0 --
gamma=0 per the S4.4-plan-draft rule "gamma applies only when fusion is ON"), so they go through
pc_train_epoch purely for the noise-augmentation loader plumbing (noise_augmentation is "only
wired through the pc_loss path", runner.py:258-259), not because any physics loss term is active.

R7 (size-matched wider RGSAGE-Max) has no width specified anywhere in the docs or code -- this
script measures it at the baseline gnn_hidden_dim=64 as a flagged placeholder, not R7's true
(larger, undefined) cost. See the printed warning and the "width_is_placeholder" field.

Usage: uv run python scripts/measure_s4_4_timing.py --arm {r0n,r1,r2,r7}
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

# Must be set before importing torch, matching experiments/runner.py:7 exactly, so this
# measurement's determinism settings (and thus speed) match a real run's.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.training import PhysicsDropoutLoader
from fault_localization.experiments.seed import seed_everything
from fault_localization.experiments.train import validate_epoch
from fault_localization.models.common import pc_loss, unlabelable_window_mask
from fault_localization.models.pc_stgnn import PCSTGNN

DATA_ROOT = "dataset_fullscale_50"
BATCH_SIZE = 32
GNN_LAYERS = 4  # frozen sweep depth, CLAUDE.md
TEMPORAL_HIDDEN = 128
GNN_HIDDEN = 64  # baseline width; R7's true width is undefined, see module docstring
LR = 0.001
WEIGHT_DECAY = 0.0001
GRAD_CLIP_NORM = 1.0
NOISE_SIGMAS = [0.0, 0.0005, 0.002, 0.01]
TRAIN_K_RANGE = [0, 1, 2, 5]
N_TRAIN_BATCHES = 200
N_VAL_BATCHES = 200
SEED = 42

ARM_SWITCHES = {
    "r0n": dict(use_type_head=False, use_decoder=False, width_is_placeholder=False),
    "r1": dict(use_type_head=True, use_decoder=False, width_is_placeholder=False),
    "r2": dict(use_type_head=False, use_decoder=True, width_is_placeholder=False),
    "r7": dict(use_type_head=False, use_decoder=False, width_is_placeholder=True),
}


def build_model(arm: str, edge_index) -> PCSTGNN:
    sw = ARM_SWITCHES[arm]
    return PCSTGNN(
        edge_index, table=None, norm_stats=None,
        use_fusion=False,
        use_type_head=sw["use_type_head"],
        use_decoder=sw["use_decoder"],
        aggregator="max",
        gnn_hidden_dim=GNN_HIDDEN,
        num_layers=GNN_LAYERS,
        temporal_hidden_dim=TEMPORAL_HIDDEN,
        num_classes=26,
        dropout=0.35,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARM_SWITCHES))
    args = p.parse_args()
    arm = args.arm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED, deterministic=True)

    train_ds = WindowGraphDataset(DATA_ROOT, "train")
    val_ds = WindowGraphDataset(DATA_ROOT, "val")
    n_train_batches_per_epoch = -(-len(train_ds) // BATCH_SIZE)
    n_val_batches_per_epoch = -(-len(val_ds) // BATCH_SIZE)

    model = build_model(arm, train_ds.edge_index).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    base_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True, seed=SEED, num_workers=0)
    train_loader = PhysicsDropoutLoader(
        base_loader, k_range=TRAIN_K_RANGE, rng_seed=SEED, policy="zero_messages",
        noise_sigmas=NOISE_SIGMAS, norm_std=train_ds.std,
    )

    # ---- time N_TRAIN_BATCHES training steps (forward_pc + pc_loss + backward + step) ----
    model.train()
    batch_seconds = []
    it = iter(train_loader)
    for i in range(N_TRAIN_BATCHES):
        batch = next(it)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        x = batch["x"].to(device, non_blocking=True)
        x_clean = batch["x_clean"].to(device, non_blocking=True)
        drop_mask = batch["drop_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        loss_mask = unlabelable_window_mask(batch["window_id"]).to(device)
        sigma0 = batch.get("sigma0")
        if sigma0 is not None:
            sigma0 = sigma0.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        out = model.forward_pc(x, x_clean, drop_mask, sigma0=sigma0)
        parts = pc_loss(out, y, lam1=0.0, lam2=0.0, loss_mask=loss_mask, aux_net_weight=0.0)
        loss = parts["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_seconds.append(time.perf_counter() - t0)

    # ---- time N_VAL_BATCHES validation forward passes (plain model(x), no grad) ----
    val_loader = make_dataloader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    val_batch_seconds = []
    it = iter(val_loader)
    with torch.no_grad():
        for i in range(N_VAL_BATCHES):
            batch = next(it)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = batch["x"].to(device, non_blocking=True)
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            val_batch_seconds.append(time.perf_counter() - t0)

    mean_train_batch_s = sum(batch_seconds) / len(batch_seconds)
    mean_val_batch_s = sum(val_batch_seconds) / len(val_batch_seconds)
    epoch_train_s = mean_train_batch_s * n_train_batches_per_epoch
    epoch_val_s = mean_val_batch_s * n_val_batches_per_epoch
    epoch_with_val_s = epoch_train_s + epoch_val_s

    result = {
        "arm": arm,
        "seed": SEED,
        "n_train_batches_measured": N_TRAIN_BATCHES,
        "n_val_batches_measured": N_VAL_BATCHES,
        "mean_train_batch_seconds": mean_train_batch_s,
        "mean_val_batch_seconds": mean_val_batch_s,
        "train_batches_per_epoch": n_train_batches_per_epoch,
        "val_batches_per_epoch": n_val_batches_per_epoch,
        "epoch_train_seconds": epoch_train_s,
        "epoch_validation_seconds": epoch_val_s,
        "epoch_with_validation_seconds": epoch_with_val_s,
        "epoch_with_validation_minutes": epoch_with_val_s / 60.0,
        "width_is_placeholder": ARM_SWITCHES[arm]["width_is_placeholder"],
        "gnn_hidden_dim_used": GNN_HIDDEN,
        "gnn_layers": GNN_LAYERS,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    if ARM_SWITCHES[arm]["width_is_placeholder"]:
        print(f"WARNING: arm {arm!r} has no defined 'wider' width anywhere in the docs/code. "
              f"This measurement uses the baseline gnn_hidden_dim={GNN_HIDDEN} and very likely "
              f"UNDERSTATES R7's true per-batch cost.")

    out_path = Path("results/s4_4_timing/timing.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing[arm] = result
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
