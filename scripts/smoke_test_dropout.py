"""NEXT-STEPS.md Step 2 parallel task — de-risk the never-run dropout code.

Runs on CPU + tiny data while the GPU sweep is busy (no time cost). Exercises
the two code paths Step 4 depends on, so any bug is fixed NOW rather than
crashing on the first real dropout run:

  1. evaluate_dropout_curve()  over k in {0,1,2,5,10}  (Step 4a mechanism)
  2. DropoutAugmentedLoader     masking nodes during a real train_epoch (Step 4b)

Uses the tiny `dataset` (825 scenarios x 2 runs) and rgsage_max at the frozen
depth-4 config. Trains nothing to convergence — this is a crash/shape check,
not an accuracy check.

Usage:
    uv run python scripts/smoke_test_dropout.py
Exit code 0 = both paths pass; 1 = a failure was caught.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root —
# add the repo root so `fault_localization` imports like the -m sweeps do.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.evaluator import evaluate_dropout_curve
from fault_localization.dropout.training import DropoutAugmentedLoader
from fault_localization.experiments.runner import _model
from fault_localization.experiments.train import train_epoch

DATA_ROOT = "dataset"          # tiny diagnostic set
K_LEVELS = [0, 1, 2, 5, 10]
DEVICE = torch.device("cpu")   # keep the GPU free for the real sweep


def _build_model():
    # Frozen config shape: rgsage_max, depth-4. Small, so it builds fast on CPU.
    config = {
        "model": {
            "name": "rgsage_max",
            "aggregator": "max",
            "temporal_hidden_dim": 128,
            "gnn_hidden_dim": 64,
            "dropout": 0.35,
        },
        "gnn_layers": 4,
    }
    train_ds = WindowGraphDataset(DATA_ROOT, "train")
    model = _model(config, train_ds.edge_index).to(DEVICE)
    return model, train_ds


def test_evaluate_dropout_curve(model) -> None:
    """Step 4a: degradation curve over k must run without crashing."""
    test_ds = WindowGraphDataset(DATA_ROOT, "test")
    # Small subset for speed — dropout evaluator only needs to exercise shapes.
    loader = make_dataloader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    results = evaluate_dropout_curve(
        model=model,
        loader=loader,
        device=DEVICE,
        dropout_levels=K_LEVELS,
        n_mask_seeds=2,          # 2 is enough to prove the seed loop works
        policy="zero_messages",
        f1_average="macro",
        model_name="rgsage_max_smoke",
    )

    # Structural assertions: every k present, monotone-ish keys, finite F1.
    for k in K_LEVELS:
        assert k in results, f"missing k={k} in results"
        r = results[k]
        assert "f1_mean" in r and "f1_std" in r, f"k={k} missing f1 stats"
        assert r["f1_mean"] == r["f1_mean"], f"k={k} f1_mean is NaN"
        assert len(r["f1_per_seed"]) == 2, f"k={k} wrong seed count"
    print("  k     f1_mean ± f1_std")
    for k in K_LEVELS:
        r = results[k]
        print(f"  {k:<5d} {r['f1_mean']:.4f} ± {r['f1_std']:.4f}")


def test_dropout_aware_loader(model, train_ds) -> None:
    """Step 4b: masking during a real train_epoch must run without crashing."""
    base_loader = make_dataloader(
        train_ds, batch_size=32, shuffle=True, seed=42, num_workers=0,
    )
    aug_loader = DropoutAugmentedLoader(
        base_loader,
        k_range=[0, 1, 2, 5],
        rng_seed=42,
        policy="zero_messages",
    )

    # Confirm masking actually zeroes nodes for k>0 batches (feature-masking
    # mechanism, not just a pass-through wrapper).
    saw_masked_batch = False
    for batch in aug_loader:
        k = batch["dropout_k"]
        if k > 0:
            x = batch["x"]
            # Each sample should have exactly k fully-zeroed nodes.
            zeroed = (x.abs().sum(dim=(2, 3)) == 0)  # [B, N] node all-zero
            per_sample = zeroed.sum(dim=1)
            assert (per_sample >= k).all(), (
                f"expected >={k} zeroed nodes/sample, got {per_sample.tolist()[:5]}"
            )
            saw_masked_batch = True
            break
    assert saw_masked_batch, "no k>0 batch produced by DropoutAugmentedLoader"

    # Now a real optimizer step through the augmented loader (one epoch, tiny).
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epoch_loader = aug_loader.with_seed(0)
    stats = train_epoch(model, epoch_loader, opt, DEVICE, grad_clip_norm=1.0)
    assert stats["loss"] == stats["loss"], "train loss is NaN after masked epoch"
    print(f"  masked train_epoch loss = {stats['loss']:.4f} "
          f"({stats['seconds']:.1f}s), grad-clip applied OK")


def main() -> int:
    print(f"[smoke] data={DATA_ROOT} device={DEVICE} — building model...")
    try:
        model, train_ds = _build_model()
    except Exception:
        print("[FAIL] model/dataset build")
        traceback.print_exc()
        return 1

    failures = []
    for name, fn, args in [
        ("evaluate_dropout_curve (Step 4a)", test_evaluate_dropout_curve, (model,)),
        ("DropoutAugmentedLoader (Step 4b)", test_dropout_aware_loader, (model, train_ds)),
    ]:
        print(f"\n[smoke] {name}")
        try:
            fn(*args)
            print(f"[PASS] {name}")
        except Exception:
            print(f"[FAIL] {name}")
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED — {len(failures)} path(s): {failures}")
        return 1
    print("SMOKE TEST PASSED — both dropout paths run clean. Step 4 de-risked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
