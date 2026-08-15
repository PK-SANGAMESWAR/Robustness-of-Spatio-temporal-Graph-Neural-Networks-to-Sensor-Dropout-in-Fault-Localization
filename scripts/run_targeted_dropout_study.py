"""Worst-case (targeted) sensor-dropout evaluation.

Companion to ``run_dropout_study.py``.  That script drops k μPMUs uniformly at
random and therefore reports average-case degradation; this one drops the k
μPMUs topologically **nearest the fault bus**, giving the worst case over
sensor subsets rather than the mean.

Targeted selection is a deterministic function of the sample label (see
``fault_localization/dropout/targeted.py``), so there are no mask seeds to
average over: the only error bar is over training seeds 42/43/44.

Usage:
    uv run python scripts/run_targeted_dropout_study.py \
        --results-dir results/fullscale_final \
        --data-root dataset_fullscale_50 \
        --models rgsage_max --seeds 42 43 44 \
        --output-dir results/dropout_study_targeted

For the dropout-aware checkpoints, point --results-dir at results/dropout_aware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.targeted import (
    apply_targeted_dropout_batch,
    build_proximity_rank,
)
from fault_localization.models.voting import soft_vote

from run_dropout_study import load_trained_model  # noqa: E402

MODELS_DEFAULT = ["rgsage_max"]


def get_proximity_rank(data_root: Path, cache: Path) -> np.ndarray:
    """Load the cached proximity rank, building it via OpenDSS on a miss."""
    if cache.exists():
        rank = np.load(cache)
        if rank.shape == (25, 25):
            print(f"[rank] loaded cached proximity rank from {cache}", flush=True)
            return rank
        print(f"[rank] cache {cache} has wrong shape {rank.shape}; rebuilding",
              flush=True)

    print("[rank] building proximity rank (compiling feeder in OpenDSS) ...",
          flush=True)
    rank = build_proximity_rank(data_root / "processed" / "node_mapping.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, rank)
    print(f"[rank] wrote {cache}", flush=True)
    return rank


def evaluate_targeted_curve(
    model, loader, device, k_levels, rank, f1_average="macro",
) -> dict[int, dict[str, float]]:
    """Evaluate one model across targeted-dropout levels in a single pass.

    Mirrors ``evaluate_dropout_curve``'s single-pass structure: the test split
    lives in per-event compressed .npz files, so looping k *inside* the batch
    loop decompresses the split once instead of len(k_levels) times.
    """
    from sklearn.metrics import f1_score

    preds: dict[int, list[int]] = {k: [] for k in k_levels}
    labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            labels.extend(batch["y"].tolist())

            zero_pred: list[int] | None = None
            for k in k_levels:
                if k == 0:
                    if zero_pred is None:
                        _, gp = soft_vote(model(x))
                        zero_pred = gp.cpu().tolist()
                    preds[k].extend(zero_pred)
                    continue

                x_dropped, _ = apply_targeted_dropout_batch(
                    x, y, k=k, rank=rank,
                    fallback_seed=10000 * k + batch_idx,
                )
                _, gp = soft_vote(model(x_dropped))
                preds[k].extend(gp.cpu().tolist())

    out: dict[int, dict[str, float]] = {}
    labels_arr = np.array(labels)
    for k in k_levels:
        p = np.array(preds[k])
        out[k] = {
            "f1": float(f1_score(labels_arr, p, average=f1_average, zero_division=0)),
            "acc": float(np.mean(p == labels_arr)),
            "n_samples": int(len(labels_arr)),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/fullscale_final")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/dropout_study_targeted")
    p.add_argument("--models", nargs="+", default=MODELS_DEFAULT)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--variant", default="standard",
                   help="Label recorded in the output (standard | dropout_aware).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    data_root = (repo / args.data_root).resolve()
    results_dir = (repo / args.results_dir).resolve()
    out_dir = (repo / args.output_dir).resolve()

    # Resolve every path before touching OpenDSS: Compile chdir's the process.
    rank = get_proximity_rank(data_root, out_dir / "proximity_rank.npy")

    device = torch.device(args.device)
    test_ds = WindowGraphDataset(str(data_root), "test")
    test_loader = make_dataloader(test_ds, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    rows: list[dict] = []
    for model_name in args.models:
        for seed in args.seeds:
            seed_dir = results_dir / model_name / f"seed_{seed:05d}"
            if not (seed_dir / "checkpoint_best.pt").exists():
                print(f"[skip] {model_name} seed={seed}: no checkpoint", flush=True)
                continue

            model, cfg = load_trained_model(seed_dir, test_ds.edge_index, device)
            f1_avg = cfg.get("metrics", {}).get("f1_average", "macro")
            print(f"[eval] {model_name} ({args.variant}) seed={seed} "
                  f"targeted k={args.k_levels} ...", flush=True)

            curve = evaluate_targeted_curve(
                model, test_loader, device, args.k_levels, rank, f1_avg,
            )
            for k in args.k_levels:
                print(f"  [targeted] {model_name} seed={seed} k={k}: "
                      f"F1={curve[k]['f1']:.4f}", flush=True)
                rows.append({
                    "model": model_name, "variant": args.variant, "seed": seed,
                    "k": k, "f1_seed": curve[k]["f1"], "acc_seed": curve[k]["acc"],
                    "policy": "targeted_nearest_fault",
                })

    if not rows:
        print("No models evaluated (no checkpoints found).")
        return 1

    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per_seed_path = out_dir / f"targeted_per_seed_{args.variant}.csv"
    df.to_csv(per_seed_path, index=False)

    agg = []
    for (model_name, k), g in df.groupby(["model", "k"], sort=False):
        f1s = g["f1_seed"].to_numpy()
        agg.append({
            "model": model_name, "variant": args.variant, "k": int(k),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "acc_mean": float(g["acc_seed"].mean()),
            "n_train_seeds": int(len(f1s)),
            "policy": "targeted_nearest_fault",
            "err_kind": "train_seed_std",
            "f1_per_train_seed": json.dumps([round(float(v), 6) for v in f1s]),
        })
    agg_df = pd.DataFrame(agg).sort_values(["model", "k"])
    agg_path = out_dir / f"targeted_curves_{args.variant}.csv"
    agg_df.to_csv(agg_path, index=False)

    print(f"\n=== Targeted dropout ({args.variant}), seeds {args.seeds} ===")
    for r in agg:
        print(f"  {r['model']:<12} k={r['k']:>2}: F1 = "
              f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}")
    print(f"\nWrote {per_seed_path}\n      {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
