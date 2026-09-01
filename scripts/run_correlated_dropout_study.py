"""Spatially correlated (contiguous) sensor-dropout evaluation.

Companion to ``run_dropout_study.py`` (scattered, independent loss) and
``run_targeted_dropout_study.py`` (worst case by proximity to the fault).  This
one drops a **connected block** of k μPMUs from the measured-only graph, the
failure shape a shared communication path, a common backhaul, or a feeder-
segment outage actually produces.

It is the third corner of the missingness space the paper reports, and the one
its Limitations section previously named as unmeasured.  Scattered loss is the
friendly case for message passing, because a surviving neighbour can relay what
the dropped node would have contributed; a contiguous outage removes that
neighbour too.  Comparing this curve against the random curve at equal k
therefore isolates the effect of *spatial correlation* alone — the number of
dropped sensors, the zeroing semantics, the k grid, the checkpoints, and the
test split are all held identical.

Nothing is retrained; frozen checkpoints are evaluated as they are.

Usage:
    uv run python scripts/run_correlated_dropout_study.py \
        --results-dir results/fullscale_final --variant standard \
        --models rgsage_max --seeds 42 43 44 \
        --output-dir results/dropout_study_correlated

    uv run python scripts/run_correlated_dropout_study.py \
        --results-dir results/dropout_aware --variant dropout_aware \
        --models rgsage_max --seeds 42 43 44 \
        --output-dir results/dropout_study_correlated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.correlated import (
    apply_correlated_dropout_batch,
    build_adjacency,
)
from fault_localization.models.voting import soft_vote

from run_dropout_study import load_trained_model  # noqa: E402

POLICY = "correlated_connected_block"


def evaluate_correlated_curve(
    model, loader, device, k_levels, adjacency,
    n_mask_seeds: int = 3, f1_average: str = "macro", max_batches: int | None = None,
) -> dict[int, dict[str, float]]:
    """One pass over the loader covering every (k, mask-offset) combo.

    Same single-pass structure as ``evaluate_dropout_curve`` and the same
    ``mask_seed = 10000*k + offset`` scheme, so every model sees an identical
    sequence of contiguous outages and the comparison against the random sweep
    is paired at the level of (k, offset, batch).
    """
    from sklearn.metrics import f1_score

    combos = [(k, off) for k in k_levels
              for off in range(1 if k == 0 else n_mask_seeds)]
    preds: dict[tuple[int, int], list[int]] = {c: [] for c in combos}
    labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["x"].to(device)
            labels.extend(batch["y"].tolist())

            for (k, off) in combos:
                if k == 0:
                    _, gp = soft_vote(model(x))
                    preds[(k, off)].extend(gp.cpu().tolist())
                    continue

                mask_seed = 10000 * k + off
                x_dropped, _ = apply_correlated_dropout_batch(
                    x, k=k, seed=mask_seed * 10000 + batch_idx,
                    adjacency=adjacency,
                )
                _, gp = soft_vote(model(x_dropped))
                preds[(k, off)].extend(gp.cpu().tolist())

    labels_arr = np.array(labels)
    out: dict[int, dict[str, float]] = {}
    for k in k_levels:
        n_off = 1 if k == 0 else n_mask_seeds
        per_off = [
            float(f1_score(labels_arr, np.array(preds[(k, off)]),
                           average=f1_average, zero_division=0))
            for off in range(n_off)
        ]
        out[k] = {
            "f1": float(np.mean(per_off)),
            "f1_mask_std": float(np.std(per_off, ddof=1)) if len(per_off) > 1 else 0.0,
            "f1_per_mask_seed": per_off,
            "n_samples": int(len(labels_arr)),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/fullscale_final")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/dropout_study_correlated")
    p.add_argument("--models", nargs="+", default=["rgsage_max"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
    p.add_argument("--variant", default="standard",
                   help="Label recorded in the output (standard | dropout_aware).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Smoke-test only: stop after N batches.")
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    data_root = (repo / args.data_root).resolve()
    results_dir = (repo / args.results_dir).resolve()
    out_dir = (repo / args.output_dir).resolve()

    device = torch.device(args.device)
    test_ds = WindowGraphDataset(str(data_root), "test")
    test_loader = make_dataloader(test_ds, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
    adjacency = build_adjacency(test_ds.edge_index, n_nodes=25)
    degrees = [len(a) for a in adjacency]
    print(f"[graph] 25 nodes, degrees min={min(degrees)} max={max(degrees)} "
          f"sum={sum(degrees)} (= 2 x |E|)", flush=True)

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
                  f"correlated k={args.k_levels} ...", flush=True)

            curve = evaluate_correlated_curve(
                model, test_loader, device, args.k_levels, adjacency,
                args.n_mask_seeds, f1_avg, args.max_batches,
            )
            for k in args.k_levels:
                print(f"  [correlated] {model_name} seed={seed} k={k}: "
                      f"F1={curve[k]['f1']:.4f}", flush=True)
                rows.append({
                    "model": model_name, "variant": args.variant, "seed": seed,
                    "k": k, "f1_seed": curve[k]["f1"],
                    "f1_mask_std": curve[k]["f1_mask_std"],
                    "f1_per_mask_seed": json.dumps(
                        [round(x, 6) for x in curve[k]["f1_per_mask_seed"]]),
                    "policy": POLICY,
                })

    if not rows:
        print("No models evaluated (no checkpoints found).")
        return 1

    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per_seed_path = out_dir / f"correlated_per_seed_{args.variant}.csv"
    df.to_csv(per_seed_path, index=False)

    agg = []
    for (model_name, k), g in df.groupby(["model", "k"], sort=True):
        f1s = g["f1_seed"].to_numpy()
        agg.append({
            "model": model_name, "variant": args.variant, "k": int(k),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "n_train_seeds": int(len(f1s)),
            "n_mask_seeds": args.n_mask_seeds,
            "policy": POLICY, "err_kind": "train_seed_std",
            "f1_per_train_seed": json.dumps([round(float(v), 6) for v in f1s]),
        })
    agg_df = pd.DataFrame(agg).sort_values(["model", "k"])
    agg_path = out_dir / f"correlated_curves_{args.variant}.csv"
    agg_df.to_csv(agg_path, index=False)

    print(f"\n=== Correlated (contiguous) dropout ({args.variant}), "
          f"seeds {args.seeds} ===")
    for r in agg:
        print(f"  {r['model']:<12} k={r['k']:>2}: F1 = "
              f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}")
    print(f"\nWrote {per_seed_path}\n      {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
