"""Persistent (per-event) sensor-dropout evaluation.

Companion to ``run_dropout_study.py``.  That script redraws the dropped μPMU set
for every window; this one draws it **once per event** and holds it across the
event's 40 windows, which is how a real μPMU failure behaves.  Everything else —
checkpoints, k grid, zeroing semantics, soft-vote metric — is identical, so the
difference between the two curves isolates the effect of mask persistence.

Selection is a deterministic function of ``(event_id, k, mask_seed)`` (see
``fault_localization/dropout/persistent.py``), so all models see identical masks
without a shared per-batch seed.  Error bars are reported over training seeds
42/43/44; the mask seeds are averaged within each training seed, mirroring the
random sweep's ``n_mask_seeds=3``.

Usage:
    uv run python scripts/run_persistent_dropout_study.py \
        --results-dir results/fullscale_final \
        --data-root dataset_fullscale_50 \
        --models rgsage_max --seeds 42 43 44 \
        --output-dir results/dropout_study_persistent

For the dropout-aware checkpoints, point --results-dir at results/dropout_aware
and pass --variant dropout_aware.
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
from fault_localization.dropout.persistent import apply_persistent_dropout_batch
from fault_localization.models.voting import soft_vote

from run_dropout_study import load_trained_model  # noqa: E402

MODELS_DEFAULT = ["rgsage_max"]


def evaluate_persistent_curve(
    model, loader, device, k_levels, n_mask_seeds=3, f1_average="macro",
) -> dict[int, dict[str, float]]:
    """Evaluate one model across persistent-dropout levels in a single pass.

    Mirrors ``evaluate_dropout_curve``'s single-pass structure: the test split
    lives in per-event compressed .npz files, so looping (k, mask_seed) *inside*
    the batch loop decompresses the split once instead of
    len(k_levels) * n_mask_seeds times.
    """
    from sklearn.metrics import f1_score

    combos = [(k, off) for k in k_levels for off in range(n_mask_seeds)]
    preds: dict[tuple[int, int], list[int]] = {c: [] for c in combos}
    labels: list[int] = []
    # Shared across batches so each event is hashed once per (k, mask_seed).
    cache: dict[tuple[str, int, int], list[int]] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            wids = batch["window_id"]
            labels.extend(batch["y"].tolist())

            # k=0 is dropout-free, so every mask seed yields the same
            # prediction: compute it once and copy.
            zero_pred: list[int] | None = None
            for (k, off) in combos:
                if k == 0:
                    if zero_pred is None:
                        _, gp = soft_vote(model(x))
                        zero_pred = gp.cpu().tolist()
                    preds[(k, off)].extend(zero_pred)
                    continue

                x_dropped, _ = apply_persistent_dropout_batch(
                    x, wids, k=k, mask_seed=off, cache=cache,
                )
                _, gp = soft_vote(model(x_dropped))
                preds[(k, off)].extend(gp.cpu().tolist())

    out: dict[int, dict[str, float]] = {}
    labels_arr = np.array(labels)
    for k in k_levels:
        f1s, accs = [], []
        for off in range(n_mask_seeds):
            p = np.array(preds[(k, off)])
            f1s.append(float(f1_score(labels_arr, p, average=f1_average,
                                      zero_division=0)))
            accs.append(float(np.mean(p == labels_arr)))
        out[k] = {
            "f1": float(np.mean(f1s)),
            "f1_mask_std": float(np.std(f1s)),
            "f1_per_mask_seed": [round(v, 6) for v in f1s],
            "acc": float(np.mean(accs)),
            "n_samples": int(len(labels_arr)),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/fullscale_final")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/dropout_study_persistent")
    p.add_argument("--models", nargs="+", default=MODELS_DEFAULT)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
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
                  f"persistent k={args.k_levels} x {args.n_mask_seeds} masks ...",
                  flush=True)

            curve = evaluate_persistent_curve(
                model, test_loader, device, args.k_levels,
                n_mask_seeds=args.n_mask_seeds, f1_average=f1_avg,
            )
            for k in args.k_levels:
                print(f"  [persistent] {model_name} seed={seed} k={k}: "
                      f"F1={curve[k]['f1']:.4f}", flush=True)
                rows.append({
                    "model": model_name, "variant": args.variant, "seed": seed,
                    "k": k, "f1_seed": curve[k]["f1"],
                    "f1_mask_std": curve[k]["f1_mask_std"],
                    "f1_per_mask_seed": json.dumps(curve[k]["f1_per_mask_seed"]),
                    "acc_seed": curve[k]["acc"],
                    "policy": "persistent_per_event",
                })

    if not rows:
        print("No models evaluated (no checkpoints found).")
        return 1

    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per_seed_path = out_dir / f"persistent_per_seed_{args.variant}.csv"
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
            "n_mask_seeds": args.n_mask_seeds,
            "policy": "persistent_per_event",
            "err_kind": "train_seed_std",
            "f1_per_train_seed": json.dumps([round(float(v), 6) for v in f1s]),
        })
    agg_df = pd.DataFrame(agg).sort_values(["model", "k"])
    agg_path = out_dir / f"persistent_curves_{args.variant}.csv"
    agg_df.to_csv(agg_path, index=False)

    print(f"\n=== Persistent per-event dropout ({args.variant}), "
          f"seeds {args.seeds} ===")
    for r in agg:
        print(f"  {r['model']:<12} k={r['k']:>2}: F1 = "
              f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}")
    print(f"\nWrote {per_seed_path}\n      {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
