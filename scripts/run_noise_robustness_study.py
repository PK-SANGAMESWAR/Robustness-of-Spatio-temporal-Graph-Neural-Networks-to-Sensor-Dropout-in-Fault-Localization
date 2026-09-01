"""Sensor dropout under additive measurement noise.

Every accuracy in this project is measured on voltages taken straight from
OpenDSS, so the full-observability scores are optimistic relative to field
μPMU data and the dropout curves describe a grid that is *incomplete* but never
*noisy*.  This script crosses the two: it re-runs the inference-time dropout
sweep with zero-mean Gaussian noise added to the available voltage channels, at
noise levels bracketing the 1 %-of-nominal total vector error that IEEE
C37.118 permits.

Nothing is retrained.  The frozen checkpoints of ``results/fullscale_final``
(standard) and ``results/dropout_aware`` are evaluated as they are, so the
question answered is the deployment question: does the robustness that
dropout-aware training buys survive on inputs that are also degraded?

Order of operations per batch is dropout first, then noise.  Noise is gated by
the availability mask (``dropout/noise.py``), so a dropped node — all six
channels zeroed — receives none, and the masking under test is never partially
undone by the perturbation.

Usage:
    uv run python scripts/run_noise_robustness_study.py \
        --results-dir results/fullscale_final --variant standard \
        --models rgsage_max --seeds 42 43 44 \
        --output-dir results/dropout_study_noise

    uv run python scripts/run_noise_robustness_study.py \
        --results-dir results/dropout_aware --variant dropout_aware \
        --models rgsage_max --seeds 42 43 44 \
        --output-dir results/dropout_study_noise
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
from fault_localization.dropout.masking import apply_sensor_dropout_batch
from fault_localization.dropout.noise import apply_measurement_noise_batch
from fault_localization.models.voting import soft_vote

from run_dropout_study import load_trained_model  # noqa: E402

POLICY = "noise_plus_zero_messages"


def evaluate_noise_grid(
    model, loader, device, k_levels, sigmas, norm_std,
    n_mask_seeds: int = 3, f1_average: str = "macro", max_batches: int | None = None,
) -> dict[tuple[float, int], dict[str, float]]:
    """One pass over the loader covering every (sigma, k, mask-offset) combo.

    Structured like ``evaluate_dropout_curve``: the test split lives in
    per-event compressed .npz files, so the pass is dominated by decompression
    and every combo must be evaluated inside the batch loop rather than by
    re-reading the split once per combo.

    The dropout mask for a given (k, offset, batch) uses the same
    ``mask_seed = 10000*k + offset`` scheme as the random sweep, so at
    ``sigma = 0`` this reproduces ``results/dropout_study`` exactly and the
    noise arm is paired with it sensor-for-sensor.
    """
    from sklearn.metrics import f1_score

    # k=0 is dropout-free, so its mask offsets are all the same draw; evaluating
    # one of them saves n_mask_seeds-1 forward passes per sigma on every batch.
    # This also means the k=0 row carries a single noise realization rather than
    # n_mask_seeds of them, so it has no mask-seed error bar.  That is
    # deliberate: one realization already averages over the ~247k windows of the
    # test split, which swamps the draw-to-draw variation, and the reported error
    # bar is over training seeds in any case.
    combos = [(k, off) for k in k_levels
              for off in range(1 if k == 0 else n_mask_seeds)]
    preds: dict[tuple[float, int, int], list[int]] = {
        (s, k, off): [] for s in sigmas for (k, off) in combos
    }
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
                    x_dropped = x
                else:
                    mask_seed = 10000 * k + off
                    x_dropped, _, _ = apply_sensor_dropout_batch(
                        x, k=k, seed=mask_seed * 10000 + batch_idx,
                        policy="zero_messages", edge_index=None,
                    )

                for s_idx, s in enumerate(sigmas):
                    # Explicit deterministic mix rather than hash(): the noise
                    # stream must be reproducible across processes and must not
                    # collide with the dropout mask stream.
                    noise_seed = (((s_idx + 1) * 97 + k) * 131 + off) * 100003 + batch_idx
                    x_in = apply_measurement_noise_batch(
                        x_dropped, sigma_pu=s, norm_std=norm_std,
                        seed=noise_seed,
                    )
                    _, gp = soft_vote(model(x_in))
                    preds[(s, k, off)].extend(gp.cpu().tolist())

    labels_arr = np.array(labels)
    out: dict[tuple[float, int], dict[str, float]] = {}
    for s in sigmas:
        for k in k_levels:
            n_off = 1 if k == 0 else n_mask_seeds
            per_off = [
                float(f1_score(labels_arr, np.array(preds[(s, k, off)]),
                               average=f1_average, zero_division=0))
                for off in range(n_off)
            ]
            out[(s, k)] = {
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
    p.add_argument("--output-dir", default="results/dropout_study_noise")
    p.add_argument("--models", nargs="+", default=["rgsage_max"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--sigma-pu", type=float, nargs="+", default=[0.0, 0.005, 0.01],
                   help="Noise std in per-unit voltage; 0.01 = 1%% of nominal.")
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
    norm_std = test_ds.std

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
                  f"sigma={args.sigma_pu} k={args.k_levels} ...", flush=True)

            grid = evaluate_noise_grid(
                model, test_loader, device, args.k_levels, args.sigma_pu,
                norm_std, args.n_mask_seeds, f1_avg, args.max_batches,
            )
            for (s, k), v in sorted(grid.items()):
                print(f"  [noise] {model_name} seed={seed} sigma={s} k={k}: "
                      f"F1={v['f1']:.4f}", flush=True)
                rows.append({
                    "model": model_name, "variant": args.variant, "seed": seed,
                    "sigma_pu": s, "k": k, "f1_seed": v["f1"],
                    "f1_mask_std": v["f1_mask_std"],
                    "f1_per_mask_seed": json.dumps(
                        [round(x, 6) for x in v["f1_per_mask_seed"]]),
                    "policy": POLICY,
                })

    if not rows:
        print("No models evaluated (no checkpoints found).")
        return 1

    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per_seed_path = out_dir / f"noise_per_seed_{args.variant}.csv"
    df.to_csv(per_seed_path, index=False)

    agg = []
    for (model_name, s, k), g in df.groupby(["model", "sigma_pu", "k"], sort=True):
        f1s = g["f1_seed"].to_numpy()
        agg.append({
            "model": model_name, "variant": args.variant,
            "sigma_pu": float(s), "k": int(k),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "n_train_seeds": int(len(f1s)),
            "n_mask_seeds": args.n_mask_seeds,
            "policy": POLICY, "err_kind": "train_seed_std",
            "f1_per_train_seed": json.dumps([round(float(v), 6) for v in f1s]),
        })
    agg_df = pd.DataFrame(agg).sort_values(["model", "sigma_pu", "k"])
    agg_path = out_dir / f"noise_curves_{args.variant}.csv"
    agg_df.to_csv(agg_path, index=False)

    print(f"\n=== Noise x dropout ({args.variant}), seeds {args.seeds} ===")
    for r in agg:
        print(f"  {r['model']:<12} sigma={r['sigma_pu']:<6} k={r['k']:>2}: "
              f"F1 = {r['f1_mean']:.3f} ± {r['f1_std']:.3f}")
    print(f"\nWrote {per_seed_path}\n      {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
