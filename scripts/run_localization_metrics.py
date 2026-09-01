"""Decompose the degradation curve into detection and localization.

The headline metric everywhere in this project is macro-F1 over 26 classes:
25 fault locations plus no-fault.  The corpus is exactly 50 % no-fault and
detection is effectively saturated (no-fault recall >= 0.9999 at k=0), so
class 0 contributes a near-perfect F1 to every average.  On a 26-class macro
that is worth up to 1/26 = 0.038 of "free" score, which barely matters at
k=0 (0.853 vs 0.847 for RGSAGE-Max) but dominates the interpretation at the
severe end of the sweep, where the whole macro-F1 is 0.048.

This script re-runs the standard random-dropout sweep and reports three
quantities per (model, seed, k) instead of one:

  * ``f1_macro26``  — the paper's metric, reproduced as a cross-check.
  * ``f1_macro25``  — the 25 fault classes only, i.e. localization proper.
  * ``f1_nofault``  — class 0 alone, i.e. detection.

Together they separate "the model can still tell that something happened" from
"the model can still say where", which a single macro-F1 conflates.  Predictions
are written to an .npz per (model, variant, seed) so any further breakdown can
be computed without another GPU pass.

The k=0 row is the standing sanity check of ``.rules/10-integrity.md``:
``f1_macro26`` at k=0 must equal that checkpoint's recorded clean test F1.

Usage:
    uv run python scripts/run_localization_metrics.py \
        --results-dir results/fullscale_final --variant standard \
        --models rgsage_max --seeds 42 43 44
    uv run python scripts/run_localization_metrics.py \
        --results-dir results/dropout_aware --variant dropout_aware \
        --models rgsage_max --seeds 42 43 44
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
from fault_localization.models.voting import soft_vote

from run_dropout_study import load_trained_model  # noqa: E402

# A classifier that always answers "no fault" on the balanced corpus:
# precision 0.5, recall 1.0 -> F1 = 2/3 on class 0, zero on the other 25.
TRIVIAL_NOFAULT_MACRO26 = (2 * 0.5 * 1.0 / 1.5) / 26


def split_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Macro-F1 over all 26 classes, over the 25 fault classes, and class 0."""
    from sklearn.metrics import f1_score

    per_class = f1_score(labels, preds, average=None,
                         labels=list(range(26)), zero_division=0)
    return {
        "f1_macro26": float(np.mean(per_class)),
        "f1_macro25": float(np.mean(per_class[1:])),
        "f1_nofault": float(per_class[0]),
    }


def evaluate_split(model, loader, device, k_levels, n_mask_seeds, max_batches):
    """Single pass over the loader, collecting predictions for every (k, offset)."""
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
                else:
                    mask_seed = 10000 * k + off
                    x_dropped, _, _ = apply_sensor_dropout_batch(
                        x, k=k, seed=mask_seed * 10000 + batch_idx,
                        policy="zero_messages", edge_index=None,
                    )
                    _, gp = soft_vote(model(x_dropped))
                preds[(k, off)].extend(gp.cpu().tolist())

    return np.array(labels), {c: np.array(v) for c, v in preds.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/fullscale_final")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/dropout_study_localization")
    p.add_argument("--models", nargs="+", default=["rgsage_max"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
    p.add_argument("--variant", default="standard")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    # Predictions are saved by default so any further breakdown (per-class,
    # event-level bootstrap) needs no second GPU pass; ~26 MB per run.
    p.add_argument("--no-save-predictions", action="store_false",
                   dest="save_predictions", default=True)
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    data_root = (repo / args.data_root).resolve()
    results_dir = (repo / args.results_dir).resolve()
    out_dir = (repo / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

            model, _ = load_trained_model(seed_dir, test_ds.edge_index, device)
            print(f"[eval] {model_name} ({args.variant}) seed={seed} ...", flush=True)
            labels, preds = evaluate_split(model, test_loader, device,
                                           args.k_levels, args.n_mask_seeds,
                                           args.max_batches)

            if args.save_predictions:
                npz = out_dir / f"preds_{model_name}_{args.variant}_seed{seed}.npz"
                np.savez_compressed(
                    npz, labels=labels,
                    **{f"k{k}_off{off}": v for (k, off), v in preds.items()},
                )

            recorded = json.loads((seed_dir / "metrics.json").read_text())
            recorded_clean = recorded["test_graph"]["f1"]

            for k in args.k_levels:
                n_off = 1 if k == 0 else args.n_mask_seeds
                per_off = [split_metrics(labels, preds[(k, off)])
                           for off in range(n_off)]
                agg = {key: float(np.mean([m[key] for m in per_off]))
                       for key in ("f1_macro26", "f1_macro25", "f1_nofault")}

                if k == 0 and args.max_batches is None:
                    delta = abs(agg["f1_macro26"] - recorded_clean)
                    status = "OK" if delta < 1e-3 else "MISMATCH"
                    print(f"  [sanity] k=0 macro26={agg['f1_macro26']:.4f} vs "
                          f"recorded {recorded_clean:.4f} -> {status}", flush=True)
                    if status == "MISMATCH":
                        raise SystemExit(
                            "k=0 F1 does not match the recorded clean F1; the "
                            "loader or checkpoint path is wrong (.rules/10-integrity.md)."
                        )

                print(f"  {model_name} seed={seed} k={k:>2}: "
                      f"macro26={agg['f1_macro26']:.4f} "
                      f"macro25={agg['f1_macro25']:.4f} "
                      f"nofault={agg['f1_nofault']:.4f}", flush=True)
                rows.append({
                    "model": model_name, "variant": args.variant, "seed": seed,
                    "k": k, **agg,
                    "macro26_over_trivial": agg["f1_macro26"] / TRIVIAL_NOFAULT_MACRO26,
                    "policy": "zero_messages",
                })

    if not rows:
        print("No models evaluated (no checkpoints found).")
        return 1

    import pandas as pd
    df = pd.DataFrame(rows)
    per_seed_path = out_dir / f"localization_per_seed_{args.variant}.csv"
    df.to_csv(per_seed_path, index=False)

    agg_rows = []
    for (model_name, k), g in df.groupby(["model", "k"], sort=True):
        entry = {"model": model_name, "variant": args.variant, "k": int(k),
                 "n_train_seeds": int(len(g))}
        for key in ("f1_macro26", "f1_macro25", "f1_nofault"):
            vals = g[key].to_numpy()
            entry[f"{key}_mean"] = float(vals.mean())
            entry[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        entry["trivial_macro26"] = round(TRIVIAL_NOFAULT_MACRO26, 6)
        agg_rows.append(entry)
    agg_path = out_dir / f"localization_curves_{args.variant}.csv"
    pd.DataFrame(agg_rows).sort_values(["model", "k"]).to_csv(agg_path, index=False)

    print(f"\nTrivial always-no-fault 26-class macro-F1 = "
          f"{TRIVIAL_NOFAULT_MACRO26:.4f}")
    print(f"Wrote {per_seed_path}\n      {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
