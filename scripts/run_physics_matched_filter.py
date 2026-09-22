"""Physics-only matched filter — a training-free fault-localization baseline.

What this is
------------
The classical impedance method, made explicit and evaluated under exactly the
protocol every learned model in this repo is evaluated under. It has **zero
learned parameters** and sees **no training data**.

On a radial feeder the sag at sensor ``i`` from a fault at bus ``b`` factorizes
as ``dv_i ~= alpha * s_{b,i}`` (``graph/sensitivity.py``): a severity scalar
``alpha`` absorbing fault resistance and loading, times a fixed spatial
signature ``s_b``. So the *direction* of the observed deviation in sensor space
identifies the fault; its magnitude does not. The decision rule is therefore::

    d, a  = observed deviation and availability over the 25 sensors
    if ||d||_a < tau :  predict 0 (no fault)
    else             :  predict  argmax_c  cos_a(d, s_c)

Only ``tau`` is fitted, and it is fitted on the **validation** split, never on
test. One scalar, chosen on held-out data — this is a baseline, not a model.

Why it belongs in the paper
---------------------------
The existing baselines answer "how much does a graph network beat a linear
model?" (logreg 0.713 vs RGSAGE-Max 0.846). They do not answer "how much does
*any* learning beat the physics we already know?", which is the question a
reviewer of a physics-informed paper will ask first. This supplies that number,
and it is also the honest floor against which the P1/P2/P3 screening arms should
be read: an arm that merely reaches this score has not learned anything the
impedance model did not already supply.

It is evaluated at every dropout level too, because a parameter-free rule that
degrades gracefully would be a genuinely useful result under sensor loss --
masked cosine simply drops terms from the sum rather than feeding a network
zeros it was never trained on.

Output: results/physics_baseline/matched_filter.json  (+ .csv per k-level)

Usage:
    uv run python scripts/run_physics_matched_filter.py
    uv run python scripts/run_physics_matched_filter.py --k-levels 0 1 2 5 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.masking import apply_sensor_dropout_batch
from fault_localization.dropout.noise import apply_measurement_noise_batch
from fault_localization.graph.sensitivity import load_sensitivity
from fault_localization.models.physics_loss import masked_cosine, observed_deviation
from sklearn.metrics import f1_score

LABELS = list(range(26))


def collect(loader, signatures, k, mask_seed, sigma_pu=0.0, norm_std=None):
    """One pass: returns (energy, best_class, y) over the split.

    Dropout is applied before noise, matching run_noise_robustness_study.py:
    noise is mask-gated, so a dropped node stays exactly zero either way.
    """
    energies, best, ys = [], [], []
    for bi, batch in enumerate(loader):
        x, y = batch["x"], batch["y"]
        if k > 0:
            # Per-batch seed derivation matches dropout/evaluator.py post-bugfix:
            # a constant seed would repeat the same node subset for every batch.
            x = apply_sensor_dropout_batch(x, k=k, seed=mask_seed * 10000 + bi)[0]
        if sigma_pu > 0:
            x = apply_measurement_noise_batch(x, sigma_pu, norm_std,
                                              seed=mask_seed * 10000 + bi)
        d, a = observed_deviation(x)
        cos = masked_cosine(d, signatures, a)          # [B, 25]
        energies.append((d * a).norm(dim=-1))
        best.append(cos.argmax(dim=-1) + 1)
        ys.append(y)
    return (torch.cat(energies).numpy(), torch.cat(best).numpy(),
            torch.cat(ys).numpy())


def predict(energy, best, tau):
    return np.where(energy < tau, 0, best)


def fit_tau(energy, best, y):
    """Choose the no-fault threshold on validation macro-F1."""
    grid = np.quantile(energy, np.linspace(0.01, 0.99, 99))
    scores = [f1_score(y, predict(energy, best, t), average="macro",
                       labels=LABELS, zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--mask-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sigma-pu", type=float, nargs="+", default=[0.0],
                   help="measurement-noise levels in per-unit voltage; the "
                        "existing study grid is 0 0.0005 0.002 0.01")
    p.add_argument("--out", default="results/physics_baseline")
    args = p.parse_args()

    sig = torch.as_tensor(load_sensitivity()["sensitivity"], dtype=torch.float32)
    norm_std = np.asarray(json.loads(
        Path(args.data_root, "processed/norm_stats.json").read_text())["std"])
    val_ds = WindowGraphDataset(args.data_root, "val")
    test_ds = WindowGraphDataset(args.data_root, "test")

    def loader(ds):
        return make_dataloader(ds, args.batch_size, shuffle=False)

    # tau is fitted at k=0 on validation and then held FIXED for every k, so
    # the degradation curve measures the rule degrading, not the rule being
    # refitted at each dropout level.
    ev, bv, yv = collect(loader(val_ds), sig, 0, 0)
    tau, val_f1 = fit_tau(ev, bv, yv)
    print(f"tau fitted on val: {tau:.5f}  (val macro-F1 {val_f1:.4f})", flush=True)

    rows, summary = [], {"tau": tau, "val_f1_k0": val_f1,
                         "policy": "physics_matched_filter",
                         "n_learned_parameters": 0, "curve": {}}
    for sigma in args.sigma_pu:
        for k in args.k_levels:
            seeds = [0] if (k == 0 and sigma == 0) else args.mask_seeds
            f1s, loc_f1s = [], []
            for ms in seeds:
                e, b, y = collect(loader(test_ds), sig, k, ms, sigma, norm_std)
                pred = predict(e, b, tau)
                f1s.append(f1_score(y, pred, average="macro", labels=LABELS,
                                    zero_division=0))
                fault = y > 0
                loc_f1s.append(float((pred[fault] == y[fault]).mean()))
                rows.append({"sigma_pu": sigma, "k": k, "mask_seed": ms,
                             "f1": f1s[-1], "fault_window_accuracy": loc_f1s[-1]})
            summary["curve"][f"sigma{sigma}_k{k}"] = {
                "sigma_pu": sigma, "k": k,
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "f1_per_mask_seed": [float(v) for v in f1s],
                "fault_window_accuracy_mean": float(np.mean(loc_f1s)),
                "n_mask_seeds": len(seeds),
            }
            print(f"sigma={sigma:<7} k={k:2d}: macro-F1 {np.mean(f1s):.4f}  "
                  f"fault-window acc {np.mean(loc_f1s):.4f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_filter.json").write_text(json.dumps(summary, indent=2) + "\n")
    import csv as _csv
    with (out / "matched_filter_curve.csv").open("w", newline="") as h:
        w = _csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}/matched_filter.json + matched_filter_curve.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
