"""Classical (non-deep, non-graph) baselines on the same corpus and splits.

The paper's only non-spatial reference is the GRU, which is still a deep
sequence model.  That leaves an obvious question unanswered: how much of the
reported accuracy needs a spatio-temporal GNN at all, as opposed to a
conventional classifier over post-fault RMS features of the kind Livani et
al.\\ use for faulted-line identification?  This script answers it on identical
data, identical splits, and the identical dropout sweep, so the comparison is
like for like.

Feature construction follows the classical recipe rather than the deep one:
each 20-step window is reduced, per $\\mu$PMU and per voltage phase, to the
minimum, mean, maximum, and standard deviation over the window, concatenated
with that node's three availability-mask values.  That is
25 nodes x (3 phases x 4 statistics + 3 mask) = 375 features per window.  The
minimum and the drop from the pre-fault level are exactly what an
impedance-style method keys on, so this is a fair rather than a strawman
encoding.

The classifier consumes all 25 nodes at once and emits the window's class
directly, so it needs no soft voting; it therefore sees the same information
the GNN sees, minus the topology.  Dropout is applied with the same
``mask_seed = 10000*k + offset`` scheme as ``evaluate_dropout_curve``, before
feature extraction, so a dropped node contributes zeroed statistics and a
(0,0,0) mask exactly as it does for the GNN.

CPU only -- it does not contend with GPU training.

Usage:
    uv run python scripts/run_classical_baseline.py --train-subsample 150000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.masking import apply_sensor_dropout_batch


def featurize(x: torch.Tensor) -> np.ndarray:
    """[B, N, S, 6] -> [B, N*15] classical summary features."""
    volts = x[..., :3]                      # [B, N, S, 3]
    mask = x[..., 0, 3:]                    # [B, N, 3] (static over the window)
    feats = torch.cat([
        volts.min(dim=2).values,
        volts.mean(dim=2),
        volts.max(dim=2).values,
        volts.std(dim=2),
        mask,
    ], dim=-1)                              # [B, N, 15]
    return feats.reshape(feats.shape[0], -1).cpu().numpy()


def collect(loader, combos, max_batches=None, train_mode=False):
    """One pass over the loader, featurizing at each explicit (k, offset).

    ``combos`` is passed explicitly rather than derived from a level list so a
    caller can request one specific offset — deriving it would silently reuse
    offset 0 and make every mask realization identical.
    """
    feats: dict[tuple[int, int], list[np.ndarray]] = {c: [] for c in combos}
    labels: list[int] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch["x"]
        labels.extend(batch["y"].tolist())
        for (k, off) in combos:
            if k == 0:
                feats[(k, off)].append(featurize(x))
            else:
                mask_seed = 10000 * k + off
                xd, _, _ = apply_sensor_dropout_batch(
                    x, k=k, seed=mask_seed * 10000 + batch_idx,
                    policy="zero_messages", edge_index=None)
                feats[(k, off)].append(featurize(xd))
        if train_mode and batch_idx % 500 == 0:
            print(f"    ... train batch {batch_idx}", flush=True)

    return (np.array(labels),
            {c: np.concatenate(v, axis=0) for c, v in feats.items()})


def main() -> int:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/classical_baseline")
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
    p.add_argument("--train-subsample", type=int, default=150_000,
                   help="Windows used to fit (the full split is 1.155M).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test-batches", type=int, default=None)
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = (repo / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = WindowGraphDataset(str(repo / args.data_root), "train")
    test_ds = WindowGraphDataset(str(repo / args.data_root), "test")
    n_train_batches = max(1, args.train_subsample // args.batch_size)

    print(f"[fit] featurizing ~{n_train_batches * args.batch_size:,} training "
          f"windows (of {len(train_ds):,}) ...", flush=True)
    t0 = time.time()
    # shuffle=True is essential, not cosmetic: the event-chunk sampler emits
    # windows in event order, and one event carries a single label, so the
    # first N unshuffled batches cover only a couple of classes.  Shuffling
    # randomizes the event order, so a prefix of N batches is a random sample
    # of events and covers all 26 classes.
    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True,
                                   seed=args.seed, num_workers=0)
    y_tr, f_tr = collect(train_loader, [(0, 0)], max_batches=n_train_batches,
                         train_mode=True)
    X_tr = f_tr[(0, 0)]
    print(f"[fit] features {X_tr.shape} in {(time.time()-t0)/60:.1f} min; "
          f"{len(set(y_tr.tolist()))} classes present", flush=True)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)

    models = {
        # Multinomial is the default for multi-class with lbfgs in current
        # scikit-learn; the explicit multi_class argument has been removed.
        "logreg": LogisticRegression(max_iter=300, n_jobs=args.n_jobs),
        # 100 trees with a leaf floor of 5 keeps the fitted forest to a few
        # hundred MB on 150k x 375 inputs; the machine has ~7 GB free and GPU
        # training is running alongside.
        "random_forest": RandomForestClassifier(
            n_estimators=100, n_jobs=args.n_jobs, random_state=args.seed,
            min_samples_leaf=5),
    }
    fitted = {}
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_tr_s if name == "logreg" else X_tr, y_tr)
        fitted[name] = clf
        print(f"[fit] {name} in {(time.time()-t0)/60:.1f} min", flush=True)

    # One (k, offset) at a time, each its own pass over the split.  Holding all
    # 13 combos in memory at once would need ~4.8 GB of float32 features for the
    # full 247k-window test split; this holds one combo (~370 MB) and a pass
    # costs only a minute or two, since featurization is cheap next to the
    # per-event decompression that dominates either way.
    scores: dict[tuple[str, int], list[float]] = {}
    for k in args.k_levels:
        n_off = 1 if k == 0 else args.n_mask_seeds
        for off in range(n_off):
            t0 = time.time()
            test_loader = make_dataloader(test_ds, args.batch_size,
                                          shuffle=False, num_workers=0)
            y_te, f_te = collect(test_loader, [(k, off)],
                                 max_batches=args.max_test_batches)
            X = f_te[(k, off)]
            X_scaled = scaler.transform(X)
            for name, clf in fitted.items():
                pred = clf.predict(X_scaled if name == "logreg" else X)
                f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
                scores.setdefault((name, k), []).append(f1)
            print(f"  [pass] k={k} off={off} in {(time.time()-t0)/60:.1f} min "
                  f"(n={len(y_te):,})", flush=True)
            del X, X_scaled, f_te

    rows = []
    for (name, k), per_off in sorted(scores.items()):
        rows.append({
            "model": name, "k": k,
            "f1_mean": float(np.mean(per_off)),
            "f1_std": float(np.std(per_off, ddof=1)) if len(per_off) > 1 else 0.0,
            "f1_per_mask_seed": json.dumps([round(v, 6) for v in per_off]),
            "n_train_windows": int(len(y_tr)),
            "policy": "zero_messages",
        })
        print(f"  {name:<14} k={k:>2}: macro-F1 = {rows[-1]['f1_mean']:.4f}",
              flush=True)

    import pandas as pd
    path = out_dir / "classical_curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nWrote {path}")
    print("\nCompare k=0 against Table 3's GNN column: the gap is what the "
          "spatio-temporal graph model buys over a conventional classifier "
          "given the same measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
