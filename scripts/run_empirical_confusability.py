"""S2.2 -- empirical confusability (``04_STAGE_C_gonogo.md`` S2.2), independent of the operator
table: leave-one-out 1-nearest-neighbour classification on raw observation vectors from the full
training split, compared against the table's own separability D (S2.1, F-C1) at k=0.

Observation vector per event: the settled post-fault snapshot (step 45, matching
``scripts/check_s1_7_discretization.py`` and ``physics/observation.py``'s "last timestep is the
settled level" convention), restricted to the 55 actually-instrumented channels
(``operator/mask.npy``) -- unmeasured channels are dropped, not zero-padded, so they cannot bias
the Euclidean distance.

For each event, find its nearest OTHER event by raw Euclidean distance (scipy cKDTree; the
corpus is noise-free within a scenario, so ties within a scenario are excluded by construction --
run_id is not part of the label). Confusion matrix C[i,j] = P(nearest neighbour's fault_bus = j |
own fault_bus = i), 25x25 (no-fault has no per-event instances at this granularity, matching
S1.4-S1.7's convention). Compare the symmetrised C against D_global_k0 from S2.1 (F-C3 scatter).

Usage:
    uv run python scripts/run_empirical_confusability.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_BUSES  # noqa: E402

SETTLED_STEP = 45
OUT_DIR = REPO / "results/pcstgnn/audit/s2_2_confusability"
D_GLOBAL_K0_PATH = REPO / "results/pcstgnn/audit/s2_1_identifiability/D_global_k0.npy"


def load_train_events(data_root: Path) -> list[Path]:
    stems = set()
    with (data_root / "splits/train.txt").open() as fh:
        for line in fh:
            stems.add(line.strip().rsplit("_w", 1)[0])
    raw_dir = data_root / "raw_sim"
    return sorted(p for p in raw_dir.glob("*.npz") if p.stem in stems)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mask = np.load(REPO / "operator/mask.npy")  # [25,3]
    mask_flat = mask.reshape(-1) > 0.5
    n_avail = int(mask_flat.sum())

    data_root = REPO / "dataset_fullscale_50"
    paths = load_train_events(data_root)
    print(f"[load] {len(paths)} training events, {n_avail} instrumented channels", flush=True)

    X = np.empty((len(paths), n_avail), dtype=np.float64)
    y = np.empty(len(paths), dtype=np.int64)
    for i, p in enumerate(paths):
        with np.load(p, allow_pickle=True) as loaded:
            rec = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        snap = rec[:, SETTLED_STEP, :3].reshape(-1)  # [75]
        X[i] = snap[mask_flat]
        y[i] = FAULT_BUSES.index(meta["fault_bus"])
    print(f"[observe] X={X.shape} ({time.time()-t0:.1f}s)", flush=True)

    t1 = time.time()
    tree = cKDTree(X)
    # query 2 neighbours: [0] is the point itself (dist 0), [1] is the true nearest other point
    dist, idx = tree.query(X, k=2, workers=-1)
    nn_idx = idx[:, 1]
    nn_dist = dist[:, 1]
    print(f"[knn] done ({time.time()-t1:.1f}s)", flush=True)

    y_pred = y[nn_idx]
    n_classes = len(FAULT_BUSES)
    C = np.zeros((n_classes, n_classes), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for true_c, pred_c in zip(y, y_pred):
        C[true_c, pred_c] += 1
        counts[true_c] += 1
    C_norm = C / counts[:, None]

    overall_accuracy = float((y == y_pred).mean())
    off_diag_confusion_rate = float(1.0 - overall_accuracy)

    C_sym = 0.5 * (C_norm + C_norm.T)

    result = {
        "n_events": len(paths),
        "n_instrumented_channels": n_avail,
        "settled_step": SETTLED_STEP,
        "nn_overall_accuracy": overall_accuracy,
        "nn_confusion_rate": off_diag_confusion_rate,
        "nn_dist_median": float(np.median(nn_dist)),
        "nn_dist_p01": float(np.quantile(nn_dist, 0.01)),
        "per_class_accuracy": [float(C_norm[c, c]) for c in range(n_classes)],
        "fault_buses": FAULT_BUSES,
    }

    np.save(OUT_DIR / "confusion_matrix_25x25.npy", C_norm)
    np.save(OUT_DIR / "confusion_matrix_symmetrised.npy", C_sym)

    # F-C3: compare against D_global_k0 (S2.1), restricted to the 25 fault-location submatrix
    # (drop row/col 0 = no-fault, which has no per-event instances here).
    if D_GLOBAL_K0_PATH.exists():
        D = np.load(D_GLOBAL_K0_PATH)
        D_locs = D[1:, 1:]  # [25,25], drop f=0
        iu = np.triu_indices(n_classes, k=1)
        d_vals = D_locs[iu]
        c_vals = C_sym[iu]
        # Spearman-style rank correlation via numpy (no scipy.stats dependency issue)
        from scipy.stats import spearmanr
        rho, pval = spearmanr(d_vals, c_vals)
        result["fc3_correlation"] = {
            "metric": "spearman(D_global_k0, empirical_confusion_rate) over 300 location pairs",
            "rho": float(rho), "p_value": float(pval),
            "expected_sign": "negative (low D <-> high confusion)",
        }
        np.savez(OUT_DIR / "fc3_scatter_data.npz", D=d_vals, confusion=c_vals)

        # Outcome classification per S2.2's table: table says separable/confusable (D>=1 / D<1)
        # vs corpus says separable/confusable (confusion rate below/above the population median).
        median_conf = float(np.median(c_vals))
        table_confusable = d_vals < 1.0
        corpus_confusable = c_vals > median_conf
        contingency = {
            "table_separable_corpus_separable": int(np.sum(~table_confusable & ~corpus_confusable)),
            "table_separable_corpus_confusable": int(np.sum(~table_confusable & corpus_confusable)),
            "table_confusable_corpus_separable": int(np.sum(table_confusable & ~corpus_confusable)),
            "table_confusable_corpus_confusable": int(np.sum(table_confusable & corpus_confusable)),
            "corpus_confusable_threshold_median_rate": median_conf,
        }
        result["s2_2_outcome_contingency"] = contingency
    else:
        result["fc3_correlation"] = None
        print("[warn] D_global_k0.npy not found yet -- run S2.1 first for F-C3", flush=True)

    result["wall_seconds"] = time.time() - t0
    (OUT_DIR / "confusability_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("per_class_accuracy",)}, indent=2))
    print(f"\nwrote {OUT_DIR / 'confusability_summary.json'} ({result['wall_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
