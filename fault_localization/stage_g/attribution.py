"""S7.1 error attribution (``PC-STGNN/09_STAGE_H_writing.md``): where do a model's errors fall on the identifiability map?

For each misclassified window with a labelable true class, the pair (true class, predicted class) is looked up in
``D_global_k{k}.npy`` (``results/pcstgnn/audit/s2_1_identifiability``; 26 x 26, index 0 = no fault, index c = fault bus c;
D = separability in nats, averaged over random sensor subsets at that k). A pair with ``D < D_THRESHOLD`` counts as
"indistinguishable" (``11_METRICS.md`` § 5). Reported per (model, k):

- ``frac_errors_in_indist_pairs``: share of the errors that fall in a pair with D < 1;
- ``chance_frac``: share of ordered pairs (a != b) with D < 1, the rate if errors went to uniformly random wrong classes;
- ``frac_indist_pair_windows_wrong``: share of windows whose true class belongs to some indistinguishable pair that
  are wrong, to show how much of the total error budget such classes can account for.

Windows with zero post-inception samples (w = 20, fault label, no fault data) are excluded: no model can be right there,
and that error is not an identifiability error (``unlabelable_window_mask``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D_DIR = REPO / "results/pcstgnn/audit/s2_1_identifiability"
D_THRESHOLD = 1.0
FAULT_START_STEP, WINDOW_LENGTH = 40, 20


def load_D(k: int, d_dir: Path = D_DIR) -> np.ndarray:
    d = np.load(Path(d_dir) / f"D_global_k{k}.npy")
    if d.shape != (26, 26) or not np.allclose(d, d.T):
        raise ValueError(f"D_global_k{k}.npy is not a symmetric 26 x 26 matrix")
    return d


def labelable(window_ids) -> np.ndarray:
    idx = np.array([int(str(w).rpartition("_w")[2]) for w in window_ids])
    return idx + WINDOW_LENGTH - FAULT_START_STEP != 0


def attribute(y: np.ndarray, pred: np.ndarray, window_ids, D: np.ndarray, thr: float = D_THRESHOLD) -> dict:
    keep = labelable(window_ids)
    y, pred = y[keep].astype(np.int64), pred[keep].astype(np.int64)
    wrong = y != pred
    n_wrong = int(wrong.sum())
    off = ~np.eye(26, dtype=bool)
    indist = (D < thr) & off
    in_pair = indist[y[wrong], pred[wrong]]
    classes_in_pair = indist.any(axis=1)
    mask_cls = classes_in_pair[y]
    return {"n_windows": int(len(y)), "n_wrong": n_wrong, "error_rate": n_wrong / len(y),
            "n_errors_in_indist_pairs": int(in_pair.sum()),
            "frac_errors_in_indist_pairs": float(in_pair.mean()) if n_wrong else float("nan"),
            "n_indist_pairs": int(indist.sum() // 2), "chance_frac": float(indist.sum() / off.sum()),
            "n_windows_in_indist_classes": int(mask_cls.sum()),
            "frac_indist_class_windows_wrong": float(wrong[mask_cls].mean()) if mask_cls.any() else float("nan"),
            "frac_other_class_windows_wrong": float(wrong[~mask_cls].mean())}


def top_confusions(y, pred, window_ids, D, n: int = 10) -> list[dict]:
    keep = labelable(window_ids)
    y, pred = y[keep].astype(np.int64), pred[keep].astype(np.int64)
    m = y != pred
    flat = np.bincount(y[m] * 26 + pred[m], minlength=676).reshape(26, 26)
    order = np.argsort(flat.ravel())[::-1][:n]
    return [{"true": int(i // 26), "pred": int(i % 26), "count": int(flat.ravel()[i]), "D": float(D[i // 26, i % 26])}
            for i in order if flat.ravel()[i] > 0]


def run(pred_dir: Path, models: dict, ks=(0, 10), seeds=(42, 43, 44), n_offsets: int = 3, sigma_key: str = "s0") -> dict:
    """``models``: ``{label: dir name under pred_dir}``. Averages the per-window counts over offsets, reports per seed."""
    out: dict = {}
    for label, name in models.items():
        out[label] = {}
        for k in ks:
            D = load_D(k)
            per_seed = []
            for s in seeds:
                p = Path(pred_dir) / name / f"seed_{s:05d}_predictions.npz"
                if not p.exists():
                    continue
                with np.load(p, allow_pickle=False) as z:
                    y, ids = z["y_true"], z["window_ids"]
                    rows = [attribute(y, z[f"pf__k{k}_{sigma_key}_o{o}"], ids, D) for o in range(n_offsets)]
                    top = top_confusions(y, z[f"pf__k{k}_{sigma_key}_o0"], ids, D)
                per_seed.append({"seed": s, "offsets": rows, "top_confusions_offset0": top})
            out[label][f"k{k}"] = per_seed
    return out
