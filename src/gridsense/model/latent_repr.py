"""Case-store representations of the per-node embeddings h [25, 64] (decisions O4, O15, 2026-10-28).

(a) concat          : h flattened in fixed node order, 1,600-d.
(b) mean_max        : mean and max over the 25 nodes, 128-d.
(c) concat_pca256   : (a) standardized per dimension, then PCA to 256-d. The scaler and the PCA are fitted on the
                      train subsample ONLY (never on val or test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

METHODS = ("a_concat", "b_mean_max", "c_concat_pca256")
PCA_DIM = 256


def fit_scaler_pca(store_h: np.ndarray, dim: int = PCA_DIM, chunk: int = 20000) -> dict[str, np.ndarray]:
    """Per-dimension mean/std of the flattened store, then the top `dim` principal axes of the standardized store."""
    flat = store_h.reshape(len(store_h), -1)
    n, d = flat.shape
    mean = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        mean += flat[i:i + chunk].sum(0, dtype=np.float64)
    mean /= n
    var = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        var += ((flat[i:i + chunk] - mean) ** 2).sum(0)
    std = np.sqrt(var / (n - 1))
    std = np.where(std < 1e-8, 1.0, std)  # constant dimension: leave unscaled (value 0 after centring)
    cov = np.zeros((d, d), dtype=np.float64)
    for i in range(0, n, chunk):
        z = (flat[i:i + chunk] - mean) / std
        cov += z.T @ z
    cov /= n - 1
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:dim]
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "components": vecs[:, order].astype(np.float32),
            "explained_variance_ratio": (vals[order] / vals.clip(min=0).sum()).astype(np.float64),
            "n_fit_windows": np.array(n)}


def save_scaler_pca(params: dict[str, np.ndarray], path: Path) -> None:
    np.savez(path, **params)


def load_scaler_pca(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def represent(h: np.ndarray, method: str, pca: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """h: [B, 25, 64] float32 -> [B, d] float32."""
    if method == "a_concat":
        return np.ascontiguousarray(h.reshape(len(h), -1), dtype=np.float32)
    if method == "b_mean_max":
        return np.concatenate([h.mean(1), h.max(1)], axis=1).astype(np.float32)
    if method == "c_concat_pca256":
        if pca is None:
            raise ValueError("c_concat_pca256 needs the scaler/PCA fitted on the train subsample")
        z = (h.reshape(len(h), -1) - pca["mean"]) / pca["std"]
        return (z @ pca["components"]).astype(np.float32)
    raise ValueError(f"unknown method {method}")
