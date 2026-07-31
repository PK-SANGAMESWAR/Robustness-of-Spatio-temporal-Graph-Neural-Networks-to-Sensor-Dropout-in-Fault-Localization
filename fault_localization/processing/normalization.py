"""
Z-score normalization for node features.

Strategy (from DATASET.md §4):
  - Compute mean and std from the **training set only**.
  - Apply to val and test sets using the same statistics.
  - Statistics are persisted so inference-time normalization is identical.

Feature format (F=6):
  Channels 0-2 : per-unit RMS voltages (V_A, V_B, V_C) — NORMALIZED.
  Channels 3-5 : binary availability mask (1=present, 0=absent) — PASS-THROUGH.

The normalization is per-phase (over channels 0-2 only), computed across all
training windows, nodes, and timesteps — matching the typical approach for
spatiotemporal datasets.  Mask channels are never included in the statistics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_NORM_STATS_FILENAME = "norm_stats.json"


# ---------------------------------------------------------------------------
# Compute statistics from training set
# ---------------------------------------------------------------------------


def compute_normalization_stats(
    train_windows_dir: Path,
    train_window_ids_file: Path,
) -> dict[str, list[float]]:
    """Compute per-phase mean and std from the training set.

    Iterates over all training window IDs, accumulates feature statistics,
    and returns a dict with ``"mean"`` and ``"std"`` lists of length F=3.

    Parameters
    ----------
    train_windows_dir:
        Directory containing per-event windowed NPZ files.
    train_window_ids_file:
        Path to ``splits/train.txt``.

    Returns
    -------
    dict
        ``{"mean": [μ_A, μ_B, μ_C], "std": [σ_A, σ_B, σ_C]}``
        Values are per-phase statistics computed over all training
        windows, nodes, and timesteps.
    """
    train_ids = _read_window_ids(train_window_ids_file)
    event_ids = {wid.rsplit("_w", 1)[0] for wid in train_ids}

    # Welford online algorithm for numerically stable mean and variance
    count = np.zeros(3, dtype=np.float64)
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    for event_id in sorted(event_ids):
        npz_path = train_windows_dir / f"{event_id}.npz"
        if not npz_path.exists():
            logger.warning("Window file not found for event %s — skipping.", event_id)
            continue

        npz = np.load(npz_path)
        windows = npz["windows"]  # [W, N, S, F=6]
        all_ids = npz["window_ids"]

        train_mask = np.fromiter(
            (str(wid) in train_ids for wid in all_ids),
            dtype=bool,
            count=len(all_ids),
        )
        if not train_mask.any():
            continue
        windows_train = windows[train_mask]  # [W', N, S, F=6]

        # Only normalize voltage channels (0:3); mask channels (3:6) are binary.
        voltage_channels = windows_train[..., :3]  # [W', N, S, 3]
        flat = voltage_channels.reshape(-1, 3)      # [W'*N*S, 3]

        # Combine batch moments with the running Welford moments.  This is
        # algebraically equivalent to scalar Welford updates but avoids a
        # Python loop over billions of training values.
        for f in range(3):  # only 3 voltage channels
            values = flat[:, f]
            values = values[np.isfinite(values)]
            batch_count = len(values)
            if not batch_count:
                continue
            batch_mean = float(values.mean())
            batch_M2 = float(((values - batch_mean) ** 2).sum())
            previous_count = count[f]
            combined_count = previous_count + batch_count
            delta = batch_mean - mean[f]
            mean[f] += delta * batch_count / combined_count
            M2[f] += batch_M2 + delta * delta * previous_count * batch_count / combined_count
            count[f] = combined_count

    # Finalize std (with Bessel's correction)
    variance = np.where(count > 1, M2 / (count - 1), 0.0)
    std = np.sqrt(variance)
    # Avoid division by zero: set zero-std features to 1
    std = np.where(std < 1e-10, 1.0, std)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": count.tolist(),
        "channels": "voltage_only_0-2",  # mask channels (3-5) not normalized
    }
    logger.info(
        "Normalization stats (voltage channels 0-2 only) — mean: %s, std: %s",
        [f"{m:.6f}" for m in stats["mean"]],
        [f"{s:.6f}" for s in stats["std"]],
    )
    return stats


# ---------------------------------------------------------------------------
# Apply normalization
# ---------------------------------------------------------------------------


def normalize_windows(
    windows: np.ndarray,
    stats: dict[str, list[float]],
) -> np.ndarray:
    """Apply Z-score normalization to voltage channels only.

    Channels 0-2 (voltages) are z-normalized; channels 3-5 (availability
    mask) are passed through unchanged.

    Parameters
    ----------
    windows:
        Float64 array of shape ``[W, N, S, F=6]`` or ``[N, S, F=6]``.
    stats:
        Normalization statistics dict with ``"mean"`` and ``"std"`` keys
        (each of length 3 — one per voltage channel).

    Returns
    -------
    np.ndarray
        Array of the same shape with voltage channels normalized.
    """
    mean = np.array(stats["mean"], dtype=np.float64)  # [3]
    std = np.array(stats["std"], dtype=np.float64)    # [3]
    result = windows.copy()
    result[..., :3] = (windows[..., :3] - mean) / std  # normalize voltages
    # channels 3-5 (mask) remain unchanged
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_norm_stats(stats: dict, output_dir: Path) -> Path:
    """Save normalization stats to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _NORM_STATS_FILENAME
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved normalization stats → %s", path)
    return path


def load_norm_stats(stats_dir: Path) -> dict[str, list[float]]:
    """Load normalization stats from JSON."""
    path = stats_dir / _NORM_STATS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _read_window_ids(path: Path) -> set[str]:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}
