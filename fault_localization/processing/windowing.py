"""
Sliding-window extraction from raw fault-event recordings.

Input  : raw NPZ files, each containing a ``data`` array of shape [25, 60, 3].
Output : per-window tensors of shape [N=25, S=20, F=3] and associated window IDs.

Window count per event:
  Mathematical derivation: (T - S) / stride + 1 = (60 - 20) / 1 + 1 = 41
  DATASET.md §4 states "40 overlapping windows per fault event" — to be
  confirmed empirically in Stage 2.2 against the paper's exact procedure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fault_localization.config.constants import (
    TOTAL_RECORD_MS,
    WINDOW_LENGTH,
    WINDOW_STRIDE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core windowing logic
# ---------------------------------------------------------------------------


def extract_windows(
    recording: np.ndarray,
    window_length: int = WINDOW_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> np.ndarray:
    """Extract overlapping sliding windows from a single fault-event recording.

    Parameters
    ----------
    recording:
        Float64 array of shape ``[N, T, F]`` (nodes, timesteps, features).
    window_length:
        Number of timesteps per window (default 20).
    stride:
        Stride between consecutive windows in timesteps (default 1).

    Returns
    -------
    np.ndarray
        Float64 array of shape ``[W, N, S, F]`` where W is the number of
        extracted windows.
    """
    if recording.ndim != 3:
        raise ValueError(
            f"Expected recording of shape [N, T, F], got {recording.shape}"
        )
    n_nodes, n_timesteps, n_features = recording.shape
    available_windows = (n_timesteps - window_length) // stride + 1
    # Benchmark v1.0 deliberately omits the final endpoint (start t=40),
    # yielding the approved 40-window protocol from a 60-step recording.
    n_windows = min(available_windows, 40) if n_timesteps == TOTAL_RECORD_MS and window_length == WINDOW_LENGTH and stride == WINDOW_STRIDE else available_windows

    if n_windows <= 0:
        raise ValueError(
            f"Recording length {n_timesteps} is too short for "
            f"window_length={window_length}."
        )

    # Use stride_tricks for zero-copy view (read-only)
    windows = np.lib.stride_tricks.sliding_window_view(
        recording, window_shape=window_length, axis=1
    )
    # sliding_window_view shape: [N, n_windows, F, window_length]
    # Reorder to [n_windows, N, window_length, F]
    windows = windows.transpose(1, 0, 3, 2)[:n_windows]  # [W, N, S, F]

    assert windows.shape == (n_windows, n_nodes, window_length, n_features), (
        f"Unexpected window array shape: {windows.shape}"
    )
    return np.ascontiguousarray(windows, dtype=np.float64)


def count_windows(
    n_timesteps: int = TOTAL_RECORD_MS,
    window_length: int = WINDOW_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> int:
    """Return the number of sliding windows for a given recording length."""
    available = max(0, (n_timesteps - window_length) // stride + 1)
    if n_timesteps == TOTAL_RECORD_MS and window_length == WINDOW_LENGTH and stride == WINDOW_STRIDE:
        return min(available, 40)
    return available


# ---------------------------------------------------------------------------
# Window ID generation
# ---------------------------------------------------------------------------


def make_window_ids(event_id: str, n_windows: int) -> list[str]:
    """Generate canonical window IDs for a given fault event.

    Parameters
    ----------
    event_id:
        Unique string identifier for the fault event (e.g.
        ``"7_AG_1p0_0000"``).
    n_windows:
        Number of windows extracted from this event.

    Returns
    -------
    list[str]
        Window IDs of the form ``"<event_id>_w<window_index:04d>"``.
    """
    return [f"{event_id}_w{i:04d}" for i in range(n_windows)]


def event_id_from_meta(meta: dict) -> str:
    """Derive a canonical event ID string from simulation metadata."""
    res_str = f"{meta['resistance']:.1f}".replace(".", "p")
    return (
        f"{meta['fault_bus']}_{meta['fault_type']}_"
        f"{res_str}_{meta['run_id']:04d}"
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_event_file(npz_path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Load an event NPZ file and extract windows + IDs.

    Parameters
    ----------
    npz_path:
        Path to a raw simulation NPZ file.

    Returns
    -------
    (windows, window_ids, meta)
        windows    : shape [W, N, S, F]
        window_ids : list of W string IDs
        meta       : simulation metadata dict
    """
    import json
    npz = np.load(npz_path, allow_pickle=True)
    recording = npz["data"]   # [N, T, F]
    meta = json.loads(str(npz["meta"]))

    windows = extract_windows(recording)
    event_id = event_id_from_meta(meta)
    window_ids = make_window_ids(event_id, len(windows))

    return windows, window_ids, meta


def save_windows(
    windows: np.ndarray,
    window_ids: list[str],
    output_dir: Path,
) -> Path:
    """Save a batch of windows for one event to a compressed NPZ file.

    Parameters
    ----------
    windows:
        Shape [W, N, S, F].
    window_ids:
        List of W string IDs.
    output_dir:
        Directory to write into.

    Returns
    -------
    Path
        Path of the saved NPZ file (named after the first window ID's event).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Derive event ID from first window ID
    event_id = window_ids[0].rsplit("_w", 1)[0]
    out_path = output_dir / f"{event_id}.npz"
    np.savez_compressed(
        out_path,
        windows=windows,
        window_ids=np.array(window_ids, dtype=str),
    )
    return out_path


def process_all_events(
    raw_sim_dir: Path,
    windows_output_dir: Path,
) -> list[tuple[str, int]]:
    """Process all raw NPZ files in *raw_sim_dir* and save windowed outputs.

    Parameters
    ----------
    raw_sim_dir:
        Directory containing per-event raw NPZ files.
    windows_output_dir:
        Directory where windowed NPZ files are saved.

    Returns
    -------
    list[(event_id, n_windows)]
        List of (event_id, window_count) tuples for downstream label
        assignment.
    """
    raw_files = sorted(raw_sim_dir.glob("*.npz"))
    if not raw_files:
        raise FileNotFoundError(f"No NPZ files found in {raw_sim_dir}")

    results: list[tuple[str, int]] = []
    for npz_path in raw_files:
        try:
            windows, window_ids, meta = process_event_file(npz_path)
            save_windows(windows, window_ids, windows_output_dir)
            event_id = event_id_from_meta(meta)
            results.append((event_id, len(windows)))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process %s: %s", npz_path.name, exc)

    logger.info(
        "Windowing complete: %d events processed, %d total windows.",
        len(results),
        sum(n for _, n in results),
    )
    return results
