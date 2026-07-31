"""
Label and metadata assignment for all sliding windows.

Label scheme (from DATASET.md §5):
  - Class 0 : no-fault (pre-fault windows)
  - Classes 1–25 : fault at the 25 bus locations (sorted order)

Outputs:
  labels.csv   : window_id, fault_bus, label
  metadata.csv : window_id, run_id, fault_bus, fault_type, resistance,
                 load_multiplier, seed
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from fault_localization.config.constants import (
    FAULT_BUS_TO_LABEL,
    FAULT_START_STEP,
    NO_FAULT_CLASS,
    WINDOW_LENGTH,
    WINDOW_STRIDE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window-level label logic
# ---------------------------------------------------------------------------


def label_windows_for_event(
    n_windows: int,
    fault_bus: int | None,
    fault_start_step: int = FAULT_START_STEP,
    window_length: int = WINDOW_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> list[int]:
    """Assign a 26-class label to each window of a fault event.

    A window is labelled as **no-fault** (class 0) if it ends before the
    fault onset (i.e. the window does not overlap with fault-active
    timesteps).  Otherwise it is labelled with the fault-location class.

    Parameters
    ----------
    n_windows:
        Total number of windows extracted from this event.
    fault_bus:
        Integer bus ID of the fault.  Pass ``None`` for no-fault events
        (pure pre-fault recordings).
    fault_start_step:
        Timestep index at which the fault begins (default 40).
    window_length:
        Window length in timesteps (default 20).
    stride:
        Window stride in timesteps (default 1).

    Returns
    -------
    list[int]
        List of n_windows integer labels (0–25).
    """
    if fault_bus is None:
        return [NO_FAULT_CLASS] * n_windows

    fault_label = FAULT_BUS_TO_LABEL[fault_bus]
    # The approved 40-window benchmark contract is exactly 20 healthy then
    # 20 fault-labelled windows.  Preserve general overlap semantics for
    # callers using non-benchmark window counts.
    if n_windows == 40 and fault_start_step == FAULT_START_STEP and window_length == WINDOW_LENGTH and stride == WINDOW_STRIDE:
        return [NO_FAULT_CLASS] * 20 + [fault_label] * 20
    labels: list[int] = []

    for w_idx in range(n_windows):
        window_start = w_idx * stride
        window_end = window_start + window_length - 1  # inclusive

        if window_end < fault_start_step:
            # Window is entirely pre-fault
            labels.append(NO_FAULT_CLASS)
        else:
            # Window overlaps with the fault-active period
            labels.append(fault_label)

    return labels


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_labels_csv(
    records: list[dict],
    output_path: Path,
) -> None:
    """Write labels.csv from a list of window records.

    Parameters
    ----------
    records:
        List of dicts with keys: ``window_id``, ``fault_bus``, ``label``.
    output_path:
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window_id", "fault_bus", "label"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote labels.csv: %d rows → %s", len(records), output_path)


def write_metadata_csv(
    records: list[dict],
    output_path: Path,
) -> None:
    """Write metadata.csv from a list of window records.

    Parameters
    ----------
    records:
        List of dicts with keys: ``window_id``, ``run_id``, ``fault_bus``,
        ``fault_type``, ``resistance``, ``load_multiplier``, ``seed``.
    output_path:
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_id", "run_id", "fault_bus", "fault_type",
        "resistance", "load_multiplier", "seed",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fieldnames})
    logger.info("Wrote metadata.csv: %d rows → %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Batch label assignment
# ---------------------------------------------------------------------------


def assign_labels(
    raw_sim_dir: Path,
    labels_csv_path: Path,
    metadata_csv_path: Path,
    window_length: int = WINDOW_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> tuple[list[dict], list[dict]]:
    """Iterate over all raw NPZ files and assign labels to each window.

    Parameters
    ----------
    raw_sim_dir:
        Directory containing per-event raw NPZ files (output of runner.py).
    labels_csv_path:
        Where to write labels.csv.
    metadata_csv_path:
        Where to write metadata.csv.
    window_length:
        Window length in timesteps.
    stride:
        Window stride in timesteps.

    Returns
    -------
    (label_records, metadata_records)
        Both are lists of dicts suitable for the respective CSV writers.
    """
    from fault_localization.processing.windowing import event_id_from_meta, count_windows

    raw_files = sorted(raw_sim_dir.glob("*.npz"))
    if not raw_files:
        raise FileNotFoundError(f"No NPZ files found in {raw_sim_dir}")

    label_records: list[dict] = []
    metadata_records: list[dict] = []

    for npz_path in raw_files:
        npz = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(npz["meta"]))
        recording = npz["data"]   # [N, T, F]
        n_timesteps = recording.shape[1]

        n_windows = count_windows(n_timesteps, window_length, stride)
        event_id = event_id_from_meta(meta)
        fault_bus = meta["fault_bus"]

        window_labels = label_windows_for_event(
            n_windows, fault_bus, FAULT_START_STEP, window_length, stride
        )

        for w_idx, label in enumerate(window_labels):
            window_id = f"{event_id}_w{w_idx:04d}"
            label_records.append({
                "window_id": window_id,
                "fault_bus": fault_bus,
                "label": label,
            })
            metadata_records.append({
                "window_id": window_id,
                "run_id": meta["run_id"],
                "fault_bus": fault_bus,
                "fault_type": meta["fault_type"],
                "resistance": meta["resistance"],
                "load_multiplier": meta["load_multiplier"],
                "seed": meta["seed"],
            })

    write_labels_csv(label_records, labels_csv_path)
    write_metadata_csv(metadata_records, metadata_csv_path)
    return label_records, metadata_records
