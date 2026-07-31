"""
Train / val / test split generator.

Split strategy (reproduced from [BASE] DATASET.md §7):
  - 70 / 15 / 15 random split at the **fault-event level** (not window level).
  - All overlapping windows from the same fault event stay together in the
    same split, preventing data leakage.
  - Stratified by fault location (fault_bus) for class balance.
  - Fixed random seed for reproducibility.

Outputs:
  splits/train.txt  — one window_id per line
  splits/val.txt
  splits/test.txt
"""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

from fault_localization.config.constants import (
    FAULT_BUSES,
    GLOBAL_RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------


def split_events(
    event_ids: list[str],
    fault_bus_per_event: dict[str, int],
    seed: int = GLOBAL_RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified random split of fault events into train/val/test sets.

    Stratification is by fault bus (fault location).  Events are shuffled
    within each stratum before splitting.

    Parameters
    ----------
    event_ids:
        Unique event identifiers (one per fault event, not per window).
    fault_bus_per_event:
        Mapping from event_id → fault bus integer.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (train_events, val_events, test_events)
        Lists of event IDs assigned to each split.
    """
    rng = random.Random(seed)

    # Group events by fault_bus for stratification
    by_bus: dict[int, list[str]] = defaultdict(list)
    for event_id in event_ids:
        bus = fault_bus_per_event[event_id]
        by_bus[bus].append(event_id)

    train_events: list[str] = []
    val_events: list[str] = []
    test_events: list[str] = []

    for bus in sorted(by_bus.keys()):
        bus_events = by_bus[bus]
        rng.shuffle(bus_events)

        n = len(bus_events)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)
        # Test gets the remainder to avoid rounding errors
        n_test = n - n_train - n_val

        train_events.extend(bus_events[:n_train])
        val_events.extend(bus_events[n_train : n_train + n_val])
        test_events.extend(bus_events[n_train + n_val : n_train + n_val + n_test])

    logger.info(
        "Split: %d train events, %d val events, %d test events "
        "(total %d events).",
        len(train_events), len(val_events), len(test_events),
        len(train_events) + len(val_events) + len(test_events),
    )
    return train_events, val_events, test_events


def expand_events_to_windows(
    event_ids: list[str],
    windows_per_event: dict[str, list[str]],
) -> list[str]:
    """Expand a list of event IDs to all their constituent window IDs."""
    window_ids: list[str] = []
    for event_id in event_ids:
        window_ids.extend(windows_per_event.get(event_id, []))
    return window_ids


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_split_file(window_ids: list[str], output_path: Path) -> None:
    """Write window IDs to a split text file (one per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(window_ids))
        if window_ids:
            f.write("\n")
    logger.info("Wrote %d window IDs → %s", len(window_ids), output_path)


def read_split_file(path: Path) -> list[str]:
    """Read window IDs from a split text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Batch split generation from labels.csv
# ---------------------------------------------------------------------------


def generate_splits(
    labels_csv_path: Path,
    splits_output_dir: Path,
    seed: int = GLOBAL_RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Read labels.csv and generate the 70/15/15 fault-event-level split.

    Parameters
    ----------
    labels_csv_path:
        Path to labels.csv (window_id, fault_bus, label).
    splits_output_dir:
        Directory where train.txt / val.txt / test.txt are written.
    seed:
        Random seed.

    Returns
    -------
    (train_window_ids, val_window_ids, test_window_ids)
    """
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv_path}")

    # Build: event_id → [window_ids]  and  event_id → fault_bus
    windows_per_event: dict[str, list[str]] = defaultdict(list)
    fault_bus_per_event: dict[str, int] = {}

    with open(labels_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            window_id = row["window_id"]
            fault_bus = int(row["fault_bus"])
            # Event ID = everything before the last "_w<digits>" suffix
            event_id = window_id.rsplit("_w", 1)[0]
            windows_per_event[event_id].append(window_id)
            fault_bus_per_event[event_id] = fault_bus

    all_event_ids = list(windows_per_event.keys())
    train_events, val_events, test_events = split_events(
        all_event_ids, fault_bus_per_event, seed=seed
    )

    train_windows = expand_events_to_windows(train_events, windows_per_event)
    val_windows = expand_events_to_windows(val_events, windows_per_event)
    test_windows = expand_events_to_windows(test_events, windows_per_event)

    write_split_file(train_windows, splits_output_dir / "train.txt")
    write_split_file(val_windows, splits_output_dir / "val.txt")
    write_split_file(test_windows, splits_output_dir / "test.txt")

    logger.info(
        "Splits: %d train windows / %d val windows / %d test windows.",
        len(train_windows), len(val_windows), len(test_windows),
    )
    return train_windows, val_windows, test_windows


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------


def verify_no_leakage(
    train_windows: list[str],
    val_windows: list[str],
    test_windows: list[str],
) -> bool:
    """Verify that no event appears in more than one split.

    Raises
    ------
    AssertionError
        If leakage is detected (shared event IDs across splits).

    Returns
    -------
    bool
        True if no leakage detected.
    """
    def get_event_ids(window_ids: list[str]) -> set[str]:
        return {wid.rsplit("_w", 1)[0] for wid in window_ids}

    train_events = get_event_ids(train_windows)
    val_events = get_event_ids(val_windows)
    test_events = get_event_ids(test_windows)

    train_val = train_events & val_events
    train_test = train_events & test_events
    val_test = val_events & test_events

    if train_val or train_test or val_test:
        raise AssertionError(
            f"Data leakage detected!\n"
            f"  train ∩ val  : {len(train_val)} events\n"
            f"  train ∩ test : {len(train_test)} events\n"
            f"  val ∩ test   : {len(val_test)} events"
        )

    logger.info("✓ No data leakage detected across splits.")
    return True
