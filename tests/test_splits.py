"""Tests for train/val/test split logic and leakage prevention (Phase 3)."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from fault_localization.processing.splits import (
    expand_events_to_windows,
    generate_splits,
    split_events,
    verify_no_leakage,
)


# ---------------------------------------------------------------------------
# Synthetic event set fixture
# ---------------------------------------------------------------------------


def _make_synthetic_labels_csv(path: Path, n_events: int = 100, windows_per_event: int = 41) -> None:
    """Write a synthetic labels.csv for testing."""
    fault_buses = [7, 13, 18, 21, 25] * (n_events // 5 + 1)
    bus_to_label = {7: 1, 13: 2, 18: 3, 21: 4, 25: 5}

    rows = []
    for event_idx in range(n_events):
        bus = fault_buses[event_idx]
        label = bus_to_label[bus]
        event_id = f"{bus}_AG_1p0_{event_idx:04d}"
        for w in range(windows_per_event):
            window_id = f"{event_id}_w{w:04d}"
            rows.append({"window_id": window_id, "fault_bus": bus, "label": label})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "fault_bus", "label"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# split_events unit tests
# ---------------------------------------------------------------------------


def test_split_events_total_coverage():
    """All events must appear in exactly one split."""
    event_ids = [f"event_{i}" for i in range(100)]
    fault_bus_per_event = {eid: (i % 5) * 10 + 7 for i, eid in enumerate(event_ids)}
    train, val, test = split_events(event_ids, fault_bus_per_event, seed=42)

    all_assigned = set(train) | set(val) | set(test)
    assert all_assigned == set(event_ids), "Some events not assigned to any split"
    assert len(train) + len(val) + len(test) == len(event_ids), "Duplicate assignment"


def test_split_events_approximate_ratios():
    """Ratios should be approximately 70/15/15."""
    n_events = 200
    event_ids = [f"event_{i}" for i in range(n_events)]
    fault_bus_per_event = {eid: 7 for eid in event_ids}  # single stratum
    train, val, test = split_events(event_ids, fault_bus_per_event, seed=42)

    assert len(train) / n_events == pytest.approx(0.70, abs=0.05)
    assert len(val) / n_events == pytest.approx(0.15, abs=0.05)
    assert len(test) / n_events == pytest.approx(0.15, abs=0.05)


def test_split_events_reproducible():
    """Same seed must produce same split."""
    event_ids = [f"event_{i}" for i in range(50)]
    fault_bus_per_event = {eid: 7 for eid in event_ids}

    train1, val1, test1 = split_events(event_ids, fault_bus_per_event, seed=99)
    train2, val2, test2 = split_events(event_ids, fault_bus_per_event, seed=99)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------


def test_no_leakage_clean():
    """verify_no_leakage should pass with correctly split events."""
    train = ["event_0_w0000", "event_0_w0001", "event_1_w0000"]
    val = ["event_2_w0000"]
    test = ["event_3_w0000"]
    assert verify_no_leakage(train, val, test) is True


def test_no_leakage_detects_event_leak():
    """verify_no_leakage must detect shared events across train and val."""
    train = ["event_0_w0000", "event_0_w0001"]
    val = ["event_0_w0002"]   # same event as train!
    test = ["event_1_w0000"]
    with pytest.raises(AssertionError, match="leakage"):
        verify_no_leakage(train, val, test)


# ---------------------------------------------------------------------------
# Integration: generate_splits from CSV
# ---------------------------------------------------------------------------


def test_generate_splits_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        labels_csv = tmp / "labels.csv"
        splits_dir = tmp / "splits"
        splits_dir.mkdir()

        _make_synthetic_labels_csv(labels_csv, n_events=50, windows_per_event=41)
        train_w, val_w, test_w = generate_splits(labels_csv, splits_dir, seed=42)

        # All splits non-empty
        assert len(train_w) > 0
        assert len(val_w) > 0
        assert len(test_w) > 0

        # Total windows = 50 events * 41 windows
        total = len(train_w) + len(val_w) + len(test_w)
        assert total == 50 * 41

        # No leakage
        verify_no_leakage(train_w, val_w, test_w)

        # Files were written
        assert (splits_dir / "train.txt").exists()
        assert (splits_dir / "val.txt").exists()
        assert (splits_dir / "test.txt").exists()
