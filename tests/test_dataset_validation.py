"""Tests for dataset validation checks (Phase 4.5).

These tests use lightweight synthetic data so they can run without
OpenDSS or a generated dataset.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fault_localization.validation.dataset_checks import (
    check_class_distribution,
    check_edge_attr_ranges,
    check_graph,
    check_no_leakage,
    check_no_nan_inf,
    check_split_sizes,
    check_total_windows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_labels_csv(path: Path, n_windows: int, n_classes: int = 26) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "fault_bus", "label"])
        writer.writeheader()
        for i in range(n_windows):
            label = i % n_classes
            fault_bus = 7 if label == 0 else label * 4  # dummy buses
            writer.writerow({
                "window_id": f"event_{i // 41:04d}_AG_1p0_0000_w{i % 41:04d}",
                "fault_bus": fault_bus,
                "label": label,
            })


def _write_synthetic_window_npz(output_dir: Path, event_id: str, n_windows: int = 41) -> None:
    windows = np.random.rand(n_windows, 25, 20, 3).astype(np.float64) * 0.8 + 0.5  # ~0.5–1.3 p.u.
    window_ids = np.array([f"{event_id}_w{i:04d}" for i in range(n_windows)], dtype=str)
    np.savez_compressed(output_dir / f"{event_id}.npz", windows=windows, window_ids=window_ids)


def _write_split_file(path: Path, window_ids: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(window_ids) + "\n")


def _write_synthetic_graph(output_dir: Path, n_nodes: int = 25, n_edges: int = 24) -> None:
    """Write a minimal edge_index and node_mapping for a path graph."""
    edges = [(i, i + 1) for i in range(n_edges)] + [(i + 1, i) for i in range(n_edges)]
    edge_index = np.array(edges, dtype=np.int64).T
    np.save(output_dir / "edge_index.npy", edge_index)

    node_mapping = {str(i): i for i in range(n_nodes)}
    with open(output_dir / "node_mapping.json", "w") as f:
        json.dump(node_mapping, f)


# ---------------------------------------------------------------------------
# Check 1: total windows
# ---------------------------------------------------------------------------


def test_check_total_windows_passes(tmp_path):
    from fault_localization.config.constants import TARGET_TOTAL_WINDOWS
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, n_windows=TARGET_TOTAL_WINDOWS)
    result = check_total_windows(labels_csv, tolerance_pct=1.0)
    assert result == TARGET_TOTAL_WINDOWS


def test_check_total_windows_fails_on_large_deviation(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, n_windows=100)  # far below 2.5M target
    with pytest.raises(AssertionError):
        check_total_windows(labels_csv, tolerance_pct=1.0)


# ---------------------------------------------------------------------------
# Check 2: class distribution
# ---------------------------------------------------------------------------


def test_check_class_distribution_returns_counts(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, n_windows=260, n_classes=26)
    counts = check_class_distribution(labels_csv)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 260


# ---------------------------------------------------------------------------
# Check 5: no NaN/Inf
# ---------------------------------------------------------------------------


def test_check_no_nan_inf_passes(tmp_path):
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    _write_synthetic_window_npz(windows_dir, "test_event_0000")
    # Should not raise
    check_no_nan_inf(windows_dir, sample_limit=5)


def test_check_no_nan_inf_detects_nan(tmp_path):
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    windows = np.ones((5, 25, 20, 3), dtype=np.float64)
    windows[0, 0, 0, 0] = np.nan
    window_ids = np.array([f"ev_w{i:04d}" for i in range(5)], dtype=str)
    np.savez_compressed(windows_dir / "ev.npz", windows=windows, window_ids=window_ids)
    with pytest.raises(AssertionError, match="NaN"):
        check_no_nan_inf(windows_dir, sample_limit=5)


# ---------------------------------------------------------------------------
# Check 7: no leakage
# ---------------------------------------------------------------------------


def test_check_no_leakage_passes(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _write_split_file(splits_dir / "train.txt", ["ev0_AG_1p0_0000_w0000"])
    _write_split_file(splits_dir / "val.txt", ["ev1_AG_1p0_0000_w0000"])
    _write_split_file(splits_dir / "test.txt", ["ev2_AG_1p0_0000_w0000"])
    # Should not raise
    check_no_leakage(splits_dir)


def test_check_no_leakage_detects_leak(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _write_split_file(splits_dir / "train.txt", ["ev0_AG_1p0_0000_w0000"])
    _write_split_file(splits_dir / "val.txt", ["ev0_AG_1p0_0000_w0001"])  # same event!
    _write_split_file(splits_dir / "test.txt", ["ev2_AG_1p0_0000_w0000"])
    with pytest.raises(AssertionError):
        check_no_leakage(splits_dir)


# ---------------------------------------------------------------------------
# Check 8: graph
# ---------------------------------------------------------------------------


def test_check_graph_connected(tmp_path):
    _write_synthetic_graph(tmp_path, n_nodes=25, n_edges=24)
    result = check_graph(tmp_path / "edge_index.npy", tmp_path / "node_mapping.json")
    assert result["connected"] is True
    assert result["n_nodes"] == 25
    assert result["n_edges"] == 24


def test_check_graph_wrong_node_count(tmp_path):
    _write_synthetic_graph(tmp_path, n_nodes=20, n_edges=19)
    with pytest.raises(AssertionError, match="Expected 25 nodes"):
        check_graph(tmp_path / "edge_index.npy", tmp_path / "node_mapping.json")


# ---------------------------------------------------------------------------
# Check 9: edge attributes
# ---------------------------------------------------------------------------


def test_check_edge_attr_ranges_passes(tmp_path):
    edge_attr = np.array([[0.1, 0.05, 0.11], [0.2, 0.1, 0.22]], dtype=np.float64)
    np.save(tmp_path / "edge_attr.npy", edge_attr)
    check_edge_attr_ranges(tmp_path / "edge_attr.npy")
