"""Tests for sliding-window extraction (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from fault_localization.config.constants import (
    NUM_PHASES,
    NUM_PMU_NODES,
    TOTAL_RECORD_MS,
    WINDOW_LENGTH,
    WINDOW_STRIDE,
)
from fault_localization.processing.windowing import (
    count_windows,
    event_id_from_meta,
    extract_windows,
    make_window_ids,
)


# ---------------------------------------------------------------------------
# Window count
# ---------------------------------------------------------------------------


def test_window_count_standard():
    """Approved 60-step benchmark protocol retains 40 windows."""
    w = count_windows(60, 20, 1)
    assert w == 40


def test_window_count_dataset_md_note():
    """The approved protocol explicitly drops the endpoint window."""
    w_math = count_windows(60, 20, 1)
    assert w_math == 40


def test_window_count_custom():
    w = count_windows(n_timesteps=100, window_length=10, stride=5)
    assert w == 19  # (100 - 10) // 5 + 1 = 19


# ---------------------------------------------------------------------------
# extract_windows shape and content
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_recording():
    """Synthetic recording of shape [N, T, F]."""
    np.random.seed(42)
    return np.random.rand(NUM_PMU_NODES, TOTAL_RECORD_MS, NUM_PHASES)


def test_extract_windows_shape(synthetic_recording):
    windows = extract_windows(synthetic_recording)
    expected_w = count_windows(TOTAL_RECORD_MS, WINDOW_LENGTH, WINDOW_STRIDE)
    assert windows.shape == (expected_w, NUM_PMU_NODES, WINDOW_LENGTH, NUM_PHASES)


def test_extract_windows_dtype(synthetic_recording):
    windows = extract_windows(synthetic_recording)
    assert windows.dtype == np.float64


def test_extract_windows_content_correctness(synthetic_recording):
    """First window should match recording[:, 0:20, :]."""
    windows = extract_windows(synthetic_recording)
    expected_first = synthetic_recording[:, 0:WINDOW_LENGTH, :]
    np.testing.assert_array_equal(windows[0], expected_first)


def test_extract_windows_last_window(synthetic_recording):
    """Last retained window starts at index 39; endpoint start 40 is omitted."""
    windows = extract_windows(synthetic_recording)
    expected_last = synthetic_recording[:, 39:39 + WINDOW_LENGTH, :]
    np.testing.assert_array_equal(windows[-1], expected_last)


def test_extract_windows_wrong_ndim():
    with pytest.raises(ValueError, match="shape"):
        extract_windows(np.zeros((10, 20)))  # 2D instead of 3D


# ---------------------------------------------------------------------------
# Window IDs
# ---------------------------------------------------------------------------


def test_make_window_ids_count():
    ids = make_window_ids("7_AG_1p0_0000", n_windows=40)
    assert len(ids) == 40


def test_make_window_ids_format():
    ids = make_window_ids("7_AG_1p0_0000", n_windows=3)
    assert ids == [
        "7_AG_1p0_0000_w0000",
        "7_AG_1p0_0000_w0001",
        "7_AG_1p0_0000_w0002",
    ]


def test_event_id_from_meta():
    meta = {
        "fault_bus": 7,
        "fault_type": "AG",
        "resistance": 1.0,
        "run_id": 0,
    }
    event_id = event_id_from_meta(meta)
    assert event_id == "7_AG_1p0_0000"
