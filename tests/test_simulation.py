"""Tests for single fault-event simulation (Phase 2 — Stage 2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from fault_localization.config.constants import (
    NUM_FEATURES,
    NUM_PMU_NODES,
    NUM_PHASES,
    PMU_BUSES,
    TOTAL_RECORD_MS,
)


def _can_import_dss() -> bool:
    try:
        import opendssdirect  # noqa: F401
        return True
    except ImportError:
        return False


pytest_skip_no_dss = pytest.mark.skipif(
    not _can_import_dss(),
    reason="opendssdirect not available",
)


@pytest.fixture(scope="module")
def smoke_test_result(tmp_path_factory):
    """Run the smoke test and return (recording, meta)."""
    from fault_localization.simulation.runner import stage_smoke_test
    output_dir = tmp_path_factory.mktemp("raw_sim")
    recording, meta = stage_smoke_test(
        fault_bus=7,
        fault_type="AG",
        resistance=1.0,
        output_dir=output_dir,
    )
    return recording, meta


@pytest_skip_no_dss
def test_recording_shape(smoke_test_result):
    """Recording shape must be [25, 60, 6] (3 voltages + 3 availability mask)."""
    recording, _ = smoke_test_result
    assert recording.shape == (NUM_PMU_NODES, TOTAL_RECORD_MS, NUM_FEATURES), (
        f"Expected shape ({NUM_PMU_NODES}, {TOTAL_RECORD_MS}, {NUM_FEATURES}), "
        f"got {recording.shape}"
    )


@pytest_skip_no_dss
def test_recording_dtype(smoke_test_result):
    recording, _ = smoke_test_result
    assert recording.dtype == np.float64


@pytest_skip_no_dss
def test_recording_no_nan_prefault(smoke_test_result):
    """Pre-fault portion (steps 0–39) should have no NaN."""
    recording, _ = smoke_test_result
    pre_fault = recording[:, :40, :]
    assert not np.isnan(pre_fault).any(), "NaN in pre-fault recording"


@pytest_skip_no_dss
def test_recording_metadata(smoke_test_result):
    _, meta = smoke_test_result
    assert meta["fault_bus"] == 7
    assert meta["fault_type"] == "AG"
    assert meta["resistance"] == pytest.approx(1.0)
    assert 0.5 <= meta["load_multiplier"] <= 1.3


@pytest_skip_no_dss
def test_voltage_sag_visible(smoke_test_result):
    """Fault-active period (steps 40–59) should show Phase A voltage sag for AG fault at PMU bus 13."""
    recording, _ = smoke_test_result
    pmu13_idx = PMU_BUSES.index(13)
    pre_fault_a_avg = np.nanmean(recording[pmu13_idx, :40, 0])
    fault_a_avg = np.nanmean(recording[pmu13_idx, 40:, 0])
    assert fault_a_avg < pre_fault_a_avg, (
        f"Expected Phase A voltage sag at PMU 13: fault avg {fault_a_avg:.4f} ≥ pre-fault avg {pre_fault_a_avg:.4f}"
    )
