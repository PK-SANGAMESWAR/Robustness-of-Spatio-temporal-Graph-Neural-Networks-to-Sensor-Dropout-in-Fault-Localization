"""Tests for feeder loading and bus validation (Phase 1)."""

from __future__ import annotations

import pytest

from fault_localization.config.constants import (
    FAULT_BUSES,
    FEEDER_FREQUENCY_HZ,
    FEEDER_NOMINAL_KV,
    PMU_BUSES,
)


# ---------------------------------------------------------------------------
# Constants sanity checks (no OpenDSS required)
# ---------------------------------------------------------------------------


def test_fault_bus_count():
    assert len(FAULT_BUSES) == 25


def test_pmu_bus_count():
    assert len(PMU_BUSES) == 25


def test_no_duplicate_fault_buses():
    assert len(FAULT_BUSES) == len(set(FAULT_BUSES))


def test_no_duplicate_pmu_buses():
    assert len(PMU_BUSES) == len(set(PMU_BUSES))


def test_feeder_nominal_kv():
    assert FEEDER_NOMINAL_KV == pytest.approx(4.16)


def test_feeder_frequency():
    assert FEEDER_FREQUENCY_HZ == pytest.approx(60.0)


def _can_import_dss() -> bool:
    try:
        import opendssdirect  # noqa: F401
        return True
    except ImportError:
        return False


pytest_skip_no_dss = pytest.mark.skipif(
    not _can_import_dss(),
    reason="opendssdirect not available or feeder files not downloaded",
)


# ---------------------------------------------------------------------------
# OpenDSS integration tests (require feeder files and opendssdirect)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_feeder():
    """Fixture: download, compile feeder, return power-flow stats."""
    from fault_localization.simulation.feeder import setup_feeder
    return setup_feeder()


@pytest_skip_no_dss
def test_feeder_converges(compiled_feeder):
    assert compiled_feeder["converged"] is True


@pytest_skip_no_dss
def test_feeder_bus_count(compiled_feeder):
    # IEEE 123-bus feeder has additional auxiliary buses; total should be ≥ 123
    assert compiled_feeder["total_buses"] >= 123


@pytest_skip_no_dss
def test_fault_buses_exist(compiled_feeder):
    """All 25 fault buses must exist in the compiled feeder."""
    from fault_localization.simulation.feeder import verify_bus_ids
    result = verify_bus_ids(FAULT_BUSES, label="fault buses")
    assert result == FAULT_BUSES


@pytest_skip_no_dss
def test_pmu_buses_exist(compiled_feeder):
    """All 25 μPMU buses must exist in the compiled feeder."""
    from fault_localization.simulation.feeder import verify_bus_ids
    result = verify_bus_ids(PMU_BUSES, label="μPMU buses")
    assert result == PMU_BUSES


@pytest_skip_no_dss
def test_impedance_extraction(compiled_feeder):
    """Line impedances must be extractable and R/X must be non-negative."""
    impedances = compiled_feeder["impedances"]
    assert len(impedances) > 0
    for entry in impedances:
        assert entry["r1_ohm"] >= 0, f"Negative R for line {entry['name']}"
        assert entry["x1_ohm"] >= 0, f"Negative X for line {entry['name']}"
        assert entry["z_mag"] >= 0, f"Negative |Z| for line {entry['name']}"
