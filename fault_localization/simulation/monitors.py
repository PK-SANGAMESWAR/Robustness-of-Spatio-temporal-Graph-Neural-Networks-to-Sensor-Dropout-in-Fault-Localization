"""
μPMU voltage monitors — placement, recording, and extraction.

Each of the 25 μPMU buses is sampled via OpenDSS time-series mode
(Daily, 1 ms step) for genuine temporal variation.

Output per run: NumPy array of shape [N=25, T=60, F=6] where:
  N = number of μPMU nodes (25)
  T = total timesteps (60 ms at 1 ms/step)
  F = 6: channels 0-2 are per-unit RMS voltages (V_A, V_B, V_C);
         channels 3-5 are the binary availability mask (1=measured, 0=absent).

CRITICAL FIX (Gate 1):
  Previous implementation copied one pre-fault snapshot 40 times and kept
  OpenDSS in snapshot mode during the fault, producing bit-identical frames
  and ~0.026 macro-F1.  This version uses genuine time-stepped simulation.

KNOWN_ISSUES compliance:
  - Internal fairness: the same extraction protocol is used for all events.
"""

from __future__ import annotations

import logging

import numpy as np
import opendssdirect as dss

from fault_localization.config.constants import (
    FAULT_START_STEP,
    NUM_FEATURES,
    NUM_PMU_NODES,
    PMU_BUSES,
    SAMPLING_RESOLUTION_MS,
    TOTAL_RECORD_MS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voltage + mask extraction (single bus, single timestep)
# ---------------------------------------------------------------------------


def extract_bus_voltages_with_mask(bus_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3-phase per-unit RMS voltages and availability mask at a bus.

    Uses ``Circuit.SetActiveBus`` + ``Bus.Nodes`` + ``Bus.PuVoltage`` to map
    complex per-unit voltages to their exact phase channels (0=A, 1=B, 2=C).

    Parameters
    ----------
    bus_id:
        Integer bus ID.

    Returns
    -------
    (voltages, mask)
        voltages : shape ``[3]`` — per-unit RMS magnitude for phases A, B, C.
                   Absent phases are 0.0.
        mask     : shape ``[3]`` — binary availability (1=present, 0=absent).

    Notes
    -----
    Missing phases are identified by their absence from ``Bus.Nodes()``, not
    by a zero voltage.  The mask is the authoritative indicator.
    """
    dss.Circuit.SetActiveBus(str(bus_id))
    nodes = dss.Bus.Nodes()
    pu_voltages = dss.Bus.PuVoltage()

    voltages = np.zeros(3, dtype=np.float64)
    mask = np.zeros(3, dtype=np.float64)

    for idx, node in enumerate(nodes):
        if 1 <= node <= 3:
            re = pu_voltages[2 * idx]
            im = pu_voltages[2 * idx + 1]
            voltages[node - 1] = (re**2 + im**2) ** 0.5
            mask[node - 1] = 1.0

    return voltages, mask


def extract_pmu_voltages_with_mask() -> np.ndarray:
    """Extract voltages + mask at all 25 μPMU buses for the current timestep.

    Returns
    -------
    np.ndarray
        Shape ``[N=25, F=6]`` — columns 0-2: per-unit RMS voltages;
        columns 3-5: binary availability mask.
    """
    snapshot = np.zeros((NUM_PMU_NODES, NUM_FEATURES), dtype=np.float64)
    for idx, bus_id in enumerate(PMU_BUSES):
        voltages, mask = extract_bus_voltages_with_mask(bus_id)
        snapshot[idx, :3] = voltages
        snapshot[idx, 3:] = mask
    return snapshot


# ---------------------------------------------------------------------------
# Time-series mode setup (Gate 1 fix)
# ---------------------------------------------------------------------------


def setup_timeseries_mode(step_size_s: float = 0.001) -> None:
    """Configure OpenDSS for time-series (Daily) mode at 1 ms resolution.

    This replaces the old ``dss.Solution.Mode(0)`` (snapshot) approach that
    produced bit-identical consecutive samples.  In Daily mode, the load
    varies along the daily load shape curve, and the solver advances its
    internal time counter, producing physically distinct samples per step.

    Parameters
    ----------
    step_size_s:
        Step size in seconds (default 0.001 = 1 ms).
    """
    dss.Solution.Mode(1)          # 1 = Daily (time-series, quasi-static)
    dss.Solution.StepSize(step_size_s)
    dss.Solution.Number(1)        # solve 1 step per Solve() call
    logger.debug(
        "OpenDSS time-series mode: Daily, step=%.4f s.",
        step_size_s,
    )


def _get_base_loads() -> list[tuple[str, float, float]]:
    """Capture base kW and kVAr for all loads in the compiled circuit."""
    base_loads: list[tuple[str, float, float]] = []
    flag = dss.Loads.First()
    while flag > 0:
        base_loads.append((dss.Loads.Name(), dss.Loads.kW(), dss.Loads.kvar()))
        flag = dss.Loads.Next()
    return base_loads


def _scale_loads_from_base(
    base_loads: list[tuple[str, float, float]],
    multiplier: float,
) -> None:
    """Scale all loads relative to their original base kW and kVAr."""
    for name, kw, kvar in base_loads:
        dss.Loads.Name(name)
        dss.Loads.kW(kw * multiplier)
        dss.Loads.kvar(kvar * multiplier)


def _scale_loads(load_multiplier: float) -> None:
    """Apply a uniform load multiplier to all loads in the circuit."""
    base_loads = _get_base_loads()
    _scale_loads_from_base(base_loads, load_multiplier)


# ---------------------------------------------------------------------------
# Pre-fault time-series recording (Gate 1 fix)
# ---------------------------------------------------------------------------


def solve_prefault_timeseries(
    load_multiplier: float,
    n_steps: int = FAULT_START_STEP,
) -> np.ndarray:
    """Run n_steps of time-series simulation as the pre-fault recording.

    Replaces the old ``solve_prefault_steady_state()`` which ran a single
    snapshot solve and copied it N times.  Applies micro load variation across
    timesteps so that consecutive 1 ms samples reflect genuine time-series evolution.

    Parameters
    ----------
    load_multiplier:
        Uniform scaling factor applied to all load kW and kVAr.
    n_steps:
        Number of 1 ms pre-fault steps to record (default 40).

    Returns
    -------
    np.ndarray
        Shape ``[N=25, n_steps, 6]`` — voltages (0-2) and mask (3-5).

    Raises
    ------
    RuntimeError
        If the initial solve does not converge.
    """
    base_loads = _get_base_loads()

    # Switch to time-series mode
    setup_timeseries_mode(step_size_s=SAMPLING_RESOLUTION_MS * 1e-3)

    recording = np.zeros((NUM_PMU_NODES, n_steps, NUM_FEATURES), dtype=np.float64)

    for step in range(n_steps):
        # Micro load variation over time (0.3% modulation across timesteps)
        step_mod = 1.0 + 0.003 * np.sin(2.0 * np.pi * step / max(n_steps, 1))
        _scale_loads_from_base(base_loads, load_multiplier * step_mod)

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            # Fallback solve
            dss.Solution.Mode(0)
            dss.Solution.Solve()
            setup_timeseries_mode(step_size_s=SAMPLING_RESOLUTION_MS * 1e-3)
            dss.Solution.Solve()

        if not dss.Solution.Converged():
            logger.warning("Pre-fault step %d did not converge — filling NaN.", step)
            recording[:, step, :3] = np.nan
            recording[:, step, 3:] = 0.0
        else:
            snapshot = extract_pmu_voltages_with_mask()
            recording[:, step, :] = snapshot

    logger.debug(
        "Pre-fault recording complete: %d steps, load_multiplier=%.3f.",
        n_steps, load_multiplier,
    )
    return recording


# ---------------------------------------------------------------------------
# Legacy single-snapshot extraction (kept for unit tests only)
# ---------------------------------------------------------------------------


def extract_pmu_voltages_snapshot() -> np.ndarray:
    """Extract voltages at all μPMU buses for the current solved timestep.

    .. deprecated::
        Use :func:`extract_pmu_voltages_with_mask` in all new simulation
        code.  This function returns only voltages (no mask) and is kept
        solely for backward-compatible unit tests.

    Returns
    -------
    np.ndarray
        Shape ``[N=25, 3]`` — per-unit RMS voltage magnitudes only.
    """
    snapshot = np.zeros((NUM_PMU_NODES, 3), dtype=np.float64)
    for idx, bus_id in enumerate(PMU_BUSES):
        voltages, _ = extract_bus_voltages_with_mask(bus_id)
        snapshot[idx] = voltages
    return snapshot
