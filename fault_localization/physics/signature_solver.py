"""One OpenDSS solve -> one row of the signature table.

The table of ``01_SPEC.md`` §2 is built by the corpus's **own** Daily-mode path,
not by a snapshot solve that happens to look similar:

    compile -> Daily mode -> one pre-fault step at ``L * PREFAULT_FREEZE``
            -> inject the fault -> one more step -> read the 25 uPMUs

Rationale (``PC-STGNN-PLAN.md`` §1 decision 2): the corpus recorded its
post-fault levels with the solver in Daily mode, the loads left frozen at the
pre-fault step-39 value, and the fault element enabled *after* the first solve.
Reproducing that sequence removes a whole class of snapshot-vs-timeseries
discrepancies (control-iteration state, regulator taps, cap-bank state) rather
than hoping they are small.  Snapshot mode is kept as :func:`solve_signature`'s
``mode="snapshot"`` and is used only as the S0.4 cross-check.

Three modes, all measured against the corpus by
``scripts/audit_snapshot_equivalence.py``:

``daily``
    Recompile the feeder for this solve.  Faithful and slow.
``reuse``
    Keep the compiled circuit between solves, recompiling only every
    ``reuse_limit`` calls.  This is the mode the full 70k-solve build wants; S0.4
    exists to show it still matches ``daily`` to 1e-3 p.u. before it is trusted.
``snapshot``
    ``Solution.Mode(0)``.  Cross-check only.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from fault_localization.config.constants import (
    NUM_PHASES,
    NUM_PMU_NODES,
    SAMPLING_RESOLUTION_MS,
)
from fault_localization.physics.grids import PREFAULT_FREEZE

logger = logging.getLogger(__name__)

SolveMode = str  # "daily" | "reuse" | "snapshot"

# ---------------------------------------------------------------------------
# Per-process circuit state
# ---------------------------------------------------------------------------
# One OpenDSS engine exists per process, so these are process-local.  A worker
# in the multiprocessing pool compiles once and then reuses; the parent process
# uses the same code path with reuse_limit=1 when it wants strict fidelity.

_STATE: dict = {"compiled": False, "master": None, "solves": 0, "fault_seq": 0, "base_loads": None}


def _dss():
    import opendssdirect as dss  # imported lazily so workers import cleanly
    return dss


def reset_engine() -> None:
    """Forget the compiled circuit so the next solve recompiles."""
    _STATE.update({"compiled": False, "master": None, "solves": 0,
                   "fault_seq": 0, "base_loads": None})


def _compile(master_path: Path | None) -> None:
    """Compile the feeder, restoring the working directory afterwards.

    OpenDSS's ``Compile`` chdir's the *process* into the directory of the .dss
    file it just read, and never changes back.  Every script here compiles the
    feeder and then writes results to a path given on the command line, so
    without this the outputs would silently land inside
    ``fault_localization/data/ieee123/``.  ``feeder.compile_feeder`` is left
    alone: the frozen corpus was generated through it and its behaviour is not
    something to change after the fact.
    """
    from fault_localization.simulation.feeder import compile_feeder, ensure_feeder_files

    path = (Path(master_path) if master_path is not None else ensure_feeder_files()).resolve()
    cwd = os.getcwd()
    try:
        compile_feeder(path)
    finally:
        os.chdir(cwd)
    from fault_localization.simulation.monitors import _get_base_loads

    _STATE.update({"compiled": True, "master": str(path), "solves": 0,
                   "fault_seq": 0, "base_loads": _get_base_loads()})


def _ensure_circuit(master_path: Path | None, mode: SolveMode, reuse_limit: int) -> None:
    """Compile the feeder if this solve may not reuse the standing circuit."""
    stale = (
        not _STATE["compiled"]
        or (master_path is not None and _STATE["master"] != str(Path(master_path).resolve()))
        or mode != "reuse"
        or _STATE["solves"] >= reuse_limit
    )
    if stale:
        _compile(master_path)


def _set_loads(multiplier: float) -> None:
    from fault_localization.simulation.monitors import _scale_loads_from_base

    _scale_loads_from_base(_STATE["base_loads"], multiplier)


def _extract() -> np.ndarray:
    from fault_localization.simulation.monitors import extract_pmu_voltages_with_mask

    return extract_pmu_voltages_with_mask()  # [25, 6]


#: How many solves to take before reading the uPMUs.
#:
#: Measured on the corpus by ``scripts/audit_window_census.py`` (S0.3): applying
#: a fault produces a **settling transient** over the first post-inception
#: steps.  On the worst of 400 events the voltage moves 1.4e-5 p.u. between
#: steps 40 and 41, 4.7e-7 between 41 and 42, and is then constant to 1e-8.
#:
#: This matters for the table even though the numbers are small.  The corpus
#: records 20 fault steps, so all but the first two windows to contain
#: post-inception samples observe the *settled* level -- and the observation
#: vector of ``01_SPEC.md`` §3 is the window's last timestep, which for w >= 23
#: is step 42 or later.  A table built from a single fault solve would therefore
#: sit on the transient while nearly every observation sits on the settled
#: value: a small, systematic, location-dependent bias that S1.4 would read as
#: operator error and try to absorb into sigma_eps.  Three solves puts the table
#: on the same level the data is on, for 3x the (negligible) build cost.
#:
#: S0.4 sweeps these and records the choice; they are parameters, not constants,
#: precisely so that it can.
DEFAULT_PREFAULT_STEPS: int = 2
DEFAULT_FAULT_STEPS: int = 3


def _solve_common(
    fault: tuple[int, str, float] | None,
    load_multiplier: float,
    mode: SolveMode,
    master_path: Path | None,
    reuse_limit: int,
    n_prefault_steps: int = DEFAULT_PREFAULT_STEPS,
    n_fault_steps: int = DEFAULT_FAULT_STEPS,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve one operating point, optionally with a fault, and read the uPMUs.

    Returns ``(voltages[25, 3], mask[25, 3], converged)``.
    """
    from fault_localization.simulation.fault_injection import clear_fault, inject_fault
    from fault_localization.simulation.monitors import setup_timeseries_mode

    dss = _dss()
    _ensure_circuit(master_path, mode, reuse_limit)
    _STATE["solves"] += 1

    if mode == "snapshot":
        dss.Solution.Mode(0)
    else:
        setup_timeseries_mode(step_size_s=SAMPLING_RESOLUTION_MS * 1e-3)

    # Pre-fault steps at the frozen load level -- this is the state the corpus's
    # fault solves inherit, so the fault element must be applied *after* them.
    _set_loads(load_multiplier)
    converged = True
    for _ in range(max(1, n_prefault_steps)):
        dss.Solution.Solve()
        converged = converged and bool(dss.Solution.Converged())

    name = None
    if fault is not None:
        bus, ftype, resistance = fault
        _STATE["fault_seq"] += 1
        name = f"pcf{_STATE['fault_seq']}"
        inject_fault(bus, ftype, resistance, fault_name=name)
        for _ in range(max(1, n_fault_steps)):
            dss.Solution.Solve()
            converged = converged and bool(dss.Solution.Converged())

    snapshot = _extract()
    voltages = snapshot[:, :NUM_PHASES].copy()
    mask = snapshot[:, NUM_PHASES:].copy()

    if name is not None:
        # Disable rather than delete: OpenDSS has no element removal, and a
        # disabled Fault contributes nothing.  The reuse_limit recompile keeps
        # the count of disabled elements bounded.
        clear_fault(name)

    if not converged:
        voltages[:] = np.nan

    return voltages, mask, converged


def solve_signature(
    fault_bus: int,
    fault_type: str,
    resistance: float,
    load_multiplier: float,
    mode: SolveMode = "daily",
    master_path: Path | None = None,
    reuse_limit: int = 1,
    apply_freeze: bool = True,
    n_prefault_steps: int = DEFAULT_PREFAULT_STEPS,
    n_fault_steps: int = DEFAULT_FAULT_STEPS,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Post-fault uPMU magnitudes for one ``(f, tau, R, L)`` hypothesis.

    Parameters
    ----------
    load_multiplier:
        The event-level ``L``.  Multiplied by :data:`grids.PREFAULT_FREEZE`
        before it reaches the loads unless ``apply_freeze=False`` (which exists
        only so S0.4 can show that the unfrozen value does *not* match).

    Returns
    -------
    ``(voltages [25, 3] p.u., mask [25, 3], converged)``
    """
    level = load_multiplier * (PREFAULT_FREEZE if apply_freeze else 1.0)
    return _solve_common((fault_bus, fault_type, resistance), level, mode,
                         master_path, reuse_limit, n_prefault_steps, n_fault_steps)


def solve_nofault(
    load_multiplier: float,
    mode: SolveMode = "daily",
    master_path: Path | None = None,
    reuse_limit: int = 1,
    apply_freeze: bool = True,
    n_prefault_steps: int = DEFAULT_PREFAULT_STEPS,
    n_fault_steps: int = DEFAULT_FAULT_STEPS,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The ``f = 0`` row: uPMU magnitudes with no fault at load ``L``."""
    level = load_multiplier * (PREFAULT_FREEZE if apply_freeze else 1.0)
    return _solve_common(None, level, mode, master_path, reuse_limit,
                         n_prefault_steps, n_fault_steps)


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------


def worker(chunk: list[tuple]) -> list[tuple]:
    """Solve a list of hypotheses in one process, reusing the circuit.

    Each item is ``(slot, fault_bus, fault_type, resistance, L, mode,
    master_path, reuse_limit)`` where ``fault_bus is None`` means the no-fault
    row.  ``slot`` is opaque to the worker and is handed back so the parent can
    scatter results into the table without relying on ordering.
    """
    out = []
    for slot, bus, ftype, resistance, load, mode, master, reuse_limit in chunk:
        if bus is None:
            v, m, ok = solve_nofault(load, mode=mode, master_path=master,
                                     reuse_limit=reuse_limit)
        else:
            v, m, ok = solve_signature(bus, ftype, resistance, load, mode=mode,
                                       master_path=master, reuse_limit=reuse_limit)
        out.append((slot, v.astype(np.float32), m.astype(np.int8), bool(ok)))
    return out


def instrumented_mask(master_path: Path | None = None) -> np.ndarray:
    """The static ``[25, 3]`` phase-instrumentation pattern of the feeder.

    Only 14 of the 25 uPMUs are three-phase; the rest present patterns like
    ``(1, 0, 0)``.  The pattern is a property of the feeder, identical in every
    window and every timestep of the corpus, so the table stores it once.
    """
    _, mask, _ = solve_nofault(1.0, mode="daily", master_path=master_path)
    return mask.astype(np.int8)
