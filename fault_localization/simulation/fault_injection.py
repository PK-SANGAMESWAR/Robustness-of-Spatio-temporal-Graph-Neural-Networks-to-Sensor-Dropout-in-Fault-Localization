"""
OpenDSS Fault object builder for all 11 fault types.

Provides a high-level :func:`inject_fault` function and a
:func:`clear_fault` cleanup helper.  Every fault type is mapped to an
appropriate OpenDSS ``Fault`` element command string and is individually
testable.

Fault-type taxonomy reproduced from [BASE]:
  LG  : AG, BG, CG
  LL  : AB, BC, CA
  LLG : ABG, BCG, CAG
  3-ph: ABC, ABCG

OpenDSS ``Fault`` object syntax (reference):
  New Fault.name  Bus1=<bus>.<phase_nodes>  Phases=<n>  R=<ohms>
  For LL / LLG / 3-ph faults a second terminal (Bus2) is used.

.. note::
   This mapping has been constructed from the OpenDSS documentation
   (OpenDSS Manual, Fault object) and is to be validated against the
   actual IEEE 123-bus implementation during Phase 1 feeder bring-up.
   Phase-node syntax may need adjustment for specific line segments.
"""

from __future__ import annotations

import logging

import opendssdirect as dss

from fault_localization.config.constants import FAULT_TYPES, FAULT_RESISTANCES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase-node specifications per fault type
# ---------------------------------------------------------------------------

# OpenDSS phase node conventions:
#   .1  = phase A
#   .2  = phase B
#   .3  = phase C
#   .0  = neutral / ground

_FAULT_PHASE_SPEC: dict[str, dict] = {
    # Single-phase to ground (LG)
    "AG":  {"bus1_nodes": ".1.0", "bus2_nodes": None, "phases": 1},
    "BG":  {"bus1_nodes": ".2.0", "bus2_nodes": None, "phases": 1},
    "CG":  {"bus1_nodes": ".3.0", "bus2_nodes": None, "phases": 1},
    # Line-to-line (LL) — two-terminal fault between phases (no ground)
    "AB":  {"bus1_nodes": ".1",   "bus2_nodes": ".2", "phases": 1},
    "BC":  {"bus1_nodes": ".2",   "bus2_nodes": ".3", "phases": 1},
    "CA":  {"bus1_nodes": ".3",   "bus2_nodes": ".1", "phases": 1},
    # Double line-to-ground (LLG)
    "ABG": {"bus1_nodes": ".1.2.0", "bus2_nodes": None, "phases": 2},
    "BCG": {"bus1_nodes": ".2.3.0", "bus2_nodes": None, "phases": 2},
    "CAG": {"bus1_nodes": ".3.1.0", "bus2_nodes": None, "phases": 2},
    # Three-phase (no ground)
    "ABC": {"bus1_nodes": ".1.2.3", "bus2_nodes": None, "phases": 3},
    # Three-phase to ground
    "ABCG": {"bus1_nodes": ".1.2.3.0", "bus2_nodes": None, "phases": 3},
}

assert set(_FAULT_PHASE_SPEC.keys()) == set(FAULT_TYPES), (
    "Phase spec keys must match FAULT_TYPES exactly"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inject_fault(
    fault_bus: int,
    fault_type: str,
    resistance_ohm: float,
    fault_name: str = "fault1",
) -> str:
    """Inject a fault into the currently compiled OpenDSS circuit.

    The fault is injected as an OpenDSS ``Fault`` element and takes effect
    on the next ``Solution.Solve()`` call.

    Parameters
    ----------
    fault_bus:
        Integer bus ID where the fault is applied.
    fault_type:
        One of the 11 fault-type codes (e.g. ``"AG"``, ``"ABC"``).
    resistance_ohm:
        Fault resistance in Ohms (e.g. 0.1, 1.0, 10.0).
    fault_name:
        Internal name for the OpenDSS Fault element (defaults to
        ``"fault1"``; use a unique name when multiple faults coexist).

    Returns
    -------
    str
        The OpenDSS command string that was executed (useful for logging).

    Raises
    ------
    ValueError
        If *fault_type* is not in the 11-type taxonomy.
    """
    if fault_type not in _FAULT_PHASE_SPEC:
        raise ValueError(
            f"Unknown fault type '{fault_type}'. "
            f"Must be one of: {list(_FAULT_PHASE_SPEC.keys())}"
        )
    if resistance_ohm not in FAULT_RESISTANCES:
        logger.warning(
            "Resistance %.3f Ω is not in the standard set %s. "
            "Proceeding anyway (may be a smoke-test value).",
            resistance_ohm,
            FAULT_RESISTANCES,
        )

    spec = _FAULT_PHASE_SPEC[fault_type]
    bus1_str = f"{fault_bus}{spec['bus1_nodes']}"
    n_phases = spec["phases"]

    if spec["bus2_nodes"] is not None:
        # Two-terminal (LL) fault
        bus2_str = f"{fault_bus}{spec['bus2_nodes']}"
        cmd = (
            f"New Fault.{fault_name} "
            f"Bus1={bus1_str} Bus2={bus2_str} "
            f"Phases={n_phases} R={resistance_ohm}"
        )
    else:
        # Single-terminal fault (ground via neutral node)
        cmd = (
            f"New Fault.{fault_name} "
            f"Bus1={bus1_str} "
            f"Phases={n_phases} R={resistance_ohm}"
        )

    logger.debug("Injecting fault: %s", cmd)
    result = dss.run_command(cmd)
    if result:
        logger.warning("OpenDSS fault command returned: %s", result)

    # Enable the fault element
    dss.run_command(f"Fault.{fault_name}.Enabled=True")
    return cmd


def clear_fault(fault_name: str = "fault1") -> None:
    """Disable and remove the named fault element from the circuit.

    Parameters
    ----------
    fault_name:
        Name of the Fault element to remove.
    """
    dss.run_command(f"Fault.{fault_name}.Enabled=False")
    logger.debug("Fault '%s' disabled.", fault_name)


def clear_all_faults() -> None:
    """Disable all Fault elements currently defined in the circuit."""
    dss.run_command("Reset Faults")
    logger.debug("All fault elements reset.")


def get_fault_description(fault_bus: int, fault_type: str, resistance_ohm: float) -> str:
    """Return a human-readable description of a fault scenario."""
    return (
        f"Bus {fault_bus} | Type {fault_type} | R={resistance_ohm} Ω"
    )
