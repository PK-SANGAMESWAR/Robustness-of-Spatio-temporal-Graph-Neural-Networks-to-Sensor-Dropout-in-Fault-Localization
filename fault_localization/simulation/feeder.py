"""
IEEE 123-bus feeder loader and validation.

Responsibilities:
  - Download the IEEE 123-bus .dss files from the dss-extensions repository
    if they are not already present in fault_localization/data/ieee123/.
  - Compile the circuit via opendssdirect.py.
  - Run a baseline (no-fault) power flow and verify convergence.
  - Enumerate all buses; verify that all 25 fault positions and 25 μPMU
    locations exist in the solved model.
  - Extract per-line impedance data (R, X per unit, converted to Ohms) for
    use as edge attributes.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import opendssdirect as dss

from fault_localization.config.constants import (
    FAULT_BUSES,
    FEEDER_FREQUENCY_HZ,
    FEEDER_NOMINAL_KV,
    PMU_BUSES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_DATA_DIR = _HERE.parent / "data" / "ieee123"

# dss-extensions mirror of the OpenDSS IEEE 123-bus test case
_DSS_BASE_URL = (
    "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/"
    "master/Version8/Distrib/IEEETestCases/123Bus/"
)
_DSS_PARENT_URL = (
    "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/"
    "master/Version8/Distrib/IEEETestCases/"
)

_REQUIRED_FILES = [
    "IEEE123Master.dss",
    "IEEELineCodes.DSS",
    "IEEE123Regulators.DSS",
    "IEEE123Loads.DSS",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def ensure_feeder_files(data_dir: Path = _DATA_DIR) -> Path:
    """Download IEEE 123-bus .dss files if not already present.

    Parameters
    ----------
    data_dir:
        Local directory where the .dss files should be stored.

    Returns
    -------
    Path
        Path to the master .dss file (``IEEE123Master.dss``).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    master = data_dir / "IEEE123Master.dss"

    for fname in _REQUIRED_FILES:
        dest = data_dir / fname
        if dest.exists():
            # If IEEELineCodes.DSS contains stub redirect, replace with full file
            if fname == "IEEELineCodes.DSS" and "redirect .." in dest.read_text().lower():
                dest.unlink()
            else:
                logger.debug("Found %s — skipping download.", fname)
                continue

        url = (_DSS_PARENT_URL if fname == "IEEELineCodes.DSS" else _DSS_BASE_URL) + fname
        logger.info("Downloading %s …", url)
        try:
            urllib.request.urlretrieve(url, dest)
            logger.info("  → saved to %s", dest)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {fname} from {url}: {exc}\n"
                "Please place the IEEE 123-bus .dss files manually in "
                f"{data_dir} and retry."
            ) from exc

    if not master.exists():
        raise FileNotFoundError(
            f"IEEE123Master.dss not found in {data_dir} after download attempt."
        )

    return master


def compile_feeder(master_path: Path | None = None) -> None:
    """Compile the IEEE 123-bus circuit in OpenDSS.

    Parameters
    ----------
    master_path:
        Path to ``IEEE123Master.dss``.  If *None*, calls
        :func:`ensure_feeder_files` automatically.
    """
    if master_path is None:
        master_path = ensure_feeder_files()

    logger.info("Compiling feeder: %s", master_path)
    dss.run_command(f"Compile [{master_path}]")

    if not dss.Solution.Converged():
        dss.Solution.Solve()

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS feeder compilation did not converge.")

    logger.info("Feeder compiled successfully.")


def run_baseline_powerflow() -> dict[str, float]:
    """Run a steady-state (no-fault) power flow and return summary stats.

    The feeder must already be compiled.

    Returns
    -------
    dict
        Keys: ``"total_buses"``, ``"converged"``, ``"total_load_kw"``,
        ``"total_loss_kw"``.

    Raises
    ------
    RuntimeError
        If the power flow does not converge.
    """
    dss.Solution.Mode(0)   # Snapshot (steady-state)
    dss.Solution.Solve()

    if not dss.Solution.Converged():
        raise RuntimeError("Baseline power flow did not converge.")

    stats = {
        "total_buses": dss.Circuit.NumBuses(),
        "converged": True,
        "total_load_kw": dss.Circuit.TotalPower()[0] * -1,  # convention: load is negative source
        "total_loss_kw": dss.Circuit.Losses()[0] / 1000.0,
    }
    logger.info(
        "Baseline power flow converged. Buses: %d, Load: %.2f kW, Loss: %.2f kW",
        stats["total_buses"],
        stats["total_load_kw"],
        stats["total_loss_kw"],
    )
    return stats


def get_all_bus_names() -> list[str]:
    """Return the list of all bus names in the compiled circuit (lowercase)."""
    return [b.lower() for b in dss.Circuit.AllBusNames()]


def verify_bus_ids(
    bus_list: list[int],
    label: str = "buses",
    all_bus_names: list[str] | None = None,
) -> list[int]:
    """Verify that every integer bus ID in *bus_list* exists in the circuit.

    Parameters
    ----------
    bus_list:
        List of integer bus IDs to check.
    label:
        Human-readable label for error messages.
    all_bus_names:
        Pre-fetched list of bus name strings.  If *None*, fetched now.

    Returns
    -------
    list[int]
        The same bus_list if all buses are found.

    Raises
    ------
    ValueError
        If any bus ID is not present in the compiled feeder.
    """
    if all_bus_names is None:
        all_bus_names = get_all_bus_names()

    missing = [b for b in bus_list if str(b).lower() not in all_bus_names]
    if missing:
        raise ValueError(
            f"The following {label} were not found in the IEEE 123-bus feeder: {missing}"
        )
    logger.info("All %d %s verified in feeder.", len(bus_list), label)
    return bus_list


def verify_fault_and_pmu_buses() -> None:
    """Verify all 25 fault buses and all 25 μPMU buses exist in the feeder."""
    all_buses = get_all_bus_names()
    verify_bus_ids(FAULT_BUSES, label="fault buses", all_bus_names=all_buses)
    verify_bus_ids(PMU_BUSES, label="μPMU buses", all_bus_names=all_buses)
    logger.info("All fault and μPMU bus IDs verified successfully.")


# ---------------------------------------------------------------------------
# Green feeder configuration (Gate 4)
# ---------------------------------------------------------------------------


def _find_switch_line(bus_a: int, bus_b: int) -> str | None:
    """Find the OpenDSS Line element connecting bus_a and bus_b.

    Searches all Line elements in the compiled circuit and returns the name
    of the first one whose endpoints match the given bus pair (in either
    direction).  Also handles OpenDSS normally-open naming conventions where
    a bus may be suffixed with ``_open`` (e.g., bus 94 → "94_open").

    Parameters
    ----------
    bus_a, bus_b:
        Integer bus IDs to search for.

    Returns
    -------
    str | None
        Line element name, or ``None`` if not found.
    """
    # Accept both the plain bus ID and the _open variant
    targets_a = {str(bus_a).lower(), f"{bus_a}_open"}
    targets_b = {str(bus_b).lower(), f"{bus_b}_open"}

    flag = dss.Lines.First()
    while flag > 0:
        # Strip phase nodes (e.g. '54.1' → '54')
        b1 = dss.Lines.Bus1().split(".")[0].lower()
        b2 = dss.Lines.Bus2().split(".")[0].lower()
        if (b1 in targets_a and b2 in targets_b) or (
            b1 in targets_b and b2 in targets_a
        ):
            return dss.Lines.Name()
        flag = dss.Lines.Next()
    return None


def configure_green_feeder() -> None:
    """Apply the green (reconfigured) feeder topology from paper §4.3.

    Operations:
      - Open tie switch / line connecting buses 60 and 160.
      - Close switch / line connecting buses 54 and 94.

    The feeder must already be compiled before calling this function.
    Line names are discovered at runtime from the live DSS model so this
    function is robust to naming differences across DSS file versions.

    Raises
    ------
    RuntimeError
        If either switch cannot be found in the compiled circuit.
    """
    from fault_localization.config.constants import (
        GREEN_FEEDER_CLOSE_SWITCH_BUSES,
        GREEN_FEEDER_OPEN_SWITCH_BUSES,
    )

    open_a, open_b = GREEN_FEEDER_OPEN_SWITCH_BUSES    # (60, 160)
    close_a, close_b = GREEN_FEEDER_CLOSE_SWITCH_BUSES  # (54, 94)

    # Find the line to open (60 – 160)
    open_line = _find_switch_line(open_a, open_b)
    if open_line is None:
        raise RuntimeError(
            f"Green feeder: could not find a Line element connecting buses "
            f"{open_a} and {open_b}.  "
            f"Check that the feeder model contains this tie switch."
        )

    # Find the line to close (54 – 94)
    close_line = _find_switch_line(close_a, close_b)
    if close_line is None:
        raise RuntimeError(
            f"Green feeder: could not find a Line element connecting buses "
            f"{close_a} and {close_b}.  "
            f"Check that the feeder model contains this switch."
        )

    # Apply switching operations
    result_open = dss.run_command(f"Open Line.{open_line} 1")
    if result_open:
        logger.warning("Open Line.%s returned: %s", open_line, result_open)

    result_close = dss.run_command(f"Close Line.{close_line} 1")
    if result_close:
        logger.warning("Close Line.%s returned: %s", close_line, result_close)

    # Re-solve to apply the topology change
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        raise RuntimeError(
            "Green feeder: power flow did not converge after switching operations "
            f"(opened Line.{open_line}, closed Line.{close_line})."
        )

    logger.info(
        "Green feeder configured: opened Line.%s (buses %d–%d), "
        "closed Line.%s (buses %d–%d).",
        open_line, open_a, open_b,
        close_line, close_a, close_b,
    )


# ---------------------------------------------------------------------------
# Impedance extraction
# ---------------------------------------------------------------------------


def get_line_impedances() -> list[dict]:
    """Extract per-line R and X data from the compiled feeder.

    Uses the Lines element API from opendssdirect.

    Returns
    -------
    list[dict]
        Each entry: ``{"name": str, "bus1": str, "bus2": str,
        "r1_ohm": float, "x1_ohm": float, "length_km": float,
        "z_mag": float}``.
        ``r1_ohm`` and ``x1_ohm`` are positive-sequence total impedances
        (not per-unit).
    """
    impedances: list[dict] = []
    flag = dss.Lines.First()
    while flag > 0:
        name = dss.Lines.Name()
        bus1 = dss.Lines.Bus1().split(".")[0].lower()
        bus2 = dss.Lines.Bus2().split(".")[0].lower()
        length_km = dss.Lines.Length()   # in km (default for OpenDSS)
        r1_per_km = dss.Lines.R1()       # Ω/km positive-sequence
        x1_per_km = dss.Lines.X1()       # Ω/km positive-sequence

        r1_ohm = r1_per_km * length_km
        x1_ohm = x1_per_km * length_km
        z_mag = (r1_ohm**2 + x1_ohm**2) ** 0.5

        impedances.append(
            {
                "name": name,
                "bus1": bus1,
                "bus2": bus2,
                "r1_ohm": r1_ohm,
                "x1_ohm": x1_ohm,
                "length_km": length_km,
                "z_mag": z_mag,
            }
        )
        flag = dss.Lines.Next()

    logger.info("Extracted impedance data for %d line segments.", len(impedances))
    return impedances


# ---------------------------------------------------------------------------
# Convenience: one-shot setup
# ---------------------------------------------------------------------------


def setup_feeder(master_path: Path | None = None) -> dict:
    """Download, compile, and validate the feeder in one call.

    Returns
    -------
    dict
        Power-flow summary from :func:`run_baseline_powerflow` plus
        ``"impedances"`` key.
    """
    master_path = ensure_feeder_files() if master_path is None else master_path
    compile_feeder(master_path)
    pf_stats = run_baseline_powerflow()
    verify_fault_and_pmu_buses()
    impedances = get_line_impedances()
    return {**pf_stats, "impedances": impedances}
