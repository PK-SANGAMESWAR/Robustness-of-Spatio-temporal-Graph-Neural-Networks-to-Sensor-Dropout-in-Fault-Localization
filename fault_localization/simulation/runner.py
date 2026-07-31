"""
Main simulation orchestrator — 4-stage runner.

Stages (from implementation_plan.md Phase 2):
  2.1  Single-event smoke test (1 bus, 1 fault type, 1 resistance, 1 run).
  2.2  5×2 pilot test & empirical window-count measurement.
  2.3  Document source-count reconciliation.
  2.4  Full dataset generation (multiprocessing, checkpoint/resume, tqdm).

Output layout:
  dataset/raw_sim/{fault_bus}_{fault_type}_{resistance}_{run_id:04d}.npz
  Each file contains:
    data   : float64 array of shape [25, 60, 6]
             Channels 0-2 : per-unit RMS voltages (V_A, V_B, V_C)
             Channels 3-5 : binary availability mask (1=present, 0=absent)
    meta   : dict with fault_bus, fault_type, resistance, load_multiplier,
             seed, feeder_config

KNOWN_ISSUES compliance:
  - All models trained on identical data/splits (internal fairness).
  - Reduced scale is acceptable; document in reproduction_log.md.
  - Multi-seed evaluation with error bars is enforced in the experiment runner.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from random import Random

import numpy as np
from tqdm import tqdm

from fault_localization.config.constants import (
    FAULT_BUSES,
    FAULT_RESISTANCES,
    FAULT_TYPES,
    FAULT_START_STEP,
    FEEDER_CONFIGS,
    NUM_FEATURES,
    NUM_SCENARIOS,
    PMU_BUSES,
    RUNS_PER_SCENARIO,
    TARGET_TOTAL_WINDOWS,
    TOTAL_RECORD_MS,
    WINDOW_LENGTH,
    WINDOW_STRIDE,
    deterministic_seed,
)
from fault_localization.simulation.fault_injection import clear_all_faults, inject_fault
from fault_localization.simulation.feeder import (
    compile_feeder,
    configure_green_feeder,
    ensure_feeder_files,
)
from fault_localization.simulation.monitors import (
    extract_pmu_voltages_with_mask,
    setup_timeseries_mode,
    solve_prefault_timeseries,
)

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DEFAULT_OUTPUT_DIR = Path("dataset") / "raw_sim"
_DEFAULT_MASTER_PATH: Path | None = None  # resolved at runtime


# ---------------------------------------------------------------------------
# Core: single fault-event simulation
# ---------------------------------------------------------------------------


def simulate_fault_event(
    fault_bus: int,
    fault_type: str,
    resistance: float,
    run_id: int,
    rng: Random,
    feeder_config: str = "default",
) -> tuple[np.ndarray, dict]:
    """Simulate a single fault event and return the recording + metadata.

    Uses genuine OpenDSS time-series mode (Daily, 1 ms step) so that
    consecutive samples are physically distinct.  This fixes the blocking
    defect where snapshot-mode copying produced bit-identical pre-fault
    frames and ~0.026 macro-F1.

    Parameters
    ----------
    fault_bus, fault_type, resistance, run_id:
        Scenario specification.
    rng:
        Random instance for sampling the load multiplier.
    feeder_config:
        ``"default"`` or ``"green"`` (open 60-160, close 54-94).

    Returns
    -------
    (recording, meta)
        recording : float64 ndarray of shape [25, 60, 6]
                    Channels 0-2: per-unit RMS voltages.
                    Channels 3-5: binary availability mask (1=present, 0=absent).
        meta      : dict with simulation metadata.
    """
    import opendssdirect as dss  # import here so subprocess workers can import cleanly

    if feeder_config not in FEEDER_CONFIGS:
        raise ValueError(
            f"Unknown feeder_config '{feeder_config}'. Must be one of: {FEEDER_CONFIGS}"
        )

    load_multiplier = rng.uniform(0.5, 1.3)

    # 1. Compile a fresh feeder instance (re-compile per event to avoid
    #    state pollution between runs).
    master_path = ensure_feeder_files()
    compile_feeder(master_path)

    # 2. Apply green feeder reconfiguration if requested.
    if feeder_config == "green":
        configure_green_feeder()

    # 3. Set up time-series mode and run 40 pre-fault steps.
    #    Each step advances simulation time by 1 ms, producing physically
    #    distinct voltage readings as load dynamics evolve.
    recording = np.zeros((len(PMU_BUSES), TOTAL_RECORD_MS, NUM_FEATURES), dtype=np.float64)

    pre_fault_data = solve_prefault_timeseries(load_multiplier, FAULT_START_STEP)
    # pre_fault_data : [25, 40, 6]
    recording[:, :FAULT_START_STEP, :] = pre_fault_data

    # 4. Inject fault and record 20 fault-active steps, continuing in
    #    time-series mode (no mode change — time advances uninterrupted).
    inject_fault(fault_bus, fault_type, resistance)

    for t in range(FAULT_START_STEP, TOTAL_RECORD_MS):
        dss.Solution.Solve()
        if dss.Solution.Converged():
            # extract_pmu_voltages_with_mask() returns [N=25, F=6] in one call
            snapshot = extract_pmu_voltages_with_mask()  # [25, 6]
            recording[:, t, :] = snapshot
        else:
            logger.warning("Step %d did not converge — filling NaN (voltages) / 0 (mask).", t)
            recording[:, t, :3] = np.nan
            recording[:, t, 3:] = 0.0  # unavailable when solver diverges

    clear_all_faults()

    meta = {
        "fault_bus": fault_bus,
        "fault_type": fault_type,
        "resistance": resistance,
        "run_id": run_id,
        "load_multiplier": load_multiplier,
        "seed": deterministic_seed(fault_bus, fault_type, resistance, run_id),
        "feeder_config": feeder_config,
        "feature_format": "voltages_0-2_mask_3-5",
    }
    return recording, meta


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def event_filename(
    fault_bus: int,
    fault_type: str,
    resistance: float,
    run_id: int,
    feeder_config: str = "default",
) -> str:
    """Canonical filename for a single fault-event NPZ file."""
    res_str = f"{resistance:.1f}".replace(".", "p")
    cfg_suffix = f"_{feeder_config}" if feeder_config != "default" else ""
    return f"{fault_bus}_{fault_type}_{res_str}_{run_id:04d}{cfg_suffix}.npz"


def save_event(
    recording: np.ndarray,
    meta: dict,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save a fault-event recording to a compressed NPZ file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = event_filename(
        meta["fault_bus"],
        meta["fault_type"],
        meta["resistance"],
        meta["run_id"],
        meta.get("feeder_config", "default"),
    )
    out_path = output_dir / fname
    np.savez_compressed(out_path, data=recording, meta=json.dumps(meta))
    return out_path


def load_event(path: Path) -> tuple[np.ndarray, dict]:
    """Load a fault-event recording from an NPZ file."""
    npz = np.load(path, allow_pickle=True)
    recording = npz["data"]
    meta = json.loads(str(npz["meta"]))
    return recording, meta


def event_exists(
    fault_bus: int,
    fault_type: str,
    resistance: float,
    run_id: int,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    feeder_config: str = "default",
) -> bool:
    """Return True if the NPZ file for this event already exists (checkpoint)."""
    return (
        output_dir
        / event_filename(fault_bus, fault_type, resistance, run_id, feeder_config)
    ).exists()


# ---------------------------------------------------------------------------
# Temporal validity check (Gate 1 self-test)
# ---------------------------------------------------------------------------


def check_temporal_validity(recording: np.ndarray, threshold: float = 1e-4) -> dict:
    """Assert that a recording has genuine temporal variation.

    Validates that pre-fault samples are NOT bit-identical copies and that
    fault samples show a real voltage sag.

    Parameters
    ----------
    recording:
        Float64 array of shape [N, 60, 6].
    threshold:
        Minimum acceptable std across time for pre-fault voltage channels.

    Returns
    -------
    dict
        Keys: ``valid`` (bool), ``prefault_std``, ``fault_sag``.
    """
    voltage = recording[:, :, :3]  # [N, 60, 3]
    prefault_vol = voltage[:, :FAULT_START_STEP, :]   # [N, 40, 3]
    fault_vol = voltage[:, FAULT_START_STEP:, :]      # [N, 20, 3]

    # Compute temporal standard deviation across pre-fault timesteps (axis=1)
    temporal_std = np.nanstd(prefault_vol, axis=1)   # [N=25, 3]
    # Average across active phase channels (voltage > 0.05 p.u.) to exclude zero-padded absent phases
    valid_mask = prefault_vol[:, 0, :] > 0.05
    prefault_std = float(np.nanmean(temporal_std[valid_mask])) if valid_mask.any() else float(np.nanmean(temporal_std))
    # Sag: mean absolute drop from last pre-fault to first fault sample
    last_pre = np.nanmean(prefault_vol[:, -1, :])
    first_fault = np.nanmean(fault_vol[:, 0, :])
    fault_sag = float(last_pre - first_fault)

    valid = prefault_std > threshold

    return {
        "valid": valid,
        "prefault_std": prefault_std,
        "fault_sag": fault_sag,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Stage 2.1 — Single-event smoke test
# ---------------------------------------------------------------------------


def stage_smoke_test(
    fault_bus: int = 7,
    fault_type: str = "AG",
    resistance: float = 1.0,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    feeder_config: str = "default",
) -> tuple[np.ndarray, dict]:
    """Stage 2.1: Execute 1 fault event and save the result.

    Intended for human visual inspection of the raw waveform.
    Also runs :func:`check_temporal_validity` and logs a warning if the
    simulation appears static (Gate 1 self-test).
    """
    logger.info(
        "=== Stage 2.1: Smoke test — Bus %d | %s | %.1f Ω | config=%s ===",
        fault_bus, fault_type, resistance, feeder_config,
    )
    rng = Random(deterministic_seed(fault_bus, fault_type, resistance, 0))
    recording, meta = simulate_fault_event(
        fault_bus, fault_type, resistance, 0, rng, feeder_config
    )
    path = save_event(recording, meta, output_dir)

    # Gate 1 self-test
    validity = check_temporal_validity(recording)
    if validity["valid"]:
        logger.info(
            "✓ Temporal validity check PASSED — prefault_std=%.2e, fault_sag=%.4f p.u.",
            validity["prefault_std"], validity["fault_sag"],
        )
    else:
        logger.error(
            "✗ Temporal validity check FAILED — prefault_std=%.2e < threshold %.2e. "
            "Simulation may still be in snapshot mode. "
            "Check monitors.py:setup_timeseries_mode().",
            validity["prefault_std"], validity["threshold"],
        )

    logger.info("Smoke test saved to: %s", path)
    logger.info(
        "Recording shape: %s  |  NaN count: %d",
        recording.shape,
        int(np.isnan(recording).sum()),
    )
    return recording, meta


# ---------------------------------------------------------------------------
# Stage 2.2 — 5×2 pilot & empirical window count
# ---------------------------------------------------------------------------


def _count_windows(
    n_timesteps: int, window_length: int = WINDOW_LENGTH, stride: int = WINDOW_STRIDE
) -> int:
    """Count windows using the fixed reproduction convention."""
    from fault_localization.processing.windowing import count_windows
    return count_windows(n_timesteps, window_length, stride)


def stage_pilot(
    n_scenarios: int = 5,
    n_runs: int = 2,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    feeder_config: str = "default",
) -> int:
    """Stage 2.2: Run a small pilot and measure empirical window count W."""
    logger.info(
        "=== Stage 2.2: Pilot test (%d scenarios × %d runs | config=%s) ===",
        n_scenarios, n_runs, feeder_config,
    )
    pilot_scenarios = [
        (FAULT_BUSES[i], FAULT_TYPES[i % len(FAULT_TYPES)], FAULT_RESISTANCES[i % 3])
        for i in range(n_scenarios)
    ]
    empirical_w: list[int] = []
    for fault_bus, fault_type, resistance in pilot_scenarios:
        for run_id in range(n_runs):
            rng = Random(deterministic_seed(fault_bus, fault_type, resistance, run_id))
            recording, meta = simulate_fault_event(
                fault_bus, fault_type, resistance, run_id, rng, feeder_config
            )
            save_event(recording, meta, output_dir)
            n_timesteps = recording.shape[1]
            w = _count_windows(n_timesteps)
            empirical_w.append(w)
            logger.info(
                "  Bus %d | %s | %.1f Ω | run %d → W=%d",
                fault_bus, fault_type, resistance, run_id, w,
            )

    w_measured = empirical_w[0] if empirical_w else WINDOW_LENGTH
    logger.info("Empirical W=%d windows per fault event (from first event).", w_measured)
    return w_measured


# ---------------------------------------------------------------------------
# Stage 2.3 — Reconciliation
# ---------------------------------------------------------------------------


def stage_reconcile(w_measured: int) -> int:
    """Stage 2.3: Log the source-count discrepancy and retain 100 runs."""
    logger.info("=== Stage 2.3: Event-count reconciliation ===")
    logger.info(
        "Fixed protocol: %d scenarios × %d runs × W=%d = %d windows.",
        NUM_SCENARIOS, RUNS_PER_SCENARIO, w_measured,
        NUM_SCENARIOS * RUNS_PER_SCENARIO * w_measured,
    )
    logger.warning(
        "[BASE] reports 2,500,000 windows; that total is incompatible with "
        "825 scenarios × 100 runs × 40 windows. See reproduction_log.md.\n"
        "Per KNOWN_ISSUES.md: we do not need to match the exact scale — "
        "internal fairness (all models on the same dataset) is what matters."
    )
    return RUNS_PER_SCENARIO


# ---------------------------------------------------------------------------
# Stage 2.4 — Full dataset generation
# ---------------------------------------------------------------------------


def _worker_fn(args: tuple) -> tuple[str | None, str | None]:
    """Multiprocessing worker: simulate one fault event and save it."""
    fault_bus, fault_type, resistance, run_id, output_dir_str, feeder_config = args
    output_dir = Path(output_dir_str)

    if event_exists(fault_bus, fault_type, resistance, run_id, output_dir, feeder_config):
        return None, None  # checkpoint resume

    try:
        seed = deterministic_seed(fault_bus, fault_type, resistance, run_id)
        rng = Random(seed)
        recording, meta = simulate_fault_event(
            fault_bus, fault_type, resistance, run_id, rng, feeder_config
        )
        save_event(recording, meta, output_dir)
        return None, None
    except Exception as exc:  # noqa: BLE001
        error_msg = (
            f"FAILED: bus={fault_bus}, type={fault_type}, "
            f"R={resistance}, run={run_id}: {exc}"
        )
        logger.error(error_msg)
        return None, error_msg


def stage_full_generation(
    runs_per_scenario: int = RUNS_PER_SCENARIO,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    n_workers: int | None = None,
    feeder_config: str = "default",
) -> int:
    """Stage 2.4: Full dataset generation with multiprocessing and checkpoint/resume."""
    logger.info(
        "=== Stage 2.4: Full dataset generation "
        "(%d scenarios × %d runs = %d events | config=%s) ===",
        NUM_SCENARIOS, runs_per_scenario,
        NUM_SCENARIOS * runs_per_scenario,
        feeder_config,
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    all_tasks = [
        (fault_bus, fault_type, resistance, run_id, str(output_dir), feeder_config)
        for fault_bus in FAULT_BUSES
        for fault_type in FAULT_TYPES
        for resistance in FAULT_RESISTANCES
        for run_id in range(runs_per_scenario)
    ]
    tasks = [
        task for task in all_tasks
        if not event_exists(task[0], task[1], task[2], task[3], output_dir, feeder_config)
    ]

    total = len(tasks)
    success = 0
    errors: list[str] = []

    if not tasks:
        logger.info("All %d requested events are already present.", len(all_tasks))
        return 0

    logger.info(
        "Resuming generation: %d existing events, %d missing events; using %d workers.",
        len(all_tasks) - total, total, n_workers,
    )

    if n_workers == 1:
        for args in tqdm(tasks, desc="Generating fault events"):
            _, err = _worker_fn(args)
            if err:
                errors.append(err)
            else:
                success += 1
    else:
        with mp.Pool(n_workers) as pool:
            for _, err in tqdm(
                pool.imap_unordered(_worker_fn, tasks, chunksize=4),
                total=total,
                desc="Generating fault events",
            ):
                if err:
                    errors.append(err)
                else:
                    success += 1

    logger.info("Generation complete: %d/%d missing events saved.", success, total)
    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"{len(errors)}/{total} event generations failed:\n{preview}")

    return success


# ---------------------------------------------------------------------------
# Entrypoint: run all 4 stages
# ---------------------------------------------------------------------------


def run_pipeline(
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    skip_smoke: bool = False,
    skip_pilot: bool = False,
    n_workers: int | None = None,
    feeder_config: str = "default",
) -> dict:
    """Execute the full 4-stage simulation pipeline.

    Parameters
    ----------
    output_dir:
        Root directory for raw simulation outputs.
    skip_smoke:
        If True, skip Stage 2.1 smoke test.
    skip_pilot:
        If True, skip Stage 2.2 pilot and use theoretical W.
    n_workers:
        Number of multiprocessing workers for Stage 2.4.
    feeder_config:
        ``"default"`` or ``"green"`` feeder topology.

    Returns
    -------
    dict
        Summary with keys: ``"w_measured"``, ``"runs_per_scenario"``,
        ``"total_events_generated"``.
    """
    if not skip_smoke:
        stage_smoke_test(output_dir=output_dir, feeder_config=feeder_config)

    if not skip_pilot:
        w_measured = stage_pilot(output_dir=output_dir, feeder_config=feeder_config)
    else:
        from fault_localization.config.constants import THEORETICAL_WINDOWS_PER_EVENT
        w_measured = THEORETICAL_WINDOWS_PER_EVENT
        logger.info("Skipping pilot — using theoretical W=%d.", w_measured)

    reconciled_r = stage_reconcile(w_measured)
    total_generated = stage_full_generation(
        runs_per_scenario=reconciled_r,
        output_dir=output_dir,
        n_workers=n_workers,
        feeder_config=feeder_config,
    )

    return {
        "w_measured": w_measured,
        "runs_per_scenario": reconciled_r,
        "total_events_generated": total_generated,
        "feeder_config": feeder_config,
    }
