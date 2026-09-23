"""Build and load the signature table ``operator/``.

``mu[f, tau, R, L, i, p]`` and ``mu0[L, i, p]`` over the axes of
:mod:`fault_localization.physics.grids`, solved by
:mod:`fault_localization.physics.signature_solver` and written to::

    operator/mu.npy          [25, 11, 15, nL, 25, 3] float32, per-unit
    operator/mu0.npy         [nL, 25, 3]             float32, per-unit
    operator/mask.npy        [25, 3]                 int8, phase instrumentation
    operator/grid.json       the axes, the chosen L step, sigma_eps, provenance
    operator/build_log.csv   one row per solve: converged flag and wall time

The table is the only thing in this project that touches OpenDSS after the
corpus was generated, and ``--master`` accepts a *perturbed* feeder file so that
the S6.2 model-mismatch study can build a deliberately wrong table with the same
code path (and therefore no risk of the two differing for an unrelated reason).
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES
from fault_localization.physics import grids
from fault_localization.physics.signature_solver import worker


def _jobs(l_values, r_values, mode, master, reuse_limit):
    """Every solve the table needs, as ``(slot, ...)`` tuples."""
    out = []
    for fi, bus in enumerate(FAULT_BUSES):
        for ti, ftype in enumerate(FAULT_TYPES):
            for ri, resistance in enumerate(r_values):
                for li, load in enumerate(l_values):
                    out.append((("mu", fi, ti, ri, li), bus, ftype, float(resistance),
                                float(load), mode, master, reuse_limit))
    for li, load in enumerate(l_values):
        out.append((("mu0", li), None, None, None, float(load), mode, master, reuse_limit))
    return out


def _chunks(jobs, n_chunks):
    """Contiguous chunks, so each worker's reuse counter sees a long run."""
    size = max(1, (len(jobs) + n_chunks - 1) // n_chunks)
    return [jobs[i:i + size] for i in range(0, len(jobs), size)]


def build_table(
    out_dir: str | Path,
    workers: int = 20,
    mode: str = "daily",
    l_step: float = grids.DEFAULT_L_STEP,
    r_grid: np.ndarray | None = None,
    master: str | Path | None = None,
    reuse_limit: int = 1,
    sigma_eps: float | None = None,
    limit: int | None = None,
) -> dict:
    """Solve the whole grid and write ``operator/``.

    ``limit`` truncates the job list -- for smoke tests only; a truncated build
    writes ``complete: false`` into ``grid.json`` so it can never be mistaken for
    the real table.

    **The circuit is recompiled for every solve, and that is not negotiable.**
    The defaults were ``mode="reuse", reuse_limit=500`` until 2026-09-23, which
    would have silently corrupted the operator: OpenDSS carries **regulator tap
    positions** across solves on a standing circuit, so a hypothesis inherits the
    tap state left by whichever hypothesis the worker happened to solve before
    it.  Measured against 1,000 corpus events at their own (f, tau, R, L):

    ======================  ==============  ===================
    build path              median error    events > 1e-3 p.u.
    ======================  ==============  ===================
    reuse, never recompile  4.9e-09         **43.3 %**
    recompile per solve     1.0e-10         **1.8 %**
    ======================  ==============  ===================

    The corpus recompiles per event (``simulation/runner.py``), so recompiling is
    what *matches the generator*; reuse is a different circuit history.  S0.4
    missed this because it sampled 5 events and reported ``reuse`` and ``daily``
    as bit-identical -- true for those 5, false for 43 % of the corpus.

    The cost is 8.1 ms/solve against 4.4 ms, i.e. a full 17-point-L table takes
    ~9 minutes serial instead of ~3.  There is no scenario in which that is worth
    a wrong operator.

    The residual 1.8 % is not solver noise: it is ~6.5e-3 p.u. -- one regulator
    tap step -- and **16 of the 18 cases sit at R = 10 ohm**, the hard case the
    paper cares about.  It is the corpus's own tap hysteresis, which depends on
    the 40-step Daily history a fresh compile cannot reproduce, and it is the one
    genuinely physical contribution to sigma_eps (S1.4, S1.6).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l_values = grids.l_grid(l_step)
    r_values = np.asarray(grids.R_GRID if r_grid is None else r_grid, dtype=np.float64)
    master_str = str(Path(master)) if master is not None else None

    jobs = _jobs(l_values, r_values, mode, master_str, reuse_limit)
    truncated = limit is not None and limit < len(jobs)
    if truncated:
        jobs = jobs[:limit]

    mu = np.full((len(FAULT_BUSES), len(FAULT_TYPES), len(r_values), len(l_values),
                  25, 3), np.nan, dtype=np.float32)
    mu0 = np.full((len(l_values), 25, 3), np.nan, dtype=np.float32)
    mask = np.zeros((25, 3), dtype=np.int8)

    started = time.perf_counter()
    log_rows: list[dict] = []
    n_bad = 0

    def absorb(results):
        nonlocal n_bad, mask
        for slot, volts, m, ok in results:
            if slot[0] == "mu":
                _, fi, ti, ri, li = slot
                mu[fi, ti, ri, li] = volts
            else:
                mu0[slot[1]] = volts
            if m.sum() > mask.sum():
                mask = m
            if not ok:
                n_bad += 1
            log_rows.append({"slot": "/".join(str(v) for v in slot), "converged": int(ok)})

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for results in pool.imap_unordered(worker, _chunks(jobs, workers * 4)):
                absorb(results)
    else:
        absorb(worker(jobs))

    elapsed = time.perf_counter() - started

    np.save(out_dir / "mu.npy", mu)
    np.save(out_dir / "mu0.npy", mu0)
    np.save(out_dir / "mask.npy", mask)
    with (out_dir / "build_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slot", "converged"])
        w.writeheader()
        w.writerows(log_rows)

    meta = {
        "axis_order": list(grids.AXIS_ORDER),
        "fault_buses": list(FAULT_BUSES),
        "fault_types": list(FAULT_TYPES),
        "r_grid": r_values.tolist(),
        "l_step": float(l_step),
        "l_grid": l_values.tolist(),
        "prefault_freeze": grids.PREFAULT_FREEZE,
        "solve_mode": mode,
        "reuse_limit": reuse_limit,
        "master": master_str,
        "sigma_eps": sigma_eps,
        "n_solves": len(jobs),
        "n_non_converged": n_bad,
        "n_nan": int(np.isnan(mu).any(axis=(-1, -2)).sum() + np.isnan(mu0).any(axis=(-1, -2)).sum()),
        "seconds": elapsed,
        "complete": not truncated,
    }
    (out_dir / "grid.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_table(out_dir: str | Path) -> dict:
    """Load ``operator/`` back, with its grid metadata."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "grid.json").read_text())
    meta["mu"] = np.load(out_dir / "mu.npy")
    meta["mu0"] = np.load(out_dir / "mu0.npy")
    mask_path = out_dir / "mask.npy"
    meta["mask"] = np.load(mask_path) if mask_path.exists() else None
    return meta


def check_monotonic_sag(mu: np.ndarray, mu0: np.ndarray) -> dict:
    """S1.3 sanity: the deepest sag must shrink as R grows, for each (f, tau, L).

    A non-monotonic run almost always means a regulator or cap-bank control
    interacted with the fault; ``03_STAGE_B_operator.md`` S1.3 says decide once
    and match the corpus rather than patching the table.
    """
    sag = (mu0[None, None, None, :, :, :] - mu).max(axis=(-1, -2))   # [f, tau, R, L]
    diffs = np.diff(sag, axis=2)
    violations = int(np.sum(diffs > 1e-6))
    return {
        "n_cells": int(diffs.size),
        "n_violations": violations,
        "fraction": violations / max(diffs.size, 1),
        "monotonic": violations == 0,
    }
