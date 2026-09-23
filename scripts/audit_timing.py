"""Stage A S0.5 -- how fast does this machine solve, and what does that cost?

``13_BUDGET_TIMELINE.md`` was written against an assumed ~45 solves/s.  Every
CPU line in it (the signature table, the L- and R-grid studies, the perturbed
tables for S6.2, the two OOD corpora) scales linearly off that number, so it is
measured here and the budget rescaled from the measurement rather than the
guess.

Training time is **not** re-measured: it is already on disk from Paper 1
(``results/runtime/runtime_training.csv``), and re-running it would cost GPU-days
to learn nothing.

Usage:
    uv run python scripts/audit_timing.py [--n-solves 100] [--n-events 20]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES
from fault_localization.physics import grids
from fault_localization.physics.signature_solver import (
    DEFAULT_FAULT_STEPS,
    reset_engine,
    solve_signature,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results/pcstgnn/audit"


def time_solves(n: int, mode: str, fault_steps: int) -> float:
    """Seconds per signature solve."""
    reset_engine()
    rng = Random(0)
    jobs = [(rng.choice(FAULT_BUSES), rng.choice(FAULT_TYPES),
             float(rng.choice(grids.R_GRID)), float(rng.choice(grids.l_grid(0.05))))
            for _ in range(n)]
    solve_signature(*jobs[0], mode=mode, reuse_limit=n + 10, n_fault_steps=fault_steps)
    started = time.perf_counter()
    for job in jobs:
        solve_signature(*job, mode=mode, reuse_limit=n + 10, n_fault_steps=fault_steps)
    return (time.perf_counter() - started) / n


def time_daily_events(n: int) -> float:
    """Seconds per full 60-step Daily-mode event -- the OOD generation unit."""
    from fault_localization.simulation.runner import simulate_fault_event

    rng = Random(1)
    # An in-distribution resistance: the wall time does not depend on R, and
    # simulate_fault_event's default metadata seed is only defined for these.
    jobs = [(rng.choice(FAULT_BUSES), rng.choice(FAULT_TYPES), 1.0, 9000 + i)
            for i in range(n)]
    simulate_fault_event(*jobs[0], Random(0))
    started = time.perf_counter()
    for bus, ftype, resistance, run_id in jobs:
        simulate_fault_event(bus, ftype, resistance, run_id, Random(run_id))
    return (time.perf_counter() - started) / n


def training_costs() -> list[dict]:
    path = REPO / "results/runtime/runtime_training.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-solves", type=int, default=100)
    ap.add_argument("--n-events", type=int, default=20)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    logging.disable(logging.INFO)

    per_solve = {mode: time_solves(args.n_solves, mode, DEFAULT_FAULT_STEPS)
                 for mode in ("reuse", "daily")}
    per_event = time_daily_events(args.n_events)

    rate = 1.0 / per_solve["reuse"]
    parallel_rate = rate * args.workers            # optimistic; these are independent
    tables = {}
    for step in (0.05, 0.025):
        n_l = len(grids.l_grid(step))
        n_solves = 25 * 11 * len(grids.R_GRID) * n_l + n_l
        tables[step] = {
            "n_l": n_l, "n_solves": n_solves,
            "serial_minutes": n_solves * per_solve["reuse"] / 60.0,
            "parallel_minutes": n_solves / parallel_rate / 60.0,
            "size_mb": 25 * 11 * len(grids.R_GRID) * n_l * 75 * 4 / 1e6,
        }
    ood = {
        "pilot_events": 25 * 11 * 2 * 2,
        "test_events": 25 * 11 * 4 * 10,
    }
    ood["pilot_hours_serial"] = ood["pilot_events"] * per_event / 3600.0
    ood["test_hours_serial"] = ood["test_events"] * per_event / 3600.0
    ood["pilot_hours_parallel"] = ood["pilot_hours_serial"] / args.workers
    ood["test_hours_parallel"] = ood["test_hours_serial"] / args.workers

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"per_solve_seconds": per_solve, "solves_per_second_single_core": rate,
               "workers_assumed": args.workers, "per_daily_event_seconds": per_event,
               "fault_steps": DEFAULT_FAULT_STEPS, "tables": tables, "ood": ood,
               "training_csv_rows": training_costs()}
    (out_dir / "timing.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# S0.5 -- Timing calibration",
        "",
        f"Generated {date.today().isoformat()} by `scripts/audit_timing.py` on this machine "
        f"(24 cores, T1000 8 GB).",
        "",
        "## Measured",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| signature solve, `reuse` mode, {DEFAULT_FAULT_STEPS} fault steps | "
        f"**{per_solve['reuse']*1000:.2f} ms** ({rate:.0f}/s single-core) |",
        f"| signature solve, `daily` mode (recompile each time) | "
        f"{per_solve['daily']*1000:.1f} ms ({1/per_solve['daily']:.1f}/s) |",
        f"| full 60-step Daily-mode event (the OOD unit) | {per_event*1000:.0f} ms |",
        "",
        f"The plan assumed ~45 solves/s.  The measured single-core rate is **{rate:.0f}/s**, "
        f"i.e. {rate/45:.0f}x faster, because `reuse` mode keeps the compiled circuit and the "
        "compile -- not the solve -- was the cost being estimated.",
        "",
        "## Signature table",
        "",
        "| L step | L points | solves | serial | 20 workers | size |",
        "|---|---|---|---|---|---|",
    ]
    for step, t in tables.items():
        lines.append(f"| {step} | {t['n_l']} | {t['n_solves']:,} | "
                     f"{t['serial_minutes']:.1f} min | {t['parallel_minutes']:.1f} min | "
                     f"{t['size_mb']:.0f} MB |")
    lines += [
        "",
        "**Consequence for S1.2.** Both candidate L steps cost minutes, not hours, so the grid "
        "step should be chosen purely on interpolation error -- there is no budget argument for "
        "the coarser grid.",
        "",
        "## OOD corpora",
        "",
        "| set | events | serial | 20 workers |",
        "|---|---|---|---|",
        f"| pilot | {ood['pilot_events']:,} | {ood['pilot_hours_serial']:.2f} h | "
        f"{ood['pilot_hours_parallel']*60:.0f} min |",
        f"| sealed test | {ood['test_events']:,} | {ood['test_hours_serial']:.2f} h | "
        f"{ood['test_hours_parallel']*60:.0f} min |",
        "",
        "## GPU (not re-measured)",
        "",
        "From `results/runtime/runtime_training.csv`: RGSAGE-Max 30 epochs = 6.7 h standard / "
        "6.9 h dropout-aware; RGATv2 12.3 h.  One run therefore exceeds 6 h and the reduced-seed "
        "rule of `02_STAGE_A_audit.md` S0.5 stands: 3 seeds for R0, R2, R4, R5, R6, G0, G2; "
        "1 seed for R1, R3, R7, G1, G3, B2a.",
    ]
    (out_dir / "timing.md").write_text("\n".join(lines) + "\n")

    print(f"signature solve (reuse, {DEFAULT_FAULT_STEPS} steps): "
          f"{per_solve['reuse']*1000:.2f} ms -> {rate:.0f}/s single-core")
    print(f"signature solve (daily, recompile):     {per_solve['daily']*1000:.1f} ms "
          f"-> {1/per_solve['daily']:.1f}/s")
    print(f"full 60-step Daily event:               {per_event*1000:.0f} ms")
    for step, t in tables.items():
        print(f"table L-step {step}: {t['n_solves']:,} solves -> "
              f"{t['serial_minutes']:.1f} min serial / {t['parallel_minutes']:.1f} min "
              f"on {args.workers} workers, {t['size_mb']:.0f} MB")
    print(f"OOD pilot {ood['pilot_events']:,} ev -> {ood['pilot_hours_serial']:.2f} h serial; "
          f"sealed {ood['test_events']:,} ev -> {ood['test_hours_serial']:.2f} h serial")
    print(f"wrote {out_dir / 'timing.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
