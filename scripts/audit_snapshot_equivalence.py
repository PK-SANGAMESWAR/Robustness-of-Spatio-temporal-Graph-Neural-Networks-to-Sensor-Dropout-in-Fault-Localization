"""Stage A S0.4 -- does a fresh solve reproduce the corpus, and at what error?

This is the gate the whole operator inherits.  The signature table describes the
corpus only as accurately as one re-solve reproduces one recorded event, and the
achieved error **is** sigma_eps, the residual floor of ``01_SPEC.md`` §4.  If it
is large, S1.4 will read it as bias and try to absorb it; if it is mis-measured,
every residual in the project is mis-scaled.

Three things are decided here:

**The freeze assertion.**  The corpus's fault steps run at ``L * 0.99953``, not
``L``.  ``monitors.solve_prefault_timeseries`` rescales the loads at the top of
each pre-fault step and the fault loop of ``simulation/runner.py`` never rescales
again, so the sinusoid is frozen one step short of its period.  This script
asserts it by solving at **both** levels and checking that the frozen one wins --
a difference of ~5e-4 p.u., which is the same order as sigma_eps itself.

**The solve mode.**  ``daily`` (recompile per solve) is faithful and slow;
``reuse`` (keep the circuit) is what the 70k-solve build wants; ``snapshot``
(Mode 0) is the cross-check.  All three are measured against the corpus so the
cheapest faithful one can be chosen on evidence.

**The settling depth.**  Applying a fault produces a transient over the first
post-inception steps (S0.3 measured 1.4e-5 p.u. between steps 40 and 41, settled
by step 42).  The corpus records 20 fault steps, so its recorded post-fault level
is the settled one.  The number of solves the table takes is swept here rather
than assumed.

Usage:
    uv run python scripts/audit_snapshot_equivalence.py [--n-events 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import date
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fault_localization.config.constants import FAULT_RESISTANCES, FAULT_START_STEP
from fault_localization.physics.grids import PREFAULT_FREEZE
from fault_localization.physics.signature_solver import reset_engine, solve_signature

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results/pcstgnn/audit"

TOLERANCE_PU = 1e-3


def pick_events(raw_dir: Path, n: int, seed: int = 20260923) -> list[Path]:
    """n events spanning all three resistances and at least three fault types."""
    files = sorted(raw_dir.glob("*.npz"))
    rng = Random(seed)
    chosen: list[Path] = []
    by_r: dict[float, list[Path]] = {r: [] for r in FAULT_RESISTANCES}
    for path in files:
        tag = path.stem.split("_")
        r = float(tag[2].replace("p", "."))
        if r in by_r:
            by_r[r].append(path)
    types_seen: set[str] = set()
    while len(chosen) < n:
        for r in FAULT_RESISTANCES:
            if len(chosen) >= n:
                break
            pool = by_r[r]
            # Prefer a fault type not yet represented.
            fresh = [p for p in pool if p.stem.split("_")[1] not in types_seen]
            pick = rng.choice(fresh or pool)
            types_seen.add(pick.stem.split("_")[1])
            chosen.append(pick)
    return chosen


def corpus_reference(record: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The settled post-fault level and the instrumentation mask."""
    settled = record[:, FAULT_START_STEP + 3:, :3].mean(axis=1)
    mask = record[:, 0, 3:] > 0.5
    return settled, mask


def compare(volts: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(volts - reference)[mask].max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-events", type=int, default=5)
    ap.add_argument("--fault-steps", type=int, nargs="+", default=[1, 2, 3, 5])
    args = ap.parse_args()
    logging.disable(logging.INFO)

    raw = Path(args.data_root) / "raw_sim"
    events = pick_events(raw, args.n_events)

    rows: list[dict] = []
    for path in events:
        with np.load(path, allow_pickle=True) as loaded:
            record = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        reference, mask = corpus_reference(record)
        load = meta["load_multiplier"]
        spec = (meta["fault_bus"], meta["fault_type"], meta["resistance"])

        for mode in ("daily", "reuse", "snapshot"):
            for steps in args.fault_steps:
                reset_engine()
                volts, _, ok = solve_signature(*spec, load, mode=mode,
                                               reuse_limit=200, n_fault_steps=steps)
                rows.append({
                    "event": path.stem, "resistance": spec[2], "fault_type": spec[1],
                    "load_multiplier": load, "mode": mode, "fault_steps": steps,
                    "freeze": "frozen", "max_abs_err_pu": compare(volts, reference, mask),
                    "converged": int(ok),
                })
        # The freeze assertion: the same solve at the unfrozen load L * 1.0.
        reset_engine()
        unfrozen, _, ok = solve_signature(*spec, load, mode="daily", apply_freeze=False,
                                          n_fault_steps=max(args.fault_steps))
        rows.append({
            "event": path.stem, "resistance": spec[2], "fault_type": spec[1],
            "load_multiplier": load, "mode": "daily", "fault_steps": max(args.fault_steps),
            "freeze": "unfrozen", "max_abs_err_pu": compare(unfrozen, reference, mask),
            "converged": int(ok),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "snapshot_equivalence.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summarise ----------------------------------------------------
    def worst(**filters) -> float:
        sel = [r["max_abs_err_pu"] for r in rows
               if all(r[k] == v for k, v in filters.items())]
        return max(sel) if sel else float("nan")

    grid = {}
    for mode in ("daily", "reuse", "snapshot"):
        for steps in args.fault_steps:
            grid[(mode, steps)] = worst(mode=mode, fault_steps=steps, freeze="frozen")

    best_steps = max(args.fault_steps)
    frozen_err = grid[("daily", best_steps)]
    unfrozen_err = worst(freeze="unfrozen")
    freeze_ok = unfrozen_err > frozen_err
    reuse_matches = abs(grid[("reuse", best_steps)] - frozen_err) < 1e-9
    tolerance_ok = frozen_err < TOLERANCE_PU
    sigma_eps = float(frozen_err)

    summary = {
        "n_events": len(events),
        "events": [p.stem for p in events],
        "tolerance_pu": TOLERANCE_PU,
        "grid_worst_err_pu": {f"{m}/{s}": v for (m, s), v in grid.items()},
        "frozen_err_pu": frozen_err,
        "unfrozen_err_pu": unfrozen_err,
        "freeze_assertion_passes": bool(freeze_ok),
        "reuse_matches_daily": bool(reuse_matches),
        "within_tolerance": bool(tolerance_ok),
        "sigma_eps_pu": sigma_eps,
        "prefault_freeze": PREFAULT_FREEZE,
        "recommended_mode": "reuse" if reuse_matches else "daily",
        "recommended_fault_steps": best_steps,
        "pass": bool(freeze_ok and tolerance_ok),
    }
    (out_dir / "snapshot_equivalence.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"events: {', '.join(summary['events'])}")
    print("worst |V_solve - V_corpus| over instrumented channels, p.u.:")
    print("  mode      " + "".join(f"{s:>12d} steps" for s in args.fault_steps))
    for mode in ("daily", "reuse", "snapshot"):
        print(f"  {mode:<10}" + "".join(f"{grid[(mode, s)]:>18.3e}" for s in args.fault_steps))
    print(f"freeze assertion: frozen (L*{PREFAULT_FREEZE:.6f}) = {frozen_err:.3e} p.u. vs "
          f"unfrozen (L*1.0) = {unfrozen_err:.3e} p.u. -> "
          f"{'FROZEN WINS' if freeze_ok else 'ASSERTION FAILED'}")
    print(f"reuse == daily: {reuse_matches}")
    print(f"sigma_eps = {sigma_eps:.3e} p.u.  (tolerance {TOLERANCE_PU:.0e}: "
          f"{'PASS' if tolerance_ok else 'FAIL'})")
    print(f"VERDICT: {'PASS' if summary['pass'] else 'FAIL'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
