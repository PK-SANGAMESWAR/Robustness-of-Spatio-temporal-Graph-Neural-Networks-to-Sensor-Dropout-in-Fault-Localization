"""Stage A/B -- does the table build path match the corpus generator?

Written after the S1.3 non-monotone investigation turned up a build-path bug that
would have silently corrupted the operator.  Every number these checks produce is
quoted in ``results/pcstgnn/audit/AUDIT-NOTE.md`` and in
``PC-STGNN/03_STAGE_B_operator.md`` S1.3/S1.4, so they are kept runnable rather
than left in a scratchpad.

    uv run python scripts/audit_build_path.py --checks modes monotonic controls

``modes``      -- table vs corpus for each solve mode.  THE important one: the
                 standing-circuit path carries regulator taps between solves and
                 disagrees with the corpus on 43% of events.
``prefault``   -- rules out pre-fault step count as the cause.
``monotonic``  -- sag-vs-R monotonicity on the real R grid, and whether the
                 corpus agrees.
``controls``   -- is the non-monotonicity control-driven?  (Mostly not.)
``settling``   -- are the three S0.3 settling outliers control-driven?  (One is.)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from random import Random

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES  # noqa: E402
from fault_localization.physics import signature_solver as S  # noqa: E402
from fault_localization.physics.grids import R_GRID  # noqa: E402

DEFAULT_OUT = REPO / "results/pcstgnn/audit"
R_CORPUS = [0.1, 1.0, 10.0]
SETTLING_PROBE = ["55_BC_1p0_0007.npz", "7_CA_10p0_0002.npz", "51_ABC_1p0_0000.npz",
                  "53_ABG_10p0_0035.npz"]
MONO_PROBE = [(80, "AB"), (80, "BC"), (80, "CA"), (89, "AB"), (89, "CA"), (93, "BC"),
              (51, "ABC"), (55, "BC"), (7, "CA")]


def _events(raw_dir: Path, n: int, seed: int):
    """(instrumented mask, settled post-fault level, meta) for n random events."""
    files = sorted(raw_dir.glob("*.npz"))
    picks = Random(seed).sample(files, min(n, len(files)))
    out = []
    for path in picks:
        with np.load(path, allow_pickle=True) as loaded:
            rec = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        out.append((path.name, rec[:, 0, 3:] > 0.5, rec[:, 45, :3], meta))
    return out


def _err(event, **kwargs) -> float:
    _, inst, settled, meta = event
    v, _m, _ok = S.solve_signature(meta["fault_bus"], meta["fault_type"],
                                   meta["resistance"], meta["load_multiplier"], **kwargs)
    return float(np.abs(v - settled)[inst].max())


def check_modes(raw_dir: Path, n: int) -> dict:
    """The build path must recompile per solve -- taps carry over otherwise."""
    events = _events(raw_dir, n, seed=2)
    rows = []
    for label, mode, limit in (("reuse_never_recompile", "reuse", 10**9),
                               ("reuse_limit_1", "reuse", 1),
                               ("daily", "daily", 1),
                               ("snapshot", "snapshot", 1)):
        S.reset_engine()
        t0 = time.time()
        errs = np.array([_err(e, mode=mode, reuse_limit=limit) for e in events])
        worst = sorted(zip((e[0] for e in events), (e[3]["resistance"] for e in events), errs),
                       key=lambda r: -r[2])[:20]
        rows.append({
            "path": label, "n_events": len(events),
            "median": float(np.median(errs)), "p95": float(np.percentile(errs, 95)),
            "max": float(errs.max()),
            "frac_gt_1e-3": float((errs > 1e-3).mean()),
            "ms_per_solve": 1000 * (time.time() - t0) / len(events),
            "worst_by_R": collections.Counter(R for _n, R, e in worst if e > 1e-4),
        })
        r = rows[-1]
        print(f"  {label:22s} median {r['median']:.2e}  p95 {r['p95']:.2e}  "
              f"max {r['max']:.2e}  >1e-3 {100*r['frac_gt_1e-3']:5.1f}%  "
              f"{r['ms_per_solve']:.1f} ms/solve")
    for row in rows:
        row["worst_by_R"] = {str(k): int(v) for k, v in row["worst_by_R"].items()}
    return {"rows": rows}


def check_prefault(raw_dir: Path, n: int) -> dict:
    """Rules out pre-fault step count: it changes nothing."""
    events = _events(raw_dir, n, seed=1)
    rows = []
    for steps in (2, 5, 10, 20, 40):
        S.reset_engine()
        errs = np.array([_err(e, mode="reuse", reuse_limit=10**9, n_prefault_steps=steps)
                         for e in events])
        rows.append({"n_prefault_steps": steps, "median": float(np.median(errs)),
                     "max": float(errs.max()), "frac_gt_1e-3": float((errs > 1e-3).mean())})
        print(f"  n_prefault={steps:3d}  median {rows[-1]['median']:.2e}  "
              f"max {rows[-1]['max']:.2e}  >1e-3 {100*rows[-1]['frac_gt_1e-3']:5.1f}%")
    return {"rows": rows}


def check_monotonic(raw_dir: Path, l_probe=(0.5, 0.9, 1.3)) -> dict:
    """Sag-vs-R monotonicity on the real grid, and whether the corpus agrees."""
    S.reset_engine()
    nb, nt, nr, nl = len(FAULT_BUSES), len(FAULT_TYPES), len(R_GRID), len(l_probe)
    mu = np.zeros((nb, nt, nr, nl, 25, 3))
    mu0 = np.zeros((nl, 25, 3))
    nonconv = 0
    t0 = time.time()
    for li, L in enumerate(l_probe):
        mu0[li] = S.solve_nofault(L, mode="daily", reuse_limit=1)[0]
    for bi, b in enumerate(FAULT_BUSES):
        for ti, t in enumerate(FAULT_TYPES):
            for ri, R in enumerate(R_GRID):
                for li, L in enumerate(l_probe):
                    v, _m, ok = S.solve_signature(b, t, float(R), L, mode="daily", reuse_limit=1)
                    mu[bi, ti, ri, li] = v
                    nonconv += (not ok)
    sag = (mu0[None, None, None] - mu).max(axis=(-1, -2))
    viol = np.diff(sag, axis=2) > 1e-6
    ung = [FAULT_TYPES.index(t) for t in FAULT_TYPES if not t.endswith("G")]
    out = {
        "n_solves": nb * nt * nr * nl + nl, "seconds": time.time() - t0,
        "n_nonconverged": int(nonconv),
        "n_cells": int(viol.size), "n_violations": int(viol.sum()),
        "fraction": float(viol.mean()),
        "violations_by_type": {FAULT_TYPES[i]: int(viol[:, i].sum()) for i in range(nt)},
        "violations_ungrounded": int(viol[:, ung].sum()),
        "buses_affected": [int(FAULT_BUSES[i]) for i in np.where(viol.any(axis=(1, 2, 3)))[0]],
    }
    print(f"  grid: {out['n_violations']}/{out['n_cells']} = {100*out['fraction']:.2f}% "
          f"non-monotone; {out['violations_ungrounded']} of them ungrounded; "
          f"{out['n_nonconverged']} non-converged ({out['seconds']:.0f}s)")

    # Does the corpus agree, at its own three resistances?
    corpus = collections.defaultdict(list)
    for path in sorted(raw_dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as loaded:
            rec = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        inst = rec[:, 0, 3:] > 0.5
        corpus[(meta["fault_bus"], meta["fault_type"], meta["resistance"])].append(
            float((rec[:, 39] - rec[:, 45])[:, :3][inst].max()))
    agree = cn = 0
    cells_nonmono = []
    for b in FAULT_BUSES:
        for t in FAULT_TYPES:
            c = [np.median(corpus[(b, t, R)]) for R in R_CORPUS]
            tab = [float((mu0[1] - mu[FAULT_BUSES.index(b), FAULT_TYPES.index(t),
                                     int(np.abs(R_GRID - R).argmin()), 1]).max())
                   for R in R_CORPUS]
            c_mono = c[0] >= c[1] >= c[2]
            agree += (c_mono == (tab[0] >= tab[1] >= tab[2]))
            if not c_mono:
                cn += 1
                cells_nonmono.append((int(b), t))
    out.update({
        "corpus_cells": len(FAULT_BUSES) * len(FAULT_TYPES),
        "corpus_table_monotonicity_agree": int(agree),
        "corpus_nonmonotone_cells": cn,
        "corpus_nonmonotone_all_ungrounded": all(not t.endswith("G") for _b, t in cells_nonmono),
    })
    print(f"  corpus: agrees with table on {agree}/{out['corpus_cells']} cells; "
          f"{cn} non-monotone, all ungrounded: {out['corpus_nonmonotone_all_ungrounded']}")
    return out


def _sags_under(mode: str, cells) -> dict:
    import opendssdirect as dss
    S.reset_engine()
    out, first = {}, True
    for b, t in cells:
        row = []
        for R in R_CORPUS:
            v, m, _ = S.solve_signature(b, t, R, 0.9, mode="reuse", reuse_limit=10**9)
            if first:
                dss.Text.Command(f"Set ControlMode={mode}")
                v, m, _ = S.solve_signature(b, t, R, 0.9, mode="reuse", reuse_limit=10**9)
                first = False
            v0 = S.solve_nofault(0.9, mode="reuse", reuse_limit=10**9)[0]
            row.append(float((v0 - v)[m > 0.5].max()))
        out[(b, t)] = row
    return out


def check_controls() -> dict:
    """Is the non-monotonicity a regulator/cap-bank interaction?  Mostly not."""
    on, off = _sags_under("STATIC", MONO_PROBE), _sags_under("OFF", MONO_PROBE)
    rows, n_on, n_off = [], 0, 0
    for key in on:
        a, b_ = on[key], off[key]
        ma, mb = a[0] >= a[1] >= a[2], b_[0] >= b_[1] >= b_[2]
        n_on += (not ma); n_off += (not mb)
        rows.append({"bus": key[0], "type": key[1], "sag_controls_on": a,
                     "sag_controls_off": b_, "monotone_on": ma, "monotone_off": mb})
        print(f"  bus {key[0]:4d} {key[1]:5s}  monotone ON {str(ma):5s}  OFF {mb}")
    delta = max(float(np.abs(np.array(on[k]) - np.array(off[k])).max()) for k in on)
    print(f"  non-monotone: {n_on}/{len(on)} with controls, {n_off}/{len(off)} without; "
          f"max |ON-OFF| sag shift {delta:.3e} p.u.")
    return {"rows": rows, "n_nonmonotone_controls_on": n_on,
            "n_nonmonotone_controls_off": n_off, "max_sag_shift_pu": delta}


def check_settling(raw_dir: Path) -> dict:
    """Are the three S0.3 settling outliers control-driven?  One provably is."""
    import opendssdirect as dss
    rows = {}
    for mode in ("STATIC", "OFF"):
        S.reset_engine()
        first = True
        for name in SETTLING_PROBE:
            with np.load(raw_dir / name, allow_pickle=True) as loaded:
                meta = json.loads(str(loaded["meta"]))
            args = dict(mode="reuse", reuse_limit=10**9)
            v1, m, _ = S.solve_signature(meta["fault_bus"], meta["fault_type"],
                                         meta["resistance"], meta["load_multiplier"],
                                         n_fault_steps=1, **args)
            if first:
                dss.Text.Command(f"Set ControlMode={mode}")
                v1, m, _ = S.solve_signature(meta["fault_bus"], meta["fault_type"],
                                             meta["resistance"], meta["load_multiplier"],
                                             n_fault_steps=1, **args)
                first = False
            v3, _, _ = S.solve_signature(meta["fault_bus"], meta["fault_type"],
                                         meta["resistance"], meta["load_multiplier"],
                                         n_fault_steps=3, **args)
            rows.setdefault(name, {})[mode] = float(np.abs(v3 - v1)[m > 0.5].max())
    for name, r in rows.items():
        verdict = ("control-driven" if r["STATIC"] > 1e-4 and r["OFF"] < 1e-4
                   else "not control-driven" if r["STATIC"] > 1e-4 else "normal")
        r["verdict"] = verdict
        print(f"  {name:26s} ON {r['STATIC']:.3e}  OFF {r['OFF']:.3e}  {verdict}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--checks", nargs="+",
                    default=["modes", "monotonic", "controls", "settling"],
                    choices=["modes", "prefault", "monotonic", "controls", "settling"])
    ap.add_argument("--n-events", type=int, default=1000)
    args = ap.parse_args()

    raw = Path(args.data_root) / "raw_sim"
    out: dict = {"data_root": args.data_root, "n_events": args.n_events}
    for name in args.checks:
        print(f"[{name}]", flush=True)
        if name == "modes":
            out[name] = check_modes(raw, args.n_events)
        elif name == "prefault":
            out[name] = check_prefault(raw, min(args.n_events, 150))
        elif name == "monotonic":
            out[name] = check_monotonic(raw)
        elif name == "controls":
            out[name] = check_controls()
        elif name == "settling":
            out[name] = check_settling(raw)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "build_path.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
