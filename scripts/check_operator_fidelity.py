"""Stage B operator fidelity checks (``PC-STGNN/03_STAGE_B_operator.md``).

    uv run python scripts/check_operator_fidelity.py --checks lgrid --workers 8

``[Fixed 2026-09-25]`` ``margin_distribution`` now chunks over the event batch too
(``chunk_b``), not just over ``chunk_f``. The R-grid fix (``grids.R_GRID`` 15->18 points,
see ``results/pcstgnn/operator/S1.5_RGRID_CORRECTION.md``) grew H = 11*NUM_R*n_l enough
that the old B=3000-at-once ``[B, chunk_f, H, 75]`` intermediate reached ~12-24GB for the
larger L-step tables and was OOM-killed repeatedly on this 15GB box. Batching over B is a
pure memory fix -- identical rho/margin/PSM per event, just computed in smaller pieces.

``lgrid`` (S1.2) **decides the L grid** and must therefore be run at full
population, not on a convenience sample -- see the sampling rule in
``12_CHECKLISTS.md``.  It produces two things:

1. the linear-interpolation error of each candidate step against a finer
   reference grid, over **every** (f, tau, R) cell -- 4,125 of them; and
2. the **top-2 rho margin distribution** per candidate step, which is what the
   step is selected on.

The second is the point.  Fidelity is free on this corpus (the operator is the
corpus's own generator, S0.4/S2.0a), so a finer L grid does not buy accuracy --
it drives rho_true toward 0, widens the margin, and pushes softmax(-rho) toward
one-hot, which *degenerates* the fusion (04_STAGE_C_gonogo.md S2.0a consequence
4).  The target is an informative, non-saturating margin, not the largest one
that runs.

Nothing here selects the step automatically.  It reports the distributions and
the human chooses, because the trade-off is a judgement about how much evidence
the physics prior should carry -- not a threshold.
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
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES  # noqa: E402
from fault_localization.physics import grids, residual as R, table as T  # noqa: E402

DEFAULT_OUT = REPO / "results/pcstgnn/operator/lgrid"
SETTLED_STEP = 45          # a settled post-fault sample (S0.3)

#: sigma_eps values the margin is reported at.  Per 01_SPEC.md §4 the floor is
#: never set below the uPMU accuracy class, so these bracket the realistic range;
#: the per-step discretisation error is reported alongside them and is normally
#: far below, which is exactly why it does not get to set the scale alone.
SIGMA_GRID = (1e-3, 5e-3, 1e-2)


def build_candidates(out_root: Path, steps, reference, workers, master=None) -> dict:
    """Build one table per candidate L step plus the reference grid."""
    built = {}
    for step in sorted({*steps, reference}, reverse=True):
        out = out_root / f"step_{str(step).replace('.', 'p')}"
        if (out / "mu.npy").exists():
            print(f"  [skip] L step {step}: already built at {out}", flush=True)
        else:
            t0 = time.perf_counter()
            # mode="daily"/reuse_limit=1 -- recompile per solve.  A standing
            # circuit carries regulator taps between hypotheses and disagrees
            # with the corpus on 43% of events (scripts/audit_build_path.py).
            T.build_table(out, workers=workers, mode="daily", reuse_limit=1,
                          l_step=step, master=master)
            print(f"  built L step {step} ({len(grids.l_grid(step))} pts) "
                  f"in {time.perf_counter() - t0:.0f}s", flush=True)
        built[step] = T.load_table(out)
    return built


def interpolation_error(cand: dict, ref: dict, step: float, reference: float) -> dict:
    """Linear-interpolation error of a candidate grid against the reference.

    Full population: every (f, tau, R, sensor, phase), at every reference L that
    the candidate grid does not already contain.
    """
    l_c, l_r = grids.l_grid(step), grids.l_grid(reference)
    mu_c, mu_r = cand["mu"], ref["mu"]                     # [f,t,R,L,25,3]
    keep = ~np.isin(np.round(l_r, 9), np.round(l_c, 9))    # only interpolated points

    # Vectorised linear interpolation along axis 3.  NB: a plain
    # ``reshape(-1, n_L)`` does NOT isolate that axis -- L is axis 3 of 6, so the
    # sensor and phase axes vary faster and the reshape would interpolate across
    # *channels*.  Index the axis explicitly instead.
    hi = np.clip(np.searchsorted(l_c, l_r, side="left"), 1, len(l_c) - 1)
    lo = hi - 1
    w = (l_r - l_c[lo]) / (l_c[hi] - l_c[lo])              # [L_ref]
    wq = w[None, None, None, :, None, None]
    interp = mu_c[:, :, :, lo] * (1.0 - wq) + mu_c[:, :, :, hi] * wq
    err = np.abs(interp - mu_r)[:, :, :, keep]             # [f,t,R,Lk,25,3]
    mask = ref["mask"] > 0 if "mask" in ref else np.ones((25, 3), bool)
    err = err[..., mask]

    ung = [i for i, t in enumerate(FAULT_TYPES) if not t.endswith("G")]
    grd = [i for i, t in enumerate(FAULT_TYPES) if t.endswith("G")]
    out = {
        "l_step": step, "n_points": len(l_c),
        "n_cells_full_population": int(mu_c[..., 0, 0].size),
        "n_interpolated_L": int(keep.sum()),
        "max": float(err.max()), "p99": float(np.percentile(err, 99)),
        "p50": float(np.median(err)),
        "by_R": {f"{r:.4f}": float(err[:, :, i].max()) for i, r in enumerate(grids.R_GRID)},
        "max_ungrounded": float(err[:, ung].max()),
        "max_grounded": float(err[:, grd].max()),
    }
    return out


def tap_discontinuities(ref: dict, reference_step: float, jump_threshold: float = 5e-3) -> dict:
    """Locate the regulator-tap steps in mu(L) -- the floor no L grid can cross.

    S1.2 originally asked for "the coarsest step whose interpolation error is
    < 1e-3 p.u.".  That criterion is **not achievable at any affordable step**,
    and this is why: mu(L) is not smooth.  A rising load eventually trips a
    regulator tap, and at that load the whole feeder's voltage steps by ~one tap.
    Linear interpolation across a step function errs by ~the step size no matter
    how fine the grid, so the tail of the interpolation error is flat in the grid
    step while the bulk converges normally.

    The signature of a tap event, as opposed to anything fault-related, is that
    it is **circuit-wide**: the same L interval moves a majority of (f, tau, R)
    cells at once, and it moves the no-fault row mu0(L) too.  Both are checked.
    """
    mu, mu0 = ref["mu"], ref["mu0"]
    mask = ref["mask"] > 0 if "mask" in ref else np.ones((25, 3), bool)
    l_values = grids.l_grid(reference_step)

    step = np.abs(np.diff(mu, axis=3))[..., mask]          # [f,t,R,L-1,C]
    big = (step > jump_threshold).any(axis=-1)             # [f,t,R,L-1]
    per_interval = big.sum(axis=(0, 1, 2))
    n_cells = big[..., 0].size
    nofault = np.abs(np.diff(mu0, axis=0))[..., mask].max(axis=-1)   # [L-1]

    boundaries = []
    for i in np.where(per_interval > 0.25 * n_cells)[0]:
        boundaries.append({
            "l_low": float(l_values[i]), "l_high": float(l_values[i + 1]),
            "cells_moved_frac": float(per_interval[i] / n_cells),
            "nofault_jump_pu": float(nofault[i]),
        })
    return {
        "reference_step": reference_step, "jump_threshold": jump_threshold,
        "max_adjacent_jump_pu": float(step.max()),
        "median_adjacent_delta_pu": float(np.median(step)),
        "n_tap_boundaries": len(boundaries),
        "boundaries": boundaries,
        "nofault_max_jump_pu": float(nofault.max()),
        "implied_sigma_eps_floor_pu": float(np.median([b["nofault_jump_pu"]
                                                       for b in boundaries]))
        if boundaries else None,
    }


def _stratified_events(raw_dir: Path, n: int, seed: int):
    """Sample events stratified by (R, grounded/ungrounded), equally per stratum."""
    files = sorted(raw_dir.glob("*.npz"))
    strata = collections.defaultdict(list)
    for path in files:
        bus, ftype, rtag, _run = path.stem.split("_")
        strata[(rtag, ftype.endswith("G"))].append(path)
    rng = Random(seed)
    per = max(1, n // len(strata))
    picks = []
    for key in sorted(strata, key=str):
        picks += rng.sample(strata[key], min(per, len(strata[key])))
    return picks, {str(k): len(v) for k, v in sorted(strata.items(), key=lambda kv: str(kv[0]))}, per


def _observe(paths):
    """(s [B,25,3] p.u., a [B,25,3], labels) at a settled post-fault sample."""
    s, a, y = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as loaded:
            rec = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        s.append(rec[:, SETTLED_STEP, :3])
        a.append((rec[:, 0, 3:] > 0.5).astype(np.float32))
        y.append(FAULT_BUSES.index(meta["fault_bus"]) + 1)
    return (torch.tensor(np.stack(s), dtype=torch.float32),
            torch.tensor(np.stack(a), dtype=torch.float32),
            np.asarray(y))


def _boot_ci(values, stat, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    draws = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def margin_distribution(tbl: dict, step: float, s, a, y, sigma_eps_disc: float,
                        chunk_f: int, chunk_b: int = 200) -> list[dict]:
    """Top-2 rho margin per candidate step, at each sigma_eps in SIGMA_GRID.

    The margin is the quantity beta and the lambda terms actually see, and the
    one S1.2 selects on (11_METRICS.md §2).

    ``[Fixed 2026-09-25]`` ``raw_residual``'s ``[B, chunk_f, H, 75]`` intermediate scales
    with H = 11*NUM_R*n_l, and NUM_R grew 15->18 in the R-grid fix. At the full
    B=3000 margin sample and the original chunk_f=2, that intermediate is ~6-12GB
    (two of them live at once) on a 15GB box -- OOM-killed every run at exactly the
    step_0.05 table (H=3366), reproducibly. Batching over B here (chunk_b events at
    a time, results concatenated before any statistic is computed) is a pure memory
    fix: every rho/margin/psm value is identical to computing them in one shot.
    """
    flat = R.flatten_table(tbl["mu"], tbl["mu0"])
    rows = []
    for sigma in (*SIGMA_GRID, sigma_eps_disc):
        rho_parts = []
        for b0 in range(0, len(y), chunk_b):
            b1 = min(b0 + chunk_b, len(y))
            r_part, _hyp = R.rho(s[b0:b1], a[b0:b1], flat, sigma, chunk_f=chunk_f)
            rho_parts.append(r_part)
        rho = torch.cat(rho_parts, dim=0)
        marg = R.margin(rho).cpu().numpy()
        pred = R.psm_predict(rho).cpu().numpy()
        rho_true = rho[torch.arange(len(y), device=rho.device),
                       torch.as_tensor(y, device=rho.device)].cpu().numpy()
        lo, hi = _boot_ci(marg, np.median)
        rows.append({
            "l_step": step, "sigma_eps": float(sigma),
            "is_discretisation_sigma": bool(sigma == sigma_eps_disc),
            "n_events": int(len(y)),
            "margin_p10": float(np.percentile(marg, 10)),
            "margin_median": float(np.median(marg)),
            "margin_median_ci95": [lo, hi],
            "margin_p90": float(np.percentile(marg, 90)),
            "frac_margin_gt_10nat": float((marg > 10).mean()),
            "frac_margin_lt_1nat": float((marg < 1).mean()),
            "rho_true_median": float(np.median(rho_true)),
            "psm_top1": float((pred == y).mean()),
        })
        r = rows[-1]
        print(f"    sigma={sigma:.2e}{'*' if r['is_discretisation_sigma'] else ' '}  "
              f"margin p10 {r['margin_p10']:8.3f}  median {r['margin_median']:9.3f}  "
              f"p90 {r['margin_p90']:10.3f}  >10nat {100*r['frac_margin_gt_10nat']:5.1f}%  "
              f"PSM {r['psm_top1']:.3f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", nargs="+", default=["lgrid"], choices=["lgrid", "taps"],
                    help="bias/load/floor/rgrid land with S1.4-S1.7")
    ap.add_argument("--l-steps", type=float, nargs="+", default=[0.1, 0.05, 0.025])
    ap.add_argument("--reference-step", type=float, default=0.0125)
    ap.add_argument("--workers", type=int, default=8,
                    help="capped below the box's 24: never take the whole machine "
                         "while another job is running")
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-margin-events", type=int, default=3000)
    ap.add_argument("--margin-seed", type=int, default=20260923)
    ap.add_argument("--chunk-f", type=int, default=2)
    ap.add_argument("--master", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if "taps" in args.checks:
        ref_dir = out_root / f"step_{str(args.reference_step).replace('.', 'p')}"
        if not (ref_dir / "mu.npy").exists():
            raise SystemExit(f"taps needs the reference table at {ref_dir}; run lgrid first")
        taps = tap_discontinuities(T.load_table(ref_dir), args.reference_step)
        print(f"[taps] {taps['n_tap_boundaries']} circuit-wide boundaries in "
              f"L in [0.5, 1.3]; max adjacent jump {taps['max_adjacent_jump_pu']:.3e} p.u.; "
              f"no-fault jump up to {taps['nofault_max_jump_pu']:.3e}")
        for b in taps["boundaries"]:
            print(f"    L {b['l_low']:.4f} -> {b['l_high']:.4f}: "
                  f"{100*b['cells_moved_frac']:5.1f}% of cells move, "
                  f"no-fault step {b['nofault_jump_pu']:.3e} p.u.")
        path = out_root / "tap_discontinuities.json"
        path.write_text(json.dumps(taps, indent=2))
        print(f"wrote {path}")
        if args.checks == ["taps"]:
            return 0

    print(f"[build] candidates {args.l_steps} + reference {args.reference_step} "
          f"at {args.workers} workers", flush=True)
    built = build_candidates(out_root, args.l_steps, args.reference_step,
                             args.workers, args.master)

    print("\n[interpolation error] full population, all (f, tau, R) cells", flush=True)
    interp = []
    for step in sorted(args.l_steps, reverse=True):
        row = interpolation_error(built[step], built[args.reference_step], step,
                                  args.reference_step)
        interp.append(row)
        print(f"  L step {step:6}  ({row['n_points']:2d} pts)  max {row['max']:.3e}  "
              f"p99 {row['p99']:.3e}  median {row['p50']:.3e}  "
              f"(ungrounded {row['max_ungrounded']:.3e} / grounded "
              f"{row['max_grounded']:.3e})", flush=True)

    picks, strata, per = _stratified_events(Path(args.data_root) / "raw_sim",
                                            args.n_margin_events, args.margin_seed)
    s, a, y = _observe(picks)
    s, a = s.to(device), a.to(device)
    print(f"\n[rho margin] {len(picks)} events, stratified by (R x grounded): "
          f"{len(strata)} strata x {per} -- seed {args.margin_seed} -- device={device}", flush=True)
    margins = []
    for step in sorted(args.l_steps, reverse=True):
        disc = next(r["p99"] for r in interp if r["l_step"] == step)
        print(f"  L step {step} ({len(grids.l_grid(step))} pts), "
              f"discretisation sigma = {disc:.3e}", flush=True)
        margins += margin_distribution(built[step], step, s, a, y, disc, args.chunk_f)

    summary = {
        "check": "lgrid", "date": time.strftime("%Y-%m-%d"),
        "candidates": args.l_steps, "reference_step": args.reference_step,
        "sampling": {
            "interpolation": "full population (all 4,125 (f,tau,R) cells x all "
                             "interpolated L x all instrumented channels)",
            "margin": {"n_events": len(picks), "scheme": "stratified by (R, grounded)",
                       "strata": strata, "per_stratum": per, "seed": args.margin_seed,
                       "ci": "median margin, 2000-sample bootstrap"},
        },
        "interpolation_error": interp,
        "margin": margins,
        "selection": "NOT automatic -- see 03_STAGE_B_operator.md S1.2",
    }
    path = out_root / "lgrid_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
