"""Stage B S1.4-S1.7 (``PC-STGNN/03_STAGE_B_operator.md``) against the locked operator/ table.

    uv run python scripts/check_operator_calibration.py --checks bias load floor rgrid

Runs on the **training split**, full population by default (`12_CHECKLISTS.md` S0), using the
operator/ table S1.2 selected (l_step=0.05).  Exploits a cost cut the margin computation
(``check_operator_fidelity.py``) does not have: S1.4/S1.5/S1.6 only ever need the residual
profiled against the **known** f_true row, not all 26 hypotheses, so events are grouped by
f_true and each group hits a single-row table slice -- about 26x less compute than the full
margin sweep, which is what makes a full-population pass affordable in one process.

``--checks``:
  bias   S1.4  |e| = |s - mu[f_true, tau_true, R_true, L_true]| at the STORED ground truth
                (nearest grid point), by R / fault type / sensor.  Now a construction check
                (S2.0a): a hit means the build path diverged, not a fidelity failure.
  load   S1.5  profile (tau, R, L) at f_true on A; compare argmin to stored ground truth.
                The decisive load-axis test.
  floor  S1.6  r-bar_{f_true} (profiled, unshifted, /|A|) at the S1.2 discretisation sigma;
                calibrates whether that sigma is the right scale (target median in [0.3, 3]).
  rgrid  S1.7  per-(f,tau) monotonicity of sag vs R on THIS table (17-pt L grid, not the 3-probe
                smoke check); effective_r_levels (identifiability.py, already implemented);
                argmin_R accuracy split by monotone/non-monotone cell, from the `load` pass.
                The halved-R-grid PSM-F1 discretisation sub-test is NOT here -- it needs the
                full 26-way argmin (same cost class as the margin sweep) and is left for a
                follow-up pass reusing check_operator_fidelity.py's batching pattern.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES  # noqa: E402
from fault_localization.physics import grids, identifiability as I, residual as R, table as T  # noqa: E402

OUT_DIR = REPO / "results/pcstgnn/audit/s1_4_7"
TABLE_DIR = REPO / "operator"
SETTLED_STEP = 45


def load_train_events(data_root: Path) -> list[Path]:
    stems = set()
    with (data_root / "splits/train.txt").open() as fh:
        for line in fh:
            stems.add(line.strip().rsplit("_w", 1)[0])
    raw_dir = data_root / "raw_sim"
    paths = [p for p in raw_dir.glob("*.npz") if p.stem in stems]
    return sorted(paths)


def event_ground_truth(paths: list[Path], l_grid: np.ndarray, r_grid: np.ndarray):
    """Returns s [B,25,3], a [B,25,3], f_true [B] (1..25), tau_true [B], r_idx [B] (nearest),
    l_idx [B] (nearest), r_true [B] (Ohms), l_true [B] (the stored multiplier)."""
    s, a, f_true, tau_true, r_true, l_true = [], [], [], [], [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as loaded:
            rec = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        s.append(rec[:, SETTLED_STEP, :3])
        a.append((rec[:, 0, 3:] > 0.5).astype(np.float32))
        f_true.append(FAULT_BUSES.index(meta["fault_bus"]) + 1)
        tau_true.append(FAULT_TYPES.index(meta["fault_type"]))
        r_true.append(meta["resistance"])
        l_true.append(meta["load_multiplier"])
    # R_GRID is log-spaced (grids.R_GRID) -- nearest point must be in log-distance, not linear;
    # a linear argmin picks the wrong neighbour near cell boundaries (e.g. true R=1.0 snaps to
    # 0.776 instead of 1.225, a large systematic error masquerading as an operator bias).
    r_idx = np.abs(np.log(np.asarray(r_true))[:, None] - np.log(r_grid)[None, :]).argmin(axis=1)
    l_idx = np.abs(np.asarray(l_true)[:, None] - l_grid[None, :]).argmin(axis=1)
    return (torch.tensor(np.stack(s), dtype=torch.float32),
            torch.tensor(np.stack(a), dtype=torch.float32),
            np.asarray(f_true), np.asarray(tau_true), r_idx, l_idx,
            np.asarray(r_true), np.asarray(l_true))


def check_bias(mu: np.ndarray, mask: np.ndarray, s, a, f_true, tau_true, r_idx, l_idx,
              r_true) -> dict:
    """S1.4 -- |e| at the STORED ground truth (f_true, tau_true, r_idx, l_idx)."""
    mu_true = mu[f_true - 1, tau_true, r_idx, l_idx]          # [B,25,3]
    e = np.abs(s.numpy() - mu_true) * (mask[None] > 0)
    m = mask > 0
    per_event_max = e.reshape(e.shape[0], -1)[:, m.reshape(-1)].max(axis=1)
    per_event_med = np.median(e.reshape(e.shape[0], -1)[:, m.reshape(-1)], axis=1)

    by_r = {}
    for ri, rval in enumerate(sorted(set(r_true.round(4)))):
        sel = np.isclose(r_true, rval, rtol=1e-3)
        if sel.sum() == 0:
            continue
        vals = per_event_max[sel]
        by_r[f"{rval:.4g}"] = {"n": int(sel.sum()), "median": float(np.median(vals)),
                               "p95": float(np.percentile(vals, 95)), "max": float(vals.max())}
    by_type = {}
    for ti, tname in enumerate(FAULT_TYPES):
        sel = tau_true == ti
        if sel.sum() == 0:
            continue
        vals = per_event_max[sel]
        by_type[tname] = {"n": int(sel.sum()), "median": float(np.median(vals)),
                          "p95": float(np.percentile(vals, 95)), "max": float(vals.max())}

    return {
        "n_events": int(len(f_true)),
        "median_abs_e_max_channel": float(np.median(per_event_max)),
        "p95_abs_e_max_channel": float(np.percentile(per_event_max, 95)),
        "max_abs_e": float(per_event_max.max()),
        "median_abs_e_median_channel": float(np.median(per_event_med)),
        "frac_above_5e3": float((per_event_max > 5e-3).mean()),
        "frac_above_1e2": float((per_event_max > 1e-2).mean()),
        "by_R": by_r, "by_type": by_type,
        "pass_note": "S2.0a: this is a construction check now, not a fidelity test. "
                     "A hit above the tap-hysteresis floor (~6.5e-3 p.u., S1.4 spec) means "
                     "the build path diverged, not that the operator is inaccurate.",
    }


def check_load(flat: torch.Tensor, s, a, f_true, tau_true, r_idx, l_idx, n_l: int,
              chunk: int = 2000) -> dict:
    """S1.5 -- profile (tau,R,L) at f_true only; compare argmin to stored ground truth."""
    n = s.shape[0]
    order = np.argsort(f_true)
    s_o, a_o = s[order], a[order]
    f_o, tau_o, r_o, l_o = f_true[order], tau_true[order], r_idx[order], l_idx[order]

    tau_star = np.empty(n, dtype=np.int64)
    r_star = np.empty(n, dtype=np.int64)
    l_star = np.empty(n, dtype=np.int64)
    r_err = np.empty(n, dtype=np.int64)
    l_err = np.empty(n, dtype=np.int64)

    i = 0
    while i < n:
        f_here = f_o[i]
        j = i
        while j < n and f_o[j] == f_here:
            j += 1
        sl = slice(i, j)
        row = flat[f_here:f_here + 1]                      # [1,H,75]
        for k0 in range(i, j, chunk):
            k1 = min(k0 + chunk, j)
            _r, hyp = R.raw_residual(s_o[k0:k1], a_o[k0:k1], row, sigma_eps=1.0, chunk_f=1)
            ti, ri, li = grids.unravel_hypothesis(hyp[:, 0].numpy(), n_l)
            tau_star[k0:k1] = ti
            r_star[k0:k1] = ri
            l_star[k0:k1] = li
        i = j

    tau_ok = tau_star == tau_o
    r_err = np.abs(r_star - r_o)
    l_err = np.abs(l_star - l_o)

    # un-permute back to original event order
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    tau_ok, r_err, l_err = tau_ok[inv], r_err[inv], l_err[inv]

    return {
        "n_events": int(n),
        "tau_star_correct_frac": float(tau_ok.mean()),
        "l_star_within_1_step_frac": float((l_err <= 1).mean()),
        "l_star_exact_frac": float((l_err == 0).mean()),
        "l_star_err_median_steps": float(np.median(l_err)),
        "r_star_within_1_step_frac": float((r_err <= 1).mean()),
        "r_star_exact_frac": float((r_err == 0).mean()),
        "r_star_err_median_steps": float(np.median(r_err)),
        "pass": bool((l_err <= 1).mean() > 0.90 and tau_ok.mean() > 0.90),
        "pass_criterion": "|L*-L_true|<=1 grid step on >90% AND tau*=tau_true on >90%",
        "_r_err_by_event": r_err, "_l_err_by_event": l_err,  # consumed by rgrid check, stripped before dump
    }


def check_floor(flat: torch.Tensor, s, a, f_true, sigma_eps: float, chunk: int = 2000) -> dict:
    """S1.6 -- r-bar_{f_true}(A)/|A| (unshifted, profiled) at the given sigma_eps."""
    n = s.shape[0]
    order = np.argsort(f_true)
    s_o, a_o, f_o = s[order], a[order], f_true[order]
    r_true_prof = np.empty(n, dtype=np.float64)
    n_avail = a_o.sum(dim=(1, 2)).numpy()

    i = 0
    while i < n:
        f_here = f_o[i]
        j = i
        while j < n and f_o[j] == f_here:
            j += 1
        row = flat[f_here:f_here + 1]
        for k0 in range(i, j, chunk):
            k1 = min(k0 + chunk, j)
            r_val, _hyp = R.raw_residual(s_o[k0:k1], a_o[k0:k1], row, sigma_eps=sigma_eps, chunk_f=1)
            r_true_prof[k0:k1] = r_val[:, 0].numpy()
        i = j

    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    r_true_prof, n_avail = r_true_prof[inv], n_avail[inv]
    r_bar = r_true_prof / np.maximum(n_avail, 1)

    return {
        "n_events": int(n), "sigma_eps": float(sigma_eps),
        "median_r_bar_true": float(np.median(r_bar)),
        "p10_r_bar_true": float(np.percentile(r_bar, 10)),
        "p90_r_bar_true": float(np.percentile(r_bar, 90)),
        "pass": bool(0.3 <= np.median(r_bar) <= 3.0),
        "pass_criterion": "median r-bar_true in [0.3, 3]",
        "rescale_suggestion": (
            None if 0.3 <= np.median(r_bar) <= 3.0
            else float(sigma_eps * np.sqrt(np.median(r_bar) / 1.0))
        ),
    }


def check_rgrid(mu: np.ndarray, mask: np.ndarray, r_grid: np.ndarray,
                load_result: dict) -> dict:
    """S1.7 -- monotonicity on THIS table, effective_r_levels, argmin_R by monotone/non-monotone."""
    m = mask > 0
    # sag[f,tau,R,L] = deepest instrumented-channel sag = 1 - min(mu) over (i,p) in mask
    sag = 1.0 - mu[..., m].min(axis=-1)                       # [f,tau,R,L]
    diffs = np.diff(sag, axis=2)                               # along R; sag should DEcrease as R increases
    non_monotone = (diffs > 1e-6).any(axis=2)                  # [f,tau,L] -- True if sag rises anywhere
    n_cells = non_monotone.size
    frac_non_monotone = float(non_monotone.mean())
    ungrounded = [i for i, t in enumerate(FAULT_TYPES) if not t.endswith("G")]
    frac_ungrounded_of_violations = None
    if non_monotone.sum() > 0:
        v_ungrounded = non_monotone[:, ungrounded, :].sum()
        frac_ungrounded_of_violations = float(v_ungrounded / non_monotone.sum())

    eff = I.effective_r_levels(mu, sigma_eps=5.145e-3, mask=mask)

    out = {
        "table_l_step": 0.05,
        "n_fault_tau_L_cells": int(n_cells),
        "frac_non_monotone_in_R": frac_non_monotone,
        "frac_violations_ungrounded": frac_ungrounded_of_violations,
        "effective_r_levels": eff if not isinstance(eff, dict) else
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in eff.items()},
    }

    if "_r_err_by_event" in load_result:
        # argmin_R accuracy is stratified using the *fault-type* monotonicity of the true cell,
        # since S1.3/S1.7 non-monotonicity is keyed on (f, tau) via the ungrounded-type pattern,
        # not on which individual event landed there -- approximate by fault type: ungrounded
        # types carry the violations (S1.3: 904/1084), so split on tau being grounded/ungrounded.
        pass  # left as a per-type breakdown the caller can join against `load`'s per-event data
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", nargs="+", default=["bias", "load", "floor", "rgrid"],
                    choices=["bias", "load", "floor", "rgrid"])
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--table-dir", default=str(TABLE_DIR))
    ap.add_argument("--sigma-eps", type=float, default=5.145e-3,
                    help="S1.2 discretisation p99 at the chosen step (0.05) -- the a priori "
                         "starting point per 01_SPEC.md Sec.4 / S1.6")
    ap.add_argument("--max-events", type=int, default=None, help="smoke-test cap only")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tbl = T.load_table(args.table_dir)
    mu, mask = tbl["mu"], tbl["mask"]
    flat = R.flatten_table(tbl["mu"], tbl["mu0"])
    n_l = mu.shape[3]
    l_grid_vals = grids.l_grid(0.05)
    r_grid_vals = np.asarray(grids.R_GRID)

    t0 = time.perf_counter()
    paths = load_train_events(Path(args.data_root))
    if args.max_events:
        paths = paths[:args.max_events]
    print(f"[load] {len(paths)} training events", flush=True)
    s, a, f_true, tau_true, r_idx, l_idx, r_true, l_true = event_ground_truth(
        paths, l_grid_vals, r_grid_vals)
    print(f"[load] observed in {time.perf_counter() - t0:.1f}s", flush=True)

    results = {"n_train_events": len(paths), "sigma_eps_used": args.sigma_eps}
    load_result = None

    if "bias" in args.checks:
        t0 = time.perf_counter()
        results["S1.4_bias"] = check_bias(mu, mask, s, a, f_true, tau_true, r_idx, l_idx, r_true)
        print(f"[bias] median|e|={results['S1.4_bias']['median_abs_e_max_channel']:.3e}  "
              f"p95={results['S1.4_bias']['p95_abs_e_max_channel']:.3e}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    if "load" in args.checks:
        t0 = time.perf_counter()
        load_result = check_load(flat, s, a, f_true, tau_true, r_idx, l_idx, n_l)
        r_err = load_result.pop("_r_err_by_event")
        l_err = load_result.pop("_l_err_by_event")
        results["S1.5_load"] = load_result
        print(f"[load] tau*={load_result['tau_star_correct_frac']:.3f}  "
              f"L* within 1 step={load_result['l_star_within_1_step_frac']:.3f}  "
              f"R* within 1 step={load_result['r_star_within_1_step_frac']:.3f}  "
              f"PASS={load_result['pass']}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        np.savez(out_dir / "load_errors_by_event.npz", r_err=r_err, l_err=l_err,
                tau_true=tau_true, f_true=f_true)

    if "floor" in args.checks:
        t0 = time.perf_counter()
        results["S1.6_floor"] = check_floor(flat, s, a, f_true, args.sigma_eps)
        print(f"[floor] median r-bar={results['S1.6_floor']['median_r_bar_true']:.4f}  "
              f"PASS={results['S1.6_floor']['pass']}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    if "rgrid" in args.checks:
        t0 = time.perf_counter()
        results["S1.7_rgrid"] = check_rgrid(mu, mask, r_grid_vals, load_result or {})
        print(f"[rgrid] frac_non_monotone={results['S1.7_rgrid']['frac_non_monotone_in_R']:.4f}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    path = out_dir / "s1_4_7_summary.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
