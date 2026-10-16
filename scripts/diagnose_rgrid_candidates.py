"""S1.5 R-grid remediation diagnostic -- author-directed, 2026-09-25.

``results/pcstgnn/audit/s1_4_7/s1_4_7_summary.json`` found S1.5 (the decisive load-axis
test, ``PC-STGNN/03_STAGE_B_operator.md``) FAILING: L* within one grid step = 38.4 % on
the full training split, against a > 90 % criterion. Root-caused (not a bug): the locked
15-point log-spaced R grid (``operator/grid.json``, ``fault_localization/physics/grids.R_GRID``)
contains none of the corpus's own training resistances {0.1, 1, 10} Ohm, so the joint
profiled argmin over (tau, R, L) lets a nearby-but-wrong grid R trade off against a
compensating L shift and fit better than the (off-grid) true R + true L -- corrupting L
recovery even though R* itself always lands on a sensible neighbour.

Two remediations were proposed and are compared here, deliberately NOT by editing the
locked ``operator/`` table:

  hybrid    ``sorted(unique(15-pt log grid, {0.1, 1, 10}))`` -> 18 points. Puts the
            corpus's own R values exactly on-grid, which should directly remove the R-L
            compensation for those events -- but conflicts with the design note in
            ``grids.py`` (R is deliberately kept off-grid so S1.7's argmin_R-error test
            is not vacuous) and is not the remedy the spec itself names.
  dense25   25-pt log grid, corpus R values still deliberately off-grid -- the S1.7
            branch-on-failure's own prescribed remedy (``03_STAGE_B_operator.md:244``,
            "densify to 25 log-spaced points").

Both candidate tables are built by ``scripts/build_signature_table.py --r-grid ...
--out-dir results/pcstgnn/operator/rgrid_{hybrid,dense25}`` (never the canonical,
S1.2-locked ``operator/``). ``fault_localization/physics/grids.NUM_R`` and
``residual.flatten_table``/``grids.unravel_hypothesis`` are hardcoded to the canonical
15-point grid and used throughout the live pipeline (PSM, fusion, LOO); rather than touch
them for a side-by-side comparison of grids that may never become canonical, this script
reimplements the small amount of S1.5 math it needs generically over whichever R-axis
length the loaded candidate table actually has. It does not modify, and is not called by,
any Gate-1-owned script.

    uv run python scripts/diagnose_rgrid_candidates.py \\
        --tables current=operator hybrid=results/pcstgnn/operator/rgrid_hybrid \\
                 dense25=results/pcstgnn/operator/rgrid_dense25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402

NUM_PMU_NODES = 25
NUM_PHASES = 3
NUM_CHANNELS = NUM_PMU_NODES * NUM_PHASES
NUM_TYPES = len(FAULT_TYPES)
SETTLED_STEP = 45

OUT_DIR = REPO / "results/pcstgnn/audit/s1_5_rgrid_diagnostic"


def load_train_events(data_root: Path) -> list[Path]:
    stems = set()
    with (data_root / "splits/train.txt").open() as fh:
        for line in fh:
            stems.add(line.strip().rsplit("_w", 1)[0])
    raw_dir = data_root / "raw_sim"
    return sorted(p for p in raw_dir.glob("*.npz") if p.stem in stems)


def event_ground_truth(paths: list[Path], l_grid: np.ndarray, r_grid: np.ndarray):
    """Same construction as ``check_operator_calibration.event_ground_truth``,
    parametrized on the candidate's own r_grid (nearest in log-distance)."""
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
    r_idx = np.abs(np.log(np.asarray(r_true))[:, None] - np.log(r_grid)[None, :]).argmin(axis=1)
    l_idx = np.abs(np.asarray(l_true)[:, None] - l_grid[None, :]).argmin(axis=1)
    return (torch.tensor(np.stack(s), dtype=torch.float32),
            torch.tensor(np.stack(a), dtype=torch.float32),
            np.asarray(f_true), np.asarray(tau_true), r_idx, l_idx,
            np.asarray(r_true), np.asarray(l_true))


def unravel(index: np.ndarray, n_r: int, n_l: int):
    """Flat (tau,R,L) index -> components, for THIS table's own (n_r, n_l) -- the
    generic form of ``grids.unravel_hypothesis``, which hardcodes n_r=15."""
    tau_idx, rem = np.divmod(index, n_r * n_l)
    r_idx, l_idx = np.divmod(rem, n_l)
    return tau_idx, r_idx, l_idx


def check_load(mu: np.ndarray, s, a, f_true, tau_true, r_idx, l_idx, chunk: int = 2000) -> dict:
    """S1.5, generalized over the table's own R-axis length.

    Profiles (tau, R, L) at f_true on the full instrumented set (weight = availability
    mask only, matching ``check_operator_calibration.check_load``'s ``sigma_eps=1.0`` --
    a constant per-channel scale does not change an argmin). Compares argmin to the
    stored ground truth.
    """
    n = s.shape[0]
    n_r, n_l = mu.shape[2], mu.shape[3]
    order = np.argsort(f_true)
    s_o, a_o = s[order], a[order]
    f_o, tau_o, r_o, l_o = f_true[order], tau_true[order], r_idx[order], l_idx[order]

    tau_star = np.empty(n, dtype=np.int64)
    r_star = np.empty(n, dtype=np.int64)
    l_star = np.empty(n, dtype=np.int64)

    mu_t = torch.from_numpy(mu.astype(np.float32)).to(s.device)  # [25,11,n_r,n_l,25,3]

    i = 0
    while i < n:
        f_here = f_o[i]
        j = i
        while j < n and f_o[j] == f_here:
            j += 1
        row = mu_t[f_here - 1].reshape(NUM_TYPES * n_r * n_l, NUM_CHANNELS)  # [H,75]
        for k0 in range(i, j, chunk):
            k1 = min(k0 + chunk, j)
            s_flat = s_o[k0:k1].reshape(k1 - k0, -1)
            a_flat = a_o[k0:k1].reshape(k1 - k0, -1)
            diff = s_flat[:, None, :] - row[None, :, :]                # [b,H,75]
            r = (diff * diff * a_flat[:, None, :]).sum(dim=-1)         # [b,H]
            hyp = r.argmin(dim=-1).cpu().numpy()
            ti, ri, li = unravel(hyp, n_r, n_l)
            tau_star[k0:k1] = ti
            r_star[k0:k1] = ri
            l_star[k0:k1] = li
        i = j

    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    tau_star, r_star, l_star = tau_star[inv], r_star[inv], l_star[inv]

    tau_ok = tau_star == tau_true
    r_err = np.abs(r_star - r_idx)
    l_err = np.abs(l_star - l_idx)
    return {
        "n_events": int(n),
        "tau_star_correct_frac": float(tau_ok.mean()),
        "l_star_within_1_step_frac": float((l_err <= 1).mean()),
        "l_star_exact_frac": float((l_err == 0).mean()),
        "r_star_within_1_step_frac": float((r_err <= 1).mean()),
        "r_star_exact_frac": float((r_err == 0).mean()),
        "pass": bool((l_err <= 1).mean() > 0.90 and tau_ok.mean() > 0.90),
        "pass_criterion": "|L*-L_true|<=1 grid step on >90% AND tau*=tau_true on >90%",
        "_l_err": l_err, "_r_err": r_err, "_tau_ok": tau_ok,
    }


def by_r_breakdown(result: dict, r_true: np.ndarray) -> dict:
    out = {}
    for rv in sorted(set(np.round(r_true, 4))):
        sel = np.isclose(r_true, rv, rtol=1e-3)
        if sel.sum() == 0:
            continue
        out[f"{rv:g}"] = {
            "n": int(sel.sum()),
            "l_star_within_1_step_frac": float((result["_l_err"][sel] <= 1).mean()),
            "r_star_within_1_step_frac": float((result["_r_err"][sel] <= 1).mean()),
            "tau_star_correct_frac": float(result["_tau_ok"][sel].mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True,
                     help="name=path pairs, e.g. current=operator hybrid=results/.../rgrid_hybrid")
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--max-events", type=int, default=None, help="smoke-test cap only")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    paths = load_train_events(Path(args.data_root))
    if args.max_events:
        paths = paths[:args.max_events]
    print(f"[load] {len(paths)} training events on {device}", flush=True)

    all_results = {}
    for spec in args.tables:
        name, table_path = spec.split("=", 1)
        tbl = T.load_table(table_path)
        mu = tbl["mu"]
        n_r, n_l = mu.shape[2], mu.shape[3]
        r_grid = np.asarray(tbl["r_grid"])
        l_grid = np.asarray(tbl["l_grid"])
        assert len(r_grid) == n_r and len(l_grid) == n_l, f"{name}: grid.json disagrees with mu.npy shape"

        t0 = time.perf_counter()
        s, a, f_true, tau_true, r_idx, l_idx, r_true, l_true = event_ground_truth(paths, l_grid, r_grid)
        s, a = s.to(device), a.to(device)
        result = check_load(mu, s, a, f_true, tau_true, r_idx, l_idx)
        by_r = by_r_breakdown(result, r_true)
        elapsed = time.perf_counter() - t0

        for k in ("_l_err", "_r_err", "_tau_ok"):
            result.pop(k)
        result["by_R"] = by_r
        result["n_r_grid_points"] = int(n_r)
        result["r_grid"] = r_grid.tolist()
        result["table_path"] = str(table_path)
        all_results[name] = result

        print(f"[{name}] n_r={n_r}  tau*={result['tau_star_correct_frac']:.3f}  "
              f"L* within 1 step={result['l_star_within_1_step_frac']:.3f}  "
              f"R* within 1 step={result['r_star_within_1_step_frac']:.3f}  "
              f"PASS={result['pass']}  ({elapsed:.1f}s)", flush=True)
        for rv, d in by_r.items():
            print(f"    R={rv:>6}  n={d['n']:5d}  L* w/in 1={d['l_star_within_1_step_frac']:.3f}  "
                  f"R* w/in 1={d['r_star_within_1_step_frac']:.3f}", flush=True)

    (out_dir / "rgrid_diagnostic.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n[done] wrote {out_dir / 'rgrid_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
