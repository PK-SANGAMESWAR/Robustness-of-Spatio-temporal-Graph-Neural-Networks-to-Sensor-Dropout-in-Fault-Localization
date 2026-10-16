"""S1.7 discretization sub-test -- author-directed, 2026-09-25.

``PC-STGNN/03_STAGE_B_operator.md`` S1.7: "Discretisation: rebuild the residual using only
every second R grid point ... re-run PSM (Stage C S2.4) and compare F1. Pass: discretisation
costs < 0.01 macro-F1." This was never implemented (``check_operator_calibration.py``'s own
docstring flags it as a follow-up needing the full 26-way argmin, the same cost class as the
lgrid margin sweep). It's the last of Gate 1's five criteria still open after the 2026-09-25
R-grid fix (``results/pcstgnn/operator/S1.5_RGRID_CORRECTION.md``).

Per the author's explicit instruction: the canonical ``operator/`` table (18-pt hybrid R grid)
and its calibrated sigma_eps (7.2555e-4) stay FROZEN for this test -- nothing here rebuilds or
touches them. Only a second, halved-grid table is built (in a side directory), and both are
scored identically.

Halving: every second point of the canonical 18-pt ``grids.R_GRID`` starting at index 0 -> 9
points. This happens to retain all three corpus resistances {0.1, 1, 10} exactly (their
insertion left them at even indices in the sorted union), so the comparison isolates pure grid
*density* loss from the alignment fix S1.5 required -- halving a grid that also lost R-alignment
would confound the two effects this and the S1.5 fix are separately about.

Evaluated on the same population as every other S1.x check in this pass: the full training
split (28,875 events, ``dataset_fullscale_50/splits/train.txt``) -- not a Stage C test-split
run, which this is not. Metric: macro-F1 over the 25 fault-location classes actually present in
per-event ground truth (these are all instrumented fault events; no true no-fault class 0
instances exist at this per-event granularity, matching S1.4-S1.6's sibling checks and the
25-class "localization" convention already used elsewhere in this repo, e.g.
``scripts/run_localization_metrics.py``).

``fault_localization.physics.residual.flatten_table`` hardcodes the canonical NUM_R (18) in its
shape assertion, so the halved (9-point) table can't go through it. ``raw_residual`` itself is
fully shape-agnostic (no reference to NUM_R), so this script flattens generically instead
(mirroring flatten_table's own logic) and calls ``residual.rho``/``residual.psm_predict``
unmodified.

    uv run python scripts/check_s1_7_discretization.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES  # noqa: E402
from fault_localization.physics import residual as R, table as T  # noqa: E402

NUM_PMU_NODES = 25
NUM_PHASES = 3
NUM_CHANNELS = NUM_PMU_NODES * NUM_PHASES
NUM_TYPES = len(FAULT_TYPES)
SETTLED_STEP = 45
CANONICAL_SIGMA_EPS = 0.0007255549105131452  # operator/grid.json rgrid_correction, frozen

OUT_DIR = REPO / "results/pcstgnn/audit/s1_7_discretization"


def flatten_generic(mu: np.ndarray, mu0: np.ndarray) -> torch.Tensor:
    """``residual.flatten_table``'s logic, without its hardcoded-NUM_R shape assertion."""
    n_f, n_t, n_r, n_l = mu.shape[:4]
    h = n_t * n_r * n_l
    flat = np.empty((n_f + 1, h, NUM_CHANNELS), dtype=np.float32)
    flat[1:] = mu.reshape(n_f, h, NUM_CHANNELS).astype(np.float32)
    flat[0] = np.broadcast_to(
        mu0.astype(np.float32).reshape(1, 1, n_l, NUM_CHANNELS), (n_t, n_r, n_l, NUM_CHANNELS)
    ).reshape(h, NUM_CHANNELS)
    return torch.from_numpy(flat)


def load_train_events(data_root: Path) -> list[Path]:
    stems = set()
    with (data_root / "splits/train.txt").open() as fh:
        for line in fh:
            stems.add(line.strip().rsplit("_w", 1)[0])
    raw_dir = data_root / "raw_sim"
    return sorted(p for p in raw_dir.glob("*.npz") if p.stem in stems)


def observe(paths: list[Path]):
    s, a, y = [], [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as loaded:
            rec = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        s.append(rec[:, SETTLED_STEP, :3])
        a.append((rec[:, 0, 3:] > 0.5).astype(np.float32))
        y.append(FAULT_BUSES.index(meta["fault_bus"]) + 1)
    return (torch.tensor(np.stack(s), dtype=torch.float32),
            torch.tensor(np.stack(a), dtype=torch.float32),
            np.asarray(y))


def psm_macro_f1(flat: torch.Tensor, s, a, y, sigma_eps: float, chunk_b: int = 200,
                 chunk_f: int = 2) -> dict:
    preds = np.empty(len(y), dtype=np.int64)
    for b0 in range(0, len(y), chunk_b):
        b1 = min(b0 + chunk_b, len(y))
        rho, _hyp = R.rho(s[b0:b1], a[b0:b1], flat, sigma_eps, chunk_f=chunk_f)
        preds[b0:b1] = R.psm_predict(rho).cpu().numpy()
    labels = list(range(1, len(FAULT_BUSES) + 1))
    f1 = f1_score(y, preds, labels=labels, average="macro", zero_division=0)
    top1 = float((preds == y).mean())
    return {"macro_f1_25class": float(f1), "top1_acc": top1, "n_events": int(len(y))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-dir", default=str(REPO / "operator"))
    ap.add_argument("--halved-dir", default=str(REPO / "results/pcstgnn/operator/rgrid_halved_s17"))
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--sigma-eps", type=float, default=CANONICAL_SIGMA_EPS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    paths = load_train_events(Path(args.data_root))
    print(f"[load] {len(paths)} training events on {device}", flush=True)
    s, a, y = observe(paths)
    s, a = s.to(device), a.to(device)

    results = {"sigma_eps": args.sigma_eps, "n_events": len(paths)}
    for name, table_dir in (("full_18pt", args.canonical_dir), ("halved_9pt", args.halved_dir)):
        tbl = T.load_table(table_dir)
        n_r = tbl["mu"].shape[2]
        flat = flatten_generic(tbl["mu"], tbl["mu0"])
        t0 = time.perf_counter()
        res = psm_macro_f1(flat, s, a, y, args.sigma_eps)
        res["n_r_grid_points"] = int(n_r)
        res["table_dir"] = str(table_dir)
        elapsed = time.perf_counter() - t0
        results[name] = res
        print(f"[{name}] n_r={n_r}  macro_f1(25-class)={res['macro_f1_25class']:.4f}  "
              f"top1={res['top1_acc']:.4f}  ({elapsed:.1f}s)", flush=True)

    delta = results["full_18pt"]["macro_f1_25class"] - results["halved_9pt"]["macro_f1_25class"]
    results["delta_f1"] = float(delta)
    results["pass"] = bool(abs(delta) < 0.01)
    results["pass_criterion"] = "|F1_full - F1_halved| < 0.01 macro-F1 (03_STAGE_B_operator.md S1.7)"
    print(f"\n[S1.7] delta F1 = {delta:+.4f}  PASS={results['pass']}", flush=True)

    path = out_dir / "s1_7_discretization.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
