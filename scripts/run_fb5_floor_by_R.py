"""F-B5 data: r-bar at the true hypothesis, by fault resistance (S1.6, ``PC-STGNN/03_STAGE_B_operator.md``).

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/run_fb5_floor_by_R.py

CPU only. Same events, table and r-bar definition as ``check_operator_calibration.check_floor`` (profiled, unshifted,
divided by |A|), at the rescaled sigma_eps recorded by S1.7 (``s1_7_discretization.json``). Splits the result by R.
Writes ``results/pcstgnn/audit/s1_6_floor_by_R/{floor_by_R.json,r_bar.npz}``; the older s1_4_7 files are not touched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("coc", REPO / "scripts/check_operator_calibration.py")
coc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coc)
from fault_localization.physics import grids, residual as R, table as T  # noqa: E402

OUT = REPO / "results/pcstgnn/audit/s1_6_floor_by_R"


def main() -> int:
    sigma = json.loads((REPO / "results/pcstgnn/audit/s1_7_discretization/s1_7_discretization.json").read_text())["sigma_eps"]
    tbl = T.load_table(REPO / "operator")
    flat = R.flatten_table(tbl["mu"], tbl["mu0"])
    paths = coc.load_train_events(REPO / "dataset_fullscale_50")
    s, a, f_true, tau_true, r_idx, l_idx, r_true, l_true = coc.event_ground_truth(
        paths, grids.l_grid(0.05), np.asarray(grids.R_GRID))
    n = s.shape[0]
    order = np.argsort(f_true)
    s_o, a_o, f_o = s[order], a[order], f_true[order]
    n_avail = a_o.sum(dim=(1, 2)).cpu().numpy()
    r_prof = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and f_o[j] == f_o[i]:
            j += 1
        row = flat[f_o[i]:f_o[i] + 1]
        for k0 in range(i, j, 2000):
            k1 = min(k0 + 2000, j)
            rv, _ = R.raw_residual(s_o[k0:k1], a_o[k0:k1], row, sigma_eps=sigma, chunk_f=1)
            r_prof[k0:k1] = rv[:, 0].cpu().numpy()
        i = j
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    r_bar = r_prof[inv] / np.maximum(n_avail[inv], 1)
    r_true = np.asarray(r_true, dtype=float)
    out = {"n_events": int(n), "sigma_eps": float(sigma),
           "median_all": float(np.median(r_bar)), "p10_all": float(np.percentile(r_bar, 10)), "p90_all": float(np.percentile(r_bar, 90)),
           "by_R": {}}
    for rv in sorted(set(r_true.tolist())):
        m = r_true == rv
        out["by_R"][str(rv)] = {"n": int(m.sum()), "median": float(np.median(r_bar[m])),
                                "p10": float(np.percentile(r_bar[m], 10)), "p90": float(np.percentile(r_bar[m], 90))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "floor_by_R.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savez(OUT / "r_bar.npz", r_bar=r_bar, r_true=r_true)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
