"""S2.1 -- identifiability matrix (``04_STAGE_C_gonogo.md`` S2.1), on the frozen corrected
operator table (canonical ``operator/``, sigma_eps=7.2555e-4 -- neither touched here).

For every location pair (f1, f2), including f=0 (no-fault), compute the separability
``D(f1,f2;A) = min over (tau,R,L) pairs of sum_{(i,p) in A} (mu_f1-mu_f2)^2 / (2*sigma_eps^2)``
(``fault_localization/physics/identifiability.py``, already implemented -- this script is the
driver + figures, not new physics).

Two views, both requested by the spec:
  - GLOBAL: D profiled over the full (tau, R, L) hypothesis space (the metric's own
    definition) -- averaged over random k-sensor-dropout subsets at k in {0,2,5,10}. -> F-C1
    heatmaps + indistinguishable-pair counts + worst-case-per-location.
  - PER-R: D profiled over (tau, L) only, at each corpus R in {0.1, 1, 10} held fixed (a
    sub-table selected via grids.unravel_hypothesis's C-order (tau,R,L) flattening) -- same k
    sweep. -> F-C2 fraction-indistinguishable vs k, per R.

``availability`` A is the real instrumented mask (operator/mask.npy, 55/75 channels), not "every
channel" -- k-dropout subsets are then drawn from *that* base, matching dropout/masking.py.

Usage:
    uv run python scripts/run_identifiability_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.physics import identifiability as I  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402
from fault_localization.physics.grids import NUM_R, NUM_TYPES, R_GRID  # noqa: E402
from fault_localization.physics.residual import flatten_table  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452
K_LEVELS = [0, 2, 5, 10]
CORPUS_R = [0.1, 1.0, 10.0]
N_SUBSETS = 8
OUT_DIR = REPO / "results/pcstgnn/audit/s2_1_identifiability"


def per_r_subtable(table: np.ndarray, n_l: int, r_idx: int) -> np.ndarray:
    """[26, n_tau*NUM_R*n_l, 75] -> [26, n_tau*n_l, 75] at fixed R index (C order tau,R,L)."""
    n_f, h, c = table.shape
    assert h == NUM_TYPES * NUM_R * n_l
    resh = table.reshape(n_f, NUM_TYPES, NUM_R, n_l, c)
    sub = resh[:, :, r_idx, :, :]
    return sub.reshape(n_f, NUM_TYPES * n_l, c)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tbl = T.load_table(REPO / "operator")
    mu, mu0, mask = tbl["mu"], tbl["mu0"], tbl["mask"]
    n_l = mu.shape[3]
    flat = flatten_table(mu, mu0).numpy()  # [26, H, 75]
    print(f"[load] table {flat.shape}, mask sum={mask.sum():.0f}/{mask.size} ({time.time()-t0:.1f}s)")

    summary = {"sigma_eps": CANONICAL_SIGMA_EPS, "k_levels": K_LEVELS, "n_subsets": N_SUBSETS,
               "global": {}, "per_r": {}}

    # ---- GLOBAL: profiled over full (tau,R,L) ----
    for k in K_LEVELS:
        t1 = time.time()
        cache_path = OUT_DIR / f"D_global_k{k}.npy"
        if cache_path.exists():
            d = np.load(cache_path)
            print(f"[global k={k}] cached, skipping compute ({time.time()-t1:.1f}s)", flush=True)
        else:
            d = I.separability_over_subsets(flat, CANONICAL_SIGMA_EPS, sigma0=0.0, base_mask=mask,
                                             k=k, n_subsets=N_SUBSETS, seed=0)
            np.save(cache_path, d)
        indist = I.indistinguishable_fraction(d, threshold=1.0)
        worst = I.worst_case_per_location(d)
        summary["global"][k] = {
            "indistinguishable_fraction": indist,
            "n_indistinguishable_pairs": int(round(indist * (26 * 25 // 2))),
            "n_pairs_total": 26 * 25 // 2,
            "worst_case_min_D_by_location": worst.tolist(),
            "worst_case_location_argmin": int(np.argmin(worst)),
            "worst_case_min_D": float(worst.min()),
            "median_D_offdiag": float(np.median(d[np.triu_indices(26, k=1)])),
        }
        print(f"[global k={k}] indist_frac={indist:.4f} worst_min={worst.min():.3f} "
              f"({time.time()-t1:.1f}s)", flush=True)

    # ---- PER-R: profiled over (tau,L) only, R held at each corpus value ----
    r_grid = np.asarray(R_GRID)
    for r_val in CORPUS_R:
        r_idx = int(np.argmin(np.abs(r_grid - r_val)))
        assert abs(r_grid[r_idx] - r_val) < 1e-9, f"R={r_val} not exactly on-grid at idx {r_idx}"
        sub = per_r_subtable(flat, n_l, r_idx)
        summary["per_r"][r_val] = {}
        for k in K_LEVELS:
            t1 = time.time()
            cache_path = OUT_DIR / f"D_R{r_val}_k{k}.npy"
            if cache_path.exists():
                d = np.load(cache_path)
                print(f"[R={r_val} k={k}] cached, skipping compute ({time.time()-t1:.1f}s)", flush=True)
            else:
                d = I.separability_over_subsets(sub, CANONICAL_SIGMA_EPS, sigma0=0.0, base_mask=mask,
                                                 k=k, n_subsets=N_SUBSETS, seed=0)
                np.save(cache_path, d)
            indist = I.indistinguishable_fraction(d, threshold=1.0)
            worst = I.worst_case_per_location(d)
            summary["per_r"][r_val][k] = {
                "indistinguishable_fraction": indist,
                "worst_case_min_D": float(worst.min()),
            }
            print(f"[R={r_val} k={k}] indist_frac={indist:.4f} worst_min={worst.min():.3f} "
                  f"({time.time()-t1:.1f}s)", flush=True)

    summary["wall_seconds"] = time.time() - t0
    (OUT_DIR / "identifiability_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'identifiability_summary.json'} ({summary['wall_seconds']:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
