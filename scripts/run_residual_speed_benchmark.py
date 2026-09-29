"""S3.4 -- residual computation layer, speed check (``05_STAGE_D_infra.md``).

Benchmarks the online (``method="direct"``, float32) path of ``physics/residual.py``
against the Gate-3 criterion: **under ~5 ms per batch of 32 on the T1000**. This is the
path a training loop or the imputation baseline would call per micro-batch -- the
``method="gram"`` float64 form is reserved for the offline corpus-wide cache
(``physics/cache.py``) and is not what this check is about.

Three quantities are timed, all on the real frozen operator table
(``operator/`` -- 26 classes x 3366 hypotheses x 75 channels, the post-R-grid-fix
size; ``05_STAGE_D_infra.md``'s "2,805 hypotheses" is the stale pre-fix count):

  - r-tilde(A)   -- ``residual.rho``        (also gives hyp_star, needed for LOO/D)
  - r-tilde(D)   -- ``residual.rho_heldout``
  - r_i^LOO      -- ``residual.rho_loo``     ("multiplies by 25" per the spec)

Timing uses random inputs (wall time of these ops depends on tensor shape, not
value) with CUDA warmup + synchronize, batch size 32, chunk_f=1 (the default
used elsewhere, e.g. ``physics/cache.py``'s ``build_rho_cache``).

Usage:
    uv run python scripts/run_residual_speed_benchmark.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452
BATCH_SIZE = 32
N_WARMUP = 5
N_ITERS = 30
THRESHOLD_MS = 5.0
OUT_DIR = REPO / "results/pcstgnn/audit/s3_4_residual_speed"


def _time_call(fn, n_warmup: int, n_iters: int, device: torch.device) -> dict:
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times_ms = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std(ddof=1)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "n_iters": n_iters,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[device] {gpu_name}", flush=True)

    tbl = T.load_table(REPO / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"]).to(device)
    print(f"[table] {tuple(table.shape)} on {device}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(0)
    s = (torch.randn(BATCH_SIZE, 25, 3, generator=g) * 0.05 + 1.0).to(device)
    # a random availability pattern spanning the dropout grid (k up to 10 zeroed
    # nodes) -- shape doesn't matter for wall time, but this keeps the benchmark
    # honest about what shape of input the layer actually sees.
    a = torch.ones(BATCH_SIZE, 25, 3)
    for i in range(BATCH_SIZE):
        k = [0, 1, 2, 5, 10][i % 5]
        if k > 0:
            drop = torch.randperm(25, generator=g)[:k]
            a[i, drop, :] = 0.0
    a = a.to(device)
    d_mask = (1.0 - a).to(device)  # held-out = whatever A doesn't cover
    f_hat = torch.randint(0, 26, (BATCH_SIZE,), generator=g).to(device)

    results: dict = {
        "device": gpu_name,
        "table_shape": list(table.shape),
        "batch_size": BATCH_SIZE,
        "method": "direct",
        "chunk_f": 1,
        "sigma_eps": CANONICAL_SIGMA_EPS,
        "threshold_ms": THRESHOLD_MS,
    }

    # r-tilde(A) -- also gives hyp_star, which rho_heldout/rho_loo consume downstream.
    rho_vals_holder = {}

    def call_rho():
        rho_vals, hyp = R.rho(s, a, table, CANONICAL_SIGMA_EPS, sigma0=0.0,
                              chunk_f=1, method="direct")
        rho_vals_holder["rho"] = rho_vals
        rho_vals_holder["hyp"] = hyp

    results["rho_A"] = _time_call(call_rho, N_WARMUP, N_ITERS, device)
    print(f"[rho(A)]      mean {results['rho_A']['mean_ms']:.3f} ms "
          f"(median {results['rho_A']['median_ms']:.3f}, n={N_ITERS})", flush=True)

    hyp_star = rho_vals_holder["hyp"]

    def call_rho_heldout():
        R.rho_heldout(s, d_mask, table, hyp_star, CANONICAL_SIGMA_EPS, sigma0=0.0)

    results["rho_D"] = _time_call(call_rho_heldout, N_WARMUP, N_ITERS, device)
    print(f"[rho(D)]      mean {results['rho_D']['mean_ms']:.3f} ms "
          f"(median {results['rho_D']['median_ms']:.3f}, n={N_ITERS})", flush=True)

    def call_rho_loo():
        R.rho_loo(s, a, table, f_hat, CANONICAL_SIGMA_EPS, sigma0=0.0)

    results["rho_LOO"] = _time_call(call_rho_loo, N_WARMUP, N_ITERS, device)
    print(f"[rho_LOO]     mean {results['rho_LOO']['mean_ms']:.3f} ms "
          f"(median {results['rho_LOO']['median_ms']:.3f}, n={N_ITERS})", flush=True)

    passed = {
        "rho_A": results["rho_A"]["mean_ms"] < THRESHOLD_MS,
        "rho_D": results["rho_D"]["mean_ms"] < THRESHOLD_MS,
        "rho_LOO": results["rho_LOO"]["mean_ms"] < THRESHOLD_MS,
    }
    results["passed"] = passed
    results["gate3_speed_criterion_met"] = all(passed.values())
    results["cache_needed"] = not passed["rho_A"]  # r-tilde(A) is the criterion's own target

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "speed_benchmark.json").write_text(json.dumps(results, indent=2))

    verdict = "PASSED (no cache needed)" if results["gate3_speed_criterion_met"] \
        else "FAILED (cache required per 05_STAGE_D_infra.md's branch)"
    print(f"\n=== Gate 3 speed criterion: {verdict} ===")
    print(f"wrote {OUT_DIR / 'speed_benchmark.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
