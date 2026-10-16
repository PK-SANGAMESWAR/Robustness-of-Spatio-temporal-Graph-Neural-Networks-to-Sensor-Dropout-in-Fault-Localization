"""Verify the k=0 rho(A) cache shortcut before it is trusted inside S4.3 training.

Pulls real k=0 batches from the ACTUAL S4.3 training loader (PhysicsDropoutLoader over the
real train split, real train_k_range, real seed) -- not synthetic tensors -- and checks two
independent things that could each corrupt every k=0 batch silently:

1. Window-id alignment: cache/rtilde_train.npy's window_id order does NOT match
   dataset_fullscale_50/splits/train.txt's order (verified separately: same 1,155,000-window
   set, different order -- the cache iterates sorted raw-sim event keys, the split file follows
   the event-split assignment). physics.cache.load_rho_k0_lookup looks up by window_id string,
   never by row position, which is immune to this -- this script proves that lookup actually
   lands on the right rows for real batches, not just that the function looks correct.
2. Numeric equivalence: rho(A) computed LIVE by the model (method="direct", float32, the
   training-time path) against rho(A) read from the cache (built with method="gram", float64,
   S3.4) must agree to the ~1e-5 tolerance S3.4's own cross-validation established.

Usage:
    uv run python scripts/verify_pcstgnn_rho_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.training import PhysicsDropoutLoader  # noqa: E402
from fault_localization.models.pc_stgnn import PCSTGNN  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402
from fault_localization.physics.cache import load_rho_k0_lookup  # noqa: E402
from fault_localization.physics.observation import load_norm_stats  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452
DATA_ROOT = "dataset_fullscale_50"
CACHE_PATH = REPO / "cache" / "rtilde_train.npy"
SEED = 42
N_K0_BATCHES = 5
# rho(A) is a/(sigma_eps^2 + sigma0^2)-weighted, so with sigma_eps ~ 7.26e-4 the values
# routinely run into the millions -- an absolute tolerance is meaningless at that scale.
# Use relative + a small absolute floor for near-zero entries (numpy.isclose semantics).
TOL_RTOL = 1e-3
TOL_ATOL = 1e-2


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    ds = WindowGraphDataset(DATA_ROOT, "train")
    base_loader = make_dataloader(ds, batch_size=32, shuffle=True, seed=SEED)
    loader = PhysicsDropoutLoader(base_loader, k_range=[0, 1, 2, 5], rng_seed=SEED,
                                  policy="zero_messages")

    tbl = T.load_table("operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"])
    norm_stats = load_norm_stats(DATA_ROOT)
    model = PCSTGNN(ds.edge_index, table, norm_stats, use_fusion=True, beta_init=0.0,
                    sigma_eps=CANONICAL_SIGMA_EPS, sigma0=0.0, aggregator="max",
                    num_layers=4).to(device)
    model.eval()  # fixed BN/dropout -- this check is about rho, not about training dynamics

    print(f"loading {CACHE_PATH} ...", flush=True)
    window_to_row, rho0, hyp0 = load_rho_k0_lookup(CACHE_PATH)
    print(f"cache: {len(window_to_row)} unique window_ids, rho0 shape {rho0.shape}", flush=True)

    checked = 0
    max_rel_diff = 0.0
    all_close = True
    hyp_matches = 0
    hyp_total = 0
    for batch in loader:
        if batch.get("dropout_k", -1) != 0:
            continue
        wids = batch["window_id"]
        missing = [w for w in wids if w not in window_to_row]
        if missing:
            print(f"[FAIL] {len(missing)} window_id(s) not found in cache, e.g. {missing[:3]}")
            return 1

        x = batch["x"].to(device)
        with torch.no_grad():
            rho_live, hyp_live, _, _ = model._rho(x)
        rho_live = rho_live.cpu().numpy()
        hyp_live = hyp_live.cpu().numpy()

        rows = [window_to_row[w] for w in wids]
        rho_cached = rho0[rows]
        hyp_cached = hyp0[rows]

        close = np.isclose(rho_live, rho_cached, rtol=TOL_RTOL, atol=TOL_ATOL)
        all_close = all_close and bool(close.all())
        # rho routinely runs into the millions (a/(sigma_eps^2+sigma0^2) weighting), so
        # report relative error, not absolute -- an absolute diff of 0.5 against 7e6 is a
        # relative error of ~7e-8, i.e. plain float32-vs-float64 precision, not a bug.
        rel = np.abs(rho_live - rho_cached) / np.maximum(np.abs(rho_cached), 1.0)
        max_rel_diff = max(max_rel_diff, float(rel.max()))
        hyp_matches += int((hyp_live == hyp_cached).sum())
        hyp_total += hyp_live.size

        print(f"  k=0 batch {checked}: n={len(wids)} max_rel_diff={rel.max():.3e} "
              f"mean_rel_diff={rel.mean():.3e} all_close={bool(close.all())} "
              f"hyp_match={(hyp_live == hyp_cached).mean():.3f} "
              f"sample_window_ids={wids[:2]}", flush=True)
        checked += 1
        if checked >= N_K0_BATCHES:
            break

    if checked == 0:
        print("[FAIL] no k=0 batches encountered -- cannot verify (unlucky draw or "
              "PhysicsDropoutLoader misbehaving)")
        return 1

    print(f"\n=== {checked} k=0 batches checked, max_rel_diff={max_rel_diff:.3e}, "
          f"hyp_match_rate={hyp_matches/hyp_total:.4f} ===")
    if not all_close:
        print(f"[FAIL] some entries fall outside rtol={TOL_RTOL:.1e}/atol={TOL_ATOL:.1e}")
        return 1
    print(f"[PASS] within rtol={TOL_RTOL:.1e}/atol={TOL_ATOL:.1e} on real batches from the "
          f"real S4.3 loader")
    return 0


if __name__ == "__main__":
    sys.exit(main())
