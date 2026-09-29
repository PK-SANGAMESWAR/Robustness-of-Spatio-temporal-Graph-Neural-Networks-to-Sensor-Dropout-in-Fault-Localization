"""S3.4 -- build the offline r-tilde(A) cache (``05_STAGE_D_infra.md`` Gate 3 branch).

The online ``method="direct"`` residual (the only form accurate enough for a
differentiable training loop) measured at ~45 ms/batch-of-32 on the T1000 --
9x over the 5 ms/batch Gate-3 budget, and compute-bound (chunk_f-invariant), so
the branch's "if slow, cache" instruction applies. Writes
``cache/rtilde_{split}.npy`` for train/val/test via
``fault_localization.physics.cache.build_rtilde_cache``.

Usage:
    uv run python scripts/build_rtilde_cache.py --split test
    uv run python scripts/build_rtilde_cache.py --split train --limit-events 20  # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.physics import cache as C  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=str(REPO / "cache"))
    ap.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    ap.add_argument("--n-mask-seeds", type=int, default=3)
    ap.add_argument("--batch-events", type=int, default=6)
    ap.add_argument("--limit-events", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    tbl = T.load_table(REPO / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"])
    print(f"[load] table {tuple(table.shape)} device={device}", flush=True)

    def progress(done, total):
        print(f"  [{args.split}] {done}/{total} events", flush=True)

    t0 = time.time()
    meta = C.build_rtilde_cache(
        args.data_root, args.split, table, CANONICAL_SIGMA_EPS,
        Path(args.out_dir) / f"rtilde_{args.split}.npy",
        k_levels=tuple(args.k_levels), n_mask_seeds=args.n_mask_seeds,
        batch_events=args.batch_events, limit_events=args.limit_events,
        device=device, progress=progress,
    )
    meta["seconds"] = time.time() - t0
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
