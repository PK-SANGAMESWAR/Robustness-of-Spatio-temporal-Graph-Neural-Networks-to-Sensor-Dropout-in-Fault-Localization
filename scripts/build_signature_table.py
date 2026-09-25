"""Stage B S1.3 -- build the physics operator's signature table (``PC-STGNN/03_STAGE_B_operator.md``).

    uv run python scripts/build_signature_table.py --workers 20 --l-step 0.05

Thin CLI over ``fault_localization.physics.table.build_table`` -- the actual solve loop, the
per-solve OpenDSS recompile (``mode="daily"``, ``reuse_limit=1``; PC-STGNN-PLAN.md §2.1 /
commit ``ee835a0``), and the ``grid.json`` metadata it writes are all there; see that module's
docstring for why recompiling per solve is not negotiable.

Default output is the canonical ``operator/`` at the repo root (`01_SPEC.md` §2, the plan's
decision 5), which S1.2 has already selected 0.05 for -- see `operator/grid.json`'s ``decision``
block and `results/pcstgnn/operator/S1.2_DECISION.md`.  Refuses to overwrite a complete table at
the default location without ``--force``, because ``build_table`` writes a fresh ``grid.json``
that would silently drop the S1.2 decision provenance recorded there.

``--master`` builds against a perturbed feeder model instead (S6.2 model-mismatch study) --
always pass a non-default ``--out-dir`` for those, never the canonical ``operator/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.physics import grids, table as T  # noqa: E402

DEFAULT_OUT = REPO / "operator"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--l-step", type=float, default=0.05,
                    help="S1.2 selected 0.05 -- see operator/grid.json's decision block")
    ap.add_argument("--r-grid", type=float, nargs="+", default=None,
                    help="defaults to grids.R_GRID (the real 15-point log-spaced grid)")
    ap.add_argument("--mode", default="daily", choices=["daily", "snapshot", "reuse"],
                    help="daily is the corpus-matching path (S0.4); reuse is a cross-check only")
    ap.add_argument("--reuse-limit", type=int, default=1,
                    help="1 = recompile per solve, matches the corpus generator; do not raise "
                         "this on the canonical build (ee835a0)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--master", default=None,
                    help="perturbed feeder .dss for S6.2 model-mismatch tables -- use a "
                         "non-default --out-dir with this, never the canonical operator/")
    ap.add_argument("--sigma-eps", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test truncation only")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing complete table at --out-dir, including any "
                         "S1.2 decision block recorded in its grid.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    grid_path = out_dir / "grid.json"
    if grid_path.exists() and not args.force:
        existing = json.loads(grid_path.read_text())
        if existing.get("complete") and args.limit is None:
            print(f"[skip] {out_dir} already holds a complete table "
                  f"(l_step={existing.get('l_step')}, n_solves={existing.get('n_solves')}). "
                  f"Pass --force to rebuild and lose its 'decision' provenance if any.")
            return 0

    t0 = time.perf_counter()
    meta = T.build_table(
        out_dir, workers=args.workers, mode=args.mode, l_step=args.l_step,
        r_grid=args.r_grid, master=args.master, reuse_limit=args.reuse_limit,
        sigma_eps=args.sigma_eps, limit=args.limit,
    )
    elapsed = time.perf_counter() - t0
    print(f"built {out_dir}: l_step={meta['l_step']} "
          f"({len(grids.l_grid(meta['l_step']))} pts), n_solves={meta['n_solves']}, "
          f"non_converged={meta['n_non_converged']}, nan={meta['n_nan']}, "
          f"complete={meta['complete']}, {elapsed:.0f}s wall")
    if meta["n_non_converged"]:
        print(f"  WARNING: {meta['n_non_converged']} solves did not converge -- "
              f"S1.3 requires these to be logged and re-solved, not silently accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
