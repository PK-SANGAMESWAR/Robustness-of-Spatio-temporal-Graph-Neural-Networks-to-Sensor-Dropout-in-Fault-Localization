"""Stage G scorer for one checkpoint on the validation or test split. Saves every prediction.

    uv run python scripts/score_stage_g.py --arm r5 --seed 42 --split val --grid mismatch \
        --table-dir operator_mismatch/tap_p1/operator --out-name r5_tap_p1

Writes ``results/stage_g/<split>/<out-name>/seed_<seed>_{scoring.json,predictions.npz}``.
Differences from ``scripts/score_s4_4_gate4_run.py`` (which the S4.4 queue has loaded and which is not
edited): predictions are saved; the grid is selectable; ``--table-dir`` swaps the signature table
(model-mismatch Phase 1); the split is selectable.  The test split is refused unless
``--approved-test-split`` is passed, because the plan must be approved first (CLAUDE.md).
This script does not know the sealed OOD set; that has its own one-run pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from fault_localization.data.dataset import WindowGraphDataset  # noqa: E402
from fault_localization.stage_g import scoring as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", help="S4.4 arm: results/s4_4/<arm>/pcstgnn/seed_<seed:05d>/")
    ap.add_argument("--run-dir", help="explicit run directory (for models outside results/s4_4)")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--approved-test-split", action="store_true")
    ap.add_argument("--grid", choices=sorted(S.GRIDS), default="full")
    ap.add_argument("--table-dir", default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--out-root", default=str(REPO / "results/stage_g"))
    ap.add_argument("--root", default="dataset_fullscale_50")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    if args.split == "test" and not args.approved_test_split:
        raise SystemExit("refused: the test split needs --approved-test-split (plan approved by the author)")
    if bool(args.arm) == bool(args.run_dir):
        raise SystemExit("give exactly one of --arm and --run-dir")
    run_dir = Path(args.run_dir) if args.run_dir else REPO / "results/s4_4" / args.arm / "pcstgnn" / f"seed_{args.seed:05d}"
    name = args.out_name or (args.arm or run_dir.parent.name)

    ds = WindowGraphDataset(args.root, args.split)
    device = torch.device(args.device)
    model, cfg = S.load_model(run_dir, ds.edge_index, device,
                              table_dir=Path(args.table_dir) if args.table_dir else None,
                              shuffle_seed=args.shuffle_seed)
    res = S.score_dataset(model, ds, S.GRIDS[args.grid], device=device, max_windows=args.max_windows,
                          progress_every=args.progress_every)
    meta = {"arm": args.arm or str(run_dir), "seed": args.seed, "split": args.split, "grid": args.grid,
            "table_dir": args.table_dir, "shuffle_seed": args.shuffle_seed, "max_windows": args.max_windows,
            "checkpoint_sha256": S.sha256_file(run_dir / "checkpoint_best.pt"), "data_root": args.root,
            "resistances": [0.1, 1.0, 10.0]}
    out_dir = Path(args.out_root) / args.split / name
    path = S.write_summary(out_dir, f"seed_{args.seed:05d}", res, meta, resistances=(0.1, 1.0, 10.0))
    print(json.dumps({"summary": str(path), "n_windows": int(len(res.y_true)),
                      "seconds": res.seconds}, indent=2))
    S.release(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
