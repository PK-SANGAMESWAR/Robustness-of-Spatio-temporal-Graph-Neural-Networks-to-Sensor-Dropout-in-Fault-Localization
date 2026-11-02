"""Sealed OOD test set: the one-run pipeline (``fault_localization/stage_g/ood_pipeline.py``).

Real run (needs the author's explicit go-ahead; opens ``data/ood_test/`` once):

    uv run python scripts/run_stage_g_ood.py --author-go-ahead "<who, when, reference>"

Pilot smoke run (never the sealed set; numbers are not reportable):

    uv run python scripts/run_stage_g_ood.py --smoke-pilot --manifest config/stage_g/smoke_models.json \
        --max-windows 96 --device cpu

Freeze the model list (writes ``config/stage_g/frozen_models.json`` once, never overwrites):

    uv run python scripts/run_stage_g_ood.py --freeze-from config/stage_g/models_to_freeze.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.stage_g import ood_pipeline as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author-go-ahead", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-pilot", action="store_true")
    ap.add_argument("--manifest", default=str(P.MANIFEST))
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freeze-from", default=None)
    args = ap.parse_args()

    if args.freeze_from:
        entries = json.loads(Path(args.freeze_from).read_text())["entries"]
        m = P.freeze_manifest(entries, P.MANIFEST)
        print(json.dumps({"frozen": str(P.MANIFEST), "n": len(m["entries"])}, indent=2))
        return 0

    if args.smoke_pilot:
        cfg = P.PipelineConfig(sealed_dir=REPO / "data/ood_pilot", hash_file=None,
                               out_dir=REPO / "results/stage_g/ood_pilot_smoke", manifest=Path(args.manifest),
                               expected_events=1100, resume=args.resume, smoke=True, device=args.device,
                               max_windows=args.max_windows, check_git=False, check_locks=False)
    else:
        cfg = P.PipelineConfig(go_ahead=args.author_go_ahead, resume=args.resume, device=args.device,
                               manifest=Path(args.manifest))
    try:
        done = P.run(cfg)
    except P.Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"completed": True, "n_models": done["n_models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
