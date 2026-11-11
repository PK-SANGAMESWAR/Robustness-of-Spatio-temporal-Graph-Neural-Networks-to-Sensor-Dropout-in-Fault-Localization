"""Gate 6 analysis of a completed sealed-set pipeline run (CPU, numpy only).

    nice -n 19 uv run python scripts/run_gate6_analysis.py --dir results/stage_g/ood

Reads ``<dir>/{r5,r0n,r0}_seed<seed>_predictions.npz`` (written by ``scripts/run_stage_g_ood.py``) and writes
``<dir>/gate6_analysis.json``.  It refuses a directory without ``COMPLETED.json``.  A smoke directory (the pilot run,
``STARTED.json`` with ``"smoke": true``) is analysed for structure only: the output is named
``gate6_analysis_SMOKE_NOT_REPORTABLE.json`` and no number is printed.  Rule: ``14_PREREGISTRATION.md`` Amendments 1, 3, 9, 11.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(REPO / "results/stage_g/ood"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    from fault_localization.stage_g.gate6 import run_analysis

    d = Path(args.dir)
    if not (d / "COMPLETED.json").exists():
        print("REFUSED: no COMPLETED.json in", d, file=sys.stderr)
        return 2
    smoke = bool(json.loads((d / "STARTED.json").read_text()).get("smoke", False))
    out = run_analysis(d, seeds=tuple(args.seeds), n_boot=args.n_boot)
    out["reportable"] = not smoke
    path = d / ("gate6_analysis_SMOKE_NOT_REPORTABLE.json" if smoke else "gate6_analysis.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if smoke:
        print(f"smoke analysis written to {path} (structure only; no number is reportable)")
    else:
        g = out["gate6"]
        print(json.dumps({"written": str(path), "gate6_evaluable": g["evaluable"], "gate6_passed": g["passed"],
                          "per_sigma": {k: {"passes": v["passes"], "ci95": v["ci95"], "delta_per_seed": v["delta_per_seed"]}
                                        for k, v in g["per_sigma"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
