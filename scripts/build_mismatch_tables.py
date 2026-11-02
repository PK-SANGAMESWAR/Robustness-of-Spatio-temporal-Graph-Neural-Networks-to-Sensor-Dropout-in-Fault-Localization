"""S6.2 model-mismatch study, Phase 1: build the three perturbed signature tables (CPU only)
and measure how large each mismatch is.

    nice -n 19 uv run python scripts/build_mismatch_tables.py --workers 4

Writes ``operator_mismatch/<name>/{feeder,operator}`` and
``results/stage_g/mismatch_tables/precheck.json``.  Nothing under ``operator/`` is touched.
The pre-check (``08_STAGE_G_eval.md`` amendment, 2026-11-02): rho at the **true class** on 320
validation fault windows (fixed sample, seed 7), k = 0, for the nominal table and each
perturbed table; the median is the reported number.  No GPU, no trained model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import yaml  # noqa: E402

from fault_localization.physics import mismatch as M  # noqa: E402

OUT_ROOT = REPO / "operator_mismatch"
REPORT = REPO / "results/stage_g/mismatch_tables/precheck.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--names", nargs="+", default=list(M.MISMATCH_LEVELS))
    ap.add_argument("--root", default="dataset_fullscale_50")
    ap.add_argument("--n-windows", type=int, default=320)
    ap.add_argument("--r5-config", default="results/s4_4/r5/pcstgnn/seed_00042/config.yaml")
    args = ap.parse_args()

    sigma_eps = float(yaml.safe_load((REPO / args.r5_config).read_text())["model"]["sigma_eps"])
    report: dict = {"sigma_eps": sigma_eps, "n_windows": args.n_windows, "tables": {}}
    nominal = REPO / "operator"

    for name in args.names:
        op_dir = OUT_ROOT / name / "operator"
        if (op_dir / "grid.json").exists() and json.loads((op_dir / "grid.json").read_text()).get("complete"):
            print(f"[{name}] table exists, skipping build", flush=True)
            spec = json.loads((OUT_ROOT / name / "perturbation.json").read_text())
        else:
            t0 = time.perf_counter()
            spec = M.build_mismatch_table(name, OUT_ROOT, workers=args.workers)
            print(f"[{name}] built in {time.perf_counter() - t0:.0f} s: {spec['table']}", flush=True)
        report["tables"][name] = {"spec": spec, "shift_vs_nominal": M.table_shift(nominal, op_dir)}
        if spec["kind"] == "tap":
            report["tables"][name]["tap_positions"] = M.tap_positions(spec["master"])

    report["tap_positions_nominal"] = M.tap_positions(M.FEEDER_DIR / M.MASTER_NAME)

    x, y, stats, wids = M.load_precheck_windows(args.root, "val", args.n_windows)
    report["precheck_windows"] = {"first": wids[:3], "n": len(wids)}
    report["true_class_residual"] = {"nominal": M.true_class_residual(nominal, x, y, stats, sigma_eps)}
    for name in args.names:
        report["true_class_residual"][name] = M.true_class_residual(
            OUT_ROOT / name / "operator", x, y, stats, sigma_eps)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for key, v in report["true_class_residual"].items():
        print(key, {s: round(d["median"], 4) for s, d in v.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
