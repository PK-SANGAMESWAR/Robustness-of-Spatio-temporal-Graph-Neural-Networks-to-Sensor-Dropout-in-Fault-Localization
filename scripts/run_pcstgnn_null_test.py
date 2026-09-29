"""S4.1 -- the null test, run before any other Stage E work (06_STAGE_E_pcstgnn.md).

Trains PCSTGNN (models/pc_stgnn.py) with every physics switch off -- lambda_1 =
lambda_2 = beta = 0, "gate" off (there is no separate gate parameter in
PCSTGNN; use_fusion=False is the whole of it) -- at seed 42, under the exact
same recipe as R0 (dropout-aware RGSAGE-Max, no noise aug):
config/dropout_aware/pcstgnn.yaml is config/dropout_aware/rgsage_max.yaml with
only model.name changed (generated via
scripts.run_dropout_aware_sweep.make_dropout_aware_config, not hand-written,
so the hyperparameters cannot drift).

With every switch off, PCSTGNN is byte-for-byte RGSAGE-Max at init
(tests/test_pc_stgnn.py::test_all_switches_off_is_exactly_rgsage) and no
physics computation ever runs inside forward() (models/pc_stgnn.py:
``if not self.use_fusion: return logits`` -- no rho, no table lookup, no extra
RNG draws), so the per-epoch training curve is the property under test: it
must reproduce R0 (results/dropout_aware/rgsage_max/seed_00042/) exactly. Any
divergence means an unintended change in the data path, seeding, or graph
construction that would confound every later attribution claim in Stage E.

Output: results/dropout_aware/pcstgnn/seed_00042/ (metrics.json,
metrics_by_epoch.csv, checkpoint_best.pt, ...) -- same benchmark_version
("dropout_aware") as R0, so both runs are checked against the same frozen
dataset manifest.

Usage:
    uv run python scripts/run_pcstgnn_null_test.py
    uv run python scripts/run_pcstgnn_null_test.py --check-only   # just compare, no training
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SEED = 42
BENCHMARK_VERSION = "dropout_aware"
R0_DIR = REPO / "results" / BENCHMARK_VERSION / "rgsage_max" / f"seed_{SEED:05d}"
NULL_DIR = REPO / "results" / BENCHMARK_VERSION / "pcstgnn" / f"seed_{SEED:05d}"
CONFIG = REPO / "config" / "dropout_aware" / "pcstgnn.yaml"


def read_metrics_by_epoch(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def compare(r0_rows: list[dict], null_rows: list[dict]) -> dict:
    """Per-epoch exact-reproduction check on train_loss and val_graph_f1.

    train_seconds is excluded -- wall-clock, not a reproduction criterion.
    """
    result = {"n_epochs_r0": len(r0_rows), "n_epochs_null": len(null_rows),
              "first_divergence_epoch": None, "max_abs_diff_loss": 0.0,
              "max_abs_diff_val_f1": 0.0, "exact": True}
    for i, (a, b) in enumerate(zip(r0_rows, null_rows)):
        epoch = a["epoch"]
        if epoch != b["epoch"]:
            result["exact"] = False
            result["first_divergence_epoch"] = epoch
            break
        d_loss = abs(float(a["train_loss"]) - float(b["train_loss"]))
        d_f1 = abs(float(a["val_graph_f1"]) - float(b["val_graph_f1"]))
        result["max_abs_diff_loss"] = max(result["max_abs_diff_loss"], d_loss)
        result["max_abs_diff_val_f1"] = max(result["max_abs_diff_val_f1"], d_f1)
        if d_loss > 0 or d_f1 > 0:
            result["exact"] = False
            if result["first_divergence_epoch"] is None:
                result["first_divergence_epoch"] = epoch
    if len(r0_rows) != len(null_rows):
        result["exact"] = False
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true",
                    help="skip training; compare NULL_DIR against R0_DIR if it already exists")
    ap.add_argument("--data-root", default="dataset_fullscale_50")
    args = ap.parse_args()

    if not R0_DIR.exists():
        print(f"R0 run not found at {R0_DIR} -- nothing to reproduce against.")
        return 1

    if not args.check_only:
        if (NULL_DIR / "metrics.json").exists():
            print(f"[skip] null-test run already complete at {NULL_DIR}", flush=True)
        else:
            cmd = [
                "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
                "--config", str(CONFIG),
                "--seed", str(SEED),
                "--data-root", args.data_root,
                "--benchmark-version", BENCHMARK_VERSION,
            ]
            print(f"[run] {' '.join(cmd)}", flush=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[FAILED] training exited with code {result.returncode}")
                return result.returncode

    r0_rows = read_metrics_by_epoch(R0_DIR / "metrics_by_epoch.csv")
    null_rows = read_metrics_by_epoch(NULL_DIR / "metrics_by_epoch.csv")
    cmp = compare(r0_rows, null_rows)

    print("\n=== S4.1 null test: PCSTGNN(all switches off) vs R0 (dropout-aware RGSAGE-Max) ===")
    print(f"  R0 epochs:   {cmp['n_epochs_r0']}")
    print(f"  null epochs: {cmp['n_epochs_null']}")
    print(f"  max |train_loss diff|: {cmp['max_abs_diff_loss']:.10f}")
    print(f"  max |val_graph_f1 diff|: {cmp['max_abs_diff_val_f1']:.10f}")
    if cmp["exact"]:
        print("  PASS -- exact reproduction.")
    else:
        print(f"  FAIL -- first divergence at epoch {cmp['first_divergence_epoch']}.")
    return 0 if cmp["exact"] else 2


if __name__ == "__main__":
    sys.exit(main())
