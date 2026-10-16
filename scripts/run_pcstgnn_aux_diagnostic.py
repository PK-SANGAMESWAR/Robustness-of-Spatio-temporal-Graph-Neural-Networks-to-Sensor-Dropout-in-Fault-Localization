"""S4.3d auxiliary network-only loss diagnostic (06_STAGE_E_pcstgnn.md), after S4.3c was unresolved.

One arm: the S4.3c setup (config/pcstgnn_noisediag/lam1_0p0.yaml) with only
    lambda_1 = 0, lambda_2 = 0, gamma = pc_loss.aux_net_loss_weight = 1
Seed 42, 5 epochs, noise + dropout augmentation on, beta learned. Validation split only:
training.skip_test_eval = true, so the test split is never loaded.

Logged every epoch on the full validation split (runner + _network_only_eval): fused macro-F1 at
k = 0, network-only macro-F1 at k = 0 and at sigma = 1e-2, network-only "no fault" share, beta,
and the auxiliary loss (train_aux, the unweighted aux CE averaged over the epoch).

Pre-registered decision rule (06_STAGE_E S4.3d), read at epoch 5:
    fixed     : network-only F1 > 0.10 and network-only no-fault share <= 0.99
    not fixed : anything else
Fused F1 is information only, compared to 0.879 (S4.3c lambda_1 = 0, epoch 5).

Resumable: skipped if the run's metrics.json exists.

Usage:
    uv run python scripts/run_pcstgnn_aux_diagnostic.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.experiments.config import load_config, validate_config  # noqa: E402

SEED = 42
GAMMA = 1.0
BASE_CONFIG = REPO / "config" / "pcstgnn_noisediag" / "lam1_0p0.yaml"
CONFIG_DIR = REPO / "config" / "pcstgnn_auxdiag"
VERSION = "pcstgnn_auxdiag_gamma1"
RUN_DIR = REPO / "results" / VERSION / "pcstgnn" / f"seed_{SEED:05d}"
SUMMARY = REPO / "results" / "pcstgnn_auxdiag" / "diagnostic_summary.json"
S43C_REF = REPO / "results" / "pcstgnn_noisediag" / "diagnostic_summary.json"

FIXED_F1, MAX_CLASS0 = 0.10, 0.99


def build_config() -> Path:
    cfg = validate_config(load_config(str(BASE_CONFIG)))
    cfg["pc_loss"].update(lambda1=0.0, lambda2=0.0, aux_net_loss_weight=GAMMA,
                          log_network_only=True)
    cfg["training"]["skip_test_eval"] = True
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "gamma1.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def main() -> int:
    if not (RUN_DIR / "metrics.json").exists():
        cmd = ["uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
               "--config", str(build_config()), "--seed", str(SEED),
               "--data-root", "dataset_fullscale_50", "--benchmark-version", VERSION]
        print(f"[run] {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print("[FAILED] training run", flush=True)
            return 1
    rows = list(csv.DictReader(open(RUN_DIR / "metrics_by_epoch.csv")))
    per_epoch = [{k: float(v) for k, v in r.items()} for r in rows]
    last = per_epoch[-1]
    fixed = last["val_net_f1"] > FIXED_F1 and last["val_net_class0_frac"] <= MAX_CLASS0
    ref = json.loads(S43C_REF.read_text())["arms"]["lam1_0p0"]["per_epoch"][-1]
    summary = {
        "setup": {"base": str(BASE_CONFIG.relative_to(REPO)), "lambda1": 0.0, "lambda2": 0.0,
                  "aux_net_loss_weight": GAMMA, "seed": SEED, "epochs": len(per_epoch),
                  "split": "val", "k": 0},
        "rule": {"fixed": f"net_f1 > {FIXED_F1} and class0 <= {MAX_CLASS0}",
                 "fused_f1": "information only, vs S4.3c lambda1=0 epoch 5"},
        "per_epoch": per_epoch,
        "s4_3c_lam1_0_epoch5": ref,
        "gpu_hours_train_loop": sum(r["train_seconds"] for r in per_epoch) / 3600,
        "verdict": "fixed" if fixed else "not fixed",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print("epoch | fused F1 | net F1 | net F1 @1e-2 | net no-fault | beta | aux", flush=True)
    for r in per_epoch:
        print(f"{int(r['epoch']):>5} | {r['val_graph_f1']:.4f} | {r['val_net_f1']:.4f} | "
              f"{r['val_net_f1_sigma1e-2']:.4f} | {r['val_net_class0_frac']:.4f} | "
              f"{r['beta']:.4f} | {r['train_aux']:.4f}", flush=True)
    print(f"verdict: {summary['verdict']}\nwrote {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
