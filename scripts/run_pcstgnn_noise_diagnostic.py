"""S4.3c collapse diagnostic (06_STAGE_E_pcstgnn.md), run before any noise-augmented re-sweep.

The noise-free S4.3 sweep collapsed in all six arms: the network branch alone votes "no fault" on
~100 % of validation windows and the fused model is PSM plus a no-fault bias. S4.3b predicts this
for noise-free training. This script answers one question cheaply: does noise augmentation stop
the collapse?

Two arms, 5 epochs each, seed 42, everything else as S4.3:
    lambda_1 = 0       (control: fusion only, no physics-consistency loss)
    lambda_1 = 0.0333  (the noise-free S4.3 winner)
lambda_2 = 0, beta learned, use_fusion=True, dropout augmentation on (k in {0,1,2,5}), noise
augmentation on: sigma drawn per window from {0, 5e-4, 2e-3, 1e-2} p.u. via dropout/noise.py,
sigma0 in the residual = the sigma injected. The noise-free k=0 rho cache is OFF (it cannot serve a
noisy window).

Logged every epoch on the full validation split (k = 0): network-only macro-F1 and no-fault
fraction (clean), network-only macro-F1 at sigma = 1e-2, and the standard fused val F1.

Pre-registered decision rule (written into 06_STAGE_E S4.3c before this script was run), read on
each arm's final epoch:
    fixed      : network-only F1 > 0.10 and no-fault fraction < 0.95
    collapsed  : no-fault fraction > 0.99
    unresolved : otherwise
    >= 1 arm fixed -> noise-augmented S4.3 rerun; both collapsed -> fix the fusion design first.

Runs serially (single T1000). Resumable: an arm with metrics.json is skipped.

Usage:
    uv run python scripts/run_pcstgnn_noise_diagnostic.py
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
EPOCHS = 5
BASE_CONFIG = "config/dropout_aware/pcstgnn.yaml"
CANONICAL_SIGMA_EPS = 0.0007255549105131452
SIGMA_GRID = [0.0, 5e-4, 2e-3, 1e-2]
LAMBDA1_ARMS = [0.0, 0.0333]
CONFIG_DIR = REPO / "config" / "pcstgnn_noisediag"
SUMMARY = REPO / "results" / "pcstgnn_noisediag" / "diagnostic_summary.json"

FIXED_F1, FIXED_CLASS0, COLLAPSED_CLASS0 = 0.10, 0.95, 0.99


def tag(lam1: float) -> str:
    return f"lam1_{str(lam1).replace('.', 'p')}"


def run_dir(lam1: float) -> Path:
    return REPO / "results" / f"pcstgnn_noisediag_{tag(lam1)}" / "pcstgnn" / f"seed_{SEED:05d}"


def build_config(lam1: float) -> Path:
    cfg = validate_config(load_config(BASE_CONFIG))
    cfg["training"]["epochs"] = EPOCHS
    cfg["model"].update(use_fusion=True, beta_init=0.0, sigma_eps=CANONICAL_SIGMA_EPS, sigma0=0.0)
    cfg["pc_loss"] = {"enabled": True, "lambda1": float(lam1), "lambda2": 0.0,
                      "log_network_only": True}
    cfg["noise_augmentation"] = {"enabled": True, "sigma_grid": SIGMA_GRID}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{tag(lam1)}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def train(lam1: float) -> bool:
    if (run_dir(lam1) / "metrics.json").exists():
        print(f"[skip] lambda1={lam1} already complete", flush=True)
        return True
    cmd = ["uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
           "--config", str(build_config(lam1)), "--seed", str(SEED),
           "--data-root", "dataset_fullscale_50",
           "--benchmark-version", f"pcstgnn_noisediag_{tag(lam1)}"]
    print(f"[run] lambda1={lam1}: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode == 0


def classify(net_f1: float, class0: float) -> str:
    if net_f1 > FIXED_F1 and class0 < FIXED_CLASS0:
        return "fixed"
    if class0 > COLLAPSED_CLASS0:
        return "collapsed"
    return "unresolved"


def main() -> int:
    print(f"S4.3c collapse diagnostic -- seed {SEED}, {EPOCHS} epochs, noise grid {SIGMA_GRID}, "
          f"arms lambda1={LAMBDA1_ARMS} (serial)", flush=True)
    summary = {"rule": {"fixed": f"net_f1 > {FIXED_F1} and class0 < {FIXED_CLASS0}",
                        "collapsed": f"class0 > {COLLAPSED_CLASS0}"}, "arms": {}}
    for lam1 in LAMBDA1_ARMS:
        if not train(lam1):
            print(f"[FAILED] lambda1={lam1}", flush=True)
            return 1
        rows = list(csv.DictReader(open(run_dir(lam1) / "metrics_by_epoch.csv")))
        per_epoch = [{k: float(v) for k, v in r.items()} for r in rows]
        last = per_epoch[-1]
        verdict = classify(last["val_net_f1"], last["val_net_class0_frac"])
        summary["arms"][tag(lam1)] = {"lambda1": lam1, "per_epoch": per_epoch, "verdict": verdict}
        print(f"\n  lambda1={lam1}:  epoch | fused val F1 | net-only F1 | net no-fault frac | "
              f"net F1 @1e-2 | beta", flush=True)
        for r in per_epoch:
            print(f"    {int(r['epoch']):>5} | {r['val_graph_f1']:.4f} | {r['val_net_f1']:.4f} | "
                  f"{r['val_net_class0_frac']:.4f} | {r['val_net_f1_sigma1e-2']:.4f} | "
                  f"{r.get('beta', float('nan')):.4f}", flush=True)
        print(f"  -> verdict: {verdict}", flush=True)

    verdicts = [a["verdict"] for a in summary["arms"].values()]
    if "fixed" in verdicts:
        decision = "noise augmentation breaks the collapse -> proceed to noise-augmented S4.3 rerun"
    elif all(v == "collapsed" for v in verdicts):
        decision = "collapse persists under noise -> fusion-design problem; do not rerun S4.3/S4.4"
    else:
        decision = "unresolved -> report; author decides"
    summary["decision"] = decision
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\n=== S4.3c diagnostic complete === {decision}\nwrote {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
