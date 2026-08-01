"""Train SDR-STGNN — the novel availability-gated, sensor-dropout-robust model.

SDR-STGNN (fault_localization/models/sdr_stgnn.py) is RGSAGE-Max with
availability-gated message passing: dead μPMUs emit no messages, so a dropped
node is represented purely by its live neighbours. It is trained WITH sensor-
dropout augmentation (dropout_study.dropout_aware_training in config/sdr_stgnn.yaml)
so the gates actually get exercised, then the runner auto-emits the degradation
curve (dropout_study.enabled → dropout_curve.json) using the same paired-mask
evaluator as scripts/run_dropout_study.py.

Frozen recipe — identical to run_fullscale_final_sweep.py (depth 4, 30 epochs,
grad-clip 1.0, warmup 3) — the ONLY architectural change is the gated aggregation.
Seed 42; output results/sdr_stgnn/sdr_stgnn/seed_00042/. Resumable + crash-isolated.

The model + config are corpus-agnostic: pass --data-root dataset_fullscale_100 to
re-run on the 100-run corpus later. Nothing here touches config/*.yaml the other
sweeps read (dry-run writes a temp config to the scratch/output dir only).

Usage (real, after the baseline sweep + Step 3 gate; GPU must be free):
    uv run python scripts/run_sdr_stgnn.py

Dry-run (CPU, tiny data, 1 epoch, tiny curve — scaffold check, no GPU):
    uv run python scripts/run_sdr_stgnn.py --dry-run \
        --output-root <scratch>/sdr_dryrun
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from fault_localization.experiments.config import load_config, validate_config

MODEL = "sdr_stgnn"
SEED = 42


def _write_dryrun_config(output_root: str) -> Path:
    """Materialise a tiny CPU config for the scaffold check (never mutates repo configs)."""
    cfg = validate_config(load_config(f"config/{MODEL}.yaml"))
    cfg["device"] = "cpu"
    ds = cfg["dropout_study"]
    ds["k_levels"] = [0, 1]      # keep the auto degradation-curve tiny/fast
    ds["n_mask_seeds"] = 1
    ds["train_k_range"] = [0, 1]
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sdr_stgnn_dryrun.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU, tiny data, 1 epoch, tiny curve — scaffold check.")
    p.add_argument("--output-root", default="results")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--benchmark-version", default="sdr_stgnn")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--gnn-layers", type=int, default=4)
    p.add_argument("--grad-clip-norm", default="1.0")
    p.add_argument("--warmup-epochs", default="3")
    args = p.parse_args()

    if args.dry_run:
        config_path = _write_dryrun_config(args.output_root)
        data_root = "dataset"
        epochs = 1
        gnn_layers = 2
        print("=== DRY-RUN: cpu / dataset (tiny) / 1 epoch / tiny curve ===", flush=True)
    else:
        config_path = Path(f"config/{MODEL}.yaml")
        data_root = args.data_root
        epochs = args.epochs
        gnn_layers = args.gnn_layers

    run_dir = Path(args.output_root) / args.benchmark_version / MODEL / f"seed_{args.seed:05d}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {MODEL} seed={args.seed} — already complete", flush=True)
        return 0

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", str(config_path),
        "--seed", str(args.seed),
        "--data-root", data_root,
        "--benchmark-version", args.benchmark_version,
        "--epochs", str(epochs),
        "--gnn-layers", str(gnn_layers),
        "--grad-clip-norm", str(args.grad_clip_norm),
        "--warmup-epochs", str(args.warmup_epochs),
    ]
    # The runner writes to config["output_root"]; override via the config for the
    # dry-run's scratch dir (real runs use the default results/).
    if args.output_root != "results":
        c = yaml.safe_load(config_path.read_text())
        c["output_root"] = args.output_root
        config_path.write_text(yaml.safe_dump(c, sort_keys=True))

    print(f"[run] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (rc={result.returncode})"
    print(f"[{status}] {MODEL} seed={args.seed} — {elapsed/60:.1f} min", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
