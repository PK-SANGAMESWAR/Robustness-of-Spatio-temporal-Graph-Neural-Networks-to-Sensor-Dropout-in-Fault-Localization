"""NEXT-STEPS.md Step 4b — dropout-aware training sweep (scaffold).

Retrains the models (seed 42) with random node masking DURING training
(DropoutAugmentedLoader, variable k per batch), same FROZEN config +
stabilization as the Step 2 baseline. The resulting dropout-aware checkpoints
are then evaluated by scripts/run_dropout_study.py (Step 4a evaluator) exactly
like the standard models, and scripts/build_dropout_comparison.py overlays the
two — answering "how much accuracy at each k does dropout-aware training
recover vs standard training?".

Frozen config (identical to run_fullscale_final_sweep.py, the ONLY difference
being masking-during-training):
  - GNN depth = 4 (GNNs), epochs = 30, weighted CE, grad-clip 1.0, warmup 3.
  - dropout_study.dropout_aware_training = True, policy = zero_messages,
    train_k_range = [0, 1, 2, 5]  (variable nodes masked per batch).
  - Same data/splits (dataset_fullscale_50), seed 42.

Generated configs are written to config/dropout_aware/<model>.yaml — NEW files,
never touching the existing config/*.yaml that the running baseline sweep reads.
Each (model, seed) is a crash-isolated subprocess, skipped if already complete
(resumable), same pattern as every sweep in this project.

IMPORTANT: The real run is GPU + full data and must wait for the Step 2 sweep to
finish and Step 3 to gate the baseline as clean. Use --dry-run to exercise the
whole scaffold on CPU + tiny data (1 epoch, 1 model) without touching the GPU or
the running sweep.

Usage (real, Step 4b — after Step 2 sweep + Step 3 gate):
    uv run python scripts/run_dropout_aware_sweep.py

Dry-run (CPU, tiny data, 1 epoch, rgsage_max only — scaffold check):
    uv run python scripts/run_dropout_aware_sweep.py --dry-run \
        --output-root <scratch>/dropout_aware_dryrun
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

MODELS = ["rgatv2", "rgcn", "gru", "rgsage_mean", "rgsage_max"]
SEEDS = [42]
BENCHMARK_VERSION = "dropout_aware"
CONFIG_OUT_DIR = Path("config/dropout_aware")


def make_dropout_aware_config(model: str, *, device: str, epochs: int,
                              gnn_layers: int, output_root: str,
                              data_root: str) -> Path:
    """Build a self-contained dropout-aware config for one model and write it.

    Fully resolves base + model config, then bakes in the frozen recipe
    (depth, epochs, stabilization) and flips on dropout-aware training. Written
    to config/dropout_aware/ — a fresh file that never mutates the baseline
    configs the running sweep depends on.
    """
    cfg = validate_config(load_config(f"config/{model}.yaml"))
    cfg["device"] = device
    cfg["data_root"] = data_root
    cfg["training"]["epochs"] = epochs
    cfg["training"]["grad_clip_norm"] = 1.0
    cfg["training"]["warmup_epochs"] = 3
    cfg["output_root"] = output_root
    if model != "gru":
        cfg["gnn_layers"] = gnn_layers

    ds = cfg.setdefault("dropout_study", {})
    ds["dropout_aware_training"] = True   # THE difference vs the baseline sweep
    ds["enabled"] = False                 # degradation eval is done by run_dropout_study.py
    ds["policy"] = "zero_messages"
    ds["train_k_range"] = [0, 1, 2, 5]

    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_OUT_DIR / f"{model}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def run_one(model: str, seed: int, cfg_path: Path, *, benchmark_version: str,
            data_root: str, output_root: str, dry_run: bool) -> tuple[bool, float]:
    run_dir = Path(output_root) / benchmark_version / model / f"seed_{seed:05d}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {model} seed={seed} — already complete", flush=True)
        return True, 0.0

    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", str(cfg_path),
        "--seed", str(seed),
        "--data-root", data_root,
        "--benchmark-version", benchmark_version,
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    if dry_run:
        print("[dry-run] executing tiny CPU training to prove the scaffold ...",
              flush=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (rc={result.returncode})"
    print(f"[{status}] {model} seed={seed} — {elapsed/60:.1f} min", flush=True)
    return ok, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU, tiny data, 1 epoch, rgsage_max only — scaffold check.")
    p.add_argument("--output-root", default="results")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--benchmark-version", default=BENCHMARK_VERSION)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="Training seeds to run (default: [42]). Resumable: any "
                        "(model, seed) with an existing metrics.json is skipped.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--gnn-layers", type=int, default=4)
    args = p.parse_args()

    if args.dry_run:
        device = "cpu"
        epochs = 1
        models = ["rgsage_max"]
        seeds = [42]
        data_root = "dataset"
        print("=== DRY-RUN: cpu / dataset (tiny) / 1 epoch / rgsage_max ===")
    else:
        device = "cuda"
        epochs = args.epochs
        models = args.models
        seeds = args.seeds
        data_root = args.data_root

    failures: list[str] = []
    total_start = time.time()
    for model in models:
        cfg_path = make_dropout_aware_config(
            model, device=device, epochs=epochs, gnn_layers=args.gnn_layers,
            output_root=args.output_root, data_root=data_root,
        )
        for seed in seeds:
            ok, _ = run_one(model, seed, cfg_path,
                            benchmark_version=args.benchmark_version,
                            data_root=data_root, output_root=args.output_root,
                            dry_run=args.dry_run)
            if not ok:
                failures.append(f"{model} seed={seed}")

    total = time.time() - total_start
    print(f"\nDropout-aware sweep done in {total/60:.1f} min. "
          f"{len(failures)} failure(s).", flush=True)
    if failures:
        print("Failed runs:\n  " + "\n  ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
