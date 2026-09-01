"""Retrain queue for the three reviewer-driven gaps that need new training.

Each of these closes a criticism that no amount of re-evaluation of the frozen
checkpoints can answer, because the thing being questioned is what the model was
trained on.  They are ordered so that the first three runs give a one-seed
answer to all three questions in ~7 GPU-hours, and the rest complete n=3.

  scenario   The frozen split partitions the 50 runs of each scenario, so every
             (bus, fault type, resistance) configuration appears in training.
             This arm retrains on ``dataset_fullscale_50_scenario``, where whole
             scenarios are held out, and asks how much of the reported accuracy
             survives when the test configurations are genuinely unseen.
             Both the standard and the dropout-aware recipe are retrained, so
             the mitigation conclusion is re-tested under the harder split too.

  kgrid10    Dropout-aware training samples k ~ Uniform{0,1,2,5} but is
             evaluated out to k=10, so the headline number extrapolates beyond
             the training distribution.  This arm adds 10 to the training grid
             and asks whether that raises the k=10 result or costs clean
             accuracy — the method's only hyperparameter, previously unablated.

  seeds      The accuracy/robustness inversion holds in all three seeds but
             reaches only p=0.076, because the *size* of the gap varies by seed.
             The bound is the seed count, not the test set, so this arm trains
             the two architectures the claim rests on at three further seeds.

Nothing here touches the frozen corpus, the frozen splits, the committed
dropout-aware configs, or any existing results directory: new configs are
written under config/reviewer/, and every run lands in its own
``results/<benchmark_version>/``.  Runs are skipped when their metrics.json
already exists, so the queue is resumable after an interruption.

Usage:
    uv run python scripts/run_reviewer_retrains.py --list
    uv run python scripts/run_reviewer_retrains.py --arms scenario kgrid10
    uv run python scripts/run_reviewer_retrains.py            # everything
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from fault_localization.experiments.config import load_config, validate_config

REPO = Path(__file__).resolve().parent.parent
CONFIG_OUT = REPO / "config" / "reviewer"

FROZEN_ROOT = "dataset_fullscale_50"
SCENARIO_ROOT = "dataset_fullscale_50_scenario"
EPOCHS = 30
GNN_LAYERS = 4


def write_config(name: str, model: str, *, data_root: str,
                 dropout_aware: bool, train_k_range: list[int] | None) -> Path:
    """Materialize a fully-resolved config so a run never depends on CLI drift."""
    cfg = validate_config(load_config(f"config/{model}.yaml"))
    cfg["device"] = "cuda"
    cfg["data_root"] = data_root
    cfg["output_root"] = "results"
    cfg["training"]["epochs"] = EPOCHS
    cfg["training"]["grad_clip_norm"] = 1.0
    cfg["training"]["warmup_epochs"] = 3
    if model != "gru":
        cfg["gnn_layers"] = GNN_LAYERS

    ds = cfg.setdefault("dropout_study", {})
    ds["enabled"] = False  # degradation curves come from the study scripts
    ds["policy"] = "zero_messages"
    ds["dropout_aware_training"] = dropout_aware
    if train_k_range is not None:
        ds["train_k_range"] = train_k_range

    CONFIG_OUT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_OUT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def build_queue(arms: list[str]) -> list[dict]:
    """Ordered run list: one seed of each arm first, then the n=3 remainder."""
    jobs: list[dict] = []

    def add(arm, name, model, seed, data_root, dropout_aware, k_range, bench, tier):
        jobs.append({
            "arm": arm, "config_name": name, "model": model, "seed": seed,
            "data_root": data_root, "dropout_aware": dropout_aware,
            "train_k_range": k_range, "benchmark_version": bench, "tier": tier,
        })

    # Tier 1 — one seed of every arm, so each question has an answer early.
    if "scenario" in arms:
        add("scenario", "scenario_std", "rgsage_max", 42, SCENARIO_ROOT,
            False, None, "scenario_heldout_std", 1)
        add("scenario", "scenario_da", "rgsage_max", 42, SCENARIO_ROOT,
            True, [0, 1, 2, 5], "scenario_heldout_da", 1)
    if "kgrid10" in arms:
        add("kgrid10", "kgrid10_da", "rgsage_max", 42, FROZEN_ROOT,
            True, [0, 1, 2, 5, 10], "dropout_aware_k10", 1)

    # Tier 2 — complete n=3 on the two arms that make claims.
    if "scenario" in arms:
        for seed in (43, 44):
            add("scenario", "scenario_std", "rgsage_max", seed, SCENARIO_ROOT,
                False, None, "scenario_heldout_std", 2)
        for seed in (43, 44):
            add("scenario", "scenario_da", "rgsage_max", seed, SCENARIO_ROOT,
                True, [0, 1, 2, 5], "scenario_heldout_da", 2)
    if "kgrid10" in arms:
        for seed in (43, 44):
            add("kgrid10", "kgrid10_da", "rgsage_max", seed, FROZEN_ROOT,
                True, [0, 1, 2, 5, 10], "dropout_aware_k10", 2)

    # Tier 3 — extra seeds for the inversion, the most expensive and the most
    # easily dropped if the budget runs out.
    if "seeds" in arms:
        for model in ("rgsage_max", "rgcn"):
            for seed in (45, 46, 47):
                add("seeds", f"extraseed_{model}", model, seed, FROZEN_ROOT,
                    False, None, "fullscale_extra_seeds", 3)

    jobs.sort(key=lambda j: j["tier"])
    return jobs


def run_one(job: dict, dry_run: bool) -> tuple[bool, float]:
    run_dir = (REPO / "results" / job["benchmark_version"] / job["model"]
               / f"seed_{job['seed']:05d}")
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {job['benchmark_version']}/{job['model']} "
              f"seed={job['seed']} — already complete", flush=True)
        return True, 0.0

    cfg_path = write_config(
        job["config_name"], job["model"], data_root=job["data_root"],
        dropout_aware=job["dropout_aware"], train_k_range=job["train_k_range"],
    )
    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", str(cfg_path.relative_to(REPO)),
        "--seed", str(job["seed"]),
        "--data-root", job["data_root"],
        "--benchmark-version", job["benchmark_version"],
        "--epochs", str(EPOCHS),
        "--grad-clip-norm", "1.0",
        "--warmup-epochs", "3",
    ]
    if job["model"] != "gru":
        cmd += ["--gnn-layers", str(GNN_LAYERS)]

    print(f"[run] ({job['arm']}) {' '.join(cmd)}", flush=True)
    if dry_run:
        return True, 0.0

    start = time.time()
    result = subprocess.run(cmd, cwd=REPO)
    elapsed = time.time() - start
    ok = result.returncode == 0
    print(f"[{'ok' if ok else f'FAILED rc={result.returncode}'}] "
          f"{job['benchmark_version']}/{job['model']} seed={job['seed']} — "
          f"{elapsed/60:.1f} min", flush=True)
    return ok, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=["scenario", "kgrid10", "seeds"],
                   choices=["scenario", "kgrid10", "seeds"])
    p.add_argument("--max-tier", type=int, default=3,
                   help="Run only jobs at or below this tier (1 = one seed each).")
    p.add_argument("--list", action="store_true", help="Print the queue and exit.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = [j for j in build_queue(args.arms) if j["tier"] <= args.max_tier]
    print(f"Queue: {len(jobs)} run(s)\n")
    for j in jobs:
        print(f"  tier{j['tier']} {j['arm']:<9} {j['model']:<12} seed={j['seed']} "
              f"data={j['data_root']:<32} DA={j['dropout_aware']} "
              f"k={j['train_k_range']}")
    if args.list:
        return 0

    failures: list[str] = []
    total = time.time()
    for j in jobs:
        ok, _ = run_one(j, args.dry_run)
        if not ok:
            failures.append(f"{j['benchmark_version']}/{j['model']} seed={j['seed']}")

    print(f"\nQueue done in {(time.time()-total)/3600:.2f} h. "
          f"{len(failures)} failure(s).", flush=True)
    for f in failures:
        print(f"  FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
