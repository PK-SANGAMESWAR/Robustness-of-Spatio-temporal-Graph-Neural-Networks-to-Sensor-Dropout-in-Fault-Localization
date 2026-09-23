"""NEXT-STEPS.md Step 4a — inference-time sensor-dropout degradation curves.

For each trained (seed-42) model in a results directory, evaluate macro-F1 with
k μPMUs dropped at inference, k ∈ {0,1,2,5,10} of 25. Emits a degradation-curve
CSV + an overlay figure (all models, F1 vs k).

Paired comparison (NEXT-STEPS.md 4a hard requirement): the mask seed for a given
(k, mask-seed-offset) is deterministic and model-independent
(``mask_seed = 10000*k + offset`` inside evaluator.py), and every model is
evaluated on the SAME shuffle=False test loader — so the SAME k nodes are dropped
for every model at a given (k, offset). This is a true paired degradation study,
not five independently-masked ones.

Mechanism (audited, locked in NEXT-STEPS.md 4a): dropout = feature masking
("zero_messages"). Dropped nodes stay in the graph; voltage channels [0:3] AND
availability-mask channels [3:6] are zeroed; edges are NOT removed. The
"drop_edges" policy is unexercised/future-work and MUST NOT be used here.

This script does NOT train anything — it loads existing checkpoints and evaluates.
Run it AFTER the baseline sweep completes and Step 3 gates the baseline as clean.

Usage (real, Step 4a — after the sweep + Step 3 gate):
    uv run python scripts/run_dropout_study.py \
        --results-dir results/fullscale_final \
        --data-root dataset_fullscale_50 \
        --output-dir results/dropout_study

Dry-run (CPU, tiny data, existing tiny-set checkpoints — shape check only):
    uv run python scripts/run_dropout_study.py \
        --results-dir results/benchmark_v1.1_samplerfix \
        --data-root dataset --device cpu --n-mask-seeds 2 \
        --output-dir <scratch>/dropout_study_dryrun
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.dropout.evaluator import evaluate_dropout_curve
from fault_localization.experiments.runner import _model

# Figure palette — matches scripts/build_fullscale_final_figures.py for a
# consistent look across every figure in the paper.
MODELS_DEFAULT = ["gru", "rgcn", "rgsage_mean", "rgsage_max", "rgatv2"]
MODEL_LABELS = {"gru": "GRU", "rgcn": "RGCN", "rgsage_mean": "RGSAGE-Mean",
                "rgsage_max": "RGSAGE-Max", "rgatv2": "RGATv2"}
MODEL_COLOR = {"gru": "#2a78d6", "rgcn": "#eb6834", "rgsage_mean": "#1baf7a",
               "rgsage_max": "#eda100", "rgatv2": "#e87ba4"}
INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"


def load_trained_model(seed_dir: Path, edge_index, device: torch.device):
    """Rebuild a model from its run config.yaml and load checkpoint_best.pt."""
    cfg = yaml.safe_load((seed_dir / "config.yaml").read_text())
    # Replicate runner's edges-off ablation handling (off for all real runs).
    disable_edges = cfg.get("ablation", {}).get("disable_edges", False)
    model_edge = np.zeros((2, 0), dtype=np.int64) if disable_edges else edge_index
    model = _model(cfg, model_edge).to(device)
    ckpt = torch.load(seed_dir / "checkpoint_best.pt", map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def aggregate_over_seeds(per_seed_rows: "list[dict]",
                         model_order: "list[str]") -> "list[dict]":
    """Collapse per-(seed, model, k) records into one row per (model, k).

    ``f1_mean`` is the mean over TRAINING seeds of each seed's (mask-mean) F1.
    ``f1_std`` is the std over training seeds (sample, ddof=1) when >1 seed is
    present — the multi-seed error bar the paper figures draw — and falls back to
    the single seed's mask-seed std when only one training seed is given, so the
    default (seed-42-only) path is byte-for-byte unchanged. Downstream figure
    scripts read only (model, k, f1_mean, f1_std) and one row per (model, k), so
    this aggregation keeps them working without any change.
    """
    import pandas as pd
    df = pd.DataFrame(per_seed_rows)
    order = {m: i for i, m in enumerate(model_order)}
    out: list[dict] = []
    for (model_name, k), g in df.groupby(["model", "k"], sort=False):
        f1s = g["f1_seed"].to_numpy(dtype=float)
        n = int(len(f1s))
        if n > 1:
            f1_std = float(f1s.std(ddof=1)); err_kind = "train_seed_std"
        else:
            f1_std = float(g["mask_std"].iloc[0]); err_kind = "mask_seed_std"
        out.append({
            "model": model_name, "k": int(k),
            "f1_mean": float(f1s.mean()), "f1_std": f1_std,
            "acc_mean": float(g["acc_seed"].mean()),
            "n_train_seeds": n, "n_mask_seeds": int(g["n_mask_seeds"].iloc[0]),
            "policy": g["policy"].iloc[0], "err_kind": err_kind,
            # The mask realisation is a pure function of (k, mask offset,
            # batch_idx, batch size) -- see dropout/evaluator.py -- so a curve
            # is only reproducible to the bit if the batch size is recorded
            # with it.  It never was, which is the whole reason the stored
            # Paper-1 curves and a fresh re-evaluation differ by ~1e-3 macro-F1
            # at k > 0 (PC-STGNN Stage A, S0.1).  Recorded from here on.
            "batch_size": (int(g["batch_size"].iloc[0])
                           if "batch_size" in g.columns else -1),
            "seeds": json.dumps(sorted(int(s) for s in g["seed"])),
            "f1_per_train_seed": json.dumps([round(float(x), 6) for x in f1s]),
        })
    out.sort(key=lambda d: (order.get(d["model"], 99), d["k"]))
    return out


def evaluate_all_models(args) -> "list[dict]":
    device = torch.device(args.device)
    results_dir = Path(args.results_dir)

    test_ds = WindowGraphDataset(args.data_root, "test")
    test_loader = make_dataloader(test_ds, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    per_seed_rows: list[dict] = []
    for seed in args.seeds:
        for model_name in args.models:
            seed_dir = results_dir / model_name / f"seed_{seed:05d}"
            ckpt = seed_dir / "checkpoint_best.pt"
            if not ckpt.exists():
                print(f"[skip] {model_name} seed={seed}: no checkpoint at {ckpt}",
                      flush=True)
                continue

            model, cfg = load_trained_model(seed_dir, test_ds.edge_index, device)
            f1_avg = cfg.get("metrics", {}).get("f1_average", "macro")
            print(f"[eval] {model_name} seed={seed} (device={device}) "
                  f"over k={args.k_levels} ...", flush=True)

            curve = evaluate_dropout_curve(
                model=model, loader=test_loader, device=device,
                dropout_levels=args.k_levels, n_mask_seeds=args.n_mask_seeds,
                policy=args.policy, f1_average=f1_avg,
                model_name=f"{model_name}/seed{seed}",
            )
            for k in args.k_levels:
                r = curve[k]
                per_seed_rows.append({
                    "model": model_name, "seed": seed, "k": k,
                    "f1_seed": r["f1_mean"], "acc_seed": r["acc_mean"],
                    "mask_std": r["f1_std"], "n_mask_seeds": r["n_mask_seeds"],
                    "policy": r["policy"], "batch_size": int(args.batch_size),
                })
    if not per_seed_rows:
        return []
    return aggregate_over_seeds(per_seed_rows, args.models)


def write_csv(rows: "list[dict]", out_dir: Path) -> Path:
    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "degradation_curves.csv"
    df.to_csv(path, index=False)
    return path


def build_figure(rows: "list[dict]", out_dir: Path, k_levels: "list[int]") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "font.family": "sans-serif",
    })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED); ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0)

    for model_name in [m for m in MODELS_DEFAULT if m in df["model"].values]:
        g = df[df["model"] == model_name].sort_values("k")
        ax.errorbar(
            g["k"], g["f1_mean"], yerr=g["f1_std"],
            marker="o", capsize=4, linewidth=2, markersize=6,
            color=MODEL_COLOR.get(model_name, INK_SECONDARY),
            label=MODEL_LABELS.get(model_name, model_name), zorder=3,
        )
    multiseed = any(r.get("err_kind") == "train_seed_std" for r in rows)
    err_over = "training seeds" if multiseed else "mask seeds"
    ax.set_xticks(k_levels)
    ax.set_xlabel("μPMUs dropped at inference (k of 25)")
    ax.set_ylabel(f"Soft-voted graph macro-F1 (test), mean ± std over {err_over}")
    ax.set_title("Sensor-dropout degradation curves (feature-masking)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out_dir / "degradation_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/fullscale_final")
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--output-dir", default="results/dropout_study")
    p.add_argument("--models", nargs="+", default=MODELS_DEFAULT)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Training seeds to evaluate and aggregate as mean ± std "
                        "over seeds (e.g. --seeds 42 43 44). Default: [42].")
    p.add_argument("--seed", type=int, default=None,
                   help="(back-compat) single training seed; ignored if --seeds "
                        "is given.")
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
    p.add_argument("--policy", default="zero_messages",
                   choices=["zero_messages"],  # drop_edges is unexercised — locked out
                   help="Only zero_messages is supported (NEXT-STEPS.md 4a).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers; 0 was the old hardcoded value "
                        "and is I/O bound on this corpus.")
    args = p.parse_args()
    if args.seeds is None:
        args.seeds = [args.seed] if args.seed is not None else [42]

    rows = evaluate_all_models(args)
    if not rows:
        print("No models evaluated (no checkpoints found).")
        return 1
    out_dir = Path(args.output_dir)
    csv_path = write_csv(rows, out_dir)
    fig_path = build_figure(rows, out_dir, args.k_levels)
    print(f"\nWrote {csv_path} ({len(rows)} rows) and {fig_path}")

    # Compact console table
    import pandas as pd
    pivot = pd.DataFrame(rows).pivot(index="model", columns="k", values="f1_mean")
    print("\n--- macro-F1 vs k (mean over mask seeds) ---")
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
