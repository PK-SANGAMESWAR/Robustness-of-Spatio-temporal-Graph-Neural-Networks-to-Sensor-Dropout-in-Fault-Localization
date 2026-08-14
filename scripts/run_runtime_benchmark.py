"""Runtime / efficiency benchmark for the paper's cost claims.

Reviewer request (CHANGES.md items 2 and 10): the paper repeatedly asserts that
dropout-aware training carries "no inference-time cost" but never measures
latency, memory or throughput, and never states training time or convergence
speed.  This script measures all of it.

Two halves, from two different sources.

**Inference (measured here, on the GPU/CPU of this machine).**  Each trained
checkpoint is rebuilt from its own ``config.yaml`` and timed on synthetic
batches of the true input shape ``[B, 25, 20, 6]``.  Synthetic input is
deliberate: the real corpus stores one compressed ``.npz`` per event, so a
loader-based timing measures decompression, not the model.  Reported per
architecture: parameter count, mean±std latency at batch size 1 (the
single-event, latency-critical case) and at the evaluation batch size,
throughput in windows/s, and peak CUDA memory allocated by the forward pass.

The "no inference-time cost" claim is tested directly by timing the *standard*
and *dropout-aware* RGSAGE-Max checkpoints as separate entries: dropout-aware
training changes only the training data, so the two must come out equal within
measurement noise, and the table shows whether they do.  SDR-STGNN is timed the
same way, and its gate is a genuine forward-pass cost, so it is expected to be
slower -- an honest cost to report next to the finding that it does not win.

**Training (read from disk, not re-measured).**  ``metrics_by_epoch.csv``
records ``train_seconds`` for every epoch of every completed run, so mean
seconds/epoch, total wall clock, and the epoch at which the best validation
soft-voted graph F1 occurred (convergence speed) are all recoverable exactly as
they happened, for both standard and dropout-aware runs.

Outputs
  results/runtime/runtime_inference.csv   (per architecture/variant/device)
  results/runtime/runtime_training.csv    (per architecture/variant, from disk)
  results/runtime/runtime_summary.json    (both, plus environment)
  paper/tables/runtime.tex                (the paper table)

Usage
  uv run python scripts/run_runtime_benchmark.py
  uv run python scripts/run_runtime_benchmark.py --repeats 100 --devices cuda cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from fault_localization.config.constants import WINDOW_LENGTH
from fault_localization.data.dataset import WindowGraphDataset
from scripts.run_dropout_study import load_trained_model

# Input feature count: 3 RMS phase voltages + 3 availability-mask channels.
NUM_FEATURES = 6

MODEL_LABEL = {
    "gru": "GRU",
    "rgcn": "RGCN",
    "rgsage_mean": "RGSAGE-Mean",
    "rgsage_max": "RGSAGE-Max",
    "rgatv2": "RGATv2",
    "sdr_stgnn": "SDR-STGNN",
}

# (label, results subdirectory, model directory, variant tag)
VARIANTS: list[tuple[str, str, str, str]] = [
    ("GRU", "fullscale_final", "gru", "standard"),
    ("RGCN", "fullscale_final", "rgcn", "standard"),
    ("RGSAGE-Mean", "fullscale_final", "rgsage_mean", "standard"),
    ("RGSAGE-Max", "fullscale_final", "rgsage_max", "standard"),
    ("RGATv2", "fullscale_final", "rgatv2", "standard"),
    ("RGSAGE-Max", "dropout_aware", "rgsage_max", "dropout-aware"),
    ("SDR-STGNN", "sdr_stgnn", "sdr_stgnn", "dropout-aware"),
]


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def time_forward(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_nodes: int,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    """Time ``repeats`` forward passes on a fixed synthetic batch.

    Returns per-batch latency statistics in milliseconds plus throughput in
    windows/s.  The same tensor is reused across repeats so that host-to-device
    transfer and allocation are excluded: what is measured is the forward pass,
    which is what the "no inference-time cost" claim is about.
    """
    model.eval()
    x = torch.randn(
        batch_size, num_nodes, WINDOW_LENGTH, NUM_FEATURES, device=device
    )
    # The three mask channels are 0/1 in the real data.  SDR-STGNN's gate reads
    # them, and a plausible availability pattern keeps its branch behaviour (and
    # therefore its timing) representative rather than degenerate.
    x[..., 3:6] = 1.0

    for _ in range(warmup):
        model(x)
    baseline_mb = float("nan")
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        # Everything resident before the timed forward: the model weights plus the
        # 32 MiB cuBLAS workspace that CUBLAS_WORKSPACE_CONFIG=":4096:8" (set for
        # deterministic kernels) reserves on first matmul.  Subtracting it isolates
        # the activation memory the forward pass itself needs.
        baseline_mb = torch.cuda.memory_allocated(device) / 2**20

    samples: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else float("nan")
    )
    mean_ms = statistics.fmean(samples)
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "latency_ms_median": statistics.median(samples),
        "latency_ms_p95": float(np.percentile(samples, 95)),
        "throughput_windows_per_s": batch_size / (mean_ms / 1e3),
        "peak_mem_mb": peak_mb,
        "baseline_mem_mb": baseline_mb,
        "forward_mem_mb": peak_mb - baseline_mb,
        "repeats": repeats,
    }


def training_stats(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate train_seconds / convergence epoch over a variant's seed runs.

    Read, not re-measured: these are the wall-clock numbers of the runs that
    produced the paper's results, on the T1000 8 GB described in Section
    "Experimental Setup".
    """
    per_seed_epoch_s: list[float] = []
    per_seed_total_h: list[float] = []
    per_seed_best_epoch: list[int] = []
    per_seed_epochs: list[int] = []
    for d in seed_dirs:
        df = pd.read_csv(d / "metrics_by_epoch.csv")
        per_seed_epoch_s.append(float(df["train_seconds"].mean()))
        per_seed_total_h.append(float(df["train_seconds"].sum()) / 3600.0)
        # Checkpoint selection is on validation soft-voted graph F1 (runner.py),
        # so the argmax epoch is the epoch the reported model came from.
        per_seed_best_epoch.append(int(df.loc[df["val_graph_f1"].idxmax(), "epoch"]))
        per_seed_epochs.append(int(len(df)))
    return {
        "n_seeds": len(seed_dirs),
        "epochs_run": per_seed_epochs[0] if per_seed_epochs else None,
        "sec_per_epoch_mean": statistics.fmean(per_seed_epoch_s),
        "sec_per_epoch_std": (
            statistics.stdev(per_seed_epoch_s) if len(per_seed_epoch_s) > 1 else 0.0
        ),
        "total_hours_mean": statistics.fmean(per_seed_total_h),
        "total_hours_std": (
            statistics.stdev(per_seed_total_h) if len(per_seed_total_h) > 1 else 0.0
        ),
        "best_epoch_mean": statistics.fmean(per_seed_best_epoch),
        "best_epoch_std": (
            statistics.stdev(per_seed_best_epoch) if len(per_seed_best_epoch) > 1 else 0.0
        ),
        "best_epoch_per_seed": per_seed_best_epoch,
        "sec_per_epoch_per_seed": per_seed_epoch_s,
    }


def build_latex(inf_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]],
                batch_size: int) -> str:
    """Runtime table: inference cost per architecture + training cost per variant."""
    train_by_key = {(r["model"], r["variant"]): r for r in train_rows}
    lines: list[str] = []
    for r in inf_rows:
        t = train_by_key.get((r["model_dir"], r["variant"]))
        epoch_min = f"{t['sec_per_epoch_mean']/60:.1f}" if t else "--"
        best_ep = f"{t['best_epoch_mean']:.1f}" if t else "--"
        total_h = f"{t['total_hours_mean']:.1f}" if t else "--"
        variant = "DA" if r["variant"] == "dropout-aware" else "Std"
        lines.append(
            f"    {r['label']} & {variant} & ${r['params']/1e3:.1f}$k "
            f"& ${r['latency_b1_ms_mean']:.2f}$ "
            f"& ${r['throughput_windows_per_s']:,.0f}$ "
            f"& ${r['forward_mem_mb']:.0f}$ "
            f"& ${epoch_min}$ & ${total_h}$ & ${best_ep}$ \\\\".replace(",", "{,}")
        )
    body = "\n".join(lines)
    return (
        "% Generated by scripts/run_runtime_benchmark.py -- do not edit by hand.\n"
        "% Source: results/runtime/runtime_summary.json\n"
        "\\begin{table*}[!tb]\n"
        "  \\centering\n"
        "  \\caption{Cost of each model and of the dropout-aware augmentation, on one\n"
        "           NVIDIA T1000 (8\\,GB). Inference is timed over synthetic batches of\n"
        "           the true input shape $[B,25,20,6]$: single-event latency at $B{=}1$\n"
        f"           and throughput at $B{{=}}{batch_size}$, with the activation memory the\n"
        "           forward pass allocates (peak minus resident weights and cuBLAS\n"
        "           workspace). Training columns are wall-clock numbers recorded by the\n"
        "           runs that produced the results (30 epochs, mean over seeds\n"
        "           42/43/44); ``best ep.'' is the epoch of highest validation\n"
        "           soft-voted graph F1, i.e.\\ the checkpoint actually reported.\n"
        "           Std\\ =\\ standard training, DA\\ =\\ dropout-aware.}\n"
        "  \\label{tab:runtime}\n"
        "  \\footnotesize\n"
        "  \\setlength{\\tabcolsep}{5pt}\n"
        "  \\begin{tabular}{@{}llccccccc@{}}\n"
        "    \\toprule\n"
        "    \\multirow{2}{*}{Model} & \\multirow{2}{*}{Train.}\n"
        "      & \\multirow{2}{*}{Params}\n"
        "      & \\multicolumn{3}{c}{Inference} & \\multicolumn{3}{c}{Training} \\\\\n"
        "    \\cmidrule(lr){4-6}\\cmidrule(l){7-9}\n"
        "      & & & Lat. (ms) & Windows/s & Act.\\ mem. (MiB)\n"
        "      & min/epoch & Total (h) & Best ep. \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table*}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--data-root", default="dataset_fullscale_50")
    ap.add_argument("--output-dir", default="results/runtime")
    ap.add_argument("--tex-out", default="paper/tables/runtime.tex")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--timing-seed",
        type=int,
        default=42,
        help="Which seed's checkpoint to time (architecture, not weights, sets latency).",
    )
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Throughput batch size; 32 is the evaluation batch size.")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--devices", nargs="+", default=["cuda"])
    args = ap.parse_args()

    root = Path(args.results_root)

    # The graph is a property of the dataset, and every model was built on it.
    # Load the test split only: it is the smallest, and nothing but edge_index
    # and the node count is needed here.
    ds = WindowGraphDataset(Path(args.data_root), split="test")
    edge_index = ds.edge_index
    num_nodes = int(np.max(edge_index) + 1) if np.size(edge_index) else 25

    inf_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    for label, sub, model_dir, variant in VARIANTS:
        base = root / sub / model_dir
        seed_dirs = [
            base / f"seed_{s:05d}" for s in args.seeds
            if (base / f"seed_{s:05d}" / "metrics_by_epoch.csv").exists()
        ]
        if not seed_dirs:
            print(f"[skip] {label} ({variant}): no completed runs under {base}")
            continue

        tstats = training_stats(seed_dirs)
        tstats.update({"label": label, "model": model_dir, "variant": variant})
        train_rows.append(tstats)

        timing_dir = base / f"seed_{args.timing_seed:05d}"
        if not (timing_dir / "checkpoint_best.pt").exists():
            timing_dir = seed_dirs[0]

        for dev_name in args.devices:
            device = torch.device(dev_name)
            model, _ = load_trained_model(timing_dir, edge_index, device)
            total, trainable = count_params(model)
            b1 = time_forward(model, device, 1, num_nodes, args.repeats, args.warmup)
            bn = time_forward(
                model, device, args.batch_size, num_nodes, args.repeats, args.warmup
            )
            row = {
                "label": label,
                "model_dir": model_dir,
                "variant": variant,
                "device": dev_name,
                "checkpoint": str(timing_dir),
                "params": total,
                "params_trainable": trainable,
                "batch_size_throughput": args.batch_size,
                "latency_b1_ms_mean": b1["latency_ms_mean"],
                "latency_b1_ms_std": b1["latency_ms_std"],
                "latency_b1_ms_p95": b1["latency_ms_p95"],
                "latency_bN_ms_mean": bn["latency_ms_mean"],
                "latency_bN_ms_std": bn["latency_ms_std"],
                "throughput_windows_per_s": bn["throughput_windows_per_s"],
                "peak_mem_mb": bn["peak_mem_mb"],
                "forward_mem_mb": bn["forward_mem_mb"],
                "peak_mem_mb_b1": b1["peak_mem_mb"],
                "forward_mem_mb_b1": b1["forward_mem_mb"],
                "weight_mem_mb": total * 4 / 2**20,
                "repeats": args.repeats,
            }
            inf_rows.append(row)
            print(
                f"[{dev_name}] {label:12s} {variant:13s} "
                f"params={total/1e3:6.1f}k  "
                f"lat@1={b1['latency_ms_mean']:6.2f}±{b1['latency_ms_std']:.2f} ms  "
                f"thr@{args.batch_size}={bn['throughput_windows_per_s']:9.0f} w/s  "
                f"fwd_mem={bn['forward_mem_mb']:6.1f} MiB  "
                f"train={tstats['sec_per_epoch_mean']/60:5.1f} min/ep "
                f"({tstats['total_hours_mean']:4.1f} h)  "
                f"best_ep={tstats['best_epoch_mean']:.1f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inf_rows).to_csv(out_dir / "runtime_inference.csv", index=False)
    pd.DataFrame(train_rows).to_csv(out_dir / "runtime_training.csv", index=False)

    # The headline overhead numbers the paper quotes, derived here so the text
    # never has to recompute a ratio by hand.
    def find_inf(model_dir: str, variant: str, dev: str = "cuda") -> dict | None:
        for r in inf_rows:
            if (r["model_dir"], r["variant"], r["device"]) == (model_dir, variant, dev):
                return r
        return None

    def find_train(model_dir: str, variant: str) -> dict | None:
        for r in train_rows:
            if (r["model"], r["variant"]) == (model_dir, variant):
                return r
        return None

    derived: dict[str, Any] = {}
    std_i, da_i = find_inf("rgsage_max", "standard"), find_inf("rgsage_max", "dropout-aware")
    if std_i and da_i:
        derived["rgsage_max_da_vs_std_latency_ratio"] = (
            da_i["latency_b1_ms_mean"] / std_i["latency_b1_ms_mean"]
        )
        derived["rgsage_max_da_vs_std_params_equal"] = std_i["params"] == da_i["params"]
    std_t, da_t = find_train("rgsage_max", "standard"), find_train("rgsage_max", "dropout-aware")
    if std_t and da_t:
        derived["rgsage_max_da_vs_std_train_overhead_pct"] = 100.0 * (
            da_t["sec_per_epoch_mean"] / std_t["sec_per_epoch_mean"] - 1.0
        )
    sdr_i = find_inf("sdr_stgnn", "dropout-aware")
    if sdr_i and da_i:
        derived["sdr_vs_da_latency_ratio"] = (
            sdr_i["latency_b1_ms_mean"] / da_i["latency_b1_ms_mean"]
        )
        derived["sdr_vs_da_throughput_ratio"] = (
            sdr_i["throughput_windows_per_s"] / da_i["throughput_windows_per_s"]
        )
        derived["sdr_extra_params"] = sdr_i["params"] - da_i["params"]

    summary = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "protocol": {
            "input_shape": [None, num_nodes, WINDOW_LENGTH, NUM_FEATURES],
            "synthetic_input": True,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "throughput_batch_size": args.batch_size,
            "timing_seed": args.timing_seed,
            "training_numbers_source": "metrics_by_epoch.csv train_seconds (recorded, not re-measured)",
        },
        "inference": inf_rows,
        "training": train_rows,
        "derived": derived,
    }
    (out_dir / "runtime_summary.json").write_text(json.dumps(summary, indent=2))

    cuda_rows = [r for r in inf_rows if r["device"] == "cuda"]
    tex_path = Path(args.tex_out)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(build_latex(cuda_rows or inf_rows, train_rows, args.batch_size))

    print(f"\nWrote {out_dir/'runtime_inference.csv'}")
    print(f"Wrote {out_dir/'runtime_training.csv'}")
    print(f"Wrote {out_dir/'runtime_summary.json'}")
    print(f"Wrote {tex_path}")
    print("\nDerived:")
    for k, v in derived.items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
