"""T12 — computational cost (``08_STAGE_G_eval.md`` S6.4, ``11_METRICS.md`` § 7).

Four columns, from three sources:

- **parameters**: counted on the model rebuilt from its own ``config.yaml`` (CPU);
- **FLOPs per window**: the learned backbone with ``torch.utils.flop_counter.FlopCounterMode`` (matrix products; one
  multiply-add = 2 FLOPs), plus the residual counted from its code: ``physics/residual.py::raw_residual`` ("direct",
  the method every run used) forms, for every table entry, one difference, one square, one weight product and one
  sum, so it costs ``4 * table.numel()`` FLOPs per window (the min-reduction and the fusion add are below 0.1 %);
- **training GPU-hours**: the sum of ``train_seconds`` in each run's ``metrics_by_epoch.csv`` (read, not re-measured);
- **inference latency**: measured only on a quiet machine (``12_CHECKLISTS.md`` § 0); ``quiet_reasons`` lists why a
  machine is not quiet, and the latency step refuses to run while that list is not empty.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RESIDUAL_FLOPS_PER_ENTRY = 4

#: (label, run dirs relative to REPO, uses the residual at inference)
COST_MODELS = (
    ("R0ⁿ", [f"results/s4_4/r0n/pcstgnn/seed_{s:05d}" for s in (42, 43, 44)]),
    ("R5 (PC-STGNN)", [f"results/s4_4/r5/pcstgnn/seed_{s:05d}" for s in (42, 43, 44)]),
    ("R2 (learned decoder)", [f"results/s4_4/r2/pcstgnn/seed_{s:05d}" for s in (42, 43, 44)]),
    ("R8 (no noise aug.)", ["results/s4_4/r8/pcstgnn/seed_00042"]),
    ("B2ᵃ (regressed R̂)", ["results/stage_g/train/b2a/pcstgnn/seed_00042"]),
    ("R0 (Paper 1)", [f"results/dropout_aware/rgsage_max/seed_{s:05d}" for s in (42, 43, 44)]),
)


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def backbone_flops(model: torch.nn.Module, n_nodes: int = 25, window: int = 20, feats: int = 6) -> int:
    """FLOPs of one window through the learned part (``network_logits`` when the model has a table)."""
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.zeros(1, n_nodes, window, feats)
    x[..., 3:] = 1.0
    fn = model.network_logits if hasattr(model, "network_logits") else model
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        fn(x)
    return int(fc.get_total_flops())


def residual_flops(model: torch.nn.Module) -> int:
    """``4 * table.numel()`` when the model computes the residual at inference, else 0."""
    table = getattr(model, "phys_table", None)
    if table is None or table.numel() == 0:
        return 0
    return RESIDUAL_FLOPS_PER_ENTRY * int(table.numel())


def training_hours(run_dir: Path) -> dict:
    """Sum of ``train_seconds`` over the epochs of one run, and the best-validation epoch."""
    rows = list(csv.DictReader((Path(run_dir) / "metrics_by_epoch.csv").open()))
    secs = [float(r["train_seconds"]) for r in rows]
    f1 = [float(r["val_graph_f1"]) for r in rows]
    return {"epochs": len(rows), "hours": sum(secs) / 3600.0, "sec_per_epoch": statistics.fmean(secs),
            "best_epoch": int(rows[int(np.argmax(f1))]["epoch"])}


def scoring_rate_ms(run_dir: Path) -> float | None:
    """Wall-clock ms per window-forward of the full validation grid (60 forwards; data loading included)."""
    p = Path(run_dir) / "scoring_full_grid.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    n_fwd = len(d["cells"]) * int(d.get("n_mask_seeds", 3))
    return 1e3 * float(d["scoring_wallclock_seconds"]) / (int(d["n_windows"]) * n_fwd)


# ------------------------------------------------------------------------------------------
# Quiet-machine check and latency
# ------------------------------------------------------------------------------------------


def quiet_reasons(repo: Path = REPO, max_load: float = 1.5) -> list[str]:
    """Why this machine is not quiet; empty list = quiet."""
    bad = []
    ood = repo / "results/stage_g/ood"
    if (ood / "STARTED.json").exists() and not (ood / "COMPLETED.json").exists() and not (ood / "FAILED.json").exists():
        bad.append("the sealed OOD run has started and not completed")
    for lock in ("results/s4_4/queue.lock", "results/stage_f/gtf2/watcher.lock", "results/stage_g/queue.lock"):
        p = repo / lock
        if p.exists():
            try:
                os.kill(int(p.read_text().split()[0]), 0)
                bad.append(f"{lock} is held by a live process")
            except (ValueError, IndexError, ProcessLookupError):
                pass
            except PermissionError:
                bad.append(f"{lock} is held by a live process")
    load1 = os.getloadavg()[0]
    if load1 > max_load:
        bad.append(f"1-min load average {load1:.2f} > {max_load}")
    return bad


@torch.no_grad()
def latency(model: torch.nn.Module, device: torch.device, batch_size: int, repeats: int = 50, warmup: int = 10) -> dict:
    """Forward latency on a fixed synthetic batch ``[B, 25, 20, 6]`` (mask channels 1); fusion included for R5."""
    x = torch.randn(batch_size, 25, 20, 6, device=device)
    x[..., 3:] = 1.0
    model.eval()
    for _ in range(warmup):
        model(x)
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"batch_size": batch_size, "ms_mean": statistics.fmean(samples), "ms_std": statistics.stdev(samples),
            "ms_p95": float(np.percentile(samples, 95)), "windows_per_s": batch_size / (statistics.fmean(samples) / 1e3),
            "repeats": repeats}


def load_any(run_dir: Path, device) -> torch.nn.Module:
    """Rebuild a PC-STGNN run (Stage G loader) or a Paper-1 run (``build_model_from_config``)."""
    from fault_localization.stage_g import scoring as S

    ei = torch.from_numpy(np.load(REPO / "dataset_fullscale_50/processed/edge_index.npy")).long()
    model, _cfg = S.load_model(REPO / run_dir, ei, device)
    return model
