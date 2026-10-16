"""Audit, 2026-09-29 (author review of S2.4/S2.5's writeup): verify S1.7's 0.8977 (training
split) and S2.4's 0.772 (validation split) are directly comparable before that "reversal" is
reported as a real train->val generalization result.

They are NOT the same metric. ``check_s1_7_discretization.py`` scores an **event-level, single
settled-step, 25-class, fault-only** task (no true no-fault class exists at that granularity, per
its own docstring). ``run_psm_full_eval.py`` scores a **window-level, 40-windows-per-event,
26-class** task via ``WindowGraphDataset`` (half the windows per event are pre-fault, label 0).

This script runs both metrics on both splits (2x2) so the "reversal" can be decomposed into a
metric effect vs a genuine split effect, holding the other fixed each time.

Usage:
    uv run python scripts/check_s2_4_metric_consistency.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_BUSES, WINDOW_LENGTH  # noqa: E402
from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.physics import observation as OBS  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402
from fault_localization.physics.residual import flatten_table  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452
SETTLED_STEP = 45
DATA_ROOT = REPO / "dataset_fullscale_50"
OUT_DIR = REPO / "results/pcstgnn/audit/s2_4_metric_check"


def load_event_paths(split: str) -> list[Path]:
    stems = set()
    with (DATA_ROOT / f"splits/{split}.txt").open() as fh:
        for line in fh:
            stems.add(line.strip().rsplit("_w", 1)[0])
    raw_dir = DATA_ROOT / "raw_sim"
    return sorted(p for p in raw_dir.glob("*.npz") if p.stem in stems)


def s1_7_style_eval(split: str, table: torch.Tensor, device: torch.device) -> dict:
    """Event-level, single settled-step, 25-class fault-only -- check_s1_7_discretization.py's
    exact protocol, applied to whichever split is passed in."""
    paths = load_event_paths(split)
    s, a, y = [], [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as loaded:
            rec = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        s.append(rec[:, SETTLED_STEP, :3])
        a.append((rec[:, 0, 3:] > 0.5).astype(np.float32))
        y.append(FAULT_BUSES.index(meta["fault_bus"]) + 1)
    s_t = torch.tensor(np.stack(s), dtype=torch.float32).to(device)
    a_t = torch.tensor(np.stack(a), dtype=torch.float32).to(device)
    y_a = np.asarray(y)

    preds = np.empty(len(y_a), dtype=np.int64)
    chunk_b = 200
    for b0 in range(0, len(y_a), chunk_b):
        b1 = min(b0 + chunk_b, len(y_a))
        rho, _hyp = R.rho(s_t[b0:b1], a_t[b0:b1], table, CANONICAL_SIGMA_EPS, chunk_f=2)
        preds[b0:b1] = R.psm_predict(rho).cpu().numpy()
    labels25 = list(range(1, len(FAULT_BUSES) + 1))
    f1_25 = f1_score(y_a, preds, labels=labels25, average="macro", zero_division=0)
    top1 = float((preds == y_a).mean())
    return {"protocol": "event_level_settled_step_25class_faultonly", "split": split,
            "n_events": int(len(y_a)), "macro_f1_25class": float(f1_25), "top1_acc": top1}


def s2_4_style_eval(split: str, table: torch.Tensor, device: torch.device,
                     mean: np.ndarray, std: np.ndarray, batch_size: int = 4096) -> dict:
    """Window-level, 40-windows-per-event, 26-class -- run_psm_full_eval.py's exact k=0
    protocol, applied to whichever split is passed in."""
    ds = WindowGraphDataset(str(DATA_ROOT), split)
    loader = make_dataloader(ds, batch_size, shuffle=False)
    preds, trues = [], []
    for bi, batch in enumerate(loader):
        x, y = batch["x"].to(device), batch["y"].to(device)
        s, a = OBS.observe(x, mean, std, reduce="last")
        s, a = s.double(), a.double()
        rho, _hyp = R.rho(s, a, table, CANONICAL_SIGMA_EPS, sigma0=0.0, method="gram")
        pred = R.psm_predict(rho)
        preds.extend(pred.tolist()); trues.extend(y.tolist())
        if bi % 30 == 0:
            print(f"  [{split} s2.4-style] batch {bi}/{len(loader)}", flush=True)
    labels26 = list(range(26))
    f1_26 = f1_score(trues, preds, average="macro", labels=labels26, zero_division=0)
    # Also report the 25-class-only average on this same window population, for the
    # metric-holding-population-fixed comparison.
    labels25 = list(range(1, 26))
    f1_25_same_pop = f1_score(trues, preds, average="macro", labels=labels25, zero_division=0)
    return {"protocol": "window_level_40perevent_26class", "split": split,
            "n_windows": len(trues), "macro_f1_26class": float(f1_26),
            "macro_f1_25class_same_population": float(f1_25_same_pop)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    tbl = T.load_table(REPO / "operator")
    table = flatten_table(tbl["mu"], tbl["mu0"])
    norm_stats = json.loads((DATA_ROOT / "processed/norm_stats.json").read_text())
    mean, std = np.asarray(norm_stats["mean"]), np.asarray(norm_stats["std"])
    print(f"[load] table {tuple(table.shape)} device={device}", flush=True)

    results = {}
    for split in ["train", "val"]:
        print(f"\n=== {split} ===", flush=True)
        r1 = s1_7_style_eval(split, table, device)
        print(f"  s1.7-style: {r1}", flush=True)
        r2 = s2_4_style_eval(split, table, device, mean, std)
        print(f"  s2.4-style: {r2}", flush=True)
        results[split] = {"s1_7_style": r1, "s2_4_style": r2}

    results["wall_seconds"] = time.time() - t0
    (OUT_DIR / "metric_consistency_check.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'metric_consistency_check.json'} ({results['wall_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
