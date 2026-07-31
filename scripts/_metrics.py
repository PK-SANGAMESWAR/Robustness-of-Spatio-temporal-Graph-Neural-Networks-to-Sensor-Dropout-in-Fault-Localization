"""Shared per-run metric extraction, used by both build_trend_summary.py and
build_fullscale_summary.py so the two summary scripts compute the same metrics
the same way (graph macro-F1, per-window macro-F1, per-resistance F1, no-fault
precision/recall) from each run's metrics.json + predictions_test.csv.

Metric mapping notes (this codebase's existing evaluate_loader/reporting
naming, reused unchanged):
  - graph_macro_f1     = metrics.json["test_graph"]["f1"]           (soft-voted, one prediction per window)
  - perwindow_macro_f1 = metrics.json["test_node_diagnostic"]["f1"] (per-node diagnostic, existing "secondary" metric)
  - f1_r0p1/f1_r1/f1_r10, nofault_precision/recall are computed here from predictions_test.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

LABELS = list(range(26))


def resistance_f1(df: pd.DataFrame, resistance: float) -> float:
    sub = df[np.isclose(df["resistance"], resistance)]
    if sub.empty:
        return float("nan")
    _, _, f1, _ = precision_recall_fscore_support(
        sub["truth"], sub["vote_prediction"], labels=LABELS, average="macro", zero_division=0
    )
    return float(f1)


def nofault_precision_recall(df: pd.DataFrame) -> tuple[float, float]:
    p, r, _, _ = precision_recall_fscore_support(
        df["truth"], df["vote_prediction"], labels=[0], average=None, zero_division=0
    )
    return float(p[0]), float(r[0])


def row_for_run(scale: str, runs_per_scenario: int, model: str, seed_dir: Path) -> dict | None:
    metrics_path = seed_dir / "metrics.json"
    preds_path = seed_dir / "predictions_test.csv"
    config_path = seed_dir / "config.yaml"
    if not metrics_path.exists() or not preds_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text())
    preds = pd.read_csv(preds_path)

    edges_enabled = True
    if config_path.exists():
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        edges_enabled = not cfg.get("ablation", {}).get("disable_edges", False)

    nf_p, nf_r = nofault_precision_recall(preds)
    seed = int(seed_dir.name.replace("seed_", ""))

    return {
        "scale": scale,
        "runs_per_scenario": runs_per_scenario,
        "model": model,
        "seed": seed,
        "graph_macro_f1": metrics["test_graph"]["f1"],
        "perwindow_macro_f1": metrics["test_node_diagnostic"]["f1"],
        "f1_r0p1": resistance_f1(preds, 0.1),
        "f1_r1": resistance_f1(preds, 1.0),
        "f1_r10": resistance_f1(preds, 10.0),
        "nofault_precision": nf_p,
        "nofault_recall": nf_r,
        "edges_enabled": edges_enabled,
    }


def collect_rows(scales: dict[str, tuple[str, int]], results_root: Path = Path("results")) -> pd.DataFrame:
    """Collect one row per (scale, model, seed) for every completed run.

    `scales` maps a scale label -> (benchmark_version dir name under
    `results_root`, runs_per_scenario). Any scale whose directory doesn't
    exist yet is silently skipped (sweeps are resumable/in-progress-friendly).
    """
    rows: list[dict] = []
    for scale, (benchmark_dir, runs_per_scenario) in scales.items():
        root = results_root / benchmark_dir
        if not root.exists():
            continue
        for model_dir in sorted(root.glob("*")):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for seed_dir in sorted(model_dir.glob("seed_*")):
                row = row_for_run(scale, runs_per_scenario, model, seed_dir)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)
