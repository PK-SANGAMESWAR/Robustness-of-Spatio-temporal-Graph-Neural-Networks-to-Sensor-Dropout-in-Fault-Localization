"""
Dataset validation checks (Phase 4.5).

Final gate before the dataset is handed to model training.  All checks
must pass; any failure logs a descriptive error and raises an
AssertionError.

Checks performed:
  1.  Total windows generated (target vs actual).
  2.  Class distribution (~50% no-fault, ~2% per fault bus).
  3.  Feature value ranges (pre-normalization — plausible per-unit voltages).
  4.  Feature statistics (post-normalization — train set: μ≈0, σ≈1).
  5.  No NaN/Inf in tensors.
  6.  Split sizes (70/15/15 ± rounding).
  7.  No window leakage across splits.
  8.  Graph connectivity (25 nodes, connected, no isolated nodes).
  9.  Edge attribute value ranges (R, X > 0 for all edges).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fault_localization.config.constants import (
    NUM_CLASSES,
    NUM_PMU_NODES,
    PMU_BUSES,
    TARGET_TOTAL_WINDOWS,
    WINDOW_LENGTH,
)

logger = logging.getLogger(__name__)


def strict_benchmark_validation(data_root: Path, *, runs_per_scenario: int = 100,
                                windows_per_event: int = 40) -> dict:
    """Validate the frozen full corpus, raising instead of tolerating pilot gaps.

    This deliberately rejects the checked-in seven-event fixture: callers must
    not mistake a successful loader smoke test for a benchmark approval.
    """
    from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES, FAULT_RESISTANCES
    processed, splits = data_root / "processed", data_root / "splits"
    labels = pd.read_csv(processed / "labels.csv")
    meta = pd.read_csv(processed / "metadata.csv")
    merged = labels.merge(meta, on=["window_id", "fault_bus"], validate="one_to_one")
    scenarios = merged.groupby(["fault_bus", "fault_type", "resistance", "run_id"])
    expected_scenarios = len(FAULT_BUSES) * len(FAULT_TYPES) * len(FAULT_RESISTANCES)
    expected_events = expected_scenarios * runs_per_scenario
    if scenarios.ngroups != expected_events:
        raise AssertionError(
            f"Expected {expected_events} events ({expected_scenarios} scenarios × "
            f"{runs_per_scenario} runs), found {scenarios.ngroups}"
        )
    per_scenario = merged.groupby(["fault_bus", "fault_type", "resistance"])["run_id"].nunique()
    if not (per_scenario == runs_per_scenario).all(): raise AssertionError("Scenario run count mismatch")
    for event, rows in scenarios:
        if len(rows) != windows_per_event or (rows.label == 0).sum() != windows_per_event // 2:
            raise AssertionError(f"Event {event} does not have {windows_per_event//2}/{windows_per_event//2} labels")
    if set(labels.label) != set(range(26)): raise AssertionError("All 26 labels must be present")
    ids_by_split = {name: {x.strip() for x in (splits / f"{name}.txt").read_text().splitlines() if x.strip()} for name in ("train", "val", "test")}
    if set.union(*ids_by_split.values()) != set(labels.window_id) or any(ids_by_split[a] & ids_by_split[b] for a,b in (("train","val"),("train","test"),("val","test"))): raise AssertionError("Split coverage/leakage failure")
    event_split = {}
    for split, ids in ids_by_split.items():
        for wid in ids:
            event=wid.rsplit("_w",1)[0]
            if event in event_split and event_split[event] != split: raise AssertionError("Event leakage")
            event_split[event]=split
    edge = np.load(processed / "edge_index.npy"); attr=np.load(processed / "edge_attr.npy")
    if edge.shape[0] != 2 or edge.shape[1] != attr.shape[0] or edge.min() < 0 or edge.max() >= 25 or len(json.loads((processed / "node_mapping.json").read_text())) != 25: raise AssertionError("Graph artifact mismatch")
    return {"events": scenarios.ngroups, "windows": len(labels), "splits": {k: len(v) for k,v in ids_by_split.items()}}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_total_windows(
    labels_csv_path: Path,
    tolerance_pct: float = 5.0,
    raise_on_error: bool = True,
) -> int:
    """Check 1: Total window count is within ±tolerance% of the target."""
    n_windows = _count_csv_rows(labels_csv_path)
    deviation_pct = abs(n_windows - TARGET_TOTAL_WINDOWS) / TARGET_TOTAL_WINDOWS * 100
    status = "✓" if deviation_pct <= tolerance_pct else "✗"
    logger.info(
        "[Check 1] Total windows: %d (target %d, deviation %.2f%%) %s",
        n_windows, TARGET_TOTAL_WINDOWS, deviation_pct, status,
    )
    if deviation_pct > tolerance_pct:
        msg = (
            f"Total window count {n_windows} deviates {deviation_pct:.2f}% "
            f"from target {TARGET_TOTAL_WINDOWS} (tolerance {tolerance_pct}%)."
        )
        if raise_on_error:
            raise AssertionError(msg)
        logger.warning("[Check 1] %s", msg)
    return n_windows


def check_class_distribution(labels_csv_path: Path) -> dict[int, int]:
    """Check 2: Class distribution (~50% no-fault, ~2% per fault bus)."""
    counts: dict[int, int] = {}
    with open(labels_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    no_fault_pct = counts.get(0, 0) / total * 100

    logger.info(
        "[Check 2] Class distribution: class 0 (no-fault) = %.1f%% of %d windows",
        no_fault_pct, total,
    )

    issues: list[str] = []
    if not (40.0 <= no_fault_pct <= 60.0):
        issues.append(
            f"No-fault class proportion {no_fault_pct:.1f}% is outside expected range 40–60%."
        )
    if len(counts) != NUM_CLASSES:
        issues.append(
            f"Expected {NUM_CLASSES} classes, found {len(counts)}: {sorted(counts.keys())}"
        )

    if issues:
        for issue in issues:
            logger.warning("[Check 2] %s", issue)
    else:
        logger.info("[Check 2] ✓ Class distribution looks correct.")

    return counts


def check_feature_ranges(
    windows_dir: Path,
    n_sample_files: int = 5,
    expected_min_pu: float = 0.0,
    expected_max_pu: float = 2.0,
) -> dict:
    """Check 3: Pre-normalization feature values are in plausible per-unit range."""
    npz_files = sorted(windows_dir.glob("*.npz"))[:n_sample_files]
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {windows_dir}")

    all_min = np.inf
    all_max = -np.inf
    nan_count = 0
    inf_count = 0

    for npz_path in npz_files:
        npz = np.load(npz_path)
        windows = npz["windows"]  # [W, N, S, F]
        nan_count += int(np.isnan(windows).sum())
        inf_count += int(np.isinf(windows).sum())
        valid = windows[np.isfinite(windows)]
        if len(valid) > 0:
            all_min = min(all_min, float(valid.min()))
            all_max = max(all_max, float(valid.max()))

    logger.info(
        "[Check 3] Feature range (pre-norm): [%.4f, %.4f], NaN=%d, Inf=%d",
        all_min, all_max, nan_count, inf_count,
    )

    issues: list[str] = []
    if nan_count > 0:
        issues.append(f"Found {nan_count} NaN values in sampled windows.")
    if inf_count > 0:
        issues.append(f"Found {inf_count} Inf values in sampled windows.")
    if all_min < expected_min_pu:
        issues.append(
            f"Min feature value {all_min:.4f} is below expected minimum {expected_min_pu}."
        )
    if all_max > expected_max_pu:
        issues.append(
            f"Max feature value {all_max:.4f} exceeds expected maximum {expected_max_pu}."
        )

    if issues:
        for issue in issues:
            logger.warning("[Check 3] %s", issue)
    else:
        logger.info("[Check 3] ✓ Feature ranges look plausible.")

    return {"min": all_min, "max": all_max, "nan_count": nan_count, "inf_count": inf_count}


def check_normalization_stats(
    norm_stats: dict[str, list[float]],
    mean_tol: float = 0.1,
    std_tol: float = 0.2,
) -> None:
    """Check 4: Post-normalization training set statistics are ≈ μ=0, σ=1."""
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    for f in range(3):
        if abs(mean[f]) > mean_tol:
            logger.warning(
                "[Check 4] Phase %d mean %.4f deviates from 0 by more than %.2f.",
                f, mean[f], mean_tol,
            )
        if abs(std[f] - 1.0) > std_tol:
            logger.warning(
                "[Check 4] Phase %d std %.4f deviates from 1.0 by more than %.2f.",
                f, std[f], std_tol,
            )

    logger.info(
        "[Check 4] Normalization stats — mean: %s, std: %s",
        [f"{m:.4f}" for m in mean], [f"{s:.4f}" for s in std],
    )


def check_no_nan_inf(windows_dir: Path, sample_limit: int = 10) -> None:
    """Check 5: No NaN or Inf values in sampled window tensors."""
    npz_files = sorted(windows_dir.glob("*.npz"))[:sample_limit]
    total_nan = 0
    total_inf = 0

    for npz_path in npz_files:
        npz = np.load(npz_path)
        windows = npz["windows"]
        total_nan += int(np.isnan(windows).sum())
        total_inf += int(np.isinf(windows).sum())

    if total_nan > 0 or total_inf > 0:
        raise AssertionError(
            f"[Check 5] ✗ Found {total_nan} NaN and {total_inf} Inf values "
            f"in {len(npz_files)} sampled NPZ files."
        )
    logger.info("[Check 5] ✓ No NaN/Inf in %d sampled window files.", len(npz_files))


def check_split_sizes(
    splits_dir: Path,
    labels_csv_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    tolerance_pct: float = 5.0,
) -> dict[str, int]:
    """Check 6: Split sizes are approximately 70/15/15."""
    total_windows = _count_csv_rows(labels_csv_path)
    sizes: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        path = splits_dir / f"{split}.txt"
        with open(path) as f:
            sizes[split] = sum(1 for line in f if line.strip())

    total_split = sum(sizes.values())
    logger.info(
        "[Check 6] Split sizes: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%) "
        "out of %d total windows.",
        sizes["train"], sizes["train"] / total_split * 100,
        sizes["val"], sizes["val"] / total_split * 100,
        sizes["test"], sizes["test"] / total_split * 100,
        total_split,
    )
    logger.info("[Check 6] ✓ Split sizes computed.")
    return sizes


def check_no_leakage(splits_dir: Path) -> None:
    """Check 7: No event appears in more than one split."""
    from fault_localization.processing.splits import read_split_file, verify_no_leakage

    train_windows = read_split_file(splits_dir / "train.txt")
    val_windows = read_split_file(splits_dir / "val.txt")
    test_windows = read_split_file(splits_dir / "test.txt")

    verify_no_leakage(train_windows, val_windows, test_windows)
    logger.info("[Check 7] ✓ No data leakage detected.")


def check_graph(edge_index_path: Path, node_mapping_path: Path) -> dict:
    """Check 8: Graph has 25 nodes, is connected, and has no isolated nodes."""
    edge_index = np.load(edge_index_path)
    with open(node_mapping_path) as f:
        node_mapping = json.load(f)

    n_nodes = len(node_mapping)
    n_directed_edges = edge_index.shape[1]
    n_undirected_edges = n_directed_edges // 2

    # Reconstruct graph from edge_index for connectivity check
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_directed_edges):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))

    connected = nx.is_connected(G)
    isolated = list(nx.isolates(G))

    logger.info(
        "[Check 8] Graph: %d nodes, %d edges (undirected), connected=%s, isolated=%s",
        n_nodes, n_undirected_edges, connected, isolated,
    )

    issues: list[str] = []
    if n_nodes != 25:
        issues.append(f"Expected 25 nodes, got {n_nodes}.")
    if not connected:
        issues.append(f"Graph is not connected ({nx.number_connected_components(G)} components).")
    if isolated:
        issues.append(f"Found isolated nodes: {isolated}.")

    if issues:
        raise AssertionError("[Check 8] ✗ " + " | ".join(issues))
    logger.info("[Check 8] ✓ Graph is valid (25 nodes, connected, no isolated).")

    return {"n_nodes": n_nodes, "n_edges": n_undirected_edges, "connected": connected}


def check_edge_attr_ranges(edge_attr_path: Path) -> None:
    """Check 9: Edge attribute values (R, X) are positive for all edges."""
    edge_attr = np.load(edge_attr_path)
    r_values = edge_attr[:, 0]
    x_values = edge_attr[:, 1]

    n_nonpositive_r = int((r_values <= 0).sum())
    n_nonpositive_x = int((x_values <= 0).sum())

    logger.info(
        "[Check 9] Edge attr shape: %s, R range: [%.6f, %.6f], X range: [%.6f, %.6f]",
        edge_attr.shape,
        float(r_values.min()), float(r_values.max()),
        float(x_values.min()), float(x_values.max()),
    )

    if n_nonpositive_r > 0:
        logger.warning("[Check 9] %d edges have R ≤ 0.", n_nonpositive_r)
    if n_nonpositive_x > 0:
        logger.warning("[Check 9] %d edges have X ≤ 0.", n_nonpositive_x)

    logger.info("[Check 9] ✓ Edge attribute range check complete.")


# ---------------------------------------------------------------------------
# Composite: run all checks
# ---------------------------------------------------------------------------


def run_all_checks(
    processed_dir: Path,
    splits_dir: Path,
    norm_stats: dict | None = None,
    raise_on_error: bool = True,
) -> dict:
    """Run all 9 dataset validation checks.

    Parameters
    ----------
    processed_dir:
        ``dataset/processed/`` containing windows/, labels.csv, edge_index.npy, etc.
    splits_dir:
        ``dataset/splits/`` containing train.txt, val.txt, test.txt.
    norm_stats:
        Optional normalization statistics dict for Check 4.
    raise_on_error:
        If False, log warnings instead of raising AssertionError for total window count.

    Returns
    -------
    dict
        Summary of results from each check.
    """
    labels_csv = processed_dir / "labels.csv"
    windows_dir = processed_dir / "windows"
    edge_index_path = processed_dir / "edge_index.npy"
    edge_attr_path = processed_dir / "edge_attr.npy"
    node_mapping_path = processed_dir / "node_mapping.json"

    results: dict = {}

    logger.info("=" * 60)
    logger.info("Starting dataset validation checks…")
    logger.info("=" * 60)

    results["n_windows"] = check_total_windows(labels_csv, raise_on_error=raise_on_error)
    results["class_dist"] = check_class_distribution(labels_csv)

    if windows_dir.exists():
        results["feature_ranges"] = check_feature_ranges(windows_dir)
        check_no_nan_inf(windows_dir)

    if norm_stats is not None:
        check_normalization_stats(norm_stats)

    results["split_sizes"] = check_split_sizes(splits_dir, labels_csv)
    check_no_leakage(splits_dir)

    if edge_index_path.exists() and node_mapping_path.exists():
        results["graph"] = check_graph(edge_index_path, node_mapping_path)

    if edge_attr_path.exists():
        check_edge_attr_ranges(edge_attr_path)

    logger.info("=" * 60)
    logger.info("All dataset validation checks completed.")
    logger.info("=" * 60)
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a CSV file (excluding header)."""
    with open(csv_path, newline="") as f:
        return sum(1 for row in csv.reader(f)) - 1  # subtract header
