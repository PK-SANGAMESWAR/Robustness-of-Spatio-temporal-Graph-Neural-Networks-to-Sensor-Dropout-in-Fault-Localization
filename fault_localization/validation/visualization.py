"""
Visualization outputs for dataset validation (Phase 4.5).

All figures are saved to ``dataset/validation/``.

Outputs:
  1. voltage_sag_{event_id}.png    — 3–5 representative fault events,
                                     3-phase voltage across 60 ms.
  2. fault_signature_comparison.png — overlay voltage profiles for
                                      different fault types at same bus.
  3. class_distribution.png        — bar chart across all 26 classes.
  4. feature_distribution.png      — raw voltage histograms + post-norm.
  5. graph_topology.png            — NetworkX plot of 25-node graph.
  6. split_balance.png             — class distribution per split.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import seaborn as sns

from fault_localization.config.constants import (
    FAULT_START_STEP,
    FAULT_TYPES,
    NUM_CLASSES,
    PMU_BUSES,
    TOTAL_RECORD_MS,
)

logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
_COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1. Voltage sag plots
# ---------------------------------------------------------------------------


def plot_voltage_sag(
    recording: np.ndarray,
    event_id: str,
    output_dir: Path,
    pmu_node_indices: list[int] | None = None,
) -> Path:
    """Plot 3-phase voltage sag for a single fault event.

    Parameters
    ----------
    recording:
        Shape [N, T, F] per-unit RMS voltages.
    event_id:
        Identifier string for filename and title.
    output_dir:
        Output directory.
    pmu_node_indices:
        Which node indices to plot (default: first 3 nodes).

    Returns
    -------
    Path
        Path to saved figure.
    """
    if pmu_node_indices is None:
        pmu_node_indices = [0, 1, 2]

    n_nodes_to_plot = len(pmu_node_indices)
    time_axis = np.arange(TOTAL_RECORD_MS)

    fig, axes = plt.subplots(
        n_nodes_to_plot, 1,
        figsize=(12, 3 * n_nodes_to_plot),
        sharex=True,
    )
    if n_nodes_to_plot == 1:
        axes = [axes]

    phase_labels = ["Phase A", "Phase B", "Phase C"]
    phase_colors = ["#e63946", "#457b9d", "#2a9d8f"]

    for ax_idx, node_idx in enumerate(pmu_node_indices):
        ax = axes[ax_idx]
        bus_id = PMU_BUSES[node_idx]

        for phase in range(3):
            voltages = recording[node_idx, :, phase]
            ax.plot(
                time_axis, voltages,
                label=phase_labels[phase],
                color=phase_colors[phase],
                linewidth=1.5,
            )

        ax.axvline(
            x=FAULT_START_STEP,
            color="gray", linestyle="--", linewidth=1.0, label="Fault onset (t=40ms)",
        )
        ax.set_ylabel("Voltage (p.u.)")
        ax.set_title(f"μPMU Bus {bus_id}")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_ylim(-0.05, 1.3)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"Voltage Sag — {event_id}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"voltage_sag_{event_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_multiple_voltage_sags(
    raw_sim_dir: Path,
    output_dir: Path,
    n_events: int = 5,
) -> list[Path]:
    """Plot voltage sag for up to *n_events* representative fault events."""
    raw_files = sorted(raw_sim_dir.glob("*.npz"))[:n_events]
    paths: list[Path] = []
    for npz_path in raw_files:
        npz = np.load(npz_path, allow_pickle=True)
        recording = npz["data"]
        meta = json.loads(str(npz["meta"]))
        event_id = (
            f"{meta['fault_bus']}_{meta['fault_type']}_"
            f"{meta['resistance']:.1f}ohm_run{meta['run_id']:04d}"
        )
        paths.append(plot_voltage_sag(recording, event_id, output_dir))
    return paths


# ---------------------------------------------------------------------------
# 2. Fault signature comparison
# ---------------------------------------------------------------------------


def plot_fault_signature_comparison(
    raw_sim_dir: Path,
    output_dir: Path,
    fault_bus: int = 7,
    node_idx: int = 0,
    phase: int = 0,
) -> Path:
    """Overlay voltage profiles for different fault types at the same bus."""
    fault_profiles: dict[str, np.ndarray] = {}

    for fault_type in FAULT_TYPES:
        # Find any matching file
        matches = list(raw_sim_dir.glob(f"{fault_bus}_{fault_type}_*.npz"))
        if matches:
            npz = np.load(matches[0], allow_pickle=True)
            fault_profiles[fault_type] = npz["data"][node_idx, :, phase]

    if not fault_profiles:
        logger.warning(
            "No fault events found for bus %d in %s", fault_bus, raw_sim_dir
        )
        return output_dir / "fault_signature_comparison.png"

    time_axis = np.arange(TOTAL_RECORD_MS)
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (fault_type, profile) in enumerate(fault_profiles.items()):
        ax.plot(
            time_axis, profile,
            label=fault_type,
            color=_COLORS[i % len(_COLORS)],
            linewidth=1.5,
        )

    ax.axvline(
        x=FAULT_START_STEP,
        color="gray", linestyle="--", linewidth=1.0, label="Fault onset",
    )
    phase_labels = ["Phase A", "Phase B", "Phase C"]
    ax.set_title(
        f"Fault Signature Comparison — Bus {fault_bus}, {phase_labels[phase]}",
        fontweight="bold",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (p.u.)")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fault_signature_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3. Class distribution histogram
# ---------------------------------------------------------------------------


def plot_class_distribution(labels_csv_path: Path, output_dir: Path) -> Path:
    """Bar chart of window counts across all 26 classes."""
    counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    with open(labels_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            counts[label] = counts.get(label, 0) + 1

    labels = list(range(NUM_CLASSES))
    values = [counts.get(l, 0) for l in labels]
    label_strs = ["No Fault"] + [f"Bus {PMU_BUSES[i]}" if i < len(PMU_BUSES) else str(i)
                                  for i in range(1, NUM_CLASSES)]
    # Shorten x-labels: use bus IDs only
    from fault_localization.config.constants import LABEL_TO_FAULT_BUS, NO_FAULT_CLASS
    x_labels = ["NF"] + [str(LABEL_TO_FAULT_BUS.get(i, i)) for i in range(1, NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, values, color=sns.color_palette("viridis", NUM_CLASSES))
    ax.set_xticks(labels)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Class (0=No Fault, 1–25=Fault Bus)")
    ax.set_ylabel("Window Count")
    ax.set_title("Class Distribution — 26-Class Labels", fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "class_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 4. Feature distribution (raw + normalized)
# ---------------------------------------------------------------------------


def plot_feature_distribution(
    windows_dir: Path,
    output_dir: Path,
    norm_stats: dict | None = None,
    n_files: int = 20,
) -> Path:
    """Voltage histograms (raw) and optional normalized distributions."""
    npz_files = sorted(windows_dir.glob("*.npz"))[:n_files]
    if not npz_files:
        logger.warning("No window files found in %s", windows_dir)
        return output_dir / "feature_distribution.png"

    all_values: list[np.ndarray] = []
    for npz_path in npz_files:
        windows = np.load(npz_path)["windows"]  # [W, N, S, F]
        all_values.append(windows.reshape(-1, 3))

    data = np.concatenate(all_values, axis=0)  # [total_samples, 3]
    data = data[np.all(np.isfinite(data), axis=1)]

    n_cols = 2 if norm_stats is not None else 1
    phase_labels = ["Phase A (V_A)", "Phase B (V_B)", "Phase C (V_C)"]
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 10))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for phase in range(3):
        ax_raw = axes[phase, 0]
        sns.histplot(data[:, phase], bins=100, ax=ax_raw, color=_COLORS[phase])
        ax_raw.set_xlabel("Voltage (p.u.)")
        ax_raw.set_title(f"Raw — {phase_labels[phase]}")
        ax_raw.set_ylabel("Count")

        if norm_stats is not None:
            ax_norm = axes[phase, 1]
            mean = norm_stats["mean"][phase]
            std = norm_stats["std"][phase]
            normalized = (data[:, phase] - mean) / std
            sns.histplot(normalized, bins=100, ax=ax_norm, color=_COLORS[phase])
            ax_norm.set_xlabel("Normalized voltage (z-score)")
            ax_norm.set_title(f"Normalized — {phase_labels[phase]}")
            ax_norm.set_ylabel("Count")

    fig.suptitle("Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "feature_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 5. Graph topology
# ---------------------------------------------------------------------------


def plot_graph_topology(
    G_measured: nx.Graph,
    output_dir: Path,
) -> Path:
    """NetworkX plot of the 25-node measured-only graph with bus ID labels."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Use spring layout (or fixed positions if available)
    pos = nx.spring_layout(G_measured, seed=42, k=2.0)

    phase_count = nx.get_node_attributes(G_measured, "phase_count")
    node_colors = [
        "#2a9d8f" if phase_count.get(n, 3) == 3 else "#e9c46a"
        for n in G_measured.nodes()
    ]

    nx.draw_networkx_nodes(
        G_measured, pos, node_color=node_colors,
        node_size=600, ax=ax, alpha=0.9,
    )
    nx.draw_networkx_labels(
        G_measured, pos, ax=ax, font_size=8, font_weight="bold"
    )
    # Edge labels: |Z| magnitude
    z_mag_labels = {
        (u, v): f"{G_measured[u][v].get('z_mag', 0):.3f}Ω"
        for u, v in G_measured.edges()
    }
    nx.draw_networkx_edges(G_measured, pos, ax=ax, alpha=0.6, width=2.0)
    nx.draw_networkx_edge_labels(G_measured, pos, z_mag_labels, ax=ax, font_size=6)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2a9d8f", label="3-phase μPMU"),
        Patch(facecolor="#e9c46a", label="1/2-phase μPMU"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title("IEEE 123-Bus: Measured-Only Graph (25 μPMU nodes)", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "graph_topology.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 6. Split balance
# ---------------------------------------------------------------------------


def plot_split_balance(
    labels_csv_path: Path,
    splits_dir: Path,
    output_dir: Path,
) -> Path:
    """Class distribution per split to verify stratification."""
    from fault_localization.processing.splits import read_split_file

    # Build window_id → label mapping
    label_map: dict[str, int] = {}
    with open(labels_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label_map[row["window_id"]] = int(row["label"])

    splits = {}
    for split_name in ["train", "val", "test"]:
        ids = read_split_file(splits_dir / f"{split_name}.txt")
        counts = [0] * NUM_CLASSES
        for wid in ids:
            label = label_map.get(wid, 0)
            counts[label] += 1
        splits[split_name] = counts

    x = np.arange(NUM_CLASSES)
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 5))

    split_colors = {"train": "#457b9d", "val": "#2a9d8f", "test": "#e63946"}
    for i, (split_name, counts) in enumerate(splits.items()):
        ax.bar(x + i * width, counts, width, label=split_name,
               color=split_colors[split_name], alpha=0.85)

    from fault_localization.config.constants import LABEL_TO_FAULT_BUS
    x_labels = ["NF"] + [str(LABEL_TO_FAULT_BUS.get(i, i)) for i in range(1, NUM_CLASSES)]
    ax.set_xticks(x + width)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Window Count")
    ax.set_title("Class Distribution Per Split", fontweight="bold")
    ax.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "split_balance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Composite: generate all validation figures
# ---------------------------------------------------------------------------


def generate_all_visualizations(
    raw_sim_dir: Path,
    processed_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    G_measured: nx.Graph | None = None,
    norm_stats: dict | None = None,
) -> list[Path]:
    """Generate all 6 validation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    labels_csv = processed_dir / "labels.csv"
    windows_dir = processed_dir / "windows"

    if raw_sim_dir.exists():
        saved.extend(plot_multiple_voltage_sags(raw_sim_dir, output_dir, n_events=5))
        saved.append(plot_fault_signature_comparison(raw_sim_dir, output_dir))

    if labels_csv.exists():
        saved.append(plot_class_distribution(labels_csv, output_dir))

    if windows_dir.exists():
        saved.append(plot_feature_distribution(windows_dir, output_dir, norm_stats))

    if G_measured is not None:
        saved.append(plot_graph_topology(G_measured, output_dir))

    if labels_csv.exists() and splits_dir.exists():
        saved.append(plot_split_balance(labels_csv, splits_dir, output_dir))

    logger.info("Generated %d validation figures → %s", len(saved), output_dir)
    return saved
