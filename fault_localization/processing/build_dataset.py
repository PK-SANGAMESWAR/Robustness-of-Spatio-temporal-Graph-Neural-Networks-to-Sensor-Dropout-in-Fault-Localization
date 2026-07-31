"""Materialize processed artifacts from resumable raw simulations.

This script orchestrates the full dataset build pipeline:
  1. Build and validate the measured-only graph (Gate 2 fix: graph provenance).
  2. Extract sliding windows from raw simulation NPZs.
  3. Assign labels and metadata.
  4. Generate train/val/test splits.
  5. Compute normalization statistics on the training split.

The graph is built FIRST so that edge_index.npy is always consistent with
the raw simulations.  The old defect was consuming whatever edge_index.npy
happened to exist without re-validating it.

Usage:
    python -m fault_localization.processing.build_dataset --data-root dataset/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build(data_root: Path, feeder_config: str = "default") -> None:
    """Run the full dataset build pipeline.

    Parameters
    ----------
    data_root:
        Root directory containing ``raw_sim/`` subdirectory and where
        ``processed/`` and ``splits/`` will be created.
    feeder_config:
        ``"default"`` or ``"green"`` — affects which graph is built.
    """
    import networkx as nx

    data_root = Path(data_root).resolve()
    raw = data_root / "raw_sim"
    processed = data_root / "processed"
    windows_dir = processed / "windows"

    # -----------------------------------------------------------------
    # Step 1: Build + validate the measured-only graph
    # Gate 4b fix: graph is built here, not consumed from stale file.
    # -----------------------------------------------------------------
    logger.info("Step 1: Building measured-only graph …")
    from fault_localization.simulation.feeder import (
        compile_feeder,
        configure_green_feeder,
        ensure_feeder_files,
    )
    from fault_localization.graph.measured_only import build_and_save_measured_only_graph
    from fault_localization.graph.edge_attributes import build_and_save_edge_attr

    master_path = ensure_feeder_files()
    compile_feeder(master_path)

    if feeder_config == "green":
        configure_green_feeder()

    G, edge_index = build_and_save_measured_only_graph(processed)
    build_and_save_edge_attr(G, processed, edge_index=edge_index)

    # Gate 2 assertion: acyclicity
    if not nx.is_forest(G):
        cycle_rank = (
            G.number_of_edges()
            - G.number_of_nodes()
            + nx.number_connected_components(G)
        )
        raise RuntimeError(
            f"GATE 2 FAILED: Measured-only graph has cycles (cycle rank={cycle_rank}). "
            "Fix the graph construction before proceeding."
        )
    logger.info(
        "✓ Graph built and validated: %d nodes, %d edges, acyclic.",
        G.number_of_nodes(), G.number_of_edges(),
    )

    # -----------------------------------------------------------------
    # Step 2: Extract sliding windows
    # -----------------------------------------------------------------
    logger.info("Step 2: Extracting sliding windows from raw NPZ files …")
    from fault_localization.processing.windowing import process_all_events
    process_all_events(raw, windows_dir)

    # -----------------------------------------------------------------
    # Step 3: Assign labels and metadata
    # -----------------------------------------------------------------
    logger.info("Step 3: Assigning labels …")
    from fault_localization.processing.labels import assign_labels
    assign_labels(raw, processed / "labels.csv", processed / "metadata.csv")

    # -----------------------------------------------------------------
    # Step 4: Generate splits
    # -----------------------------------------------------------------
    logger.info("Step 4: Generating event-level splits …")
    from fault_localization.processing.splits import generate_splits
    generate_splits(processed / "labels.csv", data_root / "splits")

    # -----------------------------------------------------------------
    # Step 5: Compute normalization stats (train split only)
    # -----------------------------------------------------------------
    logger.info("Step 5: Computing normalization statistics (train split) …")
    from fault_localization.processing.normalization import (
        compute_normalization_stats,
        save_norm_stats,
    )
    stats = compute_normalization_stats(windows_dir, data_root / "splits" / "train.txt")
    save_norm_stats(stats, processed)

    logger.info(
        "Dataset build complete. Processed artifacts written to: %s",
        processed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--feeder-config",
        choices=["default", "green"],
        default="default",
        help="Feeder topology configuration (default or green/reconfigured).",
    )
    args = p.parse_args()
    build(args.data_root, feeder_config=args.feeder_config)
