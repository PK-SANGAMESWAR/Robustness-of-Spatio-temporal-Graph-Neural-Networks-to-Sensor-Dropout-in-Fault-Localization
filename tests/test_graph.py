"""Tests for graph construction (Phase 4).

These tests work in two modes:
  - Without OpenDSS: tests graph utilities using a synthetic topology.
  - With OpenDSS   : full integration test on the compiled IEEE 123-bus feeder.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from fault_localization.config.constants import PMU_BUSES


# ---------------------------------------------------------------------------
# Synthetic graph fixture (no OpenDSS required)
# ---------------------------------------------------------------------------


def _make_synthetic_pmu_graph(n_nodes: int = 25) -> nx.Graph:
    """Build a simple path graph on the first 25 PMU buses for testing."""
    G = nx.Graph()
    buses = PMU_BUSES[:n_nodes]
    for i, bus_id in enumerate(buses):
        G.add_node(bus_id, phase_count=3 if i % 3 != 0 else 1)

    for i in range(len(buses) - 1):
        G.add_edge(
            buses[i], buses[i + 1],
            r1_ohm=0.1 * (i + 1),
            x1_ohm=0.05 * (i + 1),
            z_mag=((0.1 * (i + 1))**2 + (0.05 * (i + 1))**2) ** 0.5,
            length_km=0.5,
        )
    return G


@pytest.fixture()
def synthetic_graph():
    return _make_synthetic_pmu_graph()


# ---------------------------------------------------------------------------
# graph_to_pyg_edge_index
# ---------------------------------------------------------------------------


def test_edge_index_shape(synthetic_graph):
    from fault_localization.graph.measured_only import graph_to_pyg_edge_index

    edge_index = graph_to_pyg_edge_index(synthetic_graph)
    n_edges = synthetic_graph.number_of_edges()
    assert edge_index.shape == (2, 2 * n_edges), (
        f"Expected [2, {2 * n_edges}], got {edge_index.shape}"
    )


def test_edge_index_dtype(synthetic_graph):
    from fault_localization.graph.measured_only import graph_to_pyg_edge_index

    edge_index = graph_to_pyg_edge_index(synthetic_graph)
    assert edge_index.dtype == np.int64


def test_edge_index_valid_node_range(synthetic_graph):
    from fault_localization.graph.measured_only import graph_to_pyg_edge_index

    edge_index = graph_to_pyg_edge_index(synthetic_graph)
    n_nodes = synthetic_graph.number_of_nodes()
    assert edge_index.min() >= 0
    assert edge_index.max() < n_nodes


def test_edge_index_symmetric(synthetic_graph):
    """Each directed edge (u→v) must have a corresponding (v→u)."""
    from fault_localization.graph.measured_only import graph_to_pyg_edge_index

    edge_index = graph_to_pyg_edge_index(synthetic_graph)
    edges_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    for u, v in edges_set:
        assert (v, u) in edges_set, f"Edge ({u},{v}) has no reverse ({v},{u})"


# ---------------------------------------------------------------------------
# build_edge_attr
# ---------------------------------------------------------------------------


def test_edge_attr_shape(synthetic_graph):
    from fault_localization.graph.edge_attributes import build_edge_attr

    edge_attr = build_edge_attr(synthetic_graph, feature_names=["R", "X", "Z_mag"])
    n_directed = 2 * synthetic_graph.number_of_edges()
    assert edge_attr.shape == (n_directed, 3), (
        f"Expected [{n_directed}, 3], got {edge_attr.shape}"
    )


def test_edge_attr_r_positive(synthetic_graph):
    from fault_localization.graph.edge_attributes import build_edge_attr

    edge_attr = build_edge_attr(synthetic_graph, feature_names=["R", "X", "Z_mag"])
    assert (edge_attr[:, 0] > 0).all(), "Some edges have non-positive R"


def test_edge_attr_x_positive(synthetic_graph):
    from fault_localization.graph.edge_attributes import build_edge_attr

    edge_attr = build_edge_attr(synthetic_graph, feature_names=["R", "X", "Z_mag"])
    assert (edge_attr[:, 1] > 0).all(), "Some edges have non-positive X"


def test_edge_attr_extended_features(synthetic_graph):
    from fault_localization.graph.edge_attributes import build_edge_attr

    edge_attr = build_edge_attr(
        synthetic_graph, feature_names=["R", "X", "Z_mag", "length_km", "B"]
    )
    assert edge_attr.shape[1] == 5


# ---------------------------------------------------------------------------
# Integration: full 123-bus graph build
# ---------------------------------------------------------------------------


def _can_import_dss() -> bool:
    try:
        import opendssdirect  # noqa: F401
        return True
    except ImportError:
        return False


pytest_skip_no_dss = pytest.mark.skipif(
    not _can_import_dss(), reason="opendssdirect not available"
)


@pytest.fixture(scope="module")
def measured_graph(tmp_path_factory):
    from fault_localization.simulation.feeder import compile_feeder, ensure_feeder_files
    from fault_localization.graph.measured_only import build_and_save_measured_only_graph
    from pathlib import Path

    master = ensure_feeder_files()
    compile_feeder(master)
    output_dir = tmp_path_factory.mktemp("processed")
    # Pass the project-level metadata path explicitly (always generated by scripts/generate_pmu_metadata.py)
    metadata_path = Path(__file__).parent.parent / "config" / "pmu_phase_metadata.yaml"
    G, edge_index = build_and_save_measured_only_graph(output_dir, metadata_path=metadata_path)
    return G, edge_index, output_dir


@pytest_skip_no_dss
def test_measured_graph_node_count(measured_graph):
    G, _, _ = measured_graph
    assert G.number_of_nodes() == 25


@pytest_skip_no_dss
def test_measured_graph_connected(measured_graph):
    G, _, _ = measured_graph
    assert nx.is_connected(G), "Measured-only graph must be connected"


@pytest_skip_no_dss
def test_measured_graph_no_isolated(measured_graph):
    G, _, _ = measured_graph
    isolated = list(nx.isolates(G))
    assert isolated == [], f"Found isolated nodes: {isolated}"


@pytest_skip_no_dss
def test_measured_edge_index_saved(measured_graph):
    _, edge_index, output_dir = measured_graph
    edge_index_path = output_dir / "edge_index.npy"
    assert edge_index_path.exists()
    loaded = np.load(edge_index_path)
    np.testing.assert_array_equal(loaded, edge_index)
