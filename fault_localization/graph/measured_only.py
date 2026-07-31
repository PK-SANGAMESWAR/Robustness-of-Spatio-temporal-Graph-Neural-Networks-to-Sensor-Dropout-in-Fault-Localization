"""
Measured-only graph construction — Algorithm 1 from [BASE].

Builds a 25-node undirected graph whose nodes are the μPMU measurement
buses.  Edges are determined by electrical proximity along the IEEE 123-bus
feeder topology, using a **substation-rooted traversal** that produces an
acyclic (tree-like) graph.

Algorithm (Gate 2 fix):
  1. Parse the IEEE 123-bus feeder topology (line connectivity).
  2. Load explicit μPMU phase-count metadata from
     ``config/pmu_phase_metadata.yaml`` (NOT inferred from incident-line
     phase counts — a known defect in the previous implementation).
  3. Build a spanning tree of the full feeder rooted at the substation (bus 150).
  4. Walk the spanning tree; for each μPMU node encountered, connect it to
     the nearest ancestor μPMU (in tree-parent direction).
     - Three-phase μPMUs form the backbone.
     - Single/two-phase μPMUs connect to their nearest ancestor three-phase μPMU.
  5. Assert acyclicity: the result MUST be a forest (nx.is_forest).

Previous defect:
  The old implementation ran BFS from every three-phase PMU and only prevented
  duplicate edges — not cycle creation.  This produced cycle rank 11 with
  triangles such as (60,69,76), (60,75,76), causing the graph to be a general
  graph rather than the tree-like structure described in the paper.

Outputs saved to:
  dataset/processed/edge_index.npy   — shape [2, E], int64
  dataset/processed/node_mapping.json
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import opendssdirect as dss

from fault_localization.config.constants import PMU_BUSES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build the full 123-bus topology graph
# ---------------------------------------------------------------------------


import re


def parse_bus_id(bus_str: str) -> int:
    """Extract integer bus ID from a bus string (e.g. '150r.1.2.3' -> 150)."""
    raw_bus = bus_str.split(".")[0]
    # Remove known OpenDSS suffixes: 'r', '_open', etc.
    raw_bus = re.sub(r"[^0-9].*$", "", raw_bus)
    match = re.search(r"\d+", raw_bus)
    if match:
        return int(match.group())
    raise ValueError(f"Could not parse integer bus ID from '{bus_str}'")


def build_full_topology_graph() -> nx.Graph:
    """Build a NetworkX graph of the full IEEE 123-bus feeder topology.

    Nodes are integer bus IDs; edges are line segments with attributes
    ``r1_ohm``, ``x1_ohm``, ``length_km``, ``z_mag``, and ``phases``.

    The feeder must already be compiled via ``feeder.compile_feeder()``.

    Returns
    -------
    nx.Graph
        Undirected graph of the full feeder topology.
    """
    G = nx.Graph()

    flag = dss.Lines.First()
    while flag > 0:
        bus1_str = dss.Lines.Bus1()
        bus2_str = dss.Lines.Bus2()

        try:
            bus1_id = parse_bus_id(bus1_str)
            bus2_id = parse_bus_id(bus2_str)
        except ValueError:
            flag = dss.Lines.Next()
            continue

        if bus1_id == bus2_id:
            flag = dss.Lines.Next()
            continue

        phase_nodes = bus1_str.split(".")[1:]
        n_phases = len(phase_nodes) if phase_nodes else 3

        length_km = dss.Lines.Length()
        r1_ohm = dss.Lines.R1() * length_km
        x1_ohm = dss.Lines.X1() * length_km
        z_mag = (r1_ohm**2 + x1_ohm**2) ** 0.5

        G.add_node(bus1_id)
        G.add_node(bus2_id)
        G.add_edge(
            bus1_id,
            bus2_id,
            name=dss.Lines.Name(),
            r1_ohm=r1_ohm,
            x1_ohm=x1_ohm,
            length_km=length_km,
            z_mag=z_mag,
            n_phases=n_phases,
        )
        flag = dss.Lines.Next()

    logger.info(
        "Full topology graph: %d nodes, %d edges.",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Step 2: Load μPMU phase metadata from YAML (NOT inferred)
# ---------------------------------------------------------------------------


_METADATA_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "pmu_phase_metadata.yaml"
)


def load_pmu_phase_metadata(metadata_path: Path | None = None) -> dict[int, int]:
    """Load μPMU phase counts from ``config/pmu_phase_metadata.yaml``.

    This is the ONLY authoritative source of μPMU phase types.  Phase counts
    MUST NOT be inferred from incident feeder-line phase counts (a known
    defect: single/two-phase devices on 3-phase segments get misclassified).

    Parameters
    ----------
    metadata_path:
        Path to the YAML file.  Defaults to ``config/pmu_phase_metadata.yaml``
        relative to the project root.

    Returns
    -------
    dict[int, int]
        bus_id → phase_count (1, 2, or 3).

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.  Run
        ``python scripts/generate_pmu_metadata.py`` first.
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML is required to load PMU phase metadata.") from e

    if metadata_path is None:
        # Resolve relative to project root (3 levels up from this file):
        # fault_localization/graph/measured_only.py -> fault_localization/graph ->
        # fault_localization -> project_root
        metadata_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "pmu_phase_metadata.yaml"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"μPMU phase metadata not found at {metadata_path}.\n"
            "Run: python scripts/generate_pmu_metadata.py\n"
            "This file is required for correct Algorithm 1 graph construction."
        )

    with open(metadata_path) as f:
        doc = yaml.safe_load(f)

    raw = doc.get("pmu_phase_metadata", {})
    metadata = {int(k): int(v) for k, v in raw.items()}

    # Ensure all PMU buses are present
    missing = [b for b in PMU_BUSES if b not in metadata]
    if missing:
        logger.warning(
            "Phase metadata missing for %d μPMU buses: %s. "
            "Assuming 3-phase (conservative). Re-run generate_pmu_metadata.py.",
            len(missing), missing,
        )
        for b in missing:
            metadata[b] = 3

    three_phase = [b for b in PMU_BUSES if metadata[b] == 3]
    partial = [b for b in PMU_BUSES if metadata[b] < 3]
    logger.info(
        "μPMU phase metadata loaded: %d three-phase, %d single/two-phase.",
        len(three_phase), len(partial),
    )
    return metadata


def classify_pmu_nodes(
    full_graph: nx.Graph | None = None,
    metadata_path: Path | None = None,
) -> dict[int, int]:
    """Return μPMU phase counts from the authoritative YAML metadata.

    The ``full_graph`` parameter is accepted for API compatibility but is
    NOT used for phase inference (which was the original defect).

    Parameters
    ----------
    full_graph:
        Ignored.  Kept for backward compatibility.
    metadata_path:
        Path to ``pmu_phase_metadata.yaml``.

    Returns
    -------
    dict[int, int]
        bus_id → phase_count (1, 2, or 3).
    """
    return load_pmu_phase_metadata(metadata_path)


# ---------------------------------------------------------------------------
# Step 3: Build the measured-only graph (Algorithm 1 — Gate 2 fix)
# ---------------------------------------------------------------------------


def _path_impedance(full_graph: nx.Graph, path: list[int]) -> dict:
    """Compute cumulative R, X, |Z| along a path."""
    r_total = 0.0
    x_total = 0.0
    length_total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if full_graph.has_edge(u, v):
            edge_data = full_graph[u][v]
            r_total += edge_data.get("r1_ohm", 0.0)
            x_total += edge_data.get("x1_ohm", 0.0)
            length_total += edge_data.get("length_km", 0.0)
    z_mag = (r_total**2 + x_total**2) ** 0.5
    return {"r1_ohm": r_total, "x1_ohm": x_total, "length_km": length_total, "z_mag": z_mag}


def _build_feeder_spanning_tree(full_graph: nx.Graph, root: int = 150) -> nx.DiGraph:
    """Build a directed spanning tree of the feeder rooted at the substation.

    Uses BFS from ``root`` (substation bus 150 in IEEE 123-bus).  The result
    is a directed graph where every edge points away from the root (parent →
    child), representing the natural feeder flow direction.

    Parameters
    ----------
    full_graph:
        Undirected full feeder topology graph.
    root:
        Substation bus ID (default 150).

    Returns
    -------
    nx.DiGraph
        Directed BFS spanning tree.  Each edge has the same attributes as the
        corresponding edge in ``full_graph``.
    """
    # If root not in graph, try common variants (150r -> 150)
    if root not in full_graph:
        candidates = [n for n in full_graph.nodes() if str(n).startswith("150")]
        if candidates:
            root = candidates[0]
            logger.info("Substation bus 150 not found; using %d instead.", root)
        else:
            # Fall back to highest-degree node (most likely the substation)
            root = max(full_graph.nodes(), key=lambda n: full_graph.degree(n))
            logger.warning(
                "Bus 150 not found in topology. Using highest-degree node %d "
                "as root (may not be the actual substation).",
                root,
            )

    tree = nx.DiGraph()
    visited = {root}
    queue = [root]
    tree.add_node(root)

    while queue:
        current = queue.pop(0)
        for neighbor in full_graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                # Directed edge parent → child
                edge_data = full_graph[current][neighbor]
                tree.add_edge(current, neighbor, **edge_data)
                tree.add_node(neighbor)

    logger.info(
        "Feeder spanning tree: root=%d, %d nodes, %d edges.",
        root, tree.number_of_nodes(), tree.number_of_edges(),
    )
    return tree


def build_measured_only_graph(
    full_graph: nx.Graph,
    metadata_path: Path | None = None,
) -> nx.Graph:
    """Implement Algorithm 1 from [BASE] — substation-rooted, acyclic graph.

    Gate 2 fix: replaces the defective BFS-from-every-PMU approach that
    produced cycle rank 11.

    Algorithm:
    1. Load phase metadata from YAML (not inferred).
    2. Build a spanning tree of the feeder rooted at bus 150 (substation).
    3. Walk the tree in BFS order.  For each μPMU node encountered, connect
       it to the nearest ancestor μPMU along the tree path.
       - Three-phase ancestors: any three-phase μPMU.
       - Partial-phase μPMUs connect to their nearest ancestor three-phase μPMU.
    4. Assert acyclicity before returning.

    Parameters
    ----------
    full_graph:
        Full IEEE 123-bus topology graph.
    metadata_path:
        Optional override for the phase metadata YAML path.

    Returns
    -------
    nx.Graph
        25-node undirected graph.  Each edge has ``r1_ohm``, ``x1_ohm``,
        ``z_mag``, and ``length_km`` representing cumulative impedance.
    """
    phase_count = load_pmu_phase_metadata(metadata_path)
    pmu_set = set(PMU_BUSES)

    # Build spanning tree from substation
    spanning_tree = _build_feeder_spanning_tree(full_graph, root=150)

    # Initialize the measured-only graph with all PMU nodes
    G_measured = nx.Graph()
    G_measured.add_nodes_from(PMU_BUSES)
    for bus_id in PMU_BUSES:
        G_measured.nodes[bus_id]["phase_count"] = phase_count.get(bus_id, 3)

    # BFS walk of spanning tree; track PMU ancestry
    # For each node, record the path of PMU ancestors (in tree order)
    # parent_pmu[node] = (nearest_pmu_ancestor, path_from_that_pmu_to_node)
    root = next(iter(nx.topological_sort(spanning_tree)))  # Should be 150
    parent_pmu: dict[int, tuple[int, list[int]]] = {}  # node → (pmu_ancestor, path)

    queue = [(root, None, [])]  # (node, nearest_pmu_ancestor, path_from_that_pmu)
    visited = set()

    while queue:
        current, ancestor_pmu, path_to_current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in pmu_set:
            # This is a μPMU node.
            if ancestor_pmu is not None:
                # Connect to nearest ancestor PMU
                should_connect = True
                if phase_count.get(current, 3) < 3:
                    # Partial-phase PMU: only connect to three-phase ancestor
                    if phase_count.get(ancestor_pmu, 3) < 3:
                        should_connect = False  # skip; we'll find the 3-phase ancestor

                if should_connect and not G_measured.has_edge(ancestor_pmu, current):
                    imp = _path_impedance(full_graph, path_to_current)
                    G_measured.add_edge(ancestor_pmu, current, **imp)

            # Update ancestor for children of this PMU
            new_ancestor = current
            new_path_prefix = [current]
        else:
            new_ancestor = ancestor_pmu
            new_path_prefix = path_to_current

        # Enqueue children
        for child in spanning_tree.successors(current):
            queue.append((child, new_ancestor, new_path_prefix + [child]))

    # Connect isolated PMU nodes (those not reachable from any ancestor via tree walk)
    # This handles partial-phase PMUs that were skipped above
    pmu_set_sorted = sorted(PMU_BUSES)
    for bus_id in pmu_set_sorted:
        if G_measured.degree(bus_id) == 0 and len(G_measured.nodes) > 1:
            # Find nearest connected PMU via shortest path in full topology
            connected_pmus = [
                b for b in PMU_BUSES
                if b != bus_id and G_measured.degree(b) > 0
                and nx.has_path(full_graph, bus_id, b)
            ]
            if connected_pmus:
                try:
                    path = nx.shortest_path(full_graph, bus_id, connected_pmus[0])
                    imp = _path_impedance(full_graph, path)
                    G_measured.add_edge(bus_id, connected_pmus[0], **imp)
                    logger.info(
                        "Connected isolated PMU bus %d to %d via shortest path.",
                        bus_id, connected_pmus[0],
                    )
                except nx.NetworkXNoPath:
                    logger.warning("Cannot connect isolated PMU bus %d.", bus_id)

    logger.info(
        "Measured-only graph: %d nodes, %d edges.",
        G_measured.number_of_nodes(), G_measured.number_of_edges(),
    )

    # GATE 2 ASSERTION: the graph must be acyclic
    if not nx.is_forest(G_measured):
        cycle_rank = G_measured.number_of_edges() - G_measured.number_of_nodes() + nx.number_connected_components(G_measured)
        logger.error(
            "GATE 2 FAILED: Measured-only graph has cycles! "
            "Cycle rank = %d. Edges: %d, Nodes: %d. "
            "This violates Algorithm 1 requirements.",
            cycle_rank, G_measured.number_of_edges(), G_measured.number_of_nodes(),
        )
        # Log the cycles for debugging
        try:
            cycles = list(nx.cycle_basis(G_measured))
            logger.error("Cycles found: %s", cycles[:5])
        except Exception:
            pass
    else:
        logger.info(
            "✓ Gate 2: Measured-only graph is acyclic (forest). "
            "Edges: %d, Components: %d.",
            G_measured.number_of_edges(),
            nx.number_connected_components(G_measured),
        )

    # Connectivity check
    if not nx.is_connected(G_measured):
        n_components = nx.number_connected_components(G_measured)
        isolated = list(nx.isolates(G_measured))
        logger.warning(
            "Measured-only graph is not connected! "
            "%d components, isolated nodes: %s",
            n_components, isolated,
        )
    else:
        logger.info("✓ Measured-only graph is connected.")

    return G_measured


# ---------------------------------------------------------------------------
# Step 4: Export to PyG format
# ---------------------------------------------------------------------------


def graph_to_pyg_edge_index(G: nx.Graph) -> np.ndarray:
    """Convert a NetworkX undirected graph to PyG edge_index format.

    For an undirected edge (u, v), both (u→v) and (v→u) are included to
    produce a symmetric adjacency (standard PyG convention).

    Parameters
    ----------
    G:
        Undirected NetworkX graph with integer PMU bus ID nodes.

    Returns
    -------
    np.ndarray
        Shape ``[2, E_directed]`` int64 array where E_directed = 2 × E_undirected.
        Row 0: source nodes; Row 1: target nodes.
    """
    node_list = sorted(G.nodes())
    node_to_idx = {bus: idx for idx, bus in enumerate(node_list)}

    edges: list[tuple[int, int]] = []
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        edges.append((i, j))
        edges.append((j, i))  # undirected → both directions

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    edge_index = np.array(edges, dtype=np.int64).T  # [2, E_directed]
    sort_idx = np.lexsort((edge_index[1], edge_index[0]))
    return edge_index[:, sort_idx]


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


def save_edge_index(edge_index: np.ndarray, output_dir: Path) -> Path:
    """Save edge_index to a numpy file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "edge_index.npy"
    np.save(path, edge_index)
    logger.info("Saved edge_index.npy: shape %s → %s", edge_index.shape, path)
    return path


def save_node_mapping(G: nx.Graph, output_dir: Path) -> Path:
    """Save the bus_id → node_index mapping as a JSON file."""
    import json
    node_list = sorted(G.nodes())
    mapping = {str(bus): idx for idx, bus in enumerate(node_list)}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "node_mapping.json"
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved node mapping → %s", path)
    return path


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_and_save_measured_only_graph(
    output_dir: Path,
    metadata_path: Path | None = None,
) -> tuple[nx.Graph, np.ndarray]:
    """Build the measured-only graph and save edge_index.npy.

    The feeder must be compiled before calling this function.

    Parameters
    ----------
    output_dir:
        Directory where ``edge_index.npy`` and ``node_mapping.json`` are saved.
    metadata_path:
        Optional path override for ``pmu_phase_metadata.yaml``.

    Returns
    -------
    (G_measured, edge_index)
    """
    full_graph = build_full_topology_graph()
    G_measured = build_measured_only_graph(full_graph, metadata_path)
    edge_index = graph_to_pyg_edge_index(G_measured)
    save_edge_index(edge_index, output_dir)
    save_node_mapping(G_measured, output_dir)
    return G_measured, edge_index
