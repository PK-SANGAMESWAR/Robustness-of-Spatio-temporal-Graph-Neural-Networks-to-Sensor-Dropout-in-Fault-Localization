"""
Edge attribute computation for the measured-only graph.

Computes per-edge impedance attributes (R, X, |Z|) and optional
extended features (line length, susceptance B). Features are stored
once per graph — they are static (not per-timestep).

Output:
  dataset/processed/edge_attr.npy — shape [E_directed, F_edge]

Edge ordering strictly matches edge_index.npy by deriving edge node pairs
directly from PyG edge_index.

Default edge feature set:
  ["R", "X", "Z_mag"] → F_edge = 3

Extended feature set (optional):
  ["R", "X", "Z_mag", "length_km", "B"] → F_edge = 5
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np

from fault_localization.graph.measured_only import graph_to_pyg_edge_index

logger = logging.getLogger(__name__)

# Default features (always computed)
DEFAULT_EDGE_FEATURES: list[str] = ["R", "X", "Z_mag"]

# Extended feature set (stored, selectable at model-load time)
EXTENDED_EDGE_FEATURES: list[str] = ["R", "X", "Z_mag", "length_km", "B"]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _susceptance(r_ohm: float, x_ohm: float) -> float:
    """Compute series susceptance magnitude |B| = |X| / (R^2 + X^2) in Siemens.

    In AC circuit theory, Y = 1 / (R + jX) = R/(R^2 + X^2) - j X/(R^2 + X^2).
    Returns 0.0 if R^2 + X^2 is negligible.
    """
    denom = r_ohm**2 + x_ohm**2
    if denom < 1e-12:
        return 0.0
    return abs(x_ohm) / denom


def _edge_features_for_pair(
    G: nx.Graph,
    u: int,
    v: int,
    feature_names: list[str],
) -> list[float]:
    """Extract requested edge features for node pair (u, v).

    Parameters
    ----------
    G:
        Measured-only NetworkX graph with impedance edge attributes.
    u, v:
        Node pair (bus IDs).
    feature_names:
        Ordered list of feature names or aliases.

    Returns
    -------
    list[float]
        Feature values in the order of ``feature_names``.
    """
    if not G.has_edge(u, v):
        raise KeyError(f"Edge ({u}, {v}) not found in measured-only graph.")

    data = G[u][v]
    r = float(data.get("r1_ohm", 0.0))
    x = float(data.get("x1_ohm", 0.0))
    z_mag = float(data.get("z_mag", (r**2 + x**2) ** 0.5))
    length = float(data.get("length_km", 0.0))
    b = _susceptance(r, x)

    feature_map = {
        # Standard names
        "R": r,
        "X": x,
        "Z_mag": z_mag,
        "length_km": length,
        "B": b,
        # Case / naming aliases
        "r": r,
        "x": x,
        "r1_ohm": r,
        "x1_ohm": x,
        "z_mag": z_mag,
        "z": z_mag,
        "length": length,
        "b": b,
        "susceptance": b,
    }

    values: list[float] = []
    for name in feature_names:
        if name not in feature_map:
            raise KeyError(
                f"Unknown edge feature '{name}'. "
                f"Supported options: {list(feature_map.keys())}"
            )
        values.append(feature_map[name])

    return values


# ---------------------------------------------------------------------------
# Build edge_attr array
# ---------------------------------------------------------------------------


def build_edge_attr(
    G: nx.Graph,
    feature_names: list[str] = DEFAULT_EDGE_FEATURES,
    edge_index: np.ndarray | None = None,
) -> np.ndarray:
    """Compute edge attribute matrix matching PyG edge_index ordering.

    Parameters
    ----------
    G:
        Measured-only NetworkX graph with impedance attributes.
    feature_names:
        Ordered list of feature names to include.
    edge_index:
        Optional PyG edge_index array [2, E_directed]. If None, computed
        via :func:`graph_to_pyg_edge_index`.

    Returns
    -------
    np.ndarray
        Shape ``[E_directed, F_edge]`` float64 array. Row k matches
        column k of ``edge_index``.
    """
    if edge_index is None:
        edge_index = graph_to_pyg_edge_index(G)

    node_list = sorted(G.nodes())

    n_directed_edges = edge_index.shape[1]
    n_features = len(feature_names)
    edge_attr = np.zeros((n_directed_edges, n_features), dtype=np.float64)

    for k in range(n_directed_edges):
        src_idx = int(edge_index[0, k])
        dst_idx = int(edge_index[1, k])
        u = node_list[src_idx]
        v = node_list[dst_idx]

        edge_attr[k] = _edge_features_for_pair(G, u, v, feature_names)

    logger.info(
        "Edge attribute matrix: shape %s, features: %s",
        edge_attr.shape, feature_names,
    )
    return edge_attr


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_edge_attr(edge_attr: np.ndarray, output_dir: Path) -> Path:
    """Save edge_attr to a numpy file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "edge_attr.npy"
    np.save(path, edge_attr)
    logger.info("Saved edge_attr.npy: shape %s → %s", edge_attr.shape, path)
    return path


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_and_save_edge_attr(
    G: nx.Graph,
    output_dir: Path,
    feature_names: list[str] = DEFAULT_EDGE_FEATURES,
    edge_index: np.ndarray | None = None,
) -> np.ndarray:
    """Build and save the edge attribute matrix.

    Parameters
    ----------
    G:
        Measured-only NetworkX graph.
    output_dir:
        Directory where ``edge_attr.npy`` is saved.
    feature_names:
        Edge feature names to include.
    edge_index:
        Optional pre-computed edge index array.

    Returns
    -------
    np.ndarray
        Shape ``[E_directed, F_edge]``.
    """
    edge_attr = build_edge_attr(G, feature_names, edge_index=edge_index)
    save_edge_attr(edge_attr, output_dir)
    return edge_attr

