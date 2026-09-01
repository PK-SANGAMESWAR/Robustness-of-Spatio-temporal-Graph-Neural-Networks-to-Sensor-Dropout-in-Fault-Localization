"""Fault-to-sensor sensitivity templates and the electrical-zone partition.

Physical basis
--------------
On a radial feeder the voltage sag seen at sensor ``i`` when a fault occurs at
bus ``b`` is, to first order, governed by the mutual impedance between them,
which for a radial network is the impedance of the **path they share** from the
substation::

    Z_ib  ~=  Z( source -> lowest common ancestor of i and b )

so the sag pattern across the sensor array factorizes as::

    dv_i  ~=  a * s_{b,i}

where ``a`` is a scalar severity that absorbs fault resistance and loading, and
``s_b`` is a normalized spatial signature fixed by the network.  That
factorization is why high-impedance faults are hard (``a`` is small: 0.01 p.u.
at 10 ohm versus 0.83 p.u. at 0.1 ohm) while remaining, in principle,
localizable: the *pattern* is unchanged.

The electrical-zone partition
-----------------------------
Two fault buses whose signatures are collinear cannot be told apart by any
method that looks only at sag amplitude across the sensors, no matter how it is
trained.  Grouping the 25 classes by signature collinearity therefore yields a
partition into **electrical zones**, and gives a hard ceiling on what a purely
spatial-amplitude method can achieve.  On this feeder that ceiling is well below
what the trained networks reach, which localizes exactly where the learned part
of a model has to do its work: *within* a zone, using temporal shape and
per-phase asymmetry rather than spatial amplitude.

This is a first-order model.  It ignores the driving-point term ``Z_bb + R_f``
and load flow, so the true separability may differ; the templates it produces
are used as an *initialization* for a model that is free to move them, never as
a fixed constraint.

Requires OpenDSS to build, so results are cached beside the feeder config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_PATH = (Path(__file__).resolve().parent.parent
              / "data" / "ieee123" / "config" / "fault_sensitivity.npz")

SUBSTATION_BUS = 150
# Cosine similarity above which two class signatures are treated as
# indistinguishable by spatial amplitude alone.
COLLINEAR_TOL = 0.999


def build_sensitivity(node_mapping_path: str | Path,
                      collinear_tol: float = COLLINEAR_TOL) -> dict:
    """Build the ``[25, 25]`` sensitivity matrix and the zone partition.

    Returns
    -------
    dict with keys
        ``sensitivity``   ``[25, 25]`` float32, row ``label-1`` is the
                          L2-normalized signature of that fault class.
        ``zone_of_class`` ``[26]`` int64, zone index per class; entry 0 is the
                          no-fault class and is set to -1.
        ``zone_templates`` ``[Z, 25]`` float32, one normalized template per
                          zone (the mean of its members' signatures).
        ``n_zones``       int
    """
    import networkx as nx

    from fault_localization.config.constants import LABEL_TO_FAULT_BUS, PMU_BUSES
    from fault_localization.graph.measured_only import build_full_topology_graph
    from fault_localization.simulation.feeder import compile_feeder

    node_mapping_path = Path(node_mapping_path).resolve()
    mapping = json.loads(node_mapping_path.read_text())

    compile_feeder()
    graph = build_full_topology_graph()

    pmu_sorted = sorted(PMU_BUSES)
    # Same positional guard as the hop-distance builder: the sensor axis is
    # positional, so a mapping mismatch would silently transpose the templates.
    if any(mapping[str(bus)] != idx for idx, bus in enumerate(pmu_sorted)):
        raise ValueError(
            "node_mapping.json does not follow sorted(PMU_BUSES) order; the "
            "sensitivity templates would be misaligned with the node axis.")

    root = SUBSTATION_BUS if SUBSTATION_BUS in graph else min(graph.nodes)
    for _, _, d in graph.edges(data=True):
        d["z"] = float(d.get("z_mag",
                             (d.get("r1_ohm", 0.0) ** 2
                              + d.get("x1_ohm", 0.0) ** 2) ** 0.5))

    paths = nx.single_source_dijkstra_path(graph, root, weight="z")
    zcum = nx.single_source_dijkstra_path_length(graph, root, weight="z")

    def shared_impedance(a: int, b: int) -> float:
        """Impedance of the path a and b share from the substation."""
        shared = 0.0
        for x, y in zip(paths[a], paths[b]):
            if x != y:
                break
            shared = zcum[x]
        return shared

    raw = np.zeros((25, 25), dtype=np.float64)
    for label in range(1, 26):
        fault_bus = LABEL_TO_FAULT_BUS[label]
        raw[label - 1] = [shared_impedance(fault_bus, bus) for bus in pmu_sorted]

    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("degenerate sensitivity row (all-zero signature)")
    sensitivity = raw / norms

    # Zones = connected components of the "indistinguishable" relation.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    similarity = sensitivity @ sensitivity.T
    adjacency = csr_matrix((similarity > collinear_tol).astype(np.int8))
    n_zones, labels = connected_components(adjacency, directed=False)

    zone_templates = np.zeros((n_zones, 25), dtype=np.float64)
    for z in range(n_zones):
        members = sensitivity[labels == z]
        template = members.mean(axis=0)
        zone_templates[z] = template / np.linalg.norm(template)

    zone_of_class = np.full(26, -1, dtype=np.int64)   # class 0 = no-fault
    zone_of_class[1:] = labels

    sizes = np.bincount(labels)
    logger.info("Sensitivity: %d electrical zones over 25 classes "
                "(largest %d classes, %d singletons).",
                n_zones, sizes.max(), int((sizes == 1).sum()))
    return {
        "sensitivity": sensitivity.astype(np.float32),
        "zone_of_class": zone_of_class,
        "zone_templates": zone_templates.astype(np.float32),
        "n_zones": int(n_zones),
    }


def load_sensitivity(node_mapping_path: str | Path | None = None,
                     cache_path: str | Path = CACHE_PATH,
                     rebuild: bool = False) -> dict:
    """Load the templates and zone partition, building and caching if needed."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path)
        return {
            "sensitivity": data["sensitivity"],
            "zone_of_class": data["zone_of_class"],
            "zone_templates": data["zone_templates"],
            "n_zones": int(data["n_zones"]),
        }

    if node_mapping_path is None:
        raise ValueError(
            f"no cache at {cache_path} and no node_mapping_path given; pass the "
            "dataset's processed/node_mapping.json to build it.")

    result = build_sensitivity(node_mapping_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **result)
    logger.info("Cached sensitivity templates to %s", cache_path)
    return result
