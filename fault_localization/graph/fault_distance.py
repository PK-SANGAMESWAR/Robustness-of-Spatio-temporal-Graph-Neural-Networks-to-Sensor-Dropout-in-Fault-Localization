"""Hop-distance matrix between candidate fault buses and μPMU sensors.

The measured-only graph (25 nodes, 24 edges) is built by compressing the feeder
onto its instrumented buses, which discards the geometry the compression walked
over: 19 of the 25 candidate fault buses are not nodes of that graph at all, so
a model operating on it has no representation of *where* a candidate fault sits
relative to the sensors that would see it.  The targeted-dropout study made the
cost of that omission measurable — dropping the sensors nearest the fault is far
more damaging than dropping the same number at random — but nothing in the
architecture was ever told which sensors those are.

This module supplies that missing geometry as a static ``[26, 25]`` matrix of
hop distances over the **full 123-bus feeder topology**:

    D[q, i] = shortest-path hop count from the fault bus of class ``q``
              to the bus carrying μPMU node ``i``

Row 0 is the no-fault class, which has no fault bus and is therefore all zeros:
a uniform row adds a constant to every attention logit of that query and so
cancels in the softmax, leaving the no-fault query with no geometric prior
rather than an arbitrary one.  Rows 1..25 follow ``LABEL_TO_FAULT_BUS``.

Distances, not the rankings of :func:`fault_localization.dropout.targeted.
build_proximity_rank`: a rank tells you which sensor is nearest, a distance tells
you *how much* nearer, which is what an attention bias needs.

Requires OpenDSS to build, so the result is cached to a ``.npy`` beside the
feeder config and loaded from there afterwards.  Note that OpenDSS's ``Compile``
changes the process working directory, so paths are resolved before compiling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Static property of the feeder, so it lives with the feeder config rather than
# in any one dataset directory (the frozen corpora are never written to).
CACHE_PATH = (Path(__file__).resolve().parent.parent
              / "data" / "ieee123" / "config" / "fault_hop_distance.npy")

NUM_CLASSES = 26
NUM_SENSORS = 25


def build_fault_hop_distance(node_mapping_path: str | Path) -> np.ndarray:
    """Compute the ``[26, 25]`` fault-bus-to-μPMU hop-distance matrix.

    Parameters
    ----------
    node_mapping_path:
        ``processed/node_mapping.json`` of the dataset, mapping bus id (as a
        string) to node index in the measured-only graph.

    Returns
    -------
    np.ndarray
        Int64 array of shape ``[26, 25]``.  ``D[0]`` is all zeros (no-fault);
        ``D[label, i]`` is the hop distance from ``LABEL_TO_FAULT_BUS[label]``
        to the bus of μPMU node ``i``.

    Raises
    ------
    ValueError
        If ``node_mapping.json`` does not order nodes by ``sorted(PMU_BUSES)``.
        The matrix is indexed by the model's node axis, so a different ordering
        would silently attach every sensor to the wrong distance.
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
    # Same guard as build_proximity_rank: the node axis is positional, so an
    # ordering mismatch is silent corruption rather than an error.
    if any(mapping[str(bus)] != idx for idx, bus in enumerate(pmu_sorted)):
        raise ValueError(
            "node_mapping.json does not follow sorted(PMU_BUSES) order; the "
            "hop-distance matrix would be misaligned with the model's node axis."
        )

    distance = np.zeros((NUM_CLASSES, NUM_SENSORS), dtype=np.int64)
    for label in range(1, NUM_CLASSES):
        fault_bus = LABEL_TO_FAULT_BUS[label]
        hops = nx.single_source_shortest_path_length(graph, fault_bus)
        missing = [bus for bus in pmu_sorted if bus not in hops]
        if missing:
            raise ValueError(
                f"buses {missing} are unreachable from fault bus {fault_bus}; "
                "the feeder topology should be connected."
            )
        distance[label] = [hops[bus] for bus in pmu_sorted]

    logger.info("Built fault-to-μPMU hop distances over %d-bus topology "
                "(max %d hops).", graph.number_of_nodes(), int(distance.max()))
    return distance


def load_fault_hop_distance(
    node_mapping_path: str | Path | None = None,
    cache_path: str | Path = CACHE_PATH,
    rebuild: bool = False,
) -> np.ndarray:
    """Load the hop-distance matrix, building and caching it if needed.

    Parameters
    ----------
    node_mapping_path:
        Required only when the cache is absent or ``rebuild`` is set.
    cache_path:
        Where the ``.npy`` lives.
    rebuild:
        Ignore any existing cache and recompute.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        distance = np.load(cache_path)
        if distance.shape != (NUM_CLASSES, NUM_SENSORS):
            raise ValueError(
                f"cached hop-distance matrix at {cache_path} has shape "
                f"{distance.shape}, expected {(NUM_CLASSES, NUM_SENSORS)}."
            )
        return distance

    if node_mapping_path is None:
        raise ValueError(
            f"no cached matrix at {cache_path} and no node_mapping_path given; "
            "pass the dataset's processed/node_mapping.json to build it."
        )

    distance = build_fault_hop_distance(node_mapping_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, distance)
    logger.info("Cached hop-distance matrix to %s", cache_path)
    return distance
