"""Targeted (worst-case) sensor dropout for the μPMU robustness study.

The degradation curves of :mod:`fault_localization.dropout.evaluator` drop
μPMUs uniformly at random, so every number they produce is an *average* over
sensor subsets.  This module supplies the adversarial counterpart: drop the
``k`` μPMUs that are topologically **closest to the fault**, which is the
worst case an operator actually cares about, since only 6 of the 25 fault
buses carry a μPMU and the nearest sensors carry the strongest signature.

Proximity is hop distance over the **full 123-bus feeder topology**, not over
the 25-node measured-only graph: fault buses are generally not nodes of the
measured-only graph (only the 6 instrumented ones are), so the distance from a
fault to a sensor is only well defined on the physical feeder.

Selection is a deterministic function of the sample's label, so every model is
automatically evaluated on the identical mask for a given (label, k) — a
stronger pairing than the random sweep, which needs a shared seed to achieve
the same thing.

No-fault windows (label 0, exactly half the corpus) have no fault bus, so no
targeted selection is defined for them; they fall back to random selection
using the same per-batch seed scheme as the random sweep.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_proximity_rank(node_mapping_path: str | Path) -> np.ndarray:
    """Rank μPMU nodes by hop distance from each fault bus.

    Compiles the IEEE 123-bus feeder, builds the full topology graph, and for
    every fault-location class computes the ordering of the 25 μPMU nodes by
    shortest-path hop distance from that class's fault bus.  Ties are broken by
    ascending node index, so the result is fully deterministic.

    Requires OpenDSS.  Note that OpenDSS's ``Compile`` changes the process
    working directory, so callers must use absolute paths afterwards.

    Parameters
    ----------
    node_mapping_path:
        ``processed/node_mapping.json`` of the dataset, mapping bus id (as a
        string) to node index in the measured-only graph.

    Returns
    -------
    np.ndarray
        Int64 array of shape ``[25, 25]``.  Row ``label - 1`` holds the μPMU
        node indices ordered nearest-to-farthest from that label's fault bus,
        so ``rank[label - 1, :k]`` is the targeted drop set for ``label``.
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
    if any(mapping[str(bus)] != idx for idx, bus in enumerate(pmu_sorted)):
        raise ValueError(
            "node_mapping.json does not follow sorted(PMU_BUSES) order; the "
            "proximity rank would be misaligned with the model's node axis."
        )

    rank = np.zeros((25, 25), dtype=np.int64)
    for label in range(1, 26):
        fault_bus = LABEL_TO_FAULT_BUS[label]
        hops = nx.single_source_shortest_path_length(graph, fault_bus)
        ordered = sorted((hops[bus], mapping[str(bus)]) for bus in pmu_sorted)
        rank[label - 1] = [node for _, node in ordered]

    logger.info("Built proximity rank for 25 fault classes over %d-bus topology.",
                graph.number_of_nodes())
    return rank


def apply_targeted_dropout_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    rank: np.ndarray,
    fallback_seed: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Drop the ``k`` μPMUs nearest each sample's fault bus.

    Zeroing semantics are identical to
    :func:`fault_localization.dropout.masking.apply_sensor_dropout_batch` —
    all 6 channels of a dropped node go to zero, so its availability mask
    becomes ``(0, 0, 0)`` — so the only variable that changes relative to the
    random sweep is *which* nodes are dropped.

    Parameters
    ----------
    x:
        Batch tensor of shape ``[B, N, S, F=6]``.
    y:
        Batch labels of shape ``[B]``; 0 is the no-fault class.
    k:
        Number of sensors to drop (0 = no dropout).
    rank:
        ``[25, 25]`` proximity rank from :func:`build_proximity_rank`.
    fallback_seed:
        Seed for the random selection applied to no-fault samples.

    Returns
    -------
    (x_dropped, dropped_per_sample)
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y disagree on batch size: {x.shape[0]} vs {y.shape[0]}. "
            "Targeted selection is label-conditioned, so a mismatch would "
            "silently mask the wrong nodes."
        )

    if k <= 0:
        return x, [[] for _ in range(x.shape[0])]

    b, n, _, _ = x.shape
    k_eff = min(k, n)
    rng = random.Random(fallback_seed)
    labels = y.detach().cpu().tolist()

    dropped_per_sample: list[list[int]] = []
    for label in labels:
        if label > 0:
            dropped_per_sample.append(sorted(rank[label - 1, :k_eff].tolist()))
        else:
            # No fault bus exists for the no-fault class, so proximity is
            # undefined; fall back to the random policy.
            dropped_per_sample.append(sorted(rng.sample(range(n), k_eff)))

    x_dropped = x.clone()
    idx_b = torch.arange(b, device=x.device).repeat_interleave(k_eff)
    idx_n = torch.tensor(
        [d for ds in dropped_per_sample for d in ds],
        dtype=torch.long, device=x.device,
    )
    x_dropped[idx_b, idx_n] = 0.0

    return x_dropped, dropped_per_sample
