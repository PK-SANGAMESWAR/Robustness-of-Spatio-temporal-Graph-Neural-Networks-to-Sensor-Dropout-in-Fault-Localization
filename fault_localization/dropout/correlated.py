"""Spatially correlated (contiguous) sensor dropout.

``dropout/masking.py`` draws the k dropped μPMUs uniformly at random and
independently of one another, so the resulting degradation curve is an average
over *scattered* sensor subsets.  Real μPMU outages are rarely scattered: units
on one lateral share a communication path, a pole, and a backhaul radio, and a
substation or feeder-segment event takes out a contiguous block of them at once.

Which of the two is harder is not obvious a priori, and the experiment exists to
settle it rather than to confirm a guess.  Contiguity removes a node *and* the
neighbours that would have relayed for it, severing a region of the measured-only
graph — but it concentrates all of the damage in one region and leaves the rest
of the feeder fully observed, whereas scattered loss degrades every region at
once.  Under a macro-average over 26 classes those two effects push in opposite
directions, so the sign of the difference is an empirical question.

This module supplies that policy: the dropped set is grown as a **connected
subgraph** of the 25-node measured-only tree, from a uniformly chosen root,
by repeatedly attaching a uniformly chosen frontier neighbour.  Zeroing
semantics are byte-for-byte those of ``apply_sensor_dropout_batch`` — all six
channels of a dropped node go to zero, so its mask becomes (0, 0, 0) — so the
only variable that changes relative to the random sweep is *which* nodes go
dark, which is what makes the two curves comparable.

The measured-only graph is a tree with 24 edges, so a connected k-subset always
exists for k ≤ 25 and is never unique; the root and the growth order are both
drawn from the caller's seed, keeping the policy deterministic and shareable
across models in the same way the random policy is.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def build_adjacency(edge_index: np.ndarray | torch.Tensor, n_nodes: int = 25) -> list[list[int]]:
    """Undirected adjacency lists for the measured-only graph.

    ``edge_index`` is stored as ``[2, E]``.  It is treated as undirected here
    regardless of whether both directions are present, matching the closed
    neighbourhood the message-passing operators use.
    """
    ei = np.asarray(edge_index if isinstance(edge_index, np.ndarray)
                    else edge_index.detach().cpu().numpy())
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, E], got {ei.shape}")

    adjacency: list[set[int]] = [set() for _ in range(n_nodes)]
    for src, dst in zip(ei[0].tolist(), ei[1].tolist()):
        if src == dst:
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    return [sorted(neighbours) for neighbours in adjacency]


def sample_connected_subset(adjacency: list[list[int]], k: int,
                            rng: random.Random) -> list[int]:
    """Grow a connected set of ``k`` nodes from a uniformly chosen root.

    At each step the next node is drawn uniformly from the frontier — the
    neighbours of the current set that are not yet in it — which yields a
    contiguous outage without biasing towards any particular branch.  If the
    frontier empties before ``k`` nodes are collected (impossible on the
    connected tree used here, but guarded for reuse), growth restarts from a
    fresh unused root, so the result is a union of contiguous blocks rather
    than a failure.
    """
    n = len(adjacency)
    k = min(k, n)
    if k <= 0:
        return []

    chosen: set[int] = set()
    frontier: set[int] = set()

    while len(chosen) < k:
        if not frontier:
            candidates = [v for v in range(n) if v not in chosen]
            root = rng.choice(candidates)
            chosen.add(root)
            frontier = {u for u in adjacency[root] if u not in chosen}
            continue

        nxt = rng.choice(sorted(frontier))
        chosen.add(nxt)
        frontier.discard(nxt)
        frontier.update(u for u in adjacency[nxt] if u not in chosen)

    return sorted(chosen)


def apply_correlated_dropout_batch(
    x: torch.Tensor,
    k: int,
    seed: int,
    adjacency: list[list[int]],
) -> tuple[torch.Tensor, list[list[int]]]:
    """Drop a contiguous block of ``k`` μPMUs, drawn per sample.

    Mirrors ``apply_sensor_dropout_batch``: one subset per sample in the batch,
    drawn in sample order from a single seeded RNG, applied with one vectorised
    scatter.  Only the *selection rule* differs.

    Parameters
    ----------
    x:
        Batch tensor ``[B, N, S, 6]``.
    k:
        Number of sensors to drop (0 = no dropout).
    seed:
        Seed for the per-batch draw; vary it with the batch index.
    adjacency:
        Undirected adjacency lists from :func:`build_adjacency`.

    Returns
    -------
    (x_dropped, dropped_per_sample)
    """
    if k <= 0:
        return x, [[] for _ in range(x.shape[0])]

    b, n, _, _ = x.shape
    if len(adjacency) != n:
        raise ValueError(
            f"adjacency has {len(adjacency)} nodes but the batch has {n}; a "
            "mismatch would silently mask the wrong sensors."
        )

    rng = random.Random(seed)
    k_eff = min(k, n)
    dropped_per_sample = [sample_connected_subset(adjacency, k_eff, rng)
                          for _ in range(b)]

    x_dropped = x.clone()
    idx_b = torch.arange(b, device=x.device).repeat_interleave(k_eff)
    idx_n = torch.tensor(
        [d for ds in dropped_per_sample for d in ds],
        dtype=torch.long, device=x.device,
    )
    x_dropped[idx_b, idx_n] = 0.0

    return x_dropped, dropped_per_sample
