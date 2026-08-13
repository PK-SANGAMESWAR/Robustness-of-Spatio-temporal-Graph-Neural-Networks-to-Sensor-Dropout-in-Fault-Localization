"""
Sensor-dropout masking for μPMU nodes.

This module provides functions to apply sensor dropout at inference time
(and during dropout-aware training).

Missingness representation (Dataset-check.md §11):
  - Availability mask channels (3-5) set to 0 for dropped nodes.
  - Voltage channels (0-2) set to 0 for dropped nodes.
  - Two message-passing policies:
      * "zero_messages" (default): dropped nodes keep their edges but
        their features are zeroed.  Simpler and a reasonable baseline.
      * "drop_edges": edges to/from dropped nodes are removed from
        edge_index before message passing.  More faithful to the idea
        that a missing sensor cannot propagate information.

IMPORTANT: Zeroing features alone (without setting the mask) is NOT
sufficient — a zero voltage is a valid reading on some buses.  The
availability mask is the authoritative indicator of sensor loss.

KNOWN_ISSUES compliance:
  - Same dropout seeds across all models (internal fairness).
  - Multiple seeds and error bars are enforced in evaluator.py.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch


PolicyType = Literal["zero_messages", "drop_edges"]


def apply_sensor_dropout(
    x: torch.Tensor,
    k: int,
    rng: random.Random,
    policy: PolicyType = "zero_messages",
    edge_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list[int]]:
    """Apply sensor dropout to a single window tensor.

    Randomly selects ``k`` μPMU nodes to drop (simulate missing sensors).
    Dropped nodes have their voltage channels (0-2) zeroed and availability
    mask channels (3-5) set to 0.

    Parameters
    ----------
    x:
        Input tensor of shape ``[N, S, F=6]`` (single sample) or
        ``[B, N, S, F=6]`` (batch).
    k:
        Number of sensors to drop (0 = no dropout).
    rng:
        Random instance for reproducible node selection.
    policy:
        Message-passing policy for dropped nodes:
        - ``"zero_messages"``: features zeroed; edges kept.
        - ``"drop_edges"``: edges to/from dropped nodes removed.
    edge_index:
        Graph connectivity [2, E] — required for ``"drop_edges"`` policy.

    Returns
    -------
    (x_dropped, edge_index_dropped, dropped_nodes)
        x_dropped      : Tensor with dropped nodes zeroed out.
        edge_index_dropped : Modified edge_index (None if policy is
                          ``"zero_messages"``).
        dropped_nodes  : List of dropped node indices.
    """
    if k <= 0:
        return x, edge_index, []

    # Determine N from tensor shape
    if x.dim() == 3:
        n = x.shape[0]   # [N, S, F]
    elif x.dim() == 4:
        n = x.shape[1]   # [B, N, S, F]
    else:
        raise ValueError(f"Expected x of shape [N,S,F] or [B,N,S,F], got {x.shape}")

    k = min(k, n)
    dropped = sorted(rng.sample(range(n), k))

    x_dropped = x.clone()
    if x.dim() == 3:
        x_dropped[dropped, :, :] = 0.0   # zero all 6 channels
    else:
        x_dropped[:, dropped, :, :] = 0.0

    # Ensure mask channels reflect the dropout (0 = unavailable)
    # This is redundant if already zeroed, but makes intent explicit
    # and is correct even if voltage was legitimately zero before.
    if x.dim() == 3:
        x_dropped[dropped, :, 3:] = 0.0  # mask channels → unavailable
    else:
        x_dropped[:, dropped, :, 3:] = 0.0

    # Handle edge_index according to policy
    edge_index_out = edge_index
    if policy == "drop_edges" and edge_index is not None:
        dropped_set = set(dropped)
        src, dst = edge_index
        keep_mask = torch.tensor(
            [s.item() not in dropped_set and d.item() not in dropped_set
             for s, d in zip(src, dst)],
            dtype=torch.bool,
            device=edge_index.device,
        )
        edge_index_out = edge_index[:, keep_mask]

    return x_dropped, edge_index_out, dropped


def apply_sensor_dropout_batch(
    x: torch.Tensor,
    k: int,
    seed: int,
    policy: PolicyType = "zero_messages",
    edge_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list[list[int]]]:
    """Apply sensor dropout to a batch, dropping the same k nodes per sample.

    Parameters
    ----------
    x:
        Batch tensor of shape ``[B, N, S, F=6]``.
    k:
        Number of sensors to drop per sample in the batch.
    seed:
        Random seed for reproducibility (ensures same nodes dropped across
        models for fair comparison — KNOWN_ISSUES §3).
    policy:
        Message-passing policy (see :func:`apply_sensor_dropout`).
    edge_index:
        Graph connectivity [2, E].

    Returns
    -------
    (x_dropped, edge_index_dropped, dropped_per_sample)
        dropped_per_sample : list of k dropped node indices per sample.
    """
    rng = random.Random(seed)
    b, n, s, f = x.shape
    x_dropped = x.clone()

    # Draw the per-sample node subsets exactly as apply_sensor_dropout does
    # (``sorted(rng.sample(range(n), k))``, once per sample, in sample order), so
    # the RNG stream — and therefore every mask — is unchanged.  The old code
    # obtained these by calling apply_sensor_dropout per sample, which cloned and
    # zeroed a [N,S,F] slice whose tensor result was then thrown away (the caller
    # re-zeroed into x_dropped anyway).  That was b wasted clones plus 2b zeroing
    # ops per call, on the hottest path of the whole dropout sweep.  Here we take
    # only the sample indices and apply one vectorised scatter instead.
    k_eff = min(k, n)
    dropped_per_sample: list[list[int]] = [
        sorted(rng.sample(range(n), k_eff)) for _ in range(b)
    ]

    if k_eff > 0:
        idx_b = torch.arange(b, device=x.device).repeat_interleave(k_eff)
        idx_n = torch.tensor(
            [d for ds in dropped_per_sample for d in ds],
            dtype=torch.long, device=x.device,
        )
        x_dropped[idx_b, idx_n] = 0.0

    # For batch mode, edge_index policy is applied once (same graph for all)
    edge_index_out = edge_index
    if policy == "drop_edges" and edge_index is not None and b > 0:
        # Use the union of all dropped nodes across the batch
        all_dropped: set[int] = set()
        for d in dropped_per_sample:
            all_dropped.update(d)
        if all_dropped:
            src, dst = edge_index
            keep_mask = torch.tensor(
                [s.item() not in all_dropped and d.item() not in all_dropped
                 for s, d in zip(src, dst)],
                dtype=torch.bool,
                device=edge_index.device,
            )
            edge_index_out = edge_index[:, keep_mask]

    return x_dropped, edge_index_out, dropped_per_sample
