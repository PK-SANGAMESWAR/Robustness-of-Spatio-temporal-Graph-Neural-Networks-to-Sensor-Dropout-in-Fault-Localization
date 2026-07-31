"""
Dropout-aware training augmentation.

Wraps a DataLoader to apply random sensor dropout to each training batch.
This forces the model to learn representations robust to missing sensors.

Design:
  - For each batch, randomly select k ~ Uniform(k_range) sensors to drop.
  - Apply the same dropout policy as the inference evaluator (zero_messages
    or drop_edges).
  - The number of dropped sensors varies per batch (curriculum: start with
    small k, increase over training) if a schedule is provided.

KNOWN_ISSUES compliance:
  - Dropout-aware and standard training use IDENTICAL base data/splits.
  - The only difference is the per-batch masking applied during forward pass.
"""

from __future__ import annotations

import random
from typing import Any, Iterator

import torch

from fault_localization.dropout.masking import apply_sensor_dropout_batch


class DropoutAugmentedLoader:
    """Wraps a DataLoader to apply random sensor dropout per training batch.

    Parameters
    ----------
    base_loader:
        The underlying DataLoader (typically the training loader).
    k_range:
        List of possible dropout levels to sample from each batch.
        Example: [0, 1, 2, 5] means each batch randomly drops 0–5 nodes.
        Set to [0] to disable augmentation (equivalent to standard training).
    rng_seed:
        Base random seed.  Each epoch should use a different seed to avoid
        repeating the same dropout pattern.
    policy:
        Message-passing policy: ``"zero_messages"`` or ``"drop_edges"``.
    """

    def __init__(
        self,
        base_loader,
        k_range: list[int],
        rng_seed: int = 42,
        policy: str = "zero_messages",
    ):
        self.base_loader = base_loader
        self.k_range = k_range
        self.rng_seed = rng_seed
        self.policy = policy

    def __len__(self) -> int:
        return len(self.base_loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.rng_seed)

        for batch_idx, batch in enumerate(self.base_loader):
            x: torch.Tensor = batch["x"]      # [B, N, S, F=6]

            # Sample dropout level for this batch
            k = rng.choice(self.k_range)

            if k > 0:
                # Use a deterministic seed per batch so gradients are reproducible
                mask_seed = self.rng_seed * 10000 + batch_idx
                edge_index = batch.get("edge_index", None)

                x_dropped, _, dropped_nodes = apply_sensor_dropout_batch(
                    x, k=k, seed=mask_seed, policy=self.policy,
                    edge_index=edge_index,
                )
                batch = dict(batch)  # shallow copy to avoid mutating original
                batch["x"] = x_dropped
                batch["dropout_k"] = k
                batch["dropout_nodes"] = dropped_nodes
            else:
                batch = dict(batch)
                batch["dropout_k"] = 0
                batch["dropout_nodes"] = [[] for _ in range(x.size(0))]

            yield batch

    def with_seed(self, epoch: int) -> "DropoutAugmentedLoader":
        """Return a new loader with a different seed (for epoch-level variation).

        Calling this at the start of each epoch ensures that different batches
        in different epochs see different dropout patterns.

        Parameters
        ----------
        epoch:
            Current epoch index.

        Returns
        -------
        DropoutAugmentedLoader
            New loader with seed = base_seed + epoch.
        """
        return DropoutAugmentedLoader(
            self.base_loader,
            self.k_range,
            rng_seed=self.rng_seed + epoch,
            policy=self.policy,
        )
