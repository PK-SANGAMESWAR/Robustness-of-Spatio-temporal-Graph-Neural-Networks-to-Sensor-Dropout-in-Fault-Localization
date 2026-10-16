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


class PhysicsDropoutLoader(DropoutAugmentedLoader):
    """``DropoutAugmentedLoader`` that also exposes the pre-dropout batch.

    PC-STGNN's lambda_2 term (``PC-STGNN/01_SPEC.md`` Sec 6, ``models/common.py::pc_loss``)
    is a leave-k-out consistency score on the *dropped* sensors, so the training loop needs
    both the clean and the dropped observation plus which channels were zeroed -- none of
    which the base class retains (it overwrites ``batch["x"]`` in place). This subclass adds
    ``batch["x_clean"]`` and ``batch["drop_mask"]`` ([B, N, 3], 1 on synthetically dropped
    channels) on top of every key the base class already yields, using the same per-batch
    mask seed, so it is a strict superset -- existing consumers of ``DropoutAugmentedLoader``
    are unaffected because this is a separate class, not a change to it.

    Optional measurement-noise augmentation (``06_STAGE_E_pcstgnn.md`` S4.3ᵇ): with
    ``noise_sigmas`` given, each window draws its own sigma (p.u.) from that grid, noise is
    added by ``dropout/noise.py`` (unchanged) *before* dropout, and ``batch["sigma0"]`` ([B])
    records the level actually injected so the residual can use it. ``x_clean`` is then the
    noisy, undropped batch -- what a real sensor on D would have read. The sigma draws use
    their own RNG, so the k sequence and every dropout mask are identical to the noise-free
    loader at the same seed. With ``noise_sigmas=None`` this class behaves exactly as before.
    """

    def __init__(self, base_loader, k_range, rng_seed: int = 42, policy: str = "zero_messages",
                 noise_sigmas=None, norm_std=None):
        super().__init__(base_loader, k_range, rng_seed=rng_seed, policy=policy)
        self.noise_sigmas = list(noise_sigmas) if noise_sigmas else None
        self.norm_std = norm_std
        if self.noise_sigmas and norm_std is None:
            raise ValueError("noise augmentation needs norm_std to map p.u. sigma to z-space")

    def _add_noise(self, x: torch.Tensor, batch_idx: int, sigma_rng: random.Random):
        from fault_localization.dropout.noise import apply_measurement_noise_batch
        sigmas = [sigma_rng.choice(self.noise_sigmas) for _ in range(x.shape[0])]
        x_noisy = x.clone()
        for gi, sig in enumerate(self.noise_sigmas):
            if sig <= 0:
                continue
            rows = [i for i, s in enumerate(sigmas) if s == sig]
            if rows:
                seed = (self.rng_seed * 10000 + batch_idx) * 16 + gi + 1
                x_noisy[rows] = apply_measurement_noise_batch(x[rows], sig, self.norm_std, seed)
        return x_noisy, torch.tensor(sigmas, dtype=torch.float32)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.rng_seed)
        sigma_rng = random.Random(self.rng_seed * 7919 + 1)

        for batch_idx, batch in enumerate(self.base_loader):
            x_clean: torch.Tensor = batch["x"]      # [B, N, S, F=6]
            k = rng.choice(self.k_range)
            sigma0 = None
            if self.noise_sigmas:
                x_clean, sigma0 = self._add_noise(x_clean, batch_idx, sigma_rng)

            if k > 0:
                mask_seed = self.rng_seed * 10000 + batch_idx
                edge_index = batch.get("edge_index", None)
                x_dropped, _, dropped_nodes = apply_sensor_dropout_batch(
                    x_clean, k=k, seed=mask_seed, policy=self.policy,
                    edge_index=edge_index,
                )
                b, n = x_clean.shape[0], x_clean.shape[1]
                drop_mask = torch.zeros(b, n, 3)
                for i, dropped in enumerate(dropped_nodes):
                    if dropped:
                        drop_mask[i, dropped, :] = 1.0
                out = dict(batch)
                out["x"] = x_dropped
                out["x_clean"] = x_clean
                out["drop_mask"] = drop_mask
                out["dropout_k"] = k
                out["dropout_nodes"] = dropped_nodes
            else:
                out = dict(batch)
                out["x"] = x_clean
                out["x_clean"] = x_clean
                out["drop_mask"] = torch.zeros(x_clean.shape[0], x_clean.shape[1], 3)
                out["dropout_k"] = 0
                out["dropout_nodes"] = [[] for _ in range(x_clean.size(0))]
            if sigma0 is not None:
                out["sigma0"] = sigma0

            yield out

    def with_seed(self, epoch: int) -> "PhysicsDropoutLoader":
        return PhysicsDropoutLoader(
            self.base_loader,
            self.k_range,
            rng_seed=self.rng_seed + epoch,
            policy=self.policy,
            noise_sigmas=self.noise_sigmas,
            norm_std=self.norm_std,
        )
