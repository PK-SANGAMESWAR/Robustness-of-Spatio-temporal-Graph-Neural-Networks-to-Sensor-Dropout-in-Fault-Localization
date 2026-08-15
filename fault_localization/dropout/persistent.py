"""Persistent (per-event) sensor dropout for the μPMU robustness study.

The degradation curves of :mod:`fault_localization.dropout.evaluator` redraw the
dropped sensor set for **every window**.  With 40 windows per event that is
physically incoherent: a μPMU that fails is down for the whole event, not
re-drawn every 20 ms.  It is also optimistic, because a model that aggregates
evidence over an event sees a *different* 40-window subset of sensors missing
each time, so the union of observed sensors across an event stays close to full
observability even at large k.

This module supplies the coherent counterpart: draw the dropped set **once per
event** and hold it fixed across that event's windows.  Everything else —
zeroing semantics, the k grid, the soft-vote metric — is unchanged, so the only
variable that moves relative to the random sweep is *when* the mask is resampled.

Determinism.  The drop set is a pure function of ``(event_id, mask_seed)``,
derived through a BLAKE2b digest rather than Python's builtin ``hash()``, whose
string hashing is salted per process (``PYTHONHASHSEED``) and would therefore
produce a different sweep on every run.  Because selection is keyed on the event
id and not on batch position, every model, training seed, and batch size sees the
identical mask for a given event — the same fairness property the random sweep
buys with a shared per-batch seed, but here it holds even if the loader's
batching changes.
"""

from __future__ import annotations

import hashlib
import logging
import random

import torch

logger = logging.getLogger(__name__)


def event_id_of(window_id: str) -> str:
    """Return the event a window belongs to.

    Window ids are formed as ``<event_id>_w<index>`` by the processing pipeline,
    and :class:`~fault_localization.data.dataset.WindowGraphDataset` splits them
    back with the same ``rsplit`` used here, so this stays consistent with the
    dataset's own notion of an event (and therefore with the event-level splits).
    """
    return window_id.rsplit("_w", 1)[0]


def event_drop_set(event_id: str, k: int, n_nodes: int, mask_seed: int) -> list[int]:
    """Select the ``k`` μPMUs that are unavailable for the whole of one event.

    Parameters
    ----------
    event_id:
        Identifier of the event; the mask is constant across its windows.
    k:
        Number of sensors to drop (0 = no dropout).
    n_nodes:
        Number of μPMU nodes in the measured-only graph (25).
    mask_seed:
        Selects one realization of the failure pattern.  Averaging over several
        mask seeds gives the error bar over *which* sensors failed, exactly as
        the random sweep's mask offsets do.

    Returns
    -------
    list[int]
        Sorted node indices to drop.
    """
    if k <= 0:
        return []

    # BLAKE2b over the (event, k, seed) triple: stable across processes and
    # machines, unlike hash() on str.  k enters the key so that the k=1 drop set
    # is not merely a prefix of the k=2 one — each level is an independent draw,
    # matching the random sweep, where every (k, offset) pair redraws.
    key = f"{event_id}|{k}|{mask_seed}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    return sorted(rng.sample(range(n_nodes), min(k, n_nodes)))


def apply_persistent_dropout_batch(
    x: torch.Tensor,
    window_ids: list[str],
    k: int,
    mask_seed: int,
    cache: dict[tuple[str, int, int], list[int]] | None = None,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Zero the per-event dropped sensors for every sample in a batch.

    Zeroing semantics are identical to
    :func:`fault_localization.dropout.masking.apply_sensor_dropout_batch` — all 6
    channels of a dropped node go to zero, so its availability mask becomes
    ``(0, 0, 0)``.  Samples from the same event therefore receive the *same*
    dropped set, whether or not they land in the same batch.

    Parameters
    ----------
    x:
        Batch tensor of shape ``[B, N, S, F=6]``.
    window_ids:
        The batch's window ids, in sample order (``batch["window_id"]``).
    k:
        Number of sensors to drop per event.
    mask_seed:
        Failure-pattern realization; see :func:`event_drop_set`.
    cache:
        Optional dict reused across batches to avoid re-hashing the same event.
        The digest is cheap, but an event recurs in ~40 samples, so this removes
        nearly all of the hashing from the inner loop.

    Returns
    -------
    (x_dropped, dropped_per_sample)
    """
    if x.shape[0] != len(window_ids):
        raise ValueError(
            f"x and window_ids disagree on batch size: {x.shape[0]} vs "
            f"{len(window_ids)}.  Persistent selection is keyed on the window "
            "id, so a mismatch would silently mask the wrong nodes."
        )

    if k <= 0:
        return x, [[] for _ in range(x.shape[0])]

    b, n, _, _ = x.shape
    k_eff = min(k, n)
    if cache is None:
        cache = {}

    dropped_per_sample: list[list[int]] = []
    for wid in window_ids:
        event = event_id_of(wid)
        ck = (event, k, mask_seed)
        drop = cache.get(ck)
        if drop is None:
            drop = event_drop_set(event, k, n, mask_seed)
            cache[ck] = drop
        dropped_per_sample.append(drop)

    x_dropped = x.clone()
    idx_b = torch.arange(b, device=x.device).repeat_interleave(k_eff)
    idx_n = torch.tensor(
        [d for ds in dropped_per_sample for d in ds],
        dtype=torch.long, device=x.device,
    )
    x_dropped[idx_b, idx_n] = 0.0

    return x_dropped, dropped_per_sample
