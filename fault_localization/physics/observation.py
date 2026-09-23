"""Window tensor -> observation vector, in per-unit.

``01_SPEC.md`` §3.  The observation is the **last timestep** of the window,
de-normalised back to per-unit so it lives in the same units as the signature
table:

    s_ip = x[i, S-1, p] * std_p + mean_p
    A    = { (i, p) : m_ip = 1 }

Two constraints shape this and are easy to break by accident:

1. *Inputs only.*  The mean-of-post-inception-samples of the first draft needed
   the window's offset inside the event, which is not in the ``[25, 20, 6]``
   tensor (``01_SPEC.md`` §5).  The last timestep needs nothing: the fault
   period is quasi-static with the loads frozen, so every post-inception sample
   sits at the same level and the last one *is* that level whenever the window
   has one at all.  Window 20 has none, so its last sample is the pre-fault
   level and the residual prefers f = 0 -- the "physics disabled" policy falls
   out for free.
2. *The loader normalises, the table does not.*  ``WindowGraphDataset`` z-scores
   channels 0-2 with the training-split statistics, so anything read off a batch
   must be pushed back through ``mean``/``std`` before it meets ``mu``.

``reduce="last5"`` averages the final five timesteps.  On the clean corpus this
changes nothing (the post-fault level is flat to < 1e-6 p.u.), so it is compared
against ``last`` **only in the noise experiments**, where a single sample is
sqrt(5) noisier than the average.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # keeps the module importable for offline/NumPy use
    torch = None

REDUCERS = ("last", "last5")


def load_norm_stats(data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, std)`` over the three voltage channels, from the corpus."""
    stats = json.loads((Path(data_root) / "processed/norm_stats.json").read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("norm_stats must carry three voltage channels")
    return mean, std


def _reduce_window(x, reduce: str):
    """Collapse the time axis of ``[..., S, F]`` to ``[..., F]``."""
    if reduce == "last":
        return x[..., -1, :]
    if reduce == "last5":
        return x[..., -5:, :].mean(axis=-2) if isinstance(x, np.ndarray) else x[..., -5:, :].mean(dim=-2)
    raise ValueError(f"reduce must be one of {REDUCERS}, got {reduce!r}")


def observe(x, mean, std, reduce: str = "last"):
    """Batch of normalised windows -> ``(s [B,25,3] p.u., a [B,25,3] float)``.

    ``x`` is ``[B, N, S, 6]`` as the loader delivers it (torch or NumPy).  The
    availability ``a`` is read from the mask channels, so it is 0 both for
    phases the uPMU never had and for nodes the dropout policy zeroed -- the
    residual needs no separate notion of "dropped".

    A dropped node's voltage channels are 0, which de-normalises to ``mean``
    rather than to zero.  That is harmless *only* because every consumer
    multiplies by ``a``; nothing here may be used without it.
    """
    if x.shape[-1] != 6:
        raise ValueError(f"expected 6 feature channels, got {x.shape[-1]}")
    flat = _reduce_window(x, reduce)                      # [B, N, 6]
    if torch is not None and isinstance(flat, torch.Tensor):
        mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32), device=flat.device, dtype=flat.dtype)
        std_t = torch.as_tensor(np.asarray(std, dtype=np.float32), device=flat.device, dtype=flat.dtype)
        s = flat[..., :3] * std_t + mean_t
        a = (flat[..., 3:] > 0.5).to(flat.dtype)
        return s, a
    flat = np.asarray(flat)
    s = flat[..., :3] * np.asarray(std, dtype=flat.dtype) + np.asarray(mean, dtype=flat.dtype)
    a = (flat[..., 3:] > 0.5).astype(flat.dtype)
    return s, a


def observe_raw(record: np.ndarray, window: int, window_length: int = 20,
                reduce: str = "last") -> tuple[np.ndarray, np.ndarray]:
    """Observation straight from a raw ``[25, 60, 6]`` event recording.

    Window ``w`` covers steps ``[w, w + window_length)``, so its last timestep is
    step ``w + window_length - 1``.  Raw recordings are already in per-unit, so
    no de-normalisation happens here -- which is the point: the offline cache
    (``physics/cache.py``) never touches ``norm_stats`` and therefore cannot
    drift from it.
    """
    end = window + window_length
    if end > record.shape[1]:
        raise IndexError(f"window {window} exceeds record of {record.shape[1]} steps")
    chunk = record[:, window:end, :]
    flat = _reduce_window(chunk, reduce)
    return flat[..., :3].astype(np.float32), (flat[..., 3:] > 0.5).astype(np.float32)
