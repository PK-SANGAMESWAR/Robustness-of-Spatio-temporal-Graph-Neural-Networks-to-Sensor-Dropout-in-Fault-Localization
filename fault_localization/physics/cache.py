"""Precompute rho for a whole split, straight from the raw recordings.

The residual depends on the inputs alone (``01_SPEC.md`` §5), so for the clean,
undropped case it can be computed once and reused by every experiment -- which
is what makes the training-free Stage C affordable and keeps the physics term
out of the training loop's critical path.

Two deliberate choices:

*Read ``raw_sim``, not ``processed``.*  Raw recordings are already per-unit, so
the cache never touches ``norm_stats.json`` and therefore cannot drift from it.
Window ``w``'s observation is step ``w + 19`` of the ``[25, 60, 6]`` record --
exactly what ``observe()`` sees after the loader normalises and
:func:`observation.observe` de-normalises.

*Cache ``k = 0`` only* (:func:`build_rho_cache`).  Under synthetic dropout the
available set A changes, so ``rho`` changes: a cache would have to be keyed by
``(window, k, mask seed)`` (``12_CHECKLISTS.md`` §C).  Rather than store that
product, this cache covers the clean case and the layer stays online for
masked inputs.

*Cache the full dropout grid* (:func:`build_rtilde_cache`, ``05_STAGE_D_infra.md``
S3.4) -- added once the online ``method="direct"`` (float32, the only form
accurate enough to sit in a differentiable training loop -- see
``residual.py``'s module docstring) measured at ~45 ms/batch-of-32 on the T1000,
9x over the Gate-3 budget and **chunk_f-invariant** (compute-bound: profiling
26 classes x 3366 hypotheses x 75 channels per window, not a batching
inefficiency).  ``method="gram"`` (float64) is ~2x faster per window on this
GPU's weak fp64 throughput -- not the "~800x" the docstring's general claim
implies on faster hardware -- but is still exact to float tolerance and,
computed once offline, removes the online cost entirely for anything that only
needs to *read* rho at evaluation time (the physics-imputation baseline,
Stage E's evaluation passes).  Keyed by ``(window_id, k, mask_seed)`` via a
seed independent of batch assembly (:func:`_window_dropout_seed`, sha256-based
-- Python's built-in ``hash()`` on strings is process-randomised and would make
the cache irreproducible across runs), so it is valid under any downstream
batching or shuffling, unlike the ``apply_sensor_dropout_batch`` seeding used
for reported paper metrics (which is deliberately batch-order-dependent and is
*not* what this cache reproduces -- it is a new, independent source of
dropout masks for training/caching use, not a drop-in replacement for the
paper's evaluation protocol).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from fault_localization.config.constants import WINDOW_LENGTH
from fault_localization.physics.observation import observe_raw
from fault_localization.physics.residual import flatten_table, rho


def split_window_ids(data_root: str | Path, split: str) -> list[str]:
    path = Path(data_root) / "splits" / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def group_by_event(window_ids: list[str]) -> dict[str, list[tuple[str, int]]]:
    """``{event_id: [(window_id, window_index), ...]}`` -- one file read per event."""
    out: dict[str, list[tuple[str, int]]] = {}
    for wid in window_ids:
        event, _, idx = wid.rpartition("_w")
        out.setdefault(event, []).append((wid, int(idx)))
    for entries in out.values():
        entries.sort(key=lambda item: item[1])
    return out


def build_rho_cache(
    data_root: str | Path,
    split: str,
    table: np.ndarray | torch.Tensor,
    sigma_eps,
    out_path: str | Path,
    sigma0: float = 0.0,
    reduce: str = "last",
    chunk_f: int = 1,
    batch_events: int = 32,
    limit_events: int | None = None,
    progress=None,
) -> dict:
    """Write ``cache/rho_{split}.npz``: ``window_id``, ``rho [M, 26]``, ``hyp [M, 26]``.

    ``table`` is the flattened ``[26, H, 75]`` tensor from
    :func:`residual.flatten_table`.
    """
    data_root = Path(data_root)
    if not isinstance(table, torch.Tensor):
        table = torch.as_tensor(np.asarray(table, dtype=np.float32))
    events = group_by_event(split_window_ids(data_root, split))
    keys = sorted(events)
    if limit_events is not None:
        keys = keys[:limit_events]

    ids: list[str] = []
    rho_rows: list[np.ndarray] = []
    hyp_rows: list[np.ndarray] = []

    for start in range(0, len(keys), batch_events):
        s_batch, a_batch = [], []
        for event in keys[start:start + batch_events]:
            with np.load(data_root / "raw_sim" / f"{event}.npz", allow_pickle=True) as loaded:
                record = loaded["data"]
            for wid, idx in events[event]:
                s, a = observe_raw(record, idx, WINDOW_LENGTH, reduce)
                ids.append(wid)
                s_batch.append(s)
                a_batch.append(a)
        s_t = torch.from_numpy(np.stack(s_batch))
        a_t = torch.from_numpy(np.stack(a_batch))
        values, hyp = rho(s_t, a_t, table, sigma_eps, sigma0, chunk_f=chunk_f)
        rho_rows.append(values.numpy().astype(np.float32))
        hyp_rows.append(hyp.numpy().astype(np.int32))
        if progress is not None:
            progress(min(start + batch_events, len(keys)), len(keys))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rho_all = np.concatenate(rho_rows) if rho_rows else np.zeros((0, 26), dtype=np.float32)
    hyp_all = np.concatenate(hyp_rows) if hyp_rows else np.zeros((0, 26), dtype=np.int32)
    np.savez_compressed(
        out_path,
        window_id=np.array(ids),
        rho=rho_all,
        hyp=hyp_all,
        sigma_eps=np.asarray(sigma_eps, dtype=np.float32),
        sigma0=np.float32(sigma0),
        reduce=reduce,
        policy="clean_k0",
    )
    return {"n_windows": len(ids), "n_events": len(keys), "path": str(out_path)}


def load_rho_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {"window_id": loaded["window_id"], "rho": loaded["rho"], "hyp": loaded["hyp"]}


# ---------------------------------------------------------------------------
# Full dropout-grid cache -- rho(A) for every (window, k, mask_seed)
# ---------------------------------------------------------------------------


def _window_dropout_seed(window_id: str, k: int, mask_seed: int) -> int:
    """Deterministic 32-bit seed, independent of batch order or process.

    Python's ``hash()`` on ``str`` is salted per process (``PYTHONHASHSEED``),
    so it cannot be used here -- the whole point of this cache is that the same
    window always gets the same dropped set, rebuilt or not.
    """
    key = f"{window_id}|{k}|{mask_seed}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _dropout_indices(window_id: str, k: int, mask_seed: int, n_nodes: int = 25) -> list[int]:
    if k <= 0:
        return []
    import random

    rng = random.Random(_window_dropout_seed(window_id, k, mask_seed))
    return rng.sample(range(n_nodes), min(k, n_nodes))


def build_rtilde_cache(
    data_root: str | Path,
    split: str,
    table: np.ndarray | torch.Tensor,
    sigma_eps,
    out_path: str | Path,
    k_levels: tuple[int, ...] = (0, 1, 2, 5, 10),
    n_mask_seeds: int = 3,
    sigma0: float = 0.0,
    reduce: str = "last",
    batch_events: int = 6,
    limit_events: int | None = None,
    device: torch.device | str = "cpu",
    progress=None,
) -> dict:
    """Write ``cache/rtilde_{split}.npy``: rho/hyp for every (window, k, mask_seed).

    ``rho``/``hyp`` are ``[W, len(k_levels), n_mask_seeds, 26]``.  ``k=0`` has no
    mask seed of its own (dropout is a no-op), so its one computed slice is
    replicated across the ``n_mask_seeds`` axis to keep the array rectangular;
    consumers should read offset 0 for ``k=0`` and treat the rest as duplicates,
    not additional evidence.

    Uses ``method="gram"`` (float64) -- exact to float tolerance against
    ``method="direct"`` (Gate 3's own correctness check), and the only form fast
    enough to build this offline. Batched over ``batch_events`` raw events (each
    40 windows) to bound GPU memory: at the real operator's table size
    (26 x 3366 x 75), gram's internal ``r_full`` tensor is
    ``O(batch * 26 * 3366)`` float64 elements, which OOMs the T1000 well before
    a torch DataLoader-sized batch.
    """
    data_root = Path(data_root)
    if not isinstance(table, torch.Tensor):
        table = torch.as_tensor(np.asarray(table, dtype=np.float32))
    table = table.to(device)
    events = group_by_event(split_window_ids(data_root, split))
    keys = sorted(events)
    if limit_events is not None:
        keys = keys[:limit_events]

    n_k = len(k_levels)
    ids: list[str] = []
    rho_chunks: list[np.ndarray] = []
    hyp_chunks: list[np.ndarray] = []

    for start in range(0, len(keys), batch_events):
        chunk_ids: list[str] = []
        s_batch, a_batch = [], []
        for event in keys[start:start + batch_events]:
            with np.load(data_root / "raw_sim" / f"{event}.npz", allow_pickle=True) as loaded:
                record = loaded["data"]
            for wid, idx in events[event]:
                s0, a0 = observe_raw(record, idx, WINDOW_LENGTH, reduce)
                chunk_ids.append(wid)
                s_batch.append(s0)
                a_batch.append(a0)

        s0_t = torch.from_numpy(np.stack(s_batch)).to(device=device, dtype=torch.float64)
        a0_t = torch.from_numpy(np.stack(a_batch)).to(device=device, dtype=torch.float64)
        bc = s0_t.shape[0]
        rho_chunk = np.empty((bc, n_k, n_mask_seeds, 26), dtype=np.float32)
        hyp_chunk = np.empty((bc, n_k, n_mask_seeds, 26), dtype=np.int32)

        for ki, k in enumerate(k_levels):
            n_off = 1 if k == 0 else n_mask_seeds
            for off in range(n_off):
                if k > 0:
                    a_masked = a0_t.clone()
                    for row, wid in enumerate(chunk_ids):
                        drop = _dropout_indices(wid, k, off)
                        if drop:
                            a_masked[row, drop, :] = 0.0
                else:
                    a_masked = a0_t
                rho_vals, hyp = rho(s0_t, a_masked, table, sigma_eps, sigma0, method="gram")
                rho_chunk[:, ki, off, :] = rho_vals.detach().cpu().numpy().astype(np.float32)
                hyp_chunk[:, ki, off, :] = hyp.detach().cpu().numpy().astype(np.int32)
            for off in range(n_off, n_mask_seeds):  # k=0: replicate to stay rectangular
                rho_chunk[:, ki, off, :] = rho_chunk[:, ki, 0, :]
                hyp_chunk[:, ki, off, :] = hyp_chunk[:, ki, 0, :]

        ids.extend(chunk_ids)
        rho_chunks.append(rho_chunk)
        hyp_chunks.append(hyp_chunk)
        if progress is not None:
            progress(min(start + batch_events, len(keys)), len(keys))

    rho_full = (np.concatenate(rho_chunks, axis=0) if rho_chunks
                else np.zeros((0, n_k, n_mask_seeds, 26), dtype=np.float32))
    hyp_full = (np.concatenate(hyp_chunks, axis=0) if hyp_chunks
                else np.zeros((0, n_k, n_mask_seeds, 26), dtype=np.int32))

    payload = {
        "window_id": np.array(ids),
        "rho": rho_full,
        "hyp": hyp_full,
        "k_levels": np.array(k_levels, dtype=np.int64),
        "n_mask_seeds": n_mask_seeds,
        "sigma_eps": np.asarray(sigma_eps, dtype=np.float64),
        "sigma0": float(sigma0),
        "reduce": reduce,
        "seed_scheme": "sha256(window_id|k|mask_seed)[:8] hex -> int; batch-order independent",
        "policy": "zero_messages_equivalent",  # a masked to 0, matching apply_sensor_dropout's semantics
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, payload, allow_pickle=True)
    return {"n_windows": len(ids), "n_events": len(keys), "path": str(out_path)}


def load_rtilde_cache(path: str | Path) -> dict:
    return np.load(path, allow_pickle=True).item()
