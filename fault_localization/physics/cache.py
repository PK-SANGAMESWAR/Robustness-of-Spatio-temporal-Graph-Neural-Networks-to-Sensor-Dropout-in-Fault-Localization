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

*Cache ``k = 0`` only.*  Under synthetic dropout the available set A changes, so
``rho`` changes: a cache would have to be keyed by ``(window, k, mask seed)``
(``12_CHECKLISTS.md`` §C).  Rather than store that product, the cache covers the
clean case and the layer stays online for masked inputs.
"""

from __future__ import annotations

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
