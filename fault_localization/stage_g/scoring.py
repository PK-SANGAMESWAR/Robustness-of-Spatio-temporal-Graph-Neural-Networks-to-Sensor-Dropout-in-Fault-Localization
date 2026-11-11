"""Stage G scoring: the S4.4 evaluation loop, but it **saves every prediction**.

``scripts/score_s4_4_gate4_run.py`` keeps predictions in memory and writes only F1 values.  The
event bootstrap of ``14_PREREGISTRATION.md`` section 3 resamples events, so it needs the per-window
class.  This module repeats the numerical path of that scorer (same per-window noise and dropout
seeds, same fusion arithmetic, same macro-F1) and writes ``predictions.npz`` next to a summary.
``tests/test_stage_g_scoring.py`` checks the two scorers agree on a real checkpoint.

The old scorer is not edited: the running queue has it loaded.  Its seed helpers are imported so
that "shared seeds across models" holds by construction.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from score_s4_4_gate4_run import (  # noqa: E402  (read-only import of the Gate-4 scorer helpers)
    BATCH_SIZE,
    SIGMA_LEVELS,
    dropout_seed,
    macro_f1,
    noise_seed,
)

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.masking import apply_sensor_dropout_batch  # noqa: E402
from fault_localization.dropout.noise import apply_measurement_noise_batch  # noqa: E402
from fault_localization.experiments.runner import _model as build_model_from_config  # noqa: E402
from fault_localization.models.voting import soft_vote  # noqa: E402

N_OFFSETS = 3
FULL_GRID = [(k, s) for s in SIGMA_LEVELS for k in (0, 1, 2, 5, 10)]
#: Mismatch Phase 1 (08 amendment): the 7 H2/H4 cells plus clean k = 0.
MISMATCH_GRID = [(k, s) for s in SIGMA_LEVELS for k in (0, 10)]
#: Sealed OOD set: H3 cells (k = 0 at every sigma) plus the descriptive k = 5, 10 at sigma = 0.
OOD_GRID = [(0, s) for s in SIGMA_LEVELS] + [(5, 0.0), (10, 0.0)]
#: S4 dropout policies (persistent / contiguous / targeted): k = 1, 2, 5, 10 at sigma = 0 (k = 0 is the random k = 0 cell).
POLICY_GRID = [(k, 0.0) for k in (1, 2, 5, 10)]
GRIDS = {"full": FULL_GRID, "mismatch": MISMATCH_GRID, "ood": OOD_GRID, "policy": POLICY_GRID}
POLICIES = ("random", "persistent", "contiguous", "targeted")


def is_grounded(fault_type: str) -> bool:
    """Fault types with a ground connection: AG BG CG ABG BCG CAG ABCG (7); ungrounded: AB BC CA ABC (4)."""
    return "G" in fault_type


class _PolicyContext:
    """Per-dataset objects the non-random dropout policies need (built once)."""

    def __init__(self, policy: str, dataset):
        self.policy = policy
        self.cache: dict = {}
        if policy == "contiguous":
            from fault_localization.dropout.correlated import build_adjacency
            self.adjacency = build_adjacency(dataset.edge_index)
        if policy == "targeted":
            from fault_localization.dropout.targeted import build_proximity_rank
            self.rank = build_proximity_rank(Path(dataset.root) / "processed" / "node_mapping.json")

    def apply(self, x, y, wids, k, off, batch_idx):
        """Same seed rules as the Paper-1 studies: persistent keyed on the event, the others on (k, offset, batch)."""
        if self.policy == "persistent":
            from fault_localization.dropout.persistent import apply_persistent_dropout_batch
            return apply_persistent_dropout_batch(x, wids, k, off, cache=self.cache)[0]
        seed = dropout_seed(k, off, batch_idx)
        if self.policy == "contiguous":
            from fault_localization.dropout.correlated import apply_correlated_dropout_batch
            return apply_correlated_dropout_batch(x, k, seed, self.adjacency)[0]
        from fault_localization.dropout.targeted import apply_targeted_dropout_batch
        return apply_targeted_dropout_batch(x, y, k, self.rank, seed)[0]


def cell_key(k: int, sigma: float, off: int) -> str:
    return f"k{k}_s{sigma:g}_o{off}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------------------


def load_model(run_dir: Path, edge_index, device, *, table_dir: Path | None = None,
               shuffle_seed: int | None = None):
    """Rebuild a model from its own ``config.yaml`` and load ``checkpoint_best.pt``.

    ``table_dir``: replace the (persistent) ``phys_table`` buffer by the table in that directory
    (a mismatch table).  ``shuffle_seed``: as R6-inference.  Both keep every learned weight.
    """
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if shuffle_seed is not None:
        cfg.setdefault("model", {})["shuffle_seed"] = int(shuffle_seed)
    model = build_model_from_config(cfg, edge_index).to(device)
    kept = ({k: v.clone() for k, v in model.named_buffers() if k in ("phys_table", "shuffle_map")}
            if shuffle_seed is not None else {})
    state = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=True)["model"]
    model.load_state_dict(state)
    for k, v in kept.items():
        getattr(model, k).copy_(v)
    if table_dir is not None:
        from fault_localization.physics import residual as R
        from fault_localization.physics import table as T
        if not getattr(model, "phys_table", torch.empty(0)).numel():
            raise ValueError("this model has no signature table to replace")
        tbl = T.load_table(table_dir)
        new = R.flatten_table(tbl["mu"], tbl["mu0"]).to(device)
        if new.shape != model.phys_table.shape:
            raise ValueError(f"table shape {tuple(new.shape)} != model table {tuple(model.phys_table.shape)}")
        if torch.equal(new, model.phys_table):
            raise ValueError("replacement table equals the model's own table")
        model.phys_table.copy_(new)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------------------


@dataclass
class ScoreResult:
    window_ids: np.ndarray            # [N] records order
    y_true: np.ndarray                # [N] int8
    resistance: np.ndarray            # [N] float32
    fault_type: np.ndarray            # [N] str
    event_id: np.ndarray              # [N] str
    cells: list                       # [(k, sigma)]
    n_offsets: int
    preds: dict                       # cell_key -> {"pf": int8[N], "pn": ..., "pp": ...}
    has_table: bool
    use_fusion: bool
    beta: float | None
    seconds: float


def score_dataset(model, dataset: WindowGraphDataset, cells, *, n_offsets: int = N_OFFSETS, device,
                  batch_size: int = BATCH_SIZE, max_windows: int | None = None,
                  window_subset=None, progress_every: int = 0, policy: str = "random") -> ScoreResult:
    """Score ``model`` on ``dataset`` at every ``(k, sigma)`` cell and offset; keep every prediction.

    Noise and dropout draws are seeded per window from ``(level index, offset, global window index)``
    exactly as in the Gate-4 scorer, where the level index is the position of sigma in the canonical
    ``SIGMA_LEVELS`` -- so a cell scored here sees the same corruption as in any earlier run.

    ``max_windows`` keeps the first N windows of ``dataset.records``; ``window_subset`` keeps exactly
    the given window ids (both for tests and smoke runs only).  Seeds always use the *global* index.

    ``policy``: how sensors are lost for k > 0 -- ``random`` (the S4.4 protocol, per window), or one of
    the Paper-1 stress policies ``persistent`` (per event), ``contiguous`` (connected block), ``targeted``
    (nearest the fault).  The policy changes only which sensors are zeroed.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    pctx = None if policy == "random" else _PolicyContext(policy, dataset)
    for _k, s in cells:
        if s not in SIGMA_LEVELS:
            raise ValueError(f"sigma {s} is not in the canonical grid {SIGMA_LEVELS}")
    edge_index = dataset.edge_index
    norm_std = dataset.std
    gidx_of = {wid: i for i, (wid, *_r) in enumerate(dataset.records)}
    use_fusion = bool(getattr(model, "use_fusion", False))
    has_table = bool(getattr(model, "phys_table", torch.empty(0)).numel())
    beta = float(model.beta) if use_fusion else None

    all_recs = dataset.records
    if window_subset is not None:
        keep_ids = set(window_subset)
    elif max_windows is not None:
        keep_ids = {r[0] for r in all_recs[:max_windows]}
    else:
        keep_ids = None
    kept = [i for i, r in enumerate(all_recs) if keep_ids is None or r[0] in keep_ids]
    pos_of = {g: p for p, g in enumerate(kept)}
    n_total = len(kept)
    bufs = {cell_key(k, s, o): {"pf": np.full(n_total, -1, np.int8), "pn": np.full(n_total, -1, np.int8),
                                "pp": np.full(n_total, -1, np.int8)}
            for k, s in cells for o in range(n_offsets)}
    by_sigma: dict[float, list[int]] = {}
    for k, s in cells:
        by_sigma.setdefault(s, []).append(k)

    loader = make_dataloader(dataset, batch_size, shuffle=False)
    t0 = time.perf_counter()
    seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            wids = list(batch["window_id"])
            sel = [i for i, w in enumerate(wids) if keep_ids is None or w in keep_ids]
            if not sel:
                continue
            x_full = batch["x"][sel].to(device)
            y_batch = batch["y"][sel].to(device)
            wids = [wids[i] for i in sel]
            gidx = [gidx_of[w] for w in wids]
            pos = [pos_of[g] for g in gidx]
            b = len(wids)
            seen += b
            for sigma, ks in by_sigma.items():
                s_idx = SIGMA_LEVELS.index(sigma)
                for off in range(n_offsets):
                    if sigma > 0:
                        x_noisy = x_full.clone()
                        for i in range(b):
                            x_noisy[i:i + 1] = apply_measurement_noise_batch(
                                x_full[i:i + 1], sigma, norm_std, seed=noise_seed(s_idx, off, gidx[i]))
                    else:
                        x_noisy = x_full
                    for k in ks:
                        if k > 0 and pctx is not None:
                            x_eval = pctx.apply(x_noisy.clone(), y_batch, wids, k, off, bi)
                        elif k > 0:
                            x_eval = x_noisy.clone()
                            for i in range(b):
                                x_eval[i:i + 1], _, _ = apply_sensor_dropout_batch(
                                    x_noisy[i:i + 1], k=k, seed=dropout_seed(k, off, gidx[i]),
                                    policy="zero_messages", edge_index=edge_index)
                        else:
                            x_eval = x_noisy
                        d = bufs[cell_key(k, sigma, off)]
                        if use_fusion:
                            z = model.network_logits(x_eval)
                            rho, _, _, _ = model._rho(x_eval, sigma0=sigma)
                            pf = soft_vote(z - beta * rho[:, None, :])[1]
                            pn = soft_vote(z)[1]
                            pp = rho.argmin(-1)
                        elif has_table:
                            z = model.network_logits(x_eval)
                            rho, _, _, _ = model._rho(x_eval, sigma0=sigma)
                            pf = pn = soft_vote(z)[1]
                            pp = rho.argmin(-1)
                        else:
                            pf = soft_vote(model(x_eval))[1]
                            pn = pp = None
                        d["pf"][pos] = pf.cpu().numpy().astype(np.int8)
                        if pn is not None:
                            d["pn"][pos] = pn.cpu().numpy().astype(np.int8)
                            d["pp"][pos] = pp.cpu().numpy().astype(np.int8)
            if progress_every and (bi + 1) % progress_every == 0:
                print(f"  [score] batch {bi + 1}, {seen}/{n_total} windows, "
                      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if seen != n_total:
        raise RuntimeError(f"scored {seen} windows, expected {n_total}")
    recs = [all_recs[g] for g in kept]
    for key, d in bufs.items():
        if (d["pf"] < 0).any():
            raise RuntimeError(f"cell {key}: some windows were never scored")
        if not has_table:
            del d["pn"], d["pp"]
    return ScoreResult(
        window_ids=np.array([r[0] for r in recs]),
        y_true=np.array([r[2] for r in recs], dtype=np.int8),
        resistance=np.array([float(r[3]["resistance"]) for r in recs], dtype=np.float32),
        fault_type=np.array([r[3]["fault_type"] for r in recs]),
        event_id=np.array([r[1] for r in recs]),
        cells=[tuple(c) for c in cells], n_offsets=n_offsets, preds=bufs,
        has_table=has_table, use_fusion=use_fusion, beta=beta,
        seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------
# Summaries and I/O
# ---------------------------------------------------------------------------------------


def _group_f1(y, p, mask) -> float:
    return macro_f1(y[mask], p[mask]) if mask.any() else float("nan")


def summarize(res: ScoreResult, resistances=None) -> dict:
    """Per cell: pooled macro-F1 per offset (mean, std), F1 by resistance and by grounded/ungrounded.

    Every F1 is the project's 26-class macro-F1 over *all* windows of the stratum, including the
    no-fault windows of its events (same convention as ``f1_by_resistance`` of the Gate-4 scorer).
    """
    y = res.y_true.astype(np.int64)
    grounded = np.array([is_grounded(t) for t in res.fault_type])
    rs = sorted(set(np.round(res.resistance.astype(np.float64), 6).tolist())) if resistances is None else list(resistances)
    cells = {}
    for k, s in res.cells:
        per_off, by_r, by_g, dec = [], {r: [] for r in rs}, {"grounded": [], "ungrounded": []}, []
        for o in range(res.n_offsets):
            d = res.preds[cell_key(k, s, o)]
            pf = d["pf"].astype(np.int64)
            per_off.append(macro_f1(y, pf))
            for r in rs:
                by_r[r].append(_group_f1(y, pf, np.isclose(res.resistance, r)))
            by_g["grounded"].append(_group_f1(y, pf, grounded))
            by_g["ungrounded"].append(_group_f1(y, pf, ~grounded))
            if res.has_table:
                pn, pp = d["pn"].astype(np.int64), d["pp"].astype(np.int64)
                dec.append({"f1_network_only": macro_f1(y, pn), "f1_psm": macro_f1(y, pp),
                            "fused_eq_psm": float(np.mean(pf == pp)), "fused_eq_network_only": float(np.mean(pf == pn))})
        cell = {"f1_mean": float(np.mean(per_off)), "f1_std": float(np.std(per_off)), "f1_per_offset": per_off,
                "f1_by_resistance": {str(r): float(np.nanmean(v)) for r, v in by_r.items()},
                "f1_by_group": {g: float(np.nanmean(v)) for g, v in by_g.items()}}
        for key in (dec[0] if dec else {}):
            cell[key + "_mean"] = float(np.mean([d_[key] for d_ in dec]))
        cells[f"k{k}_sigma{s:g}"] = cell
    return cells


def save_predictions(path: Path, res: ScoreResult) -> None:
    """One compressed ``.npz``: window ids, labels, strata, and an int8 class per cell and offset."""
    arrays = {"window_ids": res.window_ids, "event_id": res.event_id, "y_true": res.y_true,
              "resistance": res.resistance, "fault_type": res.fault_type,
              "cells": np.array(json.dumps({"cells": res.cells, "n_offsets": res.n_offsets}))}
    for key, d in res.preds.items():
        for name, arr in d.items():
            arrays[f"{name}__{key}"] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_predictions(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["cells"] = json.loads(str(out["cells"]))
    return out


def write_summary(out_dir: Path, name: str, res: ScoreResult, meta: dict, resistances=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(out_dir / f"{name}_predictions.npz", res)
    summary = dict(meta)
    summary.update({"n_windows": int(len(res.y_true)), "n_events": int(len(set(res.event_id.tolist()))),
                    "cells": summarize(res, resistances), "has_table": res.has_table,
                    "use_fusion": res.use_fusion, "beta": res.beta,
                    "scoring_wallclock_seconds": res.seconds,
                    "predictions_file": f"{name}_predictions.npz",
                    "predictions_sha256": sha256_file(out_dir / f"{name}_predictions.npz")})
    path = out_dir / f"{name}_scoring.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
