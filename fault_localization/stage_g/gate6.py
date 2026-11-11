"""Gate 6 analysis: the event-level paired cluster bootstrap of ``14_PREREGISTRATION.md`` Amendments 1, 3, 9, 11.

Input: the per-window predictions saved by the one-run sealed-set pipeline
(``<name>_seed<seed:05d>_predictions.npz``, models ``r5``, ``r0n`` and optionally ``r0``).

Primary measure (Amendment 3): 26-class macro-F1 pooled over all windows of the sealed set, k = 0, at each
sigma in {5e-4, 2e-3, 1e-2}; each F1 is the mean over the 3 mask/noise offsets.
Statistic (Amendment 1, approved in Amendment 9): resample the **events** with replacement (10,000 resamples,
``RandomState(20261102)``, the same resamples for every cell and model so the comparison is paired); in each
resample compute, for each seed, F1(R5) - F1(R0n); take the mean over the 3 seeds; the interval is the 2.5th and
97.5th percentile; "excludes zero" means the lower bound is above 0.
Sign rule: F1(R5) - F1(R0n) on the full data is positive for each of the seeds 42, 43, 44 (R5 and R0n paired by seed).
A sigma passes if both hold. **Gate 6 passes if all three sigma pass** (Amendment 3).

Everything else is descriptive (Amendments 3, 4, 11): sigma = 0, k = 5 and 10 at sigma = 0, F1 for each resistance, and the
grounded / ungrounded strata; R0 appears as a descriptive row. Per-resistance and per-stratum intervals use the same
resamples restricted to the events of that stratum.

Speed: the macro-F1 of a resample is computed from per-event 26 x 26 confusion matrices, ``weights @ confusions``,
instead of from 440,000 predictions.  ``tests/test_gate6.py`` checks it against ``sklearn.metrics.f1_score``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fault_localization.stage_g.scoring import cell_key, is_grounded

N_CLASSES = 26
BOOT_SEED = 20261102
PRIMARY_SIGMAS = (0.0005, 0.002, 0.01)
DESCRIPTIVE_CELLS = ((0, 0.0), (5, 0.0), (10, 0.0))


# --------------------------------------------------------------------------------------------
# Macro-F1 from confusion matrices
# --------------------------------------------------------------------------------------------


def macro_f1_from_confusion(c: np.ndarray) -> np.ndarray:
    """``c [..., 676]`` (rows = true class, columns = predicted) -> macro-F1 ``[...]`` over the 26 classes.

    Same convention as ``sklearn.metrics.f1_score(average="macro", labels=range(26), zero_division=0)``:
    a class with no true and no predicted window scores 0 and still counts in the mean.
    """
    m = c.reshape(c.shape[:-1] + (N_CLASSES, N_CLASSES)).astype(np.float64)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    fn = m.sum(axis=-1) - tp
    fp = m.sum(axis=-2) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return f1.mean(axis=-1)


def event_confusions(y: np.ndarray, pred: np.ndarray, ev_idx: np.ndarray, n_events: int) -> np.ndarray:
    """Per-event confusion matrices ``[E, 676]`` (float32)."""
    flat = ev_idx.astype(np.int64) * (N_CLASSES * N_CLASSES) + y.astype(np.int64) * N_CLASSES + pred.astype(np.int64)
    return np.bincount(flat, minlength=n_events * N_CLASSES * N_CLASSES).reshape(n_events, -1).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


class Run:
    """One model-seed's saved predictions."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.data = {k: z[k] for k in z.files}
        self.cells = json.loads(str(self.data["cells"]))
        self.n_offsets = int(self.cells["n_offsets"])

    def pred(self, k: int, sigma: float, off: int) -> np.ndarray:
        return self.data[f"pf__{cell_key(k, sigma, off)}"]


def load_runs(out_dir: Path, models=("r5", "r0n", "r0"), seeds=(42, 43, 44)) -> dict:
    """``{model: {seed: Run}}``; a model whose files are absent is left out (R0 is optional)."""
    runs: dict = {}
    for m in models:
        found = {s: Run(Path(out_dir) / f"{m}_seed{s:05d}_predictions.npz") for s in seeds
                 if (Path(out_dir) / f"{m}_seed{s:05d}_predictions.npz").exists()}
        if found:
            runs[m] = found
    for need in ("r5", "r0n"):
        if need not in runs or set(runs[need]) != set(seeds):
            raise FileNotFoundError(f"{need}: predictions for seeds {seeds} are required in {out_dir}")
    return runs


def events_and_strata(run: Run):
    """Event index per window, and per-event resistance and grounded flag (an event has one R and one fault type)."""
    ev_names, ev_idx = np.unique(run.data["event_id"], return_inverse=True)
    first = np.zeros(len(ev_names), dtype=np.int64)
    first[ev_idx[::-1]] = np.arange(len(ev_idx))[::-1]            # first window of each event
    res = run.data["resistance"][first].astype(np.float64)
    grounded = np.array([is_grounded(str(t)) for t in run.data["fault_type"][first]])
    return ev_names, ev_idx, res, grounded


# --------------------------------------------------------------------------------------------
# Bootstrap for one cell
# --------------------------------------------------------------------------------------------


def _subsets(res: np.ndarray, grounded: np.ndarray) -> dict:
    out = {"pooled": np.ones(len(res), dtype=bool)}
    for r in sorted(set(np.round(res, 6).tolist())):
        out[f"R={r:g}"] = np.isclose(res, r)
    out["grounded"] = grounded.copy()
    out["ungrounded"] = ~grounded
    return out


def analyze_cell(runs: dict, k: int, sigma: float, ev_idx, n_events, subsets: dict, seeds, n_boot: int,
                 chunk: int = 250, boot_seed: int = BOOT_SEED) -> dict:
    """Point estimates for every model and a bootstrap interval of the seed-mean ΔF1 (R5 − R0n) per subset."""
    n_off = runs["r5"][seeds[0]].n_offsets
    y = runs["r5"][seeds[0]].data["y_true"]
    conf: dict = {}                                           # (model, seed, off) -> [E, 676]; only r5 and r0n are bootstrapped
    point: dict = {name: {m: {s: [] for s in seeds} for m in runs} for name in subsets}
    for m, by_seed in runs.items():
        for s in seeds:
            for off in range(n_off):
                c = event_confusions(y, by_seed[s].pred(k, sigma, off), ev_idx, n_events)
                for name, mask in subsets.items():
                    point[name][m][s].append(float(macro_f1_from_confusion(c[mask].sum(axis=0))))
                if m in ("r5", "r0n"):
                    conf[(m, s, off)] = c
    boot = {name: np.zeros(n_boot) for name in subsets} if n_boot else {}
    rng = np.random.RandomState(boot_seed)
    probs = np.full(n_events, 1.0 / n_events)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        w = rng.multinomial(n_events, probs, size=b).astype(np.float32)             # [b, E] event counts
        for name, mask in subsets.items():
            wm = w if mask.all() else w[:, mask]
            acc = np.zeros(b)
            for s in seeds:
                d = np.zeros(b)
                for off in range(n_off):
                    f5 = macro_f1_from_confusion(wm @ conf[("r5", s, off)][mask])
                    f0 = macro_f1_from_confusion(wm @ conf[("r0n", s, off)][mask])
                    d += (f5 - f0) / n_off
                acc += d / len(seeds)
            boot[name][done:done + b] = acc
        done += b
    result = {}
    for name, mask in subsets.items():
        f1 = {m: [float(np.mean(point[name][m][s])) for s in seeds] for m in runs}
        delta = [a - b for a, b in zip(f1["r5"], f1["r0n"])]
        entry = {"n_events": int(mask.sum()), "f1": f1, "delta_per_seed": delta, "delta_mean": float(np.mean(delta)),
                 "sign_all_seeds_positive": bool(all(d > 0 for d in delta))}
        if n_boot:
            lo, hi = np.percentile(boot[name], [2.5, 97.5])
            entry.update({"ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
                          "ci_lower_above_zero": bool(lo > 0), "boot_mean": float(boot[name].mean())})
        result[name] = entry
    return result


def run_analysis(out_dir: Path, seeds=(42, 43, 44), n_boot: int = 10000, chunk: int = 250,
                 cells=None) -> dict:
    """The whole Gate 6 analysis for a pipeline output directory."""
    runs = load_runs(out_dir, seeds=seeds)
    ref = runs["r5"][seeds[0]]
    for m, by_seed in runs.items():
        for s, r in by_seed.items():
            if not np.array_equal(r.data["window_ids"], ref.data["window_ids"]):
                raise ValueError(f"{m} seed {s}: windows differ from the reference run")
    ev_names, ev_idx, res, grounded = events_and_strata(ref)
    subsets = _subsets(res, grounded)
    cells = cells or ([(0, s) for s in PRIMARY_SIGMAS] + list(DESCRIPTIVE_CELLS))
    out_cells = {}
    for k, s in cells:
        out_cells[f"k{k}_sigma{s:g}"] = analyze_cell(runs, k, s, ev_idx, len(ev_names), subsets, list(seeds), n_boot, chunk)
    per_sigma = {}
    for s in PRIMARY_SIGMAS:
        key = f"k0_sigma{s:g}"
        if key not in out_cells or not n_boot:
            continue
        p = out_cells[key]["pooled"]
        per_sigma[f"{s:g}"] = {"delta_per_seed": p["delta_per_seed"], "delta_mean": p["delta_mean"], "ci95": p["ci95"],
                               "sign_all_seeds_positive": p["sign_all_seeds_positive"],
                               "ci_lower_above_zero": p["ci_lower_above_zero"],
                               "passes": bool(p["sign_all_seeds_positive"] and p["ci_lower_above_zero"])}
    evaluable = len(seeds) == 3 and n_boot >= 10000 and len(per_sigma) == len(PRIMARY_SIGMAS)
    gate = {"evaluable": bool(evaluable), "passed": bool(evaluable and all(v["passes"] for v in per_sigma.values())),
            "per_sigma": per_sigma,
            "rule": "Gate 6 passes if, at each sigma in {5e-4, 2e-3, 1e-2} (k=0, pooled over R), F1(R5)-F1(R0n) is positive in all "
                    "3 seeds and the 95% event-bootstrap CI of the seed-mean difference has a lower bound above 0."}
    return {"seeds": list(seeds), "n_boot": n_boot, "boot_seed": BOOT_SEED, "n_events": int(len(ev_names)),
            "n_windows": int(len(ev_idx)), "models": sorted(runs), "gate6": gate, "cells": out_cells}
