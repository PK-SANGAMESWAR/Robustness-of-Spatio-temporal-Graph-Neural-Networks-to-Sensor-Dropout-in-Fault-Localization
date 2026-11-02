"""Perturbed feeder models and signature tables for the S6.2 model-mismatch study.

``08_STAGE_G_eval.md`` S6.2 (as amended 2026-11-02): the trained models and the corpus stay
untouched; only the signature table changes.  Phase 1 builds three tables:

======== =============================================================================================
name     perturbation (applied to a *copy* of the IEEE-123 feeder files)
======== =============================================================================================
tap_p1   every ``RegControl`` ``Vreg`` raised by one tap step = 0.75 V (5/8 % of 120 V)
tap_m1   every ``RegControl`` ``Vreg`` lowered by one tap step
imp5     every non-switch ``Line``: R and X matrices multiplied by 1 + 0.05 s, s = +-1 drawn per line
======== =============================================================================================

The tap perturbation moves the regulator *set point* by one step.  The controllers (which the
table builder keeps, to match the corpus) then settle one step away from the nominal position
in many, not all, hypotheses; the realised tap change and the realised voltage shift are
measured and reported by :func:`tap_positions` and :func:`table_shift`, never assumed.

Nothing here edits ``fault_localization/data/ieee123/`` or ``operator/``: the perturbed feeder
is a copy under the output directory, and the table is written next to it.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FEEDER_DIR = REPO / "fault_localization" / "data" / "ieee123"
MASTER_NAME = "IEEE123Master.dss"
TAP_STEP_VOLTS = 0.75           # one regulator step: 0.625 % of the 120 V control base
IMPEDANCE_SEED = 20261102       # fixed before any run; see 14_PREREGISTRATION.md amendment 6

#: name -> (kind, level).  ``level`` is +-1 step for taps and the fractional size for impedance.
MISMATCH_LEVELS: dict[str, tuple[str, float]] = {
    "tap_p1": ("tap", +1.0),
    "tap_m1": ("tap", -1.0),
    "imp5": ("impedance", 0.05),
}


def copy_feeder(dst_dir: str | Path) -> Path:
    """Copy the nominal feeder files into ``dst_dir`` and return the new master path."""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for src in FEEDER_DIR.iterdir():
        if src.is_file() and src.suffix.lower() == ".dss":
            shutil.copy2(src, dst / src.name)
    master = dst / MASTER_NAME
    if not master.exists():
        raise FileNotFoundError(master)
    return master


def _matrix_text(flat: list[float], n: int) -> str:
    """Row-major ``n*n`` list -> OpenDSS lower-triangular text ``(a | b c | d e f)``."""
    rows = []
    for i in range(n):
        rows.append(" ".join(f"{flat[i * n + j]:.10g}" for j in range(i + 1)))
    return "(" + " | ".join(rows) + ")"


def _compile_nominal(master: Path):
    import opendssdirect as dss
    from fault_localization.simulation.feeder import compile_feeder
    compile_feeder(master)
    return dss


def tap_edit_lines(master: Path, steps: float) -> list[str]:
    """``Edit regcontrol.<n> vreg=<v + steps * 0.75>`` for every regulator control."""
    dss = _compile_nominal(master)
    lines = []
    for name in dss.RegControls.AllNames():
        dss.RegControls.Name(name)
        vreg = float(dss.RegControls.ForwardVreg())
        lines.append(f"Edit RegControl.{name} vreg={vreg + steps * TAP_STEP_VOLTS:.6f}")
    if not lines:
        raise RuntimeError("no RegControl found in the feeder")
    return lines


def impedance_factors(names: list[str], level: float, seed: int = IMPEDANCE_SEED) -> dict[str, float]:
    """Per-line factor ``1 + level * s`` with s = +-1 drawn independently per line (fixed seed)."""
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=len(names))
    return {n: float(1.0 + level * s) for n, s in zip(names, signs)}


def impedance_edit_lines(master: Path, level: float, seed: int = IMPEDANCE_SEED) -> tuple[list[str], dict]:
    """``Edit Line.<n> Rmatrix=.. Xmatrix=..`` for every non-switch line; C matrix untouched."""
    dss = _compile_nominal(master)
    entries = []
    dss.Lines.First()
    while True:
        if not dss.Lines.IsSwitch():
            entries.append((dss.Lines.Name(), int(dss.Lines.Phases()),
                            list(dss.Lines.RMatrix()), list(dss.Lines.XMatrix())))
        if not dss.Lines.Next():
            break
    factors = impedance_factors([e[0] for e in entries], level, seed)
    lines = []
    for name, n, r, x in entries:
        f = factors[name]
        lines.append(f"Edit Line.{name} Rmatrix={_matrix_text([f * v for v in r], n)} "
                     f"Xmatrix={_matrix_text([f * v for v in x], n)}")
    return lines, {"n_lines": len(entries), "n_plus": int(sum(v > 1 for v in factors.values())),
                   "n_minus": int(sum(v < 1 for v in factors.values())), "seed": seed, "level": level}


def make_perturbed_feeder(name: str, out_root: str | Path) -> dict:
    """Write ``<out_root>/<name>/feeder/`` (nominal copy + appended ``Edit`` lines).

    The edit lines go after ``CalcVoltageBases`` at the end of the master, so the voltage bases
    stay the nominal ones and every per-unit conversion is unchanged.
    """
    kind, level = MISMATCH_LEVELS[name]
    feeder_dir = Path(out_root) / name / "feeder"
    master = copy_feeder(feeder_dir)
    nominal_text = master.read_text()
    extra: dict = {}
    if kind == "tap":
        edits = tap_edit_lines(master, level)
    else:
        edits, extra = impedance_edit_lines(master, level)
    master.write_text(nominal_text.rstrip("\n") + "\n\n! --- S6.2 mismatch perturbation: " + name
                      + " ---\n" + "\n".join(edits) + "\n")
    spec = {"name": name, "kind": kind, "level": level, "n_edit_lines": len(edits),
            "master": str(master), **extra}
    (feeder_dir.parent / "perturbation.json").write_text(json.dumps(spec, indent=2))
    return spec


def build_mismatch_table(name: str, out_root: str | Path, workers: int = 4) -> dict:
    """Perturbed feeder + full signature table (same grid, mode and recompile-per-solve as nominal)."""
    from fault_localization.physics.table import build_table
    spec = make_perturbed_feeder(name, out_root)
    meta = build_table(Path(out_root) / name / "operator", workers=workers, mode="daily",
                       master=spec["master"], reuse_limit=1)
    spec["table"] = {k: meta[k] for k in ("n_solves", "n_non_converged", "n_nan", "seconds", "complete")}
    (Path(out_root) / name / "perturbation.json").write_text(json.dumps(spec, indent=2))
    return spec


# --------------------------------------------------------------------------------------
# Diagnostics: how large is the mismatch actually?
# --------------------------------------------------------------------------------------


def table_shift(nominal_dir: str | Path, other_dir: str | Path) -> dict:
    """Distribution of ``|mu_other - mu_nominal|`` (p.u.) over every table entry."""
    a = np.load(Path(nominal_dir) / "mu.npy", mmap_mode="r")
    b = np.load(Path(other_dir) / "mu.npy", mmap_mode="r")
    if a.shape != b.shape:
        raise ValueError(f"table shapes differ: {a.shape} vs {b.shape}")
    d = np.abs(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)).ravel()
    d = d[np.isfinite(d)]
    q = np.quantile(d, [0.5, 0.9, 0.99])
    return {"n": int(d.size), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(d.max())}


def tap_positions(master: str | Path, loads=(0.5, 0.8, 1.0, 1.3)) -> dict:
    """Regulator tap positions after the no-fault Daily solve, per load level (CPU, one worker)."""
    import opendssdirect as dss
    from fault_localization.physics import signature_solver as ss
    out: dict[str, dict[str, float]] = {}
    for level in loads:
        ss.reset_engine()
        ss.solve_nofault(float(level), mode="daily", master_path=Path(master), reuse_limit=1)
        taps = {}
        dss.RegControls.First()
        while True:
            tr = dss.RegControls.Transformer()
            dss.Transformers.Name(tr)
            taps[tr] = float(dss.Transformers.Tap())
            if not dss.RegControls.Next():
                break
        out[f"{level:g}"] = taps
    return out


# --------------------------------------------------------------------------------------
# CPU pre-check of the true-class residual (06_STAGE_E R6 record, 2026-11-02)
# --------------------------------------------------------------------------------------


def load_precheck_windows(root: str | Path, split: str = "val", n: int = 320, seed: int = 7):
    """Fixed random sample of ``n`` fault windows with >= 1 post-inception sample (w >= 21).

    Streams ``labels.csv`` instead of building the full dataset object (about 2 GB of metadata
    for the 1.65 M-window corpus).  Returns ``(x [n,25,20,6] normalised, y [n], norm_stats,
    window_ids)``; the normalisation is the dataset's own (voltage channels 0-2 only).
    """
    import csv
    root = Path(root)
    ids = [t.strip() for t in (root / "splits" / f"{split}.txt").read_text().splitlines() if t.strip()]
    eligible = [w for w in ids if int(w.rsplit("_w", 1)[1]) >= 21]
    rng = np.random.RandomState(seed)
    chosen = sorted(rng.choice(len(eligible), size=n, replace=False).tolist())
    wanted = {eligible[i] for i in chosen}
    labels: dict[str, int] = {}
    with (root / "processed" / "labels.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["window_id"] in wanted:
                labels[row["window_id"]] = int(row["label"])
    stats = json.loads((root / "processed" / "norm_stats.json").read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    xs, ys, wids = [], [], []
    cache: dict[str, tuple] = {}
    for i in chosen:
        wid = eligible[i]
        event = wid.rsplit("_w", 1)[0]
        if event not in cache:
            with np.load(root / "processed" / "windows" / f"{event}.npz") as z:
                cache = {event: (z["windows"], {str(w): j for j, w in enumerate(z["window_ids"])})}
        windows, pos = cache[event]
        w = windows[pos[wid]].astype(np.float32)
        w[..., :3] = (w[..., :3] - mean) / std
        xs.append(w)
        ys.append(labels[wid])
        wids.append(wid)
    if any(y == 0 for y in ys):
        raise RuntimeError("a pre-check window has label 0; the fault-window filter is wrong")
    return np.stack(xs), np.asarray(ys), (mean, std), wids


def true_class_residual(table_dir: str | Path, x, y, norm_stats, sigma_eps: float,
                        sigma0s=(0.0, 0.01)) -> dict:
    """rho at the true class for windows ``x`` [B,25,20,6] (normalised) and labels ``y``.

    ``rho`` is the shifted, halved residual of ``01_SPEC.md`` section 4 (>= 0, one exact zero).
    k = 0 (no synthetic dropout): the availability is the window's own static mask.
    """
    import torch
    from fault_localization.physics import residual as R
    from fault_localization.physics import table as T
    from fault_localization.physics.observation import observe
    tbl = T.load_table(table_dir)
    flat = R.flatten_table(tbl["mu"], tbl["mu0"])
    mean, std = norm_stats
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    out = {}
    with torch.no_grad():
        s, a = observe(xt, mean, std, "last")
        for sigma0 in sigma0s:
            r_raw, _ = R.raw_residual(s, a, flat, sigma_eps, sigma0)
            rho = R._shift(r_raw)
            idx = torch.arange(len(yt))
            true = rho[idx, yt].double().numpy()
            raw = (r_raw[idx, yt] * 0.5).double().numpy()      # unshifted, halved: same units as rho
            out[f"sigma0={sigma0:g}"] = {
                # rho is shifted by the minimum over classes, so it is exactly 0 whenever the true
                # class wins: its median is 0 for any table that still classifies > 50 % correctly.
                "median": float(np.median(true)), "mean": float(true.mean()),
                "p90": float(np.quantile(true, 0.9)), "max": float(true.max()),
                "psm_accuracy": float((rho.argmin(-1) == yt).double().mean()),
                "raw_half_median": float(np.median(raw)), "raw_half_mean": float(raw.mean()),
                "raw_half_p90": float(np.quantile(raw, 0.9)),
                "n": int(len(yt)),
            }
    return out
