"""G-TF -- a training-free gate on the R5 checkpoints (``PC-STGNN/stage_f_training_free_gate_draft.md``).

R5 is not retrained. The S5.1 detector (PSM refit, ``physics/residual.py`` via
``run_silent_detection.py``) flags sensors; a flagged sensor is dropped (all six channels zeroed,
the ``zero_messages`` operation of ``dropout/masking.py``). Four arms are scored on the same
corrupted validation windows: no gate, G-TF, oracle (drops exactly the corrupted sensors), and
-- on clean windows -- no gate vs G-TF (the false-suppression cost).

Pre-registered decisions (draft § 2, frozen before any G-TF result):
  a. The validation windows are split into halves A and B **by fault event** (event = the window
     id up to ``_w``), with a fixed seed (``SPLIT_SEED``). No event is in both halves.
  b. The per-sensor threshold is chosen on the **clean** windows of half A. Scoring is on half B only.
  c. Detector: LOO only if its AUROC is >= 0.01 higher than the in-sample AUROC for bias +-2 % at
     sigma0 = 1e-2 for each of s = 1, 2, 3; otherwise in-sample. The rule is applied once, by
     ``decide_detector`` after the sigma0 = 1e-2 S5.2 result is complete, and recorded in
     ``results/stage_f/gtf/detector_decision.json``.
  d. Per-sensor false-positive rate 1 % (``FPR_TARGET``). The realized rate is reported on clean
     windows and on the clean sensors of corrupted windows.
  e. Every result is reported for each corruption type separately.
  f. Core scope: all five corruption types, sigma0 = 1e-2, s = 1, 2, 3, seeds 42/43/44, plus the clean-window arm.
     The decision rule uses bias +-2 % only. rec is defined only if (oracle - no gate) >= 0.01.
  g. Full scope (5 types x s x 4 sigma0) runs only if ``results/stage_f/gtf/FULL_SCOPE_APPROVED`` exists.

Stages (each resumable; a stage skips when its output exists):
  split       CPU. Writes split.json (event lists hashed, balance report).
  thresholds  CPU. Per-sensor thresholds from clean half-A windows, and clean half-B flags.
  score       GPU. R5 passes for one seed. Needs the detector decision and the S5.1 raw scores.
  summarize   CPU. Decision rule, recovered fraction, paired t over seeds, Holm.
  --check-ready  prints whether the pre-registration is complete (used by the queue).

G-TF2 (``PC-STGNN/stage_f_gtf2_preregistration.md``, ``--gate gtf2``) adds a second detector, the temporal
flatline score (``flatline_score``: minus the standard deviation of a sensor's readings over the window), and
flags a sensor if EITHER detector fires (``detector_flags``). Each detector has its own per-sensor threshold at
``FPR_GTF2`` = 0.5 % on the clean half-A windows. ``--gate gtf2r`` scores the extra arm "residual only at 0.5 %"
(detector 1 alone, the G-TF2 thresholds, the gated arm only; no-gate and oracle come from the G-TF2 result). With
``--gate gtf`` (the default) nothing changes: one detector, 1 % threshold, the same files, the same numbers.

This script never touches the test split or the sealed OOD set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

SPLIT_SEED = 7
FPR_TARGET = 0.01
LOO_MARGIN = 0.01                 # decision c
SEEDS = (42, 43, 44)
SIGMA_ALL = (0.0, 0.0005, 0.002, 0.01)
SIGMA_CORE = (0.01,)
CORE_TYPES = ("stuck", "bias_2pct", "bias_5pct", "gauss_0.02", "swap")   # all five types at sigma0 = 1e-2
MIN_DENOMINATOR = 0.01            # rec is defined only if (oracle - no gate) >= 0.01
S_LEVELS = (1, 2, 3)
OUT_ROOT = REPO / "results/stage_f/gtf"
DECISION_PATH = OUT_ROOT / "detector_decision.json"
FULL_APPROVAL_PATH = OUT_ROOT / "FULL_SCOPE_APPROVED"
GATE_DOC = REPO / "PC-STGNN/07_STAGE_F_gate.md"
DRAFT_DOC = REPO / "PC-STGNN/stage_f_training_free_gate_draft.md"
READY_MARKER = "added after s5.1, before any g-tf result"
FPR_GTF2 = 0.005                  # G-TF2: per-sensor false-positive rate of EACH detector (the union is at most ~1 %)
OUT_ROOT_GTF2 = REPO / "results/stage_f/gtf2"
OUT_ROOT_GTF2R = REPO / "results/stage_f/gtf2r"      # residual-only arm at 0.5 % (descriptive; not in a decision rule)
GTF2_DOC = REPO / "PC-STGNN/stage_f_gtf2_preregistration.md"
READY_MARKER_GTF2 = "before any g-tf2 result"
GTF2_TYPES = ("stuck", "bias_2pct", "bias_5pct", "gauss_0.02", "swap")   # same core scope as G-TF
GTF2_PRIMARY = "stuck"
GTF2_SECONDARY = ("bias_2pct", "bias_5pct", "swap")                       # Gaussian is reported only
SUB_BATCH = 32                    # R5 forward chunk, as score_s4_4_gate4_run.BATCH_SIZE
S1_BATCH = 512                    # loader batch of the S5.1 scripts; raw arrays are indexed by it


# --------------------------------------------------------------------------------------------
# pure helpers (no torch needed; unit-tested)
# --------------------------------------------------------------------------------------------

def event_of(window_id: str) -> str:
    return window_id.rsplit("_w", 1)[0]


def split_events(events, seed: int = SPLIT_SEED):
    """Half A / half B by event. Returns (set_A, set_B). Deterministic in (events, seed)."""
    ev = sorted(set(events))
    perm = np.random.RandomState(seed).permutation(len(ev))
    n_a = len(ev) // 2
    a = {ev[i] for i in perm[:n_a]}
    b = {ev[i] for i in perm[n_a:]}
    assert not (a & b) and len(a) + len(b) == len(ev)
    return a, b


def split_hash(a_events) -> str:
    return hashlib.sha256("\n".join(sorted(a_events)).encode()).hexdigest()


def per_sensor_thresholds(scores: np.ndarray, avail: np.ndarray, fpr: float = FPR_TARGET) -> np.ndarray:
    """Per-sensor threshold so that ``fpr`` of that sensor's available clean scores exceed it."""
    thr = np.full(scores.shape[1], np.inf)
    for j in range(scores.shape[1]):
        v = scores[avail[:, j], j]
        if v.size:
            thr[j] = np.quantile(v, 1.0 - fpr)
    return thr


def flags_from_scores(scores: np.ndarray, avail: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (scores > thr[None, :]) & avail.astype(bool)


def decide_detector(s52: dict, sigma_key: str = "bias_2pct", margin: float = LOO_MARGIN) -> dict:
    """Decision c. ``s52`` is ``silent_detection.json['s5_2_psm_auroc']`` at sigma0 = 1e-2."""
    diffs = {f"s{s}": s52[f"{sigma_key}_s{s}"]["loo"] - s52[f"{sigma_key}_s{s}"]["ins"] for s in S_LEVELS}
    use_loo = all(d >= margin for d in diffs.values())
    return {"detector": "loo" if use_loo else "ins", "loo_minus_ins": diffs, "margin": margin,
            "rule": "LOO only if LOO AUROC - in-sample AUROC >= margin at bias_2pct for each of s=1,2,3"}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def recovered_fraction(f_gtf: float, f_nogate: float, f_oracle: float, min_den: float = MIN_DENOMINATOR) -> dict:
    """rec = (G-TF - no gate) / (oracle - no gate); undefined (None) if the denominator is < ``min_den``."""
    den = f_oracle - f_nogate
    return {"denominator": den, "recovered": (None if den < min_den else (f_gtf - f_nogate) / den)}


def gate_zero(x, flags):
    """Zero all channels (voltage and mask) of the flagged nodes. ``x`` [B,N,S,6], ``flags`` [B,N] bool."""
    out = x.clone()
    out[flags] = 0.0
    return out


def flatline_score(x, mean, std) -> np.ndarray:
    """Temporal flatline score, [B, N]: minus the mean over a sensor's instrumented phases of the standard deviation
    of its voltage readings over the window (p.u., population std over the S steps). A flat sensor has score 0,
    a live one is negative; HIGH score = flat, so ``score > threshold`` flags a sensor whose std is BELOW its
    threshold. ``x`` [B,N,S,6] is the loader tensor (z-scored voltages in channels 0-2, static instrumentation
    mask in 3-5); ``mean``/``std`` are the voltage normalisation statistics. Same definition as the exploratory
    ``scripts/run_temporal_detector_exploratory.py``."""
    import torch
    mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32))
    std_t = torch.as_tensor(np.asarray(std, dtype=np.float32))
    volts = x[..., :3] * std_t + mean_t                         # [B,N,S,3] p.u.
    sd = volts.std(dim=2, unbiased=False)                       # [B,N,3]
    ok = (x[:, :, 0, 3:] > 0.5).float()                         # [B,N,3] instrumented phases
    n_ok = ok.sum(-1).clamp(min=1)
    return (-(sd * ok).sum(-1) / n_ok).numpy()


def detector_flags(ins_scores, flat_scores, avail, thr_ins, thr_flat=None) -> dict:
    """Flags of each detector and of their union. ``flat_scores``/``thr_flat`` None = flatline detector OFF:
    the only entry is ``union`` and it equals ``flags_from_scores`` exactly (G-TF). With the flatline detector
    on (G-TF2) a sensor is flagged if its residual is above its threshold OR its std is below its threshold;
    an unavailable sensor is never flagged."""
    ins = flags_from_scores(ins_scores, avail, thr_ins)
    if thr_flat is None:
        return {"union": ins}
    flat = flags_from_scores(flat_scores, avail, thr_flat)
    return {"ins": ins, "flat": flat, "union": ins | flat}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def git_clean_and_tracked(path: Path) -> bool:
    rel = _rel(path)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO).returncode == 0
    return tracked and clean


def raw_dir(sigma: float) -> Path:
    import run_silent_detection as S1
    import run_silent_detection_noise as SN
    return (S1.OUT_DIR / "raw") if sigma == 0.0 else (SN._out_dir(sigma) / "raw")


def sigma_summary_path(sigma: float) -> Path:
    import run_silent_detection as S1
    import run_silent_detection_noise as SN
    return (S1.OUT_DIR / "silent_detection.json") if sigma == 0.0 else (SN._out_dir(sigma) / "silent_detection.json")


def check_ready(scope: str = "core") -> tuple[bool, list[str]]:
    """True only when the pre-registration is complete (Part 3 step 5): the detector decision is
    recorded, the G-TF arm is in 07_STAGE_F_gate.md with the 'before any G-TF result' marker, and
    both documents are committed and unmodified. Also checks the inputs exist."""
    problems: list[str] = []
    sigmas = SIGMA_CORE if scope == "core" else SIGMA_ALL
    if not GATE_DOC.exists() or READY_MARKER not in GATE_DOC.read_text(encoding="utf-8").lower():
        problems.append("07_STAGE_F_gate.md has no G-TF arm marked 'added after S5.1, before any G-TF result'")
    for doc in (GATE_DOC, DRAFT_DOC):
        if not doc.exists() or not git_clean_and_tracked(doc):
            problems.append(f"{_rel(doc)} is not committed or has uncommitted changes")
    if not DECISION_PATH.exists():
        problems.append("detector decision not recorded (results/stage_f/gtf/detector_decision.json)")
    else:
        d = json.loads(DECISION_PATH.read_text())
        if d.get("detector") not in ("loo", "ins"):
            problems.append("detector decision file has no valid 'detector'")
    for sg in sigmas:
        if not (raw_dir(sg) / "COMPLETE.json").exists():
            problems.append(f"S5.1 raw scores incomplete for sigma0={sg}")
        if not sigma_summary_path(sg).exists():
            problems.append(f"S5.1 summary missing for sigma0={sg}")
    for s in SEEDS:
        if not (REPO / f"results/s4_4/r5/pcstgnn/seed_{s:05d}/checkpoint_best.pt").exists():
            problems.append(f"R5 checkpoint missing for seed {s}")
    if scope == "full" and not FULL_APPROVAL_PATH.exists():
        problems.append("full scope needs results/stage_f/gtf/FULL_SCOPE_APPROVED (author approval)")
    return (not problems), problems


def check_ready_gtf2() -> tuple[bool, list[str]]:
    """G-TF2 may be scored only when its pre-registration is committed and unmodified and the arm is marked
    'before any G-TF2 result' in 07_STAGE_F_gate.md. This function is not called by any queue."""
    problems: list[str] = []
    if not GATE_DOC.exists() or READY_MARKER_GTF2 not in GATE_DOC.read_text(encoding="utf-8").lower():
        problems.append("07_STAGE_F_gate.md has no G-TF2 arm marked 'before any G-TF2 result'")
    for doc in (GATE_DOC, GTF2_DOC):
        if not doc.exists() or not git_clean_and_tracked(doc):
            problems.append(f"{_rel(doc)} is not committed or has uncommitted changes")
    if not (raw_dir(0.01) / "COMPLETE.json").exists():
        problems.append("S5.1 raw scores incomplete for sigma0=0.01")
    for s in SEEDS:
        if not (REPO / f"results/s4_4/r5/pcstgnn/seed_{s:05d}/checkpoint_best.pt").exists():
            problems.append(f"R5 checkpoint missing for seed {s}")
    return (not problems), problems


def check_gate_equals_training_dropout(data_root: Path | None = None) -> dict:
    """Consistency check 1: the G-TF gate action is the same operation as the dropout used in
    training (``dropout/training.py`` -> ``apply_sensor_dropout_batch``, policy zero_messages).
    For several k, drop k random nodes with the training function, build the flags from the nodes it
    reports, and require ``gate_zero`` to reproduce its tensor bit for bit, to leave the mask channels
    at 0 for the dropped nodes, and (on CPU, R5 seed 42) to give identical R5 predictions."""
    import torch
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    from fault_localization.dropout.masking import apply_sensor_dropout_batch
    from fault_localization.physics import observation as OBS
    data_root = data_root or (REPO / "dataset_fullscale_50")
    ds = WindowGraphDataset(str(data_root), "val")
    loader = make_dataloader(ds, S1_BATCH, shuffle=False)
    mean, std = OBS.load_norm_stats(data_root)
    batch = next(iter(loader))
    x = batch["x"][:64]
    res = {"tensor_equal": {}, "mask_zero": {}, "r5_pred_equal": None}
    for k in (1, 2, 3, 5, 10):
        xd, _, nodes = apply_sensor_dropout_batch(x, k=k, seed=1234 + k, policy="zero_messages", edge_index=None)
        flags = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
        for i, n in enumerate(nodes):
            flags[i, n] = True
        xg = gate_zero(x, flags)
        res["tensor_equal"][k] = bool(torch.equal(xg, xd))
        _, a = OBS.observe(xg, mean, std, reduce="last")
        res["mask_zero"][k] = bool((a.sum(dim=-1)[flags] == 0).all())
        if k == 5:
            ckpt = REPO / "results/s4_4/r5/pcstgnn/seed_00042/checkpoint_best.pt"
            if ckpt.exists():
                model = _load_r5(42, torch.device("cpu"))
                res["r5_pred_equal"] = _predict(model, xg[:24], 0.0, torch.device("cpu")) == _predict(model, xd[:24], 0.0, torch.device("cpu"))
    res["passed"] = all(res["tensor_equal"].values()) and all(res["mask_zero"].values()) and res["r5_pred_equal"] is not False
    return res


def check_stored_scores(sigma: float, batch_ids=(0, 150, 300, 450, 600, 774), data_root: Path | None = None,
                        threads: int = 2, loo_chunk: int = 128) -> dict:
    """Consistency check 2: the detector scores stored by the S5.1 run equal the scores recomputed
    here (regenerate the noisy, corrupted input of every condition for the chosen loader batches and
    run the same ``rho`` / PSM refit / LOO and in-sample code) -- and the regenerated corrupted-sensor
    labels equal the stored ones. Reports the largest absolute difference."""
    import torch
    import run_silent_detection as S1
    from fault_localization.dropout.silent import apply_silent_failure_batch
    data_root = data_root or (REPO / "dataset_fullscale_50")
    ds, loader, mean, std, table, _ = _setup(data_root, threads)
    raw = S1.open_raw(raw_dir(sigma), 0, "r")
    want = set(batch_ids)
    pos, out = 0, {"sigma0": sigma, "batches": [], "max_abs_diff": 0.0, "labels_equal": True, "n_windows": 0, "conditions": len(S1.conditions())}
    for bi, batch in enumerate(loader):
        b = batch["x"].shape[0]
        if bi in want:
            sl = slice(pos, pos + b)
            xn = _noisy(batch["x"], sigma, std, bi)
            for cname, kind, sev, n_s in S1.conditions():
                x_c, corrupted = apply_silent_failure_batch(xn, n_s, kind, seed=S1.CORRUPTION_SEED * 10000 + bi,
                                                            norm_stats=(mean, std), severity=sev)
                label = np.zeros((b, 25), bool)
                for row, nodes in enumerate(corrupted):
                    label[row, nodes] = True
                out["labels_equal"] &= bool(np.array_equal(label, np.asarray(raw[f"{cname}__label"][sl]).astype(bool)))
                lo, ins, _ = _clean_scores(x_c, sigma, mean, std, table, loo_chunk)
                for nm, new in (("psm_loo", lo), ("psm_ins", ins)):
                    d = float(np.abs(np.asarray(raw[f"{cname}__{nm}"][sl]) - new).max())
                    out["max_abs_diff"] = max(out["max_abs_diff"], d)
            out["batches"].append(bi); out["n_windows"] += b
        pos += b
    out["passed"] = out["labels_equal"] and out["max_abs_diff"] == 0.0 and bool(out["batches"])
    return out


def scope_conditions(scope: str):
    import run_silent_detection as S1
    conds = S1.conditions()
    if scope == "core":
        conds = [c for c in conds if c[0].rsplit("_s", 1)[0] in CORE_TYPES]
    return conds


def scope_sigmas(scope: str):
    return SIGMA_CORE if scope == "core" else SIGMA_ALL


# --------------------------------------------------------------------------------------------
# data plumbing shared by the stages
# --------------------------------------------------------------------------------------------

def _sigma_idx(sigma: float) -> int | None:
    import run_silent_detection_noise as SN
    return None if sigma == 0.0 else list(SN.SIGMA_GRID).index(sigma)


def _noisy(x, sigma, std, bi):
    """The S5.1 noise rerun's own noise draw: whole loader batch, seed (NOISE_SEED+idx)*10000+bi."""
    import run_silent_detection_noise as SN
    from fault_localization.dropout.noise import apply_measurement_noise_batch
    if sigma == 0.0:
        return x
    return apply_measurement_noise_batch(x, sigma, std, seed=(SN.NOISE_SEED + _sigma_idx(sigma)) * 10000 + bi)


def _load_split(ds, out_root: Path):
    a_events, b_events = split_events([r[1] for r in ds.records])
    return a_events, b_events


def stage_split(out_root: Path, data_root: Path) -> dict:
    from fault_localization.data.dataset import WindowGraphDataset
    path = out_root / "split.json"
    ds = WindowGraphDataset(str(data_root), "val")
    a, b = split_events([r[1] for r in ds.records])
    in_a = np.array([r[1] in a for r in ds.records])
    y = np.array([r[2] for r in ds.records])
    res = np.array([float(r[3]["resistance"]) for r in ds.records])
    rep = {"split_seed": SPLIT_SEED, "n_events_A": len(a), "n_events_B": len(b),
           "n_windows_A": int(in_a.sum()), "n_windows_B": int((~in_a).sum()),
           "overlap_events": len(a & b), "sha256_events_A": split_hash(a), "sha256_events_B": split_hash(b),
           "classes_A": np.bincount(y[in_a], minlength=26).tolist(),
           "classes_B": np.bincount(y[~in_a], minlength=26).tolist(),
           "resistance_windows_A": {str(r): int(np.isclose(res[in_a], r).sum()) for r in (0.1, 1.0, 10.0)},
           "resistance_windows_B": {str(r): int(np.isclose(res[~in_a], r).sum()) for r in (0.1, 1.0, 10.0)}}
    out_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=2))
    return rep


def _setup(data_root: Path, threads: int):
    import torch
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    from fault_localization.physics import observation as OBS
    from fault_localization.physics import residual as R
    from fault_localization.physics import table as T
    torch.set_num_threads(threads)
    mean, std = OBS.load_norm_stats(data_root)
    ds = WindowGraphDataset(str(data_root), "val")
    loader = make_dataloader(ds, S1_BATCH, shuffle=False)
    tbl = T.load_table(REPO / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"]).double()
    a_events, _ = split_events([r[1] for r in ds.records])
    return ds, loader, mean, std, table, a_events


def _clean_scores(x_rows, sigma, mean, std, table, loo_chunk):
    import torch
    import run_silent_detection as S1
    from fault_localization.physics import observation as OBS
    from fault_localization.physics import residual as R
    s, a = OBS.observe(x_rows, mean, std, reduce="last")
    s, a = s.double(), a.double()
    rho, _ = R.rho(s, a, table, S1.CANONICAL_SIGMA_EPS, sigma0=sigma, method="gram")
    f_hat = R.psm_predict(rho)
    loo, ins = S1.loo_and_insample(s, a, table, f_hat, loo_chunk)
    return loo.float().numpy(), ins.float().numpy(), (a.sum(dim=-1) > 0).numpy()


def stage_thresholds(sigma: float, out_root: Path, data_root: Path, threads: int,
                     max_batches: int | None, loo_chunk: int = 128, flatline: bool = False) -> Path:
    """Per-sensor thresholds from clean half-A windows; clean half-B flags for the clean arm.
    ``flatline=False`` is G-TF (1 % per sensor, in-sample and LOO residual). ``flatline=True`` is G-TF2: the
    in-sample residual and the flatline score, each at ``FPR_GTF2``, written to ``thresholds_gtf2_*.npz``."""
    fpr = FPR_GTF2 if flatline else FPR_TARGET
    out = out_root / (f"thresholds_gtf2_sigma_{sigma:g}.npz" if flatline else f"thresholds_sigma_{sigma:g}.npz")
    ds, loader, mean, std, table, a_events = _setup(data_root, threads)
    n_total = len(ds) if max_batches is None else sum(
        len(idx) for _, idx in zip(range(max_batches), loader.batch_sampler))
    loo_a, ins_a, av_a = [], [], []
    loo_b = np.zeros((n_total, 25), np.float32); ins_b = np.zeros((n_total, 25), np.float32)
    flat_b = np.zeros((n_total, 25), np.float32)
    av_all = np.zeros((n_total, 25), bool); is_a = np.zeros(n_total, bool)
    pos, t0 = 0, time.time()
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = batch["x"]; b = x.shape[0]
        rows_a = np.array([event_of(w) in a_events for w in batch["window_id"]])
        xn = _noisy(x, sigma, std, bi)
        lo, ins, av = _clean_scores(xn, sigma, mean, std, table, loo_chunk)
        sl = slice(pos, pos + b)
        loo_b[sl], ins_b[sl], av_all[sl], is_a[sl] = lo, ins, av, rows_a
        if flatline:
            flat_b[sl] = flatline_score(xn, mean, std)
        pos += b
        if bi % 20 == 0:
            print(f"  [thr sigma={sigma:g}] batch {bi} windows {pos}/{n_total} {(time.time()-t0)/60:.1f} min", flush=True)
    thr_ins = per_sensor_thresholds(ins_b[is_a], av_all[is_a], fpr)
    out_root.mkdir(parents=True, exist_ok=True)
    if flatline:
        thr_flat = per_sensor_thresholds(flat_b[is_a], av_all[is_a], fpr)
        fb = detector_flags(ins_b, flat_b, av_all, thr_ins, thr_flat)
        np.savez_compressed(out, thr_ins=thr_ins, thr_flat=thr_flat,
                            flags_clean_B_ins=fb["ins"] & ~is_a[:, None], flags_clean_B_flat=fb["flat"] & ~is_a[:, None],
                            flags_clean_B_union=fb["union"] & ~is_a[:, None], is_A=is_a, avail=av_all, n_total=n_total,
                            sigma=sigma, fpr_target=fpr, split_seed=SPLIT_SEED)
        return out
    thr_loo = per_sensor_thresholds(loo_b[is_a], av_all[is_a], fpr)
    flags_b = {"loo": flags_from_scores(loo_b, av_all, thr_loo) & ~is_a[:, None],
               "ins": flags_from_scores(ins_b, av_all, thr_ins) & ~is_a[:, None]}
    np.savez_compressed(out, thr_loo=thr_loo, thr_ins=thr_ins, flags_clean_B_loo=flags_b["loo"],
                        flags_clean_B_ins=flags_b["ins"], is_A=is_a, avail=av_all, n_total=n_total,
                        sigma=sigma, fpr_target=FPR_TARGET, split_seed=SPLIT_SEED)
    return out


# --------------------------------------------------------------------------------------------
# scoring (GPU; R5)
# --------------------------------------------------------------------------------------------

def _load_r5(seed: int, device):
    import torch
    import yaml
    from fault_localization.experiments.runner import _model as build_model
    from fault_localization.data.dataset import WindowGraphDataset
    run_dir = REPO / f"results/s4_4/r5/pcstgnn/seed_{seed:05d}"
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ds = WindowGraphDataset(cfg.get("data_root", "dataset_fullscale_50"), "val")
    model = build_model(cfg, ds.edge_index).to(device)
    model.load_state_dict(torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=True)["model"])
    model.eval()
    return model


def _predict(model, x, sigma, device):
    import torch
    from fault_localization.models.voting import soft_vote
    beta = float(model.beta)
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], SUB_BATCH):
            xb = x[i:i + SUB_BATCH].to(device)
            z = model.network_logits(xb)
            rho = model._rho(xb, sigma0=sigma)[0]
            preds.extend(soft_vote(z - beta * rho[:, None, :])[1].cpu().tolist())
    return preds


def stage_score(seed: int, scope: str, out_root: Path, data_root: Path, threads: int,
                max_batches: int | None = None, device_name: str | None = None,
                sigmas: tuple | None = None, detector: str | None = None, gate: str = "gtf",
                thr_root: Path | None = None) -> list[Path]:
    """``gate="gtf"``: G-TF, unchanged. ``gate="gtf2"``: G-TF2 (in-sample residual OR flatline score, each at 0.5 %);
    the arm is stored under the key ``gtf2`` and the per-detector rates are added under ``detectors``.
    ``gate="gtf2r"``: residual only at 0.5 % -- the gated arm only (key ``resid05``), with the G-TF2 thresholds read
    from ``thr_root``; the no-gate and oracle arms are not recomputed (the summary takes them from the G-TF2 result)."""
    import torch
    import run_silent_detection as S1
    from fault_localization.dropout.silent import apply_silent_failure_batch
    from fault_localization.physics import observation as OBS
    from score_s4_4_gate4_run import macro_f1

    gtf2 = gate == "gtf2"
    resid = gate == "gtf2r"
    arm = {"gtf": "gtf", "gtf2": "gtf2", "gtf2r": "resid05"}[gate]
    if (gtf2 or resid) and max_batches is None:        # an official G-TF2 scoring needs the committed pre-registration
        ok, problems = check_ready_gtf2()
        if not ok:
            raise RuntimeError(f"G-TF2 pre-registration is not complete: {problems}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    chk = check_gate_equals_training_dropout(data_root)
    if not chk["passed"]:
        raise RuntimeError(f"consistency check 1 failed (gate drop != training dropout): {chk}")
    if gtf2 or resid:
        det = "ins"                                    # pre-registered: the in-sample residual, never LOO
    elif detector is not None:        # smoke tests only (main() allows it with --smoke)
        det = detector
    else:
        decision = json.loads(DECISION_PATH.read_text()) if DECISION_PATH.exists() else None
        if decision is None:
            raise RuntimeError("detector decision not recorded -- run decide first (after sigma0=1e-2 is complete)")
        det = decision["detector"]
    ds, loader, mean, std, table, a_events = _setup(data_root, threads)
    model = _load_r5(seed, device)
    conds = scope_conditions(scope)
    written = []
    for sigma in (sigmas or scope_sigmas(scope)):
        out = out_root / f"sigma_{sigma:g}" / f"seed_{seed:05d}.json"
        if out.exists():
            written.append(out); continue
        thr_file = (thr_root or out_root) / (f"thresholds_gtf2_sigma_{sigma:g}.npz" if (gtf2 or resid) else f"thresholds_sigma_{sigma:g}.npz")
        if not thr_file.exists():
            raise RuntimeError(f"thresholds missing for sigma0={sigma}: run the thresholds stage first")
        T_ = np.load(thr_file)
        thr = T_[f"thr_{det}"]; clean_flags = T_["flags_clean_B_union"] if gtf2 else T_[f"flags_clean_B_{det}"]   # gtf2r: ins only
        thr_flat = T_["thr_flat"] if gtf2 else None
        raw = S1.open_raw(raw_dir(sigma), 0, "r")
        n_total = int(T_["n_total"])
        if max_batches is None:
            chk2 = check_stored_scores(sigma, batch_ids=(0, 300, 600), data_root=data_root, threads=threads)
            if not chk2["passed"]:
                raise RuntimeError(f"consistency check 2 failed (stored scores != recomputed): {chk2}")
        P = {c[0]: ({arm: []} if resid else {"nogate": [], arm: [], "oracle": []}) for c in conds}
        P["clean"] = {arm: []} if resid else {"nogate": [], arm: []}
        ytrue = []
        st = {c[0]: dict(tp=0, pos=0, fp=0, neg=0, flagged=0, win=0) for c in conds}
        st["clean"] = dict(fp=0, neg=0, flagged=0, win=0)
        dnames = ("ins", "flat", "union") if gtf2 else ()
        dst = {c: {n: dict(tp=0, fp=0, flagged=0) for n in dnames} for c in st}      # per-detector counts (G-TF2 only)
        pos, t0 = 0, time.time()
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x, y = batch["x"], batch["y"]; b = x.shape[0]
            sl = slice(pos, pos + b)
            rows_b = np.array([event_of(w) not in a_events for w in batch["window_id"]])
            if not rows_b.any():
                pos += b; continue
            if not np.array_equal(np.asarray(raw["y"][sl]).astype(np.int64), y.numpy()):
                raise RuntimeError("batch/raw alignment broken (labels differ from the S5.1 raw scores)")
            rb = torch.as_tensor(rows_b)
            xn = _noisy(x, sigma, std, bi)
            s_cl, a_cl = OBS.observe(xn, mean, std, reduce="last")
            avail_b = (a_cl.sum(dim=-1) > 0).numpy()[rows_b]
            ytrue.extend(y[rb].tolist())
            # clean-window arm
            fl = clean_flags[sl][rows_b]
            if not resid:
                P["clean"]["nogate"] += _predict(model, xn[rb], sigma, device)
            P["clean"][arm] += _predict(model, gate_zero(xn[rb], torch.as_tensor(fl)), sigma, device)
            neg = avail_b
            if gtf2:
                for n in dnames:
                    fn = T_[f"flags_clean_B_{n}"][sl][rows_b]
                    dst["clean"][n]["fp"] += int((fn & neg).sum()); dst["clean"][n]["flagged"] += int(fn.sum())
            st["clean"]["fp"] += int((fl & neg).sum()); st["clean"]["neg"] += int(neg.sum())
            st["clean"]["flagged"] += int(fl.sum()); st["clean"]["win"] += int(rows_b.sum())
            for cname, kind, sev, n_s in conds:
                x_c, corrupted = apply_silent_failure_batch(
                    xn, n_s, kind, seed=S1.CORRUPTION_SEED * 10000 + bi, norm_stats=(mean, std), severity=sev)
                label = np.zeros((b, 25), bool)
                for row, nodes in enumerate(corrupted):
                    label[row, nodes] = True
                if not np.array_equal(label, np.asarray(raw[f"{cname}__label"][sl]).astype(bool)):
                    raise RuntimeError(f"{cname}: regenerated corruption labels differ from the S5.1 raw labels")
                lab_b = label[rows_b]
                sc = np.asarray(raw[f"{cname}__psm_{det}"][sl])[rows_b]
                xb = x_c[rb]
                fdict = detector_flags(sc, flatline_score(xb, mean, std) if gtf2 else None, avail_b, thr, thr_flat)
                fl = fdict["union"]
                if not resid:
                    P[cname]["nogate"] += _predict(model, xb, sigma, device)
                P[cname][arm] += _predict(model, gate_zero(xb, torch.as_tensor(fl)), sigma, device)
                if not resid:
                    P[cname]["oracle"] += _predict(model, gate_zero(xb, torch.as_tensor(lab_b)), sigma, device)
                d = st[cname]
                d["tp"] += int((fl & lab_b).sum()); d["pos"] += int((lab_b & avail_b).sum())
                d["fp"] += int((fl & ~lab_b & avail_b).sum()); d["neg"] += int((~lab_b & avail_b).sum())
                d["flagged"] += int(fl.sum()); d["win"] += int(rows_b.sum())
                for n in dnames:
                    dst[cname][n]["tp"] += int((fdict[n] & lab_b).sum())
                    dst[cname][n]["fp"] += int((fdict[n] & ~lab_b & avail_b).sum())
                    dst[cname][n]["flagged"] += int(fdict[n].sum())
            pos += b
            if bi % 20 == 0:
                print(f"  [score seed={seed} sigma={sigma:g}] batch {bi} {(time.time()-t0)/60:.1f} min", flush=True)
        yt = np.asarray(ytrue)
        res = {"seed": seed, "sigma0": sigma, "scope": scope, "detector": ("ins+flatline" if gtf2 else det), "half": "B",
               "split_seed": SPLIT_SEED, "n_windows": int(yt.size), "fpr_target": (FPR_GTF2 if (gtf2 or resid) else FPR_TARGET),
               "gate": gate,
               "partial_run": max_batches is not None, "conditions": {}, "device": str(device),
               "scoring_seconds": time.time() - t0}
        f1_clean = None if resid else macro_f1(yt, np.asarray(P["clean"]["nogate"]))
        for c, arms in P.items():
            f1 = {k: macro_f1(yt, np.asarray(v)) for k, v in arms.items()}
            d = st[c]
            entry = {"f1": f1}
            if c == "clean":
                entry["realized_fpr_clean_windows"] = d["fp"] / max(d["neg"], 1)
                entry["flagged_per_window"] = d["flagged"] / max(d["win"], 1)
                if not resid:
                    entry[f"delta_{arm}_minus_nogate"] = f1[arm] - f1["nogate"]
                if gtf2:
                    entry["detectors"] = {n: {"realized_fpr_clean_windows": dst[c][n]["fp"] / max(d["neg"], 1),
                                              "flagged_per_window": dst[c][n]["flagged"] / max(d["win"], 1)} for n in dnames}
            else:
                if not resid:
                    entry["recovered"] = recovered_fraction(f1[arm], f1["nogate"], f1["oracle"])
                    entry["corruption_loss"] = f1_clean - f1["nogate"]       # F1(clean, no gate) - F1(corrupted, no gate)
                    entry["oracle_loss"] = f1_clean - f1["oracle"]
                    entry["f1_clean_nogate"] = f1_clean
                entry["tpr"] = d["tp"] / max(d["pos"], 1)
                entry["realized_fpr_clean_sensors_of_corrupted_windows"] = d["fp"] / max(d["neg"], 1)
                entry["flagged_per_window"] = d["flagged"] / max(d["win"], 1)
                if gtf2:
                    entry["detectors"] = {n: {"tpr": dst[c][n]["tp"] / max(d["pos"], 1),
                                              "realized_fpr_clean_sensors_of_corrupted_windows": dst[c][n]["fp"] / max(d["neg"], 1),
                                              "flagged_per_window": dst[c][n]["flagged"] / max(d["win"], 1)} for n in dnames}
            res["conditions"][c] = entry
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        written.append(out)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return written


def stage_summarize(scope: str, out_root: Path, arm: str = "gtf") -> dict:
    """``arm="gtf"``: G-TF, unchanged. ``arm="gtf2"``: the G-TF2 rules (sign rule in all 3 seeds at each s, primary
    = stuck, secondary = bias +-2 %, bias +-5 %, swap; Gaussian is reported only; paired t over seeds with Holm
    over the three s-levels within each type)."""
    from scipy import stats
    summ = {"scope": scope, "arm": arm, "sigma": {}}
    for sigma in scope_sigmas(scope):
        runs = [json.loads((out_root / f"sigma_{sigma:g}" / f"seed_{s:05d}.json").read_text()) for s in SEEDS]
        by_cond = {}
        for c in runs[0]["conditions"]:
            if c == "clean":
                continue
            ng = [r["conditions"][c]["f1"]["nogate"] for r in runs]
            gt = [r["conditions"][c]["f1"][arm] for r in runs]
            orc = [r["conditions"][c]["f1"]["oracle"] for r in runs]
            diff = [g - n for g, n in zip(gt, ng)]
            t = stats.ttest_rel(gt, ng) if len(set(diff)) > 1 else None
            by_cond[c] = {"nogate": ng, arm: gt, "oracle": orc, f"{arm}_minus_nogate": diff,
                          "sign_consistent_positive": all(d > 0 for d in diff),
                          "paired_t_p": (None if t is None else float(t.pvalue)),
                          "recovered_per_seed": [r["conditions"][c]["recovered"] for r in runs],
                          "recovered_mean_over_seeds": (None if any(r["conditions"][c]["recovered"]["recovered"] is None for r in runs)
                                                        else float(np.mean([r["conditions"][c]["recovered"]["recovered"] for r in runs]))),
                          "corruption_loss": [r["conditions"][c]["corruption_loss"] for r in runs],
                          "tpr": [r["conditions"][c]["tpr"] for r in runs],
                          "realized_fpr_clean_sensors_of_corrupted_windows":
                              [r["conditions"][c]["realized_fpr_clean_sensors_of_corrupted_windows"] for r in runs]}
        gate_conds = [c for c in by_cond if c.startswith("bias_2pct_s")]
        pv = [by_cond[c]["paired_t_p"] if by_cond[c]["paired_t_p"] is not None else 1.0 for c in gate_conds]
        for c, p in zip(gate_conds, holm(pv)):
            by_cond[c]["paired_t_p_holm_family_bias2pct"] = p
        summ["sigma"][str(sigma)] = {
            "conditions": by_cond,
            "clean": {"nogate": [r["conditions"]["clean"]["f1"]["nogate"] for r in runs],
                      arm: [r["conditions"]["clean"]["f1"][arm] for r in runs],
                      "realized_fpr_clean_windows": [r["conditions"]["clean"]["realized_fpr_clean_windows"] for r in runs]}}
    key = str(0.01)
    if arm == "gtf2" and key in summ["sigma"]:
        conds = summ["sigma"][key]["conditions"]
        rules = {}
        for typ in GTF2_TYPES:
            cs = [c for c in conds if c.rsplit("_s", 1)[0] == typ]
            pv = [conds[c]["paired_t_p"] if conds[c]["paired_t_p"] is not None else 1.0 for c in cs]
            for c, p in zip(cs, holm(pv)):
                conds[c]["paired_t_p_holm_over_s"] = p
            rules[typ] = {"role": ("primary" if typ == GTF2_PRIMARY else "secondary" if typ in GTF2_SECONDARY else "reported only"),
                          "useful (better than no gate in all 3 seeds at every s)": bool(cs) and all(conds[c]["sign_consistent_positive"] for c in cs),
                          "per_s": {c: conds[c]["sign_consistent_positive"] for c in cs}}
        summ["decision_rule_gtf2_sigma1e-2"] = rules
    elif key in summ["sigma"]:
        gc = {c: v for c, v in summ["sigma"][key]["conditions"].items() if c.startswith("bias_2pct_s")}
        summ["decision_rule_bias2pct_sigma1e-2"] = {
            "G-TF useful (better than no gate in all 3 seeds at every s)": bool(gc) and all(v["sign_consistent_positive"] for v in gc.values()),
            "per_s": {c: v["sign_consistent_positive"] for c, v in gc.items()}}
    (out_root / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def stage_summarize_gtf2r(scope: str, out_root: Path, gtf2_root: Path) -> dict:
    """Residual only at 0.5 % (descriptive; no decision rule). The no-gate and oracle F1 and the corruption loss come
    from the G-TF2 result of the same seed (same model, windows and corruption draw)."""
    summ = {"scope": scope, "arm": "resid05", "note": "descriptive; not in a decision rule", "sigma": {}}
    for sigma in scope_sigmas(scope):
        r2 = [json.loads((out_root / f"sigma_{sigma:g}" / f"seed_{s:05d}.json").read_text()) for s in SEEDS]
        r1 = [json.loads((gtf2_root / f"sigma_{sigma:g}" / f"seed_{s:05d}.json").read_text()) for s in SEEDS]
        by_cond = {}
        for c in r2[0]["conditions"]:
            if c == "clean":
                continue
            ng = [r["conditions"][c]["f1"]["nogate"] for r in r1]
            orc = [r["conditions"][c]["f1"]["oracle"] for r in r1]
            g2 = [r["conditions"][c]["f1"]["gtf2"] for r in r1]
            rs = [r["conditions"][c]["f1"]["resid05"] for r in r2]
            by_cond[c] = {"nogate": ng, "oracle": orc, "gtf2": g2, "resid05": rs,
                          "resid05_minus_nogate": [a - b for a, b in zip(rs, ng)],
                          "gtf2_minus_resid05": [a - b for a, b in zip(g2, rs)],
                          "recovered_per_seed": [recovered_fraction(a, b, o) for a, b, o in zip(rs, ng, orc)],
                          "corruption_loss": [r["conditions"][c]["corruption_loss"] for r in r1],
                          "tpr": [r["conditions"][c]["tpr"] for r in r2],
                          "realized_fpr_clean_sensors_of_corrupted_windows":
                              [r["conditions"][c]["realized_fpr_clean_sensors_of_corrupted_windows"] for r in r2]}
        summ["sigma"][str(sigma)] = {"conditions": by_cond, "clean": {
            "nogate": [r["conditions"]["clean"]["f1"]["nogate"] for r in r1],
            "resid05": [r["conditions"]["clean"]["f1"]["resid05"] for r in r2],
            "realized_fpr_clean_windows": [r["conditions"]["clean"]["realized_fpr_clean_windows"] for r in r2]}}
    (out_root / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=["split", "thresholds", "score", "summarize", "decide", "check-ready", "check-consistency"])
    ap.add_argument("--scope", choices=["core", "full"], default="core")
    ap.add_argument("--gate", choices=["gtf", "gtf2", "gtf2r"], default="gtf",
                    help="gtf2 = in-sample residual OR flatline score (PC-STGNN/stage_f_gtf2_preregistration.md); "
                         "gtf2r = residual only at 0.5 % (extra arm; reads the gtf2 thresholds); core scope only")
    ap.add_argument("--thr-root", default=None, help="gtf2r: directory with thresholds_gtf2_*.npz (default results/stage_f/gtf2/core)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="test only: CPU, 3 batches, sigma0 from --sigma, outputs under --out-root (never the official dir)")
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--detector", choices=["loo", "ins"], default=None, help="smoke tests only")
    args = ap.parse_args()
    if args.gate != "gtf" and args.scope != "core":
        sys.exit("G-TF2 is pre-registered for the core scope only")
    base = {"gtf": OUT_ROOT, "gtf2": OUT_ROOT_GTF2, "gtf2r": OUT_ROOT_GTF2R}[args.gate]
    out_root = Path(args.out_root) if args.out_root else base / args.scope
    thr_root = Path(args.thr_root) if args.thr_root else (OUT_ROOT_GTF2 / args.scope if (args.gate == "gtf2r" and not args.out_root) else None)
    if args.smoke and (args.out_root is None or (REPO / "results").resolve() in Path(args.out_root).resolve().parents
                       or Path(args.out_root).resolve() == (REPO / "results").resolve()):
        sys.exit("--smoke needs an --out-root outside results/")
    if args.stage == "check-ready":
        ok, problems = check_ready_gtf2() if args.gate != "gtf" else check_ready(args.scope)
        print(json.dumps({"ready": ok, "problems": problems}, indent=2))
        return 0 if ok else 1
    if args.stage == "check-consistency":
        sg = args.sigma if args.sigma is not None else 0.01
        r1 = check_gate_equals_training_dropout(Path(args.data_root))
        r2 = check_stored_scores(sg, data_root=Path(args.data_root), threads=args.threads)
        print(json.dumps({"check1_gate_equals_training_dropout": r1, "check2_stored_equals_recomputed": r2}, indent=2))
        return 0 if (r1["passed"] and r2["passed"]) else 1
    if args.stage == "split":
        print(json.dumps(stage_split(out_root, Path(args.data_root)), indent=2)); return 0
    if args.stage == "decide":
        if not (raw_dir(0.01) / "COMPLETE.json").exists() or not sigma_summary_path(0.01).exists():
            sys.exit("sigma0=1e-2 S5.1 result is not complete -- the detector rule is not applied before that")
        s = json.loads(sigma_summary_path(0.01).read_text())["s5_2_psm_auroc"]
        dec = decide_detector(s)
        dec.update({"sigma0": 0.01, "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        DECISION_PATH.write_text(json.dumps(dec, indent=2)); print(json.dumps(dec, indent=2)); return 0
    mb = 3 if args.smoke else None
    sigmas = (args.sigma,) if args.sigma is not None else scope_sigmas(args.scope)
    if args.stage == "thresholds":
        if args.gate == "gtf2r":
            sys.exit("gtf2r has no thresholds stage: it reads the G-TF2 thresholds (--gate gtf2 thresholds)")
        for sg in sigmas:
            tname = f"thresholds_gtf2_sigma_{sg:g}.npz" if args.gate == "gtf2" else f"thresholds_sigma_{sg:g}.npz"
            if (out_root / tname).exists():
                continue
            print(stage_thresholds(sg, out_root, Path(args.data_root), args.threads, mb, flatline=(args.gate == "gtf2")))
        return 0
    if args.stage == "score":
        seeds = (args.seed,) if args.seed else SEEDS
        for s in seeds:
            stage_score(s, args.scope, out_root, Path(args.data_root), args.threads, mb,
                        "cpu" if args.smoke else None, sigmas, args.detector if args.smoke else None, args.gate, thr_root)
        return 0
    if args.stage == "summarize":
        if args.gate == "gtf2r":
            print(json.dumps(stage_summarize_gtf2r(args.scope, out_root, Path(args.thr_root) if args.thr_root else OUT_ROOT_GTF2 / args.scope), indent=2)); return 0
        print(json.dumps(stage_summarize(args.scope, out_root, args.gate), indent=2)); return 0
    ap.print_help(); return 2


if __name__ == "__main__":
    sys.exit(main())
