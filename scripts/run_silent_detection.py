"""S5.1 + S5.2 -- training-free silent-failure detection (``07_STAGE_F_gate.md``).

S5.1: for every silent corruption (``dropout/silent.py``: stuck, bias +-2 %, +-5 %, Gaussian
sigma = 0.02 p.u., swapped phases) on s in {1, 2, 3} sensors, score every available sensor with
the leave-one-sensor-out residual rho_i^LOO (``physics/residual.py``, ``01_SPEC.md`` §7) and
measure how well it separates corrupted from clean sensors: AUROC (primary, hard gate 0.75 at
the mildest corruption, bias +-2 %), AUPRC, TPR at FPR = 0.05 (``11_METRICS.md`` §3).

S5.2: the same metrics for rho_i^in-sample (sensor i's term read at the argmin fitted on all of
A). The LOO-minus-in-sample gap is the S5.2 result.

Training-free by construction: the class used for the refit is the PSM prediction
f_hat = argmin_f rho_f(A) on the **corrupted** input (zero learned parameters, canonical frozen
operator, sigma_eps as in S2.4). A second, oracle variant refits at the true class y; it is a
diagnostic only (it separates "the detector fails" from "PSM mislocalised the fault") and is
never the gate number.

Protocol choices, stated here rather than hidden:
- Validation split only. Test and the sealed OOD set stay untouched.
- k = 0 (no dropout) and sigma0 = 0 (clean corpus): the silent corruption is the only
  perturbation. The "gauss" corruption is per-sensor, not fleet-wide noise, so sigma0 stays 0.
- One corruption realisation per window: per-batch seed ``CORRUPTION_SEED * 10000 + batch_idx``
  (``masking.py`` convention), identical across conditions, so every condition corrupts the same
  sensors of the same window wherever its candidate set allows (swap needs >= 2 phases).
- Negatives are the clean sensors of the corrupted windows -- the population a trust gate sees.

Usage:
    uv run python scripts/run_silent_detection.py                    # full val split
    uv run python scripts/run_silent_detection.py --max-batches 3    # timing check
    uv run python scripts/run_silent_detection.py --summarize-only   # recompute metrics from raw/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.silent import apply_silent_failure_batch  # noqa: E402
from fault_localization.physics import observation as OBS  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452   # same value as scripts/run_psm_full_eval.py (S2.4)
CORRUPTION_SEED = 0
N_SENSORS = (1, 2, 3)
# (label, kind, severity) -- the S3.1 grid. Severity None = no magnitude knob.
CORRUPTIONS = (
    ("stuck", "stuck", None),
    ("bias_2pct", "bias", 0.02),
    ("bias_5pct", "bias", 0.05),
    ("gauss_0.02", "gauss", 0.02),
    ("swap", "swap", None),
)
GATE_CORRUPTION = "bias_2pct"
GATE_AUROC = 0.75
FPR_TARGET = 0.05
REFITS = ("psm", "oracle")
SCORES = ("loo", "ins")
LABELS26 = list(range(26))
ROC_GRID = np.linspace(0.0, 1.0, 201)
OUT_DIR = REPO / "results/pcstgnn/audit/s5_1_silent_detection"


def conditions():
    return [(f"{label}_s{s}", kind, sev, s) for label, kind, sev in CORRUPTIONS for s in N_SENSORS]


# ---------------------------------------------------------------------------
# Pass 1: stream the split, write per-(window, sensor) scores to raw/
# ---------------------------------------------------------------------------

def open_raw(raw: Path, n: int, mode: str) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    arrs = {}

    def mm(name, dtype, shape):
        arrs[name] = np.lib.format.open_memmap(raw / f"{name}.npy", mode=mode, dtype=dtype,
                                               shape=shape if mode == "w+" else None)

    mm("y", np.int8, (n,))
    mm("window", np.int8, (n,))
    mm("resistance", np.float32, (n,))
    mm("avail", np.uint8, (n, 25))
    for cname, *_ in conditions():
        mm(f"{cname}__label", np.uint8, (n, 25))
        mm(f"{cname}__delta", np.float32, (n, 25))
        mm(f"{cname}__fhat", np.int8, (n,))
        for refit in REFITS:
            for score in SCORES:
                mm(f"{cname}__{refit}_{score}", np.float32, (n, 25))
    return arrs


def loo_and_insample(s, a, table, f_use, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``R.rho_loo`` and ``R.rho_insample`` from one per-sensor error tensor.

    Same arithmetic as the two library functions (checked against them on the first batch by
    ``check_against_library``); building the ``[B, H, 25]`` tensor once halves the cost.
    """
    b = s.shape[0]
    loo = torch.empty((b, 25), dtype=torch.float64)
    ins = torch.empty((b, 25), dtype=torch.float64)
    for c0 in range(0, b, chunk):
        c1 = min(b, c0 + chunk)
        sens = R._sensor_error_tensor(s[c0:c1], a[c0:c1], table, f_use[c0:c1],
                                      CANONICAL_SIGMA_EPS)                       # [b, H, 25]
        avail = (a[c0:c1].sum(-1) > 0).to(sens.dtype)
        total = sens.sum(dim=-1, keepdim=True)
        h_loo = (total - sens).argmin(dim=1)                                     # [b, 25]
        loo[c0:c1] = 0.5 * sens.gather(1, h_loo.unsqueeze(1)).squeeze(1) * avail
        h_ins = total.squeeze(-1).argmin(dim=1)                                  # [b]
        ins[c0:c1] = 0.5 * sens.gather(1, h_ins[:, None, None].expand(-1, 1, 25)).squeeze(1) * avail
    return loo, ins


def check_against_library(s, a, table, f_use) -> None:
    loo, ins = loo_and_insample(s, a, table, f_use, chunk=64)
    ref_loo = R.rho_loo(s, a, table, f_use, CANONICAL_SIGMA_EPS)
    ref_ins = R.rho_insample(s, a, table, f_use, CANONICAL_SIGMA_EPS)
    if not (torch.equal(loo, ref_loo) and torch.equal(ins, ref_ins)):
        raise RuntimeError("loo_and_insample disagrees with physics/residual.py")


def score_pass(data_root: Path, batch_size: int, max_batches: int | None, loo_chunk: int) -> int:
    mean, std = OBS.load_norm_stats(data_root)
    ds = WindowGraphDataset(str(data_root), "val")
    loader = make_dataloader(ds, batch_size, shuffle=False)
    # Batches are per 8-event chunk (<= 320 windows), not batch_size: count them from the sampler.
    n_total = len(ds) if max_batches is None else sum(
        len(idx) for _, idx in zip(range(max_batches), loader.batch_sampler))

    tbl = T.load_table(REPO / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"]).double()
    print(f"[load] val windows={len(ds)} scoring={n_total} table={tuple(table.shape)}", flush=True)

    arrs = open_raw(OUT_DIR / "raw", n_total, "w+")
    conds = conditions()
    pos = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = batch["x"], batch["y"]
        b = x.shape[0]
        sl = slice(pos, pos + b)
        s_clean, a = OBS.observe(x, mean, std, reduce="last")
        s_clean, a = s_clean.double(), a.double()
        arrs["y"][sl] = y.numpy().astype(np.int8)
        arrs["window"][sl] = [int(w.rsplit("_w", 1)[1]) for w in batch["window_id"]]
        arrs["resistance"][sl] = [float(m["resistance"]) for m in batch["metadata"]]
        avail = (a.sum(dim=-1) > 0)
        arrs["avail"][sl] = avail.numpy().astype(np.uint8)

        for cname, kind, sev, n_s in conds:
            x_c, corrupted = apply_silent_failure_batch(
                x, n_s, kind, seed=CORRUPTION_SEED * 10000 + bi,
                norm_stats=(mean, std), severity=sev)
            s_c, a_c = OBS.observe(x_c, mean, std, reduce="last")
            s_c, a_c = s_c.double(), a_c.double()
            if not torch.equal(a_c, a):
                raise RuntimeError("silent corruption changed the availability mask")
            label = np.zeros((b, 25), dtype=np.uint8)
            for row, nodes in enumerate(corrupted):
                label[row, nodes] = 1
            arrs[f"{cname}__label"][sl] = label
            arrs[f"{cname}__delta"][sl] = ((s_c - s_clean).abs() * a).amax(dim=-1).float().numpy()

            rho, _ = R.rho(s_c, a_c, table, CANONICAL_SIGMA_EPS, sigma0=0.0, method="gram")
            f_hat = R.psm_predict(rho)
            arrs[f"{cname}__fhat"][sl] = f_hat.numpy().astype(np.int8)
            if bi == 0:
                check_against_library(s_c[:64], a_c[:64], table, f_hat[:64])
            loo, ins = loo_and_insample(s_c, a_c, table, f_hat, loo_chunk)
            arrs[f"{cname}__psm_loo"][sl] = loo.float().numpy()
            arrs[f"{cname}__psm_ins"][sl] = ins.float().numpy()
            # Oracle refit at the true class: identical to the PSM refit wherever f_hat == y,
            # so only the mislocalised rows are recomputed.
            wrong = torch.nonzero(f_hat != y).flatten()
            if len(wrong):
                loo_w, ins_w = loo_and_insample(s_c[wrong], a_c[wrong], table, y[wrong], loo_chunk)
                loo[wrong], ins[wrong] = loo_w, ins_w
            arrs[f"{cname}__oracle_loo"][sl] = loo.float().numpy()
            arrs[f"{cname}__oracle_ins"][sl] = ins.float().numpy()
        pos += b
        if bi % 10 == 0:
            el = time.time() - t0
            eta = el / pos * (n_total - pos)
            print(f"  batch {bi} windows {pos}/{n_total}  {el/60:.1f} min  eta {eta/60:.1f} min",
                  flush=True)
    for v in arrs.values():
        v.flush()
    if pos != n_total:
        raise RuntimeError(f"scored {pos} windows, expected {n_total}")
    (OUT_DIR / "raw" / "COMPLETE.json").write_text(json.dumps(
        {"n_windows": pos, "batch_size": batch_size, "max_batches": max_batches,
         "minutes": (time.time() - t0) / 60}, indent=2))
    return pos


# ---------------------------------------------------------------------------
# Pass 2: metrics
# ---------------------------------------------------------------------------

def detection_metrics(label: np.ndarray, score: np.ndarray) -> dict:
    if label.min() == label.max():
        return {"n_pos": int(label.sum()), "n_neg": int((1 - label).sum()), "auroc": None,
                "auprc": None, "tpr_at_fpr05": None}
    fpr, tpr, _ = roc_curve(label, score)
    return {
        "n_pos": int(label.sum()),
        "n_neg": int(len(label) - label.sum()),
        "auroc": float(roc_auc_score(label, score)),
        "auprc": float(average_precision_score(label, score)),
        "prevalence": float(label.mean()),
        # Largest TPR achievable without exceeding the 5 % false-alarm rate.
        "tpr_at_fpr05": float(tpr[fpr <= FPR_TARGET].max()),
        "_roc": np.interp(ROC_GRID, fpr, tpr),
    }


def summarize() -> dict:
    raw = OUT_DIR / "raw"
    arrs = open_raw(raw, 0, "r")
    y = np.asarray(arrs["y"])
    avail = np.asarray(arrs["avail"]).astype(bool)
    strata = {"all": np.ones_like(y, dtype=bool), "fault": y > 0, "nofault": y == 0}

    rows, roc_rows = [], []
    for cname, kind, sev, n_s in conditions():
        label_full = np.asarray(arrs[f"{cname}__label"])
        delta_full = np.asarray(arrs[f"{cname}__delta"])
        fhat = np.asarray(arrs[f"{cname}__fhat"]).astype(np.int64)
        psm_f1 = f1_score(y, fhat, average="macro", labels=LABELS26, zero_division=0)
        for sname, wsel in strata.items():
            m = avail & wsel[:, None]
            lab = label_full[m]
            dpos = delta_full[m][lab == 1]
            base = {"condition": cname, "corruption": cname.rsplit("_s", 1)[0], "kind": kind,
                    "severity": sev, "n_sensors": n_s, "stratum": sname,
                    "psm_f1_macro26_all": float(psm_f1),
                    "corrupted_unchanged_frac": float((dpos == 0).mean()) if len(dpos) else None,
                    "corrupted_delta_median_pu": float(np.median(dpos)) if len(dpos) else None}
            for refit in REFITS:
                for score in SCORES:
                    sc = np.asarray(arrs[f"{cname}__{refit}_{score}"])[m]
                    met = detection_metrics(lab, sc)
                    roc = met.pop("_roc", None)
                    rows.append({**base, "refit": refit, "score": score, **met})
                    if roc is not None and sname == "all":
                        roc_rows.append({"condition": cname, "refit": refit, "score": score,
                                         "fpr": ROC_GRID.tolist(), "tpr": roc.tolist()})
        print(f"[summary] {cname} done", flush=True)

    def pick(cname, refit, score, stratum="all"):
        return next(r for r in rows if r["condition"] == cname and r["refit"] == refit
                    and r["score"] == score and r["stratum"] == stratum)

    gate = {f"{GATE_CORRUPTION}_s{s}": pick(f"{GATE_CORRUPTION}_s{s}", "psm", "loo")["auroc"]
            for s in N_SENSORS}
    s52 = {c: {"loo": pick(c, "psm", "loo")["auroc"], "ins": pick(c, "psm", "ins")["auroc"]}
           for c, *_ in conditions()}
    return {"rows": rows, "roc": roc_rows, "gate_s5_1_psm_loo_auroc": gate,
            "gate_threshold": GATE_AUROC, "s5_2_psm_auroc": s52}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--loo-chunk", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        score_pass(Path(args.data_root), args.batch_size, args.max_batches, args.loo_chunk)
    complete = json.loads((OUT_DIR / "raw" / "COMPLETE.json").read_text())
    result = summarize()
    meta = {"git_commit": git_commit(), "split": "val", "k": 0, "sigma0": 0.0,
            "sigma_eps": CANONICAL_SIGMA_EPS, "corruption_seed": CORRUPTION_SEED,
            "batch_size": complete["batch_size"], "n_windows": complete["n_windows"],
            "partial_run": complete["max_batches"] is not None,
            "f_hat": "PSM argmin rho on the corrupted input (oracle = true class, diagnostic)"}
    name = "silent_detection.json" if complete["max_batches"] is None else "silent_detection_PARTIAL.json"
    (OUT_DIR / name).write_text(json.dumps({"meta": meta, **{k: v for k, v in result.items()
                                                              if k != "roc"}}, indent=2))
    (OUT_DIR / name.replace(".json", "_roc.json")).write_text(json.dumps(result["roc"]))
    print(json.dumps({"gate_s5_1": result["gate_s5_1_psm_loo_auroc"],
                      "s5_2": result["s5_2_psm_auroc"]}, indent=2))


if __name__ == "__main__":
    main()
