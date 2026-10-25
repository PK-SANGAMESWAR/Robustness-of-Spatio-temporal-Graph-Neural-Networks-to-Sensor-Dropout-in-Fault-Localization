"""S5.1 + S5.2 again, with ambient measurement noise (``07_STAGE_F_gate.md``), 2026-10-24.

The first run (``scripts/run_silent_detection.py``, ``results/pcstgnn/audit/s5_1_silent_detection/``)
used sigma0 = 0.0 -- the silent corruption was the only perturbation on an otherwise clean
corpus. This script reruns S5.1/S5.2 at sigma0 in {5e-4, 2e-3, 1e-2} p.u. (the same noise grid
as ``noise_augmentation.sigma_grid`` elsewhere in S4.4), to ask whether the training-free
detector still clears the 0.75 hard gate once the sensors are not perfectly clean.

Same corruption types, same s in {1, 2, 3}, same validation split, same corruption seed, as the
first run -- the only thing that changes is an added measurement-noise step before both the
"clean" reference and the corrupted observation, and sigma0 in ``physics.residual.rho`` is set
to the actual injected level instead of hard-coded 0.0 (the S4.3b convention: "sigma0 in the
residual = the sigma injected", ``06_STAGE_E_pcstgnn.md``).

Noise is injected into x (``dropout.noise.apply_measurement_noise_batch``, normalized space,
gated by the availability mask) *before* the silent corruption is applied, so the silent
corruption lands on an already-noisy input -- the realistic order (ambient sensor noise is
always present; the silent failure is additionally present or not). One noise realisation per
window per sigma, seeded ``NOISE_SEED * 10000 + batch_index`` (same convention as
``CORRUPTION_SEED`` in the first script), independent across the three sigma values via
``NOISE_SEED = 1`` plus the sigma level index (so sigma=5e-4 and sigma=2e-3 do not draw the
same noise).

CPU only -- this reuses the no-grad physics residual path of run_silent_detection.py and never
touches CUDA. Run at low priority and few threads (see Usage) so GPU training is not slowed.

Usage:
    nice -n 19 ionice -c 3 uv run python scripts/run_silent_detection_noise.py --threads 6
    uv run python scripts/run_silent_detection_noise.py --sigma 0.0005 --max-batches 3  # smoke test
    uv run python scripts/run_silent_detection_noise.py --summarize-only --sigma 0.0005
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from fault_localization.data.dataset import make_dataloader  # noqa: E402
from fault_localization.dropout.noise import apply_measurement_noise_batch  # noqa: E402
from fault_localization.dropout.silent import apply_silent_failure_batch  # noqa: E402
from fault_localization.physics import observation as OBS  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402
from fault_localization.data.dataset import WindowGraphDataset  # noqa: E402

import run_silent_detection as S1  # noqa: E402  -- the sigma0=0.0 script this one extends

SIGMA_GRID = (0.0005, 0.002, 0.01)
NOISE_SEED = 1   # distinct from S1.CORRUPTION_SEED=0, and offset by sigma index below
OUT_ROOT = REPO / "results/stage_f/silent_detection_noise"


def _out_dir(sigma: float) -> Path:
    return OUT_ROOT / f"sigma_{sigma:g}"


def score_pass_noisy(sigma: float, sigma_idx: int, data_root: Path, batch_size: int,
                     max_batches: int | None, loo_chunk: int) -> int:
    mean, std = OBS.load_norm_stats(data_root)
    ds = WindowGraphDataset(str(data_root), "val")
    loader = make_dataloader(ds, batch_size, shuffle=False)
    n_total = len(ds) if max_batches is None else sum(
        len(idx) for _, idx in zip(range(max_batches), loader.batch_sampler))

    tbl = T.load_table(REPO / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"]).double()
    print(f"[sigma={sigma}] val windows={len(ds)} scoring={n_total}", flush=True)

    out_dir = _out_dir(sigma)
    arrs = S1.open_raw(out_dir / "raw", n_total, "w+")
    conds = S1.conditions()
    pos = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = batch["x"], batch["y"]
        b = x.shape[0]
        sl = slice(pos, pos + b)

        noise_seed = (NOISE_SEED + sigma_idx) * 10000 + bi
        x_noisy = apply_measurement_noise_batch(x, sigma, std, seed=noise_seed)

        s_clean, a = OBS.observe(x_noisy, mean, std, reduce="last")
        s_clean, a = s_clean.double(), a.double()
        arrs["y"][sl] = y.numpy().astype(np.int8)
        arrs["window"][sl] = [int(w.rsplit("_w", 1)[1]) for w in batch["window_id"]]
        arrs["resistance"][sl] = [float(m["resistance"]) for m in batch["metadata"]]
        avail = (a.sum(dim=-1) > 0)
        arrs["avail"][sl] = avail.numpy().astype(np.uint8)

        for cname, kind, sev, n_s in conds:
            x_c, corrupted = apply_silent_failure_batch(
                x_noisy, n_s, kind, seed=S1.CORRUPTION_SEED * 10000 + bi,
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

            rho, _ = R.rho(s_c, a_c, table, S1.CANONICAL_SIGMA_EPS, sigma0=sigma, method="gram")
            f_hat = R.psm_predict(rho)
            arrs[f"{cname}__fhat"][sl] = f_hat.numpy().astype(np.int8)
            if bi == 0:
                S1.check_against_library(s_c[:64], a_c[:64], table, f_hat[:64])
            loo, ins = S1.loo_and_insample(s_c, a_c, table, f_hat, loo_chunk)
            arrs[f"{cname}__psm_loo"][sl] = loo.float().numpy()
            arrs[f"{cname}__psm_ins"][sl] = ins.float().numpy()
            wrong = torch.nonzero(f_hat != y).flatten()
            if len(wrong):
                loo_w, ins_w = S1.loo_and_insample(s_c[wrong], a_c[wrong], table, y[wrong], loo_chunk)
                loo[wrong], ins[wrong] = loo_w, ins_w
            arrs[f"{cname}__oracle_loo"][sl] = loo.float().numpy()
            arrs[f"{cname}__oracle_ins"][sl] = ins.float().numpy()
        pos += b
        if bi % 10 == 0:
            el = time.time() - t0
            eta = el / pos * (n_total - pos)
            print(f"  [sigma={sigma}] batch {bi} windows {pos}/{n_total}  {el/60:.1f} min  "
                  f"eta {eta/60:.1f} min", flush=True)
    for v in arrs.values():
        v.flush()
    if pos != n_total:
        raise RuntimeError(f"scored {pos} windows, expected {n_total}")
    (out_dir / "raw" / "COMPLETE.json").write_text(json.dumps(
        {"n_windows": pos, "batch_size": batch_size, "max_batches": max_batches,
         "sigma0": sigma, "minutes": (time.time() - t0) / 60}, indent=2))
    return pos


def summarize(sigma: float) -> dict:
    out_dir = _out_dir(sigma)
    raw = out_dir / "raw"
    arrs = S1.open_raw(raw, 0, "r")
    y = np.asarray(arrs["y"])
    avail = np.asarray(arrs["avail"]).astype(bool)
    strata = {"all": np.ones_like(y, dtype=bool), "fault": y > 0, "nofault": y == 0}

    rows, roc_rows = [], []
    for cname, kind, sev, n_s in S1.conditions():
        label_full = np.asarray(arrs[f"{cname}__label"])
        delta_full = np.asarray(arrs[f"{cname}__delta"])
        fhat = np.asarray(arrs[f"{cname}__fhat"]).astype(np.int64)
        from sklearn.metrics import f1_score
        psm_f1 = f1_score(y, fhat, average="macro", labels=S1.LABELS26, zero_division=0)
        for sname, wsel in strata.items():
            m = avail & wsel[:, None]
            lab = label_full[m]
            dpos = delta_full[m][lab == 1]
            base = {"condition": cname, "corruption": cname.rsplit("_s", 1)[0], "kind": kind,
                    "severity": sev, "n_sensors": n_s, "stratum": sname,
                    "psm_f1_macro26_all": float(psm_f1),
                    "corrupted_unchanged_frac": float((dpos == 0).mean()) if len(dpos) else None,
                    "corrupted_delta_median_pu": float(np.median(dpos)) if len(dpos) else None}
            for refit in S1.REFITS:
                for score in S1.SCORES:
                    sc = np.asarray(arrs[f"{cname}__{refit}_{score}"])[m]
                    met = S1.detection_metrics(lab, sc)
                    roc = met.pop("_roc", None)
                    rows.append({**base, "refit": refit, "score": score, **met})
                    if roc is not None and sname == "all":
                        roc_rows.append({"condition": cname, "refit": refit, "score": score,
                                         "fpr": S1.ROC_GRID.tolist(), "tpr": roc.tolist()})
        print(f"[summary sigma={sigma}] {cname} done", flush=True)

    def pick(cname, refit, score, stratum="all"):
        return next(r for r in rows if r["condition"] == cname and r["refit"] == refit
                    and r["score"] == score and r["stratum"] == stratum)

    gate = {f"{S1.GATE_CORRUPTION}_s{s}": pick(f"{S1.GATE_CORRUPTION}_s{s}", "psm", "loo")["auroc"]
            for s in S1.N_SENSORS}
    s52 = {c: {"loo": pick(c, "psm", "loo")["auroc"], "ins": pick(c, "psm", "ins")["auroc"]}
           for c, *_ in S1.conditions()}
    passed = all(v is not None and v >= S1.GATE_AUROC for v in gate.values())
    return {"rows": rows, "roc": roc_rows, "gate_s5_1_psm_loo_auroc": gate,
            "gate_threshold": S1.GATE_AUROC, "gate_passed": passed, "s5_2_psm_auroc": s52}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--loo-chunk", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=None, help="one value; default: all of SIGMA_GRID")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sigmas = [args.sigma] if args.sigma is not None else list(SIGMA_GRID)

    for sigma in sigmas:
        sigma_idx = list(SIGMA_GRID).index(sigma) if sigma in SIGMA_GRID else 99
        out_dir = _out_dir(sigma)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.summarize_only:
            score_pass_noisy(sigma, sigma_idx, Path(args.data_root), args.batch_size,
                             args.max_batches, args.loo_chunk)
        complete = json.loads((out_dir / "raw" / "COMPLETE.json").read_text())
        result = summarize(sigma)
        meta = {"git_commit": S1.git_commit(), "split": "val", "k": 0, "sigma0": sigma,
                "sigma_eps": S1.CANONICAL_SIGMA_EPS, "corruption_seed": S1.CORRUPTION_SEED,
                "noise_seed_base": (NOISE_SEED + sigma_idx) * 10000,
                "batch_size": complete["batch_size"], "n_windows": complete["n_windows"],
                "partial_run": complete["max_batches"] is not None,
                "f_hat": "PSM argmin rho on the corrupted (and noised) input (oracle = true class, diagnostic)",
                "note": "Rerun of results/pcstgnn/audit/s5_1_silent_detection/ (sigma0=0.0) at "
                        "ambient measurement noise sigma0>0. Noise is injected into x before the "
                        "silent corruption; rho's sigma0 argument equals the injected sigma."}
        name = ("silent_detection.json" if complete["max_batches"] is None
                else "silent_detection_PARTIAL.json")
        (out_dir / name).write_text(json.dumps({"meta": meta, **{k: v for k, v in result.items()
                                                                  if k != "roc"}}, indent=2))
        (out_dir / name.replace(".json", "_roc.json")).write_text(json.dumps(result["roc"]))
        print(json.dumps({"sigma": sigma, "gate_s5_1": result["gate_s5_1_psm_loo_auroc"],
                          "gate_passed": result["gate_passed"], "s5_2": result["s5_2_psm_auroc"]},
                         indent=2))


if __name__ == "__main__":
    main()
