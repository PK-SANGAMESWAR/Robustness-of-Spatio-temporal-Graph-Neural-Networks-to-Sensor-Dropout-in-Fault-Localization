"""G-TF2 on the test split (``PC-STGNN/stage_f_gtf2_test_preregistration.md``).

    scoring (GPU, resumable, all 3 seeds in one pass):  nice -n 19 uv run python scripts/run_gtf2_test.py score
    then:   run_gtf2_test.py aggregate ; run_gtf2_test.py summarize
    smoke:  run_gtf2_test.py score --max-shards 1 --out-root <tmp dir>   (writes outside results/; do not look at the numbers)

Same design as the registered validation G-TF2 (``run_gtf.py --gate gtf2``), one change: the split is ``test`` (all 6,175 events, no halves) and the
thresholds are the frozen validation-half-A thresholds (``thresholds_gtf2_sigma_0.01.npz``, sha256 in the pre-registration). The detector-1 scores
(in-sample PSM residual) and the flatline scores are computed live per batch with the same functions as the validation run (``run_gtf._clean_scores``,
``run_gtf.flatline_score``, ``run_gtf.detector_flags``); noise and corruption draws use the validation seeds (``run_gtf._noisy``, ``S1.CORRUPTION_SEED``)
and the test loader's batch index. The residual-only arm 3b is not scored on the test split. No model is trained. Nothing is written outside
``--out-root`` (default ``results/stage_f/gtf2_test``). An official run (no ``--max-shards``) refuses to start unless the pre-registration is committed and
unmodified and the threshold file's sha256 equals the one recorded there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import run_gtf as G  # noqa: E402

PREREG = REPO / "PC-STGNN/stage_f_gtf2_test_preregistration.md"
THR_FILE = REPO / "results/stage_f/gtf2/core/thresholds_gtf2_sigma_0.01.npz"
DEFAULT_OUT = REPO / "results/stage_f/gtf2_test"
SIGMA = 0.01
BATCH = G.S1_BATCH


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_ready_test() -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not PREREG.exists() or not G.git_clean_and_tracked(PREREG):
        problems.append(f"{G._rel(PREREG)} is not committed or has uncommitted changes")
    elif f"sha256 of thresholds_gtf2_sigma_0.01.npz: {sha256(THR_FILE)}" not in PREREG.read_text(encoding="utf-8"):
        problems.append("the threshold file's sha256 is not the one recorded in the pre-registration")
    if not THR_FILE.exists():
        problems.append("frozen validation thresholds missing")
    for s in G.SEEDS:
        if not (REPO / f"results/s4_4/r5/pcstgnn/seed_{s:05d}/checkpoint_best.pt").exists():
            problems.append(f"R5 checkpoint missing for seed {s}")
    return (not problems), problems


def selfcheck(threads: int) -> dict:
    """The live scoring path on validation batches equals the stored S5.1 scores and the stored G-TF2 clean flags (difference 0)."""
    import run_silent_detection as S1
    chk = G.check_stored_scores(SIGMA, batch_ids=(0, 300), threads=threads)
    T_ = np.load(THR_FILE)
    ds, loader, mean, std, table, a_events = G._setup(REPO / "dataset_fullscale_50", threads)
    batch = next(iter(loader))
    xn = G._noisy(batch["x"], SIGMA, std, 0)
    _, ins, av = G._clean_scores(xn, SIGMA, mean, std, table, 128)
    fl = G.detector_flags(ins, G.flatline_score(xn, mean, std), av, T_["thr_ins"], T_["thr_flat"])["union"]
    rows_b = np.array([G.event_of(w) not in a_events for w in batch["window_id"]])
    stored = T_["flags_clean_B_union"][:len(rows_b)][rows_b]
    chk["clean_flags_equal_stored_G_TF2"] = bool(np.array_equal(fl[rows_b], stored))
    chk["passed"] = bool(chk["passed"] and chk["clean_flags_equal_stored_G_TF2"])
    return chk


import os  # noqa: E402
SHARD = int(os.environ.get("GTF2_TEST_SHARD", 20))        # loader batches per shard (about 35 min); a shard is written whole, so an interrupted run resumes at the next shard
SHARD_DIR = "shards"
ARMS = ("nogate", "gtf2")


def _stat_names(dnames=("ins", "flat", "union")):
    return ("tp", "pos", "fp", "neg", "flagged", "win") + tuple(f"{n}_{k}" for n in dnames for k in ("tp", "fp", "flagged"))


def stage_score(out_root: Path, threads: int, max_shards: int | None, device_name: str | None, smoke: bool) -> int:
    """All 3 seeds in one pass: the corruption, the detector scores and the flags do not depend on the seed, so they are computed once per batch.
    The residual-only arm 3b of the validation design is not scored on the test split (descriptive only; see the pre-registration)."""
    import torch
    import run_silent_detection as S1
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    from fault_localization.dropout.silent import apply_silent_failure_batch
    from fault_localization.physics import observation as OBS
    from fault_localization.physics import residual as R
    from fault_localization.physics import table as T

    if not smoke and SHARD != 20:
        raise RuntimeError("an official run uses 20 batches per shard")
    if not smoke:
        ok, problems = check_ready_test()
        if not ok:
            raise RuntimeError(f"G-TF2 test pre-registration is not complete: {problems}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    chk = G.check_gate_equals_training_dropout(REPO / "dataset_fullscale_50")
    if not chk["passed"]:
        raise RuntimeError(f"consistency check 1 failed: {chk}")
    torch.set_num_threads(threads)
    mean, std = OBS.load_norm_stats(REPO / "dataset_fullscale_50")
    ds = WindowGraphDataset(str(REPO / "dataset_fullscale_50"), "test")
    loader = make_dataloader(ds, BATCH, shuffle=False)
    tbl = T.load_table(REPO / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"]).double()
    models = {s: G._load_r5(s, device) for s in G.SEEDS}
    T_ = np.load(THR_FILE)
    thr_ins, thr_flat = T_["thr_ins"], T_["thr_flat"]
    conds = G.scope_conditions("core")
    names = [c[0] for c in conds] + ["clean"]
    sdir = out_root / SHARD_DIR
    sdir.mkdir(parents=True, exist_ok=True)
    dn = ("ins", "flat", "union")

    def new_shard():
        return {"y": [], "P": {(s, c, a): [] for s in G.SEEDS for c in names for a in (*ARMS, "oracle") if not (c == "clean" and a == "oracle")},
                "st": {c: {k: 0 for k in _stat_names()} for c in names}}

    sh, done, t0 = new_shard(), 0, time.time()
    n_batches = len(loader)
    for bi, batch in enumerate(loader):
        shard_id = bi // SHARD
        f = sdir / f"shard_{shard_id:04d}.npz"
        if f.exists():
            continue
        if max_shards is not None and done >= max_shards:
            break
        x, y = batch["x"], batch["y"]
        b = x.shape[0]
        xn = G._noisy(x, SIGMA, std, bi)
        sh["y"].extend(y.tolist())
        variants = [("clean", xn, None)]
        for cname, kind, sev, n_s in conds:
            x_c, corrupted = apply_silent_failure_batch(
                xn, n_s, kind, seed=S1.CORRUPTION_SEED * 10000 + bi, norm_stats=(mean, std), severity=sev)
            label = np.zeros((b, 25), bool)
            for row, nodes in enumerate(corrupted):
                label[row, nodes] = True
            variants.append((cname, x_c, label))
        for cname, xv, label in variants:
            _, ins, av = G._clean_scores(xv, SIGMA, mean, std, table, 128)
            fd = G.detector_flags(ins, G.flatline_score(xv, mean, std), av, thr_ins, thr_flat)
            fl = fd["union"]
            d = sh["st"][cname]
            if label is None:
                d["fp"] += int((fl & av).sum()); d["neg"] += int(av.sum())
                for n in dn:
                    d[f"{n}_fp"] += int((fd[n] & av).sum()); d[f"{n}_flagged"] += int(fd[n].sum())
            else:
                d["tp"] += int((fl & label).sum()); d["pos"] += int((label & av).sum())
                d["fp"] += int((fl & ~label & av).sum()); d["neg"] += int((~label & av).sum())
                for n in dn:
                    d[f"{n}_tp"] += int((fd[n] & label).sum()); d[f"{n}_fp"] += int((fd[n] & ~label & av).sum())
                    d[f"{n}_flagged"] += int(fd[n].sum())
            d["flagged"] += int(fl.sum()); d["win"] += b
            gated = G.gate_zero(xv, torch.as_tensor(fl))
            oracle = G.gate_zero(xv, torch.as_tensor(label)) if label is not None else None
            for s, m in models.items():
                sh["P"][(s, cname, "nogate")] += G._predict(m, xv, SIGMA, device)
                sh["P"][(s, cname, "gtf2")] += G._predict(m, gated, SIGMA, device)
                if oracle is not None:
                    sh["P"][(s, cname, "oracle")] += G._predict(m, oracle, SIGMA, device)
        if (bi + 1) % SHARD == 0 or bi == n_batches - 1:
            arrs = {"y": np.asarray(sh["y"], np.int16)}
            for (s, c, a), v in sh["P"].items():
                arrs[f"P|{s}|{c}|{a}"] = np.asarray(v, np.int16)
            for c, d in sh["st"].items():
                arrs[f"S|{c}"] = np.asarray([d[k] for k in _stat_names()], np.int64)
            np.savez_compressed(f.with_suffix(".tmp.npz"), **arrs)
            f.with_suffix(".tmp.npz").rename(f)
            print(f"  [test] shard {shard_id} written, batch {bi + 1}/{n_batches}, {(time.time() - t0) / 60:.1f} min", flush=True)
            sh, done = new_shard(), done + 1
    return done


def stage_aggregate(out_root: Path) -> list[Path]:
    from score_s4_4_gate4_run import macro_f1
    files = sorted((out_root / SHARD_DIR).glob("shard_*.npz"))
    zs = [np.load(f) for f in files]
    y = np.concatenate([z["y"] for z in zs]).astype(np.int64)
    conds = G.scope_conditions("core")
    names = [c[0] for c in conds] + ["clean"]
    snames = _stat_names()
    stat = {c: dict(zip(snames, np.sum([z[f"S|{c}"] for z in zs], axis=0).tolist())) for c in names}
    dn = ("ins", "flat", "union")
    written = []
    for seed in G.SEEDS:
        pred = {(c, a): np.concatenate([z[f"P|{seed}|{c}|{a}"] for z in zs]).astype(np.int64)
                for c in names for a in (*ARMS, "oracle") if not (c == "clean" and a == "oracle")}
        res = {"seed": seed, "sigma0": SIGMA, "scope": "core", "detector": "ins+flatline", "split": "test", "half": "all", "n_windows": int(y.size),
               "n_shards": len(files), "fpr_target": G.FPR_GTF2, "gate": "gtf2", "partial_run": False, "conditions": {},
               "threshold_sha256": sha256(THR_FILE)}
        f1_clean = macro_f1(y, pred[("clean", "nogate")])
        for c in names:
            f1 = {a: macro_f1(y, pred[(c, a)]) for a in (*ARMS, "oracle") if (c, a) in pred}
            d = stat[c]
            e = {"f1": f1}
            if c == "clean":
                e["realized_fpr_clean_windows"] = d["fp"] / max(d["neg"], 1)
                e["flagged_per_window"] = d["flagged"] / max(d["win"], 1)
                e["delta_gtf2_minus_nogate"] = f1["gtf2"] - f1["nogate"]
                e["detectors"] = {n: {"realized_fpr_clean_windows": d[f"{n}_fp"] / max(d["neg"], 1),
                                      "flagged_per_window": d[f"{n}_flagged"] / max(d["win"], 1)} for n in dn}
            else:
                e["recovered"] = G.recovered_fraction(f1["gtf2"], f1["nogate"], f1["oracle"])
                e["corruption_loss"] = f1_clean - f1["nogate"]
                e["oracle_loss"] = f1_clean - f1["oracle"]
                e["f1_clean_nogate"] = f1_clean
                e["tpr"] = d["tp"] / max(d["pos"], 1)
                e["realized_fpr_clean_sensors_of_corrupted_windows"] = d["fp"] / max(d["neg"], 1)
                e["flagged_per_window"] = d["flagged"] / max(d["win"], 1)
                e["detectors"] = {n: {"tpr": d[f"{n}_tp"] / max(d["pos"], 1),
                                      "realized_fpr_clean_sensors_of_corrupted_windows": d[f"{n}_fp"] / max(d["neg"], 1),
                                      "flagged_per_window": d[f"{n}_flagged"] / max(d["win"], 1)} for n in dn}
            res["conditions"][c] = e
        out = out_root / f"sigma_{SIGMA:g}" / f"seed_{seed:05d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        written.append(out)
    return written


def stage_summarize(out_root: Path) -> None:
    G.stage_summarize("core", out_root, arm="gtf2")                           # summary.json (G-TF2 sign rules)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["score", "aggregate", "summarize", "selfcheck", "check-ready"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-shards", type=int, default=None, help="smoke only; needs another --out-root")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out_root = Path(a.out_root)
    smoke = a.max_shards is not None
    if smoke and out_root.resolve() == DEFAULT_OUT.resolve():
        raise SystemExit("a smoke run (--max-shards) must use another --out-root")
    if a.stage == "check-ready":
        ok, problems = check_ready_test()
        print("ready" if ok else problems)
        return 0 if ok else 1
    if a.stage == "selfcheck":
        r = selfcheck(a.threads)
        print(json.dumps(r, indent=1))
        return 0 if r["passed"] else 1
    if a.stage == "score":
        print("shards written:", stage_score(out_root, a.threads, a.max_shards, a.device, smoke))
        return 0
    if a.stage == "aggregate":
        print("wrote", [str(p) for p in stage_aggregate(out_root)])
        return 0
    stage_summarize(out_root)
    print("wrote", out_root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
