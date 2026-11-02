"""Effect of the corrected targeted-dropout ranking on the seed-42 checkpoints, without retraining (author, 2026-11-01).

The paper ranks the μPMUs nearest to a fault by hop distance on `build_full_topology_graph()`, which treats the two normally open
switches Sw7 and Sw8 as closed (reports/findings/paper1_open_switch.md). Here the rank is rebuilt from the operated feeder: the open
switches removed, the same hop metric, the same tie-break by node index. Everything else is the paper's code path
(scripts/run_targeted_dropout_study.py): its loader (batch 32, no shuffle), its fallback seed `10000*k + batch_idx` for the no-fault
windows, its zeroing of all six channels, its soft vote and its macro-F1.

Each window is scored twice: with the rank that the paper saved (the control) and with the corrected rank. A window whose mask is the same
under both ranks keeps the same prediction (same input), so only windows with a changed mask are run twice. CPU, `torch_threads()` threads.
No paper file and no result file of Paper 1 is changed; outputs go to rag-for-dropout/eval/targeted_rank/ and reports/findings/.

    PYTHONPATH=src:. OMP_NUM_THREADS=2 nice -n 19 .venv/bin/python -m gridsense.eval.targeted_rank_check run --arm dropout_aware
    PYTHONPATH=src:. OMP_NUM_THREADS=2 nice -n 19 .venv/bin/python -m gridsense.eval.targeted_rank_check run --arm standard
    PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.targeted_rank_check report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gridsense.config import ROOT, TRACK, load_config, torch_threads

DATA = ROOT / "dataset_fullscale_50"
PAPER = ROOT / "results" / "dropout_study_targeted"
ARMS = {"standard": ROOT / "results/fullscale_final/rgsage_max/seed_00042",
        "dropout_aware": ROOT / "results/dropout_aware/rgsage_max/seed_00042"}
KS = (1, 2, 5, 10)
OUT = TRACK / "eval" / "targeted_rank"
RESULT_JSON = TRACK / "reports" / "findings" / "targeted_rank_seed42.json"
IDENTITY_TOL = 1e-3          # the S2 rule for k > 0: macro-F1 within 1e-3 of the paper value
C = 26


def operated_rank(node_mapping_path: Path) -> np.ndarray:
    """`build_proximity_rank` of the paper with one change: the feeder graph is the operated feeder (open switches removed)."""
    import networkx as nx

    from fault_localization.config.constants import LABEL_TO_FAULT_BUS, PMU_BUSES
    from gridsense.ingest.feeder_docs import feeder_distances

    mapping = json.loads(Path(node_mapping_path).read_text())
    graph = feeder_distances()
    pmu_sorted = sorted(PMU_BUSES)
    if any(mapping[str(b)] != i for i, b in enumerate(pmu_sorted)):
        raise ValueError("node_mapping.json does not follow sorted(PMU_BUSES)")
    rank = np.zeros((25, 25), dtype=np.int64)
    for label in range(1, 26):
        hops = nx.single_source_shortest_path_length(graph, LABEL_TO_FAULT_BUS[label])
        rank[label - 1] = [node for _, node in sorted((hops[b], mapping[str(b)]) for b in pmu_sorted)]
    return rank


def changed_labels(a: np.ndarray, b: np.ndarray, k: int) -> list[int]:
    """Fault classes (1..25) whose set of k dropped nodes differs between the two ranks."""
    return [lab for lab in range(1, 26) if set(a[lab - 1, :k].tolist()) != set(b[lab - 1, :k].tolist())]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run(arm: str, limit_batches: int | None = None) -> Path:
    import torch

    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    from fault_localization.config.constants import LABEL_TO_FAULT_BUS
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    from fault_localization.dropout.targeted import apply_targeted_dropout_batch, build_proximity_rank
    from fault_localization.models.voting import soft_vote
    from run_dropout_study import load_trained_model

    torch.set_num_threads(torch_threads())
    nm = DATA / "processed" / "node_mapping.json"
    saved = np.load(PAPER / "proximity_rank.npy")
    rebuilt = build_proximity_rank(nm)
    rank_op = operated_rank(nm)
    ident = {"saved_rank_equals_rebuilt_as_built": bool(np.array_equal(saved, rebuilt))}
    if not ident["saved_rank_equals_rebuilt_as_built"]:
        raise SystemExit("the rank saved by the paper differs from a fresh as-built rank: stop and report")
    changed = {k: changed_labels(saved, rank_op, k) for k in KS}

    ds = WindowGraphDataset(str(DATA), "test")
    model, cfg = load_trained_model(ARMS[arm], ds.edge_index, torch.device("cpu"))
    loader = make_dataloader(ds, 32, shuffle=False, num_workers=0)
    n_total = len(loader) if not limit_batches else limit_batches
    labels, ids, pred0 = [], [], []
    pb, pc, ch = ({k: [] for k in KS} for _ in range(3))
    t0 = time.perf_counter()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if limit_batches and bi >= limit_batches:
                break
            x, y = batch["x"], batch["y"]
            labels.append(y.numpy()), ids.extend(batch["window_id"])
            pred0.append(soft_vote(model(x))[1].numpy())
            for k in KS:
                seed = 10000 * k + bi                               # the paper's fallback seed for no-fault windows
                xb, _ = apply_targeted_dropout_batch(x, y, k=k, rank=saved, fallback_seed=seed)
                xc, _ = apply_targeted_dropout_batch(x, y, k=k, rank=rank_op, fallback_seed=seed)
                diff = (xb != xc).flatten(1).any(1)
                gb = soft_vote(model(xb))[1]
                gc = gb.clone()
                if bool(diff.any()):
                    gc[diff] = soft_vote(model(xc[diff]))[1]
                pb[k].append(gb.numpy()), pc[k].append(gc.numpy()), ch[k].append(diff.numpy())
            if (bi + 1) % 500 == 0:
                el = time.perf_counter() - t0
                print(f"[{arm}] batch {bi + 1}/{n_total}  {el / 60:.1f} min  ETA {el / (bi + 1) * (n_total - bi - 1) / 60:.1f} min", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_seed42" + (f"_limit{limit_batches}" if limit_batches else "")
    arrays = {"labels": np.concatenate(labels).astype(np.int8), "pred0": np.concatenate(pred0).astype(np.int8),
              "window_ids": np.array(ids)}
    for k in KS:
        arrays[f"built_k{k}"] = np.concatenate(pb[k]).astype(np.int8)
        arrays[f"corrected_k{k}"] = np.concatenate(pc[k]).astype(np.int8)
        arrays[f"changed_k{k}"] = np.concatenate(ch[k])
    np.savez_compressed(OUT / f"{tag}.npz", **arrays)
    ckpt = ARMS[arm] / "checkpoint_best.pt"
    meta = {"arm": arm, "n_windows": int(len(arrays["labels"])), "limit_batches": limit_batches, "elapsed_s": time.perf_counter() - t0,
            "torch": torch.__version__, "threads": torch_threads(), "device": "cpu", "batch_size": 32,
            "checkpoint": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": _sha(ckpt), "f1_average": cfg.get("metrics", {}).get("f1_average", "macro"),
            "rank_checks": ident, "changed_fault_labels": {str(k): v for k, v in changed.items()},
            "changed_fault_buses": {str(k): [int(LABEL_TO_FAULT_BUS[lab]) for lab in v] for k, v in changed.items()},
            "git_commit": _git("rev-parse", "--short", "HEAD"), "split_test_sha256": _sha(DATA / "splits" / "test.txt")}
    (OUT / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return OUT / f"{tag}.npz"


# ---------------------------------------------------------------------------------------------------------- analysis
def macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 over the 26 classes for confusion matrices [..., 26, 26] (rows true, columns predicted); a class with no support or
    prediction scores 0 (sklearn zero_division=0)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp, fn = cm.sum(-2) - tp, cm.sum(-1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean(-1)


def _per_run_cm(run_idx: np.ndarray, labels: np.ndarray, pred: np.ndarray, n_runs: int) -> np.ndarray:
    flat = run_idx * C * C + labels.astype(np.int64) * C + pred.astype(np.int64)
    return np.bincount(flat, minlength=n_runs * C * C).reshape(n_runs, C * C).astype(np.float64)


def paired_run_bootstrap(run_idx, labels, pred_a, pred_b, n_boot: int, seed: int = 42, level: float = 0.95) -> dict:
    """Bootstrap over test RUNS of macro-F1(b) - macro-F1(a) on the same windows. Covers the test sampling of one checkpoint only."""
    n_runs = int(run_idx.max()) + 1
    ca, cb = _per_run_cm(run_idx, labels, pred_a, n_runs), _per_run_cm(run_idx, labels, pred_b, n_runs)
    mult = np.random.default_rng(seed).multinomial(n_runs, np.full(n_runs, 1.0 / n_runs), size=n_boot).astype(np.float64)
    d = macro_f1_from_cm((mult @ cb).reshape(n_boot, C, C)) - macro_f1_from_cm((mult @ ca).reshape(n_boot, C, C))
    a = (1 - level) / 2
    return {"ci_low": float(np.quantile(d, a)), "ci_high": float(np.quantile(d, 1 - a)), "n_runs": n_runs, "n_boot": n_boot}


def analyze(arm: str, n_boot: int, tag: str | None = None) -> dict:
    from sklearn.metrics import f1_score

    tag = tag or f"{arm}_seed42"
    z = np.load(OUT / f"{tag}.npz")
    meta = json.loads((OUT / f"{tag}.meta.json").read_text())
    labels = z["labels"].astype(np.int64)
    runs = np.array([w.rsplit("_w", 1)[0] for w in z["window_ids"]])
    _, run_idx = np.unique(runs, return_inverse=True)
    paper = {}
    for row in (r.split(",") for r in (PAPER / f"targeted_per_seed_{arm}.csv").read_text().splitlines()[1:]):
        if int(row[2]) == 42:
            paper[int(row[3])] = {"f1": float(row[4]), "acc": float(row[5])}
    mean3 = {}
    for row in (r.split(",") for r in (PAPER / f"targeted_curves_{arm}.csv").read_text().splitlines()[1:]):
        mean3[int(row[2])] = {"f1_mean_3_seeds": float(row[3]), "f1_std_3_seeds": float(row[4])}
    f1 = lambda p: float(f1_score(labels, p, average=meta["f1_average"], zero_division=0))
    out = {"meta": meta, "k": {}}
    for k in (0, *KS):
        built = z["pred0"] if k == 0 else z[f"built_k{k}"]
        corr = built if k == 0 else z[f"corrected_k{k}"]
        row = {"paper_seed42_f1": paper[k]["f1"], "paper_seed42_acc": paper[k]["acc"], **mean3[k],
               "cpu_as_built_f1": f1(built), "cpu_corrected_f1": f1(corr), "cpu_as_built_acc": float(np.mean(built == labels)),
               "cpu_corrected_acc": float(np.mean(corr == labels))}
        row["as_built_minus_paper"] = row["cpu_as_built_f1"] - row["paper_seed42_f1"]
        row["identity_within_tol"] = abs(row["as_built_minus_paper"]) <= IDENTITY_TOL
        row["corrected_minus_paper"] = row["cpu_corrected_f1"] - row["paper_seed42_f1"]
        row["corrected_minus_as_built"] = row["cpu_corrected_f1"] - row["cpu_as_built_f1"]
        if k:
            chg = z[f"changed_k{k}"]
            fault = labels > 0
            row["n_changed_windows"] = int(chg.sum())
            row["share_of_fault_windows_changed"] = float(chg.sum() / fault.sum())
            row["acc_on_changed_as_built"] = float(np.mean(built[chg] == labels[chg])) if chg.any() else None
            row["acc_on_changed_corrected"] = float(np.mean(corr[chg] == labels[chg])) if chg.any() else None
            row["n_predictions_differ"] = int(np.sum(built != corr))
            row["bootstrap_corrected_minus_as_built"] = paired_run_bootstrap(run_idx, labels, built, corr, n_boot)
        out["k"][str(k)] = row
    out["vectorised_f1_check"] = float(max(abs(macro_f1_from_cm(_per_run_cm(run_idx, labels, z[f"built_k{k}"], int(run_idx.max()) + 1).sum(0).reshape(C, C))
                                              - f1(z[f"built_k{k}"])) for k in KS))
    return out


def markdown(res: dict) -> str:
    L = []
    for arm, r in res["arms"].items():
        m = r["meta"]
        L += [f"**{arm}**, seed 42 (`{m['checkpoint']}`, sha256 `{m['checkpoint_sha256'][:8]}…`), {m['n_windows']:,} test windows, CPU, {m['threads']} threads, {m['elapsed_s'] / 60:.1f} min.", "",
              "| k | paper seed 42 | CPU, paper rank (control) | CPU, corrected rank | control − paper | corrected − paper | corrected − control (95 % CI, test runs) | windows with a changed mask | accuracy on those: control → corrected |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, row in r["k"].items():
            if k == "0":
                L.append(f"| 0 | {row['paper_seed42_f1']:.4f} | {row['cpu_as_built_f1']:.4f} | (same) | {row['as_built_minus_paper']:+.4f} | – | – | – | – |")
                continue
            b = row["bootstrap_corrected_minus_as_built"]
            L.append(f"| {k} | {row['paper_seed42_f1']:.4f} | {row['cpu_as_built_f1']:.4f} | {row['cpu_corrected_f1']:.4f} | {row['as_built_minus_paper']:+.4f} | "
                     f"{row['corrected_minus_paper']:+.4f} | {row['corrected_minus_as_built']:+.4f} ({b['ci_low']:+.4f} to {b['ci_high']:+.4f}) | "
                     f"{row['n_changed_windows']:,} ({row['share_of_fault_windows_changed']:.1%} of fault windows) | "
                     f"{row['acc_on_changed_as_built']:.3f} → {row['acc_on_changed_corrected']:.3f} |")
        L.append("")
    if "dropout_aware_minus_standard" in res:
        L += ["**Dropout-aware minus standard under the same targeting** (macro-F1; one training seed; no confidence interval):", "",
              "| k | paper, seed 42 | CPU, paper rank | CPU, corrected rank | paper, mean of 3 seeds (context only) |", "|---|---|---|---|---|"]
        for k, v in res["dropout_aware_minus_standard"].items():
            L.append(f"| {k} | {v['paper_seed42']:+.4f} | {v['cpu_paper_rank']:+.4f} | {v['cpu_corrected_rank']:+.4f} | {v['paper_3_seed_mean']:+.4f} |")
        L.append("")
    return "\n".join(L)


def report(n_boot: int | None = None) -> dict:
    n_boot = n_boot or load_config()["repeats"]["bootstrap_resamples"]
    res = {"generated_by": "gridsense.eval.targeted_rank_check", "identity_tolerance_f1": IDENTITY_TOL,
           "arms": {a: analyze(a, n_boot) for a in ARMS if (OUT / f"{a}_seed42.npz").exists()}}
    if len(res["arms"]) == 2:       # derived in code, never typed by hand
        adv = {}
        for k in map(str, KS):
            sd, da = res["arms"]["standard"]["k"][k], res["arms"]["dropout_aware"]["k"][k]
            adv[k] = {"paper_seed42": da["paper_seed42_f1"] - sd["paper_seed42_f1"],
                      "cpu_paper_rank": da["cpu_as_built_f1"] - sd["cpu_as_built_f1"],
                      "cpu_corrected_rank": da["cpu_corrected_f1"] - sd["cpu_corrected_f1"],
                      "paper_3_seed_mean": da["f1_mean_3_seeds"] - sd["f1_mean_3_seeds"]}
        res["dropout_aware_minus_standard"] = adv
    res["range_corrected_minus_paper_seed42"] = {
        a: [min(v["k"][str(k)]["corrected_minus_paper"] for k in KS), max(v["k"][str(k)]["corrected_minus_paper"] for k in KS)]
        for a, v in res["arms"].items()}
    RESULT_JSON.write_text(json.dumps(res, indent=2) + "\n")
    (OUT / "table.md").write_text(markdown(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--limit-batches", type=int, default=None, help="timing/smoke only; the output is tagged and never reported")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        print(run(a.arm, a.limit_batches))
    else:
        res = report()
        print(RESULT_JSON, "arms:", list(res["arms"]))
        print((OUT / "table.md").read_text())


if __name__ == "__main__":
    main()
