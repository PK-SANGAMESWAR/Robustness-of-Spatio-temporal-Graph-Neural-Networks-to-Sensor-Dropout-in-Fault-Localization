"""Leakage checks on the real data (O19 step 2; implementation-plan.md § 13.9). CPU only; `test` is not read (val stands in for the sanity run).

1. train / val / test run sets are pairwise disjoint; the store holds only train runs; simulation seeds are unique and disjoint.
2. The FAISS store reproduces the exact torch search used for Layer 0 (same neighbours, votes, agreement) on a random val subset.
3. Sanity run: put val queries INTO the store -> kNN accuracy must jump to ~1.0; leave-one-run-out must bring it back to the clean value;
   scenario exclusion must reproduce the held-out value.
Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.leakage_checks
"""
from __future__ import annotations

import json

import numpy as np
import torch

from gridsense.cases.build import STORE_DIR, build_store, expected_fingerprint, run_seeds
from gridsense.cases.store import CaseStore
from gridsense.config import TRACK
from gridsense.data.splits import event_of, scenario_of, split_events, stratified_run_subsample
from gridsense.eval.layer0_knn import knn_path
from gridsense.eval.s2_k0 import LAT, split_arrays
from gridsense.eval.scenario_control import _vote
from gridsense.model.latent_repr import represent
from gridsense.model.predictor import checkpoint_sha256

OUT = TRACK / "reports" / "layer0"


def _vote_acc(store: CaseStore, q: np.ndarray, truth: np.ndarray, ckpt: str, k: int = 10, **excl) -> np.ndarray:
    sim, idx = store.search(q, k, query_checkpoint_sha256=ckpt, **excl)
    pred = _vote(torch.from_numpy(store.labels[idx].astype(np.int64)), torch.from_numpy(sim), k).numpy()
    return pred == truth


def main() -> None:
    torch.set_num_threads(4)
    ckpt = checkpoint_sha256(42)
    res: dict = {}
    runs = {s: set(split_events(s)) for s in ("train", "val", "test")}
    store = build_store()                                              # raises LeakageError on any forbidden run or seed
    sample = (LAT / "train_sub10" / "window_ids.txt").read_text().split()
    seeds = run_seeds()
    res["disjointness"] = {"n_runs": {s: len(v) for s, v in runs.items()},
                           "train_val": len(runs["train"] & runs["val"]), "train_test": len(runs["train"] & runs["test"]), "val_test": len(runs["val"] & runs["test"]),
                           "store_runs": len(set(store.runs)), "store_runs_not_in_train": len(set(store.runs) - runs["train"]),
                           "store_runs_in_val_or_test": len(set(store.runs) & (runs["val"] | runs["test"])),
                           "distinct_seeds_all_runs": len(set(seeds.values())), "n_runs_with_seed": len(seeds),
                           "store_seeds_in_val_or_test": len({seeds[r] for r in store.runs} & {seeds[r] for r in runs["val"] | runs["test"]}),
                           "store_fingerprint_matches_expected": store.manifest["fingerprint"] == expected_fingerprint(sample)}
    loaded = CaseStore.load(STORE_DIR, store.manifest["fingerprint"])  # saved store == rebuilt store
    res["saved_store_equals_rebuilt"] = bool(np.array_equal(loaded.vectors, store.vectors) and loaded.window_ids == store.window_ids)

    ids, keep, votes0, y, runs_all, w = split_arrays("val")
    sel = np.flatnonzero(keep)
    z = np.load(knn_path(0, 0))
    assert (z["rows"] == sel).all()
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(len(sel), size=3000, replace=False))
    rows = sel[pos]
    hq = np.load(LAT / "val" / "h_k0_o0.npy", mmap_mode="r")
    q = represent(np.asarray(hq[rows]), "b_mean_max")
    scn = [scenario_of(event_of(ids[r])) for r in rows]
    stgnn = votes0[rows].argmax(1)
    eq = {}
    for name, excl in (("standard", {}), ("heldout", {"exclude_scenarios": scn})):
        sim, idx = store.search(q, 20, query_checkpoint_sha256=ckpt, **excl)
        lab = torch.from_numpy(store.labels[idx].astype(np.int64))
        vote = _vote(lab, torch.from_numpy(sim), 10).numpy()
        agree = (store.labels[idx[:, :10]] == stgnn[:, None]).mean(1)
        eq[name] = {"vote10_identical_to_torch": float((vote == z[f"{name}_vote_10"][pos]).mean()),
                    "agree10_max_abs_diff": float(np.abs(agree - z[f"{name}_agree_10"][pos]).max()),
                    "nn_sim_max_abs_diff": float(np.abs(sim[:, 0] - z[f"{name}_nnsim"][pos]).max()),
                    "own_scenario_in_top20": int((store.scenarios[idx] == np.array(scn)[:, None]).sum())}
    res["faiss_vs_torch_3000_val_queries"] = eq

    # sanity run (§ 13.9) with 2 % of val runs
    pick = set(stratified_run_subsample(split_events("val"), 0.02, seed=42))
    m = np.array([event_of(ids[r]) in pick for r in sel])
    qrows = sel[m]
    qv = represent(np.asarray(hq[qrows]), "b_mean_max")
    truth = y[qrows]
    post = w[qrows] >= 21
    q_runs = [event_of(ids[r]) for r in qrows]
    q_scn = [scenario_of(r) for r in q_runs]
    polluted = CaseStore.build(np.concatenate([store.vectors, qv]), [*store.window_ids, *[ids[r] for r in qrows]],
                               np.concatenate([store.labels, truth]), np.concatenate([store.pred, votes0[qrows].argmax(1)]),
                               checkpoint_sha256=ckpt, note="DELIBERATELY polluted with val queries (sanity run)")
    acc = {"clean_store_standard": _vote_acc(store, qv, truth, ckpt),
           "polluted_store_no_exclusion": _vote_acc(polluted, qv, truth, ckpt),
           "polluted_store_leave_run_out": _vote_acc(polluted, qv, truth, ckpt, exclude_runs=q_runs),
           "polluted_store_scenario_held_out": _vote_acc(polluted, qv, truth, ckpt, exclude_scenarios=q_scn),
           "clean_store_scenario_held_out": _vote_acc(store, qv, truth, ckpt, exclude_scenarios=q_scn)}
    acc1 = {"clean_store_standard": _vote_acc(store, qv, truth, ckpt, k=1), "polluted_store_no_exclusion": _vote_acc(polluted, qv, truth, ckpt, k=1),
            "polluted_store_leave_run_out": _vote_acc(polluted, qv, truth, ckpt, k=1, exclude_runs=q_runs)}
    res["sanity_polluted_store"] = {"n_runs": len(pick), "n_queries": int(len(qrows)), "n_post_fault": int(post.sum()),
                                    "k_nn_10": {n: {"all": float(a.mean()), "post_fault": float(a[post].mean())} for n, a in acc.items()},
                                    "k_nn_1": {n: {"all": float(a.mean()), "post_fault": float(a[post].mean())} for n, a in acc1.items()},
                                    "self_match_k1_is_exact": bool(acc1["polluted_store_no_exclusion"].all()),
                                    "k10_vote_above_0.99_with_queries_in_store (author-free criterion I set; not met)": bool(acc["polluted_store_no_exclusion"].mean() > 0.99),
                                    "note": "k_nn=10 also counts 9 other windows, some across the fault-onset label transition, so it stays below 1; k_nn=1 must be exactly 1."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_checks_seed42.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
