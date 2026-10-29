"""Scenario-held-out control (O19 step 1, author 2026-10-29). Informational; `val` queries, k = 0, seed 42.

For each query window, all store entries of the SAME scenario (fault bus, fault type, resistance; 825 scenarios) are removed
before the kNN search. Compared with the standard store (no removal) and with the STGNN top-1 accuracy on the same windows.
Store: the 10 % train-run subsample (O15), representation (b) mean + max (O4). Only the store is held out: the encoder was
trained on these scenarios, so this is NOT a scenario-held-out model (the `scenario_heldout_*` checkpoints would be).
kNN agreement = fraction of the k_nn neighbours whose label equals the STGNN top-1 class of the query window.
Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.scenario_control
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from gridsense.config import TRACK
from gridsense.data.splits import event_of, scenario_of
from gridsense.eval.s2_k0 import K_NN, LAT, boot_ci, paired_boot_diff, split_arrays
from gridsense.model.latent_repr import represent

CHUNK = 512
OUT = TRACK / "reports" / "layer0"


def _scenario_ids(window_ids: list[str], vocab: dict[str, int]) -> np.ndarray:
    return np.array([vocab[scenario_of(event_of(w))] for w in window_ids], dtype=np.int64)


def topk_excluding_scenario(sim: torch.Tensor, store_scn: torch.Tensor, q_scn: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k store entries per query after removing every store entry of the query's own scenario. Returns (similarity, index)."""
    return sim.masked_fill(store_scn[None, :] == q_scn[:, None], float("-inf")).topk(k, dim=1)


def _vote(lab: torch.Tensor, sim: torch.Tensor, k: int) -> torch.Tensor:
    counts = torch.zeros(len(lab), 26).scatter_add_(1, lab[:, :k], torch.ones_like(sim[:, :k]))
    counts += 1e-6 * torch.zeros(len(lab), 26).scatter_add_(1, lab[:, :k], sim[:, :k])  # tie -> larger similarity sum
    return counts.argmax(1)


def run() -> dict:
    torch.set_num_threads(4)
    ids, keep, votes, y, runs, w = split_arrays("val")
    stgnn = votes.argmax(1)
    store_ids = (LAT / "train_sub10" / "window_ids.txt").read_text().split()
    vocab = {s: i for i, s in enumerate(sorted({scenario_of(event_of(i)) for i in [*store_ids, *ids]}))}
    store_scn = torch.from_numpy(_scenario_ids(store_ids, vocab))
    store_y = torch.from_numpy(np.load(LAT / "train_sub10" / "labels.npy").astype(np.int64))
    store_h = np.load(LAT / "train_sub10" / "h_k0_o0.npy")
    store_n = torch.nn.functional.normalize(torch.from_numpy(represent(store_h, "b_mean_max")), dim=1)
    del store_h
    q_scn_all = _scenario_ids(ids, vocab)
    sel = np.flatnonzero(keep)
    q_h = np.load(LAT / "val" / "h_k0_o0.npy", mmap_mode="r")
    n = len(sel)
    res = {s: {"vote": {k: np.empty(n, dtype=np.int64) for k in K_NN}, "agree": {k: np.empty(n, dtype=np.float32) for k in K_NN}}
           for s in ("standard", "heldout")}
    same_scn_share = np.empty(n, dtype=np.float32)       # share of the standard top-10 that is from the query's own scenario
    scenarios_in_store = len(set(store_scn.tolist()))
    t0 = time.time()
    for i in range(0, n, CHUNK):
        rows = sel[i:i + CHUNK]
        qq = torch.nn.functional.normalize(torch.from_numpy(represent(np.asarray(q_h[rows]), "b_mean_max")), dim=1)
        q_scn = torch.from_numpy(q_scn_all[rows])
        sim = qq @ store_n.T
        pred = torch.from_numpy(stgnn[rows])
        s_std, i_std = sim.topk(max(K_NN), dim=1)
        same = store_scn[i_std[:, :10]] == q_scn[:, None]
        same_scn_share[i:i + len(rows)] = same.float().mean(1).numpy()
        s_ho, i_ho = topk_excluding_scenario(sim, store_scn, q_scn, max(K_NN))
        for name, (s_, i_) in (("standard", (s_std, i_std)), ("heldout", (s_ho, i_ho))):
            lab = store_y[i_]
            for k in K_NN:
                res[name]["vote"][k][i:i + len(rows)] = _vote(lab, s_, k).numpy()
                res[name]["agree"][k][i:i + len(rows)] = (lab[:, :k] == pred[:, None]).float().mean(1).numpy()
        if (i // CHUNK) % 100 == 0:
            print(f"  {i + len(rows)}/{n} {time.time() - t0:.0f}s", flush=True)
    truth, r, post = y[sel], runs[sel], w[sel] >= 21
    stgnn_ok = stgnn[sel] == truth
    out: dict = {"queries": int(n), "post_fault_queries": int(post.sum()), "store_windows": int(len(store_y)),
                 "scenarios_in_store": scenarios_in_store, "scenarios_total": len(vocab),
                 "stgnn_top1": {"all": float(stgnn_ok.mean()), "post_fault": float(stgnn_ok[post].mean())},
                 "standard_top10_same_scenario_share": {"all": float(same_scn_share.mean()), "post_fault": float(same_scn_share[post].mean())},
                 "knn_vote": {}, "knn_agreement": {}, "paired": {}}
    ok = {s: {k: res[s]["vote"][k] == truth for k in K_NN} for s in res}
    for s in res:
        out["knn_vote"][s] = {f"k{k}": {"all": float(ok[s][k].mean()), "post_fault": float(ok[s][k][post].mean()),
                                        "post_fault_ci": boot_ci(ok[s][k][post], r[post])} for k in K_NN}
        out["knn_agreement"][s] = {f"k{k}": {
            "post_fault_mean": float(res[s]["agree"][k][post].mean()),
            "post_fault_when_stgnn_correct": float(res[s]["agree"][k][post & stgnn_ok].mean()),
            "post_fault_when_stgnn_wrong": float(res[s]["agree"][k][post & ~stgnn_ok].mean()),
            "all_mean": float(res[s]["agree"][k].mean())} for k in K_NN}
    for k in K_NN:
        out["paired"][f"k{k}"] = {
            "heldout_minus_standard_post_fault": paired_boot_diff(ok["heldout"][k][post], ok["standard"][k][post], r[post]),
            "standard_minus_stgnn_post_fault": paired_boot_diff(ok["standard"][k][post], stgnn_ok[post], r[post]),
            "heldout_minus_stgnn_post_fault": paired_boot_diff(ok["heldout"][k][post], stgnn_ok[post], r[post]),
            "heldout_minus_stgnn_all": paired_boot_diff(ok["heldout"][k], stgnn_ok, r)}
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT / "scenario_control_k0_val.json").write_text(json.dumps(result, indent=2) + "\n")
    print("written")
