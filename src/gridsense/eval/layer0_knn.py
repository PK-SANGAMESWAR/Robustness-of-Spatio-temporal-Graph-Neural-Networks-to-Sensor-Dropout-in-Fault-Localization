"""Layer 0, neighbour features for every (k, mask offset) of `val`, both store versions (O19 step 2, author 2026-10-30).

Store: the 10 % train-run subsample at k = 0, representation (b) mean + max (O4, O15). Two searches per query window:
  standard : all store entries;   heldout : entries of the query's own scenario removed (O20: the held-out store is the primary one).
Per store and k_nn in (5, 10, 20): vote prediction, kNN agreement (fraction of neighbours whose label equals the STGNN top-1 class
of the query window at that (k, offset)), neighbour error rate (fraction of neighbours whose STGNN prediction at k = 0 was wrong),
plus the cosine similarity of the nearest neighbour. Windows: `val` without w0020. Exact torch search (same code path as the control).
Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.layer0_knn
"""
from __future__ import annotations

import time

import numpy as np
import torch

from gridsense.config import TRACK
from gridsense.data.splits import event_of, scenario_of
from gridsense.eval.s2_k0 import K_NN, LAT, split_arrays
from gridsense.eval.scenario_control import _scenario_ids, _vote, topk_excluding_scenario
from gridsense.model.latent_repr import represent

OUT = TRACK / "eval" / "layer0"
COMBOS = [(0, 0)] + [(k, o) for k in (1, 2, 5, 10) for o in (0, 1, 2)]
CHUNK = 512


def knn_path(k: int, o: int):
    return OUT / f"knn_val_k{k}_o{o}.npz"


def main() -> None:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    ids, keep, _, y, _, _ = split_arrays("val")
    sel = np.flatnonzero(keep)
    store_ids = (LAT / "train_sub10" / "window_ids.txt").read_text().split()
    vocab = {s: i for i, s in enumerate(sorted({scenario_of(event_of(i)) for i in [*store_ids, *ids]}))}
    store_scn = torch.from_numpy(_scenario_ids(store_ids, vocab))
    store_y = torch.from_numpy(np.load(LAT / "train_sub10" / "labels.npy").astype(np.int64))
    store_err = torch.from_numpy((np.load(LAT / "train_sub10" / "votes_k0_o0.npy").argmax(1) != store_y.numpy()).astype(np.float32))
    store_n = torch.nn.functional.normalize(torch.from_numpy(represent(np.load(LAT / "train_sub10" / "h_k0_o0.npy"), "b_mean_max")), dim=1)
    q_scn_all = _scenario_ids(ids, vocab)
    print(f"store {len(store_y)} windows, STGNN train error on the store {store_err.mean():.4f}; queries {len(sel)}", flush=True)
    for (k, o) in COMBOS:
        path = knn_path(k, o)
        if path.exists():
            continue
        t0 = time.time()
        stgnn = np.load(LAT / "val" / f"votes_k{k}_o{o}.npy").argmax(1)
        h = np.load(LAT / "val" / f"h_k{k}_o{o}.npy", mmap_mode="r")
        n = len(sel)
        out = {}
        for s in ("standard", "heldout"):
            for kk in K_NN:
                out[f"{s}_vote_{kk}"] = np.empty(n, dtype=np.int8)
                out[f"{s}_agree_{kk}"] = np.empty(n, dtype=np.float32)
                out[f"{s}_nbrerr_{kk}"] = np.empty(n, dtype=np.float32)
            out[f"{s}_nnsim"] = np.empty(n, dtype=np.float32)
        for i in range(0, n, CHUNK):
            rows = sel[i:i + CHUNK]
            block = np.asarray(h[rows])
            qq = torch.nn.functional.normalize(torch.from_numpy(represent(block, "b_mean_max") if block.ndim == 3 else np.ascontiguousarray(block, dtype=np.float32)), dim=1)
            q_scn = torch.from_numpy(q_scn_all[rows])
            pred = torch.from_numpy(stgnn[rows].astype(np.int64))
            sim = qq @ store_n.T
            res = {"standard": sim.topk(max(K_NN), dim=1), "heldout": topk_excluding_scenario(sim, store_scn, q_scn, max(K_NN))}
            for s, (sc, ix) in res.items():
                lab = store_y[ix]
                out[f"{s}_nnsim"][i:i + len(rows)] = sc[:, 0].numpy()
                for kk in K_NN:
                    out[f"{s}_vote_{kk}"][i:i + len(rows)] = _vote(lab, sc, kk).numpy()
                    out[f"{s}_agree_{kk}"][i:i + len(rows)] = (lab[:, :kk] == pred[:, None]).float().mean(1).numpy()
                    out[f"{s}_nbrerr_{kk}"][i:i + len(rows)] = store_err[ix[:, :kk]].mean(1).numpy()
        np.savez_compressed(path, rows=sel.astype(np.int32), **out)
        print(f"k={k} offset={o} done in {time.time() - t0:.0f}s", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
