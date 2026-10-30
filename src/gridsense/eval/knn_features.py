"""Neighbour features against the FULL train store, both versions (O24 step 1; author 2026-10-30). CPU, exact search (torch).

Store: every train window at k = 0 (representation (b) mean + max, 128-d, L2-normalised) of the given training seed.
Queries: post-fault windows (w >= 21) of `val` or `test` at k in {0, 1, 2, 5, 10}, mask offset 0 (the gate's population; pre-fault windows are
class 0 and are not searched). Two versions from one similarity matrix: `standard` (all entries) and `heldout` (entries of the query's own
scenario removed, O20). Per version and k_nn in (5, 10, 20): vote, kNN agreement (neighbour label == STGNN top-1 of the query), neighbour
error rate (store prediction wrong), and the nearest similarity. Output: eval/layer0/knnfull_seed{seed}_{split}_k{k}_o0.npz (Git-ignored).
Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.knn_features --seed 42 --split val
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from gridsense.config import TRACK, torch_threads
from gridsense.data.splits import event_of, scenario_of, window_index
from gridsense.eval.scenario_control import _scenario_ids, _vote, topk_excluding_scenario  # noqa: F401  (same vote rule as Layer 0)
from gridsense.model.latent_repr import represent

KS = (0, 1, 2, 5, 10)
K_NN = (5, 10, 20)
OUT = TRACK / "eval" / "layer0"
CHUNK = 64


def lat_dir(seed: int) -> Path:
    return TRACK / "eval" / "latents" / f"seed{seed}"


def knn_path(seed: int, split: str, k: int) -> Path:
    return OUT / f"knnfull_seed{seed}_{split}_k{k}_o0.npz"


def post_fault_rows(window_ids: list[str]) -> np.ndarray:
    return np.flatnonzero(np.array([window_index(w) for w in window_ids]) >= 21)


def _rep(block: np.ndarray) -> np.ndarray:
    return represent(block, "b_mean_max") if block.ndim == 3 else np.ascontiguousarray(block, dtype=np.float32)


def load_store(seed: int, tag: str = "train"):
    d = lat_dir(seed) / tag
    ids = (d / "window_ids.txt").read_text().split()
    y = torch.from_numpy(np.load(d / "labels.npy").astype(np.int64))
    err = torch.from_numpy((np.load(d / "votes_k0_o0.npy").argmax(1) != y.numpy()).astype(np.float32))
    h = np.load(d / "h_k0_o0.npy")
    vecs = torch.nn.functional.normalize(torch.from_numpy(_rep(h)), dim=1)
    return ids, vecs, y, err


def run(seed: int, split: str, ks=KS, store_tag: str = "train") -> None:
    torch.set_num_threads(torch_threads())
    OUT.mkdir(parents=True, exist_ok=True)
    d = lat_dir(seed) / split
    q_ids = (d / "window_ids.txt").read_text().split()
    rows = post_fault_rows(q_ids)
    todo = [k for k in ks if not knn_path(seed, split, k).exists()]
    if not todo:
        print("all files exist", flush=True)
        return
    s_ids, store_n, store_y, store_err = load_store(seed, store_tag)
    vocab = {s: i for i, s in enumerate(sorted({scenario_of(event_of(i)) for i in s_ids} | {scenario_of(event_of(q_ids[r])) for r in rows}))}
    store_scn = torch.from_numpy(_scenario_ids(s_ids, vocab))
    q_scn = _scenario_ids([q_ids[r] for r in rows], vocab)
    print(f"seed {seed} {split}: store {len(s_ids)} windows, queries {len(rows)} post-fault windows, k {todo}", flush=True)
    for k in todo:
        t0 = time.time()
        stgnn = np.load(d / f"votes_k{k}_o0.npy")[rows].argmax(1)
        h = np.load(d / f"h_k{k}_o0.npy", mmap_mode="r")
        n = len(rows)
        out: dict = {"rows": rows.astype(np.int32)}
        for s in ("standard", "heldout"):
            for kk in K_NN:
                out[f"{s}_vote_{kk}"] = np.empty(n, dtype=np.int8)
                out[f"{s}_agree_{kk}"] = np.empty(n, dtype=np.float32)
                out[f"{s}_nbrerr_{kk}"] = np.empty(n, dtype=np.float32)
            out[f"{s}_nnsim"] = np.empty(n, dtype=np.float32)
        for i in range(0, n, CHUNK):
            sl = slice(i, i + CHUNK)
            qq = torch.nn.functional.normalize(torch.from_numpy(_rep(np.asarray(h[rows[sl]]))), dim=1)
            pred = torch.from_numpy(stgnn[sl].astype(np.int64))
            sim = qq @ store_n.T
            res = {"standard": sim.topk(max(K_NN), dim=1)}
            sim.masked_fill_(store_scn[None, :] == torch.from_numpy(q_scn[sl])[:, None], float("-inf"))   # in place: saves a 300 MB copy
            res["heldout"] = sim.topk(max(K_NN), dim=1)
            for s, (sc, ix) in res.items():
                lab = store_y[ix]
                out[f"{s}_nnsim"][sl] = sc[:, 0].numpy()
                for kk in K_NN:
                    out[f"{s}_vote_{kk}"][sl] = _vote(lab, sc, kk).numpy()
                    out[f"{s}_agree_{kk}"][sl] = (lab[:, :kk] == pred[:, None]).float().mean(1).numpy()
                    out[f"{s}_nbrerr_{kk}"][sl] = store_err[ix[:, :kk]].mean(1).numpy()
            if (i // CHUNK) % 400 == 0:
                print(f"  k={k} {i + len(pred)}/{n} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(knn_path(seed, split, k), **out)
        print(f"k={k} done in {time.time() - t0:.0f}s", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--k", type=int, nargs="*", default=list(KS))
    a = ap.parse_args()
    run(a.seed, a.split, tuple(a.k))
