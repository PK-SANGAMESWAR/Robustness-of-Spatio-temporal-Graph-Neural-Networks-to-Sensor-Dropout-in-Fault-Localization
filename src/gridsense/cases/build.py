"""Build the case store from extracted train windows (O15, O24). `sub10` = the 10 % train-run subsample; `full` = every train window.

Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.cases.build [--seed 42] [--tag train]
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from gridsense.cases.store import CaseStore, fingerprint
from gridsense.config import TRACK
from gridsense.data.splits import DATA_ROOT, event_of, split_events
from gridsense.model.latent_repr import represent
from gridsense.model.predictor import checkpoint_sha256

BASE_PARTS = dict(latent_layer="model.classifier input h [25,64]", pooling="b_mean_max", split="train", stratify="scenario", k_dropout=0,
                  policy="zero_messages", index="faiss IndexFlatIP exact, L2-normalised")
TAGS = {"train_sub10": dict(run_fraction=0.10, run_subsample_seed=42, folder="sub10"), "train": dict(run_fraction=1.0, run_subsample_seed=None, folder="full")}


def lat_dir(seed: int):
    return TRACK / "eval" / "latents" / f"seed{seed}"


def store_dir(seed: int, tag: str = "train_sub10"):
    return TRACK / "eval" / "case_store" / f"seed{seed}_{TAGS[tag]['folder']}"


def fingerprint_parts(tag: str) -> dict:
    spec = TAGS[tag]
    return {**BASE_PARTS, "run_fraction": spec["run_fraction"], "run_subsample_seed": spec["run_subsample_seed"]}


def run_seeds() -> dict[str, int]:
    m = pd.read_csv(DATA_ROOT / "processed" / "metadata.csv", usecols=["window_id", "seed"]).drop_duplicates("seed")
    return {event_of(w): int(s) for w, s in zip(m["window_id"], m["seed"])}


def build_store(seed: int = 42, tag: str = "train_sub10") -> CaseStore:
    d = lat_dir(seed) / tag
    ids = (d / "window_ids.txt").read_text().split()
    labels = np.load(d / "labels.npy").astype(np.int64)
    pred = np.load(d / "votes_k0_o0.npy").argmax(1)
    h = np.load(d / "h_k0_o0.npy")
    vec = represent(h, "b_mean_max") if h.ndim == 3 else h
    seeds = run_seeds()
    forbidden = set(split_events("val")) | set(split_events("test"))
    runs = {event_of(w) for w in ids}
    return CaseStore.build(vec, ids, labels, pred, checkpoint_sha256=checkpoint_sha256(seed), forbidden_runs=forbidden,
                           store_seeds={r: seeds[r] for r in runs}, forbidden_seeds={seeds[r] for r in forbidden}, **fingerprint_parts(tag))


def expected_fingerprint(window_ids: list[str], seed: int = 42, tag: str = "train_sub10") -> str:
    return fingerprint(checkpoint_sha256=checkpoint_sha256(seed), window_ids_sha256=hashlib.sha256("\n".join(window_ids).encode()).hexdigest(),
                       dim=128, **fingerprint_parts(tag))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", choices=list(TAGS), default="train_sub10")
    a = ap.parse_args()
    s = build_store(a.seed, a.tag)
    s.save(store_dir(a.seed, a.tag))
    print(f"case store seed {a.seed} {a.tag}: {s.manifest['n']} entries, {len(set(s.runs))} runs, {len(set(s.scenarios))} scenarios, fingerprint {s.manifest['fingerprint'][:16]}")
