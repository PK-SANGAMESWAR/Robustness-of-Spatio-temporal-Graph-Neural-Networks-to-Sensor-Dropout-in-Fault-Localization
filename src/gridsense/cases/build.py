"""Build the case store from the extracted train subsample (O15: 10 % of train runs; a full store needs the author's approval).

Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.cases.build
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gridsense.cases.store import CaseStore
from gridsense.config import TRACK
from gridsense.data.splits import DATA_ROOT, event_of, split_events
from gridsense.model.latent_repr import represent
from gridsense.model.predictor import checkpoint_sha256

LAT = TRACK / "eval" / "latents" / "seed42"
STORE_DIR = TRACK / "eval" / "case_store" / "seed42_sub10"
FINGERPRINT_PARTS = dict(latent_layer="model.classifier input h [25,64]", pooling="b_mean_max", split="train", run_fraction=0.10,
                         run_subsample_seed=42, stratify="scenario", k_dropout=0, policy="zero_messages", index="faiss IndexFlatIP exact, L2-normalised")


def run_seeds() -> dict[str, int]:
    m = pd.read_csv(DATA_ROOT / "processed" / "metadata.csv", usecols=["window_id", "seed"]).drop_duplicates("seed")
    return {event_of(w): int(s) for w, s in zip(m["window_id"], m["seed"])}


def build_store() -> CaseStore:
    ids = (LAT / "train_sub10" / "window_ids.txt").read_text().split()
    labels = np.load(LAT / "train_sub10" / "labels.npy").astype(np.int64)
    pred = np.load(LAT / "train_sub10" / "votes_k0_o0.npy").argmax(1)
    vec = represent(np.load(LAT / "train_sub10" / "h_k0_o0.npy"), "b_mean_max")
    seeds = run_seeds()
    forbidden = set(split_events("val")) | set(split_events("test"))
    return CaseStore.build(vec, ids, labels, pred, checkpoint_sha256=checkpoint_sha256(42), forbidden_runs=forbidden,
                           store_seeds={r: seeds[r] for r in {event_of(w) for w in ids}}, forbidden_seeds={seeds[r] for r in forbidden}, **FINGERPRINT_PARTS)


def expected_fingerprint(window_ids: list[str]) -> str:
    from gridsense.cases.store import fingerprint
    import hashlib
    return fingerprint(checkpoint_sha256=checkpoint_sha256(42), window_ids_sha256=hashlib.sha256("\n".join(window_ids).encode()).hexdigest(),
                       dim=128, **FINGERPRINT_PARTS)


if __name__ == "__main__":
    s = build_store()
    s.save(STORE_DIR)
    print(f"case store: {s.manifest['n']} entries, {len(set(s.runs))} runs, {len(set(s.scenarios))} scenarios, fingerprint {s.manifest['fingerprint'][:16]}")
