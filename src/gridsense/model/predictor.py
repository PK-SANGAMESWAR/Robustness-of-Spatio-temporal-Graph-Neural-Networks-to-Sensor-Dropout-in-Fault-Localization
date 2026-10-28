"""Frozen Paper-1 model behind a stable interface (IMPLEMENTATION.md § 3). Read-only use of the checkpoints.

Loading mirrors scripts/run_dropout_study.py:load_trained_model. The latent is captured with a forward pre-hook on
the node head, so the model code and its outputs are unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from gridsense.config import ROOT

RUNS = ROOT / "results" / "dropout_aware" / "rgsage_max"


def seed_dir(seed: int) -> Path:
    return RUNS / f"seed_{seed:05d}"


def checkpoint_sha256(seed: int) -> str:
    return hashlib.sha256((seed_dir(seed) / "checkpoint_best.pt").read_bytes()).hexdigest()


def load_model(seed: int, edge_index: np.ndarray, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    from fault_localization.experiments.runner import _model

    cfg = yaml.safe_load((seed_dir(seed) / "config.yaml").read_text())
    if cfg.get("ablation", {}).get("disable_edges", False):
        raise ValueError("edges-off ablation checkpoints are not supported")
    model = _model(cfg, edge_index).to(device)
    ckpt = torch.load(seed_dir(seed) / "checkpoint_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


class LatentCapture:
    """Captures the input of model.classifier: per-node embeddings h [B, 25, 64] (decision O4)."""

    def __init__(self, model: torch.nn.Module):
        self.h: torch.Tensor | None = None
        self._handle = model.classifier.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        self.h = inputs[0].detach()

    def remove(self) -> None:
        self._handle.remove()


def lean_dataset(split: str, data_root: Path, keep_events: set[str] | None = None):
    """WindowGraphDataset with a low-memory __init__ (pandas, split rows only). __getitem__ is the paper's own code.

    Records keep the split-file order, so the paper's EventChunkBatchSampler yields the same batches.
    """
    from fault_localization.data.dataset import WindowGraphDataset

    ds = WindowGraphDataset.__new__(WindowGraphDataset)
    ds.root = Path(data_root)
    ds.cache_size = 8
    ds.edge_index = np.load(ds.root / "processed/edge_index.npy").astype(np.int64)
    ds.node_mapping = json.loads((ds.root / "processed/node_mapping.json").read_text())
    ds.stats = json.loads((ds.root / "processed/norm_stats.json").read_text())
    ds.mean = np.asarray(ds.stats["mean"], dtype=np.float32)
    ds.std = np.asarray(ds.stats["std"], dtype=np.float32)
    ids = [x.strip() for x in (ds.root / "splits" / f"{split}.txt").read_text().splitlines() if x.strip()]
    if keep_events is not None:
        ids = [w for w in ids if w.rsplit("_w", 1)[0] in keep_events]
    wanted = set(ids)
    labels: dict[str, int] = {}
    with (ds.root / "processed/labels.csv").open() as f:
        for r in csv.DictReader(f):
            if r["window_id"] in wanted:
                labels[r["window_id"]] = int(r["label"])
    if len(labels) != len(wanted):
        raise ValueError(f"{split}: {len(wanted) - len(labels)} window IDs missing from labels.csv")
    ds.records = [(w, w.rsplit("_w", 1)[0], labels[w], {}) for w in ids]
    ds._cache = OrderedDict()
    return ds
