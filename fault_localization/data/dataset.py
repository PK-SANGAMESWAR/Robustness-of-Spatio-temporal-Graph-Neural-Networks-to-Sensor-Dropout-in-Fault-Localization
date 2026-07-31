"""Lazy, split-aware access to processed window files."""
from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # enables manifest/validation use before training dependencies are installed
    torch = None
    Dataset = object
    DataLoader = None


class WindowGraphDataset(Dataset):
    """Loads only requested events, retaining at most ``cache_size`` NPZ files."""
    def __init__(self, data_root: str | Path, split: str, cache_size: int = 8):
        self.root = Path(data_root)
        self.cache_size = cache_size
        self.edge_index = np.load(self.root / "processed/edge_index.npy").astype(np.int64)
        self.node_mapping = json.loads((self.root / "processed/node_mapping.json").read_text())
        self.stats = json.loads((self.root / "processed/norm_stats.json").read_text())
        self.mean = np.asarray(self.stats["mean"], dtype=np.float32)
        self.std = np.asarray(self.stats["std"], dtype=np.float32)
        if len(self.node_mapping) != 25 or self.edge_index.shape[0] != 2 or self.edge_index.min() < 0 or self.edge_index.max() >= 25:
            raise ValueError("Invalid measured-only graph or node mapping")
        labels = {r["window_id"]: r for r in csv.DictReader((self.root / "processed/labels.csv").open())}
        metadata = {r["window_id"]: r for r in csv.DictReader((self.root / "processed/metadata.csv").open())}
        ids = [x.strip() for x in (self.root / "splits" / f"{split}.txt").read_text().splitlines() if x.strip()]
        if len(ids) != len(set(ids)) or any(x not in labels or x not in metadata for x in ids):
            raise ValueError(f"Invalid {split} split IDs")
        self.records = [(wid, wid.rsplit("_w", 1)[0], int(labels[wid]["label"]), metadata[wid]) for wid in ids]
        self._cache: OrderedDict[str, tuple[np.ndarray, dict[str, int]]] = OrderedDict()

    def __len__(self) -> int: return len(self.records)

    def _event(self, event_id: str) -> tuple[np.ndarray, dict[str, int]]:
        if event_id in self._cache:
            self._cache.move_to_end(event_id); return self._cache[event_id]
        path = self.root / "processed/windows" / f"{event_id}.npz"
        with np.load(path) as loaded:
            windows = loaded["windows"]
            index = {str(w): i for i, w in enumerate(loaded["window_ids"])}
        self._cache[event_id] = (windows, index)
        if len(self._cache) > self.cache_size: self._cache.popitem(last=False)
        return self._cache[event_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        wid, event, label, metadata = self.records[index]
        windows, positions = self._event(event)
        if wid not in positions: raise KeyError(f"{wid} absent from {event}")
        w = windows[positions[wid]].astype(np.float32)
        # Normalize voltage channels (0:3) only; mask channels (3:6) stay binary.
        w[..., :3] = (w[..., :3] - self.mean) / self.std
        return {"x": w, "y": np.int64(label), "window_id": wid, "metadata": metadata}


class EventChunkBatchSampler:
    """Shuffle events while retaining a small, cache-resident event working set.

    Each split member occurs exactly once in an epoch.  The sampler only
    changes delivery order: it takes four windows from each of eight events
    per batch (32 samples with the benchmark default), so the dataset's
    eight-event cache loads each compressed event file once per epoch.
    """

    def __init__(self, dataset: WindowGraphDataset, batch_size: int, shuffle: bool, seed: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        by_event: OrderedDict[str, list[int]] = OrderedDict()
        for index, (_, event_id, _, _) in enumerate(dataset.records):
            by_event.setdefault(event_id, []).append(index)
        self.event_indices = list(by_event.values())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.active_events = min(8, batch_size)

    def __iter__(self):
        order = list(range(len(self.event_indices)))
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(order)

        for chunk_start in range(0, len(order), self.active_events):
            pending = [list(self.event_indices[i]) for i in order[chunk_start:chunk_start + self.active_events]]
            if self.shuffle:
                # Each event is stored in chronological window order: the
                # first half is healthy and the second half is fault-active.
                # Taking consecutive entries therefore produced homogeneous
                # mini-batches (several all-healthy batches, then several
                # fault-only ones).  Shuffle positions within an event as
                # well as the event order so every training batch represents
                # both sides of the classification task.  Evaluation remains
                # chronological and deterministic because shuffle=False.
                for indices in pending:
                    midpoint = len(indices) // 2
                    healthy, fault_active = indices[:midpoint], indices[midpoint:]
                    rng.shuffle(healthy)
                    rng.shuffle(fault_active)
                    # Alternate the two chronological halves.  With the
                    # default four samples per event, every full batch
                    # therefore receives two healthy and two fault-active
                    # examples from each active event.
                    indices[:] = [item for pair in zip(healthy, fault_active) for item in pair]
                    indices.extend(healthy[len(fault_active):])
                    indices.extend(fault_active[len(healthy):])
            while any(pending):
                batch: list[int] = []
                active = sum(bool(indices) for indices in pending)
                take_per_event = max(1, self.batch_size // active)
                for indices in pending:
                    batch.extend(indices[:take_per_event])
                    del indices[:take_per_event]
                if batch:
                    yield batch

    def __len__(self) -> int:
        batches = 0
        for chunk_start in range(0, len(self.event_indices), self.active_events):
            remaining = [len(indices) for indices in self.event_indices[chunk_start:chunk_start + self.active_events]]
            while any(remaining):
                active = sum(count > 0 for count in remaining)
                take_per_event = max(1, self.batch_size // active)
                remaining = [max(0, count - take_per_event) for count in remaining]
                batches += 1
        return batches


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if torch is None: raise ImportError("Install torch to construct training batches")
    return {"x": torch.as_tensor(np.stack([x["x"] for x in items])),
            "y": torch.as_tensor([x["y"] for x in items], dtype=torch.long),
            "window_id": [x["window_id"] for x in items], "metadata": [x["metadata"] for x in items]}


def make_dataloader(dataset: WindowGraphDataset, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0):
    if DataLoader is None: raise ImportError("Install torch to construct training loaders")
    return DataLoader(
        dataset,
        batch_sampler=EventChunkBatchSampler(dataset, batch_size, shuffle, seed),
        collate_fn=_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
