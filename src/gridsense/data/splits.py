"""Run-level split helpers (author decisions O12, O15; claude-rag.md § 3).

The unit of the paper is the window, but every split below is made of whole runs (events): all 40 windows of a
run stay together. A scenario is (fault bus, fault type, fault resistance); a run is one load draw of a scenario.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gridsense.config import ROOT

DATA_ROOT = ROOT / "dataset_fullscale_50"


def event_of(window_id: str) -> str:
    return window_id.rsplit("_w", 1)[0]


def window_index(window_id: str) -> int:
    return int(window_id.rsplit("_w", 1)[1])


def scenario_of(event_id: str) -> str:
    """'7_ABCG_10p0_0014' -> '7_ABCG_10p0' (bus, fault type, resistance)."""
    return event_id.rsplit("_", 1)[0]


def split_events(split: str, data_root: Path = DATA_ROOT) -> list[str]:
    """Events of a split, in first-appearance order of the split file."""
    ids = (data_root / "splits" / f"{split}.txt").read_text().split()
    return list(dict.fromkeys(event_of(w) for w in ids))


def _by_scenario(events: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for e in events:
        groups[scenario_of(e)].append(e)
    return {s: sorted(v) for s, v in sorted(groups.items())}


def stratified_run_subsample(events: list[str], fraction: float, seed: int) -> list[str]:
    """Whole runs, stratified by scenario; the total is round(fraction * n) (largest remainder, seeded ties) (O15)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    groups = _by_scenario(events)
    names = list(groups)
    quota = np.array([fraction * len(groups[s]) for s in names])
    take = np.floor(quota).astype(int)
    extra = int(round(fraction * len(events))) - int(take.sum())
    if extra > 0:
        rem = quota - take
        order = np.lexsort((rng.random(len(names)), -rem))  # largest remainder first, random tie-break
        take[order[:extra]] += 1
    chosen: list[str] = []
    for s, n in zip(names, take):
        runs = groups[s]
        chosen.extend(runs[i] for i in sorted(rng.choice(len(runs), size=n, replace=False)))
    return sorted(chosen)


def cal_split(events: list[str], seed: int) -> tuple[list[str], list[str]]:
    """50/50 by run, stratified by scenario (O12). Odd scenarios alternate the extra run between the two halves."""
    rng = np.random.default_rng(seed)
    a: list[str] = []
    b: list[str] = []
    give_a = True
    for runs in _by_scenario(events).values():
        perm = [runs[i] for i in rng.permutation(len(runs))]
        half = len(perm) // 2
        if len(perm) % 2:
            half += int(give_a)
            give_a = not give_a
        a.extend(perm[:half])
        b.extend(perm[half:])
    return sorted(a), sorted(b)


def load_window_table(windows: list[str], data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Labels and metadata for the given window IDs, in the given order."""
    lab = pd.read_csv(data_root / "processed/labels.csv", usecols=["window_id", "label"])
    meta = pd.read_csv(data_root / "processed/metadata.csv")
    df = pd.DataFrame({"window_id": windows}).merge(lab, on="window_id", how="left").merge(meta, on="window_id", how="left")
    if df["label"].isna().any():
        raise ValueError("window IDs missing from labels.csv")
    df["event_id"] = df["window_id"].map(event_of)
    df["w"] = df["window_id"].map(window_index)
    return df
