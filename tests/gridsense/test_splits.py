"""Run-level split helpers (O12, O15): whole runs, stratified by scenario, deterministic."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridsense.data.splits import cal_split, event_of, scenario_of, stratified_run_subsample, window_index  # noqa: E402

EVENTS = [f"{bus}_{ft}_{r}_{run:04d}" for bus in (7, 13, 18) for ft in ("AG", "ABCG") for r in ("0p1", "10p0")
          for run in range(35 if bus != 18 else 7)]


def test_id_parsing() -> None:
    assert event_of("7_ABCG_10p0_0014_w0021") == "7_ABCG_10p0_0014"
    assert window_index("7_ABCG_10p0_0014_w0021") == 21
    assert scenario_of("7_ABCG_10p0_0014") == "7_ABCG_10p0"


def test_subsample_whole_runs_stratified_and_deterministic() -> None:
    sub = stratified_run_subsample(EVENTS, 0.10, seed=42)
    assert sub == stratified_run_subsample(EVENTS, 0.10, seed=42)
    assert set(sub) <= set(EVENTS) and len(sub) == len(set(sub))
    assert len(sub) == round(0.10 * len(EVENTS))
    per = Counter(scenario_of(e) for e in sub)
    for s, n in Counter(scenario_of(e) for e in EVENTS).items():
        assert int(0.10 * n) <= per.get(s, 0) <= int(0.10 * n) + 1, s


def test_cal_split_disjoint_balanced_per_scenario() -> None:
    a, b = cal_split(EVENTS, seed=42)
    assert set(a).isdisjoint(b) and set(a) | set(b) == set(EVENTS)
    assert abs(len(a) - len(b)) <= 1
    ca, cb = Counter(map(scenario_of, a)), Counter(map(scenario_of, b))
    for s in Counter(map(scenario_of, EVENTS)):
        assert abs(ca[s] - cb[s]) <= 1, s
    assert (a, b) == cal_split(EVENTS, seed=42)
