"""Scenario-held-out control (O19 step 1): a query never sees a store entry of its own scenario. Synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gridsense.eval.scenario_control import topk_excluding_scenario  # noqa: E402


def test_heldout_topk_never_returns_same_scenario() -> None:
    g = torch.Generator().manual_seed(0)
    store_n = torch.nn.functional.normalize(torch.randn(500, 8, generator=g), dim=1)
    store_scn = torch.randint(0, 10, (500,), generator=g)
    q = torch.nn.functional.normalize(store_n[:40] + 0.01 * torch.randn(40, 8, generator=g), dim=1)  # near-duplicates of store rows
    q_scn = store_scn[:40].clone()
    sim = q @ store_n.T
    _, idx_std = sim.topk(5, dim=1)
    _, idx_ho = topk_excluding_scenario(sim, store_scn, q_scn, 5)
    assert (store_scn[idx_std[:, 0]] == q_scn).all()          # standard store: nearest neighbour is its own near-duplicate
    assert not (store_scn[idx_ho] == q_scn[:, None]).any()    # held-out: none of the 5 neighbours is from the query's scenario
    assert torch.isfinite(topk_excluding_scenario(sim, store_scn, q_scn, 5)[0]).all()


def test_input_similarity_not_modified() -> None:
    sim = torch.rand(4, 20)
    before = sim.clone()
    topk_excluding_scenario(sim, torch.zeros(20, dtype=torch.long), torch.zeros(4, dtype=torch.long), 3)
    assert torch.equal(sim, before)
