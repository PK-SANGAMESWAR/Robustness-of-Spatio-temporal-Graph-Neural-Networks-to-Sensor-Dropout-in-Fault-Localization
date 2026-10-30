"""Leakage rules on the real data and the saved case store (slow). Needs eval/case_store/seed42_sub10 and eval/latents."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
pytest.importorskip("faiss", reason="pending: rag dependency group")
STORE = ROOT / "rag-for-dropout" / "eval" / "case_store" / "seed42_sub10"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (STORE / "store_manifest.json").exists(), reason="pending: case store (python -m gridsense.cases.build)")]


def _store():
    import json

    from gridsense.cases.store import CaseStore
    return CaseStore.load(STORE, json.loads((STORE / "store_manifest.json").read_text())["fingerprint"])


def test_store_has_no_val_or_test_run_and_no_shared_seed() -> None:
    from gridsense.cases.build import run_seeds
    from gridsense.data.splits import split_events
    s = _store()
    forbidden = set(split_events("val")) | set(split_events("test"))
    assert not set(s.runs) & forbidden
    seeds = run_seeds()
    assert not {seeds[r] for r in s.runs} & {seeds[r] for r in forbidden}
    assert set(s.runs) <= set(split_events("train"))


def test_saved_store_fingerprint_is_the_expected_one() -> None:
    from gridsense.cases.build import expected_fingerprint
    s = _store()
    assert s.manifest["fingerprint"] == expected_fingerprint(s.window_ids)


def test_heldout_search_never_returns_own_scenario_on_real_data() -> None:
    from gridsense.data.splits import event_of, scenario_of
    from gridsense.model.latent_repr import represent
    from gridsense.model.predictor import checkpoint_sha256
    lat = ROOT / "rag-for-dropout" / "eval" / "latents" / "seed42" / "val"
    ids = (lat / "window_ids.txt").read_text().split()
    rows = np.arange(100, 241_000, 2411)[:100]
    q = represent(np.asarray(np.load(lat / "h_k0_o0.npy", mmap_mode="r")[rows]), "b_mean_max")
    scn = [scenario_of(event_of(ids[r])) for r in rows]
    s = _store()
    _, idx = s.search(q, 10, query_checkpoint_sha256=checkpoint_sha256(42), exclude_scenarios=scn)
    assert not (s.scenarios[idx] == np.array(scn)[:, None]).any()
    _, idx_std = s.search(q, 10, query_checkpoint_sha256=checkpoint_sha256(42))
    assert (s.scenarios[idx_std] == np.array(scn)[:, None]).any()          # the standard store does contain same-scenario neighbours
