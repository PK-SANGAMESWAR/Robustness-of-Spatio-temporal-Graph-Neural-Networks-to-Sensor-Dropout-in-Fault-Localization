"""Leakage and fingerprint rules of the case store (implementation-plan.md § 13.9, § 14.2; S2 author decisions). Synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
pytest.importorskip("faiss", reason="pending: rag dependency group (uv sync --group rag --inexact)")

from gridsense.cases.store import CaseStore, LeakageError, StaleIndexError, fingerprint  # noqa: E402

SCN = ["7_AG_0p1", "7_AG_1p0", "13_ABG_10p0"]


def _ids(n_runs_per_scn: int = 4, n_win: int = 5) -> list[str]:
    return [f"{s}_{r:04d}_w{w:04d}" for s in SCN for r in range(n_runs_per_scn) for w in range(n_win)]


def _store(ids=None, ckpt="abc", **kw) -> CaseStore:
    ids = ids or _ids()
    rng = np.random.default_rng(0)
    return CaseStore.build(rng.normal(size=(len(ids), 8)), ids, rng.integers(0, 3, len(ids)), rng.integers(0, 3, len(ids)),
                           checkpoint_sha256=ckpt, **kw)


def test_forbidden_run_in_store_raises() -> None:
    ids = _ids()
    with pytest.raises(LeakageError):
        _store(ids, forbidden_runs={"13_ABG_10p0_0002"})                 # a test/val run is in the store
    _store(ids, forbidden_runs={"99_AG_0p1_0000"})                       # no clash: fine


def test_forbidden_simulation_seed_in_store_raises() -> None:
    with pytest.raises(LeakageError):
        _store(store_seeds={r: i for i, r in enumerate(sorted({i.rsplit("_w", 1)[0] for i in _ids()}))}, forbidden_seeds={3})  # a store run has seed 3
    _store(store_seeds={r: 1000 + i for i, r in enumerate(sorted({i.rsplit("_w", 1)[0] for i in _ids()}))}, forbidden_seeds={3})


def test_search_excludes_own_scenario_and_own_run() -> None:
    s = _store()
    q = s.vectors[:6].copy()
    own_scn = [SCN[0]] * 6
    own_run = [s.runs[i] for i in range(6)]
    _, idx_std = s.search(q, 3, query_checkpoint_sha256="abc")
    assert (idx_std[:, 0] == np.arange(6)).all()                         # standard store: a stored window finds itself
    _, idx_scn = s.search(q, 3, query_checkpoint_sha256="abc", exclude_scenarios=own_scn)
    assert not np.isin(s.scenarios[idx_scn], [SCN[0]]).any()
    _, idx_run = s.search(q, 3, query_checkpoint_sha256="abc", exclude_runs=own_run)
    assert not (s.runs[idx_run] == np.array(own_run)[:, None]).any()     # leave-one-out: never its own run
    assert (idx_run != np.arange(6)[:, None]).all()


def test_fingerprint_changes_with_each_part_and_stale_store_is_refused(tmp_path: Path) -> None:
    base = dict(latent_layer="h", pooling="b_mean_max", split="train", fraction=0.1, seed=42)
    s = _store(**base)
    fp = s.manifest["fingerprint"]
    for change in ({"pooling": "a_concat"}, {"fraction": 0.2}, {"split": "val"}, {"latent_layer": "g"}):
        assert _store(**{**base, **change}).manifest["fingerprint"] != fp
    assert _store(ckpt="other", **base).manifest["fingerprint"] != fp
    s.save(tmp_path)
    assert CaseStore.load(tmp_path, fp).manifest["fingerprint"] == fp
    with pytest.raises(StaleIndexError):
        CaseStore.load(tmp_path, fingerprint(**{**base, "pooling": "a_concat"}))


def test_query_from_another_checkpoint_is_refused() -> None:
    s = _store()
    with pytest.raises(StaleIndexError):
        s.search(s.vectors[:2], 3, query_checkpoint_sha256="different-checkpoint")


def test_search_after_save_and_load_is_identical(tmp_path: Path) -> None:
    s = _store(split="train")
    s.save(tmp_path)
    s2 = CaseStore.load(tmp_path, s.manifest["fingerprint"])
    q = s.vectors[:5]
    a, b = s.search(q, 4, query_checkpoint_sha256="abc"), s2.search(q, 4, query_checkpoint_sha256="abc")
    assert np.array_equal(a[1], b[1]) and np.allclose(a[0], b[0])
