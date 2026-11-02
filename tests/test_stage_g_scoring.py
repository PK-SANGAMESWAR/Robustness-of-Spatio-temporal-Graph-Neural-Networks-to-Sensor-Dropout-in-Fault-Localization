"""Stage G scorer: predictions are saved, round-trip, and the numbers equal the Gate-4 scorer's.

CPU only; the equivalence tests use a tiny eval root built from six pilot events (never the sealed set).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(2)
REPO = Path(__file__).resolve().parents[1]
PILOT_RAW = REPO / "data/ood_pilot/raw_sim"

from fault_localization.stage_g import ood_pipeline as P  # noqa: E402
from fault_localization.stage_g import scoring as S  # noqa: E402


@pytest.fixture(scope="module")
def mini_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("mini")
    raw = base / "src" / "raw_sim"
    raw.mkdir(parents=True)
    for f in sorted(PILOT_RAW.glob("*.npz"))[:6]:
        shutil.copy2(f, raw / f.name)
    P.build_eval_root(raw, base / "eval_root", REPO / "dataset_fullscale_50", expected_events=6)
    return base / "eval_root"


def test_is_grounded_matches_the_registered_split():
    grounded = {"AG", "BG", "CG", "ABG", "BCG", "CAG", "ABCG"}
    ungrounded = {"AB", "BC", "CA", "ABC"}
    assert all(S.is_grounded(t) for t in grounded) and not any(S.is_grounded(t) for t in ungrounded)
    assert len(grounded) == 7 and len(ungrounded) == 4


def test_grids_contain_the_registered_cells():
    assert S.GRIDS["ood"] == [(0, 0.0), (0, 0.0005), (0, 0.002), (0, 0.01), (5, 0.0), (10, 0.0)]
    assert len(S.GRIDS["full"]) == 20 and len(S.GRIDS["mismatch"]) == 8
    assert all(s in S.SIGMA_LEVELS for _k, s in S.GRIDS["ood"])


def test_summary_and_predictions_round_trip(tmp_path):
    n = 80
    rng = np.random.RandomState(0)
    y = rng.randint(0, 26, n).astype(np.int8)
    pf = y.copy()
    pf[:20] = (pf[:20] + 1) % 26
    res = S.ScoreResult(
        window_ids=np.array([f"e{i // 40}_w{i % 40:04d}" for i in range(n)]), y_true=y,
        resistance=np.where(np.arange(n) < 40, 5.0, 20.0).astype(np.float32),
        fault_type=np.where(np.arange(n) % 2 == 0, "AG", "AB"), event_id=np.array([f"e{i // 40}" for i in range(n)]),
        cells=[(0, 0.0)], n_offsets=3, preds={S.cell_key(0, 0.0, o): {"pf": pf} for o in range(3)},
        has_table=False, use_fusion=False, beta=None, seconds=0.0)
    cells = S.summarize(res, resistances=[5.0, 20.0])
    c = cells["k0_sigma0"]
    assert c["f1_mean"] == pytest.approx(S.macro_f1(y.astype(int), pf.astype(int)))
    assert set(c["f1_by_resistance"]) == {"5.0", "20.0"} and set(c["f1_by_group"]) == {"grounded", "ungrounded"}
    path = S.write_summary(tmp_path, "toy", res, {"model": "toy"}, resistances=[5.0, 20.0])
    loaded = S.load_predictions(tmp_path / "toy_predictions.npz")
    assert np.array_equal(loaded["pf__k0_s0_o1"], pf) and np.array_equal(loaded["y_true"], y)
    assert loaded["cells"]["n_offsets"] == 3
    assert path.exists()


def _loader_order_ids(ds, n):
    """The first ``n`` windows in loader order -- what the Gate-4 scorer's ``max_windows`` keeps."""
    from fault_localization.data.dataset import make_dataloader
    ids = []
    for batch in make_dataloader(ds, S.BATCH_SIZE, shuffle=False):
        ids.extend(batch["window_id"])
        if len(ids) >= n:
            break
    return ids[:n]


def _equivalence(monkeypatch, mini_root, arm, n_windows, cells):
    import score_s4_4_gate4_run as old
    from fault_localization.data.dataset import WindowGraphDataset
    ds = WindowGraphDataset(str(mini_root), "all")
    monkeypatch.setattr(old, "WindowGraphDataset", lambda root, split: WindowGraphDataset(str(mini_root), "all"))
    summary_old = old.score_run(arm, 42, max_windows=n_windows)           # CPU; writes nothing with max_windows
    ids = _loader_order_ids(ds, n_windows)
    model, _ = S.load_model(REPO / "results/s4_4" / arm / "pcstgnn/seed_00042", ds.edge_index, torch.device("cpu"))
    res = S.score_dataset(model, ds, cells, device=torch.device("cpu"), window_subset=ids)
    new = S.summarize(res, resistances=[5.0, 20.0])
    for k, s in cells:
        a = summary_old["cells"][f"k{k}_sigma{s}"]["f1_fused_per_offset"]
        b = new[f"k{k}_sigma{s:g}"]["f1_per_offset"]
        assert a == pytest.approx(b, abs=1e-12), (k, s)
        if summary_old["has_table"]:
            oc, nc = summary_old["cells"][f"k{k}_sigma{s}"], new[f"k{k}_sigma{s:g}"]
            assert oc["f1_network_only_mean"] == pytest.approx(nc["f1_network_only_mean"], abs=1e-12)
            assert oc["f1_psm_mean"] == pytest.approx(nc["f1_psm_mean"], abs=1e-12)


def test_equals_gate4_scorer_without_table_r0n(monkeypatch, mini_root):
    _equivalence(monkeypatch, mini_root, "r0n", 48, [(0, 0.0), (10, 0.0), (5, 0.002), (0, 0.01)])


@pytest.mark.slow
def test_equals_gate4_scorer_with_fusion_r5(monkeypatch, mini_root):
    _equivalence(monkeypatch, mini_root, "r5", 32, [(0, 0.0), (10, 0.01)])


def test_test_split_is_refused_without_approval():
    r = subprocess.run([sys.executable, str(REPO / "scripts/score_stage_g.py"), "--arm", "r0n", "--seed", "42",
                        "--split", "test"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "approved-test-split" in (r.stderr + r.stdout)
