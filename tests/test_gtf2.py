"""G-TF2 helpers: the flatline detector and the union rule (CPU only; no GPU, no data, no scoring).

Pre-registration: ``PC-STGNN/stage_f_gtf2_preregistration.md``. The G-TF behaviour must not change when the
flatline detector is off.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(2)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_gtf as G  # noqa: E402
from fault_localization.dropout.silent import apply_silent_failure_batch  # noqa: E402

MEAN = np.array([1.0, 1.0, 1.0], np.float32)
STD = np.array([0.05, 0.05, 0.05], np.float32)
SIGMA0 = 0.01                                           # p.u.; the G-TF2 noise level


def _batch(b=64, n=25, s=20, seed=0, phases=None):
    """Synthetic loader tensor [B,N,S,6]: z-scored voltages with sigma0 = 1e-2 p.u. noise, static mask."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(b, n, s, 6)
    volts_pu = 1.0 + SIGMA0 * torch.randn(b, n, s, 3, generator=g)
    x[..., :3] = (volts_pu - torch.as_tensor(MEAN)) / torch.as_tensor(STD)
    mask = torch.ones(n, 3) if phases is None else phases
    x[..., 3:] = mask[None, :, None, :]
    return x


def test_gtf2_flatline_score_is_minus_the_std_in_per_unit_over_instrumented_phases():
    x = _batch(b=2, n=3, s=20, seed=1)
    ph = torch.tensor([[1., 1., 1.], [1., 0., 0.], [0., 0., 0.]])
    x[..., 3:] = ph[None, :, None, :]
    sc = G.flatline_score(x, MEAN, STD)
    assert sc.shape == (2, 3)
    volts = x[..., :3].numpy() * STD + MEAN                              # p.u.
    sd = volts.std(axis=2)                                               # [B,N,3] population std
    assert sc[0, 0] == pytest.approx(-sd[0, 0].mean(), rel=1e-4)         # three instrumented phases
    assert sc[0, 1] == pytest.approx(-sd[0, 1, 0], rel=1e-4)             # only phase a counts
    assert np.isfinite(sc[:, 2]).all()                                   # no instrumented phase: no NaN
    assert (sc <= 0).all()
    assert -sc[0, 0] == pytest.approx(SIGMA0, rel=0.4)                   # a live sensor shows about sigma0


def test_gtf2_flatline_score_is_exactly_zero_for_a_stuck_sensor_and_high_means_flat():
    x = _batch()
    xs, corrupted = apply_silent_failure_batch(x, 2, "stuck", seed=3, norm_stats=(MEAN, STD))
    sc = G.flatline_score(xs, MEAN, STD)
    lab = np.zeros(sc.shape, bool)
    for r, nodes in enumerate(corrupted):
        lab[r, nodes] = True
    assert lab.sum() == 2 * x.shape[0]
    assert (sc[lab] == 0.0).all()                                        # the stuck sensor has zero spread
    assert (sc[~lab] < 0).all()
    assert sc[lab].min() > sc[~lab].max()                                # separable: AUROC 1.0 on this construction


def test_gtf2_threshold_at_half_percent_flags_stuck_and_not_clean_sensors():
    clean = G.flatline_score(_batch(b=4000, seed=5), MEAN, STD)          # clean half-A stand-in
    avail = np.ones(clean.shape, bool)
    thr = G.per_sensor_thresholds(clean, avail, G.FPR_GTF2)
    assert G.FPR_GTF2 == 0.005 and thr.shape == (25,)
    test_clean = G.flatline_score(_batch(b=4000, seed=6), MEAN, STD)
    realized = (test_clean > thr[None, :]).mean()
    assert realized == pytest.approx(0.005, abs=0.003)                   # per-sensor rate near the target
    xs, corrupted = apply_silent_failure_batch(_batch(b=200, seed=7), 3, "stuck", seed=9, norm_stats=(MEAN, STD))
    sc = G.flatline_score(xs, MEAN, STD)
    lab = np.zeros(sc.shape, bool)
    for r, nodes in enumerate(corrupted):
        lab[r, nodes] = True
    flags = G.flags_from_scores(sc, np.ones_like(lab), thr)
    assert flags[lab].all()                                              # TPR 1.0 for a flatlined sensor
    assert flags[~lab].mean() == pytest.approx(0.005, abs=0.004)


def test_gtf2_union_rule_flags_either_detector_and_never_an_unavailable_sensor():
    ins = np.array([[5.0, 0.1, 0.1, 9.0]])
    flat = np.array([[-1.0, 0.0, -1.0, 0.0]])                            # high (0.0) = flat
    avail = np.array([[True, True, True, False]])
    thr_ins = np.full(4, 1.0)
    thr_flat = np.full(4, -0.5)
    f = G.detector_flags(ins, flat, avail, thr_ins, thr_flat)
    assert f["ins"].tolist() == [[True, False, False, False]]            # residual above threshold; sensor 3 unavailable
    assert f["flat"].tolist() == [[False, True, False, False]]           # std below threshold
    assert f["union"].tolist() == [[True, True, False, False]]           # OR
    assert (f["union"] == (f["ins"] | f["flat"])).all()
    assert not f["union"][0, 3]


def test_gtf2_with_the_flatline_detector_off_the_flags_equal_gtf_exactly():
    rng = np.random.RandomState(0)
    ins = rng.exponential(size=(500, 25))
    avail = rng.rand(500, 25) > 0.1
    thr = G.per_sensor_thresholds(ins, avail, G.FPR_TARGET)
    off = G.detector_flags(ins, None, avail, thr)                        # thr_flat omitted: G-TF
    assert list(off) == ["union"]
    assert np.array_equal(off["union"], G.flags_from_scores(ins, avail, thr))
    # a flatline detector that can never fire does not change the G-TF flags either
    never = G.detector_flags(ins, np.full_like(ins, -1.0), avail, thr, np.full(25, np.inf))
    assert np.array_equal(never["union"], off["union"])


def test_gtf2_threshold_rate_scope_and_paths_are_separate_from_gtf():
    assert G.FPR_TARGET == 0.01 and G.FPR_GTF2 == 0.005                  # G-TF unchanged
    assert G.OUT_ROOT_GTF2 != G.OUT_ROOT and "gtf2" in str(G.OUT_ROOT_GTF2)
    assert set(G.GTF2_TYPES) == set(G.CORE_TYPES) and len(G.GTF2_TYPES) == 5
    assert G.GTF2_PRIMARY == "stuck" and set(G.GTF2_SECONDARY) == {"bias_2pct", "bias_5pct", "swap"}
    assert G.scope_sigmas("core") == (0.01,) and G.SEEDS == (42, 43, 44) and G.S_LEVELS == (1, 2, 3)


def test_gtf2_not_ready_until_preregistration_is_committed(tmp_path, monkeypatch):
    gate = tmp_path / "gate.md"
    gate.write_text("no g-tf2 arm here")
    monkeypatch.setattr(G, "GATE_DOC", gate)
    monkeypatch.setattr(G, "GTF2_DOC", tmp_path / "missing.md")
    monkeypatch.setattr(G, "git_clean_and_tracked", lambda p: True)
    monkeypatch.setattr(G, "raw_dir", lambda s: tmp_path / f"raw{s}")
    ok, problems = G.check_ready_gtf2()
    assert not ok and any("marked" in p for p in problems)
    gate.write_text("[Added 2026-10-26 09:30 IST, before any G-TF2 result]")
    (tmp_path / "missing.md").write_text("x")
    (tmp_path / "raw0.01").mkdir(); (tmp_path / "raw0.01" / "COMPLETE.json").write_text("{}")
    assert G.check_ready_gtf2()[0]
    monkeypatch.setattr(G, "git_clean_and_tracked", lambda p: False)       # uncommitted documents block it
    assert not G.check_ready_gtf2()[0]


def test_gtf2_summarize_applies_the_primary_and_secondary_rules(tmp_path):
    import json
    conds = [f"{t}_s{s}" for t in G.GTF2_TYPES for s in (1, 2, 3)]
    for seed in G.SEEDS:
        c = {}
        for name in conds:
            typ = name.rsplit("_s", 1)[0]
            gain = 0.01 if typ in ("stuck", "bias_5pct") else -0.01          # stuck and bias 5 % improve, the rest do not
            c[name] = {"f1": {"nogate": 0.5 + 0.001 * seed % 7, "gtf2": 0.5 + 0.001 * seed % 7 + gain, "oracle": 0.7},
                       "recovered": {"denominator": 0.2, "recovered": 0.05}, "corruption_loss": 0.2, "tpr": 0.9,
                       "realized_fpr_clean_sensors_of_corrupted_windows": 0.01}
        c["clean"] = {"f1": {"nogate": 0.7, "gtf2": 0.69}, "realized_fpr_clean_windows": 0.01}
        d = tmp_path / "sigma_0.01"; d.mkdir(exist_ok=True)
        (d / f"seed_{seed:05d}.json").write_text(json.dumps({"seed": seed, "conditions": c}))
    summ = G.stage_summarize("core", tmp_path, "gtf2")
    rules = summ["decision_rule_gtf2_sigma1e-2"]
    assert rules["stuck"]["role"] == "primary" and rules["stuck"]["useful (better than no gate in all 3 seeds at every s)"]
    assert rules["bias_5pct"]["role"] == "secondary" and rules["bias_5pct"]["useful (better than no gate in all 3 seeds at every s)"]
    assert not rules["bias_2pct"]["useful (better than no gate in all 3 seeds at every s)"]
    assert not rules["swap"]["useful (better than no gate in all 3 seeds at every s)"]
    assert rules["gauss_0.02"]["role"] == "reported only"
    assert "gtf2_minus_nogate" in summ["sigma"]["0.01"]["conditions"]["stuck_s1"]
