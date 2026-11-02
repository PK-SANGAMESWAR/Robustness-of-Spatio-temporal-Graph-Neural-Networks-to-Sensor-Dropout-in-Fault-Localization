"""G-TF helpers and the merged S4.4 queue (CPU only; no GPU, no training)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_gtf as G  # noqa: E402
import run_s4_4_queue as Q  # noqa: E402


def test_gtf_split_is_disjoint_complete_and_deterministic():
    events = [f"e{i}" for i in range(101)]
    a, b = G.split_events(events, seed=7)
    assert not (a & b) and a | b == set(events)
    assert abs(len(a) - len(b)) <= 1
    a2, _ = G.split_events(list(reversed(events)), seed=7)
    assert a == a2 and G.split_hash(a) == G.split_hash(a2)
    a3, _ = G.split_events(events, seed=8)
    assert a3 != a


def test_gtf_event_id_strips_window_suffix():
    assert G.event_of("65_AB_0p1_0047_w23") == "65_AB_0p1_0047"


def test_gtf_per_sensor_threshold_hits_target_rate_and_skips_unavailable():
    rng = np.random.RandomState(0)
    scores = rng.exponential(size=(20000, 3)) * np.array([1.0, 5.0, 1.0])
    avail = np.ones((20000, 3), bool)
    avail[:, 2] = False                                   # sensor 2 never available
    thr = G.per_sensor_thresholds(scores, avail, 0.01)
    assert np.isinf(thr[2])
    for j in (0, 1):
        assert abs((scores[:, j] > thr[j]).mean() - 0.01) < 0.002
    assert thr[1] > 3 * thr[0]                            # thresholds are per sensor, not pooled
    flags = G.flags_from_scores(scores, avail, thr)
    assert not flags[:, 2].any()


def test_gtf_gate_zero_zeroes_all_six_channels_of_flagged_nodes_only():
    x = torch.ones(2, 4, 3, 6)
    flags = torch.zeros(2, 4, dtype=torch.bool)
    flags[0, 1] = True
    out = G.gate_zero(x, flags)
    assert (out[0, 1] == 0).all() and (x == 1).all()      # input untouched
    assert out.sum() == x.sum() - 3 * 6
    assert (out[0, 0] == 1).all() and (out[1] == 1).all()


def test_gtf_detector_rule_needs_all_three_s_levels():
    def s52(deltas):
        return {f"bias_2pct_s{s}": {"loo": 0.9 + d, "ins": 0.9} for s, d in zip((1, 2, 3), deltas)}
    assert G.decide_detector(s52([0.02, 0.02, 0.02]))["detector"] == "loo"
    assert G.decide_detector(s52([0.02, 0.02, 0.005]))["detector"] == "ins"
    assert G.decide_detector(s52([-0.0002, -0.0004, -0.0006]))["detector"] == "ins"
    assert G.decide_detector(s52([0.01, 0.01, 0.01]))["detector"] == "loo"     # "at least 0.01"


def test_gtf_recovered_fraction_needs_denominator_of_at_least_0_01_and_holm():
    r = G.recovered_fraction(0.6, 0.4, 0.8)
    assert r["recovered"] == pytest.approx(0.5) and r["denominator"] == pytest.approx(0.4)
    assert G.recovered_fraction(0.5, 0.5, 0.5)["recovered"] is None
    assert G.recovered_fraction(0.5, 0.6, 0.5)["recovered"] is None
    assert G.recovered_fraction(0.505, 0.5, 0.509)["recovered"] is None          # denominator 0.009 < 0.01
    assert G.recovered_fraction(0.505, 0.5, 0.51)["recovered"] == pytest.approx(0.5)   # exactly 0.01 is defined
    adj = G.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_gtf_core_scope_has_all_five_types_and_the_rule_uses_bias_2pct():
    conds = G.scope_conditions("core")
    assert {c[0].rsplit("_s", 1)[0] for c in conds} == {"stuck", "bias_2pct", "bias_5pct", "gauss_0.02", "swap"}
    assert len(conds) == 15 and G.scope_sigmas("core") == (0.01,)
    assert len(G.scope_sigmas("full")) == 4


def test_gtf_not_ready_until_preregistration_complete(tmp_path, monkeypatch):
    gate = tmp_path / "gate.md"
    gate.write_text("no g-tf arm here")
    monkeypatch.setattr(G, "GATE_DOC", gate)
    monkeypatch.setattr(G, "DECISION_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(G, "git_clean_and_tracked", lambda p: True)
    monkeypatch.setattr(G, "raw_dir", lambda s: tmp_path / f"raw{s}")
    monkeypatch.setattr(G, "sigma_summary_path", lambda s: tmp_path / f"s{s}.json")
    monkeypatch.setattr(G, "DRAFT_DOC", gate)
    ok, problems = G.check_ready("core")
    assert not ok and any("marked" in p for p in problems) and any("detector decision" in p for p in problems)
    # everything in place -> ready
    gate.write_text("### G-TF\n[Added after S5.1, before any G-TF result -- 2026-10-25 05:00 IST]")
    dec = tmp_path / "dec.json"; dec.write_text(json.dumps({"detector": "ins"}))
    monkeypatch.setattr(G, "DECISION_PATH", dec)
    (tmp_path / "raw0.01").mkdir(); (tmp_path / "raw0.01" / "COMPLETE.json").write_text("{}")
    (tmp_path / "s0.01.json").write_text("{}")
    ok, problems = G.check_ready("core")
    assert ok, problems
    # uncommitted documents block it
    monkeypatch.setattr(G, "git_clean_and_tracked", lambda p: False)
    assert not G.check_ready("core")[0]
    # full scope needs the approval file, which is never created here
    monkeypatch.setattr(G, "git_clean_and_tracked", lambda p: True)
    monkeypatch.setattr(G, "FULL_APPROVAL_PATH", tmp_path / "nope")
    assert any("FULL_SCOPE_APPROVED" in p for p in G.check_ready("full")[1])


def test_merged_queue_order():
    assert [(a, sd) for a, _, sd in Q.QUEUE] == [
        ("r6i", 42), ("r6i", 43), ("r6i", 44), ("r1", 42), ("r2", 42), ("r2", 43), ("r2", 44)]
    assert Q.R6I_SHUFFLE_SEED == 6 and Q.R6I_CHECKPOINT_ARM == "r5"


def test_merged_queue_lock_allows_one_process(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "LOCK_PATH", tmp_path / "q.lock")
    (tmp_path / "q.lock").write_text(str(os.getpid()))               # a live process holds it
    assert Q.acquire_lock() is False
    (tmp_path / "q.lock").write_text("999999999")                     # stale lock
    assert Q.acquire_lock() is True


def test_merged_queue_refuses_after_a_stop_and_the_old_queue_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"])
    stop = tmp_path / "stop.json"; stop.write_text("{}")
    monkeypatch.setattr(Q, "STOP_PATH", stop)
    assert Q.main() == 1
    import run_s4_4_remaining_queue as old
    assert old.main() == 1


def test_r6_lam2_0_differs_from_r6_only_in_name_and_lambda2_and_keeps_the_guard():
    import run_s4_4_remaining_queue as rq
    d = {x.split(":")[0] for x in rq._dict_diff(rq._frozen_config("r6", 42), rq._frozen_config("r6_lam2_0", 42))}
    assert d == {"benchmark_version", "pc_loss.lambda2"}
    c = rq._frozen_config("r6_lam2_0", 42)
    assert c["pc_loss"]["lambda2"] == 0.0 and c["model"]["shuffle_seed"] == 6 and c["training"]["collapse_abort"]["epoch"] == 3
    for arm in ("r5_gamma0", "r4", "r3", "r6_lam2_0", "r1", "r2"):
        assert rq._frozen_config(arm, 42)["benchmark_version"] == arm      # output dir name == arm name


def test_gate_equals_training_dropout_check_passes():
    assert G.check_gate_equals_training_dropout()["passed"]


def test_r1_r2_configs_are_r0n_plus_documented_additions_only():
    import run_s4_4_remaining_queue as rq
    r0n = rq._frozen_config("r0n", 42)
    skip = ("benchmark_version", "seed")
    d1 = {d.split(":")[0] for d in rq._dict_diff(r0n, rq._frozen_config("r1", 42)) if not d.startswith(skip)}
    d2 = {d.split(":")[0] for d in rq._dict_diff(r0n, rq._frozen_config("r2", 42)) if not d.startswith(skip)}
    assert d1 == {"model.num_fault_types", "model.use_type_head", "pc_loss.type_loss_weight"}
    assert d2 == {"model.decoder_hidden", "model.use_decoder", "pc_loss.recon_loss_weight"}
    assert rq._frozen_config("r1", 42)["pc_loss"]["type_loss_weight"] == 1.0
    assert rq._frozen_config("r2", 42)["pc_loss"]["recon_loss_weight"] == 1.0
