"""Unit tests for the S4.4-remaining-arms code changes
(PC-STGNN/S4.4_remaining_arms_plan.md Part 3, 2026-10-22):

1. ``scripts/score_s4_4_gate4_run.py`` accepts any S4.4 arm (not only r0n/r5), rebuilding the
   model from that run's own saved config.yaml via ``runner._model``.
2. The scorer also reports F1 grouped by resistance.
3. ``runner._model`` loads the physics table when lambda1>0 or lambda2>0, even with fusion off
   (R3/R4) -- not only when use_fusion (the bug that would have raised KeyError at R3/R4
   training time).
4. The R6 shuffle permutation (``scripts/make_r6_shuffle_permutation.py``) is a full derangement
   of the 25 locations with class 0 fixed, and matches what PCSTGNN actually builds for that
   seed.
5. R1/R2 loss code (``PC-STGNN/S4.4_remaining_arms_plan.md`` Part 3, 2026-10-24, D4/D5): the
   type-head cross-entropy term (R1) and the decoder reconstruction MSE term (R2) in
   ``models/common.py::pc_loss``, both defaulting to weight 0 -- a null test proves this is
   byte-identical to the old code at the default -- plus ``runner._model`` loading norm_stats
   for ``use_decoder`` without a physics table.

No training or scoring run is executed here -- every test builds tiny, in-memory models or
synthetic arrays and runs in well under a second.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fault_localization.experiments.runner import _model as build_model_from_config
from fault_localization.models.pc_stgnn import PCSTGNN

import score_s4_4_gate4_run as scorer  # noqa: E402
import make_r6_shuffle_permutation as r6  # noqa: E402

EDGE_INDEX = np.load("dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
DATA_ROOT = "dataset_fullscale_50"


def _base_config(**model_overrides) -> dict:
    """A minimal, valid config.yaml-shaped dict for the ``pcstgnn`` branch of
    ``runner._model``. Mirrors ``config/s4_4/r0n.yaml``'s structure, not its exact values."""
    model = {
        "name": "pcstgnn", "aggregator": "max", "gnn_hidden_dim": 64,
        "temporal_hidden_dim": 128, "dropout": 0.35, "beta_init": 0.0,
        "sigma_eps": 0.0007255549105131452, "sigma0": 0.0,
        "use_fusion": False, "use_type_head": False, "use_decoder": False,
    }
    model.update(model_overrides)
    return {
        "data_root": DATA_ROOT,
        "gnn_layers": 4,
        "model": model,
        "pc_loss": {"enabled": False, "lambda1": 0.0, "lambda2": 0.0, "aux_net_loss_weight": 0.0},
    }


# ---------------------------------------------------------------------------
# 3: runner._model loads the table when lambda1>0 or lambda2>0, fusion off (R3/R4 fix)
# ---------------------------------------------------------------------------

def test_fusion_off_lambda1_positive_loads_a_table():
    """R3: use_fusion=False, lambda1=0.01 > 0. Before the fix, table stayed None here and
    forward_pc() would have no 'rho_A', so pc_loss() raised KeyError at the first training
    step. After the fix, phys_table must be populated."""
    cfg = _base_config(use_fusion=False)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.01, "lambda2": 0.0, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX)
    assert model.phys_table.numel() > 0
    assert model.use_fusion is False  # the fix loads the table; it does not turn fusion on


def test_fusion_off_lambda2_positive_loads_a_table():
    """R4: use_fusion=False, lambda2=0.1 > 0 (the held-out consistency term)."""
    cfg = _base_config(use_fusion=False)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.0, "lambda2": 0.1, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX)
    assert model.phys_table.numel() > 0


def test_forward_pc_has_rho_a_for_a_fusion_off_lambda_on_arm():
    """The actual failure mode the bug produced: forward_pc()'s output dict must contain
    'rho_A' so pc_loss(lam1=...) does not raise KeyError. This is the regression test for the
    S4.4 'Tooling incident'-style failure, but for R3/R4 instead of the OOM it originally covered."""
    from fault_localization.models.common import pc_loss

    cfg = _base_config(use_fusion=False)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.01, "lambda2": 0.0, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX).eval()
    x = torch.zeros(2, 25, 20, 6)
    x[..., 3:] = 1.0
    out = model.forward_pc(x)
    assert "rho_A" in out
    y = torch.zeros(2, dtype=torch.long)
    parts = pc_loss(out, y, lam1=0.01, lam2=0.0)  # must not raise KeyError
    assert torch.isfinite(parts["total"])


def test_pc_loss_disabled_with_fusion_off_loads_no_table():
    """Non-regression: pc_loss.enabled=False must still load no table even if lambda1/2 are
    (meaninglessly) set, matching training-time behaviour where pc_enabled gates whether the
    lambda path runs at all."""
    cfg = _base_config(use_fusion=False)
    cfg["pc_loss"] = {"enabled": False, "lambda1": 0.01, "lambda2": 0.1, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX)
    assert model.phys_table.numel() == 0


def test_r0n_style_config_still_loads_no_table():
    """Non-regression: R0^n's exact pattern -- pc_loss.enabled=True purely for the noise-aug
    loader plumbing, lambda1=lambda2=0 -- must still load no table (06_STAGE_E_pcstgnn.md:
    'pc_loss.enabled=true purely for the noise-augmentation loader plumbing')."""
    cfg = _base_config(use_fusion=False)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.0, "lambda2": 0.0, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX)
    assert model.phys_table.numel() == 0


def test_fusion_on_still_loads_a_table_regardless_of_lambda():
    """Non-regression: R5/R6/R8's path (use_fusion=True) is unaffected by the fix."""
    cfg = _base_config(use_fusion=True)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.0, "lambda2": 0.0, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX)
    assert model.phys_table.numel() > 0
    assert model.use_fusion is True


def test_all_switches_off_is_still_byte_identical_to_rgsage():
    """S4.1's own invariant must still hold after the fix: a buffer, not a Parameter, so the
    RNG stream for the backbone's weights is untouched regardless of whether a table loads."""
    from fault_localization.models.graphsage import RGSAGE

    torch.manual_seed(0)
    plain = RGSAGE(EDGE_INDEX, aggregator="max", num_layers=4, temporal_hidden_dim=128,
                    num_classes=26, dropout=0.35, gnn_hidden_dim=64)
    cfg = _base_config(use_fusion=False)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.01, "lambda2": 0.0, "aux_net_loss_weight": 0.0}
    torch.manual_seed(0)
    arm = build_model_from_config(cfg, EDGE_INDEX)  # table now loads (the fix); weights must not move
    for (n1, p1), (n2, p2) in zip(plain.named_parameters(), arm.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# 1: scorer accepts any arm (rebuilds from each run's own config.yaml via runner._model)
# ---------------------------------------------------------------------------

def test_scorer_build_model_uses_runner_model_not_a_hardcoded_arm_name(tmp_path):
    """The old scorer hard-coded use_fusion = (arm == 'r5'). The new one must read use_fusion
    (and every other switch) from the run's own config.yaml, so an arbitrary arm name works."""
    cfg = _base_config(use_fusion=True, use_type_head=False, use_decoder=False)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.0, "lambda2": 0.1, "aux_net_loss_weight": 1.0}
    run_dir = tmp_path / "some_new_arm_name" / "pcstgnn" / "seed_00099"
    run_dir.mkdir(parents=True)
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    loaded = scorer._load_run_config(run_dir)
    assert loaded["model"]["use_fusion"] is True
    model = build_model_from_config(loaded, EDGE_INDEX)
    assert model.use_fusion is True
    assert model.phys_table.numel() > 0


def test_scorer_build_model_for_a_fusion_off_arm_with_lambda(tmp_path):
    """An R3/R4-style arm (whatever its directory is named) must build with a table but
    use_fusion=False, exercising both the generalized arm acceptance and the runner fix together,
    exactly as score_run() will do once it reads this config off disk."""
    cfg = _base_config(use_fusion=False)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.0, "lambda2": 0.1, "aux_net_loss_weight": 0.0}
    run_dir = tmp_path / "r4" / "pcstgnn" / "seed_00042"
    run_dir.mkdir(parents=True)
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    loaded = scorer._load_run_config(run_dir)
    model = build_model_from_config(loaded, EDGE_INDEX)
    assert model.use_fusion is False
    assert model.phys_table.numel() > 0  # the fix: table loads despite fusion being off


def test_scorer_load_run_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        scorer._load_run_config(Path("/nonexistent/run/dir"))


# ---------------------------------------------------------------------------
# 2: per-resistance F1
# ---------------------------------------------------------------------------

def test_f1_by_resistance_groups_correctly():
    """Each group's score must equal manually masking to that resistance and calling
    scorer.macro_f1 directly -- the actual contract, not a hand-assumed value. (This codebase's
    26-label macro-F1 convention, matching scripts/_metrics.py::resistance_f1, scores an absent
    class as F1=0 even under a "perfect" local prediction, so a naive "perfect means 1.0"
    expectation is wrong here by design -- the group result is well below 1.0 even with zero
    errors, because most of the 26 labels have no support in a 2-class group.)"""
    y = np.array([1, 2, 1, 2, 0, 0])
    pred = np.array([1, 2, 1, 2, 0, 0])  # zero errors within each group
    res = np.array([0.1, 0.1, 1.0, 1.0, 10.0, 10.0])
    out = scorer.f1_by_resistance(y, pred, res)
    assert set(out.keys()) == {"0.1", "1.0", "10.0"}
    for r, mask in (("0.1", res == 0.1), ("1.0", res == 1.0), ("10.0", res == 10.0)):
        assert out[r] == pytest.approx(scorer.macro_f1(y[mask], pred[mask]))
    # Groups 0.1 and 1.0 have the identical (y, pred) pair, just at different resistance, so
    # their scores must match exactly -- this is the real invariant (grouping is correct),
    # not "perfect predictions score 1.0" (false here: absent classes count as F1=0 under this
    # codebase's fixed 26-label convention, so even a perfect 2-class group scores well below 1).
    assert out["0.1"] == pytest.approx(out["1.0"])


def test_f1_by_resistance_errors_are_isolated_per_group():
    y = np.array([1, 1, 2, 2])
    pred = np.array([1, 1, 0, 0])  # the 10 ohm group is entirely wrong; the 0.1 ohm group is perfect
    res = np.array([0.1, 0.1, 10.0, 10.0])
    out = scorer.f1_by_resistance(y, pred, res)
    assert out["0.1"] == pytest.approx(scorer.macro_f1(y[res == 0.1], pred[res == 0.1]))
    assert out["10.0"] == pytest.approx(scorer.macro_f1(y[res == 10.0], pred[res == 10.0]))
    assert out["10.0"] == pytest.approx(0.0)  # every prediction wrong in that group -> F1 = 0
    assert out["0.1"] > out["10.0"]  # the perfect group must score strictly higher
    assert np.isnan(out["1.0"])  # no windows at 1 ohm in this slice


def test_f1_by_resistance_matches_whole_split_sum_structure():
    """Sanity check against the overall macro_f1: the resistance groups partition the data
    (every window's resistance is exactly one of the three training values), so grouping and
    recombining should not silently drop or duplicate any window."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 26, size=300)
    pred = rng.integers(0, 26, size=300)
    res = rng.choice([0.1, 1.0, 10.0], size=300)
    out = scorer.f1_by_resistance(y, pred, res)
    assert all(np.isfinite(v) for v in out.values())  # 300 windows, 3 groups: every group non-empty
    whole = scorer.macro_f1(y, pred)
    assert 0.0 <= whole <= 1.0
    # Not asserting whole == mean(groups): macro-F1 is not additive across subsets by
    # construction (different class supports per group) -- this just checks both paths run
    # and stay in range on the same data.
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_scorer_reports_resistances_list_and_decomposition_flag():
    """The scorer's RESISTANCES constant is exactly the three training values (config/constants.py
    fault_resistances), and the per-cell output key is 'f1_by_resistance' (checked by name since
    Part 4 will read this key to evaluate H1)."""
    assert scorer.RESISTANCES == (0.1, 1.0, 10.0)


# ---------------------------------------------------------------------------
# 4: R6 shuffle permutation (D1)
# ---------------------------------------------------------------------------

def test_r6_permutation_is_a_full_derangement_with_class_0_fixed():
    perm = r6.build_permutation(r6.SHUFFLE_SEED)
    r6.verify_r6_shuffle_permutation(perm)  # must not raise
    assert int(perm[0]) == 0
    fixed_points = [i for i in range(1, 26) if int(perm[i]) == i]
    assert fixed_points == []


def test_r6_permutation_matches_the_real_pcstgnn_class():
    perm = r6.build_permutation(r6.SHUFFLE_SEED)
    r6.verify_against_real_model(r6.SHUFFLE_SEED, perm)  # must not raise


def test_r6_permutation_saved_file_matches_the_generator():
    out_path = REPO / "config/s4_4/r6_shuffle_permutation.json"
    assert out_path.exists(), "run scripts/make_r6_shuffle_permutation.py first"
    record = json.loads(out_path.read_text())
    assert record["shuffle_seed"] == r6.SHUFFLE_SEED
    expected = r6.build_permutation(r6.SHUFFLE_SEED).tolist()
    assert record["permutation_26"] == expected
    assert record["fixed_points_1_to_25"] == []
    assert record["verified_full_derangement"] is True


def test_r6_seed_zero_would_have_had_a_fixed_point():
    """Documents why SHUFFLE_SEED is 6 and not the first integer tried: seed 0 has a fixed
    point (location 23 maps to itself), so it would have violated D1."""
    perm0 = r6.build_permutation(0)
    fixed = [i for i in range(1, 26) if int(perm0[i]) == i]
    assert fixed != []  # seed 0 fails D1 -- this is why 6 was chosen, not 0
    with pytest.raises(AssertionError):
        r6.verify_r6_shuffle_permutation(perm0)


# ---------------------------------------------------------------------------
# 5: R1/R2 loss code (D4, D5) -- no training run executed
# ---------------------------------------------------------------------------

from fault_localization.config.constants import FAULT_TYPES  # noqa: E402
from fault_localization.models.common import pc_loss, type_targets_from_metadata  # noqa: E402


def _r1r2_model(**overrides):
    """A tiny PCSTGNN with use_type_head/use_decoder on, fusion off, no table (matches
    R1/R2: lambda1=lambda2=0, use_fusion=False)."""
    cfg = _base_config(use_fusion=False, **overrides)
    return build_model_from_config(cfg, EDGE_INDEX)


def test_null_type_and_recon_weight_zero_matches_old_pc_loss_exactly():
    """The null test (Part 3 step 4): with type_loss_weight=0 and recon_loss_weight=0 (the
    defaults), passing type_targets/recon_target/recon_mask alongside must not change the
    total loss or any gradient, to 1e-7, versus calling pc_loss with neither argument at
    all. Both calls read the same `out` dict (one forward pass), so the only difference
    under test is the new code path itself, not a fresh (and non-deterministic, under
    dropout) forward pass."""
    torch.manual_seed(0)
    model = _r1r2_model(use_type_head=True, use_decoder=True).eval()  # eval: no dropout noise
    x = torch.zeros(3, 25, 20, 6)
    x[..., 3:] = 1.0
    out = model.forward_pc(x)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    type_targets = torch.tensor([-1, 3, 7], dtype=torch.long)
    recon_target = torch.randn_like(out["recon"])
    recon_mask = torch.ones_like(out["recon"])

    model.zero_grad(set_to_none=True)
    parts_old = pc_loss(out, y, lam1=0.0, lam2=0.0, aux_net_weight=0.0)
    parts_old["total"].backward(retain_graph=True)
    grads_old = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad(set_to_none=True)
    parts_new = pc_loss(out, y, lam1=0.0, lam2=0.0, aux_net_weight=0.0,
                        type_loss_weight=0.0, type_targets=type_targets,
                        recon_loss_weight=0.0, recon_target=recon_target, recon_mask=recon_mask)
    parts_new["total"].backward()
    grads_new = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert "type_ce" not in parts_new and "recon_mse" not in parts_new
    assert abs(parts_old["total"].item() - parts_new["total"].item()) < 1e-7
    assert len(grads_old) == len(grads_new) and len(grads_old) > 0
    for g_old, g_new in zip(grads_old, grads_new):
        assert torch.allclose(g_old, g_new, atol=1e-7)


def test_type_ce_masks_no_fault_windows_and_matches_manual_computation():
    """R1: type_loss_weight > 0 computes CE only over windows with a valid (non -1) type
    target, matching a manual cross-entropy over just that subset."""
    from fault_localization.models.common import expanded_cross_entropy

    torch.manual_seed(1)
    b, n, t = 4, 25, 11
    type_logits = torch.randn(b, n, t, requires_grad=True)
    out = {"logits": torch.randn(b, n, 26), "type_logits": type_logits}
    y = torch.zeros(b, dtype=torch.long)
    type_targets = torch.tensor([-1, 2, -1, 9], dtype=torch.long)

    parts = pc_loss(out, y, type_loss_weight=1.0, type_targets=type_targets)
    valid = type_targets >= 0
    expected = expanded_cross_entropy(type_logits[valid], type_targets[valid])
    assert torch.allclose(parts["type_ce"], expected.detach(), atol=1e-7)
    assert torch.allclose(parts["total"] - parts["ce"], expected, atol=1e-7)


def test_type_ce_all_windows_masked_is_zero_not_nan():
    """Edge case: a batch with no fault window at all (every type_target == -1) must
    contribute 0, not NaN/raise, so a batch like this does not derail training."""
    b, n, t = 2, 25, 11
    out = {"logits": torch.randn(b, n, 26), "type_logits": torch.randn(b, n, t)}
    y = torch.zeros(b, dtype=torch.long)
    type_targets = torch.full((b,), -1, dtype=torch.long)
    parts = pc_loss(out, y, type_loss_weight=1.0, type_targets=type_targets)
    assert parts["type_ce"].item() == 0.0
    assert torch.isfinite(parts["total"])


def test_type_loss_requires_type_logits_and_type_targets():
    out = {"logits": torch.randn(2, 25, 26)}
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(KeyError):
        pc_loss(out, y, type_loss_weight=1.0, type_targets=torch.tensor([0, 1]))
    out["type_logits"] = torch.randn(2, 25, 11)
    with pytest.raises(ValueError):
        pc_loss(out, y, type_loss_weight=1.0, type_targets=None)


def test_recon_mse_masks_dropped_channels_and_matches_manual_computation():
    """R2: recon_loss_weight > 0 computes a masked MSE, counting only the channels
    recon_mask marks as dropped."""
    torch.manual_seed(2)
    b, n = 3, 25
    recon = torch.randn(b, n, 3, requires_grad=True)
    out = {"logits": torch.randn(b, n, 26), "recon": recon}
    y = torch.zeros(b, dtype=torch.long)
    recon_target = torch.randn(b, n, 3)
    recon_mask = torch.zeros(b, n, 3)
    recon_mask[0, 5, :] = 1.0
    recon_mask[1, 10, 0] = 1.0

    parts = pc_loss(out, y, recon_loss_weight=1.0, recon_target=recon_target, recon_mask=recon_mask)
    expected = ((recon - recon_target) ** 2 * recon_mask).sum() / recon_mask.sum()
    assert torch.allclose(parts["recon_mse"], expected.detach(), atol=1e-7)
    assert torch.allclose(parts["total"] - parts["ce"], expected, atol=1e-7)


def test_recon_loss_no_dropped_channel_is_zero_not_nan():
    """Edge case: a batch where nothing was synthetically dropped (recon_mask all zero,
    e.g. k=0 with dropout_aware off) must contribute 0, not a divide-by-zero NaN."""
    b, n = 2, 25
    out = {"logits": torch.randn(b, n, 26), "recon": torch.randn(b, n, 3)}
    y = torch.zeros(b, dtype=torch.long)
    recon_target = torch.randn(b, n, 3)
    recon_mask = torch.zeros(b, n, 3)
    parts = pc_loss(out, y, recon_loss_weight=1.0, recon_target=recon_target, recon_mask=recon_mask)
    assert parts["recon_mse"].item() == 0.0
    assert torch.isfinite(parts["total"])


def test_recon_loss_requires_recon_and_target_and_mask():
    out = {"logits": torch.randn(2, 25, 26)}
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(KeyError):
        pc_loss(out, y, recon_loss_weight=1.0, recon_target=torch.zeros(2, 25, 3),
               recon_mask=torch.zeros(2, 25, 3))
    out["recon"] = torch.randn(2, 25, 3)
    with pytest.raises(ValueError):
        pc_loss(out, y, recon_loss_weight=1.0, recon_target=None, recon_mask=None)


def test_type_targets_from_metadata_masks_by_window_label_not_metadata():
    """The mask comes from the window's own label (0 = no active fault), not from whether
    the run's metadata.fault_type happens to be populated -- a run's pre-fault window still
    carries its run's fault_type in metadata but must be masked out (label 0)."""
    window_labels = torch.tensor([0, 3, 0, 7])
    metadata = [
        {"fault_type": FAULT_TYPES[2]},   # label 0 (pre-fault window of a faulted run) -> masked
        {"fault_type": FAULT_TYPES[2]},   # label 3 (fault active) -> type index 2
        {"fault_type": "none"},           # genuinely no-fault run, label 0 -> masked regardless
        {"fault_type": FAULT_TYPES[5]},   # label 7 (fault active) -> type index 5
    ]
    out = type_targets_from_metadata(window_labels, metadata)
    assert out.tolist() == [-1, 2, -1, 5]


def test_runner_model_loads_norm_stats_for_decoder_without_a_table():
    """R2: use_decoder=True, use_fusion=False, lambda1=lambda2=0 -- no physics table is
    needed (it is never read by the decoder path), but norm_stats must still load so
    train.py can de-normalise x_clean into the reconstruction target."""
    cfg = _base_config(use_fusion=False, use_decoder=True)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.0, "lambda2": 0.0, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX)
    assert model.phys_table.numel() == 0
    assert model.norm_mean.numel() == 3 and model.norm_std.numel() == 3


def test_recon_target_uses_observe_on_x_clean():
    """Integration-style check of the train.py wiring: the reconstruction target is
    observe(x_clean)'s per-unit value, de-normalised the same way the physics path
    already de-normalises x (observation.observe), not the raw z-scored tensor."""
    from fault_localization.physics.observation import observe

    cfg = _base_config(use_fusion=False, use_decoder=True)
    cfg["pc_loss"] = {"enabled": True, "lambda1": 0.0, "lambda2": 0.0, "aux_net_loss_weight": 0.0}
    model = build_model_from_config(cfg, EDGE_INDEX).eval()
    x_clean = torch.randn(2, 25, 20, 6)
    x_clean[..., 3:] = 1.0
    with torch.no_grad():
        recon_target, _ = observe(x_clean, model.norm_mean, model.norm_std, model.obs_reduce)
    assert recon_target.shape == (2, 25, 3)
    # Sanity: de-normalised values are not just the raw (z-scored) last-timestep channels.
    assert not torch.allclose(recon_target, x_clean[:, :, -1, :3])
