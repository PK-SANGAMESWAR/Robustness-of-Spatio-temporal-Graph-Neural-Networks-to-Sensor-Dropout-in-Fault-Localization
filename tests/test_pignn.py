"""Invariants for the PIGNN screening arms P1 / P2 / P3.

The tests that matter here are the ones that catch a *silent no-op*: a physics
term that is wired in but has no effect would otherwise show up as "the physics
did not help", which is a wrong scientific conclusion rather than a bug report.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fault_localization.graph.sensitivity import load_sensitivity
from fault_localization.models.graphsage import RGSAGE
from fault_localization.models.physics_loss import (
    masked_cosine, observed_deviation, physics_consistency_loss)
from fault_localization.models.pignn import PIGNNSage

EI = np.load("dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
EA = np.load("dataset_fullscale_50/processed/edge_attr.npy")
SENS = load_sensitivity()["sensitivity"]
KW = dict(aggregator="max", num_layers=4)


def _pair(**flags):
    torch.manual_seed(0)
    base = RGSAGE(EI, "max", num_layers=4).eval()
    torch.manual_seed(0)
    arm = PIGNNSage(EI, EA, SENS, **KW, **flags).eval()
    return base, arm


def _x(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 25, 20, 6, generator=g)
    x[..., 3:] = 1.0
    return x


# ---------------------------------------------------------------- reduction

def test_both_switches_off_is_exactly_rgsage():
    """The screen is only fair if the arms share a backbone bit-for-bit."""
    base, arm = _pair()
    assert torch.equal(base(_x()), arm(_x()))


def test_parameter_counts_isolate_capacity():
    """P1/P2 add exactly one scalar; P3 adds none. No capacity confound."""
    n_base = sum(p.numel() for p in RGSAGE(EI, "max", num_layers=4).parameters())
    for flags in ({"use_impedance_weights": True}, {"use_sensitivity_bias": True}):
        n = sum(p.numel() for p in PIGNNSage(EI, EA, SENS, **KW, **flags).parameters())
        assert n == n_base + 1, flags
    n_p3 = sum(p.numel() for p in PIGNNSage(EI, EA, SENS, **KW).parameters())
    assert n_p3 == n_base


# ---------------------------------------------------------------- P1

def test_p1_changes_the_output():
    base, arm = _pair(use_impedance_weights=True)
    assert not torch.allclose(base(_x()), arm(_x()))


def test_p1_kernel_is_a_decay_never_a_reward():
    """softplus(gamma) >= 0, so weight <= 1 and falls with impedance."""
    arm = PIGNNSage(EI, EA, SENS, **KW, use_impedance_weights=True)
    with torch.no_grad():
        arm.gamma_raw.fill_(-50.0)          # softplus -> ~0
        assert torch.allclose(arm.edge_weights(), torch.ones_like(arm.edge_weights()),
                              atol=1e-5), "gamma->0 must recover uniform weighting"
        arm.gamma_raw.fill_(5.0)
        w = arm.edge_weights()
    assert (w <= 1.0 + 1e-6).all()
    n_real = EI.shape[1]
    assert torch.allclose(w[n_real:], torch.ones(25), atol=1e-6), "self-loops must weigh 1"
    z = arm.edge_z[:n_real]
    order = torch.argsort(z)
    assert torch.all(w[:n_real][order].diff() <= 1e-6), "weight must be non-increasing in |Z|"


def test_p1_gamma_receives_gradient():
    arm = PIGNNSage(EI, EA, SENS, **KW, use_impedance_weights=True)
    arm(_x()).sum().backward()
    assert arm.gamma_raw.grad is not None and arm.gamma_raw.grad.abs() > 0


# ---------------------------------------------------------------- P2

def test_p2_bias_matches_the_sensitivity_matrix():
    base, arm = _pair(use_sensitivity_bias=True)
    with torch.no_grad():
        delta = arm(_x()) - base(_x())
        expected = arm.beta * arm.sens_bias.unsqueeze(0)
    assert torch.allclose(delta, expected.expand_as(delta), atol=1e-5)


def test_p2_no_fault_class_gets_no_geometric_prior():
    arm = PIGNNSage(EI, EA, SENS, **KW, use_sensitivity_bias=True)
    assert torch.all(arm.sens_bias[:, 0] == 0)


def test_p2_beta_zero_recovers_the_baseline():
    base, arm = _pair(use_sensitivity_bias=True)
    with torch.no_grad():
        arm.beta.fill_(0.0)
    assert torch.allclose(base(_x()), arm(_x()), atol=1e-6)


def test_p2_beta_receives_gradient():
    arm = PIGNNSage(EI, EA, SENS, **KW, use_sensitivity_bias=True)
    arm(_x()).sum().backward()
    assert arm.beta.grad is not None and arm.beta.grad.abs() > 0


# ---------------------------------------------------------------- P3

def _window_matching(cls: int, drop: int = 0, scale: float = 1.0):
    """A window whose per-sensor sag depth is exactly class `cls`'s signature.

    The deviation is a peak-to-trough swing, so the window must actually dip:
    the first half sits at 0 and the second half at -signature, giving a swing
    of exactly ``signature``. This mirrors the corpus, where every fault-labelled
    window straddles the fault onset at step 40.
    """
    x = torch.zeros(1, 25, 20, 6)
    x[..., 3:] = 1.0
    sig = torch.as_tensor(SENS[cls - 1]).view(1, 25, 1, 1) * scale
    x[:, :, 10:, :3] = -sig
    if drop:
        x[:, :drop] = 0.0                    # dropped sensors: all six channels
    return x


def _certain(cls: int):
    lg = torch.full((1, 25, 26), -20.0)
    lg[..., cls] = 20.0
    return lg


def test_p3_zero_when_prediction_matches_the_physics():
    assert physics_consistency_loss(_certain(7), _window_matching(7),
                                    torch.as_tensor(SENS)).item() == pytest.approx(0, abs=1e-5)


def test_p3_penalizes_a_physically_inconsistent_class():
    """Must be strictly worse than the matching class, and it must be the
    *most* distant signature that costs the most."""
    x = _window_matching(7)
    s = torch.as_tensor(SENS)
    cos = (s @ s[6])
    worst = int(torch.argmin(cos).item()) + 1
    l_match = physics_consistency_loss(_certain(7), x, s).item()
    l_worst = physics_consistency_loss(_certain(worst), x, s).item()
    assert l_worst > l_match
    assert l_worst > 0.2, f"expected a real penalty for the most distant class, got {l_worst}"


def test_p3_is_gated_off_when_the_model_says_no_fault():
    assert physics_consistency_loss(_certain(0), _window_matching(7),
                                    torch.as_tensor(SENS)).item() == pytest.approx(0, abs=1e-6)


def test_p3_ignores_dropped_sensors():
    """Masked cosine, so a matching pattern still scores 0 with sensors gone."""
    s = torch.as_tensor(SENS)
    for k in (0, 5, 10):
        loss = physics_consistency_loss(_certain(7), _window_matching(7, drop=k), s)
        assert loss.item() == pytest.approx(0, abs=1e-5), f"k={k}"


def test_p3_is_invariant_to_fault_severity():
    """The factorization says severity is a free scalar; only pattern is physical."""
    s = torch.as_tensor(SENS)
    losses = [physics_consistency_loss(_certain(9), _window_matching(3, scale=a), s).item()
              for a in (0.01, 1.0, 100.0)]
    assert max(losses) - min(losses) < 1e-5, losses


def test_p3_gradient_reaches_the_logits():
    lg = torch.randn(4, 25, 26, requires_grad=True)
    physics_consistency_loss(lg, _x(4), torch.as_tensor(SENS)).backward()
    assert lg.grad is not None and lg.grad.abs().sum() > 0


def test_p3_rejects_misshaped_signatures():
    with pytest.raises(ValueError):
        physics_consistency_loss(torch.randn(2, 25, 26), _x(2), torch.as_tensor(SENS)[:10])


def test_observed_deviation_flags_dropped_nodes():
    x = _window_matching(4, drop=6)
    d, a = observed_deviation(x)
    assert torch.all(a[:, :6] == 0) and torch.all(a[:, 6:] == 1)
    assert torch.all(d[:, :6] == 0)


def test_masked_cosine_recovers_unmasked_case():
    g = torch.Generator().manual_seed(3)
    d = torch.rand(2, 25, generator=g)
    s = torch.as_tensor(SENS)
    a = torch.ones(2, 25)
    got = masked_cosine(d, s, a)
    want = torch.nn.functional.cosine_similarity(
        d.unsqueeze(1), s.unsqueeze(0), dim=-1)
    assert torch.allclose(got, want, atol=1e-5)


# ---------------------------------------------------------------- integration

def test_train_epoch_without_physics_is_unchanged():
    """Every existing benchmark must keep a pure-CE objective."""
    from fault_localization.experiments.train import train_epoch
    model = PIGNNSage(EI, EA, SENS, **KW)
    x, y = _x(8), torch.randint(0, 26, (8,))
    loader = [{"x": x, "y": y}]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = train_epoch(model, loader, opt, torch.device("cpu"))
    assert "phys_loss" not in out

    model2 = PIGNNSage(EI, EA, SENS, **KW)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    out2 = train_epoch(model2, loader, opt2, torch.device("cpu"),
                       physics={"lam": 1.0, "signatures": torch.as_tensor(SENS)})
    assert "phys_loss" in out2 and out2["phys_loss"] >= 0


# ------------------------------------------------- regression: no degeneracy
#
# An earlier observed_deviation measured the sag against the corpus channel
# means (0.83/0.60/0.76 p.u.), which sit far below nominal because absent
# phases are stored as 0.0. That zeroed the deviation on 97.6% of node entries,
# leaving the cosine identical across all 26 classes -- so the residual was a
# constant 1.0 and the loss did nothing but push p_0 up. The unit tests above
# all passed anyway, because they use synthetic windows. These two use the real
# corpus, which is the only thing that would have caught it.

@pytest.mark.slow
def test_deviation_is_informative_on_real_windows():
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    ds = WindowGraphDataset("dataset_fullscale_50", "test")
    batch = next(iter(make_dataloader(ds, 256, shuffle=True, seed=7)))
    d, a = observed_deviation(batch["x"])
    live = a.sum(1).mean().item()
    nonzero = (d > 0).float().sum(1).mean().item()
    assert nonzero > 0.8 * live, (
        f"deviation is degenerate: only {nonzero:.1f} of {live:.1f} live sensors "
        "carry any signal; the physics term cannot discriminate")


@pytest.mark.slow
def test_matched_filter_beats_chance_on_real_windows():
    """The physics must actually rank the true class above chance (1/25 = 4%)."""
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    ds = WindowGraphDataset("dataset_fullscale_50", "test")
    it = iter(make_dataloader(ds, 256, shuffle=True, seed=7))
    xs, ys = zip(*[(b["x"], b["y"]) for b, _ in zip(it, range(20))])
    x, y = torch.cat(xs), torch.cat(ys)
    d, a = observed_deviation(x)
    cos = masked_cosine(d, torch.as_tensor(SENS), a)
    assert (cos.std(1) < 1e-8).float().mean() < 0.05, "too many all-tie windows"
    fault = y > 0
    acc = (cos.argmax(1)[fault] + 1 == y[fault]).float().mean().item()
    assert acc > 0.08, f"matched filter at {acc:.2%}, barely above 4% chance"
