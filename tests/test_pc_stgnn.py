"""Invariants for PC-STGNN.

Two failure modes are invisible in a training curve and would each cost weeks:

*The physics is wired in but has no effect.*  The run completes, the numbers
match the baseline, and the conclusion recorded is "physics does not help" --
a wrong scientific result rather than a bug report.

*The physics path quietly changes the backbone.*  Then every attribution claim
in Stage E is confounded by capacity or initialisation, and "lambda = beta = 0
recovers the Paper-1 model exactly" (S4.1) is false.

The gradient checks mirror the S4.2 audit table of ``01_SPEC.md`` §8.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fault_localization.models.common import pc_loss, unlabelable_window_mask
from fault_localization.models.graphsage import RGSAGE
from fault_localization.models.pc_stgnn import PCSTGNN
from fault_localization.physics import grids, residual as R

EI = np.load("dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
KW = dict(aggregator="max", num_layers=4)
MEAN = np.array([0.8285, 0.6046, 0.7576], dtype=np.float32)
STD = np.array([0.3752, 0.5012, 0.4369], dtype=np.float32)
STATS = (MEAN, STD)


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(5)
    n_l = 3
    mu = rng.normal(1.0, 0.05, (25, 11, grids.NUM_R, n_l, 25, 3)).astype(np.float32)
    mu0 = rng.normal(1.0, 0.05, (n_l, 25, 3)).astype(np.float32)
    return R.flatten_table(mu, mu0)


def _x(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 25, 20, 6, generator=g)
    x[..., 3:] = 1.0
    return x


def _model(table=None, **flags):
    torch.manual_seed(0)
    return PCSTGNN(EI, table, STATS, **KW, **flags)


# ---------------------------------------------------------------- reduction

def test_all_switches_off_is_exactly_rgsage(table):
    """S4.1's premise, and the reason PCSTGNN subclasses rather than reimplements."""
    torch.manual_seed(0)
    base = RGSAGE(EI, "max", num_layers=4).eval()
    arm = _model(table).eval()
    assert torch.equal(base(_x()), arm(_x()))


def test_fusion_at_beta_zero_is_also_exactly_rgsage(table):
    """beta is initialised to 0, so the fused model *starts* at the baseline."""
    torch.manual_seed(0)
    base = RGSAGE(EI, "max", num_layers=4).eval()
    arm = _model(table, use_fusion=True).eval()
    assert float(arm.beta) == 0.0
    assert torch.equal(base(_x()), arm(_x()))


def test_the_physics_path_costs_one_scalar(table):
    """No capacity confound: R5 vs R0 differs by a single parameter."""
    n_base = sum(p.numel() for p in RGSAGE(EI, "max", num_layers=4).parameters())
    n_fused = sum(p.numel() for p in _model(table, use_fusion=True).parameters())
    assert n_fused == n_base + 1


def test_fusion_changes_the_output_once_beta_is_nonzero(table):
    """Guards the silent no-op."""
    arm = _model(table, use_fusion=True).eval()
    before = arm(_x())
    with torch.no_grad():
        arm.beta.fill_(1e-4)
    assert not torch.equal(before, arm(_x()))


# ---------------------------------------------------------------- physics

def test_fusion_is_per_node_broadcast(table):
    """A product of experts per node: the same rho is subtracted at every node.

    If this ever becomes a graph-level correction, the per-node cross-entropy
    and the soft vote no longer see the prior and ``01_SPEC.md`` §6 is wrong.
    """
    arm = _model(table, use_fusion=True).eval()
    x = _x()
    with torch.no_grad():
        plain = RGSAGE.forward(arm, x)
        arm.beta.fill_(0.7)
        fused = arm(x)
        delta = plain - fused                       # should be beta * rho, per node
        rho, _, _, _ = arm._rho(x)
    for node in range(25):
        torch.testing.assert_close(delta[:, node], 0.7 * rho, rtol=1e-4, atol=1e-3)


def test_table_and_stats_are_buffers_not_parameters(table):
    """The operator is fixed: it must not appear in ``parameters()`` at all."""
    arm = _model(table, use_fusion=True)
    names = {n for n, _ in arm.named_parameters()}
    assert "phys_table" not in names and "norm_mean" not in names
    buffers = dict(arm.named_buffers())
    for key in ("phys_table", "norm_mean", "norm_std", "sigma_eps", "shuffle_map"):
        assert key in buffers
        assert not buffers[key].requires_grad


def test_shuffle_map_is_a_permutation_that_fixes_the_no_fault_class(table):
    arm = _model(table, use_fusion=True, shuffle_seed=7)
    perm = arm.shuffle_map
    assert sorted(perm.tolist()) == list(range(26))
    assert int(perm[0]) == 0                        # class 0 keeps its own signature
    assert not torch.equal(perm[1:], torch.arange(1, 26))


def test_shuffling_changes_the_residual_but_not_its_statistics(table):
    """R6's whole point: same operator, same numbers, wrong locations."""
    honest = _model(table, use_fusion=True).eval()
    shuffled = _model(table, use_fusion=True, shuffle_seed=7).eval()
    x = _x()
    rho_honest, _, _, _ = honest._rho(x)
    rho_shuffled, _, _, _ = shuffled._rho(x)
    assert not torch.allclose(rho_honest, rho_shuffled)
    # Each row is a permutation of the other, so every summary statistic matches.
    torch.testing.assert_close(rho_honest.sort(dim=-1).values,
                               rho_shuffled.sort(dim=-1).values)


def test_forward_pc_returns_a_normalised_p_graph(table):
    out = _model(table, use_fusion=True).eval().forward_pc(_x())
    torch.testing.assert_close(out["p_graph"].sum(dim=-1), torch.ones(4))
    assert out["rho_A"].shape == (4, 26)
    assert out["hyp_star"].shape == (4, 26)
    assert out["hyp_star"].max() < grids.num_hypotheses(3)


def test_rho_d_is_scored_on_held_out_sensors(table):
    arm = _model(table, use_fusion=True).eval()
    x_clean = _x()
    x = x_clean.clone()
    x[:, 5] = 0.0                                   # node 5 dropped, mask included
    drop = torch.zeros(4, 25, 3)
    drop[:, 5] = 1.0
    out = arm.forward_pc(x, x_clean=x_clean, drop_mask=drop)
    assert out["rho_D"].shape == (4, 26)
    assert (out["rho_D"] >= 0).all()


# ---------------------------------------------------------------- gradients (S4.2)

def test_beta_receives_a_gradient_and_the_table_does_not(table):
    arm = _model(table, use_fusion=True)
    with torch.no_grad():
        arm.beta.fill_(0.3)
    out = arm.forward_pc(_x())
    loss = pc_loss(out, torch.tensor([1, 2, 3, 0]))["total"]
    loss.backward()
    assert arm.beta.grad is not None and float(arm.beta.grad.abs()) > 0
    assert arm.phys_table.grad is None
    assert arm.sigma_eps.grad is None


def test_lambda1_reaches_the_network_weights(table):
    """The term is differentiable *through P(f|X)*, not through rho."""
    arm = _model(table)
    out = arm.forward_pc(_x())
    ce_only = pc_loss(out, torch.tensor([1, 2, 3, 0]), lam1=0.0)["total"]
    grad_ce = torch.autograd.grad(ce_only, arm.classifier.linear.weight,
                                  retain_graph=True)[0]
    out = arm.forward_pc(_x())
    with_l1 = pc_loss(out, torch.tensor([1, 2, 3, 0]), lam1=1.0)["total"]
    grad_l1 = torch.autograd.grad(with_l1, arm.classifier.linear.weight)[0]
    assert not torch.allclose(grad_ce, grad_l1)


def test_lambda2_reaches_the_network_weights(table):
    """If this fails, R4 is silently identical to R3 -- the bug S4.2 exists for."""
    arm = _model(table)
    x_clean = _x()
    x = x_clean.clone()
    x[:, 5] = 0.0
    drop = torch.zeros(4, 25, 3)
    drop[:, 5] = 1.0
    y = torch.tensor([1, 2, 3, 0])
    out = arm.forward_pc(x, x_clean=x_clean, drop_mask=drop)
    base = torch.autograd.grad(pc_loss(out, y)["total"],
                               arm.classifier.linear.weight, retain_graph=True)[0]
    out = arm.forward_pc(x, x_clean=x_clean, drop_mask=drop)
    with_l2 = torch.autograd.grad(pc_loss(out, y, lam2=1.0)["total"],
                                  arm.classifier.linear.weight)[0]
    assert not torch.allclose(base, with_l2)


def test_loss_mask_excludes_the_unlabelable_windows(table):
    ids = ["13_AG_1p0_0000_w0019", "13_AG_1p0_0000_w0020",
           "13_AG_1p0_0000_w0021", "13_AG_1p0_0000_w0039"]
    mask = unlabelable_window_mask(ids)
    assert mask.tolist() == [1.0, 0.0, 1.0, 1.0]
    arm = _model(table)
    out = arm.forward_pc(_x())
    y = torch.tensor([1, 2, 3, 0])
    masked = pc_loss(out, y, lam1=1.0, loss_mask=mask)["l1"]
    unmasked = pc_loss(out, y, lam1=1.0)["l1"]
    assert not torch.isclose(masked, unmasked)


def test_ablation_heads_only_exist_when_switched_on(table):
    plain = _model(table)
    assert not hasattr(plain, "type_head") and not hasattr(plain, "decoder")
    rich = _model(table, use_type_head=True, use_decoder=True)
    out = rich.forward_pc(_x())
    assert out["type_logits"].shape == (4, 25, 11)
    assert out["recon"].shape == (4, 25, 3)


def test_physics_switches_require_a_table():
    with pytest.raises(ValueError):
        PCSTGNN(EI, None, STATS, use_fusion=True, **KW)
