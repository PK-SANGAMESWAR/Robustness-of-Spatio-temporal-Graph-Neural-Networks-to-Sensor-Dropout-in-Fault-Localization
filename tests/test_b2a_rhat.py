"""B2a (learned R-hat head): the five null tests and the loss checks. CPU only, synthetic table.

N1  switch off / head ignored: the backbone is exactly PCSTGNN's (state dict, outputs).
N2  a slice that holds the whole R axis gives the profiled rho of ``physics.residual``.
N3  a window built from table entry (f, tau, R_g, L): rho at class f on the slice R_g is exactly 0
    and equals the profiled value (R-hat = true R recovers profiling).
N4  no gradient reaches the table; with weight 0 the head gets no gradient; rho carries no graph.
N5  the R-hat loss drops no-fault windows and w = 20 windows, and uses log10 R.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fault_localization.models.common import unlabelable_window_mask
from fault_localization.models.pc_stgnn import PCSTGNN
from fault_localization.models.pc_stgnn_rhat import (
    PCSTGNNRHat, b2a_batch_loss, b2a_loss, rhat_loss, rhat_targets_from_metadata)
from fault_localization.physics import grids, residual as R
from fault_localization.physics.residual_sliced import (
    raw_residual_sliced, rho_sliced, slice_table, slice_to_full_index)

torch.set_num_threads(2)
EI = np.load("dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
KW = dict(aggregator="max", num_layers=2)
STATS = (np.array([0.8285, 0.6046, 0.7576], dtype=np.float32), np.array([0.3752, 0.5012, 0.4369], dtype=np.float32))
N_L = 3
N_R = grids.NUM_R


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(5)
    mu = rng.normal(1.0, 0.05, (25, 11, N_R, N_L, 25, 3)).astype(np.float32)
    mu0 = rng.normal(1.0, 0.05, (N_L, 25, 3)).astype(np.float32)
    return R.flatten_table(mu, mu0)


def _x(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 25, 20, 6, generator=g)
    x[..., 3:] = 1.0
    return x


def _arm(cls, table, **kw):
    torch.manual_seed(0)
    return cls(EI, table, STATS, **KW, **kw)


# ----------------------------------------------------------------------------- N1
def test_N1_head_off_is_exactly_pcstgnn(table):
    base = _arm(PCSTGNN, table, use_fusion=True)
    off = _arm(PCSTGNNRHat, table, use_fusion=True, use_rhat_head=False)
    assert set(off.state_dict()) == set(base.state_dict())
    for k, v in base.state_dict().items():
        assert torch.equal(v, off.state_dict()[k]), k


def test_N1_head_on_adds_only_head_keys_and_keeps_backbone_outputs(table):
    base = _arm(PCSTGNN, table, use_fusion=True).eval()
    on = _arm(PCSTGNNRHat, table, use_fusion=True, use_rhat_head=True).eval()
    extra = set(on.state_dict()) - set(base.state_dict())
    assert extra == {"rhat_head.weight", "rhat_head.bias"}
    for k, v in base.state_dict().items():
        assert torch.equal(v, on.state_dict()[k]), k            # RNG stream not consumed by the head
    x = _x()
    assert torch.equal(base.network_logits(x), on.network_logits(x))


# ----------------------------------------------------------------------------- N2
def test_N2_full_R_axis_slice_equals_profiled_rho(table):
    """Hand the sliced code the whole table for every window: rho and the profiled index must agree."""
    m = _arm(PCSTGNN, table, use_fusion=True).eval()
    x = _x(3)
    rho_ref, hyp_ref, s, a = m._rho(x)
    n_f, h, c = table.shape
    table_b = table[None].expand(3, n_f, h, c)
    r, hyp = raw_residual_sliced(s, a, table_b, m.sigma_eps, 0.0)
    rho = (r - r.min(-1, keepdim=True).values) * 0.5
    assert torch.allclose(rho, rho_ref, atol=1e-5, rtol=1e-6)
    assert torch.equal(hyp, hyp_ref)


# ----------------------------------------------------------------------------- N3
@pytest.mark.parametrize("r_g", [3, 9, 17])                     # includes 4.8-ish, 1.0 Ohm, 30 Ohm grid points
def test_N3_window_from_table_entry_has_zero_rho_at_true_class(table, r_g):
    f, tau, l_idx = 6, 4, 1
    flat = tau * (N_R * N_L) + r_g * N_L + l_idx
    s = table[f, flat].reshape(1, 25, 3).clone()
    a = torch.ones(1, 25, 3)
    r_idx = torch.tensor([r_g])
    rho, hyp_full = rho_sliced(s, a, table, r_idx, N_R, 1e-3, 0.0)
    assert float(rho[0, f]) == pytest.approx(0.0, abs=1e-9)
    assert int(hyp_full[0, f]) == flat                           # maps back to the exact (tau, R, L) entry
    rho_p, hyp_p = R.rho(s, a, table, 1e-3, 0.0)                 # profiled
    assert float(rho_p[0, f]) == pytest.approx(0.0, abs=1e-9)
    assert int(hyp_p[0, f]) == flat
    # R-hat = true R recovers profiling at the true class; a wrong R slice does not
    wrong, _ = rho_sliced(s, a, table, torch.tensor([(r_g + 5) % N_R]), N_R, 1e-3, 0.0)
    assert float(wrong[0, f]) > 1.0


def test_slice_table_and_index_round_trip(table):
    r_idx = torch.tensor([0, 7, 17])
    sl = slice_table(table, r_idx, N_R)
    assert sl.shape == (3, 26, 11 * N_L, 75)
    hyp_slice = torch.tensor([[0, 5, 32]] * 3)
    full = slice_to_full_index(hyp_slice, r_idx, N_R, N_L)
    for b in range(3):
        for j in range(3):
            assert torch.equal(sl[b, 0, int(hyp_slice[b, j])], table[0, int(full[b, j])])


# ----------------------------------------------------------------------------- N4
def test_N4_no_gradient_to_table_rho_has_no_graph_head_zero_grad_at_weight_zero(table):
    m = _arm(PCSTGNNRHat, table, use_fusion=True)
    m.train()
    m.phys_table.requires_grad_(True)                            # would collect a grad if anything flowed
    x = _x(4)
    y = torch.tensor([3, 0, 7, 11])
    r_true = torch.tensor([1.0, 1.0, 10.0, 0.1])
    out = m.forward_pc(x, r_true=r_true)
    assert not out["rho_A"].requires_grad
    meta = [{"resistance": float(r)} for r in r_true]
    target, valid = rhat_targets_from_metadata(y, meta)
    parts = b2a_loss(out, y, rhat_target=target, rhat_valid=valid, rhat_weight=0.0, aux_net_weight=1.0)
    parts["total"].backward()
    assert m.phys_table.grad is None
    assert m.rhat_head.weight.grad is None or float(m.rhat_head.weight.grad.abs().sum()) == 0.0
    # with weight 1 the head learns, and only from its own term
    m.zero_grad()
    out = m.forward_pc(x, r_true=r_true)
    parts = b2a_loss(out, y, rhat_target=target, rhat_valid=valid, rhat_weight=1.0)
    parts["total"].backward()
    assert float(m.rhat_head.weight.grad.abs().sum()) > 0.0
    assert m.phys_table.grad is None


def test_training_uses_true_R_and_eval_uses_rhat(table):
    m = _arm(PCSTGNNRHat, table, use_fusion=True)
    x = _x(2)
    r_true = torch.tensor([10.0, 0.1])
    m.train()
    out_t = m.forward_pc(x, r_true=r_true)
    assert out_t["r_idx"].tolist() == [int(np.argmin(np.abs(np.log10(grids.R_GRID) - np.log10(r)))) for r in (10.0, 0.1)]
    with pytest.raises(ValueError):
        m.forward_pc(x)                                          # training needs the true R
    m.eval()
    out_e = m.forward_pc(x)                                      # eval: R-hat, no true R needed
    assert torch.equal(out_e["r_idx"], m.snap_index(out_e["rhat_log10"].detach()))
    with pytest.raises(ValueError):
        m.forward_pc(x, rho_a_override=torch.zeros(2, 26))       # a cached profiled rho is refused


def test_scorer_path_uses_rhat_slice(table):
    m = _arm(PCSTGNNRHat, table, use_fusion=True).eval()
    x = _x(3)
    rho, hyp, _, _ = m._rho(x)
    idx = m.snap_index(m.rhat_from_hidden(m._hidden(x)))
    rho2, _, _, _ = m._rho_at(x, idx)
    assert torch.equal(rho, rho2)
    assert m(x).shape == (3, 25, 26)


# ----------------------------------------------------------------------------- N5
def test_N5_rhat_loss_masks_no_fault_and_w20_and_uses_log10():
    rhat = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    tgt = torch.tensor([0.0, 0.0, 1.0, 1.0])
    valid = torch.tensor([False, True, True, True])             # first window: no fault
    wids = ["e_w0021", "e_w0022", "e_w0020", "e_w0030"]          # third window: w = 20
    mask = unlabelable_window_mask(wids)
    assert mask.tolist() == [1, 1, 0, 1]
    loss = rhat_loss(rhat, tgt, valid, mask)
    assert float(loss) == pytest.approx(((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2)
    loss.backward()
    assert rhat.grad.tolist()[0] == 0.0 and rhat.grad.tolist()[2] == 0.0
    assert float(rhat_loss(rhat, tgt, torch.zeros(4, dtype=torch.bool))) == 0.0     # nothing valid -> 0


def test_targets_from_metadata_is_log10_and_masks_label_zero():
    y = torch.tensor([0, 5, 9])
    t, v = rhat_targets_from_metadata(y, [{"resistance": "1.0"}, {"resistance": "10.0"}, {"resistance": "0.1"}])
    assert torch.allclose(t, torch.tensor([0.0, 1.0, -1.0]))
    assert v.tolist() == [False, True, True]


def test_batch_loss_runs_with_R5_settings(table):
    m = _arm(PCSTGNNRHat, table, use_fusion=True)
    b = 4
    batch = {"x": _x(b), "x_clean": _x(b), "drop_mask": torch.zeros(b, 25, 3),
             "y": torch.tensor([3, 0, 7, 11]), "window_id": [f"e{i}_w00{21 + i}" for i in range(b)],
             "metadata": [{"resistance": r} for r in ("1.0", "1.0", "10.0", "0.1")]}
    parts = b2a_batch_loss(m, batch, torch.device("cpu"))
    assert {"ce", "l2", "aux", "rhat_mse", "total"} <= set(parts) and "l1" not in parts
    parts["total"].backward()
