"""Tests for SDR-STGNN — the availability-gated, sensor-dropout-robust model.

Two guarantees are pinned down here:

1. **k=0 equivalence.** When every sensor is present (all availability-mask
   channels = 1), availability-weighted mean/max reduce to ordinary mean/max and
   the self-gate is the identity, so SDR-STGNN must be numerically identical to
   the standard RGSAGE backbone it is built on (same weights). This is the
   property that makes the k=0 column a clean ablation: any k=0 difference in the
   experiments is attributable to dropout-aware training, never to the gating.

2. **Robustness under dropped sensors.** With some (or all) nodes masked out, the
   forward pass stays finite and correctly shaped, and gradients still flow to
   the spatial layers so dropout-aware training can actually learn the gating.
"""
from __future__ import annotations

import torch

from fault_localization.models.graphsage import RGSAGE
from fault_localization.models.sdr_stgnn import SDRSTGNN, node_availability


def _small_ring_edge_index(n: int) -> torch.Tensor:
    """A small undirected ring (directed both ways); self-loops added in forward."""
    src = list(range(n)) + [(i + 1) % n for i in range(n)]
    dst = [(i + 1) % n for i in range(n)] + list(range(n))
    return torch.tensor([src, dst], dtype=torch.long)


def _all_available_input(b: int, n: int, s: int) -> torch.Tensor:
    """[B,N,S,6] with random voltages (0-2) and all availability masks (3-5) = 1."""
    x = torch.randn(b, n, s, 6)
    x[..., 3:] = 1.0
    return x


def _make_pair(aggregator: str, n: int, num_layers: int):
    edge_index = _small_ring_edge_index(n)
    rgsage = RGSAGE(edge_index, aggregator, temporal_hidden_dim=16,
                    gnn_hidden_dim=8, num_classes=26, dropout=0.35,
                    num_layers=num_layers)
    sdr = SDRSTGNN(edge_index, aggregator=aggregator, temporal_hidden_dim=16,
                   gnn_hidden_dim=8, num_classes=26, dropout=0.35,
                   num_layers=num_layers)
    # Identical parameter layout (encoder / spatial_layers / classifier) → weights
    # transfer 1:1 so the two forwards are directly comparable.
    sdr.load_state_dict(rgsage.state_dict())
    return rgsage.eval(), sdr.eval(), edge_index


def test_node_availability_from_mask_channels():
    x = torch.randn(2, 5, 20, 6)
    x[..., 3:] = 1.0
    x[0, 2, :, 3:] = 0.0   # drop node 2 in sample 0 (all mask channels zeroed)
    x[1, 4, :, 3:] = 0.0   # drop node 4 in sample 1
    a = node_availability(x)
    assert a.shape == (2, 5)
    assert a[0, 2] == 0.0 and a[1, 4] == 0.0
    # every other node available
    assert a.sum().item() == 2 * 5 - 2


def test_k0_equivalence_to_rgsage_max():
    torch.manual_seed(0)
    rgsage, sdr, _ = _make_pair("max", n=6, num_layers=4)
    x = _all_available_input(b=3, n=6, s=20)
    with torch.no_grad():
        assert torch.allclose(sdr(x), rgsage(x), atol=1e-5), (
            "SDR-STGNN (max) at k=0 diverges from standard RGSAGE-Max"
        )


def test_k0_equivalence_to_rgsage_mean():
    torch.manual_seed(1)
    rgsage, sdr, _ = _make_pair("mean", n=6, num_layers=4)
    x = _all_available_input(b=3, n=6, s=20)
    with torch.no_grad():
        assert torch.allclose(sdr(x), rgsage(x), atol=1e-5), (
            "SDR-STGNN (mean) at k=0 diverges from standard RGSAGE-Mean"
        )


def test_finite_output_with_dropped_nodes():
    torch.manual_seed(2)
    for aggregator in ("max", "mean"):
        _, sdr, _ = _make_pair(aggregator, n=6, num_layers=4)
        x = _all_available_input(b=4, n=6, s=20)
        # Drop nodes 1 and 3 in every sample (zero voltages + mask, as masking.py does).
        x[:, [1, 3], :, :] = 0.0
        with torch.no_grad():
            out = sdr(x)
        assert out.shape == (4, 6, 26)
        assert torch.isfinite(out).all(), f"non-finite output ({aggregator}) under dropout"


def test_finite_output_when_all_nodes_dropped():
    """Degenerate case: no live sensor anywhere → aggregate falls back to zeros, no NaN."""
    torch.manual_seed(3)
    for aggregator in ("max", "mean"):
        _, sdr, _ = _make_pair(aggregator, n=6, num_layers=4)
        x = torch.zeros(2, 6, 20, 6)  # everything dropped: a ≡ 0
        assert node_availability(x).sum() == 0
        with torch.no_grad():
            out = sdr(x)
        assert torch.isfinite(out).all(), f"non-finite output ({aggregator}) all-dropped"


def test_gradients_flow_to_spatial_layers():
    torch.manual_seed(4)
    _, sdr, _ = _make_pair("max", n=6, num_layers=4)
    sdr.train()
    x = _all_available_input(b=4, n=6, s=20)
    x[:, [2], :, :] = 0.0  # one dropped node so the gates are exercised
    loss = sdr(x).sum()
    loss.backward()
    for i, layer in enumerate(sdr.spatial_layers):
        g = layer.weight.grad
        assert g is not None and torch.isfinite(g).all(), f"bad grad at spatial_layers[{i}]"
        assert g.abs().sum() > 0, f"zero grad at spatial_layers[{i}]"
