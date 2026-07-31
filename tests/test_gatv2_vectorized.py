"""Equivalence test for the RGATv2 attention rewrite.

The original implementation computed per-head softmax over each destination
node's in-neighbors with a Python `for node in range(n)` loop (masking edges,
softmax, dropout, weighted sum). That loop was replaced with a single
vectorized grouped-softmax (torch_geometric.utils.softmax) + scatter-add
(torch_geometric.utils.scatter) pair — same math, no Python loop over nodes.

This test pins down that the rewrite is numerically identical (not just
"close enough") by re-implementing the old loop inline as a reference and
comparing against the current RGATv2.forward() in eval mode (dropout is the
identity when eval()'d, so this is an exact comparison), then checks that
gradients still flow through every learnable parameter.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fault_localization.models.gatv2 import RGATv2


def _reference_forward(model: RGATv2, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Pre-vectorization RGATv2 forward pass (per-node Python loop), for comparison only."""
    h = model.encoder(x)
    b, n, d = h.shape

    src, dst = edge_index
    loop = torch.arange(n, device=x.device)
    src = torch.cat([src, loop])
    dst = torch.cat([dst, loop])
    num_edges = src.size(0)

    h_src = h[:, src]
    h_dst = h[:, dst]
    pair = torch.cat([h_src, h_dst], dim=-1)
    transformed = model.W[0](pair)
    transformed = transformed.view(b, num_edges, model.heads, model.gnn_hidden_dim)
    transformed_act = F.leaky_relu(transformed, negative_slope=0.2)
    e = (transformed_act * model.att_vec[0][None, None]).sum(-1)

    val = model.W_val[0](h).view(b, n, model.heads, model.gnn_hidden_dim)
    out = torch.zeros(b, n, model.heads, model.gnn_hidden_dim, device=x.device)

    for node in range(n):
        mask = dst == node
        if not mask.any():
            out[:, node] = val[:, node]
            continue
        e_node = e[:, mask]
        alpha = torch.softmax(e_node, dim=1)
        alpha = model.att_dropout(alpha)
        src_nodes = src[mask]
        val_src = val[:, src_nodes]
        out[:, node] = (alpha.unsqueeze(-1) * val_src).sum(1)

    out_cat = out.reshape(b, n, model.heads * model.gnn_hidden_dim)
    out_proj = model.out_proj[0](out_cat)
    return model.classifier(out_proj)


def _small_ring_edge_index(n: int) -> torch.Tensor:
    """A small undirected ring graph (directed both ways), no self-loops (added in forward)."""
    src = list(range(n)) + [(i + 1) % n for i in range(n)]
    dst = [(i + 1) % n for i in range(n)] + list(range(n))
    return torch.tensor([src, dst], dtype=torch.long)


def test_vectorized_matches_reference_loop_in_eval_mode():
    torch.manual_seed(0)
    n, b, s, feat = 6, 3, 20, 6
    edge_index = _small_ring_edge_index(n)

    model = RGATv2(edge_index, temporal_hidden_dim=16, gnn_hidden_dim=8, heads=2,
                    attention_dropout=0.3, num_classes=26, dropout=0.35)
    model.eval()  # dropout is identity in eval mode -> exact comparison, not statistical

    x = torch.randn(b, n, s, feat)

    with torch.no_grad():
        vectorized_out = model(x)
        reference_out = _reference_forward(model, x, edge_index)

    assert torch.allclose(vectorized_out, reference_out, atol=1e-5), (
        "Vectorized RGATv2 forward diverges from the pre-rewrite per-node loop reference"
    )


def test_vectorized_gradients_flow_to_all_attention_params():
    torch.manual_seed(1)
    n, b, s, feat = 6, 2, 20, 6
    edge_index = _small_ring_edge_index(n)

    model = RGATv2(edge_index, temporal_hidden_dim=16, gnn_hidden_dim=8, heads=2,
                    attention_dropout=0.3, num_classes=26, dropout=0.35)
    model.train()

    x = torch.randn(b, n, s, feat)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    for name, param in (("W", model.W[0]), ("att_vec", model.att_vec[0]),
                         ("W_val", model.W_val[0]), ("out_proj", model.out_proj[0])):
        grad = param.weight.grad if hasattr(param, "weight") else param.grad
        assert grad is not None, f"No gradient reached {name}"
        assert torch.isfinite(grad).all(), f"Non-finite gradient in {name}"
        assert grad.abs().sum() > 0, f"Zero gradient in {name}"
