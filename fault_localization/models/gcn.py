from __future__ import annotations
import torch
from torch import nn
from .common import NodeClassifier, TemporalEncoder


class RGCN(nn.Module):
    """Recurrent GCN: GRU temporal encoding + GCN spatial layer.

    Input F=6 (3 voltage channels + 3 availability mask channels).
    Uses symmetric normalized adjacency: D^{-1/2} A D^{-1/2}.
    Self-loops are included in the aggregation.

    num_layers:
        Number of stacked spatial hops (diagnostic knob -- the frozen paper
        protocol uses 1). The degree-normalized aggregation weights depend
        only on graph structure, so they are computed once and reused across
        hops; a ReLU separates hops (not applied after the last one).
    """

    def __init__(
        self,
        edge_index,
        temporal_hidden_dim: int = 128,
        gnn_hidden_dim: int = 64,
        num_classes: int = 26,
        dropout: float = 0.35,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.encoder = TemporalEncoder(6, temporal_hidden_dim)
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        dims = [temporal_hidden_dim] + [gnn_hidden_dim] * num_layers
        self.spatial_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.classifier = NodeClassifier(gnn_hidden_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        h = self.encoder(x)              # [B, N, d]
        b, n, _ = h.shape
        src, dst = self.edge_index

        # Add self-loops
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        # Symmetric normalized weights: w_ij = 1 / sqrt(deg_i * deg_j).
        # Depends only on graph structure, so computed once and reused
        # across every hop.
        deg = torch.bincount(dst, minlength=n).to(h.dtype)
        w = (deg[src] * deg[dst]).rsqrt()

        for i, layer in enumerate(self.spatial_layers):
            d = h.shape[-1]
            out = h.new_zeros((b, n, d))
            out.index_add_(1, dst, h[:, src] * w[None, :, None])
            h = layer(out)
            if i < len(self.spatial_layers) - 1:
                h = torch.relu(h)

        return self.classifier(h)
