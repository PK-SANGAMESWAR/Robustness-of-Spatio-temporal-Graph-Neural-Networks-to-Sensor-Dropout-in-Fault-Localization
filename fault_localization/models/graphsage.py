from __future__ import annotations
import torch
from torch import nn
from .common import NodeClassifier, TemporalEncoder


class RGSAGE(nn.Module):
    """Recurrent GraphSAGE: GRU temporal encoding + SAGE spatial layer.

    Implements Eq. (5) from [BASE]: concatenation of self and neighbor aggregate,
    then a linear projection.  Self is included in the aggregate and the concat path.

    Input F=6 (3 voltage channels + 3 availability mask channels).

    Parameters
    ----------
    aggregator:
        ``"mean"`` or ``"max"`` neighborhood aggregator.
    num_layers:
        Number of stacked spatial hops (diagnostic knob — the frozen paper
        protocol uses 1; >1 lets a node's receptive field reach beyond its
        immediate μPMU neighbors, at the cost of a ReLU between hops).
    """

    def __init__(
        self,
        edge_index,
        aggregator: str = "mean",
        temporal_hidden_dim: int = 128,
        gnn_hidden_dim: int = 64,
        num_classes: int = 26,
        dropout: float = 0.35,
        num_layers: int = 1,
    ):
        super().__init__()
        if aggregator not in {"mean", "max"}:
            raise ValueError("aggregator must be 'mean' or 'max'")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.encoder = TemporalEncoder(6, temporal_hidden_dim)
        self.aggregator = aggregator
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        # Input to hop i: concat(h_self, agg(neighbors)) = 2 * dims[i]
        dims = [temporal_hidden_dim] + [gnn_hidden_dim] * num_layers
        self.spatial_layers = nn.ModuleList(
            nn.Linear(dims[i] * 2, dims[i + 1]) for i in range(num_layers)
        )
        self.classifier = NodeClassifier(gnn_hidden_dim, num_classes, dropout)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor features for each node.

        Parameters
        ----------
        x:
            Node features [B, N, d].

        Returns
        -------
        torch.Tensor
            Aggregated features [B, N, d].  Self-loops are included so every
            node aggregates its own features along with neighbors.
        """
        b, n, d = x.shape
        src, dst = self.edge_index

        # Add self-loops
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        if self.aggregator == "mean":
            out = x.new_zeros((b, n, d))
            out = out.index_add(1, dst, x[:, src])
            counts = torch.bincount(dst, minlength=n).to(x.dtype)
            return out / counts[None, :, None].clamp(min=1)
        else:  # max
            gathered = x[:, src]  # [B, E, d]
            dst_expanded = dst.view(1, -1, 1).expand(b, -1, d)
            out = x.new_full((b, n, d), float("-inf"))
            out = out.scatter_reduce(1, dst_expanded, gathered, reduce="amax", include_self=True)
            # Replace -inf for isolated nodes (shouldn't happen with self-loops)
            out = torch.where(out.isinf(), torch.zeros_like(out), out)
            return out

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        h = self.encoder(x)  # [B, N, d]
        for i, layer in enumerate(self.spatial_layers):
            agg = self.aggregate(h)  # [B, N, d]
            combined = torch.cat([h, agg], dim=-1)  # [B, N, 2d]
            h = layer(combined)
            if i < len(self.spatial_layers) - 1:
                h = torch.relu(h)
        return self.classifier(h)
