"""
SDR-STGNN — Sensor-Dropout-Robust Spatio-Temporal GNN (this paper's novel
architectural contribution).

Motivation
----------
Under sensor dropout, a missing μPMU still occupies a node in the measured-only
graph. In the mask-agnostic baselines (``RGSAGE``/``RGCN``/``RGATv2``) that dead
node's zeroed input produces a garbage temporal embedding that is nonetheless
aggregated into its live neighbours, *corrupting* them. The availability mask
(input channels 3-5, which flag which sensors are physically present) is fed to
the network but nothing structural exploits it.

Contribution: availability-gated message passing
-------------------------------------------------
SDR-STGNN derives a per-node availability scalar ``a in {0,1}`` from the mask
channels and makes aggregation *availability-weighted* so that **dead sensors
emit no messages** and a dead node is represented purely by its live neighbours:

    h  = encoder(x)                         # [B,N,H]  shared GRU (unchanged)
    a  = (mask channels present)            # [B,N]
    for each hop:
        mean:  agg_j = sum_i a_i h_i / max(sum_i a_i, eps)   over N(j) u {j}
        max :  agg_j = max_{i : a_i=1} h_i                   (dead -> -inf; 0 if none)
        combined_j = concat(a_j * h_j, agg_j)   # self path also gated
        h = ReLU(Linear(combined_j))
    logits = classifier(h)

Key property (used as a correctness test and as a clean ablation): when every
sensor is present (``a == 1``) availability-weighted mean/max reduce exactly to
ordinary mean/max and the self-gate is the identity, so **SDR-STGNN at k=0 is
identical to standard RGSAGE**. The gates only change behaviour when sensors are
missing — and therefore only *learn* to matter under dropout-aware training,
making the architecture and the augmentation complementary rather than redundant.

The parameter layout (``encoder`` / ``spatial_layers`` / ``classifier``)
deliberately mirrors :class:`RGSAGE` so weights transfer 1:1 for the equivalence
test.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import NodeClassifier, TemporalEncoder

_EPS = 1e-6


def node_availability(x: torch.Tensor) -> torch.Tensor:
    """Per-node availability in {0,1} derived from mask channels 3-5.

    A dropped sensor has its availability-mask channels zeroed for every
    timestep (see ``fault_localization.dropout.masking``), so a node is
    "available" iff any mask channel is non-zero at any timestep.

    Parameters
    ----------
    x:
        Input tensor [B, N, S, F=6] (voltages 0-2, availability mask 3-5).

    Returns
    -------
    torch.Tensor
        Availability [B, N] with values in {0.0, 1.0}.
    """
    return (x[..., 3:].amax(dim=(-1, -2)) > 0).to(x.dtype)


class SDRSTGNN(nn.Module):
    """Sensor-dropout-robust STGNN: RGSAGE backbone + availability-gated aggregation.

    Parameters
    ----------
    aggregator:
        ``"mean"`` or ``"max"`` neighbourhood aggregator (default ``"max"``,
        matching the strongest baseline this model is built on).
    num_layers:
        Number of stacked gated spatial hops (frozen protocol uses 4).
    """

    def __init__(
        self,
        edge_index,
        aggregator: str = "max",
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
        # Input to hop i: concat(gated self, gated aggregate) = 2 * dims[i]
        dims = [temporal_hidden_dim] + [gnn_hidden_dim] * num_layers
        self.spatial_layers = nn.ModuleList(
            nn.Linear(dims[i] * 2, dims[i + 1]) for i in range(num_layers)
        )
        self.classifier = NodeClassifier(gnn_hidden_dim, num_classes, dropout)

    def _self_loops(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = self.edge_index
        loop = torch.arange(n, device=device)
        return torch.cat([src, loop]), torch.cat([dst, loop])

    def _gated_aggregate(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Availability-weighted neighbour aggregate (self-loops included).

        Parameters
        ----------
        h:
            Node features [B, N, d].
        a:
            Per-node availability [B, N] in {0, 1}.

        Returns
        -------
        torch.Tensor
            Aggregated features [B, N, d]. Messages from unavailable sources
            (a_i = 0) contribute nothing; a node with no available source in
            its closed neighbourhood aggregates to zeros.
        """
        b, n, d = h.shape
        src, dst = self._self_loops(n, h.device)

        h_src = h[:, src]                     # [B, E', d]
        a_src = a[:, src]                     # [B, E']

        if self.aggregator == "mean":
            weighted = a_src.unsqueeze(-1) * h_src           # [B, E', d]
            num = h.new_zeros((b, n, d)).index_add(1, dst, weighted)
            den = a.new_zeros((b, n)).index_add(1, dst, a_src)  # [B, N]
            return num / den.clamp(min=_EPS).unsqueeze(-1)
        else:  # max — dead sources masked out so they never win the max
            neg_inf = torch.finfo(h.dtype).min
            masked = torch.where(a_src.unsqueeze(-1) > 0, h_src,
                                 h.new_full((), neg_inf))
            dst_expanded = dst.view(1, -1, 1).expand(b, -1, d)
            out = h.new_full((b, n, d), neg_inf)
            out = out.scatter_reduce(1, dst_expanded, masked, reduce="amax",
                                     include_self=True)
            # Nodes with no available source (still -inf) fall back to zeros.
            return torch.where(out <= neg_inf, torch.zeros_like(out), out)

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        a = node_availability(x)              # [B, N]  (from mask channels)
        h = self.encoder(x)                   # [B, N, temporal_hidden_dim]
        a_col = a.unsqueeze(-1)               # [B, N, 1]
        for i, layer in enumerate(self.spatial_layers):
            agg = self._gated_aggregate(h, a)         # [B, N, d]
            combined = torch.cat([a_col * h, agg], dim=-1)  # gated self + agg
            h = layer(combined)
            if i < len(self.spatial_layers) - 1:
                h = torch.relu(h)
        return self.classifier(h)
