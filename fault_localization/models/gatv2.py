"""
RGATv2 — Recurrent Graph Attention Network v2.

Implements the attention mechanism from Brody et al. (2022) GATv2 as
described in [BASE] Equation (8):

  For each directed edge (i → j):
    e_ij = a^T · LeakyReLU( W · [h_i || h_j] )   ... per head
    α_ij = softmax_j( e_ij )                        ... per head, normalized over neighbors of j
    h'_j = Σ_{i ∈ N(j)} α_ij · (W_val · h_i)      ... per head
    output_j = concat_heads( h'_j )  → projection

KEY CORRECTIONS vs. previous implementation:
  1. W is a SHARED transform applied to the CONCATENATED pair [h_i || h_j]
     BEFORE LeakyReLU and scoring — not separate projections concatenated.
  2. attention_dropout is applied to the normalized coefficients α_ij,
     NOT to the node features z.
  3. LeakyReLU is applied BEFORE the dot product with attention vector a,
     consistent with the GATv2 formulation.

KNOWN_ISSUES compliance:
  - All models (including this one) are trained on identical data/splits.
  - Internal fairness is the priority, not matching the original paper's
    absolute numbers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter, softmax

from .common import NodeClassifier, TemporalEncoder


class RGATv2(nn.Module):
    """Recurrent GATv2 for node-level fault classification.

    Parameters
    ----------
    edge_index:
        Graph connectivity, shape [2, E_directed].
    temporal_hidden_dim:
        GRU hidden size (paper: 128).
    gnn_hidden_dim:
        GNN hidden size per head (paper: 64).
    heads:
        Number of attention heads (paper: 4).
    attention_dropout:
        Dropout on normalized attention coefficients α_ij (paper: 0.3).
    num_classes:
        Number of output classes (paper: 26).
    dropout:
        NodeClassifier dropout rate (paper: 0.35).
    num_layers:
        Number of stacked attention hops (diagnostic knob -- the frozen
        paper protocol uses 1). Each hop gets its own W/att_vec/W_val/
        out_proj; a ReLU separates hops (not applied after the last one).
    """

    def __init__(
        self,
        edge_index,
        temporal_hidden_dim: int = 128,
        gnn_hidden_dim: int = 64,
        heads: int = 4,
        attention_dropout: float = 0.3,
        num_classes: int = 26,
        dropout: float = 0.35,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.encoder = TemporalEncoder(6, temporal_hidden_dim)
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))

        self.heads = heads
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_layers = num_layers

        # Layer i's input dim is temporal_hidden_dim for i=0, else gnn_hidden_dim
        # (the previous hop's out_proj output).
        dims = [temporal_hidden_dim] + [gnn_hidden_dim] * num_layers

        # W: shared transform of concatenated node-pair features → per-head representation
        # Input: 2 * dims[i] (concat [h_i || h_j]); Output: heads * gnn_hidden_dim
        self.W = nn.ModuleList(
            nn.Linear(2 * dims[i], heads * gnn_hidden_dim, bias=False) for i in range(num_layers)
        )

        # Attention vector a: one scalar per head (dot product with W's output)
        self.att_vec = nn.ParameterList(
            nn.Parameter(torch.empty(heads, gnn_hidden_dim)) for _ in range(num_layers)
        )
        for p in self.att_vec:
            nn.init.xavier_uniform_(p.unsqueeze(0))

        # Value projection: project h_i to per-head value space
        # (separate from W so value is not tied to attention computation)
        self.W_val = nn.ModuleList(
            nn.Linear(dims[i], heads * gnn_hidden_dim, bias=False) for i in range(num_layers)
        )

        self.att_dropout = nn.Dropout(attention_dropout)

        # Output projection: concat of all heads → gnn_hidden_dim
        self.out_proj = nn.ModuleList(
            nn.Linear(heads * gnn_hidden_dim, gnn_hidden_dim, bias=False) for _ in range(num_layers)
        )

        self.classifier = NodeClassifier(gnn_hidden_dim, num_classes, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape [B, N, S, F=6].
        edge_index:
            Optional override for graph connectivity [2, E].

        Returns
        -------
        torch.Tensor
            Node logits of shape [B, N, num_classes].
        """
        if edge_index is None:
            edge_index = self.edge_index

        # 1. Temporal encoding: [B, N, S, F] → [B, N, temporal_hidden_dim]
        h = self.encoder(x)
        b, n, _ = h.shape

        # Add self-loops so every node aggregates its own features
        src, dst = edge_index
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])   # [E + N]
        dst = torch.cat([dst, loop])   # [E + N]
        num_edges = src.size(0)

        for i in range(self.num_layers):
            # 2. Compute attention scores per head
            # h_src, h_dst: [B, E+N, d]
            h_src = h[:, src]  # [B, E+N, d]
            h_dst = h[:, dst]  # [B, E+N, d]

            # Concatenate and apply shared W: [B, E+N, 2d] → [B, E+N, heads*gnn_dim]
            pair = torch.cat([h_src, h_dst], dim=-1)          # [B, E+N, 2d]
            transformed = self.W[i](pair)                      # [B, E+N, heads*gnn_dim]
            transformed = transformed.view(b, num_edges, self.heads, self.gnn_hidden_dim)
            # transformed: [B, E+N, H, D]

            # LeakyReLU before dot product with attention vector (GATv2 key property)
            transformed_act = F.leaky_relu(transformed, negative_slope=0.2)

            # Attention score: dot product with per-head vector a
            # att_vec: [H, D] → [1, 1, H, D]
            e = (transformed_act * self.att_vec[i][None, None]).sum(-1)  # [B, E+N, H]

            # 3. Per-head softmax over in-neighbors of each destination node,
            # vectorized across all nodes at once via a grouped (scatter) softmax
            # instead of a Python loop over the 25 nodes. `dst` (the grouping
            # index) is shared across the batch, so grouping only the edge axis
            # (dim=0 after transposing batch out of the way) reproduces the exact
            # per-(batch, head, node) softmax the old per-node loop computed.
            # Value projection: [B, N, H*D]
            val = self.W_val[i](h).view(b, n, self.heads, self.gnn_hidden_dim)  # [B, N, H, D]

            alpha = softmax(e.transpose(0, 1), index=dst, dim=0, num_nodes=n).transpose(0, 1)  # [B, E+N, H]

            # Dropout on the normalized attention coefficients (paper Eq. 8)
            alpha = self.att_dropout(alpha)

            val_src = val[:, src]  # [B, E+N, H, D]
            contrib = alpha.unsqueeze(-1) * val_src  # [B, E+N, H, D]
            out = scatter(contrib, dst, dim=1, dim_size=n, reduce="sum")  # [B, N, H, D]

            # 4. Multi-head aggregation: concat then project → [B, N, gnn_hidden_dim]
            out_cat = out.reshape(b, n, self.heads * self.gnn_hidden_dim)
            h = self.out_proj[i](out_cat)  # [B, N, gnn_hidden_dim]
            if i < self.num_layers - 1:
                h = torch.relu(h)

        # 5. Node classification
        return self.classifier(h)
