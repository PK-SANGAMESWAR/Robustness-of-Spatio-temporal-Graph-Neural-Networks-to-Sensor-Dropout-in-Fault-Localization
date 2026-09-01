"""FaultQuery-STGNN — a physics-biased query decoder for fault localization.

Motivation
----------
Every baseline in this study, and SDR-STGNN with it, predicts all 26 classes at
every μPMU node and then pools the 25 predictions by soft voting.  Three
measured weaknesses of this study follow from that decoder, not from the encoder:

1. **The model has no representation of what it predicts.**  19 of the 25
   candidate fault buses are not nodes of the measured-only graph, so a class is
   only ever a column of a per-sensor logit vector.
2. **Its receptive field is too small for the graph.**  The measured-only tree
   has diameter 10 while the frozen protocol stacks 4 hops, so only 55.8% of
   node pairs can exchange information at all.  Soft voting cannot repair that:
   it pools predictions, not evidence.
3. **Proximity is never made available.**  The targeted-dropout study showed the
   sensors nearest a fault carry its signature (losing them costs far more than
   losing the same number at random), but nothing tells the model which those
   are, because the compression onto instrumented buses discarded the geometry.

FaultQuery-STGNN keeps the temporal encoder and the local spatial block of
RGSAGE-Max unchanged and replaces the per-node head with a cross-attention
decoder over **26 learned fault-bus queries** (25 candidate buses + no-fault).
Each query attends over the 25 sensor embeddings, with three additions to the
attention logit::

    logit(q, i) = (W_Q q_q) . (W_K h_i) / sqrt(d_head)
                  - softplus(alpha_head) * D[q, i]      physics prior
                  + beta * log r_i                      reliability
                  (overwritten by MASK_VALUE if sensor i is unavailable)

``D`` is the hop distance from query ``q``'s fault bus to sensor ``i`` over the
**full 123-bus topology** (see :mod:`fault_localization.graph.fault_distance`),
so the decoder is told the electrical geometry the graph construction threw
away.  ``alpha`` is learned per head, letting different heads settle on
different locality scales; it is initialized small so training starts close to
unbiased attention.  ``r_i`` is a learned per-sensor reliability in (0, 1),
included because measurement noise is the one stressor dropout-aware training
provably does not help against (noise leaves the availability mask at 1, so the
learned "all-zero means dark" response has nothing to fire on).

Relationship to SDR-STGNN
-------------------------
SDR-STGNN gated aggregation at *every hop*, which silenced a dropped node
permanently and blocked multi-hop routing through nodes whose information could
have been recovered from elsewhere; it matched but never beat dropout-aware
training.  Here availability masking is applied **only in the decoder**, and the
local spatial block is left ungated, so a dropped node can still be reconstructed
by message passing before the decoder decides how much to trust it.

Output contract
---------------
The decoder emits one logit per query, i.e. a single graph-level prediction, but
returns it shaped ``[B, 1, 26]``.  With ``N = 1`` the project's existing
:func:`~fault_localization.models.voting.soft_vote` is the identity and
:func:`~fault_localization.models.common.expanded_cross_entropy` is ordinary
cross-entropy, so this model is a drop-in for every existing training,
evaluation and dropout-study script with no changes to them.  Note that the
node-level diagnostic metric therefore degenerates to the graph metric for this
model and must be reported as N/A rather than as a per-window score.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import TemporalEncoder
from .sdr_stgnn import node_availability

# Finite stand-in for -inf in the attention logits.  Using a finite value keeps
# the softmax defined when *every* sensor is dropped (it degenerates to a
# uniform row) instead of producing NaN, while still underflowing to exactly
# zero weight in float32 whenever at least one sensor is available.
MASK_VALUE = -1e9

_EPS = 1e-6


class FaultQuerySTGNN(nn.Module):
    """GRU encoder + ungated local message passing + fault-query decoder.

    Parameters
    ----------
    edge_index:
        Connectivity ``[2, E]`` of the 25-node measured-only graph.
    hop_distance:
        ``[26, 25]`` fault-bus-to-sensor hop distances over the full 123-bus
        topology, from :func:`fault_localization.graph.fault_distance.
        load_fault_hop_distance`.  Row 0 (no-fault) is all zeros and therefore
        contributes a constant that cancels in the softmax.
    aggregator:
        ``"mean"`` or ``"max"`` local aggregator (default ``"max"``, matching
        the strongest baseline).
    num_heads:
        Cross-attention heads.  ``gnn_hidden_dim`` must be divisible by it.
    alpha_init:
        Initial value of the per-head distance weight, before the softplus
        reparameterisation that keeps it non-negative.
    use_distance_bias, use_availability_mask, use_reliability, use_local_gnn:
        Ablation switches, one per mechanism.  Each disables exactly one term so
        its contribution can be attributed; see
        ``scripts/run_fault_query_ablations.py``.
    """

    def __init__(
        self,
        edge_index,
        hop_distance,
        aggregator: str = "max",
        temporal_hidden_dim: int = 128,
        gnn_hidden_dim: int = 64,
        num_classes: int = 26,
        dropout: float = 0.35,
        num_layers: int = 1,
        num_heads: int = 4,
        alpha_init: float = 0.1,
        use_distance_bias: bool = True,
        use_availability_mask: bool = True,
        use_reliability: bool = True,
        use_local_gnn: bool = True,
    ):
        super().__init__()
        if aggregator not in {"mean", "max"}:
            raise ValueError("aggregator must be 'mean' or 'max'")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if gnn_hidden_dim % num_heads != 0:
            raise ValueError(
                f"gnn_hidden_dim ({gnn_hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.aggregator = aggregator
        self.num_heads = num_heads
        self.use_distance_bias = use_distance_bias
        self.use_availability_mask = use_availability_mask
        self.use_reliability = use_reliability
        self.use_local_gnn = use_local_gnn

        self.encoder = TemporalEncoder(6, temporal_hidden_dim)
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))

        distance = torch.as_tensor(hop_distance, dtype=torch.float32)
        if distance.shape != (num_classes, 25):
            raise ValueError(
                f"hop_distance must be [{num_classes}, 25], got {tuple(distance.shape)}"
            )
        self.register_buffer("hop_distance", distance)

        # Local spatial block.  Deliberately identical in structure to RGSAGE so
        # the decoder is the only thing that differs from the baseline backbone.
        if use_local_gnn:
            dims = [temporal_hidden_dim] + [gnn_hidden_dim] * num_layers
            self.spatial_layers = nn.ModuleList(
                nn.Linear(dims[i] * 2, dims[i + 1]) for i in range(num_layers)
            )
            sensor_dim = gnn_hidden_dim
        else:
            # Ablation A6: queries attend straight to the temporal embeddings.
            self.spatial_layers = nn.ModuleList()
            sensor_dim = temporal_hidden_dim

        d_model = gnn_hidden_dim
        self.d_head = d_model // num_heads
        self.queries = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(sensor_dim, d_model)
        self.w_v = nn.Linear(sensor_dim, d_model)

        # softplus(alpha_raw) >= 0 keeps the distance term a penalty, never a
        # reward for being far from the fault.
        inv_softplus = torch.log(torch.expm1(torch.tensor(float(alpha_init))))
        self.alpha_raw = nn.Parameter(inv_softplus.repeat(num_heads))
        self.beta = nn.Parameter(torch.ones(1))

        self.reliability = nn.Sequential(
            nn.Linear(sensor_dim, sensor_dim // 2),
            nn.ReLU(),
            nn.Linear(sensor_dim // 2, 1),
        )

        self.head_norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    # ------------------------------------------------------------------
    # Local spatial block (mirrors RGSAGE.aggregate; kept separate so the
    # frozen baseline implementation is never modified)
    # ------------------------------------------------------------------
    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        src, dst = self.edge_index
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        if self.aggregator == "mean":
            out = x.new_zeros((b, n, d)).index_add(1, dst, x[:, src])
            counts = torch.bincount(dst, minlength=n).to(x.dtype)
            return out / counts[None, :, None].clamp(min=1)

        gathered = x[:, src]
        dst_expanded = dst.view(1, -1, 1).expand(b, -1, d)
        out = x.new_full((b, n, d), float("-inf"))
        out = out.scatter_reduce(1, dst_expanded, gathered, reduce="amax",
                                 include_self=True)
        return torch.where(out.isinf(), torch.zeros_like(out), out)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the decoder's attention distribution ``[B, heads, 26, 25]``.

        Exposed for the unit tests and for figures: the attention matrix is the
        model's own statement about which sensors it used for which candidate
        fault, so it is directly interpretable against the feeder topology.
        """
        return self._forward(x)[1]

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        avail = node_availability(x)                     # [B, N]

        h = self.encoder(x)                              # [B, N, temporal_dim]
        for i, layer in enumerate(self.spatial_layers):
            combined = torch.cat([h, self._aggregate(h)], dim=-1)
            h = layer(combined)
            if i < len(self.spatial_layers) - 1:
                h = torch.relu(h)

        n = h.shape[1]
        q = self.w_q(self.queries).view(-1, self.num_heads, self.d_head)
        q = q.permute(1, 0, 2).unsqueeze(0)              # [1, H, 26, d_head]
        k = self.w_k(h).view(b, n, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.w_v(h).view(b, n, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        logits = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B,H,26,N]

        if self.use_distance_bias:
            alpha = nn.functional.softplus(self.alpha_raw).view(1, -1, 1, 1)
            logits = logits - alpha * self.hop_distance.unsqueeze(0).unsqueeze(0)

        if self.use_reliability:
            r = torch.sigmoid(self.reliability(h)).squeeze(-1)      # [B, N]
            logits = logits + self.beta * torch.log(r.clamp(min=_EPS))[:, None, None, :]

        if self.use_availability_mask:
            # Overwrite rather than add, so an all-dropped sample degenerates to
            # a uniform row instead of an arbitrary distance-shaped one.
            dead = (avail < 0.5)[:, None, None, :].expand_as(logits)
            logits = logits.masked_fill(dead, MASK_VALUE)

        attention = torch.softmax(logits, dim=-1)
        context = (attention @ v)                                  # [B,H,26,d_head]
        context = context.permute(0, 2, 1, 3).reshape(b, -1, self.num_heads * self.d_head)

        queries = self.queries.unsqueeze(0).expand(b, -1, -1)       # [B,26,d_model]
        fused = self.head_norm(torch.cat([context, queries], dim=-1))
        class_logits = self.head(fused).squeeze(-1)                 # [B, 26]
        return class_logits.unsqueeze(1), attention                 # [B, 1, 26]

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        return self._forward(x)[0]
