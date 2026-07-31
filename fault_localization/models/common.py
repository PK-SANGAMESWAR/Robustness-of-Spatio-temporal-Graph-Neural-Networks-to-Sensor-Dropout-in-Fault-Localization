from __future__ import annotations
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Temporal encoder (shared GRU over each node)
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Shared GRU over each node's time series.

    Input shape: [B, N, S, F]
      B = batch size
      N = number of nodes (25)
      S = window length (20 timesteps)
      F = feature dim (6: voltages 0-2 + availability mask 3-5)

    Output shape: [B, N, hidden_dim]
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, s, f = x.shape
        _, h = self.gru(x.reshape(b * n, s, f))
        return h[-1].reshape(b, n, -1)


# ---------------------------------------------------------------------------
# Node classifier head (shared across all models)
# ---------------------------------------------------------------------------

class NodeClassifier(nn.Module):
    """Per-node classification head: BN → dropout → ReLU → linear."""
    def __init__(self, in_dim: int, classes: int = 26, dropout: float = 0.35):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = self.bn(x.reshape(b * n, d)).reshape(b, n, d)
        return self.linear(self.dropout(torch.relu(x)))


# ---------------------------------------------------------------------------
# Loss (node-level cross entropy with label expansion)
# ---------------------------------------------------------------------------

def expanded_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over all nodes, sharing one event-level label.

    logits  : [B, N, C]
    targets : [B]   (one label per event, expanded to all N nodes)
    weight  : [C] optional per-class loss weights
    """
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets[:, None].expand(-1, logits.size(1)).reshape(-1),
        weight=weight,
    )


def expanded_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal Loss over all nodes with expanded event labels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    flat_logits = logits.reshape(-1, logits.size(-1))  # [B*N, C]
    flat_targets = targets[:, None].expand(-1, logits.size(1)).reshape(-1)  # [B*N]

    log_probs = nn.functional.log_softmax(flat_logits, dim=-1)
    probs = torch.exp(log_probs)

    target_log_probs = log_probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
    target_probs = probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)

    focal_weights = (1.0 - target_probs) ** gamma
    loss = -focal_weights * target_log_probs

    if alpha is not None:
        alpha_factor = alpha.gather(dim=0, index=flat_targets)
        loss = loss * alpha_factor

    return loss.mean()

