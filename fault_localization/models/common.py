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



# ---------------------------------------------------------------------------
# PC-STGNN objective (PC-STGNN/01_SPEC.md §6)
# ---------------------------------------------------------------------------

def pc_loss(
    out: dict,
    targets: torch.Tensor,
    lam1: float = 0.0,
    lam2: float = 0.0,
    loss_mask: torch.Tensor | None = None,
) -> dict:
    """``CE(z', y) + lam1 * E_P[rho(A)] + lam2 * E_P[rho(D)]``.

    ``out`` is what :meth:`models.pc_stgnn.PCSTGNN.forward_pc` returns.  The two
    physics terms are the expected residual under the model's own soft-vote
    distribution ``P(f|X)``: the gradient reaches the network **through P**, not
    through rho, which is a constant of the inputs.

    ``loss_mask`` is ``[B]``, zero on windows with no post-inception samples
    (w = 20).  Those windows carry a fault label while being physically
    identical to a healthy window, so the physics necessarily prefers class 0
    there and including them would train the model against a residual that is
    right (``01_SPEC.md`` §3, ``02_STAGE_A_audit.md`` S0.3).  It is the only
    place a window index is used anywhere, it applies at training time only, and
    it never enters the residual itself.

    Do **not** replace these with a reconstruction loss of the form
    ``||mu[f_hat] - s_true||^2``: ``f_hat`` is an arg-min index and constant in
    the weights, so that term has exactly zero gradient and the lambda_2 arm
    would be silently identical to lambda_1 (``01_SPEC.md`` §6, audited by S4.2).
    """
    logits = out["logits"]
    total = expanded_cross_entropy(logits, targets)
    parts = {"ce": total.detach().clone()}

    if lam1 or lam2:
        p_graph = out["p_graph"]                       # [B, 26], sums to 1
        weights = (torch.ones(p_graph.shape[0], device=p_graph.device)
                   if loss_mask is None else loss_mask.to(p_graph.dtype))
        denom = weights.sum().clamp(min=1.0)
        for name, key, lam in (("l1", "rho_A", lam1), ("l2", "rho_D", lam2)):
            if not lam:
                continue
            if key not in out:
                raise KeyError(f"pc_loss needs out[{key!r}] for lambda_{name[-1]}")
            term = ((p_graph * out[key]).sum(dim=-1) * weights).sum() / denom
            parts[name] = term.detach().clone()
            total = total + lam * term

    parts["total"] = total
    return parts


def unlabelable_window_mask(window_ids, fault_start_step: int = 40,
                            window_length: int = 20) -> torch.Tensor:
    """``[B]`` mask, 0 on windows with zero post-inception samples.

    Window ``w`` covers steps ``[w, w + window_length)``, so it contains
    ``w + window_length - fault_start_step`` post-inception samples.  Exactly one
    fault-labelled window per event (w = 20) has none.
    """
    keep = []
    for wid in window_ids:
        index = int(str(wid).rpartition("_w")[2])
        keep.append(0.0 if index + window_length - fault_start_step == 0 else 1.0)
    return torch.tensor(keep, dtype=torch.float32)
