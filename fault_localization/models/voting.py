from __future__ import annotations
import torch

def soft_vote(node_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum (not average) node class probabilities and select a graph class."""
    if node_logits.ndim != 3: raise ValueError("Expected [batch, nodes, classes] logits")
    scores = torch.softmax(node_logits, dim=-1).sum(dim=1)
    return scores, scores.argmax(dim=-1)
