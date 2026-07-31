from __future__ import annotations
from torch import nn
from .common import NodeClassifier, TemporalEncoder

class GRUBaseline(nn.Module):
    """GRU-only baseline: temporal encoding + node classification.

    Input F=6 (3 voltage channels + 3 availability mask channels).
    """
    def __init__(self, temporal_hidden_dim: int = 128, num_classes: int = 26, dropout: float = 0.35):
        super().__init__()
        self.encoder = TemporalEncoder(6, temporal_hidden_dim)
        self.classifier = NodeClassifier(temporal_hidden_dim, num_classes, dropout)

    def forward(self, x, edge_index=None):
        return self.classifier(self.encoder(x))
