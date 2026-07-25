"""The candidate-scoring MLP."""

import torch
import torch.nn as nn


class GroundingMLP(nn.Module):
    """Scores each candidate from its feature vector. Trained with cross-entropy over the N
    candidates per command (the GT-matched candidate is the target)."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(N, input_dim) -> (N,) scores, or (B, N, input_dim) -> (B, N)."""
        return self.net(features).squeeze(-1)
