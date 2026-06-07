# MLP that scores each candidate from its concatenated visual + language + geometric (+ extra) features.

import torch
import torch.nn as nn


class GroundingMLP(nn.Module):
    def __init__(self, visual_dim: int = 2048, language_dim: int = 384, geometric_dim: int = 5, extra_dim: int = 0, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        input_dim = visual_dim + language_dim + geometric_dim + extra_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    # (N, feat) -> (N,) scores, or (batch, N, feat) -> (batch, N).
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        scores = self.net(features).squeeze(-1)
        return scores.squeeze(0) if squeeze else scores
