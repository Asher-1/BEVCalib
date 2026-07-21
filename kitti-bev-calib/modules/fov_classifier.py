"""V60: FOV Classification auxiliary head.

Predicts whether each 3D point group falls within the camera FOV.
Provides additional training signal to stabilize cross-attention
learning (paper Eq. 10).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FoVClassifier(nn.Module):
    """Auxiliary head predicting per-point FOV membership.

    Args:
        feat_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, feat_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pc_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc_features: (B, G, D) point cloud group features.

        Returns:
            fov_logits: (B, G) logits for FOV classification.
        """
        return self.mlp(pc_features).squeeze(-1)
