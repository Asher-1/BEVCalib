"""V53b ADIR: Axis-Decoupled Iterative Refinement.

After DP-Head / CorrTransformer, apply T steps of per-axis small corrections
(Roll → Pitch → Yaw) using pooled correlation features + current quaternion.

Design: docs/V53_DESIGN.md §3.3
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _axis_delta_quat(angle_rad: torch.Tensor, axis: str) -> torch.Tensor:
    """Unit quaternion for rotation about one LiDAR axis (xyz euler convention)."""
    half = angle_rad * 0.5
    c = torch.cos(half)
    s = torch.sin(half)
    z = torch.zeros_like(c)
    if axis == "roll":
        return F.normalize(torch.stack([c, s, z, z], dim=-1), dim=-1, eps=1e-8)
    if axis == "pitch":
        return F.normalize(torch.stack([c, z, s, z], dim=-1), dim=-1, eps=1e-8)
    return F.normalize(torch.stack([c, z, z, s], dim=-1), dim=-1, eps=1e-8)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


class _AxisRefineHead(nn.Module):
    """Predict small single-axis correction (radians) from pooled features."""

    def __init__(self, feat_dim: int, axis: str, dropout: float = 0.1):
        super().__init__()
        self.axis = axis
        self.axis_tag = nn.Parameter(torch.randn(feat_dim) * 0.02)
        mid = max(feat_dim // 4, 32)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = pooled + self.axis_tag.unsqueeze(0)
        return self.mlp(x).squeeze(-1)


class ADIRRefiner(nn.Module):
    """T-step axis-decoupled iterative refinement on quaternion rotation."""

    def __init__(
        self,
        feat_dim: int = 256,
        n_steps: int = 2,
        max_step_deg: float = 1.0,
        head_dropout: float = 0.1,
        use_pitch_branch: bool = False,
    ):
        super().__init__()
        self.n_steps = max(1, int(n_steps))
        self.max_step_rad = math.radians(max_step_deg)
        self.use_pitch_branch = use_pitch_branch
        self.q_encoder = nn.Sequential(
            nn.Linear(4, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, feat_dim),
        )
        self.roll_head = _AxisRefineHead(feat_dim, "roll", head_dropout)
        self.pitch_head = _AxisRefineHead(feat_dim, "pitch", head_dropout)
        self.yaw_head = _AxisRefineHead(feat_dim, "yaw", head_dropout)

    def _refine_once(
        self,
        pooled: torch.Tensor,
        q: torch.Tensor,
        pitch_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx = pooled + self.q_encoder(q)
        dr = self.roll_head(ctx).tanh() * self.max_step_rad
        dp = self.pitch_head(ctx).tanh() * self.max_step_rad
        dy = self.yaw_head(ctx).tanh() * self.max_step_rad

        if pitch_delta is not None and self.use_pitch_branch:
            dp = dp + pitch_delta.squeeze(-1)

        q = _quat_mul(_axis_delta_quat(dr, "roll"), q)
        q = F.normalize(q, dim=-1, eps=1e-8)
        q = _quat_mul(_axis_delta_quat(dp, "pitch"), q)
        q = F.normalize(q, dim=-1, eps=1e-8)
        q = _quat_mul(_axis_delta_quat(dy, "yaw"), q)
        return F.normalize(q, dim=-1, eps=1e-8)

    def forward(
        self,
        corr_tokens: torch.Tensor,
        rotation: torch.Tensor,
        pitch_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = corr_tokens.mean(dim=1)
        q = F.normalize(rotation, dim=-1, eps=1e-8)
        for _ in range(self.n_steps):
            q = self._refine_once(pooled, q, pitch_delta=pitch_delta)
        return q
