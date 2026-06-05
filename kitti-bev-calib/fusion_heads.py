"""Fusion heads for HybridTripleCalib (HTCN)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp_head(in_dim: int, hidden: int = 128, dropout: float = 0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, 4),
    )


class GatedFusionHead(nn.Module):
    """F1: adaptive gate between BEV global and Proj projection features."""

    def __init__(self, bev_dim: int, proj_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.bev_proj = nn.Linear(bev_dim, hidden)
        self.proj_proj = nn.Linear(proj_dim, hidden)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        # 初始化Gate偏向BEV，延缓坍塌
        with torch.no_grad():
            self.gate[-1].bias.data = torch.tensor([1.5, -1.5])
        self.head = _mlp_head(hidden, hidden, dropout)

    def forward(self, f_bev: torch.Tensor, f_proj: torch.Tensor):
        fb = self.bev_proj(f_bev)
        fp = self.proj_proj(f_proj)
        logits = self.gate(torch.cat([fb, fp], dim=-1))
        w = torch.softmax(logits, dim=-1)
        fused = w[:, :1] * fb + w[:, 1:2] * fp
        q = self.head(fused)
        return F.normalize(q, dim=-1, eps=1e-6), {'gate_bev': w[:, 0], 'gate_proj': w[:, 1]}


class CascadeFusionHead(nn.Module):
    """F2: placeholder — proj coarse then BEV fine (single-step fused features)."""

    def __init__(self, bev_dim: int, proj_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj_head = _mlp_head(proj_dim, hidden, dropout)
        self.bev_head = _mlp_head(bev_dim + 4, hidden, dropout)

    def forward(self, f_bev: torch.Tensor, f_proj: torch.Tensor):
        q_coarse = F.normalize(self.proj_head(f_proj), dim=-1, eps=1e-6)
        q_fine = F.normalize(self.bev_head(torch.cat([f_bev, q_coarse], dim=-1)),
                            dim=-1, eps=1e-6)
        return q_fine, {'q_coarse': q_coarse}


class ResidualSO3FusionHead(nn.Module):
    """F4: compose two quaternion deltas (normalized product approximation)."""

    def __init__(self, bev_dim: int, proj_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.bev_head = _mlp_head(bev_dim, hidden, dropout)
        self.proj_head = _mlp_head(proj_dim, hidden, dropout)

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def forward(self, f_bev: torch.Tensor, f_proj: torch.Tensor):
        q_bev = F.normalize(self.bev_head(f_bev), dim=-1, eps=1e-6)
        q_proj = F.normalize(self.proj_head(f_proj), dim=-1, eps=1e-6)
        q = F.normalize(self._quat_mul(q_proj, q_bev), dim=-1, eps=1e-6)
        return q, {}


class SingleBranchHead(nn.Module):
    """BEV-only or Proj-only regression head."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = _mlp_head(in_dim, hidden, dropout)

    def forward(self, feat: torch.Tensor):
        q = F.normalize(self.head(feat), dim=-1, eps=1e-6)
        return q, {}


FUSION_HEADS = {
    'gated': GatedFusionHead,
    'cascade': CascadeFusionHead,
    'residual': ResidualSO3FusionHead,
}
