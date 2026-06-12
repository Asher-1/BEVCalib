"""V53 Dual-Path Pose Head (DP-Head).

BiasPath: zero-drift / sequence bias (InstanceNorm-friendly pooled features)
RecoveryPath: perturbation-sensitive correction (wraps CorrTransformerHead)
MagnitudeRouter: learnable init_err → recovery weight
JacobianGain: init_err-conditioned correction amplification (JACG)

Design: docs/V53_DESIGN.md §3.1–3.2
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_init_rot_rad(R_gt: torch.Tensor, R_init: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation error ||δ|| in radians, shape (B,)."""
    R_gt_f = R_gt[:, :3, :3].float()
    R_init_f = R_init[:, :3, :3].float()
    tr = (R_init_f @ R_gt_f.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation; t shape (B,1) broadcastable."""
    q0 = F.normalize(q0, dim=-1, eps=1e-8)
    q1 = F.normalize(q1, dim=-1, eps=1e-8)
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    q1_adj = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()
    omega = torch.acos(dot.clamp(-1 + 1e-7, 1 - 1e-7))
    sin_omega = torch.sin(omega).clamp(min=1e-7)
    w0 = torch.sin((1.0 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega
    near = (omega < 1e-4).float()
    out = w0 * q0 + w1 * q1_adj
    lin = (1.0 - t) * q0 + t * q1_adj
    out = near * lin + (1.0 - near) * out
    return F.normalize(out, dim=-1, eps=1e-8)


class BiasPath(nn.Module):
    """Zero-drift pathway: pooled correlation tokens → small Δq_bias."""

    def __init__(self, feat_dim: int, dropout: float = 0.1, use_in_norm: bool = True):
        super().__init__()
        self.token_norm = nn.LayerNorm(feat_dim) if use_in_norm else nn.Identity()
        mid = feat_dim // 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 4),
        )
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            nn.init.xavier_uniform_(self.head[-1].weight, gain=0.01)

    def forward(self, corr_tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_norm(corr_tokens)
        pooled = x.mean(dim=1)
        return F.normalize(self.head(pooled), dim=-1, eps=1e-8)


class MagnitudeRouter(nn.Module):
    """Route between bias (w→0) and recovery (w→1) from init error estimate."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        init_err_rad: torch.Tensor,
        mag_pred_rad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        init_deg = init_err_rad * (180.0 / math.pi)
        if mag_pred_rad is not None:
            mag_deg = mag_pred_rad.squeeze(-1) * (180.0 / math.pi)
        else:
            mag_deg = init_deg.detach()
        x = torch.stack([init_deg, mag_deg], dim=-1)
        return self.mlp(x).sigmoid()


class JacobianGain(nn.Module):
    """Amplify recovery branch for large init errors (JACG)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        init_err_rad: torch.Tensor,
        mag_pred_rad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        init_deg = init_err_rad * (180.0 / math.pi)
        if mag_pred_rad is not None:
            mag_deg = mag_pred_rad.squeeze(-1) * (180.0 / math.pi)
        else:
            mag_deg = init_deg.detach()
        x = torch.stack([init_deg, mag_deg], dim=-1)
        return self.mlp(x).clamp(min=0.05, max=2.0)


class DPPoseHead(nn.Module):
    """Dual-path pose head combining BiasPath + RecoveryPath (CorrTransformerHead)."""

    def __init__(
        self,
        recovery_head: nn.Module,
        feat_dim: int = 256,
        gate_deg: float = 1.5,
        use_jacg: bool = True,
        jacg_hidden_dim: int = 64,
        bias_path_in_norm: bool = True,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.recovery_head = recovery_head
        self.bias_path = BiasPath(feat_dim, dropout=head_dropout, use_in_norm=bias_path_in_norm)
        self.router = MagnitudeRouter(hidden_dim=jacg_hidden_dim)
        self.use_jacg = use_jacg
        self.gate_rad = math.radians(gate_deg)
        if use_jacg:
            self.jacg = JacobianGain(hidden_dim=jacg_hidden_dim)
        else:
            self.jacg = None
        self.token_norm_rec = nn.LayerNorm(feat_dim)

    def forward(
        self,
        corr_tokens: torch.Tensor,
        pose_queries: torch.Tensor,
        init_err_rad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        B = corr_tokens.shape[0]
        device = corr_tokens.device

        delta_q_bias = self.bias_path(corr_tokens)

        rec_tokens = self.token_norm_rec(corr_tokens)
        rec_out = self.recovery_head(rec_tokens, pose_queries)
        if isinstance(rec_out, tuple):
            delta_q_rec, mag_pred = rec_out
        else:
            delta_q_rec, mag_pred = rec_out, None

        if init_err_rad is None:
            init_err_rad = torch.zeros(B, device=device)
        if init_err_rad.dim() > 1:
            init_err_rad = init_err_rad.reshape(B)

        route_w = self.router(init_err_rad, mag_pred)

        if self.jacg is not None:
            gain = self.jacg(init_err_rad, mag_pred)
            eff_w = (route_w * gain).clamp(0.0, 1.0)
        else:
            gain = torch.ones(B, 1, device=device)
            eff_w = route_w

        delta_q = quat_slerp(delta_q_bias, delta_q_rec, eff_w)

        target_w = (init_err_rad > self.gate_rad).float().unsqueeze(-1)
        meta = {
            'route_w': route_w,
            'route_target_w': target_w,
            'delta_q_bias': delta_q_bias,
            'delta_q_rec': delta_q_rec,
            'jacg_gain': gain,
            'init_err_deg': init_err_rad * (180.0 / math.pi),
            'mag_pred': mag_pred,
        }
        return delta_q, meta
