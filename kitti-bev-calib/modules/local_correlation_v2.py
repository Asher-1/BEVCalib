"""LocalMultiHeadCorrelationV2 — forked from gmp/local_correlation.py for V42.

Key differences from V1:
1. Returns full corr_map (B, G, W², H) instead of mean-pooling over window
   → RoCR needs spatial peak information for soft_argmax
2. Accepts feature-map grid dimensions directly (feat_h, feat_w) instead
   of ViT patch-based grid computation
3. Supports adaptive window radius via forward-time override
4. Projection output (f_corr) uses per-head corr features, not head-averaged

Design: V42 CF-BEV-R (docs/V42_CF_BEV_DESIGN.md §2.2 Stage-2(B), §10.2)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalMultiHeadCorrelationV2(nn.Module):
    """Local window multi-head correlation for CF-BEV-R.

    Computes per-group correlation maps between point cloud tokens and
    nearby image tokens within a local window centred at the projected
    uv_init coordinates.

    Args:
        token_dim:      Feature dimension (must be divisible by num_heads).
        num_heads:      Number of correlation heads.
        default_radius: Default window half-size (can be overridden at forward).
        out_dim:        Output feature dimension after projection.
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_heads: int = 4,
        default_radius: int = 4,
        out_dim: int = 256,
    ):
        super().__init__()
        assert token_dim % num_heads == 0
        self.num_heads = int(num_heads)
        self.default_radius = int(default_radius)
        self.out_dim = int(out_dim)
        self.head_dim = token_dim // num_heads

        self.max_radius = 12
        max_w2 = (2 * self.max_radius + 1) ** 2
        self.proj = nn.Linear(token_dim + max_w2, out_dim)

    def forward(
        self,
        img_tokens: torch.Tensor,
        pc_tokens: torch.Tensor,
        pc_uv_init: torch.Tensor,
        feat_h: int,
        feat_w: int,
        window_radius: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            img_tokens:   (B, feat_h*feat_w, D) image feature tokens.
            pc_tokens:    (B, G, D) point cloud group tokens.
            pc_uv_init:   (B, G, 2) projected pixel coordinates at
                          **feature-map** resolution.
            feat_h:       Feature map height.
            feat_w:       Feature map width.
            window_radius: Override for adaptive window (None → default).

        Returns:
            f_corr:    (B, G, out_dim) fused correlation feature.
            corr_map:  (B, G, W²) correlation scores (head-averaged),
                       suitable for RoCR soft_argmax.
            info:      dict with valid_ratio, window_size, etc.
        """
        r = window_radius if window_radius is not None else self.default_radius
        b, p, d = img_tokens.shape
        _, g, _ = pc_tokens.shape
        assert p == feat_h * feat_w, \
            f"img_tokens has {p} tokens but feat_h*feat_w={feat_h * feat_w}"

        img_grid = img_tokens.view(b, feat_h, feat_w, d)

        W = 2 * r + 1
        wsize = W * W

        pi = pc_uv_init[..., 1].long().clamp(0, feat_h - 1)  # (B, G)
        pj = pc_uv_init[..., 0].long().clamp(0, feat_w - 1)  # (B, G)

        offs = torch.arange(-r, r + 1, device=img_tokens.device)
        dy, dx = torch.meshgrid(offs, offs, indexing='ij')
        win_y = dy.reshape(1, 1, -1)   # (1, 1, W²)
        win_x = dx.reshape(1, 1, -1)

        all_yi = (pi.unsqueeze(2) + win_y).clamp(0, feat_h - 1)  # (B, G, W²)
        all_xj = (pj.unsqueeze(2) + win_x).clamp(0, feat_w - 1)

        flat_idx = all_yi * feat_w + all_xj  # (B, G, W²)

        img_flat = img_grid.reshape(b, feat_h * feat_w, self.num_heads, self.head_dim)

        batch_idx = torch.arange(b, device=img_tokens.device).view(b, 1, 1).expand_as(flat_idx)
        gathered = img_flat[batch_idx, flat_idx]  # (B, G, W², H, head_dim)

        pc_h = pc_tokens.view(b, g, 1, self.num_heads, self.head_dim)
        corr_full = (pc_h * gathered).sum(-1) / math.sqrt(self.head_dim)  # (B, G, W², H)
        corr_map = corr_full.mean(dim=-1)  # (B, G, W²) head-averaged

        max_w2 = (2 * self.max_radius + 1) ** 2
        if wsize < max_w2:
            pad = torch.zeros(b, g, max_w2 - wsize,
                              device=corr_map.device)
            corr_padded = torch.cat([corr_map, pad], dim=2)
        elif wsize > max_w2:
            corr_padded = corr_map[:, :, :max_w2]
        else:
            corr_padded = corr_map

        corr_feat = torch.cat([pc_tokens, corr_padded], dim=-1)
        f_corr = self.proj(corr_feat)

        in_bounds = (
            (pc_uv_init[..., 0] >= 0) & (pc_uv_init[..., 0] < feat_w)
            & (pc_uv_init[..., 1] >= 0) & (pc_uv_init[..., 1] < feat_h)
        )
        valid_ratio = in_bounds.float().mean(dim=1)

        return f_corr, corr_map, {
            'corr_valid_ratio': valid_ratio,
            'window_size': wsize,
            'window_radius': r,
            'valid_mask': in_bounds,
        }
