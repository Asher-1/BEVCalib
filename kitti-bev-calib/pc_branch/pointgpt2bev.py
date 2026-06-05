"""Scatter PointGPT group features onto a BEV grid (replaces Lidar2BEV Spconv)."""

from __future__ import annotations

import torch
import torch.nn as nn

import bev_settings
from proj_head import ProjectionHead


class PointGPT2BEV(nn.Module):
    """Map PointGPT group centers + features to a dense BEV feature map."""

    def __init__(self, in_dim: int = 384, out_channels: int = 128):
        super().__init__()
        x0, x1, dx = bev_settings.xbound
        y0, y1, dy = bev_settings.ybound
        self.x0, self.x1, self.dx = float(x0), float(x1), float(dx)
        self.y0, self.y1, self.dy = float(y0), float(y1), float(dy)
        self.bev_h = int(round((self.x1 - self.x0) / self.dx))
        self.bev_w = int(round((self.y1 - self.y0) / self.dy))
        self.in_dim = in_dim
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
        )
        self.proj_head = ProjectionHead(embedding_dim=in_dim, projection_dim=out_channels)
        self.out_channels = self.proj_head.projection_dim
        print(f"[PointGPT2BEV] grid={self.bev_h}x{self.bev_w}, in_dim={in_dim}, "
              f"out_channels={self.out_channels}")

    def _xy_to_indices(self, xy: torch.Tensor):
        """xy: (B, G, 2) with x=forward, y=lateral."""
        ix = ((xy[..., 0] - self.x0) / self.dx).long()
        iy = ((xy[..., 1] - self.y0) / self.dy).long()
        valid = (
            (ix >= 0) & (ix < self.bev_h) &
            (iy >= 0) & (iy < self.bev_w)
        )
        return ix, iy, valid

    def forward(self, xyz_groups: torch.Tensor, feat_groups: torch.Tensor):
        """
        Args:
            xyz_groups: (B, G, 3)
            feat_groups: (B, G, D)
        Returns:
            bev_feats: (B, C, H, W)
            bev_mask: (B, H, W) float
        """
        B, G, D = feat_groups.shape
        device = feat_groups.device
        feat = self.feat_proj(feat_groups)

        bev_sum = feat_groups.new_zeros(B, D, self.bev_h, self.bev_w)
        bev_cnt = feat_groups.new_zeros(B, 1, self.bev_h, self.bev_w)

        ix, iy, valid = self._xy_to_indices(xyz_groups[..., :2])
        for b in range(B):
            vb = valid[b]
            if not vb.any():
                continue
            idx_x = ix[b, vb]
            idx_y = iy[b, vb]
            f = feat[b, vb]  # (N, D)
            lin = idx_x * self.bev_w + idx_y
            bev_flat = bev_sum[b].view(D, -1)
            cnt_flat = bev_cnt[b, 0].view(-1)
            bev_flat.scatter_add_(1, lin.unsqueeze(0).expand(D, -1), f.transpose(0, 1))
            cnt_flat.scatter_add_(0, lin, torch.ones(lin.shape[0], device=device))

        bev_mask = (bev_cnt.squeeze(1) > 0).float()
        bev_sum = bev_sum / bev_cnt.clamp(min=1.0)
        bev_feats = self.proj_head(bev_sum.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return bev_feats, bev_mask
