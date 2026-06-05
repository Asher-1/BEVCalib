"""CalibFormer-style local window multi-head correlation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalMultiHeadCorrelation(nn.Module):
    """Local window correlation between img tokens and pc group tokens."""

    def __init__(
        self,
        token_dim: int = 384,
        num_heads: int = 4,
        window_radius: int = 4,
        out_dim: int = 256,
        patch_size: int = 14,
    ):
        super().__init__()
        assert token_dim % num_heads == 0
        self.num_heads = int(num_heads)
        self.window_radius = int(window_radius)
        self.out_dim = int(out_dim)
        self.patch_size = int(patch_size)
        self.head_dim = token_dim // num_heads
        self.proj = nn.Linear(token_dim + num_heads, out_dim)

    def _patch_grid(self, n_tokens: int, vit_h: int, vit_w: int) -> tuple[int, int]:
        gh = max(1, vit_h // self.patch_size)
        gw = max(1, vit_w // self.patch_size)
        if gh * gw != n_tokens:
            gw = max(1, n_tokens // gh)
        return gh, gw

    def forward(
        self,
        img_tokens: torch.Tensor,
        pc_tokens: torch.Tensor,
        pc_uv_init: torch.Tensor,
        vit_h: int,
        vit_w: int,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            img_tokens: B×P×D
            pc_tokens: B×G×D
            pc_uv_init: B×G×2 pixel coords at vit resolution
        Returns:
            f_corr: B×G×out_dim
        """
        b, p, d = img_tokens.shape
        _, g, _ = pc_tokens.shape
        gh, gw = self._patch_grid(p, vit_h, vit_w)
        img_grid = img_tokens.view(b, gh, gw, d)

        r = self.window_radius
        offs = torch.arange(-r, r + 1, device=img_tokens.device)
        dy, dx = torch.meshgrid(offs, offs, indexing='ij')
        win_y = dy.reshape(-1)
        win_x = dx.reshape(-1)
        wsize = win_y.numel()

        pi = (pc_uv_init[..., 1] / max(self.patch_size, 1)).long().clamp(0, gh - 1)
        pj = (pc_uv_init[..., 0] / max(self.patch_size, 1)).long().clamp(0, gw - 1)

        corr_heads = []
        pc_h = pc_tokens.view(b, g, self.num_heads, self.head_dim)
        img_h = img_grid.view(b, gh, gw, self.num_heads, self.head_dim)

        for wi in range(wsize):
            yi = (pi + win_y[wi]).clamp(0, gh - 1)
            xj = (pj + win_x[wi]).clamp(0, gw - 1)
            batch_idx = torch.arange(b, device=img_tokens.device).view(b, 1).expand(b, g)
            gathered = img_h[batch_idx, yi, xj]
            dot = (pc_h * gathered).sum(-1) / (self.head_dim ** 0.5)
            corr_heads.append(dot)

        corr_map = torch.stack(corr_heads, dim=2).mean(dim=2)
        corr_feat = torch.cat([pc_tokens, corr_map], dim=-1)
        f_corr = self.proj(corr_feat)

        in_bounds = (
            (pc_uv_init[..., 0] >= 0) & (pc_uv_init[..., 0] < vit_w)
            & (pc_uv_init[..., 1] >= 0) & (pc_uv_init[..., 1] < vit_h)
        )
        valid_ratio = in_bounds.float().mean(dim=1)
        return f_corr, {'corr_valid_ratio': valid_ratio, 'window_size': wsize}
