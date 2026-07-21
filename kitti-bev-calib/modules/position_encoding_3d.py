"""V60: 3D Position Encoding for image features.

Implements paper Eq. 1-2: enriches 2D image features with 3D spatial
information by unprojecting pixel coordinates to 3D rays using camera
intrinsics and LID (Linear Increasing Discretization) depth sampling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoding3D(nn.Module):
    """3D Position Encoding for image features (paper Section 3.2).

    For each feature-map token at (u, v), samples depth via LID to get
    3D coordinates, then encodes into a position embedding that is
    added to the image features before cross-attention.

    Args:
        feat_dim: Output position encoding dimension (matches feature dim).
        depth_bins: Number of depth discretization bins (D in paper).
        depth_min: Minimum depth in meters.
        depth_max: Maximum depth in meters.
        patch_size: Feature map stride relative to original image.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        depth_bins: int = 16,
        depth_min: float = 1.0,
        depth_max: float = 100.0,
        patch_size: float = 4.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.depth_bins = depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.patch_size = patch_size

        depths = self._lid_discretization(depth_min, depth_max, depth_bins)
        self.register_buffer('depths', depths)

        self.pe_net = nn.Sequential(
            nn.Linear(3 * depth_bins, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        self._cached_pe = None
        self._cached_key = None

    @staticmethod
    def _lid_discretization(d_min: float, d_max: float, n_bins: int) -> torch.Tensor:
        """Linear Increasing Discretization (Reading et al., 2021)."""
        spacing = (d_max - d_min) / (n_bins * (n_bins + 1) / 2)
        depths = []
        cur = d_min
        for i in range(1, n_bins + 1):
            cur += spacing * i
            depths.append(cur)
        return torch.tensor(depths, dtype=torch.float32)

    def forward(
        self,
        feat_h: int,
        feat_w: int,
        cam_intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 3D position encoding for each image feature-map token.

        Args:
            feat_h, feat_w: Feature map spatial dimensions.
            cam_intrinsic: (B, 3, 3) camera intrinsic matrix.

        Returns:
            pe_3d: (B, feat_h * feat_w, feat_dim) position encoding.
        """
        B = cam_intrinsic.shape[0]
        device = cam_intrinsic.device

        # Only cache during eval (training needs fresh graphs for backward)
        if not self.training:
            cache_key = (feat_h, feat_w, B, device)
            if self._cached_pe is not None and self._cached_key == cache_key:
                fx_now = cam_intrinsic[:, 0, 0].mean().item()
                if hasattr(self, '_cached_fx') and abs(self._cached_fx - fx_now) < 0.1:
                    return self._cached_pe

        u_coords = (torch.arange(feat_w, device=device).float() + 0.5) * self.patch_size
        v_coords = (torch.arange(feat_h, device=device).float() + 0.5) * self.patch_size
        vv, uu = torch.meshgrid(v_coords, u_coords, indexing='ij')
        pixel_coords = torch.stack([uu.flatten(), vv.flatten()], dim=-1)

        fx = cam_intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2)
        fy = cam_intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2)
        cx = cam_intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2)
        cy = cam_intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2)

        N = pixel_coords.shape[0]
        D = self.depth_bins

        u = pixel_coords[:, 0].view(1, N, 1).expand(B, -1, D)
        v = pixel_coords[:, 1].view(1, N, 1).expand(B, -1, D)
        d = self.depths.view(1, 1, D).expand(B, N, -1)

        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d

        coords_3d = torch.cat([
            x_cam.reshape(B, N, D),
            y_cam.reshape(B, N, D),
            z_cam.reshape(B, N, D),
        ], dim=-1)

        pe_3d = self.pe_net(coords_3d)

        if not self.training:
            self._cached_pe = pe_3d
            self._cached_key = (feat_h, feat_w, B, device)
            self._cached_fx = cam_intrinsic[:, 0, 0].mean().item()

        return pe_3d
