"""RoCR — Rotation-only Correlation Refine.

Extracts a rotation-only geometric initial guess from a correlation map
by finding per-group peak offsets, constructing 2D-3D direction correspondences,
and solving for R via SVD (Procrustes on unit directions).

Design: V42 CF-BEV-R (docs/V42_CF_BEV_DESIGN.md §2.3)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoCR(nn.Module):
    """Rotation-only Correlation Refine.

    Given a per-group correlation map and the 3D group centres, compute an
    initial rotation estimate via direction-matching SVD.  The Transformer
    decoder then only needs to predict a small residual quaternion.

    Args:
        min_valid_ratio: skip SVD and return identity when fewer points are
            valid (projected inside the image).
        temperature: soft-argmax temperature; lower → sharper peak selection.
        dropout_prob: probability of skipping RoCR during training (anti-shortcut AS4).
        center_neg_bias: negative logit bias on the window centre position to
            discourage the trivial centre-peak shortcut (AS3).
    """

    def __init__(
        self,
        min_valid_ratio: float = 0.3,
        temperature: float = 1.0,
        dropout_prob: float = 0.3,
        center_neg_bias: float = 0.5,
    ):
        super().__init__()
        self.min_valid_ratio = min_valid_ratio
        self.temperature = temperature
        self.dropout_prob = dropout_prob
        self.center_neg_bias = center_neg_bias

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def forward(
        self,
        corr_map: torch.Tensor,
        xyz_groups: torch.Tensor,
        uv_init: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        valid_mask: torch.Tensor,
        window_radius: int,
        patch_size: float,
        feat_h: int,
        feat_w: int,
    ) -> dict:
        """
        Args:
            corr_map:      (B, G, W²) correlation scores per group per window cell.
                           W = 2*window_radius + 1.
            xyz_groups:    (B, G, 3) LiDAR group centres in LiDAR frame.
            uv_init:       (B, G, 2) projected coordinates at **feature-map**
                           resolution (e.g. if feat is 1/4, value range [0, feat_w)).
            cam_intrinsic: (B, 3, 3) camera intrinsic at **original** image resolution.
            valid_mask:    (B, G) bool — groups projected inside the image.
            window_radius: int, correlation window radius (in feature-map cells).
            patch_size:    feature-map stride in original pixels (e.g. 4 for 1/4 res).
            feat_h, feat_w: feature map spatial dims (not used in computation,
                           reserved for future bounds checking).

        Returns:
            dict with:
                R_geo:         (B, 3, 3)  rotation-only estimate (or Identity).
                confidence:    (B,)       fraction of valid groups used.
                skipped:       (B,)       bool, True if RoCR was skipped for a sample.
        """
        B, G, W2 = corr_map.shape
        device = corr_map.device
        W = 2 * window_radius + 1
        assert W2 == W * W, f"corr_map last dim {W2} != W²={W * W}"

        R_identity = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        skipped = torch.zeros(B, dtype=torch.bool, device=device)

        if self.training and self.dropout_prob > 0:
            drop = torch.rand(B, device=device) < self.dropout_prob
            skipped = skipped | drop

        corr_logits = corr_map / max(self.temperature, 1e-6)

        if self.center_neg_bias != 0.0:
            center_idx = W2 // 2
            corr_logits = corr_logits.clone()
            corr_logits[:, :, center_idx] -= self.center_neg_bias

        delta_uv = self._soft_argmax_offset(corr_logits, window_radius, W)

        uv_feat = uv_init + delta_uv
        uv_pixel = uv_feat * patch_size

        rays = self._pixel_to_ray(uv_pixel, cam_intrinsic)

        lidar_dirs = F.normalize(xyz_groups, dim=-1, eps=1e-8)

        R_geo = self._batch_rotation_svd(rays, lidar_dirs, valid_mask)

        if self.training and R_geo.requires_grad:
            R_geo.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))

        valid_ratio = valid_mask.float().mean(dim=1)
        low_quality = valid_ratio < self.min_valid_ratio
        skipped = skipped | low_quality

        R_out = torch.where(
            skipped.view(B, 1, 1).expand_as(R_geo),
            R_identity,
            R_geo,
        )

        return {
            'R_geo': R_out,
            'confidence': valid_ratio,
            'skipped': skipped,
            'delta_uv': delta_uv,        # in feature-map cells
            'uv_refined_px': uv_pixel,   # in original image pixels
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _soft_argmax_offset(
        logits: torch.Tensor,
        radius: int,
        W: int,
    ) -> torch.Tensor:
        """Compute sub-pixel offset from correlation logits via spatial soft-argmax.

        Args:
            logits: (B, G, W²)
            radius: window radius
            W: window side length (2*radius + 1)

        Returns:
            offset: (B, G, 2) — (Δx, Δy) in patch units, range [-radius, radius].
        """
        device = logits.device
        weights = F.softmax(logits, dim=-1)

        offsets = torch.arange(-radius, radius + 1, device=device, dtype=logits.dtype)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        dx = dx.reshape(-1)
        dy = dy.reshape(-1)

        delta_x = (weights * dx.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        delta_y = (weights * dy.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return torch.stack([delta_x, delta_y], dim=-1)

    @staticmethod
    def _pixel_to_ray(
        uv: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """Convert 2D pixel coordinates to unit ray directions in camera frame.

        Args:
            uv: (B, G, 2) pixel coordinates (at original image resolution).
            K:  (B, 3, 3) intrinsic matrix.

        Returns:
            rays: (B, G, 3) unit direction vectors.
        """
        B, G, _ = uv.shape
        fx = K[:, 0, 0].unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)

        x = (uv[..., 0] - cx) / fx.clamp(min=1e-6)
        y = (uv[..., 1] - cy) / fy.clamp(min=1e-6)
        z = torch.ones_like(x)

        rays = torch.stack([x, y, z], dim=-1)
        return F.normalize(rays, dim=-1, eps=1e-8)

    @staticmethod
    def _batch_rotation_svd(
        cam_dirs: torch.Tensor,
        lidar_dirs: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Rotation-only Procrustes: find R minimising ||cam_dirs - R @ lidar_dirs||².

        Masks out invalid groups before SVD.  Falls back to identity on
        degenerate inputs (rank < 2).

        Args:
            cam_dirs:   (B, G, 3) unit camera rays.
            lidar_dirs: (B, G, 3) unit LiDAR directions.
            valid:      (B, G) bool mask.

        Returns:
            R: (B, 3, 3) rotation matrices.
        """
        B, G, _ = cam_dirs.shape
        device = cam_dirs.device

        with torch.cuda.amp.autocast(enabled=False):
            cam_f = cam_dirs.float()
            lid_f = lidar_dirs.float()
            mask = valid.float().unsqueeze(-1)
            cam_w = cam_f * mask
            lid_w = lid_f * mask

            H = torch.bmm(lid_w.transpose(1, 2), cam_w)
            H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)

            eye = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
            h_norm = H.flatten(1).norm(dim=1)
            zero_h = h_norm < 1e-5
            if zero_h.all():
                return eye.clone()

            U, S, Vh = torch.linalg.svd(H)

            det = torch.det(torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2)))
            sign = torch.ones(B, 3, device=device)
            sign[:, 2] = torch.sign(det)
            diag = torch.diag_embed(sign)

            R = torch.bmm(torch.bmm(Vh.transpose(1, 2), diag), U.transpose(1, 2))

            degenerate = (S[:, 1] < 1e-4) | zero_h
            if degenerate.any():
                R = torch.where(degenerate.view(B, 1, 1).expand_as(R), eye, R)

        return R

    @staticmethod
    def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """Batch-convert rotation matrices to quaternions (w, x, y, z).

        Args:
            R: (B, 3, 3)

        Returns:
            q: (B, 4) unit quaternions.
        """
        B = R.shape[0]
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

        _eps = 1e-8

        mask0 = tr > 0
        s0 = torch.sqrt((tr[mask0] + 1.0).clamp(min=_eps)) * 2
        q[mask0, 0] = 0.25 * s0
        q[mask0, 1] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / s0.clamp(min=_eps)
        q[mask0, 2] = (R[mask0, 0, 2] - R[mask0, 2, 0]) / s0.clamp(min=_eps)
        q[mask0, 3] = (R[mask0, 1, 0] - R[mask0, 0, 1]) / s0.clamp(min=_eps)

        mask1 = (~mask0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s1 = torch.sqrt((1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]).clamp(min=_eps)) * 2
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1.clamp(min=_eps)
        q[mask1, 1] = 0.25 * s1
        q[mask1, 2] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / s1.clamp(min=_eps)
        q[mask1, 3] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / s1.clamp(min=_eps)

        mask2 = (~mask0) & (~mask1) & (R[:, 1, 1] > R[:, 2, 2])
        s2 = torch.sqrt((1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]).clamp(min=_eps)) * 2
        q[mask2, 0] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / s2.clamp(min=_eps)
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2.clamp(min=_eps)
        q[mask2, 2] = 0.25 * s2
        q[mask2, 3] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s2.clamp(min=_eps)

        mask3 = (~mask0) & (~mask1) & (~mask2)
        s3 = torch.sqrt((1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]).clamp(min=_eps)) * 2
        q[mask3, 0] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / s3.clamp(min=_eps)
        q[mask3, 1] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / s3.clamp(min=_eps)
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3.clamp(min=_eps)
        q[mask3, 3] = 0.25 * s3

        return F.normalize(q, dim=-1, eps=1e-8)
