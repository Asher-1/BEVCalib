"""GeoConsistency loss (V40 P0a): photometric + depth alignment under T_pred vs T_gt."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camera_geometry import (
    project_cam_to_pixel,
    sample_image_at_pixels,
    transform_points_se3,
)


class GeoConsistencyLoss(nn.Module):
    """RobustCalib-style geometric consistency on the main image resolution (640×360 + K)."""

    def __init__(
        self,
        appearance_weight: float = 0.1,
        depth_weight: float = 0.05,
        max_points: int = 2048,
    ):
        super().__init__()
        self.appearance_weight = appearance_weight
        self.depth_weight = depth_weight
        self.max_points = max_points

    def _sample_points(self, pc: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, n, _ = pc.shape
        out = []
        for i in range(b):
            if mask is not None:
                valid = mask[i] == 1
                pts = pc[i, valid]
            else:
                pts = pc[i]
            if pts.shape[0] == 0:
                pts = pc[i, :1]
            if pts.shape[0] > self.max_points:
                idx = torch.randperm(pts.shape[0], device=pts.device)[: self.max_points]
                pts = pts[idx]
            out.append(pts)
        max_n = max(p.shape[0] for p in out)
        dtype, device = pc.dtype, pc.device
        batched = torch.zeros(b, max_n, 3, device=device, dtype=dtype)
        valid_n = torch.zeros(b, max_n, dtype=torch.bool, device=device)
        for i, pts in enumerate(out):
            batched[i, : pts.shape[0]] = pts
            valid_n[i, : pts.shape[0]] = True
        return batched, valid_n

    def forward(
        self,
        img: torch.Tensor,
        pc: torch.Tensor,
        T_pred: torch.Tensor,
        T_gt: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            img: (B, 3, H, W) main pipeline image
            pc: (B, N, 3) LiDAR points
            T_pred, T_gt: (B, 4, 4) LiDAR -> camera
            cam_intrinsic: (B, 3, 3) for img resolution
        """
        if cam_intrinsic.dim() == 4:
            cam_intrinsic = cam_intrinsic.squeeze(1)
        _, _, h, w = img.shape
        pts, pt_valid = self._sample_points(pc, mask)

        cam_pred = transform_points_se3(pts, T_pred)
        cam_gt = transform_points_se3(pts, T_gt)
        u_p, v_p, z_p = project_cam_to_pixel(cam_pred, cam_intrinsic)
        u_g, v_g, z_g = project_cam_to_pixel(cam_gt, cam_intrinsic)

        in_bounds = (
            pt_valid
            & (z_p > 0.5) & (z_g > 0.5)
            & (u_p >= 0) & (u_p < w) & (v_p >= 0) & (v_p < h)
            & (u_g >= 0) & (u_g < w) & (v_g >= 0) & (v_g < h)
        )
        valid_count = in_bounds.float().sum().clamp(min=1.0)

        appearance_loss = torch.tensor(0.0, device=img.device)
        depth_loss = torch.tensor(0.0, device=img.device)
        if in_bounds.any():
            i_p = sample_image_at_pixels(img, u_p, v_p)
            i_g = sample_image_at_pixels(img, u_g, v_g)
            diff = (i_p - i_g).abs().mean(dim=-1)
            appearance_loss = (diff * in_bounds.float()).sum() / valid_count
            depth_rel = (z_p - z_g).abs() / z_g.clamp(min=0.5)
            depth_loss = (depth_rel * in_bounds.float()).sum() / valid_count

        total = self.appearance_weight * appearance_loss + self.depth_weight * depth_loss
        return {
            'geo_consistency_loss': total,
            'appearance_loss': appearance_loss,
            'depth_loss': depth_loss,
            'geo_valid_ratio': (in_bounds.float().sum() / pt_valid.float().sum().clamp(min=1.0)).detach(),
        }
