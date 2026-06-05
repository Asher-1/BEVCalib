"""Camera intrinsics scaling and projection utilities (V40 M-1)."""

from __future__ import annotations

import torch


def scale_intrinsics_for_resize(
    cam_intrinsic: torch.Tensor,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> torch.Tensor:
    """Scale K when an image is resized from (src_h, src_w) to (dst_h, dst_w).

    Args:
        cam_intrinsic: (B, 3, 3) or (3, 3)
    Returns:
        Scaled intrinsics, same shape as input.
    """
    squeeze = False
    if cam_intrinsic.dim() == 2:
        cam_intrinsic = cam_intrinsic.unsqueeze(0)
        squeeze = True
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    k = cam_intrinsic.clone()
    k[:, 0, 0] *= sx
    k[:, 1, 1] *= sy
    k[:, 0, 2] *= sx
    k[:, 1, 2] *= sy
    return k.squeeze(0) if squeeze else k


def transform_points_se3(xyz: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply T (B,4,4) to xyz (B,N,3) -> (B,N,3) in homogeneous form."""
    b, n, _ = xyz.shape
    ones = torch.ones(b, n, 1, device=xyz.device, dtype=xyz.dtype)
    pts_h = torch.cat([xyz, ones], dim=-1)
    return torch.bmm(pts_h, T.transpose(1, 2))[:, :, :3]


def project_cam_to_pixel(
    xyz_cam: torch.Tensor,
    cam_intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project camera-frame points to pixel coordinates.

    Args:
        xyz_cam: (B, N, 3)
        cam_intrinsic: (B, 3, 3)
    Returns:
        u, v, z each (B, N)
    """
    z = xyz_cam[..., 2].clamp(min=1e-3)
    fx = cam_intrinsic[:, 0, 0].unsqueeze(1)
    fy = cam_intrinsic[:, 1, 1].unsqueeze(1)
    cx = cam_intrinsic[:, 0, 2].unsqueeze(1)
    cy = cam_intrinsic[:, 1, 2].unsqueeze(1)
    u = fx * xyz_cam[..., 0] / z + cx
    v = fy * xyz_cam[..., 1] / z + cy
    return u, v, z


def pixel_to_grid(u: torch.Tensor, v: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert pixel coords to grid_sample coords in [-1, 1], shape (B, N, 1, 2)."""
    gn_x = 2.0 * u / max(w - 1, 1) - 1.0
    gn_y = 2.0 * v / max(h - 1, 1) - 1.0
    return torch.stack([gn_x, gn_y], dim=-1).unsqueeze(2)


def sample_image_at_pixels(
    img: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Bilinear sample img (B,3,H,W) at pixel u,v (B,N) -> (B,N,3)."""
    _, _, h, w = img.shape
    grid = pixel_to_grid(u, v, h, w)
    sampled = torch.nn.functional.grid_sample(
        img, grid, mode='bilinear', align_corners=True, padding_mode='border')
    return sampled.squeeze(-1).permute(0, 2, 1)
