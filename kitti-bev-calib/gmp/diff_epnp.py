"""Differentiable rotation-only EPnP via weighted Procrustes on bearing rays."""

from __future__ import annotations

import torch
import torch.nn as nn

from camera_geometry import transform_points_se3


def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _unproject_rays(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pixel uv (B,K,2) -> unit rays in camera frame (B,K,3)."""
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    x = (uv[..., 0] - cx) / fx.clamp(min=1e-6)
    y = (uv[..., 1] - cy) / fy.clamp(min=1e-6)
    rays = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    return _normalize(rays)


def _gram_schmidt_so3(R: torch.Tensor) -> torch.Tensor:
    """Project 3×3 matrices onto SO(3) without SVD (stable autograd)."""
    r0 = _normalize(R[:, :, 0])
    r1 = R[:, :, 1] - (r0 * R[:, :, 1]).sum(dim=-1, keepdim=True) * r0
    r1 = _normalize(r1)
    r2 = torch.cross(r0, r1, dim=-1)
    r = torch.stack([r0, r1, r2], dim=-1)
    det = torch.linalg.det(r)
    flip = torch.where(det < 0, -1.0, 1.0).view(-1, 1, 1)
    return r * flip


def _weighted_procrustes_rotation(
    src: torch.Tensor,
    tgt: torch.Tensor,
    weights: torch.Tensor,
    min_points: int = 4,
    svd_reg: float = 1e-5,
) -> torch.Tensor:
    """Batch weighted Kabsch: find R (B,3,3) mapping src->tgt unit rays."""
    b, _k, _ = src.shape
    w = weights.clamp(min=0.0)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    w_norm = w / w_sum

    src_c = src - (src * w_norm.unsqueeze(-1)).sum(dim=1, keepdim=True)
    tgt_c = tgt - (tgt * w_norm.unsqueeze(-1)).sum(dim=1, keepdim=True)

    h_mat = torch.bmm(
        (src_c * w_norm.unsqueeze(-1)).transpose(1, 2),
        tgt_c,
    )
    eye3 = torch.eye(3, device=h_mat.device, dtype=h_mat.dtype).unsqueeze(0)
    h_mat = h_mat + float(svd_reg) * eye3
    u, _s, vh = torch.linalg.svd(h_mat)
    r = torch.bmm(vh.transpose(1, 2), u.transpose(1, 2))
    r = _gram_schmidt_so3(r)

    valid_cnt = (w > 1e-4).sum(dim=1)
    eye = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0).expand(b, -1, -1)
    use = (valid_cnt >= min_points).float().view(b, 1, 1)
    return r * use + eye * (1.0 - use)


def _sanitize_grad(grad: torch.Tensor | None, max_norm: float = 5.0) -> torch.Tensor | None:
    if grad is None:
        return None
    grad = torch.nan_to_num(grad, nan=0.0, posinf=max_norm, neginf=-max_norm)
    return grad.clamp(-max_norm, max_norm)


class _StableProcrustesRotationFn(torch.autograd.Function):
    """Re-run Procrustes backward with sanitized grads (SVD NaN guard)."""

    @staticmethod
    def forward(ctx, src, tgt, weights, min_points, svd_reg):
        ctx.min_points = int(min_points)
        ctx.svd_reg = float(svd_reg)
        ctx.save_for_backward(src, tgt, weights)
        return _weighted_procrustes_rotation(
            src, tgt, weights, min_points=ctx.min_points, svd_reg=ctx.svd_reg)

    @staticmethod
    def backward(ctx, grad_r):
        grad_r = _sanitize_grad(grad_r, max_norm=5.0)
        src, tgt, weights = ctx.saved_tensors
        with torch.enable_grad():
            s = src.detach().requires_grad_(True)
            t = tgt.detach().requires_grad_(True)
            w = weights.detach().requires_grad_(True)
            r = _weighted_procrustes_rotation(
                s, t, w, min_points=ctx.min_points, svd_reg=ctx.svd_reg)
            gs, gt, gw = torch.autograd.grad(r, (s, t, w), grad_r, allow_unused=True)
        return (
            _sanitize_grad(gs),
            _sanitize_grad(gt),
            _sanitize_grad(gw),
            None,
            None,
        )


def _stable_weighted_procrustes_rotation(
    src: torch.Tensor,
    tgt: torch.Tensor,
    weights: torch.Tensor,
    min_points: int = 4,
    svd_reg: float = 1e-3,
) -> torch.Tensor:
    return _StableProcrustesRotationFn.apply(
        src, tgt, weights, min_points, svd_reg)


class DifferentiableEPnP(nn.Module):
    """Rotation-only PnP: estimate R_match s.t. rays through uv align with T_init-transformed 3D."""

    SVD_REG_DETACHED = 1e-5
    SVD_REG_DIFF = 1e-3

    def __init__(self, min_valid_points: int = 4, detach_rotation_grad: bool = True):
        super().__init__()
        self.min_valid_points = int(min_valid_points)
        self.detach_rotation_grad = bool(detach_rotation_grad)

    def forward(
        self,
        xyz: torch.Tensor,
        uv: torch.Tensor,
        K: torch.Tensor,
        weights: torch.Tensor,
        T_init: torch.Tensor,
        force_detach: bool | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            xyz: B×K×3 LiDAR frame
            uv: B×K×2 pixels
            K: B×3×3
            weights: B×K confidence
            T_init: B×4×4
            force_detach: override detach_rotation_grad (warmup curriculum)
        Returns:
            R_match: B×3×3 delta rotation (identity when insufficient inliers)
            meta: epnp_insufficient_ratio, epnp_mean_effective_points
        """
        detach = self.detach_rotation_grad if force_detach is None else bool(force_detach)
        svd_reg = self.SVD_REG_DETACHED if detach else self.SVD_REG_DIFF

        with torch.cuda.amp.autocast(enabled=False):
            xyz_f = xyz.float()
            uv_f = uv.float()
            k_f = K.float()
            w_f = weights.float()
            t_init_f = T_init.float()

            xyz_cam = transform_points_se3(xyz_f, t_init_f)
            src_rays = _normalize(xyz_cam)
            tgt_rays = _unproject_rays(uv_f, k_f)
            valid_cnt = (w_f > 1e-4).sum(dim=1)
            enough = valid_cnt >= self.min_valid_points

            eye = torch.eye(3, device=xyz_f.device, dtype=xyz_f.dtype).unsqueeze(0)
            procrustes = (
                _weighted_procrustes_rotation if detach
                else _stable_weighted_procrustes_rotation)
            r_all = procrustes(
                src_rays, tgt_rays, w_f,
                min_points=self.min_valid_points,
                svd_reg=svd_reg,
            )
            r_match = torch.where(enough.view(-1, 1, 1), r_all, eye)
            if self.training and detach:
                r_match = r_match.detach()
            epnp_meta = {
                'epnp_insufficient_ratio': (~enough).float().mean(),
                'epnp_mean_effective_points': valid_cnt.float().mean(),
            }
            return r_match, epnp_meta
