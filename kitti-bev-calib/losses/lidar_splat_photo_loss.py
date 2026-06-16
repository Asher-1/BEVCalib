"""V54 LiDAR Splat Photo Loss (LSP): TLC-inspired dense photo-geometric alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camera_geometry import project_cam_to_pixel, sample_image_at_pixels, transform_points_se3


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_2d = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return window_2d


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """SSIM map averaged over channels, shape (B, 1, H, W)."""
    c = img1.shape[1]
    w = window.expand(c, 1, window.shape[-2], window.shape[-1])
    pad = window.shape[-1] // 2
    mu1 = F.conv2d(img1, w, padding=pad, groups=c)
    mu2 = F.conv2d(img2, w, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=c) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim.mean(dim=1, keepdim=True)


def _bilinear_splat(
    colors: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    h: int,
    w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable bilinear splat: colors (B,N,3), u,v pixel coords -> render (B,3,H,W), weight (B,1,H,W)."""
    b, n, _ = colors.shape
    device, dtype = colors.device, colors.dtype
    render = torch.zeros(b, 3, h, w, device=device, dtype=dtype)
    weight = torch.zeros(b, 1, h, w, device=device, dtype=dtype)

    u = u.clamp(0, w - 1.001)
    v = v.clamp(0, h - 1.001)
    u0 = u.floor().long()
    v0 = v.floor().long()
    u1 = (u0 + 1).clamp(max=w - 1)
    v1 = (v0 + 1).clamp(max=h - 1)
    du = (u - u0.float()).unsqueeze(-1)
    dv = (v - v0.float()).unsqueeze(-1)

    corners = (
        (u0, v0, (1 - du) * (1 - dv)),
        (u1, v0, du * (1 - dv)),
        (u0, v1, (1 - du) * dv),
        (u1, v1, du * dv),
    )
    for ui, vi, wgt in corners:
        flat = (vi * w + ui).view(b, n)
        wgt3 = wgt.squeeze(-1)
        for ch in range(3):
            src = colors[..., ch] * wgt3
            render_ch = render[:, ch].view(b, -1)
            render_ch.scatter_add_(1, flat, src)
            render[:, ch] = render_ch.view(b, h, w)
        w_flat = weight.view(b, -1)
        w_flat.scatter_add_(1, flat, wgt3)
        weight = w_flat.view(b, 1, h, w)

    return render, weight.clamp(min=1e-6)


class LiDARSplatPhotoLoss(nn.Module):
    """Soft-splat LiDAR colors (from GT projection) using T_pred, compare to image."""

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        max_points: int = 4096,
        min_coverage: float = 50.0,
        ssim_window: int = 11,
        downsample: int = 4,
    ):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.max_points = max_points
        self.min_coverage = min_coverage
        self.ssim_window = ssim_window
        self.downsample = max(1, downsample)

    def _prepare_img(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 5:
            img = img[:, 0]
        return img

    def _sample_points(self, pc: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, c = pc.shape
        xyz = pc[..., :3]
        out, valid = [], []
        for i in range(b):
            pts = xyz[i]
            if mask is not None:
                m = mask[i] == 1
                pts = pts[m]
            if pts.shape[0] == 0:
                pts = xyz[i, :1]
            if pts.shape[0] > self.max_points:
                idx = torch.randperm(pts.shape[0], device=pts.device)[: self.max_points]
                pts = pts[idx]
            pad_n = pts.shape[0]
            buf = torch.zeros(self.max_points, 3, device=pts.device, dtype=pts.dtype)
            buf[:pad_n] = pts
            vmask = torch.zeros(self.max_points, dtype=torch.bool, device=pts.device)
            vmask[:pad_n] = True
            out.append(buf)
            valid.append(vmask)
        return torch.stack(out), torch.stack(valid)

    def forward(
        self,
        img: torch.Tensor,
        pc: torch.Tensor,
        T_pred: torch.Tensor,
        T_gt: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        img = self._prepare_img(img).float()
        if cam_intrinsic.dim() == 4:
            cam_intrinsic = cam_intrinsic.squeeze(1)
        b, _, h, w = img.shape
        ds = self.downsample
        h_s, w_s = max(1, h // ds), max(1, w // ds)
        img_s = F.interpolate(img, size=(h_s, w_s), mode='bilinear', align_corners=False)

        pts, pt_valid = self._sample_points(pc, mask)
        with torch.cuda.amp.autocast(enabled=False):
            cam_gt = transform_points_se3(pts.float(), T_gt.float())
            cam_pred = transform_points_se3(pts.float(), T_pred.float())
            u_gt, v_gt, z_gt = project_cam_to_pixel(cam_gt, cam_intrinsic.float())
            u_pr, v_pr, z_pr = project_cam_to_pixel(cam_pred, cam_intrinsic.float())

            u_gt_s = u_gt / ds
            v_gt_s = v_gt / ds
            u_pr_s = u_pr / ds
            v_pr_s = v_pr / ds

            in_bounds = (
                pt_valid
                & (z_gt > 0.5) & (z_pr > 0.5)
                & (u_gt_s >= 0) & (u_gt_s < w_s) & (v_gt_s >= 0) & (v_gt_s < h_s)
                & (u_pr_s >= 0) & (u_pr_s < w_s) & (v_pr_s >= 0) & (v_pr_s < h_s)
            )

            photo_loss = pts.new_tensor(0.0)
            depth_loss = pts.new_tensor(0.0)
            ssim_loss = pts.new_tensor(0.0)
            valid_ratio = pts.new_tensor(0.0)

            if in_bounds.any():
                colors = sample_image_at_pixels(img_s, u_gt_s, v_gt_s)
                colors = colors * in_bounds.unsqueeze(-1).float()

                render, cov = _bilinear_splat(colors, u_pr_s, v_pr_s, h_s, w_s)
                cov_mask = (cov >= self.min_coverage / (ds * ds)).float()

                l1_map = (render - img_s).abs().mean(dim=1, keepdim=True)
                photo_loss = (l1_map * cov_mask).sum() / cov_mask.sum().clamp(min=1.0)

                if self.lambda_ssim > 0 and cov_mask.sum() > 0:
                    win = _gaussian_window(self.ssim_window, 1.5, img_s.device, img_s.dtype)
                    ssim_map = _ssim_map(render, img_s, win)
                    ssim_loss = ((1.0 - ssim_map) * cov_mask).sum() / cov_mask.sum().clamp(min=1.0)

                depth_rel = (z_pr - z_gt).abs() / z_gt.clamp(min=0.5)
                depth_loss = (depth_rel * in_bounds.float()).sum() / in_bounds.float().sum().clamp(min=1.0)
                valid_ratio = in_bounds.float().sum() / pt_valid.float().sum().clamp(min=1.0)

            total = (1.0 - self.lambda_ssim) * photo_loss + self.lambda_ssim * ssim_loss
            return {
                'lsp_loss': total,
                'lsp_photo': photo_loss.detach(),
                'lsp_ssim': ssim_loss.detach(),
                'lsp_depth': depth_loss.detach(),
                'lsp_valid_ratio': valid_ratio.detach(),
            }
