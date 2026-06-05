"""Correlation alignment and sequence consistency losses for V42 CF-BEV-R."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_projection_v42(
    xyz: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
    return_depth: bool = False,
):
    """Project 3D points through extrinsic T and intrinsic K.

    Args:
        xyz: (B, G, 3) or (G, 3) points in LiDAR frame
        T: (B, 4, 4) extrinsic matrix (LiDAR to camera)
        K: (B, 3, 3) intrinsic matrix
        return_depth: if True, also return depth values

    Returns:
        uv: (B, G, 2) pixel coordinates
        depth: (B, G) depth values (only if return_depth=True)
    """
    squeeze_batch = False
    if xyz.dim() == 2:
        xyz = xyz.unsqueeze(0)
        squeeze_batch = True

    b = xyz.shape[0]
    if T.dim() == 2:
        T = T.unsqueeze(0).expand(b, -1, -1)
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(b, -1, -1)

    _, g, _ = xyz.shape
    ones = torch.ones(b, g, 1, device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    xyz_cam = torch.bmm(xyz_h, T.transpose(1, 2))[:, :, :3]

    depth = xyz_cam[..., 2].clamp(min=1e-3)
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    u = fx * xyz_cam[..., 0] / depth + cx
    v = fy * xyz_cam[..., 1] / depth + cy
    uv = torch.stack([u, v], dim=-1)

    if squeeze_batch:
        uv = uv.squeeze(0)
        depth = depth.squeeze(0)

    if return_depth:
        return uv, depth
    return uv


class CorrelationAlignmentLoss(nn.Module):
    """Supervise RoCR correlation peak offset against GT pixel displacement."""

    def __init__(self, weight: float = 0.3, warmup_epochs: int = 20, huber_delta: float = 1.0):
        super().__init__()
        self.weight = weight
        self.warmup_epochs = warmup_epochs
        self.huber_delta = huber_delta

    def _warmup_factor(self, current_epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, max(0.01, float(current_epoch) / float(self.warmup_epochs)))

    def forward(
        self,
        delta_uv_pred: torch.Tensor,
        T_gt: torch.Tensor,
        T_init: torch.Tensor,
        xyz_groups: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        valid_mask: torch.Tensor,
        patch_size: float,
        current_epoch: int = 0,
        corr_radius: int = 4,
    ) -> torch.Tensor:
        with torch.no_grad():
            uv_gt = compute_projection_v42(xyz_groups, T_gt, cam_intrinsic)
            uv_init = compute_projection_v42(xyz_groups, T_init, cam_intrinsic)
            delta_uv_gt_raw = (uv_gt - uv_init) / patch_size
            delta_uv_gt = delta_uv_gt_raw.clamp(-corr_radius, corr_radius)

        per_elem = F.huber_loss(
            delta_uv_pred,
            delta_uv_gt,
            reduction='none',
            delta=self.huber_delta,
        )
        per_group = per_elem.sum(dim=-1)

        mask = valid_mask.float()
        loss = (per_group * mask).sum() / mask.sum().clamp(min=1.0)

        return self.weight * self._warmup_factor(current_epoch) * loss


class SequenceConsistencyLoss(nn.Module):
    """Encourage smooth quaternion predictions across consecutive frames (S3+)."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, q_pred_list: list[torch.Tensor]) -> torch.Tensor:
        if len(q_pred_list) < 2:
            ref = q_pred_list[0] if q_pred_list else None
            if ref is None:
                return torch.tensor(0.0)
            return ref.new_tensor(0.0)

        losses = []
        for i in range(len(q_pred_list) - 1):
            losses.append(F.l1_loss(q_pred_list[i], q_pred_list[i + 1]))
        total = torch.stack(losses).mean()

        return self.weight * total
