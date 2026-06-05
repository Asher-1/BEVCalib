"""CorrespondenceHead: sparse 2D-3D matching from fusion tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camera_geometry import project_cam_to_pixel, transform_points_se3


class CorrespondenceHead(nn.Module):
    """Predict K sparse correspondences from group tokens + T_init anchor."""

    def __init__(
        self,
        token_dim: int = 384,
        num_points: int = 64,
        hidden: int = 256,
        use_gt_supervision: bool = True,
        confidence_threshold: float = 0.2,
        corr_validity_mode: str = 'gt',
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.use_gt_supervision = bool(use_gt_supervision)
        self.confidence_threshold = float(confidence_threshold)
        self.corr_validity_mode = str(corr_validity_mode)

        self.score_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.offset_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    MAX_UV_DELTA = 32.0

    @staticmethod
    def _project_groups(
        xyz_lidar: torch.Tensor,
        T: torch.Tensor,
        K: torch.Tensor,
        sensor_h: int,
        sensor_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz_cam = transform_points_se3(xyz_lidar, T)
        u, v, z = project_cam_to_pixel(xyz_cam, K)
        in_bounds = (
            (u >= 0) & (u < sensor_w) & (v >= 0) & (v < sensor_h) & (z > 0.5)
        )
        return torch.stack([u, v], dim=-1), in_bounds

    def forward(
        self,
        pc_groups_xyz: torch.Tensor,
        fusion_tokens: torch.Tensor,
        T_init: torch.Tensor,
        K: torch.Tensor,
        sensor_h: int,
        sensor_w: int,
        T_gt: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            pc_groups_xyz: B×G×3 LiDAR (denormalized)
            fusion_tokens: B×G×D
        Returns dict with uv, xyz, confidence, validity, match_valid_ratio_init/gt
        """
        b, g, _ = fusion_tokens.shape
        k = min(self.num_points, g)

        scores = self.score_mlp(fusion_tokens).squeeze(-1)
        _, top_idx = torch.topk(scores, k=k, dim=1)
        batch_idx = torch.arange(b, device=fusion_tokens.device).unsqueeze(1).expand(-1, k)
        tok_k = fusion_tokens[batch_idx, top_idx]
        xyz_k = pc_groups_xyz[batch_idx, top_idx]

        uv_init, valid_init = self._project_groups(
            xyz_k, T_init, K, sensor_h, sensor_w)

        offset_raw = self.offset_mlp(tok_k)
        delta_uv = torch.tanh(offset_raw[..., :2]) * self.MAX_UV_DELTA
        log_conf = offset_raw[..., 2]
        confidence = torch.sigmoid(log_conf)
        uv_pred = uv_init + delta_uv
        uv_pred = torch.stack([
            uv_pred[..., 0].clamp(0.0, float(sensor_w - 1)),
            uv_pred[..., 1].clamp(0.0, float(sensor_h - 1)),
        ], dim=-1)

        finite = (
            torch.isfinite(uv_init).all(dim=-1)
            & torch.isfinite(uv_pred).all(dim=-1)
        )
        validity = valid_init & finite & (confidence > self.confidence_threshold)
        match_valid_ratio_init = validity.float().mean(dim=1)

        out = {
            'uv': uv_pred,
            'xyz': xyz_k,
            'confidence': confidence,
            'validity': validity,
            'match_valid_ratio_init': match_valid_ratio_init,
            'top_idx': top_idx,
        }

        if self.use_gt_supervision and T_gt is not None:
            with torch.no_grad():
                uv_gt, valid_gt = self._project_groups(
                    xyz_k, T_gt, K, sensor_h, sensor_w)
            out['uv_gt'] = uv_gt
            out['validity_gt'] = valid_gt
            validity_gt = validity & valid_gt
            out['validity_gt'] = validity_gt
            if self.corr_validity_mode == 'init':
                # L_corr: supervise vs T_gt UV but keep points visible under T_init.
                out['validity'] = validity
            else:
                out['validity'] = validity_gt
            match_valid_ratio_gt = validity_gt.float().mean(dim=1)
        else:
            match_valid_ratio_gt = match_valid_ratio_init

        out['match_valid_ratio_gt'] = match_valid_ratio_gt
        # Alias: gate/EPnP weights use GT-masked ratio when supervision is on.
        out['match_valid_ratio'] = match_valid_ratio_gt

        return out


def correspondence_loss(
    uv_pred: torch.Tensor,
    uv_gt: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 on valid correspondences."""
    if validity.any():
        diff = F.smooth_l1_loss(uv_pred[validity], uv_gt[validity], reduction='mean')
    else:
        diff = uv_pred.sum() * 0.0
    return diff
