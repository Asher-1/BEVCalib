"""PointGPT pretrained encoder wrapper (loads from ProjFusion)."""

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_PROJFUSION_ROOT = os.environ.get(
    "PROJFUSION_ROOT",
    "/mnt/drtraining/user/dahailu/code/ProjFusion",
)
DEFAULT_POINTGPT_CONFIG = os.path.join(
    DEFAULT_PROJFUSION_ROOT, "cfg/pointgpt/finetune_kitti_tiny.yaml")
DEFAULT_POINTGPT_CKPT = os.path.join(
    DEFAULT_PROJFUSION_ROOT, "pretrained/kitti_pointgpt_tiny.pth")


def _load_pointgpt_model(config_path: str, checkpoint_path: str):
    config_path = os.path.abspath(config_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    proj_root = os.environ.get("PROJFUSION_ROOT") or os.path.dirname(
        os.path.dirname(os.path.dirname(config_path)))
    proj_root = os.path.abspath(proj_root)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    cwd = os.getcwd()
    try:
        os.chdir(proj_root)
        from models.pointgpt import load_pointgpt
        return load_pointgpt(config_path, checkpoint_path)
    finally:
        os.chdir(cwd)


class PointGPTEncoder(nn.Module):
    """Frozen PointGPT feature extractor aligned with ProjFusion."""

    def __init__(self,
                 config_path: str = DEFAULT_POINTGPT_CONFIG,
                 checkpoint_path: str = DEFAULT_POINTGPT_CKPT,
                 max_depth: float = 50.0,
                 n_points: int = 8192,
                 freeze: bool = True):
        super().__init__()
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"PointGPT checkpoint not found: {checkpoint_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"PointGPT config not found: {config_path}")

        model, cfg_max_depth = _load_pointgpt_model(config_path, checkpoint_path)
        self.model = model
        self.max_depth = float(cfg_max_depth if max_depth is None else max_depth)
        self.n_points = n_points
        self.n_groups = model.num_group
        self.out_dim = model.trans_dim

        if freeze:
            self._freeze = True
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            self._freeze = False

        print(f"[PointGPTEncoder] loaded ckpt={checkpoint_path}, "
              f"groups={self.n_groups}, dim={self.out_dim}, "
              f"max_depth={self.max_depth}, freeze={freeze}")

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, '_freeze', False):
            self.model.eval()
        return self

    def _prepare_points(self, pts: torch.Tensor,
                        mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, device=pts.device)
            valid = mask.bool()
            if valid.any():
                pts = pts[valid]
        sentinel = (pts.abs() < 900.0).all(dim=-1)
        pts = pts[sentinel]
        if pts.shape[0] == 0:
            pts = torch.zeros(1, 3, device=pts.device, dtype=pts.dtype)
        n = pts.shape[0]
        if n >= self.n_points:
            idx = torch.linspace(0, n - 1, self.n_points, device=pts.device).long()
            pts = pts[idx]
        elif n < self.n_points:
            pad = pts[-1:].expand(self.n_points - n, -1)
            pts = torch.cat([pts, pad], dim=0)
        return pts

    def forward(self, pcd: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pcd: (B, N, 3) metric point cloud
            mask: (B, N) optional validity mask
        Returns:
            xyz_groups: (B, G, 3) metric group centers
            feat_groups: (B, G, D)
        """
        batch_pts = []
        for b in range(pcd.shape[0]):
            m = mask[b] if mask is not None else None
            batch_pts.append(self._prepare_points(pcd[b], m))
        pts_stack = torch.stack(batch_pts, dim=0)
        pts_norm = pts_stack / self.max_depth

        frozen = not any(p.requires_grad for p in self.model.parameters())
        ctx = torch.no_grad() if frozen else torch.enable_grad()
        with ctx:
            center, feat = self.model.extraction(pts_norm)

        return center * self.max_depth, feat
