"""ProjFusion AttenDualFusion branch for HTCN."""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from camera_geometry import scale_intrinsics_for_resize

DEFAULT_PROJFUSION_ROOT = os.environ.get(
    "PROJFUSION_ROOT", "/mnt/drtraining/user/dahailu/code/ProjFusion")


def build_camera_info(cam_intrinsic: torch.Tensor, sensor_h: int, sensor_w: int) -> Dict:
    """Build ProjFusion BatchedCameraInfoDict from BEVCalib intrinsics."""
    if cam_intrinsic.dim() == 4:
        cam_intrinsic = cam_intrinsic.squeeze(1)
    B = cam_intrinsic.shape[0]
    return {
        'fx': cam_intrinsic[:, 0, 0],
        'fy': cam_intrinsic[:, 1, 1],
        'cx': cam_intrinsic[:, 0, 2],
        'cy': cam_intrinsic[:, 1, 2],
        'sensor_h': sensor_h,
        'sensor_w': sensor_w,
        'projection_mode': 'perspective',
    }


def _ensure_projfusion_path(proj_root: str):
    proj_root = os.path.abspath(proj_root)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    return proj_root


class ProjFusionBranch(nn.Module):
    """AttenDualFusion feature extractor (trainable cross-attn, frozen encoders)."""

    def __init__(self,
                 image_hw: Tuple[int, int] = (224, 448),
                 pointgpt_ckpt: str | None = None,
                 pointgpt_config: str | None = None,
                 projfusion_root: str = DEFAULT_PROJFUSION_ROOT,
                 freeze_encoders: bool = True,
                 margin: float = 2.0,
                 skip_internal_pointgpt: bool = False,
                 pointgpt_embed_dim: int = 384,
                 pointgpt_max_depth: float = 60.0,
                 explicit_k_vit_scale: bool = False):
        super().__init__()
        self.explicit_k_vit_scale = explicit_k_vit_scale
        proj_root = _ensure_projfusion_path(projfusion_root)
        if pointgpt_ckpt is None:
            pointgpt_ckpt = os.path.join(proj_root, 'pretrained/fleet_pointgpt_L20.pth')
        if pointgpt_config is None:
            pointgpt_config = os.path.join(proj_root, 'cfg/pointgpt/finetune_fleet_L20.yaml')
        pointgpt_ckpt = os.path.abspath(pointgpt_ckpt)
        pointgpt_config = os.path.abspath(pointgpt_config)

        from models.tools.core import AttenDualFusionNet

        cwd = os.getcwd()
        try:
            os.chdir(proj_root)
            encoder_argv = dict(
                image_hw=image_hw,
                use_coord=True,
                use_harmonic=True,
                harmonic_args=dict(
                    n_harmonic_functions=6, omega_0=0.33, logspace=True, append_input=True),
                margin=float(margin),
                use_mask=False,
                attention_type='Attention',
                attention_argv=dict(heads=6, dim_head=64, dropout=0.0),
                freeze_encoders=freeze_encoders,
                output_type='1d',
                img_encoder_type='vit',
                img_encoder_args=dict(
                    cache_dir=os.path.join(proj_root, '.cache/torch/hub/'),
                    modelname='dinov2_vits14',
                    source='local',
                ),
                pointgpt_config=pointgpt_config,
                pointgpt_checkpoint=pointgpt_ckpt,
            )
            if skip_internal_pointgpt:
                encoder_argv.update(
                    skip_pointgpt=True,
                    pointgpt_embed_dim=int(pointgpt_embed_dim),
                    pointgpt_max_depth_override=float(pointgpt_max_depth),
                )
            self.encoder = AttenDualFusionNet(**encoder_argv)
            self.encoder._lazy_init()
            _vit = self.encoder.fnet_2d
            if hasattr(_vit, 'module'):
                _vit = _vit.module
            if hasattr(_vit, '_net'):
                print(f"[ProjViTEncoder] hub=dinov2_vits14 (local), "
                      f"embed_dim={_vit.get_output_dim()}, VERIFY PASS")
        finally:
            os.chdir(cwd)
        self.out_dim = self.encoder.out_dim
        self.image_hw = image_hw
        self.register_buffer(
            '_imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self._requires_shared_pointgpt = skip_internal_pointgpt
        if skip_internal_pointgpt:
            print(f"[ProjFusionBranch] image_hw={image_hw}, out_dim={self.out_dim}, "
                  f"margin={margin}, shared PointGPT (embed={pointgpt_embed_dim}, "
                  f"max_depth={pointgpt_max_depth})")
        else:
            print(f"[ProjFusionBranch] image_hw={image_hw}, out_dim={self.out_dim}, "
                  f"margin={margin}, pointgpt={pointgpt_ckpt}")

    def preprocess_image(self, img: torch.Tensor) -> torch.Tensor:
        """Resize BEVCalib image (B,3,H,W) to ViT input."""
        img_vit, _, _, _ = self.preprocess_image_and_intrinsics(img, None)
        return img_vit

    def preprocess_image_and_intrinsics(
        self,
        img: torch.Tensor,
        cam_intrinsic: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        """Resize image to ViT resolution; optionally scale K to match (V40 M-1).

        Returns:
            img_vit, K_for_proj, sensor_h, sensor_w for AttenDualFusion projection.
            When explicit_k_vit_scale=False (V39 default), K is unchanged and
            sensor_h/w refer to the **input** image size (legacy ProjFusion path).
        """
        th, tw = self.image_hw
        x = F.interpolate(img, size=(th, tw), mode='bilinear', align_corners=False)
        img_vit = (x - self._imagenet_mean) / self._imagenet_std
        if not self.explicit_k_vit_scale or cam_intrinsic is None:
            sensor_h, sensor_w = img.shape[2], img.shape[3]
            return img_vit, cam_intrinsic, sensor_h, sensor_w
        k_vit = scale_intrinsics_for_resize(
            cam_intrinsic, img.shape[2], img.shape[3], th, tw)
        return img_vit, k_vit, th, tw

    def _proj_camera_info(
        self,
        img: torch.Tensor,
        cam_intrinsic: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, dict]:
        if cam_intrinsic.dim() == 4:
            cam_intrinsic = cam_intrinsic.squeeze(1)
        img_vit, k_proj, sensor_h, sensor_w = self.preprocess_image_and_intrinsics(
            img, cam_intrinsic)
        camera_info = build_camera_info(k_proj, sensor_h, sensor_w)
        return img_vit, k_proj, sensor_h, sensor_w, camera_info

    def _build_cache(self, img_vit: torch.Tensor, pcd: torch.Tensor,
                     xyz_groups: torch.Tensor | None = None,
                     feat_groups: torch.Tensor | None = None):
        """Build AttenDualFusion encoder cache, optionally reusing shared PointGPT groups."""
        from einops import rearrange

        with torch.no_grad():
            feat_2d = self.encoder.fnet_2d(img_vit)
            feat_2d = rearrange(feat_2d, 'b c n h -> b (n h) c')
            if xyz_groups is not None and feat_groups is not None:
                max_depth = float(getattr(self.encoder, 'fnet_3d_max_depth', 60.0))
                xyz_norm = xyz_groups / max_depth
                return {
                    'feat_2d': feat_2d.detach(),
                    'feat_3d': feat_groups.detach(),
                    'xyz': xyz_norm.detach(),
                }
            if getattr(self, '_requires_shared_pointgpt', False) or self.encoder.fnet_3d is None:
                raise RuntimeError(
                    "ProjFusionBranch has no internal PointGPT; "
                    "use forward_with_shared_point_features()")
            pcd_in = pcd if pcd.shape[-1] == 3 else pcd.transpose(1, 2).contiguous()
            return self.encoder.encoder_cache(img_vit, pcd_in)

    @staticmethod
    def _pool_proj_feat(rot_feat: torch.Tensor) -> torch.Tensor:
        if rot_feat.dim() == 4:
            return rot_feat.mean(dim=(2, 3))
        return rot_feat.mean(dim=1)

    def forward_from_cache(
        self,
        cache: dict,
        t_init: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        sensor_h: int,
        sensor_w: int,
    ) -> tuple[torch.Tensor, dict]:
        """Cross-attn + pool only; reuse encoder cache from a prior forward on same img/pc.

        Does not run DINOv2/PointGPT or clear feat_buffer. Grad flows through
        cache_forward (T_init-dependent projection) and downstream heads only.
        """
        if t_init.dim() == 4:
            t_init = t_init.squeeze(1)
        if cam_intrinsic.dim() == 4:
            cam_intrinsic = cam_intrinsic.squeeze(1)
        camera_info = build_camera_info(cam_intrinsic, sensor_h, sensor_w)
        rot_feat, _tsl_feat = self.encoder.cache_forward(cache, t_init, camera_info)
        return self._pool_proj_feat(rot_feat), cache

    def forward_with_shared_point_features(self, img_vit: torch.Tensor,
                                           t_init: torch.Tensor,
                                           cam_intrinsic: torch.Tensor,
                                           sensor_h: int, sensor_w: int,
                                           xyz_groups: torch.Tensor,
                                           feat_groups: torch.Tensor):
        """Proj forward with PointGPT groups computed once by HTCN."""
        if t_init.dim() == 4:
            t_init = t_init.squeeze(1)
        if cam_intrinsic.dim() == 4:
            cam_intrinsic = cam_intrinsic.squeeze(1)
        camera_info = build_camera_info(cam_intrinsic, sensor_h, sensor_w)
        self.encoder.clear_buffer()
        cache = self._build_cache(
            img_vit, None, xyz_groups=xyz_groups, feat_groups=feat_groups)
        rot_feat, _tsl_feat = self.encoder.cache_forward(cache, t_init, camera_info)
        return self._pool_proj_feat(rot_feat), cache

    def forward(self, img_or_vit: torch.Tensor, pcd: torch.Tensor,
                t_init: torch.Tensor, cam_intrinsic: torch.Tensor,
                sensor_h: int, sensor_w: int):
        """
        Returns:
            f_proj: (B, out_dim) pooled projection-aligned features
            cache: encoder cache for optional reuse

        When explicit_k_vit_scale=True, pass raw img (B,3,H,W); K is scaled internally.
        When False (V39), pass preprocessed img_vit from preprocess_image().
        """
        if getattr(self, '_requires_shared_pointgpt', False):
            raise RuntimeError(
                "ProjFusionBranch internal PointGPT stripped; "
                "use forward_with_shared_point_features()")
        if t_init.dim() == 4:
            t_init = t_init.squeeze(1)
        if self.explicit_k_vit_scale:
            img_vit, _, _, _, camera_info = self._proj_camera_info(
                img_or_vit, cam_intrinsic)
        else:
            img_vit = img_or_vit
            if cam_intrinsic.dim() == 4:
                cam_intrinsic = cam_intrinsic.squeeze(1)
            camera_info = build_camera_info(cam_intrinsic, sensor_h, sensor_w)
        pcd_in = pcd if pcd.shape[-1] == 3 else pcd.transpose(1, 2).contiguous()

        self.encoder.clear_buffer()
        cache = self._build_cache(img_vit, pcd_in)
        rot_feat, _tsl_feat = self.encoder.cache_forward(cache, t_init, camera_info)
        return self._pool_proj_feat(rot_feat), cache
