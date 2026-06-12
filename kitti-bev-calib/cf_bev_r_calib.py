"""CF-BEV-R: CalibFormer-aligned Rotation-only Calibration Model (V42).

fusion_backend = 'cf_bev_r'

Architecture (docs/V42_CF_BEV_DESIGN.md §2):
  Stage-1: Swin-Tiny+FPN+DLA → F_rgb;  FPS+kNN → (xyz_g, F_pc);  T_init project → uv_init
  Stage-2: ExtrinsicAware CrossAttn + LocalMultiHeadCorrelationV2 → F_cross, corr_map
  Stage-3: RoCR → R_geo;  SwinCorrEncoder+PoseQueryDecoder → Δq;  compose q_pred
  Stage-3b: FrontViewPitchBranch (optional)
  Output: rotation quaternion (B,4), translation zeros (B,3)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from native_cross_attention import (
    HarmonicEmbedding,
    PointEncoder,
    CrossAttentionBlock,
)
from img_branch.img_encoders import SwinT_tiny_Encoder
from losses.quat_tools import batch_quat2mat, batch_tvector2mat
from losses.losses import realworld_loss

from modules.dla_aggregation import DLAAggregation
from modules.local_correlation_v2 import LocalMultiHeadCorrelationV2
from modules.rocr_refine import RoCR
from modules.pose_query_init import PoseQueryInit
from modules.corr_transformer_decoder import CorrTransformerHead
from losses.corr_alignment_loss import compute_projection_v42


class GatedInstanceNorm(nn.Module):
    """Gated Instance Normalization (V45) with optional Partial GIN (V48).

    Learns per-channel gate to balance IN (domain removal) and identity (domain preservation).
    gate ≈ 0 → full IN (cross-domain generalization, lower ZD)
    gate ≈ 1 → bypass IN (within-domain consistency, better recovery)

    Partial GIN (V48): when gin_channels < channels, only the first gin_channels
    go through GIN; the remaining channels use LayerNorm to preserve perturbation
    sensitivity for better recovery while the GIN portion controls zero-drift.
    """

    def __init__(self, channels: int, init_gate: float = 0.5,
                 gin_channels: int = 0, gate_reg_target: float = 0.0):
        super().__init__()
        self.channels = channels
        self.gin_channels = gin_channels if gin_channels > 0 else channels
        self.bypass_channels = channels - self.gin_channels
        self.gate_reg_target = gate_reg_target

        self.in_norm = nn.InstanceNorm2d(self.gin_channels, affine=True)
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, self.gin_channels),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_fc[-2].bias, -math.log(1.0 / init_gate - 1.0))

        if self.bypass_channels > 0:
            self.bypass_norm = nn.LayerNorm(self.bypass_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bypass_channels <= 0:
            x_normed = self.in_norm(x)
            gate = self.gate_fc(x).unsqueeze(-1).unsqueeze(-1)
            self._last_gate_stats = {
                "mean": gate.detach().mean().item(),
                "std": gate.detach().std().item(),
                "min": gate.detach().min().item(),
                "max": gate.detach().max().item(),
            }
            return x * gate + x_normed * (1 - gate)

        x_gin = x[:, :self.gin_channels]
        x_bypass = x[:, self.gin_channels:]

        x_gin_normed = self.in_norm(x_gin)
        gate = self.gate_fc(x).unsqueeze(-1).unsqueeze(-1)
        self._last_gate_stats = {
            "mean": gate.detach().mean().item(),
            "std": gate.detach().std().item(),
            "min": gate.detach().min().item(),
            "max": gate.detach().max().item(),
        }
        x_gin_out = x_gin * gate + x_gin_normed * (1 - gate)

        x_bypass_out = self.bypass_norm(x_bypass.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return torch.cat([x_gin_out, x_bypass_out], dim=1)

    def gate_reg_loss(self) -> torch.Tensor:
        """Regularization loss to prevent gate from collapsing to extremes."""
        if self.gate_reg_target <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        gate_bias = self.gate_fc[-2].bias
        gate_vals = torch.sigmoid(gate_bias)
        return ((gate_vals - self.gate_reg_target) ** 2).mean()


class CFBevRCalib(nn.Module):
    """CF-BEV-R main model.

    Args:
        img_shape: (H, W) original image size.
        feat_dim: unified feature dimension (default 256).
        n_groups: FPS point groups (default 128).
        knn: k-nearest neighbours for PointEncoder.
        num_corr_heads: local correlation heads.
        default_corr_radius: default correlation window radius.
        num_queries: pose query count.
        encoder_layers: SwinCorrEncoder layers.
        decoder_layers: PoseQueryDecoder layers.
        use_rocr: enable RoCR geometric initialization.
        rocr_dropout: RoCR training dropout (AS4).
        rocr_center_bias: center negative bias (AS3).
        use_pitch_branch: enable FrontViewPitchBranch.
        rotation_only: only predict rotation (fixed translation).
        backbone_lr_scale: learning rate multiplier for Swin backbone.
        use_dla: enable DLA multi-scale aggregation (V44 P0-B).
        use_pitch_fusion: enable Pitch inference fusion (V44 P0-A).
        use_instance_norm: enable Instance Normalization for domain alignment (V44 Phase 2).
        use_gated_instance_norm: enable Gated IN for V45 (overrides use_instance_norm).
        gin_init_gate: initial gate value for GIN (0.5 = balanced).
        gin_channels: channels to apply GIN to (0 = all). V48 Partial GIN uses gin_channels < feat_dim.
        gin_gate_reg_target: regularization target for gate values (0 = disabled).
    """

    def __init__(
        self,
        img_shape: Tuple[int, int] = (360, 640),
        feat_dim: int = 256,
        n_groups: int = 128,
        knn: int = 8,
        num_corr_heads: int = 4,
        default_corr_radius: int = 4,
        num_queries: int = 6,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        use_rocr: bool = True,
        rocr_dropout: float = 0.3,
        rocr_center_bias: float = 0.5,
        use_pitch_branch: bool = True,
        pitch_aux_weight: float = 0.3,
        rotation_only: bool = True,
        backbone_lr_scale: float = 0.1,
        use_dla: bool = False,
        use_pitch_fusion: bool = False,
        use_instance_norm: bool = False,
        use_gated_instance_norm: bool = False,
        gin_init_gate: float = 0.5,
        gin_channels: int = 0,
        gin_gate_reg_target: float = 0.0,
        enable_axis_loss: bool = True,
        weight_axis_rotation: float = 0.3,
        axis_weights: Tuple[float, ...] = (1.0, 4.0, 1.0),
        use_geodesic_loss: bool = False,
        weight_quat_norm: float = 0.5,
        head_dropout: float = 0.1,
        use_magnitude_head: bool = False,
        pitch_vertical_bands: int = 3,
        decoder_pool_mode: str = "mean",
        use_dp_head: bool = False,
        dp_gate_deg: float = 1.5,
        route_loss_weight: float = 0.0,
        use_jacg: bool = True,
        jacg_hidden_dim: int = 64,
        bias_path_in_norm: bool = True,
        use_adir: bool = False,
        adir_steps: int = 2,
        adir_max_step_deg: float = 1.0,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.feat_dim = feat_dim
        self.rotation_only = rotation_only
        self.use_rocr = use_rocr
        self.use_pitch_branch = use_pitch_branch
        self.pitch_aux_weight = pitch_aux_weight
        self.backbone_lr_scale = backbone_lr_scale
        self.use_dla = use_dla
        self.use_pitch_fusion = use_pitch_fusion
        self.use_instance_norm = use_instance_norm
        self.use_gated_instance_norm = use_gated_instance_norm

        img_H, img_W = img_shape
        self.feat_h = img_H // 4
        self.feat_w = img_W // 4
        self.patch_size = 4.0

        fpn_in_channels = [192, 384, 768]
        fpn_out_channels = 256
        feat_shape = (fpn_out_channels, img_H // 8, img_W // 8)

        self.img_encoder = SwinT_tiny_Encoder(
            output_indices=[1, 2, 3],
            featureShape=feat_shape,
            out_channels=fpn_out_channels,
            FPN_in_channels=fpn_in_channels,
            FPN_out_channels=fpn_out_channels,
        )

        self.img_proj = (
            nn.Conv2d(fpn_out_channels, feat_dim, 1)
            if fpn_out_channels != feat_dim else nn.Identity()
        )

        if use_dla:
            self.dla = DLAAggregation(
                in_channels_list=fpn_in_channels,
                out_channels=feat_dim,
            )
            self.img_encoder._return_multiscale = True
        else:
            self.dla = DLAAggregation(
                in_channels_list=[fpn_out_channels] * 3,
                out_channels=feat_dim,
            )

        if use_gated_instance_norm:
            self.feat_in = GatedInstanceNorm(
                feat_dim, init_gate=gin_init_gate,
                gin_channels=gin_channels, gate_reg_target=gin_gate_reg_target)
            gin_ch = gin_channels if gin_channels > 0 else feat_dim
            bypass_ch = feat_dim - gin_ch
            if bypass_ch > 0:
                print(f"[CFBevRCalib] Partial GIN enabled: {gin_ch}/{feat_dim} channels GIN, "
                      f"{bypass_ch} channels LayerNorm (init_gate={gin_init_gate})")
            else:
                print(f"[CFBevRCalib] Gated Instance Norm enabled (D={feat_dim}, init_gate={gin_init_gate})")
        elif use_instance_norm:
            self.feat_in = nn.Sequential(
                nn.InstanceNorm2d(feat_dim, affine=True),
                nn.GELU(),
                nn.Conv2d(feat_dim, feat_dim, 1, bias=False),
            )
        else:
            self.feat_in = None

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

        self.point_encoder = PointEncoder(
            in_dim=3, hidden_dim=64, out_dim=feat_dim,
            n_groups=n_groups, knn=knn, use_fps=True,
        )

        self.harmonic = HarmonicEmbedding(n_harmonic_functions=6)

        _heads, _dim_head = 8, 32
        _attn_out = _heads * _dim_head
        cross_blocks = []
        for i in range(2):
            in_dim = feat_dim if i == 0 else _attn_out
            cross_blocks.append(CrossAttentionBlock(
                img_feat_dim=in_dim, pc_feat_dim=feat_dim,
                n_harmonic=6, heads=_heads, dim_head=_dim_head,
                dropout=head_dropout, ffn_mult=4,
            ))
        self.cross_attn_blocks = nn.ModuleList(cross_blocks)
        self._cross_out_dim = _attn_out

        cross_out_dim = 8 * 32  # heads * dim_head from CrossAttentionBlock
        self.cross_proj = nn.Linear(cross_out_dim, feat_dim) \
            if cross_out_dim != feat_dim else nn.Identity()

        self.local_corr = LocalMultiHeadCorrelationV2(
            token_dim=feat_dim,
            num_heads=num_corr_heads,
            default_radius=default_corr_radius,
            out_dim=feat_dim,
        )

        if use_rocr:
            self.rocr = RoCR(
                min_valid_ratio=0.3,
                temperature=1.0,
                dropout_prob=rocr_dropout,
                center_neg_bias=rocr_center_bias,
            )

        self.pose_query_init = PoseQueryInit(
            rgb_dim=feat_dim,
            num_queries=num_queries,
            query_dim=feat_dim,
        )

        self.use_magnitude_head = use_magnitude_head
        self.corr_head = CorrTransformerHead(
            d_model=feat_dim,
            nhead=8,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dim_feedforward=feat_dim * 4,
            dropout=head_dropout,
            use_magnitude_head=self.use_magnitude_head,
            pool_mode=decoder_pool_mode,
        )

        self.use_dp_head = use_dp_head
        self.route_loss_weight = route_loss_weight
        self.dp_head = None
        if use_dp_head:
            from modules.dp_pose_head import DPPoseHead
            self.dp_head = DPPoseHead(
                recovery_head=self.corr_head,
                feat_dim=feat_dim,
                gate_deg=dp_gate_deg,
                use_jacg=use_jacg,
                jacg_hidden_dim=jacg_hidden_dim,
                bias_path_in_norm=bias_path_in_norm,
                head_dropout=head_dropout,
            )
            print(f"[CFBevRCalib] V53 DP-Head enabled (gate={dp_gate_deg}°, jacg={use_jacg})")

        self.use_adir = use_adir
        self.adir_refiner = None
        if use_adir:
            from modules.adir_refine import ADIRRefiner
            self.adir_refiner = ADIRRefiner(
                feat_dim=feat_dim,
                n_steps=adir_steps,
                max_step_deg=adir_max_step_deg,
                head_dropout=head_dropout,
                use_pitch_branch=use_pitch_branch,
            )
            print(f"[CFBevRCalib] V53b ADIR enabled (steps={adir_steps}, max_step={adir_max_step_deg}°)")

        self.use_pitch_vertical_bands = pitch_vertical_bands
        if use_pitch_branch:
            try:
                from bev_calib import FrontViewPitchBranch
                _use_vb = self.use_pitch_vertical_bands > 1
                self.pitch_branch = FrontViewPitchBranch(
                    z_feat_dim=feat_dim, img_feat_dim=feat_dim,
                    use_vertical_bands=_use_vb,
                    n_vertical_bands=self.use_pitch_vertical_bands if _use_vb else 1)
            except ImportError:
                self.pitch_branch = None
                self.use_pitch_branch = False
        else:
            self.pitch_branch = None

        if use_pitch_fusion and self.pitch_branch is not None:
            self.pitch_conf_net = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, 1),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                self.pitch_conf_net[-2].bias.fill_(-1.0)

        self.loss_fn = realworld_loss(
            rotation_only=rotation_only,
            enable_axis_loss=enable_axis_loss,
            weight_axis_rotation=weight_axis_rotation,
            axis_weights=axis_weights,
            use_geodesic_loss=use_geodesic_loss,
            weight_quat_norm=weight_quat_norm,
        )

    @classmethod
    def from_args(cls, args, img_shape: Tuple[int, int] = (360, 640)):
        axis_weights_str = getattr(args, 'axis_weights', '1.0,4.0,1.0')
        axis_weights = tuple(float(x) for x in axis_weights_str.split(','))
        return cls(
            img_shape=img_shape,
            feat_dim=getattr(args, 'cf_feat_dim', 256),
            n_groups=getattr(args, 'cf_n_groups', 128),
            knn=getattr(args, 'cf_knn', 8),
            num_corr_heads=getattr(args, 'cf_corr_heads', 4),
            default_corr_radius=getattr(args, 'cf_corr_radius', 4),
            num_queries=getattr(args, 'cf_num_queries', 6),
            encoder_layers=getattr(args, 'cf_encoder_layers', 2),
            decoder_layers=getattr(args, 'cf_decoder_layers', 4),
            use_rocr=getattr(args, 'use_rocr', 1) > 0,
            rocr_dropout=getattr(args, 'rocr_dropout', 0.3),
            rocr_center_bias=getattr(args, 'rocr_center_bias', 0.5),
            use_pitch_branch=getattr(args, 'use_pitch_branch', 1) > 0,
            pitch_aux_weight=getattr(args, 'pitch_aux_weight', 0.3),
            rotation_only=getattr(args, 'rotation_only', True),
            backbone_lr_scale=getattr(args, 'backbone_lr_scale', 0.1),
            use_dla=getattr(args, 'use_dla', 0) > 0,
            use_pitch_fusion=getattr(args, 'use_pitch_fusion', 0) > 0,
            use_instance_norm=getattr(args, 'use_instance_norm', 0) > 0,
            use_gated_instance_norm=getattr(args, 'use_gated_instance_norm', 0) > 0,
            gin_init_gate=getattr(args, 'gin_init_gate', 0.5),
            gin_channels=getattr(args, 'gin_channels', 0),
            gin_gate_reg_target=getattr(args, 'gin_gate_reg_target', 0.0),
            enable_axis_loss=getattr(args, 'enable_axis_loss', 1) > 0,
            weight_axis_rotation=getattr(args, 'weight_axis_rotation', 0.3),
            axis_weights=axis_weights,
            use_geodesic_loss=getattr(args, 'use_geodesic_loss', 0) > 0,
            weight_quat_norm=getattr(args, 'quat_norm_weight', 0.5),
            head_dropout=getattr(args, 'head_dropout', 0.1),
            use_magnitude_head=getattr(args, 'use_magnitude_head', 0) > 0,
            pitch_vertical_bands=getattr(args, 'pitch_vertical_bands', 3),
            decoder_pool_mode=getattr(args, 'decoder_pool_mode', 'mean'),
            use_dp_head=getattr(args, 'use_dp_head', 0) > 0,
            dp_gate_deg=getattr(args, 'dp_gate_deg', 1.5),
            route_loss_weight=getattr(args, 'route_loss_weight', 0.0),
            use_jacg=getattr(args, 'use_jacg', 1) > 0,
            jacg_hidden_dim=getattr(args, 'jacg_hidden_dim', 64),
            bias_path_in_norm=getattr(args, 'bias_path_in_norm', 1) > 0,
            use_adir=getattr(args, 'use_adir', 0) > 0,
            adir_steps=getattr(args, 'adir_steps', 2),
            adir_max_step_deg=getattr(args, 'adir_max_step_deg', 1.0),
        )

    def get_param_groups(self, base_lr: float):
        """Return parameter groups with separate LR for backbone."""
        backbone_params = list(self.img_encoder.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {'params': other_params, 'lr': base_lr},
            {'params': backbone_params, 'lr': base_lr * self.backbone_lr_scale},
        ]

    def _compute_uv_feat(
        self, xyz_groups, T_init, cam_intrinsic,
    ):
        """Project 3D groups to feature-map coordinates."""
        uv_px = compute_projection_v42(xyz_groups, T_init, cam_intrinsic)
        uv_feat = uv_px / self.patch_size
        return uv_feat, uv_px

    def forward(
        self,
        imgs: torch.Tensor,
        pcd: torch.Tensor,
        gt_T: torch.Tensor = None,
        T_init: torch.Tensor = None,
        post_cam2ego_T: torch.Tensor = None,
        cam_intrinsic: torch.Tensor = None,
        masks: Optional[torch.Tensor] = None,
        out_init_loss: bool = False,
        domain_ids: Optional[torch.Tensor] = None,
        pcd_mask: Optional[torch.Tensor] = None,
        corr_window_radius: Optional[int] = None,
        rocr_detach: bool = False,
    ):
        """Training-compatible forward.

        When gt_T is provided, returns (T_pred, init_loss, loss_dict)
        matching the BEVCalib training interface.
        When gt_T is None, returns raw output dict for inference.
        """
        if T_init is None:
            T_init = gt_T

        init_err_rad = None
        if self.use_dp_head and gt_T is not None and T_init is not None:
            from modules.dp_pose_head import compute_init_rot_rad
            init_err_rad = compute_init_rot_rad(gt_T, T_init)

        result = self._core_forward(
            imgs, pcd, T_init, cam_intrinsic,
            pcd_mask=masks if pcd_mask is None else pcd_mask,
            corr_window_radius=corr_window_radius,
            rocr_detach=rocr_detach,
            init_err_rad=init_err_rad,
        )

        if gt_T is None:
            return result

        rot_q = result['rotation']
        tsl = result['translation']

        loss_dict, T_composed = self.loss_fn(
            pred_translation=tsl,
            pred_rotation=rot_q,
            pcs=pcd,
            gt_T_to_camera=gt_T,
            init_T_to_camera=T_init,
            mask=masks,
        )

        if self.use_pitch_branch and self.pitch_branch is not None:
            from bev_calib import FrontViewPitchBranch
            z_summary = result.get('z_summary')
            rgb_pitch_feat = result.get('rgb_pitch_feat', result.get('rgb_gap'))
            if z_summary is not None:
                pitch_pred = self.pitch_branch(z_summary, rgb_pitch_feat)
                pitch_loss = FrontViewPitchBranch.compute_loss(pitch_pred, gt_T, T_init)
                loss_dict['pitch_aux_loss'] = pitch_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + self.pitch_aux_weight * pitch_loss

        mag_pred = result.get('mag_pred')
        if mag_pred is not None and gt_T is not None and T_init is not None:
            R_gt = gt_T[:, :3, :3]
            R_init = T_init[:, :3, :3]
            tr_init = (R_init @ R_gt.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1)
            gt_angle = torch.acos(torch.clamp((tr_init - 1) / 2, -1 + 1e-7, 1 - 1e-7))
            mag_loss = F.mse_loss(mag_pred.squeeze(-1), gt_angle)
            loss_dict['magnitude_loss'] = mag_loss
            loss_dict['magnitude_pred_deg'] = float(mag_pred.mean().item() * 180 / 3.14159)
            loss_dict['magnitude_gt_deg'] = float(gt_angle.mean().item() * 180 / 3.14159)

        rocr_info = result.get('rocr_info', {})
        if 'delta_uv' in rocr_info:
            loss_dict['v42_delta_uv'] = rocr_info['delta_uv']
        loss_dict['v42_xyz_groups'] = result.get('xyz_groups', None)
        corr_info = result.get('corr_info', {})
        loss_dict['v42_valid_mask'] = corr_info.get('valid_mask', None)
        loss_dict['v42_patch_size'] = self.patch_size
        loss_dict['v42_rotation'] = rot_q

        if self.use_dp_head and self.route_loss_weight > 0 and 'dp_route_w' in result:
            route_w = result['dp_route_w']
            route_tgt = result['dp_route_target_w']
            route_loss = F.binary_cross_entropy(route_w, route_tgt)
            loss_dict['route_loss'] = route_loss.item()
            loss_dict['route_w_mean'] = float(route_w.mean().item())
            loss_dict['total_loss'] = loss_dict['total_loss'] + self.route_loss_weight * route_loss

        init_loss = None
        return T_composed, init_loss, loss_dict

    @staticmethod
    def _quat_compose(q1, q2):
        """Hamilton product q1 * q2. Both (B, 4) as (w, x, y, z)."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def _core_forward(
        self,
        imgs: torch.Tensor,
        pcd: torch.Tensor,
        T_init: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        pcd_mask: Optional[torch.Tensor] = None,
        corr_window_radius: Optional[int] = None,
        rocr_detach: bool = False,
        init_err_rad: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            imgs:           (B, 1, 3, H, W) images.
            pcd:            (B, N, 3) point clouds.
            T_init:         (B, 4, 4) initial extrinsic.
            cam_intrinsic:  (B, 3, 3) intrinsic matrix.
            pcd_mask:       (B, N) optional point mask.
            corr_window_radius: override for adaptive window.
            rocr_detach:    if True, detach R_geo gradient (early training).

        Returns:
            dict with:
                rotation:     (B, 4) predicted quaternion.
                translation:  (B, 3) zeros (rotation_only).
                rocr_info:    dict from RoCR module.
                corr_info:    dict from LocalCorrV2.
        """
        B = imgs.shape[0]
        device = imgs.device

        imgs_4d = imgs[:, 0] if imgs.dim() == 5 else imgs
        imgs_norm = (imgs_4d - self.img_mean.squeeze(1)) / self.img_std.squeeze(1)

        if self.use_dla:
            fpn_out, raw_scales = self.img_encoder(imgs_norm.unsqueeze(1))
            F_rgb = self.dla(raw_scales)
        else:
            fpn_out = self.img_encoder(imgs_norm.unsqueeze(1))
            fpn_feat = fpn_out[:, 0]
            fpn_feat = self.img_proj(fpn_feat)
            F_rgb = F.interpolate(
                fpn_feat, size=(self.feat_h, self.feat_w),
                mode='bilinear', align_corners=False)

        if self.feat_in is not None:
            F_rgb = self.feat_in(F_rgb)

        cur_feat_h, cur_feat_w = F_rgb.shape[2], F_rgb.shape[3]
        F_rgb_flat = F_rgb.flatten(2).permute(0, 2, 1)  # (B, feat_h*feat_w, D)
        rgb_gap = F_rgb_flat.mean(dim=1)                 # (B, D)

        _effective_mask = pcd_mask
        if _effective_mask is not None:
            if isinstance(_effective_mask, (list, tuple)):
                _effective_mask = None
            elif _effective_mask.all():
                _effective_mask = None
        xyz_groups, F_pc = self.point_encoder(pcd, mask=_effective_mask)

        uv_feat, uv_px = self._compute_uv_feat(xyz_groups, T_init, cam_intrinsic)

        valid_mask = (
            (uv_feat[..., 0] >= 0) & (uv_feat[..., 0] < cur_feat_w)
            & (uv_feat[..., 1] >= 0) & (uv_feat[..., 1] < cur_feat_h)
        )

        all_invalid = ~valid_mask.any(dim=1)
        if all_invalid.any():
            valid_mask = valid_mask.clone()
            valid_mask[all_invalid, 0] = True

        uv_norm = torch.zeros_like(uv_feat)
        uv_norm[..., 0] = 2.0 * uv_feat[..., 0] / max(cur_feat_w - 1, 1) - 1.0
        uv_norm[..., 1] = 2.0 * uv_feat[..., 1] / max(cur_feat_h - 1, 1) - 1.0
        proj_pos_emb = self.harmonic(uv_norm)

        grid_y = torch.linspace(-1, 1, cur_feat_h, device=device)
        grid_x = torch.linspace(-1, 1, cur_feat_w, device=device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        img_grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        img_pos_emb = self.harmonic(img_grid).unsqueeze(0).expand(B, -1, -1)

        n_img = cur_feat_h * cur_feat_w
        n_pc = F_pc.shape[1]
        attn_mask = valid_mask.unsqueeze(1).expand(-1, n_img, -1).float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, 0.0)

        F_cross = F_rgb_flat
        for block in self.cross_attn_blocks:
            F_cross = block(F_cross, F_pc, img_pos_emb, proj_pos_emb, attn_mask)
        F_cross = self.cross_proj(F_cross)

        f_corr, corr_map, corr_info = self.local_corr(
            img_tokens=F_cross,
            pc_tokens=F_pc,
            pc_uv_init=uv_feat,
            feat_h=cur_feat_h,
            feat_w=cur_feat_w,
            window_radius=corr_window_radius,
        )

        rocr_info = {
            'R_geo': torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1),
            'confidence': torch.ones(B, device=device),
            'skipped': torch.ones(B, dtype=torch.bool, device=device),
            'delta_uv': torch.zeros(B, xyz_groups.shape[1], 2, device=device),
        }
        R_geo = rocr_info['R_geo']

        if self.use_rocr:
            r = corr_window_radius or self.local_corr.default_radius
            rocr_out = self.rocr(
                corr_map=corr_map,
                xyz_groups=xyz_groups,
                uv_init=uv_feat,
                cam_intrinsic=cam_intrinsic,
                valid_mask=corr_info.get('valid_mask', valid_mask),
                window_radius=r,
                patch_size=self.patch_size,
                feat_h=cur_feat_h,
                feat_w=cur_feat_w,
            )
            rocr_info = rocr_out
            R_geo = rocr_out['R_geo']
            if rocr_detach:
                R_geo = R_geo.detach()

        pose_queries = self.pose_query_init(rgb_gap, T_init, R_geo)

        corr_tokens = f_corr + F_pc
        dp_meta = {}
        if self.use_dp_head and self.dp_head is not None:
            delta_q, dp_meta = self.dp_head(
                corr_tokens, pose_queries, init_err_rad=init_err_rad)
            mag_pred = dp_meta.get('mag_pred')
        else:
            head_out = self.corr_head(corr_tokens, pose_queries)
            if isinstance(head_out, tuple):
                delta_q, mag_pred = head_out
            else:
                delta_q, mag_pred = head_out, None

        if self.use_rocr and not rocr_info['skipped'].all():
            R_geo_q = RoCR.matrix_to_quaternion(R_geo)
            rotation = self._quat_compose(delta_q, R_geo_q)
        else:
            rotation = delta_q

        rotation = F.normalize(rotation, dim=-1, eps=1e-6)
        translation = torch.zeros(B, 3, device=device)

        if self.use_adir and self.adir_refiner is not None:
            rotation = self.adir_refiner(corr_tokens, rotation)

        z_hist = None
        rgb_pitch_feat = None
        if self.use_pitch_branch and self.pitch_branch is not None:
            # Q3: T_init-dependent projected-v histogram instead of raw z_hist
            # Projects LiDAR points to image using T_init, histograms v-coordinates
            # This makes pitch branch sensitive to the current pose estimate
            _n_bins = self.feat_dim
            z_hist = torch.zeros(B, _n_bins, device=device)
            if uv_px is not None:
                for b in range(B):
                    _v_vals = uv_px[b, valid_mask[b], 1]
                    if _v_vals.numel() > 0:
                        z_hist[b] = torch.histc(_v_vals.float(), bins=_n_bins,
                                                min=0, max=float(cur_feat_h * self.patch_size))
            empty_mask = z_hist.sum(dim=1) < 1
            if empty_mask.any():
                z_vals = xyz_groups[..., 2]
                for b in range(B):
                    if empty_mask[b]:
                        z_hist[b] = torch.histc(z_vals[b], bins=_n_bins, min=0, max=60)
            z_hist = z_hist / z_hist.sum(dim=1, keepdim=True).clamp(min=1)

            # Q2: Vertical band pooling instead of global average
            if self.use_pitch_vertical_bands > 1:
                from bev_calib import FrontViewPitchBranch
                rgb_pitch_feat = FrontViewPitchBranch.vertical_band_pool(
                    F_rgb_flat, cur_feat_h, cur_feat_w, self.use_pitch_vertical_bands)
            else:
                rgb_pitch_feat = rgb_gap

            if self.use_pitch_fusion and hasattr(self, 'pitch_conf_net'):
                pitch_pred = self.pitch_branch(z_hist, rgb_pitch_feat)
                pitch_delta = pitch_pred.squeeze(-1) if pitch_pred.dim() > 1 else pitch_pred
                half_p = pitch_delta * 0.5
                pitch_q = torch.stack([
                    torch.cos(half_p),
                    torch.sin(half_p),
                    torch.zeros_like(half_p),
                    torch.zeros_like(half_p),
                ], dim=-1)
                pitch_q = F.normalize(pitch_q, dim=-1, eps=1e-8)

                conf = self.pitch_conf_net(
                    torch.cat([rgb_gap, z_hist], dim=-1)
                )  # (B, 1)

                composed = self._quat_compose(pitch_q, rotation)
                rotation = rotation + conf * (composed - rotation)
                rotation = F.normalize(rotation, dim=-1, eps=1e-6)

        output = {
            'rotation': rotation,
            'translation': translation,
            'rocr_info': rocr_info,
            'corr_info': corr_info,
            'rgb_gap': rgb_gap,
            'xyz_groups': xyz_groups,
        }
        if rgb_pitch_feat is not None:
            output['rgb_pitch_feat'] = rgb_pitch_feat
        if z_hist is not None:
            output['z_summary'] = z_hist
        if mag_pred is not None:
            output['mag_pred'] = mag_pred
        if dp_meta:
            output['dp_route_w'] = dp_meta.get('route_w')
            output['dp_route_target_w'] = dp_meta.get('route_target_w')
            output['dp_init_err_deg'] = dp_meta.get('init_err_deg')
            output['dp_jacg_gain'] = dp_meta.get('jacg_gain')

        return output

    @staticmethod
    def _quat_compose(delta_q: torch.Tensor, base_q: torch.Tensor) -> torch.Tensor:
        """Hamilton product: q_out = delta_q * base_q (w, x, y, z convention)."""
        w1, x1, y1, z1 = delta_q.unbind(-1)
        w2, x2, y2, z2 = base_q.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    @torch.no_grad()
    def iterative_inference(
        self, imgs, pcd, T_init, cam_intrinsic, n_iters=1, pcd_mask=None,
    ):
        """Iterative refinement: update T_init with predicted rotation."""
        T_current = T_init.clone()
        for _ in range(n_iters):
            out = self._core_forward(imgs, pcd, T_current, cam_intrinsic, pcd_mask=pcd_mask)
            rot_q = out['rotation']
            tsl = out['translation']
            T_pred = torch.bmm(batch_tvector2mat(tsl), batch_quat2mat(rot_q))
            T_current = torch.bmm(torch.linalg.inv(T_pred.float()), T_current.float())
            if self.rotation_only:
                T_current = T_current.clone()
                T_current[:, :3, 3] = T_init[:, :3, 3]
        return T_current
