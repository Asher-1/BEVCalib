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
from losses.quat_tools import batch_quat2mat, batch_tvector2mat, quaternion_distance
from losses.losses import realworld_loss

from modules.dla_aggregation import DLAAggregation
from modules.local_correlation_v2 import LocalMultiHeadCorrelationV2
from modules.rocr_refine import RoCR
from modules.pose_query_init import PoseQueryInit
from modules.corr_transformer_decoder import CorrTransformerHead
from losses.corr_alignment_loss import compute_projection_v42

from modules.sim_cross_attention import SimCrossAttention
from modules.position_encoding_3d import PositionEncoding3D
from modules.fov_classifier import FoVClassifier
from modules.coarse_rotation_head import CoarseRotationHead
from losses.similarity_loss import (
    build_gt_correspondence, similarity_loss, fov_classification_loss
)


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
        route_zd_penalty_weight: float = 0.0,
        dp_route_input_mode: str = "gt_or_pred",
        dp_jacg_input_mode: str = "gt_or_pred",
        dp_recovery_quat_loss_weight: float = 0.0,
        use_jacg: bool = True,
        jacg_hidden_dim: int = 64,
        bias_path_in_norm: bool = True,
        recovery_path_layer_norm: bool = True,
        dp_train_hard_route: bool = False,
        use_hard_route_eval: bool = False,
        use_adir: bool = False,
        adir_steps: int = 2,
        adir_max_step_deg: float = 1.0,
        # V60: Implicit Alignment parameters
        use_sim_loss: bool = False,
        sim_loss_weight: float = 1.0,
        sim_loss_warmup: int = 10,
        sim_loss_soft_radius: float = 1.0,
        sim_n_layers: int = 2,
        use_registry_token: bool = False,
        use_fov_cls_loss: bool = False,
        fov_cls_weight: float = 0.3,
        use_3d_pos_encoding: bool = False,
        pos_enc_depth_bins: int = 16,
        use_coarse_refine: bool = False,
        coarse_refine_detach_epoch: int = 30,
        # Backbone selection
        backbone_type: str = 'swin',
        backbone_variant: str = 'dinov2-small',
        freeze_backbone: bool = True,
        backbone_freeze_layers: str = None,
        backbone_weights: str = None,
    ):
        super().__init__()
        self._backbone_type = backbone_type
        self._backbone_variant = backbone_variant
        self._freeze_backbone = freeze_backbone
        self._backbone_freeze_layers = backbone_freeze_layers
        self._backbone_weights = backbone_weights
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

        # V60 flags
        self.use_sim_loss = use_sim_loss
        self.sim_loss_weight = sim_loss_weight
        self.sim_loss_warmup = sim_loss_warmup
        self.sim_loss_soft_radius = sim_loss_soft_radius
        self.use_registry_token = use_registry_token
        self.use_fov_cls_loss = use_fov_cls_loss
        self.fov_cls_weight = fov_cls_weight
        self.use_3d_pos_encoding = use_3d_pos_encoding
        self.use_coarse_refine = use_coarse_refine
        self.coarse_refine_detach_epoch = coarse_refine_detach_epoch

        img_H, img_W = img_shape

        fpn_out_channels = 256
        feat_shape = (fpn_out_channels, img_H // 8, img_W // 8)

        backbone_type = getattr(self, '_backbone_type', 'swin')
        if backbone_type == 'dinov2':
            from img_branch.dinov2_encoder import DINOv2Encoder
            self.img_encoder = DINOv2Encoder(
                featureShape=feat_shape,
                out_channels=fpn_out_channels,
                variant=getattr(self, '_backbone_variant', 'dinov2-small'),
                freeze_backbone=getattr(self, '_freeze_backbone', True),
                freeze_layers=getattr(self, '_backbone_freeze_layers', None),
                weights_path=getattr(self, '_backbone_weights', None),
            )
            self.feat_h = feat_shape[1]
            self.feat_w = feat_shape[2]
            self.patch_size = 8.0
        else:
            fpn_in_channels = [192, 384, 768]
            self.img_encoder = SwinT_tiny_Encoder(
                output_indices=[1, 2, 3],
                featureShape=feat_shape,
                out_channels=fpn_out_channels,
                FPN_in_channels=fpn_in_channels,
                FPN_out_channels=fpn_out_channels,
            )
            self.feat_h = img_H // 4
            self.feat_w = img_W // 4
            self.patch_size = 4.0

        self.img_proj = (
            nn.Conv2d(fpn_out_channels, feat_dim, 1)
            if fpn_out_channels != feat_dim else nn.Identity()
        )

        if use_dla:
            if backbone_type == 'dinov2':
                self.dla = DLAAggregation(
                    in_channels_list=[fpn_out_channels] * 3,
                    out_channels=feat_dim,
                )
            else:
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
                use_registry_token=use_registry_token,
            ))
        self.cross_attn_blocks = nn.ModuleList(cross_blocks)
        self._cross_out_dim = _attn_out

        # V60: Similarity Cross-Attention (reverse 3D→2D for L_sim)
        self.sim_cross_attn = None
        if use_sim_loss:
            self.sim_cross_attn = SimCrossAttention(
                feat_dim=feat_dim,
                n_layers=sim_n_layers,
                n_heads=1,
                use_registry_token=use_registry_token,
                dropout=head_dropout,
            )
            print(f"[CFBevRCalib] V60 SimCrossAttention enabled "
                  f"(layers={sim_n_layers}, registry={use_registry_token})")

        # V60: 3D Position Encoding
        self.pos_enc_3d = None
        if use_3d_pos_encoding:
            self.pos_enc_3d = PositionEncoding3D(
                feat_dim=feat_dim,
                depth_bins=pos_enc_depth_bins,
                depth_min=1.0,
                depth_max=100.0,
                patch_size=4.0,
            )
            print(f"[CFBevRCalib] V60 3D Position Encoding enabled (bins={pos_enc_depth_bins})")

        # V60: FOV Classifier
        self.fov_classifier = None
        if use_fov_cls_loss:
            self.fov_classifier = FoVClassifier(feat_dim=feat_dim, hidden_dim=feat_dim // 2)
            print("[CFBevRCalib] V60 FOV Classifier enabled")

        # V60: Coarse Rotation Head (predict R_coarse from SimCrossAttn output)
        self.coarse_rot_head = None
        if use_coarse_refine and use_sim_loss:
            self.coarse_rot_head = CoarseRotationHead(
                feat_dim=feat_dim, hidden_dim=feat_dim, use_attention_pool=True
            )
            print("[CFBevRCalib] V60 Coarse Rotation Head enabled (coarse-to-fine)")

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
        self.route_zd_penalty_weight = route_zd_penalty_weight
        self.dp_route_input_mode = dp_route_input_mode
        self.dp_jacg_input_mode = dp_jacg_input_mode
        self.dp_recovery_quat_loss_weight = dp_recovery_quat_loss_weight
        self.dp_gate_rad = math.radians(dp_gate_deg)
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
                use_hard_route_eval=use_hard_route_eval,
                route_input_mode=dp_route_input_mode,
                jacg_input_mode=dp_jacg_input_mode,
                recovery_path_layer_norm=recovery_path_layer_norm,
                train_hard_route=dp_train_hard_route,
            )
            print(f"[CFBevRCalib] V53 DP-Head enabled (gate={dp_gate_deg}°, jacg={use_jacg}, "
                  f"hard_eval={use_hard_route_eval}, route_input={dp_route_input_mode}, "
                  f"jacg_input={dp_jacg_input_mode}, rec_ln={recovery_path_layer_norm}, "
                  f"train_hard_route={dp_train_hard_route})")

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
            route_zd_penalty_weight=getattr(args, 'route_zd_penalty_weight', 0.0),
            dp_route_input_mode=getattr(args, 'dp_route_input_mode', 'gt_or_pred'),
            dp_jacg_input_mode=getattr(args, 'dp_jacg_input_mode', 'gt_or_pred'),
            dp_recovery_quat_loss_weight=getattr(args, 'dp_recovery_quat_loss_weight', 0.0),
            use_jacg=getattr(args, 'use_jacg', 1) > 0,
            jacg_hidden_dim=getattr(args, 'jacg_hidden_dim', 64),
            bias_path_in_norm=getattr(args, 'bias_path_in_norm', 1) > 0,
            recovery_path_layer_norm=getattr(args, 'recovery_path_layer_norm', 1) > 0,
            dp_train_hard_route=getattr(args, 'dp_train_hard_route', 0) > 0,
            use_hard_route_eval=getattr(args, 'use_hard_route_eval', 0) > 0,
            use_adir=getattr(args, 'use_adir', 0) > 0,
            adir_steps=getattr(args, 'adir_steps', 2),
            adir_max_step_deg=getattr(args, 'adir_max_step_deg', 1.0),
            # V60
            use_sim_loss=getattr(args, 'use_sim_loss', 0) > 0,
            sim_loss_weight=getattr(args, 'sim_loss_weight', 1.0),
            sim_loss_warmup=getattr(args, 'sim_loss_warmup', 10),
            sim_loss_soft_radius=getattr(args, 'sim_loss_soft_radius', 1.0),
            sim_n_layers=getattr(args, 'sim_n_layers', 2),
            use_registry_token=getattr(args, 'use_registry_token', 0) > 0,
            use_fov_cls_loss=getattr(args, 'use_fov_cls_loss', 0) > 0,
            fov_cls_weight=getattr(args, 'fov_cls_weight', 0.3),
            use_3d_pos_encoding=getattr(args, 'use_3d_pos_encoding', 0) > 0,
            pos_enc_depth_bins=getattr(args, 'pos_enc_depth_bins', 16),
            use_coarse_refine=getattr(args, 'use_coarse_refine', 0) > 0,
            coarse_refine_detach_epoch=getattr(args, 'coarse_refine_detach_epoch', 30),
            # Backbone selection
            backbone_type=getattr(args, 'backbone_type', 'swin'),
            backbone_variant=getattr(args, 'backbone_variant', 'dinov2-small'),
            freeze_backbone=getattr(args, 'freeze_backbone', 1) > 0,
            backbone_freeze_layers=getattr(args, 'backbone_freeze_layers', None),
            backbone_weights=getattr(args, 'backbone_weights', None),
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

        if self.use_dp_head and self.route_loss_weight > 0 and 'dp_route_pred_w' in result:
            route_w = result['dp_route_pred_w']
            route_tgt = result['dp_route_target_w']
            route_loss = F.binary_cross_entropy(route_w, route_tgt)
            loss_dict['route_loss'] = route_loss.item()
            loss_dict['route_w_mean'] = float(route_w.mean().item())
            loss_dict['total_loss'] = loss_dict['total_loss'] + self.route_loss_weight * route_loss

        if (self.use_dp_head and self.route_zd_penalty_weight > 0
                and 'dp_route_w' in result and 'dp_init_err_deg' in result):
            init_err_rad = result['dp_init_err_deg'] * (math.pi / 180.0)
            zd_mask = (init_err_rad <= self.dp_gate_rad).float().unsqueeze(-1)
            route_w = result['dp_route_w']
            route_zd_penalty = (route_w * zd_mask).sum() / zd_mask.sum().clamp(min=1.0)
            loss_dict['route_zd_penalty'] = route_zd_penalty.item()
            loss_dict['total_loss'] = (
                loss_dict['total_loss'] + self.route_zd_penalty_weight * route_zd_penalty
            )

        if (self.use_dp_head and self.dp_recovery_quat_loss_weight > 0
                and 'dp_delta_q_rec' in result and gt_T is not None and T_init is not None):
            with torch.cuda.amp.autocast(enabled=False):
                target_R = torch.bmm(
                    T_init[:, :3, :3].float(),
                    gt_T[:, :3, :3].float().transpose(1, 2))
                target_q = RoCR.matrix_to_quaternion(target_R).to(result['dp_delta_q_rec'].device)
                rec_dist = quaternion_distance(
                    result['dp_delta_q_rec'].float(), target_q.float(),
                    result['dp_delta_q_rec'].device)
                if 'dp_route_target_w' in result and result['dp_route_target_w'] is not None:
                    rec_mask = result['dp_route_target_w'].detach().reshape(-1).float()
                else:
                    init_err = result.get('dp_init_err_deg')
                    if init_err is None:
                        rec_mask = torch.ones_like(rec_dist)
                    else:
                        rec_mask = (init_err.reshape(-1) * (math.pi / 180.0) > self.dp_gate_rad).float()
                denom = rec_mask.sum().clamp(min=1.0)
                rec_loss = (rec_dist * rec_mask).sum() / denom
            loss_dict['dp_recovery_quat_loss'] = rec_loss.detach() / math.pi * 180.0
            loss_dict['dp_recovery_active_ratio'] = float(rec_mask.mean().item())
            loss_dict['total_loss'] = (
                loss_dict['total_loss'] + self.dp_recovery_quat_loss_weight * rec_loss
            )

        # V60: Similarity Loss + FOV Classification Loss
        _v60_needs_gt = (
            (self.use_sim_loss and 'v60_sim_matrices' in result)
            or (self.use_fov_cls_loss and 'v60_fov_logits' in result)
        )
        if _v60_needs_gt and gt_T is not None:
            feat_h, feat_w = result['v60_feat_hw']
            gt_corr, gt_fov = build_gt_correspondence(
                xyz_groups=result.get('xyz_groups'),
                T_gt=gt_T,
                cam_intrinsic=cam_intrinsic,
                feat_h=feat_h,
                feat_w=feat_w,
                patch_size=self.patch_size,
                use_registry_token=self.use_registry_token,
                soft_radius=self.sim_loss_soft_radius,
            )

            if self.use_sim_loss and 'v60_sim_matrices' in result:
                sim_matrices = result['v60_sim_matrices']
                sim_loss_val = similarity_loss(
                    sim_matrices=sim_matrices,
                    gt_corr=gt_corr,
                    fov_mask=gt_fov,
                )
                loss_dict['v60_sim_loss'] = sim_loss_val.detach().item()
                loss_dict['v60_fov_ratio'] = float(gt_fov.float().mean().item())
                loss_dict['total_loss'] = (
                    loss_dict['total_loss'] + self.sim_loss_weight * sim_loss_val
                )

            if self.use_fov_cls_loss and 'v60_fov_logits' in result:
                fov_logits = result['v60_fov_logits']
                fov_loss_val = fov_classification_loss(fov_logits, gt_fov)
                loss_dict['v60_fov_cls_loss'] = fov_loss_val.detach().item()
                loss_dict['total_loss'] = (
                    loss_dict['total_loss'] + self.fov_cls_weight * fov_loss_val
                )

        # V60: Coarse Rotation Loss (geodesic distance between R_coarse and R_gt)
        if self.use_coarse_refine and 'v60_R_coarse' in result and gt_T is not None:
            R_coarse = result['v60_R_coarse']  # (B, 3, 3)
            R_gt = gt_T[:, :3, :3]  # (B, 3, 3)
            R_diff = torch.bmm(R_coarse.transpose(1, 2), R_gt)
            trace_val = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
            cos_angle = (trace_val - 1.0) / 2.0
            cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            coarse_geo_loss = torch.acos(cos_angle).mean()
            loss_dict['v60_coarse_rot_loss'] = coarse_geo_loss.detach().item()
            coarse_weight = 0.3
            loss_dict['total_loss'] = (
                loss_dict['total_loss'] + coarse_weight * coarse_geo_loss
            )

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
            enc_out = self.img_encoder(imgs_norm.unsqueeze(1))
            if isinstance(enc_out, tuple):
                fpn_out, raw_scales = enc_out
                F_rgb = self.dla(raw_scales)
            else:
                fpn_feat = enc_out[:, 0] if enc_out.dim() == 5 else enc_out
                fpn_feat = self.img_proj(fpn_feat)
                fpn_feat = F.interpolate(
                    fpn_feat, size=(self.feat_h, self.feat_w),
                    mode='bilinear', align_corners=False)
                F_rgb = fpn_feat
        else:
            fpn_out = self.img_encoder(imgs_norm.unsqueeze(1))
            fpn_feat = fpn_out[:, 0] if fpn_out.dim() == 5 else fpn_out
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
        attn_mask = valid_mask.unsqueeze(1).expand(-1, n_img, -1)

        # V60: 3D Position Encoding for image features
        img_pe_3d = None
        if self.pos_enc_3d is not None:
            img_pe_3d = self.pos_enc_3d(cur_feat_h, cur_feat_w, cam_intrinsic)

        # V60: SimCrossAttention (3D→2D) for similarity loss
        v60_sim_matrices = None
        v60_fov_logits = None
        v60_R_coarse = None
        if self.sim_cross_attn is not None:
            sim_img_feat = F_rgb_flat
            if img_pe_3d is not None:
                sim_img_feat = sim_img_feat + img_pe_3d
            sim_pc_out, v60_sim_matrices = self.sim_cross_attn(
                pc_features=F_pc,
                img_features=sim_img_feat,
                pc_pos_emb=None,
                img_pos_emb=None,
            )

            # V60: Coarse Rotation Head (predict R_coarse from refined 3D features)
            if self.coarse_rot_head is not None:
                v60_R_coarse = self.coarse_rot_head(
                    sim_pc_out, valid_mask, img_features=sim_img_feat
                )

                # Coarse-to-Fine: re-project points using corrected T_init
                T_corrected = T_init.clone()
                # The coarse head is supervised against absolute R_gt.
                T_corrected[:, :3, :3] = v60_R_coarse
                uv_feat_c, _ = self._compute_uv_feat(xyz_groups, T_corrected, cam_intrinsic)
                valid_mask_c = (
                    (uv_feat_c[..., 0] >= 0) & (uv_feat_c[..., 0] < cur_feat_w)
                    & (uv_feat_c[..., 1] >= 0) & (uv_feat_c[..., 1] < cur_feat_h)
                )
                all_invalid_c = ~valid_mask_c.any(dim=1)
                if all_invalid_c.any():
                    valid_mask_c = valid_mask_c.clone()
                    valid_mask_c[all_invalid_c, 0] = True
                # Update projection pos emb and masks for main cross-attention
                uv_norm_c = torch.zeros_like(uv_feat_c)
                uv_norm_c[..., 0] = 2.0 * uv_feat_c[..., 0] / max(cur_feat_w - 1, 1) - 1.0
                uv_norm_c[..., 1] = 2.0 * uv_feat_c[..., 1] / max(cur_feat_h - 1, 1) - 1.0
                proj_pos_emb = self.harmonic(uv_norm_c)
                valid_mask = valid_mask_c
                n_img = cur_feat_h * cur_feat_w
                attn_mask = valid_mask.unsqueeze(1).expand(-1, n_img, -1)
                # All downstream correspondence geometry must use T_coarse.
                uv_feat = uv_feat_c
                uv_px = uv_feat_c * self.patch_size

        # V60: FOV Classification
        if self.fov_classifier is not None:
            v60_fov_logits = self.fov_classifier(F_pc)

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
            output['dp_route_pred_w'] = dp_meta.get('route_pred_w')
            output['dp_route_target_w'] = dp_meta.get('route_target_w')
            output['dp_init_err_deg'] = dp_meta.get('init_err_deg')
            output['dp_jacg_gain'] = dp_meta.get('jacg_gain')
            output['dp_delta_q_bias'] = dp_meta.get('delta_q_bias')
            output['dp_delta_q_rec'] = dp_meta.get('delta_q_rec')

        # V60: pass through similarity and FOV info for loss computation
        if v60_sim_matrices is not None:
            output['v60_sim_matrices'] = v60_sim_matrices
        if v60_fov_logits is not None:
            output['v60_fov_logits'] = v60_fov_logits
        if v60_R_coarse is not None:
            output['v60_R_coarse'] = v60_R_coarse
        output['v60_valid_mask'] = valid_mask
        output['v60_feat_hw'] = (cur_feat_h, cur_feat_w)

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
