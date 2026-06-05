"""HybridTripleCalibNet (HTCN): BEV + ProjFusion dual-branch fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import bev_settings
from img_branch.cam2bev_query import Cam2BEVQuery
from pc_branch.pc_branch import Lidar2BEV
from pc_branch.pointgpt2bev import PointGPT2BEV
from pointgpt_wrapper import PointGPTEncoder
from projfusion_branch import ProjFusionBranch
from fusion_heads import FUSION_HEADS, SingleBranchHead
from losses.losses import realworld_loss
from bev_calib import BEVDiffFuser, BEVEncoder


class HybridTripleCalib(nn.Module):
    """BEVCalib Query-BEV + optional PointGPT2BEV + ProjFusion AttenDualFusion."""

    FUSION_BACKENDS = (
        'bev_only', 'proj_only', 'hybrid_dual', 'hybrid_triple',
        'camera_bev_triple',  # v39.1: Camera-BEV Cross-Attention
    )

    def __init__(self,
                 img_shape=(640, 360),
                 fusion_backend='hybrid_triple',
                 pc_encoder_mode='pointgpt2bev',
                 fusion_variant='gated',
                 rotation_only=True,
                 enable_axis_loss=True,
                 weight_axis_rotation=0.5,
                 axis_weights=(1.0, 1.0, 1.0),
                 use_balanced_axis_loss=False,
                 use_geodesic_loss=False,
                 weight_quat_norm=0.5,
                 head_dropout=0.1,
                 fuser_type='diff',
                 bev_encoder=True,
                 bev_pool_factor=4,
                 num_heads=8,
                 num_layers=2,
                 cam2bev_mode='query',
                 backbone_type='dinov2',
                 backbone_variant='dinov2-small',
                 freeze_backbone=True,
                 freeze_layers=None,
                 backbone_weights=None,
                 projfusion_image_hw=(224, 448),
                 pointgpt_ckpt=None,
                 pointgpt_config=None,
                 pointgpt_max_depth=60.0,
                 deep_supervision_weight=0.2,
                 gate_entropy_weight=0.2,  # Anti-collapse: 强正则化
                 iterative_refine=0,
                 projfusion_margin=2.0,
                 **kwargs):
        super().__init__()
        if fusion_backend not in self.FUSION_BACKENDS:
            raise ValueError(f"Unknown fusion_backend={fusion_backend}")
        if cam2bev_mode != 'query':
            raise ValueError("HTCN requires cam2bev_mode=query (LSS disabled)")
        if pc_encoder_mode not in ('spconv', 'pointgpt2bev'):
            raise ValueError(f"Unknown pc_encoder_mode={pc_encoder_mode}")

        self.fusion_backend = fusion_backend
        self.pc_encoder_mode = pc_encoder_mode
        self.fusion_variant = fusion_variant
        self.rotation_only = rotation_only
        self.deep_supervision_weight = deep_supervision_weight
        self.gate_entropy_weight = gate_entropy_weight
        self.iterative_refine = iterative_refine
        self.bev_pool_factor = bev_pool_factor or 4
        self.use_camera_bev = fusion_backend == 'camera_bev_triple'
        self.use_bev = fusion_backend in ('bev_only', 'hybrid_dual', 'hybrid_triple')
        self.use_proj = fusion_backend in ('proj_only', 'hybrid_dual', 'hybrid_triple', 'camera_bev_triple')
        self._shared_pointgpt = (
            self.use_bev and self.use_proj and pc_encoder_mode == 'pointgpt2bev'
        )
        
        # 预先初始化为 None，避免 proj_only 模式下 AttributeError
        self.pointgpt_encoder = None

        if self.use_camera_bev:
            # Camera-BEV Cross-Attention分支（v39.1新架构）
            from camera_bev_fusion import build_camera_bev_branch
            
            # 复用DINOv2作为图像编码器
            self.img_branch = Cam2BEVQuery(
                img_shape=img_shape,
                backbone_type=backbone_type,
                backbone_variant=backbone_variant,
                freeze_backbone=freeze_backbone,
                freeze_layers=freeze_layers,
                backbone_weights=backbone_weights,
                query_downsample=bev_pool_factor or 4,
            )
            
            # 复用PointGPT作为点云编码器
            self.pointgpt_encoder = PointGPTEncoder(
                checkpoint_path=pointgpt_ckpt,
                config_path=pointgpt_config,
                max_depth=pointgpt_max_depth,
                freeze=True,
            )
            
            # Camera-BEV分支：直接cross-attention，无BEV投影
            self.camera_bev_branch = build_camera_bev_branch(
                img_encoder=self.img_branch.CamEncode,  # 共享DINOv2
                hidden_dim=256,
                num_heads=8,
                num_layers=2,
                dropout=head_dropout
            )
            self.bev_feat_dim = self.camera_bev_branch.out_dim
            
        elif self.use_bev:
            self.img_branch = Cam2BEVQuery(
                img_shape=img_shape,
                backbone_type=backbone_type,
                backbone_variant=backbone_variant,
                freeze_backbone=freeze_backbone,
                freeze_layers=freeze_layers,
                backbone_weights=backbone_weights,
                query_downsample=bev_pool_factor or 4,
            )
            if pc_encoder_mode == 'pointgpt2bev':
                self.pointgpt_encoder = PointGPTEncoder(
                    checkpoint_path=pointgpt_ckpt,
                    config_path=pointgpt_config,
                    max_depth=pointgpt_max_depth,
                    freeze=True,
                )
                self.pc_branch = PointGPT2BEV(
                    in_dim=self.pointgpt_encoder.out_dim,
                    out_channels=self.img_branch.out_channels,
                )
            else:
                self.pointgpt_encoder = None
                self.pc_branch = Lidar2BEV(
                    to_bev_mode=kwargs.get('to_bev_mode', 'concat'),
                    voxel_mode=kwargs.get('voxel_mode', 'hard'),
                    scatter_reduce=kwargs.get('scatter_reduce', 'sum'),
                )
            self.bev_encoder_use = bev_encoder
            if self.bev_encoder_use:
                self.bev_encoder = BEVEncoder()
            embed_dim = self.img_branch.out_channels + self.pc_branch.out_channels
            if fuser_type == 'diff':
                self.conv_fuser = BEVDiffFuser(
                    self.img_branch.out_channels,
                    self.pc_branch.out_channels,
                    embed_dim,
                    cam_drop_aware=False,
                )
            else:
                from bev_calib import ConvFuser
                self.conv_fuser = ConvFuser(
                    self.img_branch.out_channels,
                    self.pc_branch.out_channels,
                    embed_dim,
                )
            self.bev_shape = (self.img_branch.nx_x, self.img_branch.nx_y)
            self.pose_embed = nn.Parameter(
                torch.zeros(1, embed_dim, self.bev_shape[0], self.bev_shape[1]))
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=4 * embed_dim,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=num_layers * 4,
            )
            self.bev_feat_dim = embed_dim
            self.head_drop = nn.Dropout(head_dropout)

        if self.use_proj:
            _proj_kwargs = dict(
                image_hw=projfusion_image_hw,
                pointgpt_ckpt=pointgpt_ckpt,
                pointgpt_config=pointgpt_config,
                margin=projfusion_margin,
            )
            if self._shared_pointgpt:
                _proj_kwargs.update(
                    skip_internal_pointgpt=True,
                    pointgpt_embed_dim=self.pointgpt_encoder.out_dim,
                    pointgpt_max_depth=pointgpt_max_depth,
                )
            self.proj_branch = ProjFusionBranch(**_proj_kwargs)
            self.proj_feat_dim = self.proj_branch.out_dim
        else:
            self.proj_branch = None
            self.proj_feat_dim = 0

        if (self.use_bev or self.use_camera_bev) and self.use_proj:
            head_cls = FUSION_HEADS[fusion_variant]
            self.fusion_head = head_cls(self.bev_feat_dim, self.proj_feat_dim,
                                        hidden=128, dropout=head_dropout)
        elif self.use_bev or self.use_camera_bev:
            self.fusion_head = SingleBranchHead(self.bev_feat_dim, dropout=head_dropout)
        elif self.use_proj:
            self.fusion_head = SingleBranchHead(self.proj_feat_dim, dropout=head_dropout)
        else:
            raise ValueError("At least one branch must be enabled")

        if deep_supervision_weight > 0 and (self.use_bev or self.use_camera_bev) and self.use_proj:
            self.aux_bev_head = SingleBranchHead(self.bev_feat_dim, dropout=head_dropout)
            self.aux_proj_head = SingleBranchHead(self.proj_feat_dim, dropout=head_dropout)
        else:
            self.aux_bev_head = None
            self.aux_proj_head = None

        self.loss_fn = realworld_loss(
            rotation_only=rotation_only,
            enable_axis_loss=enable_axis_loss,
            weight_axis_rotation=weight_axis_rotation,
            axis_weights=axis_weights,
            use_geodesic_loss=use_geodesic_loss,
            use_balanced_axis_loss=use_balanced_axis_loss,
            weight_quat_norm=weight_quat_norm,
        )
        print(f"[HybridTripleCalib] backend={fusion_backend}, pc={pc_encoder_mode}, "
              f"fusion={fusion_variant}, bev={self.use_bev}, proj={self.use_proj}, "
              f"shared_pointgpt={self._shared_pointgpt}")
        self._profile_modules = False
        self._profile_events = []

    def get_module_profile(self, reset=True):
        """BEVCalib-compatible profiling hook (HTCN detailed breakdown TBD)."""
        if not getattr(self, '_profile_events', None):
            return {}
        if reset:
            self._profile_events = []
        return {}

    def _pool_bev(self, x, cam_bev_mask):
        B, C, H, W = x.shape
        if self.bev_pool_factor > 1:
            pf = self.bev_pool_factor
            x = F.avg_pool2d(x, pf)
            cam_bev_mask = F.max_pool2d(
                cam_bev_mask.reshape(B, 1, H, W).float(), pf).squeeze(1)
            H, W = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        bev_mask = cam_bev_mask.reshape(B, H * W).bool()
        max_valid = int(bev_mask.sum(dim=1).max().item())
        max_valid = max(max_valid, 1)
        masked_x = torch.zeros(B, max_valid, C, device=x.device, dtype=x.dtype)
        padding_mask = torch.zeros(B, max_valid, dtype=torch.bool, device=x.device)
        valid_counts = torch.zeros(B, dtype=torch.long, device=x.device)
        for i in range(B):
            cnt = int(bev_mask[i].sum().item())
            valid_counts[i] = cnt
            if cnt > 0:
                masked_x[i, :cnt] = x[i, bev_mask[i]]
                padding_mask[i, cnt:] = True
        x = self.transformer(masked_x, src_key_padding_mask=padding_mask)
        pooled = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        for i in range(B):
            cnt = valid_counts[i].item()
            if cnt > 0:
                pooled[i] = x[i, :cnt].mean(dim=0)
        return pooled

    def _forward_bev_features(self, img, pc, init_T_to_camera, cam_intrinsic, post_cam2ego_T,
                              xyz_g=None, feat_g=None):
        img = img.unsqueeze(1)
        cam2ego_T = torch.linalg.inv(init_T_to_camera.float()).unsqueeze(1)
        if cam_intrinsic.dim() == 3:
            cam_intrinsic = cam_intrinsic.unsqueeze(1)
        if post_cam2ego_T.dim() == 3:
            post_cam2ego_T = post_cam2ego_T.unsqueeze(1)
        cam_bev_feats, cam_bev_mask = self.img_branch(
            cam2ego_T=cam2ego_T,
            cam_intrins=cam_intrinsic,
            post_cam2ego_T=post_cam2ego_T,
            imgs=img,
        )
        if self.pointgpt_encoder is not None:
            if xyz_g is None or feat_g is None:
                pc_xyz = pc if pc.shape[-1] == 3 else pc.transpose(1, 2).contiguous()
                xyz_g, feat_g = self.pointgpt_encoder(pc_xyz)
            pc_bev_feats, pc_bev_mask = self.pc_branch(xyz_g, feat_g)
        else:
            pc_t = pc.permute(0, 2, 1).contiguous()
            pc_bev_feats = self.pc_branch(pc_t)
            pc_bev_mask = cam_bev_mask
        x = self.conv_fuser(cam_bev_feats, pc_bev_feats)
        if self.bev_encoder_use:
            x = self.bev_encoder(x)
        x = x + self.pose_embed
        f_bev = self._pool_bev(x, cam_bev_mask)
        f_bev = self.head_drop(f_bev)
        return f_bev

    def _predict_rotation(self, f_bev, f_proj):
        if (self.use_bev or self.use_camera_bev) and self.use_proj:
            rotation, _meta = self.fusion_head(f_bev, f_proj)
        elif self.use_bev or self.use_camera_bev:
            rotation, _meta = self.fusion_head(f_bev)
        else:
            rotation, _meta = self.fusion_head(f_proj)
        self._last_fusion_meta = _meta if isinstance(_meta, dict) else {}
        return rotation

    def forward(self, img, pc, gt_T_to_camera, init_T_to_camera, post_cam2ego_T,
                cam_intrinsic, masks=None, out_init_loss=False, domain_ids=None):
        if gt_T_to_camera.dim() == 4:
            gt_T_to_camera = gt_T_to_camera.squeeze(1)
        if init_T_to_camera.dim() == 4:
            init_T_to_camera = init_T_to_camera.squeeze(1)
        if cam_intrinsic.dim() == 4:
            cam_intrinsic = cam_intrinsic.squeeze(1)

        B = img.shape[0]
        pc_xyz = pc if pc.shape[-1] == 3 else pc.transpose(1, 2).contiguous()
        pc_for_loss = pc_xyz
        sensor_h, sensor_w = img.shape[2], img.shape[3]
        img_vit = self.proj_branch.preprocess_image(img) if self.use_proj else None
        xyz_g = feat_g = None
        if self.pointgpt_encoder is not None and (self.use_bev or self.use_camera_bev or self._shared_pointgpt):
            xyz_g, feat_g = self.pointgpt_encoder(pc_xyz)

        n_iter = max(1, self.iterative_refine) if self.iterative_refine > 0 else 1
        t_current = init_T_to_camera
        total_loss = None
        t_expected = None

        for step_i in range(n_iter):
            f_bev = f_proj = None
            if self.use_camera_bev:
                # Camera-BEV forward（v39.1新架构）
                if xyz_g is None:
                    xyz_g, feat_g = self.pointgpt_encoder(pc_xyz)
                T_cam2lidar = torch.linalg.inv(t_current.float())
                f_bev = self.camera_bev_branch(
                    img=img,
                    pc_groups_xyz=xyz_g,
                    pc_groups_feat=feat_g,
                    T_cam2lidar=T_cam2lidar,
                    cam_intrinsic=cam_intrinsic
                )
                f_bev = self.head_drop(f_bev)
            elif self.use_bev:
                f_bev = self._forward_bev_features(
                    img, pc, t_current, cam_intrinsic, post_cam2ego_T,
                    xyz_g=xyz_g, feat_g=feat_g)
            if self.use_proj:
                if self._shared_pointgpt and xyz_g is not None:
                    f_proj, _ = self.proj_branch.forward_with_shared_point_features(
                        img_vit, t_current, cam_intrinsic, sensor_h, sensor_w,
                        xyz_g, feat_g)
                else:
                    f_proj, _ = self.proj_branch(
                        img_vit, pc_xyz, t_current, cam_intrinsic, sensor_h, sensor_w)
            rotation = self._predict_rotation(f_bev, f_proj)

            translation = torch.zeros(B, 3, device=img.device)
            loss_init_t = t_current if n_iter > 1 else init_T_to_camera
            step_loss, t_expected = self.loss_fn(
                pred_translation=translation,
                pred_rotation=rotation,
                pcs=pc_for_loss,
                gt_T_to_camera=gt_T_to_camera,
                init_T_to_camera=loss_init_t,
                mask=masks,
            )
            if self.aux_bev_head is not None and f_bev is not None:
                q_bev, _ = self.aux_bev_head(f_bev)
                aux_loss, _ = self.loss_fn(
                    pred_translation=translation, pred_rotation=q_bev,
                    pcs=pc_for_loss, gt_T_to_camera=gt_T_to_camera,
                    init_T_to_camera=loss_init_t, mask=masks)
                step_loss = {k: v + self.deep_supervision_weight * aux_loss[k]
                             if torch.is_tensor(v) else v
                             for k, v in step_loss.items()}
            if self.aux_proj_head is not None and f_proj is not None:
                q_proj, _ = self.aux_proj_head(f_proj)
                aux_loss, _ = self.loss_fn(
                    pred_translation=translation, pred_rotation=q_proj,
                    pcs=pc_for_loss, gt_T_to_camera=gt_T_to_camera,
                    init_T_to_camera=loss_init_t, mask=masks)
                step_loss = {k: v + self.deep_supervision_weight * aux_loss[k]
                             if torch.is_tensor(v) else v
                             for k, v in step_loss.items()}

            if self.gate_entropy_weight > 0 and self.fusion_variant == 'gated':
                if 'gate_bev' in self._last_fusion_meta and 'gate_proj' in self._last_fusion_meta:
                    g_b = self._last_fusion_meta['gate_bev']
                    g_p = self._last_fusion_meta['gate_proj']
                    gate_weights = torch.stack([g_b, g_p], dim=-1)
                    entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1).mean()
                    entropy_penalty = -self.gate_entropy_weight * entropy
                    step_loss['total_loss'] = step_loss['total_loss'] + entropy_penalty

            if total_loss is None:
                total_loss = step_loss
            else:
                gamma = 0.8
                w = gamma ** (n_iter - 1 - step_i)
                total_loss = {k: total_loss[k] + w * step_loss[k] for k in total_loss}
            if n_iter > 1 and step_i < n_iter - 1:
                from losses.quat_tools import batch_quat2mat, batch_tvector2mat
                rot_q = F.normalize(rotation, dim=-1)
                t_pred = torch.bmm(batch_tvector2mat(translation), batch_quat2mat(rot_q))
                t_current = torch.bmm(torch.linalg.inv(t_pred.float()), t_current.float())
                if self.rotation_only:
                    t_current = t_current.clone()
                    t_current[:, :3, 3] = init_T_to_camera[:, :3, 3]

        init_loss = None
        if out_init_loss:
            with torch.no_grad():
                t_zero = torch.zeros(B, 3, device=img.device)
                r_id = torch.zeros(B, 4, device=img.device)
                r_id[:, 0] = 1
                init_loss, _ = self.loss_fn(
                    pred_translation=t_zero, pred_rotation=r_id,
                    pcs=pc_for_loss, gt_T_to_camera=gt_T_to_camera,
                    init_T_to_camera=init_T_to_camera, mask=masks)

        return t_expected, init_loss, total_loss


def build_calib_model(args, device, img_shape, rotation_only, is_main=False, tprint=print):
    """Factory: CFBevRCalib, GeoMatchProjCalib, HybridTripleCalib, or legacy BEVCalib."""
    fusion_backend = getattr(args, 'fusion_backend', 'bev') or 'bev'
    if fusion_backend == 'cf_bev_r':
        from cf_bev_r_calib import CFBevRCalib
        model = CFBevRCalib.from_args(args, img_shape=img_shape).to(device)
        if is_main:
            tprint(f"[CF-BEV-R] fusion_backend=cf_bev_r, "
                   f"use_rocr={getattr(args, 'use_rocr', 1)}, "
                   f"groups={getattr(args, 'cf_n_groups', 128)}")
        return model
    if fusion_backend == 'geo_match_proj':
        from geo_match_proj_calib import GeoMatchProjCalib
        model = GeoMatchProjCalib.from_args(args, img_shape=img_shape).to(device)
        if is_main:
            tprint(f"[GMP] fusion_backend=geo_match_proj, "
                   f"proj_hw={getattr(args, 'projfusion_image_hw', [252, 448])}")
        return model
    if fusion_backend in HybridTripleCalib.FUSION_BACKENDS:
        model = HybridTripleCalib(
            img_shape=img_shape,
            fusion_backend=fusion_backend,
            pc_encoder_mode=getattr(args, 'pc_encoder_mode', 'pointgpt2bev'),
            fusion_variant=getattr(args, 'fusion_variant', 'gated'),
            rotation_only=rotation_only,
            enable_axis_loss=getattr(args, 'enable_axis_loss', 1) > 0,
            weight_axis_rotation=getattr(args, 'weight_axis_rotation', 0.5),
            axis_weights=tuple(float(x) for x in args.axis_weights.split(',')),
            use_balanced_axis_loss=getattr(args, 'use_balanced_axis_loss', 0) > 0,
            use_geodesic_loss=getattr(args, 'use_geodesic_loss', 0) > 0,
            weight_quat_norm=getattr(args, 'quat_norm_weight', 0.5),
            head_dropout=getattr(args, 'head_dropout', 0.1),
            fuser_type=getattr(args, 'fuser_type', 'diff'),
            bev_encoder=True,
            bev_pool_factor=getattr(args, 'bev_pool_factor', 4) or 4,
            cam2bev_mode='query',
            backbone_type=getattr(args, 'backbone_type', 'dinov2'),
            backbone_variant=getattr(args, 'backbone_variant', 'dinov2-small'),
            freeze_backbone=getattr(args, 'freeze_backbone', 1) > 0,
            freeze_layers=getattr(args, 'backbone_freeze_layers', None),
            backbone_weights=getattr(args, 'backbone_weights', None),
            projfusion_image_hw=tuple(getattr(args, 'projfusion_image_hw', [224, 448])),
            pointgpt_ckpt=getattr(args, 'native_cross_pointgpt_ckpt', None),
            pointgpt_config=getattr(args, 'native_cross_pointgpt_config', None),
            pointgpt_max_depth=getattr(args, 'native_cross_pointgpt_max_depth', 60.0),
            deep_supervision_weight=float(getattr(args, 'deep_supervision_weight', 0.2)),
            gate_entropy_weight=float(getattr(args, 'gate_entropy_weight', 0.2)),
            iterative_refine=getattr(args, 'iterative_refine', 0),
            projfusion_margin=float(getattr(args, 'native_cross_extend_ratio', 2.0) or 2.0),
            to_bev_mode=getattr(args, 'to_bev_mode', 'concat'),
            voxel_mode=getattr(args, 'voxel_mode', 'hard'),
            scatter_reduce=getattr(args, 'scatter_reduce', 'sum'),
        ).to(device)
        if is_main:
            tprint(f"[HTCN] fusion_backend={fusion_backend}, "
                   f"pc_encoder={getattr(args, 'pc_encoder_mode', 'pointgpt2bev')}, "
                   f"variant={getattr(args, 'fusion_variant', 'gated')}")
        return model

    from bev_calib import BEVCalib
    enable_axis_loss = args.enable_axis_loss > 0
    use_geodesic_loss = args.use_geodesic_loss > 0
    use_mlp_head = args.use_mlp_head > 0
    axis_weights_tuple = tuple(float(x) for x in args.axis_weights.split(','))
    use_foundation_depth = args.use_foundation_depth > 0
    fd_mode = args.fd_mode if use_foundation_depth else "replace"
    _num_domains = args.num_domains
    if args.domain_adversarial > 0 and _num_domains == 0:
        _num_domains = 21
    return BEVCalib(
        deformable=getattr(args, 'deformable', 0) > 0 if hasattr(args, 'deformable') else False,
        bev_encoder=True,
        img_shape=img_shape,
        rotation_only=rotation_only,
        enable_axis_loss=enable_axis_loss,
        weight_axis_rotation=args.weight_axis_rotation,
        axis_weights=axis_weights_tuple,
        drop_path_rate=args.drop_path_rate,
        head_dropout=args.head_dropout,
        use_geodesic_loss=use_geodesic_loss,
        use_mlp_head=use_mlp_head,
        bev_pool_factor=args.bev_pool_factor,
        use_foundation_depth=use_foundation_depth,
        depth_model_type=args.depth_model_type,
        fd_mode=fd_mode,
        voxel_mode=args.voxel_mode,
        to_bev_mode=args.to_bev_mode,
        scatter_reduce=args.scatter_reduce,
        fuser_type=args.fuser_type,
        cam_drop_prob=args.cam_drop_prob,
        cam_drop_mode=args.cam_drop_mode,
        intrinsic_input=args.intrinsic_input,
        use_pitch_branch=args.use_pitch_branch > 0,
        pitch_aux_weight=args.pitch_aux_weight,
        bev_instance_norm=args.bev_instance_norm > 0,
        use_contrastive_extrinsic=args.use_contrastive_extrinsic > 0,
        contrastive_weight=args.contrastive_weight,
        use_balanced_axis_loss=args.use_balanced_axis_loss > 0,
        weight_quat_norm=getattr(args, 'quat_norm_weight', 0.5),
        domain_adversarial=args.domain_adversarial > 0,
        domain_adversarial_weight=args.domain_adversarial_weight,
        num_domains=_num_domains,
        cam2bev_mode=args.cam2bev_mode,
        backbone_type=args.backbone_type,
        backbone_variant=args.backbone_variant,
        freeze_backbone=args.freeze_backbone > 0,
        freeze_layers=args.backbone_freeze_layers,
        backbone_weights=args.backbone_weights,
        correlation_fusion=getattr(args, 'correlation_fusion', 0) > 0,
        cross_correlation_fusion=getattr(args, 'cross_correlation_fusion', 0) > 0,
        explicit_tinit=getattr(args, 'explicit_tinit', 0) > 0,
        tinit_sensitivity_weight=getattr(args, 'tinit_sensitivity_weight', 0.0),
        iterative_refine=getattr(args, 'iterative_refine', 0),
        native_cross=getattr(args, 'native_cross', 0) > 0,
        native_cross_pc_groups=getattr(args, 'native_cross_pc_groups', 128),
        native_cross_n_harmonic=getattr(args, 'native_cross_n_harmonic', 6),
        native_cross_n_layers=getattr(args, 'native_cross_n_layers', 1),
        native_cross_dual_branch=getattr(args, 'native_cross_dual_branch', 1) > 0,
        native_cross_knn=getattr(args, 'native_cross_knn', 8),
        native_cross_use_fps=getattr(args, 'native_cross_use_fps', 1) > 0,
        native_cross_use_pointgpt=getattr(args, 'native_cross_use_pointgpt', 0) > 0,
        native_cross_pointgpt_ckpt=getattr(args, 'native_cross_pointgpt_ckpt', None),
        native_cross_pointgpt_config=getattr(args, 'native_cross_pointgpt_config', None),
        native_cross_pointgpt_max_depth=getattr(args, 'native_cross_pointgpt_max_depth', 50.0),
        native_cross_extend_ratio=getattr(args, 'native_cross_extend_ratio', 1.0),
        native_cross_iter_steps=getattr(args, 'native_cross_iter_steps', 0),
    ).to(device)
