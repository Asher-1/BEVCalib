"""V40 GeoMatch-ProjCalib (GMP): proj + geo + optional P1 Match/Correlation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_heads import SingleBranchHead
from gmp.hybrid_pose_head import HybridPoseHead
from losses.geo_consistency_loss import GeoConsistencyLoss
from losses.losses import realworld_loss
from losses.quat_tools import batch_quat2mat, batch_tvector2mat
from projfusion_branch import ProjFusionBranch


class GeoMatchProjCalib(nn.Module):
    """AttenDualFusion proj path with V40 geometry fixes and optional geo/match losses."""

    FUSION_BACKENDS = ('geo_match_proj',)

    def __init__(
        self,
        img_shape=(640, 360),
        projfusion_image_hw=(252, 448),
        pointgpt_ckpt: str | None = None,
        pointgpt_config: str | None = None,
        pointgpt_max_depth: float = 60.0,
        projfusion_margin: float = 2.0,
        iterative_refine: int = 0,
        appearance_loss_weight: float = 0.0,
        depth_loss_weight: float = 0.0,
        geo_loss_start_epoch: int = 5,
        rotation_only: bool = True,
        enable_axis_loss: bool = True,
        weight_axis_rotation: float = 0.5,
        axis_weights=(1.0, 2.5, 1.0),
        use_balanced_axis_loss: bool = False,
        use_geodesic_loss: bool = False,
        weight_quat_norm: float = 0.5,
        head_dropout: float = 0.15,
        use_match_head: bool = False,
        use_local_correlation: bool = False,
        num_correspondences: int = 64,
        compose_mode: str = 'match_then_refine',
        correspondence_loss_weight: float = 0.0,
        correspondence_loss_start_epoch: int = 0,
        correspondence_loss_warmup_epochs: int = 0,
        correspondence_supervision: bool = True,
        match_valid_ratio_min: float = 0.3,
        match_disable_fallback: bool = False,
        match_gate_use_init_ratio: bool = False,
        match_confidence_threshold: float = 0.2,
        match_corr_validity_mode: str = 'init',
        match_epnp_min_points: int = 4,
        differentiable_epnp: bool = False,
        diff_epnp_warmup_epochs: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.rotation_only = rotation_only
        self.iterative_refine = iterative_refine
        self.appearance_loss_weight = float(appearance_loss_weight)
        self.depth_loss_weight = float(depth_loss_weight)
        self.geo_loss_start_epoch = int(geo_loss_start_epoch)
        self.use_match_head = bool(use_match_head)
        self.use_local_correlation = bool(use_local_correlation)
        self.correspondence_loss_weight = float(correspondence_loss_weight)
        self.correspondence_loss_start_epoch = int(correspondence_loss_start_epoch)
        self.correspondence_loss_warmup_epochs = int(correspondence_loss_warmup_epochs)
        self.compose_mode = compose_mode
        self.differentiable_epnp = bool(differentiable_epnp)
        self.diff_epnp_warmup_epochs = int(diff_epnp_warmup_epochs)
        self._training_epoch = 0
        self._profile_modules = False

        self.proj_branch = ProjFusionBranch(
            image_hw=tuple(projfusion_image_hw),
            pointgpt_ckpt=pointgpt_ckpt,
            pointgpt_config=pointgpt_config,
            margin=projfusion_margin,
            pointgpt_max_depth=pointgpt_max_depth,
            explicit_k_vit_scale=True,
        )
        proj_dim = self.proj_branch.out_dim
        token_dim = proj_dim
        self._max_depth = float(
            getattr(self.proj_branch.encoder, 'fnet_3d_max_depth', pointgpt_max_depth))

        if use_match_head or use_local_correlation:
            self.fusion_head = HybridPoseHead(
                proj_dim=proj_dim,
                dropout=head_dropout,
                use_match_head=use_match_head,
                use_local_correlation=use_local_correlation,
                num_correspondences=num_correspondences,
                compose_mode=compose_mode,
                match_valid_ratio_min=match_valid_ratio_min,
                correspondence_supervision=correspondence_supervision,
                match_disable_fallback=match_disable_fallback,
                match_gate_use_init_ratio=match_gate_use_init_ratio,
                match_confidence_threshold=match_confidence_threshold,
                match_corr_validity_mode=match_corr_validity_mode,
                match_epnp_min_points=match_epnp_min_points,
                token_dim=token_dim,
                vit_hw=tuple(projfusion_image_hw),
                differentiable_epnp=differentiable_epnp,
                diff_epnp_warmup_epochs=diff_epnp_warmup_epochs,
            )
        else:
            self.fusion_head = SingleBranchHead(proj_dim, dropout=head_dropout)

        self.loss_fn = realworld_loss(
            rotation_only=rotation_only,
            enable_axis_loss=enable_axis_loss,
            weight_axis_rotation=weight_axis_rotation,
            axis_weights=axis_weights,
            use_geodesic_loss=use_geodesic_loss,
            use_balanced_axis_loss=use_balanced_axis_loss,
            weight_quat_norm=weight_quat_norm,
        )
        geo_w = self.appearance_loss_weight + self.depth_loss_weight
        self.geo_loss_fn = GeoConsistencyLoss(
            appearance_weight=self.appearance_loss_weight,
            depth_weight=self.depth_loss_weight,
        ) if geo_w > 0 else None
        print(f"[GeoMatchProjCalib] proj_hw={projfusion_image_hw}, "
              f"iter={iterative_refine}, geo_w=({appearance_loss_weight},"
              f"{depth_loss_weight}), match={use_match_head}, "
              f"corr={use_local_correlation}, compose={compose_mode}, "
              f"diff_epnp={differentiable_epnp}, "
              f"diff_epnp_warmup={diff_epnp_warmup_epochs}")

    @classmethod
    def from_args(cls, args, img_shape=(640, 360)):
        return cls(
            img_shape=img_shape,
            projfusion_image_hw=tuple(getattr(args, 'projfusion_image_hw', [252, 448])),
            pointgpt_ckpt=getattr(args, 'native_cross_pointgpt_ckpt', None),
            pointgpt_config=getattr(args, 'native_cross_pointgpt_config', None),
            pointgpt_max_depth=float(getattr(args, 'native_cross_pointgpt_max_depth', 60.0)),
            projfusion_margin=float(getattr(args, 'native_cross_extend_ratio', 2.0) or 2.0),
            iterative_refine=int(getattr(args, 'iterative_refine', 0)),
            appearance_loss_weight=float(getattr(args, 'appearance_loss_weight', 0.0)),
            depth_loss_weight=float(getattr(args, 'depth_loss_weight', 0.0)),
            geo_loss_start_epoch=int(getattr(args, 'geo_loss_start_epoch', 5)),
            rotation_only=getattr(args, 'rotation_only', True),
            enable_axis_loss=getattr(args, 'enable_axis_loss', 1) > 0,
            weight_axis_rotation=float(getattr(args, 'weight_axis_rotation', 0.5)),
            axis_weights=tuple(float(x) for x in args.axis_weights.split(',')),
            use_balanced_axis_loss=getattr(args, 'use_balanced_axis_loss', 0) > 0,
            use_geodesic_loss=getattr(args, 'use_geodesic_loss', 0) > 0,
            weight_quat_norm=getattr(args, 'quat_norm_weight', 0.5),
            head_dropout=float(getattr(args, 'head_dropout', 0.15)),
            use_match_head=getattr(args, 'use_match_head', 0) > 0,
            use_local_correlation=getattr(args, 'use_local_correlation', 0) > 0,
            num_correspondences=int(getattr(args, 'num_correspondences', 64)),
            compose_mode=str(getattr(args, 'compose_mode', 'match_then_refine')),
            correspondence_loss_weight=float(getattr(args, 'correspondence_loss_weight', 0.0)),
            correspondence_loss_start_epoch=int(
                getattr(args, 'correspondence_loss_start_epoch', 0)),
            correspondence_loss_warmup_epochs=int(
                getattr(args, 'correspondence_loss_warmup_epochs', 0)),
            correspondence_supervision=getattr(args, 'correspondence_supervision', 1) > 0,
            match_valid_ratio_min=float(getattr(args, 'match_valid_ratio_min', 0.3)),
            match_disable_fallback=getattr(args, 'match_disable_fallback', 0) > 0,
            match_gate_use_init_ratio=getattr(args, 'match_gate_use_init_ratio', 0) > 0,
            match_confidence_threshold=float(
                getattr(args, 'match_confidence_threshold', 0.2)),
            match_corr_validity_mode=str(
                getattr(args, 'match_corr_validity_mode', 'init')),
            match_epnp_min_points=int(getattr(args, 'match_epnp_min_points', 4)),
            differentiable_epnp=getattr(args, 'differentiable_epnp', 0) > 0,
            diff_epnp_warmup_epochs=int(getattr(args, 'diff_epnp_warmup_epochs', 5)),
        )

    def set_training_epoch(self, epoch: int):
        self._training_epoch = int(epoch)
        if isinstance(self.fusion_head, HybridPoseHead):
            self.fusion_head.set_training_epoch(epoch)

    @staticmethod
    def _index_proj_cache(cache: dict, batch_indices: torch.Tensor) -> dict:
        return {k: v.index_select(0, batch_indices) for k, v in cache.items()}

    def _stash_proj_cache_for_jacobian(self, cache: dict, k_proj: torch.Tensor,
                                       sensor_h: int, sensor_w: int):
        """Save detached encoder cache from main forward for jac probe reuse."""
        self._jac_stash = {
            'cache': {k: v.detach() for k, v in cache.items()},
            'k_proj': k_proj.detach(),
            'sensor_h': int(sensor_h),
            'sensor_w': int(sensor_w),
        }

    def forward_pose_from_cache(
        self,
        init_T_to_camera: torch.Tensor,
        gt_T_to_camera: torch.Tensor,
        pcs: torch.Tensor,
        masks=None,
        batch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pose path only: cache_forward + fusion head (no DINOv2/PointGPT/geo/match loss).

        Requires _stash_proj_cache_for_jacobian() from the preceding main forward.
        Returns the same t_expected tensor as forward() (for jacobian correction).
        """
        stash = getattr(self, '_jac_stash', None)
        if stash is None:
            raise RuntimeError(
                "forward_pose_from_cache: no encoder cache stash; run main forward first")
        cache = stash['cache']
        k_proj = stash['k_proj']
        if batch_indices is not None:
            cache = self._index_proj_cache(cache, batch_indices)
            k_proj = k_proj.index_select(0, batch_indices)
        if init_T_to_camera.dim() == 4:
            init_T_to_camera = init_T_to_camera.squeeze(1)
        if gt_T_to_camera.dim() == 4:
            gt_T_to_camera = gt_T_to_camera.squeeze(1)
        b = init_T_to_camera.shape[0]
        pc_xyz = pcs if pcs.shape[-1] == 3 else pcs.transpose(1, 2).contiguous()
        f_proj, cache = self.proj_branch.forward_from_cache(
            cache, init_T_to_camera, k_proj, stash['sensor_h'], stash['sensor_w'])
        rotation, _ = self._predict_rotation(
            f_proj, cache, k_proj, init_T_to_camera, gt_T_to_camera)
        translation = torch.zeros(b, 3, device=init_T_to_camera.device)
        _, t_expected = self.loss_fn(
            pred_translation=translation,
            pred_rotation=rotation,
            pcs=pc_xyz,
            gt_T_to_camera=gt_T_to_camera,
            init_T_to_camera=init_T_to_camera,
            mask=masks,
        )
        return t_expected

    @staticmethod
    def _compose_T_pred(translation, rotation):
        rot_q = F.normalize(rotation, dim=-1)
        return torch.bmm(batch_tvector2mat(translation), batch_quat2mat(rot_q))

    def _get_k_proj(self, img, cam_intrinsic):
        _, k_proj, sensor_h, sensor_w, _ = self.proj_branch._proj_camera_info(
            img, cam_intrinsic)
        return k_proj, sensor_h, sensor_w

    def _predict_rotation(
        self,
        f_proj,
        cache,
        k_proj,
        t_init,
        t_gt,
    ):
        if isinstance(self.fusion_head, HybridPoseHead):
            rotation, meta = self.fusion_head(
                f_proj, cache, self._max_depth, k_proj, t_init,
                T_gt=t_gt, training=self.training)
            return rotation, meta
        rotation, meta = self.fusion_head(f_proj)
        return rotation, meta

    def _maybe_geo_loss(
        self,
        img,
        pc_for_loss,
        gt_T_to_camera,
        init_T_to_camera,
        cam_intrinsic,
        translation,
        rotation,
        masks,
    ):
        if self.geo_loss_fn is None:
            return {}
        if self._training_epoch < self.geo_loss_start_epoch:
            return {}
        t_pred = self._compose_T_pred(translation, rotation)
        with torch.cuda.amp.autocast(enabled=False):
            t_gt_expected = torch.matmul(
                torch.linalg.inv(t_pred.float()), init_T_to_camera.float())
        if self.rotation_only:
            t_gt_expected = t_gt_expected.clone()
            t_gt_expected[:, :3, 3] = init_T_to_camera[:, :3, 3]
        geo = self.geo_loss_fn(
            img=img,
            pc=pc_for_loss,
            T_pred=t_gt_expected,
            T_gt=gt_T_to_camera,
            cam_intrinsic=cam_intrinsic,
            mask=masks,
        )
        return geo

    def _effective_correspondence_weight(self) -> float:
        base = self.correspondence_loss_weight
        if base <= 0:
            return 0.0
        ep = self._training_epoch
        start = self.correspondence_loss_start_epoch
        if ep < start:
            return 0.0
        warm = self.correspondence_loss_warmup_epochs
        if warm > 0:
            frac = min(1.0, (ep - start + 1) / float(warm))
            return base * frac
        return base

    def forward(
        self,
        img,
        pc,
        gt_T_to_camera,
        init_T_to_camera,
        post_cam2ego_T,
        cam_intrinsic,
        masks=None,
        out_init_loss=False,
        domain_ids=None,
    ):
        del post_cam2ego_T, domain_ids
        if gt_T_to_camera.dim() == 4:
            gt_T_to_camera = gt_T_to_camera.squeeze(1)
        if init_T_to_camera.dim() == 4:
            init_T_to_camera = init_T_to_camera.squeeze(1)
        if cam_intrinsic.dim() == 4:
            cam_intrinsic = cam_intrinsic.squeeze(1)

        b = img.shape[0]
        pc_xyz = pc if pc.shape[-1] == 3 else pc.transpose(1, 2).contiguous()
        pc_for_loss = pc_xyz
        k_proj, sensor_h, sensor_w = self._get_k_proj(img, cam_intrinsic)

        n_iter = max(1, self.iterative_refine) if self.iterative_refine > 0 else 1
        t_current = init_T_to_camera
        total_loss = None
        t_expected = None
        last_meta = {}

        for step_i in range(n_iter):
            f_proj, cache = self.proj_branch(
                img, pc_xyz, t_current, cam_intrinsic, sensor_h, sensor_w)
            rotation, head_meta = self._predict_rotation(
                f_proj, cache, k_proj, t_current, gt_T_to_camera)
            last_meta = head_meta

            translation = torch.zeros(b, 3, device=img.device)
            loss_init_t = t_current if n_iter > 1 else init_T_to_camera
            step_loss, t_expected = self.loss_fn(
                pred_translation=translation,
                pred_rotation=rotation,
                pcs=pc_for_loss,
                gt_T_to_camera=gt_T_to_camera,
                init_T_to_camera=loss_init_t,
                mask=masks,
            )

            corr_w = self._effective_correspondence_weight()
            if corr_w > 0 and 'correspondence_loss' in head_meta:
                corr = head_meta['correspondence_loss'] * corr_w
                step_loss['correspondence_loss'] = head_meta['correspondence_loss'].detach()
                step_loss['correspondence_loss_weighted'] = corr.detach()
                step_loss['total_loss'] = step_loss['total_loss'] + corr

            for _ratio_key in (
                'match_valid_ratio',
                'match_valid_ratio_init',
                'match_valid_ratio_gt',
            ):
                if _ratio_key in head_meta:
                    step_loss[_ratio_key] = head_meta[_ratio_key].detach().mean()

            for _mk, _sk in (
                ('match_fallback_ratio', 'match_fallback_ratio'),
                ('epnp_grad_detached', 'epnp_grad_detached'),
                ('epnp_insufficient_ratio', 'epnp_insufficient_ratio'),
                ('epnp_mean_effective_points', 'epnp_mean_effective_points'),
                ('corr_valid_ratio', 'corr_valid_ratio'),
            ):
                if _mk in head_meta:
                    v = head_meta[_mk]
                    step_loss[_sk] = v.detach().mean() if torch.is_tensor(v) else v

            geo = self._maybe_geo_loss(
                img, pc_for_loss, gt_T_to_camera, loss_init_t, cam_intrinsic,
                translation, rotation, masks)
            if geo:
                step_loss['total_loss'] = step_loss['total_loss'] + geo['geo_consistency_loss']
                step_loss['appearance_loss'] = geo['appearance_loss'].detach()
                step_loss['depth_loss'] = geo['depth_loss'].detach()
                step_loss['geo_valid_ratio'] = geo['geo_valid_ratio']

            if total_loss is None:
                total_loss = step_loss
            else:
                gamma = 0.8
                w = gamma ** (n_iter - 1 - step_i)
                total_loss = {
                    k: total_loss[k] + w * step_loss[k]
                    if torch.is_tensor(total_loss[k]) else total_loss[k]
                    for k in total_loss
                }
            if n_iter > 1 and step_i < n_iter - 1:
                t_pred = self._compose_T_pred(translation, rotation)
                t_current = torch.bmm(
                    torch.linalg.inv(t_pred.float()), t_current.float())
                if self.rotation_only:
                    t_current = t_current.clone()
                    t_current[:, :3, 3] = init_T_to_camera[:, :3, 3]
            elif self.training and step_i == n_iter - 1:
                self._stash_proj_cache_for_jacobian(cache, k_proj, sensor_h, sensor_w)

        init_loss = None
        if out_init_loss:
            with torch.no_grad():
                t_zero = torch.zeros(b, 3, device=img.device)
                r_id = torch.zeros(b, 4, device=img.device)
                r_id[:, 0] = 1
                init_loss, _ = self.loss_fn(
                    pred_translation=t_zero, pred_rotation=r_id,
                    pcs=pc_for_loss, gt_T_to_camera=gt_T_to_camera,
                    init_T_to_camera=init_T_to_camera, mask=masks)

        return t_expected, init_loss, total_loss
