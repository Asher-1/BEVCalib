"""HybridPoseHead: RefineHead + optional Match/Correlation path."""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion_heads import SingleBranchHead
from gmp.diff_epnp import DifferentiableEPnP
from gmp.local_correlation import LocalMultiHeadCorrelation
from gmp.match_head import CorrespondenceHead, correspondence_loss
from gmp.pose_composer import PoseComposer


class HybridPoseHead(nn.Module):
    """Unified RefineHead + Match EPnP + optional local correlation."""

    def __init__(
        self,
        proj_dim: int,
        hidden: int = 128,
        dropout: float = 0.15,
        use_match_head: bool = False,
        use_local_correlation: bool = False,
        num_correspondences: int = 64,
        compose_mode: str = 'match_then_refine',
        match_valid_ratio_min: float = 0.3,
        correspondence_supervision: bool = True,
        match_disable_fallback: bool = False,
        match_gate_use_init_ratio: bool = False,
        match_confidence_threshold: float = 0.2,
        match_corr_validity_mode: str = 'init',
        match_epnp_min_points: int = 4,
        token_dim: int = 384,
        vit_hw: tuple[int, int] = (252, 448),
        differentiable_epnp: bool = False,
        diff_epnp_warmup_epochs: int = 5,
    ):
        super().__init__()
        self.use_match_head = bool(use_match_head)
        self.use_local_correlation = bool(use_local_correlation)
        self.compose_mode = compose_mode
        self.match_valid_ratio_min = float(match_valid_ratio_min)
        self.match_disable_fallback = bool(match_disable_fallback)
        self.match_gate_use_init_ratio = bool(match_gate_use_init_ratio)
        self.num_correspondences = int(num_correspondences)
        self.vit_hw = tuple(vit_hw)
        self.token_dim = int(token_dim)
        self.differentiable_epnp = bool(differentiable_epnp)
        self.diff_epnp_warmup_epochs = int(diff_epnp_warmup_epochs)
        self._training_epoch = 0

        corr_out = 256
        self.local_corr = LocalMultiHeadCorrelation(
            token_dim=token_dim, out_dim=corr_out) if use_local_correlation else None
        match_in_dim = corr_out if use_local_correlation else token_dim

        self.refine_head = SingleBranchHead(proj_dim, hidden=hidden, dropout=dropout)
        self.match_head = CorrespondenceHead(
            token_dim=match_in_dim,
            num_points=num_correspondences,
            use_gt_supervision=correspondence_supervision,
            confidence_threshold=match_confidence_threshold,
            corr_validity_mode=match_corr_validity_mode,
        ) if use_match_head else None
        self.epnp = DifferentiableEPnP(
            min_valid_points=match_epnp_min_points,
            detach_rotation_grad=not differentiable_epnp,
        ) if use_match_head else None
        self.composer = PoseComposer()

    def set_training_epoch(self, epoch: int):
        self._training_epoch = int(epoch)

    def _epnp_force_detach(self, training: bool) -> bool | None:
        if not training or not self.differentiable_epnp:
            return None
        if self._training_epoch < self.diff_epnp_warmup_epochs:
            return True
        return False

    def _denorm_xyz(self, xyz_norm: torch.Tensor, max_depth: float) -> torch.Tensor:
        return xyz_norm * max_depth

    def forward(
        self,
        f_proj: torch.Tensor,
        cache: dict,
        max_depth: float,
        K: torch.Tensor,
        T_init: torch.Tensor,
        T_gt: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        meta: dict = {}
        xyz_norm = cache['xyz']
        fusion_tokens = cache['feat_3d']
        pc_xyz = self._denorm_xyz(xyz_norm, max_depth)
        vit_h, vit_w = self.vit_hw

        match_tokens = fusion_tokens
        if self.local_corr is not None:
            from camera_geometry import project_cam_to_pixel, transform_points_se3
            xyz_cam = transform_points_se3(pc_xyz, T_init)
            u, v, _z = project_cam_to_pixel(xyz_cam, K)
            pc_uv = torch.stack([u, v], dim=-1)
            match_tokens, corr_aux = self.local_corr(
                cache['feat_2d'], fusion_tokens, pc_uv, vit_h, vit_w)
            meta.update(corr_aux)

        q_refine, refine_meta = self.refine_head(f_proj)
        meta.update(refine_meta)

        R_match = None
        corr_loss = None
        mode = self.compose_mode

        if self.match_head is not None and self.epnp is not None:
            match_out = self.match_head(
                pc_xyz, match_tokens, T_init, K, vit_h, vit_w, T_gt=T_gt if training else None)
            meta['match_valid_ratio_init'] = match_out['match_valid_ratio_init']
            meta['match_valid_ratio_gt'] = match_out['match_valid_ratio_gt']
            meta['match_valid_ratio'] = match_out['match_valid_ratio']

            R_match, epnp_meta = self.epnp(
                match_out['xyz'],
                match_out['uv'],
                K,
                match_out['confidence'] * match_out['validity'].float(),
                T_init,
                force_detach=self._epnp_force_detach(training),
            )
            meta.update(epnp_meta)
            if training and self.differentiable_epnp:
                meta['epnp_grad_detached'] = float(
                    self._epnp_force_detach(training) or False)

            gate_ratio = (
                match_out['match_valid_ratio_init']
                if self.match_gate_use_init_ratio
                else match_out['match_valid_ratio_gt'])
            if not self.match_disable_fallback:
                low = gate_ratio < self.match_valid_ratio_min
                if low.any():
                    eye = torch.eye(
                        3, device=R_match.device, dtype=R_match.dtype)
                    R_match = torch.where(
                        low.view(-1, 1, 1),
                        eye,
                        R_match,
                    )
                    meta['match_fallback_ratio'] = low.float().mean()
                else:
                    meta['match_fallback_ratio'] = torch.zeros(1, device=R_match.device)
            else:
                meta['match_fallback_ratio'] = (
                    (gate_ratio < self.match_valid_ratio_min).float().mean())

            if training and 'uv_gt' in match_out:
                corr_loss = correspondence_loss(
                    match_out['uv'], match_out['uv_gt'], match_out['validity'])

        q_pred, compose_meta = self.composer(q_refine, R_match, mode=mode)
        meta.update(compose_meta)
        if corr_loss is not None:
            meta['correspondence_loss'] = corr_loss
        return q_pred, meta
