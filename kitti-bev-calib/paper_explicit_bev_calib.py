"""Paper implementation: implicit correspondences then explicit BEV alignment.

The two pose quantities deliberately have different semantics:

* ``T_coarse`` is an absolute LiDAR-to-camera transform.
* the fine head predicts a left-multiplied residual rotation from per-axis
  sine/cosine values and composes it with ``T_coarse`` exactly once.

This module does not inherit the repository's generic BEVCalib transformer.  The
explicit stage follows the paper: align the camera BEV with ``T_coarse``,
concatenate it with the LiDAR BEV, and regress the refinement with ResNet-18.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from native_cross_attention import PointEncoder
from modules.fov_classifier import FoVClassifier
from modules.position_encoding_3d import PositionEncoding3D
from modules.sim_cross_attention import SimCrossAttention
from losses.similarity_loss import (
    build_gt_correspondence,
    fov_classification_loss,
    similarity_loss,
)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.body(x) + identity)


class _PaperResNet50(nn.Module):
    """Paper ResNet-50 C5 backbone with an FPN P4 output at H/16 x W/16."""

    def __init__(self, out_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.in_channels = 64
        self.layer1 = self._layer(64, 3, 1)
        self.layer2 = self._layer(128, 4, 2)
        self.layer3 = self._layer(256, 6, 2)
        self.layer4 = self._layer(512, 3, 2)
        self.fpn_c4 = nn.Conv2d(1024, out_dim, 1)
        self.fpn_c5 = nn.Conv2d(2048, out_dim, 1)
        self.fpn_p4 = nn.Conv2d(out_dim, out_dim, 3, padding=1)

    def _layer(self, channels, blocks, stride):
        layers = [_Bottleneck(self.in_channels, channels, stride)]
        self.in_channels = channels * _Bottleneck.expansion
        layers.extend(_Bottleneck(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c4 = self.layer3(x)
        c5 = self.layer4(c4)
        p4 = self.fpn_c4(c4) + F.interpolate(
            self.fpn_c5(c5), size=c4.shape[-2:], mode='nearest')
        return self.fpn_p4(p4)


class _ExplicitBEVResNet18(nn.Module):
    """ResNet-18 encoder used by the paper's explicit alignment stage."""

    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.in_channels = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)

    def _make_layer(self, channels, blocks, stride):
        layers = [_BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(_BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def _matrix_to_sincos_xyz(rotation):
    """Return [sin/cos roll, pitch, yaw] for R=Rz(yaw)Ry(pitch)Rx(roll)."""
    sy = torch.sqrt(rotation[:, 0, 0].square() + rotation[:, 1, 0].square() + 1e-8)
    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = torch.atan2(-rotation[:, 2, 0], sy)
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return torch.stack([
        roll.sin(), roll.cos(), pitch.sin(), pitch.cos(), yaw.sin(), yaw.cos()
    ], dim=-1)


def _sincos_to_matrix(values):
    pairs = values.reshape(values.shape[0], 3, 2)
    pairs = torch.nn.functional.normalize(pairs, dim=-1)
    sr, cr = pairs[:, 0, 0], pairs[:, 0, 1]
    sp, cp = pairs[:, 1, 0], pairs[:, 1, 1]
    sy, cy = pairs[:, 2, 0], pairs[:, 2, 1]
    one = torch.ones_like(sr)
    zero = torch.zeros_like(sr)
    Rx = torch.stack([one, zero, zero, zero, cr, -sr, zero, sr, cr], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cp, zero, sp, zero, one, zero, -sp, zero, cp], -1).reshape(-1, 3, 3)
    Rz = torch.stack([cy, -sy, zero, sy, cy, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    return torch.bmm(torch.bmm(Rz, Ry), Rx), pairs.reshape(values.shape[0], 6)


def _materialize_misregistered_points(pc_xyz, gt_T, init_T):
    """Convert the repository's (raw points, init pose) API to the paper's P."""
    delta_T = torch.bmm(torch.linalg.inv(gt_T), init_T)
    target_T = torch.bmm(gt_T, torch.linalg.inv(delta_T))
    ones = torch.ones_like(pc_xyz[..., :1])
    pc_misaligned = torch.bmm(
        delta_T, torch.cat([pc_xyz, ones], dim=-1).transpose(1, 2)
    )[:, :3].transpose(1, 2)
    return pc_misaligned, delta_T, target_T


class _PaperImplicitPoseHead(nn.Module):
    """Paper Eq. (5)-(6): two MLP fusion stages, T2, then sin/cos head."""

    def __init__(self, feat_dim):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(2 * feat_dim, 2 * feat_dim), nn.ReLU(inplace=True),
            nn.Linear(2 * feat_dim, 2 * feat_dim), nn.ReLU(inplace=True))
        self.mlp2 = nn.Sequential(
            nn.Linear(4 * feat_dim, 4 * feat_dim), nn.ReLU(inplace=True),
            nn.Linear(4 * feat_dim, 4 * feat_dim), nn.ReLU(inplace=True))
        model_dim = 4 * feat_dim
        self.query = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.t2_attention = nn.ModuleList([
            nn.MultiheadAttention(model_dim, 8, batch_first=True) for _ in range(3)
        ])
        self.t2_norm1 = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(3)])
        self.t2_norm2 = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(3)])
        self.t2_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(model_dim, 4 * model_dim), nn.ReLU(inplace=True),
                          nn.Linear(4 * model_dim, model_dim)) for _ in range(3)
        ])
        self.rotation = nn.Sequential(
            nn.Linear(4 * feat_dim, 4 * feat_dim), nn.ReLU(inplace=True),
            nn.Linear(4 * feat_dim, 6))
        nn.init.normal_(self.rotation[-1].weight, mean=0.0, std=1e-4)
        with torch.no_grad():
            self.rotation[-1].bias.copy_(torch.tensor([0., 1., 0., 1., 0., 1.]))

    def forward(self, original_points, refined_points):
        first = self.mlp1(torch.cat([original_points, refined_points], dim=-1))
        global_max = first.max(dim=1, keepdim=True).values.expand(-1, first.shape[1], -1)
        fused = self.mlp2(torch.cat([first, global_max], dim=-1))
        aggregate = self.query.expand(fused.shape[0], -1, -1)
        for attention, norm1, norm2, ffn in zip(
                self.t2_attention, self.t2_norm1, self.t2_norm2, self.t2_ffn):
            attended, _ = attention(norm1(aggregate), fused, fused, need_weights=False)
            aggregate = aggregate + attended
            aggregate = aggregate + ffn(norm2(aggregate))
        raw = self.rotation(aggregate[:, 0])
        rotation, normalized = _sincos_to_matrix(raw)
        return rotation, normalized


class PaperExplicitBEVCalib(nn.Module):
    """Rotation-only implementation of the paper's complete two-stage model."""

    def __init__(self, img_shape, feat_dim=256, n_groups=256, knn=8,
                 sim_layers=3, depth_bins=16, sim_loss_weight=1.0,
                 fov_loss_weight=0.5, coarse_loss_weight=1.0,
                 rotation_only=True, enable_axis_loss=True,
                 weight_axis_rotation=0.5, axis_weights=(1.0, 1.0, 1.0),
                 use_geodesic_loss=True, weight_quat_norm=0.5,
                 head_dropout=0.1, backbone_type='swin',
                 backbone_variant='dinov2-small', freeze_backbone=False,
                 freeze_layers=None, backbone_weights=None,
                 voxel_mode='hard', to_bev_mode='concat',
                 scatter_reduce='sum', **_unused):
        super().__init__()
        if not rotation_only:
            raise ValueError("PaperExplicitBEVCalib supports rotation_only=True")
        self.rotation_only = True
        self.bev_encoder_use = True
        self.fusion_backend = 'paper_explicit_bev'
        self.paper_feat_dim = int(feat_dim)
        self.sim_loss_weight = float(sim_loss_weight)
        self.fov_loss_weight = float(fov_loss_weight)
        self.coarse_loss_weight = float(coarse_loss_weight)

        del (backbone_type, backbone_variant, freeze_backbone, freeze_layers,
             backbone_weights, voxel_mode, to_bev_mode, scatter_reduce,
             enable_axis_loss, weight_axis_rotation, axis_weights,
             use_geodesic_loss, weight_quat_norm)
        self.img_shape = tuple(img_shape)
        self.img_encoder = _PaperResNet50(feat_dim)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # FPS plus local PointNet grouping is the paper's fast PointNet++ path.
        self.paper_point_encoder = PointEncoder(
            in_dim=3, hidden_dim=128, out_dim=feat_dim,
            n_groups=n_groups, knn=knn, use_fps=True)
        self.paper_pos_enc = PositionEncoding3D(
            feat_dim=feat_dim, depth_bins=depth_bins,
            depth_min=1.0, depth_max=100.0,
            patch_size=16.0)
        self.paper_sim_cross_attn = SimCrossAttention(
            feat_dim=feat_dim, n_layers=sim_layers, n_heads=1,
            use_registry_token=True, dropout=0.1)
        self.paper_fov_classifier = FoVClassifier(feat_dim, hidden_dim=feat_dim // 2)
        self.paper_coarse_head = _PaperImplicitPoseHead(feat_dim)

        explicit_dim = 128
        self.paper_rgb_volume_proj = nn.Conv2d(feat_dim, explicit_dim, 1)
        self.paper_point_volume_proj = nn.Linear(feat_dim, explicit_dim)
        # Paper Sec. 4.2: X,Y,Z=200,8,200. Internally the tensor is ordered
        # north/east/down, so the vertical paper-Y dimension is stored last.
        self.volume_nx, self.volume_ny, self.volume_nz = 200, 200, 8
        self.volume_xbound = (-25.0, 25.0, 0.25)
        self.volume_ybound = (-25.0, 25.0, 0.25)
        self.volume_zbound = (-5.0, 5.0, 1.25)
        xs = torch.arange(self.volume_nx).float() * 0.25 - 25.0 + 0.125
        ys = torch.arange(self.volume_ny).float() * 0.25 - 25.0 + 0.125
        zs = torch.arange(self.volume_nz).float() * 1.25 - 5.0 + 0.625
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        self.register_buffer('paper_volume_grid', torch.stack([gx, gy, gz], dim=-1))

        bev_channels = 2 * explicit_dim * self.volume_nz
        self.paper_bev_fuser = nn.Sequential(
            nn.Conv2d(bev_channels, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        self.paper_bev_encoder = _ExplicitBEVResNet18(128)
        self.paper_c5_reduce = nn.Sequential(
            nn.Conv2d(512, 512, 7, bias=False), nn.ReLU(inplace=True),
        )
        self.paper_rotation_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(512, 6),
        )
        nn.init.normal_(self.paper_rotation_head[-1].weight, mean=0.0, std=1e-4)
        with torch.no_grad():
            self.paper_rotation_head[-1].bias.copy_(
                torch.tensor([0., 1., 0., 1., 0., 1.]))

    @staticmethod
    def _geodesic(R_a, R_b):
        rel = torch.bmm(R_a.transpose(1, 2).float(), R_b.float())
        trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        return torch.acos(((trace - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)).mean()

    def _implicit_alignment(self, img, pc, target_T, cam_intrinsic, masks):
        # The shared data pipeline emits uint8-range BGR tensors.
        image_rgb = img[:, [2, 1, 0]] / 255.0
        image_map = self.img_encoder((image_rgb - self.image_mean) / self.image_std)
        batch, _, feat_h, feat_w = image_map.shape
        image_tokens = image_map.flatten(2).transpose(1, 2)

        pc_xyz = pc if pc.shape[-1] == 3 else pc.transpose(1, 2).contiguous()
        mask_bool = masks.bool() if torch.is_tensor(masks) else None
        xyz_groups, point_tokens = self.paper_point_encoder(pc_xyz, mask=mask_bool)
        feature_stride = (float(img.shape[-2]) / float(feat_h),
                          float(img.shape[-1]) / float(feat_w))
        self.paper_pos_enc.patch_size = feature_stride
        image_tokens = (image_tokens
                        + self.paper_pos_enc(feat_h, feat_w, cam_intrinsic)
                        + self._sinusoidal_2d(feat_h, feat_w, image_tokens))
        original_point_tokens = point_tokens
        point_tokens, sim_matrices, layer_outputs = self.paper_sim_cross_attn(
            pc_features=point_tokens, img_features=image_tokens,
            return_layer_outputs=True)

        _, gt_fov = build_gt_correspondence(
            xyz_groups, target_T, cam_intrinsic, feat_h, feat_w,
            patch_size=feature_stride,
            use_registry_token=True)
        R_coarse, coarse_sincos = self.paper_coarse_head(
            original_point_tokens, point_tokens)
        T_coarse = target_T.new_zeros(batch, 4, 4)
        T_coarse[:, 3, 3] = 1.0
        T_coarse[:, :3, :3] = R_coarse
        return (T_coarse, coarse_sincos, xyz_groups, original_point_tokens,
                pc_xyz, sim_matrices, gt_fov, layer_outputs, image_map, feat_h, feat_w)

    @staticmethod
    def _sinusoidal_2d(height, width, reference):
        dim = reference.shape[-1]
        quarter = max(1, dim // 4)
        omega = torch.arange(quarter, device=reference.device, dtype=reference.dtype)
        omega = 1.0 / (10000 ** (omega / max(1, quarter - 1)))
        y = torch.arange(height, device=reference.device, dtype=reference.dtype)[:, None] * omega[None]
        x = torch.arange(width, device=reference.device, dtype=reference.dtype)[:, None] * omega[None]
        y_enc = torch.cat([y.sin(), y.cos()], dim=-1)[:, None].expand(-1, width, -1)
        x_enc = torch.cat([x.sin(), x.cos()], dim=-1)[None].expand(height, -1, -1)
        enc = torch.cat([x_enc, y_enc], dim=-1).reshape(1, height * width, -1)
        if enc.shape[-1] < dim:
            enc = F.pad(enc, (0, dim - enc.shape[-1]))
        return enc[..., :dim].expand(reference.shape[0], -1, -1)

    def _rgb_bev(self, image_map, cam_intrinsic, image_h, image_w):
        """Paper Eq. (12): project camera-centred volume and bilinearly sample RGB features."""
        batch = image_map.shape[0]
        bev_xyz = self.paper_volume_grid.reshape(-1, 3).to(dtype=image_map.dtype)
        # Stored grid is (north, east, down); camera coordinates are (east, down, north).
        camera_xyz = bev_xyz[:, [1, 2, 0]].unsqueeze(0).expand(batch, -1, -1)
        uvw = torch.bmm(cam_intrinsic.to(image_map.dtype), camera_xyz.transpose(1, 2)).transpose(1, 2)
        depth = uvw[..., 2]
        u = uvw[..., 0] / depth.clamp(min=1e-6)
        v = uvw[..., 1] / depth.clamp(min=1e-6)
        sample_grid = torch.stack([
            2.0 * (u + 0.5) / float(image_w) - 1.0,
            2.0 * (v + 0.5) / float(image_h) - 1.0,
        ], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            self.paper_rgb_volume_proj(image_map), sample_grid,
            mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(-1)
        valid = ((depth > 0) & (u >= 0) & (u < image_w)
                 & (v >= 0) & (v < image_h)).unsqueeze(1)
        sampled = sampled * valid
        volume = sampled.reshape(batch, -1, self.volume_nx, self.volume_ny, self.volume_nz)
        return volume, valid.float().mean()

    @staticmethod
    def _propagate_point_features(pc_xyz, xyz_groups, group_features, masks):
        """PointNet++ feature propagation from sampled groups back to input points."""
        distances = torch.cdist(pc_xyz, xyz_groups)
        k = min(3, xyz_groups.shape[1])
        nearest_dist, nearest_idx = distances.topk(k, dim=-1, largest=False)
        weights = 1.0 / nearest_dist.clamp(min=1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        batch_idx = torch.arange(pc_xyz.shape[0], device=pc_xyz.device)[:, None, None]
        neighbors = group_features[batch_idx, nearest_idx]
        propagated = (neighbors * weights.unsqueeze(-1)).sum(dim=2)
        if torch.is_tensor(masks):
            propagated = propagated * masks.bool().unsqueeze(-1)
        return propagated

    def _point_bev(self, point_xyz, point_features, T_coarse, masks=None):
        """Paper Eq. (7): transform and scatter downsampled point features to the 3D grid."""
        batch, _, _ = point_xyz.shape
        ones = torch.ones_like(point_xyz[..., :1])
        camera_xyz = torch.bmm(
            T_coarse, torch.cat([point_xyz, ones], dim=-1).transpose(1, 2))[:, :3].transpose(1, 2)
        bev_xyz = camera_xyz[..., [2, 0, 1]]
        ix = torch.floor((bev_xyz[..., 0] - self.volume_xbound[0]) / self.volume_xbound[2]).long()
        iy = torch.floor((bev_xyz[..., 1] - self.volume_ybound[0]) / self.volume_ybound[2]).long()
        iz = torch.floor((bev_xyz[..., 2] - self.volume_zbound[0]) / self.volume_zbound[2]).long()
        valid = ((ix >= 0) & (ix < self.volume_nx) & (iy >= 0) & (iy < self.volume_ny)
                 & (iz >= 0) & (iz < self.volume_nz))
        if torch.is_tensor(masks):
            valid = valid & masks.bool()
        linear = ((ix.clamp(0, self.volume_nx - 1) * self.volume_ny
                   + iy.clamp(0, self.volume_ny - 1)) * self.volume_nz
                  + iz.clamp(0, self.volume_nz - 1))
        features = self.paper_point_volume_proj(point_features) * valid.unsqueeze(-1)
        channels = features.shape[-1]
        volume = features.new_zeros(batch, self.volume_nx * self.volume_ny * self.volume_nz, channels)
        counts = features.new_zeros(batch, self.volume_nx * self.volume_ny * self.volume_nz, 1)
        volume.scatter_add_(1, linear.unsqueeze(-1).expand(-1, -1, channels), features)
        counts.scatter_add_(1, linear.unsqueeze(-1), valid.unsqueeze(-1).to(features.dtype))
        volume = volume / counts.clamp(min=1.0)
        volume = volume.transpose(1, 2).reshape(
            batch, channels, self.volume_nx, self.volume_ny, self.volume_nz)
        denominator = (masks.bool().sum().clamp(min=1) if torch.is_tensor(masks)
                       else valid.new_tensor(valid.numel()).clamp(min=1))
        valid_ratio = valid.sum().float() / denominator
        return volume, valid_ratio

    @staticmethod
    def _flatten_height(volume):
        batch, channels, nx, ny, nz = volume.shape
        return volume.permute(0, 1, 4, 2, 3).reshape(batch, channels * nz, nx, ny)

    def forward(self, img, pc, gt_T_to_camera, init_T_to_camera,
                post_cam2ego_T, cam_intrinsic, masks=None,
                out_init_loss=False, domain_ids=None,
                gnc_mu=None, gnc_axis_weights=None, gnc_noise_bound_sq=None,
                gnc_use_irls=False, **_unused_fwd):
        del domain_ids
        # The paper operates on a mis-registered point cloud P and estimates the
        # transform that registers that P to the camera.  The repository passes
        # raw LiDAR points plus an initial calibration, so materialize the
        # canonical right-multiplied LiDAR-frame perturbation here:
        #   delta = inv(T_gt) @ T_init, P_mis = delta @ P.
        # The paper target is then T_target = T_gt @ inv(delta), because
        # T_target @ P_mis == T_gt @ P.  Returning T_final @ delta converts the
        # paper prediction back to the repository's absolute-extrinsic API.
        pc_xyz_raw = pc if pc.shape[-1] == 3 else pc.transpose(1, 2).contiguous()
        pc_misaligned, delta_T, target_T = _materialize_misregistered_points(
            pc_xyz_raw, gt_T_to_camera, init_T_to_camera)

        (T_coarse, coarse_sincos, xyz_groups, group_features, pc_xyz,
         sim_matrices, gt_fov, layer_outputs, image_map, feat_h, feat_w) = self._implicit_alignment(
            img, pc_misaligned, target_T, cam_intrinsic, masks)
        T_coarse[:, :3, 3] = init_T_to_camera[:, :3, 3]

        del post_cam2ego_T
        rgb_volume, rgb_volume_valid = self._rgb_bev(
            image_map, cam_intrinsic, img.shape[-2], img.shape[-1])
        point_features = self._propagate_point_features(
            pc_xyz, xyz_groups, group_features, masks)
        point_volume, point_volume_valid = self._point_bev(
            pc_xyz, point_features, T_coarse, masks)
        cam_bev = self._flatten_height(rgb_volume)
        lidar_bev = self._flatten_height(point_volume)
        fused_bev = self.paper_bev_fuser(torch.cat([cam_bev, lidar_bev], dim=1))
        c5 = self.paper_bev_encoder(fused_bev)
        fine_features = self.paper_c5_reduce(c5).flatten(1)
        fine_raw = self.paper_rotation_head(fine_features)
        R_fine, _ = _sincos_to_matrix(fine_raw)
        R_final = torch.bmm(R_fine, T_coarse[:, :3, :3])
        T_final = T_coarse.clone()
        T_final[:, :3, :3] = R_final

        gt_corr, gt_fov = build_gt_correspondence(
            xyz_groups, target_T, cam_intrinsic, feat_h, feat_w,
            patch_size=(float(img.shape[-2]) / float(feat_h),
                        float(img.shape[-1]) / float(feat_w)),
            use_registry_token=True)
        sim = similarity_loss(sim_matrices, gt_corr, gt_fov)
        fov = sum(fov_classification_loss(self.paper_fov_classifier(q), gt_fov)
                  for q in layer_outputs) / len(layer_outputs)
        coarse_target = _matrix_to_sincos_xyz(target_T[:, :3, :3])
        target_R = target_T[:, :3, :3]
        _use_gnc = (
            gnc_mu is not None
            and float(gnc_mu) > 0
            and gnc_axis_weights is not None
        )
        if _use_gnc:
            from losses.gnc_loss import axis_error_deg, weighted_gnc_axis_mean
            _aw = tuple(float(x) for x in gnc_axis_weights)
            _nb = float(gnc_noise_bound_sq) if gnc_noise_bound_sq is not None else 0.0025
            coarse = weighted_gnc_axis_mean(
                axis_error_deg(target_R, T_coarse[:, :3, :3]).abs(),
                float(gnc_mu), _aw, noise_bound_sq=_nb, use_irls_weight=bool(gnc_use_irls))
            fine = weighted_gnc_axis_mean(
                axis_error_deg(target_R, R_final).abs(),
                float(gnc_mu), _aw, noise_bound_sq=_nb, use_irls_weight=bool(gnc_use_irls))
        else:
            coarse = torch.nn.functional.l1_loss(coarse_sincos, coarse_target)
            # Paper Eq. (14)-(16): supervise the composed final calibration so the
            # BEV-alignment loss reaches both the fine and coarse pose predictors.
            final_sincos = _matrix_to_sincos_xyz(R_final)
            fine = torch.nn.functional.l1_loss(final_sincos, coarse_target)
        T_absolute = torch.bmm(T_final, delta_T)
        rotation_deg = self._geodesic(
            T_absolute[:, :3, :3], gt_T_to_camera[:, :3, :3]) * (180.0 / torch.pi)
        loss = {
            'total_loss': (fine + self.sim_loss_weight * sim
                           + self.fov_loss_weight * fov
                           + self.coarse_loss_weight * coarse),
            'rotation_loss': rotation_deg.detach(),
            'geodesic_loss': rotation_deg.detach(),
            'translation_loss': fine.new_zeros(()),
            'quat_norm_loss': fine.new_zeros(()),
            'PC_reproj_loss': fine.new_zeros(()),
            'paper_fine_rotation_loss': fine.detach(),
            'paper_rgb_volume_valid_ratio': rgb_volume_valid.detach(),
            'paper_point_volume_valid_ratio': point_volume_valid.detach(),
        }
        loss['paper_similarity_loss'] = sim.detach()
        loss['paper_fov_loss'] = fov.detach()
        loss['paper_coarse_rotation_loss'] = coarse.detach()

        init_loss = None
        if out_init_loss:
            with torch.no_grad():
                init_deg = self._geodesic(
                    init_T_to_camera[:, :3, :3], gt_T_to_camera[:, :3, :3])
                init_loss = {'rotation_loss': init_deg * (180.0 / torch.pi)}
        return T_absolute, init_loss, loss

    @torch.no_grad()
    def iterative_inference(self, imgs, pcd, T_init, cam_intrinsic,
                            n_iters=1, pcd_mask=None):
        # Coarse pose is absolute, so repeated passes do not compose it again.
        identity = torch.eye(4, device=imgs.device, dtype=imgs.dtype)
        identity = identity.unsqueeze(0).expand(imgs.shape[0], -1, -1)
        current = T_init
        for _ in range(max(1, int(n_iters))):
            current, _, _ = self.forward(
                imgs, pcd, current, current, identity, cam_intrinsic,
                masks=pcd_mask, out_init_loss=False)
        return current

    @classmethod
    def from_args(cls, args, img_shape):
        return cls(
            img_shape=img_shape,
            feat_dim=getattr(args, 'paper_feat_dim', 256),
            n_groups=getattr(args, 'paper_n_groups', 256),
            knn=getattr(args, 'paper_knn', 8),
            sim_layers=getattr(args, 'paper_sim_layers', 3),
            depth_bins=getattr(args, 'paper_depth_bins', 16),
            sim_loss_weight=getattr(args, 'paper_sim_loss_weight', 1.0),
            fov_loss_weight=getattr(args, 'paper_fov_loss_weight', 0.5),
            coarse_loss_weight=getattr(args, 'paper_coarse_loss_weight', 1.0),
            rotation_only=True,
            enable_axis_loss=getattr(args, 'enable_axis_loss', 1) > 0,
            weight_axis_rotation=getattr(args, 'weight_axis_rotation', 0.5),
            axis_weights=tuple(float(x) for x in str(args.axis_weights).split(',')),
            use_geodesic_loss=getattr(args, 'use_geodesic_loss', 1) > 0,
            weight_quat_norm=getattr(args, 'quat_norm_weight', 0.5),
            head_dropout=getattr(args, 'head_dropout', 0.1),
            backbone_type=getattr(args, 'backbone_type', 'swin'),
            backbone_variant=getattr(args, 'backbone_variant', 'dinov2-small'),
            freeze_backbone=getattr(args, 'freeze_backbone', 0) > 0,
            freeze_layers=getattr(args, 'backbone_freeze_layers', None),
            backbone_weights=getattr(args, 'backbone_weights', None),
            voxel_mode=getattr(args, 'voxel_mode', 'hard'),
            to_bev_mode=getattr(args, 'to_bev_mode', 'concat'),
            scatter_reduce=getattr(args, 'scatter_reduce', 'sum'),
        )
