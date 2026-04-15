"""
Foundation Depth V2 — fixes for per-image min-max, aligner capacity, and
support for depth-supervision and dual-path fusion modes.

Modes (selected by fd_mode):
  "replace"       : (方案A-fixed) FD depth distribution replaces LSS depth_net
  "dual_path"     : (方案A-fusion) FD depth concatenated with features as extra input
  "supervision"   : (方案C) Standard LSS depth_net + auxiliary depth supervision loss

Key fixes vs V1:
  1. Remove per-image min-max; keep raw disparity scale (global log-domain)
  2. SpatialAligner with learned spatial scale/shift (conv-based, not 2-param)
  3. Dual-path option: FD depth as an extra channel to depth_net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


class SpatialDepthAligner(nn.Module):
    """Spatially-varying affine alignment from relative depth to metric depth.

    Unlike V1's 2-parameter global affine, this uses a lightweight CNN to
    predict per-pixel scale and shift, allowing the network to compensate
    for perspective distortion and scene-dependent depth ranges.
    """

    def __init__(self, init_scale=50.0, init_shift=5.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 2, 1),  # 2 channels: scale_logit, shift
        )
        nn.init.constant_(self.net[-1].bias[0], float(torch.tensor(init_scale).log()))
        nn.init.constant_(self.net[-1].bias[1], init_shift)

    def forward(self, relative_depth):
        """
        Args:
            relative_depth: (B, 1, H, W) raw disparity-derived depth (NOT min-max normalized)
        Returns:
            metric_depth: (B, 1, H, W) estimated metric depth in meters
        """
        params = self.net(relative_depth)
        scale = params[:, 0:1].exp()
        shift = params[:, 1:2]
        metric = scale * relative_depth + shift
        return metric.clamp(min=0.5, max=150.0)


class DepthBinConverterV2(nn.Module):
    """Convert continuous depth to bin distribution.
    Same Laplace kernel as V1, but with per-pixel learnable temperature.
    """

    def __init__(self, d_start, d_end, d_step, init_sigma=2.0):
        super().__init__()
        depth_bins = torch.arange(d_start, d_end, d_step, dtype=torch.float)
        self.register_buffer('depth_bins', depth_bins)
        self.D = len(depth_bins)
        self.log_sigma = nn.Parameter(torch.tensor(init_sigma).log())

    @property
    def sigma(self):
        return self.log_sigma.exp().clamp(min=0.1, max=20.0)

    def forward(self, depth_map):
        sigma = self.sigma
        bins = self.depth_bins.view(1, -1, 1, 1)
        dist = -(depth_map - bins).abs() / sigma
        return F.softmax(dist, dim=1)


class FoundationDepthProviderV2(nn.Module):
    """Wraps frozen MiDaS with FIXED depth output (no per-image min-max).

    Key fix: outputs raw 1/disparity in log-domain, preserving cross-image
    depth ordering. The downstream SpatialAligner handles metric conversion.
    """

    def __init__(self, model_type="midas_small"):
        super().__init__()
        self.model_type = model_type
        self._model = None
        self._loaded = False

        self.midas_repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ckpt', 'MiDaS-master'
        )
        self.weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ckpt'
        )
        self.weight_files = {
            'midas_small': 'midas_v21_small_256.pt',
            'dpt_swin2_t': 'dpt_swin2_tiny_256.pt',
            'dpt_beit_l': 'dpt_beit_large_512.pt',
        }

    def _load_model(self, device):
        if self._loaded:
            return
        weight_name = self.weight_files.get(self.model_type)
        weight_path = os.path.join(os.path.realpath(self.weights_dir), weight_name) if weight_name else None

        if not weight_path or not os.path.isfile(weight_path):
            print(f"[FDv2] ERROR: weights not found at {weight_path}", file=sys.stderr, flush=True)
            self._model = None
            self._loaded = True
            return

        midas_pkg = os.path.realpath(self.midas_repo)
        if midas_pkg not in sys.path:
            sys.path.insert(0, midas_pkg)

        if self.model_type == 'midas_small':
            from midas.midas_net_custom import MidasNet_small
            model = MidasNet_small(
                path=weight_path, features=64, backbone="efficientnet_lite3",
                exportable=True, non_negative=True, blocks={'expand': True},
            )
        else:
            from midas.dpt_depth import DPTDepthModel
            cfgs = {
                'dpt_swin2_t': dict(path=weight_path, backbone="swin2t16_256", non_negative=True),
                'dpt_beit_l': dict(path=weight_path, backbone="beitl16_512", non_negative=True),
            }
            model = DPTDepthModel(**cfgs[self.model_type])

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model.to(device)
        self._loaded = True
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[FDv2] {self.model_type} loaded ({n_params:.1f}M params, frozen)")

    @torch.no_grad()
    def forward(self, imgs, target_h, target_w):
        """
        Returns: (B*N, 1, target_h, target_w) — log(1/disp) preserving cross-image ordering.
        NOT min-max normalized.
        """
        device = imgs.device
        self._load_model(device)
        if self._model is None:
            return self._fallback(imgs, target_h, target_w)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        imgs_rgb = imgs * std + mean  # denormalize to [0,1]

        size = (256, 256) if "small" in self.model_type else (384, 384)
        inp = F.interpolate(imgs_rgb, size=size, mode='bilinear', align_corners=False)

        disp = self._model(inp)
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)

        disp = F.interpolate(disp, size=(target_h, target_w), mode='bilinear', align_corners=False)
        depth = torch.log(1.0 / (disp + 1e-6) + 1.0)
        return depth

    def _fallback(self, imgs, target_h, target_w):
        BN = imgs.shape[0]
        ramp = torch.linspace(1.0, 0.0, target_h, device=imgs.device).view(1, 1, target_h, 1)
        return ramp.expand(BN, 1, target_h, target_w)


class FoundationDepthLSSv2(nn.Module):
    """V2 FD-LSS: replace mode (方案A-fixed). Uses SpatialAligner instead of 2-param."""

    def __init__(self, transformedImgShape=None, featureShape=None,
                 d_conf=None, out_channels=128, depth_model_type="midas_small"):
        super().__init__()
        if transformedImgShape is None:
            transformedImgShape = (3, 256, 704)
        if featureShape is None:
            featureShape = (256, transformedImgShape[1] // 8, transformedImgShape[2] // 8)
        if d_conf is None:
            from bev_settings import d_conf as default_d_conf
            d_conf = default_d_conf

        _, self.orfH, self.orfW = transformedImgShape
        self.fC, self.fH, self.fW = featureShape
        self.d_st, self.d_end, self.d_step = d_conf
        self.D = len(torch.arange(self.d_st, self.d_end, self.d_step))
        self.out_channels = out_channels

        self.frustum = self._create_frustum()
        self.depth_provider = FoundationDepthProviderV2(model_type=depth_model_type)
        self.depth_aligner = SpatialDepthAligner()
        self.depth_to_bins = DepthBinConverterV2(self.d_st, self.d_end, self.d_step)

        self.feature_net = nn.Sequential(
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, out_channels, 1),
        )
        print(f"[FDv2-Replace] D={self.D}, out={out_channels}, model={depth_model_type}")

    def _create_frustum(self):
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype=torch.float) \
            .view(-1, 1, 1).expand(-1, self.fH, self.fW)
        xs = torch.linspace(0, self.orfW - 1, self.fW).view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        ys = torch.linspace(0, self.orfH - 1, self.fH).view(1, self.fH, 1).expand(self.D, self.fH, self.fW)
        return nn.Parameter(torch.stack((xs, ys, ds), -1), requires_grad=False)

    def get_geometry(self, cam2ego_rot, cam2ego_trans, cam_intrins,
                     post_cam2ego_rot, post_cam2ego_trans):
        B, N, _ = cam2ego_trans.shape
        points = self.frustum - post_cam2ego_trans.view(B, N, 1, 1, 1, 3)
        points = torch.linalg.inv(post_cam2ego_rot).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]), 5)
        combine = cam2ego_rot.matmul(torch.linalg.inv(cam_intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += cam2ego_trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feature(self, img_feats, raw_imgs=None):
        B, N, C, H, W = img_feats.shape
        img_feats_flat = img_feats.view(B * N, C, H, W)

        if raw_imgs is not None:
            raw_flat = raw_imgs.view(B * N, 3, raw_imgs.shape[3], raw_imgs.shape[4])
            rel_depth = self.depth_provider(raw_flat, H, W)
        else:
            rel_depth = self.depth_provider._fallback(img_feats_flat, H, W)

        metric_depth = self.depth_aligner(rel_depth)
        depth_dist = self.depth_to_bins(metric_depth)

        features = self.feature_net(img_feats_flat)
        out = depth_dist.unsqueeze(1) * features.unsqueeze(2)
        out = out.view(B, N, self.out_channels, self.D, H, W)
        return out.permute(0, 1, 3, 4, 5, 2)

    def forward(self, cam2ego_rot, cam2ego_trans, cam_intrins,
                post_cam2ego_rot, post_cam2ego_trans, img_feats, raw_imgs=None):
        geometry = self.get_geometry(cam2ego_rot, cam2ego_trans, cam_intrins,
                                     post_cam2ego_rot, post_cam2ego_trans)
        img_depth_feature = self.get_cam_feature(img_feats, raw_imgs=raw_imgs)
        return geometry, img_depth_feature


class DualPathLSS(nn.Module):
    """方案A-fusion: FD depth as extra input channel to standard depth_net.

    The key insight: don't throw away the learned depth head — just give it a
    better prior. depth_net input becomes [img_feats; fd_depth] → (fC+1) channels.
    """

    def __init__(self, transformedImgShape=None, featureShape=None,
                 d_conf=None, out_channels=128, depth_model_type="midas_small"):
        super().__init__()
        if transformedImgShape is None:
            transformedImgShape = (3, 256, 704)
        if featureShape is None:
            featureShape = (256, transformedImgShape[1] // 8, transformedImgShape[2] // 8)
        if d_conf is None:
            from bev_settings import d_conf as default_d_conf
            d_conf = default_d_conf

        _, self.orfH, self.orfW = transformedImgShape
        self.fC, self.fH, self.fW = featureShape
        self.d_st, self.d_end, self.d_step = d_conf
        self.D = len(torch.arange(self.d_st, self.d_end, self.d_step))
        self.out_channels = out_channels

        self.frustum = self._create_frustum()
        self.depth_provider = FoundationDepthProviderV2(model_type=depth_model_type)

        self.depth_net = nn.Sequential(
            nn.Conv2d(self.fC + 1, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, out_channels + self.D, 1),
        )
        print(f"[DualPathLSS] D={self.D}, in_channels={self.fC}+1(FD), out={out_channels}")

    def _create_frustum(self):
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype=torch.float) \
            .view(-1, 1, 1).expand(-1, self.fH, self.fW)
        xs = torch.linspace(0, self.orfW - 1, self.fW).view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        ys = torch.linspace(0, self.orfH - 1, self.fH).view(1, self.fH, 1).expand(self.D, self.fH, self.fW)
        return nn.Parameter(torch.stack((xs, ys, ds), -1), requires_grad=False)

    def get_geometry(self, cam2ego_rot, cam2ego_trans, cam_intrins,
                     post_cam2ego_rot, post_cam2ego_trans):
        B, N, _ = cam2ego_trans.shape
        points = self.frustum - post_cam2ego_trans.view(B, N, 1, 1, 1, 3)
        points = torch.linalg.inv(post_cam2ego_rot).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]), 5)
        combine = cam2ego_rot.matmul(torch.linalg.inv(cam_intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += cam2ego_trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feature(self, img_feats, raw_imgs=None):
        B, N, C, H, W = img_feats.shape
        img_feats_flat = img_feats.view(B * N, C, H, W)

        if raw_imgs is not None:
            raw_flat = raw_imgs.view(B * N, 3, raw_imgs.shape[3], raw_imgs.shape[4])
            fd_depth = self.depth_provider(raw_flat, H, W)
        else:
            fd_depth = self.depth_provider._fallback(img_feats_flat, H, W)

        fused = torch.cat([img_feats_flat, fd_depth], dim=1)  # (BN, fC+1, H, W)
        out = self.depth_net(fused)

        depth = out[:, :self.D].softmax(dim=1)
        features = out[:, self.D:self.D + self.out_channels]
        lifted = depth.unsqueeze(1) * features.unsqueeze(2)
        lifted = lifted.view(B, N, self.out_channels, self.D, H, W)
        return lifted.permute(0, 1, 3, 4, 5, 2)

    def forward(self, cam2ego_rot, cam2ego_trans, cam_intrins,
                post_cam2ego_rot, post_cam2ego_trans, img_feats, raw_imgs=None):
        geometry = self.get_geometry(cam2ego_rot, cam2ego_trans, cam_intrins,
                                     post_cam2ego_rot, post_cam2ego_trans)
        img_depth_feature = self.get_cam_feature(img_feats, raw_imgs=raw_imgs)
        return geometry, img_depth_feature


class DepthSupervisionHelper(nn.Module):
    """方案C: Auxiliary depth supervision for standard LSS depth head.

    Uses frozen FD model to generate pseudo-GT depth bins. During training,
    adds KL divergence loss between LSS depth_net output and FD-derived bins.

    This module does NOT modify the LSS architecture or inference path.
    It only provides the auxiliary loss signal during training.
    """

    def __init__(self, d_conf=None, depth_model_type="midas_small"):
        super().__init__()
        if d_conf is None:
            from bev_settings import d_conf as default_d_conf
            d_conf = default_d_conf

        self.depth_provider = FoundationDepthProviderV2(model_type=depth_model_type)
        self.depth_aligner = SpatialDepthAligner()
        self.depth_to_bins = DepthBinConverterV2(d_conf[0], d_conf[1], d_conf[2])
        print(f"[DepthSupervision] model={depth_model_type}, D={self.depth_to_bins.D}")

    def get_target_depth_dist(self, raw_imgs, target_h, target_w):
        """Generate pseudo-GT depth distribution from FD model.

        Args:
            raw_imgs: (B, N, 3, H, W) ImageNet-normalized images
            target_h, target_w: feature map size
        Returns:
            target_dist: (B*N, D, target_h, target_w) pseudo-GT depth distribution (detached)
        """
        B, N = raw_imgs.shape[:2]
        raw_flat = raw_imgs.view(B * N, 3, raw_imgs.shape[3], raw_imgs.shape[4])

        with torch.no_grad():
            rel_depth = self.depth_provider(raw_flat, target_h, target_w)

        metric_depth = self.depth_aligner(rel_depth)
        target_dist = self.depth_to_bins(metric_depth)
        return target_dist.detach()

    def compute_loss(self, pred_depth_logits, raw_imgs, target_h, target_w, alpha=0.5):
        """Compute KL divergence between predicted and FD-derived depth distributions.

        Args:
            pred_depth_logits: (B*N, D, H, W) raw logits from depth_net (before softmax)
            raw_imgs: (B, N, 3, H, W)
            target_h, target_w: feature map size
            alpha: loss weight
        Returns:
            loss: scalar
        """
        target = self.get_target_depth_dist(raw_imgs, target_h, target_w)
        pred = F.log_softmax(pred_depth_logits, dim=1)
        target_clamped = target.clamp(min=1e-8)
        kl = F.kl_div(pred, target_clamped, reduction='batchmean', log_target=False)
        return alpha * kl
