"""
Foundation depth model integration for LSS-style BEV projection.

Replaces LSS's learned depth distribution with a frozen pre-trained depth model,
converting metric (or scaled-relative) depth into discrete bin distributions.

The key insight: LSS's depth_net learns depth statistics specific to the training
camera configuration. A foundation depth model (trained on millions of diverse images)
provides depth priors that generalize across camera mountings.

Supported models (via torch.hub):
  - "midas_small"  : MiDaS v2.1 small (fast, relative depth)
  - "dpt_swin2_t"  : DPT with Swin2-Tiny (better quality, relative depth)
  - "dpt_beit_l"   : DPT with BEiT-Large (best quality, relative depth)

For all models, depth is relative (not metric). We learn a per-channel scale/shift
to align with the frustum depth bins, allowing end-to-end fine-tuning of alignment
while keeping the depth backbone frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


class DepthBinConverter(nn.Module):
    """Convert continuous depth values to discrete bin distributions.
    
    Given a depth map D(h,w) and frustum depth bins [d_0, d_1, ..., d_{K-1}],
    produce a soft distribution over bins using a Laplace kernel:
        p(bin_k | pixel) = exp(-|D(pixel) - d_k| / sigma) / Z
    
    sigma is learnable, starting from a reasonable default.
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
        """
        Args:
            depth_map: (B, 1, H, W) continuous depth in meters
        Returns:
            depth_dist: (B, D, H, W) soft distribution over depth bins
        """
        sigma = self.sigma
        bins = self.depth_bins.view(1, -1, 1, 1)
        dist = -(depth_map - bins).abs() / sigma
        return F.softmax(dist, dim=1)


class RelativeDepthAligner(nn.Module):
    """Learn scale and shift to convert relative depth to metric depth.
    
    Foundation models often output relative (inverse) depth. This module learns
    a per-image affine transform: metric_depth = scale * relative_depth + shift.
    
    Initialized with reasonable defaults for outdoor driving scenes.
    """
    
    def __init__(self, init_scale=50.0, init_shift=5.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(init_scale).log())
        self.shift = nn.Parameter(torch.tensor(init_shift))
    
    def forward(self, relative_depth):
        """
        Args:
            relative_depth: (B, 1, H, W) relative/inverse depth from foundation model
        Returns:
            metric_depth: (B, 1, H, W) estimated metric depth in meters
        """
        scale = self.log_scale.exp()
        metric = scale * relative_depth + self.shift
        return metric.clamp(min=0.5, max=150.0)


MIDAS_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ckpt', 'MiDaS-master')
MIDAS_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ckpt')

MIDAS_WEIGHT_FILES = {
    'midas_small': 'midas_v21_small_256.pt',
    'dpt_swin2_t': 'dpt_swin2_tiny_256.pt',
    'dpt_beit_l': 'dpt_beit_large_512.pt',
}


class FoundationDepthProvider(nn.Module):
    """Wraps a frozen foundation depth model.
    
    Handles: model loading, input preprocessing, output postprocessing,
    and resolution adaptation to match LSS feature map size.
    """
    
    def __init__(self, model_type="midas_small", device=None):
        super().__init__()
        self.model_type = model_type
        self._model = None
        self._transform = None
        self._loaded = False
        self._target_device = device


    def _load_model(self, device):
        """Lazy-load the depth model on first forward pass.
        
        Fully offline loading strategy (no network access required):
          1. Import MidasNet_small directly from local MiDaS repo
          2. Build model with path=<weights> so backbone skips download
        """
        if self._loaded:
            return
        
        midas_repo = os.path.realpath(MIDAS_REPO)
        weight_name = MIDAS_WEIGHT_FILES.get(self.model_type)
        weight_path = os.path.join(os.path.realpath(MIDAS_WEIGHTS_DIR), weight_name) if weight_name else None

        if not weight_path or not os.path.isfile(weight_path):
            print(
                f"[FoundationDepth] ERROR: weights not found at {weight_path}, "
                f"falling back to vertical depth ramp",
                file=sys.stderr, flush=True,
            )
            self._model = None
            self._loaded = True
            return

        print(f"[FoundationDepth] Loading {self.model_type} (local: {weight_path})")

        midas_pkg = os.path.join(midas_repo)
        if midas_pkg not in sys.path:
            sys.path.insert(0, midas_pkg)

        if self.model_type == 'midas_small':
            from midas.midas_net_custom import MidasNet_small as _Net
            model = _Net(
                path=weight_path, features=64, backbone="efficientnet_lite3",
                exportable=True, non_negative=True, blocks={'expand': True},
            )
        else:
            from midas.dpt_depth import DPTDepthModel as _DPT
            model_configs = {
                'dpt_swin2_t': dict(
                    path=weight_path, backbone="swin2t16_256",
                    non_negative=True,
                ),
                'dpt_beit_l': dict(
                    path=weight_path, backbone="beitl16_512",
                    non_negative=True,
                ),
            }
            cfg = model_configs.get(self.model_type)
            if cfg is None:
                print(f"[FoundationDepth] ERROR: unsupported model_type={self.model_type}",
                      file=sys.stderr, flush=True)
                self._model = None
                self._loaded = True
                return
            model = _DPT(**cfg)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model.to(device)
        self._loaded = True
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[FoundationDepth] {self.model_type} loaded ({n_params:.1f}M params, frozen)")
    
    @torch.no_grad()
    def forward(self, imgs, target_h, target_w):
        """
        Args:
            imgs: (B*N, 3, H, W) normalized images (ImageNet mean/std already applied)
            target_h, target_w: output resolution to match feature maps
        Returns:
            depth: (B*N, 1, target_h, target_w) normalized depth in [0, 1]
                   where 0 = near, 1 = far (consistent with LSS depth bins)
        """
        device = imgs.device
        self._load_model(device)
        
        if self._model is None:
            return self._fallback_depth(imgs, target_h, target_w)
        
        BN, C, H, W = imgs.shape
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        imgs_denorm = imgs * std + mean
        
        if "small" in self.model_type:
            inp = F.interpolate(imgs_denorm, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            inp = F.interpolate(imgs_denorm, size=(384, 384), mode='bilinear', align_corners=False)
        
        disp = self._model(inp)
        
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        
        disp = F.interpolate(disp, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # MiDaS outputs inverse depth (disparity): higher = closer.
        # Convert to depth: lower disparity = further away = higher depth value.
        # This ensures depth is monotonically aligned with LSS depth bins (1m, 2m, ...).
        depth = 1.0 / (disp + 1e-6)
        
        d_min = depth.amin(dim=(2, 3), keepdim=True)
        d_max = depth.amax(dim=(2, 3), keepdim=True)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        
        return depth
    
    def _fallback_depth(self, imgs, target_h, target_w):
        """Vertical position prior when no depth model is available.
        In driving scenes, lower pixels are typically closer (road surface).
        Produces a linear ramp from 0 (bottom=near) to 1 (top=far)."""
        BN = imgs.shape[0]
        device = imgs.device
        ramp = torch.linspace(1.0, 0.0, target_h, device=device).view(1, 1, target_h, 1)
        depth = ramp.expand(BN, 1, target_h, target_w)
        return depth


class FoundationDepthLSS(nn.Module):
    """LSS variant using foundation depth model instead of learned depth distribution.
    
    Architecture:
        1. Frozen depth model → relative depth map (B*N, 1, fH, fW)
        2. Learnable scale/shift → metric depth (B*N, 1, fH, fW)  
        3. Laplace bin conversion → depth distribution (B*N, D, fH, fW)
        4. Learnable feature head → BEV features (B*N, out_channels, fH, fW)
        5. depth_dist * features → (B*N, out_channels, D, fH, fW)
    
    The depth backbone is frozen; only the aligner, bin converter, and feature
    head are trained. This preserves the foundation model's generalization while
    learning task-specific alignment and features.
    """
    
    def __init__(self,
                 transformedImgShape=None,
                 featureShape=None,
                 d_conf=None,
                 out_channels=128,
                 depth_model_type="midas_small"):
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
        self.D = torch.arange(self.d_st, self.d_end, self.d_step, dtype=torch.float).shape[0]
        self.out_channels = out_channels
        
        self.frustum = self._create_frustum()
        
        self.depth_provider = FoundationDepthProvider(model_type=depth_model_type)
        self.depth_aligner = RelativeDepthAligner()
        self.depth_to_bins = DepthBinConverter(self.d_st, self.d_end, self.d_step)
        
        self.feature_net = nn.Sequential(
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, out_channels, 1),
        )
        
        print(f"[FoundationDepthLSS] depth_model={depth_model_type}, D={self.D}, "
              f"out_channels={out_channels}, feature_shape={featureShape}")
    
    def _create_frustum(self):
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype=torch.float) \
            .view(-1, 1, 1).expand(-1, self.fH, self.fW)
        xs = torch.linspace(0, self.orfW - 1, self.fW) \
            .view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        ys = torch.linspace(0, self.orfH - 1, self.fH) \
            .view(1, self.fH, 1).expand(self.D, self.fH, self.fW)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)
    
    def get_geometry(self, cam2ego_rot, cam2ego_trans, cam_intrins,
                     post_cam2ego_rot, post_cam2ego_trans):
        """Same geometry computation as standard LSS — depends on extrinsics only."""
        img_rots, img_trans = cam2ego_rot, cam2ego_trans
        img_post_rots, img_post_trans = post_cam2ego_rot, post_cam2ego_trans
        B, N, _ = img_trans.shape
        
        points = self.frustum - img_post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.linalg.inv(img_post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]),
            5,
        )
        combine = img_rots.matmul(torch.linalg.inv(cam_intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += img_trans.view(B, N, 1, 1, 1, 3)
        return points
    
    def get_cam_feature(self, img_feats, raw_imgs=None):
        """
        Args:
            img_feats: (B, N, C, fH, fW) encoded image features from SwinT
            raw_imgs: (B, N, 3, H, W) raw normalized images for depth model
        Returns:
            (B, N, D, fH, fW, out_channels)
        """
        B, N, C, H, W = img_feats.shape
        img_feats_flat = img_feats.view(B * N, C, H, W)
        
        if raw_imgs is not None:
            raw_flat = raw_imgs.view(B * N, 3, raw_imgs.shape[3], raw_imgs.shape[4])
            rel_depth = self.depth_provider(raw_flat, H, W)
        else:
            rel_depth = self.depth_provider._fallback_depth(img_feats_flat, H, W)
        
        metric_depth = self.depth_aligner(rel_depth)
        depth_dist = self.depth_to_bins(metric_depth)
        
        features = self.feature_net(img_feats_flat)
        
        out = depth_dist.unsqueeze(1) * features.unsqueeze(2)
        out = out.view(B, N, self.out_channels, self.D, H, W)
        out = out.permute(0, 1, 3, 4, 5, 2)
        return out
    
    def forward(self, cam2ego_rot, cam2ego_trans, cam_intrins,
                post_cam2ego_rot, post_cam2ego_trans, img_feats, raw_imgs=None):
        geometry = self.get_geometry(
            cam2ego_rot=cam2ego_rot, cam2ego_trans=cam2ego_trans,
            cam_intrins=cam_intrins, post_cam2ego_rot=post_cam2ego_rot,
            post_cam2ego_trans=post_cam2ego_trans
        )
        img_depth_feature = self.get_cam_feature(img_feats, raw_imgs=raw_imgs)
        return geometry, img_depth_feature
