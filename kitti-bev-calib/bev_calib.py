import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from img_branch.img_branch import Cam2BEV
from pc_branch.pc_branch import Lidar2BEV
from losses.losses import realworld_loss
from losses.quat_tools import quaternion_from_matrix
from deformable_attention import DeformableAttention
from BEVEncoder.BEVEncoder import BEVEncoder
import bev_settings


class _GradientReversal(Function):
    """Gradient Reversal Layer for Domain Adversarial Training."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainClassifier(nn.Module):
    """Classifies domain (sequence ID) from pooled BEV features.
    Used with gradient reversal to encourage domain-invariant representations."""

    def __init__(self, in_dim, num_domains, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, x, alpha=1.0):
        x_rev = _GradientReversal.apply(x, alpha)
        return self.classifier(x_rev)


class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in residual blocks)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

class ConvFuser(nn.Sequential):
    def __init__(self, img_in_channel, pc_in_channel, out_channel):
        self.img_in_channel = img_in_channel
        self.pc_in_channel = pc_in_channel
        self.out_channel = out_channel
        super(ConvFuser, self).__init__(
            nn.Conv2d(self.img_in_channel + self.pc_in_channel, self.out_channel, 1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True)
        )

    def forward(self, img_bev_feat, pc_bev_feat, cam_dropped=False):
        return super().forward(torch.cat([img_bev_feat, pc_bev_feat], dim=1))


class BEVDiffFuser(nn.Module):
    """BEV-space Difference Map Fuser inspired by DST-Calib.

    Instead of simple concat, explicitly encodes cross-modal differences:
      - pc_bev_feat:           LiDAR geometric features (anchor)
      - |cam - pc|:            absolute difference (calibration error signal)
      - cam * pc:              element-wise interaction (alignment reinforcement)

    When cam_drop_aware=True (fuser_type="diff_v2"), provides a dedicated
    pc_only_conv path for when camera features are completely absent (cam_dropped=True).
    This avoids the degenerate [pc, |pc|, 0] input that occurs with v1 + zero dropout.
    """
    def __init__(self, img_in_channel, pc_in_channel, out_channel,
                 cam_drop_aware=False):
        super().__init__()
        assert img_in_channel == pc_in_channel, (
            f"BEVDiffFuser requires equal channel dims for diff/interact ops, "
            f"got img={img_in_channel} vs pc={pc_in_channel}")
        in_ch = pc_in_channel * 3
        self.cam_drop_aware = cam_drop_aware
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        if cam_drop_aware:
            self.pc_only_conv = nn.Sequential(
                nn.Conv2d(pc_in_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
            )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, img_bev_feat, pc_bev_feat, cam_dropped=False):
        if self.cam_drop_aware and cam_dropped:
            return self.pc_only_conv(pc_bev_feat)
        diff = (img_bev_feat - pc_bev_feat).abs()
        interact = img_bev_feat * pc_bev_feat
        return self.conv(torch.cat([pc_bev_feat, diff, interact], dim=1))
    
class FrontViewPitchBranch(nn.Module):
    """Dual-flow Pitch prediction branch combining Z-aware BEV and 2D image features.

    Flow 1 (Z-aware): Globally-pooled pre-projection BEV features (B, C*nZ) that
    retain vertical distribution information compressed away by ProjectionHead.

    Flow 2 (2D front-view): Aggregated 2D image features from FPN backbone,
    capturing vertical edge cues and horizon-line signals that directly correlate
    with Pitch rotation in the camera frame.

    Both flows are fused via gated attention before final Pitch regression.
    """

    def __init__(self, z_feat_dim, img_feat_dim=128, hidden_dim=128):
        super().__init__()
        self.z_encoder = nn.Sequential(
            nn.Linear(z_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_summary, img_summary=None):
        z_feat = self.z_encoder(z_summary)
        if img_summary is not None:
            img_feat = self.img_encoder(img_summary)
            gate_input = torch.cat([z_feat, img_feat], dim=-1)
            g = self.gate(gate_input)
            fused = g * z_feat + (1 - g) * img_feat
        else:
            fused = z_feat
        return self.head(fused)

    @staticmethod
    def compute_loss(pitch_pred, gt_T_to_camera):
        R_gt = gt_T_to_camera[:, :3, :3].float()
        sy = torch.sqrt(R_gt[:, 0, 0] ** 2 + R_gt[:, 1, 0] ** 2)
        pitch_gt = torch.atan2(-R_gt[:, 2, 0], sy)
        return F.smooth_l1_loss(pitch_pred.squeeze(-1), pitch_gt)


class ContrastiveExtrinsicHead(nn.Module):
    """Auxiliary head that forces the BEV difference map to encode geometric offset.

    Takes the absolute difference between camera and LiDAR BEV features, encodes
    it into a compact extrinsic embedding, then applies two auxiliary losses:

    1. Perturbation regression: predict the actual RPY offset from the embedding.
       This forces the diff map to encode the geometric residual, not scene content.

    2. In-batch contrastive: embeddings with similar perturbation directions should
       be closer than embeddings with dissimilar perturbations (InfoNCE-style).
       This builds a smooth, transferable embedding space.
    """

    def __init__(self, in_channels, embed_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(in_channels // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.rpy_regressor = nn.Linear(embed_dim, 3)

    def forward(self, cam_bev, pc_bev):
        diff = (cam_bev - pc_bev).abs()
        feat = self.encoder(diff).flatten(1)       # (B, C//2)
        embedding = self.projector(feat)            # (B, embed_dim)
        rpy_pred = self.rpy_regressor(embedding)    # (B, 3) — predicted RPY offset
        return embedding, rpy_pred

    @staticmethod
    def compute_loss(embedding, rpy_pred, gt_T, init_T, temperature=0.1,
                     regression_weight=1.0, contrastive_weight=0.5):
        R_gt = gt_T[:, :3, :3].float()
        R_init = init_T[:, :3, :3].float()
        R_delta = R_init @ R_gt.transpose(-2, -1)  # perturbation rotation

        sy = torch.sqrt(R_delta[:, 0, 0]**2 + R_delta[:, 1, 0]**2)
        roll = torch.atan2(R_delta[:, 2, 1], R_delta[:, 2, 2])
        pitch = torch.atan2(-R_delta[:, 2, 0], sy)
        yaw = torch.atan2(R_delta[:, 1, 0], R_delta[:, 0, 0])
        rpy_gt = torch.stack([roll, pitch, yaw], dim=-1)  # (B, 3) radians

        reg_loss = F.smooth_l1_loss(rpy_pred, rpy_gt)

        B = embedding.shape[0]
        if B < 4:
            return regression_weight * reg_loss

        emb_norm = F.normalize(embedding, dim=-1)
        sim_matrix = emb_norm @ emb_norm.t() / temperature  # (B, B)

        rpy_dist = torch.cdist(rpy_gt, rpy_gt, p=2)  # (B, B)
        median_dist = rpy_dist.median()
        labels = (rpy_dist < median_dist).float()
        labels.fill_diagonal_(0)

        pos_count = labels.sum(dim=1).clamp(min=1)
        log_sum_exp = torch.logsumexp(sim_matrix - 1e9 * torch.eye(B, device=sim_matrix.device), dim=1)
        pos_sim = (sim_matrix * labels).sum(dim=1) / pos_count
        contrastive_loss = (log_sum_exp - pos_sim).mean()

        return regression_weight * reg_loss + contrastive_weight * contrastive_loss


class deformable_transformer_layer(nn.Module):
    def __init__(self, 
                 dim = 512, 
                 dim_head = 64, 
                 heads = 8, 
                 dropout = 0., 
                 downsample_factor = 4, 
                 offset_scale = 4, 
                 offset_groups = None,
                 offset_kernel_size = 6,
                 drop_path_rate = 0.,
                 ):
        super(deformable_transformer_layer, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.deformable_attention = DeformableAttention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            dropout = dropout,
            downsample_factor = downsample_factor,
            offset_scale = offset_scale,
            offset_groups = offset_groups,
            offset_kernel_size = offset_kernel_size,
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 1),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, 1),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        """
        Args: 
            x: (B, C, H, W)
        Returns:
            x: (B, C, H, W)
        """
        x = x + self.drop_path(self.deformable_attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BEVCalib(nn.Module):
    def __init__(self, 
                 num_heads = 8,
                 num_layers = 2,
                 deformable = True,
                 bev_encoder = False,
                 img_shape = None,
                 rotation_only = False,
                 enable_axis_loss = False,
                 weight_axis_rotation = 0.3,
                 axis_weights = (3.0, 1.5, 1.0),
                 drop_path_rate = 0.1,
                 head_dropout = 0.1,
                 use_geodesic_loss = False,
                 use_mlp_head = True,
                 bev_pool_factor = 0,
                 use_foundation_depth = False,
                 depth_model_type = "midas_small",
                 fd_mode = "replace",
                 voxel_mode = "hard",
                 to_bev_mode = "concat",
                 scatter_reduce = "sum",
                 fuser_type = "concat",
                 cam_drop_prob = 0.0,
                 cam_drop_mode = "zero",
                 intrinsic_input = False,
                 use_pitch_branch = False,
                 pitch_aux_weight = 0.3,
                 bev_instance_norm = False,
                 use_contrastive_extrinsic = False,
                 contrastive_weight = 0.1,
                 use_balanced_axis_loss = False,
                 domain_adversarial = False,
                 domain_adversarial_weight = 0.1,
                 num_domains = 21,
                 cam2bev_mode = "lss",
                 backbone_type = "swin",
                 backbone_variant = "dinov2-small",
                 freeze_backbone = False,
                 freeze_layers = None,
                 backbone_weights = None,
                ):
        super(BEVCalib, self).__init__()
        self.use_mlp_head = use_mlp_head
        self.use_pitch_branch = use_pitch_branch
        self.pitch_aux_weight = pitch_aux_weight
        self.bev_instance_norm = bev_instance_norm
        self.use_contrastive_extrinsic = use_contrastive_extrinsic
        self.contrastive_weight = contrastive_weight
        self.rotation_only = rotation_only
        self.domain_adversarial = domain_adversarial
        self.domain_adversarial_weight = domain_adversarial_weight
        if bev_pool_factor == 0 and cam2bev_mode == "query":
            bev_pool_factor = 4
        self.bev_pool_factor = bev_pool_factor
        self.intrinsic_input = intrinsic_input
        self.cam_drop_mode = cam_drop_mode
        self._profile_modules = False
        self._profile_events = []
        self._profile_accum = {}
        self._profile_count = 0
        self.cam2bev_mode = cam2bev_mode

        if cam2bev_mode == "query":
            from img_branch.cam2bev_query import Cam2BEVQuery
            self.img_branch = Cam2BEVQuery(
                img_shape=img_shape,
                backbone_type=backbone_type,
                backbone_variant=backbone_variant,
                freeze_backbone=freeze_backbone,
                freeze_layers=freeze_layers,
                backbone_weights=backbone_weights,
            )
        else:
            self.img_branch = Cam2BEV(
                img_shape=img_shape,
                use_foundation_depth=use_foundation_depth,
                depth_model_type=depth_model_type,
                fd_mode=fd_mode,
                backbone_type=backbone_type,
                backbone_variant=backbone_variant,
                freeze_backbone=freeze_backbone,
                freeze_layers=freeze_layers,
                backbone_weights=backbone_weights,
            )
        self.pc_branch = Lidar2BEV(
            to_bev_mode=to_bev_mode,
            voxel_mode=voxel_mode,
            scatter_reduce=scatter_reduce,
        )
        self.bev_encoder_use = bev_encoder
        if self.bev_encoder_use:
            self.bev_encoder = BEVEncoder()
        if hasattr(self.img_branch, 'nx'):
            self.bev_shape = (self.img_branch.nx[0].item(), self.img_branch.nx[1].item())
        else:
            self.bev_shape = (self.img_branch.nx_x, self.img_branch.nx_y)
        self.embed_dim = self.img_branch.out_channels + self.pc_branch.out_channels
        self.num_heads = num_heads
        self.cam_drop_prob = cam_drop_prob
        if fuser_type == "diff_v2":
            self.conv_fuser = BEVDiffFuser(
                self.img_branch.out_channels,
                self.pc_branch.out_channels,
                self.embed_dim,
                cam_drop_aware=True,
            )
        elif fuser_type == "diff":
            self.conv_fuser = BEVDiffFuser(
                self.img_branch.out_channels,
                self.pc_branch.out_channels,
                self.embed_dim,
                cam_drop_aware=False,
            )
        else:
            self.conv_fuser = ConvFuser(
                self.img_branch.out_channels,
                self.pc_branch.out_channels,
                self.embed_dim
            )
        self.pose_embed = nn.Parameter(
                        torch.zeros(1,
                                    self.embed_dim,
                                    self.bev_shape[0],
                                    self.bev_shape[1]
                                    )
        )
        self.deformable = deformable
        if self.deformable:
            self.deformable_transformer = self.make_deformable_transformer(num_layers, drop_path_rate)
        else:
            self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = self.embed_dim, 
                                       nhead = num_heads,
                                       dim_feedforward = 4 * self.embed_dim,
                                       activation= "gelu",
                                       batch_first = True,
                                       norm_first = True,
                                       ),
            num_layers = num_layers * 4
        )
        self.head_drop = nn.Dropout(head_dropout)
        head_in_dim = self.embed_dim
        if self.intrinsic_input:
            self.intrinsic_proj = nn.Sequential(
                nn.Linear(4, 32),
                nn.LayerNorm(32),
                nn.GELU(),
            )
            self.register_buffer('intr_mean',
                torch.tensor(bev_settings.intrinsic_stats['mean']))
            self.register_buffer('intr_std',
                torch.tensor(bev_settings.intrinsic_stats['std']))
            head_in_dim = self.embed_dim + 32
        if self.use_mlp_head:
            if not self.rotation_only:
                self.translation_pred = self._build_regression_head(
                    head_in_dim, 3, head_dropout)
            self.rotation_pred = self._build_regression_head(
                head_in_dim, 4, head_dropout)
        else:
            if not self.rotation_only:
                self.translation_pred = nn.Linear(head_in_dim, 3)
            self.rotation_pred = nn.Linear(head_in_dim, 4)
        self.loss_fn = realworld_loss(
            rotation_only=rotation_only,
            enable_axis_loss=enable_axis_loss,
            weight_axis_rotation=weight_axis_rotation,
            axis_weights=axis_weights,
            use_geodesic_loss=use_geodesic_loss,
            use_balanced_axis_loss=use_balanced_axis_loss,
        )
        if self.bev_instance_norm:
            self.cam_bev_norm = nn.InstanceNorm2d(self.img_branch.out_channels, affine=True)
            print(f"[BEVCalib] BEV Instance Normalization enabled (C={self.img_branch.out_channels})")
        if self.use_pitch_branch:
            nz = self.img_branch.nx[2].item()
            z_feat_dim = self.img_branch.lss.out_channels * nz
            img_feat_dim = self.img_branch.CamEncode.out_channels
            self.pitch_branch = FrontViewPitchBranch(
                z_feat_dim=z_feat_dim, img_feat_dim=img_feat_dim)
            print(f"[BEVCalib] Dual-flow Pitch branch enabled: "
                  f"z_feat_dim={z_feat_dim}, img_feat_dim={img_feat_dim}, "
                  f"aux_weight={pitch_aux_weight}")
        if self.use_contrastive_extrinsic:
            self.contrastive_head = ContrastiveExtrinsicHead(
                in_channels=self.img_branch.out_channels,
                embed_dim=64,
            )
            print(f"[BEVCalib] Contrastive Extrinsic Head enabled: "
                  f"in_channels={self.img_branch.out_channels}, weight={contrastive_weight}")
        if self.domain_adversarial:
            self.domain_classifier = DomainClassifier(
                in_dim=self.embed_dim, num_domains=num_domains)
            self._dann_epoch_ratio = 0.0
            print(f"[BEVCalib] Domain Adversarial Training enabled: "
                  f"num_domains={num_domains}, weight={domain_adversarial_weight}")

    @staticmethod
    def _build_regression_head(in_dim, out_dim, dropout=0.1):
        """Multi-layer MLP regression head with residual-style bottleneck."""
        mid_dim = in_dim // 2
        return nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
        )

    def get_module_profile(self, reset=True):
        """Return per-module average forward time (ms). Enable with model._profile_modules = True.
        Deferred sync: events are recorded without synchronize during forward;
        synchronize happens here at epoch end (zero overhead during training)."""
        if not self._profile_events:
            return {}
        torch.cuda.synchronize()
        names = ["img_branch", "pc_branch", "fuser", "transformer", "head", "loss"]
        accum = {}
        for ev_list in self._profile_events:
            for i, name in enumerate(names):
                ms = ev_list[i].elapsed_time(ev_list[i + 1])
                accum[name] = accum.get(name, 0.0) + ms
            accum["total"] = accum.get("total", 0.0) + ev_list[0].elapsed_time(ev_list[6])
        count = len(self._profile_events)
        result = {k: v / count for k, v in accum.items()}
        if reset:
            self._profile_events = []
        return result

    def make_deformable_transformer(self, num_layers, drop_path_rate=0.1):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        layers = []
        for i in range(num_layers):
            layers.append(deformable_transformer_layer(
                dim=self.embed_dim,
                dim_head=self.embed_dim // self.num_heads,
                heads=self.num_heads,
                downsample_factor=15,
                offset_kernel_size=15,
                offset_scale=10,
                drop_path_rate=dpr[i],
            ))
        return nn.Sequential(*layers)


    def quaternion_to_rotation_matrix(self, q):
        """
        Args:
            q: (B, 4)
        Returns:
            R: (B, 3, 3)
        """
        q = q.float()
        q_norm = q.norm(dim=1, keepdim=True).clamp(min=1e-6)
        q = q / q_norm
        B = q.shape[0]
        R = torch.zeros(B, 3, 3, dtype=torch.float32, device=q.device)
        R[:, 0, 0] = 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
        R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        R[:, 1, 1] = 1 - 2 * (q[:, 1] ** 2 + q[:, 3] ** 2)
        R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
        R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        R[:, 2, 2] = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return R

    def get_T_matrix(self, translation, rotation):
        """
        Args:
            translation: (B, 3)
            rotation: (B, 4)
        Returns:
            T: (B, 4, 4)
        """
        B = translation.shape[0]
        T = torch.zeros(B, 4, 4).to(translation.device)
        T[:, :3, :3] = self.quaternion_to_rotation_matrix(rotation)
        T[:, :3, 3] = translation
        T[:, 3, 3] = 1
        return T
    
    def forward(self, img, pc, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, cam_intrinsic, masks = None, out_init_loss = False, domain_ids = None):
        """
        We use Lidar as ego here.
        Args:
            Input:
            img: (B, 3, H, W), original image (with data aug).
            pc: (B, N, 3), point cloud.
            gt_T_to_camera: (B, 4, 4), ground truth transformation matrix from Lidar to camera.
            init_T_to_camera: (B, 4, 4), initial transformation matrix from Lidar to camera.
            post_cam2ego_T: (B, 4, 4), after data aug.
            cam_intrinsic: (B, 3, 3), camera intrinsic matrix.
            out_init_loss: bool, whether to output the loss between init_T and gt_T.
        """
        profiling = self._profile_modules and torch.cuda.is_available()
        if profiling:
            _ev = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
            _ev[0].record()

        img = img.unsqueeze(1)
        gt_T_to_camera = gt_T_to_camera.unsqueeze(1)
        init_T_to_camera = init_T_to_camera.unsqueeze(1)
        post_cam2ego_T = post_cam2ego_T.unsqueeze(1)
        cam_intrinsic = cam_intrinsic.unsqueeze(1)
        cam2ego_T = torch.linalg.inv(init_T_to_camera.float())

        z_summary = None
        img_feat_2d = None
        if self.use_pitch_branch:
            cam_bev_feats, cam_bev_mask, z_summary, img_feat_2d = self.img_branch(
                cam2ego_T=cam2ego_T, cam_intrins=cam_intrinsic,
                post_cam2ego_T=post_cam2ego_T, imgs=img,
                return_z_features=True)
        else:
            cam_bev_feats, cam_bev_mask = self.img_branch(
                cam2ego_T=cam2ego_T, cam_intrins=cam_intrinsic,
                post_cam2ego_T=post_cam2ego_T, imgs=img)

        if profiling:
            _ev[1].record()

        pc = pc.permute(0, 2, 1).contiguous() # (B, 3, N)
        pc_bev_feats = self.pc_branch(pc) # B, C, H, W

        if profiling:
            _ev[2].record()

        cam_dropped = False
        if self.training and self.cam_drop_prob > 0:
            drop_flag = torch.rand(1, device=cam_bev_feats.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(drop_flag, src=0)
            if drop_flag.item() < self.cam_drop_prob:
                if self.cam_drop_mode == "noise":
                    noise_scale = cam_bev_feats.std().detach().clamp(min=1e-6) * 0.1
                    cam_bev_feats = cam_bev_feats * 0 + torch.randn_like(cam_bev_feats) * noise_scale
                else:
                    cam_bev_feats = cam_bev_feats * 0
                    cam_dropped = True

        if self.bev_instance_norm and not cam_dropped:
            cam_bev_feats = self.cam_bev_norm(cam_bev_feats)
        x = self.conv_fuser(cam_bev_feats, pc_bev_feats, cam_dropped=cam_dropped)
        if self.bev_encoder_use:
            x = self.bev_encoder(x) # B, C, H, W
        x = x + self.pose_embed

        if profiling:
            _ev[3].record()

        if self.deformable:
            x = self.deformable_transformer(x) # B, C, H, W
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H*W, C) # B, H * W, C
            bev_mask = cam_bev_mask.reshape(B, H * W).unsqueeze(-1) # B, H * W, 1
            x = x * bev_mask # B, H * W, C
            valid_count = bev_mask.sum(dim=1).clamp(min=1) # B, 1
            x = x.sum(dim=1) / valid_count # B, C
        else:
            B, C, H, W = x.shape
            if self.bev_pool_factor > 1:
                pf = self.bev_pool_factor
                x = nn.functional.avg_pool2d(x, pf)
                cam_bev_mask = nn.functional.max_pool2d(
                    cam_bev_mask.reshape(B, 1, H, W).float(), pf).squeeze(1)
                _, _, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C) # B, H * W, C
            bev_mask = cam_bev_mask.reshape(B, H * W).bool()
            max_valid_cnt = int(bev_mask.sum(dim=1).max().item()) # int, max number of valid points in a batch
            valid_counts = torch.zeros(B, dtype=torch.long, device=x.device)
            masked_x = torch.zeros(B, max_valid_cnt, C, device=x.device)
            padding_mask = torch.zeros(B, max_valid_cnt, dtype=torch.bool, device=x.device)
            for i in range(B):
                cnt = int(bev_mask[i].sum().item())
                valid_counts[i] = cnt
                masked_x[i, :cnt, :] = x[i, bev_mask[i]]
                padding_mask[i, cnt:] = True
            x = self.transformer(masked_x, src_key_padding_mask=padding_mask) # B, max_valid_cnt, C
            x_pooled = torch.zeros(B, C, device=x.device)
            for i in range(B):
                cnt = valid_counts[i].item()
                if cnt > 0:
                    x_pooled[i] = x[i, :cnt, :].mean(dim=0)
            x = x_pooled

        if profiling:
            _ev[4].record()

        x_bev_pooled = x
        x = self.head_drop(x)
        if self.intrinsic_input:
            K = cam_intrinsic.squeeze(1)  # (B, 3, 3)
            intr_vec = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], dim=-1)  # (B, 4)
            intr_normed = (intr_vec - self.intr_mean) / self.intr_std
            intr_feat = self.intrinsic_proj(intr_normed)  # (B, 32)
            x = torch.cat([x, intr_feat], dim=-1)  # (B, C+32)
        if not self.rotation_only:
            translation = self.translation_pred(x)
        else:
            translation = torch.zeros(B, 3, device=x.device)
        rotation = self.rotation_pred(x)

        pred_T = self.get_T_matrix(translation=translation, rotation=rotation)

        if profiling:
            _ev[5].record()

        gt_T_to_camera = gt_T_to_camera.squeeze(1)
        init_T_to_camera = init_T_to_camera.squeeze(1)
        pc = pc.permute(0, 2, 1).contiguous() # (B, N, 3)
    
        loss, T_gt_expected = self.loss_fn(pred_translation = translation, pred_rotation = rotation,
                            pcs = pc, gt_T_to_camera = gt_T_to_camera, init_T_to_camera = init_T_to_camera, mask = masks)

        if self.use_pitch_branch and z_summary is not None:
            img_summary = None
            if img_feat_2d is not None:
                img_summary = img_feat_2d.mean(dim=1)  # avg over cameras: (B, C, fH, fW)
                img_summary = F.adaptive_avg_pool2d(img_summary, 1).flatten(1)  # (B, C)
            pitch_pred = self.pitch_branch(z_summary, img_summary)
            pitch_aux_loss = FrontViewPitchBranch.compute_loss(
                pitch_pred, gt_T_to_camera)
            loss["total_loss"] = loss["total_loss"] + self.pitch_aux_weight * pitch_aux_loss
            loss["pitch_aux_loss"] = pitch_aux_loss / torch.pi * 180.0

        if self.training and self.use_contrastive_extrinsic and not cam_dropped:
            ctr_emb, rpy_pred = self.contrastive_head(cam_bev_feats, pc_bev_feats)
            contrast_loss = ContrastiveExtrinsicHead.compute_loss(
                ctr_emb, rpy_pred, gt_T_to_camera, init_T_to_camera,
                temperature=self.contrastive_head.temperature,
            )
            loss["total_loss"] = loss["total_loss"] + self.contrastive_weight * contrast_loss
            loss["contrastive_loss"] = contrast_loss.detach()

        if self.training and self.domain_adversarial and domain_ids is not None:
            alpha = 2.0 / (1.0 + math.exp(-10.0 * self._dann_epoch_ratio)) - 1.0
            domain_logits = self.domain_classifier(x_bev_pooled, alpha=alpha)
            domain_labels = domain_ids.to(domain_logits.device)
            dann_loss = F.cross_entropy(domain_logits, domain_labels)
            loss["total_loss"] = loss["total_loss"] + self.domain_adversarial_weight * dann_loss
            loss["dann_loss"] = dann_loss.detach()
            loss["dann_alpha"] = torch.tensor(alpha)

        if profiling:
            _ev[6].record()
            self._profile_events.append(_ev)

        if out_init_loss:
            with torch.no_grad():
                B, _, _ = pc.shape
                translation = torch.zeros(B, 3).to(pc.device)
                rotation = torch.zeros(B, 4).to(pc.device)
                rotation[:, 0] = 1
                init_loss, _ = self.loss_fn(pred_translation = translation, pred_rotation = rotation,
                            pcs = pc, gt_T_to_camera = gt_T_to_camera, init_T_to_camera = init_T_to_camera, mask = masks)
        else:
            init_loss = None
                
        pred_T = T_gt_expected
        
        return (pred_T, init_loss, loss)
        
