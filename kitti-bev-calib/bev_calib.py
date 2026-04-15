import torch
import torch.nn as nn
from img_branch.img_branch import Cam2BEV
from pc_branch.pc_branch import Lidar2BEV
from losses.losses import realworld_loss
from losses.quat_tools import quaternion_from_matrix
from deformable_attention import DeformableAttention
from BEVEncoder.BEVEncoder import BEVEncoder


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

    def forward(self, img_bev_feat, pc_bev_feat):
        return super().forward(torch.cat([img_bev_feat, pc_bev_feat], dim=1))
    
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
                ):
        super(BEVCalib, self).__init__()
        self.use_mlp_head = use_mlp_head
        self.rotation_only = rotation_only
        self.bev_pool_factor = bev_pool_factor
        self._profile_modules = False
        self._profile_events = []
        self._profile_accum = {}
        self._profile_count = 0
        self.img_branch = Cam2BEV(
            img_shape=img_shape,
            use_foundation_depth=use_foundation_depth,
            depth_model_type=depth_model_type,
            fd_mode=fd_mode,
        )
        self.pc_branch = Lidar2BEV()
        self.bev_encoder_use = bev_encoder
        if self.bev_encoder_use:
            self.bev_encoder = BEVEncoder()
        self.bev_shape = (self.img_branch.nx[0].item(), self.img_branch.nx[1].item())
        self.embed_dim = self.img_branch.out_channels + self.pc_branch.out_channels
        self.num_heads = num_heads
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
        if self.use_mlp_head:
            if not self.rotation_only:
                self.translation_pred = self._build_regression_head(
                    self.embed_dim, 3, head_dropout)
            self.rotation_pred = self._build_regression_head(
                self.embed_dim, 4, head_dropout)
        else:
            if not self.rotation_only:
                self.translation_pred = nn.Linear(self.embed_dim, 3)
            self.rotation_pred = nn.Linear(self.embed_dim, 4)
        self.loss_fn = realworld_loss(
            rotation_only=rotation_only,
            enable_axis_loss=enable_axis_loss,
            weight_axis_rotation=weight_axis_rotation,
            axis_weights=axis_weights,
            use_geodesic_loss=use_geodesic_loss,
        )
    
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
        q_norm = q.norm(dim=1, keepdim=True).clamp(min=1e-8)
        q = q / q_norm
        B = q.shape[0]
        R = torch.zeros(B, 3, 3).to(q.device)
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
    
    def forward(self, img, pc, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, cam_intrinsic, masks = None, out_init_loss = False):
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
        cam2ego_T = torch.linalg.inv(init_T_to_camera)
        cam_bev_feats, cam_bev_mask = self.img_branch(cam2ego_T=cam2ego_T, cam_intrins=cam_intrinsic, post_cam2ego_T=post_cam2ego_T, imgs=img) # B, C, H, W

        if profiling:
            _ev[1].record()

        pc = pc.permute(0, 2, 1).contiguous() # (B, 3, N)
        pc_bev_feats = self.pc_branch(pc) # B, C, H, W

        if profiling:
            _ev[2].record()

        x = self.conv_fuser(cam_bev_feats, pc_bev_feats) # B, C, H, W
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
            masked_x = torch.zeros(B, max_valid_cnt, C).to(x.device)
            padding_mask = torch.zeros(B, max_valid_cnt).to(x.device)
            for i in range(B):
                cnt = int(bev_mask[i].sum().item())
                valid_counts[i] = cnt
                masked_x[i, :cnt, :] = x[i, bev_mask[i]]
                padding_mask[i, cnt:] = 1
            x = self.transformer(masked_x, src_key_padding_mask=padding_mask.bool()) # B, max_valid_cnt, C
            x_pooled = torch.zeros(B, C, device=x.device)
            for i in range(B):
                cnt = valid_counts[i].item()
                if cnt > 0:
                    x_pooled[i] = x[i, :cnt, :].mean(dim=0)
            x = x_pooled

        if profiling:
            _ev[4].record()

        x = self.head_drop(x)
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
        
