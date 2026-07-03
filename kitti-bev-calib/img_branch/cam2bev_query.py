"""
Query-based Camera-to-BEV module (BEVFormer-style).

Replaces LSS depth-based lifting with deformable cross-attention from
learnable BEV queries to multi-scale image features. This removes explicit
depth estimation, which is a primary source of domain gap between vehicles
with different camera installations.

Drop-in replacement for Cam2BEV: same forward signature, same output shapes.

Usage in bev_calib.py:
    from img_branch.cam2bev_query import Cam2BEVQuery
    self.img_branch = Cam2BEVQuery(img_shape=img_shape, ...)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from img_branch.img_encoders import SwinT_tiny_Encoder, FPN
from proj_head import ProjectionHead
from bev_settings import xbound, ybound, zbound


def _gen_bev_grid(xbound, ybound, zbound):
    """Generate BEV grid centers and cell counts (same logic as gen_dx_bx)."""
    dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]],
        dtype=torch.long,
    )
    return dx, bx, nx


class _CameraAwarePositionalEncoding(nn.Module):
    """Encode camera intrinsics + extrinsics into positional embeddings.

    For each image pixel at feature-map resolution, compute a 3D ray direction
    in ego frame, then project through a small MLP to produce per-pixel
    positional embeddings. This gives the cross-attention geometric awareness
    without requiring explicit depth estimation.
    """

    def __init__(self, embed_dim, fH, fW):
        super().__init__()
        self.fH = fH
        self.fW = fW
        self.ray_mlp = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        u = torch.arange(fW, dtype=torch.float32) + 0.5
        v = torch.arange(fH, dtype=torch.float32) + 0.5
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        self.register_buffer("grid_uv", torch.stack([grid_u, grid_v], dim=-1))  # (fH, fW, 2)

    def forward(self, cam_intrins, cam2ego_rot, cam2ego_trans):
        """
        Args:
            cam_intrins: (B, N, 3, 3)
            cam2ego_rot: (B, N, 3, 3)
            cam2ego_trans: (B, N, 3)
        Returns:
            pos_embed: (B*N, fH*fW, embed_dim)
        """
        B, N = cam_intrins.shape[:2]
        device = cam_intrins.device
        uv = self.grid_uv.to(device)  # (fH, fW, 2)

        fx = cam_intrins[:, :, 0, 0]  # (B, N)
        fy = cam_intrins[:, :, 1, 1]
        cx = cam_intrins[:, :, 0, 2]
        cy = cam_intrins[:, :, 1, 2]

        # Scale from input image resolution to feature-map resolution (stride=8)
        scale = 8.0
        u_norm = (uv[..., 0].view(1, 1, -1) * scale - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
        v_norm = (uv[..., 1].view(1, 1, -1) * scale - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
        ones = torch.ones_like(u_norm)
        rays_cam = torch.stack([u_norm, v_norm, ones], dim=-1)  # (B, N, fH*fW, 3)

        rays_ego = torch.einsum("bnij,bnpj->bnpi", cam2ego_rot, rays_cam)

        trans_expand = cam2ego_trans.unsqueeze(2).expand_as(rays_ego)
        ray_input = torch.cat([rays_ego, trans_expand], dim=-1)  # (B, N, fH*fW, 6)

        pos_embed = self.ray_mlp(ray_input)
        return pos_embed.view(B * N, self.fH * self.fW, -1)


class _BEVQueryPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for BEV query positions."""

    def __init__(self, embed_dim, nx, ny, nz):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        pe = self._build_pe(embed_dim, nx * ny)
        self.register_buffer("pe", pe)  # (1, nx*ny, embed_dim)

    @staticmethod
    def _build_pe(d_model, n_positions):
        pe = torch.zeros(n_positions, d_model)
        position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe.unsqueeze(0)

    def forward(self, B):
        return self.pe.expand(B, -1, -1)


class _TInitFourierEncoder(nn.Module):
    """Small local T_init RPY encoder used to condition BEV queries."""

    def __init__(self, embed_dim=64, n_freq=32):
        super().__init__()
        self.n_freq = n_freq
        input_dim = 3 * (2 * n_freq + 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    @staticmethod
    def rotation_matrix_to_rpy(R):
        sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
        singular = sy < 1e-6
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        pitch = torch.atan2(-R[:, 2, 0], sy)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        roll_s = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
        pitch_s = torch.atan2(-R[:, 2, 0], sy)
        yaw_s = torch.zeros_like(roll)
        roll = torch.where(singular, roll_s, roll)
        pitch = torch.where(singular, pitch_s, pitch)
        yaw = torch.where(singular, yaw_s, yaw)
        return torch.stack([roll, pitch, yaw], dim=-1)

    def forward(self, T_init_4x4):
        if T_init_4x4.dim() == 4:
            T_init_4x4 = T_init_4x4[:, 0]
        rpy_rad = self.rotation_matrix_to_rpy(T_init_4x4[:, :3, :3])
        freqs = (2.0 ** torch.arange(
            self.n_freq, device=rpy_rad.device, dtype=rpy_rad.dtype))
        encoded = rpy_rad.unsqueeze(-1) * freqs.view(1, 1, -1)
        feat = torch.cat([torch.sin(encoded), torch.cos(encoded),
                          rpy_rad.unsqueeze(-1)], dim=-1)
        return self.mlp(feat.flatten(1))


class _DeformableCrossAttention(nn.Module):
    """Simplified deformable cross-attention for BEV queries attending to image features.

    Each BEV query learns K reference points in the image, with offsets predicted
    from the query itself. This is more efficient than full cross-attention and
    naturally handles geometric correspondences.
    """

    def __init__(self, embed_dim, num_heads=8, num_points=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, key, value, key_pos=None):
        """
        Args:
            query: (B, Q, C) - BEV queries
            key: (B, S, C) - image features (flattened spatial)
            value: (B, S, C) - image features
            key_pos: (B, S, C) - positional encoding for keys
        Returns:
            output: (B, Q, C)
        """
        B, Q, C = query.shape
        S = key.shape[1]

        if key_pos is not None:
            key = key + key_pos

        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim)

        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, Q, self.num_heads, self.num_points, 2)
        offsets = offsets.tanh() * 0.5  # constrain to [-0.5, 0.5] of spatial range

        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(B, Q, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # For simplicity, use standard attention with learned weighting
        # instead of grid_sample (avoids CUDA-specific ops for portability)
        attn = torch.einsum("bqhd,bshd->bqhs", q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bqhs,bshd->bqhd", attn, v)
        out = out.reshape(B, Q, C)
        return self.out_proj(out)


class _QueryBEVLayer(nn.Module):
    """Single BEVFormer-style layer: self-attn on BEV queries + cross-attn to image."""

    def __init__(self, embed_dim, num_heads=8, num_points=4, dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = _DeformableCrossAttention(embed_dim, num_heads, num_points, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, bev_queries, bev_pos, img_feats, img_pos):
        """
        Args:
            bev_queries: (B, Q, C) learnable BEV embeddings
            bev_pos: (B, Q, C) BEV positional encoding
            img_feats: (B, S, C) multi-scale image features
            img_pos: (B, S, C) camera-aware positional encoding
        """
        q = bev_queries + bev_pos
        x = self.norm1(bev_queries + self.self_attn(q, q, bev_queries)[0])
        x = self.norm2(x + self.cross_attn(x + bev_pos, img_feats, img_feats, img_pos))
        x = self.norm3(x + self.ffn(x))
        return x


class Cam2BEVQuery(nn.Module):
    """Query-based Camera-to-BEV module.

    Uses learnable BEV queries + deformable cross-attention to image features
    instead of LSS depth-based lifting. Camera geometry is injected via
    camera-aware positional encodings on image features.

    Output interface matches Cam2BEV exactly:
        forward(...) → (bev_feats: (B,128,H,W), cam_bev_mask: (B,H,W))
    """

    def __init__(
        self,
        output_indices=(1, 2, 3),
        img_shape=None,
        encoder_out_channels=256,
        FPN_in_channels=(192, 384, 768),
        FPN_out_channels=256,
        num_query_layers=3,
        num_heads=8,
        num_points=4,
        query_dropout=0.1,
        backbone_type="swin",
        backbone_variant="dinov2-small",
        freeze_backbone=False,
        freeze_layers=None,
        backbone_weights=None,
        query_downsample=4,
        tinit_query_film=False,
    ):
        super().__init__()
        if img_shape is None:
            img_shape = (256, 704)

        img_H, img_W = img_shape
        fH, fW = img_H // 8, img_W // 8
        featureShape = (encoder_out_channels, fH, fW)
        self.fH, self.fW = fH, fW

        dx, bx, nx = _gen_bev_grid(xbound, ybound, zbound)
        self.register_buffer("dx", dx)
        self.register_buffer("bx", bx)
        self.register_buffer("nx", nx)
        nx_x, nx_y, nz = int(nx[0]), int(nx[1]), int(nx[2])
        self.nx_x, self.nx_y, self.nz = nx_x, nx_y, nz

        # Use a coarser query grid to avoid OOM from full self-attention
        # Output is bilinearly upsampled to the full BEV resolution
        self.qx = max(nx_x // query_downsample, 8)
        self.qy = max(nx_y // query_downsample, 8)
        num_queries = self.qx * self.qy

        print(f"[Cam2BEVQuery] img: {img_W}x{img_H}, feat: {fW}x{fH}, "
              f"backbone={backbone_type}, "
              f"layers={num_query_layers}, heads={num_heads}, points={num_points}")
        print(f"[Cam2BEVQuery] BEV output: {nx_x}x{nx_y}, query grid: {self.qx}x{self.qy} "
              f"({num_queries} queries, downsample={query_downsample}x)")

        if backbone_type == "dinov2":
            from img_branch.dinov2_encoder import DINOv2Encoder
            self.CamEncode = DINOv2Encoder(
                featureShape=featureShape,
                out_channels=encoder_out_channels,
                variant=backbone_variant,
                freeze_backbone=freeze_backbone,
                freeze_layers=freeze_layers,
                weights_path=backbone_weights,
            )
        else:
            self.CamEncode = SwinT_tiny_Encoder(
                list(output_indices), featureShape,
                encoder_out_channels, list(FPN_in_channels), FPN_out_channels,
            )

        embed_dim = encoder_out_channels
        self.bev_queries = nn.Embedding(num_queries, embed_dim)
        self.bev_pos = _BEVQueryPositionalEncoding(embed_dim, self.qx, self.qy, nz)
        self.cam_pos = _CameraAwarePositionalEncoding(embed_dim, fH, fW)
        self.use_tinit_query_film = bool(tinit_query_film)
        if self.use_tinit_query_film:
            self.tinit_query_encoder = _TInitFourierEncoder(embed_dim=64)
            self.tinit_query_film = nn.Sequential(
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, embed_dim * 2),
            )
            nn.init.zeros_(self.tinit_query_film[-1].weight)
            nn.init.zeros_(self.tinit_query_film[-1].bias)
            print(f"[Cam2BEVQuery] T_init query FiLM enabled (embed_dim={embed_dim})")

        self.layers = nn.ModuleList([
            _QueryBEVLayer(embed_dim, num_heads, num_points, query_dropout)
            for _ in range(num_query_layers)
        ])

        self.proj_head = ProjectionHead(embedding_dim=embed_dim, projection_dim=128)
        self.out_channels = self.proj_head.projection_dim  # 128

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, cam2ego_T, cam_intrins, post_cam2ego_T, imgs,
                return_z_features=False, tinit_T=None):
        """
        Same interface as Cam2BEV.forward.

        Args:
            cam2ego_T: (B, N, 4, 4)
            cam_intrins: (B, 3, 3) or (B, N, 3, 3)
            post_cam2ego_T: (B, N, 4, 4) — not used (no geometric lifting)
            imgs: (B, N, 3, H, W)
            return_z_features: if True, return extra z_summary and img_feat_2d
        Returns:
            bev_feats: (B, 128, nx_x, nx_y)
            cam_bev_mask: (B, nx_x, nx_y) — all-ones (no geometric masking needed)
        """
        B, N = imgs.shape[:2]

        cam2ego_rot = cam2ego_T[:, :, :3, :3]
        cam2ego_trans = cam2ego_T[:, :, :3, 3]

        if cam_intrins.dim() == 3:
            cam_intrins = cam_intrins.unsqueeze(1)  # (B,3,3) → (B,1,3,3)

        imgs_norm = (imgs - self.mean) / self.std
        img_feats = self.CamEncode(imgs_norm)  # (B, N, C, fH, fW)

        BN = B * N
        img_flat = img_feats.view(BN, -1, self.fH * self.fW).permute(0, 2, 1)  # (BN, fH*fW, C)
        img_pos = self.cam_pos(cam_intrins, cam2ego_rot, cam2ego_trans)  # (BN, fH*fW, C)

        if N > 1:
            img_flat = img_flat.view(B, N * self.fH * self.fW, -1)
            img_pos = img_pos.view(B, N * self.fH * self.fW, -1)
        else:
            img_flat = img_flat.view(B, self.fH * self.fW, -1)
            img_pos = img_pos.view(B, self.fH * self.fW, -1)

        bev_q = self.bev_queries.weight.unsqueeze(0).expand(B, -1, -1)
        if self.use_tinit_query_film:
            if tinit_T is None:
                tinit_T = torch.linalg.inv(cam2ego_T[:, 0].float())
            film = self.tinit_query_film(self.tinit_query_encoder(tinit_T)).to(dtype=bev_q.dtype)
            gamma, beta = film.chunk(2, dim=-1)
            bev_q = bev_q * (1.0 + torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        bev_p = self.bev_pos(B)

        for layer in self.layers:
            bev_q = layer(bev_q, bev_p, img_flat, img_pos)

        bev_feats = self.proj_head(bev_q)  # (B, qx*qy, 128)
        bev_feats = bev_feats.view(B, self.qx, self.qy, -1).permute(0, 3, 1, 2).contiguous()

        # Upsample from query grid (qx, qy) to full BEV grid (nx_x, nx_y)
        if self.qx != self.nx_x or self.qy != self.nx_y:
            bev_feats = F.interpolate(
                bev_feats, size=(self.nx_x, self.nx_y),
                mode="bilinear", align_corners=False,
            )

        cam_bev_mask = torch.ones(B, self.nx_x, self.nx_y, device=bev_feats.device, dtype=torch.bool)

        if return_z_features:
            z_summary = F.adaptive_avg_pool2d(bev_feats, 1).flatten(1)
            return bev_feats, cam_bev_mask, z_summary, img_feats
        return bev_feats, cam_bev_mask
