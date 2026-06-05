"""
V36: Native-Domain Extrinsic-Aware Cross-Attention for Camera-LiDAR Calibration.

Inspired by ProjFusion (IROS 2025), this module performs cross-attention between
image patch features and point cloud features in their native representation spaces,
using the current extrinsic hypothesis to compute projection-based positional embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List


class HarmonicEmbedding(nn.Module):
    """NeRF-style positional encoding."""

    def __init__(self, n_harmonic_functions: int = 6, omega_0: float = 1.0,
                 logspace: bool = True, append_input: bool = True):
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions, dtype=torch.float32)
        self.register_buffer("_frequencies", frequencies * omega_0 * math.pi)
        self.register_buffer("_zero_half_pi", torch.tensor([0.0, 0.5 * math.pi]))
        self.append_input = append_input
        self.n_harmonic_functions = n_harmonic_functions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = x[..., None] * self._frequencies
        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]
        embed = embed.sin()
        embed = embed.reshape(*x.shape[:-1], -1)
        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed

    def get_output_dim(self, input_dims: int = 2) -> int:
        return input_dims * (2 * self.n_harmonic_functions + int(self.append_input))


def farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """FPS on a single point set, GPU-friendly (no .item() sync).
    xyz: (N, 3) -> (G, 3)."""
    n = xyz.shape[0]
    if n_samples >= n:
        return xyz
    selected = torch.zeros(n_samples, dtype=torch.long, device=xyz.device)
    dist = torch.full((n,), 1e10, device=xyz.device, dtype=xyz.dtype)
    farthest = torch.randint(0, n, (1,), device=xyz.device)
    for i in range(n_samples):
        selected[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        d = ((xyz - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(dim=0, keepdim=True)
    return xyz[selected]


def farthest_point_sample_batch(pcd: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Batch FPS — all samples processed in parallel, no Python for-loop over B.
    pcd: (B, N, 3) -> (B, G, 3).  ~40x faster than per-sample FPS."""
    B, N, _ = pcd.shape
    if n_samples >= N:
        return pcd[:, :n_samples]
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=pcd.device)
    dist = torch.full((B, N), 1e10, device=pcd.device, dtype=pcd.dtype)
    farthest = torch.randint(0, N, (B, 1), device=pcd.device)
    batch_idx = torch.arange(B, device=pcd.device)
    for i in range(n_samples):
        selected[:, i:i+1] = farthest
        centroid = pcd[batch_idx, farthest.squeeze(1)].unsqueeze(1)
        d = ((pcd - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(dim=1, keepdim=True)
    gather_idx = selected.unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(pcd, 1, gather_idx)


class PointEncoder(nn.Module):
    """Point cloud encoder with FPS grouping + local kNN geometry."""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, out_dim: int = 128,
                 n_groups: int = 128, knn: int = 8, use_fps: bool = True):
        super().__init__()
        self.n_groups = n_groups
        self.knn = knn
        self.use_fps = use_fps
        # center xyz (3) + mean offset (3) + std offset (3) + max dist (1) = 10
        self.mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def _encode_groups(self, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """pts: (M, 3) valid points -> centers (G, 3), feats (G, D)."""
        g = min(self.n_groups, pts.shape[0])
        if g <= 0:
            z = pts.new_zeros(1, 3)
            return z, self.mlp(torch.cat([z, z, z, z.new_zeros(1)], dim=-1))

        if self.use_fps and g < pts.shape[0]:
            centers = farthest_point_sample(pts, g)
        else:
            stride = max(1, pts.shape[0] // g)
            idx = torch.arange(0, pts.shape[0], stride, device=pts.device)[:g]
            centers = pts[idx]

        return self._knn_features(centers, pts)

    def _knn_features(self, centers: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute kNN local geometry features for given centers."""
        k = min(self.knn, pts.shape[0])
        dist = torch.cdist(centers, pts)
        knn_idx = dist.topk(k, largest=False, dim=-1).indices
        neighbors = pts[knn_idx]
        rel = neighbors - centers.unsqueeze(-2 if centers.dim() == 3 else 1)
        mean_off = rel.mean(dim=-2)
        std_off = rel.std(dim=-2).clamp(min=1e-6)
        max_dist = rel.norm(dim=-1).max(dim=-1).values.unsqueeze(-1)
        feat_in = torch.cat([centers, mean_off, std_off, max_dist], dim=-1)
        return centers, self.mlp(feat_in)

    def _batch_forward(self, pcd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch-parallel FPS + kNN — eliminates Python for-loop over batch.

        Uses batch FPS (GPU-friendly, no .item() sync) when use_fps=True,
        or stride sampling when use_fps=False.
        """
        B, N, _ = pcd.shape
        G = min(self.n_groups, N)

        if self.use_fps and G < N:
            centers = farthest_point_sample_batch(pcd, G)  # (B, G, 3)
        else:
            stride = max(1, N // G)
            idx = torch.arange(0, N, stride, device=pcd.device)[:G]
            centers = pcd[:, idx]

        k = min(self.knn, N)
        dist = torch.cdist(centers, pcd)  # (B, G, N)
        knn_idx = dist.topk(k, largest=False, dim=-1).indices  # (B, G, k)

        batch_idx = torch.arange(B, device=pcd.device).view(B, 1, 1).expand(-1, G, k)
        neighbors = pcd[batch_idx, knn_idx]  # (B, G, k, 3)
        rel = neighbors - centers.unsqueeze(2)
        mean_off = rel.mean(dim=2)
        std_off = rel.std(dim=2).clamp(min=1e-6)
        max_dist = rel.norm(dim=-1).max(dim=2).values.unsqueeze(-1)
        feat_in = torch.cat([centers, mean_off, std_off, max_dist], dim=-1)
        feats = self.mlp(feat_in)
        return centers, feats

    def forward(self, pcd: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pcd: (B, N, 3)
            mask: (B, N) optional validity mask
        Returns:
            xyz_groups: (B, G, 3), feat_groups: (B, G, D)
        """
        if mask is None:
            return self._batch_forward(pcd)

        B, N, _ = pcd.shape
        valid_counts = mask.sum(dim=1).long()  # (B,)
        min_valid = valid_counts.min().item()

        if min_valid >= self.n_groups:
            order = mask.float().argsort(dim=1, descending=True, stable=True)
            pcd_sorted = torch.gather(pcd, 1, order.unsqueeze(-1).expand(-1, -1, 3))
            pcd_compact = pcd_sorted[:, :min_valid]  # (B, min_valid, 3)
            return self._batch_forward(pcd_compact)

        xyz_groups: List[torch.Tensor] = []
        feat_groups: List[torch.Tensor] = []
        for b in range(B):
            pts = pcd[b]
            valid = mask[b].bool()
            if valid.any():
                pts = pts[valid]
            if pts.shape[0] == 0:
                pts = pcd[b, :1]
            c, f = self._encode_groups(pts)
            xyz_groups.append(c)
            feat_groups.append(f)

        max_g = max(x.shape[0] for x in xyz_groups)
        xyz_out = torch.zeros(B, max_g, 3, device=pcd.device, dtype=pcd.dtype)
        feat_out = torch.zeros(B, max_g, self.out_dim, device=pcd.device, dtype=pcd.dtype)
        for b in range(B):
            g = xyz_groups[b].shape[0]
            xyz_out[b, :g] = xyz_groups[b]
            feat_out[b, :g] = feat_groups[b]
        return xyz_out, feat_out


class ExtrinsicAwareCrossAttention(nn.Module):
    """Cross-attention: image queries, point cloud keys/values with extrinsic-aware pos emb."""

    def __init__(self, img_feat_dim: int, pc_feat_dim: int,
                 n_harmonic: int = 6, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = dropout
        inner_dim = heads * dim_head

        harmonic_dim = HarmonicEmbedding(n_harmonic).get_output_dim(input_dims=2)
        q_input_dim = img_feat_dim + harmonic_dim
        kv_input_dim = pc_feat_dim + harmonic_dim

        self.harmonic = HarmonicEmbedding(n_harmonic)
        self.norm_q = nn.LayerNorm(q_input_dim)
        self.norm_kv = nn.LayerNorm(kv_input_dim)
        self.to_q = nn.Linear(q_input_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_input_dim, inner_dim * 2, bias=False)
        self.q_norm = nn.LayerNorm(dim_head)
        self.k_norm = nn.LayerNorm(dim_head)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = inner_dim

    def forward(self, feat_2d: torch.Tensor, feat_3d: torch.Tensor,
                img_pos_emb: torch.Tensor, proj_pos_emb: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_in = torch.cat([feat_2d, img_pos_emb], dim=-1)
        kv_in = torch.cat([feat_3d, proj_pos_emb], dim=-1)
        q_in = self.norm_q(q_in)
        kv_in = self.norm_kv(kv_in)
        q = self.to_q(q_in)
        kv = self.to_kv(kv_in)
        k, v = kv.chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=0.0 if not self.training else self.attn_dropout)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with FFN and residual."""

    def __init__(self, img_feat_dim: int, pc_feat_dim: int,
                 n_harmonic: int, heads: int, dim_head: int,
                 dropout: float, ffn_mult: int = 4):
        super().__init__()
        self.cross_attn = ExtrinsicAwareCrossAttention(
            img_feat_dim=img_feat_dim, pc_feat_dim=pc_feat_dim,
            n_harmonic=n_harmonic, heads=heads, dim_head=dim_head,
            dropout=dropout)
        attn_out_dim = heads * dim_head
        self.proj_residual = nn.Linear(img_feat_dim, attn_out_dim) \
            if img_feat_dim != attn_out_dim else nn.Identity()
        self.ffn = nn.Sequential(
            nn.LayerNorm(attn_out_dim),
            nn.Linear(attn_out_dim, attn_out_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_out_dim * ffn_mult, attn_out_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = attn_out_dim

    def forward(self, feat_2d, feat_3d, img_pos_emb, proj_pos_emb,
                attn_mask=None):
        residual = self.proj_residual(feat_2d)
        x = self.cross_attn(feat_2d, feat_3d, img_pos_emb, proj_pos_emb, attn_mask)
        x = x + residual
        x = x + self.ffn(x)
        return x


def _build_cross_stack(n_layers, img_feat_dim, pc_feat_dim, n_harmonic,
                       heads, dim_head, dropout, ffn_mult):
    attn_out_dim = heads * dim_head
    if n_layers == 1:
        return nn.ModuleList([ExtrinsicAwareCrossAttention(
            img_feat_dim=img_feat_dim, pc_feat_dim=pc_feat_dim,
            n_harmonic=n_harmonic, heads=heads, dim_head=dim_head,
            dropout=dropout)]), attn_out_dim, False
    layers = nn.ModuleList()
    for i in range(n_layers):
        in_dim = img_feat_dim if i == 0 else attn_out_dim
        layers.append(CrossAttentionBlock(
            img_feat_dim=in_dim, pc_feat_dim=pc_feat_dim,
            n_harmonic=n_harmonic, heads=heads, dim_head=dim_head,
            dropout=dropout, ffn_mult=ffn_mult))
    return layers, attn_out_dim, True


class NativeCrossCalibHead(nn.Module):
    """V36 calibration head with optional dual-branch cross-attention."""

    def __init__(self, img_feat_dim: int = 384, pc_groups: int = 128,
                 pc_feat_dim: int = 128, n_harmonic: int = 6,
                 heads: int = 8, dim_head: int = 64,
                 n_layers: int = 1, ffn_mult: int = 4,
                 dropout: float = 0.1, rotation_only: bool = True,
                 dual_branch: bool = True, knn: int = 8, use_fps: bool = True,
                 use_pointgpt: bool = False,
                 pointgpt_config: Optional[str] = None,
                 pointgpt_ckpt: Optional[str] = None,
                 pointgpt_max_depth: float = 50.0,
                 pointgpt_freeze: bool = True,
                 extend_ratio: float = 1.0):
        super().__init__()
        self.rotation_only = rotation_only
        self.use_pointgpt = use_pointgpt
        self.n_layers = n_layers
        self.dual_branch = dual_branch and rotation_only
        self.extend_ratio = float(extend_ratio)

        if use_pointgpt:
            from pointgpt_wrapper import PointGPTEncoder, DEFAULT_POINTGPT_CKPT, DEFAULT_POINTGPT_CONFIG
            self.point_encoder = PointGPTEncoder(
                config_path=pointgpt_config or DEFAULT_POINTGPT_CONFIG,
                checkpoint_path=pointgpt_ckpt or DEFAULT_POINTGPT_CKPT,
                max_depth=pointgpt_max_depth,
                freeze=pointgpt_freeze,
            )
            pc_groups = self.point_encoder.n_groups
            pc_feat_dim = self.point_encoder.out_dim
        else:
            self.point_encoder = PointEncoder(
                in_dim=3, hidden_dim=64, out_dim=pc_feat_dim,
                n_groups=pc_groups, knn=knn, use_fps=use_fps)
        self.pc_groups = pc_groups
        self.harmonic = HarmonicEmbedding(n_harmonic)

        self.rot_layers, attn_out_dim, self._use_blocks = _build_cross_stack(
            n_layers, img_feat_dim, pc_feat_dim, n_harmonic,
            heads, dim_head, dropout, ffn_mult)

        if self.dual_branch:
            self.aux_layers, _, self._aux_use_blocks = _build_cross_stack(
                n_layers, img_feat_dim, pc_feat_dim, n_harmonic,
                heads, dim_head, dropout, ffn_mult)
            agg_in = attn_out_dim * 2
        else:
            self.aux_layers = None
            agg_in = attn_out_dim

        self.aggregation = nn.Sequential(
            nn.LayerNorm(agg_in),
            nn.Linear(agg_in, agg_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_dim = agg_in // 2
        self.rot_head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, 4),
        )
        if not rotation_only:
            self.tsl_layers, tsl_out, self._tsl_use_blocks = _build_cross_stack(
                n_layers, img_feat_dim, pc_feat_dim, n_harmonic,
                heads, dim_head, dropout, ffn_mult)
            self.tsl_head = nn.Sequential(
                nn.Linear(tsl_out // 2, tsl_out // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tsl_out // 4, 3),
            )
            self.tsl_aggregation = nn.Sequential(
                nn.LayerNorm(tsl_out),
                nn.Linear(tsl_out, tsl_out // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def _run_cross_stack(self, layers, use_blocks, img_feat, feat_3d,
                         img_pos_emb, proj_pos_emb, attn_mask):
        if not use_blocks:
            return layers[0](img_feat, feat_3d, img_pos_emb, proj_pos_emb, attn_mask)
        x = img_feat
        for layer in layers:
            x = layer(x, feat_3d, img_pos_emb, proj_pos_emb, attn_mask=attn_mask)
        return x

    def compute_projection(self, xyz: torch.Tensor, T_init: torch.Tensor,
                           cam_intrinsic: torch.Tensor,
                           img_h: int, img_w: int,
                           feat_h: int, feat_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project to feature-map resolution (ProjFusion-style scaled intrinsics)."""
        B, G, _ = xyz.shape
        xyz_h = torch.cat([xyz, torch.ones(B, G, 1, device=xyz.device)], dim=-1)
        xyz_cam = torch.bmm(xyz_h, T_init.transpose(1, 2))[:, :, :3]

        depth = xyz_cam[:, :, 2].clamp(min=1e-3)
        sx = feat_w / float(max(img_w - 1, 1))
        sy = feat_h / float(max(img_h - 1, 1))
        ex = max(self.extend_ratio, 1.0)
        ey = ex
        fx = cam_intrinsic[:, 0, 0].unsqueeze(1) * sx
        fy = cam_intrinsic[:, 1, 1].unsqueeze(1) * sy
        # ProjFusion-style: expand principal point for large init bias projection
        cx = cam_intrinsic[:, 0, 2].unsqueeze(1) * sx * ex
        cy = cam_intrinsic[:, 1, 2].unsqueeze(1) * sy * ey

        u = fx * xyz_cam[:, :, 0] / depth + cx
        v = fy * xyz_cam[:, :, 1] / depth + cy
        u_norm = 2.0 * u / max(feat_w - 1, 1) - 1.0
        v_norm = 2.0 * v / max(feat_h - 1, 1) - 1.0
        uv_bound = 2.0 * max(ex, ey)
        proj_uv_norm = torch.stack([u_norm, v_norm], dim=-1).clamp(-uv_bound, uv_bound)

        valid = (xyz_cam[:, :, 2] > 0) & (u_norm.abs() < uv_bound) & (v_norm.abs() < uv_bound)
        empty = valid.sum(dim=1) == 0
        if empty.any():
            valid = valid.clone()
            valid[empty] = True
        return proj_uv_norm, valid

    def compute_img_grid_coords(self, feat_h: int, feat_w: int,
                                device: torch.device) -> torch.Tensor:
        yi = torch.linspace(-1, 1, feat_h, device=device)
        xi = torch.linspace(-1, 1, feat_w, device=device)
        grid_y, grid_x = torch.meshgrid(yi, xi, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    def forward(self, img_feat: torch.Tensor, pcd: torch.Tensor,
                T_init: torch.Tensor, cam_intrinsic: torch.Tensor,
                img_h: int, img_w: int,
                feat_h: int, feat_w: int,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = img_feat.shape[0]
        device = img_feat.device

        xyz_groups, feat_3d = self.point_encoder(pcd, mask=mask)
        proj_uv, valid_mask = self.compute_projection(
            xyz_groups, T_init, cam_intrinsic, img_h, img_w, feat_h, feat_w)
        proj_pos_emb = self.harmonic(proj_uv)
        img_grid = self.compute_img_grid_coords(feat_h, feat_w, device)
        img_pos_emb = self.harmonic(img_grid).unsqueeze(0).expand(B, -1, -1)

        n_img = feat_h * feat_w
        n_pc = feat_3d.shape[1]
        attn_mask = valid_mask.unsqueeze(1).expand(-1, n_img, -1).float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, 0.0)

        rot_feat = self._run_cross_stack(
            self.rot_layers, self._use_blocks,
            img_feat, feat_3d, img_pos_emb, proj_pos_emb, attn_mask)

        if self.dual_branch:
            aux_feat = self._run_cross_stack(
                self.aux_layers, self._aux_use_blocks,
                img_feat, feat_3d, img_pos_emb, proj_pos_emb, attn_mask)
            cross_feat = torch.cat([rot_feat, aux_feat], dim=-1)
        else:
            cross_feat = rot_feat

        x = cross_feat.mean(dim=1)
        x = self.aggregation(x)
        rotation = self.rot_head(x)
        rotation = F.normalize(rotation, dim=-1, eps=1e-6)

        if not self.rotation_only:
            tsl_feat = self._run_cross_stack(
                self.tsl_layers, self._tsl_use_blocks,
                img_feat, feat_3d, img_pos_emb, proj_pos_emb, attn_mask)
            tsl_x = self.tsl_aggregation(tsl_feat.mean(dim=1))
            translation = self.tsl_head(tsl_x)
        else:
            translation = torch.zeros(B, 3, device=device)

        return rotation, translation

    @torch.no_grad()
    def iterative_inference(self, img_feat, pcd, T_init, cam_intrinsic,
                            img_h, img_w, feat_h, feat_w,
                            n_iters: int = 3, mask=None):
        """Iterative refinement using the same composition as realworld_loss.

        T_gt_expected = inv(T_pred) @ T_init, where T_pred = tvec_mat @ quat_mat.
        """
        from losses.quat_tools import batch_quat2mat, batch_tvector2mat

        T_current = T_init.clone()
        for _ in range(n_iters):
            rot_q, tsl = self.forward(
                img_feat, pcd, T_current, cam_intrinsic,
                img_h, img_w, feat_h, feat_w, mask=mask)
            rot_q = F.normalize(rot_q, dim=-1, eps=1e-6)
            T_pred = torch.bmm(batch_tvector2mat(tsl), batch_quat2mat(rot_q))
            T_current = torch.bmm(
                torch.linalg.inv(T_pred.float()), T_current.float())
            if self.rotation_only:
                T_current = T_current.clone()
                T_current[:, :3, 3] = T_init[:, :3, 3]
        return T_current
