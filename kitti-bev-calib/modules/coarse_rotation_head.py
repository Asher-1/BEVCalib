"""
V60 Coarse Rotation Head (IJCV 2026 paper Section 3.2.3).

Predicts R_coarse from SimCrossAttention output features.
Architecture:
  1. T2 single-query cross-attention: learnable query cross-attends to
     image features for global context aggregation (paper Sec 3.2.3)
  2. Attention pooling over refined 3D features
  3. Concatenate T2 output + pooled 3D → MLP → 6D rotation → R_coarse

Uses the 6D continuous rotation representation (Zhou et al., CVPR 2019)
for regressing rotation, avoiding gimbal lock and discontinuities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CoarseRotationHead(nn.Module):
    """
    Predicts a coarse rotation (R_coarse) from aggregated 3D features
    output by SimCrossAttention, optionally enriched by a T2 cross-attention
    back to image features.

    Paper flow:
      sim_pc_out (B, G, D) → T2 cross-attn to img_feat → pool → MLP → R_coarse (3x3)
    """

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 256,
        use_attention_pool: bool = True,
        use_t2_cross_attn: bool = True,
        n_t2_heads: int = 4,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.use_attention_pool = use_attention_pool
        self.use_t2_cross_attn = use_t2_cross_attn

        # T2: single learnable query cross-attends to image features
        if use_t2_cross_attn:
            self.t2_query = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
            self.t2_q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
            self.t2_k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
            self.t2_v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
            self.t2_out_proj = nn.Linear(feat_dim, feat_dim)
            self.t2_norm = nn.LayerNorm(feat_dim)
            self.n_t2_heads = n_t2_heads
            self.t2_dim_head = feat_dim // n_t2_heads

        # Attention pooling over 3D features
        if use_attention_pool:
            self.attn_query = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
            self.attn_proj_k = nn.Linear(feat_dim, feat_dim, bias=False)

        # MLP: input is [pooled_3d, t2_output] if T2 enabled, else [pooled_3d]
        mlp_input_dim = feat_dim * 2 if use_t2_cross_attn else feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6),  # 6D rotation representation
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize final layer to predict near-identity rotation."""
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(
        self,
        pc_features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        img_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pc_features: (B, G, D) output from SimCrossAttention.
            valid_mask: (B, G) boolean mask for valid groups.
            img_features: (B, N_img, D) image features for T2 cross-attention.

        Returns:
            R_coarse: (B, 3, 3) predicted coarse rotation matrix.
        """
        B, G, D = pc_features.shape

        # T2 cross-attention: single query → image features
        t2_out = None
        if self.use_t2_cross_attn and img_features is not None:
            query = self.t2_query.expand(B, -1, -1)
            Q = self.t2_q_proj(query)
            K = self.t2_k_proj(img_features)
            V = self.t2_v_proj(img_features)

            N_img = img_features.shape[1]
            h = self.n_t2_heads
            d = self.t2_dim_head

            Q = Q.view(B, 1, h, d).transpose(1, 2)      # (B, h, 1, d)
            K = K.view(B, N_img, h, d).transpose(1, 2)   # (B, h, N_img, d)
            V = V.view(B, N_img, h, d).transpose(1, 2)   # (B, h, N_img, d)

            attn = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, V)                    # (B, h, 1, d)
            out = out.transpose(1, 2).contiguous().view(B, D)
            t2_out = self.t2_norm(self.t2_out_proj(out))

        # Attention pooling over 3D features
        if self.use_attention_pool:
            query = self.attn_query.expand(B, -1, -1)
            keys = self.attn_proj_k(pc_features)
            attn = torch.bmm(query, keys.transpose(1, 2)) / (D ** 0.5)
            if valid_mask is not None:
                attn = attn.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            pooled = torch.bmm(attn, pc_features).squeeze(1)  # (B, D)
        else:
            if valid_mask is not None:
                pc_masked = pc_features.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))
            else:
                pc_masked = pc_features
            pooled = pc_masked.max(dim=1)[0]

        # Concat and predict
        if t2_out is not None:
            feat = torch.cat([pooled, t2_out], dim=-1)
        else:
            if self.use_t2_cross_attn:
                feat = torch.cat([pooled, torch.zeros_like(pooled)], dim=-1)
            else:
                feat = pooled

        rot_6d = self.mlp(feat)
        R_coarse = self._6d_to_rotation_matrix(rot_6d)
        return R_coarse

    @staticmethod
    def _6d_to_rotation_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D rotation representation to 3x3 rotation matrix (Gram-Schmidt)."""
        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:6]

        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        return torch.stack([b1, b2, b3], dim=-1)
