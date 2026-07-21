"""V60: Similarity Cross-Attention for Implicit Alignment.

Reverse cross-attention (3D query → 2D key/value) that produces a
(B, G, N_tokens) similarity matrix for direct supervision with GT
point-pixel correspondences (paper Eq. 4, 9).

This module is independent of the existing ExtrinsicAwareCrossAttention
which uses 2D queries → 3D KV for feature fusion.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCrossAttention(nn.Module):
    """3D-query → 2D-KV cross-attention for similarity matrix supervision.

    Produces per-layer similarity matrices that can be supervised with
    GT point-pixel correspondences via cross-entropy (L_sim).

    Args:
        feat_dim: Feature dimension for both modalities.
        n_layers: Number of decoder layers (paper uses 3).
        n_heads: Attention heads (paper uses 1 for clean similarity matrix).
        use_registry_token: Add learnable KV token for out-of-FOV points.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 1,
        use_registry_token: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_registry_token = use_registry_token
        self.dim_head = feat_dim // n_heads

        self.q_proj_layers = nn.ModuleList()
        self.k_proj_layers = nn.ModuleList()
        self.v_proj_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_q_layers = nn.ModuleList()
        self.norm_out_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.q_proj_layers.append(nn.Linear(feat_dim, feat_dim, bias=False))
            self.k_proj_layers.append(nn.Linear(feat_dim, feat_dim, bias=False))
            self.v_proj_layers.append(nn.Linear(feat_dim, feat_dim, bias=False))
            self.norm_q_layers.append(nn.LayerNorm(feat_dim))
            self.norm_out_layers.append(nn.LayerNorm(feat_dim))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Dropout(dropout),
            ))

        if use_registry_token:
            self.registry_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)

        self.dropout = dropout

    def forward(
        self,
        pc_features: torch.Tensor,
        img_features: torch.Tensor,
        pc_pos_emb: Optional[torch.Tensor] = None,
        img_pos_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            pc_features: (B, G, D) point cloud group features (queries).
            img_features: (B, N_img, D) image token features (keys/values).
            pc_pos_emb: (B, G, D) optional position encoding for 3D.
            img_pos_emb: (B, N_img, D) optional position encoding for 2D.

        Returns:
            updated_pc: (B, G, D) refined 3D features after cross-attention.
            sim_matrices: list of (B, G, N_kv) similarity matrices per layer.
        """
        B, G, D = pc_features.shape
        N_img = img_features.shape[1]

        kv_input = img_features
        if img_pos_emb is not None:
            kv_input = kv_input + img_pos_emb

        if self.use_registry_token:
            reg = self.registry_token.expand(B, -1, -1)
            kv_input = torch.cat([kv_input, reg], dim=1)

        N_kv = kv_input.shape[1]

        q = pc_features
        if pc_pos_emb is not None:
            q = q + pc_pos_emb

        sim_matrices = []
        scale = math.sqrt(self.dim_head)

        for layer_idx in range(self.n_layers):
            q_normed = self.norm_q_layers[layer_idx](q)

            Q = self.q_proj_layers[layer_idx](q_normed)
            K = self.k_proj_layers[layer_idx](kv_input)
            V = self.v_proj_layers[layer_idx](kv_input)

            if self.n_heads > 1:
                Q = Q.view(B, G, self.n_heads, self.dim_head).transpose(1, 2)
                K = K.view(B, N_kv, self.n_heads, self.dim_head).transpose(1, 2)
                V = V.view(B, N_kv, self.n_heads, self.dim_head).transpose(1, 2)
                attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale
                attn_weights = F.softmax(attn_logits, dim=-1)
                sim_matrix = attn_weights.mean(dim=1)
                if self.training and self.dropout > 0:
                    attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
                out = torch.matmul(attn_weights, V)
                out = out.transpose(1, 2).contiguous().view(B, G, D)
            else:
                attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale
                attn_weights = F.softmax(attn_logits, dim=-1)
                sim_matrix = attn_weights
                if self.training and self.dropout > 0:
                    attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
                out = torch.matmul(attn_weights, V)

            sim_matrices.append(sim_matrix)

            q = q + out
            q = q + self.ffn_layers[layer_idx](self.norm_out_layers[layer_idx](q))

        return q, sim_matrices
