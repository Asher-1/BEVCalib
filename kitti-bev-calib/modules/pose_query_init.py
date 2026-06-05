"""Pose query initialization for the CF-BEV-R Transformer decoder.

Generates initial pose queries from global RGB features, the current extrinsic
hypothesis, and the RoCR geometric rotation estimate, following CalibFormer §III-D
with T_init and R_geo enhancements.

Design: V42 CF-BEV-R (docs/V42_CF_BEV_DESIGN.md §2.2 Stage-3(F))
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from native_cross_attention import HarmonicEmbedding


class PoseQueryInit(nn.Module):
    """RGB-guided pose query initialization for the correlation decoder.

    Computes ``Q_0 = MLP(GAP(F_rgb) ⊕ Harmonic(T_init) ⊕ Harmonic(R_geo))``.

    Args:
        rgb_dim: channel dimension of global-average-pooled RGB features.
        num_queries: number of decoder pose queries (CalibFormer uses 6).
        query_dim: per-query embedding dimension.
        n_harmonic: number of harmonic frequency bands.
    """

    def __init__(
        self,
        rgb_dim: int = 256,
        num_queries: int = 6,
        query_dim: int = 256,
        n_harmonic: int = 6,
    ):
        super().__init__()
        self.rgb_dim = int(rgb_dim)
        self.num_queries = int(num_queries)
        self.query_dim = int(query_dim)

        self.harmonic_t = HarmonicEmbedding(n_harmonic)
        self.harmonic_r = HarmonicEmbedding(n_harmonic)

        t_dim = self.harmonic_t.get_output_dim(input_dims=16)
        r_dim = self.harmonic_r.get_output_dim(input_dims=9)
        mlp_in = self.rgb_dim + t_dim + r_dim
        hidden_dim = self.query_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_queries * self.query_dim),
        )

    def forward(
        self,
        rgb_gap: torch.Tensor,
        T_init: torch.Tensor,
        R_geo: torch.Tensor,
    ) -> torch.Tensor:
        """Generate initial pose queries.

        Args:
            rgb_gap: ``(B, D)`` global average pooled image features.
            T_init: ``(B, 4, 4)`` initial extrinsic matrix.
            R_geo: ``(B, 3, 3)`` RoCR geometric rotation estimate.

        Returns:
            Initial pose queries ``(B, num_queries, query_dim)``.
        """
        t_flat = T_init.reshape(T_init.shape[0], -1)
        r_flat = R_geo.reshape(R_geo.shape[0], -1)

        encoded = torch.cat([
            rgb_gap,
            self.harmonic_t(t_flat),
            self.harmonic_r(r_flat),
        ], dim=-1)

        out = self.mlp(encoded)
        return out.reshape(-1, self.num_queries, self.query_dim)
