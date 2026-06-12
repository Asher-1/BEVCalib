"""Correlation Transformer Decoder — Stage-3 head for V42 CF-BEV-R.

Encodes correlation token grids with a lightweight Swin-style encoder, then
decodes pose queries that cross-attend to the encoded features to predict a
quaternion rotation residual.

Design: V42 CF-BEV-R (docs/V42_CF_BEV_DESIGN.md §2.2 Stage-3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinCorrEncoder(nn.Module):
    """Lightweight encoder for correlation token sequences.

    Processes (B, G, D) correlation features with standard Transformer encoder
    layers (pre-norm, GELU).  For point-cloud groups the tokens are treated as
    a 1D sequence rather than a spatial grid.

    Args:
        d_model:         Feature dimension.
        nhead:           Number of attention heads.
        num_layers:      Number of encoder layers.
        dim_feedforward: FFN hidden dimension.
        dropout:         Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-5,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, corr_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            corr_tokens: (B, G, D) correlation features.

        Returns:
            (B, G, D) encoded correlation features.
        """
        return self.encoder(corr_tokens)


class PoseQueryDecoder(nn.Module):
    """Transformer decoder with pose queries cross-attending to correlation memory.

    Takes initial pose queries from PoseQueryInit, cross-attends to encoded
    correlation features, mean-pools across queries, and projects to a unit
    quaternion rotation residual.

    Optionally includes a magnitude estimation head (V46+) that predicts the
    perturbation angle magnitude in radians, parallel to the quaternion head.
    This teaches the model explicit magnitude awareness, decoupling "how much
    to correct" from "in which direction".

    Args:
        d_model:              Feature dimension.
        nhead:                Number of attention heads.
        num_layers:           Number of decoder layers.
        dim_feedforward:      FFN hidden dimension.
        dropout:              Dropout rate.
        use_magnitude_head:   Enable magnitude estimation branch.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_magnitude_head: bool = False,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.use_magnitude_head = use_magnitude_head
        self.pool_mode = pool_mode

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-5,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if pool_mode == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1),
            )

        mid_dim = d_model // 2
        self.quat_head = nn.Sequential(
            nn.Linear(d_model, mid_dim),
            nn.LayerNorm(mid_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 4),
        )
        with torch.no_grad():
            self.quat_head[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            nn.init.xavier_uniform_(self.quat_head[-1].weight, gain=0.01)

        if use_magnitude_head:
            mag_mid = d_model // 4
            self.mag_head = nn.Sequential(
                nn.Linear(d_model, mag_mid),
                nn.LayerNorm(mag_mid, eps=1e-5),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mag_mid, 1),
                nn.Softplus(),
            )
            with torch.no_grad():
                nn.init.zeros_(self.mag_head[-2].bias)
                nn.init.xavier_uniform_(self.mag_head[-2].weight, gain=0.01)

    def forward(
        self,
        pose_queries: torch.Tensor,
        memory: torch.Tensor,
    ):
        """
        Args:
            pose_queries: (B, num_queries, D) initial pose queries from PoseQueryInit.
            memory:       (B, G, D) encoded correlation features from SwinCorrEncoder.

        Returns:
            If use_magnitude_head=False: (B, 4) quaternion residual.
            If use_magnitude_head=True:  tuple of (delta_q (B,4), mag_pred (B,1)).
        """
        decoded = self.decoder(tgt=pose_queries, memory=memory)
        if self.pool_mode == "attention":
            attn_logits = self.attn_pool(decoded).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)
            pooled = (decoded * attn_weights).sum(dim=1)
        else:
            pooled = decoded.mean(dim=1)
        delta_q = self.quat_head(pooled)
        delta_q = F.normalize(delta_q, dim=-1)

        if self.use_magnitude_head:
            mag_pred = self.mag_head(pooled)
            return delta_q, mag_pred

        return delta_q


class CorrTransformerHead(nn.Module):
    """Stage-3 head: encode correlation tokens, decode pose queries → Δq.

    Combines SwinCorrEncoder and PoseQueryDecoder into a single module used
    after RoCR geometry initialization in the CF-BEV-R pipeline.

    Args:
        d_model:              Feature dimension.
        nhead:                Number of attention heads.
        encoder_layers:       Number of correlation encoder layers.
        decoder_layers:       Number of pose-query decoder layers.
        dim_feedforward:      FFN hidden dimension.
        dropout:              Dropout rate.
        use_magnitude_head:   Enable magnitude estimation branch.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_magnitude_head: bool = False,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.corr_encoder = SwinCorrEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.pose_decoder = PoseQueryDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_magnitude_head=use_magnitude_head,
            pool_mode=pool_mode,
        )

    def forward(
        self,
        corr_tokens: torch.Tensor,
        pose_queries: torch.Tensor,
    ):
        """
        Args:
            corr_tokens:  (B, G, D) correlation features from Stage-2 fusion.
            pose_queries: (B, num_queries, D) initial queries from PoseQueryInit.

        Returns:
            Without magnitude head: (B, 4) quaternion residual.
            With magnitude head: tuple of (delta_q (B,4), mag_pred (B,1)).
        """
        memory = self.corr_encoder(corr_tokens)
        return self.pose_decoder(pose_queries, memory)
