"""DLA multi-scale feature aggregation for CF-BEV-R.

Aggregates Swin-Tiny FPN features at 1/8, 1/16, and 1/32 into a single
1/4-resolution feature map following CalibFormer §III-B.

Design: V42 CF-BEV-R (docs/V42_CF_BEV_DESIGN.md §2.5)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLAAggregation(nn.Module):
    """DLA skip aggregation: 1/8 + upsample(1/16) + upsample(1/32) → 1/4.

    Each FPN scale is optionally projected to a common channel width, bilinearly
    upsampled to H/4 × W/4, summed, and passed through a 1×1 conv.

    Args:
        in_channels_list: per-scale input channel counts
            ``[C_1/8, C_1/16, C_1/32]``.
        out_channels: unified output channel width (default 256).
    """

    def __init__(
        self,
        in_channels_list: list[int] | None = None,
        out_channels: int = 256,
    ):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [256, 256, 256]
        self.in_channels_list = list(in_channels_list)
        self.out_channels = int(out_channels)

        self.scale_convs = nn.ModuleList([
            nn.Conv2d(c, self.out_channels, kernel_size=1, bias=False)
            if c != self.out_channels
            else nn.Identity()
            for c in self.in_channels_list
        ])
        self.out_conv = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=1, bias=False,
        )

    def forward(self, feature_list: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate multi-scale FPN features to 1/4 resolution.

        Args:
            feature_list: ``[feat_1_8, feat_1_16, feat_1_32]`` with shapes
                ``(B, C_i, H/8, W/8)``, ``(B, C_i, H/16, W/16)``,
                ``(B, C_i, H/32, W/32)``.

        Returns:
            Unified feature map ``(B, out_channels, H/4, W/4)``.
        """
        assert len(feature_list) == len(self.in_channels_list), (
            f"expected {len(self.in_channels_list)} features, "
            f"got {len(feature_list)}"
        )

        feat_1_8 = feature_list[0]
        target_h = feat_1_8.shape[2] * 2
        target_w = feat_1_8.shape[3] * 2
        target_size = (target_h, target_w)

        aggregated = None
        for feat, conv in zip(feature_list, self.scale_convs):
            x = conv(feat)
            x = F.interpolate(
                x, size=target_size, mode="bilinear", align_corners=False,
            )
            aggregated = x if aggregated is None else aggregated + x

        return self.out_conv(aggregated)
