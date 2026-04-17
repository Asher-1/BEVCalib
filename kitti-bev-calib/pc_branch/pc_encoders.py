"""Sparse 3D encoder for point cloud BEV feature extraction.

Uses ``drcv.ops.torch_sparse`` backend (same as pillarnet) for drinfer
trace compatibility.  Falls back to ``drcv.ops.spconv`` when
``USE_DRCV_BACKEND=0``.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn

USE_DRCV = os.environ.get("USE_DRCV_BACKEND", "1") == "1"

if USE_DRCV:
    # drcv.ops.torch_sparse uses deprecated collections.Sequence (removed in Python 3.10+)
    import collections, collections.abc
    if not hasattr(collections, "Sequence"):
        collections.Sequence = collections.abc.Sequence

    import drcv.ops.torch_sparse.nn as spnn
    from drcv.ops.torch_sparse.sparse_tensor import SparseTensor
    from drcv.model_zoo.common.ops import (
        SparseConvBnAct,
        SparseResBlock,
    )
    _SPARSE_NORM_CFG = dict(type="SparseBN1d", eps=1e-3, momentum=0.01)
else:
    import spconv.pytorch as spconv


# ---- helpers ----

def _replace_feature(st, new_features):
    st.F = new_features
    return st


def _conv_output_size(input_size, kernel_size, stride, padding):
    if isinstance(input_size, (list, tuple)):
        return [_conv_output_size(s, k, st, p)
                for s, k, st, p in zip(input_size, kernel_size, stride, padding)]
    return (input_size + 2 * padding - kernel_size) // stride + 1


# ---- spconv fallback (only when USE_DRCV=0) ----

if not USE_DRCV:
    def _spconv_conv_bn_act(in_ch, out_ch, kernel_size,
                            stride=1, padding=0, bias=False,
                            indice_key=None, conv_type='SubMConv3d'):
        conv_cls = spconv.SubMConv3d if conv_type == 'SubMConv3d' else spconv.SparseConv3d
        return spconv.SparseSequential(
            conv_cls(in_ch, out_ch, kernel_size,
                     stride=stride, padding=padding, bias=bias,
                     indice_key=indice_key),
            nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    class _SpconvBasicBlock(nn.Module):
        def __init__(self, inplanes, planes, stride=1, indice_key=None):
            super().__init__()
            self.conv1 = spconv.SubMConv3d(
                inplanes, planes, 3, stride=stride,
                padding=1, bias=False, indice_key=indice_key)
            self.bn1 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = spconv.SubMConv3d(
                planes, planes, 3, stride=1,
                padding=1, bias=False, indice_key=indice_key)
            self.bn2 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)
            if inplanes != planes:
                self.downsample = spconv.SparseSequential(
                    spconv.SubMConv3d(inplanes, planes, 1, bias=False),
                    nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01),
                )
            else:
                self.downsample = None
            self.bottle_neck = spconv.SparseSequential(
                self.conv1, self.bn1, self.relu, self.conv2, self.bn2,
            )

        def forward(self, x):
            identity = x.features
            out = self.bottle_neck(x)
            if self.downsample is not None:
                identity = self.downsample(x).features
            if hasattr(out, 'replace_feature'):
                out = out.replace_feature(out.features + identity)
                out = out.replace_feature(self.relu(out.features))
            else:
                out.features = out.features + identity
                out.features = self.relu(out.features)
            return out


# ---------------------------------------------------------------------------
#  Sparse -> Dense BEV conversion
# ---------------------------------------------------------------------------

class SpconvToDenseBEV(nn.Module):
    """Convert sparse features to dense BEV.

    For drcv backend with ``learned`` / ``sum`` mode: wraps
    ``spnn.ToDenseBEVConvolution`` / ``ToDenseBEVHeightAdd``.
    For ``concat`` mode or spconv fallback: manual scatter.
    """

    def __init__(self, in_channels, out_channels, n_z, bev_shape, mode='concat'):
        super().__init__()
        self.in_channels = in_channels
        self.n_z = n_z
        self.bev_h, self.bev_w = bev_shape
        self.mode = mode

        self.bev_op = None
        if mode == 'concat':
            self.out_channels = n_z * in_channels
        elif mode == 'learned':
            self.out_channels = out_channels
            self.kernel = nn.Parameter(torch.zeros(n_z, in_channels, out_channels))
            self.bias = nn.Parameter(torch.zeros(1, out_channels))
            std = 1.0 / math.sqrt(in_channels)
            self.kernel.data.uniform_(-std, std)
        elif mode == 'sum':
            self.out_channels = in_channels

    def forward(self, x):
        if self.bev_op is not None:
            return self.bev_op(x)

        if USE_DRCV:
            features = x.F
            indices = x.C.long()
            s = x.s
            batch_idx = indices[:, 3]
            d0_idx = indices[:, 0] // s
            d1_idx = indices[:, 1] // s
            z_idx = indices[:, 2] // s
            batch_size = batch_idx.max().item() + 1
        else:
            features = x.features
            indices = x.indices.long()
            batch_idx = indices[:, 0]
            d0_idx = indices[:, 1]
            d1_idx = indices[:, 2]
            z_idx = indices[:, 3]
            batch_size = x.batch_size

        z_idx = torch.clamp(z_idx, 0, self.n_z - 1)
        flat_bev = batch_idx * (self.bev_h * self.bev_w) + d0_idx * self.bev_w + d1_idx
        bev_size = batch_size * self.bev_h * self.bev_w

        if self.mode == 'concat':
            out_c = self.n_z * self.in_channels
            bev = features.new_zeros(bev_size * out_c)
            ch_offset = z_idx * self.in_channels
            col_base = ch_offset.unsqueeze(1) + torch.arange(
                self.in_channels, device=features.device).unsqueeze(0)
            flat_idx = flat_bev.unsqueeze(1).expand_as(col_base) * out_c + col_base
            bev.scatter_add_(0, flat_idx.reshape(-1), features.reshape(-1))
            bev = bev.view(bev_size, out_c)
        elif self.mode == 'learned':
            z_kernels = self.kernel[z_idx]
            bev_feat = torch.bmm(
                features.unsqueeze(1), z_kernels).squeeze(1) + self.bias
            out_c = self.out_channels
            bev = bev_feat.new_zeros(bev_size, out_c)
            bev.scatter_add_(0, flat_bev.unsqueeze(1).expand_as(bev_feat), bev_feat)
        else:
            out_c = self.in_channels
            bev = features.new_zeros(bev_size, out_c)
            bev.scatter_add_(0, flat_bev.unsqueeze(1).expand_as(features), features)

        return bev.view(batch_size, self.bev_h, self.bev_w, out_c) \
                  .permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
#  Encoder
# ---------------------------------------------------------------------------

class SparseEncoder(nn.Module):
    """3D sparse encoder: voxel features -> dense BEV map.

    When ``USE_DRCV=1`` (default): uses ``drcv.ops.torch_sparse`` modules
    (``SparseConvBnAct``, ``SparseResBlock``) — drinfer trace compatible.

    Architecture (drcv mode, pillarnet-style)::

        conv_input(k=3,s=1) -> stage0[ResBlock, ResBlock, ConvBnAct(s=2)]
                             -> stage1[ResBlock, ResBlock, ConvBnAct(s=2)]
                             -> stage2[ResBlock, ResBlock, ConvBnAct(s=2)]
                             -> stage3[ResBlock, ResBlock]
                             -> conv_out(k=1,s=1)     # channel projection only
                             -> to_bev (dense BEV)

    Total spatial downsample = 2^3 = 8x in each dimension.
    """

    def __init__(self, sparse_shape, in_channels=3, base_channels=16, out_channels=128,
                 layer_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
                 layer_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
                 to_bev_mode='concat',
                 ):
        super(SparseEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_channels = layer_channels
        self.layer_paddings = layer_paddings
        self.sparse_shape = sparse_shape
        self.base_channels = base_channels
        self.to_bev_mode = to_bev_mode

        if USE_DRCV:
            self.conv_input = SparseConvBnAct(in_channels, base_channels, kernel_size=3, stride=1, norm_cfg=_SPARSE_NORM_CFG)
        else:
            self.conv_input = _spconv_conv_bn_act(
                in_channels, base_channels, 3, padding=1, indice_key="subm1")

        self._build_layers()

        out_spatial = self._compute_output_spatial(sparse_shape)
        n_z = max(out_spatial[2], 1)
        bev_shape = (out_spatial[0], out_spatial[1])
        self.to_bev = SpconvToDenseBEV(
            out_channels, out_channels, n_z, bev_shape, mode=to_bev_mode)

    def _build_layers(self):
        self.conv_layers = nn.ModuleList()
        ch = self.base_channels

        for i, blocks in enumerate(self.layer_channels):
            stage = nn.ModuleList()
            for j, out_ch in enumerate(tuple(blocks)):
                pad = tuple(self.layer_paddings[i])[j]
                is_downsample = (j == len(blocks) - 1) and (i < len(self.layer_channels) - 1)

                if is_downsample:
                    if USE_DRCV:
                        stage.append(SparseConvBnAct(ch, out_ch, kernel_size=3, stride=2, norm_cfg=_SPARSE_NORM_CFG))
                    else:
                        stage.append(_spconv_conv_bn_act(
                            ch, out_ch, 3, stride=(2, 2, 2), padding=pad,
                            bias=True, indice_key=f"spconv{i+1}",
                            conv_type='SparseConv3d'))
                else:
                    if USE_DRCV:
                        stage.append(SparseResBlock(ch, out_ch, norm_cfg=_SPARSE_NORM_CFG))
                    else:
                        stage.append(_SpconvBasicBlock(
                            ch, out_ch, indice_key=f"subm{i+1}_{j}"))
                ch = out_ch
            self.conv_layers.append(stage)

        if USE_DRCV:
            self.conv_out = nn.Sequential(
                spnn.Conv3d(ch, self.out_channels, kernel_size=1, stride=1, bias=False),
                spnn.BatchNorm(self.out_channels, eps=1e-3, momentum=0.01),
                spnn.ReLU(inplace=True),
            )
        else:
            self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(
                    ch, self.out_channels,
                    kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 0),
                    indice_key="spconv_down2", bias=False),
                nn.BatchNorm1d(self.out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )

    def _compute_output_spatial(self, sparse_shape):
        shape = list(sparse_shape)

        if USE_DRCV:
            n_downsamples = sum(
                1 for i, blocks in enumerate(self.layer_channels)
                for j in range(len(blocks))
                if (j == len(blocks) - 1) and (i < len(self.layer_channels) - 1)
            )
            ds_factor = 2 ** n_downsamples
            shape = [max(s // ds_factor, 1) for s in shape]
            return shape

        for i, blocks in enumerate(self.layer_channels):
            for j in range(len(blocks)):
                is_downsample = (j == len(blocks) - 1) and (i < len(self.layer_channels) - 1)
                if is_downsample:
                    pad_raw = tuple(self.layer_paddings[i])[j]
                    if isinstance(pad_raw, (list, tuple)):
                        pad = list(pad_raw)
                    else:
                        pad = [pad_raw] * 3
                    shape = _conv_output_size(shape, [3, 3, 3], [2, 2, 2], pad)
        shape = _conv_output_size(shape, [3, 3, 3], [1, 1, 2], [1, 1, 0])
        return shape

    def forward(self, features, coors, batch_size):
        if USE_DRCV:
            coors_inv = torch.stack(
                [coors[:, 1], coors[:, 2], coors[:, 3], coors[:, 0]], dim=-1)
            x = SparseTensor(features, coors_inv)
        else:
            coors = coors.int()
            x = spconv.SparseConvTensor(features, coors, self.sparse_shape, batch_size)

        x = self.conv_input(x)

        for stage in self.conv_layers:
            for block in stage:
                x = block(x)

        x = self.conv_out(x)
        return self.to_bev(x)
