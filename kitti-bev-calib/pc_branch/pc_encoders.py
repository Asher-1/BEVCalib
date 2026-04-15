import os
import torch
import torch.nn as nn

# Backend switch: set USE_DRCV_BACKEND=1 to use drcv ops instead of spconv
USE_DRCV = os.environ.get("USE_DRCV_BACKEND", "1") == "1"

if USE_DRCV:
    import drcv.ops.spconv as spconv
else:
    import spconv.pytorch as spconv


def _replace_feature(tensor, new_features):
    """Cross-backend feature replacement on SparseConvTensor.
    spconv v2 provides replace_feature(); drcv (spconv v1) uses direct assignment.
    """
    if hasattr(tensor, 'replace_feature'):
        return tensor.replace_feature(new_features)
    tensor.features = new_features
    return tensor


class SparseBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)
        # Bottleneck sequence
        self.bottle_neck = spconv.SparseSequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
        )

    def forward(self, x):
        residual = x
        out = self.bottle_neck(x)
        out = _replace_feature(out, out.features + residual.features)
        out = _replace_feature(out, self.relu(out.features))
        return out


class SparseEncoder(nn.Module):
    def __init__(self, sparse_shape, in_channels=3, base_channels=16, out_channels=128,
                 layer_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
                 layer_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]]
                 ):
        super(SparseEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_channels = layer_channels
        self.layer_paddings = layer_paddings
        self.sparse_shape = sparse_shape
        self.base_channels = base_channels
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.in_channels, self.base_channels, 3, padding=1, bias=False, indice_key="subm1"),
            nn.BatchNorm1d(self.base_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.make_conv_layers()

    def make_conv_layers(self):
        self.conv_layers = nn.ModuleList()
        in_channels = self.in_channels
        for i, blocks in enumerate(self.layer_channels):
            block_list = nn.ModuleList()
            for j, out_channels in enumerate(tuple(blocks)):
                pad = tuple(self.layer_paddings[i])[j]
                if j == len(blocks) - 1 and i != len(self.layer_channels) - 1:
                    block_list.append(
                        spconv.SparseSequential(
                            spconv.SparseConv3d(
                                in_channels,
                                out_channels,
                                3,
                                padding=pad,
                                stride=(2, 2, 2),
                                indice_key=f"spconv{i + 1}",
                            ),
                            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    block_list.append(
                        SparseBasicBlock(out_channels, out_channels)
                    )
                in_channels = out_channels
            self.conv_layers.append(block_list)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                out_channels,
                self.out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 2),
                padding=(1, 1, 0),
                indice_key="spconv_down2",
                bias=False
            ),
            nn.BatchNorm1d(self.out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(
            features,
            coors,
            self.sparse_shape,
            batch_size
        )
        x = self.conv_input(x)

        for layer in self.conv_layers:
            for stage in layer:
                ### Could work now
                x = stage(x)

        x = self.conv_out(x)

        x = x.dense(False)  # B, X, Y, Z, C
        B, X, Y, Z, C = x.shape
        x = x.view(B, X, Y, Z * C).permute(0, 3, 1, 2).contiguous()
        return x