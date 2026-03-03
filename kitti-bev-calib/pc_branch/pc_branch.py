import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
from .pc_encoders import SparseEncoder

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proj_head import ProjectionHead
from bev_settings import xbound, ybound, zbound, down_ratio, sparse_shape, vsize_xyz


class Lidar2BEV(nn.Module):
    def __init__(self):
        super(Lidar2BEV, self).__init__()
        self.ptvoxel = PointToVoxel(
            vsize_xyz = vsize_xyz,
            coors_range_xyz=(xbound[0], ybound[0], zbound[0], xbound[1], ybound[1], zbound[1]),
            num_point_features=3,
            max_num_voxels=120000,
            max_num_points_per_voxel=10,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.sparse_encoder = SparseEncoder(sparse_shape=sparse_shape)
        self.proj_head = ProjectionHead()
        self.voxelize_reduce = True
        self.out_channels = self.proj_head.projection_dim

    @torch.no_grad()
    def voxelize(self, pc):
        B, _, _ = pc.shape
        vox_list, coors_list, num_points_list = [], [], []
        for i in range(B):
            vox_, coors_, num_points_ = self.ptvoxel(pc[i])
            if self.voxelize_reduce:
                vox_ = vox_.sum(dim=1)
            batch_id = torch.full([vox_.shape[0], 1], i, dtype=torch.int32, device=vox_.device)
            coors_list.append(torch.cat([batch_id, coors_], dim=1))
            vox_list.append(vox_)
            num_points_list.append(num_points_)
        return torch.cat(vox_list, 0), torch.cat(coors_list, 0), torch.cat(num_points_list, 0)

    def forward(self, pc):
        """
        Args:
            pc: (B, C, N)
        Returns:
            bev feats: (B, C, H, W)
        """
        B, C, N = pc.shape
        pc = pc.permute(0, 2, 1).contiguous()
        vox, coors, num_points = self.voxelize(pc)

        z_part, y_part, x_part, c_part = vox[:, 0:1], vox[:, 1:2], vox[:, 2:3], vox[:, 3:C]
        vox = torch.cat([c_part, x_part, y_part, z_part], dim=1)
        coors = coors[:, [0, 3, 2, 1]] # zyx to xyz
        out = self.sparse_encoder(vox, coors, B)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B*H*W, C)
        out = self.proj_head(out)
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return out