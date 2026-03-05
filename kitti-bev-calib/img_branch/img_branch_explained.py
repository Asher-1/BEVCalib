"""
LSS (Lift-Splat-Shoot) 图像分支详细注释版本
============================================

输入配置（基于您的设置）:
- 图像尺寸: 640×360 (W×H)
- BEV范围: X[0, 200m], Y[-100, 100m], Z[-10, 10m]
- BEV分辨率: 2m/格
- BEV网格: X=100格, Y=100格, Z=5格
- 深度采样: 1m到100m，每1m一个，共99层

核心思想：
1. 将2D图像"提升(Lift)"到3D空间
2. 在多个深度假设下，将像素投影到3D
3. 将3D点"投射(Splat)"到BEV网格
4. 用于下游任务(Shoot)

可视化工作流：
            
    2D图像 (640×360)
         ↓
    [图像编码器]
         ↓
    特征图 (80×45) ← 下采样8倍
         ↓
    [深度预测网络] ← 预测每个像素的深度分布
         ↓
    深度加权特征 (99深度层)
         ↓
    [视锥体变换] ← 图像空间→相机空间→ego空间
         ↓
    3D点云 (约35万个点)
         ↓
    [BEV池化] ← 将点投影到俯视图网格
         ↓
    BEV特征图 (100×100)
"""

import torch
import torch.nn as nn
try:
    from .bev_pool import bev_pool
    from .img_encoders import SwinT_tiny_Encoder
except:
    import bev_pool.bev_pool as bev_pool
    from img_encoders import SwinT_tiny_Encoder

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proj_head import ProjectionHead
from bev_settings import xbound, ybound, zbound, down_ratio, sparse_shape, vsize_xyz, d_conf


def gen_dx_bx(xbound, ybound, zbound):
    """
    生成BEV网格参数
    
    示例输入:
        xbound = (0, 200, 2.0)     # X轴: 0到200米，步长2米
        ybound = (-100, 100, 2.0)  # Y轴: -100到100米，步长2米
        zbound = (-10, 10, 4.0)    # Z轴: -10到10米，步长4米
    
    返回:
        dx: [2.0, 2.0, 4.0]        # 体素尺寸 (米)
        bx: [1.0, -99.0, -8.0]     # 网格起点（体素中心对齐）
        nx: [100, 100, 5]          # 网格数量
    
    可视化:
        X轴: |--2m--|--2m--|...|--2m--| (100格)
             ↑                      ↑
             0m                     200m
        
        Z轴: |---4m---|---4m---|...| (5格)
             ↑                   ↑
            -10m                10m
    """
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    # dx = [2.0, 2.0, 4.0] - 每个体素的物理尺寸
    
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    # bx = [0+1.0, -100+1.0, -10+2.0] = [1.0, -99.0, -8.0]
    # 为什么加半个体素？让体素中心对齐到整数倍位置
    
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    # nx = [(200-0)/2, (100-(-100))/2, (10-(-10))/4]
    #    = [100, 100, 5]
    
    return dx, bx, nx


class LSS(nn.Module):
    """
    Lift-Splat-Shoot 核心模块
    
    功能: 将2D图像特征提升到3D空间，通过深度预测
    """
    
    def __init__(self, 
                 transformedImgShape = None,  # (C, H, W) = (3, 360, 640)
                 featureShape = None,         # (C, fH, fW) = (256, 45, 80)
                 d_conf = d_conf,             # (1.0, 100.0, 1.0) 深度配置
                 out_channels = 128,          # 输出特征维度
                ):
        super(LSS, self).__init__()
        
        # ========== 尺寸配置 ==========
        if transformedImgShape is None:
            transformedImgShape = (3, 360, 640)  # 默认值
        if featureShape is None:
            # 特征图是原图下采样8倍: 360//8=45, 640//8=80
            featureShape = (256, transformedImgShape[1] // 8, transformedImgShape[2] // 8)
        
        _, self.orfH, self.orfW = transformedImgShape
        # orfH = 360 (原始特征高度)
        # orfW = 640 (原始特征宽度)
        
        self.fC, self.fH, self.fW = featureShape
        # fC = 256  (特征通道数)
        # fH = 45   (特征图高度: 360/8)
        # fW = 80   (特征图宽度: 640/8)
        
        # ========== 深度配置 ==========
        self.d_st, self.d_end, self.d_step = d_conf
        # d_st = 1.0    (起始深度: 1米)
        # d_end = 100.0 (结束深度: 100米)
        # d_step = 1.0  (深度步长: 每1米)
        
        self.D = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float).shape[0]
        # D = 99 (深度层数: 从1m到100m，共99层)
        # 深度列表: [1, 2, 3, ..., 99] 米
        
        self.out_channels = out_channels
        # out_channels = 128 (输出特征维度)
        
        # ========== 创建视锥体 ==========
        self.frustum = self.create_frustum()
        # frustum.shape = (99, 45, 80, 3)
        # 含义: 每个深度层(99) × 每个特征像素(45×80) × 3D坐标(x,y,d)
        
        # ========== 深度预测网络 ==========
        self.depth_net = nn.Sequential(
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            # 输入: (256, 45, 80) → 输出: (256, 45, 80)
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            # 输入: (256, 45, 80) → 输出: (256, 45, 80)
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            
            nn.Conv2d(self.fC, self.out_channels + self.D, 1),
            # 输入: (256, 45, 80) → 输出: (227, 45, 80)
            #                              ↑
            #                         128 + 99 = 227
            #                         特征  深度
        )
        # 输出通道解释:
        #   - 前99通道: 每个像素在99个深度层的概率分布
        #   - 后128通道: 上下文特征（语义信息）
    
    def create_frustum(self):
        """
        创建视锥体 - 图像空间的3D采样网格
        
        返回: frustum.shape = (D, fH, fW, 3) = (99, 45, 80, 3)
        
        可视化:
            图像平面 (640×360)
                |
                | 深度=1m
                |-----------> 每个像素在1m处的3D点 (80×45个点)
                | 深度=2m
                |-----------> 每个像素在2m处的3D点 (80×45个点)
                | ...
                | 深度=99m
                |-----------> 每个像素在99m处的3D点 (80×45个点)
        
        总共: 99 × 45 × 80 = 356,400 个3D采样点
        """
        # ========== 深度维度 ==========
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float)
        # ds.shape = (99,) 
        # ds = [1., 2., 3., ..., 99.]
        
        ds = ds.view(-1, 1, 1).expand(-1, self.fH, self.fW)
        # ds.shape = (99, 45, 80)
        # ds[0,:,:] = 1.0  (所有位置深度都是1米)
        # ds[1,:,:] = 2.0  (所有位置深度都是2米)
        # ...
        
        # ========== 水平坐标 (X轴) ==========
        xs = torch.linspace(0, self.orfW - 1, self.fW)
        # xs.shape = (80,)
        # xs = [0, 8.1, 16.2, ..., 639]
        # 解释: 在原图宽度640上均匀采样80个点
        #      每个特征像素对应原图的8个像素
        
        xs = xs.view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        # xs.shape = (99, 45, 80)
        # xs[d,h,:] = [0, 8.1, 16.2, ..., 639] (所有深度层和高度相同)
        
        # ========== 垂直坐标 (Y轴) ==========
        ys = torch.linspace(0, self.orfH - 1, self.fH)
        # ys.shape = (45,)
        # ys = [0, 8.2, 16.4, ..., 359]
        # 解释: 在原图高度360上均匀采样45个点
        
        ys = ys.view(1, self.fH, 1).expand(self.D, self.fH, self.fW)
        # ys.shape = (99, 45, 80)
        # ys[d,:,w] = [0, 8.2, ..., 359] (所有深度层和宽度相同)
        
        # ========== 组合成3D网格 ==========
        frustum = torch.stack((xs, ys, ds), -1)
        # frustum.shape = (99, 45, 80, 3)
        # frustum[d, h, w] = [x坐标, y坐标, 深度]
        
        # 示例: frustum[0, 0, 0] = [0, 0, 1]
        #      ↑ 深度1米，左上角像素(0,0)的3D位置
        #      frustum[98, 44, 79] = [639, 359, 99]
        #      ↑ 深度99米，右下角像素的3D位置
        
        return nn.Parameter(frustum, requires_grad = False)
        # 不需要梯度，这是固定的几何结构

    def get_geometry(self, 
                     cam2ego_rot,      # (B, N, 3, 3) 相机到ego的旋转
                     cam2ego_trans,    # (B, N, 3) 相机到ego的平移
                     cam_intrins,      # (B, N, 3, 3) 相机内参矩阵
                     post_cam2ego_rot, # (B, N, 3, 3) 数据增强后的旋转
                     post_cam2ego_trans, # (B, N, 3) 数据增强后的平移
                     ):
        """
        几何变换: 图像空间 → 相机空间 → ego空间
        
        输入示例 (B=2, N=1):
            cam2ego_rot: (2, 1, 3, 3)      相机朝向
            cam2ego_trans: (2, 1, 3)       相机位置
            cam_intrins: (2, 1, 3, 3)      焦距、主点
            post_cam2ego_rot: (2, 1, 3, 3) 图像旋转
            post_cam2ego_trans: (2, 1, 3)  图像平移
        
        返回:
            points: (2, 1, 99, 45, 80, 3) - ego空间的3D坐标
        
        坐标系变换链:
            增强图像空间 → 原始图像空间 → 相机空间 → ego空间
               [撤销增强]    [透视逆变换]   [外参变换]
        """
        img_rots, img_trans, cam_intrins, img_post_rots, img_post_trans = \
            cam2ego_rot, cam2ego_trans, cam_intrins, post_cam2ego_rot, post_cam2ego_trans
        
        B, N, _ = img_trans.shape
        # B = 2 (batch size)
        # N = 1 (相机数量，这里是单相机)
        
        # ===================================================================
        # 步骤1: 撤销数据增强 (增强图像空间 → 原始图像空间)
        # ===================================================================
        
        # -------- 1.1 撤销平移 --------
        points = self.frustum - img_post_trans.view(B, N, 1, 1, 1, 3)
        # self.frustum.shape = (99, 45, 80, 3)
        # img_post_trans.view = (2, 1, 1, 1, 1, 3)
        # points.shape = (2, 1, 99, 45, 80, 3)
        
        # 广播后: frustum自动扩展到 (2, 1, 99, 45, 80, 3)
        # 每个batch、每个相机都有独立的frustum副本
        
        # -------- 1.2 撤销旋转 --------
        points = torch.inverse(img_post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # inverse(img_post_rots).shape = (2, 1, 3, 3)
        # .view(2, 1, 1, 1, 1, 3, 3) - 扩展维度用于广播
        # points.unsqueeze(-1).shape = (2, 1, 99, 45, 80, 3, 1) - 变成列向量
        # 
        # 矩阵乘法: (3×3) @ (3×1) = (3×1)
        # 结果: points.shape = (2, 1, 99, 45, 80, 3, 1)
        
        # 现在 points 在原始图像空间（像素坐标）
        
        # ===================================================================
        # 步骤2: 透视逆变换 (图像空间 → 相机空间)
        # ===================================================================
        
        # 当前 points = (x_pixel, y_pixel, depth)
        # 目标: 转换为相机3D坐标 (X_cam, Y_cam, Z_cam)
        
        # 透视投影公式:
        #   x_pixel = fx * (X_cam / Z_cam) + cx
        #   y_pixel = fy * (Y_cam / Z_cam) + cy
        # 
        # 逆变换:
        #   X_cam = (x_pixel - cx) * Z_cam / fx
        #   Y_cam = (y_pixel - cy) * Z_cam / fy
        #   Z_cam = depth
        
        # 简化: X_cam = x_pixel * Z_cam (相机内参后面会处理)
        
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                # 前两维(x,y) 乘以 深度(z)
                # (2,1,99,45,80,2,1) × (2,1,99,45,80,1,1) = (2,1,99,45,80,2,1)
                # 结果: (x*z, y*z)
                
                points[:, :, :, :, :, 2:3]
                # 保持深度不变
                # 结果: (z)
            ),
            5,  # 在倒数第二个维度拼接
        )
        # points.shape = (2, 1, 99, 45, 80, 3, 1)
        # points = (x*z, y*z, z)
        
        # 可视化理解:
        #   原始: (像素x=320, 像素y=180, 深度z=10m)
        #   变换: (320*10, 180*10, 10) = (3200, 1800, 10)
        #   
        #   为什么这样？因为相机投影是除以深度，逆过程就是乘以深度
        #   后面会用相机内参矩阵的逆来恢复真实的米制坐标
        
        # ===================================================================
        # 步骤3: 应用相机内参和外参 (相机空间 → ego空间)
        # ===================================================================
        
        # -------- 3.1 组合变换矩阵 --------
        combine = img_rots.matmul(torch.inverse(cam_intrins))
        # img_rots.shape = (2, 1, 3, 3)      - 相机到ego的旋转
        # inverse(cam_intrins).shape = (2, 1, 3, 3) - 相机内参的逆
        # combine.shape = (2, 1, 3, 3)
        
        # 相机内参矩阵 K:
        #   K = [fx  0  cx]
        #       [ 0 fy  cy]
        #       [ 0  0   1]
        # 
        # K^{-1} 的作用: 将 (x*z, y*z, z) → (X_cam, Y_cam, Z_cam)
        # 然后 img_rots 将相机坐标旋转到ego坐标系
        
        # -------- 3.2 应用旋转变换 --------
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # combine.view = (2, 1, 1, 1, 1, 3, 3)
        # points = (2, 1, 99, 45, 80, 3, 1)
        # 矩阵乘法: (3×3) @ (3×1) = (3×1)
        # squeeze(-1) 移除最后一维
        # points.shape = (2, 1, 99, 45, 80, 3)
        
        # -------- 3.3 添加平移 --------
        points += img_trans.view(B, N, 1, 1, 1, 3)
        # img_trans.shape = (2, 1, 3)
        # .view(2, 1, 1, 1, 1, 3) - 广播到所有点
        # points.shape = (2, 1, 99, 45, 80, 3)
        
        # ===================================================================
        # 最终结果
        # ===================================================================
        # points[b, n, d, h, w] = (X_ego, Y_ego, Z_ego)
        # 
        # 含义: 
        #   - batch b 的样本
        #   - 相机 n
        #   - 在深度假设 d 米时
        #   - 特征图位置 (h, w) 对应的像素
        #   - 在ego坐标系(激光雷达中心)的3D坐标 (X, Y, Z)
        
        return points

    def get_cam_feature(self, img_feats):
        """
        深度预测和特征加权
        
        输入:
            img_feats: (B, N, C, H, W) = (2, 1, 256, 45, 80)
        
        输出:
            img_feats: (B, N, D, H, W, C) = (2, 1, 99, 45, 80, 128)
        
        核心思想:
            1. 为每个像素预测99个深度的概率分布
            2. 用深度概率加权上下文特征
            3. 每个像素变成99个带特征的3D点
        
        可视化:
                                预测深度分布
            像素(h,w) ────────→ [p₁, p₂, ..., p₉₉]
               ↓                    ↓
            特征向量f          加权: f×p₁, f×p₂, ..., f×p₉₉
               ↓                    ↓
            1个2D点           99个3D点（不同深度）
        """
        B, N, C, H, W = img_feats.shape
        # B=2, N=1, C=256, H=45, W=80
        
        # ========== 重塑为批处理 ==========
        img_feats = img_feats.view(B * N, C, H, W)
        # img_feats.shape = (2, 256, 45, 80)
        # 将batch和相机维度合并，方便卷积处理
        
        # ========== 深度预测网络 ==========
        img_feats = self.depth_net(img_feats)
        # 输入: (2, 256, 45, 80)
        # 输出: (2, 227, 45, 80)
        #           ↑
        #        99 + 128
        #       深度 + 特征
        
        # ========== 提取深度概率 ==========
        depth = img_feats[:, :self.D].softmax(dim = 1)
        # 取前99个通道
        # depth.shape = (2, 99, 45, 80)
        # softmax确保每个像素在99个深度上的概率和为1
        
        # depth[b, d, h, w] = 像素(h,w)在深度d的概率
        # 
        # 示例: depth[0, :, 20, 40] = [0.001, 0.002, ..., 0.15, ..., 0.001]
        #       表示像素(20,40)最可能在深度15米处（概率0.15最大）
        
        # ========== 深度加权特征 ==========
        img_feats = depth.unsqueeze(1) * img_feats[:, self.D : self.D + self.out_channels].unsqueeze(2)
        
        # 拆解这个操作:
        # 1. depth.unsqueeze(1)
        #    (2, 99, 45, 80) → (2, 1, 99, 45, 80)
        #    添加通道维度
        
        # 2. img_feats[:, 99:227]
        #    取后128个通道（上下文特征）
        #    (2, 128, 45, 80)
        
        # 3. .unsqueeze(2)
        #    (2, 128, 45, 80) → (2, 128, 1, 45, 80)
        #    添加深度维度
        
        # 4. 广播乘法:
        #    (2, 1, 99, 45, 80) × (2, 128, 1, 45, 80)
        #    ↓
        #    (2, 128, 99, 45, 80)
        
        # 物理意义:
        #   对于每个像素(h,w)的特征向量f[128维]:
        #   - 在深度1米: f × p₁
        #   - 在深度2米: f × p₂
        #   - ...
        #   - 在深度99米: f × p₉₉
        #   
        #   深度不确定的像素（概率分布平坦）→ 特征分散到多个深度
        #   深度确定的像素（概率集中）→ 特征集中在某个深度
        
        # ========== 恢复维度 ==========
        img_feats = img_feats.view(B, N, self.out_channels, self.D, H, W)
        # img_feats.shape = (2, 1, 128, 99, 45, 80)
        
        img_feats = img_feats.permute(0, 1, 3, 4, 5, 2)
        # 调整维度顺序: (B, N, C, D, H, W) → (B, N, D, H, W, C)
        # img_feats.shape = (2, 1, 99, 45, 80, 128)
        
        # 最终形状解释:
        #   (2, 1, 99, 45, 80, 128)
        #    ↑  ↑  ↑   ↑   ↑   ↑
        #    B  N  深  特  特  特
        #          度  征  征  征
        #          层  图  图  维
        #             高   宽   度
        
        return img_feats

    def forward(self, 
                cam2ego_rot,        # (B, N, 3, 3)
                cam2ego_trans,      # (B, N, 3)
                cam_intrins,        # (B, N, 3, 3)
                post_cam2ego_rot,   # (B, N, 3, 3)
                post_cam2ego_trans, # (B, N, 3)
                img_feats           # (B, N, C, fH, fW) = (2, 1, 256, 45, 80)
                ):
        """
        LSS前向传播
        
        输入:
            img_feats: (2, 1, 256, 45, 80) - 图像特征
            外参和内参矩阵
        
        输出:
            geometry: (2, 1, 99, 45, 80, 3) - 3D坐标（ego空间）
            img_depth_feature: (2, 1, 99, 45, 80, 128) - 深度加权特征
        
        数据统计:
            总点数: 2 × 1 × 99 × 45 × 80 = 712,800 个3D点
            每个点: 3维坐标 + 128维特征
        """
        # 计算几何变换
        geometry = self.get_geometry(
            cam2ego_rot=cam2ego_rot, 
            cam2ego_trans=cam2ego_trans, 
            cam_intrins=cam_intrins, 
            post_cam2ego_rot=post_cam2ego_rot, 
            post_cam2ego_trans=post_cam2ego_trans
        )
        # geometry.shape = (2, 1, 99, 45, 80, 3)
        
        # 计算深度加权特征
        img_depth_feature = self.get_cam_feature(img_feats)
        # img_depth_feature.shape = (2, 1, 99, 45, 80, 128)
        
        return geometry, img_depth_feature


class Cam2BEV(nn.Module):
    """
    完整的相机到BEV转换模块
    
    流程:
        图像 → 编码器 → LSS → BEV池化 → BEV特征图
    
    最终输出:
        BEV特征图: (B, 128, 100, 100) - 俯视图特征
        BEV掩码: (B, 100, 100) - 标记有图像覆盖的区域
    """
    
    def __init__(self, 
                 output_indices = [1, 2, 3],      # SwinT的输出层
                 img_shape = None,                # (H, W) = (360, 640)
                 encoder_out_channels = 256,      # 编码器输出通道
                 FPN_in_channels = [192, 384, 768],  # FPN输入通道
                 FPN_out_channels = 256,          # FPN输出通道
                ):
        super(Cam2BEV, self).__init__()
        
        # ========== 图像尺寸配置 ==========
        if img_shape is None:
            img_shape = (360, 640)  # (H, W)
        
        img_H, img_W = img_shape
        # img_H = 360 (图像高度)
        # img_W = 640 (图像宽度)
        
        transformedImgShape = (3, img_H, img_W)
        # transformedImgShape = (3, 360, 640)
        
        featureShape = (encoder_out_channels, img_H // 8, img_W // 8)
        # featureShape = (256, 45, 80)
        # 特征图下采样8倍: 360/8=45, 640/8=80
        
        print(f"[Cam2BEV] 输入图像尺寸: {img_W}x{img_H}, 特征尺寸: {featureShape[2]}x{featureShape[1]}")
        # 输出: [Cam2BEV] 输入图像尺寸: 640x360, 特征尺寸: 80x45
        
        # ========== 核心模块初始化 ==========
        
        # LSS模块
        self.lss = LSS(
            transformedImgShape=transformedImgShape,  # (3, 360, 640)
            featureShape=featureShape                 # (256, 45, 80)
        )
        
        # 图像编码器（Swin Transformer）
        self.CamEncode = SwinT_tiny_Encoder(
            output_indices, 
            featureShape, 
            encoder_out_channels, 
            FPN_in_channels, 
            FPN_out_channels
        )
        # 输入: (B, N, 3, 360, 640)
        # 输出: (B, N, 256, 45, 80)
        
        # ========== BEV网格参数 ==========
        # 从配置文件读取:
        #   xbound = (0, 200, 2.0)    → 100格
        #   ybound = (-100, 100, 2.0) → 100格
        #   zbound = (-10, 10, 4.0)   → 5格
        
        dx, bx, nx = gen_dx_bx(xbound=xbound, ybound=ybound, zbound=zbound)
        # dx = [2.0, 2.0, 4.0]    - 体素尺寸（米）
        # bx = [1.0, -99.0, -8.0] - 网格起点
        # nx = [100, 100, 5]      - 网格数量
        
        self.dx = nn.Parameter(dx, requires_grad = False)
        self.bx = nn.Parameter(bx, requires_grad = False)
        self.nx = nn.Parameter(nx, requires_grad = False)
        
        # ========== 投影头配置 ==========
        # 
        # BEV池化后的维度变化:
        #   池化输出: (B, 128, 5, 100, 100)
        #                    ↑   ↑    ↑
        #                  特征  Z  X,Y
        #   
        #   压缩Z维度: cat(unbind(dim=2))
        #   → (B, 128×5, 100, 100) = (B, 640, 100, 100)
        #   
        # 问题: 下游网络期望128维，但现在是640维
        # 解决: 用ProjectionHead压缩回128维
        
        nz = nx[2].item()
        # nz = 5 (Z方向的体素数量)
        
        proj_embedding_dim = self.lss.out_channels * nz
        # proj_embedding_dim = 128 × 5 = 640
        
        self.proj_head = ProjectionHead(embedding_dim=proj_embedding_dim)
        # ProjectionHead: 640维 → 128维
        # 使用MLP将多层BEV特征压缩到固定维度
        
        self.out_channels = self.proj_head.projection_dim
        # out_channels = 128 (最终输出维度)
        
        # ========== 图像归一化参数 ==========
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        # ImageNet均值: (B, N, 3, 1, 1)
        
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        # ImageNet标准差: (B, N, 3, 1, 1)
        
        print(f"cam bev resolution: {nx[0].item()} x {nx[1].item()} x {nx[2].item()}")
        # 输出: cam bev resolution: 100 x 100 x 5
        
        print(f"[Cam2BEV] Z体素数: {nz}, ProjectionHead输入维度: {proj_embedding_dim} → {self.out_channels}")
        # 输出: [Cam2BEV] Z体素数: 5, ProjectionHead输入维度: 640 → 128

    def bev_pool(self, img_depth_feature, geometry):
        """
        BEV池化 - 将3D点云投影到俯视图网格
        
        输入:
            img_depth_feature: (B, N, D, H, W, C) = (2, 1, 99, 45, 80, 128)
            geometry: (B, N, D, H, W, 3) = (2, 1, 99, 45, 80, 3)
        
        输出:
            cam_bev: (B, C*nZ, nX, nY) = (2, 640, 100, 100)
        
        可视化:
            3D点云（ego空间）              BEV网格（俯视图）
            
                 Z ↑                         Y
                   |                         ↑
                   |  • • •                  |
                   | • • • •             [100x100]
                   |• • • • •                |
                   +------→ X                +-----→ X
                   
            每个点投影到对应的网格格子
            同一格子的点特征求和/平均
        """
        img_pc = img_depth_feature
        geom_feats = geometry
        # 简化变量名
        
        B, N, D, H, W, C = img_pc.shape
        # B=2, N=1, D=99, H=45, W=80, C=128
        
        # ========== 展平所有点 ==========
        Nprime = B * N * D * H * W
        # Nprime = 2 × 1 × 99 × 45 × 80 = 712,800
        # 总共712,800个3D点
        
        img_pc = img_pc.reshape(Nprime, C)
        # img_pc.shape = (712800, 128)
        # 每个点的128维特征
        
        # ========== 量化3D坐标到体素网格 ==========
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        # 
        # 公式: voxel_idx = (position - grid_origin) / voxel_size
        # 
        # 示例计算（X方向）:
        #   position = 50.0 米
        #   grid_origin = bx - dx/2 = 1.0 - 1.0 = 0.0
        #   voxel_size = dx = 2.0
        #   voxel_idx = (50.0 - 0.0) / 2.0 = 25
        #   
        #   含义: 50米的位置对应第25个格子
        # 
        # geom_feats.shape = (2, 1, 99, 45, 80, 3)
        # geom_feats[..., 0] ∈ [0, 99]  (X方向: 0-100格)
        # geom_feats[..., 1] ∈ [0, 99]  (Y方向: 0-100格)
        # geom_feats[..., 2] ∈ [0, 4]   (Z方向: 0-5格)
        
        geom_feats = geom_feats.view(Nprime, 3)
        # geom_feats.shape = (712800, 3)
        # 每个点的体素索引 (x_idx, y_idx, z_idx)
        
        # ========== 添加batch索引 ==========
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=img_pc.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        # batch_ix.shape = (712800, 1)
        # batch_ix = [0, 0, ..., 0, 1, 1, ..., 1]
        #            ↑ 前356400个  ↑ 后356400个
        #            batch 0      batch 1
        
        geom_feats = torch.cat([geom_feats, batch_ix], 1)
        # geom_feats.shape = (712800, 4)
        # 每个点: (x_idx, y_idx, z_idx, batch_idx)
        
        # ========== 过滤边界外的点 ==========
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])  # X ∈ [0, 100)
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])  # Y ∈ [0, 100)
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])  # Z ∈ [0, 5)
        )
        # kept.shape = (712800,) - bool数组
        # kept = [True, False, True, ...]
        
        img_pc = img_pc[kept]
        geom_feats = geom_feats[kept]
        # 假设保留约50%的点（视野内）
        # img_pc.shape ≈ (356400, 128)
        # geom_feats.shape ≈ (356400, 4)
        
        # ========== BEV池化操作 ==========
        try:
            cam_bev = bev_pool(img_pc, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
            # 
            # bev_pool是CUDA加速的自定义操作
            # 功能: 将点特征聚合到体素网格
            # 
            # 伪代码:
            #   for each point (x_idx, y_idx, z_idx, b_idx):
            #       cam_bev[b_idx, :, z_idx, x_idx, y_idx] += feature
            # 
            # 输出: cam_bev.shape = (2, 128, 5, 100, 100)
            #                        ↑   ↑   ↑  ↑    ↑
            #                        B   C   Z  X    Y
            
        except:
            # 如果池化失败（如所有点都在边界外），返回全0
            cam_bev = torch.zeros(
                B, C, 
                self.nx[2].item(), 
                self.nx[0].item(), 
                self.nx[1].item(), 
                device = img_pc.device, 
                dtype = img_pc.dtype
            )
            # cam_bev.shape = (2, 128, 5, 100, 100)
        
        # ========== 压缩Z维度 ==========
        cam_bev = torch.cat(cam_bev.unbind(dim = 2), 1)
        # 
        # unbind(dim=2) 将Z维度分解:
        #   (2, 128, 5, 100, 100) → 5个 (2, 128, 100, 100)
        # 
        # cat(..., dim=1) 沿通道维度拼接:
        #   → (2, 128*5, 100, 100) = (2, 640, 100, 100)
        # 
        # 可视化:
        #   层0 [128通道]  |  层1 [128通道]  | ... | 层4 [128通道]
        #   高度-10到-6m   |  高度-6到-2m    | ... | 高度6到10m
        #   ─────────────────────────────────────────────────
        #   拼接成640通道，每组128通道对应一个高度层
        
        return cam_bev
        # cam_bev.shape = (2, 640, 100, 100)
    
    def ref_bev_pool(self, img_depth_feature, geometry):
        """
        参考BEV池化 - 用于生成掩码
        
        与bev_pool相同，但:
        1. 输入特征全为1（不计算梯度）
        2. 输出用于判断哪些BEV格子有图像覆盖
        
        返回:
            binary_bev: (B, 100, 100) - bool掩码
        """
        with torch.no_grad():  # 不需要梯度
            img_pc = img_depth_feature
            geom_feats = geometry
            B, N, D, H, W, C = img_pc.shape
            Nprime = B * N * D * H * W
            img_pc = img_pc.reshape(Nprime, C)
            
            # 量化和过滤（与bev_pool相同）
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat(
                [
                    torch.full([Nprime // B, 1], ix, device=img_pc.device, dtype=torch.long)
                    for ix in range(B)
                ]
            )   
            geom_feats = torch.cat([geom_feats, batch_ix], 1)
            
            kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2])
            )
            img_pc = img_pc[kept]
            geom_feats = geom_feats[kept]
            
            try:
                cam_bev = bev_pool(img_pc, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
                # cam_bev.shape = (2, 128, 5, 100, 100)
            except:
                cam_bev = torch.zeros(B, C, self.nx[2].item(), self.nx[0].item(), self.nx[1].item(), device = img_pc.device, dtype = img_pc.dtype)
            
            # ========== 提取掩码 ==========
            binary_bev = torch.max(cam_bev, dim=2)[0]
            # 取Z维度的最大值: (2, 128, 5, 100, 100) → (2, 128, 100, 100)
            # 只要任何一个Z层有值，该位置就认为有覆盖
            
            binary_bev = (binary_bev != 0).float()
            # 转换为二值掩码: 非零→1, 零→0
            # binary_bev.shape = (2, 128, 100, 100)
            
        return binary_bev

    def forward(self, 
                cam2ego_T,      # (B, N, 4, 4) = (2, 1, 4, 4)
                cam_intrins,    # (B, N, 3, 3) = (2, 1, 3, 3)
                post_cam2ego_T, # (B, N, 4, 4) = (2, 1, 4, 4)
                imgs            # (B, N, 3, H, W) = (2, 1, 3, 360, 640)
                ):
        """
        完整的前向传播
        
        输入:
            imgs: (2, 1, 3, 360, 640) - 原始图像
            外参和内参矩阵
        
        输出:
            bev_feats: (2, 128, 100, 100) - BEV特征图
            cam_bev_mask: (2, 100, 100) - 有效区域掩码
        
        完整数据流:
            (2,1,3,360,640) 图像
                ↓ [归一化]
            (2,1,3,360,640)
                ↓ [SwinT编码器]
            (2,1,256,45,80) 特征图
                ↓ [LSS]
            (2,1,99,45,80,3) 几何 + (2,1,99,45,80,128) 特征
                ↓ [BEV池化]
            (2,640,100,100) BEV
                ↓ [投影头]
            (2,128,100,100) 最终BEV特征
        """
        # ========== 分解变换矩阵 ==========
        cam2ego_rot = cam2ego_T[:, :, :3, :3]
        # cam2ego_rot.shape = (2, 1, 3, 3)
        # 相机到ego的旋转矩阵
        
        cam2ego_trans = cam2ego_T[:, :, :3, 3]
        # cam2ego_trans.shape = (2, 1, 3)
        # 相机到ego的平移向量
        
        post_cam2ego_rot = post_cam2ego_T[:, :, :3, :3]
        post_cam2ego_trans = post_cam2ego_T[:, :, :3, 3]
        # 数据增强后的旋转和平移
        
        # ========== 图像归一化 ==========
        imgs = (imgs - self.mean) / self.std
        # imgs.shape = (2, 1, 3, 360, 640)
        # 标准化到ImageNet分布: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        # ========== 图像编码 ==========
        img_feats = self.CamEncode(imgs)
        # 输入: (2, 1, 3, 360, 640)
        # 输出: (2, 1, 256, 45, 80)
        # SwinT提取多尺度特征，下采样8倍
        
        # ========== LSS处理 ==========
        geometry, img_depth_feature = self.lss(
            cam2ego_rot=cam2ego_rot, 
            cam2ego_trans=cam2ego_trans, 
            cam_intrins=cam_intrins, 
            post_cam2ego_rot=post_cam2ego_rot, 
            post_cam2ego_trans=post_cam2ego_trans, 
            img_feats=img_feats
        )
        # geometry.shape = (2, 1, 99, 45, 80, 3)
        #   每个点的3D坐标（ego空间）
        # img_depth_feature.shape = (2, 1, 99, 45, 80, 128)
        #   每个点的深度加权特征
        
        # ========== BEV池化 ==========
        bev_feats = self.bev_pool(geometry=geometry, img_depth_feature=img_depth_feature)
        # bev_feats.shape = (2, 640, 100, 100)
        # 3D点云聚合到2D BEV网格
        
        # ========== 生成掩码 ==========
        with torch.no_grad():
            geom_detach = geometry.detach()
            # 分离梯度，掩码不参与训练
            
            ones_feat = torch.ones_like(img_depth_feature).to(img_depth_feature.device).detach()
            # 全1特征: (2, 1, 99, 45, 80, 128)
            
            ref_feats = self.ref_bev_pool(geometry=geom_detach, img_depth_feature=ones_feat)
            # ref_feats.shape = (2, 128, 100, 100)
            # 值为1的位置表示有图像覆盖
        
        B, C, H, W = bev_feats.shape
        # B=2, C=640, H=100, W=100
        
        cam_bev_mask = ref_feats != 0
        # cam_bev_mask.shape = (2, 128, 100, 100) - bool类型
        
        ref_mask = cam_bev_mask[:, 0:1, :, :]
        # 取第一个通道的掩码
        # ref_mask.shape = (2, 1, 100, 100)
        
        try:
            assert torch.all(cam_bev_mask == ref_mask)
            # 验证所有通道的掩码都相同（应该是这样）
        except:
            # 如果不同，保存数据用于调试
            print(f"ERROR: cam_bev_mask != ref_mask")
            torch.save(cam2ego_T, "cam2ego_T.pt")
            torch.save(cam_intrins, "cam_intrins.pt")
            torch.save(post_cam2ego_T, "post_cam2ego_T.pt")
            torch.save(imgs, "imgs.pt")
        
        # ========== 投影头处理 ==========
        bev_feats = bev_feats.permute(0, 2, 3, 1).reshape(B*H*W, C)
        # (2, 640, 100, 100) → (2, 100, 100, 640) → (20000, 640)
        # 将所有BEV格子展平，每个格子640维特征
        
        bev_feats = self.proj_head(bev_feats)
        # (20000, 640) → (20000, 128)
        # 投影头将640维压缩到128维
        # 
        # 作用: 
        #   - 整合5个Z层的信息
        #   - 降低维度，减少计算量
        #   - 保持下游网络兼容性
        
        bev_feats = bev_feats.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # (20000, 128) → (2, 100, 100, 128) → (2, 128, 100, 100)
        # 恢复BEV网格形状
        
        # ========== 返回结果 ==========
        return bev_feats, cam_bev_mask[:, 0, :, :]
        # bev_feats.shape = (2, 128, 100, 100) - BEV特征图
        # cam_bev_mask.shape = (2, 100, 100) - 有效区域掩码


# ============================================================
# 可视化总结
# ============================================================
"""
完整数据流（具体尺寸）:

1. 输入图像
   (2, 1, 3, 360, 640)
   ↓ 标准化
   
2. 图像编码（SwinT）
   (2, 1, 3, 360, 640) → (2, 1, 256, 45, 80)
   下采样8倍: 360/8=45, 640/8=80
   
3. 视锥体构建
   Frustum: (99, 45, 80, 3)
   - 99个深度层（1m到99m）
   - 45×80个特征图位置
   - 每个位置的(x, y, depth)坐标
   
4. 深度预测
   (2, 1, 256, 45, 80) → 深度网络 → (2, 227, 45, 80)
   分解: 99通道深度 + 128通道特征
   
5. 深度加权
   深度(2,1,99,45,80) × 特征(2,1,128,45,80)
   = (2, 1, 99, 45, 80, 128)
   总点数: 2×1×99×45×80 = 712,800个3D点
   
6. 几何变换
   (2, 1, 99, 45, 80, 3)
   图像空间 → 相机空间 → ego空间
   
7. BEV池化
   712,800个点 → 过滤 → 约356,400个有效点
   → 聚合到 (2, 128, 5, 100, 100)
   → 压缩Z → (2, 640, 100, 100)
   
8. 投影头
   (2, 640, 100, 100) → (2, 128, 100, 100)
   640维（5层×128）→ 128维
   
9. 输出
   BEV特征: (2, 128, 100, 100)
   覆盖范围: X[0,200m], Y[-100,100m]
   分辨率: 2m/格
   掩码: (2, 100, 100) - 标记有图像覆盖的区域

关键参数总结:
- 图像尺寸: 640×360
- 特征尺寸: 80×45 (下采样8倍)
- 深度层数: 99 (1m到100m)
- BEV网格: 100×100×5 (X×Y×Z)
- BEV范围: [0,200]×[-100,100]×[-10,10] 米
- BEV分辨率: 2m/格
- 总3D点数: ~71万个
- 有效点数: ~36万个（50%在视野内）
"""
