"""Camera-BEV Cross-Attention Fusion (v39.1架构重构)

目标：解决当前BEV分支坍塌问题
根因：BEV投影链路太长，信息损失严重，无法与Proj强预训练对抗

新方案：
1. 跳过BEV俯视投影
2. 保持Camera坐标系下的cross-attention
3. 与Proj形成真正互补：不同query视角的2D-3D关联

架构对比：
【原BEV】img → Query投影到BEV → 与pc_bev融合 → pool
【原Proj】img → ViT patches → cross-attn查询pc groups
【新Camera-BEV】img → DINOv2 patches（camera系）→ cross-attn查询pc groups → 不同于Proj的query方式
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraBEVCrossAttention(nn.Module):
    """Camera坐标系下的图像-点云cross-attention（与Proj形成互补）
    
    关键差异：
    - Proj：ResNet/ViT patches (224×448) + projection投影查询
    - Camera-BEV：DINOv2 patches (640×360) + spatial query（不同分辨率和视角）
    """
    
    def __init__(self,
                 img_feat_dim: int = 384,  # DINOv2-small输出
                 pc_feat_dim: int = 384,   # PointGPT group特征
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_feat_dim = img_feat_dim
        self.pc_feat_dim = pc_feat_dim
        self.hidden_dim = hidden_dim
        
        # 1. Feature projection（对齐到hidden_dim）
        self.img_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.pc_proj = nn.Linear(pc_feat_dim, hidden_dim)
        
        # 2. Positional encoding for image patches
        self.img_pos_embed = nn.Parameter(torch.randn(1, 45*80, hidden_dim) * 0.02)
        
        # 3. Cross-attention layers (img queries pc)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # 4. Output head (global pooling + projection)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self,
                img_feats: torch.Tensor,      # (B, H*W, C_img) DINOv2输出
                pc_groups_xyz: torch.Tensor,  # (B, G, 3) PointGPT group centers
                pc_groups_feat: torch.Tensor, # (B, G, C_pc) PointGPT features
                T_cam2lidar: torch.Tensor,    # (B, 4, 4) 用于点云投影到camera系
                cam_intrinsic: torch.Tensor,  # (B, 3, 3) 相机内参
                img_shape: tuple = (360, 640)) -> torch.Tensor:
        """
        Args:
            img_feats: DINOv2 patch embeddings (B, 45*80, 384)
            pc_groups_xyz: PointGPT group centers in lidar frame
            pc_groups_feat: PointGPT semantic features
            T_cam2lidar: Camera to Lidar transform (for projection)
            cam_intrinsic: Camera intrinsic matrix
            img_shape: (H, W) for spatial positional encoding
            
        Returns:
            f_camera_bev: Global feature vector (B, hidden_dim)
        """
        B = img_feats.shape[0]
        
        # 1. Project features to hidden_dim
        img_feat = self.img_proj(img_feats)  # (B, H*W, hidden_dim)
        pc_feat = self.pc_proj(pc_groups_feat)  # (B, G, hidden_dim)
        
        # 2. Add positional encoding to img
        img_feat = img_feat + self.img_pos_embed
        
        # 3. 投影点云到图像平面（几何约束的核心）
        pc_xyz_homo = torch.cat([
            pc_groups_xyz,
            torch.ones(B, pc_groups_xyz.shape[1], 1, device=pc_groups_xyz.device)
        ], dim=-1)  # (B, G, 4)
        
        pc_cam = torch.matmul(
            T_cam2lidar,  # (B, 4, 4)
            pc_xyz_homo.transpose(1, 2)  # (B, 4, G)
        ).transpose(1, 2)[:, :, :3]  # (B, G, 3)
        
        # 投影到2D图像坐标
        H, W = img_shape
        fx = cam_intrinsic[:, 0, 0]  # (B,)
        fy = cam_intrinsic[:, 1, 1]
        cx = cam_intrinsic[:, 0, 2]
        cy = cam_intrinsic[:, 1, 2]
        
        # 透视投影
        u = (pc_cam[:, :, 0] / (pc_cam[:, :, 2] + 1e-8)) * fx.unsqueeze(1) + cx.unsqueeze(1)  # (B, G)
        v = (pc_cam[:, :, 1] / (pc_cam[:, :, 2] + 1e-8)) * fy.unsqueeze(1) + cy.unsqueeze(1)
        
        # 归一化到[-1, 1]（用于spatial attention bias）
        u_norm = (u / W) * 2 - 1  # (B, G)
        v_norm = (v / H) * 2 - 1
        proj_uv = torch.stack([u_norm, v_norm], dim=-1)  # (B, G, 2)
        
        # 有效性mask（投影在图像内 + 深度>0）
        valid_mask = (
            (u_norm >= -1.2) & (u_norm <= 1.2) &
            (v_norm >= -1.2) & (v_norm <= 1.2) &
            (pc_cam[:, :, 2] > 0)
        )  # (B, G)
        
        # 为pc_feat添加位置编码（使用投影坐标）
        # 简化版：直接concat归一化的uv坐标
        proj_pos_embed = torch.cat([
            proj_uv,
            pc_cam[:, :, 2:3] / 60.0  # 归一化深度
        ], dim=-1)  # (B, G, 3)
        
        # MLP编码位置信息
        pos_proj_mlp = nn.Sequential(
            nn.Linear(3, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        ).to(pc_feat.device)
        
        pc_pos_feat = pos_proj_mlp(proj_pos_embed)  # (B, G, hidden_dim)
        pc_feat_with_pos = pc_feat + pc_pos_feat
        
        # 4. Geometry-aware Cross-attention
        x = img_feat
        for i in range(len(self.cross_attn_layers)):
            # ⭐ 关键改进：使用spatial attention bias
            # 计算img patches到pc groups的几何距离
            # img patches grid: (H'=45, W'=80)
            H_feat, W_feat = 45, 80
            patch_y, patch_x = torch.meshgrid(
                torch.linspace(-1, 1, H_feat, device=x.device),
                torch.linspace(-1, 1, W_feat, device=x.device),
                indexing='ij'
            )
            patch_uv = torch.stack([patch_x, patch_y], dim=-1).reshape(-1, 2)  # (H*W, 2)
            
            # 计算每个patch到每个pc投影点的距离
            # patch_uv: (H*W, 2), proj_uv: (B, G, 2)
            dist = torch.cdist(
                patch_uv.unsqueeze(0).expand(B, -1, -1),  # (B, H*W, 2)
                proj_uv  # (B, G, 2)
            )  # (B, H*W, G)
            
            # 几何衰减权重（距离越近，权重越高）
            spatial_bias = torch.exp(-dist * 2.0)  # (B, H*W, G)
            
            # 结合有效性mask
            attn_mask = valid_mask.unsqueeze(1).expand(-1, H_feat * W_feat, -1) * spatial_bias  # (B, H*W, G)
            attn_mask = torch.where(attn_mask > 0.1, attn_mask, torch.zeros_like(attn_mask))
            
            # Cross-attention with geometric bias
            attn_out, attn_weights = self.cross_attn_layers[i](
                query=x,                    # img patches作为query
                key=pc_feat_with_pos,      # pc groups with position
                value=pc_feat              # pc semantic features
            )
            
            # 应用几何权重（soft masking）
            attn_out = attn_out * (attn_mask.sum(dim=-1, keepdim=True) / (attn_mask.sum(dim=-1, keepdim=True).max() + 1e-8))
            
            x = self.layer_norms[i*2](x + attn_out)
            
            # FFN
            ffn_out = self.ffn_layers[i](x)
            x = self.layer_norms[i*2+1](x + ffn_out)
        
        # 5. Global pooling
        f_global = x.mean(dim=1)  # (B, hidden_dim) 平均池化所有patches
        
        # 6. Output projection
        f_camera_bev = self.out_proj(f_global)  # (B, hidden_dim)
        
        return f_camera_bev


class CameraBEVBranch(nn.Module):
    """Camera-BEV分支：替代原BEV分支的高层wrapper
    
    优势：
    1. 梯度路径短：DINOv2 → cross-attn → pool（vs 原BEV的5层链路）
    2. 保留几何：camera坐标系（vs BEV俯视投影损失）
    3. 互补Proj：不同分辨率(640×360 vs 224×448)和query方式
    4. 信息密集：所有patches参与（vs BEV的mask 30-40%）
    """
    
    def __init__(self,
                 img_encoder,  # 复用现有DINOv2Encoder
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_encoder = img_encoder  # DINOv2 frozen
        
        # Camera-BEV cross-attention
        self.camera_bev_attn = CameraBEVCrossAttention(
            img_feat_dim=img_encoder.out_channels,  # 384 for DINOv2-small
            pc_feat_dim=384,  # PointGPT output
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.out_dim = hidden_dim
    
    def forward(self,
                img: torch.Tensor,               # (B, C, H, W)
                pc_groups_xyz: torch.Tensor,     # (B, G, 3)
                pc_groups_feat: torch.Tensor,    # (B, G, D)
                T_cam2lidar: torch.Tensor,       # (B, 4, 4)
                cam_intrinsic: torch.Tensor):    # (B, 3, 3)
        """
        Args:
            img: Input image (B, 3, 360, 640)
            pc_groups_xyz: PointGPT group centers in lidar frame
            pc_groups_feat: PointGPT semantic features
            T_cam2lidar: Camera to Lidar extrinsic (inverse of T_init)
            cam_intrinsic: Camera intrinsic matrix
            
        Returns:
            f_camera_bev: Global feature (B, hidden_dim)
        """
        # 1. Extract image features with DINOv2
        img_feats = self.img_encoder.forward_features(img)  # (B, C, H', W')
        
        # Reshape to sequence for attention
        B, C, H, W = img_feats.shape
        img_feats = img_feats.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 2. Cross-attention with point cloud
        f_camera_bev = self.camera_bev_attn(
            img_feats=img_feats,
            pc_groups_xyz=pc_groups_xyz,
            pc_groups_feat=pc_groups_feat,
            T_cam2lidar=T_cam2lidar,
            cam_intrinsic=cam_intrinsic,
            img_shape=(H, W)
        )
        
        return f_camera_bev


def build_camera_bev_branch(img_encoder, **kwargs):
    """工厂函数：构建Camera-BEV分支
    
    用于在hybrid_triple_calib.py中替换原BEV分支
    """
    return CameraBEVBranch(
        img_encoder=img_encoder,
        hidden_dim=kwargs.get('hidden_dim', 256),
        num_heads=kwargs.get('num_heads', 8),
        num_layers=kwargs.get('num_layers', 2),
        dropout=kwargs.get('dropout', 0.1)
    )
