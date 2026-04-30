# BEVCalib 泛化性能改进实施计划

**基于**: 诊断报告 + DST-Calib 论文调研
**目标**: 突破当前 0.65° 架构地板，向 < 0.3° RPY 推进
**日期**: 2026-04-25

---

## 当前架构数据流

```
img (B,3,H,W) → Cam2BEV(SwinT+LSS) → cam_bev_feats (B,128,96,96)
                                                                    ↘
                                                      ConvFuser(1x1, 256→256) → + pose_embed
                                                                    ↗           ↓
pc  (B,N,3)   → Lidar2BEV(spconv)    → pc_bev_feats  (B,128,96,96)   DeformableTransformer(×2)
                                                                                ↓
                                                                        masked_avg_pool → head_drop
                                                                                ↓
                                                                    rotation_pred(4) / translation_pred(3)
```

**诊断问题**: `cam_bev_feats` 跨场景完全一致（变化 <0.04%），实为固定 FOV 模板，不编码标定信号。

---

## Phase 1: BEV Difference Map (低成本快速验证)

**改造成本**: 低 (修改 ~50 行代码，新增 1 个类)
**预期收益**: 中-高（强制跨模态关联学习）
**需要重新训练**: 是

### 1.1 核心思路

将 `ConvFuser(cat([cam, pc]))` 替换为 `DiffFuser(pc, cam-pc_diff)`，让网络直接从 BEV 空间的模态差异中学习标定信号。

### 1.2 Difference Map 构建

```python
class BEVDiffFuser(nn.Module):
    """在 BEV 空间构建 Difference Map 替代简单 concat。
    
    输入: cam_bev_feats (B, C_cam, H, W), pc_bev_feats (B, C_pc, H, W)
    输出: fused (B, C_out, H, W)
    
    Difference Map channels:
      - pc_bev_feats (128ch): LiDAR BEV 特征 (保留几何信息)
      - |cam - pc|    (128ch): 绝对差异 (标定误差信号)
      - cam * pc      (128ch): 交互项 (对齐区域增强)
    共 384ch → 1x1 Conv → 256ch
    """
    def __init__(self, cam_ch, pc_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cam_ch + pc_ch + pc_ch, out_ch, 1),  # 384 → 256
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    
    def forward(self, cam_bev_feat, pc_bev_feat):
        diff = (cam_bev_feat - pc_bev_feat).abs()  # 标定误差信号
        interact = cam_bev_feat * pc_bev_feat       # 对齐增强
        x = torch.cat([pc_bev_feat, diff, interact], dim=1)
        return self.conv(x)
```

### 1.3 修改位置

`bev_calib.py` 中:
1. 在 `__init__` 中将 `self.conv_fuser = ConvFuser(...)` 替换为 `self.conv_fuser = BEVDiffFuser(...)`
2. `embed_dim` 保持不变 (256)，下游 Transformer 和 head 不需要任何修改
3. `forward()` 调用保持一致: `x = self.conv_fuser(cam_bev_feats, pc_bev_feats)`

### 1.4 训练配置

```yaml
# 基于当前最佳模型 v20-v8recipe-pitch-wt3 的配置
# 仅修改 fuser 类型
fuser_type: "diff"   # "concat" (当前) | "diff" (新)
# 其他超参保持不变
angle_range: 5.0
z_voxels: 5
use_drcv: 0
voxel_mode: scatter
scatter_reduce: mean
axis_weights: [1, 3, 1]
```

### 1.5 验证标准

- 训练收敛速度是否与 baseline 相当
- Camera bypass 实验: 去掉 Camera 后是否比 baseline 退化更多（证明 Camera 贡献增加）
- test_data_v2 泛化误差 < 0.60° (比 0.646° 提升 >7%)

---

## Phase 2: Camera Branch Dropout (低成本正则化)

**改造成本**: 极低 (~10 行代码)
**预期收益**: 中（迫使 Transformer 学习跨模态注意力）
**可与 Phase 1 并行**

### 2.1 核心思路

训练时以概率 p 将 `cam_bev_feats` 全部置零，迫使模型不能 100% 依赖 Camera，同时在 Camera 可用时学习利用它。

### 2.2 实现

```python
# 在 BEVCalib.forward() 中，conv_fuser 调用前添加:
if self.training and hasattr(self, 'cam_drop_prob') and self.cam_drop_prob > 0:
    if torch.rand(1).item() < self.cam_drop_prob:
        cam_bev_feats = torch.zeros_like(cam_bev_feats)
```

### 2.3 建议超参

- `cam_drop_prob = 0.3` (30% 概率屏蔽 Camera)
- 变体: `cam_drop_prob = 0.5` (50%)

### 2.4 验证标准

- Camera bypass 实验的退化应更显著（证明 Camera 在训练时被有效利用）
- 泛化误差 < 0.64°

---

## Phase 3: 双边 BEV 增强 (中等改造)

**改造成本**: 中-高 (新增数据增强 pipeline)
**预期收益**: 高（DST-Calib 核心创新）
**依赖 Phase 1 结果**

### 3.1 核心思路

在训练时，不仅扰动 LiDAR→Camera 变换（当前做法），还扰动 Camera 的虚拟视角。这需要：
1. 用 Depth Anything V2 估计每帧的稠密深度
2. 将深度图转换为 3D 点云
3. 从随机虚拟视角重新渲染 Camera 深度投影
4. 与 LiDAR 的随机投影配对训练

### 3.2 预处理步骤 (离线)

```bash
# 1. 为所有训练数据预计算深度图 (一次性)
python precompute_depth.py \
    --data_dir /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --model depth_anything_v2_vitl \
    --output_dir depth_maps/

# 2. DAR 深度校正 (使用 LiDAR 稀疏引导)
python depth_anchor_refine.py \
    --depth_dir depth_maps/ \
    --lidar_dir /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --output_dir refined_depth_maps/
```

### 3.3 训练时增强逻辑

```python
class DoubleSidedAugmentation:
    """在训练时对 Camera 和 LiDAR 都进行视角扰动。"""
    
    def __init__(self, cam_range_deg=3.0, lidar_range_deg=5.0):
        self.cam_range = cam_range_deg
        self.lidar_range = lidar_range_deg
    
    def __call__(self, img, pc, depth_map, gt_T, intrinsics):
        # 1. 随机扰动 Camera 视角
        delta_T_cam = random_perturbation(self.cam_range)
        new_cam_T = delta_T_cam @ gt_T
        
        # 2. 用深度图+新视角重新渲染 Camera 观测
        cam_depth_cloud = depth_to_3d(depth_map, intrinsics)
        new_img_depth = render_from_viewpoint(cam_depth_cloud, new_cam_T)
        
        # 3. 随机扰动 LiDAR 投影视角
        delta_T_lidar = random_perturbation(self.lidar_range)
        init_T = delta_T_lidar @ new_cam_T
        
        # 4. GT = new_cam_T 和 init_T 之间的变换
        gt_calib = new_cam_T @ inv(init_T)
        
        return img, pc, new_img_depth, init_T, gt_calib
```

### 3.4 改造 custom_dataset.py

需要修改 dataset 以:
1. 加载预计算的 refined depth maps
2. 在 `__getitem__` 中应用 DoubleSidedAugmentation
3. 返回额外的 depth 信息给模型

### 3.5 改造 Cam2BEV

Camera 分支输入从 RGB 图像 `(B, 3, H, W)` 扩展为 RGB + Depth `(B, 4, H, W)`:
- 通道 0-2: RGB 图像
- 通道 3: 校正后的深度图

SwinTransformer 第一层卷积需要适配 4 通道输入。

### 3.6 验证标准

- 泛化误差 < 0.50° (比 baseline 提升 >23%)
- Camera bypass 退化 > 40% (证明 Camera 真正有贡献)
- cross-sequence std < 0.12°/axis

---

## Phase 4: 完整 DST-Calib 式架构 (大改造，可选)

**改造成本**: 高 (几乎重写模型)
**预期收益**: 很高（与 DST-Calib 对齐）
**依赖 Phase 3 结果和分析**

### 4.1 核心改造

1. 从 BEV 空间迁移到图像空间 Difference Map
2. 用 ResNet+CBAM 替代 SwinT+DeformableTransformer
3. Block Processing 替代全局 attention
4. 自监督损失 (Chamfer Distance)

### 4.2 决策点

Phase 3 的结果将决定是否需要 Phase 4:
- 如果 Phase 3 达到 < 0.3°，则不需要 Phase 4
- 如果 Phase 3 仅到 ~0.4°，则考虑 Phase 4

---

## 实施优先级和时间线

| 阶段 | 改造内容 | 工作量 | 预计时间 | 前置依赖 |
|---|---|---|---|---|
| **Phase 1** | BEV Diff Fuser | 0.5天代码 + 训练 | 2天 | 无 |
| **Phase 2** | Camera Dropout | 0.5小时代码 + 训练 | 1天 | 无 |
| **Phase 1+2 评估** | 泛化测试 + Camera bypass | 0.5天 | 0.5天 | Phase 1, 2 训练完成 |
| **Phase 3** | 双边增强 | 2天代码 + 训练 | 4天 | Phase 1/2 结果分析 |
| **Phase 4** | 完整重构 (可选) | 5天+ | 1周+ | Phase 3 结果 |

---

## 建议立即执行

1. **Phase 1 + Phase 2 并行启动**: 
   - Phase 1 (BEV Diff Fuser) 修改代码 + 开始训练
   - Phase 2 (Camera Dropout) 在 baseline 配置上训练
   
2. **对照实验矩阵** (4个模型):

| 实验 | Fuser | Cam Dropout | 预期 |
|---|---|---|---|
| baseline | concat | 0 | 0.646° (已有) |
| diff-only | diff | 0 | < 0.60° |
| dropout-only | concat | 0.3 | < 0.64° |
| **diff+dropout** | **diff** | **0.3** | **< 0.55°** |

3. **评估**: 对每个模型运行 test_data_v2 泛化评估 + Camera bypass

---

## 风险和缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Diff Map 训练不收敛 | 低 | 高 | 从 baseline checkpoint 热启动 |
| Camera Dropout 过大导致欠拟合 | 低 | 中 | 尝试 p=0.1, 0.2, 0.3 |
| 深度估计质量不足 | 中 | 高 (Phase 3) | DAR 深度校正 + 质量筛选 |
| BEV 空间 Diff Map 效果弱于图像空间 | 中 | 中 | Phase 4 回退到图像空间 |
