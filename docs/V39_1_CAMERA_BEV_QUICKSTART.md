# V39.1 Camera-BEV Fusion 快速启动

## 背景

V39 M1训练发现Gate坍塌问题：
- 两个训练（原始+修复）都完全坍塌（bev=0.000）
- 但MEDW仍优秀（0.31-0.42°，达到v36水平）
- **结论**：当前BEV架构无法提供互补价值

## 新架构：Camera-BEV Cross-Attention

### 核心改进

| 维度 | 原BEV | Camera-BEV |
|------|-------|-----------|
| 坐标系 | BEV俯视投影（破坏camera几何） | Camera坐标系（保留前向对应） |
| 链路长度 | 5层（DINOv2→Query→PointGPT2BEV→Fuser→Transformer→Pool） | 2层（DINOv2→CrossAttn→Pool） |
| 信息利用 | mask 30-40%（cam FOV内） | 100%（所有patches） |
| 与Proj互补 | 同任务不同视角（但更弱） | **不同分辨率+query方式** |

### 快速测试

```bash
# 1. 修改hybrid_triple_calib.py添加Camera-BEV支持
# 见下方集成代码

# 2. 新建配置 v39_camera_bev.yaml
cp configs/v39_minimal.yaml configs/v39_camera_bev.yaml

# 修改关键参数：
# - fusion_backend: camera_bev_triple  # 新后端
# - max_frames_per_seq: 500  # 快速验证
# - num_epochs: 100  # 快速对比

# 3. 启动训练
bash batch_train.sh configs/v39_camera_bev.yaml
```

### 预期结果

**成功标志**：
1. ✅ Gate保持平衡（bev=0.3-0.6, entropy>0.6）到Epoch 20+
2. ✅ MEDW与纯Proj相比有改善（或至少持平）
3. ✅ 训练精度收敛正常（Train Rot < 2.5°）

**如果失败**：
- Gate仍坍塌 → 说明Proj预训练优势无法克服，考虑放弃双分支
- MEDW无改善 → Camera-BEV与Proj确实无互补性，简化为proj_only

## 集成代码

### 步骤1：修改hybrid_triple_calib.py

在`HybridTripleCalib.__init__`中添加：

```python
# hybrid_triple_calib.py Line ~60

FUSION_BACKENDS = (
    'bev_only', 'proj_only', 'hybrid_dual', 'hybrid_triple',
    'camera_bev_triple',  # 新增
)

# Line ~73
self.use_camera_bev = fusion_backend == 'camera_bev_triple'
self.use_bev = fusion_backend in ('bev_only', 'hybrid_dual', 'hybrid_triple')
self.use_proj = fusion_backend in ('proj_only', 'hybrid_dual', 'hybrid_triple', 'camera_bev_triple')

# Line ~79-120（原BEV分支构建）
if self.use_bev:
    # 保持原有BEV逻辑
    ...
elif self.use_camera_bev:
    # 新增Camera-BEV分支
    from camera_bev_fusion import build_camera_bev_branch
    
    self.img_branch = Cam2BEVQuery(...)  # 复用DINOv2
    self.pointgpt_encoder = PointGPTEncoder(...)  # 复用PointGPT
    
    self.camera_bev_branch = build_camera_bev_branch(
        img_encoder=self.img_branch.CamEncode,  # 共享DINOv2
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=head_dropout
    )
    self.bev_feat_dim = self.camera_bev_branch.out_dim
```

### 步骤2：修改forward逻辑

```python
# Line ~296-302
if self.use_bev:
    f_bev = self._forward_bev_features(...)
elif self.use_camera_bev:
    # Camera-BEV forward
    xyz_g, feat_g = self.pointgpt_encoder(pc_xyz) if xyz_g is None else (xyz_g, feat_g)
    T_cam2lidar = torch.linalg.inv(t_current.float())
    
    f_bev = self.camera_bev_branch(
        img=img,
        pc_groups_xyz=xyz_g,
        pc_groups_feat=feat_g,
        T_cam2lidar=T_cam2lidar,
        cam_intrinsic=cam_intrinsic
    )
    f_bev = self.head_drop(f_bev)
```

### 步骤3：修改aux head（deep supervision）

```python
# Line ~172-177
if deep_supervision_weight > 0 and (self.use_bev or self.use_camera_bev) and self.use_proj:
    self.aux_bev_head = SingleBranchHead(self.bev_feat_dim, dropout=head_dropout)
    self.aux_proj_head = SingleBranchHead(self.proj_feat_dim, dropout=head_dropout)
```

### 步骤4：创建配置文件

```yaml
# configs/v39_camera_bev.yaml

experiments:
  - name: "v39_M2_camera_bev_quick"
    description: "Camera-BEV fusion: direct cross-attn without BEV projection"
    dataset: "all"
    version: "v39_M2_camera_bev_quick"
    params:
      fusion_backend: camera_bev_triple  # 新后端
      pc_encoder_mode: pointgpt2bev  # 保持兼容（仅Proj用）
      fusion_variant: gated
      max_frames_per_seq: 500
      num_epochs: 100
      eval_epoches: 20
      # 其他参数与v39_minimal保持一致
```

## 监控指标

```bash
# 实时监控Gate状态
tail -f logs/*/v39_M2_camera_bev_quick/train.log | grep "Gate mean"

# 预期健康状态：
# Epoch [1-5]: bev=0.4-0.6, entropy>0.8
# Epoch [10-20]: bev=0.3-0.5, entropy>0.6
# Epoch [50+]: bev=0.2-0.4, entropy>0.5（允许轻微倾斜但不完全坍塌）
```

## 决策树

```
Camera-BEV训练到Epoch 20
  |
  ├─ Gate保持平衡(bev>0.2, entropy>0.5)
  |    └─> ✅ 继续到Epoch 100，对比MEDW
  |          ├─ MEDW优于纯Proj → 成功！Camera-BEV有效
  |          └─ MEDW持平/更差 → 放弃，简化为proj_only
  |
  └─ Gate坍塌(bev<0.05)
       └─> ❌ Epoch 40停止，放弃双分支架构
            → 正式结论：Proj预训练优势无法克服
            → 简化为fusion_backend='proj_only'
```
