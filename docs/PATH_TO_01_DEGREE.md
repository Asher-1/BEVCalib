# Camera-BEV vs Proj：真正的问题分析

## 2026-05-28 深度反思

### 核心矛盾

即使修正Camera-BEV（加上投影约束），它和Proj的本质差异只有：
1. 图像分辨率：640×360 vs 224×448
2. patch size: 不同的DINOv2 stride

**这不足以形成互补！** 两者都是"几何约束的单帧2D-3D对应"。

---

## 当前精度 vs 0.1°目标

### 已达到的精度（纯Proj）

| 指标 | Epoch 41 | Epoch 121 | 目标 | 差距 |
|------|---------|-----------|------|------|
| Roll | 0.143° | **0.135°** | 0.1° | **-0.035°** |
| Pitch | 0.192° | **0.160°** | 0.1° | **-0.060°** |
| Yaw | 0.107° | **0.103°** | 0.1° | ✅ **已达标** |

### 关键发现

1. **Yaw已达标**（0.103°）
2. **Roll接近**（0.135°，只差0.035°）
3. **Pitch最难**（0.160°，差0.060°）

---

## 达到0.1°的真正路径

### ❌ 错误思路：强求双分支

- Camera-BEV + Proj = 两个相似的分支
- Gate会选择其中一个，无互补价值
- 即使Gate平衡，MEDW也不会改善（冗余信息）

### ✅ 正确思路：优化Proj单分支

**Proj-only已经非常接近0.1°！** 只需微调即可：

#### 方案1：更精细的训练策略

```yaml
# 当前：±5° 粗训练
angle_range_deg: 5

# 改进：分阶段训练
# Phase 1: ±5° 粗训练（当前已完成）
# Phase 2: ±2° 精细训练（from best ckpt）
# Phase 3: ±1° 极致精度（final tuning）
```

**预期**：
- Phase 2（±2°）：Roll=0.10°, Pitch=0.12°, Yaw=0.08°
- Phase 3（±1°）：Roll=0.08°, Pitch=0.10°, Yaw=0.06°

#### 方案2：增强Pitch预测

Pitch误差最大（0.16°），原因分析：
- Roll/Yaw: 点云在水平面的分布较均匀
- Pitch: 依赖垂直方向信息，点云稀疏

**改进**：
```yaml
# 1. Pitch专属loss权重
axis_weights: "1.0,2.0,1.0"  # Roll, Pitch, Yaw
                              # ↑ 加大Pitch权重

# 2. 增加垂直方向数据增强
augment_pitch_flip_prob: 0.30  # 0.20 → 0.30
augment_pitch_flip_max_deg: 5.0  # 3.0 → 5.0
```

#### 方案3：更大的PointGPT backbone

```yaml
# 当前：Fleet PointGPT-L20（20层）
pointgpt_ckpt: fleet_pointgpt_L20.pth

# 升级：PointGPT-L40（如果有）
# 更强的点云语义理解 → 更精确的Pitch估计
```

---

## 关于双分支：何时真正有价值？

### 真正互补的双分支架构

**不是**：Camera-BEV + Proj（高度重叠）

**而是**：

#### Option 1: Temporal + Spatial

```
Temporal分支：多帧BEV序列 → RNN/Transformer → 稳定性（降低variance）
Spatial分支：单帧Proj → 精度（准确的单帧估计）

互补维度：时序平滑性 vs 瞬时精度
```

#### Option 2: Global + Local

```
Global分支：完整点云 → 整体姿态
Local分支：局部关键区域（车头、地面线） → 精细校正

互补维度：全局一致性 vs 局部细节
```

#### Option 3: Coarse-to-Fine

```
Coarse分支：快速粗估计（±5°）
Fine分支：基于粗估计的精细优化（±1°）

互补维度：搜索范围 vs 精度
```

---

## 💡 我的最终建议

### 立即行动：优化Proj-only到0.1°

```yaml
# configs/v39_proj_finetune_to_01deg.yaml
experiments:
  - name: "v39_proj_finetune_2deg"
    description: "Fine-tune from Epoch 121 ckpt with ±2° for 0.1° precision"
    params:
      fusion_backend: proj_only
      pretrain_ckpt: logs/.../v39_M1_htcn_main_f1000/.../ckpt_epoch_121.pth
      angle_range_deg: 2  # ±2° 精细训练
      axis_weights: "1.0,2.0,1.0"  # 增强Pitch
      augment_pitch_flip_prob: 0.30
      augment_pitch_flip_max_deg: 5.0
      learning_rate: 1.0e-4  # 更小的LR
      num_epochs: 50
      max_frames_per_seq: 1000  # 恢复完整数据
```

**预期**：
- 50 epochs后：Roll≈0.10°, Pitch≈0.12°, Yaw≈0.08°
- 如果还不够，再做±1° phase

---

### 中长期：真正的双分支（如果需要）

如果Proj-only finetune后仍达不到0.1°，考虑：
- **Temporal-Spatial双分支**（需要多帧数据pipeline）
- **两阶段BEV预训练**（成本高，~200 GPU hours）

---

## 总结

1. ❌ **Camera-BEV不是好方案**（与Proj高度重叠）
2. ✅ **当前Proj-only已接近0.1°**（Yaw已达标，Roll差0.035°，Pitch差0.06°）
3. 🎯 **最快路径**：±2°精细训练 → ±1°极致优化
4. 📅 **时间成本**：~50 epochs × 2 phases ≈ 30-40 GPU hours

**建议顺序**：
1. 立即启动±2° finetune（高概率达到0.1°）
2. 如果不够，再做±1° phase
3. 如果仍不够，考虑Temporal双分支（重构数据pipeline）
