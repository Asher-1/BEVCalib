# V39训练速度瓶颈分析与加速方案

## 当前训练速度（batch_size=24）

```
每epoch时间：356秒（6分钟）
每step时间：4.1秒
Steps/epoch：87步
吞吐量：0.24 steps/s

100 epochs预计：356s × 100 = 35,600s ≈ 10小时
```

## 时间分配（瓶颈分析）

| 阶段 | 耗时 | 占比 | 瓶颈等级 |
|------|------|------|---------|
| **prep（数据增强）** | **63s** | **18%** | 🔴 **主瓶颈** |
| compute（forward+backward） | 283s | 80% | 🟡 正常 |
| data_load | 8s | 2% | ✅ OK |

### prep瓶颈详解

```python
# 数据增强pipeline（CPU密集）
prep包含：
1. augment_mount_jitter（旋转+平移扰动）← 占prep 40%
2. augment_pitch_flip（Pitch翻转）
3. augment_color_jitter（图像增强）
4. augment_pc_jitter + dropout（点云增强）
5. 坐标系变换（camera→lidar）

问题：全部在CPU上运行，成为瓶颈
```

## 🚀 加速方案（目标：5小时/100epochs）

### 方案对比

| 方案 | 改进 | 加速比 | 100ep预计 | 风险 |
|------|------|--------|-----------|------|
| **A: 激进batch_size**⭐ | 24→64 | **2.7×** | **3.7h** ✅ | 显存可能不足 |
| B: 减少epochs | 100→50 | 2× | 5h | 收敛不充分 |
| C: 简化数据增强 | 减少prep | 1.2× | 8.3h | 泛化能力下降 |
| D: 减少评估频率 | eval每20ep | 1.05× | 9.5h | ⚠️ 错过关键信息 |

### 方案A详解：激进batch_size（推荐）

#### 显存估算

```python
# HTCN双分支（bs=24）：30-38GB
# proj_only单分支（bs=24）：预计20-25GB

# L20显存：48GB
# 可用空间：48 - 25 = 23GB

# 增加batch_size到64：
# 预计显存：25GB × (64/24) = 66GB → ❌ 超出

# 安全方案：batch_size=48
# 预计显存：25GB × (48/24) = 50GB → ⚠️ 勉强（需要测试）

# 保守方案：batch_size=40
# 预计显存：25GB × (40/24) = 41.7GB → ✅ 安全
```

#### 加速效果

```python
# 当前：batch_size=24, global_bs=192
# Steps/epoch = dataset_size / global_bs
# ≈ 16704 / 192 = 87 steps

# 方案A1：batch_size=40, global_bs=320
# Steps/epoch = 16704 / 320 = 52 steps
# 加速比：87 / 52 = 1.67×
# 100 epochs：10h / 1.67 = 6h ⚠️ 仍超过5h

# 方案A2：batch_size=48, global_bs=384
# Steps/epoch = 16704 / 384 = 43 steps
# 加速比：87 / 43 = 2.02×
# 100 epochs：10h / 2.02 = 4.95h ✅ 达标

# 方案A3：batch_size=64, global_bs=512
# Steps/epoch = 16704 / 512 = 33 steps
# 加速比：87 / 33 = 2.64×
# 100 epochs：10h / 2.64 = 3.8h ✅✅ 理想
```

### 方案B：减少epochs（备选）

```yaml
# Stage 1: 100 → 60 epochs
# 预期：Jacobian从-0.2提升到0.7-0.8（vs 0.85目标略低）
# 时间：6h

# Stage 2: 80 → 50 epochs  
# Stage 3: 50 → 30 epochs
```

### 方案C：简化数据增强（不推荐）

```yaml
# 减少CPU augmentation
augment_mount_jitter_prob: 0.3  # 0.6 → 0.3
augment_pitch_flip_prob: 0.15  # 0.25 → 0.15
# 预期加速：18% → 10%，总加速1.1×

# 问题：泛化能力会下降
```

## 🎯 最终推荐方案

### 激进方案（推荐⭐，如果显存允许）

```yaml
batch_size: 48  # 冒险尝试
learning_rate: 5.7e-4  # 按√(48/24)=1.41缩放：4e-4×1.41
num_epochs: 60  # 100 → 60（双保险）

# 预期：
# - 2× 加速（batch_size）
# - 0.6× epochs
# - 总时间：10h / 2 × 0.6 = 3h ✅✅
```

### 保守方案（如果激进方案OOM）

```yaml
batch_size: 40  # 安全
learning_rate: 5.2e-4  # 按√(40/24)=1.29缩放
num_epochs: 70  # 适当减少

# 预期：
# - 1.67× 加速
# - 0.7× epochs
# - 总时间：10h / 1.67 × 0.7 = 4.2h ✅
```

## 关于Jacobian/MEDW评估耗时

```
每次Jacobian评估：~77秒（1.3分钟）
每次MEDW评估：~2秒
总评估时间：~79秒/次

如果eval_epoches=10：
- 100 epochs需要10次评估
- 总耗时：79s × 10 = 790s ≈ 13分钟
- 占比：13min / 600min = 2.2% ← ✅ 可接受

如果eval_epoches=20：
- 100 epochs需要5次评估  
- 总耗时：79s × 5 = 395s ≈ 6.6分钟
- 占比：1.1% ← ✅ 更好
```

**结论**：评估耗时很少（1-2%），**不是瓶颈**。

## Cross-attn容量分析

### 当前参数量

```
proj_branch: 1.24M trainable
  └─ AttenDualFusion cross-attn: ~1.0M
      ├─ rot_cross_attention: 512K
      └─ tsl_cross_attention: 512K
      
fusion_head: 66.8K
  └─ GatedFusionHead: 3层MLP

Total trainable: 1.31M (3.13%)
```

### 容量评估

**对比其他模型**：
```
ResNet-18: 11M params
ViT-Small: 22M params  
我们的cross-attn: 1.3M params ← 偏少

但考虑到：
- Encoder frozen（40M预训练知识）
- 任务相对简单（3自由度旋转）
- 数据规模：~16K samples
```

**容量是否足够？**

| 判据 | 当前状态 | 评估 |
|------|---------|------|
| 过拟合风险 | Train精度高，Val精度也高 | ✅ 无过拟合 |
| 欠拟合风险 | Jacobian [WEAK] | ⚠️ **可能容量不足** |
| 数据/参数比 | 16K / 1.3M = 0.012 | ⚠️ 偏低（理想>1） |

**结论**：
1. **对于MEDW任务**：1.3M参数**足够**（已达0.29°）
2. **对于Jacobian任务**：1.3M参数**可能不足**（需要泛化到±10°）

**改进建议**：
- 方案1：解冻PointGPT最后2层 → 3-4M trainable
- 方案2：增加cross-attn层数（Fleet经验中有更深的attn）
- 方案3：先用当前容量验证，不行再扩容
