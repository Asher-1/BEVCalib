# BEVCalib 泛化性能瓶颈诊断报告

> **日期**: 2026-04-26
> **版本**: V22 诊断实验设计
> **仓库**: `/mnt/drtraining/user/dahailu/code/BEVCalib`
> **状态**: 实验设计完成，待运行

---

## 1. 问题描述

BEVCalib 模型经过 v16~v21 共计 20+ 组实验迭代，泛化角度误差（R/P/Y）始终无法突破 **0.1度** 瓶颈。已尝试的策略包括：

- 多种数据增强组合 (pc_jitter, color_jitter, intrinsic, pitch_flip, sign_flip)
- 不同 z 分辨率 (z=5, z=10)
- 不同点云后端 (spconv vs DRCV)
- 不同 voxel 模式 (hard vs scatter)
- 不同 Loss 设计 (quaternion, axis_rotation, geodesic)

以上策略均未能实现突破性提升。

---

## 2. 根因分析

### 2.1 梯度诊断实验

编写了 `analyze_gradient_flow.py` 对 v21 最优模型进行梯度流向分析，核心发现：

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| Head 梯度 L2 | ~78 | Head 参数获得大量梯度 |
| Backbone 梯度 L2 | ~10 | Backbone (Swin-T) 梯度较小 |
| 自然梯度比 | ~7.5x | Head 梯度天然就是 Backbone 的 7.5 倍 |
| **有效更新比** | **~55x** | 考虑 `backbone_lr_scale=0.1` 后，Head 更新是 Backbone 的 55 倍 |

**根因结论**: `backbone_lr_scale=0.1` 使 Swin-T backbone 有效学习率仅 `1e-5`，处于半冻结状态，特征提取能力严重不足，是泛化瓶颈的首要原因。

### 2.2 Per-Module 梯度分布

| Module | 梯度 L2 | 占比 |
| --- | --- | --- |
| rotation_pred | 70~97 | **~90%** |
| bev_encoder | ~10 | ~8% |
| transformer | ~5 | ~2% |
| img_branch | ~8 | (backbone) |
| pc_branch | ~5 | (backbone) |
| translation_pred | 0 | rotation_only 未参与 |

存在 **梯度漏斗效应**: `rotation_pred` 的 2 个参数 (weight + bias) 吃掉了 head 绝大部分梯度。原因是 quaternion→rotation matrix 转换中 `torch.linalg.inv` 的反向传播产生梯度放大。

### 2.3 与官方仓库的关键差异

| 维度 | 当前仓库 | 官方 UCR-CISL/BEVCalib | 影响 |
| --- | --- | --- | --- |
| **Backbone LR** | `lr × 0.1 = 1e-5` | `5e-5` 统一 LR | Backbone 欠训练 |
| **优化目标** | rotation_only | 旋转 + 平移联合 | 丢失平移梯度的隐性监督 |
| **扰动范围** | ±5°, 0.15m | ±20°, 1.5m | 可能限制学习空间 |
| **Loss 设计** | 复合 loss (axis + quat + geodesic) | 简单 quat + trans MSE | 梯度方向冲突风险 |
| **数据增强** | 多种增强 | 无增强 | 增强可能引入噪声 |

---

## 3. V22 实验方案

### 3.1 设计思路

基于梯度诊断结果，设计 6 组控制变量实验，分两阶段逐步验证：

**Phase A**: 仅改 Backbone LR（隔离 backbone 欠训练问题）
**Phase B**: 开启联合优化 + 提高 LR（测试平移监督的独立贡献）

### 3.2 实验矩阵

| 实验 | backbone_lr_scale | rotation_only | 角度范围 | 平移范围 | axis_loss | 增强 | 对比目标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A1** (基线) | 0.1 → `1e-5` | Yes | 5° | 0.15m | Yes | v21 | v21 基线复现 |
| **A2** | 0.5 → `5e-5` | Yes | 5° | 0.15m | Yes | v21 | Backbone LR 提升 5x |
| **A3** | 1.0 → `1e-4` | Yes | 5° | 0.15m | Yes | v21 | 统一 LR (官方策略) |
| **B1** | 0.5 → `5e-5` | **No** | 5° | 0.15m | Yes | v21 | 联合优化 + 中等 LR |
| **B2** | 1.0 → `1e-4` | **No** | 10° | 0.5m | Yes | v21 | 联合优化 + 中等扰动 |
| **B3** | 1.0 → `1e-4` | **No** | 20° | 1.5m | **No** | **无** | 复现官方极简配置 |

### 3.3 关键对比链

```
1. Backbone LR 影响:      A1 → A2 → A3
2. 联合优化 vs 仅旋转:     A2 vs B1, A3 vs B2
3. 扰动范围影响:            B1 → B2 → B3
4. 复杂 loss+增强 vs 极简:  A3 vs B3
5. 全面最优 vs v21 基线:    winner vs A1
```

### 3.4 预期结果

- **A2/A3 > A1**: 提高 backbone LR 应直接改善泛化（验证根因假设）
- **B 系列 > A 系列**: 联合优化提供额外梯度信号
- **B3 双面性**: 小数据 + 大扰动可能不匹配，也可能因极简设计意外有效
- 若 A3 即可显著提升 → backbone 欠训练是唯一主因
- 若 B 系列进一步提升 → 联合优化是独立贡献因子

---

## 4. 工程改进

### 4.1 梯度监控集成

在 `train_kitti.py` 中新增 per-group 梯度监控功能：

- 每 50 steps 采样一次梯度 L2 范数（开销极低）
- 分组统计: backbone / head / per-module
- TensorBoard 可视化: `GradNorm/backbone`, `GradNorm/head`, `GradNorm/ratio_hd_bb`
- Epoch 结束时输出汇总到 `train.log`

**健康指标参考**:

| 指标 | 含义 | 健康范围 |
| --- | --- | --- |
| `ratio` | head/backbone 自然梯度比 | 3~15x |
| `eff_update_ratio` | 考虑 LR scale 后的有效更新比 | < 20x |
| Per-module grad | 各模块梯度贡献 | rotation_pred 不应超过总量的 80% |

### 4.2 诊断工具

新增 `analyze_gradient_flow.py`:
- 加载任意 checkpoint，跑 N 个 batch 的 forward + backward
- 输出 backbone/head 梯度比、有效更新比、per-module 分解
- 自动给出风险等级判定 (SEVERE / WARNING / OK)

---

## 5. 配置文件清单

| 文件 | 用途 |
| --- | --- |
| `configs/eval_v21_generalization.yaml` | v21 全部 8 模型评估配置 |
| `configs/batch8_train_all_v22_diagnosis.yaml` | v22 诊断实验训练配置 (6 组) |
| `configs/eval_v22_diagnosis.yaml` | v22 诊断实验评估配置 |
| `analyze_gradient_flow.py` | 梯度流向分析工具 |

---

## 6. 运行命令

```bash
# V22 训练 (6 组实验自动串行)
bash batch_train.sh configs/batch8_train_all_v22_diagnosis.yaml

# V22 评估 (训练完成后)
HF_HUB_OFFLINE=1 python run_generalization_eval.py --config configs/eval_v22_diagnosis.yaml --parallel -1

# V21 基线评估 (对照)
HF_HUB_OFFLINE=1 python run_generalization_eval.py --config configs/eval_v21_generalization.yaml --parallel -1
```

---

## 7. 下一步计划

1. **运行 V22 实验** → 验证 backbone LR 和联合优化假设
2. **对比 V21 基线** → 量化改进幅度
3. **根据 V22 结果决定 V23 方向**:
   - 若 backbone LR 是主因 → V23 聚焦 LR schedule 精调
   - 若联合优化有贡献 → V23 探索 6D rotation 表示 + 更稳健的 loss
   - 若大扰动有效 → V23 探索课程学习 (从小扰动渐进到大扰动)
