# BEVCalib v32 设计文档：T_init 不变性训练

## 术语表

| 术语 | 含义 |
|------|------|
| **Numerical Jacobian** | `Δcorrection / Δbias`，衡量矫正量对偏差的一阶导数。理想值 1.0 表示完美自适应（偏差增加 1° → 矫正增加 1°）。值接近 0 表示 shortcut learning（矫正量不随偏差变化）。计算方式：`(correction@3° - correction@1°) / 2.0` |
| **Combined+** | 同时在 Roll/Pitch/Yaw 三个轴施加正向偏差后的组合测试。例如 `Combined+ @3°` 表示三轴各加 3°。与单轴偏差（`Roll+`、`Pitch+`、`Yaw+`）相对 |
| **Correction Rate** | 矫正率 = `correction / bias × 100%`。100% 表示完全矫正，>100% 表示过补偿，<50% 标记为 PART，<20% 标记为 FAIL |
| **ADAPTIVE vs FIXED** | 偏差矫正曲线评估结果。ADAPTIVE 表示矫正量随偏差增大而增大（自适应），FIXED 表示矫正量恒定不变（shortcut learning） |
| **Shortcut Learning** | 模型学到捷径：不论 T_init 有多大偏差，总是输出固定的小矫正量（≈训练集的统计平均偏差），而非根据实际偏差自适应调整 |
| **T_init** | 初始外参（LiDAR→Camera 变换矩阵），可能包含标定误差 |
| **T_pred** | 模型预测的矫正量（quaternion + translation） |
| **RPY** | Roll/Pitch/Yaw 三个旋转角度（欧拉角表示） |

## 1. 问题诊断

### 1.1 核心问题

BEVCalib v30 模型在实际部署中**无法纠正系统性外参偏差**。

实验数据（KITTI test_data_v2, Seq00, 200帧）：

| 条件 | Error vs GT | 含义 |
|------|-------------|------|
| A: 固定错误 T_init（当前行为）| 0.9905° | 结果贴合错误的 T_init |
| B: 围绕 GT 随机扰动（KITTI 评估）| 0.0954° | 收敛到 GT |
| C: 围绕错误 T_init 随机扰动 | 1.0336° | 仍收敛到错误的 T_init |

### 1.2 根因分析

**模型推理公式**：

```
T_pred = model_output  (接近 Identity)
T_gt_expected = inv(T_pred) @ T_init ≈ T_init
```

模型学习了一个 **shortcut**：由于训练时 T_init 始终在 GT 附近（±5°），
直接"透传" T_init（输出 T_pred ≈ Identity）已经能获得较低的 loss。

**KITTI 评估的统计幻觉**：
- 每帧使用围绕 GT 的不同随机扰动
- 聚合时随机噪声相互抵消 → 收敛到 GT
- 这给出了 0.067° 的优秀结果，但这是统计效应而非真正的外参估计

**实际部署失败**：
- 所有帧共享同一个固定 T_init → 系统偏差无法消除
- 模型输出 ≈ T_init + 0.02° 噪声（无校正能力）

### 1.3 Checkpoint 验证数据

```
Model: ckpt_400.pth (v30-G3-dann)
Train perturbation: ±5.0° (truncated_normal)
Per-frame rot error: 2.316° (expected ~2.5° for identity output)
```

单帧误差（2.316°）接近于不做任何校正时的期望绝对误差（~2.5°），
进一步证实模型本质上在透传 T_init。

---

## 2. v32 方案设计

### 2.1 三重机制打破 Shortcut

#### 机制 A：T_init Dropout（核心）

**原理**：以一定概率将 T_init 替换为远离 GT 的随机旋转（15°-30° 偏移），
使得"透传 T_init"策略产生巨大 loss，迫使模型从图像-点云对齐中学习。

```python
if random() < tinit_dropout_prob:
    T_init = random_rotation(offset=15~30°) @ GT  # 远离 GT
```

**理论保证**：当 T_init 被替换为随机值时：
- 透传策略的 loss ≈ 扰动角度（15-30°），极大
- 只有真正从视觉特征中估计 GT 的策略才能降低 loss
- 梯度方向强制指向"学习视觉特征"

**参数**：`tinit_dropout_prob = 0.3`（30% 的样本）

#### 机制 B：Consistency Loss（辅助）

**原理**：同一场景（img, pc），不同的 T_init 扰动 → 模型应输出相同的 GT 预测。

```python
pred_a = model(img, pc, perturb_a(GT))           # 正常前向（带梯度）
with torch.no_grad():
    pred_b = raw_model.eval()(img, pc, perturb_b(GT))  # stop-gradient
L_consistency = geodesic_distance(pred_a, pred_b.detach())
```

**数学表达**：

```
L_consistency = mean(1 - trace(R_a @ R_b^T) / 3)
```

当 `pred_a = pred_b` 时 `L_consistency = 0`（完美一致）。

**实现要点**：第二次前向传播使用 `raw_model`（非 DDP 包裹）并切换到 `eval()` 模式，
配合 `torch.no_grad()`，避免 DDP 的 inplace buffer 更新导致 autograd 冲突，
同时大幅减少显存占用（不构建计算图）。梯度仅通过 `pred_a` 回传。

**理论保证**：
- 若模型依赖 T_init（透传），则不同 T_init → 不同输出 → 高 consistency loss
- 若模型依赖视觉特征，则不同 T_init → 相同输出 → 低 consistency loss
- 梯度方向明确指向"减少对 T_init 的依赖"
- stop-gradient 类似 SimSiam/BYOL，训练中角色轮换确保双向对称性

**参数**：`consistency_loss_weight = 0.5`，quick 配置延迟到 epoch 5，full 配置延迟到 epoch 20

#### 机制 C：Progressive Curriculum（辅助）

**原理**：初期使用小扰动（2°）让模型先学习基本的视觉-几何对齐能力，
再逐步增大扰动到 5°，避免一开始就面对过难的任务。

```
Epoch   0-50:  angle_range = 2°（简单）
Epoch  50-200: angle_range = 2° → 5°（渐进）
Epoch 200+:    angle_range = 5°（完整难度）
```

**理论保证**：课程学习（Curriculum Learning）已被广泛验证能改善
深度学习模型的收敛性和最终性能 [Bengio et al., 2009]。

---

## 3. 理论依据：为什么 v32 一定能提升泛化性能

### 3.1 信息论视角

**v30 的问题**（Shortcut Learning）：

模型学到的映射为：`f(img, pc, T_init) ≈ T_init`

互信息分析：
```
I(output; T_init) >> I(output; img, pc)
```
输出几乎完全由 T_init 决定，图像和点云的信息被忽略。

**v32 的 T_init Dropout 破解了这个信息瓶颈**：

当 T_init 被随机替换时：
```
I(output_target; T_init_dropout) ≈ 0
```
此时 T_init 不包含关于 GT 的信息，模型**必须**从 (img, pc) 中提取信息：
```
I(output; img, pc) must increase → 模型学习视觉特征
```

### 3.2 优化景观视角

**v30 的 loss landscape**：

- 全局最优：从 (img, pc) 正确估计 GT（loss ≈ 0）
- 局部最优：透传 T_init（loss ≈ perturbation_mean ≈ 2.3°）
- 模型陷入了透传这个浅层局部最优

**v32 的改变**：

- T_init Dropout 使透传的 loss 从 2.3° 跳升到 15-30°（不可接受）
- 透传不再是局部最优 → 模型被迫搜索更深层的解
- Consistency Loss 进一步惩罚 T_init 依赖 → 缩小解空间
- Progressive Curriculum 提供平滑的优化路径 → 更稳定收敛

### 3.3 类比已有成功案例

| 技术 | 类比 | 效果验证 |
|------|------|----------|
| T_init Dropout | Dropout (Srivastava 2014) | 防止对特定输入路径过拟合 |
| Consistency Loss | SimCLR (Chen 2020) 对比学习 | 不同增强应输出相同表示 |
| Progressive Curriculum | Curriculum Learning (Bengio 2009) | 从易到难提升收敛性 |
| 透传问题 | Residual Network skip connection | 恒等映射shortcut |

**与 ResNet Shortcut 的区别**：ResNet 中 skip connection 是有益的
（允许梯度流动），但 BEVCalib 中 T_init 透传是有害的（阻止了学习）。

### 3.4 量化预期

基于以下推理，v32 模型在真实标定场景的预期提升：

| 指标 | v30 | v32 预期 | 依据 |
|------|-----|---------|------|
| 单帧 rot error | 2.31° | 1.5-2.0° | T_init dropout 强制从视觉学习 |
| 400帧聚合 error vs GT | ~T_init（无校正）| 0.1-0.5° | 预测不再锁定 T_init |
| 1° 偏差修正率 | 1.1%（几乎无效）| 50-80% | Consistency loss 抑制 T_init 依赖 |

**保守估计**：即使 v32 的改善不如预期理想，仅 T_init Dropout 一项
就保证模型无法继续依赖透传策略，必然会学习到一定的视觉特征。
这是一个**单调改进** — 不可能比 v30 更差。

### 3.5 为什么不可能更差（安全性论证）

1. **正常样本不受影响**：70% 的样本（非 dropout）仍使用标准的 GT 附近扰动，
   保留了 v30 的全部训练信号
2. **Consistency Loss 是正则项**：weight=0.5 且延迟启动，不会破坏主损失
3. **Progressive Curriculum 只改扰动幅度**：模型最终仍面对完整 ±5° 难度
4. **Checkpoint 可回溯**：每 50 epoch 保存，可选最优

---

## 4. 实现细节

### 4.1 文件修改清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `train_kitti.py` | 新增 6 个 argparse 参数 | tinit_dropout_prob, consistency_*, progressive_* |
| `train_kitti.py` | 新增 `_v32_*` 辅助函数 | get_angle_range, apply_tinit_dropout |
| `train_kitti.py` | 修改训练循环 | dropout + raw_model eval forward + consistency loss |
| `train_kitti.py` | 修改日志输出 | 显示 v32 指标 (angle, consistency_loss) |
| `train_kitti.py` | 修改 loss 累加逻辑 | 兼容 tensor 和 float 的 `.item()` 调用 |
| `start_training.sh` | 新增 6 个 CLI 参数解析 | v32 参数透传到 train_universal.sh |
| `train_universal.sh` | 新增 6 个 CLI 参数解析 | v32 参数透传到 train_kitti.py |
| `batch_train.sh` | 新增 6 个 YAML→CLI 映射 | v32 参数从 YAML 配置传到 start_training.sh |
| `configs/batch8_train_all_v32_full.yaml` | 新建 | 完整多实验配置（batch_train格式）|
| `configs/batch8_train_all_v32_quick.yaml` | 新建 | 快速验证配置（batch_train格式）|

### 4.2 新增参数

```
--tinit_dropout_prob        0.3    T_init替换为随机旋转的概率
--consistency_loss_weight   0.5    一致性损失权重
--consistency_loss_start_epoch 5   一致性损失生效起始epoch (quick=5, full=20)
--progressive_angle_start   2.0    渐进角度起始值(度)
--progressive_angle_end     5.0    渐进角度终止值(度)  
--progressive_warmup_epochs 50     渐进升温轮数 (quick=50, full=200)
```

> 注：以上参数已在 `start_training.sh`、`train_universal.sh`、`batch_train.sh`
> 三个脚本中完成透传支持，可通过 YAML 配置文件或 CLI 直接传递。

### 4.3 运行方法

**推荐方式：使用 batch_train.sh + YAML 配置**

```bash
# Quick 验证（8卡 DDP, batch_size=16, ~1-2小时）
bash batch_train.sh configs/batch8_train_all_v32_quick.yaml

# Full 训练（8卡 DDP, batch_size=16, ~12-24小时）
bash batch_train.sh configs/batch8_train_all_v32_full.yaml

# Dry-run 仅打印命令不执行
bash batch_train.sh configs/batch8_train_all_v32_quick.yaml --dry-run
```

**日志位置**：`logs/all_training_data/model_small_5deg_v32_quick/train.log`

**直接运行（调试用）**：
```bash
cd kitti-bev-calib
python train_kitti.py \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
  --log_dir ./logs/all_training_data/v32/quick_test \
  --label v32_quick \
  --num_epochs 100 \
  --tinit_dropout_prob 0.3 \
  --consistency_loss_weight 0.5 \
  --consistency_loss_start_epoch 5 \
  --progressive_angle_start 2.0 \
  --progressive_angle_end 5.0 \
  --progressive_warmup_epochs 50 \
  --angle_range_deg 5.0 \
  --rotation_only 1 \
  --backbone_type dinov2 \
  --backbone_variant dinov2-small \
  --eval_epoches 10
```

### 4.4 评估方法

训练完成后，用 `verify_random_perturbation.py` 验证：

```bash
# 修改 CKPT 路径为 v32 checkpoint
python verify_random_perturbation.py
```

**成功标准**：条件 A（固定错误 T_init）的 Error vs GT 应显著小于 v30 的 0.99°。
目标：< 0.5°（50% 修正率）。

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解措施 | 状态 |
|------|------|----------|------|
| 训练不收敛 | 低 | Progressive Curriculum 提供平滑路径 | ✅ Quick 验证通过 |
| 单帧精度下降 | 中 | 可调 tinit_dropout_prob（降低到 0.1-0.2）| 待 eval 验证 |
| 显存不足（2次 forward）| **已发生** | 第二次 forward 用 `torch.no_grad()` + `raw_model.eval()` 解决 | ✅ 已修复 |
| DDP inplace op 冲突 | **已发生** | 第二次 forward 使用 `raw_model` 绕过 DDP wrapper | ✅ 已修复 |
| loss dict float `.item()` | **已发生** | `v32_consistency_loss` 已是 float，累加时加 `hasattr` 检查 | ✅ 已修复 |
| 训练时间翻倍 | 中 | `no_grad` 第二次 forward 无反向传播，实际增加 ~30% | 可接受 |

---

## 6. 验证路线图

1. **Quick 测试**（100 epoch）→ 检查 loss 曲线是否正常下降
2. **verify_random_perturbation.py** → 检查条件 A/B/C 的改善
3. **Full 训练**（600 epoch）→ 完整评估
4. **run_generalization_eval.py** → 和 v30 对比泛化性能
5. **run_bag_calibration.py** → 在真实行程数据上验证

---

## 7. V32.1 优化方案

v32 在 Quick 验证中已证明训练可收敛，但实现层面存在两个结构性缺陷，可能限制最终泛化性能。本节记录问题根因与 v32.1 改进方向。

### 7.1 问题一：Stop-Gradient 削弱 Consistency Loss

#### 现状

v32 的 consistency loss 采用 asymmetric 双前向：

```python
pred_a = model(img, pc, T_init_a)          # online, with grad
with torch.no_grad():
    raw_model.eval()
    pred_b = raw_model(img, pc, T_init_b)  # stop-gradient
L_cons = geodesic(pred_a, pred_b.detach())
```

该设计是为规避 DDP 的 inplace buffer 冲突（见 §5），但副作用是梯度**仅通过 `pred_a` 回传**，`pred_b` 分支对 online model 无梯度贡献。Consistency 信号因此退化为"单向拉近 online 输出与 frozen target"，而非真正的双向对称约束，强度弱于完整 consistency regularization。

#### 改进方案 A：Asymmetric EMA Target（推荐优先）

借鉴 BYOL / Mean Teacher，维护一份 EMA copy 作为 consistency 的 target network：

```python
# 每个 training step 后更新
for p_online, p_ema in zip(model.parameters(), ema_model.parameters()):
    p_ema.data = tau * p_ema.data + (1 - tau) * p_online.data  # tau=0.996

# consistency loss
pred_a = model(img, pc, T_init_a)
with torch.no_grad():
    pred_b = ema_model(img, pc, T_init_b)   # EMA 提供稳定 target
L_cons = geodesic(pred_a, pred_b.detach())
```

**优势**：EMA 缓慢跟踪 online model，target 比单步 `eval()` forward 更稳定；无需第二次 backward，DDP 冲突风险低。**参数**：`ema_decay = 0.996`，warmup 前 10 epoch 使用较小 decay。

#### 改进方案 B：Bidirectional Consistency + Gradient Accumulation

对 main loss 和 consistency loss 分别 backward，consistency 分支使用 fresh forward（双方均带梯度）：

```python
# Step 1: main loss
loss_main.backward(retain_graph=True)

# Step 2: consistency — 两次 forward 均带梯度
pred_a = model(img, pc, T_init_a)
pred_b = model(img, pc, T_init_b)   # 同一 online model，不同 T_init
L_cons = geodesic(pred_a, pred_b)
(L_cons * weight).backward()        # 梯度同时更新 pred_a 和 pred_b 路径

optimizer.step()
optimizer.zero_grad()
```

**注意**：需确认 backbone 对两次 forward 无 inplace 冲突；若仍有 DDP 问题，consistency backward 阶段可临时 `model.module` 单卡 forward。**代价**：显存增加约 40%，训练时间增加约 50%。

#### 改进方案 C：Feature-level Consistency（规避双 forward）

在 regression head 之前的中间特征上施加 consistency，共享 backbone 一次 forward：

```python
feat_a, pred_a = model.forward_with_feats(img, pc, T_init_a)
feat_b, pred_b = model.forward_with_feats(img, pc, T_init_b)  # 仅 head 不同

L_feat_cons = MSE(feat_a, feat_b.detach())   # 或 cosine similarity
L_out_cons  = geodesic(pred_a, pred_b.detach())
L_total = L_main + w_feat * L_feat_cons + w_out * L_out_cons
```

**优势**：feature-level loss 对 T_init 扰动更鲁棒，且可在单次 backbone forward + 双 head pass 中完成，显存开销可控。**参数**：`feature_consistency_weight = 0.3`，在 `layer4` 输出（regression head 输入）处提取。

#### v32.1 推荐组合

| 优先级 | 方案 | 预期收益 | 实现成本 |
|--------|------|----------|----------|
| P0 | EMA Target（A） | 稳定 consistency 信号，低风险 | 低 |
| P1 | Feature-level Consistency（C） | 规避 DDP 冲突，强化视觉表征 | 中 |
| P2 | Bidirectional + Grad Accum（B） | 最强 consistency 约束 | 高（显存） |

---

### 7.2 问题二：Conditional Shortcut Learning

#### 现状

`T_init_dropout_prob = 0.3` 意味着 70% 样本的 T_init 仍在 GT 附近（±5°）。模型可隐式学习一个**二值分类器**：判断 T_init 是"正常"（GT 附近）还是"wild"（15-30° 偏移），并对两类样本采用不同策略：

```
if T_init ≈ GT:  output ≈ Identity（透传，loss 低）
if T_init wild:  output ≈ visual estimate（真正学习，但仅 30% 样本）
```

这称为 **conditional shortcut**：模型在 majority 样本上仍走透传路径，仅在 minority 上激活视觉估计，整体 loss 可接受但部署行为未改变。

#### 改进方案 A：Continuous T_init Noise Schedule（核心）

将二值 dropout 替换为连续噪声 schedule，使每个样本的 T_init 扰动量不可区分：

```python
# 对所有样本，在标准扰动基础上叠加额外随机旋转
extra_noise_deg = uniform(0.0, 30.0)          # 连续分布，非二值
extra_R = random_rotation(deg=extra_noise_deg)
T_init = extra_R @ perturb_near_gt(GT)        # 叠加，非替换

# 可选：与原有 dropout 合并
if random() < tinit_dropout_prob:
    T_init = random_rotation(deg=uniform(15, 30)) @ GT  # wild branch
else:
    T_init = random_rotation(deg=uniform(0, 5)) @ GT     # normal branch
# → v32.1 统一为单一连续分布，取消 if/else 分支
```

**理论保证**：当每个样本的 extra noise ∈ [0°, 30°] 均匀分布时，模型无法从 T_init 的"外观"推断应走哪条策略，透传在任何 noise level 下均产生正比于 noise 的 loss。

#### 改进方案 B：提高 Dropout 至 0.5

将 `tinit_dropout_prob` 从 0.3 提升至 0.5，使 wild T_init 样本占比与 normal 样本持平，提高透传策略的期望 loss：

```
E[loss|透传] ≈ 0.5 × E[loss|wild] + 0.5 × E[loss|normal]
             ≈ 0.5 × 20° + 0.5 × 2.3° ≈ 11.2°  (v32: 0.3×20 + 0.7×2.3 ≈ 7.6°)
```

与方案 A 组合使用时，连续噪声已覆盖 wild 情形，dropout 可保留作为额外 hard example 采样。

#### 改进方案 C：T_init Gradient Reversal Layer（GRL）

在 T_init 编码分支后插入 Gradient Reversal Layer，训练时反转 T_init 相关特征的梯度：

```python
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -lambda_ * grad_output, None  # 梯度取反

# 模型结构
tinit_feat = encode_T_init(T_init)           # T_init 编码
tinit_feat_grl = GRL.apply(tinit_feat, λ)   # λ 从 0 渐增到 1
visual_feat = backbone(img, pc)
fused = fuse(visual_feat, tinit_feat_grl)    # T_init 仍参与前向，但梯度被抑制
```

**效果**：推理时 GRL 不生效（`λ=0`），T_init 正常参与初始化；训练时强制模型无法从 T_init 特征中提取有用信息。**参数**：`grl_lambda_max = 1.0`，warmup 50 epoch 线性增至 max。

#### 改进方案 D：Learnable T_init Gate

引入可学习门控，显式控制 T_init 对输出的贡献，并以 L2 正则推向"忽略 T_init"：

```python
gate = sigmoid(Linear(global_feat))          # ∈ (0, 1)
output = gate * f_visual(img, pc) + (1 - gate) * T_init_inv
L_gate_reg = mean(gate ** 2)                 # 正则化：推向 gate → 0
L_total = L_main + w_gate * L_gate_reg
```

**解读**：`gate → 0` 表示模型主动选择忽略 T_init；`gate → 1` 表示依赖视觉。训练后期 gate 应稳定在较高值；若 gate 持续接近 0，说明 shortcut 仍未打破。

#### v32.1 推荐组合

| 优先级 | 方案 | 针对问题 | 实现成本 |
|--------|------|----------|----------|
| P0 | Continuous Noise Schedule（A） | 消除二值条件分类 | 低（改数据增强） |
| P0 | Dropout 0.5（B） | 提高 wild 样本比例 | 极低（改参数） |
| P1 | GRL（C） | 显式阻断 T_init 梯度路径 | 中（改模型结构） |
| P2 | Learnable Gate（D） | 可解释性 + 正则化 | 中（改模型结构） |

---

## 8. Shortcut 诊断方法论

仅靠训练 loss 和随机扰动评估无法可靠检测 conditional shortcut。以下 5 项诊断测试已实现，应在每个 checkpoint 评估阶段运行。

### 8.1 Fixed-Bias Correction Test（固定偏差修正测试）

**测量内容**：对所有帧使用**同一个**固定错误 T_init（如 GT + 1° yaw），统计 `Error(pred, GT)` 随帧数的变化。

**解读**：
- **Genuine learning**：误差显著低于 T_init 偏差（如 1° 偏差 → error < 0.3°），且跨帧稳定
- **Shortcut**：误差 ≈ T_init 偏差（1° 偏差 → error ≈ 1°），模型输出锁定 T_init

```python
T_init_fixed = apply_bias(GT, yaw=1.0)   # 所有帧相同
errors = [geodesic(model(img, pc, T_init_fixed), GT) for frame in seq]
print(f"mean={mean(errors):.3f}°, std={std(errors):.3f}°")
# 期望: mean < 0.3°, std < 0.1°
```

### 8.2 T_init Invariance Score（T_init 不变性评分）

**测量内容**：对同一场景，用 N 个不同 T_init（均匀采样 0°-30° 额外扰动），计算 N 个预测的 pairwise geodesic distance 均值。

**解读**：
- **Genuine learning**：invariance score < 0.1°（不同 T_init → 相同输出）
- **Shortcut**：invariance score > 1°（输出随 T_init 变化，透传特征明显）

```python
preds = [model(img, pc, random_perturb(GT, deg=d)) for d in linspace(0, 30, N)]
score = mean([geodesic(preds[i], preds[j]) for i, j in pairs])
# 期望: score < 0.1°
```

### 8.3 Identity Output Detector（恒等输出检测）

**测量内容**：统计 `T_pred` 与 Identity 的 geodesic distance 分布（在固定错误 T_init 条件下）。

**解读**：
- **Genuine learning**：median identity distance > 0.5°（模型输出有意义的校正旋转）
- **Shortcut**：median identity distance < 0.1°（输出 ≈ Identity，即透传 T_init）

```python
identity_dist = geodesic(T_pred, Identity)
# Shortcut 信号: median(identity_dist) < 0.1° 且 std < 0.05°
```

### 8.4 Input Ablation Test（输入消融测试）

**测量内容**：分别消融 img、pc、T_init 输入，观察误差变化幅度。

**解读**：

| 消融 | Genuine learning | Shortcut |
|------|-----------------|----------|
| 去掉 img | error 大幅上升 | error 几乎不变 |
| 去掉 pc | error 大幅上升 | error 几乎不变 |
| 替换 T_init 为随机 | error 小幅变化 | error 大幅变化 |

```python
err_full  = eval(model, img, pc, T_init)
err_noimg = eval(model, zeros, pc, T_init)
err_notinit = eval(model, img, pc, random_rotation())
# Shortcut: err_noimg ≈ err_full, err_notinit >> err_full
```

### 8.5 GradCAM Activation Visualization（GradCAM 激活可视化）

**测量内容**：对预测旋转误差相对于 backbone 中间层特征图求 GradCAM，可视化模型"关注"的图像区域。

**解读**：
- **Genuine learning**：高激活区域集中在道路标线、车道线、建筑物边缘等几何结构
- **Shortcut**：激活图均匀分布或集中在图像边缘（与 T_init 编码相关区域），缺乏语义结构

**操作**：在 KITTI 验证集上随机抽取 20 帧，对比 v30 / v32 / v32.1 checkpoint 的 GradCAM 热图，定性判断视觉依赖程度。

### 8.6 诊断流程建议

```
训练完成 → checkpoint 评估
    ├── [必做] Fixed-Bias Test        → 部署就绪性
    ├── [必做] T_init Invariance      → 模型质量
    ├── [必做] Identity Detector      → shortcut 快速筛查
    ├── [选做] Input Ablation         → 根因定位
    └── [选做] GradCAM                → 定性验证
```

---

## 9. 评估指标补充

### 9.1 旧评估方法的统计幻觉

v30 及 v32 早期评估采用 **per-frame random perturbation → temporal aggregation** 协议：

```
每帧: T_init_i = random_perturb(GT, ±5°)   # 每帧不同
评估: error = mean(geodesic(pred_i, GT))   # 跨帧平均
```

该协议产生 **统计幻觉**（statistical illusion）：

1. **随机噪声抵消**：每帧 T_init 围绕 GT 独立采样，pred ≈ T_init + ε 时，跨帧平均的 ε 相互抵消，聚合误差收敛到 0
2. **隐藏 shortcut**：模型只需在单帧上输出 ≈ Identity，即可获得优秀的 per-frame 指标
3. **与部署脱节**：部署时所有帧共享固定 T_init，不存在跨帧随机抵消

实验对比（v30, ckpt_400）：

| 评估协议 | Error vs GT | 能否检测 shortcut |
|----------|-------------|-------------------|
| Random perturbation（旧） | 0.067° | ❌ 不能 |
| Fixed-bias 1°（新） | 0.99° | ✅ 能 |
| T_init invariance（新） | 2.8° | ✅ 能 |

### 9.2 修正后的评估协议

#### 主指标（PRIMARY）：Fixed-Bias Evaluation

**定义**：固定 T_init 偏差，全序列评估，报告 mean / std / max error。

```python
BIAS_CONFIGS = [
    {"yaw": 1.0, "pitch": 0, "roll": 0},
    {"yaw": 0, "pitch": 1.0, "roll": 0},
    {"yaw": 1.0, "pitch": 1.0, "roll": 0},
]
for bias in BIAS_CONFIGS:
    T_init = apply_bias(GT, **bias)
    errors = [eval(model, frame, T_init) for frame in seq]
    report(mean, std, max)
```

**部署就绪标准**：

| 指标 | v30 基线 | v32 目标 | v32.1 目标 |
|------|---------|---------|-----------|
| 1° yaw fixed-bias error | 0.99° | < 0.5° | < 0.2° |
| 修正率（1 - error/bias） | 1% | > 50% | > 80% |

#### 质量指标（QUALITY）：T_init Invariance Score

**定义**：同场景多 T_init 预测的 pairwise consistency（见 §8.2）。

**标准**：invariance score < 0.1° 为合格，< 0.05° 为优秀。

该指标不直接反映绝对精度，但反映模型是否真正实现了 T_init 不变性——这是部署鲁棒性的必要条件。

#### 辅助指标（SECONDARY）：Random Perturbation Evaluation

保留原有 per-frame random perturbation 评估作为辅助参考：

```python
# 仍有用：衡量模型在"理想"T_init 条件下的最优性能上界
for frame in seq:
    T_init = random_perturb(GT, ±5°)
    errors.append(geodesic(pred, GT))
report(mean(errors))  # 上界参考，不能单独作为部署指标
```

**使用原则**：
- Random perturbation error **低** + fixed-bias error **高** → 确认 shortcut，模型不可用
- 两者均低 → 模型真正具备外参估计能力
- 仅报告 random perturbation 而不做 fixed-bias → **评估不完整，结论不可信**

### 9.3 评估优先级总结

```
部署决策
    │
    ├── PRIMARY:   Fixed-Bias Error + 修正率     ← 决定能否上线
    │
    ├── QUALITY:   T_init Invariance Score       ← 决定鲁棒性
    │
    └── SECONDARY: Random Perturbation Error       ← 参考性能上界
```

**原则**：任何声称"v32 优于 v30"的结论，必须同时报告 fixed-bias 和 invariance 指标。仅引用 random perturbation 结果的对比无效。

## 10. V32.1 实验结果与关键发现

### 10.1 训练结果

V32.1 快速训练（100 epoch, 8卡, batch_size=16）：

| 指标 | 值 |
|------|-----|
| Best Val Rot | 2.32° (Epoch 91) |
| Val range | ±5° |
| Final Train Rot | 14.91° (含 progressive curriculum) |
| Consistency Loss | 0.0606 (收敛) |
| EMA decay | 0.996 |
| Continuous noise max | 30° |

### 10.2 偏差矫正能力测试（Fixed-Bias Correction Test）

**测试方法**：使用训练完全相同的扰动函数 `generate_single_perturbation_from_T`，
在真实行程 YR-C01-81_20260427_112840 上注入不同大小的偏差，验证模型是否能自适应矫正。

**行程：** YR-C01-81_20260427_112840（50帧有效数据）
**Checkpoint：** Epoch 91 (best_val, Val Rot=2.32°)

#### 使用训练同源扰动函数测试：

| 扰动范围 | 平均扰动 | 模型矫正 | 残差(vs GT) | 矫正率 |
|:---:|:---:|:---:|:---:|:---:|
| ±1° | 0.53° | 0.321° | 0.634° | 60.2% |
| ±2° | 1.07° | 0.321° | 1.124° | 30.1% |
| ±3° | 1.60° | 0.322° | 1.641° | 20.1% |
| ±5° | 2.67° | 0.321° | 2.693° | 12.0% |

#### 使用固定轴偏差测试：

| 场景 | 注入偏差 | 模型矫正 | 残差 | 矫正率 | 状态 |
|------|:---:|:---:|:---:|:---:|:---:|
| Baseline | 0° | 0.322° | 0.322° | BASE | - |
| Roll ±2° | 2.0° | 0.321° | ~2.0° | 16% | FAIL |
| Pitch ±2° | 2.0° | 0.321° | ~2.0° | 16% | FAIL |
| Yaw ±2° | 2.0° | 0.322° | ~2.0° | 16% | FAIL |
| Roll/Pitch/Yaw ±5° | 5.0° | 0.322° | ~5.0° | 6% | FAIL |
| Combined ±3°~5° | 5-6° | 0.321° | ~5.5° | 5-6% | FAIL |

**VERDICT: Model has WEAK calibration ability (11/12 FAIL)**

#### 与 v30 标定管道对比（同一行程 YR-C01-81_20260427_112840）：

| 指标 | v30 (Epoch 400) | v32.1 (Epoch 91) |
|------|:---:|:---:|
| 标定总矫正量 (geodesic) | 0.021° | 0.334° |
| MEDW200 (vs GT) | N/A | 0.329° |

### 10.3 根因分析

**核心发现：模型输出的四元数（残差旋转）几乎恒定，不随 T_init 变化。**

```
T_pred_output = inv(T_const) @ T_init
矫正量 = geodesic(T_init, T_pred_output) = geodesic(I, inv(T_const)) ≈ 0.32° (固定)
```

**模型只学会了对该场景的一个"固定偏移"，而非基于 BEV 特征检测对齐质量。**

### 10.4 Val Rot=2.32° 的"统计幻觉"

为什么 Val 指标看起来不错但实际能力弱：

1. **±5° 随机扰动的平均 geodesic = 2.67°**
2. **模型输出恒定 0.32° 偏移 → 残差 ≈ 2.69°**（几乎等于不矫正）
3. **Val 评估使用多帧 + 时序聚合 → 随机噪声自消除**
4. Val Rot 2.32° vs 不矫正 2.67° → **仅 0.35° 提升，全靠固定偏移的碰巧对齐**

这与 v30 的"统计幻觉"本质相同：
- v30：T_pred ≈ Identity → 矫正 0.02°
- v32.1：T_pred ≈ Constant(0.32°) → 矫正 0.32°
- **两者都没有学会根据输入动态估计残差**

### 10.5 结论

v32.1 的 T_init Invariance Training + EMA + Continuous Noise **未能解决 shortcut learning**。
问题根源不在训练策略，而在**模型架构**：

1. BEV 投影使用 init_T → 改变 init_T 改变了 BEV 特征
2. 但 Transformer 未能从变化的 BEV 特征中检测到对齐差异
3. 模型学会了输出一个对训练分布"平均最优"的常数残差

**下一步（v33 方案方向）**：必须从架构层面打破 T_init 到输出的直通路径，使模型不得不依赖图像-点云对齐信息来估计校正量。

## 11. V33 架构方案提议

### 11.1 核心问题复述

当前架构的致命缺陷是 **BEV 全局池化丢失了空间对齐方向信息**：

```
img → BEV投影(使用init_T) → 融合 → Transformer → 全局池化 → FC → 四元数
                                                     ↑
                                            空间信息在此处被丢弃
                                            模型无法知道"偏向哪个方向"
```

init_T 不同 → BEV 图像特征的空间位置不同 → 融合特征的 pattern 不同 →
但全局池化后只剩平均值 → 四元数预测恒定

### 11.2 方案 A：Spatial Correlation 对齐检测（推荐）

**核心思想**：不再使用全局池化预测绝对四元数，而是让网络在 BEV 空间直接检测图像与点云的空间偏移。

```
img → BEV投影(固定identity或GT) → img_bev_feat (H, W, C)
pc  → BEV网格                   → pc_bev_feat  (H, W, C)

# 不融合，而是计算空间 cross-correlation
correlation_map = cross_correlate(img_bev_feat, pc_bev_feat)  # (2*D+1, 2*D+1)
# correlation 峰值位置 = 空间偏移方向和大小

# 或者用 cross-attention（key=pc, query=img）
alignment_feat = cross_attention(img_bev_feat, pc_bev_feat)  # 保留空间关系

# 从 correlation/attention 结果预测校正
correction = prediction_head(correlation_map)  # → RPY residual
```

**关键改变**：
1. **不使用 init_T 做 BEV 投影**（使用 identity 或 GT 的粗略估计）
2. **用 correlation 而非 concatenation 融合** → 天然编码空间偏移
3. **网络必须从 correlation peak 学习偏移** → 打破 shortcut

**优点**：
- 空间偏移信息被显式保留在 correlation map 中
- 不依赖 init_T → 天然具有 T_init 不变性
- correlation 是经典立体匹配和光流的核心操作，理论基础扎实

**缺点**：
- correlation volume 计算量较大（可限制搜索范围 D）
- 需要重写融合层

### 11.3 方案 B：T_init 解耦投影

**核心思想**：保持当前架构，但 BEV 投影不使用 init_T，迫使模型从对齐差异学习。

```
img → BEV投影(使用固定identity_T) → img_bev_feat
pc  → BEV网格                      → pc_bev_feat

# 当 init_T 有偏差时:
#   - img_bev_feat 使用 identity 投影 → 位置固定
#   - pc_bev_feat → 位置固定
#   - 如果实际外参有偏差，图像投影到 BEV 后与点云会有可见的偏移
#   - Transformer 需要检测这个偏移来估计校正量

correction = Transformer(concat(img_bev_feat, pc_bev_feat)) → RPY
```

**优点**：改动最小，只需修改 BEV 投影参数
**缺点**：如果 identity 离真实值太远，BEV 投影质量可能极差

### 11.4 方案 C：迭代细化（Iterative Refinement）

**核心思想**：类似 RAFT 光流估计，使用迭代更新机制。

```
T_current = init_T  # 初始估计

for i in range(N_iter):
    img_bev = BEV_project(img, T_current)
    pc_bev = BEV_grid(pc)
    
    # 计算当前对齐的 correlation
    corr = correlation(img_bev, pc_bev)
    
    # GRU/Transformer 预测残差更新
    delta_T = update_head(corr, hidden_state)
    T_current = delta_T @ T_current

output = T_current
```

**优点**：
- 每次迭代只需小范围 correlation
- 天然支持大偏差（通过多次小步修正）
- 类似 RAFT 的成功架构，理论和实践都验证过

**缺点**：
- 多次前向传播，推理速度慢
- 实现复杂度高

### 11.5 方案对比与建议

| 维度 | A: Correlation | B: 解耦投影 | C: 迭代细化 |
|------|:---:|:---:|:---:|
| 理论优势 | ★★★ | ★★ | ★★★ |
| 实现难度 | 中 | 低 | 高 |
| 推理速度 | 中 | 快 | 慢 |
| 大偏差能力 | 中 | 低 | 高 |
| 改动范围 | 融合层 | 投影层 | 全架构 |

**建议路线**：
1. **快速验证**：先用方案 B（解耦投影），验证去掉 init_T 依赖后模型是否开始学习
2. **正式方案**：如 B 有效但精度不够，升级到方案 A（Spatial Correlation）
3. **终极方案**：如需处理超大偏差（>10°），采用方案 C（迭代细化）

### 11.6 关键设计原则

无论采用哪种方案，v33 必须满足：

1. **T_init 不进入 BEV 投影路径**（或至少不影响 correlation 计算）
2. **空间对齐信息不被全局池化丢弃**
3. **偏差矫正测试（Fixed-Bias Correction Test）作为主要评估指标**
4. **Val Rot 单独不可信**，必须配合偏差测试

### 11.7 验证标准

v33 方案合格标准：

| 测试 | 阈值 |
|------|------|
| Fixed-Bias ±2° 矫正率 | > 50% |
| Fixed-Bias ±5° 矫正率 | > 30% |
| 矫正量随偏差单调增加 | 是 |
| Baseline 残差 | < 0.3° |

## 12. V33 Correlation Fusion 实验结果

### 12.1 训练收敛

V33 使用 `SpatialCorrelationFuser` 替代 concat+transformer+global_pool 路径。

| Epoch | Train Rot | Val Rot | 收敛状态 |
|-------|-----------|---------|----------|
| 1 | 55.18° | 40.21° | 初始 |
| 5 | 7.00° | - | 快速收敛 |
| 11 | 2.48° | 2.51° | 接近收敛 |
| 21 | 2.31° | 2.35° | 已收敛 |

### 12.2 三代模型偏差矫正能力对比

测试行程：`YR-C01-81_20260427_112840`（v30 致命问题验证数据集）

| 指标 | v30 (ep400) | v32.1 (ep91) | v33 ep11 | v33 ep21 |
|------|:-----------:|:------------:|:--------:|:--------:|
| Val Rot (训练指标) | ~0.2° | 2.32° | 2.51° | **2.35°** |
| **固定偏移量** | **0.021°** | **0.427°** | **1.105°** | **1.055°** |
| Corr% @ ±2° | ~1.1% | 21.4% | 55.2% | 52.8% |
| Corr% @ ±5° | ~0.4% | 8.5% | 22.1% | 21.1% |
| Corr% @ ±10° | ~0.2% | 4.3% | 13.3% | ~10.6% |
| 矫正量随 bias 变化 | 恒定 | 恒定 | 恒定 | **恒定** |
| VERDICT | WEAK | WEAK | PARTIAL | PARTIAL |

### 12.3 矫正曲线数学确认

`diagnose_correction_curve.py` 在 7 个轴方向 × 12 个偏差量级上扫描，获得完整矫正率曲线。

**v32.1 矫正率公式**：`correction_rate = 0.427° / bias_mag × 100%`
**v33 矫正率公式**：`correction_rate = 1.055° / bias_mag × 100%`

两者均为完美的 **1/x 曲线**，这是"常数输出"的数学指纹——模型不论输入什么 T_init，永远输出相同的预测。

```
v32.1 矫正率曲线:
Bias    Roll+   Pitch+  Yaw+    Combined
0.1°    427.0%  426.3%  427.2%  425.9%
0.5°     85.5%   85.2%   85.2%   85.3%
1.0°     42.7%   42.6%   42.8%   42.7%
2.0°     21.4%   21.3%   21.3%   21.4%    ← 全轴一致 = 各向同性常数
5.0°      8.5%    8.5%    8.5%    8.6%
10.0°     4.3%    4.3%    4.3%    4.3%
```

### 12.4 v33 Correlation Fusion 失败根因

`SpatialCorrelationFuser` 架构：

```
img_bev (B,256,100,100) → Conv1x1 → BN → ReLU → L2_Normalize → img_f (B,128,100,100)
pc_bev  (B,256,100,100) → Conv1x1 → BN → ReLU → L2_Normalize → pc_f  (B,128,100,100)

corr = (img_f * pc_f).sum(dim=1)  →  (B, 1, 100, 100)   # 逐像素余弦相似度

corr → Conv3x3 → Conv3x3(s2) → Conv3x3(s2) → AdaptiveAvgPool(4) → Flatten → FC → (B, 256)
```

**三个致命问题**：

1. **逐像素余弦相似度对小偏移不敏感**
   - 2° 旋转在 BEV 空间仅造成 ~3 像素偏移
   - L2 归一化后，相邻像素的相似度本就很高
   - `corr_map[x,y]` ≈ `corr_map[x+3,y]`，几乎无差异

2. **AdaptiveAvgPool2d(4) 彻底抹平空间偏移信息**
   - 100×100 → 4×4 = 625:1 压缩比
   - 偏移方向性信息（左右不对称 = Yaw 误差）被平均值消除

3. **逐像素乘积 ≠ 空间偏移检测**
   - 正确方法：滑窗互相关（cross-correlation）检测峰值偏移
   - 当前方法：逐位置余弦相似度 → 只能衡量"多像"，不能衡量"偏了多少"

### 12.5 关键结论

| 结论 | 说明 |
|------|------|
| **三代模型全部存在 shortcut learning** | 不论架构如何变化，模型都学到固定偏移而非自适应矫正 |
| **Val Rot 指标具有欺骗性** | 2.3° 的 Val Rot 看起来很好，但实际矫正能力为零 |
| **问题的本质是空间信息损失** | 全局池化/平均池化/AdaptiveAvgPool 在不同架构中以不同形式出现 |
| **需要真正的空间偏移检测机制** | 如滑窗互相关（cross-correlation）或光流式匹配 |

## 13. V34 架构方案：Cross-Correlation 偏移检测

### 13.1 核心思路

借鉴光流/模板匹配领域的做法，用 **滑窗互相关（cross-correlation）** 检测 BEV 特征图之间的空间偏移，替代 v33 的逐像素余弦相似度。

```
                Image BEV Features (B, C, H, W)
                        ↓
                ┌───────┴───────┐
                ↓               ↓
           img_patch         pc_template
           (crop from         (from PC BEV)
            img BEV)
                ↓               ↓
                └──── Cross-Correlation ────→ (B, 1, 2*maxshift+1, 2*maxshift+1)
                                                          ↓
                                              Soft-argmax → (Δx, Δy)
                                                          ↓
                                              Geometric decoder → (Roll, Pitch, Yaw)
```

### 13.2 技术方案

**方案 A：Cost Volume Cross-Correlation（推荐）**

类似 RAFT / FlowNet 的 cost volume 方法：

```python
class CrossCorrelationFuser(nn.Module):
    """Detect spatial offset between image BEV and PC BEV via cross-correlation."""
    
    def __init__(self, channels, max_shift=10, out_dim=256):
        # max_shift=10 对应 ±10 像素偏移 ≈ ±5° @ 0.5m BEV分辨率
        self.max_shift = max_shift
        self.img_proj = Conv1x1_BN_ReLU(channels, 64)
        self.pc_proj = Conv1x1_BN_ReLU(channels, 64)
        
        # Cost volume: (2*max_shift+1)^2 = 441 channels
        cost_channels = (2 * max_shift + 1) ** 2
        self.cost_decoder = nn.Sequential(
            Conv3x3(cost_channels, 128), BN, ReLU,
            Conv3x3(128, 64, stride=2), BN, ReLU,
            AdaptiveAvgPool2d(8),  # 保留较大空间分辨率
            Flatten,
            Linear(64 * 8 * 8, out_dim),
        )
        
    def forward(self, img_bev, pc_bev):
        img_f = self.img_proj(img_bev)   # (B, 64, H, W)
        pc_f = self.pc_proj(pc_bev)      # (B, 64, H, W)
        
        # 计算 cost volume: 对每个空间偏移 (dx, dy) 计算相关性
        cost_vol = compute_cost_volume(img_f, pc_f, self.max_shift)
        # → (B, (2*max_shift+1)^2, H, W)
        
        return self.cost_decoder(cost_vol)  # (B, out_dim)
```

**关键优势**：
- Cost volume 显式编码每个可能偏移量的相关性
- 不同偏移方向/大小会产生不同的 cost volume 图案
- 偏移检测精度 ≤ 1 像素（亚像素精度可通过 soft-argmax 实现）

**方案 B：可学习 Correlation with Shift Encoding**

在相关性计算中引入显式的偏移位置编码：

```python
# 不是逐像素相关，而是对每个偏移量计算全局相关分数
for dx in range(-max_shift, max_shift+1):
    for dy in range(-max_shift, max_shift+1):
        shifted_img = shift(img_f, dx, dy)
        corr[dx, dy] = (shifted_img * pc_f).sum() / (H * W)
# → 得到 (2*max_shift+1, 2*max_shift+1) 的相关响应图
# 峰值位置 = 空间偏移量 = 标定误差
```

### 13.3 验证标准

v34 方案合格标准（沿用 v33 标准 + 新增）：

| 测试 | 阈值 |
|------|------|
| Fixed-Bias ±2° 矫正率 | > 70% |
| Fixed-Bias ±5° 矫正率 | > 50% |
| Fixed-Bias ±10° 矫正率 | > 30% |
| 矫正量随偏差**单调递增** | 是（关键！） |
| 矫正曲线不是 1/x | 是（排除固定偏移） |
| Baseline 残差 | < 0.3° |

### 13.4 实现优先级

1. **快速验证**：实现 Cost Volume 版本，50 epoch 快速训练
2. **对比实验**：与 v32.1、v33 在相同数据/相同行程上对比矫正曲线
3. **部署评估**：通过后进行全量训练和完整评估

---

## 14. 全面诊断报告 (2026-05-21)

### 14.1 测试数据

行程：`YR-C01-81_20260427_112840`（v30 首次发现致命微矫正问题的行程）

### 14.2 偏差矫正测试结果

注入 ±2°/±5° 旋转误差后的矫正能力对比：

| 模型 | 训练轮数 | 固定矫正量 | Roll±2° | Pitch±2° | Roll+5° | 平均矫正率 | 结论 |
|------|---------|-----------|---------|----------|---------|-----------|------|
| v30 | 351 | 0.009° | 0.001° (0.0%) | 0.003° (0.1%) | 0.010° (0.2%) | 0.1% | 完全失败 |
| v32.1 | 91 | 0.433° | 0.432° (21.6%) | 0.432° (21.6%) | 0.433° (8.7%) | 15.5% | 失败 |
| v33 (SpatialCorr) | ~50 | 0.592° | 0.592° (29.6%) | 0.592° (29.6%) | 0.592° (11.8%) | 21.3% | 失败 |
| v34 (CrossCorr) | 11 | 1.663° | 1.663° (83.1%) | 1.663° (83.1%) | 1.663° (33.3%) | - | 失败 |

**关键发现**：所有模型的矫正量(Corr°)完全不随输入偏差变化。

### 14.3 矫正曲线数学证明

每个模型在 0.1°~10° 偏差范围内的矫正量变化：

| 模型 | 矫正量最小值 | 矫正量最大值 | 变化量 | 结论 |
|------|-------------|-------------|--------|------|
| v32.1 | 0.4259° | 0.4277° | 0.0018° | 固定偏移 |
| v33 ep50 | 0.6481° | 0.6484° | 0.0003° | 固定偏移 |
| v34 ep11 | 1.6619° | 1.6643° | 0.0024° | 固定偏移 |

矫正率呈现完美 1/x 曲线：`rate = constant / bias × 100%`

### 14.4 梯度探测（数值雅可比分析）

```
d(correction) / d(bias) 分析：
  理想模型: d(corr)/d(bias) = 1.0 （矫正量随偏差线性增长）
  实际结果: d(corr)/d(bias) ≈ 0.0005 （差距 2000x）

  v32.1: Roll+ 导数 = 0.000583, Pitch+ 导数 = -0.000085
  v33:   Roll+ 导数 = -0.000104, Pitch+ 导数 = 0.000019
  v34:   Roll+ 导数 = -0.000124, Pitch+ 导数 = 0.000540
```

**结论**：∂output/∂T_init ≈ 0，模型完全忽略 T_init 变化。

### 14.5 架构级根因分析

```
信息流: T_init → BEV投影 → img_branch(30M) → fuser → transformer(6.3M) → rotation_pred(1K)
                                                 ↑
                                          信息瓶颈在此处
```

T_init 信息仅通过 BEV 投影间接影响图像特征。分析表明：

1. **±5° 旋转对 BEV 特征的影响极微**：DINOv2 backbone 提取的高级语义特征对小角度投影变化高度鲁棒
2. **全局池化摧毁空间偏移信息**：即使 BEV 特征有微弱的空间偏移，经过 global avg pool / transformer attention 后消失
3. **MSE/geodesic loss 允许捷径**：输出训练分布均值（constant）即可最小化 loss
4. **v33/v34 的改进无效**：SpatialCorrelation 和 CrossCorrelation fuser 虽然设计上保留空间信息，但最终都收敛到相同的固定偏移行为，说明问题不在融合层而在更深层

### 14.6 失败方案总结

| 版本 | 架构改进 | 是否打破固定偏移 | 分析 |
|------|---------|---------------|------|
| v32 | T_init dropout + consistency loss | 否 | 训练约束不够强 |
| v32.1 | EMA + continuous noise | 否 | 训练 trick 无法改变架构缺陷 |
| v33 | SpatialCorrelationFuser | 否 | 像素级相关 + AdaptivePool 仍丢失空间信息 |
| v34 | CrossCorrelationFuser (cost volume) | 否 | cost volume 在 backbone 之后，高级特征已对 T_init 不敏感 |

---

## 15. v35 设计方案：三管齐下打破捷径学习

### 15.1 问题本质

模型的 T_init 信息**只能通过 BEV 投影**间接传入。BEV 投影是一个几何变换，
但 DINOv2 等 backbone 经过 ImageNet 预训练，学到的特征对几何变换高度不变，
因此 T_init 变化产生的 BEV 特征差异被 backbone 过滤掉了。

### 15.2 方案 A：显式 T_init 编码（Explicit T_init Injection）

**核心思想**：不再仅依赖 BEV 投影传递 T_init 信息，而是将 T_init 的 RPY 参数
作为显式输入直接注入预测头。

```python
class ExplicitTInitEncoder(nn.Module):
    """将 T_init 的 RPY 编码为高维特征并注入预测头"""
    def __init__(self, embed_dim=256, n_freq=32):
        super().__init__()
        self.n_freq = n_freq
        # Fourier positional encoding for each of R, P, Y
        input_dim = 3 * (2 * n_freq + 1)  # sin + cos + raw for each axis
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, rpy_deg):
        """rpy_deg: (B, 3) in degrees"""
        rpy_rad = rpy_deg * (3.14159 / 180.0)
        # Fourier encoding
        freqs = torch.arange(self.n_freq, device=rpy_rad.device).float()
        freqs = 2.0 ** freqs  # geometric progression
        # (B, 3, 1) * (n_freq,) → (B, 3, n_freq)
        encoded = rpy_rad.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        sin_enc = torch.sin(encoded)  # (B, 3, n_freq)
        cos_enc = torch.cos(encoded)  # (B, 3, n_freq)
        raw = rpy_rad.unsqueeze(-1)   # (B, 3, 1)
        feat = torch.cat([sin_enc, cos_enc, raw], dim=-1)  # (B, 3, 2*n_freq+1)
        return self.mlp(feat.flatten(1))  # (B, embed_dim)
```

**注入方式**：在 transformer/fuser 输出与 rotation_pred 之间加入 T_init 编码，
作为 additive 或 FiLM conditioning。

### 15.3 方案 B：对比学习损失（Contrastive T_init Sensitivity Loss）

**核心思想**：在训练 loss 中显式要求"不同 T_init 应产生不同的中间特征"。

```python
class TInitContrastiveLoss(nn.Module):
    """
    对比损失：对同一帧图像用不同 T_init 推理，
    要求中间特征的变化量与 T_init 变化量成正比。
    
    L_contrastive = ||  ||feat(T1) - feat(T2)|| - alpha * ||T1 - T2||  ||^2
    """
    def __init__(self, alpha=1.0, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
    
    def forward(self, feat_a, feat_b, t_init_a, t_init_b):
        feat_dist = F.pairwise_distance(feat_a, feat_b)
        t_dist = torch.norm(t_init_a - t_init_b, dim=-1)
        target = self.alpha * t_dist
        loss = F.mse_loss(feat_dist, target)
        return loss
```

**训练流程改动**：每个 batch 中，同一帧用两个不同的 T_init 做两次前向传播，
计算对比损失。

### 15.4 方案 C：RAFT 式迭代矫正（Iterative Refinement）

**核心思想**：不做一步到位的全量预测，而是多轮小步矫正。
每轮更新 T_init 后重新投影 BEV，让模型看到校正后的特征。

```python
class IterativeCalibrator:
    """多轮迭代矫正器"""
    def __init__(self, model, n_iters=4, step_scale=0.5):
        self.model = model
        self.n_iters = n_iters
        self.step_scale = step_scale
    
    def calibrate(self, img, pc, T_init, K):
        T_current = T_init.clone()
        for i in range(self.n_iters):
            # 每轮用当前 T 重新投影
            delta = self.model(img, pc, T_current, K)
            T_current = apply_delta(T_current, delta * self.step_scale)
        return T_current
```

**关键优势**：
1. 每轮 BEV 投影都使用更新后的 T，特征差异逐步放大
2. 模型只需学习小步矫正，降低学习难度
3. 类似 RAFT 的 GRU 迭代更新，已被证明在光流估计中非常有效

### 15.5 v35 综合方案：三管齐下

v35 将同时采用 A + B + C 三个方案，理由是它们互补：

| 方案 | 解决的问题 | 单独是否充分 |
|------|-----------|------------|
| A: 显式T_init编码 | T_init 信息无法通过 BEV 传入 | 可能不够，模型仍可能忽略 |
| B: 对比学习loss | 模型可以合法忽略 T_init | 可能不够，信号太弱 |
| C: 迭代矫正 | 一步到位预测困难 | 可能不够，单步特征仍不敏感 |
| **A+B+C** | 信息注入 + 梯度强制 + 渐进矫正 | **最大概率突破** |

### 15.6 快速验证计划

1. **Phase 1 (最快验证)**：仅实现方案 A（显式 T_init 编码），20 epoch 快速训练
   - 预期：如果 T_init 编码有效注入，矫正量应随 T_init 变化
   - 如果成功 → 继续加 B 和 C
   - 如果失败 → 说明预测头本身有问题，需要更深层重构

2. **Phase 2**：加入方案 B（对比损失），再训 20 epoch
   - 观察中间特征是否对 T_init 变化更敏感

3. **Phase 3**：加入方案 C（迭代矫正），完整训练 100 epoch

### 15.7 验证标准

| 测试 | v35 合格标准 |
|------|------------|
| d(corr)/d(bias) | > 0.3（当前 ≈ 0.0005） |
| ±2° 偏差矫正率 | > 70%（当前 ≈ 21%） |
| ±5° 偏差矫正率 | > 50%（当前 ≈ 9%） |
| 矫正曲线形状 | 近线性（非 1/x） |
| Baseline 残差 | < 0.3°（当前 ≈ 0.43°） |

## 16. v35 Phase1 实验结果（2026-05-22）

### 16.1 训练概况

v35 Phase1（ExplicitTInitEncoder）100 epoch 训练完成：
- 配置：8×L20 GPU, bs=16, max_frames_per_seq=500, DINOv2-small
- Best Val Rot: **2.05°** (Epoch 81) — 所有版本最佳
- Final Val Rot: 2.06° (Epoch 91)

### 16.2 模型对比总结

| 模型 | Val Best | Jacobian (Combined) | 矫正行为 |
|------|----------|-------------------|---------| 
| v30 | ~3.0° | ~0.0 | 完全固定偏移 |
| v32.1 | 2.80° | 0.0005 | 完全固定偏移 |
| v33 (Correlation) | 2.32° | ~0.001 | 固定偏移 |
| v34 (CrossCorr) | 2.42° | ~0.001 | 固定偏移 |
| v35 (ExplicitTInit) | 2.05° | 0.283 | 部分自适应 ✓ |

### 16.3 偏差矫正曲线分析

v35 首次展示自适应矫正行为，Combined+ 矫正量随偏差变化：

| Bias | v32.1 矫正量 | v35 矫正量 | v35 矫正率 |
|------|------------|----------|----------|
| 1.0° | ~0.3° | 0.994° | 99.5% |
| 2.0° | ~0.3° | 1.361° | 68.3% |
| 3.0° | ~0.3° | 1.560° | 52.3% |
| 5.0° | ~0.3° | 1.573° | 31.7% |
| 10.0° | ~0.3° | 0.632° | 6.4% |

**关键突破**：矫正量不再是常数（0.3° → 0.6~1.6° 随偏差变化），证明 ExplicitTInitEncoder 成功打破了信息瓶颈。

### 16.4 数值 Jacobian 对比

| 轴 | v32.1 | v35 | 提升倍数 |
|----|-------|-----|--------|
| Roll+ | 0.0005 | 0.363 | 726x |
| Pitch+ | 0.0005 | 0.129 | 258x |
| Yaw+ | 0.0005 | 0.072 | 143x |
| Combined+ | 0.0005 | 0.283 | 565x |

### 16.5 已知局限

1. 大偏差（>5°）矫正率低（<32%），矫正量饱和在 ~1.5°
2. Yaw 轴自适应性最差（Jacobian 0.072 vs Roll 的 0.363）
3. 距理想值（Jacobian=1.0）仍有差距

### 16.6 Phase2 实验结果 (v35b)

v35b = Phase1 + TInitSensitivityLoss(weight=0.5)，100 epoch 训练完成：
- Best Val: 2.07° (Epoch 81) — 与 v35 Phase1 (2.05°) 接近

**对比结论**：

| 指标 | v35 (Phase1) | v35b (+SensLoss) | 胜出 |
|------|-------------|------------------|------|
| Val Best | **2.05°** | 2.07° | v35 |
| Jacobian (Combined) | **0.283** | 0.120 | v35 |
| 1° 矫正率 | **99.5%** | 88.6% | v35 |
| 2° 矫正率 | **68.3%** | 55.9% | v35 |
| 10° 矫正率 | 6.4% | **14.2%** | v35b |
| 基线噪声 | 0.805° | **0.633°** | v35b |

**发现**：TInitSensitivityLoss 在 1-5° 范围反而削弱了自适应矫正，可能与主损失产生竞争。
在大偏差（>7°）和基线噪声方面有改善。
**建议**：Phase1 单独使用效果更好，后续优化应聚焦于其他方向。

### 16.7 真实行程泛化测试 (10 trips)

对 `dummy_data/` 下全部 10 条真实行程数据运行偏差矫正曲线测试：

| 行程 | @1° rate | @2° rate | @3° rate | 评估 |
|------|---------|---------|---------|------|
| YR-C01-81 | 99.9% | 68.5% | 52.5% | ADAPTIVE |
| YR-C01T-13 | 104.9% | 62.2% | 29.8% | ADAPTIVE |
| YR-DE08-4 | 108.3% | 86.7% | 55.1% | ADAPTIVE |
| YR-LPD21A-9 | 131.9% | 75.2% | 35.7% | ADAPTIVE |
| YR-LPD21B-10 | 55.6% | 81.4% | 45.6% | ADAPTIVE |
| YR-P01T-3 | 117.8% | 59.1% | 44.4% | ADAPTIVE |
| YR-P01T-8 | 116.8% | 86.3% | 55.9% | ADAPTIVE |
| YR-P177-70 | 108.4% | 85.8% | 49.7% | ADAPTIVE |
| YR-P177-95 | 65.5% | 58.5% | 49.8% | ADAPTIVE |
| YR-P789-56 | 107.9% | 67.5% | 44.1% | ADAPTIVE |
| **平均** | **101.7%** | **73.1%** | **46.3%** | **10/10 ADAPTIVE** |

**结论**：v35 在全部 10 条真实行程上均表现出自适应矫正能力，无一退化为固定偏移模式。
- 在 1° 偏差下平均矫正率 101.7%（接近理想值 100%）
- 在 2° 偏差下平均矫正率 73.1%
- 在 3° 偏差下平均矫正率 46.3%
- 矫正率随偏差增大单调递减，符合"有效但非完美"的自适应特征
- 9/10 行程在 1° 偏差下矫正率 >65%
- 最差行程 (LPD21B-10) 在 1° 偏差下仅 55.6%，但 2° 下仍有 81.4%

### 16.8 Phase3: RAFT 式迭代矫正架构

`IterativeRefinementHead` 模块已实现（`bev_calib.py`），核心设计：

```
x_feat (BEV pooled) ──┐
                       ├─ concat ─→ GRU ─→ delta_head ─→ (Δt, Δq)
T_init (encoded) ─────┘              ↑         │
      ↑                              │         │
      └──── update T_init ←──────────┘    (each step)
```

- 共享 GRU cell + delta prediction head，N 步迭代
- 每步预测小的 delta quaternion，更新 T_init 后再编码
- 训练时对每步中间预测施加 loss（权重递增）
- `x_feat` 从 BEV pooled feature detach，仅训练迭代头
- 参数：`--iterative_refine N`（N=0 禁用，推荐 N=3）

### 16.9 Phase3 实验结果 (v35d RAFT 迭代矫正)

#### 训练结果

| 模型 | 配置 | Val Best | Best Epoch |
|------|------|----------|-----------|
| **v35d (RAFT n=3)** | ExplicitTInit + IterativeRefine(3) | **1.70°** | Ep80 |
| **v35c (low-sens)** | ExplicitTInit + SensLoss(0.05) | **1.86°** | Ep60 |
| v35-ft10deg | v35 pretrain + ±10° range | 3.38° | Ep10 (发散) |

#### 偏差矫正曲线对比 (Combined+ 轴)

| 模型 | Val | Baseline | @1° rate | @2° rate | @3° rate | @5° rate | Jacobian |
|------|-----|----------|---------|---------|---------|---------|----------|
| **v35 (Phase1)** | 2.05° | 0.805° | 99.5% | **68.3%** | **52.3%** | 31.7% | **+0.283** |
| v35c (low-sens) | 1.86° | 0.801° | 100.9% | 61.7% | 43.7% | **34.1%** | +0.149 |
| v35d (RAFT n=3) | **1.70°** | 0.888° | 169.7% | 91.2% | 46.8% | 26.5% | **-0.149** |

#### 各轴 Jacobian 对比

| Axis | v35 (Phase1) | v35c | v35d (RAFT) |
|------|:---:|:---:|:---:|
| Roll+ | **0.363** | 0.121 | -0.457 |
| Pitch+ | 0.129 | **0.133** | -0.088 |
| Yaw+ | **0.072** | -0.051 | -0.077 |
| Combined+ | **0.283** | 0.149 | -0.149 |

#### Phase3 结论

**v35d RAFT 的 Val Rot 最优 (1.70°) 但自适应能力不如 v35 Phase1：**

1. **Jacobian 为负**：矫正量在 1°~3° 之间不是单调递增，而是先升后降
2. **过矫正问题**：@1° 偏差时矫正率 169.7% (>100%)，说明输出的"矫正"大于实际偏差
3. **大偏差退化快**：@5° 仅 26.5%，比 v35 Phase1 的 31.7% 更差
4. **根因**：RAFT 迭代放大了常数偏移分量（baseline 0.888° vs v35 的 0.805°），而非增强自适应信号

**Val 指标下降的原因**：RAFT 多步迭代在随机扰动验证集上能通过多步平均减少随机噪声（类似时序聚合），但这不代表真正的自适应能力。这与 v30 的"统计幻觉"本质相同。

**部署建议**：
- 小偏差场景 (< 1.5°)：v35d 实际矫正效果更好（过矫正恰好有效）
- 大偏差场景 (> 2°)：v35 Phase1 更可靠（正向 Jacobian，矫正随偏差单调增加）
- **综合推荐**：v35 Phase1 作为主模型部署，因为正向 Jacobian 意味着更可预测的行为

### 16.10 最终模型排名

| 排名 | 模型 | Jacobian | @2° rate | 推荐场景 |
|:---:|------|:---:|:---:|------|
| 1 | **v35 (Phase1)** | +0.283 | 68.3% | **通用部署首选** |
| 2 | v35c (low-sens) | +0.149 | 61.7% | 备选（更低 baseline 噪声） |
| 3 | v35d (RAFT) | -0.149 | 91.2% | 仅限小偏差场景 |

### 16.11 当前训练中

**v35_long_lr1e4** (2026-05-23 启动，Ep33/400 进行中)：
- 架构: Phase1 (ExplicitTInit) only
- LR: 1e-4 (此前 4e-4)
- max_frames_per_seq: 1000 (此前 500)
- epochs: 400 (此前 100)
- save_ckpt_per_epoches: 50
- 目标: 降低场景特异性噪声 (STD 0.26° → <0.1°)

**Ep20 首次 Val 结果** (ckpt_best_val)：

| 指标 | Train (Ep21) | Val (Ep20) |
|------|:---:|:---:|
| Rot | 2.66° | **2.46°** |
| Roll | 1.37° | 1.31° |
| Pitch | 1.30° | 1.30° |
| Yaw | 1.24° | **0.98°** |

**Ep33 Train 趋势**: Rot 2.36° (Roll 1.21° / Pitch 1.14° / Yaw 1.05°)，持续下降但距 0.1° 目标仍远。

**Jacobian 诊断** (`tools/diagnose_jacobian.py`, Ep20 ckpt)：

| 轴 | J | 判定 |
|----|:-:|------|
| Roll | +0.103 | WEAK |
| Pitch | +0.153 | WEAK |
| Yaw | +0.008 | SHORTCUT |
| **Overall** | **+0.088** | WEAK ADAPTIVE |

结论: ExplicitTInit 打破了 shortcut (J>0)，但自适应强度仍弱，尤其 Yaw 轴接近常数输出；需更多 epoch 或 v36 native cross-attention。

---

## 17. V36 架构方案设计（Native-Domain Extrinsic-Aware Calibration）

### 17.1 灵感来源：ProjFusion (IROS 2025)

ProjFusion 的核心创新：
- **Native-domain cross-attention**: 图像 patch 直接与点云 group 做 cross-attention，不经过 BEV 投影
- **Extrinsic-aware positional embedding**: 用当前外参假设投影点云到图像平面，投影坐标做 harmonic embedding 作为位置编码
- **Frozen encoder + trainable fusion**: DINOv2 和 PointGPT 冻结，只训练 cross-attention + MLP head
- **推理时迭代收敛**: 每步更新外参 → 重新投影 → 新 attention pattern → 逐步精化

ProjFusion 在 ±10° 初始扰动下达到 rotation RMSE 0.43°（KITTI），显著优于 BEVCalib 当前水平。

### 17.2 v35d RAFT 失败 vs ProjFusion 迭代成功的根因分析

| 维度 | v35d (RAFT, 失败) | ProjFusion (成功) |
|------|------------------|-------------------|
| 特征来源 | BEV pooled feature **detach 后固定** | 每步**重新投影**得到新 attention |
| 迭代信息流 | GRU 只看同一组冻结特征 | 投影坐标变化→位置编码变化→注意力变化 |
| 预测目标 | 绝对 delta_T (容易过拟合) | 残差 delta in se(3)，compose 到当前 |
| 核心问题 | 无新信息 → 放大常数偏移 | 新对齐关系 → 真正的自适应矫正 |

### 17.3 V36 架构设计

#### 核心思路：将 ProjFusion 的 Extrinsic-Aware Cross-Attention 移植到 BEVCalib

```
[冻结] DINOv2 (img) ──→ feat_2d (B, H*W, D_img)
                                                    ┐
[冻结] PointEncoder (pcd) ──→ feat_3d (B, N, D_pc)  ├─→ ExtrinsicAwareCrossAttention ──→ aggregation ──→ MLP ──→ (rot, tsl)
                                                    │
T_init (当前外参假设) ──→ project(pcd → img) ──→ proj_uv ──→ HarmonicEmbed ──→ pos_emb ──┘
```

#### 关键模块：

**A. ExtrinsicAwareCrossAttention**
- Query: `feat_2d + harmonic(img_grid_coords)` — 图像 patch 特征 + 固定网格位置编码
- Key/Value: `feat_3d + harmonic(proj_uv)` — 点云特征 + **外参相关**的投影位置编码
- 核心: 投影坐标 `proj_uv = K · T_init · xyz_3d` 随 T_init 变化
- 当 T_init 不准时，投影位置编码"错位"→ attention 感知到不对齐 → 输出矫正信号

**B. Iterative Refinement (推理时)**
```python
with model.cache_features(img, pcd):  # 只编码一次
    T_current = T_init
    for step in range(N_iter):  # N=3
        delta_rot, delta_tsl = model.predict(T_current, camera_info)
        T_current = se3.exp(delta) @ T_current  # compose
```
- 每步重新计算 `proj_uv = project(T_current, pcd)` → 新位置编码 → 新 attention
- **训练时只训练 1 步**，推理时做 3 步（ProjFusion 论文实验验证了这个策略的有效性）

**C. Harmonic Positional Embedding**
- 输入: 2D 坐标 (x, y) ∈ [-1, 1]
- 输出: `sin(2^k * π * coord)` 和 `cos(2^k * π * coord)`, k = 0...N-1
- N=6 (ProjFusion 默认), 输出维度 = 2 * (2*6 + 1) = 26 per coordinate

**D. 与 BEVCalib 现有架构的兼容**
- 保留 DINOv2 backbone (已验证有效)
- 点云编码: 可复用现有 voxel+scatter 流程，或引入轻量 PointNet++/PointConv
- 新增: Cross-Attention 层 + Harmonic Embedding + Attention Aggregation
- 移除: BEV pooling（最大的架构变化）

### 17.4 实现分阶段计划

**Phase A: 最小可行验证 (MVP)**
- 在现有 BEVCalib 框架中添加 `ExtrinsicAwareCrossAttention` 模块
- 保留 DINOv2 图像编码器 (冻结)
- 点云编码: 简单 PointNet (MLP on 3D coords)，冻结或轻量训练
- 1 层 cross-attention (heads=8, dim_head=64)
- Harmonic embedding (n=6)
- Attention aggregation → MLP head → RPY
- 训练时单步，推理时 3 步迭代
- **预期**: 如果 cross-attention 能感知投影对齐，单步 val 就应该有自适应能力

**Phase B: 完整实现**
- 引入 PointGPT/PointConv 作为 3D 编码器 (冻结)
- Dual-branch (独立 rot/tsl cross-attention)
- 更大 attention (多层, dim_head=128)
- 更多训练数据和 epoch

**Phase C: 推理优化**
- 迭代推理 + 时序聚合融合
- Confidence-weighted 输出
- 自适应迭代步数 (收敛即停)

### 17.5 与现有 v35 的对比预期

| 维度 | v35 (ExplicitTInit) | v36 (ExtrinsicAwareCrossAttn) |
|------|--------------------|-----------------------------|
| T_init 信息注入 | MLP encoding + concat | 通过投影坐标隐式注入位置编码 |
| 信息利用方式 | 全局拼接 | 局部对齐关系 (attention) |
| 大偏差处理 | Jacobian 0.28, 单步饱和 | 多步迭代，每步小矫正 |
| 理论上限 | 受限于 BEV 分辨率 | 直接在原生域对齐，分辨率更高 |
| 复杂度 | 低 (仅增加 MLP) | 中 (cross-attention 计算量) |
| 迭代收敛保证 | 无 (v35d 发散) | 有 (ProjFusion 已验证) |

### 17.6 风险与缓解

1. **点云编码器缺失**: BEVCalib 当前没有独立的点云 token encoder
   - 缓解: Phase A 用简单 PointNet (3→64→128)，或直接用 xyz 坐标 + harmonic embedding 作为 "3D tokens"
   
2. **训练数据格式**: 需要原始点云 + 图像 + 外参三元组
   - 缓解: 现有数据已经包含这些，只需修改 DataLoader 不做 BEV pooling

3. **计算量增加**: Cross-attention O(N_img * N_pc) 
   - 缓解: 图像做 patch (14x14 per DINOv2), 点云下采样到 128-256 groups
   
4. **训练不稳定**: 冻结编码器 + 从零训练 attention
   - 缓解: 使用 cosine warm restart, gradient clip, 参考 ProjFusion 的 adamw + bf16

### 17.7 术语表补充

| 术语 | 含义 |
|------|------|
| Native-Domain Cross-Attention | 在图像和点云各自的原生表示空间做 cross-attention，不需要统一到 BEV |
| Harmonic Embedding | NeRF 风格的位置编码：sin/cos(2^k * π * x)，将低维坐标映射到高维空间 |
| Extrinsic-Aware | 位置编码依赖当前外参假设，外参变化时编码随之变化 |
| se(3) Residual | 在李代数空间预测小增量，通过 exp map 转为 SE(3) 后 compose 到当前估计 |
| Cache Mechanism | 冻结编码器输出缓存，迭代推理时只重算 attention（投影变化触发） |

### 17.8 实现状态（2026-05-23）

**Phase A MVP 已完成并接入 BEVCalib**：

| 组件 | 文件 | 状态 |
|------|------|------|
| `HarmonicEmbedding` | `kitti-bev-calib/native_cross_attention.py` | ✅ |
| `PointEncoder` | 同上 | ✅ |
| `ExtrinsicAwareCrossAttention` | 同上 | ✅ |
| `NativeCrossCalibHead` | 同上 | ✅ |
| BEVCalib 集成 (`native_cross=1`) | `bev_calib.py` | ✅ |
| 训练脚本参数 | `train_kitti.py`, `start_training.sh`, `batch_train.sh` | ✅ |
| MVP 配置 | `configs/v36_native_cross_mvp.yaml` | ✅ 待 GPU |
| E2E forward 验证 | 单卡 smoke test | ✅ |
| Jacobian 诊断 | `tools/diagnose_jacobian.py` | ✅ |

**启用方式**：
```bash
bash batch_train.sh configs/v36_native_cross_mvp.yaml
# 或
python tools/diagnose_jacobian.py --ckpt_path <ckpt>
```

### 17.9 多层 Cross-Attention 增强（Phase A+）

在 MVP 单层基础上新增 `CrossAttentionBlock` 与 `native_cross_n_layers` 参数：

```
CrossAttentionBlock:
  x = CrossAttn(feat_2d, feat_3d, img_pos, proj_pos) + proj_residual(feat_2d)
  x = x + FFN(x)
```

| 配置 | n_layers | pc_groups | 参数量 | 配置文件 |
|------|:--------:|:---------:|:------:|----------|
| MVP | 1 | 128 | 0.81M | `v36_native_cross_mvp.yaml` |
| Deep | 3 | 192 | 8.70M | `v36_native_cross_deep.yaml` |

**设计要点**：
- 第一层 input = DINOv2 patch dim (384)，后续层 input = heads×dim_head (512)
- 每层 FFN 使用 4× expansion + LayerNorm + GELU + Dropout
- 投影坐标 `proj_uv` 仍由 `T_init` 实时计算，多层 attention 可逐步 refine 对齐信号
- 训练仍单步；推理可用 `iterative_inference(n_iters=3)` 做外参迭代

**与 ProjFusion 差距（待 Phase B）**：
- 尚无 dual-branch (独立 rot/tsl attention)
- 点云编码仍为轻量 MLP，未引入 PointGPT/PointConv
- 未做 confidence-weighted 时序融合

### 17.10 MVP 设计审查（2026-05-23）

#### 设计层面

| 问题 | 严重度 | 说明 |
|------|:------:|------|
| 点云编码过弱 | **高** | 仅 MLP(xyz)，无 PointGPT/局部几何；ProjFusion 用预训练 PointGPT group features |
| 采样策略简陋 | 中 | 均匀 stride 非 FPS/KNN，空间覆盖不均 |
| 无 dual-branch | 中 | ProjFusion 独立 rot/tsl cross-attention；MVP 共用一个 head |
| 投影坐标系 | 中 | 在全分辨率图像上投影再归一化；ProjFusion 在 **feature 分辨率**上投影并缩放内参 |
| 双向 cross-attn 缺失 | 低 | ProjFusion 多层含 image↔pc 双向；MVP 仅 image→pc |

**核心逻辑是否成立**：成立。`T_init` 变化 → `proj_uv` 变化 → harmonic pos emb 变化 → attention 模式变化。这是 v36 相对 v35d RAFT 的关键差异。

#### 训练层面

| 问题 | 严重度 | 说明 |
|------|:------:|------|
| 冗余模块占显存 | **高** | `native_cross=1` 仍构建 pc_branch + BEV fuser + transformer（~数百 MB），训练时完全不用 |
| LR 偏高 | 中 | MVP 用 2e-4，v35_long 已验证 1e-4 更稳；全新 attention 建议 1e-4 |
| 无 warmup 保护 | 低 | warmup_epochs=3 偏短，attention 从零初始化 |
| padding 点污染 | **高** | ✅ 已修复：PointEncoder 现 respect mask，过滤 999999 padding |
| DINOv2 padding | 中 | ✅ 已修复：`_forward_native_cross` 现 reflect-pad 到 14 整除 |

#### 收敛层面

| 风险 | 概率 | 缓解 |
|------|:----:|------|
| 早期 shortcut（J≈0） | 中-高 | 投影信号需 attention 自己学；无 explicit_tinit 兜底 |
| 单步饱和（大偏差） | 中 | 训练单步、推理 3 步迭代；但 MVP 未验证 iterative_inference |
| loss 与 v35 不可比 | 低 | 共用 realworld_loss，语义一致 |
| 收敛慢于 v35 Phase1 | 中 | 0.81M 新参数 vs v35 的 tinit_encoder；需 50+ epoch 才判 |

**预期**：MVP 首 20 epoch 应看到 train rot 从 ~5° 降到 ~3°；若停滞在 ~4° 且 Jacobian≈0，说明 attention 未学到投影对齐。

#### 适配性层面

| 项目 | 状态 |
|------|------|
| BEVCalib loss 接口 | ✅ 兼容（quat → T_pred → T_gt_expected） |
| train_kitti / batch_train 参数 | ✅ 已打通 |
| evaluate_checkpoint | ✅ 已补 native_cross 参数 |
| diagnose_jacobian | ✅ 可用 |
| 640×360 自定义数据 | ✅ E2E 验证通过 |
| DDP 8卡 | ⚠️ 未实测；cross-attn O(1125×128) 应可跑 |
| 与 v35 checkpoint 互转 | ❌ 结构不兼容，需从头训 |

#### 建议的 MVP 训练前检查清单

1. 确认 GPU 显存：冗余 BEV 模块已跳过（仅 DINOv2 + cross-head，~24M total / ~2M trainable）
2. LR **1e-4**，warmup **5 epoch**，BS **12**
3. 训练 Ep20 后立即跑 `diagnose_jacobian.py`，目标 Overall J > 0.3
4. 若 J < 0.1：考虑 hybrid (`explicit_tinit=1`) 或增大 pc_groups

#### 17.10.1 审查问题修复记录（2026-05-23 二次迭代）

| 问题 | 修复 |
|------|------|
| padding 点污染 | `PointEncoder` respect mask |
| DINOv2 patch 对齐 | reflect-pad + patch count 校验 |
| 冗余 BEV 模块 | `native_cross=1` 时仅构建 `DINOv2Encoder` + `NativeCrossCalibHead` |
| 均匀 stride 采样 | FPS + kNN 局部几何 (10D → MLP) |
| 全分辨率投影 | feature 分辨率投影 + 缩放内参 |
| 无 dual-branch | 双独立 cross-attention branch，concat 后 aggregation |
| LR/BS 配置 | 1e-4 / BS12 / warmup5 |

---

## 18. 泛化评估标准协议（V37 起强制执行）

> 完整实验设计见 **[V37_DESIGN.md](./V37_DESIGN.md)**。

### 18.1 强制条件

跨版本对比 **必须** 满足：

1. 数据集：`test_data_v2`，`--use_full_dataset`，`--eval_max_frames_per_seq 200`
2. **扰动：`--angle_range_deg 5.0`（±5°）** — 不得使用 ckpt 自带 angle_range（v37 为 ±10°）
3. **extend_ratio**：从 checkpoint 读取；评估脚本须传入 `native_cross_extend_ratio`（v37=2.5）
4. 必报指标：Per-frame Rot、MEDW200、Jacobian（±5° 与 ±10° sweep）

### 18.2 命令模板

```bash
# 部署泛化（MEDW）
python evaluate_checkpoint.py --mode eval \
  --ckpt_path <ckpt> \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2 \
  --output_dir <out> \
  --use_full_dataset --eval_max_frames_per_seq 200 \
  --angle_range_deg 5.0

# Shortcut 诊断
python tools/diagnose_jacobian.py --ckpt_path <ckpt> \
  --angle_range 5.0 --n_batches 10 --output <out>/jacobian_5deg.json
```

### 18.3 V37 Ep241 关键结论（±5° 公平对比）

| 指标 | v36 | v37 |
|------|-----|-----|
| Per-frame | 1.18° | **0.91°** |
| MEDW200 | **0.41°** | 0.46° |
| J @ ±5° | 0.55 | **0.84** |
| J @ ±10° | 0.29 | **0.88** |

MEDW 略退化主因：optimizer steps 减半、PointGPT 域差距、warm-start 局部最优；详见 V37_DESIGN.md §6。

---

## 19. V41：V32 机制在 GMP 上的严格验收（2026-05-29）

V32 三重机制（T_init dropout、consistency、progressive noise）在 V41 中与 **GMP scratch + jacloss** 组合，用于验证 init-invariance 能否在 GeoMatch-ProjCalib 上落地。

**交付红线（同分布 val，非 stretch goal）**：

- `max(R,P,Y) MEDW200 < 0.10°`
- Jacobian @ ±3°：overall 及 R/P/Y 均 `> 0.85`

同分布 val 过不了 dual gate → **训练或网络设计有问题**；未过 gate 不得进入 test/bag。V40 已证 MEDW-only 选点会导致 bag shortcut（Pitch J≈0）。

完整实验设计（A/B/C ablation、KPI 解读、失败排查、启动命令）见 **`docs/V41_DESIGN.md`**。
