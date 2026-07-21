# Recovery >95% 全路径设计: TTA + 自监督方案

## 域差异根因总结

经分析发现, test_data_c1 与 train 的域差异**不在图像级别**:
- 亮度/对比度/分辨率: 差异 <1%
- 内参 (fx/fy/cx/cy): 范围高度重叠

**真正的域差异在 mounting geometry (安装外参)**:
- Train: Roll ∈ [-93°, -87°], 23个安装配置
- Test: Roll ∈ [-92°, -88°], 12个安装配置
- D类 trip (seq03): 距最近训练 seq 1.0° → 直接映射为 ZD

**结论**: ZD 本质是"模型学到的默认修正方向"与实际安装不匹配。

---

## 方案 C: Test-Time Adaptation (TTA)

### 核心思想

在部署时, 利用目标域的少量帧对模型做轻量级在线适应, 使模型的"默认修正"对齐新安装。

### 方案 C1: ZD 滑窗补偿 (已实现, 最轻量)

```
Pipeline:
1. 收集前 30-50 帧的模型预测
2. 估计当前 trip 的 ZD 偏置
3. 后续帧的模型输出减去 ZD
4. 每 N 帧更新 ZD 估计 (滑窗)
```

- 优点: 零训练, 即插即用
- 缺点: 假设 ZD 在 trip 内恒定 (实际随场景变化)
- 预计效果: ZD从0.87°降到0.2-0.3° (不能完全消除)

### 方案 C2: Affine Head Adaptation (推荐)

```
Architecture:
  Model = Backbone(frozen) + Head(frozen) + Affine(trainable, 6 params)
  
  Output = Affine(Model_output) = scale * prediction + bias
  
Deploy TTA:
  1. 在第一段trip上 (100帧), 用自监督信号训练 Affine 层
  2. 自监督信号: 时序一致性 (连续帧预测应平滑)
  3. 训练完成后, Affine 吸收了 mounting bias
  4. 后续推理: 只需 Model + Affine (零额外推理开销)
```

**自监督信号设计**:
```python
# 时序一致性: 连续帧的标定结果应该一致
loss_temporal = MSE(pred[t], pred[t-1])  # 理想情况下连续帧标定相同

# 迭代一致性: 多轮推理应收敛到不动点
loss_fixpoint = MSE(model(apply(pred, input)), pred)

# 组合
loss_tta = loss_temporal + λ * loss_fixpoint
```

- 优点: 只训6个参数, 极快收敛 (<100帧), 零额外推理开销
- 缺点: 线性假设可能不够
- 预计效果: ZD从0.87°降到 <0.1° (直接对齐)

### 方案 C3: BatchNorm Adaptation (中等复杂度)

```
Deploy TTA:
  1. 在目标域数据上 forward 100帧 (不做反向传播)
  2. 更新 BatchNorm 的 running_mean/running_var
  3. 使模型的 BN 统计量适应新域
```

- 优点: 无需额外训练, 只需 forward
- 缺点: CF-BEV-R 可能没有足够的 BN 层 (SwinT用LayerNorm)
- 预计效果: 有限, 因为 LN 不存储 running stats

### 方案 C4: LoRA Adaptation (最强但最重)

```
Deploy TTA:
  1. 在 SwinT 最后2层加 LoRA (rank=4, ~2K params)
  2. 在目标域 100帧上用自监督 loss 训练 LoRA
  3. 推理时 LoRA merge 到权重 (零额外延迟)
```

- 优点: 最强表达力, 能适应复杂域变化
- 缺点: 需要在部署时训练, warm-up时间 ~30s
- 预计效果: 可将 ZD 降至 <0.05°

### TTA 方案推荐

```
优先级: C2 (Affine) > C1 (ZD补偿) > C4 (LoRA) > C3 (BN)

理由: 
- C2 兼具简洁和有效: 6参数对齐就能消除大部分 mounting bias
- C1 是兜底方案, 已实现
- C4 是C2不够时的升级路径
```

---

## 方案 E: 无 GT 自监督/半监督微调

### 为什么可以无 GT 训练?

标定任务有天然的自监督信号:
1. **时序一致性**: 同 trip 内标定结果应恒定 (车辆安装不变)
2. **光度一致性**: 正确标定后, LiDAR投影到图像应对齐
3. **几何一致性**: 正确标定后, 重投影误差最小

### 方案 E1: 自监督微调 (Self-Supervised Fine-Tuning)

```python
def self_supervised_loss(model, frames_batch):
    """无需 GT 的自监督 loss"""
    predictions = [model(frame) for frame in frames_batch]
    
    # Loss 1: 时序一致性 (同 trip 标定应相同)
    temporal_loss = variance(predictions)
    
    # Loss 2: 光度一致性 (LiDAR 投影后 RGB 应匹配)
    for pred in predictions:
        T_calibrated = apply_correction(pred, T_init)
        projected_pc = project(pcd, T_calibrated, intrinsic)
        photo_loss += photometric_consistency(image, projected_pc)
    
    # Loss 3: 不动点 (迭代推理应收敛)
    fixpoint_loss = ||model(apply(pred, input)) - 0||
    
    return temporal_loss + photo_loss + fixpoint_loss
```

**训练方式**:
```
1. 用 test_data_c1 的全量帧 (无GT) + 上述自监督 loss
2. 从 V62/V67 权重开始微调
3. 只更新最后几层 (或用 LoRA)
4. 训练 5-10 epoch
```

- 预计效果: ZD 大幅降低, Jacobian 略有改善
- 风险: 光度 loss 对遮挡/反射敏感

### 方案 E2: 伪标签半监督

```
Pipeline:
1. 用 V62 模型在 test_data_c1 上预测
2. 选择高置信度帧 (方差小、temporal一致的) 作为伪GT
3. 用伪GT做常规监督训练
4. 迭代: 新模型重新生成伪标签, 筛选, 再训练
```

- 优点: 可利用现有训练pipeline
- 缺点: 初始伪标签可能有 ZD bias → 需要去偏
- 预计效果: 如去偏成功, 可显著降ZD

### 方案 E3: Temporal Consistency Training (最推荐)

```
核心思想: 
  同一 trip 内, 相机安装不变
  所以正确的模型应该对同 trip 所有帧输出相同的标定参数

Loss = Var(predictions over trip) → 最小化 trip 内预测方差
     + λ * KL(pred_distribution, prior)  → 防止坍塌到零
```

**实现步骤**:
```python
# Step 1: 在 test_data_c1 上运行 V62, 收集每 trip 的预测分布
# Step 2: 计算每 trip 的 median prediction 作为"锚点"
# Step 3: 用 (锚点 - model_prediction) 作为监督信号
# Step 4: 在 train_data + test_data 混合训练
#   - train_data: 正常监督 loss
#   - test_data: temporal consistency loss
```

- 优点: 不需要任何 GT, 利用标定任务天然的时序不变性
- 缺点: 锚点本身有 ZD bias (但 median 可减少)
- 预计效果: 配合 V67 的迭代监督, 可将 test-domain ZD 降至 0.1-0.2°

---

## 综合路径: 如何稳定达到 >95% Recovery

### 最优路径 (依据投入/回报比)

```
Phase 1 (当前): V67 训练 + ZD补偿部署
  ├─ V67: 迭代监督+域增强+Jacobian loss
  ├─ mount_jitter_sigma=3.0 覆盖 1° mounting gap
  ├─ ZD 在线补偿 (方案C1, 已实现)
  └─ 预期: Recovery 85-92%

Phase 2 (如不够): Affine TTA (方案C2)
  ├─ 加6参数 Affine 层, 部署时 100帧适应
  ├─ 自监督: temporal+fixpoint consistency
  └─ 预期: Recovery 92-97%

Phase 3 (如仍不够): Temporal Consistency 自监督 (方案E3)
  ├─ 在 test_data_c1 上做无GT微调
  ├─ 从 V67 权重开始, 只更新最后2层
  └─ 预期: Recovery >97%

Phase 4 (终极保障): 加入 test_data_c1 的 GT 数据训练
  ├─ 哪怕 10% 的 trip 有 GT 也够
  └─ 预期: Recovery >99%
```

### 为什么这个路径可行?

根据域差异分析:
- 域差异只在 mounting (0.5-1°), 不在视觉特征
- mount_jitter=3° 的训练增强已覆盖此范围
- TTA 的 Affine 层可直接对齐 mounting offset
- Recovery = f(effective_Jacobian, ZD_residual, iterations)

数学上:
- 若 V67 将 test-domain J 从 0.39 提升到 0.65
- 若 TTA/ZD补偿将 ZD 从 0.87° 降到 0.15°
- 则 5 轮迭代: effective_residual ≈ 0.15 × (1-0.65)^5 + 3×(1-0.65)^5
  = 0.15 × 0.005 + 3 × 0.005 = 0.016° 
  → Recovery = (3-0.016)/3 = **99.5%** ✓
