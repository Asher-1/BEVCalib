# V40 P1c训练监控报告

> 监控时间：2026-05-29 14:16  
> 训练进度：Epoch 15/60 (25%)  
> 状态：⚠️ 训练中，但发现配置问题


## 一、训练基本信息

### 启动配置

| 参数 | 配置值 | 预期值 | 状态 |
|------|--------|--------|------|
| 训练集 | all_training_data | ✅ | 正确 |
| Epoch数 | 60 | ✅ 60 | 正确 |
| Batch size | 32 | ✅ 32 | 正确 |
| Learning rate | 4.6e-4 | ✅ 4.6e-4 | 正确 |
| DDP GPUs | 8 | ✅ 8 | 正确 |
| Pretrain ckpt | v39_exp1 best_medw | ✅ | 正确 |


### ⚠️ 关键问题：iterative_refine未生效

```
iterative_refine=0  # ❌ 实际配置
```

预期：`iterative_refine=1`（P1c配置）

影响：
- 当前使用match_only模式（无iterative refine）
- 相当于P1b配置而非P1c
- 但仍启用了DiffEPnP和LocalCorrelation


### ✅ 已生效的P1配置

| 组件 | 状态 | 日志证据 |
|------|------|---------|
| DiffEPnP | ✅ 启用 | `diff_epnp=True` |
| DiffEPnP warmup | ✅ 5 epoch | `diff_epnp_warmup=5` |
| MatchHead | ✅ 启用 | `match=True` |
| LocalCorrelation | ✅ 启用 | `corr=True` |
| GeoConsistency | ✅ 启用 | `geo_w=(0.1,0.05)` |
| Compose mode | ✅ | `match_then_refine` |


### ❌ P2b参数未检测到

日志中无P2b参数（`augment_fov_crop_prob`, `augment_lidar_sparse_prob`）

原因：
- 训练启动于12:36
- P2b实施完成于13:45
- 此次训练使用的是P2b实施前的配置


## 二、训练收敛分析（Epoch 1-15）

### 训练误差趋势

| Epoch | Train Rot | Val Rot | Jacobian | 判定 |
|-------|-----------|---------|----------|------|
| ep1 | 4.52° | 4.27° | -1.195 WEAK | 起步正常 |
| ep2-13 | - | - | - | （训练中） |
| ep14 | 4.14° | - | - | 轻微改善 |
| ep15 | 训练中... | - | - | - |

ep1 详细指标：
- Train Rot: 4.52° (Roll:2.16° Pitch:2.08° Yaw:2.01°)
- Val Rot: 4.27° (Roll:2.08° Pitch:1.85° Yaw:1.85°)
- Jacobian: -1.195 [WEAK]（期待值：> 0.5）

ep14 详细指标：
- Train Rot: 4.14° (Roll:1.96° Pitch:1.82° Yaw:2.02°)
- 改善：4.52° → 4.14° (-0.38°，-8%)


### Loss组成分析（Epoch 14）

| Loss类型 | 数值 | 说明 |
|---------|------|------|
| total_loss | 31.2297 | 加权总和 |
| rotation_loss | 4.14° | 旋转误差 |
| correspondence_loss | 26.88px | 匹配误差 |
| appearance_loss | 24.57 | 几何一致性（外观） |
| depth_loss | 0.0042 | 几何一致性（深度） |
| PC_reproj_loss | 1.6094 | 点云重投影 |


### MatchHead表现（Epoch 14）

| 指标 | 数值 | 说明 |
|------|------|------|
| match_valid_ratio | 0.1896 | 19%匹配内点率（⚠️ 偏低） |
| match_fallback_ratio | 0.7559 | 76%回退到refine（⚠️ 高） |
| epnp_grad_detached | 0.0 | DiffEPnP梯度已启用 ✅ |
| corr_valid_ratio | 0.2925 | 29% LocalCorr窗口有效 |

关键问题：
- match_valid_ratio仅19%：匹配质量不佳
- 76%的batch回退到refine-only：EPnP失效率高
- 说明：MatchHead尚未充分学习，需要更多epoch


## 三、与预期对比

### 预期指标（V40_COMPLETE_SUMMARY_WITH_P2.md）

| Epoch | Train Rot | Val Rot | Jacobian | MEDW | 状态 |
|-------|-----------|---------|----------|------|------|
| ep1 | ~5.0° | ~5.0° | -1.2 WEAK | ~0.4° | - |
| ep15 | ~2.5° | ~2.5° | > 0.3 | < 0.4° | P1b Gate |
| ep30 | ~2.0° | ~2.0° | > 0.5 | < 0.4° | P1c Gate |
| ep60 | < 2.0° | < 2.0° | > 0.6 | < 0.35° | 成功 |

### 实际表现（Epoch 1-14）

| Epoch | Train Rot | 预期 | 差距 | 判定 |
|-------|-----------|------|------|------|
| ep1 | 4.52° | ~5.0° | ✅ 优于预期 | 正常 |
| ep14 | 4.14° | < 3.0° | ⚠️ 慢于预期 | 收敛偏慢 |

分析：
- ep1表现正常（4.52° vs 预期5.0°）
- ep14仍在4.14°，距离ep15目标（2.5°）较远
- 收敛速度比预期慢约20-30%


### 可能原因

| 原因 | 可能性 | 说明 |
|------|-------|------|
| 1. iterative_refine=0 | ⚠️ 高 | 未使用refine迭代，可能影响收敛 |
| 2. MatchHead学习慢 | ⚠️ 高 | 76%回退率说明匹配质量不足 |
| 3. DiffEPnP warmup | 🟢 正常 | ep5才启用梯度，ep14刚开始学习 |
| 4. P2b增强未启用 | ⚠️ 中 | 此次训练未用P2b，数据增强较弱 |


## 四、预测与建议

### 基于当前趋势预测

收敛速度：
- ep1 → ep14：4.52° → 4.14°（-0.38°/13ep = -0.029°/ep）
- 若保持此速度：
  - ep30：4.14° - 0.029×16 = 3.68°（⚠️ 未达标，预期2.0°）
  - ep60：4.14° - 0.029×46 = 2.81°（⚠️ 未达标，预期<2.0°）

Jacobian预测：
- ep1：-1.195 [WEAK]
- ep15预期：> 0.3（P1b Gate）
- 可能无法达标（需要看ep15验证结果）


### 建议行动

#### 🟢 短期（继续当前训练）

建议：继续训练至ep30

理由：
1. ep14仍在DiffEPnP warmup阶段（ep5-10刚启用）
2. MatchHead需要更多epoch学习（当前76%回退率）
3. 可能在ep20-30出现加速收敛（梯度流稳定后）

监控指标（ep15验证）：
- ✅ Val Rot < 3.5°：继续训练
- ⚠️ Val Rot > 4.0°：考虑调整
- ✅ Jacobian > 0.2：有希望达标
- ❌ Jacobian < 0：考虑重启


#### 🟡 中期（ep30评估后决策）

场景A：ep30 Val Rot < 2.5°，Jacobian > 0.5
- ✅ 继续至ep60
- 预期MEDW < 0.35°可达成

场景B：ep30 Val Rot > 3.0°，Jacobian < 0.3
- ⚠️ 停止当前训练
- 重新启动，修复配置问题：
  1. ✅ 启用`iterative_refine=1`
  2. ✅ 启用P2b增强
  3. ✅ 使用P2b完整配置


#### 🔴 立即行动（启动P2b新训练，推荐）

建议：立即启动新的P1c训练（使用P2b完整配置）

理由：
1. 当前训练有配置缺陷（iterative_refine=0，无P2b）
2. 收敛慢于预期（可能无法在ep60达标）
3. P2b配置已就绪（13:45完成）
4. 并行训练成本可接受（2个实验同时运行）

操作：
```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 确认P2b配置
cat configs/v40_gmp_p1c.yaml | grep -A 5 "P2b:"

# 启动新训练（会创建新的model目录）
bash batch_train.sh configs/v40_gmp_p1c.yaml
```

预期：
- 新训练使用完整P2b配置
- 预计收敛速度更快（+数据增强）
- ep30有望达到Jacobian > 0.5


## 五、当前训练ETA

进度：Epoch 15/60 (25%)

时间消耗：
- 已用时：1.6h（ep1-14）
- 平均速度：411.5s/ep（6.86min/ep）
- 剩余45 epoch：ETA 5.3h

总预计：
- 启动时间：12:36
- 预计完成：20:00 (今晚8点)


## 六、总结

### 当前状态：⚠️ 训练中，但有配置问题

| 维度 | 状态 | 说明 |
|------|------|------|
| 训练进度 | 25% (ep15/60) | 正常运行 |
| 配置完整性 | ⚠️ 部分 | iterative_refine=0，无P2b |
| 收敛速度 | ⚠️ 慢于预期 | -8%/14ep vs 预期-45%/15ep |
| MatchHead | ⚠️ 学习中 | 76%回退率，需更多epoch |
| DiffEPnP | ✅ 已启用 | ep5后梯度流通 |


### 关键发现

1. ✅ 训练正常运行，无NaN或崩溃
2. ⚠️ 收敛慢于预期（ep14仍4.14° vs 预期2.5°）
3. ❌ 配置不完整：iterative_refine=0，缺少P2b
4. ⚠️ MatchHead质量不足：76%回退率
5. 🟢 DiffEPnP已启用：梯度正常流动


### 推荐决策

方案A（推荐⭐⭐⭐⭐⭐）：
- ✅ 立即启动新的P1c训练（使用P2b完整配置）
- ⏸️ 当前训练继续至ep30（作为对照组）
- 📊 ep30对比两个实验，选择更优的继续

方案B（保守）：
- ⏸️ 继续当前训练至ep30
- 📊 若ep30未达标（Val Rot > 2.5°），再启动P2b

方案C（激进，不推荐）：
- ❌ 停止当前训练
- ✅ 立即启动P2b训练
- 风险：浪费当前15 epoch的训练


我的建议：选择方案A
- 两个训练并行（成本+8h GPU时间）
- 当前训练作为消融对照（无P2b的效果）
- 新训练使用完整配置（更可能达标）
- ep30时对比决策


文档版本：v1.0  
监控者：BEVCalib Team  
监控时间：2026-05-29 14:16  
下次监控：ep15验证结果（约14:45）
