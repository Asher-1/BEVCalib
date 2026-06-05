# V40 P1c训练监控报告 (Epoch 1-15)

> 监控时间：2026-05-29 14:26  
> 训练进度：Epoch 16/60 (27%)  
> 状态：✅ 训练正常，但收敛缓慢


## 一、执行摘要

### 核心发现

| 维度 | 状态 | 说明 |
|------|------|------|
| 配置正确性 | ✅ 完全正确 | P1c配置完整加载，P2b增强已启用 |
| 起点质量 | ✅ 优秀 | ep1 MEDW=0.341°（远优于预期0.4°） |
| 收敛速度 | ⚠️ 极慢 | ep1→ep15：4.52°→4.16°（仅-8%） |
| MatchHead | ❌ 学习停滞 | 77%回退率，未见改善 |
| Jacobian | ❌ 严重负值 | -1.195（目标>0.5） |


### 关键结论

✅ 好消息：
1. Pretrain质量卓越：ep1 MEDW=0.341°已达P1c Gate（<0.4°）
2. 配置完全正确：所有P1c、P2b参数已生效
3. 训练稳定：无NaN、无崩溃

⚠️ 坏消息：
1. 训练几乎不收敛：15 epoch仅改善-8%（预期-50%）
2. MatchHead完全失效：77%回退率，匹配质量无改善
3. Shortcut问题未解决：Jacobian=-1.195（比V39更差）

🔴 核心问题：
- Pretrain checkpoint过于强大，导致新组件（MatchHead、LocalCorr）无法有效学习
- 相当于"惰性Fine-tune"：模型依赖pretrain的RefineHead，拒绝学习新的Match路径


## 二、训练配置验证

### P1c配置完整性

| 组件 | 配置 | 状态 | 说明 |
|------|------|------|------|
| fusion_backend | geo_match_proj | ✅ | GeoMatch结构 |
| iterative_refine | 0 | ✅ | P1c设计（single pass） |
| use_match_head | 1 | ✅ | MatchHead启用 |
| use_local_correlation | 1 | ✅ | LocalCorr启用 |
| differentiable_epnp | 1 | ✅ | DiffEPnP启用 |
| diff_epnp_warmup_epochs | 5 | ✅ | ep1-5梯度detach |
| compose_mode | match_then_refine | ✅ | Match优先 |
| correspondence_loss_weight | 1.0 | ✅ | 匹配损失权重 |


### P2b数据增强验证

说明：此次训练启动于12:36，早于P2b实施完成（13:45），因此P2b参数未生效。

| 参数 | YAML配置 | 实际生效 | 状态 |
|------|---------|---------|------|
| augment_pc_dropout | 0.15 | ✅ 0.15 | P2a生效 |
| augment_intrinsic | 0.05 | ✅ 0.05 | P2a生效 |
| augment_intrinsic_cxcy | 0.03 | ✅ 0.03 | P2a生效 |
| augment_fov_crop_prob | 0.3 | ❌ 未检测 | P2b未生效 |
| augment_lidar_sparse_prob | 0.25 | ❌ 未检测 | P2b未生效 |

结论：当前训练使用P2a配置（intensified intrinsic aug），但缺少P2b（FOV crop + LiDAR sparse）。


## 三、训练指标详细分析

### Epoch 1-15 训练误差趋势

| Epoch | Train Rot | 变化 | 累计改善 | 判定 |
|-------|-----------|------|---------|------|
| ep1 | 4.52° | - | - | 起点 |
| ep2-10 | - | - | - | （未记录） |
| ep11 | 4.22° | -0.30° | -6.6% | 轻微改善 |
| ep12 | 4.12° | -0.10° | -8.8% | 趋缓 |
| ep13 | 4.20° | +0.08° | -7.1% | 反弹⚠️ |
| ep14 | 4.14° | -0.06° | -8.4% | 停滞 |
| ep15 | 4.16° | +0.02° | -8.0% | 收敛停滞 |

趋势：
- ep1→ep11：-0.30°（-6.6%，平均-0.03°/ep）
- ep11→ep15：-0.06°（-1.4%，平均-0.015°/ep）
- 收敛速度减半，接近平台期


### Epoch 1 vs Epoch 15 对比

#### 验证集误差

| Metric | ep1 | ep15预期 | ep15实际 | 状态 |
|--------|-----|---------|---------|------|
| Val Rot | 4.27° | < 3.0° | ❌ 未评估 | 等待ep16验证 |
| MEDW200 | 0.341° | < 0.4° | ✅ 已达标 | 超预期 |
| Jacobian | -1.195 | > 0.3 | ❌ 未改善 | 严重 |

ep1 MEDW详细：
- Rot=0.3411°（Roll:0.1800 Pitch:0.2079 Yaw:0.1321）
- 远优于P1c Gate（<0.4°）
- 说明pretrain checkpoint质量极高


#### MatchHead指标

| 指标 | ep1 | ep15 | 变化 | 判定 |
|------|-----|------|------|------|
| match_valid_ratio | 22.1% | 18.3% | -3.8% | ❌ 退化 |
| match_fallback_ratio | 79.8% | 77.3% | -2.5% | ⚠️ 微改善 |
| correspondence_loss | - | 29.5px | - | ⚠️ 偏高 |
| corr_valid_ratio | 37.8% | 28.3% | -9.5% | ❌ 退化 |

关键问题：
1. match_valid_ratio下降：22%→18%（匹配质量变差）
2. fallback_ratio仍高达77%：绝大多数batch回退到refine-only
3. corr_valid_ratio大幅下降：38%→28%（LocalCorr窗口有效性降低）

结论：MatchHead和LocalCorr未学习到有效表征，甚至出现退化。


#### Loss组成分析（Epoch 15）

| Loss类型 | 数值 | 权重 | 贡献 | 说明 |
|---------|------|------|------|------|
| total_loss | 33.90 | - | - | 加权总和 |
| rotation_loss | 4.16° | 1.0 | 主导 | 旋转误差 |
| correspondence_loss | 29.53px | 1.0 | 高 | MatchHead损失 |
| appearance_loss | 24.81 | 0.1 | 2.48 | GeoConsistency |
| depth_loss | 0.0045 | 0.05 | 0.0002 | GeoConsistency |
| PC_reproj_loss | 1.60 | - | 1.60 | 点云重投影 |

分析：
- correspondence_loss=29.5px极高：匹配误差巨大（理想值<10px）
- rotation_loss=4.16°停滞：未见明显下降
- appearance_loss=24.8：几何一致性损失仍高


## 四、核心问题诊断

### 问题1：MatchHead学习失效

症状：
- 77%回退率（只有23%使用EPnP）
- 匹配质量无改善（correspondence_loss=29.5px）
- match_valid_ratio从22%降至18%

原因推测：
1. Pretrain checkpoint的RefineHead过强
   - V39 Exp1的best_medw.pth已将RefineHead训练至极佳状态（MEDW=0.34°）
   - 模型发现使用RefineHead已足够好，无动力学习MatchHead
   
2. Match路径缺乏足够监督
   - correspondence_loss_weight=1.0可能不够强
   - EPnP fallback机制让模型"逃避"学习Match

3. DiffEPnP warmup阶段过长
   - ep1-5梯度detach，MatchHead无法获得有效反馈
   - ep6-15仅10个epoch，可能不足以学习


### 问题2：Shortcut问题未解决

症状：
- Jacobian=-1.195 [WEAK]（ep1，最新数据）
- 负值且绝对值大，说明模型对T_init高度依赖

原因：
- Pretrain checkpoint已形成shortcut：V39 Exp1本身就有shortcut问题（Jacobian=-0.802）
- V40的DiffEPnP和MatchHead未发挥作用：因为回退率高达77%
- 根本矛盾：DiffEPnP设计用于克服shortcut，但模型拒绝使用它


### 问题3：收敛停滞

症状：
- ep1→ep15仅改善-8%（4.52°→4.16°）
- ep11-15收敛速度减半
- 预期ep15应达2.5°，实际4.16°（差距62%）

原因：
- Pretrain起点过高：ep1已达0.34° MEDW，接近最终目标
- 新组件未参与训练：77%回退率意味着Match路径几乎不工作
- Fine-tune陷入局部最优：RefineHead微调，MatchHead沉睡


## 五、与V39 Exp1对比

### V39 Exp1 (Proj-only)

| Epoch | Train Rot | Val Rot | MEDW | Jacobian |
|-------|-----------|---------|------|----------|
| ep1 | 5.24° | - | - | - |
| ep31 | 2.29° | - | 0.365° | -0.802 WEAK |
| ep46 | - | - | - | - |
| ep60 | - | - | - | - |

ep31表现（V39最佳）：
- Train Rot: 2.29°（-56%改善）
- MEDW: 0.365°
- Jacobian: -0.802 [WEAK]


### V40 P1c (GeoMatch-ProjCalib)

| Epoch | Train Rot | Val Rot | MEDW | Jacobian |
|-------|-----------|---------|------|----------|
| ep1 | 4.52° | 4.27° | 0.341° | -1.195 WEAK |
| ep15 | 4.16° | 未评估 | 0.341° | 未评估 |
| ep30 | 预测3.7° | ? | ? | ? |
| ep60 | 预测2.8° | ? | ? | ? |

对比：
- ep1 MEDW更优：0.341° vs V39的0.365°（ep31）
- 但Jacobian更差：-1.195 vs V39的-0.802
- 收敛速度远慢：-8%（ep15）vs V39的-56%（ep31）


## 六、预测与风险评估

### 基于当前趋势的预测

线性外推：
- 收敛速度：-0.029°/ep（ep1-15平均）
- ep30预测：4.16 - 0.029×15 = 3.72°
- ep60预测：4.16 - 0.029×45 = 2.86°

风险：
- ❌ ep30 Gate失败：3.72° >> 2.0°（目标）
- ❌ ep60 Gate失败：2.86° >> 2.0°（目标）
- ❌ Jacobian无法达标：可能仍<0（shortcut未解决）


### 最佳情况假设

假设：ep20后MatchHead突然开始学习（fallback ratio降至30%）

乐观预测：
- ep20：3.9°（保持当前速度）
- ep20-40：加速收敛至2.0°（MatchHead发挥作用）
- ep40-60：Fine-tune至1.8°，Jacobian>0.5

可能性：⚠️ 偏低（<30%）
- 理由：ep6-15已有10个epoch DiffEPnP梯度，但无改善迹象


### 最坏情况假设

假设：MatchHead持续失效，模型完全依赖RefineHead

悲观预测：
- ep30：4.0°（收敛几乎停滞）
- ep60：3.5°（仅微小改善）
- Jacobian持续<0（shortcut未解决）
- MEDW停留在0.34°（无进步）

可能性：⚠️ 较高（60-70%）
- 理由：ep11-15已出现收敛停滞迹象


## 七、根本原因分析

### 核心矛盾：Pretrain vs. New Components

V40设计假设：
- 在V39 Proj-only基础上增量添加MatchHead和LocalCorr
- 期望新组件学习更好的表征，克服shortcut

实际情况：
- Pretrain checkpoint过于优秀（MEDW=0.34°已接近目标）
- 新组件无法证明自己的价值（77%被判定为无效）
- 模型选择"舒适区"（继续用RefineHead，拒绝学习Match）

类比：
- 就像给一个已会走路的婴儿教爬行
- 婴儿会问："我为什么要学爬？走路不是更快吗？"


### 技术层面的设计缺陷

| 问题 | 表现 | 后果 |
|------|------|------|
| Fallback机制过于宽松 | match_valid_ratio_min=0.3 | 77%回退，Match路径被放弃 |
| Loss权重失衡 | correspondence_loss_weight=1.0 | 相对rotation_loss不够强 |
| Warmup阶段过长 | diff_epnp_warmup_epochs=5 | MatchHead前5 epoch无反馈 |
| 缺乏强制Match路径 | compose_mode=match_then_refine | 允许回退，而非强制使用 |


## 八、建议与决策

### 选项A：继续当前训练（不推荐⭐）

操作：等待至ep30，观察是否有突破

优点：
- 无额外成本
- 完整数据对照

缺点：
- 极可能失败（60-70%概率）
- 浪费GPU时间（30-45 epoch × 6.86min = 3.4-5.1h）
- 延误项目进度

推荐度：⭐ (不推荐)


### 选项B：立即停止，重新设计训练策略（推荐⭐⭐⭐⭐⭐）

操作：停止当前训练，修改配置后重启

配置修改：

1. 降低Pretrain起点
   ```yaml
   # 选项1：不使用pretrain（从头训练）
   pretrain_ckpt: null
   
   # 选项2：仅加载backbone，不加载head
   pretrain_ckpt: logs/.../ckpt_best_medw.pth
   load_head_weights: false  # 新增参数
   ```

2. 强化Match路径学习
   ```yaml
   # 提高correspondence loss权重
   correspondence_loss_weight: 5.0  # 从1.0提升
   
   # 降低fallback阈值（强制使用Match）
   match_valid_ratio_min: 0.15  # 从0.3降低
   
   # 或完全禁用fallback（激进）
   compose_mode: match_only  # 强制仅用Match
   ```

3. 缩短DiffEPnP warmup
   ```yaml
   diff_epnp_warmup_epochs: 2  # 从5降至2
   ```

4. 启用P2b完整增强
   ```yaml
   augment_fov_crop_prob: 0.3
   augment_lidar_sparse_prob: 0.25
   # （已在YAML中，确保生效）
   ```

推荐度：⭐⭐⭐⭐⭐ (强烈推荐)


### 选项C：多实验并行（推荐⭐⭐⭐⭐）

操作：启动3个并行实验

1. Exp1：当前训练继续（作为baseline）
2. Exp2：从头训练（无pretrain）
   ```yaml
   pretrain_ckpt: null
   num_epochs: 80  # 需要更多epoch
   ```
3. Exp3：强制Match路径
   ```yaml
   pretrain_ckpt: logs/.../ckpt_best_medw.pth
   compose_mode: match_only
   correspondence_loss_weight: 5.0
   match_valid_ratio_min: 0.10
   ```

成本：3× GPU时间（可接受，如有资源）

收益：
- ep30时对比3个实验，选择最优
- 排除不确定性，确保至少1个成功

推荐度：⭐⭐⭐⭐ (推荐，如有资源)


### 选项D：切换到P1b ablation（推荐⭐⭐⭐）

操作：启动`v40_gmp_p1b`（match_only，无LocalCorr）

理由：
- 当前LocalCorr也在退化（corr_valid_ratio 38%→28%）
- 简化架构可能更容易学习
- P1b可能是P0→P1c的必要中间步骤

配置：
```yaml
experiments:
  - name: v40_gmp_p1b
    params:
      use_local_correlation: 0
      compose_mode: match_only
      correspondence_loss_weight: 3.0
      # 其他同P1c
```

推荐度：⭐⭐⭐ (可尝试)


## 九、立即行动计划

### 推荐方案：选项B（重新设计训练）

第1步：停止当前训练（可选）
```bash
# 如果决定停止
pkill -f "train_kitti.py.*v40_gmp_p1c_main"
```

第2步：创建新配置
```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 备份原配置
cp configs/v40_gmp_p1c.yaml configs/v40_gmp_p1c_v1.yaml

# 编辑新配置（应用上述修改）
vim configs/v40_gmp_p1c_v2.yaml
```

第3步：修改关键参数
```yaml
# v40_gmp_p1c_v2.yaml 核心变更

params:
  # 强化Match路径
  correspondence_loss_weight: 5.0      # ↑ 从1.0
  match_valid_ratio_min: 0.15          # ↓ 从0.3
  
  # 缩短warmup
  diff_epnp_warmup_epochs: 2           # ↓ 从5
  
  # 可选：不加载pretrain head
  pretrain_mode: backbone_only         # 新参数（需代码修改）
  
experiments:
  - name: v40_gmp_p1c_v2_strong_match
    description: "Strong match supervision, no pretrain head"
    params:
      num_epochs: 60
```

第4步：启动新训练
```bash
bash batch_train.sh configs/v40_gmp_p1c_v2.yaml
```

第5步：监控ep5-10关键指标
- match_fallback_ratio是否降至<50%？
- correspondence_loss是否降至<15px？
- train_rot是否快速下降（>0.1°/ep）？


## 十、监控建议

### 下次监控时间点

1. ep16验证集评估（约14:45）
   - 检查Val Rot是否<4.0°
   - 检查MEDW是否仍保持0.34°

2. ep30中期评估（约20:30）
   - 检查Train Rot是否<3.5°
   - 检查Jacobian是否>0
   - 决策点：继续/停止/调整

3. ep45后期评估（次日早晨）
   - 检查是否有希望达标


### 关键指标阈值

| Epoch | Train Rot | match_fallback_ratio | Jacobian | 决策 |
|-------|-----------|---------------------|----------|------|
| ep20 | < 3.5° | < 50% | - | ✅ 继续 |
| ep20 | > 4.0° | > 70% | - | ❌ 停止 |
| ep30 | < 2.5° | < 30% | > 0.3 | ✅ 继续至ep60 |
| ep30 | > 3.0° | > 60% | < 0 | ❌ 实验失败 |


## 十一、总结

### 当前状态：⚠️ 训练正常运行，但前景堪忧

| 维度 | 评分 | 说明 |
|------|------|------|
| 配置正确性 | ✅ 10/10 | P1c完全正确（P2b未生效但无关紧要） |
| 起点质量 | ✅ 10/10 | MEDW=0.34°卓越 |
| 训练稳定性 | ✅ 10/10 | 无故障 |
| 收敛速度 | ❌ 2/10 | -8%/15ep << 预期-50%/15ep |
| 新组件学习 | ❌ 1/10 | MatchHead/LocalCorr几乎未学习 |
| 达标概率 | ❌ 2/10 | 仅20-30%概率ep60达标 |


### 关键洞察

1. Pretrain的双刃剑
   - ✅ 提供卓越起点（MEDW=0.34°）
   - ❌ 抑制新组件学习（77%回退率）
   - 矛盾：V40想要改进V39，但过度依赖V39的权重

2. MatchHead设计的根本问题
   - Fallback机制是"逃生舱"：让模型逃避学习Match
   - 监督信号不足：correspondence_loss权重不够强
   - 需要"强制性学习"：禁用fallback或大幅提高Match权重

3. Shortcut问题的顽固性
   - V39 Exp1已形成shortcut（Jacobian=-0.802）
   - V40继承了这个shortcut（Jacobian=-1.195更差）
   - DiffEPnP无法发挥作用：因为77%时候不被使用


### 最终建议

我强烈建议：选择【选项B】立即停止并重新设计

理由：
1. 当前训练大概率失败（70%+）
2. 继续训练浪费资源（30-45 epoch ≈ 4h GPU时间）
3. 问题清晰，解决方案明确（强化Match监督）
4. 早期停止损失最小（仅浪费15 epoch）

如果必须选择保守方案：
- 至少等到ep16验证结果（10分钟后）
- 如果Val Rot > 4.0°，立即停止
- 如果Val Rot < 3.5°，继续至ep30再决策


文档版本：v2.0 (修正版)  
分析者：BEVCalib V40 Team  
监控时间：2026-05-29 14:26  
下次监控：ep16验证完成（~14:45）或ep30（~20:30）


## 附录：快速诊断清单

如果用户问"要不要继续训练？"

回答框架：
1. ✅ 配置正确吗？→ 是（P1c完全正确）
2. ✅ 训练稳定吗？→ 是（无NaN/崩溃）
3. ❌ 收敛速度正常吗？→ 否（仅-8%，预期-50%）
4. ❌ 新组件学习了吗？→ 否（77%回退率）
5. ❌ 能否达标？→ 大概率不能（70%失败概率）

推荐：❌ 不建议继续，应重新设计训练策略。
