# V55 Main Failure Root-Cause Summary

**日期**: 2026-06-24  
**对象**: `v55_deploy_generalization` 主线失败复盘  
**结论**: 当前主线的主要问题不是“训练没收敛”，而是 **路由塌缩 + 扰动/增强过强 + GIN 约束不足** 共同导致跨序列灾难样本。

---

## 1. 直接观测到的症状

### 训练期

- `route_w_mean` 长期停留在 `0.96~0.97`
- `route_zd_penalty` 一直偏高，但没有把路由拉回到 bias path
- `smoke` 的 `best_dual` 出现在 `epoch 1`，之后 proxy 持续退化
- `smoke` 后期仍出现 `NaN GUARD`

### 泛化评估期

- `v55-main-best-dual` / `best-medw` 出现大量 `80°+` 级别灾难样本
- `v55-main-best-val` 也出现大量 `68°+` 级别灾难样本
- 时序聚合虽然能显著压低误差，但最终 `MEDW200/400` 仍约 `0.51~0.53°`
- 远高于部署目标 `MEDW400 <= 0.18°`

---

## 2. 根因判断

### Root Cause A: DP-Head 路由仍然塌向 RecoveryPath

虽然已经开启：

- `route_loss`
- `route_zd_penalty`
- `use_hard_route_eval`

但训练日志表明：

- 小扰动样本并没有充分走到 `BiasPath`
- `RecoveryPath` 仍承担了过多本应由稳定路径处理的样本

结果就是：

- 小偏差场景的固有漂移没有真正压稳
- 一旦遇到跨域样本，RecoveryPath 会输出灾难性大角度误差

### Root Cause B: 扰动与域增强叠加过强

V55 主线同时启用了：

- `continuous_noise_max_deg=8`
- `mount jitter=0.45`
- `FOV crop=0.25`
- `LiDAR sparse=0.15`
- `per_axis_prob=0.35`
- `Yaw` 高权重采样

这些增强单独都合理，但叠加在一起会造成：

- 训练分布比真实部署更“野”
- Router 更频繁地学习到“大扰动优先”
- RecoveryPath 的校正幅度更容易被放大

### Root Cause C: Partial GIN 约束不够强

`gin_channels=96` + `gin_init_gate=0.40` 对主线来说偏激进：

- 域不变性不够强，跨序列风格差仍大量泄漏进 recovery 通路
- `DP-Head + 轻 GIN` 的组合更容易让 RecoveryPath 学会“场景相关 shortcut”

---

## 3. 对策

### 对策 1: 立即切到 `V55-safe full`

目标：

- 先把稳定底盘跑出来
- 优先保住 `MEDW / ZD`
- 暂时不再把风险压给 DP-Head

### 对策 2: 开一版 `V55b`

V55b 的方向不是推翻主线，而是把主线收紧：

- **Route 更紧**
  - 更高 `route_loss_weight`
  - 更高 `route_zd_penalty_weight`
  - 更小 `dp_gate_deg`
- **Perturb 更紧**
  - 更低 `continuous_noise_max_deg`
  - 降低 `inject_dedicated_ratio`
  - 降低 `mount/FOV/sparse` 增强强度
- **GIN 更紧**
  - 提高 `gin_channels`
  - 降低 `gin_init_gate`
  - 更强 gate regularization

---

## 4. 执行结论

推荐执行顺序：

1. 立即跑 `V55-safe full`
2. 同时准备 `V55b` 收紧版配置
3. `V55-safe full` 完成后接 `V55b smoke`

这样可以同时保证：

- 有一条稳定底盘线持续产出
- 主线结构思路不被过早放弃
