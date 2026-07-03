# V55 设计方案：按 `calibration_explore` 落地的车端泛化训练主线

**日期**: 2026-06-22  
**状态**: 设计完成，待训练  
**依据**: `/mnt/drtraining/user/dahailu/calibration_explore/camera_lidar_rpy_calibration_full_plan.md`  
**目标**: 提升 BEVCalib 在车端部署口径下的泛化稳定性，而不是继续做单一 loss 的离线最优

---

## 1. 为什么需要 V55

`calibration_explore` 对路线已经给得很清楚：

1. 车端目标不是单帧论文精度，而是**同车型多场景泛化**。
2. 训练主线要优先解决 **域差 / 安装差 / 时序稳定性 / Yaw 弱轴**。
3. Yaw 最终需要 **DNN 基座 + IMU/消失点融合**，DNN 本身的职责是把部署前的基座做稳。

仓库现状里：

- `V53f` 说明 Partial GIN 对部署 MEDW 很有价值。
- `V54d` 说明 DP-Head + router fix 对 Genuine Recovery 更有价值。
- `V54a` 的 LSP/photo 路线训练更脆，且没有稳定证明对部署泛化持续增益。

**因此 V55 的核心决策是：**

> 不再把主线定义为 “再叠一个新 loss”，而是把 `Partial GIN + DP-Head + Hard Route Eval + 域泛化增强` 组合成一个面向车端部署的统一训练方案。

---

## 2. V55 核心目标

### 2.1 主目标

| 指标 | 目标 |
|------|------|
| MEDW400 | `<= 0.18°` |
| ZD max(R/P/Y) | `<= 0.10°` |
| GenuineRec @2° | `>= 70%` |
| 部署策略 | 单 ckpt 主推，保留 gate 评估 |

### 2.2 Stretch 目标

| 指标 | Stretch |
|------|---------|
| MEDW400 | `<= 0.17°` |
| ZD max(R/P/Y) | `<= 0.09°` |
| GenuineRec @2° | `>= 75%` |

### 2.3 边界说明

- `Yaw < 0.1°` 不要求 V55 单模型独立完成。
- V55 负责把 DNN 基座训练到可部署水平。
- 车端最终方案仍建议按 `calibration_explore` 接 `IMU + 消失点` 的 Yaw 融合模块。

### 2.4 两条执行路线

| 路线 | 用途 | 推荐时机 |
|------|------|---------|
| **推荐首跑参数** | 追求在 `V54d` 基础上进一步提升部署泛化 | 默认先跑 |
| **保守备线参数** | 降低结构耦合和训练不稳定性，优先保住可训练性与 MEDW/ZD 底盘 | 首跑 smoke 不稳时切换 |

---

## 3. 结构选择

### 3.1 主结构

```
CF-BEV-R
  + Partial GIN
  + DP-Head
  + JACG
  + Hard Route Eval
  + MGDA(pose + zd + inject)
```

### 3.2 为什么这样选

| 组件 | V55 作用 |
|------|-----------|
| Partial GIN | 把一部分通道做域对齐，优先稳住 ZD / MEDW |
| DP-Head | 显式拆分小偏差 identity 与大偏差 recovery |
| Hard Route Eval | 让部署时的路由逻辑与训练目标一致 |
| JACG | 避免 recovery 路径在大扰动时增益不足 |
| MGDA | 减少手工调权重时 pose / ZD / inject 相互压制 |

### 3.3 不做什么

- 不把 `LSP/photo` 作为 V55 主线。
- 不把 `DANN` 作为默认开启项，避免再引入不稳定变量。
- 不新开专门的 Yaw 网络分支，先通过采样分布和训练课程补 Yaw。

### 3.4 推荐首跑参数

目标是把 `V53f` 的 Partial GIN 稳定性和 `V54d` 的 recovery 能力合到同一条主线里。

```
CF-BEV-R
  + Partial GIN(96ch)
  + DP-Head
  + Hard Route Eval
  + MGDA(pose + zd + inject)
  + 中等强度域泛化增强
```

特点：

- 最有机会直接超过 `v54d-router-fix`
- 训练目标和部署路由完全一致
- 对 GenuineRec 的提升最积极
- 风险是 `GIN + DP-Head + route penalty` 的组合更复杂

### 3.5 保守备线参数

目标是先稳住训练和泛化底盘，再决定是否重新引入 DP-Head。

```
CF-BEV-R
  + Partial GIN(128ch)
  - DP-Head
  - Hard Route Eval
  + MGDA(pose + zd + inject)
  + 较温和的域泛化增强
```

特点：

- 明显减少结构耦合
- 更接近 `v53f` 成功路径
- 更容易把 MEDW / ZD 做稳
- GenuineRec 上限通常低于推荐首跑参数

### 3.6 两条路线的参数差异

| 参数 | 推荐首跑 | 保守备线 |
|------|----------|----------|
| `use_dp_head` | `1` | `0` |
| `use_hard_route_eval` | `1` | `0` |
| `use_gated_instance_norm` | `1` | `1` |
| `gin_channels` | `96` | `128` |
| `gin_init_gate` | `0.40` | `0.35` |
| `learning_rate` | `1.2e-5` | `1.0e-5` |
| `route_loss_weight` | `0.25` | `0.0` |
| `route_zd_penalty_weight` | `0.35` | `0.0` |
| `jacobian_loss_weight` | `0.12` | `0.10` |
| `zero_drift_loss_weight` | `0.45` | `0.42` |
| `inject_recovery_loss_weight` | `0.60` | `0.50` |
| `augment_mount_jitter_prob` | `0.45` | `0.35` |
| `augment_fov_crop_prob` | `0.25` | `0.15` |
| `augment_lidar_sparse_prob` | `0.15` | `0.10` |
| `per_axis_weights` | `"0.25,0.20,0.55"` | `"0.30,0.25,0.45"` |

### 3.7 推荐使用顺序

建议按下面顺序执行：

1. 先跑推荐首跑参数的 smoke
2. smoke 稳定后继续 full
3. 若 smoke 不稳，再切到保守备线

对应配置文件：

- 推荐首跑：`configs/v55_deploy_generalization_cf_bev_r.yaml`
- 保守备线：`configs/v55_safe_deploy_generalization_cf_bev_r.yaml`

---

## 4. 数据与采样策略

V55 的数据策略直接对齐 `calibration_explore` 的“同车型多场景”原则。

### 4.1 采样策略

- `pose_aware_sampling=true`
- `max_frames_per_seq=400`
- 保留多 sequence，但减少同一轨迹里的高重复帧

### 4.2 训练分布

- 总扰动范围仍用 `5°`
- 课程学习从 `1.5° -> 5°`
- 同时保留 `0.5° / 1.0° / 2.0° / 3.5° / 5.0°` 多尺度扰动
- 增加 `per_axis_prob`，并把 `Yaw` 采样权重调高

### 4.3 域泛化增强

V55 默认开启以下增强：

- `mount jitter`
- `intrinsic jitter`
- `color jitter`
- `FOV crop`
- `LiDAR sparsification`

这几项不是为了论文视觉效果，而是为了模拟：

- 不同装车姿态
- 轻微内参偏移
- 光照变化
- 视场裁切差异
- 不同线数/有效点密度的点云

---

## 5. Loss 设计

### 5.1 主损失

| Loss | 作用 |
|------|------|
| `pose` | 基础姿态回归 |
| `zero_drift_loss` | 压小偏差场景的固有漂移 |
| `inject_recovery_loss` | 直接对齐部署 Genuine Recovery |
| `jacobian_loss` | 保证扰动增大时 correction slope 不塌 |
| `route_loss` | 监督 DP-Head 的 bias / recovery 路由 |
| `route_zd_penalty` | 防止 router 在小扰动时塌到 recovery |

### 5.2 V55 的关键取舍

- `zero_drift_loss` 继续 ramp，避免压垮 recovery
- `inject_recovery_loss` 保持高权重，继续和 `gdiag Fixed-Inject` 对齐
- `photo/LSP` 关闭
- `rig_consistency` 默认关闭在主线外，只在需要时做补充实验

---

## 6. 训练日程

### Stage A: Smoke

- 15 epochs
- 目标：
  - 无 NaN
  - `route_w` 不塌缩
  - `hard route eval` 链路可用
  - Yaw 采样/增强不会显著拖垮训练

### Stage B: Full

- 60 epochs
- 从 `v54d_router_fix` 的 `best_dual` 起训
- 主目标是把：
  - `V53f` 的 Partial GIN 稳定性
  - `V54d` 的 recovery 能力
  合并到一个 ckpt 里

### Stage C: Refine

- 20 epochs
- 从 V55 full `best_dual` 继续
- 冻结 backbone / bev_branch
- 更高 `zero_perturbation` 和 `inject` 比例
- 只冲部署 gate

---

## 7. 关键超参建议

### 7.1 结构相关

| 参数 | 建议值 |
|------|--------|
| `use_dp_head` | `1` |
| `use_hard_route_eval` | `1` |
| `dp_gate_deg` | `1.5` |
| `use_gated_instance_norm` | `1` |
| `gin_channels` | `96` |
| `gin_init_gate` | `0.40` |

### 7.2 训练相关

| 参数 | 建议值 |
|------|--------|
| `learning_rate` | `1.2e-5` |
| `batch_size` | `16` |
| `backbone_lr_scale` | `0.08` |
| `zero_drift_loss_weight` | `0.45` |
| `inject_recovery_loss_weight` | `0.60` |
| `jacobian_loss_weight` | `0.12` |
| `route_loss_weight` | `0.25` |
| `route_zd_penalty_weight` | `0.35` |

### 7.3 泛化增强相关

| 参数 | 建议值 |
|------|--------|
| `augment_mount_jitter_prob` | `0.45` |
| `augment_mount_jitter_rot_sigma` | `1.5` |
| `augment_intrinsic` | `0.03` |
| `augment_color_jitter` | `0.25` |
| `augment_fov_crop_prob` | `0.25` |
| `augment_lidar_sparse_prob` | `0.15` |

### 7.4 扰动分布

| 参数 | 建议值 |
|------|--------|
| `progressive_angle_start` | `1.5` |
| `progressive_angle_end` | `5.0` |
| `progressive_warmup_epochs` | `25` |
| `per_axis_prob` | `0.35` |
| `per_axis_weights` | `"0.25,0.20,0.55"` |

---

## 8. 评估方案

V55 只用现有仓库的部署口径验收，不新发明评估标准。

### 8.1 主评估

- `run_generalization_eval.py`
- `generalization_diag`
- `MEDW400`
- deploy gate summary

### 8.2 候选 ckpt

- `best_dual`: 主部署候选
- `best_medw`: 时序聚合优先候选
- `best_val`: recovery 候选

### 8.3 对照组

- `v54d-router-fix`
- `v54d-refine2`
- `v53f-gin-jacg`
- `v54a-nolsp`

---

## 9. 风险与回退

### 9.1 主要风险

| 风险 | 说明 | 应对 |
|------|------|------|
| GIN + DP-Head 组合不稳定 | 新组合可能互相干扰 | 先 smoke；保守设置 `gin_channels=96` |
| route 仍塌缩 | 小扰动样本仍走 recovery | 保留 `route_zd_penalty` + `hard route eval` |
| 域增强过强 | recovery 被拖垮 | 先从中等增强强度起步 |
| Yaw 改善有限 | DNN 单独难破物理瓶颈 | 后续接 IMU/VP 融合，不把压力全压给训练 |

### 9.2 回退方案

如果 V55 smoke 明显不稳定，退回 `V55-safe`：

```
V53f 风格
  Partial GIN
  + MGDA
  + 域泛化增强
  - DP-Head
```

即先保住车端泛化底盘，再决定是否继续合并 DP-Head。

### 9.3 何时切到保守备线

满足任一条就建议从推荐首跑切到保守备线：

1. smoke 前 `10-15ep` 内出现 `NaN` 或 loss 持续爆炸
2. `route_w_mean` 长时间贴近 `0.0` 或 `1.0`，说明 router 仍明显塌缩
3. `ZD max(R/P/Y)` 比 `v54d-router-fix` smoke 明显更差，且 `inject` 没有同步提升
4. 域增强打开后 `MEDW` 和 `GenuineRec` 同时显著恶化

---

## 10. 结论

V55 的本质不是 “V54 再加一个 loss”，而是：

> 依据 `calibration_explore` 的部署思路，把 BEVCalib 训练成一个更稳的车端泛化基座。

它重点解决四件事：

1. 安装差和场景差引起的域偏移
2. 小偏差时的零漂移稳定性
3. 大偏差时的真实恢复能力
4. 训练路由与部署路由不一致的问题

对应落地文件：

- `configs/v55_deploy_generalization_cf_bev_r.yaml`
- `configs/v55_safe_deploy_generalization_cf_bev_r.yaml`
- `configs/eval_generalization_v55.yaml`
- `run_v55_eval.sh`
