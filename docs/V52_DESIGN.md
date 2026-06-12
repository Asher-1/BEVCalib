# V52 设计方案：解耦双路径并行实验

**日期**: 2026-06-10  
**状态**: 已实现训练配置 + inject_recovery_loss  
**前置**: V51 泛化评估 (`generalization_v51`)

---

## 1. V51 复盘结论

V51 同时启用 `zero_drift_loss` + `jacobian_loss` 导致：

| 指标 | v51a | v50a | Phase A 目标 |
|------|------|------|-------------|
| MEDW400 | **0.188°** ✅ | 0.263° | < 0.25° |
| ZD max(R,P,Y) | **0.146°** ✅ | 0.185° | < 0.20° |
| Genuine Recovery | **29.2%** ❌ | 57.3% | ≥ 60% |

**根因**：非纯捷径（PredIndep=0.701 GENUINE），而是 ZD loss 压过矫正幅度 → 全局 under-correction。Jacobian 局部斜率代理与 gdiag Genuine Rec 脱节。

**V52 策略**：不再线性混合双 proxy，拆成 **两条独立实验线**，两台机器并行：

| 线路 | 配置 | 机器 | 假设 |
|------|------|------|------|
| **v52a** | 两阶段解耦 (Recovery→ZD) | 机器 A | 先保 Recovery，再 ZD 微调 |
| **v52b** | inject_recovery 直接监督 | 机器 B | 对齐 gdiag Fixed-Inject 指标 |

---

## 2. 验收目标

### Phase A（第一版）

| 指标 | 阈值 | 评估 |
|------|------|------|
| MEDW400 | **≤ 0.22°** | test_data_v2 |
| ZD max(R,P,Y) | **≤ 0.18°** | gdiag |
| Genuine Recovery | **≥ 50%** | gdiag Fixed-Inject |
| 小量级 Recovery (0.5°+1.0°) | **> 0%** | gdiag multi-mag |

### Phase B（stretch）

| 指标 | 阈值 |
|------|------|
| MEDW400 | ≤ 0.20° |
| Genuine Recovery | ≥ 55% |
| GS_medw | ≤ 0.60 |

---

## 3. 新增 Loss：inject_recovery_loss

对齐 gdiag **Fixed-Inject**（全轴 +inject_deg）：

```
init_R = gt_R @ dR(inject, inject, inject)   # LiDAR RPY, 右乘 gt
out_err = euler(R_pred @ R_gt^T)
L_inj = SmoothL1(out_err, 0)
```

| 参数 | v52b 默认 | 说明 |
|------|----------|------|
| `inject_recovery_loss_weight` | **0.5** | 独立于主 pose loss |
| `inject_recovery_loss_start_epoch` | **10** | 主 loss 稳定后 |
| `inject_recovery_magnitude_deg` | **2.0** | 与 gdiag inject 对齐 |
| `inject_recovery_dedicated_ratio` | **0.10** | 强制 Fixed-Inject batch |

实现：`train_kitti.py::_apply_fixed_inject_batch` + `_zero_drift_axis_error_deg`

---

## 4. 线路 A：两阶段解耦 (v52a)

### Stage 1 — Recovery Base (~100ep)

```
L = L_pose + λ_jac·L_jacobian (0.15)
关闭: zero_drift_loss, inject_recovery_loss
zero_perturbation_prob: 0.10
multi_scale: 0.5:0.30, 1.0:0.25, 2.0:0.15
progressive_angle: 2° → 5° over 50ep
corr_alignment_weight: 0.3
pretrain: v50a ckpt_best_val
```

### Stage 2 — ZD Fine-tune (~30ep)

```
pretrain: Stage1 ckpt_best_val  (不用 dual)
backbone_lr_scale: 0.01  (Swin 软冻结; freeze_backbone 对 cf_bev_r 无效)
LR: ~1/5 Stage1
L = L_pose + λ_jac·L_jacobian (0.05) + λ_zd·L_zero_drift (ramp 0.3→1.0 / 10ep)
zero_drift_dedicated_ratio: 0.15, zero_perturbation_prob: 0.25
jacobian_early_stop: overall < 0.85 连续 2 eval → 停训
ckpt: ckpt_best_val + ckpt_best_jacobian + ckpt_best_dual
```

### Smoke 优化 (2026-06-10)

基于 smoke 结论调整 S2：
- ZD 权重线性 ramp，避免一步到位压垮 Recovery
- S2 保留轻量 Jacobian loss (0.05) 作 Recovery 护栏
- 新增 `ckpt_best_jacobian.pth` 与 Jacobian 早停
- 移除无效的 `freeze_backbone=1`（Swin/cf_bev_r 不生效）

---

## 5. 线路 B：inject_recovery 单阶段 (v52b)

```
L = L_pose + λ_inj·L_inject_recovery (0.5) + λ_jac·L_jacobian (0.05)
关闭: zero_drift_loss
inject_recovery_dedicated_ratio: 0.10
multi_scale / corr_alignment 同 v52a S1
pretrain: v50a ckpt_best_val
epochs: 100 (full) / 20 (smoke)
```

Jacobian 降为辅助（0.05），主 Recovery 监督由 inject_loss 承担。

---

## 6. 实验矩阵与并行启动

### 配置文件

| 文件 | 机器 | 内容 |
|------|------|------|
| `configs/v52a_cf_bev_r.yaml` | **机器 A** | S1/S2 smoke + full |
| `configs/v52b_cf_bev_r.yaml` | **机器 B** | smoke + full |

### 启动命令

**机器 A（v52a 两阶段）：**
```bash
# Smoke 链: S1(15ep) → S2(5ep)
bash batch_train.sh configs/v52a_cf_bev_r.yaml --skip-pattern full

# Full 链: S1(100ep) → S2(30ep) — 等 S1 smoke 通过后再跑
bash batch_train.sh configs/v52a_cf_bev_r.yaml --skip-pattern smoke
```

**机器 B（v52b inject）：**
```bash
bash batch_train.sh configs/v52b_cf_bev_r.yaml --skip-pattern full   # smoke 20ep
bash batch_train.sh configs/v52b_cf_bev_r.yaml --skip-pattern smoke   # full 100ep
```

### 实验列表

| 实验 | version | epochs | 依赖 |
|------|---------|--------|------|
| v52a_S1_smoke | v52a_S1_smoke | 15 | v50a pretrain |
| v52a_S2_smoke | v52a_S2_smoke | 5 | v52a_S1_smoke ckpt |
| v52a_S1_full | v52a_S1_full | 100 | v50a pretrain |
| v52a_S2_full | v52a_S2_full | 30 | v52a_S1_full ckpt |
| v52b_smoke | v52b_smoke | 20 | v50a pretrain |
| v52b_full | v52b_full | 100 | v50a pretrain |

日志：`logs/all_training_data/model_small_5deg_v52*/`

---

## 7. 泛化评估（训练完成后）

### 7.1 配置文件与入口

| 文件 | 说明 |
|------|------|
| `configs/eval_generalization_v52.yaml` | 11 个 V52 主模型 + 4 个历史对照 |
| `run_v52_final_eval.sh` | 评估入口（支持 wait / precheck / report-only） |
| `scripts/monitor_v52_training.sh` | 训练进度监控 |

```bash
# 监控训练（单次 / 每 5min 刷新）
bash scripts/monitor_v52_training.sh
bash scripts/monitor_v52_training.sh --watch

# 训练完成后 — 先 precheck
bash run_v52_final_eval.sh --precheck-only

# 正式泛化评估（~12 模型，约 40–60min）
bash run_v52_final_eval.sh

# 或：阻塞等待 S1+S2+v52b 全部完成再自动评估
bash run_v52_final_eval.sh --wait-training

# 仅重生成报告
bash run_v52_final_eval.sh --report-only
```

输出目录：`logs/evaluations/generalization_v52/GENERALIZATION_REPORT.md`

### 7.2 评估模型清单（12 个）

| 分组 | 模型 | ckpt | 解读重点 |
|------|------|------|---------|
| **v52a-S2** | best-val / best-jacobian / best-dual | 各 ckpt | **主候选**；S2 可能早停 <30ep |
| **v52a-S1** | best-val | ckpt_best_val.pth | Recovery 基线（无 ZD） |
| **v52b** | best-val / best-jacobian / best-dual | 各 ckpt | inject_recovery 线路 |
| **对照** | v51a / v50a / v49b / v45c | 历史 ckpt | Pareto 锚点 |

### 7.3 Phase A 验收判据（gdiag + MEDW400）

| 指标 | 阈值 | 报告章节 |
|------|------|---------|
| MEDW400 | ≤ **0.22°** | 四、时序聚合 |
| ZD max(R,P,Y) | ≤ **0.18°** | 十、gdiag Zero-Drift |
| Genuine Recovery @2° | ≥ **50%** | 十、Fixed-Inject |
| 0.5° + 1.0° Recovery | **> 0%** | 十、Multi-Magnitude |
| GS_medw | ≤ **0.65**（参考） | 十、GS_medw |

**通过逻辑**（不要求单 ckpt 全达标）：
- **v52a 成功**：S2-best-val MEDW≤0.22 且 S2-best-jacobian GenuineRec≥50%
- **v52b 成功**：best-val GenuineRec≥50% 且 MEDW400≤0.25
- **最优部署**：在达标 ckpt 中取 MEDW400 最小者

### 7.4 评估注意点（必读）

1. **不要用 multi-pass gdiag**  
   与 v51 一致，仅单帧 forward + MEDW 聚合；迭代 pass 已从 `evaluate_checkpoint.py` 移除。

2. **v52a-S2 可能提前结束**  
   `jacobian_early_stop` 或 val patience 触发后 epoch <30；评估以实际存在的 `ckpt_best_*.pth` 为准，勿假设 `ckpt_30.pth`。

3. **三个 ckpt 角色不同，必须分开评**  
   | ckpt | 适用场景 |
   |------|---------|
   | `ckpt_best_val.pth` | 部署 MEDW / 单帧 val 最优 |
   | `ckpt_best_jacobian.pth` | Recovery / Fixed-Inject 优先 |
   | `ckpt_best_dual.pth` | MEDW+Jacobian 联合 gate（可能锁 ep1，需对照解读） |

4. **联合解读 gdiag 三指标**  
   Genuine Recovery、Shortcut Proportion、Prediction Independence 必须一起看；高 PredIndep + 低 GenuineRec = under-correction（V51 模式）。

5. **Multi-Magnitude 是 V52 关键新增观测**  
   重点看 0.5°/1.0° Recovery 是否仍为负；V51 在此失败。

6. **训练依赖链**  
   `v52a_S2_full` 仅在 `batch_train.sh` 顺序跑完 S1 后才有 ckpt；eval precheck 会显示 missing 直到 S2 完成。

7. **对照组 pitch_vertical_bands 不同**  
   v45c=1, v49b=3, v52/v51/v50=5；对比 Per-frame 时注意架构差异。

8. **exclude_seqs=07**  
   与 v50/v51 评估一致，保证横向可比。

9. **NaN GUARD 偶发**  
   训练日志出现 `[NaN GUARD]` 单 batch skip 可接受；若累计 >10 次需检查该 run 的 ckpt 可靠性。

10. **S1 vs S2 对比**  
    若 S2 的 GenuineRec 相对 S1 下降 >15pp 但 MEDW 改善 <0.03°，判定 S2 过度 ZD，回退 S1 ckpt 部署。

### 7.5 训练监控命令

```bash
# 当前进度快照
bash scripts/monitor_v52_training.sh

# v52a 链: S1(100ep) → S2(30ep)  预计 ~3h + ~0.5h
# v52b: 100ep  预计 ~1.5h
# 并行两台机器时 v52b 通常先完成
```

### 7.6 结果决策树（评估后）

```
v52a-S2-best-jacobian: GenuineRec≥50% ?
  ├─ YES + S2-best-val MEDW≤0.22 → ★ v52a 生产候选
  └─ NO  → 对比 v52b / 保留 v50a 作 Recovery 路径

v52b-best-val: GenuineRec≥50% 且 MEDW≤0.25 ?
  ├─ YES → v52b 作 inject 线路候选
  └─ NO  → v52ab (S1 inject + S2 ZD) Phase 3

两者各优一项 → 双 ckpt 部署 (|init_err| 门控)
```

---

## 8. Phase 3 实验（V52 评估后，2026-06-10）

### 8.1 Phase A 结论摘要

| 线路 | 最佳 ckpt | MEDW400 | GenuineRec | 判定 |
|------|-----------|---------|------------|------|
| v52a-S1 | best-val | 0.290° | **62.9%** | Recovery ✅ MEDW ❌ |
| v52a-S2 | best-val | 0.286° | 62.2% | S2 几乎无效 |
| v52a-S2 | best-jacobian | **0.213°** | 33.9% | MEDW ✅ Recovery ❌ |
| v52b | best-val | 0.283° | 55.1% | 均未达标 |

**Phase A 未通过**；Recovery 解耦成功，MEDW 未追上 v49b/v51a (~0.17°)。

### 8.2 两台机器并行配置

| 机器 | 配置文件 | 实验链 | 假设 |
|------|---------|--------|------|
| **A** | `configs/v52ab_cf_bev_r.yaml` | S1 inject+jac → S2 轻 ZD | 在 v52a-S1 Recovery 上补 MEDW |
| **B** | `configs/v52c_cf_bev_r.yaml` | S1-v2 (v51a→jac) + multimag (inject+小角度) | MEDW 锚点拉回 Rec / 强化小角度 |

**机器 A 启动：**
```bash
bash batch_train.sh configs/v52ab_cf_bev_r.yaml --skip-pattern full    # smoke
bash batch_train.sh configs/v52ab_cf_bev_r.yaml --skip-pattern smoke   # full
```

**机器 B 启动：**
```bash
bash batch_train.sh configs/v52c_cf_bev_r.yaml --skip-pattern full
bash batch_train.sh configs/v52c_cf_bev_r.yaml --skip-pattern smoke
```

### 8.3 实验矩阵

| 实验 | version | ep | pretrain | 关键差异 |
|------|---------|-----|----------|---------|
| v52ab_S1_smoke/full | v52ab_S1_* | 15/100 | **v52a_S1_full** best-val | inject(0.4)+jac(0.15) |
| v52ab_S2_smoke/full | v52ab_S2_* | 5/20 | v52ab_S1 ckpt | ZD ramp **0.2→0.5**, jac≥0.80 早停 |
| v52a_S1_v2_smoke/full | v52a_S1_v2_* | 15/100 | **v51a** best-dual | jac(0.15), LR×0.8, 无 ZD/inject |
| v52c_multimag_smoke/full | v52c_multimag_* | 20/100 | v52a_S1_full | inject(0.5), multi_scale **0.35/0.35/0.30** |

### 8.4 Phase 3 验收判据

| 实验 | 成功条件 |
|------|---------|
| **v52ab** | S2-best-val MEDW≤0.22° **且** S1/S2 GenuineRec≥50% |
| **v52a-S1-v2** | MEDW≤0.25° **且** GenuineRec≥55% |
| **v52c** | 0.5° Recv≥40%, 1.0° Recv≥45%, GenuineRec≥55% |

### 8.5 Phase 3 全量评估结论（2026-06-11）

报告: `logs/evaluations/generalization_v52_all/GENERALIZATION_REPORT.md`

| 实验 | ckpt | MEDW400 | GenuineRec | Per-frame | 判定 |
|------|------|---------|------------|-----------|------|
| **v52a-S1-v2** | **best-dual** | **0.174°** ✅ | 28.7% ❌ | 1.747° | **部署候选** |
| **v52a-S1-v2** | best-val | 0.267° | **58.9%** ✅ | **1.091°** | **Recovery 候选** |
| v52ab-S1 | best-val | 0.385° | 62.5% | 1.099° | inject 损害 MEDW ❌ |
| v52ab-S2 | 三 ckpt 相同 | 0.254° | 37.0% | **1.587°** | S2 崩溃 ❌ |
| v52c multimag | best-val | 0.381° | 60.3% | 1.149° | 未达目标 ❌ |
| v52c multimag | best-jac | 0.289° | 58.9% | 1.158° | 未达目标 ❌ |

**Phase 3 总判定**:
- **v52a-S1-v2 部分成功**：MEDW 与 Recovery 分属不同 ckpt，与 v51 under-correction 模式一致（dual=聚合优，val=矫正优）
- **v52ab / v52c 终止**：不建议同方向继续微调
- **V52 系列 BEST 第 3**：v52a-S1-v2-dual (0.174°) 仅次于 v49b/v51a

---

## 9. 生产部署：v52a-S1-v2 双 ckpt 门控

### 9.1 模型资产

| 角色 | 文件 | 训练 version |
|------|------|-------------|
| **MEDW 聚合 / 小初值** | `ckpt_best_dual.pth` | `v52a_S1_v2_full` |
| **大扰动 Recovery** | `ckpt_best_val.pth` | `v52a_S1_v2_full` |

```text
logs/all_training_data/model_small_5deg_v52a_S1_v2_full/
  all_training_data_scratch/checkpoint/
    ckpt_best_dual.pth   # MEDW400=0.174°, GenuineRec=28.7%
    ckpt_best_val.pth    # MEDW400=0.267°, GenuineRec=58.9%, Per-frame=1.091°
```

**备选 Recovery**（GenuineRec 略高 63%，MEDW 略差）:
`model_small_5deg_v52a_S1_full/.../ckpt_best_val.pth`（v52a-S1，无 v51 锚点）

### 9.2 门控逻辑

部署时无 GT，用 **初始外参偏差估计** + **聚合帧数** 选 ckpt：

```
init_rot = geodesic_angle(init_extrinsic, nominal_extrinsic)   # 相对工厂/上次标定
n_frames = 当前 sequence 可用帧数

if n_frames >= MEDW_N:                    # 推荐 MEDW_N=200~400
    ckpt = ckpt_best_dual               # 在线聚合标定主路径
elif init_rot > GATE_DEG:               # 推荐 GATE_DEG=1.5°（可调 1.0~2.0°）
    ckpt = ckpt_best_val                # 大扰动 Recovery
else:
    ckpt = ckpt_best_dual               # 默认小偏差精细标定
```

| 场景 | 选 ckpt | 依据 |
|------|---------|------|
| 正常在线 MEDW 聚合（≥200 帧） | **dual** | MEDW 0.174°，与 v51a 同级 |
| 首帧 / 大角度初值（>1.5°） | **val** | GenuineRec 59%，Per-frame 优于 dual |
| 小偏差单帧 | dual | ZD 更低 (0.146° vs 0.167° max axis) |

**注意**：dual 与 v51a-dual 同属 under-correction ckpt，不可用于 2° Fixed-Inject 场景；大 inject 必须切 val。

### 9.3 推理参数（与训练一致）

```yaml
fusion_backend: cf_bev_r
rotation_only: true
bev_zbound_step: 4.0
pitch_vertical_bands: 5
backbone_type: swin
# 聚合：MEDW400 robust median on axis-angle
```

### 9.4 验收（上车前）

| 项 | 阈值 | 参考 eval |
|----|------|-----------|
| dual MEDW400 | ≤ 0.18° | 0.174° |
| val GenuineRec @2° | ≥ 55% | 58.9% |
| val 0.5°/1.0° Recv | > 0% | 34%/41% |
| 门控切换 | 大 inject 场景用 val | gdiag Fixed-Inject |

### 9.5 与 v49b/v51a 的关系

| 模型 | MEDW400 | GenuineRec | 建议 |
|------|---------|------------|------|
| v49b | **0.172°** | 49.5% | 纯 MEDW 极致 |
| v51a dual | 0.173° | 29.2% | 可被 v52a-S1-v2-dual 替代 |
| **v52a-S1-v2 dual+val** | 0.174° / 0.267° | 29% / **59%** | **推荐**双 ckpt |

### 9.6 泛化评估与部署的关系

| 脚本 | 作用 | 是否含双 ckpt 门控 |
|------|------|-------------------|
| `run_v52_final_eval.sh` | Phase A+3 全量对比 | ❌ 每 ckpt 独立跑 |
| `run_v52_deploy_eval.sh` | 部署 ckpt + `DEPLOY_ACCEPTANCE.md` | ❌ 分路径验收，非合并门控 |
| `evaluate_checkpoint.py` §4 | **部署模拟** MEDW uniform sample | ✅ 单 ckpt 多帧聚合 |
| `generalization_diag` | ZD / Fixed-Inject / Multi-Mag | ✅ 单 ckpt；**非**车端门控 |

**缺失能力**：`|init_err|>1.5° → val，否则 dual` 的**同窗联合 eval** 尚未实现（需双模型推理或离线合并 per-frame 预测）。

部署评估命令：
```bash
bash run_v52_deploy_eval.sh
bash run_v52_deploy_eval.sh --report-only   # 已有 eval 结果时
```

### 9.7 指标解读与上车门槛（诚实评估）

用户硬门槛 vs 当前 V52 最佳（v52a-S1-v2）：

| 指标 | 硬门槛 | dual (MEDW路径) | val (Recovery路径) | 说明 |
|------|--------|-----------------|-------------------|------|
| MEDW400 | ≤0.18° | **0.174°** ✅ | 0.267° | 仅聚合部署路径达标 |
| ZD max(R,P,Y) | ≤0.10° | **0.146°** ❌ | 0.167° | gdiag init=GT 无扰动 |
| GenuineRec @2° | ≥90% | **28.7%** ❌ | **58.9%** ❌ | 2° 全轴 inject 后 MEDW 残差 |
| 0.5° Recv | >0% | **-3.5%** ❌ | 34.4° ✅ | dual 小角度欠矫正 |

**为何看起来「泛化很差」**：
1. **Recovery% 与 MEDW 是不同任务**：MEDW 测「多帧聚合后离 GT 多近」；GenuineRec 测「2° 注入后矫正了多少」。dual 为 MEDW 优化，必然 Rec 低（~29%，同 v51a）。
2. **90% Recovery 当前全系列未达标**：历史最高 v45c **75%**；V52 val **59%**。2° inject 后 genuine 残差仍 ~1.4°，距 90%（残差<0.34°）差距大。
3. **ZD 0.1° 未达标**：best dual ZD max axis **0.146°**；这是无扰动固有偏差，与 MEDW 聚合误差独立。
4. **eval 未跑部署门控**：报告里同时列出 19 个 ckpt 的 gdiag，容易用 dual 的 Rec 或 val 的 MEDW 误判「整体失败」——应**分路径**看 §9.2。

**结论**：V52 双 ckpt 是 **MEDW 与 Recovery 的工程折中**，不是单模型全面达标；若硬要求 Rec≥90% 且 ZD≤0.1°，需 v52d 或新训练目标，不能靠 eval 门控 alone。

---

| 模型 | MEDW400 | GenuineRec | 建议 |
|------|---------|------------|------|
| v49b | **0.172°** | 49.5% | 纯 MEDW 极致，Recovery 弱 |
| v51a dual | 0.173° | 29.2% | 被 v52a-S1-v2-dual 替代（同 MEDW，可换 val 做 Recovery） |
| **v52a-S1-v2 dual+val** | 0.174° / 0.267° | 29% / **59%** | **推荐**：双 ckpt 覆盖 MEDW+Recovery |

---

## 10. v52d MGDA（备选，单 ckpt Pareto 突破）

**触发条件**：双 ckpt 门控在车端不可接受（内存/切换延迟），且需单模型同时 MEDW≤0.22° + GenuineRec≥50%。

### 10.1 目标

| 指标 | 目标 |
|------|------|
| MEDW400 | ≤ 0.22° |
| Genuine Recovery @2° | ≥ 50% |
| 0.5° + 1.0° Recv | > 0% |
| 单 ckpt | 是 |

### 10.2 方法

```
L_tasks = { L_pose, L_jacobian(0.10), L_zero_drift(λ_zd) }
λ_zd 从 0 线性 ramp 至 0.3（10ep），禁止 >0.5

每 step:
  g_i = ∇_θ L_i
  g*  = MGDA(g_1, g_2, g_3)          # 最小范数凸组合
  θ   ← θ - lr * g*

约束（val proxy，非 hard loss）:
  jacobian_eval ≥ 0.85  → 否则增大 w_jac
  medw_eval ≤ 0.25      → 否则增大 w_zd（上限 0.3）
```

**pretrain**: `v52a_S1_v2_full/ckpt_best_val.pth`（已有 Recovery 59%）或 `v51a/ckpt_best_dual.pth`（MEDW 锚点）

### 10.3 实验矩阵（smoke → full）

| 实验 | version | ep | 说明 |
|------|---------|-----|------|
| v52d_mgda_smoke | v52d_mgda_smoke | 15 | MGDA 3-loss 不 NaN |
| v52d_mgda_full | v52d_mgda_full | 80 | 单 ckpt 主实验 |

**实现前置**：`train_kitti.py` 增加 `--use_mgda` + 梯度收集；可参考 `torch.autograd.grad` 多 loss backward。

### 10.4 成功 / 失败决策

```
v52d MEDW≤0.22 且 GenuineRec≥50%  → 替换双 ckpt 为单 ckpt 生产
否则                              → 维持 §9 双 ckpt 门控，MGDA 归档
```

---

## 11. 实现清单

| 项 | 文件 | 状态 |
|----|------|------|
| inject_recovery_loss | `train_kitti.py` | ✅ |
| zero_drift ramp + jac early-stop + ckpt_best_jacobian | `train_kitti.py` | ✅ |
| CLI / batch_train 映射 | `batch_train.sh`, `start_training.sh` | ✅ |
| v52a 配置 (S2 优化) | `configs/v52a_cf_bev_r.yaml` | ✅ |
| v52b 配置 | `configs/v52b_cf_bev_r.yaml` | ✅ |
| 设计文档 | `docs/V52_DESIGN.md` | ✅ |
| v52 泛化 eval | `configs/eval_generalization_v52.yaml` | ✅ |
| 评估入口 | `run_v52_final_eval.sh` | ✅ |
| 训练监控 | `scripts/monitor_v52_training.sh` | ✅ |
| Phase 3 机器 A | `configs/v52ab_cf_bev_r.yaml` | ✅ |
| Phase 3 机器 B | `configs/v52c_cf_bev_r.yaml` | ✅ |
| Phase 3 全量 eval | `configs/eval_generalization_v52.yaml` → `generalization_v52_all` | ✅ |
| 双 ckpt 部署规格 | `docs/V52_DESIGN.md` §9 | ✅ |
| v52d MGDA 设计 | `docs/V52_DESIGN.md` §10 | 📋 待实现 |
| v52d 训练配置 | `configs/v52d_cf_bev_r.yaml` | ✅ Phase1 手动 λ |
| 部署 eval | `configs/eval_generalization_v52_deploy.yaml` | ✅ |
| 部署 eval 入口 | `run_v52_deploy_eval.sh` | ✅ |
| 部署验收摘要 | `scripts/summarize_v52_deploy_metrics.py` | ✅ |

---

## 12. 参考

- V51 泛化报告: `logs/evaluations/generalization_v51/GENERALIZATION_REPORT.md`
- V51 设计: `docs/V51_DESIGN.md`
- v50a 预训练: `logs/all_training_data/model_small_5deg_v50a_optuna_no_cons_S1_full/`
