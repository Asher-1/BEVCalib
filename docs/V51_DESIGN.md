# V51 设计方案：Zero-Drift + Genuine Recovery 双目标训练

**日期**: 2026-06-09  
**状态**: 已实现训练配置 + loss；待启动实验  
**前置**: v50 泛化评估 (`generalization_v50_final`)

---

## 1. 背景与动机

v50 泛化评估表明 **Zero-Drift (ZD) 与 Genuine Recovery 存在 Pareto 权衡**，无单一 checkpoint 同时满足部署 ZD 与高 Recovery：

| 模型 | ZD max(R,P,Y) | Genuine Rec | MEDW400 | 模式 |
|------|--------------|-------------|---------|------|
| v49b-no-cons | **0.156°** | 49.5% | **0.196°** | Shortcut（低 ZD，低 Recovery） |
| v45c-S1 | 0.332° | **75.4%** | 0.374° | Genuine（高 Recovery，高 ZD） |
| v50a-no-cons | 0.185° | 57.3% | 0.263° | 折中 |

**v50 结论**:
- 关闭 `consistency_loss` 显著改善部署 MEDW400（v50a 0.263° vs T32 0.333°）
- 多 pass 迭代推理对 Recovery 提升有限（~+7%），且会偏袒 shortcut 模型 → **已从 gdiag 评估中移除**
- 部署精度由 **单帧 forward + MEDW 时序聚合** 决定，不依赖同帧迭代 pass

**V51 目标**: 在 v50a 基础上，通过显式双代理 loss 同时优化 ZD 与 Recovery，寻找 Pareto knee。

---

## 2. 验收目标

### Phase A（第一版可验收）

| 指标 | 阈值 | 评估方式 |
|------|------|---------|
| ZD max(R,P,Y) | **< 0.20°** | test_data_v2 gdiag |
| Genuine Recovery | **≥ 60%** | test_data_v2 gdiag Fixed-Inject |
| MEDW400 | **< 0.25°** | test_data_v2 时序聚合 |
| Jacobian overall | **> 0.85** | 训练 val KPI |

### Phase B（stretch 目标）

| 指标 | 阈值 |
|------|------|
| ZD max(R,P,Y) | < 0.15° |
| Genuine Recovery | ≥ 70% |
| MEDW400 | < 0.20° |

---

## 3. 评估协议变更（V51）

### 3.1 移除：gdiag 多 pass 迭代推理

**原因**:
- 1-pass → 3-pass Recovery 仅 +7%，无法达到 70%
- 未训练的多 pass 会虚抬 shortcut 模型 Recovery
- 与部署协议不一致（部署 = 单帧 forward + MEDW）

**变更** (`evaluate_checkpoint.py`):
- 删除 `gdiag_iterative_passes` 参数及 `iterative_inference` 测试块
- `_run_single_pass` 仅保留 **单次 forward**
- **保留** MEDW per-sequence 聚合（gdiag 与主评估均不变）

### 3.2 保留：MEDW 时序聚合

部署与泛化主指标继续使用 `TemporalCalibrationAggregator` / MEDW200/400/800。

---

## 4. 训练 Loss 设计

### 4.1 总损失

```
L_total = L_pose                          # 主 pose + axis + reproj
        + λ_jac  · L_jacobian             # Recovery 代理 (V51 新增启用)
        + λ_zd   · L_zero_drift           # ZD 代理 (V51 新增)
        + λ_oc   · overcorrection_scale   # 防 zero-drift 过校正
        + λ_mag  · L_magnitude            # 已有
        + λ_gin  · L_gin_gate_reg         # GIN 开启时
```

### 4.2 Recovery 代理：Jacobian Supervision Loss

**已有实现** (`train_kitti.py::_compute_jacobian_supervision_loss`)

约束 `d(correction)/d(bias) ≈ 1`，与 gdiag 的 correction 定义一致：

```
correction = init_err - out_err   (per-axis degrees)
J_est = Δcorrection / Δbias
L_jac = SmoothL1(J_est, 1.0)
```

| 参数 | V51 默认值 | 说明 |
|------|-----------|------|
| `jacobian_loss_weight` | **0.10** | Optuna 搜 [0.05, 0.25] |
| `jacobian_loss_start_epoch` | **10** | 主 loss 稳定后启用 |
| `jacobian_loss_probe_deg` | **2.0** | 与 gdiag inject=2° 对齐 |
| `jacobian_loss_interval` | **4** | 每 4 batch 算一次（省算力） |

### 4.3 Zero-Drift 代理：Zero-Drift Loss

**V51 新增** (`train_kitti.py::_zero_drift_axis_error_deg`)

在 `init_T = gt_T` 的 batch 上，直接惩罚 `R_pred` 相对 `R_gt` 的 per-axis 偏差：

```python
zd_rpy = euler(R_pred @ R_gt^T)   # 目标 → 0
L_zd = SmoothL1(zd_rpy, 0)
```

| 参数 | V51 默认值 | 说明 |
|------|-----------|------|
| `zero_drift_loss_weight` | **1.0** | 独立于主 pose loss 的 ZD 权重 |
| `zero_drift_loss_start_epoch` | **5** | 较早启用 |
| `zero_drift_dedicated_ratio` | **0.15** | 强制 15% batch 纯 init=gt |
| `zero_perturbation_prob` | **0.25** | 随机 init=gt 采样（叠加 dedicated） |

与 v43 `zero_perturbation_prob` 的区别：V51 在 init=gt batch 上增加 **独立加权** 的 `L_zd`，而非仅靠主 loss 间接约束。

### 4.4 其他关键训练策略（继承 v50a）

| 参数 | 值 | 理由 |
|------|-----|------|
| `consistency_loss_weight` | **0.0** | v50a 证：关 consistency 利部署 |
| `overcorrection_penalty` | **2.0** | 防 init=gt 时过校正（v47） |
| `quat_bias_reset_alpha` | 0.1 | Optuna T32 |
| `axis_weights` | 1.5,5.5,1.5 | Optuna T32 Pitch 加权 |
| `cosine_T0` | 40 | Optuna T32 |
| pretrain | v50a best-val ckpt | 热启动 |

---

## 5. 训练阶段调度

```
Epoch 1–4:   仅 L_pose（warmup）
Epoch 5+:    + L_zero_drift（init=gt batch）
Epoch 10+:   + L_jacobian（每 4 batch）
全程:        overcorrection_penalty, magnitude_loss
```

---

## 6. 实验矩阵

配置文件: `configs/v51_cf_bev_r.yaml`

| 实验 | version | epochs | 用途 |
|------|---------|--------|------|
| **v51a_dual_loss_S1_smoke** | v51a_dual_loss_S1_smoke | 10 | loss 稳定性 smoke |
| **v51a_dual_loss_S1_full** | v51a_dual_loss_S1_full | 150 | Phase A 主实验 |

### 启动命令

```bash
# Smoke（先跑，确认无 NaN）
bash batch_train.sh configs/v51_cf_bev_r.yaml --skip-pattern full

# Full 150ep
bash batch_train.sh configs/v51_cf_bev_r.yaml --skip-pattern smoke
```

日志目录: `logs/all_training_data/model_small_5deg_v51a_dual_loss_S1_*/`

---

## 7. 泛化评估计划

训练完成后，在 test_data_v2 上跑 gdiag（单 pass + MEDW）：

```bash
# 待 v51 训练完成后创建 eval yaml 并执行
bash run_v51_final_eval.sh   # 或扩展 run_v50_final_eval.sh
```

**主排名指标**（与 v50 一致，无 multi-pass）:
1. MEDW400（部署）
2. GS_medw（综合）
3. ZD max(R,P,Y) + Genuine Recovery（联合解读）

---

## 8. 后续：V51b Optuna HPO（Phase 2）

在 v51a 验证双 loss 有效后，扩展 `optuna_search.py`：

### 新增搜索维度

```python
"jacobian_loss_weight": [0.05, 0.10, 0.15, 0.20, 0.25]
"zero_drift_loss_weight": [0.5, 1.0, 1.5, 2.0, 3.0]
"zero_drift_dedicated_ratio": [0.10, 0.15, 0.20]
"zero_perturbation_prob": [0.15, 0.25, 0.35]
```

### 约束目标函数

```python
if zd_max_rpy > 0.20:
    return inf  # hard prune
score = 0.40 * medw/0.20 + 0.35 * (1 - gen_rec/100) + 0.25 * zd/0.20
```

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| λ_zd 过大 → Recovery 下降 | smoke 监控 Jacobian eval；Optuna 搜权重 |
| λ_jac 过大 → MEDW 单帧变差 | start_epoch=10，weight≤0.25 |
| 双 loss 梯度冲突 | 分阶段启用；jacobian interval=4 |
| Phase B (70% Rec) 未达 | Phase 2 加 Fixed-Inject recovery loss |

---

## 10. 实现清单

| 项 | 文件 | 状态 |
|----|------|------|
| 移除 gdiag multi-pass | `evaluate_checkpoint.py` | ✅ |
| `zero_drift_loss_*` | `train_kitti.py` | ✅ |
| CLI / batch_train 映射 | `batch_train.sh`, `start_training.sh` | ✅ |
| 训练配置 | `configs/v51_cf_bev_r.yaml` | ✅ |
| 设计文档 | `docs/V51_DESIGN.md` | ✅ |
| v51 泛化 eval yaml | `configs/eval_generalization_v51.yaml` | 待训练完成后 |
| Optuna v51 HPO | `optuna_search.py` | 待 v51a 结果 |

---

## 11. 参考

- v50 泛化报告: `logs/evaluations/generalization_v50_final/GENERALIZATION_REPORT.md`
- Optuna T32 报告: `logs/all_training_data/optuna_trials/reports/OPTUNA_T32_REPORT.md`
- v50 训练配置: `configs/v50_cf_bev_r.yaml`
- Jacobian loss 设计: `docs/V41_DESIGN.md` § Jacloss
