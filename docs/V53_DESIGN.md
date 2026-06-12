# V53 设计方案：双通路 Pose Head + 轴解耦迭代（结构突破）

**日期**: 2026-06-11  
**状态**: Phase 3 完成 — 全量训练 + 泛化 eval 已出报告（2026-06-11）  
**前置**: V52 双 ckpt 门控 eval 未达标；v52d MGDA full 已完成  
**基座**: CF-BEV-R (`fusion_backend=cf_bev_r`)，Stage-1/2 不变

---

## 1. 动机：V52 为何到顶

### 1.1 门控 eval 最终结果（2026-06-11）

| 指标 | 硬门槛 | 双 ckpt 门控实测 | 判定 |
|------|--------|-----------------|------|
| MEDW400 | ≤ 0.18° | **0.215°** | ❌ |
| ZD max(R,P,Y) | ≤ 0.10° | **0.146°** | ❌ |
| GenuineRec @2° | ≥ 90% | **62.0%** | ❌ |
| 0.5° / 1.0° Recv | > 0% | 34.4% / 41.0% | ✅ |

### 1.2 根因（结构层，非纯调参）

| 现象 | gdiag 证据 | 结构根因 |
|------|-----------|----------|
| ZD 0.146° | init=GT 仍有偏差 | **单一 Δq 头**无法同时拟合 identity 映射 |
| Rec 62% | PredIndep=**GENUINE** | 非 shortcut，是 **矫正增益不足**；Δq 与 \|δ_init\| 未显式耦合 |
| MEDW 0.21° | seq03 outlier ~0.65° | 帧间系统偏差 + 全局头无序列偏差建模 |
| 双 ckpt 仍失败 | dual ZD 好 / val Rec 好 | Pareto 被拆到两个完整模型，**前沿仍填不满 90%×0.1° 矩形** |

V52 关闭 GIN（`gin_channels=0`）；V48 Partial GIN 的设计意图（IN→ZD, LN→Recovery）从未在 V52 主线上验证。

**结论**：继续调 λ / 双 ckpt / MGDA（v52d）只能在 **同结构 Pareto 前沿** 上移动，无法突破 90% GenuineRec + 0.1° ZD。**必须改 Stage-3 Pose Head。**

---

## 2. V53 核心思想

> **显式分解「序列零漂移偏差」与「扰动条件化矫正」，网络内可学习路由，替代车端双 ckpt。**

```
Stage-1/2 (不变):  Swin+DLA → CrossAttn → LocalCorr → F_corr

Stage-3 (V53 新增):
  F_bias  = BiasPath(F_corr)              # 慢变 / 序列级偏差，ZD 专用
  F_rec   = RecoveryPath(F_corr)          # 扰动敏感，无 IN
  δ       = estimate_init_error(T_init)   # magnitude_head 或 axis-angle
  w       = Router(δ, n_frames_proxy)   # 小 δ → bias；大 δ → recovery
  Δq_bias = Head_bias(F_bias)
  Δq_rec  = Head_rec(F_rec, δ) · gain(δ)   # JACG 增益
  q       = compose(q_init, w·Δq_rec + (1-w)·Δq_bias)

Stage-3b (可选 ADIR): 分轴 2-step 迭代 refine q
```

与 V52 双 ckpt 对照：

| | V52 双 ckpt | V53 DP-Head |
|--|------------|-------------|
| 路由 | 车端 \|init_err\|>1.5° | 网络内可微 Router |
| 内存 | 2× 完整模型 | +~5M 参数（双头） |
| 训练 | 两个 ckpt 独立优化 | 联合 loss + 路由监督 |
| MEDW+Rec | 分路径达标，合并不达标 | 目标：单 ckpt 同时逼近 |

---

## 3. 模块设计

### 3.1 Dual-Path Pose Head（DP-Head）— 主模块

**替换** `cf_bev_r_calib.py` 中 `CorrTransformerHead` 输出的单一 `rotation` quaternion。

#### BiasPath（零漂移通路）

```python
class BiasPath(nn.Module):
    """序列级慢变偏差：深层 IN 去域 + 全局 pool → 小幅度 Δq_bias"""
    def __init__(self, feat_dim):
        self.norm = nn.InstanceNorm2d(feat_dim, affine=True)  # 或 Partial GIN
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), nn.GELU(),
            nn.Linear(feat_dim // 2, 4),  # quaternion delta
        )
        # 初始化接近 identity（小权重）
```

- **训练信号**：`zero_drift_loss` + `zero_perturbation_prob` batch 路由 w→0
- **目标**：ZD max(R,P,Y) ≤ 0.10°

#### RecoveryPath（扰动矫正通路）

```python
class RecoveryPath(nn.Module):
    """保留 perturbation 敏感度：LayerNorm + init_err 条件化"""
    def __init__(self, feat_dim):
        self.norm = nn.LayerNorm(feat_dim)  # 不用 IN
        self.corr_head = CorrTransformerHead(...)  # 复用现有结构
        self.err_encoder = nn.Linear(3, feat_dim)  # δ_init RPY (rad)
```

- **训练信号**：inject_recovery_loss + jacobian_loss + 大扰动 batch 路由 w→1
- **目标**：GenuineRec @2° ≥ 75%（Phase A）→ 85%（Phase B）

#### MagnitudeRouter（可学习门控）

```python
class MagnitudeRouter(nn.Module):
    """替代车端 gate_deg=1.5° 硬阈值"""
    def forward(self, delta_init_deg, mag_pred_deg, training=True):
        # delta_init: (B,3) axis-angle or RPY error estimate
        # 训练: 用 GT init error；推理: magnitude_head + T_init 一致性
        x = torch.cat([delta_init_deg, mag_pred_deg.unsqueeze(-1)], dim=-1)
        w = self.mlp(x).sigmoid()  # w→1: recovery, w→0: bias
        return w
```

**路由监督**（训练期，无需新标注）：

```
L_route = BCE(w, target_w)
target_w = 1 if ||δ_init|| > gate_deg else 0
gate_deg = 1.5° (与 §9 部署策略对齐)
```

#### 输出合成

```python
q_bias = normalize(quat_mul(q_init, delta_q_bias))
q_rec  = normalize(quat_mul(q_init, delta_q_rec))
q_pred = slerp(q_bias, q_rec, w)  # 或可学习 compose
```

### 3.2 Jacobian-Aware Correction Gain（JACG）

在 RecoveryPath 输出上施加 **init_err 条件增益**：

```
gain = softplus(MLP([||δ_init||, mag_pred]))   # (B,1) 或 (B,3) 逐轴
Δq_rec = gain ⊙ direction(F_rec)               # direction 来自 CorrTransformerHead
```

- 小 δ → gain≈0 → 不破坏 ZD
- 大 δ → gain 放大 → 提升 GenuineRec
- 与现有 `jacobian_loss` 对齐：gain 的 ∂/∂δ 应 ≥ 0.85

### 3.3 Axis-Decoupled Iterative Refinement（ADIR）— Phase B 可选

在 DP-Head 输出后增加 **2-step 分轴 refine**（推理时可固定 T=2）：

```
T ← q_pred from DP-Head
for t in range(T):
    Δq_r = RollQuery(F_corr, T)      # 水平条带 pool
    Δq_p = PitchBranch(z_bands, F_rgb)  # 升级现有 FrontViewPitchBranch
    Δq_y = YawQuery(F_corr, T)
    T ← compose(Δq_r, Δq_p, Δq_y) · T
    if max(|Δq_*|) < ε: break
```

- **与 v36 native_cross_iter 区别**：每步 **轴-specific query**，非同一 head 重复
- **目标**：单步矫正 ~0.7°/轴 → 两步达 90% Rec（残差 <0.34°/轴）

### 3.4 Online Bias Subtractor（OBS）— 部署 wrapper，非训练必须

不改主网络，部署层在线估计序列偏差：

```
b_seq = robust_median({q_pred_i - q_anchor})   # 前 N 帧
q_deploy = q_pred ⊗ inv(b_seq)^α               # α 随 n_frames ramp
```

- 专攻 MEDW400；对 inject Rec 无直接帮助
- 可与 DP-Head 的 BiasPath 互补（OBS 处理 seq03 类 outlier）

---

## 4. 损失函数

### 4.1 任务损失（继承 V52）

| Loss | 权重 | 路由 |
|------|------|------|
| L_pose | 1.0 | 全局 |
| L_jacobian | 0.10 | Recovery batch (w>0.5) |
| L_zero_drift | ramp 0.1→0.3 | Bias batch (w<0.5) |
| L_inject_recovery | 0.5 | Recovery dedicated 10% |
| L_route | 0.1 | 全 batch |

### 4.2 路由 batch 构造

```
zero_perturbation_prob:     0.20   → target_w=0, 监督 BiasPath
multi_scale 2°:             0.15   → target_w=1, 监督 RecoveryPath
continuous 0~5°:            0.65   → target_w=sigmoid((||δ||-1.5)/0.5)
```

### 4.3 分阶段验收 proxy（训练中）

| Proxy | 阈值 | 频率 |
|-------|------|------|
| MEDW200 | ≤ 0.25° | 每 5 ep |
| jacobian_eval | ≥ 0.85 | 每 5 ep |
| val GenuineRec proxy | inject 2° batch residual | 每 10 ep |

---

## 5. 验收目标（分阶段，诚实）

### Phase A — V53 smoke（15ep）

| 指标 | 阈值 | 说明 |
|------|------|------|
| 训练 | 无 NaN | 三 loss + route 可训 |
| ZD max(R,P,Y) | ≤ 0.12° | 优于 V52 0.146° |
| GenuineRec @2° | ≥ 65% | 优于 V52 val 59% |
| 路由 w 分布 | 双峰清晰 | 可视化 w vs \|\|δ\|\| |

### Phase B — V53 full（80ep）

| 指标 | 阈值 |
|------|------|
| MEDW400 | ≤ 0.20° |
| ZD max(R,P,Y) | ≤ 0.10° |
| GenuineRec @2° | ≥ 75% |
| 0.5°+1.0° Recv | > 0% |
| 单 ckpt | 是 |

### Phase C — Stretch（需 ADIR + OBS）

| 指标 | 阈值 | 备注 |
|------|------|------|
| MEDW400 | ≤ 0.18° | 部署硬门槛 |
| GenuineRec @2° | ≥ 85% | 历史 v45c 最高 75% |
| GenuineRec @2° | **≥ 90%** | **全系列未验证可行**；若 Phase B 未达，需重新评估门槛或 eval 协议 |

**建议分阶段上车标准**（替代单一 90% 硬门槛）：

| 部署场景 | MEDW400 | ZD | GenuineRec |
|----------|---------|-----|------------|
| 在线聚合标定 | ≤0.18° | ≤0.10° | — |
| 大扰动 Recovery | — | — | ≥75%（Phase B） |
| 理想单 ckpt | ≤0.18° | ≤0.10° | ≥85%（Stretch） |

---

## 6. 实验矩阵

| 实验 | version | ep | 结构 | pretrain | 说明 |
|------|---------|-----|------|----------|------|
| **v53a_dphead_smoke** | v53a_dphead_smoke | 15 | DP-Head only | v52a-S1-v2-val | ✅ CONVERGED |
| **v53a_dphead_full** | v53a_dphead_full | 80 | DP-Head + JACG | v52a-S1-v2-val | ✅ ep71 dual MEDW200=0.104° |
| v53b_adir_smoke | v53b_adir_smoke | 15 | DP-Head + ADIR T=2 | v53a best | ✅ |
| v53b_adir_full | v53b_adir_full | 60 | 同上 | v53a best | ✅ dual@ep1；val@ep51 |
| v53c_partial_gin | v53c_partial_gin_full | 80 | Partial GIN 对照 | v52a-val | ✅ ep31 MEDW200=0.030° |
| v53d_obs_deploy | — | — | OBS wrapper | v53a ckpt | 📋 待 eval 后决定 |

**并行策略**：

| 机器 | 实验 | 命令 |
|------|------|------|
| A | v53a smoke → full | `bash batch_train.sh configs/v53a_dphead_cf_bev_r.yaml --skip-pattern full/full` |
| B | v53c Partial GIN smoke | `bash batch_train.sh configs/v53c_partial_gin_cf_bev_r.yaml --skip-pattern full` |
| B | v52d full（已在跑） | 保持不变 |
| A | v53b ADIR（v53a 完成后） | `bash batch_train.sh configs/v53b_adir_cf_bev_r.yaml --skip-pattern smoke` |
| 任一 | 泛化 eval | `bash run_v53_eval.sh` |

---

## 7. 实现清单

| 项 | 文件 | 状态 |
|----|------|------|
| `BiasPath` / `RecoveryPath` / `MagnitudeRouter` | `kitti-bev-calib/modules/dp_pose_head.py` | ✅ |
| CF-BEV-R Stage-3 集成 | `cf_bev_r_calib.py` | ✅ |
| `--use_dp_head` / `--route_loss_weight` CLI | `train_kitti.py`, `start_training.sh` | ✅ |
| `L_route` 路由监督 | `cf_bev_r_calib.py` forward | ✅ |
| v53a smoke/full 配置 | `configs/v53a_dphead_cf_bev_r.yaml` | ✅ 训练完成 |
| v53c Partial GIN 配置 | `configs/v53c_partial_gin_cf_bev_r.yaml` | ✅ 训练完成 |
| v53b ADIR 配置 | `configs/v53b_adir_cf_bev_r.yaml` | ✅ 训练完成 |
| ADIR 模块 | `kitti-bev-calib/modules/adir_refine.py` | ✅ |
| 泛化 eval | `configs/eval_generalization_v53.yaml`, `run_v53_eval.sh` | ✅ 2026-06-11 |
| v53 门控 eval | `run_v53_eval.sh --gate-eval` | ✅ MEDW 验收已出 |
| gdiag DP-Head 路由可视化 | `evaluate_checkpoint.py` | 📋 Phase 4（分析 route_w 塌缩） |
| OBS 部署 wrapper | `utils/obs_bias_subtractor.py` | 📋 Phase C（seq03 outlier） |
| eval DP-Head 加载修复 | `evaluate_checkpoint.py` auto-detect | ✅ 2026-06-11 |

### 7.1 开关隔离

```python
# hybrid_triple_calib.build_calib_model
if fusion_backend == 'cf_bev_r':
    if getattr(args, 'use_dp_head', 0):
        model = CFBevRCalibV53.from_args(...)  # 或 CFBevRCalib(use_dp_head=True)
    else:
        model = CFBevRCalib.from_args(...)
```

不影响 v52/v52d 既有实验。

---

## 8. 与 v52d 的关系

| | v52d MGDA | V53 DP-Head |
|--|-----------|-------------|
| 改动层 | 损失梯度平衡 | **网络结构** |
| 目标 | MEDW≤0.22 & Rec≥50% | ZD≤0.10 & Rec≥75%+ |
| 能否替代双 ckpt | 可能（若达标） | **设计目标** |
| 能否达 90% Rec | unlikely | Phase C + ADIR，**待验证** |

**决策树**：

```
v52d full 达标 (MEDW≤0.22, Rec≥50%)
  ├─ v53a smoke ZD/Rec 优于 v52d → 继续 v53a full，v52d 归档为 baseline
  └─ v53a 无提升 → 维持 v52d 单 ckpt 或 V52 双 ckpt 部署

v52d 未达标
  └─ v53a 为主攻方向；v52 双 ckpt 维持 MEDW 路径
```

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Router 塌缩（恒 w=0 或 w=1） | L_route + 强制 dedicated batch 比例 |
| BiasPath 过强压 Recovery | gain(δ) 下限 + jacobian early-stop |
| ADIR 推理延迟 | T=2 固定，仅 Refine 步；或蒸馏到单步 |
| 90% Rec 仍不可达 | 分阶段上车标准（§5）；考虑分轴 Rec 门槛 |
| seq03 outlier | OBS + robust median；或 per-seq bias 估计 |

---

## 12. Smoke 训练监控 Checklist

训练日志路径：`logs/all_training_data/model_small_5deg_<version>/train.log`

### 12.1 通用（所有 V53 实验）

| 检查项 | 位置 / 关键字 | 通过标准 |
|--------|--------------|----------|
| 无 NaN | `Loss:` 行 | 无 `nan` / `inf` |
| DDP 启动 | 开头 | `DDP enabled: 8 GPUs` |
| Pretrain 加载 | epoch 0 前 | `Load pretrain model from ...` 无报错 |
| MEDW200 eval | 每 5 ep | 有数值且 < 0.35°（smoke 粗门槛） |
| Jacobian eval | 每 5 ep | overall > 0.80（smoke） |

### 12.2 v53a DP-Head 专项

| 检查项 | 关键字 | 通过标准 |
|--------|--------|----------|
| DP-Head 启用 | 启动 banner | `V53 DP-Head enabled` |
| route_loss | `route_loss` | ep5+ 存在且 < 0.5 |
| route_w_mean | `route_w_mean` | 0.2–0.8 之间（非塌缩到 0 或 1） |
| zero_drift_loss | dedicated batch | ep5+ 有值 |
| inject_recovery_loss | dedicated batch | ep5+ 有值 |
| magnitude_loss | 每 epoch | 收敛下降趋势 |

**Smoke 通过判定**（15ep 结束）：
- CONVERGED 或 train loss 稳定下降
- 无 NaN / OOM 中断
- `route_w_mean` 在 zero_perturb batch 附近 < 0.3，inject batch 附近 > 0.5（抽查 tensorboard 或 log）

### 12.3 v53c Partial GIN 专项

| 检查项 | 关键字 | 通过标准 |
|--------|--------|----------|
| Partial GIN | 启动 banner | `Partial GIN enabled: 128/256` |
| gin_gate | gate stats（若有） | mean ∈ [0.3, 0.7] |

**对照结论（训练 proxy）**：v53c val MEDW200 **0.030°** 远优于 v53a **0.104°**，但 v53c **未启用 DP-Head**——需在 test gdiag 上确认：Partial GIN  alone 是否已够，或 DP-Head 是否 over-engineered。

### 12.4 v53b ADIR 专项

| 检查项 | 关键字 | 通过标准 |
|--------|--------|----------|
| ADIR 启用 | 启动 banner | `V53b ADIR enabled (steps=2)` |
| GenuineRec proxy | inject_recovery_loss 下降 | 低于 v53a 同 epoch 基准 |

### 12.5 Smoke 通过后下一步

```bash
# v53a full
bash batch_train.sh configs/v53a_dphead_cf_bev_r.yaml --skip-pattern smoke

# 泛化 eval
bash run_v53_eval.sh --precheck-only
bash run_v53_eval.sh
bash run_v53_eval.sh --gate-eval    # dual/val ckpt 门控 gdiag
```

---

## 13. 参考

- V52 设计与门控 eval: `docs/V52_DESIGN.md` §9–§10
- 门控验收: `logs/evaluations/generalization_v52_deploy/DEPLOY_GATE_ACCEPTANCE.md`
- CF-BEV-R 架构: `docs/V42_CF_BEV_DESIGN.md`, `cf_bev_r_calib.py`
- V48 Partial GIN: `cf_bev_r_calib.py::GatedInstanceNorm`
- 历史 Recovery 最高: v45c GenuineRec ~75%

---

## 14. Phase 3 训练结论（2026-06-11）

### 14.1 Val MEDW200 proxy（训练集 split，非 test gdiag）

| 实验 | Best ep | MEDW200 max(R,P,Y) | Jacobian | 备注 |
|------|---------|-------------------:|---------:|------|
| v53a DP-Head | 71 | **0.104°** | 0.934 | route_w_mean 塌缩 → **0.98**（几乎恒 Recovery） |
| v53b ADIR | 1 (dual) / 51 (val) | **0.051°** (dual@ep1) | 0.932 | dual gate 停在 ep1，后续 val 更好但未更新 dual |
| v53c Partial GIN | 31 | **0.030°** | 0.985 | 无 DP-Head；val proxy 最优 |

### 14.2 关键训练信号

1. **Router 塌缩（v53a/v53b）**：ep1 `route_w_mean≈0.62` → ep71 **0.98**。BiasPath 在训练中几乎未被使用；网络退化为「单 Recovery 头 + 微弱 bias 分支」。
2. **v53c 优于 v53a（proxy）**：Partial GIN（128/256）在 val MEDW 上显著更好，说明 V48 遗留结构对 ZD 有效，且不一定需要 DP-Head 复杂度。
3. **v53b dual gate 早停**：MEDW200 最佳在 ep1（0.051°），之后 val pose 继续改善但 dual ckpt 未刷新——eval 应同时看 **best-val** 与 **best-dual**。
4. **Smoke 全部通过**：三实验 80ep（v53b 60ep）均无 NaN，CONVERGED。

### 14.3 与 V52 门控 baseline 对比（test，V52 已完成）

| 指标 | V52 双 ckpt 门控 | Phase B 目标 |
|------|-----------------|-------------|
| MEDW400 | 0.215° ❌ | ≤0.20° |
| ZD max(R,P,Y) | 0.146° ❌ | ≤0.10° |
| GenuineRec@2° | 62.0% ❌ | ≥75% |

V53 能否在 test 上突破上述三项，以 §15 eval 报告为准。

---

## 15. Phase 3 泛化 eval 结论（2026-06-11）

**报告**: `logs/evaluations/generalization_v53/GENERALIZATION_REPORT.md`  
**门控验收**: `logs/evaluations/generalization_v53/DEPLOY_GATE_ACCEPTANCE.md`

> 注：首轮 eval 因 `evaluate_checkpoint.py` 未加载 `use_dp_head` 导致 v53a MEDW≈29° 假象，已修复（state_dict 自动检测 + ckpt args 透传）并重跑。

### 15.1 Phase B 门槛对照（test gdiag + MEDW400）

| 指标 | Phase B 门槛 | v53a dual | v53c GIN | v53b ADIR | v52d MGDA | v52 双 ckpt 门控 |
|------|-------------|----------:|---------:|----------:|----------:|-----------------:|
| MEDW400 | ≤0.20° | **0.248°** ❌ | **0.209°** ❌ | 0.255° ❌ | **0.180°** ✅ | 0.215° ❌ |
| ZD max(R,P,Y) | ≤0.10° | **0.203°** ❌ | 0.156° ❌ | 0.213° ❌ | **0.136°** ❌ | 0.146° ❌ |
| GenuineRec@2° | ≥75% | **31.0%** ❌ | 38.7% ❌ | 37.6% ❌ | 43.7% ❌ | 62.0% ❌ |
| 0.5°/1.0° Recv | >0% | -2.8% / 8.4% ✅ | -3.9% / 7.6% ✅ | -0.4% / 6.6% ✅ | 2.2% / 11.9% ✅ | 34% / 41% ✅ |

**v53a 门控（dual→ZD / val→Inject）** — 全部 FAIL：

| 指标 | 门槛 | 门控值 |
|------|------|-------:|
| MEDW400 | ≤0.18° | **0.248°** ❌ |
| ZD max(R,P,Y) | ≤0.10° | **0.203°** ❌ |
| GenuineRec@2° | ≥75% | **31.0%** ❌ |
| 0.5° Recv | >0% | -2.8% ❌ |

### 15.2 系列内排名（V53）

| 维度 | 最优 | 说明 |
|------|------|------|
| MEDW400 | **v53c Partial GIN** 0.209° | 无 DP-Head，优于 v53a 0.248° |
| ZD | v53c 0.156° | v53a 0.203° 劣于 V52 门控 0.146° |
| GenuineRec | v53b 37.6% ≈ v53c 38.7% | 均远低于 v45c 75.4%、V52 62% |
| Per-frame | v53b 1.454° | 略优于 v53a/v53c ~1.48° |

**核心结论**：V53 结构突破 **未通过 Phase B 任一门禁**。DP-Head（v53a）在 test 上 **全面劣于** Partial GIN 对照（v53c），且 Rec 大幅倒退；ADIR（v53b）无 Rec 增益。

### 15.3 与 V52 / v52d 对照

| 部署场景 | 推荐模型 | 依据 |
|----------|----------|------|
| MEDW 在线聚合 | **v52d MGDA** 0.180° | V53 全线未达 0.18° 硬门槛 |
| MEDW 备选 | **v53c Partial GIN** 0.209° | V53 内最优，结构更简单 |
| GenuineRec | **v45c** 75.4% / **v52a** 59% | V53 最高仅 ~39% |
| 单 ckpt 综合 | **v52d** 或 **v53c** | 放弃 v53a DP-Head 主线 |

Router 塌缩（训练 `route_w_mean→0.98`）与 test 上 BiasPath 未改善 ZD/MEDW 一致：**双通路设计未在推理中生效**。

### 15.4 seq03 outlier（MEDW 瓶颈）

| 模型 | seq03 MEDW400 Rot |
|------|----------------:|
| v52d | 0.693° |
| v53c | 0.713° |
| v53a | 0.649° |

seq03 仍是 MEDW 主要拖累；OBS / per-seq bias 仍为 Phase C 必要项。

---

## 16. 后续优化方向（eval 驱动，按优先级）

### P0 — 主线决策（已明确）

1. **归档 v53a DP-Head 为 failed branch** — test 全面劣于 v53c，Rec 倒退至 31%。
2. **MEDW 部署继续 v52d**（0.180°）或切换 **v53c Partial GIN**（0.209°，V53 内最优）。
3. **Rec 部署仍用 v45c / v52a** — V53 未带来 Recovery 提升。
4. **与上车方确认 Phase B 75% Rec 门槛** — 全系列历史最高 ~75%（v45c），90% 不可行。

### P1 — v53c+ 低成本迭代（推荐下一实验）

| 实验 | 改动 | 预期 |
|------|------|------|
| **v53c_full_tune** | v53c 基座 + v52d 式 MGDA loss | MEDW 向 0.18° 靠拢 |
| **v53e_gin_mgda** | Partial GIN 128/256 + MGDA | 合并 v53c ZD 与 v52d MEDW 优势 |
| **v53f_gin_jacg** | v53c + JACG 增益（无 DP-Head） | 提升 Rec 而不引入 router |

### P2 — DP-Head v2（仅当坚持双通路）

- Router 防塌缩：分阶段训练、增大 `route_loss_weight`、router 熵正则
- BiasPath 用 Partial GIN norm（合并 v53c 有效结构）
- 训练/推理一致性验证：`route_w` 在 zero_perturb batch 上 <0.3

### P3 — Phase C Stretch

| 方向 | 触发 | 预期 |
|------|------|------|
| **OBS wrapper** | seq03 ~0.65–0.71° | MEDW400 → ≤0.18° |
| **v53b ADIR 蒸馏** | 单步延迟约束 | 暂不建议（Rec 无增益） |
| **per-seq robust median** | 部署层 | 降低 seq03 outlier |

### P4 — 不建议

- 继续 v53a 同配置 full 重训（router 塌缩已复现）
- V52 双 ckpt 微调（test 三项 FAIL）
- 纯 λ 调参无结构变化

---

## 17. V53e 方案（Partial GIN + MGDA）— Phase 4

**状态**: 已实现，待训练  
**配置**: `configs/v53e_gin_mgda_cf_bev_r.yaml`  
**设计动机**: v53c (GIN) test MEDW 0.209° + v52d (Pareto loss) test MEDW 0.180° → 合并两者优势

### 17.1 结构

| 组件 | 来源 | 说明 |
|------|------|------|
| Partial GIN 128/256 | v53c | `use_gated_instance_norm=1`, `gin_channels=128` |
| DP-Head | **关闭** | `use_dp_head=0` |
| MGDA 三任务 backward | **新增** | `utils/mgda.py` + `--use_mgda 1` |
| pretrain | v53c dual | 同架构热启动 |

### 17.2 Loss / MGDA 任务

```
L_tasks = { L_pose, L_zd (ramp 0.1→0.3), L_inject (w=0.25, dedicated 8%) }
每 step: g* = MGDA(g_pose, g_zd, g_inject)   # 最小范数凸组合
L_jacobian: 仍独立 backward（与 v52d 一致）
```

### 17.3 验收目标

| 指标 | smoke (15ep) | full (80ep) |
|------|-------------|-------------|
| 训练 | 无 NaN | CONVERGED |
| val MEDW200 | < 0.15° | < 0.10° |
| test MEDW400 | — | ≤ 0.18° (stretch) |
| GenuineRec@2° | — | ≥ 50% (stretch 75%) |

### 17.4 启动命令

```bash
# smoke
bash batch_train.sh configs/v53e_gin_mgda_cf_bev_r.yaml --skip-pattern full

# full
bash batch_train.sh configs/v53e_gin_mgda_cf_bev_r.yaml --skip-pattern smoke

# eval（训练完成后取消 eval_generalization_v53.yaml 中 v53e 注释）
bash run_v53_eval.sh
```
