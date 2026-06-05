# V41 实验设计：GMP × V32 × 严格 Dual Gate 验收

> 状态：执行中（2026-05-30，已吸收 V40 P1d 经验）  
> 前置：`docs/V40_DESIGN.md`（GMP 架构）、`docs/V32_DESIGN.md`（T_init 不变性）、`docs/V40_JACOBIAN_MEDW_CURVES.md`（V40 KPI 分裂）  
> 配置：`configs/v41_gmp_a_v32_baseline.yaml` / `v41_gmp_b_v32_jacloss.yaml` / `v41_gmp_c_v32_jacloss_match.yaml`  
> 训后泛化：`configs/eval_generalization_v40_v41.yaml`

---

## 0. 一句话目标

在 **GMP scratch 训练**（不嫁接 v20 ckpt）上，叠加 **V32 init-invariance + Jacobian 训练监督**，通过三组 ablation 验证 match/corr 与 diff_epnp 能否让模型在 **与 train 同分布的 val 划分** 上同时满足 **MEDW 精度** 与 **Jacobian 自适应** 两条红线。

**同分布 val 过不了 dual gate = 训练或网络设计有问题**，不是「预期可能不过、先看相对改善」。

---

## 1. 必过红线（Red Line Gate）

### 1.1 验收标准（val 划分，与 train 同数据集 / 同扰动分布）

| KPI | 条件 | 说明 |
|-----|------|------|
| **MEDW200** | `max(Roll, Pitch, Yaw) < 0.10°` | 逐轴检查，**禁止**仅用合成角 `rot` 代替 |
| **Jacobian @ ±3°** | `overall` 及 `R/P/Y` **均** `> 0.85` | init 扰动 sweep：`correction = init_err - out_err`，`J = d(correction)/d(bias)` |

- val 与 train 共享 `all_training_data`，仅序列/帧划分不同；扰动均为 **±5°**（`eval_angle_range_deg=5`）。
- 此处过不了 → 模型未学到可用的 init→GT 映射，和/或 init 敏感性（Jacobian）未建立 → **架构或训练目标有问题**。
- **未过 gate 的 ckpt 禁止进入 test/bag 泛化评估**（避免用失败模型做误导性结论）。

### 1.2 Verdict 判定（训练结束自动输出）

训练结束后 `log_dir` 自动生成：

| 文件 | 内容 |
|------|------|
| `CONVERGENCE_REPORT.md` | 人类可读：verdict、验收标准、每 eval epoch KPI 表 |
| `convergence_report.json` | 机器可读：`converged` / `best_dual` / `kpi_history` |
| `training_summary.md` | 扩展摘要：Best Dual Gate 或 `NOT CONVERGED` |

| Verdict | 含义 | 后续 |
|---------|------|------|
| **CONVERGED** | 至少一个 epoch 通过 dual gate → 存在 `ckpt_best_dual.pth` | 可进 `run_generalization_eval.py` |
| **NOT_CONVERGED** | 有 KPI 记录但从未通过 | **训练/网络不达标**，需改 loss/架构/权重，不得放宽 0.10° |
| **NO_KPI_EVAL** | 未启用 MEDW/Jacobian eval | 配置错误 |

### 1.3 Dual Gate 实现（`train_kitti.py`，2026-05-29 起）

**旧逻辑（已废弃）**：`medw_result['rot'] < 0.10` 且 `jacobian overall > 0.85` — 合成角可掩盖单轴超标，Pitch Jacobian 可长期 ≈0 仍存 ckpt。

**新逻辑**：

```python
# MEDW：max(R,P,Y) < dual_gate_medw_max（默认 0.10°）
# Jacobian：overall 及 roll/pitch/yaw 均 > dual_gate_jacobian_min（默认 0.85）
_dual_gate_pass(medw_result, jac_result, medw_max_deg, jac_min)
```

- `ckpt_best_medw.pth`：按 `max(R,P,Y)` 最小化选取（不再按合成 `rot`）。
- `ckpt_best_dual.pth`：仅 gate PASS 时保存，含完整 R/P/Y 与 Jacobian 分轴字段。
- 每 eval epoch train.log 打印 `max(R,P,Y)=...` 与 `Dual gate PASS/FAIL` 明细。

> **注意**：V40 等在旧代码下启动的训练不含上述逻辑；V41 必须用新版 `train_kitti.py` **重新启动**。

---

## 2. 背景：V40 为何不够

| 现象 | 数据 | 含义 |
|------|------|------|
| val MEDW 尚可 | v40 p0ab ep21 ≈ **0.17°** | 同分布精度「看起来还行」 |
| Jacobian 全程 WEAK | Pitch J ≈ **-0.1（≈0 灵敏度）** | init 扰动几乎不改变输出 → **shortcut** |
| bag inject Recover 低 | v40 GMP **11–13%** | MEDW 与 Jacobian **脱钩**，部署 ckpt 不可信 |
| v20 pitch-wt3 对照 | bag Recover **42–71%** | 旧 BEV 路径在同任务上远优于 GMP scratch |

V40 设计文档中 P0 阶段 Jacobian 仅监控、不设硬门槛；V41 将 **同分布 dual gate 提升为唯一交付红线**，不再用「MEDW 趋势 OK 即可进 P1」的宽松策略。

---

## 3. 三组实验设计（A / B / C）

三台 8 卡机器并行，**除下表外参数完全一致**（含 `seed=42`）。

| 机器 | 配置 | version | 相对上一组的增量 |
|------|------|---------|------------------|
| **A** | `v41_gmp_a_v32_baseline.yaml` | `v41_gmp_a_v32_baseline` | V32 + geo + **jacloss 0.1**；**无 match** |
| **B** | `v41_gmp_b_v32_jacloss.yaml` | `v41_gmp_b_v32_jacloss` | A + **Match/Corr**（`correspondence_loss_weight=5.0`）；**无 diff_epnp** |
| **C** | `v41_gmp_c_v32_jacloss_match.yaml` | `v41_gmp_c_v32_jacloss_match` | B + **`differentiable_epnp=1`**（`diff_epnp_warmup_epochs=2`） |

### 3.1 公共配方（v20_v8recipe_pitch_wt3 + GMP scratch）

| 类别 | 参数 | 值 |
|------|------|-----|
| 扰动 | `angle_range_deg` / `eval_angle_range_deg` | 5° |
| 轴 loss | `axis_weights` | `1.0,3.0,1.0`（Pitch 加权） |
| 数据 | `dataset` | `all` → `all_training_data` |
| 采样 | `max_frames_per_seq` | **1000**（与 `sample_step` 互斥；替代 v20 的 `sample_step=2`） |
| 采样 | `pose_aware_sampling` | true（`poses/` 自动位于 dataset_root 下） |
| 训练 | `num_epochs` / `eval_epoches` / `save_ckpt_per_epoches` | 200 / 20 / 20 |
| LR | `lr_schedule` / `step_size` / `learning_rate` | step / 50 / 1e-4 |
| GMP | `fusion_backend` | `geo_match_proj` |
| GMP | `freeze_backbone` / `pretrain_ckpt` | 1 / null（**scratch，不嫁接 v20 ckpt**） |
| V32 | `continuous_tinit_noise` / `tinit_dropout_prob` | 1 / 0.3 |
| V32 | `consistency_loss_weight` / start | 0.5 / **epoch 10** |
| Jacloss | `jacobian_loss_weight` / start / probe | 0.1 / **epoch 10** / **±3.0°**（与 eval 对齐） |
| Geo | `appearance_loss_weight` / start | 0.1 / **epoch 5** |
| Gate | `enable_dual_gate_ckpt` / thresholds | 1 / MEDW max<**0.10°**, Jac>**0.85** |
| KPI | `enable_medw_eval` / `enable_jacobian_eval` | 1 / 1 |
| KPI | `jacobian_eval_angle_deg` / batches | ±**3.0°** / **2 batches** |

### 3.2 B/C 专有（Match 路径）

| 参数 | B / C |
|------|-------|
| `use_match_head` / `use_local_correlation` | 1 / 1 |
| `compose_mode` | `match_then_refine` |
| `num_correspondences` | 64 |
| `correspondence_supervision` | 1 |
| `correspondence_loss_weight` | **5.0**（V40 P1d 验证；P1c 1.0 时 fallback ~77%） |
| `match_valid_ratio_min` | 0.15（低于默认 0.3，减少 match fallback） |
| `differentiable_epnp` | B: 0；C: **1** |
| `diff_epnp_warmup_epochs` | C: **2** |

### 3.3 A 组 intentionally 关闭 Match

`use_match_head=0` 时 `HybridPoseHead` 不建 match/epnp 分支，`PoseComposer` 在 `R_match=None` 时走 **refine_only**。A 组回答：**仅靠 geo + V32 + jacloss，GMP 能否过 gate**。

---

## 4. 启动与监控

### 4.1 启动命令

```bash
# 机器 A / B / C 各跑一条
bash batch_train.sh configs/v41_gmp_a_v32_baseline.yaml
bash batch_train.sh configs/v41_gmp_b_v32_jacloss.yaml
bash batch_train.sh configs/v41_gmp_c_v32_jacloss_match.yaml
```

日志目录：`logs/all_training_data/model_small_5deg_{version}/`

### 4.2 训练中检查

```bash
# 确认新版 dual gate 描述已出现在 train.log 开头
grep "max(MEDW200 R,P,Y)" logs/all_training_data/model_small_5deg_v41_gmp_a_v32_baseline/train.log

# 每 20 epoch 的 gate 状态
grep -E "max\(R,P,Y\)|Dual gate" logs/.../train.log

# B/C：Match 路径是否学起来（V40 P1d 目标 fallback<40%、corr<15px）
grep -E "match_fallback|corr=" logs/all_training_data/model_small_5deg_v41_gmp_b_v32_jacloss/train.log | tail -20
```

### 4.3 训练结束

```bash
cat logs/all_training_data/model_small_5deg_v41_gmp_a_v32_baseline/CONVERGENCE_REPORT.md
```

### 4.4 仅 CONVERGED 后跑泛化

```bash
python run_generalization_eval.py --config configs/eval_generalization_v40_v41.yaml
```

`eval_generalization_v40_v41.yaml` 中 V41 三组默认取 `ckpt_best_dual.pth`，fallback `ckpt_best_medw.pth`。

---

## 5. 结果解读矩阵（严格口径）

| 结果模式 | 诊断 |
|----------|------|
| **A/B/C 全部 NOT_CONVERGED** | GMP scratch + 当前 V32/jacloss 配方 **整体不达标** → 改训练目标或可训练范围，不是改 gate |
| **仅 B 或 C CONVERGED** | Match（± diff_epnp）是必要条件；A 证明 geo-only 不够 |
| **A 过、B/C 不过** | Match 分支引入干扰或新 shortcut |
| **MEDW 过、Jacobian 不过** | 回归能拟合 val，但对 init 不敏感 → **shortcut 未消除** |
| **Jacobian 过、MEDW 不过** | 局部线性响应有，整体标定精度不够 |
| **Pitch Jac 长期 < 0.85** | 与 V40 同病；pitch 信息未进入可训练 head → **网络路径问题** |

**禁止的解读方式**：

- ❌ 「gate 太严，0.17° 已经不错」— 0.10° 是同分布交付红线，不是 stretch goal  
- ❌ 「先看 B 比 A 好就行」— 相对比较不能替代 gate PASS  
- ❌ 「没过 gate 也先跑 bag」— 会重复 V40 MEDW/Jacobian 脱钩误导  

---

## 6. 失败根因假设（训练 / 网络侧）

以下均为 **需在 NOT_CONVERGED 时排查的方向**，不是放宽 gate 的理由。

### 6.1 Shortcut 未打掉（最可能，V40 已验证）

- val MEDW 下降但 Pitch Jacobian ≈ 0 → 模型仍「透传 T_init + 小修正」  
- V32 三件套（dropout + consistency + jacloss）在 GMP 上可能权重不足；**consistency/jacloss 已自 ep10 开启**（原 ep20 过晚）

### 6.2 Jacobian 监督（2026-05-30 已与 eval 对齐）

| 项 | 训练 jacloss | eval gate | 状态 |
|----|--------------|-----------|------|
| 角度 | ±**3.0°** | ±**3.0°** | ✅ 已对齐 |
| eval 采样 | — | **2 batches** | ✅ 降低 KPI 方差 |
| 频率 | 每 **4 batch** 一次（代码默认） | — | ⚠️ 仍偏 sparse；NOT_CONVERGED 时可改代码 |

### 6.3 Loss ramp 时间表

| Epoch | 开启项 |
|-------|--------|
| 5 | geo（appearance + depth） |
| **10** | consistency(0.5) + jacloss(0.1) |

ep 10–30 为 V32+Jac 主学习窗；KPI 震荡需看分轴趋势，不能单点否定。

### 6.4 架构约束

- `freeze_backbone=1`：可训练参数主要在 fusion / match / refine  
- GMP scratch、无 v20 ckpt：比 v20 pitch-wt3 **更难**，但 **不能因此降低 gate**

### 6.5 数据采样（三组公平，但与 v20 不可比绝对值）

- V41：`max_frames_per_seq=1000`（pose-aware 后再 cap）  
- v20：`sample_step=2` → 长序列 often **>1000 帧/seq**  

三组之间公平；若 train loss 仍高且 KPI 无下降趋势，属于 **训练配置未给够学习信号**，应加数据量/epoch，**不是改 0.10°**。

### 6.6 已知无效/低影响配置

| 参数 | 说明 |
|------|------|
| `warmup_epochs: 5` | `lr_schedule=step` 时不走 LinearWarmup，**当前无效** |
| `early_stopping_patience: 30` | 30×20=600 epoch > 200，**实际不触发** |

---

## 7. NOT_CONVERGED 后的改法方向（待实验验证）

仅在 **三组结果 + CONVERGENCE_REPORT KPI 表** 出齐后选型：

| 优先级 | 动作 | 说明 |
|--------|------|------|
| 代码 | jacloss 每 batch 计算 | 去掉 `train_kitti.py` 中 `batch_index % 4` 限制 |
| 数据 | `max_frames_per_seq` → 2500 或 `sample_step: 2` | 学习信号不足时 |
| 时长 | `num_epochs` → 400 | 对齐 v40 长训 |
| 可选 | B/C 加 progressive 5°→3° | 参考 v40 dual_gate；需新对照实验 |
| 末手段 | 解冻部分 backbone | 显存/稳定性成本高 |

**已纳入默认配置（§11）**：corr_w=5.0、V32/jacloss ep10、probe/eval ±3°、jacobian_eval_batches=2。

---

## 8. 与 V40 实验链关系

```
V40 P0ab/P1d          →  MEDW 趋势 OK，Jacobian WEAK，bag 失败
        ↓
V41 A/B/C             →  同分布 dual gate 必过红线 + V32/jacloss 全员开启
        ↓ CONVERGED
run_generalization    →  test_data + bag inject（异分布）
```

V40 `configs/v40_gmp_paper_v20recipe_dual_gate.yaml` 中 F0/F1/E1 链仍保留作历史对照；V41 是 **更严格的同分布交付层**，不替代 V40 架构文档中的 P0–P3 分 phase 叙述，但 **覆盖其「Jacobian 仅监控」的宽松 gate**。

---

## 11. V40 经验迁移（2026-05-30 已落地配置）

V40 P1c/P1d 训练结论及 V41 采纳方式：

| V40 教训 | 数据 / 现象 | V41 采纳 | 不采纳 |
|----------|-------------|----------|--------|
| Match 需强监督 | P1c corr_w=1.0 → fallback **77%**；P1d corr_w=**5.0** → corr **~10px**、fallback **~30%** | B/C：`correspondence_loss_weight: **5.0**` | 以为 match 开就行 |
| Match ≠ 消 shortcut | P1d Pitch J **≈0**，bag Recover **11–13%** | 全员 V32 + jacloss + **0.10° gate** | 只靠 MEDW 选 ckpt |
| Jacobian 训练/评估一致 | v40 probe 与 ±10° eval 脱节 | `jacobian_loss_probe_deg: **3.0**` = eval | — |
| V32 不宜过晚 | 前 20 epoch 无 consistency/jacloss 易走 shortcut | start epoch **10**（geo 仍 ep5） | — |
| KPI 方差 | 单 batch Jacobian sweep 抖动 | `jacobian_eval_batches: **2**` | — |
| Progressive 降扰动 | dual_gate 10°→3° 压 MEDW 至 **~0.16°** | **暂不默认**（固定 5° 保持三组可比） | 直接抄 v40 10° 配方 |
| v39 pretrain | ep1 MEDW ~0.34° | **scratch 不变**（V41 目标） | 嫁接 v40/v39 ckpt |
| P2b 增广 | 提升异分布鲁棒 | gate 阶段 **关闭**增广 | gate 前加 augment |

### B/C 训练期 Match 监控目标（来自 P1d）

| 指标 | 目标 | 来源 |
|------|------|------|
| `match_fallback_ratio`（train） | **< 40%** | P1d ep45+ |
| `correspondence_loss`（L_corr） | **< 15px** | P1d ep54 |
| Pitch Jacobian | **> 0.85** | V41 红线（v40 未达成） |

> **重要**：Match 学起来是 B/C 的必要条件，**不是** gate PASS 的充分条件；A 组无 match，专门验证「geo + V32 + jacloss  alone 是否够」。

---

## 12. Match 修复 v41.1（2026-06-01）

### 为何 V41 里 Match 比 P1d 更难

| 因素 | P1d | V41 B/C |
|------|-----|---------|
| 训练目标 | 60ep Match 专项 | scratch + geo + V32 ±30° + jacloss |
| L_corr 有效点 | valid_gt 足够 | **valid_gt≈2%**（init 错时 GT 投影 mask 过严） |
| Fallback | corr_w=5 后 ~30% | **valid_gt<0.15 → 几乎 100% fallback** |
| EPnP | 通常 ≥4 有效点 | **~1–2 有效点 → identity** |

根因：**不是 corr_w 太小，而是 gate + GT-mask 让 Match 路径被绕过，L_corr 经常无梯度。**

### v41.1 代码 + 配置（B/C yaml 已更新）

| 参数 | 值 | 作用 |
|------|-----|------|
| `correspondence_loss_start_epoch` | 20 | ep1–20 先 pose+geo，再开 corr |
| `correspondence_loss_warmup_epochs` | 10 | ep21–30 线性 ramp corr_w 0→5 |
| `match_corr_validity_mode` | **init** | L_corr 用 T_init 可见点监督 T_gt UV（非 valid_gt 交集） |
| `match_disable_fallback` | 1 | 不再把 R_match 置 identity |
| `match_gate_use_init_ratio` | 1 | fallback 统计用 init 可见率 |
| `match_valid_ratio_min` | 0.05 | 对齐实际 init 可见率 |
| `match_confidence_threshold` | 0.1 | 更多 correspondence 参与 |
| `match_epnp_min_points` | 2 | 稀疏 valid 时仍跑 EPnP |
| `match_phase_noise_max_deg` | 10 | Match 阶段 cap V32 连续扰动 |

### 续训后监控（ep25/40/60）

```bash
grep -E "correspondence_loss:|match_fallback|match_valid_ratio_init|epnp_insufficient" \
  logs/all_training_data/model_small_5deg_v41_gmp_b_v32_jacloss/train.log | tail -20
```

目标：`L_corr` **下降** toward 15px，`fallback` **<40%**，`valid_init` **>10%**。

---

## 9. 文档与配置索引

| 资源 | 路径 |
|------|------|
| A 配置 | `configs/v41_gmp_a_v32_baseline.yaml` |
| B 配置 | `configs/v41_gmp_b_v32_jacloss.yaml` |
| C 配置 | `configs/v41_gmp_c_v32_jacloss_match.yaml` |
| 泛化 eval | `configs/eval_generalization_v40_v41.yaml` |
| 训练入口 | `batch_train.sh` → `train_universal.sh` → `kitti-bev-calib/train_kitti.py` |
| V32 机制 | `docs/V32_DESIGN.md` |
| V40 GMP 架构 | `docs/V40_DESIGN.md` |
| V40 KPI 曲线 | `docs/V40_JACOBIAN_MEDW_CURVES.md` |

---

## 10. 变更记录

| 日期 | 变更 |
|------|------|
| 2026-06-01 | Match v41.1：分阶段 corr、init 可见性 L_corr、disable fallback、gate/noise/epnp 调参；§12 |
| 2026-05-30 | 吸收 V40 P1d：corr_w=5.0（B/C）；V32/jacloss ep10；jac probe/eval ±3°；jacobian_eval_batches=2；新增 §11 |
| 2026-05-29 | 初版：三组 A/B/C 设计、同分布 dual gate 必过红线、结果解读矩阵、V40 背景、train_kitti dual gate 逐轴实现与 CONVERGENCE_REPORT 自动输出 |
