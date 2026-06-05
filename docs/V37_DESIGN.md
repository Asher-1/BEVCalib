# V37 实验设计报告：Long Train + Large Batch + ProjFusion 对齐

> 状态：**训练已完成（Ep400）**；终态泛化评估见 **`docs/V37_GENERALIZATION_REPORT.md`**  
> 对比基线：v36 `model_small_5deg_v36_native_cross_pointgpt` Ep191 best val  
> 评估数据：`/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2`（12 序列 × 200 帧）  
> 后续方案：**v38 暂停 → v39 见 `docs/V39_DESIGN.md`**

---

## 1. 实验动机

v36 在训练集 val 上达到 **1.15°（±5°）**，但 `test_data_v2` 泛化仍有差距（单帧 ~1.17°，MEDW200 ~0.41°）。v37 目标：

1. 对齐 ProjFusion 长训配方：大 batch、±10° 扰动、`extend_ratio=2.5`
2. 提高 GPU 利用率（BS=128/GPU，~20GB/46GB L20）
3. 从 v36 Ep191 warm-start，加速收敛
4. 验证 shortcut 克服能力（Jacobian）与部署指标（MEDW）

**配置文件**：`configs/v37_native_cross_pointgpt_long.yaml`

---

## 2. v36 vs v37 配置对比

| 参数 | v36 | v37 |
|------|-----|-----|
| batch/GPU | 16 | **128** |
| 全局 BS | 128 | **1024** |
| LR | 1e-4 | **4e-4**（线性缩放） |
| 扰动（训练） | ±5° | **±10°** |
| max_frames_per_seq | 500 | **1000** |
| num_epochs | 200 | **400** |
| steps/epoch | **~65** | **16** |
| 总 optimizer steps | **~13,000** | **~6,400** |
| extend_ratio | 1.0 | **2.5** |
| pretrain | scratch | **v36 Ep191 ckpt** |
| PointGPT | KITTI tiny | KITTI tiny（未换） |

---

## 3. 训练收敛对比（截至 Ep241）

> **注意**：v36 val 为 ±5°，v37 val 为 ±10°，val 数值不可直接比大小，看趋势即可。

### 3.1 Val best rot 轨迹

| Epoch | v36 Val (±5°) | v37 Val (±10°) |
|------:|--------------:|---------------:|
| 21 | 2.13° | 1.87° |
| 81 | 1.63° | 1.38° |
| 141 | 1.31° | 1.27° |
| 191/241 | **1.15°** | **1.26°** |

v37 在更宽 val 域下 Ep241 已接近 v36 最终水平，说明 **in-domain 学习能力 OK**。

### 3.2 Train rot（Ep240 附近）

| | v36 Ep191 | v37 Ep241 |
|--|-----------|-----------|
| Train rot | ~2.0° | ~2.7° |

v37 train 偏高是 ±10° 扰动的正常现象，不代表欠拟合。

### 3.3 Cosine restart 波动

`cosine_T0=50, Tmult=2` 导致 Ep61、Ep161 等 restart 后 val **暂时回弹**（Ep161 val=1.67°），Ep221→241 再次下降至 1.26°。判收敛应看 **best val 曲线**，不应看单次 val。

---

## 4. 标准泛化评估协议（强制）

**所有跨版本对比必须使用同一协议**，否则结论无效。

| 项 | 规定 |
|----|------|
| 数据集 | `test_data_v2` |
| 采样 | `--use_full_dataset --eval_max_frames_per_seq 200` |
| **扰动** | **`--angle_range_deg 5.0`（强制 ±5°）** |
| extend_ratio | 从 ckpt 读取（v37=2.5）；`evaluate_checkpoint.py` / `diagnose_jacobian.py` 必须传入 |
| 指标 | Per-frame Rot、MEDW200、Jacobian ±5°/±10° sweep |
| Shortcut | `tools/diagnose_jacobian.py`，Overall J > 0.3 = ADAPTIVE |

```bash
# MEDW + 时序聚合
python evaluate_checkpoint.py --mode eval \
  --ckpt_path <ckpt> \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2 \
  --output_dir <out> \
  --use_full_dataset --eval_max_frames_per_seq 200 \
  --angle_range_deg 5.0 --batch_size 8

# Jacobian
python tools/diagnose_jacobian.py --ckpt_path <ckpt> \
  --angle_range 5.0 --n_batches 10 --batch_size 4 \
  --output <out>/jacobian_5deg.json
```

**反面教训（v37 Ep241 首次 MEDW）**：
- 未传 `--angle_range_deg 5.0` → 使用 ckpt 自带 ±10° → MEDW200 **虚高 6.67°**（Pitch 轴崩溃）
- `evaluate_checkpoint.py` 曾遗漏 `extend_ratio` → 已修复（2026-05-26）

---

## 5. 泛化评估结果（Ep241，±5° 公平对比）

| 指标 | v36 Ep191 | v37 Ep241 | Δ |
|------|-----------|-----------|---|
| Per-frame Rot | 1.175° | **0.906°** | **-23%** ✓ |
| MEDW200 | **0.413°** | 0.457° | +11% ✗ |
| MEDW50 | 0.406° | 0.453° | +12% |

### 5.1 Shortcut / Jacobian

| 扰动 sweep | v36 Overall J | v37 Overall J |
|------------|--------------:|--------------:|
| **±5°** | 0.549 (ADAPTIVE) | **0.840** (ADAPTIVE) ✓ |
| **±10°** | 0.290 (WEAK) | **0.879** (ADAPTIVE) ✓ |

v37 在 **shortcut 克服** 上显著优于 v36，尤其 ±10° 域外从 0.29→0.88。

### 5.2 PointGPT 预训练 A/B（v36 ckpt，±5°，5905 帧）

| PointGPT 权重 | single_forward | iter3 |
|---------------|---------------:|------:|
| KITTI | **1.454°** | 0.988° |
| nuScenes swap | 2.944° (+102%) | 5.568° |

**结论**：在已用 KITTI PointGPT 训好的 v36 权重上 swap nuScenes **严重退化**；KITTI 与当前 head 已耦合，swap 无效。nuScenes 价值需 **从头重训 calib head** 才可评估。

产物目录：
- `logs/all_training_data/pointgpt_ab_v36_test_v2/ab_summary.json`
- `logs/.../medw200_test_v2_eval_5deg/`
- `logs/.../checkpoint/jacobian_5deg_ep241.json`

---

## 6. 根因分析：为何单帧改善但 MEDW200 略退化？

### 6.1 【高】有效 optimizer steps 减半（主因之一）

| | v36 | v37 |
|--|-----|-----|
| steps/epoch | ~65 | **16** |
| total steps | ~13,000 | **~6,400** |

v37 虽然每 epoch 见过更多帧（1024×16 ≈ 16k vs 128×65 ≈ 8k），但 **参数更新次数仅为 v36 的 49%**。大 batch 提升梯度估计质量，但 **无法完全补偿 step 数减半**。

对 MEDW 这类 **序列级统计量**，需要模型在多种轨迹段上稳定收敛；step 不足时，单帧均值可能改善（±10° 训练增强自适应），但 **跨序列一致性**（MEDW）可能滞后。

**量化估算**：
- 若目标是 step parity（~13k steps），v37 在 16 step/ep 下需 **~800 epoch**，或改为 **BS=64（32 step/ep）+ 400 epoch ≈ 12.8k steps**。

### 6.2 【高】max_frames=1000 仍远低于序列全长

`all_training_data` 单序列长度 6k–26k 帧（见 `data_len.json`），`max_frames=1000` + pose_aware 后训练集约 **16,800 帧/epoch**。

| max_frames | 估计 train 帧/epoch | steps/ep (BS=1024) |
|:----------:|:--------------------:|:------------------:|
| 500 (v36) | ~8,400 | ~65 (BS=128) |
| 1000 (v37) | ~16,800 | **16** |
| **5000 (提议)** | **~84,000** | **~82** |

**max_frames→5000 的预期收益**：
- 每 epoch 覆盖更长轨迹，pose 多样性 ↑
- steps/epoch 从 16 → ~82，**与 v36 step 密度相当**
- 有利于 MEDW 学习序列级稳定输出

**风险**：
- epoch 时间 ×~5（数据加载 已占 10–30% profiling）
- pose_aware 在 5000 cap 下仍可能只保留 5–15% 关键帧，需验证实际采样率
- 需配合 **BS=64 或 grad_accum** 避免 step 过多导致总训练时间过长

**建议 v38**：`max_frames_per_seq=3000~5000`，`batch_size=64/GPU`（global 512），LR=2e-4，目标 steps/ep ≈ 40–80。

### 6.3 【高】PointGPT 域差距：KITTI 360° vs 车队前向稀疏点云

**现象**：
- PointGPT 在 KITTI/nuScenes 上预训练，点云为 **车载 360° 扫描**，左右/后方均有稠密点
- `all_training_data` 为 **前向 ROI 点云**（`foreground=true`），侧后方面点稀少或缺失
- FPS 128 groups 在 KITTI 上学的是 **全向几何分布**；在车队数据上 group 中心大量落在前方路面/车辆，**特征语义错位**

**A/B 证据**：
- KITTI PointGPT + v36 head：可用（1.45° @ test_v2）
- nuScenes swap（不 retrain head）：灾难性 2.94°
- 说明 PointGPT 特征与 cross-attn head **强耦合**，预训练分布至关重要

**建议：基于车队数据重训 PointGPT（V38 子项）**

| 步骤 | 说明 |
|------|------|
| 1. 数据 | 从 `all_training_data` 导出 ProjFusion PointGPT 格式（8192 pts，max_depth=50m） |
| 2. 配置 | 复制 `ProjFusion/cfg/pointgpt/finetune_kitti_tiny.yaml` → `finetune_fleet_tiny.yaml`，SEQ 指向车队序列 |
| 3. 训练 | 在 ProjFusion 框架内 MAE/Contrastive 预训练 PointGPT-tiny（**架构必须与现网一致**：128 groups, dim=384） |
| 4. 接入 | v38 calib 训练使用 `native_cross_pointgpt_ckpt=fleet_pointgpt_L20.pth` |

**关于 Pointnet2_PyTorch**：
- 仓库 `/mnt/drtraining/user/dahailu/code/Pointnet2_PyTorch` 为经典 PointNet++ 实现，**与 ProjFusion PointGPT（PointTransformer）架构不同**
- **不建议**直接用 PointNet2 替换 PointGPT encoder（需重写 `native_cross_attention` 接口）
- 可行路径：用 Pointnet2 做 **点云预处理/采样实验**，或作为 **辅助 ablation**；主路径仍应在 ProjFusion PointGPT 上 finetune

**前向稀疏点云的特殊处理（建议在 fleet PointGPT 预训练中加入）**：
- 训练时 **不做 360° rotation augmentation**
- 明确 `foreground` mask，与 BEVCalib 一致
- 可考虑 **sector-aware FPS**：优先在前方扇区采 group

### 6.4 【中】Warm-start 天花板

v37 从 v36 Ep191 启动，**cross-attn head 已收敛到 ±5° 邻域**。后续 ±10° 长训主要改善：
- Jacobian / 大角度自适应（已验证 ✓）
- 单帧精度（已验证 ✓）

但 **MEDW 依赖的序列级偏差模式** 可能在 v36 权重中已形成局部最优；大 batch + 少 step 不易跳出。

**验证方法**：v38 做 **scratch vs warm-start** 对照（同 step 预算）。

### 6.5 【中】MEDW 与单帧指标的优化目标不一致

| 指标 | 优化目标 |
|------|----------|
| Per-frame | 每帧独立扰动下恢复 GT |
| MEDW200 | 200 帧窗口 axis-angle median，**抑制零均值噪声** |

v37 提升单帧 + Jacobian，说明 **帧级自适应** 变强；MEDW 略差 0.04° 可能在误差范围内，也可能来自：
- 某些序列（如 seq00）MEDW 偏高拉均值
- ±10° 训练使单帧 correction 幅度变大，**序列级 median 对 outlier 更敏感**

**Ep400 完成后** 应复测；若 MEDW 仍差，优先查 **per-seq breakdown**（`deploy_simulation.json`）。

### 6.6 【中】架构仍非 ProjFusion 完整配方

| 差距项 | v37 状态 | ProjFusion |
|--------|----------|------------|
| extend_ratio | 2.5 ✓ | 2.5 |
| 扰动 | ±10° ✓ | ±10° |
| 分辨率 | 640×360 | 224×448 |
| cross-attn 层数 | 1 | 2–3+ |
| dual-branch | 伪 dual（concat） | 真独立 rot/tsl |
| PointGPT | KITTI 预训练 | 数据集匹配预训练 |
| 迭代推理 | 训练单步 | 训练即多步 |

分辨率 / 层数 / 真 dual-branch 未对齐，**MEDW 上限可能仍受架构 cap**。

### 6.7 【低】其他因素

- **评估代码 bug**：extend_ratio 遗漏已修复；首次 MEDW 用 ±10° 已废弃
- **显存未吃满**：BS=128 约 20GB/46GB；增大 batch 进一步减少 steps，**不推荐**；应优先 **降 BS 增 steps** 或 **增 max_frames**
- **数据增强**：v37 mount_jitter / pitch_flip 与 v36 相同，不是主因
- **test_data_v2 与 train 分布**：12 序列 hold-out，seq00 等长序列 MEDW 方差大

---

## 7. 效果总结矩阵

| 维度 | v37 vs v36 | 判定 |
|------|------------|------|
| Train val（各自扰动域） | 接近 | ✓ |
| 单帧泛化（±5°） | 0.91° vs 1.18° | **✓ 明显改进** |
| MEDW200（±5°） | 0.46° vs 0.41° | △ 略退化（Ep241 未完成训练） |
| Jacobian ±5° | 0.84 vs 0.55 | **✓ 显著改进** |
| Jacobian ±10° | 0.88 vs 0.29 | **✓ 显著改进** |
| PointGPT swap | nuScenes 不可用 | KITTI 仍最优 |

**一句话**：v37 在 **shortcut 克服 + 单帧精度** 上优先达成目标；**MEDW 部署指标** 受 step 密度、PointGPT 域差距、warm-start 局部最优共同制约，需 v38 结构性调整。

---

## 8. V38 实验建议（优先级排序）

### P0 — 不增加总训练时间的前提下提高 step 密度

```yaml
# configs/v38_native_cross_fleet_pointgpt.yaml（已更新，含 3 臂对照 + Fleet L20 PointGPT）
batch_size: 64          # global 512, steps/ep ~40–80（配合 max_frames=5000）
learning_rate: 2e-4     # sqrt/linear 缩放
max_frames_per_seq: 5000
num_epochs: 300         # 目标 total steps ~12k–15k
native_cross_extend_ratio: 2.5
angle_range_deg: 10
native_cross_pointgpt_ckpt: .../fleet_pointgpt_L20.pth
native_cross_pc_groups: 256
native_cross_pointgpt_max_depth: 60.0
# 三臂: fleet L20 scratch | fleet L20 warm-start v37 | kitti scratch
```

### P1 — 车队 PointGPT 预训练

1. 编写 `all_training_data` → ProjFusion PointGPT 数据导出脚本
2. `finetune_fleet_tiny.yaml` + 50 epoch 预训练
3. v38 calib 使用 fleet PointGPT ckpt（与 KITTI / nuScenes A/B 三臂对照，**均 scratch calib head**）

### P2 — 架构对齐（可选，工作量大）

- 224×448 训练分辨率 或 多尺度
- `native_cross_n_layers=2`
- 真 dual-branch rot/tsl

### P3 — v37 续训至 Ep400 后终评 ✅ 已完成

Ep400 完成后按 **§4 协议** 复跑全套。**完整结果见 `docs/V37_GENERALIZATION_REPORT.md`**。

要点：Ep350 val 最优 (0.90°)，但 MEDW200 Ep241 (0.457°) 优于 Ep350 (0.493°)；Jacobian 全 checkpoint ADAPTIVE。

---

## 9. 产物索引

| 产物 | 路径 |
|------|------|
| **泛化评估报告** | **`docs/V37_GENERALIZATION_REPORT.md`** |
| v39 设计（待确认） | `docs/V39_DESIGN.md` |
| v37 配置 | `configs/v37_native_cross_pointgpt_long.yaml` |
| v38 配置（暂停） | `configs/v38_native_cross_fleet_pointgpt.yaml` |
| fleet PointGPT L20 预训练 | `ProjFusion/pretrained/fleet_pointgpt_L20.pth` |
| 训练 log | `logs/all_training_data/model_small_10deg_v37_native_cross_pointgpt_long/train.log` |
| best val ckpt | `.../checkpoint/ckpt_350.pth` (Ep350, val 0.90°) |
| best MEDW ckpt | `.../checkpoint/ckpt_best_val.pth` (Ep241, MEDW200 0.457°) |
| MEDW Ep241/300/350 | `.../medw200_test_v2_eval_5deg{,_ep300,_ep350}/` |
| Jacobian Ep241/300/350 | `.../checkpoint/jacobian_{5,10}deg_ep*.json` |
| PointGPT A/B | `logs/all_training_data/pointgpt_ab_v36_test_v2/` |

---

## 10. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-05-26 | 初版：Ep241 泛化/Jacobian/A/B 结果 + 根因分析 + V38 建议 |
| 2026-05-26 | 修复 `evaluate_checkpoint.py` / `diagnose_jacobian.py` extend_ratio 传递 |
| 2026-05-27 | Ep400 完成；新增 `V37_GENERALIZATION_REPORT.md`；v38→v39 路线更新 |
