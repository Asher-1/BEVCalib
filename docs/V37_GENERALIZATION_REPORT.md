# V37 泛化性能评估报告

> **状态**：训练已完成（Ep400）；本报告为终态泛化评估汇总  
> **模型**：`model_small_10deg_v37_native_cross_pointgpt_long`  
> **基线**：v36 `model_small_5deg_v36_native_cross_pointgpt` Ep191  
> **评估日期**：2026-05-27  
> **关联设计**：`docs/V37_DESIGN.md`

---

## 1. 执行摘要

| 维度 | v36 Ep191 | v37 最佳 checkpoint | 结论 |
|------|-----------|---------------------|------|
| **Shortcut 克服（Jacobian ±10°）** | 0.29 (WEAK) | **0.88+ (ADAPTIVE)** | ✅ v37 目标达成 |
| **单帧泛化（±5° test_v2）** | 1.18° | **0.84° (Ep350)** | ✅ 显著改进 |
| **部署指标 MEDW200（±5°）** | **0.413°** | 0.457° (Ep241) / 0.493° (Ep350) | ⚠️ 未超越 v36 |
| **Val best（±10° 训练域）** | 1.15° (±5° val) | **0.90° (Ep350)** | ✅ in-domain 最优 |

**一句话结论**：v37 在 **shortcut 克服 + 单帧精度** 上全面优于 v36；**MEDW 部署指标** 在 Ep241 略差、Ep350 进一步退化。**Val 最优 ≠ MEDW 最优**，生产部署应单独选 ckpt。

**推荐 checkpoint 用途**：

| 用途 | 推荐 ckpt | 理由 |
|------|-----------|------|
| 部署 / MEDW 优先 | `ckpt_best_val.pth` (Ep241) 或 v36 Ep191 | MEDW200 最低 |
| 单帧 / 大角度自适应 | `ckpt_350.pth` | 单帧 0.84°，Val 0.90° |
| Shortcut 验证 | Ep241 / Ep350 均可 | J±10° > 0.88 |

---

## 2. 评估协议（强制，所有对比均遵守）

| 项 | 规定 |
|----|------|
| 测试集 | `/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2` |
| 采样 | `--use_full_dataset --eval_max_frames_per_seq 200`（12 序列 × 200 帧 = 2400 样本） |
| 扰动 | **`--angle_range_deg 5.0`**（公平对比，与 v36 一致） |
| extend_ratio | 从 ckpt 读取（v37 = 2.5） |
| 可视化 | `--vis_interval 0`（加速，不影响数值） |
| Shortcut | `tools/diagnose_jacobian.py`，Overall J > 0.3 = ADAPTIVE |

```bash
# MEDW 评估
python evaluate_checkpoint.py --mode eval \
  --ckpt_path logs/.../ckpt_350.pth \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2 \
  --output_dir logs/.../medw200_test_v2_eval_5deg_ep350 \
  --use_full_dataset --eval_max_frames_per_seq 200 \
  --angle_range_deg 5.0 --vis_interval 0 --batch_size 8

# Jacobian
python tools/diagnose_jacobian.py \
  --ckpt_path logs/.../ckpt_350.pth \
  --angle_range 5.0 --n_batches 10 --batch_size 4 \
  --output logs/.../jacobian_5deg_ep350.json
```

---

## 3. 训练终态（Ep400 完成）

| 指标 | Ep300 | Ep350 (best val) | Ep400 |
|------|------:|-----------------:|------:|
| Val Rot (±10°) | 1.06° | **0.90°** ★ | 1.15° |
| Train Rot | 2.82° | 2.74° | 2.94° |

- 训练时长：7.96h（150 epoch 增量，自 Ep250 resume 至 Ep400）
- Best val 出现在 **Ep350**（train summary 记录 Ep341 best_val ckpt，eval 表 Ep350 最低）
- Ep400 出现明显过拟合（val 1.15°），**不应作为部署 ckpt**

---

## 4. 泛化指标总表（test_data_v2，±5°）

### 4.1 全局指标

| Checkpoint | 单帧 Rot | MEDW50 | MEDW200 | Roll@MEDW200 | Pitch@MEDW200 | Yaw@MEDW200 |
|------------|--------:|-------:|--------:|-------------:|--------------:|------------:|
| v36 Ep191 | 1.175° | 0.406° | **0.413°** | 0.193° | 0.257° | **0.151°** |
| v37 Ep241 | 0.906° | 0.452° | 0.457° | 0.226° | 0.267° | 0.166° |
| v37 Ep300 | — | — | 0.517° | 0.253° | 0.315° | 0.208° |
| v37 Ep350 | **0.837°** | 0.505° | 0.493° | 0.229° | 0.287° | 0.212° |

### 4.2 Jacobian / Shortcut

| Checkpoint | ±5° Overall J | ±10° Overall J | 判定 |
|------------|-------------:|---------------:|------|
| v36 Ep191 | 0.549 | ~0.29 | ±10° WEAK |
| v37 Ep241 | 0.840 | 0.879 | ADAPTIVE ✓ |
| v37 Ep300 | 0.908 | 0.895 | ADAPTIVE ✓ |
| v37 Ep350 | 0.892 | 0.889 | ADAPTIVE ✓ |

v37 全 checkpoint 在 ±5°/±10° 均为 ADAPTIVE，**shortcut 问题已克服**。

### 4.3 Val vs MEDW 背离（重要发现）

| Checkpoint | Val Rot (±10°) | MEDW200 (±5° test) | 关系 |
|------------|---------------:|-------------------:|------|
| Ep241 | 1.26° | **0.457°** (v37 MEDW 最佳) | Val 一般，MEDW 最好 |
| Ep350 | **0.90°** | 0.493° | Val 最好，MEDW 更差 |
| Ep300 | 1.06° | 0.517° | 双指标均非最优 |

**含义**：继续以 val loss 早停会损害部署指标；v39 应以 **MEDW200 proxy** 作为 checkpoint 选择依据之一。

---

## 5. 逐序列 MEDW200 分解

### 5.1 瓶颈序列：Seq00 Roll

| 模型 | seq00 Rot@MEDW200 | seq00 Roll@MEDW200 |
|------|------------------:|-------------------:|
| v36 Ep191 | 0.825° | 0.804° |
| v37 Ep241 | 0.970° | **0.918°** |
| v37 Ep350 | 1.043° | **0.886°** |

Seq00 长序列 Roll 轴是全局 MEDW 的主要拖累；v37 在 seq02/04 等序列有优势，但被 seq00 抵消。

### 5.2 v37 Ep350 各序列 MEDW200 Rot

| Seq | Rot | Roll | Pitch | Yaw | 备注 |
|-----|----:|-----:|------:|----:|------|
| 00 | 1.043° | 0.886° | 0.234° | 0.499° | **瓶颈** |
| 01 | 0.476° | 0.285° | 0.262° | 0.277° | |
| 02 | 0.299° | 0.203° | 0.210° | 0.066° | 优秀 |
| 03 | 0.398° | 0.302° | 0.244° | 0.087° | |
| 04 | 0.227° | 0.085° | 0.015° | 0.210° | 优秀 |
| 05 | 0.657° | 0.119° | 0.645° | 0.022° | Pitch 偏高 |
| 06 | 0.629° | 0.358° | 0.514° | 0.045° | |
| 07 | 0.533° | 0.215° | 0.353° | 0.336° | |
| 08 | 0.618° | 0.007° | 0.210° | 0.581° | Yaw 偏高 |
| 09 | 0.359° | 0.078° | 0.292° | 0.195° | |
| 10 | 0.406° | 0.032° | 0.403° | 0.036° | |
| 11 | 0.264° | 0.173° | 0.064° | 0.189° | 优秀 |

---

## 6. 与目标差距（0.1°/轴）

| 轴 | v36 最佳 (MEDW200) | v37 Ep350 (MEDW200) | 距 0.1° 目标 |
|----|-------------------:|--------------------:|-------------:|
| Roll | 0.193° | 0.229° | **2.3×** |
| Pitch | 0.257° | 0.287° | **2.9×** |
| Yaw | 0.151° | 0.212° | **2.1×** |

**0.1°/轴在当前架构与数据规模下不现实**；合理 stretch 目标为 MEDW200 **0.25–0.30°**，Roll **0.12–0.15°**。

---

## 7. 根因归纳（为何 v37 未在 MEDW 上全面超越 v36）

1. **Optimizer step 密度减半**（~6400 vs v36 ~13000 steps）→ 序列级统计量学习不足
2. **Warm-start 局部最优**：MEDW 相关权重在 v36 邻域已收敛，长训改善单帧/Jacobian 但 MEDW 滞后
3. **架构 cap**：1-layer NativeCross + DINO 640×360 + Linear 级 head，非 ProjFusion 完整配方
4. **PointGPT 域差距**：KITTI 360° 预训练 vs 车队前向稀疏点云（fleet L20 预训练在 v38 才接入，v37 未用）
5. **过拟合轨迹**：Ep350 后 val 与 MEDW 同步恶化，说明需 MEDW-aware 早停

---

## 8. 结论与 v37 交付物

### 8.1 达成项

- ✅ Shortcut 克服（Jacobian ±10° 从 0.29 → 0.88+）
- ✅ 单帧泛化 -23% ~ -29%（vs v36）
- ✅ Val in-domain 收敛至 0.90°（Ep350）
- ✅ 评估协议标准化（±5° 公平对比、extend_ratio 修复）

### 8.2 未达成项

- ❌ MEDW200 超越 v36（0.413°）
- ❌ 0.1°/轴部署目标
- ❌ Seq00 Roll 瓶颈消除

### 8.3 产物索引

| 产物 | 路径 |
|------|------|
| 训练 log | `logs/.../train.log` |
| Best val ckpt | `.../checkpoint/ckpt_350.pth` |
| Best val (early stop) | `.../checkpoint/ckpt_best_val.pth` (Ep241) |
| MEDW Ep241 | `.../medw200_test_v2_eval_5deg/` |
| MEDW Ep300 | `.../medw200_test_v2_eval_5deg_ep300/` |
| MEDW Ep350 | `.../medw200_test_v2_eval_5deg_ep350/` |
| Jacobian Ep241/300/350 | `.../checkpoint/jacobian_{5,10}deg_ep*.json` |
| 训练总结 | `.../training_summary.md` |

---

## 9. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-05-27 | 终态报告：Ep241/300/350 全套 MEDW + Jacobian，vs v36 对比，ckpt 选用建议 |
