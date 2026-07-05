# V45 A/B/C 实验设计文档

**日期**: 2026-06-03
**基线**: V44-S1 (BEST 0.582°, per-frame 0.649°) — 首次超越 V20 baseline

---

## 一、设计动机

V43/V44 评估揭示了三个关键洞察：

1. **V44 架构 (DLA + PitchFusion) 是核心优势**：V44-S1 per-frame 0.649° vs V43 最好的 0.832°（-22%）
2. **InstanceNorm 不是 "破坏泛化"，而是改变了误差特征**：V44-S2/S3 per-frame 差（1.0°）但 bag 精度极好（0.05-0.07°）
3. **128 FPS groups 可能是空间分辨率瓶颈**：PointEncoder 输出固定 128 token，可能限制了精细校准

## 二、实验矩阵

| 变体 | InstanceNorm | cf_n_groups | batch_size | 核心假设 |
| --- | --- | --- | --- | --- |
| **V45a** | S1/S2/S3 全关 | 128 | 16 | 强增强+正则化可替代 IN 的域不变性 |
| **V45b** | S1 关, S2/S3 开 | 128 | 16 | IN 的低偏差高方差对部署有利 |
| **V45c** | S1 关, S2/S3 开 | **256** | 12 | 更高空间分辨率 + IN 组合 |

## 三、共同改进（相比 V44）

| 参数 | V44 值 | V45 值 | 依据 |
| --- | --- | --- | --- |
| `head_dropout` | 0.1 (hardcoded) | **0.15** | CrossAttn + CorrTransformer 正则化 |
| `axis_weights` | 1.0,4.0,1.0 | **1.5,4.0,1.5** | 提升 Roll/Yaw，Pitch 保持 4x |
| `pitch_aux_weight` | 0.3 | **0.4** | Pitch 是瓶颈 (0.386° vs Roll 0.275°) |
| `mount_jitter_prob` | 0.3 | **0.4** | 更强域增强 |
| `mount_jitter_rot_sigma` | 1.0 | **1.5** | 更大安装姿态变异 |
| `color_jitter` | 0.2 | **0.3** | 更强颜色/亮度变异 |
| `augment_intrinsic` | 0.02 | **0.03** | 更大焦距变异 |
| S1 `num_epochs` | 150 | **120** | V44-S1 Ep51 收敛，120 足够 |
| S1 `lr_schedule` | step(V44)/cosine(v44opt) | **step** | V44-S1 用 step 已验证 |
| S2 `continuous_noise_max_deg` | 8.0 | **5.0** | V43 数据支持 |
| S3 `continuous_noise_max_deg` | 5.0 | **2.0** | V43 数据支持 |
| S3 `axis_weights` | 2.0,2.0,2.0 | **1.5,3.0,1.5** | 避免过强轴分离 |

## 四、各阶段 Checkpoint 策略

全系使用 **`ckpt_best_dual.pth`** 作为 pretrain：

| 阶段 | 理由 |
| --- | --- |
| S1 pretrain | Dual-gate 保证 Jacobian>0.85 + MEDW<0.10°，地基需要精度+鲁棒双保 |
| S2 pretrain | 防止 shortcut 向下传播（V44 用 best_medw 时 S2 恢复率从 94%降到 90%） |
| S3 pretrain | 安全默认；如果 S2 best_dual vs best_medw 差距>0.01° 可做对照 |

## 五、V45c cf_n_groups=256 兼容性分析

### 权重兼容

`n_groups` 仅影响 FPS 采样数量（运行时 token 数），不影响 MLP 权重维度：
- PointEncoder.mlp: 输入 10 维（固定），输出 feat_dim 维（固定）
- Cross-attention: 处理可变长度 token 序列，权重不依赖 token 数
- **128-group checkpoint 的每个 key 都能被 256-group 模型完整加载**

### 计算量影响

| 组件 | 128 groups | 256 groups | 倍数 |
| --- | --- | --- | --- |
| FPS: `cdist(B,G,N)` | (16,128,16384) | (12,256,16384) | **1.5×** |
| kNN topk: `(B,G,N)` | (16,128,16384) | (12,256,16384) | **1.5×** |
| PointEncoder MLP: `(B,G,10)→(B,G,D)` | (16,128) | (12,256) | **1.5×** |
| Cross-Attention (self): `O(G²·D)` | 128²=16K | 256²=65K | **4×** |
| Cross-Attention (cross, img→pc): `O(G·H·W·D)` | 128·(H·W) | 256·(H·W) | **2×** |

### 训练速度预估

| 阶段 | 128 groups (bs=16) | 256 groups (bs=12) | 预估慢 |
| --- | --- | --- | --- |
| 数据加载 | 基准 | 不变 | 0% |
| Forward (PointEncoder) | ~3ms | ~4.5ms | 50% |
| Forward (Cross-Attn) | ~12ms | ~30ms | 150% |
| Forward (其他) | ~15ms | ~15ms | 0% |
| **总 step 时间** | ~30ms | ~50ms | **~60%** |

### 显存预估

| 组件 | 128 groups (bs=16) | 256 groups (bs=12) | 差异 |
| --- | --- | --- | --- |
| Point features | 16·128·256·4 = 2MB | 12·256·256·4 = 3MB | +1MB |
| FPS cdist | 16·128·16384·4 = 128MB | 12·256·16384·4 = 192MB | +64MB |
| Cross-attn QKV | ~400MB | ~600MB | +200MB |
| **总增量** | - | - | **~300-500MB** |

结论: batch_size=12 有充分安全余量。如果实测 OOM 可降至 10；如果显存充裕可试 14。

## 六、执行计划

### Quick 版（单节点 8 GPU）

```
Phase 1: V45a/b 共享 S1 → 预计 1 天
Phase 2: 分叉 → V45a S2 + V45b S2 → 预计各 0.5 天
Phase 3: V45a S3 + V45b S3 → 预计各 0.3 天
Phase 4: V45c S1 → S2 → S3 → 预计 2 天（独立训练）
```

总计: 约 4-5 天完成 quick 验证

### Full 版（16 节点 128 GPU）

Quick 版确认改进方向后再启动。

## 七、评估对比指标

训练完成后使用同一套评估 pipeline 对比：

| 评估类型 | 工具 | 核心指标 |
| --- | --- | --- |
| Test data 泛化 | `run_generalization_eval.py` | Per-frame Mean Rot, BEST Agg, P95 |
| Bag 零扰动 | `run_bag_calibration.py` | Residual, Conf%, total_std |
| Bag shortcut | `run_bag_calibration.py --inject 2,2,2` | Recover%, Shortcut判定 |
| Bag 小扰动 | `run_bag_calibration.py --inject 0.5,0.5,0.5` | Residual, Recover% |

### 对比基线

| 基线 | Per-frame | BEST Agg | Bag Baseline | Bag inject_small |
| --- | --- | --- | --- | --- |
| V44-S1 | 0.649° | 0.582° | 0.104° | 0.119° |
| V44-S2 | 1.001° | 0.787° | 0.078° | 0.050° |
| V44-S3 | 1.011° | 0.781° | 0.072° | 0.053° |

### 成功标准

- **V45a/b/c S1** 应达到: per-frame ≤ 0.60°, BEST ≤ 0.55°
- **V45a S3** 应达到: bag baseline ≤ 0.08°, inject_small ≤ 0.10°
- **V45b/c S3** 应达到: bag baseline ≤ 0.06°, inject_small ≤ 0.04°

## 八、Bug Fix 记录（2026-06-03）

在参数审查过程中发现 `cf_bev_r_calib.py`（V42+ CF-BEV-R 模型）存在严重的硬编码问题，导致 YAML 配置中的部分参数**从未生效**。

### 修复的硬编码问题

| # | 文件 | 硬编码内容 | 影响 | 修复方式 |
|---|------|-----------|------|---------|
| 1 | `cf_bev_r_calib.py` | `weight_axis_rotation=0.3` | YAML 设 0.5 实际用 0.3 | 从 `args` 读取 |
| 2 | `cf_bev_r_calib.py` | `axis_weights=(1.0,4.0,1.0)` | YAML 设 `1.5,4.0,1.5` 实际用 `1.0,4.0,1.0` | 从 `args` 读取 |
| 3 | `cf_bev_r_calib.py` | `use_geodesic_loss=False` | YAML 配置无效 | 从 `args` 读取 |
| 4 | `cf_bev_r_calib.py` | `weight_quat_norm` 未显式传递 | 仅使用构造函数默认值 0.5 | 从 `args` 读取 |
| 5 | `cf_bev_r_calib.py` | `CrossAttentionBlock dropout=0.1` | YAML 设 0.15 实际用 0.1 | 从 `args.head_dropout` 读取 |
| 6 | `cf_bev_r_calib.py` | `CorrTransformerHead dropout=0.1` | YAML 设 0.15 实际用 0.1 | 从 `args.head_dropout` 读取 |
| 7 | `native_cross_attention.py` | `ExtrinsicAwareCrossAttention` SDPA `dropout_p=0.1` | 构造函数传入的 dropout 在 attention 中被忽略 | 使用 `self.attn_dropout` |
| 8 | `cf_bev_r_calib.py` | `realworld_loss` 构造 `enable_axis_loss=True` | YAML 中设 false 无效，始终为 True | 从 `args.enable_axis_loss` 读取 |

### 新增的参数传递支持

| 文件 | 改动 |
|------|------|
| `train_kitti.py` | 新增 `--quat_norm_weight` argparse 参数 |
| `batch_train.sh` | 新增 `quat_norm_weight` → `--quat_norm_weight` 映射 |

### 对历史评估的影响

V43/V44 **实际训练时**使用的参数值（而非 YAML 中配置的值）：
- `axis_weights` = (1.0, 4.0, 1.0)，不是 YAML 中的自定义值
- `weight_axis_rotation` = 0.3，不是 YAML 中的自定义值
- `head_dropout` = 0.1，不是任何自定义值

**V45 是首批真正使用 YAML 配置参数的实验。**

### `drop_path_rate` 说明

`drop_path_rate` 仅在 `bev_calib.py` 的 `deformable_transformer_layer` 中使用。
CF-BEV-R (V42+) 使用标准 Transformer，不含 deformable 层。
因此 V45 YAML 中的 `drop_path_rate: 0.15` **不会产生任何效果**，已从所有 V45 配置中移除。

### 向后兼容性

所有修复完全向后兼容：新参数默认值等同于之前硬编码值，旧配置不受影响。

## 九、配置文件索引

| 文件 | 变体 | 节点 |
| --- | --- | --- |
| `configs/v45a_cf_bev_r_quick.yaml` | V45a (无 IN) | 1 |
| `configs/v45a_cf_bev_r_full.yaml` | V45a (无 IN) | 16 |
| `configs/v45b_cf_bev_r_quick.yaml` | V45b (S2/S3 IN) | 1 |
| `configs/v45b_cf_bev_r_full.yaml` | V45b (S2/S3 IN) | 16 |
| `configs/v45c_cf_bev_r_quick.yaml` | V45c (S2/S3 IN, 256g) | 1 |
| `configs/v45c_cf_bev_r_full.yaml` | V45c (S2/S3 IN, 256g) | 16 |
