# BEVCalib 部署指南

## 1. 最佳模型选择

### 推荐部署模型

| 优先级 | 模型 | MEDW800 | 训练配置 | Checkpoint |
|---|---|---|---|---|
| **首选** | V30opt-G3-small | **0.063°** | 1节点/500帧/dinov2-small | `ckpt_best_val.pth` |
| 备选 | V30-G3-dann-ckpt400 | 0.067° | 16节点/10K帧/dinov2-small+DANN | `ckpt_400.pth` |
| 备选 | V30-G3-small | 0.069° | 16节点/10K帧/dinov2-small | `ckpt_best_val.pth` |
| 参考 | V29-G3 | 0.092° | 1节点/500帧/dinov2-small | `ckpt_best_val.pth` |

### 模型文件路径

```
# 首选 (V30opt-G3-small)
logs/all_training_data/v30_opt_quick/model_small_5deg_v30_opt_G3_dinov2small_quick/
  all_training_data_scratch/checkpoint/ckpt_best_val.pth

# 备选 (V30-G3-dann-ckpt400)
logs/all_training_data/v30/model_small_5deg_v30_G3_dinov2small_dann/
  all_training_data_scratch/checkpoint/ckpt_400.pth
```

## 2. 推理配置

### 核心参数

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `backbone_type` | `dinov2` | 自动从 checkpoint 检测 |
| `backbone_variant` | `dinov2-small` | 自动从 checkpoint 检测 |
| `img_shape` | `(360, 640)` | 输入图像尺寸 (H, W) |
| `rotation_only` | `true` | 仅校准旋转 |
| `BEV_ZBOUND_STEP` | `4.0` | BEV 网格 Z 轴步长 |

### 时序聚合参数

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `agg_method` | `axis_angle_median` | MEDW 方法 (优于 SVD-Mean) |
| `max_frames` | `800` | 最少需要 800 帧才能达到 sub-0.1° |
| `angle_range_deg` | `5.0` | 扰动范围 (评估/部署一致) |
| `eval_seed` | `42` | 固定种子确保可复现 |

### 帧数与精度关系

| 帧数 | V30opt-G3 MEDW | 达标状态 |
|---|---|---|
| 50 帧 | ~0.308° | 未达标 |
| 200 帧 | ~0.138° | 未达标 |
| 400 帧 | 0.096° | 达标 (<0.1°) |
| 800 帧 | **0.063°** | 达标 |
| 1600 帧 | ~0.045° | 达标 (需要更多数据) |

## 3. 使用方式

### 方式一: Python API (推荐)

```python
from utils.bevcalib_inference import infer_sequence

results = infer_sequence(
    ckpt_path="path/to/ckpt_best_val.pth",
    data_dir="/path/to/kitti_format_data",
    max_frames=800,
    agg_method='axis_angle_median',
    batch_size=8,
)

for r in results:
    print(f"Seq {r['seq_id']}: Rot={r['rot_error']:.4f}°")
    print(f"  Calibration T:\n{r['agg_T']}")
```

### 方式二: CLI

```bash
cd code/BEVCalib/utils

python drinfer_infer.py \
  --config drinfer_config.yaml \
  --mode sequence \
  --data-dir /path/to/test_data \
  --max-frames 800 \
  --agg-method axis_angle_median
```

### 方式三: 单帧推理 + 外部聚合

```python
from utils.bevcalib_inference import load_bevcalib_inference, TemporalCalibrationAggregator

wrapper, epoch = load_bevcalib_inference("path/to/ckpt.pth")
agg = TemporalCalibrationAggregator(max_frames=800, method='axis_angle_median')

for frame_data in stream:
    pred_T = wrapper(img, pc, init_T, post_T, K)
    agg.add(pred_T.cpu().numpy())

calibration = agg.aggregate()
confidence = agg.get_confidence()
```

## 4. 训练建议

### 已验证的最优训练配方 (G3)

| 参数 | 值 | 说明 |
|---|---|---|
| `backbone_type` | `dinov2` | DINOv2 预训练特征 |
| `backbone_variant` | `dinov2-small` | 22.1M 参数, 性价比最高 |
| `freeze_backbone` | 1 | 冻结大部分层 |
| `backbone_freeze_layers` | `0:-2` | 仅放开最后 2 层 |
| `cam2bev_mode` | `query` | Query-based BEV 投影 |
| `augment_mount_jitter_prob` | 0.7 | 强 mount jitter |
| `augment_mount_jitter_rot_sigma` | 3.0 | 3° 旋转抖动 |
| `angle_range_deg` | 5 | 5° 扰动范围 |
| `rotation_only` | true | 仅旋转校准 |
| `perturb_distribution` | `truncated_normal` | 截断正态分布 |
| `per_axis_prob` | 0.3 | 单轴扰动概率 |

### 训练规模与泛化

| 训练规模 | 最佳 MEDW800 | 建议 |
|---|---|---|
| 1 节点 (8 GPU), 500帧/seq | **0.063°** | **泛化最优** |
| 16 节点 (128 GPU), 10K帧/seq | 0.069° | 训练域过拟合 |
| 16 节点, ddp_auto_scale=0 | 0.082° | LR 不匹配 batch 大小 |

**结论**: 单节点小规模训练产生的模型泛化性能最好。多节点训练的价值在于缩短训练时间，但不提升跨域泛化精度。

### 不推荐的配置

| 配置 | 原因 |
|---|---|
| `dinov2-base` | 模型大 3.9×, 泛化仅提升 5.4%, 过拟合严重 |
| `DANN` | 小规模训练有害, 大规模仅微弱正面 |
| `angle_range_deg: 10` | 10° 训练 5° 评估效果差 (0.226° vs 0.063°) |
| `ddp_auto_scale: 0` (多节点) | LR/batch 比例失调 |

## 5. 评估验证

### 运行评估

```bash
python run_generalization_eval.py \
  --config configs/eval_generalization_v30_opt_quick.yaml \
  --parallel -1
```

### 关键指标解读

| 指标 | 含义 | 目标 |
|---|---|---|
| Per-frame Rot | 单帧旋转误差 (geodesic) | ~2.3° (正常范围) |
| MEDW400 | 400 帧中值聚合误差 | <0.1° |
| MEDW800 | 800 帧中值聚合误差 | <0.07° |
| 泛化衰退 | 测试/训练误差比 | <1.2× |

### 达标标准

- **Sub-0.1°**: MEDW400 ≤ 0.1° (400 帧即可达标)
- **最优精度**: MEDW800 ≈ 0.063° (推荐部署配置)
- **各轴均衡**: Roll/Pitch/Yaw 各 <0.04°
