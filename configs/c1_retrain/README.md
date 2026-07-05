# C1 重训配置 (camera_1 数据集)

## 背景

之前使用 `all_training_data` 训练，该数据集使用**长焦相机**(FOV仅15.2°)与LiDAR组合：
- 极少点云能投影到图像中 (~10%利用率)
- Roll角缺乏足够观测（窄FOV导致roll变化几乎不改变像素位置）
- 大量点云被浪费

现在使用 `all_training_data_c1`，采用 **camera_1** (fisheye, FOV=97.7°)：
- 点云利用率提升到 ~85%
- Roll/Pitch/Yaw三轴都有充分观测
- 23个序列, 184,274帧, 340GB

## BEV网格兼容性分析

| 参数 | 当前配置 | camera_1覆盖 | 结论 |
|------|---------|-------------|------|
| X方向 | [0, 200]m | 前方200m内均可覆盖 | 完全兼容 |
| Y方向 | [-100, 100]m | 200m处可视±229m | 远超覆盖 |
| Z方向 | [-10, 10]m | 100m处可视±64m | 完全兼容 |

**结论**: 当前BEV网格配置完全适配camera_1数据，无需修改。

## 训练目标

1. Recovery > 95% (3-pass迭代推理)
2. 泛化RPY误差 < 0.1°
3. 固定平移，只优化旋转 (`rotation_only=true, trans_range=0.0`)

## 配置文件说明

### 优先训练 (Baseline)

| 配置 | 原型 | 特点 | 优先级 |
|------|------|------|--------|
| `c1_baseline_v45c_cf_bev_r.yaml` | v45c | 256 groups, 无GIN, 官方baseline | P0 |
| `c1_v48a_multiscale_cf_bev_r.yaml` | v48a | Multi-Scale + 无consistency | P0 |

### 备选训练 (泛化对照)

| 配置 | 原型 | 特点 | 优先级 |
|------|------|------|--------|
| `c1_v45a_no_in_cf_bev_r.yaml` | v45a | 128 groups, 泛化综合最优 | P1 |
| `c1_v50a_optuna_no_cons_cf_bev_r.yaml` | v50a | Optuna超参+无consistency | P1 |
| `c1_v44_high_recovery_cf_bev_r.yaml` | v44 | Recovery最强(82.1%) | P1 |

## 训练命令

```bash
# P0: 先跑baseline (作为参照)
bash batch_train.sh configs/c1_retrain/c1_baseline_v45c_cf_bev_r.yaml

# P0: 同时跑v48a (预期最优)
bash batch_train.sh configs/c1_retrain/c1_v48a_multiscale_cf_bev_r.yaml

# P1: 备选对照实验
bash batch_train.sh configs/c1_retrain/c1_v45a_no_in_cf_bev_r.yaml
bash batch_train.sh configs/c1_retrain/c1_v50a_optuna_no_cons_cf_bev_r.yaml
bash batch_train.sh configs/c1_retrain/c1_v44_high_recovery_cf_bev_r.yaml
```

## 版本配置关键差异

| 维度 | v44 | v45a | v45c(baseline) | v48a | v50a |
|------|-----|------|----------------|------|------|
| cf_n_groups | 128 | 128 | 256 | 256 | 128 |
| GIN | 无 | 无 | 无 | 无 | 无 |
| consistency_loss | 0.5 | 0.5 | 0.5 | **0.0** | **0.0** |
| multi_scale_perturb | 无 | 无 | 无 | **有** | **有** |
| lr_schedule | step | step | cosine | cosine | cosine |
| overcorrection | 无 | 无 | 无 | 2.0 | 1.5 |
| pitch_vertical_bands | - | - | - | 3 | 5 |

## 预期效果

camera_1 宽FOV数据的优势：
1. **Roll观测增强**: 从FOV=15°到98°，roll变化产生的像素位移增大~6x
2. **点云密度提升**: 投影到图像的点从~10%增至~85%
3. **更丰富的几何约束**: 覆盖更大空间范围的点对应关系

预期改善：
- Recovery: 从旧数据65-82% → 新数据 80-95%+ (单pass)
- 3-pass Recovery: 从91% → 97%+ (因为单pass更高)
- ZD: 从0.25-0.45° → <0.15° (更多观测约束)
- MED400 RPY: 从0.1-0.2° → <0.1° (目标)
