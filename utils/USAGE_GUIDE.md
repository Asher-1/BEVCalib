# BEVCalib 推理工具使用指南

本文档详细说明 `bevcalib_inference.py` (Python API) 和 `drinfer_infer.py` (CLI 工具) 的完整使用方法。

---

## 目录

1. [快速开始](#1-快速开始)
2. [bevcalib_inference.py — Python API](#2-bevcalib_inferencepy--python-api)
3. [drinfer_infer.py — CLI 工具](#3-drinfer_inferpy--cli-工具)
4. [配置文件参考](#4-配置文件参考)
5. [场景示例](#5-场景示例)
6. [FAQ](#6-faq)

---

## 1. 快速开始

### 30 秒上手: 对一组序列数据运行标定

```bash
# 方式一: 纯命令行传参
cd code/BEVCalib/utils
python drinfer_infer.py \
  --config ../configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data \
  --max-frames 800

# 方式二: 所有参数写在配置文件里
python drinfer_infer.py --config my_config.yaml --mode sequence
```

### 30 秒上手: Python 代码中调用

```python
from bevcalib_inference import infer_sequence

results = infer_sequence(
    ckpt_path="path/to/ckpt_best_val.pth",
    data_dir="/path/to/test_data",
    max_frames=800,
)
for r in results:
    print(f"Seq {r['seq_id']}: Rot={r['rot_error']:.4f}°")
    print(f"标定外参 T:\n{r['agg_T']}")
```

---

## 2. bevcalib_inference.py — Python API

### 2.1 核心类: `BEVCalibInference`

推理-only 封装器，去掉 loss 计算，只输出预测的 LiDAR→Camera 变换矩阵。

```python
from bevcalib_inference import load_bevcalib_inference

# 加载模型 (大部分参数自动从 checkpoint 检测)
wrapper, epoch = load_bevcalib_inference("path/to/ckpt.pth")

# 单帧推理
pred_T = wrapper(img, pc, init_T, post_T, K)
# img:     (B, 3, H, W) float tensor — RGB 图像
# pc:      (B, N, 3) float tensor   — 点云 XYZ
# init_T:  (B, 4, 4) float tensor   — 初始 LiDAR→Camera 外参 (含扰动)
# post_T:  (B, 4, 4) float tensor   — 后处理矩阵 (推理时传 identity)
# K:       (B, 3, 3) float tensor   — 相机内参
# 返回:    (B, 4, 4) float tensor   — 预测的 LiDAR→Camera 外参
```

#### `load_bevcalib_inference` 参数表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ckpt_path` | str | 必填 | checkpoint 文件路径 |
| `device` | str | `"cuda"` | `"cuda"` 或 `"cpu"` |
| `img_shape` | tuple | `(360, 640)` | 输入图像尺寸 (H, W) |
| `rotation_only` | bool/None | `None` | `None` 自动检测, `True` 仅旋转, `False` 旋转+平移 |
| `backbone_type` | str/None | `None` | `None` 自动检测, `"swin"` 或 `"dinov2"` |
| `backbone_variant` | str/None | `None` | `None` 自动检测, 如 `"dinov2-small"` |
| `deformable` | bool | `False` | 是否使用 deformable attention |
| `bev_encoder` | bool | `True` | 是否使用 BEV encoder |
| `use_mlp_head` | bool/None | `None` | `None` 自动检测 MLP/Linear 回归头 |
| `voxel_mode` | str | `"scatter"` | `"hard"` 或 `"scatter"` |
| `to_bev_mode` | str | `"concat"` | `"concat"`, `"learned"`, 或 `"sum"` |
| `scatter_reduce` | str | `"sum"` | `"sum"` 或 `"mean"` |
| `bev_pool_factor` | int | `0` | BEV pooling 因子 |
| `max_attn_tokens` | int | `0` | 最大 attention token 数 (0=全部) |

> **自动检测**: `backbone_type`, `backbone_variant`, `rotation_only`, `use_mlp_head` 都能从 checkpoint 自动识别。大部分场景只需传 `ckpt_path`。

---

### 2.2 核心类: `TemporalCalibrationAggregator`

生产级时序聚合器。将多帧预测聚合为一个稳健的标定结果。

#### 流式模式 (逐帧累积)

```python
from bevcalib_inference import load_bevcalib_inference, TemporalCalibrationAggregator

wrapper, epoch = load_bevcalib_inference("ckpt.pth")
agg = TemporalCalibrationAggregator(min_frames=50, max_frames=800)

for img, pc, init_T, post_T, K in data_stream:
    pred_T = wrapper(img, pc, init_T, post_T, K)
    agg.add(pred_T)

    if agg.ready:   # count >= min_frames
        calibration = agg.aggregate()          # (4,4) numpy
        confidence = agg.get_confidence()      # {'roll_std': ..., 'pitch_std': ..., ...}
        print(f"帧数: {agg.count}, 置信度: {confidence['total_std']:.3f}°")

agg.reset()  # 下一个序列前重置
```

#### 批量模式 (一次传入所有帧)

```python
from bevcalib_inference import TemporalCalibrationAggregator

pred_Ts = [model(img_i, ...).cpu().numpy() for img_i in sequence]
calibration = TemporalCalibrationAggregator.aggregate_batch(pred_Ts)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `min_frames` | int | `50` | 最少帧数 (不足时 `ready=False`) |
| `max_frames` | int | `1600` | 最大帧数 (超出则淘汰最早帧) |
| `method` | str | `"axis_angle_median"` | `"axis_angle_median"` (推荐) 或 `"svd_mean"` |

#### 帧数 vs 精度参考

| 帧数 | MEDW 精度 | 说明 |
|---|---|---|
| 50 | ~0.3° | 粗略估计 |
| 200 | ~0.14° | 中等精度 |
| 400 | ~0.096° | 达标 (<0.1°) |
| 800 | ~0.063° | 推荐部署配置 |
| 1600 | ~0.045° | 最高精度 (收益递减) |

---

### 2.3 核心函数: `infer_sequence`

端到端序列推理: 加载模型 → 加载数据 → 生成扰动 → 逐帧推理 → MEDW 聚合 → 计算误差。流程与 `evaluate_checkpoint.py` 一致，保证结果可复现。

```python
from bevcalib_inference import infer_sequence

results = infer_sequence(
    ckpt_path="path/to/ckpt_best_val.pth",
    data_dir="/path/to/kitti_format_data",
    seq_ids=["00", "01", "02"],     # 指定序列, None=全部
    max_frames=800,                 # 每序列最大帧数
    agg_method="axis_angle_median", # MEDW 聚合方法
    angle_range_deg=5.0,            # 扰动范围 (度)
    eval_seed=42,                   # 固定种子, 保证可复现
    batch_size=8,                   # 推理 batch_size
    device="cuda",
    img_shape=(360, 640),           # 图像尺寸 (H, W)
)

# 返回值: list of dict
for r in results:
    print(f"序列 {r['seq_id']}:")
    print(f"  帧数:     {r['n_frames']}")
    print(f"  旋转误差: {r['rot_error']:.4f}° (Roll={r['roll_error']:.3f}° "
          f"Pitch={r['pitch_error']:.3f}° Yaw={r['yaw_error']:.3f}°)")
    print(f"  标定外参:\n{r['agg_T']}")
    print(f"  置信度:   {r['confidence']}")
```

#### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ckpt_path` | str | 必填 | checkpoint 路径 |
| `data_dir` | str | 必填 | KITTI 格式数据根目录 |
| `seq_ids` | list/None | `None` | 序列 ID 列表, `None` 表示全部 |
| `max_frames` | int | `800` | 每序列最大帧数 |
| `agg_method` | str | `"axis_angle_median"` | 聚合方法 |
| `angle_range_deg` | float | `5.0` | 扰动角度范围 |
| `eval_seed` | int | `42` | 随机种子 |
| `batch_size` | int | `8` | 推理批大小 |
| `device` | str | `"cuda"` | 设备 |
| `rotation_only` | bool/None | `None` | 自动检测 |
| `img_shape` | tuple | `(360, 640)` | 图像尺寸 (H, W) |
| `verbose` | bool | `True` | 打印进度 |

#### 返回值结构

```python
[
    {
        'seq_id':       '00',                     # 序列 ID
        'agg_T':        np.ndarray (4,4),          # 聚合后的 LiDAR→Camera 外参
        'gt_T':         np.ndarray (4,4),          # Ground Truth 外参
        'n_frames':     800,                       # 使用的帧数
        'rot_error':    0.055,                     # 总旋转误差 (度, geodesic)
        'roll_error':   0.037,                     # Roll 误差 (度)
        'pitch_error':  0.039,                     # Pitch 误差 (度)
        'yaw_error':    0.030,                     # Yaw 误差 (度)
        'confidence':   {                          # 置信度 (越小越好)
            'roll_std': 0.98, 'pitch_std': 1.02,
            'yaw_std': 0.95, 'total_std': 1.68,
            'n_frames': 800
        },
    },
    ...
]
```

---

### 2.4 便捷函数: `load_bevcalib_with_aggregation`

一行代码同时加载模型和聚合器:

```python
from bevcalib_inference import load_bevcalib_with_aggregation

model, agg, epoch = load_bevcalib_with_aggregation(
    "ckpt_best_val.pth",
    min_frames=50,
    max_frames=800,
)

for img, pc, init_T, post_T, K in sequence:
    pred_T = model(img, pc, init_T, post_T, K)
    agg.add(pred_T)

calibration = agg.aggregate()
confidence = agg.get_confidence()
agg.reset()
```

---

## 3. drinfer_infer.py — CLI 工具

支持 4 种运行模式，所有参数均可通过**命令行传参**或**配置文件**指定，命令行参数优先级更高。

### 3.1 命令行参数总表

```
python drinfer_infer.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | str | `utils/drinfer_config.yaml` | YAML 配置文件路径 |
| `--mode` | str | `eval` | 运行模式: `eval` / `compare` / `temporal` / `sequence` |
| `--backend` | str | 配置文件决定 | 推理后端: `pytorch` / `drinfer` |
| `--output` | str | `None` | JSON 报告输出路径 |
| `--vis-dir` | str | `None` | 可视化图片保存目录 |
| `--agg-method` | str | `axis_angle_median` | 聚合方法: `axis_angle_median` / `svd_mean` |
| `--data-dir` | str | `None` | 数据集根目录 (sequence 模式, 覆盖配置文件) |
| `--seq-ids` | str | `None` | 逗号分隔的序列 ID (sequence 模式) |
| `--max-frames` | int | `None` | 每序列最大帧数 (sequence 模式) |

---

### 3.2 模式一: `eval` — 单后端精度+性能评估

评估单个推理后端的精度和延迟。

```bash
# 使用 PyTorch 后端评估
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode eval \
  --backend pytorch \
  --output results/eval_pytorch.json

# 使用 DrInfer 后端评估
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode eval \
  --backend drinfer \
  --output results/eval_drinfer.json

# 同时保存投影可视化
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode eval \
  --vis-dir results/vis/
```

**输出示例:**

```
======================================================================
EVALUATION REPORT [pytorch]  (500 samples, 50 batches)
======================================================================
Metric               Mean     Median        P95        Max
--------------------------------------------------------------
rot_error            2.3160   2.2069     4.6600     5.0050  deg
roll_error           1.0403   0.8891     2.7631     3.9780  deg
pitch_error          1.0474   0.8752     2.8036     4.2651  deg
yaw_error            1.0463   0.8755     2.7580     3.8850  deg
```

---

### 3.3 模式二: `compare` — PyTorch vs DrInfer 对比

同一数据集上对比两种后端的精度差异、延迟和显存占用。

```bash
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode compare \
  --output results/compare_report.json
```

**输出示例:**

```
================================================================================
COMPARISON REPORT: PyTorch (BEVCalibInference) vs DrInfer
================================================================================
Metric               PyTorch    DrInfer      Delta  RelDiff%
--------------------------------------------------------------
rot_error            2.3160     2.3158    -0.0002      0.01%  deg
```

---

### 3.4 模式三: `temporal` — 时序聚合评估 (验证集)

在验证集上运行时序聚合评估。按 GT 外参变化自动分割序列。

```bash
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode temporal \
  --agg-method axis_angle_median \
  --output results/temporal_report.json
```

---

### 3.5 模式四: `sequence` — 序列级 MEDW 推理 (生产部署)

对指定数据目录运行端到端序列推理，流程与 `evaluate_checkpoint.py` 一致。**这是生产部署推荐模式。**

```bash
# 评估全部序列, 每序列最多 800 帧
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data_v2 \
  --max-frames 800 \
  --output results/sequence_report.json

# 只评估特定序列
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data_v2 \
  --seq-ids "00,01,05,08" \
  --max-frames 800

# 使用 SVD-Mean 聚合方法 (通常 MEDW 更好)
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data_v2 \
  --agg-method svd_mean
```

**输出示例:**

```
======================================================================
SEQUENCE INFERENCE REPORT  (12 sequences, MEDW800)
======================================================================
   Seq  Frames      Rot     Roll    Pitch      Yaw
--------------------------------------------------
    00     800   0.0550°  0.0070°  0.0330°  0.0430°
    01     800   0.0810°  0.0410°  0.0200°  0.0670°
  ...
--------------------------------------------------
  Mean          0.0630°  0.0370°  0.0390°  0.0300°
   Std          0.0250°  0.0200°  0.0220°  0.0150°
======================================================================
```

---

## 4. 配置文件参考

### 4.1 完整配置文件模板

下面是一个完整的 YAML 配置文件，包含所有可用参数及说明:

```yaml
# ================================================================
# BEVCalib 推理配置 — 支持 eval / compare / temporal / sequence
# ================================================================

# -- 模型 checkpoint (必填) -------------------------------------------
ckpt_path: "logs/all_training_data/v30_opt_quick/model_.../ckpt_best_val.pth"

# -- 数据集 (eval/compare/temporal 模式必填) -------------------------
dataset_root: "/path/to/training_data"
validate_sample_ratio: 0.2        # 验证集比例

# -- 数据集 (sequence 模式) ------------------------------------------
# 命令行 --data-dir 优先; 也可在此指定:
# dataset_root: "/path/to/test_data_v2"  # sequence 模式使用

# -- 模型架构 (大部分可自动检测) ------------------------------------
deformable: false
bev_encoder: true
rotation_only: true               # null 或不写 = 自动检测
# backbone_type: dinov2            # 自动检测
# backbone_variant: dinov2-small   # 自动检测
# use_mlp_head: null               # 自动检测

# -- 输入维度 -------------------------------------------------------
img_height: 360
img_width: 640
max_num_points: 200000

# -- BEV 配置 -------------------------------------------------------
bev_zbound_step: "4.0"            # 必须与训练一致
voxel_mode: "hard"                # "hard" (训练) 或 "scatter" (drinfer)
to_bev_mode: "concat"             # 必须与训练一致
scatter_reduce: "sum"
bev_pool_factor: 0
max_attn_tokens: 0                # 0=全部 token, >0=packing 加速

# -- 评估参数 -------------------------------------------------------
angle_range_deg: 5.0              # 扰动角度范围
trans_range: 0.15                 # 扰动平移范围
batch_size: 8                     # 推理 batch size
num_workers: 4
max_batches: 0                    # 0=评估全部, >0=限制 batch 数
eval_seed: 42                     # 固定种子

# -- 时序聚合 (temporal/sequence 模式) -------------------------------
max_frames: 800                   # 每序列最大帧数

# -- DrInfer 导出/推理 -----------------------------------------------
export_dir: "logs/.../drinfer"
model_name: "bevcalib_fusion_head"
model_version: "v2"
inference_backend: "pytorch"      # "pytorch" 或 "drinfer"

# -- 通用 ------------------------------------------------------------
device: "cuda"
report_output: "results/report.json"
```

### 4.2 最小配置文件 (sequence 模式)

```yaml
ckpt_path: "path/to/ckpt_best_val.pth"
dataset_root: "/path/to/test_data"
bev_zbound_step: "4.0"
```

只需 3 行即可运行:

```bash
python drinfer_infer.py --config minimal.yaml --mode sequence
```

### 4.3 命令行参数 vs 配置文件优先级

| 参数 | 命令行 | 配置文件 | 优先级 |
|---|---|---|---|
| 数据目录 | `--data-dir` | `dataset_root` | 命令行 > 配置文件 |
| 序列 ID | `--seq-ids` | 不支持 | 仅命令行 |
| 最大帧数 | `--max-frames` | `max_frames` | 命令行 > 配置文件 |
| 聚合方法 | `--agg-method` | 不支持 | 仅命令行 |
| 推理后端 | `--backend` | `inference_backend` | 命令行 > 配置文件 |
| 报告路径 | `--output` | `report_output` | 命令行 > 配置文件 |
| 模式 | `--mode` | 不支持 | 仅命令行 |

---

## 5. 场景示例

### 场景 A: 生产部署 — 对新车辆数据运行标定

```bash
# 数据准备: KITTI 格式目录结构
# /data/vehicle_001/
#   sequences/
#     00/
#       image_2/   ← PNG 图片
#       velodyne/  ← 点云 .bin
#       calib.txt  ← 相机内参+外参

python drinfer_infer.py \
  --config configs/deploy_config.yaml \
  --mode sequence \
  --data-dir /data/vehicle_001 \
  --max-frames 800 \
  --output /data/vehicle_001/calibration_report.json
```

### 场景 B: Python 集成 — 嵌入到标定 Pipeline

```python
import os
os.environ["BEV_ZBOUND_STEP"] = "4.0"

from bevcalib_inference import load_bevcalib_with_aggregation
import torch
import numpy as np

model, agg, epoch = load_bevcalib_with_aggregation(
    "ckpt_best_val.pth",
    max_frames=800,
    device="cuda",
)

def calibrate_sequence(image_list, pc_list, init_T, intrinsic):
    """对一个序列运行标定, 返回聚合后的外参."""
    agg.reset()
    post_T = torch.eye(4).unsqueeze(0).cuda()

    for img_np, pc_np in zip(image_list, pc_list):
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).cuda()
        pc_t = torch.from_numpy(pc_np[:, :3]).float().unsqueeze(0).cuda()
        init_t = torch.from_numpy(init_T).float().unsqueeze(0).cuda()
        K_t = torch.from_numpy(intrinsic).float().unsqueeze(0).cuda()

        with torch.no_grad():
            pred_T = model(img_t, pc_t, init_t, post_T, K_t)
        agg.add(pred_T)

    if agg.ready:
        return agg.aggregate(), agg.get_confidence()
    return None, None
```

### 场景 C: 模型性能对比 — 对比两个 checkpoint

```bash
# 评估 checkpoint A
python drinfer_infer.py \
  --config config_A.yaml --mode sequence \
  --data-dir /data/test \
  --output results/ckpt_A.json

# 评估 checkpoint B
python drinfer_infer.py \
  --config config_B.yaml --mode sequence \
  --data-dir /data/test \
  --output results/ckpt_B.json

# 结果在 JSON 中, 对比 summary.rot_mean 即可
```

### 场景 D: PyTorch vs DrInfer 部署验证

```bash
python drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode compare \
  --output results/pt_vs_dr.json
```

### 场景 E: 快速验证 — 只跑几个 batch

```yaml
# quick_eval.yaml
ckpt_path: "path/to/ckpt.pth"
dataset_root: "/path/to/data"
bev_zbound_step: "4.0"
max_batches: 10           # 只跑 10 个 batch
batch_size: 4
```

```bash
python drinfer_infer.py --config quick_eval.yaml --mode eval
```

### 场景 F: 只标定特定序列

```bash
python drinfer_infer.py \
  --config my_config.yaml \
  --mode sequence \
  --data-dir /data/test \
  --seq-ids "00,05,08" \
  --max-frames 400
```

---

## 6. FAQ

### Q: 环境变量 `BEV_ZBOUND_STEP` 怎么设？

在导入 `bevcalib_inference` 之前设置:

```python
import os
os.environ["BEV_ZBOUND_STEP"] = "4.0"
```

CLI 工具从配置文件的 `bev_zbound_step` 字段自动设置。**这个值必须与训练时一致**, 否则模型输出会有偏差。常见取值: `"4.0"` (V29/V30), `"2.0"` (旧版本)。

### Q: 需要安装哪些依赖？

```
torch >= 2.0
numpy, scipy, opencv-python, pyyaml
transformers (Swin backbone)
```

DINOv2 backbone 需要在 `ckpt/checkpoints/` 目录放置预训练权重。

### Q: 数据集目录格式要求？

KITTI-Odometry 格式:

```
data_root/
  sequences/
    00/
      image_2/        *.png (RGB 图像)
      velodyne/        *.bin (点云, float32, N×4)
      calib.txt        标定参数
    01/
      ...
```

### Q: 如何判断标定结果好不好？

查看 `rot_error` (总旋转误差, geodesic 距离):

| 精度等级 | rot_error | 评价 |
|---|---|---|
| < 0.05° | 极好 | 超出预期 |
| 0.05° - 0.1° | 优秀 | 满足部署要求 |
| 0.1° - 0.2° | 良好 | 可接受 |
| > 0.2° | 一般 | 帧数不足或模型不匹配 |

同时检查 `confidence.total_std`: 越小越稳定。一般 < 2.0° 表示聚合质量好。

### Q: sequence 模式和 temporal 模式有什么区别？

| 特性 | `sequence` 模式 | `temporal` 模式 |
|---|---|---|
| 数据源 | 指定的测试数据目录 | 训练集的验证子集 |
| 序列分割 | 按目录结构 (KITTI sequences) | 按 GT 外参变化自动检测 |
| 流程 | 与 evaluate_checkpoint.py 一致 | 简化版 |
| 用途 | **生产评估/部署** | 快速验证 |
| 推荐 | 正式评估用这个 | 开发调试时快速查看 |

### Q: 推理速度怎么样？

单帧推理 (640x360, L20 GPU):
- PyTorch: ~15-25ms/帧
- DrInfer: ~8-12ms/帧 (1.5-2x 加速)

800 帧序列完整推理 (含数据加载):
- batch_size=8: ~2-3 分钟/序列
- batch_size=16: ~1-2 分钟/序列
