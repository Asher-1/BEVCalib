# bevcalib_inference.py 使用文档

BEVCalib 推理核心库。提供模型加载、单帧推理、时序聚合、序列级标定等功能。

本模块是一个 **Python 库** (import 调用)，不是 CLI 脚本。CLI 入口见 `drinfer_infer.py`。

---

## 目录

- [快速开始](#快速开始)
- [核心 API](#核心-api)
  - [load_bevcalib_inference — 加载模型](#1-load_bevcalib_inference)
  - [load_bevcalib_with_aggregation — 加载模型+聚合器](#2-load_bevcalib_with_aggregation)
  - [infer_sequence — 序列级推理](#3-infer_sequence)
  - [TemporalCalibrationAggregator — 时序聚合器](#4-temporalcalibrationaggregator)
  - [BEVCalibInference — 推理 wrapper](#5-bevcalibinference)
- [调用场景](#调用场景)
- [参数详解](#参数详解)
- [环境变量](#环境变量)
- [常见问题](#常见问题)

---

## 快速开始

### 30 秒上手: 序列级标定

```python
from utils.bevcalib_inference import infer_sequence

# 一行搞定: 加载模型 → 推理全序列 → MEDW 聚合 → 返回标定结果
results = infer_sequence(
    ckpt_path="logs/model/checkpoint/ckpt_best_val.pth",
    data_dir="/data/bevcalib/test_data_v2",
    max_frames=800,
)

for r in results:
    print(f"Seq {r['seq_id']}: Rot={r['rot_error']:.4f}°")
    print(f"  标定外参 T (4x4):\n{r['agg_T']}")
```

### 30 秒上手: 流式在线标定

```python
from utils.bevcalib_inference import load_bevcalib_with_aggregation

model, agg, epoch = load_bevcalib_with_aggregation("ckpt_best_val.pth")

for img, pc, init_T, post_T, K in data_stream:
    pred_T = model(img, pc, init_T, post_T, K)
    agg.add(pred_T)
    if agg.ready:  # 已积累足够帧数
        calibration = agg.aggregate()      # (4,4) 标定外参
        confidence = agg.get_confidence()   # 置信度
        print(f"标定完成: {agg.count} 帧, 置信度: {confidence}")

agg.reset()  # 重置，准备下一个序列
```

---

## 核心 API

### 1. load_bevcalib_inference

**加载 checkpoint 并返回推理 wrapper**。自动检测 backbone 类型、旋转模式等。

```python
from utils.bevcalib_inference import load_bevcalib_inference

wrapper, epoch = load_bevcalib_inference(
    ckpt_path="path/to/ckpt_best_val.pth",
    device="cuda",
    img_shape=(360, 640),
    # 以下参数全部支持自动检测，通常无需手动指定
    rotation_only=None,      # None=自动检测
    backbone_type=None,      # None=自动检测 (swin / dinov2)
    backbone_variant=None,   # None=自动检测 (dinov2-small / dinov2-base)
)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ckpt_path` | str | 必填 | `.pth` checkpoint 文件路径 |
| `device` | str | `"cuda"` | 运行设备: `cuda` / `cpu` |
| `img_shape` | tuple | `(360, 640)` | 输入图像尺寸 `(H, W)` |
| `rotation_only` | bool/None | `None` | `None`=自动检测, `True`=仅旋转, `False`=旋转+平移 |
| `deformable` | bool | `False` | 是否使用 Deformable Attention |
| `bev_encoder` | bool | `True` | 是否使用 BEV Encoder |
| `use_mlp_head` | bool/None | `None` | `None`=自动检测, `True`=MLP 回归头, `False`=Linear 回归头 |
| `voxel_mode` | str | `"scatter"` | 体素化模式: `hard` / `scatter` |
| `to_bev_mode` | str | `"concat"` | BEV 投影模式: `concat` / `learned` / `sum` |
| `scatter_reduce` | str | `"sum"` | scatter 聚合方式: `sum` / `mean` |
| `bev_pool_factor` | int | `0` | BEV 池化因子 (0=不池化) |
| `max_attn_tokens` | int | `0` | Transformer token 数限制 (0=不限) |
| `backbone_type` | str/None | `None` | `None`=自动检测, `swin` / `dinov2` |
| `backbone_variant` | str/None | `None` | `None`=自动检测, 如 `dinov2-small` |

**返回**:

| 返回值 | 类型 | 说明 |
|---|---|---|
| `wrapper` | `BEVCalibInference` | 推理 wrapper (可直接调用 `wrapper(img, pc, init_T, post_T, K)`) |
| `epoch` | int | checkpoint 对应的训练 epoch |

---

### 2. load_bevcalib_with_aggregation

**加载模型 + 创建时序聚合器**。一次调用，生产部署直接用。

```python
from utils.bevcalib_inference import load_bevcalib_with_aggregation

model, aggregator, epoch = load_bevcalib_with_aggregation(
    ckpt_path="ckpt_best_val.pth",
    min_frames=50,       # 最少需要 50 帧才允许聚合
    max_frames=1600,     # 滑动窗口最大帧数
    device="cuda",
    img_shape=(360, 640),
)
```

**参数**: 除了 `min_frames` 和 `max_frames` 外，其余参数与 `load_bevcalib_inference` 相同。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `min_frames` | int | `50` | 最少帧数 (低于此值 `agg.ready` 返回 `False`) |
| `max_frames` | int | `1600` | 滑动窗口最大帧数 (超出时自动丢弃最旧帧) |

**返回**:

| 返回值 | 类型 | 说明 |
|---|---|---|
| `wrapper` | `BEVCalibInference` | 推理 wrapper |
| `aggregator` | `TemporalCalibrationAggregator` | 时序聚合器实例 |
| `epoch` | int | 训练 epoch |

---

### 3. infer_sequence

**一键完成序列级推理**: 加载模型 → 加载数据 → 生成扰动 → 逐帧推理 → MEDW 聚合 → 计算误差。

```python
from utils.bevcalib_inference import infer_sequence

results = infer_sequence(
    ckpt_path="path/to/ckpt_best_val.pth",
    data_dir="/path/to/kitti_format_data",
    seq_ids=None,          # None=全部序列, 或 ["00", "03", "05"]
    max_frames=800,        # 每序列最大帧数
    agg_method='axis_angle_median',  # MEDW 聚合 (推荐)
    angle_range_deg=5.0,   # 扰动范围 (需匹配评估配置)
    eval_seed=42,          # 随机种子 (保证可复现)
    batch_size=8,          # 推理 batch size
    device='cuda',
    img_shape=(360, 640),
    verbose=True,
)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ckpt_path` | str | 必填 | checkpoint 路径 |
| `data_dir` | str | 必填 | KITTI 格式数据目录 (包含 `sequences/` 子目录) |
| `seq_ids` | list/None | `None` | 序列 ID 列表, `None` 表示全部 |
| `max_frames` | int | `800` | 每序列最大帧数 |
| `agg_method` | str | `'axis_angle_median'` | 聚合方法: `axis_angle_median` / `svd_mean` |
| `angle_range_deg` | float | `5.0` | 扰动角度范围 (度) |
| `eval_seed` | int | `42` | 随机种子 |
| `batch_size` | int | `8` | 推理 batch size |
| `device` | str | `'cuda'` | 运行设备 |
| `rotation_only` | bool/None | `None` | `None`=自动检测 |
| `img_shape` | tuple | `(360, 640)` | 图像尺寸 `(H, W)` |
| `verbose` | bool | `True` | 是否打印进度 |

**返回**: `list[dict]`，每个序列一个字典:

```python
{
    'seq_id': '00',                    # 序列 ID
    'agg_T': np.ndarray(4, 4),         # 聚合后的 LiDAR→Camera 标定外参
    'gt_T': np.ndarray(4, 4),          # Ground Truth 外参
    'n_frames': 800,                   # 使用的帧数
    'rot_error': 0.055,                # 总旋转误差 (度, geodesic)
    'roll_error': 0.007,               # Roll 误差 (度)
    'pitch_error': 0.033,              # Pitch 误差 (度)
    'yaw_error': 0.043,                # Yaw 误差 (度)
    'confidence': {                    # 置信度 (各轴标准差)
        'roll_std': 0.85,
        'pitch_std': 0.92,
        'yaw_std': 0.78,
        'total_std': 1.47,
        'n_frames': 800,
    },
}
```

---

### 4. TemporalCalibrationAggregator

**时序聚合器**: 累积多帧预测，通过 axis-angle median (MEDW) 方法输出鲁棒的标定结果。

#### 流式使用 (推荐用于在线标定)

```python
from utils.bevcalib_inference import TemporalCalibrationAggregator

agg = TemporalCalibrationAggregator(
    min_frames=50,                    # 最少 50 帧
    max_frames=1600,                  # 滑动窗口 1600 帧
    method='axis_angle_median',       # MEDW 方法
)

# 逐帧添加
for pred_T in predictions:
    agg.add(pred_T)                   # 接受 numpy (4,4) 或 torch Tensor

# 查询状态
print(f"已累积: {agg.count} 帧")
print(f"可聚合: {agg.ready}")         # count >= min_frames

# 聚合
if agg.ready:
    calibration = agg.aggregate()     # (4,4) numpy
    confidence = agg.get_confidence() # dict: roll_std, pitch_std, yaw_std, total_std

# 重置
agg.reset()
```

#### 批量使用 (一次性聚合)

```python
from utils.bevcalib_inference import TemporalCalibrationAggregator

pred_Ts = [model(img_i, ...).cpu().numpy() for img_i in sequence]

# 静态方法，无需实例化
calibration = TemporalCalibrationAggregator.aggregate_batch(
    pred_Ts,
    method='axis_angle_median',
)
```

#### API

| 方法 | 说明 |
|---|---|
| `add(pred_T)` | 添加一帧预测。接受 `(4,4)` numpy/torch, 或 `(B,4,4)` 批量 |
| `aggregate()` | 返回聚合后的 `(4,4)` 标定矩阵 |
| `get_confidence()` | 返回各轴一致性指标 (roll/pitch/yaw std) |
| `reset()` | 清空缓冲区，准备下一个序列 |
| `ready` (property) | `count >= min_frames` 时返回 `True` |
| `count` (property) | 当前缓冲区中的帧数 |
| `aggregate_batch(pred_Ts, method)` (静态) | 一次性聚合，无需实例化 |

#### 聚合方法对比

| 方法 | 参数值 | 精度 | 适用场景 |
|---|---|---|---|
| **MEDW (推荐)** | `axis_angle_median` | 0.063° (800帧) | 零均值噪声, 生产部署 |
| SVD-Mean | `svd_mean` | 0.069° (800帧) | 对比实验 |

#### 帧数与精度关系

| 帧数 | MEDW 精度 | 相对 400 帧提升 |
|---|---|---|
| 400 | 0.096° | 基准 |
| 800 | 0.063° | 34% |
| 1600 | 0.045° | 53% |

---

### 5. BEVCalibInference

**推理 wrapper**: 封装 BEVCalib 模型的纯推理路径 (无 loss 计算)。

```python
# 通常不直接构造，由 load_bevcalib_inference 返回
wrapper, epoch = load_bevcalib_inference("ckpt.pth")

# 直接调用
pred_T = wrapper(
    img,               # (B, 3, H, W) RGB 图像 tensor
    pc,                # (B, N, 3) 点云 tensor
    init_T_to_camera,  # (B, 4, 4) 初始 LiDAR→Camera 变换
    post_cam2ego_T,    # (B, 4, 4) 后处理变换 (推理时传单位矩阵)
    cam_intrinsic,     # (B, 3, 3) 相机内参
)
# pred_T: (B, 4, 4) 预测的 LiDAR→Camera 标定外参
```

---

## 调用场景

### 场景 1: 离线批量标定

对多个序列做标定，保存结果到文件。

```python
import json
from utils.bevcalib_inference import infer_sequence

results = infer_sequence(
    ckpt_path="ckpt_best_val.pth",
    data_dir="/data/sequences",
    max_frames=800,
)

output = {}
for r in results:
    output[r['seq_id']] = {
        'calibration': r['agg_T'].tolist(),
        'rot_error': r['rot_error'],
        'confidence': r['confidence'],
    }
with open("calibrations.json", "w") as f:
    json.dump(output, f, indent=2)
```

### 场景 2: 在线实时标定 (流式)

接入车端数据流，逐帧推理并持续更新标定。

```python
import os
os.environ["BEV_ZBOUND_STEP"] = "4.0"

from utils.bevcalib_inference import load_bevcalib_with_aggregation

model, agg, _ = load_bevcalib_with_aggregation(
    "ckpt_best_val.pth",
    min_frames=100,
    max_frames=800,
)

for img, pc, init_T, post_T, K in sensor_stream():
    pred_T = model(img, pc, init_T, post_T, K)
    agg.add(pred_T)

    if agg.count % 100 == 0:
        calib = agg.aggregate()
        conf = agg.get_confidence()
        print(f"[{agg.count} frames] total_std={conf['total_std']:.2f}°")

        if conf['total_std'] < 1.0:  # 收敛判断
            print("标定已收敛!")
            break
```

### 场景 3: 与 evaluate_checkpoint.py 结果对比验证

确保推理接口与评估管道输出一致。

```python
from utils.bevcalib_inference import infer_sequence

results = infer_sequence(
    ckpt_path="logs/model/checkpoint/ckpt_best_val.pth",
    data_dir="/data/bevcalib/test_data_v2",
    max_frames=800,
    angle_range_deg=5.0,
    eval_seed=42,
)

import numpy as np
rots = [r['rot_error'] for r in results]
print(f"MEDW800 Mean Rot = {np.mean(rots):.4f}°")
# 应与 evaluate_checkpoint.py 报告中的 MEDW800 值接近
```

### 场景 4: 只获取标定矩阵 (不关心误差)

实际部署时没有 GT，只需要标定结果。

```python
from utils.bevcalib_inference import load_bevcalib_inference, TemporalCalibrationAggregator
import torch, numpy as np

wrapper, _ = load_bevcalib_inference("ckpt.pth", device="cuda")
agg = TemporalCalibrationAggregator(min_frames=100, max_frames=800)

# 假设你已经有预处理好的数据
for img_tensor, pc_tensor, init_T_tensor, K_tensor in my_dataloader:
    post_T = torch.eye(4).unsqueeze(0).expand(img_tensor.shape[0], -1, -1).cuda()
    pred_T = wrapper(img_tensor, pc_tensor, init_T_tensor, post_T, K_tensor)
    agg.add(pred_T)

calibration_T = agg.aggregate()  # (4,4) numpy — 这就是最终标定外参
print("LiDAR→Camera 标定外参:")
print(calibration_T)
```

### 场景 5: 对比不同 checkpoint 的泛化性能

```python
from utils.bevcalib_inference import infer_sequence
import numpy as np

checkpoints = {
    "V29-G3": "logs/v29_quick/model.../ckpt_best_val.pth",
    "V30-G3": "logs/v30/model.../ckpt_best_val.pth",
}

for name, ckpt in checkpoints.items():
    results = infer_sequence(ckpt, "/data/test_data_v2", max_frames=800)
    rots = [r['rot_error'] for r in results]
    print(f"{name}: MEDW800 = {np.mean(rots):.4f}° ± {np.std(rots):.4f}°")
```

---

## 环境变量

| 变量 | 说明 | 示例 |
|---|---|---|
| `BEV_ZBOUND_STEP` | BEV Z 轴体素步长。**必须在 import 前设置**，且与训练配置一致 | `export BEV_ZBOUND_STEP=4.0` |
| `HF_HUB_OFFLINE` | 离线模式 (DINOv2 权重从本地加载) | `export HF_HUB_OFFLINE=1` |

```python
import os
os.environ["BEV_ZBOUND_STEP"] = "4.0"
os.environ["HF_HUB_OFFLINE"] = "1"

from utils.bevcalib_inference import load_bevcalib_inference  # 之后再 import
```

---

## 常见问题

**Q: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`**

A: Checkpoint 使用了 DINOv2 backbone，但模型默认初始化为 Swin。解决方法: 让 `backbone_type` 和 `backbone_variant` 自动检测 (设为 `None`)，不要手动指定。

**Q: infer_sequence 的结果与 evaluate_checkpoint.py 有少量差异 (~0.01°)**

A: 正常现象。两者使用相同的聚合逻辑，但 batch_size 不同会导致随机扰动序列略有差异 (固定 seed + 不同 batch 分组 → 不同的 RNG 状态)。差异在 0.01° 级别不影响结论。

**Q: 如何选择 max_frames?**

A: 推荐 800 帧 (精度 0.063°，耗时适中)。追求极致精度用 1600 帧 (0.045°)。最少需要 400 帧才能达到 sub-0.1° 目标。

**Q: `TemporalCalibrationAggregator` 是否线程安全?**

A: 不是。每个线程/序列应使用独立的 aggregator 实例。
