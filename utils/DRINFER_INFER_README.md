# drinfer_infer.py 使用文档

BEVCalib 推理评估与性能对比工具。支持 PyTorch / DrInfer 两种后端，提供 4 种运行模式。

---

## 目录

- [快速开始](#快速开始)
- [运行方式](#运行方式)
  - [方式一: YAML 配置文件](#方式一-yaml-配置文件)
  - [方式二: 命令行参数](#方式二-命令行参数)
  - [方式三: 混合 (配置文件 + 命令行覆盖)](#方式三-混合)
- [四种运行模式](#四种运行模式)
  - [eval 模式: 单后端精度评估](#1-eval-模式)
  - [compare 模式: PyTorch vs DrInfer 对比](#2-compare-模式)
  - [temporal 模式: 时序聚合评估](#3-temporal-模式)
  - [sequence 模式: 序列级 MEDW 推理](#4-sequence-模式)
- [YAML 配置文件详解](#yaml-配置文件详解)
- [命令行参数详解](#命令行参数详解)
- [调用场景](#调用场景)
- [输出说明](#输出说明)

---

## 快速开始

```bash
# 进入项目目录
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 场景 1: 用 PyTorch 评估一个 checkpoint 的精度
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --backend pytorch

# 场景 2: 对测试数据做序列级标定 (最常用的生产部署模式)
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data \
  --max-frames 800
```

---

## 运行方式

### 方式一: YAML 配置文件

将所有参数写入 YAML 文件，只需一个 `--config` 参数。

```bash
python utils/drinfer_infer.py --config configs/my_config.yaml
```

YAML 文件示例 (`configs/my_config.yaml`):

```yaml
ckpt_path: "logs/model/checkpoint/ckpt_best_val.pth"
dataset_root: "/path/to/dataset"
img_height: 360
img_width: 640
rotation_only: true
angle_range_deg: 5.0
batch_size: 8
bev_zbound_step: "4.0"
inference_backend: "pytorch"
device: "cuda"
```

### 方式二: 命令行参数

所有核心参数都可以通过命令行直接传递，无需配置文件。此时 `--config` 指向一个包含最少必要字段 (`ckpt_path`, `dataset_root`) 的 YAML。

```bash
python utils/drinfer_infer.py \
  --config configs/minimal_config.yaml \
  --mode sequence \
  --backend pytorch \
  --data-dir /path/to/test_data \
  --seq-ids "00,01,02" \
  --max-frames 800 \
  --agg-method axis_angle_median \
  --output results/report.json
```

### 方式三: 混合

YAML 文件定义基础配置，命令行参数覆盖特定项。命令行优先级高于 YAML。

```bash
# YAML 中 inference_backend: drinfer，命令行覆盖为 pytorch
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --backend pytorch \
  --output /tmp/my_eval.json
```

---

## 四种运行模式

### 1. eval 模式

**用途**: 使用单个后端 (PyTorch 或 DrInfer) 评估 checkpoint 的推理精度和耗时。

```bash
# PyTorch 后端评估
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode eval \
  --backend pytorch

# DrInfer 后端评估
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode eval \
  --backend drinfer

# 保存可视化投影图
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode eval \
  --vis-dir results/vis/
```

**输出**: 每个样本的 RPY 旋转误差 + 平移误差 + 推理耗时统计 (Mean/Median/P95/Max)。

### 2. compare 模式

**用途**: 同时运行 PyTorch 和 DrInfer，对比精度差异、延迟加速比、显存占用。

```bash
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode compare \
  --output results/compare_report.json
```

**前置条件**: 需要先用 `torch2drinfer.py` 导出 DrInfer 模型。

**输出**: 两个后端的精度、耗时、显存侧对侧对比表。

### 3. temporal 模式

**用途**: 在验证集上做时序聚合评估，模拟在线标定场景。自动检测序列边界，逐序列聚合。

```bash
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode temporal \
  --agg-method axis_angle_median

# 使用 SVD-Mean 聚合方法
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode temporal \
  --agg-method svd_mean
```

**输出**: 每个序列的聚合旋转误差 (RPY) + 全局均值/标准差。

### 4. sequence 模式

**用途**: 对指定目录的 KITTI 格式数据做序列级推理，输出每个序列的聚合标定外参和误差。与 `evaluate_checkpoint.py` 的评估管道完全一致。

```bash
# 评估所有序列，800 帧聚合
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data_v2 \
  --max-frames 800

# 仅评估指定序列
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data_v2 \
  --seq-ids "00,03,05" \
  --max-frames 1600

# 保存 JSON 报告
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /path/to/test_data_v2 \
  --output results/sequence_report.json
```

**输出**: 每个序列的帧数、Rot/Roll/Pitch/Yaw 误差 + 整体均值/标准差。

---

## YAML 配置文件详解

```yaml
# ======================== 必填 ========================

ckpt_path: "path/to/ckpt_best_val.pth"      # checkpoint 文件路径
dataset_root: "/path/to/dataset"             # KITTI 格式数据集根目录

# ======================== 模型架构 ========================

deformable: false          # 是否使用 Deformable Attention (默认 false)
bev_encoder: true          # 是否使用 BEV Encoder (默认 true)
rotation_only: true        # 仅旋转校准 (默认 true, 自动从 checkpoint 检测)
# backbone_type / backbone_variant 自动从 checkpoint 检测，无需手动指定

# ======================== 输入尺寸 ========================

img_height: 360            # 输入图像高度 (默认 360)
img_width: 640             # 输入图像宽度 (默认 640)
max_num_points: 200000     # 最大点云点数 (DrInfer 模式必填)

# ======================== BEV 参数 ========================

bev_zbound_step: "4.0"    # BEV Z 轴步长 (必须与训练一致, 常用 2.0 或 4.0)
voxel_mode: "hard"         # 体素化模式: hard / scatter (默认 scatter)
to_bev_mode: "concat"      # BEV 投影模式: concat / learned / sum
scatter_reduce: "sum"      # scatter 聚合方式: sum / mean
max_attn_tokens: 0         # Transformer token 数限制 (0=不限)

# ======================== 评估参数 ========================

angle_range_deg: 5.0       # 扰动角度范围 (度)
trans_range: 0.15          # 扰动平移范围 (米)
batch_size: 8              # 推理 batch size
num_workers: 4             # 数据加载线程数
max_batches: 0             # 最大 batch 数 (0=全部)
validate_sample_ratio: 0.2 # 验证集比例 (eval/compare/temporal 模式使用)

# ======================== sequence 模式专用 ========================

max_frames: 800            # 每序列最大帧数 (sequence 模式)
eval_seed: 42              # 扰动随机种子

# ======================== DrInfer 参数 ========================

export_dir: "path/to/drinfer"   # DrInfer 模型导出目录
model_name: "bevcalib_fusion_head"
model_version: "v2"

# ======================== 输出 ========================

inference_backend: "pytorch"    # 默认后端: pytorch / drinfer
device: "cuda"                  # 运行设备
report_output: "results/report.json"  # JSON 报告输出路径
```

---

## 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | str | `utils/drinfer_config.yaml` | YAML 配置文件路径 |
| `--mode` | str | `eval` | 运行模式: `eval` / `compare` / `temporal` / `sequence` |
| `--backend` | str | 配置文件中的值 | 推理后端: `pytorch` / `drinfer` |
| `--output` | str | None | JSON 报告输出路径 |
| `--vis-dir` | str | None | 投影可视化图片保存目录 (eval 模式) |
| `--agg-method` | str | `axis_angle_median` | 时序聚合方法: `axis_angle_median` / `svd_mean` |
| `--data-dir` | str | None | 数据集目录 (sequence 模式, 覆盖配置文件) |
| `--seq-ids` | str | None | 逗号分隔的序列 ID (sequence 模式, 默认全部) |
| `--max-frames` | int | 800 | 每序列最大帧数 (sequence 模式) |

---

## 调用场景

### 场景 1: 训练后快速验证精度

训练完成后，用 eval 模式快速检查 checkpoint 的推理精度。

```bash
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode eval \
  --backend pytorch \
  --output logs/quick_eval.json
```

### 场景 2: 生产部署前精度验证

用 sequence 模式在测试数据上做完整的序列级评估，确认 MEDW800 精度达标。

```bash
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /data/bevcalib/test_data_v2 \
  --max-frames 800 \
  --output results/deploy_check.json
```

### 场景 3: DrInfer 模型导出后验证

导出 DrInfer 模型后，用 compare 模式确认精度一致性和加速比。

```bash
# 先导出
python utils/torch2drinfer.py --config configs/drinfer_config_all.yaml

# 再对比
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode compare \
  --output results/pt_vs_dr.json
```

### 场景 4: 评估不同聚合帧数的效果

对比 400/800/1600 帧聚合的精度差异。

```bash
for N in 400 800 1600; do
  python utils/drinfer_infer.py \
    --config configs/drinfer_config_all.yaml \
    --mode sequence \
    --data-dir /data/bevcalib/test_data_v2 \
    --max-frames $N \
    --output results/medw_${N}.json
done
```

### 场景 5: 只评估特定序列

```bash
python utils/drinfer_infer.py \
  --config configs/drinfer_config_all.yaml \
  --mode sequence \
  --data-dir /data/bevcalib/test_data_v2 \
  --seq-ids "00,05,08" \
  --max-frames 800
```

---

## 输出说明

### eval 模式输出

```
======================================================================
EVALUATION REPORT [pytorch]  (1000 samples, 50 batches)
======================================================================

Metric               Mean     Median        P95        Max
--------------------------------------------------------------
rot_error            2.3160     2.2060     4.6600     5.0150  deg
roll_error           1.0410     0.9500     2.1200     3.5000  deg
...

Timing (ms)          Mean     Median        P95        Min        Max
------------------------------------------------------------------------
total (e2e)          45.2       44.8       48.1       42.0       52.3
```

### sequence 模式输出

```
======================================================================
SEQUENCE INFERENCE REPORT  (12 sequences, MEDW800)
======================================================================

   Seq  Frames      Rot     Roll    Pitch      Yaw
--------------------------------------------------
    00     800   0.0550°   0.0070°   0.0330°   0.0430°
    01     800   0.0810°   0.0410°   0.0200°   0.0670°
    ...
--------------------------------------------------
  Mean            0.0630°   0.0370°   0.0370°   0.0270°
   Std            0.0250°   0.0200°   0.0220°   0.0150°
======================================================================
```

### JSON 报告字段

sequence 模式的 JSON 报告包含:

```json
{
  "mode": "sequence",
  "method": "axis_angle_median",
  "max_frames": 800,
  "checkpoint": "path/to/ckpt.pth",
  "data_dir": "/path/to/data",
  "summary": {
    "n_sequences": 12,
    "rot_mean": 0.063,
    "rot_std": 0.025,
    "roll_mean": 0.037,
    "pitch_mean": 0.037,
    "yaw_mean": 0.027
  },
  "sequences": [
    {
      "seq_id": "00",
      "n_frames": 800,
      "rot_error": 0.055,
      "agg_T": [[...], ...],
      "gt_T": [[...], ...]
    }
  ]
}
```
