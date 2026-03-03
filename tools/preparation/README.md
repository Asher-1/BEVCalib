# tools/preparation - 数据准备工具集

## 流程概览

```
原始 trips 目录          数据准备                    图像 Resize
  /trips/              batch_prepare_trips.py       resize_images.py
  ├── trip_A/   ──→    sequences/*/image_2/*.png    sequences/*/image_2_640x360/*.jpg
  ├── trip_B/          sequences/*/velodyne/*.bin
  └── trip_C/          sequences/*/calib.txt
```

**步骤 1** 从原始 bag 数据提取 PNG 图像 + 点云 + 标定文件（KITTI 格式）

**步骤 2** 将 4K PNG 批量 resize 为训练用 JPEG（4.3 MB → 40 KB，加载速度提升 10 倍）

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `batch_prepare_trips.py` | 步骤 1：批量数据准备 |
| `prepare_custom_dataset.py` | 步骤 1：单 trip 数据准备（被 batch 调用） |
| `resize_images.py` | 步骤 2：图像 resize 核心 |
| `run_preparation_pipeline.sh` | 一键脚本：步骤 1 + 步骤 2 串联 |
| `run_resize_only.sh` | 快捷脚本：仅执行步骤 2 |

---

## 使用方法

### 方式 1：一键完成（推荐）

```bash
./run_preparation_pipeline.sh <trips_dir> <output_dir> [width] [height] [camera] [fps] [--force-config]
```

**示例：**

```bash
cd tools/preparation

# 默认行为：优先使用bag中的合格lidar外参，不合格则fallback到lidars.cfg
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_data \
    640 360

# 强制使用lidars.cfg外参（当已知bag中的外参不准确时使用）
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_data_force_config \
    640 360 traffic_2 10.0 --force-config
```

| 参数 | 说明 | 默认值 |
|----------|------|--------|
| `$1` trips_dir | trips 根目录（包含多个 trip 子目录） | 必填 |
| `$2` output_dir | 输出目录 | 必填 |
| `$3` width | resize 目标宽度 | 640 |
| `$4` height | resize 目标高度 | 360 |
| `$5` camera_name | 相机名称 | traffic_2 |
| `$6` target_fps | 目标帧率 | 10.0 |
| `--force-config` | 强制使用 lidars.cfg 中的外参 | 不启用 |

---

### 方式 2：分步执行

#### 步骤 1：批量数据准备

```bash
python batch_prepare_trips.py \
    --trips_dir /path/to/trips \
    --output_dir /path/to/output \
    --camera_name traffic_2 \
    --target_fps 10.0 \
    --start_sequence 0 \
    --force-config  # 可选：强制使用lidars.cfg外参
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--trips_dir` | trips 根目录（包含多个 trip 子目录） | 必填 |
| `--output_dir` | 输出目录 | 必填 |
| `--camera_name` | 相机名称 | traffic_2 |
| `--target_fps` | 目标帧率 | 10.0 |
| `--start_sequence` | 起始 sequence ID | 0 |
| `--force-config` | 强制使用 lidars.cfg 中的外参替代 bag 外参 | 不启用 |

执行完成后脚本会自动提示步骤 2 的命令。

#### 步骤 2：图像 Resize

**快捷脚本：**

```bash
./run_resize_only.sh <dataset_root> [width] [height] [workers] [quality]
```

```bash
./run_resize_only.sh /path/to/output 640 360
```

| 位置参数 | 说明 | 默认值 |
|----------|------|--------|
| `$1` dataset_root | 数据集根目录（含 sequences/） | 必填 |
| `$2` width | 目标宽度 | 640 |
| `$3` height | 目标高度 | 360 |
| `$4` workers | 并行进程数 | 32 |
| `$5` quality | JPEG 质量 0-100 | 95 |

**或直接调用 Python：**

```bash
python resize_images.py \
    --dataset_root /path/to/output \
    --width 640 \
    --height 360 \
    --workers 32 \
    --quality 95
```

---

### 方式 3：仅 Resize（数据已存在）

```bash
./run_resize_only.sh /path/to/existing_dataset 640 360
```

可以对同一数据集生成多个分辨率：

```bash
./run_resize_only.sh /path/to/output 640 360
./run_resize_only.sh /path/to/output 800 448
./run_resize_only.sh /path/to/output 1024 576
```

---

## 输出目录结构

```
output_dir/
├── sequences/
│   ├── 00/
│   │   ├── image_2/              ← 原始 PNG（步骤 1）
│   │   ├── image_2_640x360/      ← resize JPEG（步骤 2）
│   │   ├── velodyne/             ← 点云 .bin（步骤 1）
│   │   ├── calib.txt             ← 标定参数（步骤 1）
│   │   └── times.txt
│   ├── 01/
│   │   └── ...
│   └── ...
├── image_2_640x360_meta.json     ← resize 元数据
└── batch_processing_*.log        ← 处理日志
```

---

## 完整示例

```bash
# 进入工具目录
cd /mnt/drtraining/user/dahailu/code/BEVCalib/tools/preparation

# === 方式 A：一键 ===
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_data \
    640 360

# === 方式 B：分步 ===
# 步骤 1
python batch_prepare_trips.py \
    --trips_dir /mnt/drtraining/user/dahailu/data/bevcalib/test_trips \
    --output_dir /mnt/drtraining/user/dahailu/data/bevcalib/test_data

# 检查生成结果
ls /mnt/drtraining/user/dahailu/data/bevcalib/test_data/sequences/00/image_2/

# 步骤 2
./run_resize_only.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_data \
    640 360

# === 训练 ===
python ../../kitti-bev-calib/train_kitti.py \
    --data_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data \
    --img_H 360 \
    --img_W 640
```

---

## Lidar 外参来源（`--force-config`）

calib.txt 中的 Tr 矩阵（LiDAR→Camera 变换）依赖 `sensor_to_lidar` 外参。外参来源优先级：

| 模式 | sensor_to_lidar 来源 | 适用场景 |
|------|----------------------|----------|
| 默认 | BAG 优先（通过验证后使用），不合格则 fallback 到 `lidars.cfg` | BAG 中有准确标定数据 |
| `--force-config` | 始终使用 `lidars.cfg` | 已知 BAG 中的外参不准确 |

**BAG 外参验证规则：**
- `frame_id` 不能是 `lidar_uncalibrated`
- `sensor_to_lidar` 不能接近 identity 矩阵（旋转+平移均≈0 视为未标定）

---

## 常见问题

**Q: `$'\r': command not found`**
A: 换行符问题，运行 `sed -i 's/\r$//' *.sh`

**Q: resize 太慢**
A: 增加并行进程 `./run_resize_only.sh /data/output 640 360 64`

**Q: 可以只 resize 不重新准备数据吗？**
A: 可以，直接用 `./run_resize_only.sh`，已 resize 的图会自动跳过。

**Q: 支持多分辨率吗？**
A: 支持，对同一数据集多次运行不同尺寸即可，各自生成独立目录。

**Q: 什么时候用 `--force-config`？**
A: 当你发现 BAG 中的 lidar 外参（如 `sensor_to_lidar`）不准确时使用。可以先不加此选项生成一份，再加 `--force-config` 生成一份，对比投影效果来决定。
