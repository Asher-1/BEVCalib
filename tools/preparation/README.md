# tools/preparation - 数据准备工具集

## 流程概览

```
输入方式 A（批量）:        输入方式 B（单 trip）:
  /trips/                    /trip_A/
  ├── trip_A/                ├── bags/important/
  ├── trip_B/                └── configs/
  └── trip_C/

          ↓  步骤 1                    ↓  步骤 2
  batch_prepare_trips.py       resize_images.py
  或 prepare_custom_dataset.py
          ↓                            ↓
  sequences/*/image_2/*.png    sequences/*/image_2_640x360/*.jpg
  sequences/*/velodyne/*.bin
  sequences/*/calib.txt
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
./run_preparation_pipeline.sh <input_dir> <output_dir> [width] [height] [camera] [fps] [--force-config]
```

脚本会自动检测 `input_dir` 是**单个 trip 目录**（含 `bags/` 和 `configs/`）还是**trips 根目录**（含多个 trip 子目录），并选择对应的处理方式。

**示例：**

```bash
cd tools/preparation

# ---- 批量模式：处理 trips_dir 下所有 trip ----
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_data \
    640 360

# 批量 + 并行 + 强制config
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_data \
    640 360 traffic_2 10.0 --force-config -j 4

# ---- 单 trip 模式：直接指定一个 trip 目录 ----
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips/YR-C061-9_20260305_055658 \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_single \
    640 360

# 单 trip + 指定 sequence ID
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips/YR-C061-9_20260305_055658 \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_single \
    640 360 traffic_2 10.0 --sequence-id 5
```

| 参数 | 说明 | 默认值 |
|----------|------|--------|
| `$1` input_dir | trips 根目录（含多个 trip 子目录）**或**单个 trip 目录（含 `bags/` 和 `configs/`） | 必填 |
| `$2` output_dir | 输出目录 | 必填 |
| `$3` width | resize 目标宽度 | 640 |
| `$4` height | resize 目标高度 | 360 |
| `$5` camera_name | 相机名称 | traffic_2 |
| `$6` target_fps | 目标帧率 | 10.0 |
| `--force-config` | 强制使用 lidars.cfg 中的外参 | 不启用 |
| `--sequence-id N` | 单 trip 模式的 sequence 编号 | 0 |
| `-j N` | 批量模式并行处理 trip 数量 | 1 |

---

### 方式 2：分步执行

#### 步骤 1a：批量数据准备（多 trip）

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

#### 步骤 1b：单 trip 数据准备

```bash
python prepare_custom_dataset.py \
    --bag_dir /path/to/trip/bags/important \
    --config_dir /path/to/trip/configs \
    --output_dir /path/to/output \
    --sequence_id 0 \
    --camera_name traffic_2 \
    --target_fps 10.0
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bag_dir` | bag 文件目录 | 必填 |
| `--config_dir` | 配置文件目录（含 cameras.cfg, lidars.cfg） | 必填 |
| `--output_dir` | 输出目录 | 必填 |
| `--sequence_id` | 生成的 sequence 编号 | 0 |
| `--camera_name` | 相机名称 | traffic_2 |
| `--target_fps` | 目标帧率 | 10.0 |

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

# === 方式 A：一键批量处理所有 trips ===
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_data \
    640 360

# === 方式 B：一键处理单个 trip ===
./run_preparation_pipeline.sh \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_trips/YR-C061-9_20260305_055658 \
    /mnt/drtraining/user/dahailu/data/bevcalib/test_single \
    640 360

# === 方式 C：分步 ===
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

## 坐标系与 calib.txt 格式

### 坐标系说明

- **velodyne/** 中的点云保存在 **LiDAR 系**（KITTI-Odometry 标准）
- 当 BAG header `frame_id == "lidar"` 时，点云从 Sensing 系自动转换到 LiDAR 系
- 当 `frame_id` 为其他值时，点云已在 LiDAR 系，直接保存

### calib.txt 格式

| 字段 | 含义 | 来源 |
|------|------|------|
| P0-P3 | 相机投影矩阵 (3x4) | cameras.cfg 内参 |
| Tr | Camera → LiDAR (KITTI 标准，3x4) | lidars.cfg `sensor_to_lidar` + cameras.cfg `sensor_to_cam` 合成 |
| T_cam2sensing | Camera → Sensing (系统已知相机外参，3x4) | cameras.cfg `sensor_to_cam` |
| D | 畸变系数 | cameras.cfg 畸变参数 |
| camera_model | 相机模型 (pinhole/fisheye) | cameras.cfg |

**重要**: Tr 始终由 `configs/lidars.cfg` + `configs/cameras.cfg` 合成，不使用 BAG 中的外参。

---

## `--force-config` 说明

`--force-config` 仅影响 **点云坐标转换**（当 BAG 中点云在 Sensing 系时）：

| 模式 | 点云 Sensing→LiDAR 转换来源 | calib.txt Tr 来源 |
|------|-----|------|
| 默认 | BAG 中的 `sensor_to_lidar`（录制时实际使用的外参） | 始终由 configs 合成 |
| `--force-config` | `lidars.cfg` 的 `sensor_to_lidar` | 始终由 configs 合成 |

注意：当 BAG header `frame_id != "lidar"` 时（点云已在 LiDAR 系），`--force-config` 无效果。

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
A: 当 BAG header `frame_id == "lidar"`（点云在 Sensing 系）且 BAG 中的 `sensor_to_lidar` 不准确时使用。可以先不加此选项生成一份，再加 `--force-config` 生成一份，对比投影效果来决定。

**Q: calib.txt 的 Tr 和之前有什么不同？**
A: 之前 Tr = Camera→Sensing（仅使用 cameras.cfg），现在 Tr = Camera→LiDAR（由 lidars.cfg + cameras.cfg 合成），符合 KITTI-Odometry 标准。
