# BEVCalib 自定义数据集制作 Pipeline

## 1. 数据格式要求

### 1.1 目录结构

自定义数据集需遵循 KITTI-like 格式：

```
your-dataset/
├── sequences/
│   ├── 00/
│   │   ├── image_2/           # 相机图像（PNG）
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   ├── image_2_640x360/   # 预缩放图像（可选，加速训练）
│   │   ├── velodyne/          # 点云数据（BIN）
│   │   │   ├── 000000.bin
│   │   │   └── ...
│   │   └── calib.txt          # 标定文件
│   └── 01/
│       └── ...
```

### 1.2 图像格式

- **格式**：PNG（推荐）或 JPG
- **命名**：6 位数字 + 扩展名（`000000.png`）
- **模式**：RGB
- **尺寸**：训练时自动 resize 到 640×360（Custom）或 704×256（KITTI）

可选：预缩放图像放在 `image_2_{W}x{H}/` 目录，`CustomDataset` 会自动检测并使用。

### 1.3 点云格式

- **格式**：BIN 文件（`np.float32`）
- **形状**：`(N, 4)` 含 x, y, z, intensity；或 `(N, 3)` 仅 xyz
- **坐标系**：LiDAR 坐标系（x 前，y 左，z 上）
- **单位**：米

```python
import numpy as np
points = np.array([[x, y, z, intensity], ...], dtype=np.float32)
points.tofile('sequences/00/velodyne/000000.bin')
```

### 1.4 标定文件（calib.txt）

KITTI 格式：

```
P2: fx 0 cx 0 0 fy cy 0 0 0 1 0
Tr: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1
```

- **P2**：相机投影矩阵（3×4），前 3×3 为内参 K
- **Tr**：LiDAR→Camera 变换矩阵（4×4）

`CustomDataset` 还支持扩展标定格式（distortion、camera_model、T_cam2sensing）。

## 2. 数据准备工具

### 2.1 从 ROS bag 转换

```bash
python tools/preparation/prepare_custom_dataset.py \
    --bag_path /path/to/rosbag \
    --output_dir /path/to/output \
    --image_topic /camera/image_raw \
    --lidar_topic /lidar/points
```

### 2.2 图像预缩放

```bash
python tools/preparation/resize_images.py \
    --input_dir sequences/00/image_2 \
    --output_dir sequences/00/image_2_640x360 \
    --width 640 --height 360
```

### 2.3 数据验证

```bash
python tools/validation/validate_dataset.py \
    --dataset_root /path/to/dataset
```

检查项：目录结构、文件匹配、图像可读性、点云格式、标定文件解析。

## 3. 使用自定义数据集训练

### 3.1 环境变量配置

```bash
export BEV_DATASET_TYPE=custom
export BEV_ZBOUND_STEP=4.0      # Z 方向体素步长
export BEV_XBOUND_MIN=0          # 可选：覆盖 X 范围
export BEV_XBOUND_MAX=200
```

### 3.2 训练命令

```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /path/to/your-dataset \
    --use_custom_dataset \
    --log_dir ./logs/custom_exp \
    --num_epochs 400 \
    --batch_size 8 \
    --rotation_only \
    --enable_axis_loss
```

### 3.3 数据利用率验证

`CustomDataset` 支持预训练数据利用率检查：

```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /path/to/data \
    --validate_data \
    --validate_sample_ratio 0.1 \
    --min_point_utilization 0.3 \
    --min_valid_ratio 0.8
```

## 4. 坐标系约定

| 坐标系 | X | Y | Z |
|--------|---|---|---|
| LiDAR (Ego) | 前 | 左 | 上 |
| Camera | 右 | 下 | 前 |

变换关系：`T_lidar2cam` 将 LiDAR 坐标系中的点变换到相机坐标系。
