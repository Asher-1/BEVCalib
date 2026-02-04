# BEVCalib 数据集工具集

本目录包含用于生成、验证和可视化自定义 KITTI-Odometry 格式数据集的工具脚本。

## 📋 工具列表

### 1. prepare_custom_dataset.py
**主要的数据集准备脚本**

从 ROS bag 文件生成 KITTI-Odometry 格式的数据集，包括：
- 图像提取与同步
- 点云提取与去畸变（参考C++实现）
- 位姿插值与转换（李代数插值）
- 标定文件生成

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bag_dir` | 必填 | ROS bag文件目录 |
| `--config_dir` | 必填 | 配置文件目录（包含cameras.cfg, extrinsics.yaml等） |
| `--output_dir` | 必填 | 输出数据集目录 |
| `--camera_name` | traffic_2 | 相机名称 |
| `--target_fps` | 10.0 | 目标帧率（用于降采样） |
| `--num_workers` | 4 | 并行工作线程数 |
| `--batch_size` | 200 | 批处理大小 |
| `--max_frames` | None | 最大处理帧数（用于测试） |
| `--save_debug_samples` | 0 | 保存调试样本数量（未去畸变点云，用于对比可视化） |
| `--max_pose_gap` | 0.5 | 最大允许的pose间隔（秒），用于处理不连续bag数据 |

**使用示例：**
```bash
# 基本用法
python prepare_custom_dataset.py \
  --bag_dir /path/to/bag/dir \
  --config_dir /path/to/config/dir \
  --output_dir /path/to/output/dir \
  --camera_name traffic_2

# 完整参数（推荐）
python prepare_custom_dataset.py \
  --bag_dir /path/to/bag/dir \
  --config_dir /path/to/config/dir \
  --output_dir /path/to/output/dir \
  --camera_name traffic_2 \
  --target_fps 10.0 \
  --num_workers 8 \
  --batch_size 200 \
  --save_debug_samples 20

# 快速测试（只处理100帧）
python prepare_custom_dataset.py \
  --bag_dir /path/to/bag/dir \
  --config_dir /path/to/config/dir \
  --output_dir /path/to/output/dir \
  --camera_name traffic_2 \
  --max_frames 100
```

**输出目录结构：**
```
output_dir/
├── sequences/00/
│   ├── image_2/          # PNG图像 (000000.png, 000001.png, ...)
│   ├── velodyne/         # 去畸变后的点云 (000000.bin, ...)
│   ├── debug_raw_pointclouds/  # 未去畸变点云样本（如果启用--save_debug_samples）
│   ├── calib.txt         # 标定文件
│   └── times.txt         # 时间戳文件
├── poses/00.txt          # 位姿文件
└── temp/                 # 临时文件（可删除）
```

**详细文档：** 参见 `../docs/自定义数据集制作Pipeline.md`

---

### 2. visualize_projection.py
**点云投影可视化工具**

支持两种模式：
- **project**: 单纯的点云投影到图像
- **compare**: 对比去畸变前后的效果

**特性：**
- ✅ 支持 PINHOLE 和 KANNALA_BRANDT 两种相机模型
- ✅ 自动处理相机畸变系数
- ✅ FOV 过滤（对齐 C++ 实现）
- ✅ 深度着色渲染

**使用示例：**
```bash
# 投影单帧点云
python visualize_projection.py \
  --mode project \
  --dataset_root /path/to/dataset \
  --frame 0

# 对比去畸变效果（使用debug_raw_pointclouds目录）
python visualize_projection.py \
  --mode compare \
  --dataset_root /path/to/dataset \
  --frame 0 \
  --debug_sample 0

# 批量对比多帧
python visualize_projection.py \
  --mode compare \
  --dataset_root /path/to/dataset \
  --frame 0 \
  --num_frames 5
```

**注意：** 对比去畸变效果需要在生成数据时使用 `--save_debug_samples` 参数保存未去畸变的点云样本。

---

### 3. validate_kitti_odometry.py
**KITTI-Odometry 格式验证器**

严格验证数据集是否符合 KITTI-Odometry 标准格式。

**检查项：**
- ✅ 目录结构（sequences/, poses/）
- ✅ 标定文件格式（P0-P3, Tr）
- ✅ 位姿文件格式（每行12个数）
- ✅ 图像和点云命名格式
- ✅ 数据对齐（数量一致性）
- ✅ 坐标范围合理性

**使用示例：**
```bash
python validate_kitti_odometry.py /path/to/dataset --sequence 00
```

**输出：** 详细的验证报告，包括通过项、警告和错误

---

### 4. visualize_kitti_structure.py
**数据集结构可视化工具**

快速浏览数据集的整体结构和统计信息。

**功能：**
- 📊 序列统计（帧数、时长、FPS）
- 📐 图像和点云尺寸/范围
- 🔧 标定参数预览
- ✅ 数据完整性检查

**使用示例：**
```bash
# 分析整个数据集
python visualize_kitti_structure.py /path/to/dataset

# 分析特定序列
python visualize_kitti_structure.py /path/to/dataset --sequence 00

# 分析多个序列
python visualize_kitti_structure.py /path/to/dataset --sequence 00 01 02
```

---

### 5. view_pointcloud.py
**点云查看工具**

支持查看 PLY 和 BIN 格式的点云文件。

**支持格式：**
- `.ply`: PLY 格式（ASCII）
- `.bin`: KITTI BIN 格式（每点 4 或 5 个 float32）

**可视化后端：**
- **Open3D**（推荐）：交互式 3D 查看
- **Matplotlib**（备选）：简单的 3D 散点图

**使用示例：**
```bash
# 查看 PLY 格式点云
python view_pointcloud.py temp/pointclouds/000000.ply

# 查看 BIN 格式点云
python view_pointcloud.py sequences/00/velodyne/000000.bin

# 查看多个点云（对比）
python view_pointcloud.py temp/pointclouds/000000.ply sequences/00/velodyne/000000.bin

# 只显示统计信息
python view_pointcloud.py temp/pointclouds/000000.ply --info

# 指定后端
python view_pointcloud.py temp/pointclouds/000000.ply --backend matplotlib
```

---

## 🔄 典型工作流程

### 1. 生成数据集（完整流程）
```bash
# 生成完整数据集（包含调试样本用于验证去畸变效果）
python tools/prepare_custom_dataset.py \
  --bag_dir /mnt/drtraining/user/dahailu/data/bevcalib/bags/unimportant \
  --config_dir /mnt/drtraining/user/dahailu/data/bevcalib/config \
  --output_dir /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data \
  --camera_name traffic_2 \
  --target_fps 10.0 \
  --num_workers 32 \
  --batch_size 800 \
  --save_debug_samples 20
```

**预期输出：**
- 数据提取: ~2分钟
- 数据同步: <1秒
- 去畸变保存: ~4-5分钟
- **总计: ~6-7分钟**

### 2. 验证数据集
```bash
python validate_kitti_odometry.py /data/kitti_dataset --sequence 00
```

### 3. 可视化检查
```bash
# 查看数据集结构
python visualize_kitti_structure.py /data/kitti_dataset --sequence 00

# 验证投影效果
python visualize_projection.py \
  --mode project \
  --dataset_root /data/kitti_dataset \
  --frame 0

# 对比去畸变效果（需要--save_debug_samples）
python visualize_projection.py \
  --mode compare \
  --dataset_root /data/kitti_dataset \
  --frame 0 \
  --debug_sample 0
```

### 4. 查看点云
```bash
# 查看去畸变后的点云
python view_pointcloud.py /data/kitti_dataset/sequences/00/velodyne/000000.bin

# 对比去畸变前后（需要--save_debug_samples）
python view_pointcloud.py \
  /data/kitti_dataset/sequences/00/debug_raw_pointclouds/000000_raw.bin \
  /data/kitti_dataset/sequences/00/velodyne/000000.bin
```

### 5. 开始训练
```bash
cd kitti-bev-calib
python train_kitti.py --dataset_root /data/kitti_dataset
```

---

## 📚 相关文档

- **数据集制作指南**: `../docs/自定义数据集制作Pipeline.md`
- **训练和测试流程**: `../docs/训练和测试流程文档.md`
- **原理解析**: `../docs/原理解析文档.md`
- **代码架构**: `../docs/代码架构文档.md`

---

## 🛠️ 依赖项

```bash
# 核心依赖
pip install numpy opencv-python scipy

# 可选依赖（用于点云可视化）
pip install open3d matplotlib

# ROS bag 处理（二选一）
pip install rosbag  # ROS1
pip install rosbags  # ROS2 或独立使用
```

---

## 💡 提示

1. **性能优化**：
   - 使用 `--batch_size` 和 `--num_workers` 参数加速数据生成
   - 对于大型数据集，先用 `--max_frames 10` 测试

2. **调试**：
   - 使用 `--keep_temp` 保留中间文件以便检查
   - 使用 `visualize_projection.py --mode compare` 验证去畸变效果

3. **相机模型**：
   - **PINHOLE**: 标准针孔模型（5个畸变系数）
   - **KANNALA_BRANDT**: 鱼眼模型（4个畸变系数）
   - 畸变系数会自动从 `cameras.cfg` 提取并应用

---

## 📞 问题反馈

如果遇到问题，请检查：
1. 配置文件格式是否正确（`cameras.cfg`, `lidars.cfg`）
2. ROS bag 文件是否包含所需的 topics
3. 相机名称是否与配置文件中的 `camera_dev` 匹配

更多详细信息请参考项目文档。
