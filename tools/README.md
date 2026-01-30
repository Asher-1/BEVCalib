# BEVCalib 数据集工具集

本目录包含用于生成、验证和可视化自定义 KITTI-Odometry 格式数据集的工具脚本。

## 📋 工具列表

### 1. prepare_custom_dataset.py
**主要的数据集准备脚本**

从 ROS bag 文件生成 KITTI-Odometry 格式的数据集，包括：
- 图像提取与同步
- 点云提取与去畸变
- 位姿插值与转换
- 标定文件生成

**使用示例：**
```bash
python prepare_custom_dataset.py \
  --bag_dir /path/to/bag/dir \
  --config_dir /path/to/config/dir \
  --output_dir /path/to/output/dir \
  --camera_name camera_1 \
  --max_frames 100
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

# 对比去畸变效果
python visualize_projection.py \
  --mode compare \
  --dataset_root /path/to/dataset \
  --frame 0

# 批量对比多帧
python visualize_projection.py \
  --mode compare \
  --dataset_root /path/to/dataset \
  --frame 0 \
  --num_frames 5
```

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

### 1. 生成数据集
```bash
python prepare_custom_dataset.py \
  --bag_dir /data/bag/dir \
  --config_dir /data/bag/dir/config \
  --output_dir /data/kitti_dataset \
  --camera_name traffic_2 \
  --max_frames 100
```

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

# 对比去畸变效果
python visualize_projection.py \
  --mode compare \
  --dataset_root /data/kitti_dataset \
  --frame 0
```

### 4. 查看点云
```bash
# 查看去畸变后的点云
python view_pointcloud.py /data/kitti_dataset/sequences/00/velodyne/000000.bin

# 对比去畸变前后
python view_pointcloud.py \
  /data/kitti_dataset/temp/pointclouds/000000.ply \
  /data/kitti_dataset/sequences/00/velodyne/000000.bin
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
