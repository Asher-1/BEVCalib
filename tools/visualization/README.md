# 可视化工具 (Visualization Tools)

数据可视化和图像生成工具集。

---

## 📋 工具列表

### `visualize_projection.py` - 交互式投影可视化

交互式查看点云投影到图像的效果。

**功能**:
- 实时交互式查看
- 支持键盘控制切换帧
- 深度着色可视化
- 投影统计显示

**使用方法**:
```bash
python tools/visualization/visualize_projection.py \
    --dataset_root /path/to/dataset \
    --sequence 00 \
    --start_frame 0
```

**交互控制**:
- `→` / `Space`: 下一帧
- `←` / `Backspace`: 上一帧
- `q` / `Esc`: 退出
- `s`: 保存当前帧

**显示内容**:
- 原始图像 + 点云投影
- 深度颜色映射
- 可见点统计
- 深度范围信息

---

### `view_pointcloud.py` - 点云查看器

3D点云可视化工具。

**功能**:
- 3D交互式查看
- 支持多种点云格式
- 颜色/强度渲染
- 视角自由旋转

**使用方法**:
```bash
# 查看单个点云文件
python tools/visualization/view_pointcloud.py \
    /path/to/pointcloud.bin

# 查看数据集中的点云
python tools/visualization/view_pointcloud.py \
    --dataset_root /path/to/dataset \
    --sequence 00 \
    --frame 0
```

**交互控制**:
- 鼠标拖动: 旋转视角
- 滚轮: 缩放
- `r`: 重置视角
- `c`: 切换颜色模式（深度/强度）
- `q`: 退出

---

### `visualize_kitti_structure.py` - KITTI结构可视化

可视化KITTI数据集的目录结构和内容。

**功能**:
- 树形显示目录结构
- 统计文件数量
- 检查文件完整性
- 生成结构报告

**使用方法**:
```bash
python tools/visualization/visualize_kitti_structure.py \
    --dataset_root /path/to/dataset \
    --output structure_report.txt
```

**输出示例**:
```
dataset/
└── sequences/
    ├── 00/ (1544 frames) ✅
    │   ├── image_2/ (1544 images)
    │   ├── velodyne/ (1544 point clouds)
    │   ├── calib.txt ✅
    │   └── poses.txt ✅
    ├── 01/ (9000 frames) ✅
    └── ...
```

---

### `batch_generate_projections.py` - 批量生成投影图

批量生成点云投影可视化图像。

**功能**:
- 批量处理多个序列
- 自定义采样策略
- 并行生成加速
- 自动组织输出

**使用方法**:
```bash
# 生成所有序列的投影图
python tools/visualization/batch_generate_projections.py \
    --dataset_root /path/to/dataset \
    --output_dir projections/ \
    --sample_rate 0.1 \
    --workers 4

# 仅生成特定序列
python tools/visualization/batch_generate_projections.py \
    --dataset_root /path/to/dataset \
    --output_dir projections/ \
    --sequences 00 01 02 \
    --frames 0 100 200 300
```

**输出结构**:
```
projections/
├── sequence_00/
│   ├── frame_000000.png
│   ├── frame_000100.png
│   └── ...
├── sequence_01/
└── ...
```

---

## 🎯 使用场景

### 场景1: 快速检查投影质量

```bash
# 交互式查看
python tools/visualization/visualize_projection.py \
    --dataset_root dataset/ --sequence 00
```

### 场景2: 生成展示图片

```bash
# 批量生成关键帧投影
python tools/visualization/batch_generate_projections.py \
    --dataset_root dataset/ \
    --output_dir showcase/ \
    --sequences 00 05 10 \
    --frames 0 500 1000
```

### 场景3: 调试点云数据

```bash
# 查看原始点云
python tools/visualization/view_pointcloud.py \
    dataset/sequences/00/velodyne/000000.bin

# 对比投影效果
python tools/visualization/visualize_projection.py \
    --dataset_root dataset/ --sequence 00 --start_frame 0
```

### 场景4: 数据集概览

```bash
# 生成结构报告
python tools/visualization/visualize_kitti_structure.py \
    --dataset_root dataset/ \
    --output dataset_structure.txt
```

---

## 🎨 可视化效果说明

### 点云投影可视化

**颜色映射**:
- 蓝色（冷色）: 近距离（< 20m）
- 绿色（中间）: 中距离（20-100m）
- 红色（暖色）: 远距离（> 100m）

**显示信息**:
- 总点数
- 可见点数
- 可见率
- 深度范围

### 3D点云可视化

**渲染模式**:
- **深度模式**: 按距离着色
- **强度模式**: 按反射强度着色
- **高度模式**: 按Z坐标着色

**视角控制**:
- 默认: 俯视45°
- 可自由旋转和缩放
- 支持保存视角配置

---

## 📊 批量生成策略

### 采样策略

**均匀采样**:
```bash
python tools/visualization/batch_generate_projections.py \
    ... --sample_rate 0.1  # 每10帧采样1帧
```

**关键帧采样**:
```bash
python tools/visualization/batch_generate_projections.py \
    ... --key_frames  # 开始、1/4、中间、3/4、结束
```

**自定义帧列表**:
```bash
python tools/visualization/batch_generate_projections.py \
    ... --frames 0 10 20 50 100 200
```

### 性能优化

**并行处理**:
```bash
python tools/visualization/batch_generate_projections.py \
    ... --workers 8  # 使用8个进程
```

**输出质量**:
```bash
python tools/visualization/batch_generate_projections.py \
    ... --dpi 150 --figsize 16 9  # 高分辨率输出
```

---

## ⚠️ 注意事项

### 1. GUI依赖

**交互式工具需要显示器**:
- `visualize_projection.py`
- `view_pointcloud.py`

如在服务器上使用，需要：
```bash
# 使用X11转发
ssh -X user@server

# 或使用虚拟显示
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

**非交互式工具（可在服务器运行）**:
- `batch_generate_projections.py`
- `visualize_kitti_structure.py`

### 2. 内存使用

**大规模批量生成**:
- 调整 `--workers` 避免内存不足
- 分批次处理大数据集
- 使用 `--sample_rate` 减少输出量

### 3. 输出大小

生成的图片可能占用较大空间：
```bash
# 单张投影图: ~500KB
# 1000帧 × 12序列 = 12,000张 ≈ 6GB
```

建议：
- 使用合理的采样率
- 定期清理不需要的图片
- 压缩存档

---

## 🔗 相关工具

### 与验证工具配合

```bash
# 1. 先验证投影质量
python tools/validation/validate_dataset.py projection-full dataset/ \
    --output-dir validation_proj/

# 2. 查看验证报告
cat validation_proj/PROJECTION_VALIDATION_REPORT.md

# 3. 针对问题序列生成更多投影
python tools/visualization/batch_generate_projections.py \
    --dataset_root dataset/ \
    --sequences 03 05  # 问题序列
    --output_dir debug_projections/ \
    --sample_rate 0.05  # 密集采样
```

### 与分析工具配合

```bash
# 1. 生成投影图
python tools/visualization/batch_generate_projections.py \
    --dataset_root dataset/ --output_dir proj/

# 2. 统计分析（如需要）
python tools/analysis/analyze_perturbation_training.py \
    --projections proj/
```

---

## 💡 最佳实践

### 1. 首次查看数据集

```bash
# 快速浏览
python tools/visualization/visualize_projection.py \
    --dataset_root dataset/ --sequence 00

# 使用方向键快速翻页，了解数据质量
```

### 2. 制作演示材料

```bash
# 生成高质量图片
python tools/visualization/batch_generate_projections.py \
    --dataset_root dataset/ \
    --output_dir presentation/ \
    --sequences 00 05 10 \
    --frames 0 500 1000 \
    --dpi 300 \
    --figsize 20 11.25
```

### 3. 调试标定问题

```bash
# 1. 查看原始点云
python tools/visualization/view_pointcloud.py \
    dataset/sequences/00/velodyne/000000.bin

# 2. 查看投影效果
python tools/visualization/visualize_projection.py \
    --dataset_root dataset/ --sequence 00 --start_frame 0

# 3. 如发现偏移，检查Tr矩阵
python tools/validation/verify_dataset_tr_fix.py \
    --dataset_root dataset/
```

---

## 🔗 相关文档

- [主文档](../README.md)
- [验证工具文档](../validation/README.md)
- [数据准备文档](../preparation/README.md)

---

**最后更新**: 2026-03-01
