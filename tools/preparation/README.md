# 数据准备工具 (Preparation Tools)

数据集格式转换和准备工具集。

---

## 📋 工具列表

### `prepare_custom_dataset.py` - 自定义数据集准备

将自定义格式的数据转换为KITTI-Odometry格式。

**功能**:
- 支持多种输入格式
- 自动生成标定文件
- 位姿转换和对齐
- 图像和点云时间戳同步
- 数据质量检查

**使用方法**:
```bash
python tools/preparation/prepare_custom_dataset.py \
    --source /path/to/raw/data \
    --output /path/to/output \
    --config config.yaml
```

**配置文件示例** (`config.yaml`):
```yaml
camera:
  width: 1920
  height: 1080
  fx: 1000.0
  fy: 1000.0
  cx: 960.0
  cy: 540.0

lidar:
  type: velodyne
  channels: 64

calibration:
  camera_to_lidar:
    rotation: [...]
    translation: [...]
```

**输出结构**:
```
output/
└── sequences/
    ├── 00/
    │   ├── image_2/        # 图像序列
    │   ├── velodyne/       # 点云序列
    │   ├── calib.txt       # 标定文件
    │   └── poses.txt       # 位姿文件
    ├── 01/
    └── ...
```

---

### `batch_prepare_trips.py` - 批量数据准备

批量处理多个数据集或行程（trip）。

**功能**:
- 自动发现多个数据源
- 并行处理加速
- 统一配置管理
- 批处理日志记录

**使用方法**:
```bash
python tools/preparation/batch_prepare_trips.py \
    --source_dir /path/to/multiple/trips \
    --output_dir /path/to/output \
    --config config.yaml \
    --workers 4
```

**目录结构要求**:
```
source_dir/
├── trip_001/
│   ├── images/
│   ├── pointclouds/
│   └── poses.txt
├── trip_002/
└── ...
```

**输出**:
```
output_dir/
└── sequences/
    ├── 00/    # trip_001
    ├── 01/    # trip_002
    └── ...
```

---

## 🎯 使用场景

### 场景1: 准备单个数据集

```bash
# 1. 准备配置文件
cat > config.yaml << EOF
camera:
  width: 1920
  height: 1080
  fx: 1000.0
  fy: 1000.0
  cx: 960.0
  cy: 540.0
EOF

# 2. 运行转换
python tools/preparation/prepare_custom_dataset.py \
    --source raw_data/ \
    --output dataset/ \
    --config config.yaml

# 3. 验证结果
python tools/validation/validate_dataset.py summary dataset/
```

### 场景2: 批量准备多个行程

```bash
# 批量处理
python tools/preparation/batch_prepare_trips.py \
    --source_dir multiple_trips/ \
    --output_dir dataset/ \
    --config config.yaml \
    --workers 4

# 验证所有序列
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation/ --full
```

---

## 📊 数据准备流程

### 标准流程

```
原始数据
    ↓
1. 数据读取和解析
    ↓
2. 时间戳对齐
    ↓
3. 坐标系转换
    ↓
4. 标定文件生成
    ↓
5. KITTI格式输出
    ↓
6. 质量检查
    ↓
KITTI-Odometry数据集
```

### 关键步骤说明

**1. 时间戳对齐**
- 找到图像和点云的最佳匹配
- 处理时间偏移
- 剔除孤立帧

**2. 坐标系转换**
- 相机坐标系 ↔ 雷达坐标系
- 车体坐标系 ↔ 世界坐标系
- 保持右手坐标系

**3. 标定文件生成**
- 计算P0-P3投影矩阵
- 生成Tr变换矩阵（Velodyne→Camera）
- 验证矩阵正确性

---

## ⚠️ 注意事项

### 1. 输入数据要求

**图像**:
- 格式: PNG, JPG
- 命名: 连续编号或时间戳
- 建议分辨率: >= 640x480

**点云**:
- 格式: BIN (KITTI), PCD, LAS
- 坐标系: 必须已知
- 点格式: (x, y, z, intensity)

**位姿**:
- 格式: TXT, CSV
- 内容: 4x4变换矩阵或7D (x,y,z,qw,qx,qy,qz)

### 2. 常见问题

**Q: 时间戳不对齐怎么办？**
```bash
# 使用时间偏移参数
python tools/preparation/prepare_custom_dataset.py \
    ... \
    --time_offset 0.05  # 50ms偏移
```

**Q: 坐标系不一致？**
```yaml
# 在config.yaml中指定变换
calibration:
  transform:
    rotation: [roll, pitch, yaw]  # 欧拉角（度）
    translation: [x, y, z]          # 平移（米）
```

**Q: 数据量过大？**
```bash
# 使用采样
python tools/preparation/prepare_custom_dataset.py \
    ... \
    --sample_rate 0.5  # 保留50%数据
```

### 3. 性能优化

**大数据集处理**:
- 使用 `batch_prepare_trips.py` 并行处理
- 调整 `--workers` 参数（建议: CPU核心数-2）
- 考虑分批次处理

**内存优化**:
- 避免一次性加载所有数据
- 使用流式处理
- 及时释放大对象

---

## 📈 质量检查

准备完成后，务必进行质量检查：

```bash
# 1. 快速摘要
python tools/validation/validate_dataset.py summary dataset/

# 2. 格式验证
python tools/validation/validate_dataset.py format dataset/ --all

# 3. Tr矩阵检查
python tools/validation/verify_dataset_tr_fix.py --dataset_root dataset/

# 4. 投影效果测试
python tools/validation/check_projection_headless.py \
    --dataset_root dataset/ --sequence 00 --frame 0 \
    --output test_projection.png
```

检查投影图：
- 点云应精确覆盖物体轮廓
- 深度着色应连续合理
- 无明显偏移或扭曲

---

## 🔗 相关文档

- [主文档](../README.md)
- [验证工具文档](../validation/README.md)
- [可视化工具文档](../visualization/README.md)

---

**最后更新**: 2026-03-01
