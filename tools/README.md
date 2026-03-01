# BEVCalib Tools 工具集

本目录包含 BEVCalib 数据集准备、验证、可视化和分析的完整工具集。

为便于使用和维护，所有工具已按功能分类组织到不同子目录中。

---

## 📂 目录结构

```
tools/
├── README.md                    # 本文档
├── docs/                        # 📚 文档和指南
├── preparation/                 # 📊 数据准备工具
├── validation/                  # ✅ 验证工具
├── visualization/               # 🎨 可视化工具
├── analysis/                    # 📈 分析工具
├── utils/                       # 🔧 修复与调试工具
└── scripts/                     # 🔄 Shell 脚本工具
```

---

## 🚀 快速开始

### 1. 准备数据集

```bash
# 准备自定义数据集（KITTI-Odometry格式）
python tools/preparation/prepare_custom_dataset.py \
    --source /path/to/raw/data \
    --output /path/to/output \
    --config config.yaml
```

### 2. 验证数据集

```bash
# 方式A: 快速摘要（5秒）
python tools/validation/validate_dataset.py summary /path/to/dataset

# 方式B: 快速验证（17秒）- 日常检查
python tools/validation/validate_dataset.py full /path/to/dataset \
    --output-dir validation_results

# 方式C: 完整验证（15分钟）- 首次验证
python tools/validation/validate_dataset.py full /path/to/dataset \
    --output-dir validation_results --full

# 方式D: 仅投影验证（10分钟）- 每序列10帧
python tools/validation/validate_dataset.py projection-full /path/to/dataset \
    --output-dir projection_validation
```

### 3. 可视化

```bash
# 交互式点云投影可视化
python tools/visualization/visualize_projection.py \
    --dataset_root /path/to/dataset \
    --sequence 00

# 查看单个点云
python tools/visualization/view_pointcloud.py /path/to/pointcloud.bin
```

---

## 📚 详细文档

### 核心文档

- **[快速开始指南](docs/QUICK_START.md)** - 1分钟上手
- **[架构说明](docs/ARCHITECTURE.md)** - 工具设计理念
- **[验证模式详解](docs/VALIDATION_MODES.md)** - 不同验证模式对比

---

## 🛠️ 工具分类说明

### 📊 数据准备工具 (`preparation/`)

数据集格式转换和准备工具。

**主要工具**:
- `prepare_custom_dataset.py` - 转换自定义数据为KITTI-Odometry格式
- `batch_prepare_trips.py` - 批量准备多个数据集

[查看详细文档 →](preparation/README.md)

---

### ✅ 验证工具 (`validation/`)

数据集质量验证和检查工具。

**主要工具**:
- **`validate_dataset.py`** ⭐ - 统一验证入口（推荐）
- `validate_kitti_odometry.py` - KITTI格式验证
- `verify_dataset_tr_fix.py` - Tr矩阵验证
- `comprehensive_projection_validation.py` - 完整投影验证
- `check_projection_headless.py` - 单帧投影测试
- `show_dataset_summary.py` - 数据集摘要

**快速使用**:
```bash
# 所有验证功能已整合到 validate_dataset.py
python tools/validation/validate_dataset.py --help
```

[查看详细文档 →](validation/README.md)

---

### 🎨 可视化工具 (`visualization/`)

数据可视化和图像生成工具。

**主要工具**:
- `visualize_projection.py` - 交互式投影可视化
- `view_pointcloud.py` - 点云查看器
- `visualize_kitti_structure.py` - KITTI数据结构可视化
- `batch_generate_projections.py` - 批量生成投影图

**快速使用**:
```bash
# 交互式查看点云投影
python tools/visualization/visualize_projection.py \
    --dataset_root /path/to/dataset --sequence 00
```

[查看详细文档 →](visualization/README.md)

---

### 📈 分析工具 (`analysis/`)

训练数据分析和统计工具。

**主要工具**:
- `analyze_perturbation_training.py` - 扰动训练效果分析

[查看详细文档 →](analysis/README.md)

---

### 🔧 修复与调试工具 (`utils/`)

数据修复和问题调试工具。

**主要工具**:
- `fix_calib_tr_inversion.py` - 修复Tr矩阵反向问题
- `debug_undistortion.py` - 调试点云去畸变算法

**使用场景**:
- 修复标定矩阵格式问题
- 对比C++/Python去畸变实现
- 诊断数据质量问题

[查看详细文档 →](utils/README.md)

---

### 🔄 Shell 脚本 (`scripts/`)

批处理管理和监控脚本。

**主要脚本**:
- `monitor_batch_processing.sh` - 监控批处理任务
- `stop_batch_processing.sh` - 停止批处理任务

[查看详细文档 →](scripts/README.md)

---

## 📋 常见工作流

### 工作流1: 准备新数据集

```bash
# 1. 准备数据
python tools/preparation/prepare_custom_dataset.py \
    --source raw_data/ --output dataset/

# 2. 验证数据集
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation/ --full

# 3. 查看验证报告
cat validation/VALIDATION_SUMMARY.md
cat validation/projection_validation/PROJECTION_VALIDATION_REPORT.md
```

### 工作流2: 日常数据检查

```bash
# 快速摘要
python tools/validation/validate_dataset.py summary dataset/

# 快速验证
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation_quick/
```

### 工作流3: 投影质量分析

```bash
# 1. 生成完整投影验证
python tools/validation/validate_dataset.py projection-full dataset/ \
    --output-dir projections/

# 2. 交互式查看特定序列
python tools/visualization/visualize_projection.py \
    --dataset_root dataset/ --sequence 00
```

### 工作流4: 问题诊断

```bash
# 1. 检查Tr矩阵
python tools/validation/verify_dataset_tr_fix.py --dataset_root dataset/

# 2. 测试单帧投影
python tools/validation/check_projection_headless.py \
    --dataset_root dataset/ --sequence 00 --frame 0 \
    --output test_projection.png

# 3. 如发现问题，使用修复工具
python tools/utils/fix_calib_tr_inversion.py --dataset_root dataset/
```

---

## 🎯 推荐最佳实践

### 1. 首次使用新数据集

```bash
# Step 1: 快速摘要（了解数据集概况）
python tools/validation/validate_dataset.py summary dataset/

# Step 2: 完整验证（确保数据质量）
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation_complete/ --full

# Step 3: 查看报告，确认无问题
cat validation_complete/VALIDATION_SUMMARY.md
```

### 2. 日常开发验证

```bash
# 快速模式即可（17秒）
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation_quick/
```

### 3. 发布前检查

```bash
# 运行完整验证
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation_release/ --full
```

---

## ❓ 常见问题

### Q1: 应该使用哪个验证命令？

- **日常检查**: `validate_dataset.py full dataset/` (快速模式，17秒)
- **首次验证**: `validate_dataset.py full dataset/ --full` (完整模式，15分钟)
- **仅看投影**: `validate_dataset.py projection-full dataset/` (10分钟)

详见 [VALIDATION_MODES.md](docs/VALIDATION_MODES.md)

### Q2: 为什么要分目录组织？

- **便于查找**: 按功能分类，快速定位工具
- **降低复杂度**: 每个目录职责单一，易于理解
- **便于维护**: 相关工具集中管理，减少耦合
- **模块化**: 各工具独立开发和测试

### Q3: 如何从旧路径迁移？

旧路径 → 新路径：
```bash
# 验证工具
tools/validate_dataset.py              → tools/validation/validate_dataset.py
tools/validate_kitti_odometry.py       → tools/validation/validate_kitti_odometry.py

# 可视化工具
tools/visualize_projection.py          → tools/visualization/visualize_projection.py
tools/view_pointcloud.py                → tools/visualization/view_pointcloud.py

# 数据准备
tools/prepare_custom_dataset.py        → tools/preparation/prepare_custom_dataset.py

# 工具函数
tools/fix_calib_tr_inversion.py        → tools/utils/fix_calib_tr_inversion.py
```

### Q4: 在哪里查看各工具的详细用法？

每个子目录都有独立的 `README.md`，包含详细的工具说明和使用示例。

---

## 🤝 贡献指南

添加新工具时，请：

1. 选择合适的分类目录
2. 添加脚本文档字符串
3. 更新对应目录的 README.md
4. 如需要，添加使用示例到主 README

---

## 📞 问题反馈

如遇到问题或有改进建议，请联系维护团队。

---

**最后更新**: 2026-03-01  
**维护者**: BEVCalib Team
