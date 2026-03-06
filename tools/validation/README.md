# 验证工具 (Validation Tools)

数据集质量验证和检查工具集。

---

## 📋 工具列表

### ⭐ `validate_dataset.py` - 统一验证入口（推荐）

整合所有验证功能的统一入口工具。

**功能**:
- 数据集摘要
- KITTI格式验证
- Tr矩阵验证
- 投影效果验证
- 生成完整验证报告

**使用方法**:

```bash
# 查看帮助
python tools/validation/validate_dataset.py --help

# 数据集摘要
python tools/validation/validate_dataset.py summary /path/to/dataset

# ===== 两种验证模式（二选一） =====

# 快速验证 (~17秒，前3序列，各1帧投影)
python tools/validation/validate_dataset.py quick /path/to/dataset \
    --output-dir validation_results/

# 完整验证 (~15分钟，所有序列 + 完整投影)
python tools/validation/validate_dataset.py full /path/to/dataset \
    --output-dir validation_results/

# ===== 单项验证 =====

# 验证单个序列格式
python tools/validation/validate_dataset.py format /path/to/dataset --sequence 00

# 验证所有序列格式
python tools/validation/validate_dataset.py format /path/to/dataset --all

# 验证Tr矩阵
python tools/validation/validate_dataset.py tr /path/to/dataset

# 单帧投影测试
python tools/validation/validate_dataset.py projection /path/to/dataset \
    --sequence 00 --frame 0 --output test.png

# 多序列投影验证（每序列多帧采样）
python tools/validation/validate_dataset.py projection-full /path/to/dataset \
    --output-dir projections/
```

**验证模式对比**:

| 命令 | 耗时 | 格式验证范围 | 投影覆盖 | 适用场景 |
|------|------|-------------|----------|----------|
| `quick` | ~17秒 | 前3序列 | 前3序列，各1帧 | 日常检查 |
| `full` | ~15分钟 | 所有序列 | 所有序列，各5帧 | 首次验证、发布前 |
| `projection-full` | ~10分钟 | — | 所有序列，各10帧 | 仅投影分析 |

详见 [../docs/VALIDATION_MODES.md](../docs/VALIDATION_MODES.md)

---

### `validate_kitti_odometry.py` - KITTI格式验证

验证数据集是否符合KITTI-Odometry格式规范。

**检查项**:
- 目录结构完整性
- 标定文件格式（P0-P3, Tr）
- 位姿文件格式
- 图像文件命名和格式
- 点云文件命名和格式
- 数据对齐性（图像/点云/位姿数量一致）

**使用方法**:
```bash
python tools/validation/validate_kitti_odometry.py \
    --dataset_root /path/to/dataset \
    --sequence 00
```

**输出示例**:
```
✅ 序列00验证通过
  - 22/22 项检查通过
  - 帧数: 1544
  - 标定: 正常
  - 位姿: 正常
```

---

### `verify_dataset_tr_fix.py` - Tr矩阵验证

专门验证标定文件中的 Tr 矩阵（Velodyne → Camera）。

**检查项**:
- Tr矩阵格式（3x4）
- 旋转矩阵正交性
- 旋转矩阵行列式（应≈1.0）
- 位移向量合理性

**使用方法**:
```bash
python tools/validation/verify_dataset_tr_fix.py \
    --dataset_root /path/to/dataset
```

**输出示例**:
```
序列 00:
  ✅ Tr矩阵格式正确
  ✅ 旋转矩阵正交性: OK (误差 < 0.001)
  ✅ 行列式: 1.000 (OK)
  ✅ 位移向量: [-0.02, -0.06, -0.33] (合理)
```

---

### `comprehensive_projection_validation.py` - 完整投影验证

对所有序列进行采样投影验证，生成详细报告。

**特性**:
- 每序列采样10帧（均匀分布）
- 自动生成投影可视化图
- 统计投影质量指标
- 按序列分类存储结果

**使用方法**:
```bash
python tools/validation/comprehensive_projection_validation.py \
    --dataset_root /path/to/dataset \
    --output_dir projections/
```

**输出结构**:
```
projections/
├── PROJECTION_VALIDATION_REPORT.md    # 详细报告
├── summary.json                        # 统计数据
├── sequence_00/
│   ├── frame_000000.png               # 10张投影图
│   ├── frame_000171.png
│   ├── ...
│   └── statistics.json                 # 该序列统计
├── sequence_01/
└── ...
```

**报告内容**:
- 每序列投影统计（可见率、深度范围）
- 投影质量评估
- 异常帧检测

---

### `check_projection_headless.py` - 单帧投影测试

在无头环境中测试单帧点云投影效果。

**特性**:
- 无需GUI，适合服务器环境
- 生成带深度着色的投影图
- 输出投影统计信息

**使用方法**:
```bash
python tools/validation/check_projection_headless.py \
    --dataset_root /path/to/dataset \
    --sequence 00 \
    --frame 0 \
    --output projection_test.png
```

**输出**:
- PNG投影图像
- 投影统计（总点数、可见点数、可见率、深度范围）

---

### `show_dataset_summary.py` - 数据集摘要

快速显示数据集的基本统计信息。

**显示内容**:
- 序列数量和编号
- 各序列帧数
- 数据对齐状态
- Tr矩阵状态

**使用方法**:
```bash
python tools/validation/show_dataset_summary.py \
    --dataset_root /path/to/dataset
```

**输出示例**:
```
📊 数据集摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

序列信息:
  • 总序列数: 12
  • 总帧数: 67,032

各序列详情:
  00: 1,544 帧 ✅
  01: 9,000 帧 ✅
  ...
```

---

## 🎯 推荐使用方式

### 场景1: 首次验证新数据集

```bash
# 1. 快速摘要
python tools/validation/validate_dataset.py summary /path/to/dataset

# 2. 完整验证（~15分钟）
python tools/validation/validate_dataset.py full /path/to/dataset \
    --output-dir validation/

# 3. 查看报告
cat validation/VALIDATION_SUMMARY.md
cat validation/projection_validation/PROJECTION_VALIDATION_REPORT.md
```

### 场景2: 日常快速检查

```bash
# 快速验证（~17秒）
python tools/validation/validate_dataset.py quick /path/to/dataset \
    --output-dir validation_quick/
```

### 场景3: 仅验证投影质量

```bash
# 完整投影验证（~10分钟）
python tools/validation/validate_dataset.py projection-full /path/to/dataset \
    --output-dir projections/
```

### 场景4: 诊断特定问题

```bash
# 检查Tr矩阵
python tools/validation/verify_dataset_tr_fix.py --dataset_root /path/to/dataset

# 测试特定帧投影
python tools/validation/check_projection_headless.py \
    --dataset_root /path/to/dataset --sequence 00 --frame 100 \
    --output debug.png
```

---

## 📊 验证结果解读

### KITTI格式验证结果

**通过条件**:
- ✅ 所有22项检查通过
- ✅ 图像、点云、位姿数量一致
- ✅ 文件命名连续无缺失

**常见问题**:
- ❌ 文件数量不匹配 → 检查数据准备流程
- ❌ Tr矩阵格式错误 → 使用 `../utils/fix_calib_tr_inversion.py` 修复
- ❌ 位姿文件格式错误 → 检查位姿矩阵维度

### Tr矩阵验证结果

**健康指标**:
- 旋转矩阵正交性误差 < 0.01
- 行列式 ∈ [0.99, 1.01]
- 位移向量 < 5.0m（典型值）

**问题诊断**:
- 行列式 ≈ -1.0 → 坐标系反向，需修复
- 正交性误差 > 0.1 → 矩阵损坏或格式错误
- 位移向量异常大 → 可能是单位错误或配置问题

### 投影质量评估

**正常范围**:
- 可见率: 10% - 40%（取决于场景）
- 深度范围: 3m - 200m
- 每帧可见点: 5,000 - 30,000

**异常情况**:
- 可见率 < 5% → 标定可能错误
- 可见率 > 60% → 检查投影逻辑
- 深度异常 → 点云坐标系问题

---

## ⚠️ 注意事项

1. **首次验证推荐使用完整模式**
   ```bash
   python tools/validation/validate_dataset.py full dataset/ --output-dir results/
   ```

2. **无头环境**
   - 所有投影工具已支持无头模式
   - 自动使用 `matplotlib.use('Agg')`

3. **大数据集验证**
   - 完整验证耗时较长（~15分钟）
   - 可先用快速模式检查
   - 投影验证会生成大量PNG图片（每序列10张）

4. **验证结果保存**
   - 建议保留验证报告用于追溯
   - JSON文件可用于自动化分析

---

## 🔗 相关文档

- [主文档](../README.md)
- [验证模式详解](../docs/VALIDATION_MODES.md)
- [快速开始](../docs/QUICK_START.md)
- [架构说明](../docs/ARCHITECTURE.md)

---

**最后更新**: 2026-03-01
