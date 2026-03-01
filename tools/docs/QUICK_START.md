# 🚀 BEVCalib 工具快速开始

## 一分钟快速验证

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 1. 查看数据集摘要（5秒）
python tools/validate_dataset.py summary /path/to/dataset

# 2. 快速验证（1分钟）
python tools/validate_dataset.py full /path/to/dataset \
    --output-dir quick_validation

# 3. 查看结果
cat quick_validation/VALIDATION_SUMMARY.md
```

---

## 常用命令速查

### 📊 数据集摘要
```bash
python tools/validate_dataset.py summary /path/to/dataset
```

### ✅ 格式验证
```bash
# 单个序列
python tools/validate_dataset.py format /path/to/dataset --sequence 00

# 所有序列
python tools/validate_dataset.py format /path/to/dataset --all
```

### 🎯 投影测试
```bash
# 单帧测试
python tools/validate_dataset.py projection /path/to/dataset \
    --sequence 00 --frame 0 --output test.png

# 完整投影验证（每序列5帧）
python tools/validate_dataset.py projection-full /path/to/dataset \
    --output-dir validation/projections
```

### 🔍 Tr矩阵验证
```bash
python tools/validate_dataset.py tr /path/to/dataset
```

### 📋 完整验证
```bash
# 快速验证（前3个序列）
python tools/validate_dataset.py full /path/to/dataset \
    --output-dir validation

# 完整验证（所有序列）
python tools/validate_dataset.py full /path/to/dataset \
    --output-dir validation --full
```

---

## 实际使用示例

### 场景1: 新数据集首次验证

```bash
# 步骤1: 快速摘要
python tools/validate_dataset.py summary \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data

# 步骤2: 完整验证
python tools/validate_dataset.py full \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --output-dir validation_results

# 步骤3: 查看报告
cat validation_results/VALIDATION_SUMMARY.md
```

### 场景2: 快速检查某个序列

```bash
# 检查序列05
python tools/validate_dataset.py format \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --sequence 05

# 测试投影
python tools/validate_dataset.py projection \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --sequence 05 --frame 0 --output seq05_test.png
```

### 场景3: 验证投影质量

```bash
# 完整投影验证（所有序列，每序列5帧）
python tools/validate_dataset.py projection-full \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --output-dir projection_validation

# 查看结果
ls -lh projection_validation/sequence_00/
cat projection_validation/PROJECTION_VALIDATION_REPORT.md
```

---

## 输出文件说明

### 完整验证输出

```
validation_results/
├── VALIDATION_SUMMARY.md        # 📋 主报告（从这里开始）
├── validation_summary.json      # JSON格式结果
├── tr_matrix_validation.log     # Tr矩阵验证日志
├── format_seq00.log             # 序列格式验证日志
├── format_seq01.log
├── ...
└── sample_projections/          # 投影测试图像
    ├── seq00_frame000000.png
    ├── seq01_frame000000.png
    └── ...
```

### 投影验证输出

```
projection_validation/
├── PROJECTION_VALIDATION_REPORT.md  # 📊 投影报告
├── summary.json                     # JSON汇总
├── sequence_00/                     # 序列00结果
│   ├── frame_000000.png            # 5张投影图
│   ├── frame_000386.png
│   ├── frame_000772.png
│   ├── frame_001158.png
│   ├── frame_001543.png
│   └── statistics.json             # 统计信息
├── sequence_01/
└── ...
```

---

## 快速诊断流程

### 问题：数据集无法训练

```bash
# 1. 检查基本信息
python tools/validate_dataset.py summary /path/to/dataset

# 2. 验证格式
python tools/validate_dataset.py format /path/to/dataset --sequence 00

# 3. 检查Tr矩阵
python tools/validate_dataset.py tr /path/to/dataset

# 4. 测试投影
python tools/validate_dataset.py projection /path/to/dataset \
    --sequence 00 --frame 0 --output debug.png
```

### 问题：投影不对齐

```bash
# 1. 验证Tr矩阵
python tools/validate_dataset.py tr /path/to/dataset

# 2. 查看投影图像
python tools/validate_dataset.py projection /path/to/dataset \
    --sequence 00 --frame 0 --output test.png

# 3. 多帧测试
python tools/validate_dataset.py projection-full /path/to/dataset \
    --output-dir projection_test --sequences 00
```

---

## 性能提示

| 命令 | 耗时 | 说明 |
|------|------|------|
| `summary` | ~5秒 | 快速统计 |
| `format` (单序列) | ~1秒 | 格式检查 |
| `tr` | ~3秒 | Tr矩阵验证 |
| `projection` (单帧) | ~5秒 | 单帧投影测试 |
| `projection-full` | ~10分钟 | 60帧投影（12序列×5帧） |
| `full` (快速) | ~2分钟 | 验证前3个序列 |
| `full --full` | ~15分钟 | 验证所有序列 |

---

## 常见错误处理

### 错误1: "Tr矩阵格式错误"

```bash
# 检查Tr矩阵
python tools/validate_dataset.py tr /path/to/dataset

# 如果是旧数据集（2025-02-04之前），运行修复
python tools/fix_calib_tr_inversion.py --dataset_root /path/to/dataset
```

### 错误2: "数据不对齐"

```bash
# 查看详细信息
python tools/validate_dataset.py format /path/to/dataset --sequence XX

# 检查文件数量
ls sequences/XX/image_2/ | wc -l
ls sequences/XX/velodyne/ | wc -l
```

### 错误3: "投影失败"

```bash
# 查看详细错误
python tools/check_projection_headless.py \
    --dataset_root /path/to/dataset \
    --sequence 00 --frame 0 --output debug.png
```

---

## 更多帮助

```bash
# 查看完整文档
cat tools/README.md

# 查看命令帮助
python tools/validate_dataset.py --help
python tools/validate_dataset.py full --help
```

---

**提示**: 首次验证新数据集，建议直接运行：
```bash
python tools/validate_dataset.py full /path/to/dataset \
    --output-dir validation_results
```
