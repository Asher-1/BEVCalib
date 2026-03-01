# BEVCalib 训练脚本使用说明

## 📋 概述

BEVCalib 训练脚本已重构，支持多数据集训练并改进了日志目录结构。

## 🚀 快速开始

### 训练 all_training_data 数据集

```bash
bash start_training.sh all v1
```

### 训练 B26A 数据集

```bash
bash start_training.sh B26A v1
```

## 📂 日志结构

训练日志按数据集分级组织：

```
logs/
├── B26A/
│   ├── model_small_10deg_v1/
│   └── model_small_5deg_v1/
├── all_training_data/
│   ├── model_small_10deg_v1/
│   └── model_small_5deg_v1/
└── README.md
```

## 🔧 可用脚本

### 1. start_training.sh（推荐）

自动配置并启动多GPU训练。

**用法**:
```bash
bash start_training.sh [dataset] [version]

# 数据集选项:
#   B26A   - B26A 数据集
#   all    - all_training_data 数据集
#   custom - 自定义数据集

# 示例:
bash start_training.sh B26A v1
bash start_training.sh all v1
CUSTOM_DATASET=/path/to/data bash start_training.sh custom v1
```

### 2. train_universal.sh

通用训练脚本，支持详细配置。

**用法**:
```bash
bash train_universal.sh [mode] --dataset_root PATH [options]

# 模式:
#   scratch   - 从头训练
#   finetune  - 微调
#   resume    - 恢复训练

# 选项:
#   --dataset_root PATH     数据集路径（必需）
#   --dataset_name NAME     数据集名称（可选）
#   --cuda_device ID        GPU ID
#   --angle_range_deg DEG   旋转扰动（默认20）
#   --trans_range M         平移扰动（默认1.5）
#   --log_suffix SUFFIX     日志后缀

# 示例:
bash train_universal.sh scratch \
  --dataset_root /path/to/all_training_data \
  --cuda_device 0 \
  --angle_range_deg 10 \
  --trans_range 0.5 \
  --log_suffix small_10deg_v1
```

### 3. train_B26A.sh（旧脚本）

向后兼容的 B26A 专用脚本，仍可使用。

## 📊 监控训练

### 查看日志

```bash
# 实时日志
tail -f logs/all_training_data/model_small_10deg_v1/train.log

# GPU 状态
nvidia-smi -l 1

# 训练进程
ps aux | grep train_kitti
```

### TensorBoard

```bash
# 查看所有数据集
tensorboard --logdir logs/ --port 6006

# 查看特定数据集
tensorboard --logdir logs/all_training_data/ --port 6006
```

## 🛑 停止训练

```bash
bash stop_training.sh
```

## 📚 相关文档

- `QUICK_START_TRAINING.md` - 快速开始指南
- `TRAINING_REFACTOR_SUMMARY.md` - 重构总结
- `SCRIPT_CHANGES_SUMMARY.txt` - 变更摘要
- `logs/README.md` - 日志目录说明

## 💡 推荐工作流

1. 快速验证（B26A）
   ```bash
   bash start_training.sh B26A v1
   ```

2. 完整训练（all_training_data）
   ```bash
   bash start_training.sh all v1
   ```

3. 监控训练
   ```bash
   tail -f logs/all_training_data/model_small_10deg_v1/train.log
   tensorboard --logdir logs/all_training_data/ --port 6006
   ```

## ⚠️ 注意事项

1. 新脚本的日志位置: `logs/{dataset_name}/model_*/`
2. 旧脚本仍可用，向后兼容
3. 建议使用新脚本以获得更好的日志组织

---

**更新日期**: 2026-03-01  
**推荐使用**: `start_training.sh` + `train_universal.sh`
