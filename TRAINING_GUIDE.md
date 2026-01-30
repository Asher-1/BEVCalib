# BEVCalib 训练指南

本文档提供详细的训练参数说明和建议。

---

## 📋 快速开始

### KITTI 数据集训练
```bash
python kitti-bev-calib/train_kitti.py \
    --log_dir ./logs/kitti \
    --dataset_root /path/to/kitti-odometry \
    --batch_size 16 \
    --num_epochs 500
```

### 自定义数据集训练
```bash
python kitti-bev-calib/train_kitti.py \
    --log_dir ./logs/custom_model \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --batch_size 4 \
    --num_epochs 100
```

---

## 🎯 训练参数详解

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset_root` | 数据集根目录 | `/path/to/dataset` |
| `--log_dir` | 日志输出目录 | `./logs/my_model` |

### 核心训练参数

| 参数 | 默认值 | KITTI 推荐 | 自定义数据集推荐 | 说明 |
|------|--------|-----------|----------------|------|
| `--batch_size` | 16 | 16 | 4-8 | 批大小 |
| `--num_epochs` | 500 | 500 | 100-200 | 训练轮数 |
| `--lr` | 1e-4 | 1e-4 | 1e-4 / 5e-5 | 学习率 |
| `--scheduler` | 0 | 1 | 1 | 学习率调度器 |
| `--step_size` | 80 | 80 | 40-80 | 学习率衰减步长 |

### 模型架构参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--bev_encoder` | 1 | 1 | 使用 BEV 编码器 |
| `--deformable` | 0 | 0 | 可变形注意力 |
| `--xyz_only` | 1 | 1 | 只使用 XYZ 坐标（不使用强度） |

### 标定参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--angle_range_deg` | 20.0 | 旋转扰动范围（度）|
| `--trans_range` | 1.5 | 平移扰动范围（米）|
| `--label` | - | 实验标签 |

### 保存参数

| 参数 | 默认值 | KITTI 推荐 | 自定义推荐 | 说明 |
|------|--------|-----------|----------|------|
| `--save_ckpt_per_epoches` | 40 | 40 | 20 | 保存检查点间隔 |
| `--pretrain_ckpt` | None | - | `./ckpt/kitti.pth` | 预训练模型路径 |

---

## 📊 训练配置推荐

### 场景 1: 小数据集（< 1000 帧）

```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /path/to/small_dataset \
    --log_dir ./logs/small_model \
    --batch_size 4 \
    --num_epochs 100 \
    --save_ckpt_per_epoches 10 \
    --lr 1e-4 \
    --scheduler 1 \
    --step_size 30
```

**特点**：
- 小批量（避免过拟合）
- 较少轮数（数据量小）
- 频繁保存（监控训练）

### 场景 2: 中等数据集（1000-5000 帧）

```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --log_dir ./logs/medium_model \
    --batch_size 8 \
    --num_epochs 200 \
    --save_ckpt_per_epoches 20 \
    --lr 1e-4 \
    --scheduler 1 \
    --step_size 60 \
    --angle_range_deg 20 \
    --trans_range 1.5 \
    --label B26A_20_1.5
```

**特点**：
- 适中批量
- 充足训练轮数
- 标准学习率调度

### 场景 3: 大数据集（> 5000 帧）

```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /path/to/large_dataset \
    --log_dir ./logs/large_model \
    --batch_size 16 \
    --num_epochs 500 \
    --save_ckpt_per_epoches 40 \
    --lr 1e-4 \
    --scheduler 1 \
    --step_size 80
```

**特点**：
- 大批量（加速训练）
- 更多轮数（充分学习）
- KITTI 标准配置

### 场景 4: 从 KITTI 微调

```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --log_dir ./logs/finetuned_model \
    --pretrain_ckpt ./ckpt/kitti.pth \
    --batch_size 4 \
    --num_epochs 50 \
    --save_ckpt_per_epoches 10 \
    --lr 5e-5 \
    --scheduler 1 \
    --step_size 20
```

**特点**：
- 加载预训练权重
- 较小学习率（微调）
- 较少轮数（快速适应）

---

## 🔧 参数调优指南

### 学习率（`--lr`）

**建议值**：
- 从头训练：`1e-4`
- 微调：`5e-5` 或 `1e-5`

**调整策略**：
- 损失震荡：降低学习率
- 损失下降慢：增大学习率
- 使用 `--scheduler 1` 自动衰减

### 批大小（`--batch_size`）

**建议值**：
- 16GB GPU：4-8
- 24GB GPU：8-16
- 多 GPU：16-32

**影响**：
- 大批量：训练快，但可能欠拟合
- 小批量：泛化好，但训练慢

### 训练轮数（`--num_epochs`）

**建议值**：
- 小数据集：50-100
- 中等数据集：100-200
- 大数据集：200-500

**判断标准**：
- 验证损失不再下降时停止
- 使用早停策略

### 扰动范围

**旋转（`--angle_range_deg`）**：
- 低噪声：10-15 度
- 标准：20 度
- 高噪声：25-30 度

**平移（`--trans_range`）**：
- 低噪声：0.5-1.0 米
- 标准：1.5 米
- 高噪声：2.0-3.0 米

---

## 📈 训练监控

### TensorBoard

```bash
tensorboard --logdir ./logs
```

访问：http://localhost:6006

**关键指标**：
- `train/loss`：训练损失
- `train/rotation_error`：旋转误差
- `train/translation_error`：平移误差

### WandB（可选）

如果启用了 WandB：
```bash
wandb login
# 训练时会自动上传
```

---

## 🎯 实际案例

### 案例 1: B26A 车载数据集

**数据集信息**：
- 路径：`/home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data`
- 序列：自动检测
- 场景：车载 LiDAR-Camera 标定

**训练命令**：
```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --log_dir ./logs/B26A_model \
    --label B26A_20_1.5 \
    --batch_size 8 \
    --num_epochs 150 \
    --save_ckpt_per_epoches 15 \
    --angle_range_deg 20 \
    --trans_range 1.5 \
    --bev_encoder 1 \
    --deformable 0 \
    --xyz_only 1 \
    --scheduler 1 \
    --lr 1e-4 \
    --step_size 50
```

**微调版本**（使用 KITTI 预训练）：
```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --log_dir ./logs/B26A_finetuned \
    --pretrain_ckpt ./ckpt/kitti.pth \
    --label B26A_finetuned \
    --batch_size 4 \
    --num_epochs 50 \
    --save_ckpt_per_epoches 10 \
    --lr 5e-5 \
    --scheduler 1 \
    --step_size 20
```

### 案例 2: 多序列数据集

**数据集结构**：
```
dataset/
└── sequences/
    ├── 00/  # 1000 帧
    ├── 01/  # 800 帧
    └── 02/  # 1200 帧
```

**训练命令**：
```bash
python kitti-bev-calib/train_kitti.py \
    --dataset_root /path/to/multi_sequence_dataset \
    --log_dir ./logs/multi_seq_model \
    --batch_size 8 \
    --num_epochs 200 \
    --save_ckpt_per_epoches 20
```

---

## ⚠️ 常见问题

### Q1: CUDA Out of Memory

**解决方案**：
1. 减小 `--batch_size`（如 16 → 8 → 4）
2. 减小图像尺寸（修改数据集）
3. 使用梯度累积

### Q2: 训练损失不下降

**可能原因**：
1. 学习率过大或过小
2. 数据集问题（标定不准）
3. 批量过大

**解决方案**：
1. 调整 `--lr`（尝试 5e-5 或 2e-4）
2. 检查数据集质量
3. 减小 `--batch_size`

### Q3: 过拟合

**现象**：训练损失低，但验证损失高

**解决方案**：
1. 增加数据（更多序列）
2. 减少训练轮数
3. 增加扰动范围
4. 使用预训练模型微调

### Q4: 训练太慢

**解决方案**：
1. 增大 `--batch_size`
2. 使用多 GPU
3. 减少数据增强
4. 降低图像分辨率

---

## 📝 训练检查清单

开始训练前，确认：

- [ ] 数据集路径正确
- [ ] 数据集格式符合 KITTI 标准
- [ ] 已检查数据质量（使用 `validate_kitti_odometry.py`）
- [ ] GPU 内存足够（根据 batch_size）
- [ ] 日志目录已创建
- [ ] 选择合适的超参数
- [ ] （可选）下载了预训练模型

---

## 🔗 相关文档

- [数据集准备](README.md#custom-dataset)
- [自定义数据集训练](CUSTOM_DATASET_TRAINING.md)
- [KITTI 数据集结构](README.md#kitti-odometry)

---

**更新时间**: 2026-01-28  
**版本**: v1.0
