# 🚀 BEVCalib 训练快速开始指南

## 📋 前提条件

### 1. 数据集准备

确保数据集已准备好并符合 KITTI-Odometry 格式：

```bash
# B26A 数据集
/mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data/

# 全量数据集
/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data/

# 数据集结构
dataset_root/
├── sequences/
│   ├── 00/
│   │   ├── image_2/
│   │   ├── velodyne/
│   │   └── calib.txt
│   ├── 01/
│   └── ...
└── poses/
    ├── 00.txt
    ├── 01.txt
    └── ...
```

### 2. 环境准备

```bash
# 激活 conda 环境
conda activate bevcalib

# 检查环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
nvidia-smi
```

## 🎯 最简单的方式：使用 start_training.sh

### B26A 数据集训练

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 启动训练（版本 v1）
bash start_training.sh B26A v1
```

**说明**:
- 自动启动 2 个训练进程（GPU 0 和 GPU 1）
- GPU 0: 10° 扰动，0.5m 平移
- GPU 1: 5° 扰动，0.3m 平移
- 日志位置: `./logs/B26A/model_*_v1/`

### all_training_data 数据集训练

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 启动训练（版本 v1）
bash start_training.sh all v1
```

**说明**:
- 自动启动 2 个训练进程（GPU 0 和 GPU 1）
- GPU 0: 10° 扰动，0.5m 平移
- GPU 1: 5° 扰动，0.3m 平移
- 日志位置: `./logs/all_training_data/model_*_v1/`

### 自定义数据集训练

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 设置数据集路径并启动
CUSTOM_DATASET=/path/to/your/dataset bash start_training.sh custom v1
```

## 📊 查看训练状态

### 实时监控

```bash
# 查看 GPU 使用情况
nvidia-smi -l 1

# 查看训练进程
ps aux | grep train_kitti

# 查看训练日志（B26A 数据集）
tail -f logs/B26A/model_small_10deg_v1/train.log

# 查看训练日志（all_training_data 数据集）
tail -f logs/all_training_data/model_small_10deg_v1/train.log
```

### TensorBoard 可视化

```bash
# 查看所有训练
tensorboard --logdir logs/ --port 6006

# 只查看 B26A 数据集
tensorboard --logdir logs/B26A/ --port 6006

# 只查看 all_training_data 数据集
tensorboard --logdir logs/all_training_data/ --port 6007
```

然后在浏览器中访问: `http://localhost:6006`

## 🛑 停止训练

```bash
# 使用停止脚本
bash stop_training.sh

# 或手动停止
pkill -f train_kitti
```

## 🔧 高级用法：单独训练

### 使用 train_universal.sh

```bash
# 单个 GPU 训练，10° 扰动
bash train_universal.sh scratch \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
  --dataset_name all_training_data \
  --cuda_device 0 \
  --angle_range_deg 10 \
  --trans_range 0.5 \
  --log_suffix small_10deg_v1

# 日志位置: logs/all_training_data/model_small_10deg_v1/
```

### 完整参数说明

```bash
bash train_universal.sh [mode] [options]

# 模式 (mode):
#   scratch   - 从头训练（默认）
#   finetune  - 从 KITTI 预训练模型微调
#   resume    - 从最后的检查点恢复

# 选项 (options):
#   --dataset_root PATH     - 数据集根目录（必需）
#   --dataset_name NAME     - 数据集名称（可选，自动检测）
#   --cuda_device ID        - CUDA 设备 ID（如 0, 1, 2）
#   --tensorboard_port PORT - TensorBoard 端口（默认 6006）
#   --log_suffix SUFFIX     - 日志目录后缀
#   --angle_range_deg DEG   - 旋转扰动范围（默认 20）
#   --trans_range M         - 平移扰动范围（默认 1.5）
```

### 示例：不同扰动级别

```bash
# 小扰动 (5°, 0.3m) - 适合已标定数据的微调
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 0 \
  --angle_range_deg 5 \
  --trans_range 0.3 \
  --log_suffix small_5deg_v1

# 中等扰动 (10°, 0.5m) - 推荐用于初始标定
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 1 \
  --angle_range_deg 10 \
  --trans_range 0.5 \
  --log_suffix medium_10deg_v1

# 大扰动 (20°, 1.5m) - 适合大误差标定
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 2 \
  --angle_range_deg 20 \
  --trans_range 1.5 \
  --log_suffix large_20deg_v1
```

## 📂 日志目录结构

训练后，日志按数据集分级组织：

```
logs/
├── B26A/                           # B26A 数据集
│   ├── model_small_10deg_v1/
│   │   ├── train.log               # 训练日志
│   │   ├── events.out.tfevents.*   # TensorBoard 事件
│   │   ├── epoch_40.pth            # 检查点
│   │   ├── epoch_80.pth
│   │   └── ...
│   └── model_small_5deg_v1/
│       └── ...
│
├── all_training_data/              # 全量数据集
│   ├── model_small_10deg_v1/
│   └── model_small_5deg_v1/
│
└── README.md
```

## 🎓 典型工作流

### 1. 快速验证（B26A 小数据集）

```bash
# 使用小数据集快速验证
bash start_training.sh B26A v1

# 查看训练进度
tail -f logs/B26A/model_small_10deg_v1/train.log

# 训练 50-100 个 epoch 后检查结果
tensorboard --logdir logs/B26A/ --port 6006
```

### 2. 完整训练（all_training_data 全量数据）

```bash
# 使用全量数据集训练最终模型
bash start_training.sh all v1

# 监控训练（需要更长时间）
tail -f logs/all_training_data/model_small_10deg_v1/train.log

# 训练 200-500 个 epoch
tensorboard --logdir logs/all_training_data/ --port 6007
```

### 3. 对比实验

```bash
# 同时训练两个数据集对比
bash start_training.sh B26A v1      # 后台运行
bash start_training.sh all v2       # 后台运行

# TensorBoard 同时查看
tensorboard --logdir logs/ --port 6006
# 在浏览器中可以按数据集筛选对比
```

## 🔍 常见问题

### Q1: 如何选择扰动级别？

**答**: 根据初始标定误差选择
- **小扰动 (5°, 0.3m)**: 标定误差 < 5°
- **中等扰动 (10°, 0.5m)**: 标定误差 5-10°（推荐）
- **大扰动 (20°, 1.5m)**: 标定误差 > 10°

### Q2: 训练需要多长时间？

**答**: 取决于数据集大小和 GPU
- B26A (3178 帧): ~2-4 小时 / 100 epoch (单 GPU)
- all_training_data: 需要查看具体帧数

### Q3: 如何恢复中断的训练？

**答**: 使用 resume 模式
```bash
bash train_universal.sh resume \
  --dataset_root /path/to/data \
  --dataset_name all_training_data \
  --cuda_device 0
```

### Q4: 如何修改批次大小？

**答**: 目前需要修改脚本中的 `--batch_size` 参数
```bash
# 编辑 train_universal.sh
# 将 --batch_size 16 改为其他值
```

### Q5: 日志目录占用空间太大怎么办？

**答**: 定期清理和归档
```bash
# 归档旧日志
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/B26A/

# 删除原始日志
rm -rf logs/B26A/
```

## 📝 训练检查清单

### 训练前

- [ ] 数据集准备完成，格式正确
- [ ] Conda 环境已激活 (`bevcalib`)
- [ ] GPU 可用且显存充足 (`nvidia-smi`)
- [ ] bev_pool CUDA 扩展已编译
- [ ] 确定训练配置（数据集、扰动级别、版本号）

### 训练中

- [ ] 训练进程正常运行 (`ps aux | grep train_kitti`)
- [ ] GPU 利用率正常 (`nvidia-smi`)
- [ ] 日志正常写入 (`tail -f logs/.../train.log`)
- [ ] Loss 正常下降（TensorBoard）
- [ ] 无异常错误信息

### 训练后

- [ ] 检查点文件已保存（`*.pth`）
- [ ] TensorBoard 曲线正常
- [ ] 评估模型性能
- [ ] 备份重要模型

## 🎯 快速命令备忘

```bash
# ============ 启动训练 ============
# B26A 数据集
bash start_training.sh B26A v1

# all_training_data 数据集
bash start_training.sh all v1

# ============ 监控训练 ============
# GPU 状态
nvidia-smi -l 1

# 训练进程
ps aux | grep train_kitti

# 实时日志
tail -f logs/all_training_data/model_small_10deg_v1/train.log

# TensorBoard
tensorboard --logdir logs/all_training_data/ --port 6006

# ============ 停止训练 ============
bash stop_training.sh
# 或
pkill -f train_kitti

# ============ 检查点管理 ============
# 查找最新检查点
find logs/all_training_data/model_small_10deg_v1/ -name "*.pth" | sort -V | tail -1

# 列出所有检查点
ls -lh logs/all_training_data/model_small_10deg_v1/*.pth
```

## 📚 相关文档

- `TRAINING_SCRIPT_MIGRATION.md` - 脚本重构详细说明
- `train_universal.sh` - 通用训练脚本
- `start_training.sh` - 启动脚本
- `logs/README.md` - 日志目录说明

## 💡 最佳实践

1. **从小数据集开始**: 先用 B26A 验证配置正确
2. **版本管理**: 使用有意义的版本号（如 `v1_baseline`, `v2_tuned`）
3. **定期检查**: 每 50-100 epoch 查看一次 TensorBoard
4. **保存重要模型**: 及时备份表现好的检查点
5. **日志归档**: 定期清理和归档旧日志

---

**创建日期**: 2026-03-01  
**适用版本**: train_universal.sh + start_training.sh  
**推荐工作流**: start_training.sh → 监控 → TensorBoard
