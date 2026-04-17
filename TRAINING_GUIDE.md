# 🚀 BEVCalib 训练完整指南

完整的BEVCalib模型训练指南，涵盖数据准备、训练启动、参数调优、监控和最佳实践。

## 📋 目录

- [前提条件](#前提条件)
- [快速开始](#快速开始)
- [训练脚本说明](#训练脚本说明)
- [高级用法](#高级用法)
- [监控训练](#监控训练)
- [日志结构](#日志结构)
- [典型工作流](#典型工作流)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)

## 前提条件

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate bevcalib

# 检查环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
nvidia-smi

# 确保 bev_pool 扩展已编译
cd kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

### 2. 数据集准备

确保数据集符合 KITTI-Odometry 格式：

```
dataset_root/
├── sequences/
│   ├── 00/
│   │   ├── image_2/       # 左相机图像
│   │   ├── velodyne/      # 点云文件
│   │   └── calib.txt      # 标定文件
│   ├── 01/
│   └── ...
└── poses/
    ├── 00.txt             # 位姿文件
    ├── 01.txt
    └── ...
```

**可用数据集**:
- `B26A`: `/mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data/` (小数据集，快速验证)
- `all_training_data`: `/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data/` (完整训练)
- 自定义数据集: 参考 [PREPARE_CUSTOM_DATASET.md](PREPARE_CUSTOM_DATASET.md)

## 快速开始

### 最简单的方式：使用 start_training.sh

#### 训练 B26A 数据集

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

#### 训练 all_training_data 数据集

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

#### 训练自定义数据集

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 设置数据集路径并启动
CUSTOM_DATASET=/path/to/your/dataset bash start_training.sh custom v1
```

## 训练脚本说明

### 1. start_training.sh（推荐）

快速启动脚本，自动配置并启动多GPU训练。

**用法**:
```bash
bash start_training.sh [dataset] [version] [--ddp] [--lr LR]

# 数据集选项:
#   B26A   - B26A 数据集
#   all    - all_training_data 数据集
#   custom - 自定义数据集（需设置 CUSTOM_DATASET 环境变量）

# 选项:
#   --ddp           - 使用分布式数据并行（DDP）模式
#   --lr LR         - 设置初始学习率（默认：2e-4 for scratch）

# 示例:
bash start_training.sh B26A v1
bash start_training.sh all v1 --ddp
bash start_training.sh B26A v2 --lr 0.0001
CUSTOM_DATASET=/path/to/data bash start_training.sh custom v1
```

**特性**:
- ✅ 自动选择可用GPU
- ✅ 自动配置数据集路径
- ✅ 支持DDP分布式训练
- ✅ 自定义学习率
- ✅ 后台运行，日志重定向

### 2. train_universal.sh

通用训练脚本，支持详细配置和多种训练模式。

**用法**:
```bash
bash train_universal.sh [mode] --dataset_root PATH [options]

# 模式:
#   scratch   - 从头训练（默认）
#   finetune  - 从 KITTI 预训练模型微调
#   resume    - 从最后的检查点恢复

# 必需参数:
#   --dataset_root PATH     - 数据集根目录

# 可选参数:
#   --dataset_name NAME     - 数据集名称（自动检测）
#   --cuda_device ID        - CUDA 设备 ID（如 0, 1, 2）
#   --angle_range_deg DEG   - 旋转扰动范围（默认 20）
#   --trans_range M         - 平移扰动范围（默认 1.5）
#   --log_suffix SUFFIX     - 日志目录后缀
#   --tensorboard_port PORT - TensorBoard 端口（默认 6006）
#   --learning_rate LR      - 初始学习率
```

**示例**:

```bash
# 1. 从头训练，10° 扰动
bash train_universal.sh scratch \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
  --cuda_device 0 \
  --angle_range_deg 10 \
  --trans_range 0.5 \
  --log_suffix small_10deg_v1

# 2. 微调KITTI预训练模型
bash train_universal.sh finetune \
  --dataset_root /path/to/custom_data \
  --cuda_device 1 \
  --angle_range_deg 5 \
  --trans_range 0.3

# 3. 恢复训练
bash train_universal.sh resume \
  --dataset_root /path/to/data \
  --cuda_device 0

# 4. 自定义学习率
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 0 \
  --learning_rate 0.0001
```

### 3. batch_train.sh

**配置文件驱动的批量训练脚本**，支持一次性启动多组对比实验。

**用法**:
```bash
bash batch_train.sh [config_file]

# 使用默认配置（5deg Z分辨率对比）
bash batch_train.sh

# 使用指定配置
bash batch_train.sh configs/batch_train_5deg.yaml
bash batch_train.sh configs/batch_train_10deg_rotation.yaml
bash batch_train.sh configs/batch_train_lr_ablation.yaml

# 查看会执行的命令（不实际运行）
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml

# 强制重新训练（忽略已存在的实验目录）
bash batch_train.sh --force configs/batch_train_5deg.yaml

# 跳过名称匹配的实验
bash batch_train.sh --skip-pattern "baseline" configs/batch_train_5deg.yaml

# 后台运行
nohup bash batch_train.sh configs/batch_train_5deg.yaml > batch.log 2>&1 &
```

**预配置实验组**:
| 配置文件 | 实验内容 | 实验数 |
| --- | --- | --- |
| `configs/batch_train_5deg.yaml` | 5度扰动 + Z分辨率对比 (z=1,5,10) | 3组 |
| `configs/batch_train_10deg_rotation.yaml` | 10度rotation-only + Z对比 | 3组 |
| `configs/batch_train_lr_ablation.yaml` | 学习率消融 (1e-4, 2e-4, 5e-4, 1e-3) | 4组 |

**特性**:
- ✅ 配置文件驱动，无需修改脚本代码
- ✅ **defaults 继承机制**（消除重复配置，v2.1+）
- ✅ 支持 `start_training.sh` 的所有参数
- ✅ 串行执行，自动管理GPU资源
- ✅ **智能TensorBoard管理**（监控当前实验具体目录，v2.1+）
- ✅ **自动跳过已完成实验**（检测输出目录是否存在 `train.log`）
- ✅ **实验级 skip 标志**（YAML 中 `skip: true` 跳过单个实验）
- ✅ **`--force` 强制重跑** / **`--skip-pattern` 按名称正则过滤**
- ✅ 实验间自动等待释放资源
- ✅ 详细的日志记录

**创建自定义配置**:
```bash
# 复制模板
cp configs/batch_train_5deg.yaml configs/my_experiments.yaml

# 编辑配置文件
vim configs/my_experiments.yaml

# 运行
bash batch_train.sh configs/my_experiments.yaml
```

**配置文件格式**: 见 [configs/README.md](configs/README.md)

### 4. train_B26A.sh（向后兼容）

B26A 专用训练脚本，保持向后兼容。

```bash
bash train_B26A.sh scratch --cuda_device 0
```

## 批量训练配置系统

### 配置文件驱动

`batch_train.sh` 现在支持通过YAML配置文件定义批量训练实验，无需修改脚本代码。

#### 预配置实验模板

| 配置文件 | 实验内容 | 实验数 | 用途 |
| --- | --- | --- | --- |
| `configs/batch_train_5deg.yaml` | 5度扰动 + Z分辨率对比 | 3组 | Z消融实验 |
| `configs/batch_train_10deg_rotation.yaml` | 10度rotation-only + Z对比 | 3组 | Rotation模式实验 |
| `configs/batch_train_lr_ablation.yaml` | 学习率对比 | 4组 | 学习率消融 |
| `configs/batch_train_multinode.yaml` | 多机DDP训练示例 | 2组 | 多机训练模板 |
| `configs/batch_train_multinode_slurm.yaml` | SLURM集群训练 | 2组 | SLURM自动化 |
| `configs/batch_train_template.yaml` | 通用模板 | - | 自定义实验 |

#### 快速使用

```bash
# 使用默认配置（5deg Z对比）
bash batch_train.sh

# 使用指定配置
bash batch_train.sh configs/batch_train_10deg_rotation.yaml

# 查看会执行的命令（不实际运行）
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml

# 后台运行
nohup bash batch_train.sh configs/batch_train_5deg.yaml > batch.log 2>&1 &
```

#### 创建自定义配置

```bash
# 1. 复制模板
cp configs/batch_train_5deg.yaml configs/my_experiments.yaml

# 2. 编辑配置文件
vim configs/my_experiments.yaml

# 3. 运行
bash batch_train.sh configs/my_experiments.yaml
```

#### 配置文件示例

```yaml
# 全局配置
global:
  dry_run: false                 # 仅打印命令不执行
  force_rerun: false             # 强制重新训练（忽略已存在的实验目录）
  wait_between_experiments: 10   # 实验间等待时间（秒）

# 实验组配置
experiments:
  - name: "实验1"                # 实验名称
    description: "实验描述"       # 可选描述
    
    dataset: "B26A"              # 数据集: B26A / all / custom
    version: "v1"                # 版本标签
    
    env:
      BEV_ZBOUND_STEP: 2.0       # 环境变量
    
    params:
      angle_range_deg: 5         # 旋转扰动
      trans_range: 0.3           # 平移扰动
      batch_size: 16             # Batch size
      learning_rate: 0.0002      # 学习率
      use_ddp: true              # 使用DDP
      rotation_only: false       # 仅优化旋转
      foreground: true           # 前台执行（批量训练推荐）
      no_tensorboard: true       # 不启动TB（批量训练统一管理）

  - name: "实验2"
    dataset: "B26A"
    version: "v2"
    params:
      angle_range_deg: 10

  - name: "已完成的实验"
    skip: true                   # 跳过此实验
    skip_reason: "结果已归档"     # 可选：跳过原因
    dataset: "B26A"
    version: "v3"
      use_ddp: true
```

**支持的所有参数**:

配置文件支持 `start_training.sh` 的所有命令行选项：

| 配置参数 | 命令行选项 | 说明 |
| --- | --- | --- |
| `angle_range_deg` | `--angle` | 旋转扰动范围 |
| `trans_range` | `--trans` | 平移扰动范围 |
| `batch_size` | `--bs` | Batch size |
| `learning_rate` | `--lr` | 学习率 |
| `use_ddp` | `--ddp` | 使用DDP |
| `ddp_gpus` | `--ddp N` | DDP GPU数 |
| `use_compile` | `--compile` | torch.compile |
| `rotation_only` | `--rotation_only` | 仅优化旋转 |
| `foreground` | `--fg` | 前台执行 |
| `no_tensorboard` | `--no-tb` | 不启动TB |
| `tensorboard_port` | `--tb_port` | TB端口 |
| `nnodes` | `--nnodes` | 多机DDP-机器数 |
| `node_rank` | `--node_rank` | 多机DDP-机器编号 |
| `master_addr` | `--master_addr` | 多机DDP-master IP |
| `master_port` | `--master_port` | 多机DDP-master端口 |

详细配置说明见: [configs/README.md](configs/README.md)

#### 配置文件优势

**之前（硬编码）**:
```bash
# 需要修改脚本代码
EXPERIMENTS=(
    "20.0|B26A|v4-z1|--ddp --angle 5"
    "4.0|B26A|v4-z5|--ddp --angle 5"
)
```

**现在（配置文件）**:
```yaml
# 清晰的YAML格式，易于阅读和修改
experiments:
  - name: "z=1"
    dataset: "B26A"
    version: "v4-z1"
    env:
      BEV_ZBOUND_STEP: 20.0
    params:
      angle_range_deg: 5
      use_ddp: true
```

## 高级用法

### 不同扰动级别

根据初始标定误差选择合适的扰动范围：

```bash
# 小扰动 (5°, 0.3m) - 标定误差 < 5°
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 0 \
  --angle_range_deg 5 \
  --trans_range 0.3 \
  --log_suffix small_5deg_v1

# 中等扰动 (10°, 0.5m) - 标定误差 5-10°（推荐）
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 1 \
  --angle_range_deg 10 \
  --trans_range 0.5 \
  --log_suffix medium_10deg_v1

# 大扰动 (20°, 1.5m) - 标定误差 > 10°
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 2 \
  --angle_range_deg 20 \
  --trans_range 1.5 \
  --log_suffix large_20deg_v1
```

### BEV Z分辨率实验

通过环境变量 `BEV_ZBOUND_STEP` 控制BEV高度方向的体素分辨率：

```bash
# z=1: 更细粒度（20.0步长，200个体素层）
BEV_ZBOUND_STEP=20.0 bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 0 \
  --log_suffix z1_experiment

# z=5: 平衡（4.0步长，40个体素层，推荐）
BEV_ZBOUND_STEP=4.0 bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 1 \
  --log_suffix z5_experiment

# z=10: 粗粒度（2.0步长，20个体素层）
BEV_ZBOUND_STEP=2.0 bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 2 \
  --log_suffix z10_experiment
```

**推荐**: z=5 (BEV_ZBOUND_STEP=4.0) 在精度和泛化能力之间取得最佳平衡。

### Rotation-Only模式

仅标定旋转参数，固定平移：

```bash
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 0 \
  --angle_range_deg 10 \
  --trans_range 0.5 \
  --log_suffix rotation_only \
  --rotation_only 1  # 启用rotation-only模式
```

### 分布式训练（DDP）

使用多GPU加速训练：

```bash
# 使用start_training.sh的DDP模式
bash start_training.sh B26A v1 --ddp

# 或手动配置DDP
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --use_ddp \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 29500
```

## 监控训练

### 实时监控

```bash
# 查看 GPU 使用情况
nvidia-smi -l 1

# 查看训练进程
ps aux | grep train_kitti

# 实时日志
tail -f logs/B26A/model_small_10deg_v1/train.log

# 查看最近的epoch统计
tail -100 logs/B26A/model_small_10deg_v1/train.log | grep "Epoch \["
```

### TensorBoard 可视化

```bash
# 查看所有训练
tensorboard --logdir logs/ --port 6006

# 只查看 B26A 数据集
tensorboard --logdir logs/B26A/ --port 6006

# 只查看 all_training_data 数据集
tensorboard --logdir logs/all_training_data/ --port 6007

# 对比多个实验
tensorboard --logdir logs/B26A/model_small_5deg_v1/:5deg,logs/B26A/model_small_10deg_v1/:10deg --port 6006
```

然后在浏览器访问: `http://localhost:6006`

### 检查训练状态

```bash
# 查找最新检查点
find logs/B26A/model_small_10deg_v1/ -name "*.pth" | sort -V | tail -1

# 列出所有检查点
ls -lht logs/B26A/model_small_10deg_v1/*/checkpoint/*.pth

# 查看训练进度
grep "Epoch \[" logs/B26A/model_small_10deg_v1/train.log | tail -20
```

## 日志结构

训练后，日志按数据集分级组织：

```
logs/
├── B26A/                              # B26A 数据集
│   ├── model_small_10deg_v1/
│   │   ├── train.log                  # 训练日志
│   │   ├── B26A_scratch/
│   │   │   ├── events.out.tfevents.* # TensorBoard 事件
│   │   │   └── checkpoint/
│   │   │       ├── ckpt_40.pth       # 检查点
│   │   │       ├── ckpt_80.pth
│   │   │       └── ckpt_400.pth
│   │   └── test_data_eval/           # 测试评估结果（如果运行）
│   │
│   ├── model_small_5deg_v1/
│   │   └── ...
│   │
│   └── model_small_5deg_v4-z1/       # Z分辨率实验
│       ├── model_small_5deg_v4-z5/
│       └── model_small_5deg_v4-z10/
│
├── all_training_data/                 # 全量数据集
│   ├── model_small_10deg_v1/
│   └── model_small_5deg_v1/
│
└── README.md
```

## 典型工作流

### 1. 快速验证（B26A 小数据集）

```bash
# Step 1: 使用小数据集快速验证
bash start_training.sh B26A v1

# Step 2: 查看训练进度
tail -f logs/B26A/model_small_10deg_v1/train.log

# Step 3: 训练 50-100 个 epoch 后检查结果
tensorboard --logdir logs/B26A/ --port 6006

# Step 4: 评估检查点（可选）
python evaluate_checkpoint.py \
    --ckpt_path logs/B26A/model_small_10deg_v1/B26A_scratch/checkpoint/ckpt_100.pth \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data \
    --angle_range_deg 10.0 \
    --trans_range 0.5
```

### 2. 完整训练（all_training_data）

```bash
# Step 1: 使用全量数据集训练最终模型
bash start_training.sh all v1

# Step 2: 监控训练（需要更长时间）
tail -f logs/all_training_data/model_small_10deg_v1/train.log

# Step 3: 训练 200-500 个 epoch
tensorboard --logdir logs/all_training_data/ --port 6007

# Step 4: 备份重要模型
cp logs/all_training_data/model_small_10deg_v1/all_training_data_scratch/checkpoint/ckpt_400.pth \
   backups/best_model_v1.pth
```

### 3. 批量对比实验

使用 `batch_train.sh` 一次性运行多组实验：

```bash
# Z分辨率消融实验（串行执行3组）
bash batch_train.sh configs/batch_train_5deg.yaml

# Rotation-only实验
bash batch_train.sh configs/batch_train_10deg_rotation.yaml

# 学习率消融实验
bash batch_train.sh configs/batch_train_lr_ablation.yaml

# TensorBoard 查看所有实验
tensorboard --logdir logs/B26A/ --port 6006
```

**查看会执行的命令**:
```bash
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml
```

**自定义实验组**:
```bash
# 1. 复制配置模板
cp configs/batch_train_5deg.yaml configs/my_experiments.yaml

# 2. 编辑配置（添加/修改实验）
vim configs/my_experiments.yaml

# 3. 运行
bash batch_train.sh configs/my_experiments.yaml
```

### 4. 超参数消融实验

**使用配置文件批量运行**（推荐）:

```bash
# 学习率消融（串行执行4组）
bash batch_train.sh configs/batch_train_lr_ablation.yaml

# Z分辨率消融
bash batch_train.sh configs/batch_train_5deg.yaml

# Rotation-only消融
bash batch_train.sh configs/batch_train_10deg_rotation.yaml
```

**手动单个实验**:

```bash
# 学习率对比
bash start_training.sh B26A v1_lr1e4 --lr 0.0001
bash start_training.sh B26A v1_lr2e4 --lr 0.0002
bash start_training.sh B26A v1_lr5e4 --lr 0.0005

# 扰动范围对比
bash train_universal.sh scratch --dataset_root /path/to/data --angle_range_deg 5 --log_suffix 5deg
bash train_universal.sh scratch --dataset_root /path/to/data --angle_range_deg 10 --log_suffix 10deg
bash train_universal.sh scratch --dataset_root /path/to/data --angle_range_deg 20 --log_suffix 20deg
```

## 停止训练

```bash
# 使用停止脚本
bash stop_training.sh

# 或手动停止
pkill -f train_kitti

# 停止特定训练
pkill -f "train_kitti.py --log_dir ./logs/B26A"
```

## 常见问题

### Q1: 如何选择扰动级别？

**答**: 根据初始标定误差选择
- **小扰动 (5°, 0.3m)**: 标定误差 < 5°
- **中等扰动 (10°, 0.5m)**: 标定误差 5-10°（推荐）
- **大扰动 (20°, 1.5m)**: 标定误差 > 10°

### Q2: 训练需要多长时间？

**答**: 取决于数据集大小和 GPU
- B26A (约1544帧): ~1-2 小时 / 100 epoch (单 RTX 3090)
- all_training_data: 根据帧数而定
- 推荐训练 200-400 epochs

### Q3: 如何恢复中断的训练？

**答**: 使用 resume 模式
```bash
bash train_universal.sh resume \
  --dataset_root /path/to/data \
  --dataset_name B26A \
  --cuda_device 0
```

### Q4: 如何修改批次大小？

**答**: 修改脚本中的 `--batch_size` 参数
```bash
# 编辑 train_universal.sh
# 将 --batch_size 16 改为其他值（如 8, 32）
# 或直接传参（需修改脚本支持）
```

### Q5: 日志目录占用空间太大怎么办？

**答**: 定期清理和归档
```bash
# 归档旧日志
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/B26A/

# 删除原始日志（小心！）
rm -rf logs/B26A/model_old_*/

# 只保留重要检查点
cd logs/B26A/model_*/*/checkpoint/
rm ckpt_{20,60,100,140,180}.pth  # 保留40的倍数
```

### Q6: 如何选择BEV_ZBOUND_STEP？

**答**: 
- **z=1 (20.0)**: 最细粒度，训练精度最高，但可能过拟合
- **z=5 (4.0)**: 推荐，精度和泛化的最佳平衡
- **z=10 (2.0)**: 粗粒度，训练快但精度较低

根据泛化评估结果，**推荐使用 z=5**。

### Q7: 多个实验如何避免端口冲突？

**答**: 为每个实验指定不同的TensorBoard端口
```bash
# 终端1
bash train_universal.sh scratch --dataset_root /data1 --tensorboard_port 6006

# 终端2
bash train_universal.sh scratch --dataset_root /data2 --tensorboard_port 6007

# 终端3
bash train_universal.sh scratch --dataset_root /data3 --tensorboard_port 6008
```

## 最佳实践

### 训练前检查清单

- [ ] Conda 环境已激活 (`bevcalib`)
- [ ] GPU 可用且显存充足 (`nvidia-smi`)
- [ ] bev_pool CUDA 扩展已编译
- [ ] 数据集准备完成，格式正确
- [ ] 确定训练配置（数据集、扰动级别、版本号）

### 训练中检查清单

- [ ] 训练进程正常运行 (`ps aux | grep train_kitti`)
- [ ] GPU 利用率正常 (80-95%)
- [ ] 日志正常写入
- [ ] Loss 正常下降（TensorBoard）
- [ ] 无 OOM 或 CUDA 错误

### 训练后检查清单

- [ ] 检查点文件已保存（`*.pth`）
- [ ] TensorBoard 曲线正常
- [ ] 评估模型性能（`evaluate_checkpoint.py`）
- [ ] 备份重要模型
- [ ] 记录最佳配置

### 推荐设置

**1. 数据集选择**
- 快速验证: B26A
- 最终模型: all_training_data
- 泛化测试: 多数据集交叉验证

**2. 训练参数**
- Batch size: 8-16 (根据GPU显存调整)
- Learning rate: 2e-4 (scratch), 7.5e-5 (finetune)
- Epochs: 200-400 (B26A), 400-500 (all_training_data)
- 扰动范围: 10°/0.5m（推荐）

**3. 版本管理**
使用有意义的版本号：
- `v1`: 基线实验
- `v2_lr1e4`: 学习率调优
- `v3_z5`: Z分辨率实验
- `v4_rotation`: Rotation-only模式

**4. 日志管理**
- 定期检查训练进度（每50-100 epochs）
- 保存重要检查点到独立目录
- 定期清理和归档旧日志

**5. 从小到大**
1. 先用B26A验证配置正确（1-2小时）
2. 再用all_training_data训练最终模型（数小时）
3. 最后进行泛化测试

## 快速命令备忘

```bash
# ============ 启动训练 ============
# B26A 数据集（快速验证）
bash start_training.sh B26A v1

# all_training_data 数据集（完整训练）
bash start_training.sh all v1

# 批量实验（配置文件驱动）
bash batch_train.sh configs/batch_train_5deg.yaml          # Z分辨率对比
bash batch_train.sh configs/batch_train_lr_ablation.yaml   # 学习率消融

# 自定义配置
bash train_universal.sh scratch \
  --dataset_root /path/to/data \
  --cuda_device 0 \
  --angle_range_deg 10 \
  --trans_range 0.5

# ============ 监控训练 ============
# GPU 状态
nvidia-smi -l 1

# 训练进程
ps aux | grep train_kitti

# 实时日志
tail -f logs/B26A/model_small_10deg_v1/train.log

# TensorBoard
tensorboard --logdir logs/B26A/ --port 6006

# ============ 停止训练 ============
bash stop_training.sh
# 或
pkill -f train_kitti

# ============ 检查点管理 ============
# 查找最新检查点
find logs/B26A/model_small_10deg_v1/ -name "*.pth" | sort -V | tail -1

# 评估检查点
python evaluate_checkpoint.py \
    --ckpt_path logs/.../ckpt_400.pth \
    --dataset_root /path/to/data

# ============ 分析工具 ============
# 分析训练性能
bash utils/scripts/quick_analyze.sh 5deg --skip-test

# 生成完整报告
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml
```

## 批量训练配置文件详解

### 配置文件结构（v2.1+ 支持 defaults 继承）

配置文件使用YAML格式，包含三部分：全局配置、默认参数（v2.1+）和实验组配置。

#### 1. 全局配置

```yaml
global:
  dry_run: false                 # 仅打印命令不执行
  batch_log_dir: "logs"          # 批量日志目录
  wait_between_experiments: 10   # 实验间等待（秒）
```

#### 2. 默认参数（v2.1+ 新增）

所有实验继承的公共参数，消除重复配置：

```yaml
defaults:
  env: {}                        # 默认环境变量
  
  params:
    angle_range_deg: 5          # 所有实验使用5度扰动
    trans_range: 0.3
    batch_size: 16              # 所有实验使用batch=16
    learning_rate: null
    use_ddp: true               # 所有实验使用DDP
    foreground: true
    no_tensorboard: true
    # ... 其他公共参数
```

#### 3. 实验配置（只需指定差异化参数）

每个实验只需指定与 `defaults` 不同的参数：

```yaml
experiments:
  - name: "z=5 (5deg)"          # 显示名称
    description: "实验描述"      # 可选
    dataset: "B26A"              # 必需
    version: "v4-z5"             # 必需
    env:
      BEV_ZBOUND_STEP: 4.0       # 实验特定环境变量
    # params 自动继承 defaults（无需重复）
  
  - name: "z=5 bs=32 (5deg)"
    dataset: "B26A"
    version: "v4-z5-bs32"
    env:
      BEV_ZBOUND_STEP: 4.0
    params:
      batch_size: 32             # 覆盖defaults中的16
```

**参数继承规则**:
- 实验首先继承 `defaults` 中的所有 `env` 和 `params`
- 实验中指定的参数会覆盖 `defaults` 中的同名参数
- 优先级：`experiment.params` > `defaults.params`

### 参数完整列表

| 参数路径 | 类型 | 默认值 | 对应命令行 | 说明 |
| --- | --- | --- | --- | --- |
| `global.dry_run` | bool | false | `--dry-run` | 仅打印命令 |
| `global.force_rerun` | bool | false | `--force` | 强制重跑（忽略已存在目录） |
| `global.wait_between_experiments` | int | 10 | - | 实验间等待 |
| `skip` | bool | false | - | 跳过此实验 |
| `skip_reason` | str | "" | - | 跳过原因（日志中显示） |
| `dataset` | str | - | 位置参数1 | 数据集名称 |
| `version` | str | - | 位置参数2 | 版本标签 |
| `env.BEV_ZBOUND_STEP` | float | - | 环境变量 | BEV Z步长 |
| `params.angle_range_deg` | float | null | `--angle` | 旋转扰动 |
| `params.trans_range` | float | null | `--trans` | 平移扰动 |
| `params.batch_size` | int | null | `--bs` | Batch size |
| `params.learning_rate` | float | null | `--lr` | 学习率 |
| `params.use_ddp` | bool | false | `--ddp` | DDP模式 |
| `params.ddp_gpus` | int | null | `--ddp N` | DDP GPU数 |
| `params.use_compile` | bool | false | `--compile` | torch.compile |
| `params.rotation_only` | bool | false | `--rotation_only` | 仅旋转 |
| `params.foreground` | bool | false | `--fg` | 前台执行 |
| `params.no_tensorboard` | bool | false | `--no-tb` | 不启动TB |
| `params.tensorboard_port` | int | null | `--tb_port` | TB端口 |
| `params.nnodes` | int | null | `--nnodes` | 机器数 |
| `params.node_rank` | int | null | `--node_rank` | 机器编号 |
| `params.master_addr` | str | null | `--master_addr` | Master IP |
| `params.master_port` | int | null | `--master_port` | Master端口 |

**注意**: `null` 表示使用 `start_training.sh` 的默认值。

### 典型配置场景（使用 defaults 继承）

#### 场景1: Z分辨率消融（最简配置）

```yaml
defaults:
  params:
    angle_range_deg: 5          # 公共参数
    use_ddp: true
    foreground: true
    no_tensorboard: true

experiments:
  - name: "z=1"
    dataset: "B26A"
    version: "v4-z1"
    env:
      BEV_ZBOUND_STEP: 20.0     # 只需指定差异

  - name: "z=5"
    dataset: "B26A"
    version: "v4-z5"
    env:
      BEV_ZBOUND_STEP: 4.0

  - name: "z=10"
    dataset: "B26A"
    version: "v4-z10"
    env:
      BEV_ZBOUND_STEP: 2.0
```

**优势**: 3个实验配置从原来的45行缩减到15行。

#### 场景2: 学习率消融

```yaml
defaults:
  env:
    BEV_ZBOUND_STEP: 4.0        # 固定z=5
  params:
    angle_range_deg: 5
    use_ddp: true
    batch_size: 16
    foreground: true
    no_tensorboard: true

experiments:
  - name: "lr=1e-4"
    dataset: "B26A"
    version: "v5-lr1e4"
    params:
      learning_rate: 0.0001     # 只覆盖learning_rate

  - name: "lr=2e-4"
    dataset: "B26A"
    version: "v5-lr2e4"
    params:
      learning_rate: 0.0002
```

**优势**: 每个实验只需3行，清晰展示实验变量。

#### 场景3: Rotation-Only vs 完整标定

```yaml
defaults:
  env:
    BEV_ZBOUND_STEP: 4.0
  params:
    angle_range_deg: 5
    trans_range: 0.3
    use_ddp: true
    rotation_only: false        # 默认完整标定
    foreground: true
    no_tensorboard: true

experiments:
  - name: "完整标定"
    dataset: "B26A"
    version: "v1-full"
    # 完全使用defaults

  - name: "仅旋转"
    dataset: "B26A"
    version: "v1-rotation"
    params:
      angle_range_deg: 10       # 覆盖部分参数
      rotation_only: true
```

**优势**: "完整标定"实验仅需4行，无需重复公共参数。

#### 场景4: 多数据集对比

```yaml
experiments:
  - name: "B26A"
    dataset: "B26A"
    version: "v1"
    params:
      use_ddp: true

  - name: "全量数据"
    dataset: "all"
    version: "v1"
    params:
      use_ddp: true
```

### 使用技巧

#### 1. 快速验证配置

```bash
# 使用 --dry-run 查看会执行的命令
bash batch_train.sh --dry-run configs/my_config.yaml
```

#### 2. 跳过已完成的实验

批量训练会自动检测输出目录，若已存在 `train.log` 则跳过该实验：

```
⏭️  跳过实验 [2/7]: v8_z10
  原因: 输出目录已存在且包含训练日志
  路径: /path/to/logs/all_training_data/model_small_5deg_v8_z10_quick
  如需重新训练，请删除该目录或使用 --force 参数
```

**跳过控制方式：**

| 方式 | 用法 | 场景 |
| --- | --- | --- |
| 自动跳过 | 默认行为 | 断点续跑批量实验 |
| `--force` | `bash batch_train.sh --force config.yaml` | 强制全部重跑 |
| `force_rerun: true` | YAML `global` 中设置 | 配置文件级别强制重跑 |
| `skip: true` | YAML 单个实验中设置 | 永久标记某实验不运行 |
| `--skip-pattern` | `bash batch_train.sh --skip-pattern "baseline\|v8" config.yaml` | 按名称正则跳过 |

#### 3. 参数省略

只指定需要修改的参数，其他使用默认值：

```yaml
experiments:
  - dataset: "B26A"
    version: "v1"
    params:
      use_ddp: true
      foreground: true
      no_tensorboard: true
      # 其他参数使用默认值
```

#### 3. 环境变量使用

```yaml
env:
  BEV_ZBOUND_STEP: 2.0           # 必须匹配训练时的值
  CUSTOM_VAR: "value"            # 可以设置任何环境变量
```

#### 4. 版本命名规范

建议使用清晰的版本标识：

```yaml
version: "v4-z5"        # v4版本, z=5配置
version: "v5-lr1e4"     # v5版本, lr=1e-4
version: "v1-rotation"  # v1版本, rotation-only模式
```

#### 5. TensorBoard 监控（v2.1+ 智能管理）

`batch_train.sh` 会自动为每个实验启动 TensorBoard：

**特性**:
- ✅ **精确监控**: 只监控当前实验的具体目录
- ✅ **自动推导**: 根据 dataset/version/angle 自动推导目录
- ✅ **清晰聚焦**: 不受其他实验干扰
- ✅ **日志分离**: TensorBoard 日志存放在实验目录中

**示例**:

```yaml
defaults:
  params:
    angle_range_deg: 5

experiments:
  - name: "z=1"
    dataset: "B26A"
    version: "v4-z1"
    # TensorBoard 监控: logs/B26A/model_small_5deg_v4-z1/
  
  - name: "z=5"
    dataset: "B26A"
    version: "v4-z5"
    # TensorBoard 监控: logs/B26A/model_small_5deg_v4-z5/
```

**优化前** (❌ 错误):
- 监控 `logs/B26A/`（整个数据集）
- 同时显示所有实验曲线
- 难以聚焦当前训练

**优化后** (✅ 正确):
- 监控 `logs/B26A/model_small_5deg_v4-z1/`（具体实验）
- 只显示当前实验曲线
- 清晰聚焦，实时监控

**禁用 TensorBoard**:

如果不需要实时监控，可在 defaults 中设置：

```yaml
defaults:
  params:
    no_tensorboard: true
```

### 常见错误

#### 错误1: YAML语法错误

```yaml
# 错误: 缩进不正确
experiments:
- name: "exp1"
  params:
    use_ddp: true
```

```yaml
# 正确: 使用2空格缩进
experiments:
  - name: "exp1"
    params:
      use_ddp: true
```

#### 错误2: 布尔值格式

```yaml
# 错误: 使用字符串
use_ddp: "true"

# 正确: 使用布尔值
use_ddp: true
```

#### 错误3: 数值类型

```yaml
# 错误: 字符串数值
angle_range_deg: "5"

# 正确: 数值
angle_range_deg: 5
```

## DrInfer 模型部署

### 概述

训练完成后，可将 PyTorch 模型导出为 DrInfer 格式，用于高效推理部署。工具链包括：

| 工具 | 用途 |
| --- | --- |
| `utils/torch2drinfer.py` | PyTorch → DrInfer 模型导出（支持 flat/pmodel 两种布局） |
| `utils/drinfer_infer.py` | DrInfer 推理 & PyTorch vs DrInfer 对比评估 |
| `evaluate_drinfer.py` | DrInfer 模型泛化能力评估 |

### Step 1: 准备 DrInfer 配置文件

```bash
# 已有预配置模板 (在 configs/ 目录下)
ls configs/drinfer_config_*.yaml

# 或从模板创建
cp configs/drinfer_config_v8_scatter_mean_concat.yaml configs/drinfer_config_my_model.yaml
vim configs/drinfer_config_my_model.yaml
```

**关键配置项**（必须与训练配置一致）:
- `bev_zbound_step`: BEV Z 方向步长
- `voxel_mode`: 体素化模式（`hard` / `scatter`）
- `scatter_reduce`: Scatter 聚合方式（`sum` / `mean`）
- `to_bev_mode`: BEV 转换模式（`concat` / `learned` / `sum`）

### Step 2: 导出 DrInfer 模型

```bash
# 快速导出 (flat 布局: .bin + .txt)
python utils/torch2drinfer.py --config configs/drinfer_config_my_model.yaml

# 完整部署导出 (pmodel 布局: 含 nn_param.cfg、input/output_data)
python utils/torch2drinfer.py --config configs/drinfer_config_my_model.yaml --layout pmodel

# 指定输出目录
python utils/torch2drinfer.py --config configs/drinfer_config_my_model.yaml --layout pmodel --output_dir /path/to/output
```

**布局对比**:

| 布局 | 输出目录结构 | 用途 |
| --- | --- | --- |
| `flat` | `export_dir/{model_name}.bin` + `.txt` | 快速导出、开发验证 |
| `pmodel` | `export_dir/bevcalib_fusion_head/engine_graph/` + `input_data/` + `output_data/` | 正式部署、`pmodel forward` 兼容 |

### Step 3: 验证导出精度

```bash
# PyTorch vs DrInfer 对比 (精度 + 耗时 + 显存)
python utils/drinfer_infer.py --config configs/drinfer_config_my_model.yaml --mode compare

# 单后端评估
python utils/drinfer_infer.py --config configs/drinfer_config_my_model.yaml --backend drinfer
python utils/drinfer_infer.py --config configs/drinfer_config_my_model.yaml --backend pytorch

# pmodel 布局验证 (可选)
pmodel forward logs/.../drinfer/bevcalib_fusion_head
```

### Step 4: 泛化能力评估

```bash
# 单数据集评估
python evaluate_drinfer.py \
    --ckpt_path logs/.../checkpoint/ckpt_best_val.pth \
    --export_dir logs/.../drinfer \
    --dataset_root /path/to/test_data \
    --compare_pytorch

# 批量泛化评估 (通过 run_generalization_eval.py)
# 在 eval 配置中设置 backend: drinfer
python run_generalization_eval.py --config configs/eval_generalization.yaml
```

### 完整部署流程示例

```bash
# 1. 训练模型
bash batch_train.sh configs/batch8_train_all_v8_quick.yaml

# 2. 导出为 DrInfer
python utils/torch2drinfer.py \
    --config configs/drinfer_config_v8_scatter_mean_concat.yaml \
    --layout pmodel

# 3. 精度对比验证
python utils/drinfer_infer.py \
    --config configs/drinfer_config_v8_scatter_mean_concat.yaml \
    --mode compare

# 4. 泛化评估
python evaluate_drinfer.py \
    --ckpt_path logs/all_training_data/model_small_5deg_v8_scatter_mean_quick/all_training_data_scratch/checkpoint/ckpt_best_val.pth \
    --export_dir logs/all_training_data/model_small_5deg_v8_scatter_mean_quick/drinfer \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --compare_pytorch
```

**详细 DrInfer 配置说明**: 见 [configs/README.md](configs/README.md) 中的 "DrInfer 转换配置" 章节。

## 相关文档

- **数据准备**: [PREPARE_CUSTOM_DATASET.md](PREPARE_CUSTOM_DATASET.md)
- **模型评估**: [README.md#evaluation](README.md#evaluation)
- **性能分析**: [utils/ANALYSIS_GUIDE.md](utils/ANALYSIS_GUIDE.md)
- **批量训练配置**: [configs/README.md](configs/README.md)
- **DrInfer 配置详解**: [configs/README.md](configs/README.md) (DrInfer 转换配置章节)

## 故障排除

### CUDA Out of Memory

```bash
# 减小batch size
# 编辑 train_universal.sh，将 --batch_size 从 16 改为 8 或 4
```

### TensorBoard端口被占用

```bash
# 查看端口占用
lsof -i :6006

# 使用其他端口
tensorboard --logdir logs/ --port 6007
```

### 训练卡住不动

```bash
# 检查GPU状态
nvidia-smi

# 检查进程
ps aux | grep train_kitti

# 查看最新日志
tail -100 logs/B26A/model_*/train.log
```

### bev_pool编译失败

```bash
# 重新编译
cd kitti-bev-calib/img_branch/bev_pool
python setup.py clean
python setup.py build_ext --inplace

# 检查CUDA版本匹配
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

---

**创建日期**: 2026-03-09  
**适用版本**: start_training.sh + train_universal.sh + batch_train.sh  
**推荐工作流**: start_training.sh → 监控 → 评估 → 分析
