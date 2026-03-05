# [CoRL 2025] BEVCalib: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representation

[![arXiv](https://img.shields.io/badge/arXiv-2506.02587-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2506.02587) [![Website](https://img.shields.io/badge/Website-BEVCalib-blue?style=for-the-badge)](https://cisl.ucr.edu/BEVCalib) [![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/cisl-hf/BEVCalib) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) [![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

> **🎉 NEW (2026-01-27)**: `prepare_custom_dataset.py` v2.0 发布！  
> - 🚀 **5-8x 速度提升**（并行处理 + 批处理优化）  
> - 👁️ **PLY 格式临时点云**（可用 CloudCompare 直接查看验证）  
> - 📋 查看详情: [QUICKSTART_PERFORMANCE.md](QUICKSTART_PERFORMANCE.md) | [CHANGELOG_v2.0.md](CHANGELOG_v2.0.md)

<hr style="border: 2px solid gray;"></hr>

## Getting Started

### Prerequistes
First create a conda environment:
```bash
conda create -n bevcalib python=3.11
conda activate bevcalib
pip3 install -r requirements.txt
```

The code is built with following libraries:

- Python = 3.11
- Pytorch = 2.2.2
- CUDA = 11.8
- cuda-toolkit = 11.8
- [spconv-cu118](https://github.com/traveller59/spconv)
- OpenCV
- pandas
- open3d
- transformers
- [deformable_attention](https://github.com/lucidrains/deformable-attention)
- tensorboard
- wandb
- pykitti

We recommend using the following command to install cuda-toolkit=11.8:
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

After installing the above dependencies, please run the following command to install [bev_pool](https://github.com/mit-han-lab/bevfusion) operation
```bash
cd ./kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

We also provide a [Dockerfile](Dockerfile/Dockerfile) for easy setup, please execute the following command to build the docker image and install cuda extensions:
```bash
docker build -f Dockerfile/Dockerfile -t bevcalib .
docker run --gpus all -it -v$(pwd):/workspace bevcalib
### In the docker, run the following command to install cuda extensions
cd ./kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

## Dataset Preparation
### KITTI-Odometry
Coordinates reference: https://developer.aliyun.com/article/855136
We release the code to reproduce our results on the KITTI-Odometry dataset. Please download the KITTI-Odometry dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). After downloading the dataset, the directory structure should look like
```tree
kitti-odometry/
├── sequences/         
│   ├── 00/            
│   │   ├── image_2/  
│   │   ├── image_3/   
│   │   ├── velodyne/
│   │   └── calib.txt 
│   ├── 01/
│   │   ├── ...
│   └── 21/
│       └── ...
└── poses/            
    ├── 00.txt        
    ├── 01.txt
    └── ...
```

### CalibDB
Coming soon!

### Custom Dataset
We provide a tool to prepare your custom dataset from ROS bags. See [PREPARE_CUSTOM_DATASET.md](PREPARE_CUSTOM_DATASET.md) for detailed instructions.

**Quick Start:**
```bash
# Prepare dataset from ROS bags (optimized for speed)
python tools/prepare_custom_dataset.py \
    --bag_dir /path/to/bags \
    --config_dir /path/to/config \
    --output_dir ./data/custom_dataset \
    --batch_size 500 \
    --num_workers 4

# View extracted point clouds (PLY format for easy visualization)
python tools/view_pointcloud.py ./data/custom_dataset/temp/pointclouds/000000.ply

# Validate the dataset
python tools/validate_kitti_odometry.py --dataset_root ./data/custom_dataset
```

**Key Features:**
- 🚀 **5-8x faster** with parallel processing and optimized batch size
- 👁️ **PLY format** for temporary point clouds (easy to view with CloudCompare/MeshLab)
- 🔄 **Automatic conversion** to BIN format for training
- 📊 **Built-in visualization** tool for data verification
- 🎯 **Auto-detect sequences** - automatically finds all available sequences for training

**Training on Custom Dataset:**
```bash
# Prepare dataset
python tools/prepare_custom_dataset.py \
    --bag_dir /path/to/bags \
    --config_dir /path/to/config \
    --output_dir ./data/custom_dataset

# Train (automatically detects all sequences)
python kitti-bev-calib/train_kitti.py \
    --dataset_root ./data/custom_dataset \
    --log_dir ./logs/custom_model \
    --batch_size 4 \
    --num_epochs 100
```

See [CUSTOM_DATASET_TRAINING.md](CUSTOM_DATASET_TRAINING.md) for detailed training guide.

## Pretrained Model
We release our pretrained model on the KITTI-Odometry dataset. We provide two ways to download our models.
### Google cloud
Please find the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1r9RkZATm9-7vh5buoB1YSDuL3_DslxZ3?usp=share_link) and place it in the `./ckpt` directory. For your convenience, you can also run `pip3 install gdown` and run the following command to download the KITTI checkpoint in the command line.

```bash
gdown https://drive.google.com/uc\?id\=1gWO-Z4NXG2uWwsZPecjWByaZVtgJ0XNb
```
### Hugging face
We also release our pretrained model on [Hugging Face page](https://huggingface.co/cisl-hf/BEVCalib). You should download huggingface-cli by `pip install -U "huggingface_hub[cli]"` and then download the pretrained model by running the following command:
```bash
huggingface-cli download cisl-hf/BEVCalib --revision kitti-bev-calib --local-dir YOUR_LOCAL_PATH
```

## Evaluation

### evaluate_checkpoint.py - Checkpoint 评估工具

评估已保存的 checkpoint，生成详细的误差报告和可视化结果。

#### 功能特性
- ✅ 在验证集上评估模型性能
- ✅ 生成详细的误差统计（平移、旋转、RPY分解）
- ✅ 为每个样本生成可视化图像（Init/GT/Pred 三列对比）
- ✅ 保存外参矩阵和误差报告（文本文件）
- ✅ 支持自定义扰动范围
- ✅ 支持批量处理（可限制评估样本数）

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt_path` | str | 必填 | Checkpoint 文件路径 (*.pth) |
| `--dataset_root` | str | 必填 | 数据集根目录路径 |
| `--angle_range_deg` | float | 20.0 | 评估时的扰动角度范围 (度) |
| `--trans_range` | float | 1.5 | 评估时的扰动平移范围 (米) |
| `--target_width` | int | 640 | 目标图像宽度 |
| `--target_height` | int | 360 | 目标图像高度 |
| `--batch_size` | int | 8 | Batch size |
| `--max_batches` | int | 5 | 最多评估的batch数（0=全部，5=快速测试）|
| `--validate_sample_ratio` | float | 0.1 | 验证集比例（0.0-1.0）|
| `--deformable` | int | 0 | 是否使用 deformable attention (与训练时保持一致) |
| `--bev_encoder` | int | 1 | 是否使用 BEV encoder (与训练时保持一致) |
| `--xyz_only` | int | 1 | 是否只使用 XYZ 坐标 (与训练时保持一致) |
| `--vis_points` | int | 80000 | 可视化的最大点数 |
| `--vis_point_radius` | int | 1 | 可视化点的半径（像素）|

#### 使用示例

**基础用法：**
```bash
python evaluate_checkpoint.py \
    --ckpt_path logs/model/checkpoint/ckpt_100.pth \
    --dataset_root ./data/custom_dataset \
    --angle_range_deg 20.0 \
    --trans_range 1.5
```

**自定义数据集评估（B26A）：**
```bash
python evaluate_checkpoint.py \
    --ckpt_path logs/B26A_model_small_3deg_v6.5/checkpoint/ckpt_500.pth \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data_fix \
    --angle_range_deg 3.0 \
    --trans_range 0.15 \
    --target_width 640 \
    --target_height 360 \
    --batch_size 8 \
    --max_batches 10
```

**快速测试（少量样本）：**
```bash
python evaluate_checkpoint.py \
    --ckpt_path logs/model/checkpoint/ckpt_100.pth \
    --dataset_root ./dataset \
    --max_batches 2
```

**完整评估（所有验证集）：**
```bash
python evaluate_checkpoint.py \
    --ckpt_path logs/model/checkpoint/ckpt_100.pth \
    --dataset_root ./dataset \
    --max_batches 0  # 0 表示评估全部
```

#### 输出结果

评估完成后会在 checkpoint 同级目录生成评估文件夹：
```
logs/model/checkpoint/
├── ckpt_100.pth
└── ckpt_100_eval/                    # 评估结果目录
    ├── sample_0000_projection.png    # 可视化图像（3列：Init/GT/Pred）
    ├── sample_0001_projection.png
    ├── sample_0002_projection.png
    ├── ...
    └── extrinsics_and_errors.txt     # 详细误差报告
```

**extrinsics_and_errors.txt 内容示例：**
```
Checkpoint: epoch_100
Evaluation on validation set (perturbation: 20.0deg, 1.5m)
================================================================================

Ground Truth Extrinsics (LiDAR → Camera):
  ...4×4 矩阵...

================================================================================

Sample 0000
--------------------------------------------------------------------------------

Predicted Extrinsics (LiDAR → Camera):
  ...4×4 矩阵...

Translation Errors (in LiDAR coordinate system):
  Total:   0.015234 m
  X (Fwd): 0.008123 m
  Y (Lat): 0.003456 m
  Z (Ht):  0.012890 m

Rotation Errors (axis-angle):
  Total:       0.234567 deg
  Roll (X):    0.123456 deg
  Pitch (Y):   0.089012 deg
  Yaw (Z):     0.178901 deg

================================================================================
...更多样本...

================================================================================
AVERAGE ERRORS ACROSS ALL SAMPLES
================================================================================

Total samples evaluated: 40

Average Translation Errors (in LiDAR coordinate system):
  Total:   0.012345 ± 0.006789 m
  X (Fwd): 0.007890 ± 0.004321 m
  Y (Lat): 0.003210 ± 0.001987 m
  Z (Ht):  0.010234 ± 0.005678 m

Average Rotation Errors (axis-angle):
  Total:       0.198765 ± 0.089012 deg
  Roll (X):    0.098765 ± 0.045678 deg
  Pitch (Y):   0.067890 ± 0.032109 deg
  Yaw (Z):     0.134567 ± 0.056789 deg

================================================================================
```

#### 终端输出示例

```
================================================================================
评估 Checkpoint: logs/model/checkpoint/ckpt_100.pth
================================================================================

1. 加载模型配置...
   ✓ 从 checkpoint 加载参数
   ✓ 训练噪声: 20.0°, 1.5m
   ✓ 评估噪声: 20.0°, 1.5m

2. 创建模型...
   ✓ 模型结构: BEVCalib
   ✓ Deformable Attention: 否
   ✓ BEV Encoder: 是

3. 加载数据集...
   ✓ 自动检测到 1 个序列: ['00']
   ✓ 数据集: 1234 个样本
   ✓ 验证集: 123 个样本

4. 开始评估...
   输出目录: logs/model/checkpoint/ckpt_100_eval
   扰动参数: 20.0°, 1.5m
   
   处理样本 0... ✓ (Trans: 0.0152m, Rot: 0.23°)
   处理样本 1... ✓ (Trans: 0.0134m, Rot: 0.19°)
   处理样本 2... ✓ (Trans: 0.0167m, Rot: 0.28°)
   ...

5. 评估完成！
   ✓ 总样本数: 40
   ✓ 平均平移误差: 0.0123 ± 0.0068 m
   ✓ 平均旋转误差: 0.199 ± 0.089 deg
   ✓ 结果已保存到: logs/model/checkpoint/ckpt_100_eval
   
================================================================================
```

#### 提示与技巧

**1. 快速测试 vs 完整评估**
```bash
# 快速测试（2个batch，约16个样本）
--max_batches 2

# 完整评估（所有验证集）
--max_batches 0
```

**2. 调整扰动范围**
```bash
# 小扰动（精度测试）
--angle_range_deg 3.0 --trans_range 0.15

# 大扰动（鲁棒性测试）
--angle_range_deg 20.0 --trans_range 1.5
```

**3. 查看可视化结果**
```bash
# 使用图像查看器打开
eog logs/model/checkpoint/ckpt_100_eval/sample_0000_projection.png

# 或批量查看
cd logs/model/checkpoint/ckpt_100_eval
ls sample_*.png
```

**4. 分析误差报告**
```bash
# 查看汇总统计
tail -30 logs/model/checkpoint/ckpt_100_eval/extrinsics_and_errors.txt

# 查看具体样本
grep "Sample 0000" -A 30 logs/model/checkpoint/ckpt_100_eval/extrinsics_and_errors.txt
```

#### 跨数据集泛化测试

使用在 A 数据集上训练的模型，评估其在 B 数据集（不同车辆/路线）上的表现，以测试模型泛化能力。

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | str | None | 自定义输出目录（避免覆盖原有 eval 结果）|
| `--use_full_dataset` | flag | False | 使用全量数据（跨数据集测试不需要 train/val split）|
| `--vis_interval` | int | 1 | 每隔 N 个样本保存可视化（全量评估时建议 50-100）|

**泛化测试用法：**
```bash
export HF_HUB_OFFLINE=1  # 离线环境需要设置

python evaluate_checkpoint.py \
    --ckpt_path logs/test_arch_validation/.../checkpoint/ckpt_400.pth \
    --dataset_root /path/to/unseen_test_data \
    --output_dir /path/to/unseen_test_data/generalization_eval/ckpt_400 \
    --use_full_dataset \
    --max_batches 0 \
    --angle_range_deg 5.0 \
    --trans_range 0.3 \
    --vis_interval 50
```

**关键要点：**
- `--use_full_dataset`：跨数据集测试应使用全部数据，而非 train/val split
- `--output_dir`：指定独立输出目录，不覆盖训练时的 eval 结果
- `--vis_interval 50`：每 50 帧保存一张可视化（1000+ 帧时避免大量 IO）
- `--angle_range_deg` / `--trans_range`：应与训练时的扰动参数保持一致

**输出目录结构：**
```
generalization_eval/ckpt_400/
├── extrinsics_and_errors.txt     # 逐帧误差 + 全量平均统计
├── sample_0000_projection.png    # 每 vis_interval 帧的可视化
├── sample_0050_projection.png
└── ...
```

---

### inference_kitti.py 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_root` | str | 必填 | 数据集根目录路径 |
| `--ckpt_path` | str | 必填 | 训练好的模型检查点路径 |
| `--log_dir` | str | `./logs/inference` | 推理日志保存目录 |
| `--batch_size` | int | 1 | 批量大小 |
| `--xyz_only` | int | 1 | 是否只使用xyz坐标 (1=是, 0=否) |
| `--angle_range_deg` | float | 20.0 | 扰动角度范围 (度) |
| `--trans_range` | float | 1.5 | 扰动平移范围 (米) |

### 使用示例

**评估 KITTI 数据集:**
```bash
python kitti-bev-calib/inference_kitti.py \
    --log_dir ./logs/inference \
    --dataset_root /path/to/kitti-odometry \
    --ckpt_path ./ckpt/kitti.pth \
    --angle_range_deg 20.0 \
    --trans_range 1.5 \
    --batch_size 16
```

**评估自定义数据集 (B26A):**
```bash
# 查看可用的检查点
ls ./logs/B26A_model_B26A_fix/B26A_scratch/checkpoint/

# 使用最新的检查点进行评估
python kitti-bev-calib/inference_kitti.py \
    --log_dir ./logs/inference_origin \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data_fix \
    --ckpt_path ./code/BEVCalib/logs/B26A_model_B26A_origin/B26A_scratch/checkpoint/ckpt_500.pth \
    --angle_range_deg 20.0 \
    --trans_range 1.5 \
    --batch_size 16
```

**快速获取最新检查点并评估:**
```bash
# 找到最新的检查点
LATEST_CKPT=$(ls -t ./logs/B26A_model_*/*/checkpoint/ckpt_*.pth 2>/dev/null | head -1)
echo "Latest checkpoint: $LATEST_CKPT"

# 运行评估
python kitti-bev-calib/inference_kitti.py \
    --dataset_root /path/to/dataset \
    --ckpt_path $LATEST_CKPT
```

### 输出说明

评估完成后会输出以下指标:
- **Translation Loss**: 平移损失 (米)
- **Rotation Loss**: 旋转损失 (度)
- **Translation xyz error**: X/Y/Z 各轴平移误差 (米)
- **Rotation ypr error**: Yaw/Pitch/Roll 各轴旋转误差 (度)

## Training

### Training on KITTI-Odometry
We provide instructions to reproduce our results on the KITTI-Odometry dataset:
```bash
python kitti-bev-calib/train_kitti.py \
        --log_dir ./logs/kitti \
        --dataset_root YOUR_PATH_TO_KITTI/kitti-odometry \
        --save_ckpt_per_epoches 40 \
        --num_epochs 500 \
        --label 20_1.5 \
        --angle_range_deg 20 \
        --trans_range 1.5 \
        --deformable 0 \
        --bev_encoder 1 \
        --batch_size 16 \
        --xyz_only 1 \
        --scheduler 1 \
        --lr 1e-4 \
        --step_size 80
```

### Training on Custom Dataset

**Quick Start (B26A Dataset):**
```bash
# Train from scratch (basic usage)
bash train_B26A.sh scratch > logs/train_B26A_scratch_$(date 
+%Y%m%d_%H%M%S).log 2>&1 &

# Train with specific GPU and TensorBoard port
nohup bash train_B26A.sh scratch --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data --log_suffix B26A_origin --cuda_device 4 --tensorboard_port 6006  > logs/train_B26A_scratch_$(date +%Y%m%d_%H%M%S)_origin.log 2>&1 &

# train2
nohup bash train_B26A.sh scratch --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data_fix --log_suffix B26A_fix --cuda_device 5 --tensorboard_port 6007  > logs/train_B26A_scratch_$(date +%Y%m%d_%H%M%S)_fix.log 2>&1 &

# train3
nohup bash train_B26A.sh scratch --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data_fix --log_suffix B26A_opt --cuda_device 6 --tensorboard_port 6008  > logs/train_B26A_scratch_$(date +%Y%m%d_%H%M%S)_opt.log 2>&1 &

# train4 - scratch
nohup bash train_B26A.sh scratch --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data_fix --log_suffix B26A_finetune --cuda_device 7 --tensorboard_port 6009  > logs/train_B26A_finetune_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# stop trainnning 
pkill -9 -f "train_kitti.py" && pkill -9 -f "train_B26A.sh" 

# Train with custom dataset path and log suffix
bash train_B26A.sh scratch \
    --dataset_root /path/to/your/dataset \
    --cuda_device 0 \
    --log_suffix my_experiment

# Use tensorboard to monitor training
tensorboard --logdir ./logs/B26A_model --port 6006

# Fine-tune from KITTI pretrained model
bash train_B26A.sh finetune --cuda_device 0

# Resume from last checkpoint
bash train_B26A.sh resume --cuda_device 0
```

**🚀 Multi-Terminal Parallel Training:**

The script now supports training multiple datasets in parallel across different terminals with automatic port and GPU management:

```bash
# Terminal 1: Train dataset1 on GPU 0
bash train_B26A.sh scratch \
    --cuda_device 0 \
    --dataset_root /path/to/dataset1 \
    --log_suffix dataset1 \
    --tensorboard_port 6006

# Terminal 2: Train dataset2 on GPU 1
bash train_B26A.sh scratch \
    --cuda_device 1 \
    --dataset_root /path/to/dataset2 \
    --log_suffix dataset2 \
    --tensorboard_port 6007

# Terminal 3: Fine-tune dataset3 on GPU 2
bash train_B26A.sh finetune \
    --cuda_device 2 \
    --dataset_root /path/to/dataset3 \
    --log_suffix dataset3 \
    --tensorboard_port 6008
```

**Script Options:**
- `--cuda_device ID`: Specify CUDA device ID (e.g., 0, 1, 2). If not specified, uses all available GPUs.
- `--tensorboard_port PORT`: Specify TensorBoard port (default: 6006, auto-increments if in use).
- `--dataset_root PATH`: Custom dataset root directory.
- `--log_suffix SUFFIX`: Add suffix to log directory (useful for distinguishing multiple runs).

**Features:**
- ✅ **Automatic port detection**: Finds available TensorBoard ports automatically
- ✅ **GPU isolation**: Each training instance can use a specific GPU
- ✅ **Log separation**: Use `--log_suffix` to keep logs organized
- ✅ **Port conflict detection**: Warns if specified port is already in use

**Manual Command:**
```bash
python kitti-bev-calib/train_kitti.py \
        --log_dir ./logs/custom_model \
        --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
        --save_ckpt_per_epoches 20 \
        --num_epochs 100 \
        --label custom_20_1.5 \
        --angle_range_deg 20 \
        --trans_range 1.5 \
        --deformable 0 \
        --bev_encoder 1 \
        --batch_size 4 \
        --xyz_only 1 \
        --scheduler 1 \
        --lr 1e-4 \
        --step_size 40
```

**Parameter Recommendations for Custom Dataset:**
- `--batch_size`: 4-8 (smaller than KITTI due to potentially different data size)
- `--num_epochs`: 100-200 (adjust based on dataset size)
- `--save_ckpt_per_epoches`: 20-40 (save checkpoints more frequently)
- `--step_size`: 40-80 (learning rate decay step)
- `--lr`: 1e-4 (default learning rate)

**Fine-tuning from KITTI Pretrained Model:**
```bash
python kitti-bev-calib/train_kitti.py \
        --log_dir ./logs/custom_finetuned \
        --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
        --pretrain_ckpt ./ckpt/kitti.pth \
        --num_epochs 50 \
        --batch_size 4 \
        --lr 5e-5
```

**Notes:**
- Change `--angle_range_deg` and `--trans_range` to train under different noise settings
- Use `--pretrain_ckpt` to load a pretrained model for fine-tuning
- The dataset loader automatically detects all sequences in the dataset
- For parallel training, ensure each terminal uses a different GPU (`--cuda_device`) and TensorBoard port (`--tensorboard_port`)
- Use `--log_suffix` to distinguish logs from different training runs

**📚 Documentation:**
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training parameters and recommendations
- [CUSTOM_DATASET_TRAINING.md](CUSTOM_DATASET_TRAINING.md) - Custom dataset preparation guide

## Acknowledgement
BEVCalib appreciates the following great open-source projects: [BEVFusion](https://github.com/mit-han-lab/bevfusion?tab=readme-ov-file), [LCCNet](https://github.com/IIPCVLAB/LCCNet), [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [spconv](https://github.com/traveller59/spconv), and [Deformable Attention](https://github.com/lucidrains/deformable-attention).

## Citation
```
@inproceedings{bevcalib,
      title={BEVCALIB: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representations}, 
      author={Weiduo Yuan and Jerry Li and Justin Yue and Divyank Shah and Konstantinos Karydis and Hang Qiu},
      booktitle={9th Annual Conference on Robot Learning},
      year={2025},
}
```
