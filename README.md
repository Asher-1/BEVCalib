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
- Pytorch = 2.6.0
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
