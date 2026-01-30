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
Please run the following command to evaluate the model:
```bash
python kitti-bev-calib/inference_kitti.py \
         --log_dir ./logs/kitti \
         --dataset_root ~/develop/data/slam_data/KITTI/kitti-odometry \
         --ckpt_path ./ckpt/kitti.pth \
         --angle_range_deg 20.0 \
         --trans_range 1.5 \
         --batch_size 16
```

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
# Train from scratch
bash train_B26A.sh scratch

# Fine-tune from KITTI pretrained model
bash train_B26A.sh finetune

# Resume from last checkpoint
bash train_B26A.sh resume
```

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
