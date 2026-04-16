# [CoRL 2025] BEVCalib: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representation

[![arXiv](https://img.shields.io/badge/arXiv-2506.02587-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2506.02587) [![Website](https://img.shields.io/badge/Website-BEVCalib-blue?style=for-the-badge)](https://cisl.ucr.edu/BEVCalib) [![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/cisl-hf/BEVCalib) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) [![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

> **🎉 NEW (2026-03-09)**: v2.1 - 配置文件驱动的批量训练系统 + defaults 继承机制！  
> - 🚀 **YAML配置驱动**: `batch_train.sh` 支持所有 `start_training.sh` 参数（15个，含多机DDP）
> - ✨ **defaults 继承**: 消除重复配置，配置文件减少50-60%行数，可维护性大幅提升
> - 📊 **预配置模板**: Z消融、Rotation-only、学习率消融、多机训练（7+组开箱即用）
> - 🔧 **无需改代码**: 通过配置文件定义实验，支持dry-run预览
> - 🖥️ **多机训练**: 自动生成工具 + SLURM集成，30秒配置多机DDP
> - 📋 查看详情: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | [configs/README.md](configs/README.md) | [MULTINODE_TRAINING.md](MULTINODE_TRAINING.md)

> **v2.0 (2026-03-09)**: 完整的训练和分析工具链！  
> - 🚀 **统一训练脚本**: `start_training.sh` + `batch_train.sh` 支持DDP、自定义学习率
> - 📊 **性能分析工具**: 配置驱动的实验对比、泛化评估、Feishu报告生成  
> - 📋 查看详情: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | [utils/ANALYSIS_GUIDE.md](utils/ANALYSIS_GUIDE.md)

> **🎉 NEW (2026-01-27)**: `prepare_custom_dataset.py` v2.0 发布！  
> - 🚀 **5-8x 速度提升**（并行处理 + 批处理优化）  
> - 👁️ **PLY 格式临时点云**（可用 CloudCompare 直接查看验证）  
> - 📋 查看详情: [QUICKSTART_PERFORMANCE.md](QUICKSTART_PERFORMANCE.md) | [CHANGELOG_v2.0.md](CHANGELOG_v2.0.md)

<hr style="border: 2px solid gray;"></hr>

## 📚 文档导航

### 快速开始
- [Installation](#installation) - 环境安装
- [Dataset Preparation](#dataset-preparation) - 数据集准备

### 训练相关
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - 🌟 **完整训练指南**（推荐）
  - 训练脚本使用说明（`start_training.sh`, `train_universal.sh`, `batch_train.sh`）
  - 批量训练配置系统（配置文件驱动）
  - 参数配置详解
  - 监控和调试技巧
  - 最佳实践
- **[configs/README.md](configs/README.md)** - 🌟 **批量训练配置文档**
  - 配置文件格式说明
  - 预配置模板使用
  - 自定义配置指南
  - 多机训练配置
- **[MULTINODE_TRAINING.md](MULTINODE_TRAINING.md)** - **多机分布式训练指南**
  - 前置条件和环境检查
  - 自动配置生成工具
  - SLURM集群集成
  - 故障排除和性能优化
  - [快速开始 →](MULTINODE_QUICKSTART.md)
  - 多机训练配置
- **[MULTINODE_TRAINING.md](MULTINODE_TRAINING.md)** - 🌟 **多机分布式训练指南**
  - 前置条件和环境检查
  - 自动配置生成
  - SLURM集群集成
  - 故障排除和性能优化

### 评估相关
- [Evaluation](#evaluation) - 模型评估
  - `evaluate_checkpoint.py` - Checkpoint评估工具
  - `inference_kitti.py` - KITTI推理脚本
  - 跨数据集泛化测试

### 分析工具
- **[utils/ANALYSIS_GUIDE.md](utils/ANALYSIS_GUIDE.md)** - 🌟 **性能分析完整指南**
  - 训练日志自动解析
  - 测试集泛化评估
  - 多组实验对比可视化
  - Feishu兼容报告生成

### 自定义数据集
- [PREPARE_CUSTOM_DATASET.md](PREPARE_CUSTOM_DATASET.md) - 从ROS bags准备数据集
- [CUSTOM_DATASET_TRAINING.md](CUSTOM_DATASET_TRAINING.md) - 自定义数据集训练

<hr style="border: 2px solid gray;"></hr>

## Installation

### Prerequisites
First create a conda environment:
```bash
conda create -n bevcalib310 python=3.10
conda activate bevcalib310
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

After installing the above dependencies, please run the following command to install [bev_pool](https://github.com/mit-han-lab/bevfusion) operation:
```bash
cd ./kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

We also provide a [Dockerfile](Dockerfile/Dockerfile) for easy setup:
```bash
docker build -f Dockerfile/Dockerfile -t bevcalib310 .
docker run --gpus all -it -v$(pwd):/workspace bevcalib310
### In the docker, run the following command to install cuda extensions
cd ./kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

## Dataset Preparation

### KITTI-Odometry
Coordinates reference: https://developer.aliyun.com/article/855136

We release the code to reproduce our results on the KITTI-Odometry dataset. Please download the KITTI-Odometry dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). After downloading the dataset, the directory structure should look like:

```
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

## Pretrained Model

We release our pretrained model on the KITTI-Odometry dataset.

### Google cloud
Please find the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1r9RkZATm9-7vh5buoB1YSDuL3_DslxZ3?usp=share_link) and place it in the `./ckpt` directory. You can also run `pip3 install gdown` and run the following command:

```bash
gdown https://drive.google.com/uc\?id\=1gWO-Z4NXG2uWwsZPecjWByaZVtgJ0XNb
```

### Hugging Face
We also release our pretrained model on [Hugging Face](https://huggingface.co/cisl-hf/BEVCalib):
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download cisl-hf/BEVCalib --revision kitti-bev-calib --local-dir YOUR_LOCAL_PATH
```

## Training

### Quick Start

**🚀 Easiest way - Use start_training.sh:**

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# Train on B26A dataset (quick validation)
bash start_training.sh B26A v1

# Train on all_training_data (full training)
bash start_training.sh all v1

# Train with DDP (distributed data parallel)
bash start_training.sh B26A v1 --ddp

# full bash command
export BEV_ZBOUND_STEP=2.0 && bash start_training.sh all v4_z10_rotation --bs 16 --lr 1e-4 --ddp --nnodes 2 --fg --angle 10 --trans 0.3 --rotation_only --enable_axis_loss --weight_axis_rotation 0.3

# Train with custom learning rate
bash start_training.sh B26A v2 --lr 0.0001
```

**📊 Batch training - Config-driven multiple experiments:**

```bash
# Train multiple experiments using config file (Z-resolution ablation)
bash batch_train.sh configs/batch_train_5deg.yaml

# Rotation-only experiments
bash batch_train.sh configs/batch_train_10deg_rotation.yaml

# Learning rate ablation
bash batch_train.sh configs/batch_train_lr_ablation.yaml

# Dry-run to preview commands
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml

# Force re-run (ignore existing experiment directories)
bash batch_train.sh --force configs/batch_train_5deg.yaml

# Skip experiments matching a regex pattern
bash batch_train.sh --skip-pattern "baseline" configs/batch_train_5deg.yaml

# Create custom config
cp configs/batch_train_5deg.yaml configs/my_experiments.yaml
bash batch_train.sh configs/my_experiments.yaml
```

**Config file features:**
- ✅ YAML-driven (no need to edit script code)
- ✅ **defaults 继承机制**: 消除重复配置，配置文件减少50-60%（v2.1+）
- ✅ **智能 TensorBoard**: 监控当前实验具体目录，清晰聚焦（v2.1+）
- ✅ **自动跳过已完成实验**: 检测 `train.log` 存在则跳过，`--force` 可覆盖
- ✅ **实验级控制**: YAML `skip: true` / `--skip-pattern` 正则过滤
- ✅ Supports all `start_training.sh` options (including multi-node DDP)
- ✅ Serial execution with auto resource management
- ✅ Pre-configured templates for common experiments
- ✅ Multi-node training support with auto-config generation

**Multi-node training:**
```bash
# Auto-generate configs for 2 machines
cd configs && bash generate_multinode_configs.sh

# Or use SLURM
sbatch configs/run_batch_train.slurm configs/batch_train_multinode_slurm.yaml
```

See [MULTINODE_TRAINING.md](MULTINODE_TRAINING.md) for complete multi-node guide.

**🔧 Advanced - Use train_universal.sh for detailed configuration:**

```bash
# Train from scratch with custom parameters
bash train_universal.sh scratch \
  --dataset_root /path/to/dataset \
  --cuda_device 0 \
  --angle_range_deg 10 \
  --trans_range 0.5 \
  --log_suffix small_10deg_v1 \
  --learning_rate 0.0002

# Fine-tune from KITTI pretrained model
bash train_universal.sh finetune \
  --dataset_root /path/to/dataset \
  --cuda_device 1

# Resume training from last checkpoint
bash train_universal.sh resume \
  --dataset_root /path/to/dataset
```

**📚 Complete training guide:** See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for:
- Training script documentation
- Parameter recommendations
- Monitoring and debugging
- Best practices
- FAQ

### Training on KITTI-Odometry

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

```bash
# Prepare dataset
python tools/prepare_custom_dataset.py \
    --bag_dir /path/to/bags \
    --config_dir /path/to/config \
    --output_dir ./data/custom_dataset

# Train (automatically detects all sequences)
bash start_training.sh custom v1
# Set dataset path:
CUSTOM_DATASET=./data/custom_dataset bash start_training.sh custom v1
```

See [CUSTOM_DATASET_TRAINING.md](CUSTOM_DATASET_TRAINING.md) for detailed training guide.

### Monitoring Training

```bash
# GPU status
nvidia-smi -l 1

# Training process
ps aux | grep train_kitti

# Real-time log
tail -f logs/B26A/model_small_10deg_v1/train.log

# TensorBoard
tensorboard --logdir logs/ --port 6006
```

### Stop Training

```bash
bash stop_training.sh
# or
pkill -f train_kitti
```

## Evaluation

### Quick Checkpoint Evaluation

**evaluate_checkpoint.py** - Evaluate saved checkpoints with detailed error reports and visualizations.

**Basic usage:**
```bash
python evaluate_checkpoint.py \
    --ckpt_path logs/model/checkpoint/ckpt_400.pth \
    --dataset_root ./data/custom_dataset \
    --angle_range_deg 10.0 \
    --trans_range 0.5 \
    --batch_size 8 \
    --max_batches 0  # 0 = evaluate all
```

**Features:**
- ✅ Detailed error statistics (translation, rotation, RPY decomposition)
- ✅ Per-sample visualizations (Init/GT/Pred comparison)
- ✅ Extrinsics matrices and error reports saved to text file
- ✅ Custom perturbation ranges
- ✅ Batch processing support

**Output:**
```
logs/model/checkpoint/ckpt_400_eval/
├── sample_0000_projection.png     # Init/GT/Pred comparison
├── sample_0001_projection.png
├── ...
└── extrinsics_and_errors.txt      # Detailed error report
```

### Cross-Dataset Generalization Testing

Evaluate model trained on dataset A on unseen dataset B:

```bash
export HF_HUB_OFFLINE=1  # For offline environments

python evaluate_checkpoint.py \
    --ckpt_path logs/train_model/checkpoint/ckpt_400.pth \
    --dataset_root /path/to/unseen_test_data \
    --output_dir /path/to/generalization_eval/ckpt_400 \
    --use_full_dataset \
    --max_batches 0 \
    --angle_range_deg 5.0 \
    --trans_range 0.3 \
    --vis_interval 50
```

**Key parameters:**
- `--use_full_dataset`: Use all data (no train/val split for cross-dataset testing)
- `--output_dir`: Custom output directory (avoid overwriting training eval results)
- `--vis_interval 50`: Save visualization every 50 frames (reduce IO for large datasets)

### KITTI Inference

```bash
python kitti-bev-calib/inference_kitti.py \
    --log_dir ./logs/inference \
    --dataset_root /path/to/kitti-odometry \
    --ckpt_path ./ckpt/kitti.pth \
    --angle_range_deg 20.0 \
    --trans_range 1.5 \
    --batch_size 16
```

## Performance Analysis Tools

### Quick Analysis

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# Analyze 5deg experiments (fast - use existing test results)
bash utils/scripts/quick_analyze.sh 5deg --skip-test

# Analyze 10deg rotation experiments (training only)
bash utils/scripts/quick_analyze.sh 10deg --only-train
```

### Complete Analysis

```bash
# Full analysis: training + test evaluation + visualization + report
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml

# Training analysis only (2-3 seconds)
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --only-train

# Skip test evaluation, use existing results (2-3 seconds)
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --skip-test
```

### Features

- ✅ **Automatic training log parsing** (full calibration & rotation-only modes)
- ✅ **Test set generalization evaluation** (automatic checkpoint evaluation)
- ✅ **Multi-experiment comparison** (visualize any number of experiments)
- ✅ **Feishu-compatible reports** (detailed Markdown reports)
- ✅ **Config-driven** (quickly switch experiment groups)

### Output

Analysis generates in configured `output_dir`:

**Report**
- `ANALYSIS_REPORT.md` - Complete Feishu-compatible Markdown report

**Visualizations**
- `convergence_curves.png` - Convergence curves (4 subplots)
- `component_breakdown.png` - Error component breakdown
- `generalization_comparison.png` - Generalization comparison (if test eval)

**Test Evaluation**
- `logs/B26A/{model_dir}/test_data_eval/` - Detailed evaluation results

### Custom Experiments

Create `utils/configs/my_experiment.yaml`:

```yaml
experiments:
  - name: "exp1"
    model_dir: "model_small_5deg_v1"
    zbound_step: 4.0
    checkpoint: "ckpt_400.pth"
  
  - name: "exp2"
    model_dir: "model_small_10deg_v1"
    zbound_step: 4.0
    checkpoint: "ckpt_400.pth"
```

Run:
```bash
python utils/scripts/analyze_experiments.py --config utils/configs/my_experiment.yaml
```

**📊 Complete analysis guide:** See [utils/ANALYSIS_GUIDE.md](utils/ANALYSIS_GUIDE.md) for:
- Configuration file system
- Module API documentation
- Usage examples
- Best practices

## Directory Structure

```
BEVCalib/
├── README.md                         # This file
├── TRAINING_GUIDE.md                 # 🌟 Complete training guide
├── requirements.txt
│
├── kitti-bev-calib/                  # Core training code
│   ├── train_kitti.py                # Main training script
│   ├── inference_kitti.py            # Inference script
│   └── ...
│
├── tools/                            # Dataset preparation tools
│   ├── prepare_custom_dataset.py     # Convert ROS bags to KITTI format
│   ├── validate_kitti_odometry.py
│   └── view_pointcloud.py
│
├── configs/                          # 🌟 Training & analysis configs
│   ├── batch_train_5deg.yaml         # 5deg Z-ablation training
│   ├── batch_train_10deg_rotation.yaml  # Rotation-only training
│   ├── batch_train_lr_ablation.yaml  # Learning rate ablation
│   └── README.md                     # Config documentation
│
├── utils/                            # Utilities and analysis tools
│   ├── scripts/
│   │   ├── analyze_experiments.py    # Analysis entry script
│   │   └── quick_analyze.sh          # Quick analysis shortcut
│   │
│   ├── configs/                      # Analysis configs
│   │   ├── experiment_config.yaml
│   │   └── experiment_config_rotation.yaml
│   │
│   ├── analysis/                     # Analysis modules
│   │   ├── training_analyzer.py
│   │   ├── test_evaluator.py
│   │   ├── report_generator.py
│   │   ├── visualizer.py
│   │   └── README.md
│   │
│   └── ANALYSIS_GUIDE.md             # 🌟 Complete analysis guide
│
├── start_training.sh                 # 🚀 Quick start training
├── train_universal.sh                # 🔧 Universal training script
├── batch_train.sh                    # 📊 Config-driven batch training
├── stop_training.sh
│
├── evaluate_checkpoint.py            # Checkpoint evaluation tool
│
├── logs/                             # Training logs (auto-generated)
│   ├── B26A/
│   │   ├── model_small_5deg_v1/
│   │   └── model_small_10deg_v1/
│   └── all_training_data/
│
├── analysis_results/                 # Analysis results (auto-generated)
│   ├── ANALYSIS_REPORT.md
│   ├── convergence_curves.png
│   ├── component_breakdown.png
│   └── generalization_comparison.png
│
└── ckpt/                             # Pretrained models
    └── kitti.pth
```

## Quick Command Reference

```bash
# ========== Training ==========
# Quick start (B26A)
bash start_training.sh B26A v1

# Quick start (all_training_data)
bash start_training.sh all v1

# Batch training (config-driven)
bash batch_train.sh configs/batch_train_5deg.yaml
bash batch_train.sh configs/batch_train_lr_ablation.yaml

# Custom training
bash train_universal.sh scratch --dataset_root /path/to/data --cuda_device 0

# ========== Monitoring ==========
# GPU status
nvidia-smi -l 1

# Real-time log
tail -f logs/B26A/model_small_10deg_v1/train.log

# TensorBoard
tensorboard --logdir logs/ --port 6006

# ========== Evaluation ==========
# Evaluate checkpoint
python evaluate_checkpoint.py \
    --ckpt_path logs/.../ckpt_400.pth \
    --dataset_root /path/to/data

# ========== Analysis ==========
# Quick analysis (5deg)
bash utils/scripts/quick_analyze.sh 5deg --skip-test

# Quick analysis (10deg rotation)
bash utils/scripts/quick_analyze.sh 10deg --only-train

# Complete analysis
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml

# ========== Stop Training ==========
bash stop_training.sh
```

## Acknowledgement

BEVCalib appreciates the following great open-source projects: [BEVFusion](https://github.com/mit-han-lab/bevfusion), [LCCNet](https://github.com/IIPCVLAB/LCCNet), [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [spconv](https://github.com/traveller59/spconv), and [Deformable Attention](https://github.com/lucidrains/deformable-attention).

## Citation

```bibtex
@inproceedings{bevcalib310,
      title={BEVCALIB: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representations}, 
      author={Weiduo Yuan and Jerry Li and Justin Yue and Divyank Shah and Konstantinos Karydis and Hang Qiu},
      booktitle={9th Annual Conference on Robot Learning},
      year={2025},
}
```

---

**License**: See [LICENSE](LICENSE)  
**Contact**: For questions or issues, please open an issue on GitHub

**📚 Important Guides**:
- **Training**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Analysis**: [utils/ANALYSIS_GUIDE.md](utils/ANALYSIS_GUIDE.md)
- **Custom Dataset**: [PREPARE_CUSTOM_DATASET.md](PREPARE_CUSTOM_DATASET.md)
