#!/bin/bash
# BEVCalib Training Script for B26A Dataset
# 
# Usage:
#   bash train_B26A.sh [mode] [options]
#
# Modes:
#   scratch   - Train from scratch (default)
#   finetune  - Fine-tune from KITTI pretrained model
#   resume    - Resume from last checkpoint
#
# Options:
#   --dataset_root PATH     - Dataset root directory (default: /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data)
#   --cuda_device ID        - CUDA device ID (e.g., 0, 1, 2) (default: auto-detect)
#   --tensorboard_port PORT - TensorBoard port (default: 6006, will auto-increment if in use)
#   --log_suffix SUFFIX     - Suffix for log directory (useful for multiple runs)
#   --angle_range_deg DEG   - Rotation perturbation range in degrees (default: 20)
#   --trans_range M         - Translation perturbation range in meters (default: 1.5)
#
# Examples:
#   # Train on GPU 0 with default dataset
#   bash train_B26A.sh scratch --cuda_device 0
#
#   # Train on GPU 1 with custom dataset and TensorBoard port
#   bash train_B26A.sh scratch --cuda_device 1 --dataset_root /path/to/dataset --tensorboard_port 6007
#
#   # Train multiple datasets in parallel
#   bash train_B26A.sh scratch --cuda_device 0 --dataset_root /path/to/dataset1 --log_suffix dataset1
#   bash train_B26A.sh scratch --cuda_device 1 --dataset_root /path/to/dataset2 --log_suffix dataset2
#
#   # Train with different perturbation settings (based on paper)
#   # Small perturbation (10°, 0.5m) - for fine calibration
#   bash train_B26A.sh scratch --cuda_device 0 --angle_range_deg 10 --trans_range 0.5 --log_suffix small_10deg
#   # Medium perturbation (15°, 1.0m)
#   bash train_B26A.sh scratch --cuda_device 1 --angle_range_deg 15 --trans_range 1.0 --log_suffix medium_15deg
#   # Large perturbation (20°, 1.5m) - default, for initial calibration
#   bash train_B26A.sh scratch --cuda_device 2 --angle_range_deg 20 --trans_range 1.5 --log_suffix large_20deg

set -e  # Exit on error

# Activate conda environment if needed
if ! python -c "import torch" &> /dev/null; then
    echo "⚠️  PyTorch not found in current environment"
    
    # Try to activate bevcalib conda environment
    if command -v conda &> /dev/null; then
        echo "Activating conda environment 'bevcalib'..."
        
        # Source conda.sh to enable conda in script
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate bevcalib 2>/dev/null || {
                echo "❌ Failed to activate 'bevcalib' environment"
                echo ""
                echo "Please run this script manually after activating the environment:"
                echo "  conda activate bevcalib"
                echo "  bash train_B26A.sh $@"
                exit 1
            }
            echo "✓ Environment activated"
        fi
    else
        echo "❌ Conda not found"
        echo ""
        echo "Please activate your Python environment manually, then run:"
        echo "  bash train_B26A.sh $@"
        exit 1
    fi
fi

# Parse arguments
MODE=${1:-scratch}
if [ $# -gt 0 ]; then
    shift  # Remove mode from arguments
fi

# Default configuration
DATASET_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data"
LOG_DIR="./logs/B26A_model"
KITTI_PRETRAIN="./ckpt/kitti.pth"
CUDA_DEVICE=""
TENSORBOARD_PORT=""
LOG_SUFFIX=""
ANGLE_RANGE_DEG="20"
TRANS_RANGE="1.5"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_root)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --dataset_root requires a path"
                exit 1
            fi
            DATASET_ROOT="$2"
            shift 2
            ;;
        --cuda_device)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --cuda_device requires a device ID"
                exit 1
            fi
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --tensorboard_port)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --tensorboard_port requires a port number"
                exit 1
            fi
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        --log_suffix)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --log_suffix requires a suffix string"
                exit 1
            fi
            LOG_SUFFIX="$2"
            shift 2
            ;;
        --angle_range_deg)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --angle_range_deg requires a value"
                exit 1
            fi
            ANGLE_RANGE_DEG="$2"
            shift 2
            ;;
        --trans_range)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --trans_range requires a value"
                exit 1
            fi
            TRANS_RANGE="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Available options: --dataset_root, --cuda_device, --tensorboard_port, --log_suffix, --angle_range_deg, --trans_range"
            exit 1
            ;;
    esac
done

# Set CUDA device if specified
if [ -n "$CUDA_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    echo "✓ Using CUDA device: $CUDA_DEVICE"
else
    # Auto-detect available GPU
    if command -v nvidia-smi &> /dev/null; then
        AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ "$AVAILABLE_GPUS" -gt 0 ]; then
            echo "ℹ️  No CUDA device specified, using all available GPUs"
            echo "   Use --cuda_device to specify a specific GPU (e.g., --cuda_device 0)"
        fi
    fi
fi

# Find available TensorBoard port if not specified
if [ -z "$TENSORBOARD_PORT" ]; then
    TENSORBOARD_PORT=6006
    while lsof -Pi :$TENSORBOARD_PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
        TENSORBOARD_PORT=$((TENSORBOARD_PORT + 1))
    done
    echo "ℹ️  TensorBoard port not specified, using: $TENSORBOARD_PORT"
else
    # Check if specified port is available
    if lsof -Pi :$TENSORBOARD_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Warning: TensorBoard port $TENSORBOARD_PORT is already in use"
        echo "   Consider using a different port with --tensorboard_port"
    fi
fi

# Add suffix to log directory if specified
if [ -n "$LOG_SUFFIX" ]; then
    LOG_DIR="${LOG_DIR}_${LOG_SUFFIX}"
fi

# 创建训练目录（如果不存在）
mkdir -p "$LOG_DIR"

# 设置日志文件路径
TRAIN_LOG_FILE="$LOG_DIR/train.log"

echo "========================================"
echo "BEVCalib Training for B26A Dataset"
echo "========================================"
echo "Dataset: $DATASET_ROOT"
echo "Mode: $MODE"
if [ -n "$CUDA_DEVICE" ]; then
    echo "CUDA Device: $CUDA_DEVICE"
fi
echo "TensorBoard Port: $TENSORBOARD_PORT"
if [ -n "$LOG_SUFFIX" ]; then
    echo "Log Suffix: $LOG_SUFFIX"
fi
echo "Log Directory: $LOG_DIR"
echo "Train Log File: $TRAIN_LOG_FILE"
echo "Perturbation: ±${ANGLE_RANGE_DEG}°, ${TRANS_RANGE}m"
echo "========================================"
echo ""

# 如果不是交互式终端（即通过 nohup 运行），自动重定向日志到文件
if [ ! -t 1 ]; then
    # 重定向 stdout 和 stderr 到日志文件，同时保留副本
    exec > >(tee -a "$TRAIN_LOG_FILE") 2>&1
    echo "日志将写入: $TRAIN_LOG_FILE"
fi

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ Error: Dataset not found at $DATASET_ROOT"
    echo "Please check the path or prepare the dataset first:"
    echo "  python tools/prepare_custom_dataset.py --bag_dir ... --output_dir $DATASET_ROOT"
    exit 1
fi

# Check if sequences exist
if [ ! -d "$DATASET_ROOT/sequences" ]; then
    echo "❌ Error: sequences/ directory not found in dataset"
    exit 1
fi

echo "✓ Dataset found"
echo ""

# Check if bev_pool extension is compiled
echo "Checking bev_pool CUDA extension..."
BEV_POOL_DIR="./kitti-bev-calib/img_branch/bev_pool"
BEV_POOL_SO=$(find "$BEV_POOL_DIR" -name "*.so" 2>/dev/null)

if [ -z "$BEV_POOL_SO" ]; then
    echo "⚠️  bev_pool CUDA extension not compiled"
    echo "Compiling bev_pool extension..."
    
    cd "$BEV_POOL_DIR"
    python setup.py build_ext --inplace
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Error: Failed to compile bev_pool extension"
        echo ""
        echo "Please make sure:"
        echo "  1. PyTorch is installed: pip install torch"
        echo "  2. CUDA toolkit is installed and configured"
        echo "  3. Run manually: cd $BEV_POOL_DIR && python setup.py build_ext --inplace"
        exit 1
    fi
    
    cd - > /dev/null
    echo "✓ bev_pool extension compiled successfully"
else
    echo "✓ bev_pool extension already compiled"
fi
echo ""

case $MODE in
    scratch)
        echo "Training from scratch..."
        python kitti-bev-calib/train_kitti.py \
            --log_dir "$LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --label B26A_scratch \
            --batch_size 8 \
            --num_epochs 500 \
            --save_ckpt_per_epoches 40 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr 1e-4 \
            --step_size 80 \
            --use_custom_dataset 1
        ;;
    
    finetune)
        echo "Fine-tuning from KITTI pretrained model..."
        
        # Check if pretrained model exists
        if [ ! -f "$KITTI_PRETRAIN" ]; then
            echo "❌ Error: KITTI pretrained model not found at $KITTI_PRETRAIN"
            echo "Please download it first:"
            echo "  gdown https://drive.google.com/uc?id=1gWO-Z4NXG2uWwsZPecjWByaZVtgJ0XNb"
            exit 1
        fi
        
        echo "✓ Pretrained model found"
        echo ""
        
        FINETUNE_LOG_DIR="${LOG_DIR}_finetuned"
        
        python kitti-bev-calib/train_kitti.py \
            --log_dir "$FINETUNE_LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$KITTI_PRETRAIN" \
            --label B26A_finetuned \
            --batch_size 16 \
            --num_epochs 50 \
            --save_ckpt_per_epoches 10 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --lr 5e-5 \
            --scheduler 1 \
            --step_size 20 \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --use_custom_dataset 1
        ;;
    
    resume)
        echo "Resuming from last checkpoint..."
        
        # Find the last checkpoint
        LAST_CKPT=$(find "$LOG_DIR" -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
        
        if [ -z "$LAST_CKPT" ]; then
            echo "❌ Error: No checkpoint found in $LOG_DIR"
            echo "Cannot resume training."
            exit 1
        fi
        
        echo "✓ Found checkpoint: $LAST_CKPT"
        echo ""
        
        python kitti-bev-calib/train_kitti.py \
            --log_dir "$LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$LAST_CKPT" \
            --label B26A_resume \
            --batch_size 16 \
            --num_epochs 150 \
            --save_ckpt_per_epoches 15 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr 1e-4 \
            --step_size 50 \
            --use_custom_dataset 1
        ;;
    
    *)
        echo "❌ Error: Unknown mode '$MODE'"
        echo "Available modes: scratch, finetune, resume"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "✓ Training completed!"
echo "========================================"
echo "Logs: $LOG_DIR"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir ./logs --port $TENSORBOARD_PORT"
echo ""
echo "To evaluate (replace N with actual epoch number, e.g., 500):"
echo "  python kitti-bev-calib/inference_kitti.py \\"
echo "    --dataset_root $DATASET_ROOT \\"
echo "    --ckpt_path $LOG_DIR/checkpoint/ckpt_N.pth \\"
echo "    --angle_range_deg $ANGLE_RANGE_DEG \\"
echo "    --trans_range $TRANS_RANGE"
echo ""
echo "Available checkpoints:"
echo "  ls $LOG_DIR/checkpoint/"
echo ""
echo "Quick inference example:"
LATEST_CKPT=\$(ls -t $LOG_DIR/checkpoint/ckpt_*.pth 2>/dev/null | head -1)
if [ -n "\$LATEST_CKPT" ]; then
    echo "  python kitti-bev-calib/inference_kitti.py \\"
    echo "    --dataset_root $DATASET_ROOT \\"
    echo "    --ckpt_path \$LATEST_CKPT"
fi
