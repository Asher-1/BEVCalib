#!/bin/bash
# BEVCalib Training Script for B26A Dataset
# 
# Usage:
#   bash train_B26A.sh [mode]
#
# Modes:
#   scratch   - Train from scratch (default)
#   finetune  - Fine-tune from KITTI pretrained model
#   resume    - Resume from last checkpoint

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
                echo "  bash train_B26A.sh $1"
                exit 1
            }
            echo "✓ Environment activated"
        fi
    else
        echo "❌ Conda not found"
        echo ""
        echo "Please activate your Python environment manually, then run:"
        echo "  bash train_B26A.sh $1"
        exit 1
    fi
fi

# Dataset configuration
DATASET_ROOT="/home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data"
LOG_DIR="./logs/B26A_model"
KITTI_PRETRAIN="./ckpt/kitti.pth"

# Training mode
MODE=${1:-scratch}

echo "========================================"
echo "BEVCalib Training for B26A Dataset"
echo "========================================"
echo "Dataset: $DATASET_ROOT"
echo "Mode: $MODE"
echo "========================================"
echo ""

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
            --label B26A_20_1.5 \
            --batch_size 4 \
            --num_epochs 500 \
            --save_ckpt_per_epoches 40 \
            --angle_range_deg 20 \
            --trans_range 1.5 \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr 1e-4 \
            --step_size 80 \
            --use_custom_dataset 1 \
            --max_range 90.0
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
        
        python kitti-bev-calib/train_kitti.py \
            --log_dir "${LOG_DIR}_finetuned" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$KITTI_PRETRAIN" \
            --label B26A_finetuned \
            --batch_size 4 \
            --num_epochs 50 \
            --save_ckpt_per_epoches 10 \
            --lr 5e-5 \
            --scheduler 1 \
            --step_size 20 \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --use_custom_dataset 1 \
            --max_range 90.0
        ;;
    
    resume)
        echo "Resuming from last checkpoint..."
        
        # Find the last checkpoint
        LAST_CKPT=$(find "$LOG_DIR/checkpoints" -name "*.pth" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
        
        if [ -z "$LAST_CKPT" ]; then
            echo "❌ Error: No checkpoint found in $LOG_DIR/checkpoints"
            echo "Cannot resume training."
            exit 1
        fi
        
        echo "✓ Found checkpoint: $LAST_CKPT"
        echo ""
        
        python kitti-bev-calib/train_kitti.py \
            --log_dir "$LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$LAST_CKPT" \
            --label B26A_20_1.5 \
            --batch_size 8 \
            --num_epochs 150 \
            --save_ckpt_per_epoches 15 \
            --angle_range_deg 20 \
            --trans_range 1.5 \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr 1e-4 \
            --step_size 50 \
            --use_custom_dataset 1 \
            --max_range 90.0
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
echo "  tensorboard --logdir ./logs"
echo ""
echo "To evaluate:"
echo "  python kitti-bev-calib/inference_kitti.py \\"
echo "    --dataset_root $DATASET_ROOT \\"
echo "    --ckpt_path $LOG_DIR/checkpoints/best_model.pth"
