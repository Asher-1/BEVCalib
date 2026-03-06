#!/bin/bash
# BEVCalib Universal Training Script
# Supports multiple datasets, single/multi-node DDP, organized log structure
# 
# Usage:
#   bash train_universal.sh [mode] [options]
#
# Modes:
#   scratch   - Train from scratch (default)
#   finetune  - Fine-tune from KITTI pretrained model
#   resume    - Resume from last checkpoint
#
# Options:
#   --dataset_root PATH      - Dataset root directory (required)
#   --dataset_name NAME      - Dataset name for log organization (auto-detected from path if not specified)
#   --cuda_device ID         - CUDA device ID (e.g., 0, 1, 2) (default: auto-detect)
#   --tensorboard_port PORT  - TensorBoard port (default: 6006, will auto-increment if in use)
#   --log_suffix SUFFIX      - Suffix for log directory (useful for multiple runs)
#   --angle_range_deg DEG    - Rotation perturbation range in degrees (default: 20)
#   --trans_range M          - Translation perturbation range in meters (default: 1.5)
#   --batch_size N           - Batch size per GPU (default: 16)
#   --ddp N                  - Enable DDP with N GPUs per node
#   --nnodes N               - Number of nodes for multi-node DDP (default: 1)
#   --node_rank R            - Rank of current node (0=master)
#   --master_addr ADDR       - Master node IP address
#   --master_port PORT       - Master node port (default: 29500)
#   --compile                - Enable torch.compile
#
# Examples:
#   # Single GPU training
#   bash train_universal.sh scratch --dataset_root /path/to/data --dataset_name B26A
#
#   # Single-node DDP (2 GPUs)
#   bash train_universal.sh scratch --dataset_root /path/to/data --ddp 2
#
#   # Multi-node DDP (2 nodes x 2 GPUs)
#   # On master:
#   bash train_universal.sh scratch --dataset_root /path/to/data --ddp 2 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1
#   # On worker:
#   bash train_universal.sh scratch --dataset_root /path/to/data --ddp 2 --nnodes 2 --node_rank 1 --master_addr 10.0.0.1

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
                echo "  bash train_universal.sh $@"
                exit 1
            }
            echo "✓ Environment activated"
        fi
    else
        echo "❌ Conda not found"
        echo ""
        echo "Please activate your Python environment manually, then run:"
        echo "  bash train_universal.sh $@"
        exit 1
    fi
fi

# Parse arguments
MODE=${1:-scratch}
if [ $# -gt 0 ]; then
    shift  # Remove mode from arguments
fi

# Default configuration
DATASET_ROOT=""  # Required, must be specified
DATASET_NAME=""  # Auto-detected from dataset path if not specified
LOG_BASE_DIR="./logs"  # Base log directory
KITTI_PRETRAIN="./ckpt/kitti.pth"
CUDA_DEVICE=""
TENSORBOARD_PORT=""
LOG_SUFFIX=""
ANGLE_RANGE_DEG="10"
TRANS_RANGE="0.3"
BATCH_SIZE="16"
DDP_NGPUS=""
USE_COMPILE=0
NNODES="1"
NODE_RANK="0"
MASTER_ADDR=""
MASTER_PORT="29500"
RDZV_TIMEOUT="600"

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
        --dataset_name)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --dataset_name requires a name"
                exit 1
            fi
            DATASET_NAME="$2"
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
        --batch_size)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --batch_size requires a value"
                exit 1
            fi
            BATCH_SIZE="$2"
            shift 2
            ;;
        --ddp)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --ddp requires number of GPUs"
                exit 1
            fi
            DDP_NGPUS="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --rdzv_timeout)
            RDZV_TIMEOUT="$2"
            shift 2
            ;;
        --compile)
            USE_COMPILE=1
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Available options: --dataset_root, --dataset_name, --cuda_device, --tensorboard_port, --log_suffix, --angle_range_deg, --trans_range, --batch_size, --ddp, --nnodes, --node_rank, --master_addr, --master_port, --rdzv_timeout, --compile"
            exit 1
            ;;
    esac
done

# Check if dataset_root is specified
if [ -z "$DATASET_ROOT" ]; then
    echo "❌ Error: --dataset_root is required"
    echo ""
    echo "Usage:"
    echo "  bash train_universal.sh [mode] --dataset_root PATH [--dataset_name NAME] [options]"
    echo ""
    echo "Examples:"
    echo "  # B26A dataset"
    echo "  bash train_universal.sh scratch --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data"
    echo ""
    echo "  # All training data"
    echo "  bash train_universal.sh scratch --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data"
    exit 1
fi

# Auto-detect dataset name from path if not specified
if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME=$(basename "$DATASET_ROOT")
    echo "ℹ️  Dataset name auto-detected: $DATASET_NAME"
fi

# Construct log directory with dataset hierarchy
# Structure: logs/{dataset_name}/model_{suffix}/
LOG_DIR="${LOG_BASE_DIR}/${DATASET_NAME}/model"

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

PYTORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda or 'N/A')" 2>/dev/null || echo "unknown")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
AVAIL_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")

if [ -n "$DDP_NGPUS" ]; then
    if [ "$NNODES" -gt 1 ]; then
        COMPUTE_STR="DDP Multi-Node (${NNODES} nodes x ${DDP_NGPUS} GPUs = $((NNODES * DDP_NGPUS)) total)"
    else
        COMPUTE_STR="DDP Standalone (${DDP_NGPUS} GPUs)"
    fi
elif [ -n "$CUDA_DEVICE" ]; then
    COMPUTE_STR="Single GPU (cuda:${CUDA_DEVICE})"
else
    COMPUTE_STR="Single GPU (auto)"
fi

_W=70
_print_row() { printf "│  %-18s %-${_W}s│\n" "$1" "$2"; }
_print_sep() { printf "├"; printf '─%.0s' $(seq 1 $((_W + 21))); printf "┤\n"; }
_print_top() { printf "┌"; printf '─%.0s' $(seq 1 $((_W + 21))); printf "┐\n"; }
_print_bot() { printf "└"; printf '─%.0s' $(seq 1 $((_W + 21))); printf "┘\n"; }

echo ""
_print_top
printf "│%*s│\n" $((_W + 21)) ""
printf "│%*s%s%*s│\n" $(( (_W + 21 - 32) / 2 )) "" "BEVCalib Training Configuration" $(( (_W + 21 - 32 + 1) / 2 )) ""
printf "│%*s│\n" $((_W + 21)) ""
_print_sep
_print_row "Dataset:"       "$DATASET_NAME"
_print_row "Dataset Path:"  "$DATASET_ROOT"
_print_row "Mode:"          "$MODE"
_print_sep
_print_row "Batch Size:"    "$BATCH_SIZE"
_print_row "Angle Range:"   "±${ANGLE_RANGE_DEG}°"
_print_row "Trans Range:"   "${TRANS_RANGE}m"
_print_sep
_print_row "Compute:"       "$COMPUTE_STR"
_print_row "GPU:"           "${GPU_NAME} (${GPU_MEM}MB) x${AVAIL_GPUS}"
_print_row "PyTorch:"       "$PYTORCH_VER"
_print_row "CUDA:"          "$CUDA_VER"
_print_row "torch.compile:" "$([ "$USE_COMPILE" -eq 1 ] && echo 'enabled' || echo 'disabled')"
if [ "$NNODES" -gt 1 ]; then
_print_sep
_print_row "Node Rank:"     "$NODE_RANK"
_print_row "Master:"        "${MASTER_ADDR}:${MASTER_PORT}"
_print_row "RDZV Timeout:"  "${RDZV_TIMEOUT}s"
fi
_print_sep
_print_row "Log Directory:"     "$LOG_DIR"
_print_row "Log File:"          "$TRAIN_LOG_FILE"
_print_row "TensorBoard Port:"  "$TENSORBOARD_PORT"
_print_bot
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

COMPILE_FLAG=""
if [ "$USE_COMPILE" -eq 1 ]; then
    COMPILE_FLAG="--compile 1"
fi

if [ -n "$DDP_NGPUS" ]; then
    if [ "$NNODES" -gt 1 ]; then
        # 检测NCCL使用的网络接口
        # 优先级: 默认路由接口 > 任意有IP的非loopback接口 > 排除模式
        DETECTED_IF=$(ip -4 route show default 2>/dev/null | awk '{print $5}' | head -1)
        if [ -z "$DETECTED_IF" ]; then
            DETECTED_IF=$(ip -4 addr show scope global 2>/dev/null | grep -oP '^\d+:\s+\K[^:@\s]+' | head -1)
        fi
        if [ -n "$DETECTED_IF" ]; then
            # 验证接口确实存在
            if ip link show "$DETECTED_IF" >/dev/null 2>&1; then
                export NCCL_SOCKET_IFNAME=$DETECTED_IF
            else
                export NCCL_SOCKET_IFNAME="^lo,docker0"
            fi
        else
            # 无法检测具体接口，使用排除模式让NCCL自动选择
            export NCCL_SOCKET_IFNAME="^lo,docker0"
        fi
        export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,NET

        LAUNCHER="torchrun \
            --nproc_per_node=$DDP_NGPUS \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            --rdzv_conf timeout=$RDZV_TIMEOUT"
        echo "Multi-node DDP: ${NNODES} nodes x ${DDP_NGPUS} GPUs/node, node_rank=$NODE_RANK"
        echo "Master: ${MASTER_ADDR}:${MASTER_PORT}, rdzv_timeout=${RDZV_TIMEOUT}s"
        echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    else
        LAUNCHER="torchrun --standalone --nproc_per_node=$DDP_NGPUS"
    fi
else
    LAUNCHER="python"
fi

case $MODE in
    scratch)
        echo "Training from scratch..."
        $LAUNCHER kitti-bev-calib/train_kitti.py \
            --log_dir "$LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --label ${DATASET_NAME}_scratch \
            --batch_size $BATCH_SIZE \
            --num_epochs 400 \
            --save_ckpt_per_epoches 40 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr 2e-4 \
            --step_size 60 \
            --use_custom_dataset 1 \
            $COMPILE_FLAG
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
        
        $LAUNCHER kitti-bev-calib/train_kitti.py \
            --log_dir "$FINETUNE_LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$KITTI_PRETRAIN" \
            --label ${DATASET_NAME}_finetuned \
            --batch_size $BATCH_SIZE \
            --num_epochs 50 \
            --save_ckpt_per_epoches 10 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --lr 7.5e-5 \
            --scheduler 1 \
            --step_size 20 \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --use_custom_dataset 1 \
            $COMPILE_FLAG
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
        
        $LAUNCHER kitti-bev-calib/train_kitti.py \
            --log_dir "$LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$LAST_CKPT" \
            --label ${DATASET_NAME}_resume \
            --batch_size $BATCH_SIZE \
            --num_epochs 100 \
            --save_ckpt_per_epoches 10 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --deformable 0 \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr 1.5e-4 \
            --step_size 20 \
            --use_custom_dataset 1 \
            $COMPILE_FLAG
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
echo "To evaluate (replace N with actual epoch number, e.g., 400):"
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
LATEST_CKPT=$(ls -t $LOG_DIR/checkpoint/ckpt_*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "  python kitti-bev-calib/inference_kitti.py \\"
    echo "    --dataset_root $DATASET_ROOT \\"
    echo "    --ckpt_path $LATEST_CKPT"
fi
