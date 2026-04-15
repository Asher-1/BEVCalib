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
#   --learning_rate LR       - Initial learning rate (default: mode-specific, scratch=2e-4, finetune=7.5e-5, resume=1.5e-4)
#   --ddp N                  - Enable DDP with N GPUs per node
#   --nnodes N               - Number of nodes for multi-node DDP (default: 1)
#   --node_rank R            - Rank of current node (0=master)
#   --master_addr ADDR       - Master node IP address
#   --master_port PORT       - Master node port (default: 29500)
#   --compile                - Enable torch.compile
#   --rotation_only          - Only optimize rotation (skip translation optimization, use design values)
#   --enable_axis_loss       - Enable per-axis rotation loss (Roll/Pitch/Yaw independent supervision)
#   --weight_axis_rotation W - Weight for per-axis rotation loss (default: 0.3)
#   --lr_schedule TYPE       - LR scheduler: step or cosine_warm_restarts (default: step)
#   --warmup_epochs N        - Linear warmup epochs (default: 5)
#   --backbone_lr_scale S    - LR multiplier for pretrained backbone (default: 0.1)
#   --cosine_T0 N            - CosineAnnealingWarmRestarts T_0 (default: 50)
#   --cosine_Tmult N         - CosineAnnealingWarmRestarts T_mult (default: 2)
#   --drop_path_rate R       - Stochastic depth rate (default: 0.1)
#   --head_dropout R         - Dropout before prediction heads (default: 0.1)
#   --perturb_distribution D - Perturbation distribution: uniform or truncated_normal (default: uniform)
#   --per_axis_prob P        - Probability of single-axis perturbation (default: 0.0)
#   --augment_pc_jitter S    - Point cloud jitter sigma in meters (default: 0.0)
#   --augment_pc_dropout R   - Point cloud dropout ratio (default: 0.0)
#   --augment_color_jitter S - Image color jitter strength (default: 0.0)
#   --augment_intrinsic S    - Camera intrinsic augmentation strength (default: 0.0, e.g. 0.05=±5%)
#   --eval_angle_range_deg D - Evaluation perturbation angle (default: same as training angle)
#   --early_stopping_patience N - Early stopping patience in eval cycles (default: 0)
#   --seed N                 - Global random seed for reproducibility (default: 42)
#   --pretrain_ckpt PATH     - Pretrained checkpoint for finetune/refine training
#   --num_epochs N           - Total training epochs (overrides default per mode)
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
            conda activate bevcalib 2>/dev/null || conda activate bevcalib310 2>/dev/null || {
                echo "❌ Failed to activate 'bevcalib' or 'bevcalib310' environment"
                echo ""
                echo "Please run this script manually after activating the environment:"
                echo "  conda activate bevcalib310"
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
LEARNING_RATE=""
DDP_NGPUS=""
USE_COMPILE=0
ROTATION_ONLY=0
ENABLE_AXIS_LOSS=0
WEIGHT_AXIS_ROTATION=""
LR_SCHEDULE=""
WARMUP_EPOCHS=""
BACKBONE_LR_SCALE=""
COSINE_T0=""
COSINE_TMULT=""
DROP_PATH_RATE=""
HEAD_DROPOUT=""
PERTURB_DISTRIBUTION=""
PER_AXIS_PROB=""
AUGMENT_PC_JITTER=""
AUGMENT_PC_DROPOUT=""
AUGMENT_COLOR_JITTER=""
AUGMENT_INTRINSIC=""
EVAL_ANGLE_RANGE_DEG=""
EARLY_STOPPING_PATIENCE=""
SEED=""
PRETRAIN_CKPT=""
NUM_EPOCHS_OVERRIDE=""
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
        --learning_rate|--lr)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --learning_rate requires a value"
                exit 1
            fi
            LEARNING_RATE="$2"
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
        --rotation_only)
            ROTATION_ONLY=1
            shift
            ;;
        --enable_axis_loss)
            ENABLE_AXIS_LOSS=1
            shift
            ;;
        --weight_axis_rotation)
            if [ $# -lt 2 ]; then
                echo "❌ Error: --weight_axis_rotation requires a value"
                exit 1
            fi
            WEIGHT_AXIS_ROTATION="$2"
            shift 2
            ;;
        --axis_weights)
            AXIS_WEIGHTS="$2"; shift 2 ;;
        --lr_schedule)
            LR_SCHEDULE="$2"; shift 2 ;;
        --warmup_epochs)
            WARMUP_EPOCHS="$2"; shift 2 ;;
        --backbone_lr_scale)
            BACKBONE_LR_SCALE="$2"; shift 2 ;;
        --cosine_T0)
            COSINE_T0="$2"; shift 2 ;;
        --cosine_Tmult)
            COSINE_TMULT="$2"; shift 2 ;;
        --drop_path_rate)
            DROP_PATH_RATE="$2"; shift 2 ;;
        --head_dropout)
            HEAD_DROPOUT="$2"; shift 2 ;;
        --perturb_distribution)
            PERTURB_DISTRIBUTION="$2"; shift 2 ;;
        --per_axis_prob)
            PER_AXIS_PROB="$2"; shift 2 ;;
        --per_axis_weights)
            PER_AXIS_WEIGHTS="$2"; shift 2 ;;
        --augment_pc_jitter)
            AUGMENT_PC_JITTER="$2"; shift 2 ;;
        --augment_pc_dropout)
            AUGMENT_PC_DROPOUT="$2"; shift 2 ;;
        --augment_color_jitter)
            AUGMENT_COLOR_JITTER="$2"; shift 2 ;;
        --augment_intrinsic)
            AUGMENT_INTRINSIC="$2"; shift 2 ;;
        --eval_angle_range_deg)
            EVAL_ANGLE_RANGE_DEG="$2"; shift 2 ;;
        --early_stopping_patience)
            EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --pretrain_ckpt)
            PRETRAIN_CKPT="$2"; shift 2 ;;
        --num_epochs)
            NUM_EPOCHS_OVERRIDE="$2"; shift 2 ;;
        --use_geodesic_loss)
            USE_GEODESIC_LOSS="$2"; shift 2 ;;
        --use_mlp_head)
            USE_MLP_HEAD="$2"; shift 2 ;;
        --use_deformable)
            USE_DEFORMABLE="$2"; shift 2 ;;
        --use_foundation_depth)
            USE_FOUNDATION_DEPTH="$2"; shift 2 ;;
        --depth_model_type)
            DEPTH_MODEL_TYPE="$2"; shift 2 ;;
        --fd_mode)
            FD_MODE="$2"; shift 2 ;;
        --depth_sup_alpha)
            DEPTH_SUP_ALPHA="$2"; shift 2 ;;
        --bev_pool_factor)
            BEV_POOL_FACTOR="$2"; shift 2 ;;
        --max_frames_per_seq)
            MAX_FRAMES_PER_SEQ="$2"; shift 2 ;;
        --eval_epoches)
            EVAL_EPOCHES="$2"; shift 2 ;;
        --grad_accum_steps)
            GRAD_ACCUM_STEPS="$2"; shift 2 ;;
        *)
            echo "❌ Unknown option: $1"
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


mkdir -p "$LOG_DIR"

# 日志文件路径 (train_kitti.py 的 tprint() 直接写入 $LOG_DIR/train.log，
# 不依赖 shell tee, 确保多机 DDP/torchrun 环境下日志完整)
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

_BOX_W=91
_hline() { printf '%0.s─' $(seq 1 $_BOX_W); }
_print_top() { printf "┌"; _hline; printf "┐\n"; }
_print_bot() { printf "└"; _hline; printf "┘\n"; }
_print_sep() { printf "├"; _hline; printf "┤\n"; }
_print_empty() { printf "│%-${_BOX_W}s│\n" ""; }
_print_center() {
    local text="$1"
    local dw=${#text}
    local pad_l=$(( (_BOX_W - dw) / 2 ))
    local pad_r=$(( _BOX_W - dw - pad_l ))
    printf "│%*s%s%*s│\n" "$pad_l" "" "$text" "$pad_r" ""
}
_print_row() {
    local label="$1" value="$2"
    local content
    content=$(printf "  %-18s %s" "$label" "$value")
    local byte_len=${#content}
    local char_len
    char_len=$(printf "%s" "$content" | LC_ALL=en_US.UTF-8 wc -m)
    local extra=$(( byte_len - char_len ))
    printf "│%-$((_BOX_W + extra))s│\n" "$content"
}
_NEED_SEP=0
_maybe_sep() {
    if [ "$_NEED_SEP" -eq 1 ]; then
        _print_sep
    fi
    _NEED_SEP=0
}
_section_end() { _NEED_SEP=1; }

_print_config_box() {
echo ""
_print_top
_print_empty
_print_center "BEVCalib Training Configuration"
_print_empty
_print_sep

_print_row "Dataset:"       "$DATASET_NAME"
_print_row "Dataset Path:"  "$DATASET_ROOT"
_print_row "Mode:"          "$MODE"
[ -n "$LOG_SUFFIX" ] && \
_print_row "Version:"       "$LOG_SUFFIX"
_section_end

_maybe_sep
_print_row "Batch Size:"    "$BATCH_SIZE"
_print_row "Angle Range:"   "+/-${ANGLE_RANGE_DEG} deg"
_print_row "Trans Range:"   "${TRANS_RANGE}m"
[ -n "$LEARNING_RATE" ] && \
_print_row "Learning Rate:" "$LEARNING_RATE"
_print_row "Rotation Only:" "$([ "$ROTATION_ONLY" -eq 1 ] && echo 'yes (skip translation)' || echo 'no (optimize both)')"
_section_end

_maybe_sep
_print_row "Axis Loss:"    "$([ "$ENABLE_AXIS_LOSS" -eq 1 ] && echo "enabled (weight=${WEIGHT_AXIS_ROTATION:-0.3})" || echo 'disabled')"
[ -n "$AXIS_WEIGHTS" ] && \
_print_row "Axis Weights:"  "R/P/Y = ${AXIS_WEIGHTS}"
_section_end

_HAS_LR_SECTION=0
if [ -n "$LR_SCHEDULE" ]; then
    _maybe_sep; _HAS_LR_SECTION=1
    _print_row "LR Schedule:"   "$LR_SCHEDULE"
fi
[ -n "$WARMUP_EPOCHS" ] && {
    [ "$_HAS_LR_SECTION" -eq 0 ] && { _maybe_sep; _HAS_LR_SECTION=1; }
    _print_row "Warmup Epochs:" "$WARMUP_EPOCHS"
}
[ -n "$BACKBONE_LR_SCALE" ] && {
    [ "$_HAS_LR_SECTION" -eq 0 ] && { _maybe_sep; _HAS_LR_SECTION=1; }
    _print_row "Backbone LR:"   "x${BACKBONE_LR_SCALE}"
}
[ "$LR_SCHEDULE" = "cosine_warm_restarts" ] && [ -n "$COSINE_T0" ] && \
    _print_row "Cosine T0/Tm:"  "T0=${COSINE_T0}, Tmult=${COSINE_TMULT:-2}"
[ "$_HAS_LR_SECTION" -eq 1 ] && _section_end

_HAS_REG=0
[ -n "$DROP_PATH_RATE" ] && { _maybe_sep; _HAS_REG=1; _print_row "Drop Path:" "$DROP_PATH_RATE"; }
[ -n "$HEAD_DROPOUT" ] && {
    [ "$_HAS_REG" -eq 0 ] && { _maybe_sep; _HAS_REG=1; }
    _print_row "Head Dropout:" "$HEAD_DROPOUT"
}
[ "$_HAS_REG" -eq 1 ] && _section_end

_HAS_PERTURB=0
[ -n "$PERTURB_DISTRIBUTION" ] && { _maybe_sep; _HAS_PERTURB=1; _print_row "Perturbation:" "$PERTURB_DISTRIBUTION"; }
[ -n "$PER_AXIS_PROB" ] && {
    [ "$_HAS_PERTURB" -eq 0 ] && { _maybe_sep; _HAS_PERTURB=1; }
    _print_row "Per-Axis Prob:" "$PER_AXIS_PROB"
}
[ -n "$PER_AXIS_WEIGHTS" ] && {
    [ "$_HAS_PERTURB" -eq 0 ] && { _maybe_sep; _HAS_PERTURB=1; }
    _print_row "Per-Axis Wts:" "R/P/Y = ${PER_AXIS_WEIGHTS}"
}
[ "$_HAS_PERTURB" -eq 1 ] && _section_end

_maybe_sep
_AUG_STR=""
[ -n "$AUGMENT_PC_JITTER" ] && [ "$AUGMENT_PC_JITTER" != "0" ] && [ "$AUGMENT_PC_JITTER" != "0.0" ] && \
    _AUG_STR="${_AUG_STR}jitter=${AUGMENT_PC_JITTER} "
[ -n "$AUGMENT_PC_DROPOUT" ] && [ "$AUGMENT_PC_DROPOUT" != "0" ] && [ "$AUGMENT_PC_DROPOUT" != "0.0" ] && \
    _AUG_STR="${_AUG_STR}dropout=${AUGMENT_PC_DROPOUT} "
[ -n "$AUGMENT_COLOR_JITTER" ] && [ "$AUGMENT_COLOR_JITTER" != "0" ] && [ "$AUGMENT_COLOR_JITTER" != "0.0" ] && \
    _AUG_STR="${_AUG_STR}color=${AUGMENT_COLOR_JITTER} "
[ -n "$AUGMENT_INTRINSIC" ] && [ "$AUGMENT_INTRINSIC" != "0" ] && [ "$AUGMENT_INTRINSIC" != "0.0" ] && \
    _AUG_STR="${_AUG_STR}intrinsic=±${AUGMENT_INTRINSIC}"
_print_row "Augmentation:"  "${_AUG_STR:-disabled}"
[ -n "$EARLY_STOPPING_PATIENCE" ] && [ "$EARLY_STOPPING_PATIENCE" != "0" ] && \
_print_row "Early Stop:"    "patience=$EARLY_STOPPING_PATIENCE"
_section_end

_maybe_sep
_print_row "Compute:"       "$COMPUTE_STR"
_print_row "GPU:"           "${GPU_NAME} (${GPU_MEM}MB) x${AVAIL_GPUS}"
_print_row "PyTorch:"       "$PYTORCH_VER"
_print_row "CUDA:"          "$CUDA_VER"
_print_row "torch.compile:" "$([ "$USE_COMPILE" -eq 1 ] && echo 'enabled' || echo 'disabled')"
_print_row "BEV Z-Step:"    "${BEV_ZBOUND_STEP:-4.0}"
if [ "$NNODES" -gt 1 ]; then
    _section_end
    _maybe_sep
    _print_row "Node Rank:"     "$NODE_RANK"
    _print_row "Master:"        "${MASTER_ADDR}:${MASTER_PORT}"
    _print_row "RDZV Timeout:"  "${RDZV_TIMEOUT}s"
fi
_section_end

_maybe_sep
_print_row "Log Directory:"     "$LOG_DIR"
if [ "$NODE_RANK" = "0" ]; then
    _print_row "Log File:"      "$TRAIN_LOG_FILE"
else
    _print_row "Log File:"      "$TRAIN_LOG_FILE (master only)"
fi
_print_row "TensorBoard Port:"  "$TENSORBOARD_PORT"
_print_bot
echo ""
}

_CONFIG_OUTPUT=$(_print_config_box 2>&1)
echo "$_CONFIG_OUTPUT"

if [ "$NODE_RANK" = "0" ]; then
    echo "$_CONFIG_OUTPUT" >> "$TRAIN_LOG_FILE"
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

ROTATION_ONLY_FLAG=""
if [ "$ROTATION_ONLY" -eq 1 ]; then
    ROTATION_ONLY_FLAG="--rotation_only 1"
fi

AXIS_LOSS_FLAG=""
if [ "$ENABLE_AXIS_LOSS" -eq 1 ]; then
    AXIS_LOSS_FLAG="--enable_axis_loss 1"
fi
WEIGHT_AXIS_FLAG=""
if [ -n "$WEIGHT_AXIS_ROTATION" ]; then
    WEIGHT_AXIS_FLAG="--weight_axis_rotation $WEIGHT_AXIS_ROTATION"
fi

OPTIM_FLAGS=""
[ -n "$LR_SCHEDULE" ] && OPTIM_FLAGS="$OPTIM_FLAGS --lr_schedule $LR_SCHEDULE"
[ -n "$WARMUP_EPOCHS" ] && OPTIM_FLAGS="$OPTIM_FLAGS --warmup_epochs $WARMUP_EPOCHS"
[ -n "$BACKBONE_LR_SCALE" ] && OPTIM_FLAGS="$OPTIM_FLAGS --backbone_lr_scale $BACKBONE_LR_SCALE"
[ -n "$COSINE_T0" ] && OPTIM_FLAGS="$OPTIM_FLAGS --cosine_T0 $COSINE_T0"
[ -n "$COSINE_TMULT" ] && OPTIM_FLAGS="$OPTIM_FLAGS --cosine_Tmult $COSINE_TMULT"
[ -n "$DROP_PATH_RATE" ] && OPTIM_FLAGS="$OPTIM_FLAGS --drop_path_rate $DROP_PATH_RATE"
[ -n "$HEAD_DROPOUT" ] && OPTIM_FLAGS="$OPTIM_FLAGS --head_dropout $HEAD_DROPOUT"
[ -n "$PERTURB_DISTRIBUTION" ] && OPTIM_FLAGS="$OPTIM_FLAGS --perturb_distribution $PERTURB_DISTRIBUTION"
[ -n "$PER_AXIS_PROB" ] && OPTIM_FLAGS="$OPTIM_FLAGS --per_axis_prob $PER_AXIS_PROB"
[ -n "$PER_AXIS_WEIGHTS" ] && OPTIM_FLAGS="$OPTIM_FLAGS --per_axis_weights $PER_AXIS_WEIGHTS"
[ -n "$AXIS_WEIGHTS" ] && OPTIM_FLAGS="$OPTIM_FLAGS --axis_weights $AXIS_WEIGHTS"
[ -n "$AUGMENT_PC_JITTER" ] && OPTIM_FLAGS="$OPTIM_FLAGS --augment_pc_jitter $AUGMENT_PC_JITTER"
[ -n "$AUGMENT_PC_DROPOUT" ] && OPTIM_FLAGS="$OPTIM_FLAGS --augment_pc_dropout $AUGMENT_PC_DROPOUT"
[ -n "$AUGMENT_COLOR_JITTER" ] && OPTIM_FLAGS="$OPTIM_FLAGS --augment_color_jitter $AUGMENT_COLOR_JITTER"
[ -n "$AUGMENT_INTRINSIC" ] && OPTIM_FLAGS="$OPTIM_FLAGS --augment_intrinsic $AUGMENT_INTRINSIC"
[ -n "$EVAL_ANGLE_RANGE_DEG" ] && OPTIM_FLAGS="$OPTIM_FLAGS --eval_angle_range_deg $EVAL_ANGLE_RANGE_DEG"
[ -n "$EARLY_STOPPING_PATIENCE" ] && OPTIM_FLAGS="$OPTIM_FLAGS --early_stopping_patience $EARLY_STOPPING_PATIENCE"
[ -n "$SEED" ] && OPTIM_FLAGS="$OPTIM_FLAGS --seed $SEED"
[ -n "$PRETRAIN_CKPT" ] && OPTIM_FLAGS="$OPTIM_FLAGS --pretrain_ckpt $PRETRAIN_CKPT"
[ -n "$USE_GEODESIC_LOSS" ] && OPTIM_FLAGS="$OPTIM_FLAGS --use_geodesic_loss $USE_GEODESIC_LOSS"
[ -n "$USE_MLP_HEAD" ] && OPTIM_FLAGS="$OPTIM_FLAGS --use_mlp_head $USE_MLP_HEAD"
[ -n "$USE_FOUNDATION_DEPTH" ] && OPTIM_FLAGS="$OPTIM_FLAGS --use_foundation_depth $USE_FOUNDATION_DEPTH"
[ -n "$DEPTH_MODEL_TYPE" ] && OPTIM_FLAGS="$OPTIM_FLAGS --depth_model_type $DEPTH_MODEL_TYPE"
[ -n "$FD_MODE" ] && OPTIM_FLAGS="$OPTIM_FLAGS --fd_mode $FD_MODE"
[ -n "$DEPTH_SUP_ALPHA" ] && OPTIM_FLAGS="$OPTIM_FLAGS --depth_sup_alpha $DEPTH_SUP_ALPHA"
[ -n "$MAX_FRAMES_PER_SEQ" ] && OPTIM_FLAGS="$OPTIM_FLAGS --max_frames_per_seq $MAX_FRAMES_PER_SEQ"
[ -n "$EVAL_EPOCHES" ] && OPTIM_FLAGS="$OPTIM_FLAGS --eval_epoches $EVAL_EPOCHES"
[ -n "$GRAD_ACCUM_STEPS" ] && OPTIM_FLAGS="$OPTIM_FLAGS --grad_accum_steps $GRAD_ACCUM_STEPS"

DEFORMABLE_VAL=${USE_DEFORMABLE:-0}
[ -n "$BEV_POOL_FACTOR" ] && OPTIM_FLAGS="$OPTIM_FLAGS --bev_pool_factor $BEV_POOL_FACTOR"

SCRATCH_EPOCHS=${NUM_EPOCHS_OVERRIDE:-400}
FINETUNE_EPOCHS=${NUM_EPOCHS_OVERRIDE:-50}
RESUME_EPOCHS=${NUM_EPOCHS_OVERRIDE:-100}

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
        export NCCL_BLOCKING_WAIT=1
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export DDP_TIMEOUT_MINUTES=${DDP_TIMEOUT_MINUTES:-$(( RDZV_TIMEOUT / 60 + 10 ))}

        RDZV_ID="${RDZV_ID:-bevcalib_${MASTER_PORT}}"
        if [ "$NODE_RANK" -eq 0 ]; then
            IS_HOST=1
        else
            IS_HOST=0
        fi

        LAUNCHER="torchrun \
            --nproc_per_node=$DDP_NGPUS \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            --rdzv_id=$RDZV_ID \
            --rdzv_conf timeout=${RDZV_TIMEOUT},is_host=${IS_HOST}"
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
        LR_SCRATCH=${LEARNING_RATE:-1e-4}
        $LAUNCHER kitti-bev-calib/train_kitti.py \
            --log_dir "$LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --label ${DATASET_NAME}_scratch \
            --batch_size $BATCH_SIZE \
            --num_epochs $SCRATCH_EPOCHS \
            --save_ckpt_per_epoches 40 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --deformable $DEFORMABLE_VAL \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr $LR_SCRATCH \
            --step_size 80 \
            --use_custom_dataset 1 \
            $COMPILE_FLAG \
            $ROTATION_ONLY_FLAG \
            $AXIS_LOSS_FLAG \
            $WEIGHT_AXIS_FLAG \
            $OPTIM_FLAGS
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
        LR_FINETUNE=${LEARNING_RATE:-4e-5}
        
        $LAUNCHER kitti-bev-calib/train_kitti.py \
            --log_dir "$FINETUNE_LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$KITTI_PRETRAIN" \
            --label ${DATASET_NAME}_finetuned \
            --batch_size $BATCH_SIZE \
            --num_epochs $FINETUNE_EPOCHS \
            --save_ckpt_per_epoches 10 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --lr $LR_FINETUNE \
            --scheduler 1 \
            --step_size 20 \
            --deformable $DEFORMABLE_VAL \
            --bev_encoder 1 \
            --xyz_only 1 \
            --use_custom_dataset 1 \
            $COMPILE_FLAG \
            $ROTATION_ONLY_FLAG \
            $AXIS_LOSS_FLAG \
            $WEIGHT_AXIS_FLAG \
            $OPTIM_FLAGS
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
        
        LR_RESUME=${LEARNING_RATE:-1e-4}
        
        $LAUNCHER kitti-bev-calib/train_kitti.py \
            --log_dir "$LOG_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --pretrain_ckpt "$LAST_CKPT" \
            --label ${DATASET_NAME}_resume \
            --batch_size $BATCH_SIZE \
            --num_epochs $RESUME_EPOCHS \
            --save_ckpt_per_epoches 10 \
            --angle_range_deg $ANGLE_RANGE_DEG \
            --trans_range $TRANS_RANGE \
            --deformable $DEFORMABLE_VAL \
            --bev_encoder 1 \
            --xyz_only 1 \
            --scheduler 1 \
            --lr $LR_RESUME \
            --step_size 20 \
            --use_custom_dataset 1 \
            $COMPILE_FLAG \
            $ROTATION_ONLY_FLAG \
            $AXIS_LOSS_FLAG \
            $WEIGHT_AXIS_FLAG \
            $OPTIM_FLAGS
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
