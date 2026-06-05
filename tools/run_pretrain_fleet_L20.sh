#!/usr/bin/env bash
# Fleet PointGPT pretrain - L20 48GB optimized launcher
# batch 512 (64/GPU), 16k points, 60m depth
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJFUSION="${PROJFUSION_ROOT:-/mnt/drtraining/user/dahailu/code/ProjFusion}"
PYTHON="${PYTHON:-/opt/conda/envs/bevcalib310/bin/python}"

export PROJFUSION_ROOT="$PROJFUSION"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

cd "$PROJFUSION"

# L20-optimized config
CONFIG="${CONFIG:-cfg/pointgpt/pretrain_fleet_L20.yaml}"
INIT_CKPT="${INIT_CKPT:-pretrained/kitti_pointgpt_tiny.pth}"
OUTPUT="${OUTPUT:-pretrained/fleet_pointgpt_L20.pth}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-4e-4}"

echo "============================================"
echo "Fleet PointGPT Pretrain - L20 Optimized"
echo "============================================"
echo "Config      : $CONFIG"
echo "Init ckpt   : $INIT_CKPT"
echo "Output      : $OUTPUT"
echo "Epochs      : $EPOCHS"
echo "Per-GPU BS  : $BATCH_SIZE (global: $((BATCH_SIZE * 8)))"
echo "Points      : 16384 (2x denser sampling)"
echo "Max depth   : 60m (vs 50m, +10% coverage)"
echo "LR          : $LR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""
echo "Fleet data characteristics:"
echo "   - Points/frame: 120k (vs KITTI 10-20k)"
echo "   - Depth 95%: 76m"
echo "   - Config retains: 93.4% points"
echo "============================================"
echo ""

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT:-29501}" \
  "$ROOT/tools/pretrain_fleet_pointgpt_ddp.py" \
  --config "$CONFIG" \
  --init_ckpt "$INIT_CKPT" \
  --output "$OUTPUT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  "$@"

echo ""
echo "============================================"
echo "Training completed!"
echo "Best model: $OUTPUT"
echo "============================================"
