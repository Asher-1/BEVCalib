#!/usr/bin/env bash
# Fleet PointGPT pretrain with 8-GPU DDP
# Fast domain adaptation: KITTI PointGPT -> Fleet PointGPT
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

CONFIG="${CONFIG:-cfg/pointgpt/pretrain_fleet_8gpu.yaml}"
INIT_CKPT="${INIT_CKPT:-pretrained/kitti_pointgpt_tiny.pth}"
OUTPUT="${OUTPUT:-pretrained/fleet_pointgpt_tiny_8gpu.pth}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2.8e-4}"

echo "============================================"
echo "Fleet PointGPT Pretrain - 8 GPU DDP"
echo "============================================"
echo "Config      : $CONFIG"
echo "Init ckpt   : $INIT_CKPT"
echo "Output      : $OUTPUT"
echo "Epochs      : $EPOCHS"
echo "Per-GPU BS  : $BATCH_SIZE (global: $((BATCH_SIZE * 8)))"
echo "LR          : $LR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
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
