#!/usr/bin/env bash
# Quick sanity check: train 2 epochs to verify DDP setup
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJFUSION="${PROJFUSION_ROOT:-/mnt/drtraining/user/dahailu/code/ProjFusion}"
PYTHON="${PYTHON:-/opt/conda/envs/bevcalib310/bin/python}"

export PROJFUSION_ROOT="$PROJFUSION"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=4

cd "$PROJFUSION"

echo "============================================"
echo "Fleet PointGPT DDP Sanity Check (2 epochs)"
echo "============================================"

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29502 \
  "$ROOT/tools/pretrain_fleet_pointgpt_ddp.py" \
  --config cfg/pointgpt/pretrain_fleet_8gpu.yaml \
  --init_ckpt pretrained/kitti_pointgpt_tiny.pth \
  --output pretrained/fleet_pointgpt_sanity.pth \
  --epochs 2 \
  --batch_size 16 \
  --lr 2.8e-4

echo ""
echo "============================================"
echo "Sanity check completed!"
echo "If training succeeded for 2 epochs, you can:"
echo "  bash tools/run_pretrain_fleet_pointgpt_8gpu.sh"
echo "============================================"
