#!/usr/bin/env bash
# Fleet PointGPT finetune (PointTransformer, ProjFusion-compatible output).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJFUSION="${PROJFUSION_ROOT:-/mnt/drtraining/user/dahailu/code/ProjFusion}"
PYTHON="${PYTHON:-/opt/conda/envs/bevcalib310/bin/python}"

export PROJFUSION_ROOT="$PROJFUSION"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$PROJFUSION"

"$PYTHON" "$ROOT/tools/pretrain_fleet_pointgpt.py" \
  --config cfg/pointgpt/finetune_fleet_tiny.yaml \
  --init_ckpt pretrained/kitti_pointgpt_tiny.pth \
  --output pretrained/fleet_pointgpt_tiny.pth \
  --epochs "${EPOCHS:-50}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --gpu 0 \
  "$@"
