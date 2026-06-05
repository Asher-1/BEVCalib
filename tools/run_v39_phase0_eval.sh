#!/usr/bin/env bash
# Phase-0: 锚定 baseline MEDW（不训练）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/conda/envs/bevcalib310/bin/python}"
TEST_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
EVAL_FLAGS="--use_full_dataset --eval_max_frames_per_seq 200 --angle_range_deg 5.0 --vis_interval 0 --batch_size 8"

cd "$ROOT"

run_eval() {
  local ckpt="$1" out="$2"
  echo "=== MEDW eval: $ckpt -> $out ==="
  "$PYTHON" evaluate_checkpoint.py --mode eval \
    --ckpt_path "$ckpt" \
    --dataset_root "$TEST_ROOT" \
    --output_dir "$out" \
    $EVAL_FLAGS
}

# B0 v36
run_eval \
  "logs/all_training_data/model_small_5deg_v36_native_cross_pointgpt/all_training_data_scratch/checkpoint/ckpt_best_val.pth" \
  "logs/v39_phase0/B0_v36_medw"

# B1 v37 ep241
run_eval \
  "logs/all_training_data/model_small_10deg_v37_native_cross_pointgpt_long/all_training_data_scratch/checkpoint/ckpt_best_val.pth" \
  "logs/v39_phase0/B1_v37_ep241_medw"

# B2 v37 ep350
run_eval \
  "logs/all_training_data/model_small_10deg_v37_native_cross_pointgpt_long/all_training_data_scratch/checkpoint/ckpt_350.pth" \
  "logs/v39_phase0/B2_v37_ep350_medw"

echo "Phase-0 baseline eval done. See logs/v39_phase0/"
