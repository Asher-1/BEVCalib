#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310 2>/dev/null || true

export HF_HUB_OFFLINE=1 USE_DRCV_BACKEND=0 PROJFUSION_ROOT=/mnt/drtraining/user/dahailu/code/ProjFusion
CKPT="${CKPT:-logs/all_training_data/model_small_5deg_v36_native_cross_pointgpt/all_training_data_scratch/checkpoint/ckpt_best_val.pth}"
DATASET="${DATASET:-/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2}"
OUT_DIR="${OUT_DIR:-logs/all_training_data/pointgpt_ab_v36_test_v2}"
arm_dir="$OUT_DIR/nusc_pointgpt"
mkdir -p "$arm_dir/shards"

NUM_SHARDS=8
BATCH_SIZE=16
pids=()
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python tools/eval_native_cross_iterative.py \
    --ckpt_path "$CKPT" \
    --dataset_root "$DATASET" \
    --validate_sample_ratio 1.0 \
    --angle_range_deg 5.0 \
    --batch_size "$BATCH_SIZE" \
    --n_iters_list 1,3 \
    --shard_id "$i" \
    --num_shards "$NUM_SHARDS" \
    --gpu_id 0 \
    --pointgpt_ckpt /mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/nusc_pointgpt_tiny.pth \
    --pointgpt_config /mnt/drtraining/user/dahailu/code/ProjFusion/cfg/pointgpt/finetune_kitti_tiny.yaml \
    --output_json "$arm_dir/shards/shard_${i}.json" \
    > "$arm_dir/shard_${i}.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

python tools/merge_iterative_eval_shards.py \
  --shard_glob "$arm_dir/shards/shard_*.json" \
  --output_json "$arm_dir/merged.json"

python tools/summarize_pointgpt_ab.py --out_dir "$OUT_DIR"
