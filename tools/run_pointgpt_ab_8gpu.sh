#!/usr/bin/env bash
# PointGPT KITTI vs nuScenes A/B eval on fixed v36 checkpoint (8-GPU sharded).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310 2>/dev/null || true

CKPT="${CKPT:-logs/all_training_data/model_small_5deg_v36_native_cross_pointgpt/all_training_data_scratch/checkpoint/ckpt_best_val.pth}"
DATASET="${DATASET:-/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2}"
OUT_DIR="${OUT_DIR:-logs/all_training_data/pointgpt_ab_v36_test_v2}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ANGLE="${ANGLE:-5.0}"
N_ITERS="${N_ITERS:-1,3}"

KITTI_CKPT="/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/kitti_pointgpt_tiny.pth"
KITTI_CFG="/mnt/drtraining/user/dahailu/code/ProjFusion/cfg/pointgpt/finetune_kitti_tiny.yaml"
NUSC_CKPT="/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/nusc_pointgpt_tiny.pth"
NUSC_CFG="/mnt/drtraining/user/dahailu/code/ProjFusion/cfg/pointgpt/finetune_kitti_tiny.yaml"

mkdir -p "$OUT_DIR"
export HF_HUB_OFFLINE=1 USE_DRCV_BACKEND=0 PROJFUSION_ROOT=/mnt/drtraining/user/dahailu/code/ProjFusion

run_arm() {
  local arm_name="$1"
  local pg_ckpt="$2"
  local pg_cfg="$3"
  local arm_dir="$OUT_DIR/$arm_name"
  mkdir -p "$arm_dir/shards"

  echo "=== [$arm_name] launching ${NUM_SHARDS} shards on ${NUM_SHARDS} GPUs ==="
  pids=()
  for ((i=0; i<NUM_SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python tools/eval_native_cross_iterative.py \
      --ckpt_path "$CKPT" \
      --dataset_root "$DATASET" \
      --validate_sample_ratio 1.0 \
      --angle_range_deg "$ANGLE" \
      --batch_size "$BATCH_SIZE" \
      --n_iters_list "$N_ITERS" \
      --shard_id "$i" \
      --num_shards "$NUM_SHARDS" \
      --gpu_id 0 \
      --pointgpt_ckpt "$pg_ckpt" \
      --pointgpt_config "$pg_cfg" \
      --output_json "$arm_dir/shards/shard_${i}.json" \
      > "$arm_dir/shard_${i}.log" 2>&1 &
    pids+=($!)
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[FATAL] $arm_name eval failed; see $arm_dir/shard_*.log"
    exit 1
  fi

  python tools/merge_iterative_eval_shards.py \
    --shard_glob "$arm_dir/shards/shard_*.json" \
    --output_json "$arm_dir/merged.json"
}

echo "Checkpoint: $CKPT"
echo "Dataset:    $DATASET"
echo "Output:     $OUT_DIR"

run_arm "kitti_pointgpt" "$KITTI_CKPT" "$KITTI_CFG"
run_arm "nusc_pointgpt" "$NUSC_CKPT" "$NUSC_CFG"

python tools/summarize_pointgpt_ab.py --out_dir "$OUT_DIR"
echo "Done: $OUT_DIR/ab_summary.json"
