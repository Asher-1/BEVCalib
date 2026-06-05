#!/usr/bin/env bash
# 后台流水线:
#   B) 等待 v37 ckpt_300.pth → 单卡 GPU0 ±5° MEDW + Jacobian（不停 8 卡训练）
#   A) 等待 fleet PointGPT 预训练完成 → 启动 v38 主实验（scratch）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/conda/envs/bevcalib310/bin/python}"
LOG_DIR="$ROOT/logs/all_training_data/pipeline_v37_ep300_v38"
mkdir -p "$LOG_DIR"

V37_CKPT_DIR="$ROOT/logs/all_training_data/model_small_10deg_v37_native_cross_pointgpt_long/all_training_data_scratch/checkpoint"
CKPT_300="$V37_CKPT_DIR/ckpt_300.pth"
FLEET_LOG="$ROOT/logs/all_training_data/fleet_pointgpt_pretrain/train.log"
FLEET_CKPT="/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_L20.pth"
FLEET_LOG_ALT="/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_L20_train.log"
TEST_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

run_ep300_eval() {
  local ckpt="$1"
  local out="$ROOT/logs/all_training_data/model_small_10deg_v37_native_cross_pointgpt_long/medw200_test_v2_eval_5deg_ep300"
  log "Ep300 eval start: $ckpt"
  export CUDA_VISIBLE_DEVICES=0
  cd "$ROOT"
  "$PYTHON" evaluate_checkpoint.py --mode eval \
    --ckpt_path "$ckpt" \
    --dataset_root "$TEST_ROOT" \
    --output_dir "$out" \
    --use_full_dataset --eval_max_frames_per_seq 200 \
    --angle_range_deg 5.0 --batch_size 8 \
    >> "$LOG_DIR/ep300_medw.log" 2>&1

  "$PYTHON" tools/diagnose_jacobian.py \
    --ckpt_path "$ckpt" \
    --angle_range 5.0 --n_batches 10 --batch_size 4 \
    --output "$V37_CKPT_DIR/jacobian_5deg_ep300.json" \
    >> "$LOG_DIR/ep300_jacobian_5deg.log" 2>&1

  "$PYTHON" tools/diagnose_jacobian.py \
    --ckpt_path "$ckpt" \
    --angle_range 10.0 --n_batches 10 --batch_size 4 \
    --output "$V37_CKPT_DIR/jacobian_10deg_ep300.json" \
    >> "$LOG_DIR/ep300_jacobian_10deg.log" 2>&1

  log "Ep300 eval done -> $out"
}

wait_ep300_and_eval() {
  log "Waiting for $CKPT_300 ..."
  while [ ! -f "$CKPT_300" ]; do
    sleep 120
  done
  run_ep300_eval "$CKPT_300"
}

wait_fleet_and_launch_v38() {
  log "Waiting for fleet PointGPT L20 pretrain ($FLEET_CKPT) ..."
  while true; do
    if [ -f "$FLEET_CKPT" ]; then
      if grep -q "^Done\." "$FLEET_LOG" 2>/dev/null || grep -q "^Done\." "$FLEET_LOG_ALT" 2>/dev/null; then
        break
      fi
      if grep -q "Ep 30/30" "$FLEET_LOG_ALT" 2>/dev/null; then
        break
      fi
    fi
    sleep 300
  done
  log "Fleet L20 pretrain ready. Launching v38 main (scratch) ..."
  cd "$ROOT"
  bash batch_train.sh configs/v38_native_cross_fleet_pointgpt.yaml \
    >> "$LOG_DIR/v38_launch.log" 2>&1
  log "v38 training finished or exited."
}

log "Pipeline started (pid $$)"
wait_ep300_and_eval &
PID_EVAL=$!
wait_fleet_and_launch_v38 &
PID_V38=$!
wait "$PID_EVAL" "$PID_V38"
log "Pipeline all tasks complete."
