#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/logs/pipeline_v55"
OUT_LOG="$OUT_DIR/monitor.log"
mkdir -p "$OUT_DIR"

LOGS=(
  "$ROOT/logs/all_training_data/model_small_5deg_v55_deploy_generalization/train.log"
  "$ROOT/logs/all_training_data/model_small_5deg_v55_safe_deploy_generalization/train.log"
  "$ROOT/logs/all_training_data/model_small_5deg_v55b_smoke/train.log"
  "$ROOT/logs/all_training_data/model_small_5deg_v55b_route_tight/train.log"
)

_stamp() { date '+%Y-%m-%d %H:%M:%S'; }

_summarize_one() {
  local log="$1"
  local name
  name="$(basename "$(dirname "$log")")"
  if [ ! -f "$log" ]; then
    echo "[$(_stamp)] $name: waiting for train.log"
    return
  fi

  local last_epoch_line
  local last_rot
  local last_route
  local last_nan
  local last_ckpt_eval
  local nan_count

  last_epoch_line="$(rg 'Epoch \[[0-9]+/[0-9]+\].*Step|Epoch \[[0-9]+/[0-9]+\] completed' "$log" | tail -1 || true)"
  last_rot="$(rg 'Train Pose Error - Rot:' "$log" | tail -1 || true)"
  last_route="$(rg 'route_w_mean:' "$log" | tail -1 || true)"
  last_nan="$(rg 'NaN GUARD' "$log" | tail -1 || true)"
  last_ckpt_eval="$(rg 'Checkpoint [0-9]+ Eval Pose Error - Rot:' "$log" | tail -1 || true)"
  nan_count="$(rg -c 'NaN GUARD' "$log" || true)"

  echo "[$(_stamp)] $name"
  [ -n "$last_epoch_line" ] && echo "  last_epoch: $last_epoch_line"
  [ -n "$last_rot" ] && echo "  train_rot:  $last_rot"
  [ -n "$last_route" ] && echo "  route:      $last_route"
  echo "  nan_count:   ${nan_count:-0}"
  [ -n "$last_nan" ] && echo "  last_nan:    $last_nan"
  [ -n "$last_ckpt_eval" ] && echo "  ckpt_eval:   $last_ckpt_eval"
}

while true; do
  {
    echo "============================================================"
    echo "[$(_stamp)] V55 convergence monitor tick"
    for log in "${LOGS[@]}"; do
      _summarize_one "$log"
    done
    echo ""
  } | tee -a "$OUT_LOG"
  sleep 300
done
