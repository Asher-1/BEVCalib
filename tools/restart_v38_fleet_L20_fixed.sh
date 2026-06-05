#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${ROOT}/configs/v38_native_cross_fleet_pointgpt.yaml"
LOG="${ROOT}/logs/all_training_data/v38_launch_$(date +%m%d_%H%M).log"

echo "Stopping existing v38 batch_train / train_kitti processes..."
pkill -f "batch_train.sh.*v38_native_cross_fleet_pointgpt" 2>/dev/null || true
pkill -f "model_small_10deg_v38_native_cross_fleet_pointgpt" 2>/dev/null || true
sleep 5

echo "Starting v38 (Arm1 → Arm4 auto chain)..."
cd "${ROOT}"
nohup bash batch_train.sh "${CONFIG}" > "${LOG}" 2>&1 &

echo "Launched PID=$!"
echo "Monitor: tail -f ${LOG}"
echo "Arm1 log: tail -f ${ROOT}/logs/all_training_data/model_small_10deg_v38_native_cross_fleet_pointgpt_L20/train.log"
echo "Arm4 log: tail -f ${ROOT}/logs/all_training_data/model_small_5deg_v38_native_cross_fleet_pointgpt_L20_finetune_5deg/train.log"
