#!/usr/bin/env bash
# Monitor Fleet PointGPT training progress
set -e

LOG="${1:-/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_tiny_8gpu_train.log}"

if [[ ! -f "$LOG" ]]; then
    echo "Log file not found: $LOG"
    echo ""
    echo "Usage: $0 [log_file]"
    echo "Default: pretrained/fleet_pointgpt_tiny_8gpu_train.log"
    exit 1
fi

echo "============================================"
echo "Fleet PointGPT Training Monitor"
echo "============================================"
echo "Log: $LOG"
echo ""

tail -20 "$LOG"

echo ""
echo "============================================"
echo "Training Statistics"
echo "============================================"

BEST_VAL=$(grep -oP 'saved.*val=\K[\d.]+' "$LOG" | tail -1 || echo "")
if [[ -n "$BEST_VAL" ]]; then
    echo "Best val loss: $BEST_VAL"
else
    echo "Best val loss: N/A (no checkpoint saved yet)"
fi

EPOCHS=$(grep -c "^Ep " "$LOG" || echo 0)
echo "Epochs completed: $EPOCHS"

LATEST=$(tail -5 "$LOG" | grep "^Ep " | tail -1 || echo "")
if [[ -n "$LATEST" ]]; then
    echo "Latest: $LATEST"
fi

echo ""
echo "============================================"
echo "Real-time updates (Ctrl+C to stop)"
echo "============================================"
tail -f "$LOG"
