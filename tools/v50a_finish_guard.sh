#!/bin/bash
# v50a 训练完成守护：确保本机 batch_train 在 v50a 结束后自动退出，
# 不会启动 v50b/v50c（远端机器训练）。
#
# 用法（后台运行）:
#   nohup bash tools/v50a_finish_guard.sh >> logs/v50a_finish_guard.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

TRAIN_LOG="logs/all_training_data/model_small_5deg_v50a_optuna_no_cons_S1_full/train.log"
BATCH_LOG="logs/v50a_train.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [v50a-guard] $*"; }

log "启动守护，监控 v50a 训练完成..."

# 等待训练完成
while true; do
    if [ -f "$TRAIN_LOG" ] && grep -q "训练完成总结" "$TRAIN_LOG" 2>/dev/null; then
        log "检测到 v50a 训练完成 (训练完成总结)"
        break
    fi
    sleep 30
done

log "等待 batch_train 自然退出（应跳过 v50b/v50c 后结束）..."

# 最多等 5 分钟让 batch_train 正常收尾
for _ in $(seq 1 30); do
    if ! pgrep -f "batch_train\.sh.*v50_cf_bev_r\.yaml" >/dev/null 2>&1; then
        log "batch_train 已正常退出 ✓"
        exit 0
    fi

    # 若检测到 v50b/v50c 实际开始训练，立即阻断
    if pgrep -f "start_training\.sh.*v50[bcd]_" >/dev/null 2>&1; then
        log "⚠️  检测到 v50b/v50c 在本机启动，强制终止 batch_train"
        pkill -f "batch_train\.sh.*v50_cf_bev_r\.yaml" || true
        sleep 5
        pkill -f "start_training\.sh.*v50[bcd]_" || true
        log "已阻断非 v50a 训练"
        exit 1
    fi

    sleep 10
done

# 超时仍未退出：强制结束 batch_train（训练本身已完成）
if pgrep -f "batch_train\.sh.*v50_cf_bev_r\.yaml" >/dev/null 2>&1; then
    log "batch_train 超时未退出，强制终止（v50a 已完成，释放资源）"
    pkill -f "batch_train\.sh.*v50_cf_bev_r\.yaml" || true
    # 停止 TensorBoard
    pkill -f "tensorboard.*v50a_optuna_no_cons" || true
fi

log "守护结束"
