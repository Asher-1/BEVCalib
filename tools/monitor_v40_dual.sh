#!/usr/bin/env bash
# 双路 V40 训练收敛监控：full vs P0ab
set -euo pipefail

FULL_LOG="${1:-logs/all_training_data/model_small_5deg_v40_gmp_full_5deg_main/train.log}"
P0_LOG="${2:-logs/all_training_data/model_small_5deg_v40_gmp_p0ab_5deg_main/train.log}"

parse_run() {
    local name="$1" log="$2"
    if [[ ! -f "$log" ]]; then
        echo "[$name] 日志不存在: $log"
        return
    fi

    local mtime epoch step train_rot medw jac nan guard intrinsic
    mtime=$(stat -c '%y' "$log" 2>/dev/null | cut -d. -f1)
    epoch=$(grep -oP 'Epoch \[\K[0-9]+(?=/80\])' "$log" | tail -1 || echo "?")
    step=$(grep -oP 'Step \[\K[0-9]+(?=/32\])' "$log" | tail -1 || echo "?")
    train_rot=$(grep 'Train Pose Error - Rot:' "$log" | tail -1 | grep -oP 'Rot: \K[0-9.]+' || echo "-")
    medw=$(grep 'MEDW200 (val reuse):' "$log" | tail -1 || true)
    jac=$(grep 'Jacobian.*from val' "$log" | tail -1 || true)
    nan=$(grep -c 'NaN GUARD\|CUDA GUARD' "$log" 2>/dev/null || echo 0)
    guard=$(grep 'CUDA GUARD' "$log" | tail -1 || true)
    intrinsic=$(grep 'Intrinsic augmentation' "$log" | tail -1 || echo "(无内参扰动)")

    local last_line
    last_line=$(tail -1 "$log")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $name"
    echo "  日志: $log"
    echo "  更新时间: $mtime"
    echo "  进度: Ep ${epoch}/80  Step ${step}/32"
    echo "  最新 Train Rot: ${train_rot}°"
    echo "  内参: ${intrinsic}"
    echo "  NaN/CUDA GUARD 次数: ${nan}"
    if [[ -n "$medw" ]]; then
        echo "  最近 MEDW: ${medw#*] }"
    else
        echo "  最近 MEDW: (尚无 eval，Ep10 起)"
    fi
    if [[ -n "$jac" ]]; then
        echo "  最近 Jacobian: ${jac#*] }"
    fi
    if [[ -n "$guard" ]]; then
        echo "  ⚠️  $guard"
    fi
    echo "  末行: $last_line"
}

echo "V40 双路训练监控  $(date '+%Y-%m-%d %H:%M:%S')"
parse_run "FULL (P0ab+Match+Corr)" "$FULL_LOG"
parse_run "P0ab (iter3+geo)" "$P0_LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "刷新: watch -n 60 bash tools/monitor_v40_dual.sh"
