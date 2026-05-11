#!/bin/bash
# =============================================================================
# BEVCalib 训练停止脚本 — 彻底停止所有训练（含批量队列和重试）
#
# 用法:
#   ./stop_training.sh           # 交互确认后停止
#   ./stop_training.sh --force   # 直接停止，不确认
#   ./stop_training.sh --status  # 仅查看状态，不停止
# =============================================================================

set -euo pipefail

FORCE=0
STATUS_ONLY=0
case "${1:-}" in
    --force|-f)  FORCE=1 ;;
    --status|-s) STATUS_ONLY=1 ;;
esac

# 所有需要匹配的进程模式（父进程在前，子进程在后）
PATTERNS=(
    "batch_train.sh"
    "start_training.sh"
    "train_universal.sh"
    "train_B26A.sh"
    "torchrun"
    "train_kitti.py"
    "tensorboard.*--logdir"
)
GREP_PATTERN=$(IFS='|'; echo "${PATTERNS[*]}")

echo "========================================"
echo "BEVCalib 训练进程管理"
echo "========================================"
echo ""

# ── 1. 收集所有相关进程 ──
collect_pids() {
    local pattern="$1"
    ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | sort -n
}

show_status() {
    echo "── 当前训练进程 ──"
    echo ""

    local has_any=0

    for pat in "${PATTERNS[@]}"; do
        local pids
        pids=$(collect_pids "$pat" 2>/dev/null || true)
        local count=0
        if [ -n "$pids" ]; then
            count=$(echo "$pids" | wc -l)
        fi
        if [ "$count" -gt 0 ]; then
            has_any=1
            printf "  %-30s %d 个进程\n" "$pat" "$count"
        fi
    done

    if [ "$has_any" -eq 0 ]; then
        echo "  ✅ 没有运行中的训练进程"
        echo ""
        echo "GPU 状态:"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null || echo "  无法获取GPU状态"
        return 1
    fi

    echo ""
    echo "── 进程详情 ──"
    ps aux | grep -E "$GREP_PATTERN" | grep -v grep | \
        awk '{printf "  PID=%-8s CPU=%5s MEM=%5s CMD=%s\n", $2, $3, $4, $11" "$12" "$13}' || true
    echo ""
    return 0
}

# ── 2. 显示状态 ──
if ! show_status; then
    exit 0
fi

if [ "$STATUS_ONLY" -eq 1 ]; then
    echo "GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null || echo "  无法获取GPU状态"
    exit 0
fi

# ── 3. 确认 ──
if [ "$FORCE" -eq 0 ]; then
    echo "是否停止所有训练进程（含批量队列和重试）? (y/n)"
    read -t 10 -p "> " CONFIRM || CONFIRM="n"
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
fi

echo ""
echo "正在停止所有训练进程..."

# ── 4. 先杀父进程（调度器），阻止新实验和重试 ──
PARENT_PATTERNS=("batch_train.sh" "start_training.sh" "train_universal.sh" "train_B26A.sh")
for pat in "${PARENT_PATTERNS[@]}"; do
    local_pids=$(collect_pids "$pat" 2>/dev/null || true)
    for pid in $local_pids; do
        [ -z "$pid" ] && continue
        echo "  [TERM] 停止调度进程: $pid ($pat)"
        kill -TERM "$pid" 2>/dev/null || true
        # 同时杀掉其所有子进程
        pkill -TERM -P "$pid" 2>/dev/null || true
    done
done

sleep 1

# ── 5. 杀 torchrun 及其子进程树 ──
TORCHRUN_PIDS=$(collect_pids "torchrun" 2>/dev/null || true)
for pid in $TORCHRUN_PIDS; do
    [ -z "$pid" ] && continue
    echo "  [TERM] 停止 torchrun: $pid"
    kill -TERM "$pid" 2>/dev/null || true
    pkill -TERM -P "$pid" 2>/dev/null || true
done

# ── 6. 杀 train_kitti.py 工作进程 ──
TRAIN_PIDS=$(collect_pids "train_kitti.py" 2>/dev/null || true)
for pid in $TRAIN_PIDS; do
    [ -z "$pid" ] && continue
    echo "  [TERM] 停止训练进程: $pid"
    kill -TERM "$pid" 2>/dev/null || true
done

# ── 7. 杀 TensorBoard ──
TB_PIDS=$(collect_pids "tensorboard.*--logdir" 2>/dev/null || true)
for pid in $TB_PIDS; do
    [ -z "$pid" ] && continue
    echo "  [TERM] 停止 TensorBoard: $pid"
    kill -TERM "$pid" 2>/dev/null || true
done

# ── 8. 等待优雅退出 ──
echo ""
echo "等待进程退出 (3s)..."
sleep 3

# ── 9. 检查残留，强制杀死 ──
REMAINING=$(ps aux | grep -E "$GREP_PATTERN" | grep -v grep | wc -l)

if [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  还有 $REMAINING 个进程未退出，强制终止 (SIGKILL)..."
    for pat in "${PATTERNS[@]}"; do
        pkill -9 -f "$pat" 2>/dev/null || true
    done
    sleep 2
fi

# ── 10. 最终验证 ──
FINAL=$(ps aux | grep -E "$GREP_PATTERN" | grep -v grep | wc -l)

echo ""
if [ "$FINAL" -eq 0 ]; then
    echo "✅ 所有训练进程已彻底停止（含批量队列和重试调度）"
else
    echo "❌ 仍有 $FINAL 个进程运行中，请手动检查:"
    ps aux | grep -E "$GREP_PATTERN" | grep -v grep
fi

echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null || echo "  无法获取GPU状态"
echo ""
