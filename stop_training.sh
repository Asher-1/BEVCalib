#!/bin/bash
# =============================================================================
# BEVCalib 训练停止脚本
# 用法: ./stop_training.sh [--force]
# =============================================================================

FORCE=0
if [ "$1" == "--force" ] || [ "$1" == "-f" ]; then
    FORCE=1
fi

echo "========================================"
echo "BEVCalib 训练停止"
echo "========================================"
echo ""

# 查找训练进程
echo "查找训练进程..."
TRAIN_PIDS=$(ps aux | grep -E "train_kitti.py" | grep -v grep | awk '{print $2}')
BASH_PIDS=$(ps aux | grep -E "train_B26A.sh|train_universal.sh|start_training.sh" | grep -v grep | awk '{print $2}')

# 显示当前进程
echo ""
echo "当前训练进程:"
ps aux | grep -E "train_kitti.py|train_B26A.sh|train_universal.sh|start_training.sh" | grep -v grep || echo "  无运行中的训练进程"
echo ""

# 统计进程数量
TRAIN_COUNT=$(echo "$TRAIN_PIDS" | grep -v "^$" | wc -l)
BASH_COUNT=$(echo "$BASH_PIDS" | grep -v "^$" | wc -l)
TOTAL_COUNT=$((TRAIN_COUNT + BASH_COUNT))

if [ "$TOTAL_COUNT" -eq 0 ]; then
    echo "✅ 没有找到运行中的训练进程"
    exit 0
fi

echo "找到 $TRAIN_COUNT 个 Python 训练进程"
echo "找到 $BASH_COUNT 个 Bash 训练脚本进程"
echo ""

# 确认停止
if [ "$FORCE" -eq 0 ]; then
    echo "是否停止所有训练进程? (y/n)"
    read -t 10 -p "> " CONFIRM || CONFIRM="n"
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
fi

echo ""
echo "正在停止训练进程..."

# 停止 Python 训练进程
if [ -n "$TRAIN_PIDS" ] && [ "$TRAIN_PIDS" != "" ]; then
    for PID in $TRAIN_PIDS; do
        if [ -n "$PID" ]; then
            echo "  停止 Python 进程: $PID"
            kill -TERM $PID 2>/dev/null || true
        fi
    done
fi

# 停止 Bash 脚本进程
if [ -n "$BASH_PIDS" ] && [ "$BASH_PIDS" != "" ]; then
    for PID in $BASH_PIDS; do
        if [ -n "$PID" ]; then
            echo "  停止 Bash 进程: $PID"
            kill -TERM $PID 2>/dev/null || true
        fi
    done
fi

# 等待进程结束
echo ""
echo "等待进程结束..."
sleep 3

# 检查是否还有残留进程
REMAINING=$(ps aux | grep -E "train_kitti.py|train_B26A.sh|train_universal.sh|start_training.sh" | grep -v grep | wc -l)

if [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  还有 $REMAINING 个进程未停止，强制终止..."
    pkill -9 -f "train_kitti.py" 2>/dev/null || true
    pkill -9 -f "train_B26A.sh" 2>/dev/null || true
    pkill -9 -f "train_universal.sh" 2>/dev/null || true
    pkill -9 -f "start_training.sh" 2>/dev/null || true
    sleep 2
fi

# 最终检查
FINAL=$(ps aux | grep -E "train_kitti.py|train_B26A.sh|train_universal.sh|start_training.sh" | grep -v grep | wc -l)

echo ""
if [ "$FINAL" -eq 0 ]; then
    echo "✅ 所有训练进程已停止"
else
    echo "❌ 仍有 $FINAL 个进程运行中，请手动检查"
    ps aux | grep -E "train_kitti.py|train_B26A.sh|train_universal.sh|start_training.sh" | grep -v grep
fi

echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null || echo "无法获取GPU状态"
echo ""

