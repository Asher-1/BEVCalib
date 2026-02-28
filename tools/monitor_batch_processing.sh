#!/bin/bash
#
# 监控批量处理进度
# 实时显示处理状态和日志
#

OUTPUT_DIR="/mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data"

echo "========================================"
echo "批量处理进度监控"
echo "========================================"
echo ""

# 检查进程是否在运行
batch_pid=$(pgrep -f "batch_prepare_trips.py")
prepare_pid=$(pgrep -f "prepare_custom_dataset.py")

if [ -z "$batch_pid" ] && [ -z "$prepare_pid" ]; then
    echo "✗ 没有找到正在运行的批量处理进程"
    echo ""
    echo "检查最近的日志:"
    latest_log=$(ls -t $OUTPUT_DIR/batch_processing_*.log 2>/dev/null | head -1)
    if [ ! -z "$latest_log" ]; then
        echo "  日志文件: $latest_log"
        echo ""
        echo "最后 20 行:"
        tail -20 "$latest_log"
    fi
    exit 1
fi

echo "✓ 批量处理正在运行中"
echo ""

if [ ! -z "$batch_pid" ]; then
    echo "主进程 PID: $batch_pid"
fi

if [ ! -z "$prepare_pid" ]; then
    echo "子进程 PID: $prepare_pid"
fi

echo ""
echo "----------------------------------------"
echo ""

# 查找最新的日志文件
latest_log=$(ls -t $OUTPUT_DIR/batch_processing_*.log 2>/dev/null | head -1)

if [ -z "$latest_log" ]; then
    echo "等待日志文件生成..."
else
    echo "日志文件: $latest_log"
    echo ""
    
    # 显示实时日志
    echo "实时日志 (Ctrl+C 停止监控):"
    echo "========================================"
    tail -f "$latest_log"
fi
