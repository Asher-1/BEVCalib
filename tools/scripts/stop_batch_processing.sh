#!/bin/bash
#
# 停止批量处理脚本
# 用于停止 batch_prepare_trips.py 及其所有子进程
#

echo "========================================"
echo "停止批量处理进程"
echo "========================================"
echo ""

# 查找所有相关进程
echo "正在查找批量处理进程..."
batch_pids=$(pgrep -f "batch_prepare_trips.py")
prepare_pids=$(pgrep -f "prepare_custom_dataset.py")

if [ -z "$batch_pids" ] && [ -z "$prepare_pids" ]; then
    echo "✓ 没有找到正在运行的批量处理进程"
    exit 0
fi

# 显示找到的进程
if [ ! -z "$batch_pids" ]; then
    echo ""
    echo "找到批量处理主进程:"
    ps aux | grep "batch_prepare_trips.py" | grep -v grep | awk '{print "  PID: " $2 " | " $11 " " $12 " " $13}'
fi

if [ ! -z "$prepare_pids" ]; then
    echo ""
    echo "找到数据准备子进程:"
    ps aux | grep "prepare_custom_dataset.py" | grep -v grep | awk '{print "  PID: " $2 " | " $11 " " $12 " " $13}'
fi

echo ""
read -p "确认停止这些进程? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

echo ""
echo "正在停止进程..."

# 先尝试优雅停止 (SIGTERM)
if [ ! -z "$batch_pids" ]; then
    echo "  发送 SIGTERM 到 batch_prepare_trips.py..."
    pkill -15 -f "batch_prepare_trips.py"
fi

if [ ! -z "$prepare_pids" ]; then
    echo "  发送 SIGTERM 到 prepare_custom_dataset.py..."
    pkill -15 -f "prepare_custom_dataset.py"
fi

# 等待 5 秒
echo "  等待进程退出..."
sleep 5

# 检查是否还有进程在运行
remaining_batch=$(pgrep -f "batch_prepare_trips.py")
remaining_prepare=$(pgrep -f "prepare_custom_dataset.py")

if [ ! -z "$remaining_batch" ] || [ ! -z "$remaining_prepare" ]; then
    echo "  进程仍在运行，强制终止 (SIGKILL)..."
    
    if [ ! -z "$remaining_batch" ]; then
        pkill -9 -f "batch_prepare_trips.py"
    fi
    
    if [ ! -z "$remaining_prepare" ]; then
        pkill -9 -f "prepare_custom_dataset.py"
    fi
    
    sleep 2
fi

# 最终验证
final_check_batch=$(pgrep -f "batch_prepare_trips.py")
final_check_prepare=$(pgrep -f "prepare_custom_dataset.py")

echo ""
if [ -z "$final_check_batch" ] && [ -z "$final_check_prepare" ]; then
    echo "✓ 所有批量处理进程已成功停止"
    exit 0
else
    echo "✗ 警告: 某些进程可能仍在运行"
    echo ""
    echo "请手动检查:"
    echo "  ps aux | grep -E 'batch_prepare_trips|prepare_custom_dataset' | grep -v grep"
    exit 1
fi
