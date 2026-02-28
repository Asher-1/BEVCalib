#!/bin/bash
# =============================================================================
# BEVCalib 多GPU训练启动脚本
# 用法: ./start_training.sh [版本号]
# 示例: ./start_training.sh v5
# =============================================================================

set -e

# 版本号（用于区分不同实验）
VERSION=${1:-v1}

# 数据集路径
DATASET=${DATASET:-/mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data_fix}

# 检查数据集是否存在
if [ ! -d "$DATASET" ]; then
    echo "❌ 错误: 数据集不存在: $DATASET"
    echo "请设置 DATASET 环境变量或修改脚本中的默认路径"
    exit 1
fi

echo "========================================"
echo "BEVCalib 多GPU训练启动"
echo "========================================"
echo "版本: $VERSION"
echo "数据集: $DATASET"
echo "========================================"
echo ""

# 检查是否有正在运行的训练
RUNNING=$(ps aux | grep -E "train_kitti.py|train_B26A.sh" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "⚠️  警告: 检测到 $RUNNING 个正在运行的训练进程"
    echo "是否继续启动新训练? (y/n)"
    read -t 10 -p "> " CONFIRM || CONFIRM="n"
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
fi

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib

echo "启动训练..."
echo ""

# 训练1: 小扰动 (3°, 0.1m) - GPU 0
echo "[GPU 0] 小扰动训练 (3°, 0.1m)..."
nohup bash train_B26A.sh scratch \
    --dataset_root $DATASET \
    --cuda_device 0 \
    --angle_range_deg 3 \
    --trans_range 0.1 \
    --log_suffix small_3deg_${VERSION} \
    &
PID1=$!
echo "  PID: $PID1"
echo "  日志: ./logs/B26A_model_small_3deg_${VERSION}/train.log"

sleep 2

# 训练2: 标准扰动 (5°, 0.3m) - GPU 1
echo "[GPU 1] 标准扰动训练 (5°, 0.3m)..."
nohup bash train_B26A.sh scratch \
    --dataset_root $DATASET \
    --cuda_device 1 \
    --angle_range_deg 5 \
    --trans_range 0.3 \
    --log_suffix small_5deg_${VERSION} \
    &
PID2=$!
echo "  PID: $PID2"
echo "  日志: ./logs/B26A_model_small_5deg_${VERSION}/train.log"

echo ""
echo "========================================"
echo "✅ 所有训练已启动"
echo "========================================"
echo ""
echo "训练进程PID:"
echo "  GPU 0 (小扰动): $PID1"
echo "  GPU 1 (标准扰动): $PID2"
echo ""
echo "查看训练状态:"
echo "  nvidia-smi"
echo "  ps aux | grep train_kitti"
echo ""
echo "查看日志:"
echo "  tail -f ./logs/B26A_model_small_3deg_${VERSION}/train.log"
echo "  tail -f ./logs/B26A_model_small_5deg_${VERSION}/train.log"
echo ""
echo "停止训练:"
echo "  ./stop_training.sh"
echo ""

