#!/bin/bash
# =============================================================================
# BEVCalib 多GPU训练启动脚本
# 支持多数据集，自动组织日志结构
#
# Usage:
#   bash start_training.sh [dataset_choice] [version] [options]
#
# Dataset Choices:
#   B26A         - B26A 数据集 (bevcalib_training_data)
#   all          - 全量数据集 (all_training_data) 
#   custom       - 自定义数据集路径
#
# Examples:
#   # 使用 B26A 数据集
#   bash start_training.sh B26A v1
#
#   # 使用全量数据集
#   bash start_training.sh all v1
#
#   # 使用自定义数据集
#   CUSTOM_DATASET=/path/to/dataset bash start_training.sh custom v1
# =============================================================================

set -e

# ============================================================================
# 参数解析
# ============================================================================
DATASET_CHOICE=${1:-B26A}  # B26A, all, 或 custom
VERSION=${2:-v1}

# ============================================================================
# 数据集配置
# ============================================================================
case $DATASET_CHOICE in
    B26A|b26a)
        DATASET_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data"
        DATASET_NAME="B26A"
        echo "ℹ️  使用 B26A 数据集"
        ;;
    all|ALL)
        DATASET_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data"
        DATASET_NAME="all_training_data"
        echo "ℹ️  使用全量数据集 (all_training_data)"
        ;;
    custom|CUSTOM)
        if [ -z "$CUSTOM_DATASET" ]; then
            echo "❌ 错误: 使用自定义数据集时，请设置 CUSTOM_DATASET 环境变量"
            echo ""
            echo "示例:"
            echo "  CUSTOM_DATASET=/path/to/dataset bash start_training.sh custom v1"
            exit 1
        fi
        DATASET_ROOT="$CUSTOM_DATASET"
        DATASET_NAME=$(basename "$DATASET_ROOT")
        echo "ℹ️  使用自定义数据集: $DATASET_NAME"
        ;;
    *)
        echo "❌ 错误: 未知的数据集选择 '$DATASET_CHOICE'"
        echo ""
        echo "可用选项:"
        echo "  B26A   - B26A 数据集"
        echo "  all    - 全量数据集"
        echo "  custom - 自定义数据集 (需设置 CUSTOM_DATASET 环境变量)"
        echo ""
        echo "示例:"
        echo "  bash start_training.sh B26A v1"
        echo "  bash start_training.sh all v1"
        echo "  CUSTOM_DATASET=/path bash start_training.sh custom v1"
        exit 1
        ;;
esac

# 检查数据集是否存在
if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ 错误: 数据集不存在: $DATASET_ROOT"
    exit 1
fi

if [ ! -d "$DATASET_ROOT/sequences" ]; then
    echo "❌ 错误: 数据集格式错误，未找到 sequences/ 目录"
    exit 1
fi

echo "========================================"
echo "BEVCalib 多GPU训练启动"
echo "========================================"
echo "数据集: $DATASET_NAME"
echo "数据集路径: $DATASET_ROOT"
echo "版本: $VERSION"
echo "日志目录: ./logs/${DATASET_NAME}/"
echo "========================================"
echo ""

# 检查是否有正在运行的训练
RUNNING=$(ps aux | grep -E "train_kitti.py|train_universal.sh" | grep -v grep | wc -l)
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

# ============================================================================
# 训练配置（根据论文设置不同扰动级别）
# ============================================================================

# 训练1: 小扰动 (10°, 0.5m) - GPU 0
echo "[GPU 0] 小扰动训练 (10°, 0.5m)..."
nohup bash train_universal.sh scratch \
    --dataset_root $DATASET_ROOT \
    --dataset_name $DATASET_NAME \
    --cuda_device 0 \
    --angle_range_deg 10 \
    --trans_range 0.5 \
    --log_suffix small_10deg_${VERSION} \
    > /dev/null 2>&1 &
PID1=$!
echo "  PID: $PID1"
echo "  日志: ./logs/${DATASET_NAME}/model_small_10deg_${VERSION}/train.log"

sleep 2

# 训练2: 标准扰动 (5°, 0.3m) - GPU 1
echo "[GPU 1] 标准扰动训练 (5°, 0.3m)..."
nohup bash train_universal.sh scratch \
    --dataset_root $DATASET_ROOT \
    --dataset_name $DATASET_NAME \
    --cuda_device 1 \
    --angle_range_deg 5 \
    --trans_range 0.3 \
    --log_suffix small_5deg_${VERSION} \
    > /dev/null 2>&1 &
PID2=$!
echo "  PID: $PID2"
echo "  日志: ./logs/${DATASET_NAME}/model_small_5deg_${VERSION}/train.log"

# 可选：训练3 - 大扰动 (20°, 1.5m) - GPU 2
# 如果有第三块GPU，可以取消注释
# echo "[GPU 2] 大扰动训练 (20°, 1.5m)..."
# nohup bash train_universal.sh scratch \
#     --dataset_root $DATASET_ROOT \
#     --dataset_name $DATASET_NAME \
#     --cuda_device 2 \
#     --angle_range_deg 20 \
#     --trans_range 1.5 \
#     --log_suffix large_20deg_${VERSION} \
#     > /dev/null 2>&1 &
# PID3=$!
# echo "  PID: $PID3"
# echo "  日志: ./logs/${DATASET_NAME}/model_large_20deg_${VERSION}/train.log"

echo ""
echo "========================================"
echo "✅ 所有训练已启动"
echo "========================================"
echo ""
echo "训练进程PID:"
echo "  GPU 0 (小扰动 10°): $PID1"
echo "  GPU 1 (标准扰动 5°): $PID2"
# echo "  GPU 3 (大扰动 20°): $PID3"
echo ""
echo "日志结构:"
echo "  logs/"
echo "  └── ${DATASET_NAME}/"
echo "      ├── model_small_10deg_${VERSION}/"
echo "      │   ├── train.log"
echo "      │   ├── events.out.tfevents.*"
echo "      │   └── epoch_*.pth"
echo "      └── model_small_5deg_${VERSION}/"
echo "          ├── train.log"
echo "          └── ..."
echo ""
echo "查看训练状态:"
echo "  nvidia-smi"
echo "  ps aux | grep train_kitti"
echo ""
echo "查看日志:"
echo "  tail -f ./logs/${DATASET_NAME}/model_small_10deg_${VERSION}/train.log"
echo "  tail -f ./logs/${DATASET_NAME}/model_small_5deg_${VERSION}/train.log"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir ./logs/${DATASET_NAME}/ --port 6006"
echo ""
echo "停止训练:"
echo "  ./stop_training.sh"
echo ""
