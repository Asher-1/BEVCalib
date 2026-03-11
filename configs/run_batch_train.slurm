#!/bin/bash
#SBATCH --job-name=bevcalib_batch      # 作业名称
#SBATCH --nodes=2                      # 2台机器
#SBATCH --gres=gpu:2                   # 每台2个GPU
#SBATCH --ntasks-per-node=1            # 每台1个任务（运行batch_train.sh）
#SBATCH --cpus-per-task=16             # 每个任务16个CPU核心
#SBATCH --time=48:00:00                # 最长运行48小时
#SBATCH --output=slurm_%j.log          # 标准输出日志
#SBATCH --error=slurm_%j.err           # 错误日志

# =============================================================================
# BEVCalib SLURM 批量训练作业脚本
#
# 用法:
#   sbatch run_batch_train.slurm [config_file]
#
# 示例:
#   sbatch run_batch_train.slurm configs/batch_train_multinode_slurm.yaml
#   sbatch run_batch_train.slurm  # 使用默认配置
# =============================================================================

set -e

# 配置文件（第一个参数或使用默认值）
CONFIG_FILE=${1:-"configs/batch_train_multinode_slurm.yaml"}

# BEVCalib 项目路径（根据实际情况修改）
PROJECT_DIR="/mnt/drtraining/user/dahailu/code/BEVCalib"

# Conda 环境名称
CONDA_ENV="bevcalib"

# =============================================================================
# 环境检查
# =============================================================================

echo "================================================================"
echo "BEVCalib SLURM 批量训练作业"
echo "================================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "节点数:          $SLURM_NNODES"
echo "每节点GPU数:     $SLURM_GPUS_ON_NODE"
echo "节点列表:        $SLURM_NODELIST"
echo "当前节点:        $SLURMD_NODENAME"
echo "节点Rank:        $SLURM_NODEID"
echo "配置文件:        $CONFIG_FILE"
echo "================================================================"
echo ""

# 激活 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "Conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python 版本: $(python --version)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# 进入项目目录
cd "$PROJECT_DIR"

# =============================================================================
# 运行批量训练
# =============================================================================

echo "开始批量训练..."
echo "配置文件: $CONFIG_FILE"
echo ""

# 运行 batch_train.sh
# SLURM 环境变量会被 start_training.sh 自动检测
bash batch_train.sh "$CONFIG_FILE"

# =============================================================================
# 完成
# =============================================================================

echo ""
echo "================================================================"
echo "批量训练作业完成"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"
