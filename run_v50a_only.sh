#!/bin/bash
# 本机仅训练 v50a，完成后 batch_train 自动退出。
# v50b/v50c 见 configs/v50_bc_cf_bev_r.yaml（远端机器）
#
# 用法:
#   bash run_v50a_only.sh
#   nohup bash run_v50a_only.sh > logs/v50a_train.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

echo "========================================"
echo "V50a 单实验训练 (本机)"
echo "配置: configs/v50_cf_bev_r.yaml"
echo "v50b/v50c 请在远端运行 configs/v50_bc_cf_bev_r.yaml"
echo "========================================"

# 启动完成守护（后台）
nohup bash tools/v50a_finish_guard.sh >> logs/v50a_finish_guard.log 2>&1 &
GUARD_PID=$!
echo "完成守护已启动 (PID: $GUARD_PID, 日志: logs/v50a_finish_guard.log)"

bash batch_train.sh configs/v50_cf_bev_r.yaml

echo "V50a 训练流程结束"
