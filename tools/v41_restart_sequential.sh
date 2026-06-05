#!/usr/bin/env bash
# 单节点 8 卡串行跑 V41 A(续训) → B v2 → C v2（避免 DDP/GPU 冲突）
#   bash tools/v41_restart_sequential.sh           # A 续训 + B/C 从头训
#   bash tools/v41_restart_sequential.sh b         # 仅 B v2
#   bash tools/v41_restart_sequential.sh c         # 仅 C v2
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

declare -a CONFIGS=(
    "configs/v41_gmp_a_v32_baseline.yaml"
    "configs/v41_gmp_b_v32_jacloss.yaml"
    "configs/v41_gmp_c_v32_jacloss_match.yaml"
)

start_idx=0
case "${1:-}" in
    b|B) start_idx=1 ;;
    c|C) start_idx=2 ;;
esac

for ((i=start_idx; i<${#CONFIGS[@]}; i++)); do
    cfg="${CONFIGS[$i]}"
    echo "========================================"
    echo "[$(date '+%F %T')] 启动: $cfg"
    echo "========================================"
    bash batch_train.sh --force "$cfg"
    ec=$?
    echo "[$(date '+%F %T')] 结束: $cfg exit=$ec"
    if [[ $ec -ne 0 ]]; then
        echo "训练异常退出，串行队列停止"
        exit "$ec"
    fi
done

echo "[$(date '+%F %T')] V41 串行队列全部完成"
