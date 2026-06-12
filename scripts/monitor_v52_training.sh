#!/bin/bash
# V52 训练进度监控 (单次快照或循环)
#
#   bash scripts/monitor_v52_training.sh          # 打印一次
#   bash scripts/monitor_v52_training.sh --watch  # 每 5min 刷新

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(dirname "$0")")" && pwd)"
cd "$SCRIPT_DIR"

WATCH=0
[ "${1:-}" = "--watch" ] && WATCH=1

print_status() {
    echo "========== V52 Training Monitor @ $(date '+%Y-%m-%d %H:%M:%S') =========="
    for name in v52a_S1_full v52a_S2_full v52b_full; do
        log="logs/all_training_data/model_small_5deg_${name}/train.log"
        ckpt_dir="logs/all_training_data/model_small_5deg_${name}/all_training_data_scratch/checkpoint"
        echo ""
        echo "[$name]"
        if [ ! -f "$log" ]; then
            echo "  status: NOT STARTED"
            continue
        fi
        if grep -q "训练完成总结 / Training Summary" "$log"; then
            echo "  status: COMPLETED"
            grep -E "Best Val|Best Dual Gate|Best Jacobian|Training Time" "$log" | tail -6 | sed 's/^/  /'
        else
            ep=$(grep -oE 'Epoch \[[0-9]+/[0-9]+\]' "$log" | tail -1 || echo "?")
            eta=$(grep "ETA=" "$log" | tail -1 | grep -oE 'ETA=[0-9.]+h' || true)
            echo "  status: RUNNING  $ep  ${eta:-}"
        fi
        nan=$(grep -c "NaN GUARD" "$log" 2>/dev/null || true)
        nan=${nan:-0}
        if [ "${nan}" -gt 0 ] 2>/dev/null; then
            echo "  warn: NaN GUARD events=${nan}"
        fi
        if [ -d "$ckpt_dir" ]; then
            ls -1 "$ckpt_dir"/ckpt_*.pth 2>/dev/null | xargs -n1 basename | tr '\n' ' ' | sed 's/^/  ckpts: /' || true
            echo ""
        fi
    done
    echo ""
    echo "================================================================"
}

print_status

if [ "$WATCH" -eq 1 ]; then
    while true; do
        sleep 300
        print_status
    done
fi
