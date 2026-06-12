#!/bin/bash
# V52 泛化评估入口
#
# 用法:
#   bash run_v52_final_eval.sh                  # 评估就绪模型 + 生成报告
#   bash run_v52_final_eval.sh --wait-training  # 等 v52a S1/S2 + v52b 训练完成后评估
#   bash run_v52_final_eval.sh --precheck-only  # 仅检查 ckpt 就绪情况
#   bash run_v52_final_eval.sh --report-only    # 仅从已有结果重生成报告

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

MODE="eval"
WAIT_TRAINING=0
while [ $# -gt 0 ]; do
    case "$1" in
        --report-only) MODE="report" ;;
        --precheck-only) MODE="precheck" ;;
        --wait-training) WAIT_TRAINING=1 ;;
    esac
    shift
done

EVAL_CONFIG="configs/eval_generalization_v52.yaml"
LOG_DIR="logs/evaluations/generalization_v52_all"
MODELS_ROOT="logs/all_training_data"
mkdir -p "$LOG_DIR"

_log_epoch() {
    local log="$1"
    if [ -f "$log" ]; then
        grep -oE 'Epoch \[[0-9]+/[0-9]+\]' "$log" | tail -1 || echo "no epoch yet"
    else
        echo "log missing"
    fi
}

_training_done() {
    local log="$1"
    [ -f "$log" ] && grep -q "训练完成总结 / Training Summary" "$log"
}

_wait_for_training() {
    local logs=(
        "${MODELS_ROOT}/model_small_5deg_v52a_S1_full/train.log"
        "${MODELS_ROOT}/model_small_5deg_v52a_S2_full/train.log"
        "${MODELS_ROOT}/model_small_5deg_v52b_full/train.log"
    )
    local ckpts=(
        "${MODELS_ROOT}/model_small_5deg_v52a_S2_full/all_training_data_scratch/checkpoint/ckpt_best_val.pth"
        "${MODELS_ROOT}/model_small_5deg_v52b_full/all_training_data_scratch/checkpoint/ckpt_best_val.pth"
    )
    echo "Waiting for v52 training to complete..."
    while true; do
        local all_done=1
        echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---"
        for log in "${logs[@]}"; do
            local name
            name=$(basename "$(dirname "$log")")
            if _training_done "$log"; then
                echo "  ✓ $name: DONE"
            else
                echo "  ⏳ $name: $(_log_epoch "$log")"
                all_done=0
            fi
        done
        local ckpt_ok=1
        for ckpt in "${ckpts[@]}"; do
            if [ ! -f "$ckpt" ]; then
                ckpt_ok=0
            fi
        done
        if [ "$all_done" -eq 1 ] && [ "$ckpt_ok" -eq 1 ]; then
            echo "All training finished and key checkpoints exist."
            break
        fi
        sleep 300
    done
}

echo "========================================"
echo "V52 Generalization Evaluation"
echo "Mode: $MODE"
echo "Config: $EVAL_CONFIG"
echo "========================================"

if [ "$WAIT_TRAINING" -eq 1 ] && [ "$MODE" = "eval" ]; then
    _wait_for_training
fi

if [ "$MODE" = "precheck" ]; then
    python run_generalization_eval.py --config "$EVAL_CONFIG" --report_only 2>&1 | head -80 || true
    python - <<'PY'
import yaml, os
cfg = yaml.safe_load(open("configs/eval_generalization_v52.yaml"))
root = cfg["bevcalib_root"]
models_dir = os.path.join(root, cfg["models_dir"])
print("\nCheckpoint precheck:")
for m in cfg["models"]:
    base = os.path.join(models_dir, m["dir_name"], "all_training_data_scratch/checkpoint")
    ckpt = m["ckpt"]
    path = os.path.join(base, ckpt)
    ok = os.path.isfile(path)
    mark = "✓" if ok else "✗"
    print(f"  {mark} {m['label']}: {ckpt} {'(ready)' if ok else '(missing)'}")
PY
    exit 0
fi

if [ "$MODE" = "report" ]; then
    python run_generalization_eval.py --config "$EVAL_CONFIG" --report_only
    echo "Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
    exit 0
fi

python run_generalization_eval.py \
    --config "$EVAL_CONFIG" \
    --parallel -1 \
    --eval_max_frames_per_seq 400 \
    --generalization_diag \
    2>&1 | tee "${LOG_DIR}/eval_run.log"

echo ""
echo "========================================"
echo "V52 evaluation complete."
echo "  Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
echo "  Charts: ${LOG_DIR}/charts/"
echo "  Log:    ${LOG_DIR}/eval_run.log"
echo "========================================"
