#!/bin/bash
# V51 泛化评估入口
#
# 用法:
#   bash run_v51_final_eval.sh
#   bash run_v51_final_eval.sh --report-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

MODE="eval"
while [ $# -gt 0 ]; do
    case "$1" in
        --report-only) MODE="report" ;;
    esac
    shift
done

EVAL_CONFIG="configs/eval_generalization_v51.yaml"
LOG_DIR="logs/evaluations/generalization_v51"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "V51 Generalization Evaluation"
echo "Mode: $MODE"
echo "Config: $EVAL_CONFIG"
echo "========================================"

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
echo "V51 evaluation complete."
echo "  Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
echo "  Charts: ${LOG_DIR}/charts/"
echo "========================================"
