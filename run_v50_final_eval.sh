#!/bin/bash
# V50 Final 统一评估入口脚本
#
# 功能:
#   1. test_data_v2 泛化评估 (run_generalization_eval.py)
#   2. BAG 实车泛化评估 (run_bag_calibration)
#   3. 自动生成对比报告
#
# 用法:
#   bash run_v50_final_eval.sh                  # 全量评估 (跳过缺失 checkpoint)
#   bash run_v50_final_eval.sh --report-only    # 仅重新生成报告
#   bash run_v50_final_eval.sh --bag-only       # 仅 BAG 评估
#   bash run_v50_final_eval.sh --eval-only      # 仅 test_data_v2 评估
#   bash run_v50_final_eval.sh --only v50a      # 仅评估 v50a 相关模型

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

MODE="all"
ONLY_FILTER=""

while [ $# -gt 0 ]; do
    case "$1" in
        --report-only) MODE="report" ;;
        --bag-only)    MODE="bag" ;;
        --eval-only)   MODE="eval" ;;
        --only)
            shift
            ONLY_FILTER="${1:-}"
            ;;
    esac
    shift
done

EVAL_CONFIG="configs/eval_generalization_v50_final.yaml"
LOG_DIR="logs/evaluations/generalization_v50_final"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "V50 Final Unified Evaluation"
echo "Mode: $MODE"
echo "Config: $EVAL_CONFIG"
echo "========================================"

if [ "$MODE" = "report" ]; then
    echo "[1/1] Generating report..."
    python run_generalization_eval.py --config "$EVAL_CONFIG" --report_only
    echo "Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
    exit 0
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "eval" ]; then
    echo "[1/2] Running test_data_v2 generalization eval..."
    python run_generalization_eval.py \
        --config "$EVAL_CONFIG" \
        --parallel -1 \
        --eval_max_frames_per_seq 400 \
        --generalization_diag \
        2>&1 | tee "${LOG_DIR}/eval_run.log"
    echo "Eval report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "bag" ]; then
    echo "[2/2] Running BAG calibration eval..."
    if [ -n "$ONLY_FILTER" ]; then
        bash run_bag_calibration_v50_final_batch.sh --only "$ONLY_FILTER" \
            2>&1 | tee "${LOG_DIR}/bag_run.log"
    else
        bash run_bag_calibration_v50_final_batch.sh \
            2>&1 | tee "${LOG_DIR}/bag_run.log"
    fi
    echo "BAG results: /mnt/drtraining/user/dahailu/data/bevcalib/calibration_outputs/v50_final_generalization/"

    echo "Regenerating report with BAG summary..."
    python run_generalization_eval.py --config "$EVAL_CONFIG" --report_only
fi

echo ""
echo "========================================"
echo "V50 Final evaluation complete."
echo "  Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
echo "  Charts: ${LOG_DIR}/charts/"
echo "  BAG:    calibration_outputs/v50_final_generalization/"
echo "========================================"
