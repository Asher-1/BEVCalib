#!/bin/bash
# 机器 B 评估流水线（串行，每步占满 8 GPU）
#
# 步骤:
#   1. v54d 泛化 eval + gate
#   2. v53f 泛化 eval + gate
#
# 用法:
#   bash run_machine_b_eval_pipeline.sh --precheck-only
#   nohup bash run_machine_b_eval_pipeline.sh > logs/pipeline_machine_b_eval/master.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

PRECHECK_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --precheck-only) PRECHECK_ONLY=1 ;;
    esac
    shift
done

LOG_ROOT="logs/pipeline_machine_b_eval"
mkdir -p "$LOG_ROOT"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(_ts)] Machine B eval pipeline"
echo "  Step 1: v54d generalization + gate"
echo "  Step 2: v53f generalization + gate"

if [ "$PRECHECK_ONLY" -eq 1 ]; then
    bash run_v54d_eval.sh --precheck-only
    bash run_v53f_eval.sh --precheck-only
    exit 0
fi

echo ""
echo "======== STEP 1: v54d eval ========"
bash run_v54d_eval.sh --gate-eval 2>&1 | tee "${LOG_ROOT}/step1_v54d_eval.log"

echo ""
echo "======== STEP 2: v53f eval ========"
bash run_v53f_eval.sh --gate-eval 2>&1 | tee "${LOG_ROOT}/step2_v53f_eval.log"

echo ""
echo "[$(_ts)] Machine B eval pipeline DONE"
echo "  v54d: logs/evaluations/generalization_v54d/GENERALIZATION_REPORT.md"
echo "  v53f: logs/evaluations/generalization_v53f/GENERALIZATION_REPORT.md"
