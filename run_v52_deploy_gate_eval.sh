#!/bin/bash
# V52 双 ckpt 门控泛化评估 + 验收报告
#
# 用法:
#   bash run_v52_deploy_gate_eval.sh
#   bash run_v52_deploy_gate_eval.sh --dry-run
#   bash run_v52_deploy_gate_eval.sh --report-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

MODE="eval"
while [ $# -gt 0 ]; do
    case "$1" in
        --report-only) MODE="report" ;;
        --dry-run) MODE="dry-run" ;;
    esac
    shift
done

EVAL_CONFIG="configs/eval_generalization_v52_deploy.yaml"
LOG_DIR="logs/evaluations/generalization_v52_deploy"
GATE_LABEL="v52a-S1-v2-deploy-gate"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "V52 Deploy GATE Evaluation"
echo "Mode: $MODE"
echo "Config: $EVAL_CONFIG"
echo "========================================"

if [ "$MODE" = "dry-run" ]; then
    python scripts/evaluate_deploy_gate.py --config "$EVAL_CONFIG" --dry-run
    exit 0
fi

if [ "$MODE" = "report" ]; then
    python scripts/summarize_v52_deploy_metrics.py \
        --config "$EVAL_CONFIG" \
        --report-dir "$LOG_DIR" \
        --gate-label "$GATE_LABEL"
    echo "Gate acceptance: ${LOG_DIR}/DEPLOY_GATE_ACCEPTANCE.md"
    exit 0
fi

python scripts/evaluate_deploy_gate.py \
    --config "$EVAL_CONFIG" \
    --output-label "$GATE_LABEL" \
    2>&1 | tee "${LOG_DIR}/gate_eval_run.log"

python scripts/summarize_v52_deploy_metrics.py \
    --config "$EVAL_CONFIG" \
    --report-dir "$LOG_DIR" \
    --gate-label "$GATE_LABEL"

echo ""
echo "========================================"
echo "V52 deploy GATE evaluation complete."
echo "  Gate JSON: ${LOG_DIR}/${GATE_LABEL}/generalization_diagnostics_gate.json"
echo "  Gate acceptance: ${LOG_DIR}/DEPLOY_GATE_ACCEPTANCE.md"
echo "  Log: ${LOG_DIR}/gate_eval_run.log"
echo "========================================"
