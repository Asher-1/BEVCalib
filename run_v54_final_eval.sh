#!/bin/bash
# V54 最终泛化评估（全训练完成后统一对比）
#
# 用法:
#   bash run_v54_final_eval.sh --precheck-only
#   bash run_v54_final_eval.sh --gate-eval
#   nohup bash run_v54_final_eval.sh --gate-eval > logs/evaluations/generalization_v54_final/nohup.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

MODE="eval"
GATE_EVAL=0
GATE_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --report-only) MODE="report" ;;
        --precheck-only) MODE="precheck" ;;
        --gate-eval) GATE_EVAL=1 ;;
        --gate-eval-only) GATE_EVAL=1; GATE_ONLY=1 ;;
    esac
    shift
done

EVAL_CONFIG="configs/eval_generalization_v54_final.yaml"
LOG_DIR="logs/evaluations/generalization_v54_final"
GATE_LABEL="v54-final-gate"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "V54 Final Generalization Evaluation"
echo "Mode: $MODE  gate_eval=$GATE_EVAL"
echo "Config: $EVAL_CONFIG"
echo "========================================"

_run_gate_eval() {
    echo ""
    echo "--- Final deploy gate eval ---"
    if python scripts/evaluate_deploy_gate.py \
        --config "$EVAL_CONFIG" \
        --output-label "$GATE_LABEL" \
        2>&1 | tee "${LOG_DIR}/gate_eval_run.log"; then
        python scripts/summarize_v52_deploy_metrics.py \
            --config "$EVAL_CONFIG" \
            --report-dir "$LOG_DIR" \
            --gate-label "$GATE_LABEL"
        echo "Gate acceptance: ${LOG_DIR}/DEPLOY_GATE_ACCEPTANCE.md"
    else
        echo "[WARN] Gate eval failed"
    fi
}

if [ "$MODE" = "precheck" ]; then
    python - <<'PY'
import yaml, os
cfg = yaml.safe_load(open("configs/eval_generalization_v54_final.yaml"))
root = cfg["bevcalib_root"]
models_dir = os.path.join(root, cfg["models_dir"])
print("\nCheckpoint precheck:")
ok_all = True
for m in cfg["models"]:
    base = os.path.join(models_dir, m["dir_name"], "all_training_data_scratch/checkpoint")
    path = os.path.join(base, m["ckpt"])
    ok = os.path.isfile(path)
    ok_all = ok_all and ok
    print(f"  {'✓' if ok else '✗'} {m['label']}: {m['ckpt']}")
print(f"\n{'All OK' if ok_all else 'MISSING checkpoints — fix before eval'}")
PY
    exit 0
fi

if [ "$MODE" = "report" ]; then
    python run_generalization_eval.py --config "$EVAL_CONFIG" --report_only
    python scripts/summarize_v52_deploy_metrics.py \
        --config "$EVAL_CONFIG" \
        --report-dir "$LOG_DIR" \
        --gate-label "$GATE_LABEL" || true
    echo "Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
    exit 0
fi

if [ "$GATE_ONLY" -eq 0 ]; then
    python run_generalization_eval.py \
        --config "$EVAL_CONFIG" \
        --parallel -1 \
        --eval_max_frames_per_seq 400 \
        --generalization_diag \
        2>&1 | tee "${LOG_DIR}/eval_run.log"

    python scripts/summarize_v52_deploy_metrics.py \
        --config "$EVAL_CONFIG" \
        --report-dir "$LOG_DIR" \
        --gate-label "$GATE_LABEL" || true
fi

if [ "$GATE_EVAL" -eq 1 ]; then
    _run_gate_eval
fi

echo ""
echo "========================================"
echo "V54 final evaluation complete."
echo "  Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
if [ "$GATE_EVAL" -eq 1 ]; then
    echo "  Gate acceptance: ${LOG_DIR}/DEPLOY_GATE_ACCEPTANCE.md"
fi
echo "========================================"
