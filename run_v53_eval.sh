#!/bin/bash
# V53 泛化评估 + 可选门控 eval
#
# 用法:
#   bash run_v53_eval.sh                      # 全量单 ckpt eval
#   bash run_v53_eval.sh --gate-eval          # 额外跑 v53a dual+val 门控 gdiag
#   bash run_v53_eval.sh --gate-eval-only     # 仅门控 eval（已有主 eval 时）
#   bash run_v53_eval.sh --precheck-only
#   bash run_v53_eval.sh --report-only

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

EVAL_CONFIG="configs/eval_generalization_v53.yaml"
LOG_DIR="logs/evaluations/generalization_v53"
GATE_LABEL="v53a-dphead-gate"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "V53 Generalization Evaluation"
echo "Mode: $MODE  gate_eval=$GATE_EVAL"
echo "Config: $EVAL_CONFIG"
echo "========================================"

_run_gate_eval() {
    echo ""
    echo "--- V53a deploy gate eval (dual MEDW + val Recovery gdiag) ---"
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
        echo "[WARN] Gate eval failed — ckpt 可能尚未训练完成，跳过验收报告"
    fi
}

if [ "$MODE" = "precheck" ]; then
    python - <<'PY'
import yaml, os
cfg = yaml.safe_load(open("configs/eval_generalization_v53.yaml"))
root = cfg["bevcalib_root"]
models_dir = os.path.join(root, cfg["models_dir"])
print("\nCheckpoint precheck:")
for m in cfg["models"]:
    base = os.path.join(models_dir, m["dir_name"], "all_training_data_scratch/checkpoint")
    path = os.path.join(base, m["ckpt"])
    ok = os.path.isfile(path)
    print(f"  {'✓' if ok else '✗'} {m['label']}: {m['ckpt']}")
if cfg.get("deploy_policy"):
    p = cfg["deploy_policy"]
    print(f"\nDeploy gate policy: gate_deg={p.get('gate_deg')}")
    for k in ("ckpt_medw", "ckpt_recovery"):
        e = p.get(k) or {}
        print(f"  {k}: {e.get('dir_name')}/{e.get('ckpt')}")
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
echo "V53 evaluation complete."
echo "  Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
if [ "$GATE_EVAL" -eq 1 ]; then
    echo "  Gate JSON: ${LOG_DIR}/${GATE_LABEL}/generalization_diagnostics_gate.json"
    echo "  Gate acceptance: ${LOG_DIR}/DEPLOY_GATE_ACCEPTANCE.md"
fi
echo "  Log: ${LOG_DIR}/eval_run.log"
echo "========================================"
