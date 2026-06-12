#!/bin/bash
# V52 部署导向泛化评估
#
# 用法:
#   bash run_v52_deploy_eval.sh
#   bash run_v52_deploy_eval.sh --precheck-only
#   bash run_v52_deploy_eval.sh --report-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

MODE="eval"
while [ $# -gt 0 ]; do
    case "$1" in
        --report-only) MODE="report" ;;
        --precheck-only) MODE="precheck" ;;
    esac
    shift
done

EVAL_CONFIG="configs/eval_generalization_v52_deploy.yaml"
LOG_DIR="logs/evaluations/generalization_v52_deploy"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "V52 Deploy Generalization Evaluation"
echo "Mode: $MODE"
echo "Config: $EVAL_CONFIG"
echo "========================================"

if [ "$MODE" = "precheck" ]; then
    python - <<'PY'
import yaml, os
cfg = yaml.safe_load(open("configs/eval_generalization_v52_deploy.yaml"))
root = cfg["bevcalib_root"]
models_dir = os.path.join(root, cfg["models_dir"])
print("\nCheckpoint precheck:")
for m in cfg["models"]:
    base = os.path.join(models_dir, m["dir_name"], "all_training_data_scratch/checkpoint")
    path = os.path.join(base, m["ckpt"])
    ok = os.path.isfile(path)
    print(f"  {'✓' if ok else '✗'} {m['label']}: {m['ckpt']}")
if cfg.get("deploy_policy"):
    print("\nDeploy policy (文档/ summarize 用，eval 主流程不自动门控):")
    p = cfg["deploy_policy"]
    print(f"  gate_deg={p.get('gate_deg')}  medw_n={p.get('medw_n_frames')}")
PY
    exit 0
fi

if [ "$MODE" = "report" ]; then
    python run_generalization_eval.py --config "$EVAL_CONFIG" --report_only
    python scripts/summarize_v52_deploy_metrics.py --config "$EVAL_CONFIG" \
        --report-dir "$LOG_DIR" || true
    echo "Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
    echo "Deploy summary: ${LOG_DIR}/DEPLOY_ACCEPTANCE.md"
    exit 0
fi

python run_generalization_eval.py \
    --config "$EVAL_CONFIG" \
    --parallel -1 \
    --eval_max_frames_per_seq 400 \
    --generalization_diag \
    2>&1 | tee "${LOG_DIR}/eval_run.log"

python scripts/summarize_v52_deploy_metrics.py \
    --config "$EVAL_CONFIG" \
    --report-dir "$LOG_DIR"

echo ""
echo "========================================"
echo "V52 deploy evaluation complete."
echo "  Report: ${LOG_DIR}/GENERALIZATION_REPORT.md"
echo "  Deploy acceptance: ${LOG_DIR}/DEPLOY_ACCEPTANCE.md"
echo "  Log: ${LOG_DIR}/eval_run.log"
echo "========================================"
