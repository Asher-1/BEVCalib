#!/bin/bash
# V55 全量串行训练流水线（单机 8 GPU 顺序占用）
#
# 步骤:
#   1. v55 main smoke
#   2. v55-safe smoke
#   3. v55 main full
#   4. v55-safe full
#   5. v55 main refine
#   6. v55-safe refine
#
# 用法:
#   nohup bash run_all_v55_pipeline.sh > logs/pipeline_v55/nohup.log 2>&1 &
#   tail -f logs/pipeline_v55/nohup.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

LOG_ROOT="logs/pipeline_v55"
mkdir -p "$LOG_ROOT"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
_step() {
    echo ""
    echo "================================================================"
    echo "[$(_ts)] STEP $1: $2"
    echo "================================================================"
}

_run_train() {
    local cfg="$1"
    local skip="${2:-}"
    local step_no="$3"
    local log="$LOG_ROOT/$(basename "${cfg%.yaml}")_step${step_no}.log"
    _step "$step_no" "batch_train $cfg ${skip}"
    if [ -n "$skip" ]; then
        bash batch_train.sh --skip-pattern "$skip" "$cfg" 2>&1 | tee "$log"
    else
        bash batch_train.sh "$cfg" 2>&1 | tee "$log"
    fi
}

_step "0" "V55 pipeline start — main + safe unified training queue"

# --- 1. main smoke ---
_run_train "configs/v55_deploy_generalization_cf_bev_r.yaml" "full|refine" "1"

# --- 2. safe smoke ---
_run_train "configs/v55_safe_deploy_generalization_cf_bev_r.yaml" "full|refine" "2"

# --- 3. main full ---
_run_train "configs/v55_deploy_generalization_cf_bev_r.yaml" "smoke|refine" "3"

# --- 4. safe full ---
_run_train "configs/v55_safe_deploy_generalization_cf_bev_r.yaml" "smoke|refine" "4"

# --- 5. main refine ---
_run_train "configs/v55_deploy_generalization_cf_bev_r.yaml" "smoke|full" "5"

# --- 6. safe refine ---
_run_train "configs/v55_safe_deploy_generalization_cf_bev_r.yaml" "smoke|full" "6"

_step "DONE" "All V55 pipeline steps completed"
echo "Logs: $LOG_ROOT/"
