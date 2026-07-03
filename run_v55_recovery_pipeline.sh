#!/bin/bash
# V55 恢复流水线:
#   1. 立即切到 V55-safe full
#   2. 跑完后自动接 V55b smoke

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
    local skip="$2"
    local step_no="$3"
    local log="$LOG_ROOT/$(basename "${cfg%.yaml}")_recovery_step${step_no}.log"
    _step "$step_no" "batch_train $cfg --skip-pattern '$skip'"
    bash batch_train.sh --skip-pattern "$skip" "$cfg" 2>&1 | tee "$log"
}

_step "0" "V55 recovery pipeline start"

# 1. 稳定底盘先跑出来
_run_train "configs/v55_safe_deploy_generalization_cf_bev_r.yaml" "smoke|refine" "1"

# 2. 收紧版主线先做 smoke
_run_train "configs/v55b_route_tight_cf_bev_r.yaml" "full" "2"

_step "DONE" "V55 recovery pipeline completed"
echo "Logs: $LOG_ROOT/"
