#!/bin/bash
# V54 后续全量执行流水线（串行，每步占满 8 GPU）
#
# 步骤:
#   1. v54c 硬门控复测 (gate-eval-only)
#   2. v54d smoke (15ep)
#   3. v54b DP-Head refine (20ep)
#   4. v53f smoke (15ep)
#   5. v54d full (40ep)
#   6. v53f full (60ep)
#
# 用法:
#   nohup bash run_all_v54_pipeline.sh > logs/pipeline_v54_next.log 2>&1 &
#   tail -f logs/pipeline_v54_next.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

LOG_ROOT="logs/pipeline_v54_next"
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
    local log="$LOG_ROOT/$(basename "${cfg%.yaml}").log"
    _step "$3" "batch_train $cfg ${skip}"
    if [ -n "$skip" ]; then
        bash batch_train.sh --skip-pattern "$skip" "$cfg" 2>&1 | tee "$log"
    else
        bash batch_train.sh "$cfg" 2>&1 | tee "$log"
    fi
}

_step "0" "Pipeline start — GPU queue (8-GPU DDP per job)"

# --- 1. 硬门控复测（验证推理硬门控对现有 v54c ckpt 的 ZD 改善）---
_step "1" "v54c hard-gate re-eval"
bash run_v54c_eval.sh --gate-eval-only 2>&1 | tee "$LOG_ROOT/step1_v54c_gate_reeval.log"

# --- 2. v54d smoke ---
_run_train "configs/v54d_router_fix_cf_bev_r.yaml" "full" "2"

# --- 3. v54b DP-Head refine ---
_run_train "configs/v54b_dphead_refine_cf_bev_r.yaml" "" "3"

# --- 4. v53f smoke ---
_run_train "configs/v53f_gin_jacg_mgda_cf_bev_r.yaml" "full" "4"

# --- 5. v54d full ---
_run_train "configs/v54d_router_fix_cf_bev_r.yaml" "smoke" "5"

# --- 6. v53f full ---
_run_train "configs/v53f_gin_jacg_mgda_cf_bev_r.yaml" "smoke" "6"

_step "DONE" "All pipeline steps completed"
echo "Logs: $LOG_ROOT/"
echo "  step1: step1_v54c_gate_reeval.log"
echo "  v54d:  v54d_router_fix_cf_bev_r.log"
echo "  v54b:  v54b_dphead_refine_cf_bev_r.log"
echo "  v53f:  v53f_gin_jacg_mgda_cf_bev_r.log"
