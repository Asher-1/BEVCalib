#!/bin/bash
# V50 Final 统一 BAG 泛化评估
#
# 评估模型:
#   v45c-S1-quick-best-val, v45c-S2-quick-best-dual
#   v49b-no-cons-best-dual
#   v49-optuna-t32-full (best_dual + best_val)
#   v50a/b/c (best_dual + best_val, 训练完成后自动纳入)
#
# 三种测试场景: baseline / shortcut(2,2,2) / inject_small(0.5,0.5,0.5)
#
# 用法:
#   bash run_bag_calibration_v50_final_batch.sh
#   bash run_bag_calibration_v50_final_batch.sh --only v50a   # 仅评估 v50a

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

ONLY_FILTER="${1:-}"
if [ "${ONLY_FILTER}" = "--only" ] && [ -n "${2:-}" ]; then
    ONLY_FILTER="$2"
else
    ONLY_FILTER=""
fi

BASE_CONFIG="configs/calibration_v50_final_generalization.yaml"
OUTPUT_BASE="/mnt/drtraining/user/dahailu/data/bevcalib/calibration_outputs"
LOGS_BASE="logs/all_training_data"

declare -a MODEL_LABELS=(
    "v45c-S1-quick"
    "v45c-S2-quick"
    "v49b-no-cons"
    "v49-optuna-t32-full"
    "v50a-optuna-no-cons-full"
    "v50b-optuna-256g-no-cons-full"
    "v50c-optuna-cond-cons-full"
)

declare -a MODEL_DIRS=(
    "model_small_5deg_v45c_cf_bev_r_S1_quick"
    "model_small_3deg_v45c_cf_bev_r_S2_quick"
    "model_small_5deg_v49b_no_cons_S1_quick"
    "model_small_5deg_v49_optuna_best_t32_S1_full"
    "model_small_5deg_v50a_optuna_no_cons_S1_full"
    "model_small_5deg_v50b_optuna_256g_no_cons_S1_full"
    "model_small_5deg_v50c_optuna_cond_cons_S1_full"
)

# 仅评估指定 checkpoint 类型 (dual / val / both)
declare -a CKPT_TYPES=(
    "best_dual|ckpt_best_dual.pth"
    "best_val|ckpt_best_val.pth"
)

# v45c-S1 和 v49b 用户指定了特定 ckpt, 用 SKIP 标记控制
declare -a CKPT_FILTER=(
    "val_only"
    "dual_only"
    "dual_only"
    "both"
    "both"
    "both"
    "both"
)

declare -a TEST_SCENARIOS=(
    "baseline|"
    "shortcut|2,2,2"
    "inject_small|0.5,0.5,0.5"
)

should_run_ckpt() {
    local filter="$1"
    local ckpt_label="$2"
    case "$filter" in
        val_only)  [ "$ckpt_label" = "best_val" ] ;;
        dual_only) [ "$ckpt_label" = "best_dual" ] ;;
        both)      true ;;
        *)         true ;;
    esac
}

run_single() {
    local label="$1"
    local model_dir="$2"
    local ckpt_label="$3"
    local ckpt_file="$4"
    local scenario_name="$5"
    local inject_rpy="$6"

    local ckpt_path="${LOGS_BASE}/${model_dir}/all_training_data_scratch/checkpoint/${ckpt_file}"
    if [ ! -f "$ckpt_path" ]; then
        echo "[SKIP] ${label}/${ckpt_label}/${scenario_name}: checkpoint not found"
        return 0
    fi

    local output_dir="${OUTPUT_BASE}/v50_final_generalization/${label}_${ckpt_label}_${scenario_name}"
    if [ -f "${output_dir}/calibration_summary.md" ]; then
        echo "[SKIP] ${label}/${ckpt_label}/${scenario_name}: already completed"
        return 0
    fi

    echo "[RUN] ${label}/${ckpt_label}/${scenario_name}"
    echo "  ckpt: ${ckpt_path}"
    echo "  output: ${output_dir}"

    mkdir -p "${output_dir}"

    local cmd="python run_bag_calibration.py \
        --config ${BASE_CONFIG} \
        --ckpt_path ${ckpt_path} \
        --output_dir ${output_dir}"

    if [ -n "$inject_rpy" ]; then
        cmd="${cmd} --inject_lidar_rpy_deg ${inject_rpy}"
        echo "  inject: ${inject_rpy}"
    fi

    eval "$cmd" 2>&1 | tee "${output_dir}/run.log" || {
        echo "[FAIL] ${label}/${ckpt_label}/${scenario_name} exited with $?"
        return 0
    }
    echo "[DONE] ${label}/${ckpt_label}/${scenario_name}"
}

N_MODELS=${#MODEL_LABELS[@]}
TOTAL=0
for ((i=0; i<N_MODELS; i++)); do
    if [ -n "$ONLY_FILTER" ] && [[ ! "${MODEL_LABELS[$i]}" == *"$ONLY_FILTER"* ]]; then
        continue
    fi
    for ckpt_str in "${CKPT_TYPES[@]}"; do
        IFS='|' read -r ckpt_label ckpt_file <<< "$ckpt_str"
        if should_run_ckpt "${CKPT_FILTER[$i]}" "$ckpt_label"; then
            TOTAL=$((TOTAL + ${#TEST_SCENARIOS[@]}))
        fi
    done
done

echo "========================================"
echo "V50 Final BAG Calibration Evaluation"
echo "Models: $N_MODELS, Total runs: $TOTAL"
echo "Output: ${OUTPUT_BASE}/v50_final_generalization/"
if [ -n "$ONLY_FILTER" ]; then
    echo "Filter: --only $ONLY_FILTER"
fi
echo "========================================"

count=0
for ((i=0; i<N_MODELS; i++)); do
    if [ -n "$ONLY_FILTER" ] && [[ ! "${MODEL_LABELS[$i]}" == *"$ONLY_FILTER"* ]]; then
        continue
    fi
    for ckpt_str in "${CKPT_TYPES[@]}"; do
        IFS='|' read -r ckpt_label ckpt_file <<< "$ckpt_str"
        if ! should_run_ckpt "${CKPT_FILTER[$i]}" "$ckpt_label"; then
            continue
        fi
        for scenario_str in "${TEST_SCENARIOS[@]}"; do
            IFS='|' read -r scenario_name inject_rpy <<< "$scenario_str"
            count=$((count + 1))
            echo ""
            echo "===== [$count/$TOTAL] ====="
            run_single "${MODEL_LABELS[$i]}" "${MODEL_DIRS[$i]}" \
                "$ckpt_label" "$ckpt_file" "$scenario_name" "$inject_rpy"
        done
    done
done

echo ""
echo "========================================"
echo "All evaluations complete."
echo "Results: ${OUTPUT_BASE}/v50_final_generalization/"
echo "========================================"
