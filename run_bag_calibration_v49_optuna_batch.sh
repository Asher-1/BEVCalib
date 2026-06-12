#!/bin/bash
# V49 Optuna T32 bag calibration 泛化评估
# 评估 v49-optuna-t32-full best_dual + best_val, 三种测试场景
#
# 用法:
#   bash run_bag_calibration_v49_optuna_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE_CONFIG="configs/calibration_v49_optuna_generalization.yaml"
OUTPUT_BASE="/mnt/drtraining/user/dahailu/data/bevcalib/calibration_outputs"
LOGS_BASE="logs/all_training_data"

declare -a MODEL_LABELS=(
    "v49-optuna-t32-full"
    "v45c-S1-quick"
    "v49b-no-cons"
)

declare -a MODEL_DIRS=(
    "model_small_5deg_v49_optuna_best_t32_S1_full"
    "model_small_5deg_v45c_cf_bev_r_S1_quick"
    "model_small_5deg_v49b_no_cons_S1_quick"
)

declare -a CKPT_TYPES=(
    "best_dual|ckpt_best_dual.pth"
    "best_val|ckpt_best_val.pth"
)

declare -a TEST_SCENARIOS=(
    "baseline|"
    "shortcut|2,2,2"
    "inject_small|0.5,0.5,0.5"
)

N_MODELS=${#MODEL_LABELS[@]}
N_CKPTS=${#CKPT_TYPES[@]}
N_SCENARIOS=${#TEST_SCENARIOS[@]}
TOTAL=$((N_MODELS * N_CKPTS * N_SCENARIOS))

echo "========================================"
echo "V49 Optuna BAG Calibration Evaluation"
echo "Models: $N_MODELS, Ckpt types: $N_CKPTS, Scenarios: $N_SCENARIOS"
echo "Total evaluations: $TOTAL"
echo "Output: ${OUTPUT_BASE}/v49_optuna_generalization/"
echo "========================================"

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

    local output_dir="${OUTPUT_BASE}/v49_optuna_generalization/${label}_${ckpt_label}_${scenario_name}"
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

echo "Serial mode (run_bag_calibration.py uses internal parallelism)"
count=0
for ((i=0; i<N_MODELS; i++)); do
    for ckpt_str in "${CKPT_TYPES[@]}"; do
        IFS='|' read -r ckpt_label ckpt_file <<< "$ckpt_str"
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
echo "Results: ${OUTPUT_BASE}/v49_optuna_generalization/"
echo "========================================"
