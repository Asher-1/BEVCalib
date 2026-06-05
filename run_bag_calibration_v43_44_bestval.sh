#!/bin/bash
# V43 + V44 bag calibration: best_val checkpoint 补充评估
# 修复：不限制 CUDA_VISIBLE_DEVICES，让 parallel:-1 使用全部 GPU
#
# 用法:
#   bash run_bag_calibration_v43_44_bestval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

BASE_CONFIG="configs/calibration_v43_44_generalization.yaml"
OUTPUT_BASE="/mnt/drtraining/user/dahailu/data/bevcalib/calibration_outputs"
LOGS_BASE="logs/all_training_data"

declare -a MODEL_LABELS=(
    "v43-S1-bestval"
    "v43-S2-bestval"
    "v43-S3-bestval"
    "v43-from-v42S1-S2-bestval"
    "v43-from-v42S1-S3-bestval"
    "v44-S1-bestval"
    "v44-S2-bestval"
    "v44-S3-bestval"
)

declare -a MODEL_DIRS=(
    "model_small_5deg_v43_cf_bev_r_S1_quick"
    "model_small_3deg_v43_cf_bev_r_S2_quick"
    "model_small_1deg_v43_cf_bev_r_S3_quick"
    "model_small_3deg_v43_from_v42S1_S2_quick"
    "model_small_1deg_v43_from_v42S1_S3_quick"
    "model_small_5deg_v44_cf_bev_r_S1_quick"
    "model_small_3deg_v44_cf_bev_r_S2_quick"
    "model_small_1deg_v44_cf_bev_r_S3_quick"
)

declare -a TEST_SCENARIOS=(
    "baseline|"
    "shortcut|2,2,2"
    "inject_small|0.5,0.5,0.5"
)

N_MODELS=${#MODEL_LABELS[@]}
N_SCENARIOS=${#TEST_SCENARIOS[@]}
TOTAL=$((N_MODELS * N_SCENARIOS))

echo "========================================"
echo "V43+V44 Best-Val Bag Calibration"
echo "Models: $N_MODELS, Scenarios: $N_SCENARIOS, Total: $TOTAL"
echo "Config parallel:-1 will use ALL available GPUs"
echo "========================================"

run_single() {
    local label="$1"
    local model_dir="$2"
    local scenario_name="$3"
    local inject_rpy="$4"

    local ckpt_path="${LOGS_BASE}/${model_dir}/all_training_data_scratch/checkpoint/ckpt_best_val.pth"
    if [ ! -f "$ckpt_path" ]; then
        echo "[SKIP] ${label}/${scenario_name}: no best_val checkpoint"
        return 0
    fi

    local output_dir="${OUTPUT_BASE}/v43_44_generalization/${label}_${scenario_name}"
    if [ -f "${output_dir}/SUMMARY_REPORT.md" ]; then
        echo "[SKIP] ${label}/${scenario_name}: already completed"
        return 0
    fi

    echo ""
    echo "[RUN] ${label}/${scenario_name}"
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
        echo "[FAIL] ${label}/${scenario_name} exited with $?"
        return 0
    }
    echo "[DONE] ${label}/${scenario_name}"
}

count=0
for ((i=0; i<N_MODELS; i++)); do
    for scenario_str in "${TEST_SCENARIOS[@]}"; do
        IFS='|' read -r scenario_name inject_rpy <<< "$scenario_str"
        count=$((count + 1))
        echo ""
        echo "===== [$count/$TOTAL] ====="
        run_single "${MODEL_LABELS[$i]}" "${MODEL_DIRS[$i]}" "$scenario_name" "$inject_rpy"
    done
done

echo ""
echo "========================================"
echo "All best_val bag calibration evaluations complete"
echo "Results: ${OUTPUT_BASE}/v43_44_generalization/"
echo "========================================"
