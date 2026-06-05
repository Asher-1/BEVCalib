#!/bin/bash
# V47 全变种 bag calibration 泛化评估批量脚本
# 含 V45-GIN Full 基线 + V47/V47-512g S1/S2 (best_dual + best_val)
# 三种测试场景：零扰动基线、shortcut检测(inject 2,2,2)、中等扰动注入(inject 0.5,0.5,0.5)
#
# 用法:
#   bash run_bag_calibration_v47_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE_CONFIG="configs/calibration_v47_generalization.yaml"
OUTPUT_BASE="/mnt/drtraining/user/dahailu/data/bevcalib/calibration_outputs"
LOGS_BASE="logs/all_training_data"

declare -a MODEL_LABELS=(
    "v45-gin-S1-full"
    "v45-gin-S2-full"
    "v45-gin-S3-full"
    "v47-S1-quick"
    "v47-S2-quick"
    "v47-512g-S1-quick"
    "v47-512g-S2-quick"
    "v47-512g-S1-full"
)

declare -a MODEL_DIRS=(
    "model_small_5deg_v45_gin_cf_bev_r_S1_full"
    "model_small_3deg_v45_gin_cf_bev_r_S2_full"
    "model_small_1deg_v45_gin_cf_bev_r_S3_full"
    "model_small_5deg_v47_cf_bev_r_S1_quick"
    "model_small_3deg_v47_cf_bev_r_S2_quick"
    "model_small_5deg_v47_512g_cf_bev_r_S1_quick"
    "model_small_3deg_v47_512g_cf_bev_r_S2_quick"
    "model_small_5deg_v47_512g_cf_bev_r_S1_full"
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
echo "V47 Bag Calibration Evaluation"
echo "Models: $N_MODELS, Ckpt types: $N_CKPTS, Scenarios: $N_SCENARIOS"
echo "Total evaluations: $TOTAL"
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

    local output_dir="${OUTPUT_BASE}/v47_generalization/${label}_${ckpt_label}_${scenario_name}"
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
            run_single "${MODEL_LABELS[$i]}" "${MODEL_DIRS[$i]}" "$ckpt_label" "$ckpt_file" "$scenario_name" "$inject_rpy"
        done
    done
done

echo ""
echo "========================================"
echo "All V47 bag calibration evaluations complete"
echo "Results: ${OUTPUT_BASE}/v47_generalization/"
echo "========================================"
