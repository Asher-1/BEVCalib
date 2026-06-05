#!/bin/bash
# V43 + V44 bag calibration 泛化评估批量脚本
# 三种测试场景：零扰动基线、shortcut检测(inject 2,2,2)、中等扰动注入(inject 0.5,0.5,0.5)
#
# 用法:
#   bash run_bag_calibration_v43_44_batch.sh          # 串行
#   bash run_bag_calibration_v43_44_batch.sh parallel  # 并行 (每个模型一个GPU)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

BASE_CONFIG="configs/calibration_v43_44_generalization.yaml"
OUTPUT_BASE="/mnt/drtraining/user/dahailu/data/bevcalib/calibration_outputs"
LOGS_BASE="logs/all_training_data"
PARALLEL_MODE="${1:-serial}"

declare -a MODEL_LABELS=(
    "v43-S1"
    "v43-S2"
    "v43-S3"
    "v43-from-v42S1-S2"
    "v43-from-v42S1-S3"
    "v44-S1"
    "v44-S2"
    "v44-S3"
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
echo "V43+V44 Bag Calibration Generalization"
echo "Models: $N_MODELS, Scenarios: $N_SCENARIOS, Total: $TOTAL"
echo "========================================"

run_single() {
    local label="$1"
    local model_dir="$2"
    local scenario_name="$3"
    local inject_rpy="$4"
    local gpu_id="${5:-0}"

    local ckpt_path="${LOGS_BASE}/${model_dir}/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
    if [ ! -f "$ckpt_path" ]; then
        ckpt_path="${LOGS_BASE}/${model_dir}/all_training_data_scratch/checkpoint/ckpt_best_medw.pth"
    fi
    if [ ! -f "$ckpt_path" ]; then
        echo "[SKIP] ${label}/${scenario_name}: no checkpoint found"
        return 0
    fi

    local output_dir="${OUTPUT_BASE}/v43_44_generalization/${label}_${scenario_name}"
    if [ -f "${output_dir}/calibration_summary.md" ]; then
        echo "[SKIP] ${label}/${scenario_name}: already completed"
        return 0
    fi

    echo "[RUN] ${label}/${scenario_name} (GPU ${gpu_id})"
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

if [ "$PARALLEL_MODE" = "parallel" ]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    GPU_COUNT=${GPU_COUNT:-1}
    echo "Parallel mode: ${GPU_COUNT} GPUs available"

    job_idx=0
    for ((i=0; i<N_MODELS; i++)); do
        for scenario_str in "${TEST_SCENARIOS[@]}"; do
            IFS='|' read -r scenario_name inject_rpy <<< "$scenario_str"
            gpu_id=$((job_idx % GPU_COUNT))
            run_single "${MODEL_LABELS[$i]}" "${MODEL_DIRS[$i]}" "$scenario_name" "$inject_rpy" "$gpu_id" &
            job_idx=$((job_idx + 1))
            if [ $((job_idx % GPU_COUNT)) -eq 0 ]; then
                wait
            fi
        done
    done
    wait
else
    echo "Serial mode"
    count=0
    for ((i=0; i<N_MODELS; i++)); do
        for scenario_str in "${TEST_SCENARIOS[@]}"; do
            IFS='|' read -r scenario_name inject_rpy <<< "$scenario_str"
            count=$((count + 1))
            echo ""
            echo "===== [$count/$TOTAL] ====="
            run_single "${MODEL_LABELS[$i]}" "${MODEL_DIRS[$i]}" "$scenario_name" "$inject_rpy" "0"
        done
    done
fi

echo ""
echo "========================================"
echo "All bag calibration evaluations complete"
echo "Results: ${OUTPUT_BASE}/v43_44_generalization/"
echo "========================================"
