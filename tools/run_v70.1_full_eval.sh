#!/bin/bash
# V70.1 全量主评估 (12序列, ~24k帧)
# 泛化诊断请单独运行: bash tools/run_v70.1_gdiag_full.sh
set -euo pipefail

ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
TEST_DATA="/mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1"
OUT_BASE="$ROOT/logs/evaluations/generalization_c1_v70.1_full"

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310
export HF_HUB_OFFLINE=1
export USE_DRCV_BACKEND=0
cd "$ROOT"

run_full() {
    local gpu=$1 label=$2 ckpt=$3 iter_steps=${4:-3}
    local out_dir="$OUT_BASE/$label"
    mkdir -p "$out_dir"
    echo "[GPU $gpu] full eval iter=$iter_steps -> $label"
    CUDA_VISIBLE_DEVICES=$gpu python evaluate_checkpoint.py \
        --ckpt_path "$ckpt" \
        --dataset_root "$TEST_DATA" \
        --use_full_dataset \
        --max_batches 0 \
        --vis_interval 99999 \
        --output_dir "$out_dir" \
        --rotation_only 1 \
        --target_width 960 --target_height 540 \
        --batch_size 8 \
        --angle_range_deg 5.0 \
        --trans_range 0.0 \
        --pitch_vertical_bands 3 \
        2>&1 | tee "$out_dir/full_eval.log"
}

S4Z="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S4Z/all_training_data_c1_scratch/checkpoint"
S4R="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S4R/all_training_data_c1_scratch/checkpoint"
S3="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S3/all_training_data_c1_scratch/checkpoint"
V68="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v68_ultimate_S1/all_training_data_c1_scratch/checkpoint/ckpt_best_dual.pth"

run_full 0 "c1-v70.1-S4Z-best-dual-iter3" "$S4Z/ckpt_best_dual.pth" 3 &
run_full 1 "c1-v70.1-S4Z-best-recovery-iter3" "$S4Z/ckpt_best_recovery.pth" 3 &
run_full 2 "c1-v70.1-S4Z-best-zd-iter0" "$S4Z/ckpt_best_zd.pth" 0 &
run_full 3 "c1-v70.1-S4R-best-dual-iter3" "$S4R/ckpt_best_dual.pth" 3 &
run_full 4 "c1-v70.1-S3-best-dual-iter3" "$S3/ckpt_best_dual.pth" 3 &
wait
echo "=== V70.1 full main eval complete ==="
echo "下一步: bash tools/run_v70.1_gdiag_full.sh"
