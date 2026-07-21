#!/bin/bash
# V70.1 快速泛化诊断 (1 batch smoke + 全量 gdiag 串行)
set -euo pipefail

ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
TEST_DATA="/mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1"
OUT_BASE="$ROOT/logs/evaluations/generalization_c1_v70.1"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310
export HF_HUB_OFFLINE=1
export USE_DRCV_BACKEND=0
cd "$ROOT"

run_gdiag() {
    local label=$1 ckpt=$2 iter_steps=${3:-3}
    local out_dir="$OUT_BASE/$label"
    mkdir -p "$out_dir"
    echo "[GPU $GPU] gdiag iter=$iter_steps -> $label"
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_checkpoint.py \
        --ckpt_path "$ckpt" \
        --dataset_root "$TEST_DATA" \
        --use_full_dataset \
        --max_batches 1 \
        --vis_interval 99999 \
        --generalization_diag \
        --gdiag_max_batches 0 \
        --gdiag_inject_deg 2.0 \
        --cf_bev_r_iter_steps "$iter_steps" \
        --output_dir "$out_dir" \
        --rotation_only 1 \
        --target_width 960 --target_height 540 \
        --batch_size 8 \
        --angle_range_deg 5.0 \
        --trans_range 0.0 \
        --pitch_vertical_bands 3 \
        2>&1 | tee "$out_dir/gdiag_run.log"
}

S4Z="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S4Z/all_training_data_c1_scratch/checkpoint"
S4R="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S4R/all_training_data_c1_scratch/checkpoint"
S3="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S3/all_training_data_c1_scratch/checkpoint"
V62="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v62_pure_recovery_S1/all_training_data_c1_scratch/checkpoint/ckpt_best_dual.pth"

run_gdiag "c1-v70.1-S4Z-best-dual-iter3" "$S4Z/ckpt_best_dual.pth" 3
run_gdiag "c1-v70.1-S4Z-best-recovery-iter3" "$S4Z/ckpt_best_recovery.pth" 3
run_gdiag "c1-v70.1-S4Z-best-zd-iter0" "$S4Z/ckpt_best_zd.pth" 0
run_gdiag "c1-v70.1-S4R-best-dual-iter3" "$S4R/ckpt_best_dual.pth" 3
run_gdiag "c1-v70.1-S4R-best-recovery-iter3" "$S4R/ckpt_best_recovery.pth" 3
run_gdiag "c1-v70.1-S3-best-dual-iter3" "$S3/ckpt_best_dual.pth" 3
run_gdiag "c1-v62-baseline-iter3" "$V62" 3
run_gdiag "c1-v70.1-S4Z-best-zd-iter3" "$S4Z/ckpt_best_zd.pth" 3

echo "=== V70.1 gdiag complete ==="
