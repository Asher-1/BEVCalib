#!/bin/bash
# V68 gdiag 补跑：跳过主评估（max_batches=1），全量 test_data gdiag
set -euo pipefail

ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
TEST_DATA="/mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1"
OUT_BASE="$ROOT/logs/evaluations/generalization_c1_v68"
CKPT_DIR="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v68_ultimate_S1/all_training_data_c1_scratch/checkpoint"
V62_CKPT="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v62_pure_recovery_S1/all_training_data_c1_scratch/checkpoint/ckpt_best_dual.pth"

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310
export HF_HUB_OFFLINE=1
export USE_DRCV_BACKEND=0
cd "$ROOT"

run_gdiag() {
    local gpu=$1 label=$2 ckpt=$3 iter_steps=${4:-0}
    local out_dir="$OUT_BASE/$label"
    mkdir -p "$out_dir"
    echo "[GPU $gpu] gdiag iter=$iter_steps -> $label"
    CUDA_VISIBLE_DEVICES=$gpu python evaluate_checkpoint.py \
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
        2>&1 | tee "$out_dir/gdiag_run_iter${iter_steps}.log"
}

run_gdiag 0 "c1-v68-ultimate-S1-best-medw" "$CKPT_DIR/ckpt_best_medw.pth" 0 &
run_gdiag 1 "c1-v68-ultimate-S1-best-val" "$CKPT_DIR/ckpt_best_val.pth" 0 &
run_gdiag 2 "c1-v68-ultimate-S1-last" "$CKPT_DIR/ckpt_290.pth" 0 &
run_gdiag 3 "c1-v62-baseline" "$V62_CKPT" 0 &
wait
echo "=== Single-step gdiag done ==="

run_gdiag 0 "c1-v68-ultimate-S1-best-medw-iter2" "$CKPT_DIR/ckpt_best_medw.pth" 2 &
run_gdiag 1 "c1-v68-ultimate-S1-best-val-iter2" "$CKPT_DIR/ckpt_best_val.pth" 2 &
run_gdiag 2 "c1-v68-ultimate-S1-last-iter2" "$CKPT_DIR/ckpt_290.pth" 2 &
run_gdiag 3 "c1-v62-baseline-iter2" "$V62_CKPT" 2 &
wait
echo "=== Iter2 gdiag done ==="

run_gdiag 0 "c1-v68-ultimate-S1-best-medw-iter3" "$CKPT_DIR/ckpt_best_medw.pth" 3
echo "=== All V68 gdiag complete ==="
