#!/bin/bash
# V70.1 全量泛化诊断 (gdiag_only, 串行执行避免 OOM)
# 前提: generalization_c1_v70.1 主评估已完成 (extrinsics_and_errors.txt)
set -euo pipefail

ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
TEST_DATA="/mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1"
OUT_BASE="$ROOT/logs/evaluations/generalization_c1_v70.1"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MIN_SEQS=10

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310
export HF_HUB_OFFLINE=1
export USE_DRCV_BACKEND=0
cd "$ROOT"

is_gdiag_complete() {
    local json="$1"
    [[ -f "$json" ]] || return 1
    python3 - "$json" "$MIN_SEQS" <<'PY'
import json, sys
path, min_seqs = sys.argv[1], int(sys.argv[2])
try:
    n = int(json.load(open(path)).get("composite", {}).get("raw", {}).get("n_seqs_evaluated", 0))
    sys.exit(0 if n >= min_seqs else 1)
except Exception:
    sys.exit(1)
PY
}

run_gdiag_only() {
    local label=$1 ckpt=$2 iter_steps=${3:-3}
    local out_dir="$OUT_BASE/$label"
    local gdiag_json="$out_dir/generalization_diagnostics.json"
    if [[ ! -f "$out_dir/extrinsics_and_errors.txt" ]]; then
        echo "[SKIP] $label: 缺少主评估结果"
        return 0
    fi
    if is_gdiag_complete "$gdiag_json"; then
        echo "[SKIP] $label: gdiag 已完成 (n_seqs>=$MIN_SEQS)"
        return 0
    fi
    echo "[GPU $GPU] gdiag_only iter=$iter_steps -> $label"
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_checkpoint.py \
        --ckpt_path "$ckpt" \
        --dataset_root "$TEST_DATA" \
        --use_full_dataset \
        --eval_max_frames_per_seq 400 \
        --gdiag_only \
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
        2>&1 | tee -a "$out_dir/gdiag_run.log"
}

S4Z="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S4Z/all_training_data_c1_scratch/checkpoint"
S4R="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S4R/all_training_data_c1_scratch/checkpoint"
S3="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v70.1_corr_domain_S3/all_training_data_c1_scratch/checkpoint"
V62="$ROOT/logs/all_training_data_c1/model_small_5deg_c1_v62_pure_recovery_S1/all_training_data_c1_scratch/checkpoint/ckpt_best_dual.pth"

run_gdiag_only "c1-v70.1-S4Z-best-dual-iter3" "$S4Z/ckpt_best_dual.pth" 3
run_gdiag_only "c1-v70.1-S4Z-best-recovery-iter3" "$S4Z/ckpt_best_recovery.pth" 3
run_gdiag_only "c1-v70.1-S4Z-best-zd-iter0" "$S4Z/ckpt_best_zd.pth" 0
run_gdiag_only "c1-v70.1-S4R-best-dual-iter3" "$S4R/ckpt_best_dual.pth" 3
run_gdiag_only "c1-v70.1-S4R-best-recovery-iter3" "$S4R/ckpt_best_recovery.pth" 3
run_gdiag_only "c1-v70.1-S3-best-dual-iter3" "$S3/ckpt_best_dual.pth" 3
run_gdiag_only "c1-v62-baseline-iter3" "$V62" 3

echo "=== V70.1 gdiag complete ==="
