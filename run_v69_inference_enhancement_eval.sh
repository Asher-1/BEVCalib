#!/usr/bin/env bash
set -euo pipefail

# V69: inference-side deployment generalization check.
# Runs P0 sequence-median aggregation under the same test_data_v2 / Seq07-excluded
# protocol used by the recent formal generalization reports.

MODE="${1:-smoke}"      # smoke | full | fixed_smoke | fixed
MODEL="${2:-all}"      # v29 | v60 | v60jac | v70dual | v70jac | v72dual | v72jac | v72bdual | v72bjac | v73dual | v73jac | v73val | v73e6 | v73e8 | v73bdual | v73bjac | v73bval | v73be6 | v73be8 | v73be10 | v74dual | v74jac | v74e1 | v74e2 | v74e4 | v74e6 | v75dual | v75jac | v75e1 | v75e2 | v76dual | v76jac | v76e1 | v76e2 | v76e3 | v76e4 | all
GPU="${GPU:-0}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
ADAPTIVE_ITER2="${ADAPTIVE_ITER2:-0}"
ADAPTIVE_SECOND_PASS_MODE="${ADAPTIVE_SECOND_PASS_MODE:-seq_median_anchor}"
ADAPTIVE_PROBE_FRAMES="${ADAPTIVE_PROBE_FRAMES:-12}"
ADAPTIVE_SEQ_RESIDUAL_DEG="${ADAPTIVE_SEQ_RESIDUAL_DEG:-0.35}"
ADAPTIVE_SEQ_RESIDUAL_LO_DEG="${ADAPTIVE_SEQ_RESIDUAL_LO_DEG:-0.0}"
ADAPTIVE_SEQ_RPY_STD_DEG="${ADAPTIVE_SEQ_RPY_STD_DEG:-0.20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/opt/conda/envs/bevcalib310/bin/python}"
DATA="${DATA:-/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/evaluations/v69_inference_enhancement}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export BEV_ZBOUND_STEP="${BEV_ZBOUND_STEP:-4.0}"
export USE_DRCV_BACKEND="${USE_DRCV_BACKEND:-0}"

case "${MODE}" in
  smoke)
    MAX_FRAMES=4
    MAX_BATCH_ARGS=(--max_batches 4)
    BATCH_SIZE=2
    MODE_SUFFIX="smoke"
    FIXED_ARGS=()
    ;;
  full)
    MAX_FRAMES=400
    MAX_BATCH_ARGS=()
    BATCH_SIZE=8
    MODE_SUFFIX="full"
    FIXED_ARGS=()
    ;;
  fixed_smoke)
    MAX_FRAMES=4
    MAX_BATCH_ARGS=(--max_batches 4)
    BATCH_SIZE=2
    MODE_SUFFIX="fixed2_smoke"
    FIXED_ARGS=(--fixed_inject_rpy "${FIXED_INJECT_RPY:-2,2,2}")
    ;;
  fixed)
    MAX_FRAMES=400
    MAX_BATCH_ARGS=()
    BATCH_SIZE=8
    MODE_SUFFIX="fixed2_full"
    FIXED_ARGS=(--fixed_inject_rpy "${FIXED_INJECT_RPY:-2,2,2}")
    ;;
  *)
    echo "Usage: $0 [smoke|full|fixed_smoke|fixed] [v29|v60|v60jac|v70dual|v70jac|v72dual|v72jac|v72bdual|v72bjac|v73dual|v73jac|v73val|v73e6|v73e8|v73bdual|v73bjac|v73bval|v73be6|v73be8|v73be10|v74dual|v74jac|v74e1|v74e2|v74e4|v74e6|v75dual|v75jac|v75e1|v75e2|v76dual|v76jac|v76e1|v76e2|v76e3|v76e4|all]" >&2
    exit 2
    ;;
esac

if [[ "${NUM_ITERATIONS}" != "1" ]]; then
  MODE_SUFFIX="${MODE_SUFFIX}_iter${NUM_ITERATIONS}"
fi

ADAPTIVE_ARGS=()
if [[ "${ADAPTIVE_ITER2}" != "0" ]]; then
  if [[ "${NUM_ITERATIONS}" == "1" ]]; then
    NUM_ITERATIONS=2
    MODE_SUFFIX="${MODE_SUFFIX}_iter2"
  fi
  MODE_SUFFIX="${MODE_SUFFIX}_adapt"
  ADAPTIVE_ARGS=(
    --adaptive_iter2
    --adaptive_second_pass_mode "${ADAPTIVE_SECOND_PASS_MODE}"
    --adaptive_probe_frames "${ADAPTIVE_PROBE_FRAMES}"
    --adaptive_seq_residual_deg "${ADAPTIVE_SEQ_RESIDUAL_DEG}"
    --adaptive_seq_residual_lo_deg "${ADAPTIVE_SEQ_RESIDUAL_LO_DEG}"
    --adaptive_seq_rpy_std_deg "${ADAPTIVE_SEQ_RPY_STD_DEG}"
  )
fi

run_p0() {
  local label="$1"
  local ckpt="$2"
  local out_dir="${OUT_ROOT}/${label}_${MODE_SUFFIX}_p0_seqmedian"

  echo "[V69] ${label}: mode=${MODE}, gpu=${GPU}, out=${out_dir}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ROOT}/tools/analysis/evaluate_p0_refinement.py" \
    --ckpt_path "${ROOT}/${ckpt}" \
    --dataset_root "${DATA}" \
    --output_dir "${out_dir}" \
    --use_full_dataset \
    --exclude_seqs 07 \
    --eval_max_frames_per_seq "${MAX_FRAMES}" \
    --angle_range_deg 5.0 \
    --trans_range 0.0 \
    --batch_size "${BATCH_SIZE}" \
    --num_iterations "${NUM_ITERATIONS}" \
    --sequence_median \
    "${ADAPTIVE_ARGS[@]}" \
    "${FIXED_ARGS[@]}" \
    "${MAX_BATCH_ARGS[@]}"
}

run_v29() {
  run_p0 \
    "v29_g3_best_val" \
    "logs/all_training_data/v29_quick/model_small_5deg_v29_G3_partial_unfreeze_dinov2_quick/all_training_data_scratch/checkpoint/ckpt_best_val.pth"
}

run_v60() {
  run_p0 \
    "v60_ckpt15" \
    "logs/all_training_data/model_small_5deg_v60_seq_balance_pitch_guard/all_training_data_scratch/checkpoint/ckpt_15.pth"
}

run_v60jac() {
  run_p0 \
    "v60_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v60_seq_balance_pitch_guard/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v70dual() {
  run_p0 \
    "v70_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v70_fixed_recovery_hardseq_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v70jac() {
  run_p0 \
    "v70_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v70_fixed_recovery_hardseq_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v72dual() {
  run_p0 \
    "v72_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v72_adir_adapter_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v72jac() {
  run_p0 \
    "v72_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v72_adir_adapter_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v72bdual() {
  run_p0 \
    "v72b_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v72b_adir_strong_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v72bjac() {
  run_p0 \
    "v72b_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v72b_adir_strong_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v73dual() {
  run_p0 \
    "v73_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v73_selective_head_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v73jac() {
  run_p0 \
    "v73_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v73_selective_head_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v73val() {
  run_p0 \
    "v73_smoke_best_val" \
    "logs/all_training_data/model_small_5deg_v73_selective_head_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_val.pth"
}

run_v73e6() {
  run_p0 \
    "v73_smoke_ckpt6" \
    "logs/all_training_data/model_small_5deg_v73_selective_head_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_6.pth"
}

run_v73e8() {
  run_p0 \
    "v73_smoke_ckpt8" \
    "logs/all_training_data/model_small_5deg_v73_selective_head_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_8.pth"
}

run_v73bdual() {
  run_p0 \
    "v73b_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v73b_selective_head_guard_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v73bjac() {
  run_p0 \
    "v73b_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v73b_selective_head_guard_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v73bval() {
  run_p0 \
    "v73b_smoke_best_val" \
    "logs/all_training_data/model_small_5deg_v73b_selective_head_guard_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_best_val.pth"
}

run_v73be6() {
  run_p0 \
    "v73b_smoke_ckpt6" \
    "logs/all_training_data/model_small_5deg_v73b_selective_head_guard_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_6.pth"
}

run_v73be8() {
  run_p0 \
    "v73b_smoke_ckpt8" \
    "logs/all_training_data/model_small_5deg_v73b_selective_head_guard_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_8.pth"
}

run_v73be10() {
  run_p0 \
    "v73b_smoke_ckpt10" \
    "logs/all_training_data/model_small_5deg_v73b_selective_head_guard_recovery_smoke/all_training_data_scratch/checkpoint/ckpt_10.pth"
}

run_v74dual() {
  run_p0 \
    "v74_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v74_v73e8_hardseq_polish_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v74jac() {
  run_p0 \
    "v74_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v74_v73e8_hardseq_polish_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v74e1() {
  run_p0 \
    "v74_smoke_ckpt1" \
    "logs/all_training_data/model_small_5deg_v74_v73e8_hardseq_polish_smoke/all_training_data_scratch/checkpoint/ckpt_1.pth"
}

run_v74e2() {
  run_p0 \
    "v74_smoke_ckpt2" \
    "logs/all_training_data/model_small_5deg_v74_v73e8_hardseq_polish_smoke/all_training_data_scratch/checkpoint/ckpt_2.pth"
}

run_v74e4() {
  run_p0 \
    "v74_smoke_ckpt4" \
    "logs/all_training_data/model_small_5deg_v74_v73e8_hardseq_polish_smoke/all_training_data_scratch/checkpoint/ckpt_4.pth"
}

run_v74e6() {
  run_p0 \
    "v74_smoke_ckpt6" \
    "logs/all_training_data/model_small_5deg_v74_v73e8_hardseq_polish_smoke/all_training_data_scratch/checkpoint/ckpt_6.pth"
}

run_v75dual() {
  run_p0 \
    "v75_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v75_v74e2_hardseq_micro_polish_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v75jac() {
  run_p0 \
    "v75_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v75_v74e2_hardseq_micro_polish_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v75e1() {
  run_p0 \
    "v75_smoke_ckpt1" \
    "logs/all_training_data/model_small_5deg_v75_v74e2_hardseq_micro_polish_smoke/all_training_data_scratch/checkpoint/ckpt_1.pth"
}

run_v75e2() {
  run_p0 \
    "v75_smoke_ckpt2" \
    "logs/all_training_data/model_small_5deg_v75_v74e2_hardseq_micro_polish_smoke/all_training_data_scratch/checkpoint/ckpt_2.pth"
}

run_v76dual() {
  run_p0 \
    "v76_smoke_best_dual" \
    "logs/all_training_data/model_small_5deg_v76_v75e2_soft_axis_balance_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
}

run_v76jac() {
  run_p0 \
    "v76_smoke_best_jacobian" \
    "logs/all_training_data/model_small_5deg_v76_v75e2_soft_axis_balance_smoke/all_training_data_scratch/checkpoint/ckpt_best_jacobian.pth"
}

run_v76e1() {
  run_p0 \
    "v76_smoke_ckpt1" \
    "logs/all_training_data/model_small_5deg_v76_v75e2_soft_axis_balance_smoke/all_training_data_scratch/checkpoint/ckpt_1.pth"
}

run_v76e2() {
  run_p0 \
    "v76_smoke_ckpt2" \
    "logs/all_training_data/model_small_5deg_v76_v75e2_soft_axis_balance_smoke/all_training_data_scratch/checkpoint/ckpt_2.pth"
}

run_v76e3() {
  run_p0 \
    "v76_smoke_ckpt3" \
    "logs/all_training_data/model_small_5deg_v76_v75e2_soft_axis_balance_smoke/all_training_data_scratch/checkpoint/ckpt_3.pth"
}

run_v76e4() {
  run_p0 \
    "v76_smoke_ckpt4" \
    "logs/all_training_data/model_small_5deg_v76_v75e2_soft_axis_balance_smoke/all_training_data_scratch/checkpoint/ckpt_4.pth"
}

case "${MODEL}" in
  v29) run_v29 ;;
  v60) run_v60 ;;
  v60jac) run_v60jac ;;
  v70dual) run_v70dual ;;
  v70jac) run_v70jac ;;
  v72dual) run_v72dual ;;
  v72jac) run_v72jac ;;
  v72bdual) run_v72bdual ;;
  v72bjac) run_v72bjac ;;
  v73dual) run_v73dual ;;
  v73jac) run_v73jac ;;
  v73val) run_v73val ;;
  v73e6) run_v73e6 ;;
  v73e8) run_v73e8 ;;
  v73bdual) run_v73bdual ;;
  v73bjac) run_v73bjac ;;
  v73bval) run_v73bval ;;
  v73be6) run_v73be6 ;;
  v73be8) run_v73be8 ;;
  v73be10) run_v73be10 ;;
  v74dual) run_v74dual ;;
  v74jac) run_v74jac ;;
  v74e1) run_v74e1 ;;
  v74e2) run_v74e2 ;;
  v74e4) run_v74e4 ;;
  v74e6) run_v74e6 ;;
  v75dual) run_v75dual ;;
  v75jac) run_v75jac ;;
  v75e1) run_v75e1 ;;
  v75e2) run_v75e2 ;;
  v76dual) run_v76dual ;;
  v76jac) run_v76jac ;;
  v76e1) run_v76e1 ;;
  v76e2) run_v76e2 ;;
  v76e3) run_v76e3 ;;
  v76e4) run_v76e4 ;;
  all)
    run_v29
    run_v60
    ;;
  *)
    echo "Usage: $0 [smoke|full|fixed_smoke|fixed] [v29|v60|v60jac|v70dual|v70jac|v72dual|v72jac|v72bdual|v72bjac|v73dual|v73jac|v73val|v73e6|v73e8|v73bdual|v73bjac|v73bval|v73be6|v73be8|v73be10|v74dual|v74jac|v74e1|v74e2|v74e4|v74e6|v75dual|v75jac|v75e1|v75e2|v76dual|v76jac|v76e1|v76e2|v76e3|v76e4|all]" >&2
    exit 2
    ;;
esac
