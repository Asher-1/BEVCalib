#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/opt/conda/envs/bevcalib310/bin/python}"
GPU="${GPU:-0}"
DATA="${DATA:-/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2}"
CKPT_PATH="${CKPT_PATH:-${ROOT}/logs/all_training_data/model_small_5deg_v74_v73e8_hardseq_polish_smoke/all_training_data_scratch/checkpoint/ckpt_2.pth}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/evaluations/adaptive_default_formal/v74e2_adaptive_default}"
RPY_THRESHOLD_DEG="${RPY_THRESHOLD_DEG:-0.2}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export BEV_ZBOUND_STEP="${BEV_ZBOUND_STEP:-4.0}"
export USE_DRCV_BACKEND="${USE_DRCV_BACKEND:-0}"

run_eval() {
  local out_dir="$1"
  local max_frames="$2"
  local batch_size="$3"
  shift 3
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ROOT}/tools/analysis/evaluate_p0_refinement.py" \
    --ckpt_path "${CKPT_PATH}" \
    --dataset_root "${DATA}" \
    --output_dir "${out_dir}" \
    --use_full_dataset --exclude_seqs 07 \
    --eval_max_frames_per_seq "${max_frames}" \
    --angle_range_deg 5.0 --trans_range 0.0 \
    --rpy_threshold_deg "${RPY_THRESHOLD_DEG}" \
    --batch_size "${batch_size}" \
    --num_iterations 2 --sequence_median --adaptive_iter2 \
    --adaptive_second_pass_mode "${ADAPTIVE_SECOND_PASS_MODE:-seq_median_anchor}" \
    --adaptive_probe_frames "${ADAPTIVE_PROBE_FRAMES:-12}" \
    --adaptive_seq_residual_deg "${ADAPTIVE_SEQ_RESIDUAL_DEG:-0.35}" \
    --adaptive_seq_residual_lo_deg "${ADAPTIVE_SEQ_RESIDUAL_LO_DEG:-0.0}" \
    --adaptive_seq_rpy_std_deg "${ADAPTIVE_SEQ_RPY_STD_DEG:-0.20}" \
    "$@"
}

mkdir -p "${OUT_ROOT}"
case "${MODE}" in
  full)
    run_eval "${OUT_ROOT}/random_full" 400 8
    run_eval "${OUT_ROOT}/fixed222_full" 400 8 --fixed_inject_rpy "${FIXED_INJECT_RPY:-2,2,2}"
    ;;
  smoke)
    run_eval "${OUT_ROOT}/random_smoke" 4 2 --max_batches 4
    run_eval "${OUT_ROOT}/fixed222_smoke" 4 2 --fixed_inject_rpy "${FIXED_INJECT_RPY:-2,2,2}" --max_batches 4
    ;;
  summarize)
    echo "Random report: ${OUT_ROOT}/random_full/p0_refinement_report.txt"
    echo "Fixed report: ${OUT_ROOT}/fixed222_full/p0_refinement_report.txt"
    ;;
  *)
    echo "Usage: $0 [full|smoke|summarize]" >&2
    exit 2
    ;;
esac

echo "Adaptive-default formal eval root: ${OUT_ROOT}"
