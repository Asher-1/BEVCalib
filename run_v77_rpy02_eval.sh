#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"  # smoke | full | fixed_smoke | fixed | both_full
CKPT="${2:-logs/all_training_data/model_small_5deg_v77_rpy02_tail_guard_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth}"
LABEL="${LABEL:-v77_rpy02_tail_guard}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/opt/conda/envs/bevcalib310/bin/python}"
GPU="${GPU:-0}"
DATA="${DATA:-/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/evaluations/v77_rpy02_tail_guard/${LABEL}}"
RPY_THRESHOLD_DEG="${RPY_THRESHOLD_DEG:-0.2}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export BEV_ZBOUND_STEP="${BEV_ZBOUND_STEP:-4.0}"
export USE_DRCV_BACKEND="${USE_DRCV_BACKEND:-0}"

case "${MODE}" in
  smoke)
    MAX_FRAMES=4
    BATCH_SIZE=2
    EXTRA=(--max_batches 4)
    OUT_SUFFIX=random_smoke
    FIXED=()
    ;;
  full)
    MAX_FRAMES=400
    BATCH_SIZE=8
    EXTRA=()
    OUT_SUFFIX=random_full
    FIXED=()
    ;;
  fixed_smoke)
    MAX_FRAMES=4
    BATCH_SIZE=2
    EXTRA=(--max_batches 4)
    OUT_SUFFIX=fixed222_smoke
    FIXED=(--fixed_inject_rpy "${FIXED_INJECT_RPY:-2,2,2}")
    ;;
  fixed)
    MAX_FRAMES=400
    BATCH_SIZE=8
    EXTRA=()
    OUT_SUFFIX=fixed222_full
    FIXED=(--fixed_inject_rpy "${FIXED_INJECT_RPY:-2,2,2}")
    ;;
  both_full)
    bash "$0" full "$CKPT"
    bash "$0" fixed "$CKPT"
    exit 0
    ;;
  *)
    echo "Usage: $0 [smoke|full|fixed_smoke|fixed|both_full] [ckpt_path]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${ROOT}/${CKPT}" && ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ -f "${ROOT}/${CKPT}" ]]; then
  CKPT_PATH="${ROOT}/${CKPT}"
else
  CKPT_PATH="${CKPT}"
fi

OUT_DIR="${OUT_ROOT}/${OUT_SUFFIX}"
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ROOT}/tools/analysis/evaluate_p0_refinement.py" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_root "${DATA}" \
  --output_dir "${OUT_DIR}" \
  --use_full_dataset \
  --exclude_seqs 07 \
  --eval_max_frames_per_seq "${MAX_FRAMES}" \
  --angle_range_deg 5.0 \
  --trans_range 0.0 \
  --batch_size "${BATCH_SIZE}" \
  --num_iterations 2 \
  --sequence_median \
  --adaptive_iter2 \
  --adaptive_second_pass_mode "${ADAPTIVE_SECOND_PASS_MODE:-seq_median_anchor}" \
  --adaptive_probe_frames "${ADAPTIVE_PROBE_FRAMES:-12}" \
  --adaptive_seq_residual_deg "${ADAPTIVE_SEQ_RESIDUAL_DEG:-0.35}" \
  --adaptive_seq_residual_lo_deg "${ADAPTIVE_SEQ_RESIDUAL_LO_DEG:-0.0}" \
  --adaptive_seq_rpy_std_deg "${ADAPTIVE_SEQ_RPY_STD_DEG:-0.20}" \
  --rpy_threshold_deg "${RPY_THRESHOLD_DEG}" \
  "${FIXED[@]}" \
  "${EXTRA[@]}"

echo "V77 RPY<${RPY_THRESHOLD_DEG} eval output: ${OUT_DIR}"
