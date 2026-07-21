#!/bin/bash
# V70.1 补跑 gdiag (400帧/序列, gdiag_only, 多卡并行)
set -euo pipefail

ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
cd "$ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310
export HF_HUB_OFFLINE=1
export USE_DRCV_BACKEND=0

OUT="$ROOT/logs/evaluations/generalization_c1_v70.1"
mkdir -p "$OUT"
LOG="$OUT/nohup_gdiag_parallel.log"

echo "=== V70.1 gdiag parallel retry $(date) ===" | tee "$LOG"

python run_generalization_eval.py \
  --config configs/c1_retrain/eval_generalization_c1_v70.1.yaml \
  --gdiag_only \
  --parallel -1 \
  --generalization_diag \
  --cf_bev_r_iter_steps 3 \
  --eval_max_frames_per_seq 400 \
  2>&1 | tee -a "$LOG"

echo "=== done $(date) ===" | tee -a "$LOG"
