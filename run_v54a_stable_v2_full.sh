#!/bin/bash
# v54a-stable-v2 80ep full 训练
set -euo pipefail
cd "$(dirname "$0")"
LOG_ROOT="logs/v54a_stable_v2"
mkdir -p "$LOG_ROOT"
sed -i 's/\r$//' run_v54a_stable_v2_full.sh batch_train.sh 2>/dev/null || true

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310

echo "[$(date +%F\ %T)] Starting v54a_lsp_stable_v2_full (80ep)"
echo "  Fix: ZD/inj/rig/LSP @ ep15 (0-indexed epoch>=15 → display ep16+)"
echo "  Smoke: 3x15ep all NaN=0 on ep1-14; ep15+ validated in full run"

bash batch_train.sh configs/v54a_lsp_stable_v2_cf_bev_r.yaml --force --skip-pattern 'smoke|ep20' \
  2>&1 | tee "$LOG_ROOT/full_batch.log"

echo "[$(date +%F\ %T)] Full training finished or stopped."
echo "Monitor NaN at ep15+: grep 'NaN GUARD' logs/all_training_data/model_small_5deg_v54a_lsp_stable_v2_full/train.log | wc -l"
