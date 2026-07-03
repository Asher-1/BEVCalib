#!/bin/bash
# Continue pipeline: cam0 TLC benchmark -> nojac full 80ep
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' run_tlc_benchmark_cam0_compare.sh run_continue_pipeline.sh 2>/dev/null || true

echo "[$(date +%F\ %T)] Step 1: TLC benchmark cam0 (3 models, GPU 0-2)"
bash run_tlc_benchmark_cam0_compare.sh

echo "[$(date +%F\ %T)] Step 2: nojac full 80ep (8 GPU DDP)"
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310
bash batch_train.sh configs/v54a_lsp_stable_jac_ablation_cf_bev_r.yaml --skip-pattern smoke --force \
  2>&1 | tee logs/batch_v54a_nojac_full.log

echo "[$(date +%F\ %T)] Pipeline complete."
