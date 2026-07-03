#!/bin/bash
# v54a_lsp_full deploy gate eval + ep6 NaN ablation smokes
set -euo pipefail
cd "$(dirname "$0")"
LOG_ROOT="logs/deploy_and_ep6_ablation"
mkdir -p "$LOG_ROOT"
sed -i 's/\r$//' run_v54_eval.sh run_deploy_and_ep6_ablation.sh 2>/dev/null || true

echo "[$(date +%F\ %T)] Step 1: v54a_lsp_full gate eval (test_data_v2)"
bash run_v54_eval.sh --gate-eval-only 2>&1 | tee "$LOG_ROOT/gate_eval.log"

echo "[$(date +%F\ %T)] Step 2: ep6 NaN ablation smokes (3x15ep)"
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310
bash batch_train.sh configs/v54a_lsp_stable_ep6_nan_ablation_cf_bev_r.yaml --force \
  2>&1 | tee "$LOG_ROOT/ep6_ablation_batch.log"

echo "[$(date +%F\ %T)] NaN summary:"
BASE="/mnt/drtraining/user/dahailu/code/BEVCalib/logs/all_training_data"
python3 - <<PY
from pathlib import Path
runs = [
    ("baseline_stable", Path("$BASE/model_small_5deg_v54a_lsp_stable_smoke/train.log")),
    ("delay_zd_inj", Path("$BASE/model_small_5deg_v54a_stable_delay_zd_inj_smoke/train.log")),
    ("delay_mgda", Path("$BASE/model_small_5deg_v54a_stable_delay_mgda_smoke/train.log")),
    ("delay_all_aux", Path("$BASE/model_small_5deg_v54a_stable_delay_all_aux_smoke/train.log")),
]
print("| run | NaN GUARD | ep15 rot |")
print("|-----|-----------|----------|")
for name, p in runs:
    if not p.exists():
        print(f"| {name} | — | — |")
        continue
    text = p.read_text(errors='ignore')
    nan = text.count('NaN GUARD')
    rot = "?"
    for line in reversed(text.splitlines()):
        if 'Epoch [15/15]' in line and 'Train Pose Error - Rot:' in line:
            rot = line.split('Rot:')[1].split('°')[0].strip() + '°'
            break
    print(f"| {name} | {nan} | {rot} |")
out = Path("$LOG_ROOT/EP6_NAN_ABLATION.md")
out.write_text("See console table\\n", encoding='utf-8')
PY

echo "Done. Gate: logs/evaluations/generalization_v54/DEPLOY_GATE_ACCEPTANCE.md"
echo "Ablation: $LOG_ROOT/EP6_NAN_ABLATION.md"
