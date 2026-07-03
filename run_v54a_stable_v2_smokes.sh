#!/bin/bash
# v54a-stable-v2 smoke batch (3x15ep) + NaN 对比表
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' run_v54a_stable_v2_smokes.sh batch_train.sh 2>/dev/null || true

LOG_ROOT="logs/v54a_stable_v2"
mkdir -p "$LOG_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310

echo "[$(date +%F\ %T)] Starting v54a stable v2 smokes (skip full)"
bash batch_train.sh configs/v54a_lsp_stable_v2_cf_bev_r.yaml --force --skip-pattern full \
  2>&1 | tee "$LOG_ROOT/smoke_batch.log"

echo "[$(date +%F\ %T)] NaN summary:"
python3 - <<'PY'
from pathlib import Path
BASE = Path("/mnt/drtraining/user/dahailu/code/BEVCalib/logs/all_training_data")
runs = [
    ("stable_smoke (baseline)", "model_small_5deg_v54a_lsp_stable_smoke"),
    ("v2_smoke", "model_small_5deg_v54a_lsp_stable_v2_smoke"),
    ("v2_nophoto", "model_small_5deg_v54a_lsp_stable_v2_nophoto_smoke"),
    ("v2_lowlsp", "model_small_5deg_v54a_lsp_stable_v2_lowlsp_smoke"),
    ("delay_zd_inj", "model_small_5deg_v54a_stable_delay_zd_inj_smoke"),
]
lines = ["# v54a-stable-v2 Smoke NaN 对比\n", "| run | NaN GUARD | ep15 Rot | best dual ep |", "|-----|-----------|----------|--------------|"]
for name, d in runs:
    p = BASE / d / "train.log"
    if not p.exists():
        lines.append(f"| {name} | — | — | — |")
        continue
    text = p.read_text(errors="ignore")
    nan = text.count("NaN GUARD")
    rot = ep = "?"
    for line in reversed(text.splitlines()):
        if "Epoch [15/15]" in line and "Train Pose Error - Rot:" in line:
            rot = line.split("Rot:")[1].split("°")[0].strip() + "°"
            break
    for line in reversed(text.splitlines()):
        if "Saved best dual gate ckpt" in line or "ckpt_best_dual" in line:
            import re
            m = re.search(r"ep(\d+)", line)
            if m:
                ep = f"ep{m.group(1)}"
                break
    lines.append(f"| {name} | {nan} | {rot} | {ep} |")
report = Path("/mnt/drtraining/user/dahailu/code/BEVCalib/logs/v54a_stable_v2/SMOKE_NAN_REPORT.md")
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nWrote {report}")
PY

echo "Done. Report: logs/v54a_stable_v2/SMOKE_NAN_REPORT.md"
