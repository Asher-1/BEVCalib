#!/bin/bash
# Phase3: ep20 stack smoke (3x) → 自动选胜者 → 80ep full
set -euo pipefail
cd "$(dirname "$0")"
LOG_ROOT="logs/v54a_phase3"
mkdir -p "$LOG_ROOT"
sed -i 's/\r$//' run_v54a_phase3_pipeline.sh batch_train.sh 2>/dev/null || true

source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310

if pgrep -f 'train_kitti.*v2_full' >/dev/null 2>&1; then
  echo "[$(date +%F\ %T)] 停止 v2_full (仅 ep1)，改跑 ep20 stack smoke"
  pkill -f 'train_kitti.*v2_full' || true
  pkill -f 'torchrun.*v2_full' || true
  sleep 8
fi

echo "[$(date +%F\ %T)] Step 1: ep20 stack smokes (3x20ep)"
bash batch_train.sh configs/v54a_lsp_stable_v3_ep20_cf_bev_r.yaml --force \
  2>&1 | tee "$LOG_ROOT/ep20_smoke_batch.log"

echo "[$(date +%F\ %T)] Step 2: 选胜者"
python3 - <<'PY'
from pathlib import Path

BASE = Path("/mnt/drtraining/user/dahailu/code/BEVCalib/logs/all_training_data")
candidates = [
    ("stack", "model_small_5deg_v54a_v3_ep20_stack_smoke", "v54a_v3_stack_full"),
    ("nophoto", "model_small_5deg_v54a_v3_ep20_nophoto_smoke", "v54a_v3_nophoto_full"),
    ("safe", "model_small_5deg_v54a_v3_ep20_safe_smoke", "v54a_v3_safe_full"),
]

def stats(logpath):
    text = logpath.read_text(errors="ignore")
    nan_total = text.count("NaN GUARD")
    nan_post16 = 0
    seen16 = False
    rot = "?"
    mgda16 = "?"
    for line in text.splitlines():
        if "Epoch [16/" in line:
            seen16 = True
        if seen16 and "NaN GUARD" in line:
            nan_post16 += 1
        if "Epoch [16/" in line and "mgda_n_tasks:" in line:
            mgda16 = line.split("mgda_n_tasks:")[1].strip().split()[0]
        if "Epoch [20/20]" in line and "Train Pose Error - Rot:" in line:
            rot = line.split("Rot:")[1].split("°")[0].strip() + "°"
    return nan_total, nan_post16, rot, mgda16

rows = []
for tag, logdir, full_name in candidates:
    p = BASE / logdir / "train.log"
    if not p.exists():
        rows.append({"tag": tag, "full": full_name, "ok": False})
        continue
    nan_total, nan_post16, rot, mgda16 = stats(p)
    rows.append({
        "tag": tag, "full": full_name, "ok": True,
        "nan_total": nan_total, "nan_post16": nan_post16,
        "rot": rot, "mgda16": mgda16,
    })

lines = [
    "# Phase3 ep20 Stack Smoke\n",
    "| variant | total NaN | NaN@ep16+ | ep20 Rot | mgda@ep16 | full run |",
    "|---------|-----------|-----------|----------|-----------|----------|",
]
for r in rows:
    if not r["ok"]:
        lines.append(f"| {r['tag']} | — | — | — | — | {r['full']} |")
    else:
        lines.append(
            f"| {r['tag']} | {r['nan_total']} | {r['nan_post16']} | {r['rot']} | {r['mgda16']} | {r['full']} |"
        )

out = Path("/mnt/drtraining/user/dahailu/code/BEVCalib/logs/v54a_phase3/EP20_STACK_REPORT.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

ok_rows = [r for r in rows if r.get("ok")]
zero_post16 = [r for r in ok_rows if r["nan_post16"] == 0]
if zero_post16:
    winner = sorted(zero_post16, key=lambda r: (r["nan_total"], r["tag"]))[0]
else:
    winner = sorted(ok_rows, key=lambda r: (r["nan_post16"], r["nan_total"], r["tag"]))[0]
    print(f"\nWARN: ep16+ NaN present, fallback to lowest: {winner['tag']}")

print(f"\nWinner: {winner['tag']} -> {winner['full']} (NaN@ep16+={winner['nan_post16']}, ep20 Rot={winner['rot']})")
Path("/mnt/drtraining/user/dahailu/code/BEVCalib/logs/v54a_phase3/WINNER.txt").write_text(winner["full"] + "\n", encoding="utf-8")
PY

WINNER=$(head -1 "$LOG_ROOT/WINNER.txt")
echo "[$(date +%F\ %T)] Step 3: launch 80ep $WINNER"

case "$WINNER" in
  v54a_v3_nophoto_full) SKIP='stack|safe' ;;
  v54a_v3_safe_full)    SKIP='stack|nophoto' ;;
  *)                    SKIP='nophoto|safe' ;;
esac

bash batch_train.sh configs/v54a_lsp_stable_v3_fallback_cf_bev_r.yaml --force --skip-pattern "$SKIP" \
  2>&1 | tee "$LOG_ROOT/full_batch.log"

echo "[$(date +%F\ %T)] Phase3 pipeline done. Winner=$WINNER"
