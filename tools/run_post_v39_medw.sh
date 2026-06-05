#!/usr/bin/env bash
# 训练完成后：对 checkpoint 目录跑 MEDW200 eval，选出 ckpt_best_medw.pth
#
# Usage:
#   bash tools/run_post_v39_medw.sh <checkpoint_dir>
#
# Example:
#   bash tools/run_post_v39_medw.sh \
#     logs/all_training_data/v39_M1_htcn_main/all_training_data_scratch/checkpoint

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/conda/envs/bevcalib310/bin/python}"
TEST_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
EVAL_FLAGS="--use_full_dataset --eval_max_frames_per_seq 200 --angle_range_deg 5.0 --vis_interval 0 --batch_size 8"

CKPT_DIR="${1:-}"
if [ -z "$CKPT_DIR" ] || [ ! -d "$CKPT_DIR" ]; then
  echo "Usage: bash tools/run_post_v39_medw.sh <checkpoint_dir>"
  exit 1
fi

cd "$ROOT"
MEDW_LOG="$CKPT_DIR/medw_eval_summary.json"
BEST_ROT="999"
BEST_CKPT=""
BEST_OUT=""

mapfile -t CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -name 'ckpt_*.pth' | sort -V)
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "No ckpt_*.pth found in $CKPT_DIR"
  exit 1
fi

echo "=== MEDW eval on ${#CKPTS[@]} checkpoints ==="
for ckpt in "${CKPTS[@]}"; do
  base="$(basename "$ckpt" .pth)"
  out="$CKPT_DIR/medw_${base}"
  echo "--- $base ---"
  "$PYTHON" evaluate_checkpoint.py --mode eval \
    --ckpt_path "$ckpt" \
    --dataset_root "$TEST_ROOT" \
    --output_dir "$out" \
    $EVAL_FLAGS 2>&1 | tail -5

  rot="$("$PYTHON" - <<PY
import json, os
out = "$out"
deploy = os.path.join(out, "deploy_simulation.json")
if os.path.isfile(deploy):
    with open(deploy) as f:
        d = json.load(f)
    entry = d.get("200", {})
    print(entry.get("mean_rot", entry.get("rot", 999)))
else:
    files = glob.glob(os.path.join(out, "**", "eval_summary*.json"), recursive=True)
    if not files:
        print("999")
    else:
        with open(files[0]) as f:
            d = json.load(f)
        medw = d.get("medw200", d.get("overall", {}))
        if isinstance(medw, dict):
            print(medw.get("rot_error", medw.get("rot", 999)))
        else:
            print(999)
PY
)"
  echo "  MEDW rot: $rot"
  if awk "BEGIN {exit !($rot < $BEST_ROT)}"; then
    BEST_ROT="$rot"
    BEST_CKPT="$ckpt"
    BEST_OUT="$out"
  fi
done

if [ -z "$BEST_CKPT" ]; then
  echo "Failed to determine best MEDW checkpoint"
  exit 1
fi

BEST_LINK="$CKPT_DIR/ckpt_best_medw.pth"
cp -f "$BEST_CKPT" "$BEST_LINK"
cat > "$MEDW_LOG" <<EOF
{
  "best_medw_rot": $BEST_ROT,
  "best_ckpt": "$(basename "$BEST_CKPT")",
  "best_ckpt_path": "$BEST_CKPT",
  "best_eval_dir": "$BEST_OUT",
  "symlink": "$BEST_LINK"
}
EOF

echo ""
echo "=== Best MEDW: ${BEST_ROT}° ==="
echo "  ckpt: $BEST_CKPT"
echo "  linked: $BEST_LINK"
echo "  summary: $MEDW_LOG"
