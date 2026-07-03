#!/bin/bash
# TLC benchmark: v54a_lsp vs v54a_nolsp vs v53c
set -euo pipefail
cd "$(dirname "$0")"
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310

TLC_ROOT="../TLC-Calib/data/TLC-Calib"
OUT_BASE="logs/evaluations/tlc_benchmark_compare"
mkdir -p "$OUT_BASE"

run_one() {
  local name="$1" ckpt="$2" gpu="$3"
  echo "[$(date +%H:%M:%S)] START $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 USE_DRCV_BACKEND=0 \
    python3 tools/tlc_bevcalib_benchmark.py \
      --tlc_root "$TLC_ROOT" \
      --ckpt_path "$ckpt" \
      --output_dir "$OUT_BASE/$name" \
      --work_dir "$OUT_BASE/$name/work" \
      --angle_deg 5 --trans_range 0 --max_frames 200 --batch_size 8 \
      > "$OUT_BASE/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name"
}

CKPT_V54A_LSP="logs/all_training_data/model_small_5deg_v54a_lsp_full/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
CKPT_V54A_NOLSP="logs/all_training_data/model_small_5deg_v54a_nolsp/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"
CKPT_V53C="logs/all_training_data/model_small_5deg_v53c_partial_gin_full/all_training_data_scratch/checkpoint/ckpt_best_dual.pth"

run_one v54a_lsp "$CKPT_V54A_LSP" 0 &
run_one v54a_nolsp "$CKPT_V54A_NOLSP" 1 &
run_one v53c "$CKPT_V53C" 2 &
wait

python3 - <<'PY'
import json
from pathlib import Path
base = Path('logs/evaluations/tlc_benchmark_compare')
models = ['v54a_lsp', 'v54a_nolsp', 'v53c']
rows = []
for m in models:
    p = base / m / 'tlc_benchmark_results.json'
    if not p.exists():
        rows.append((m, 'MISSING', '', '', '', ''))
        continue
    data = json.loads(p.read_text())
    ok = [r for r in data if 'rig_rot_deg' in r]
    if not ok:
        rows.append((m, 'FAIL', '', '', '', ''))
        continue
    mr = sum(r['rig_rot_deg'] for r in ok) / len(ok)
    mt = sum(r['rig_trans_m'] for r in ok) / len(ok)
    mf = sum(r['frame_rot_median_deg'] for r in ok) / len(ok)
    rows.append((m, len(ok), f'{mr:.4f}', f'{mt:.4f}', f'{mf:.4f}', 'OK'))

lines = [
    '# TLC Benchmark 三模型对照',
    '',
    '| Model | Scenes | Mean Rig Rot (°) | Mean Rig Trans (m) | Mean Frame Rot Med (°) | Status |',
    '|-------|--------|------------------|--------------------|------------------------|--------|',
]
for r in rows:
    lines.append(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |')
out = base / 'COMPARE_REPORT.md'
out.write_text('\n'.join(lines) + '\n')
print(out)
PY

echo "All benchmarks done. See $OUT_BASE/COMPARE_REPORT.md"
