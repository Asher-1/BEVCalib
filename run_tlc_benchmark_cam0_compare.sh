#!/bin/bash
# TLC benchmark (cam0 default): v54a_lsp vs v54a_nolsp vs v53c
set -euo pipefail
cd "$(dirname "$0")"
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310

TLC_ROOT="../TLC-Calib/data/TLC-Calib"
OUT_BASE="logs/evaluations/tlc_benchmark_cam0"
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
from datetime import datetime

base = Path('logs/evaluations/tlc_benchmark_cam0')
models = ['v54a_lsp', 'v54a_nolsp', 'v53c']
rows = []
subset_rows = []
for m in models:
    p = base / m / 'tlc_benchmark_results.json'
    if not p.exists():
        rows.append((m, 'MISSING', '', '', '', ''))
        continue
    ok = [r for r in json.loads(p.read_text()) if 'rig_rot_deg' in r]
    mr = sum(r['rig_rot_deg'] for r in ok) / len(ok)
    mf = sum(r['frame_rot_median_deg'] for r in ok) / len(ok)
    rows.append((m, len(ok), f'{mr:.4f}', f'{sum(r["rig_trans_m"] for r in ok)/len(ok):.4f}', f'{mf:.4f}', 'OK'))
    fl = [r for r in ok if r['dataset'] == 'FAST-LIVO2']
    k3 = [r for r in ok if r['dataset'] == 'KITTI-360']
    if fl and k3:
        subset_rows.append((
            m,
            f'{sum(r["rig_rot_deg"] for r in fl)/len(fl):.4f}',
            f'{sum(r["rig_rot_deg"] for r in k3)/len(k3):.4f}',
        ))

lines = [
    '# TLC Benchmark 三模型对照 (cam0)',
    '',
    f'- Generated: {datetime.now().isoformat(timespec="seconds")}',
    '- KITTI-360 默认 cam0（见 docs/TLC_BENCHMARK_NOTES.md）',
    '- Perturbation: 5° rot, 0m trans, max_frames=200',
    '',
    '## 全场景 8 scenes',
    '',
    '| Model | Scenes | Mean Rig Rot (°) | Mean Rig Trans (m) | Mean Frame Rot Med (°) | Status |',
    '|-------|--------|------------------|--------------------|------------------------|--------|',
]
for r in rows:
    lines.append(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |')

lines += ['', '## 子集', '', '| Model | FAST-LIVO2 Rig Rot (°) | KITTI-360 Rig Rot (°) |', '|-------|------------------------|----------------------|']
for m, fl, k3 in subset_rows:
    lines.append(f'| {m} | {fl} | {k3} |')

(base / 'COMPARE_REPORT.md').write_text('\n'.join(lines) + '\n')
print(base / 'COMPARE_REPORT.md')
PY

echo "Done: $OUT_BASE/COMPARE_REPORT.md"
