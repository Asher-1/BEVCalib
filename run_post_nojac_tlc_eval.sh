#!/bin/bash
# Post-training: TLC cam0 benchmark for nojac_full + 4-model compare
set -euo pipefail
cd "$(dirname "$0")"
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib310

TLC_ROOT="../TLC-Calib/data/TLC-Calib"
OUT="logs/evaluations/tlc_benchmark_cam0"
CKPT_NOJAC="logs/all_training_data/model_small_5deg_v54a_lsp_stable_nojac_full/all_training_data_scratch/checkpoint/ckpt_best_medw.pth"

echo "[$(date +%F\ %T)] TLC benchmark nojac_full (cam0)"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 USE_DRCV_BACKEND=0 \
  python3 tools/tlc_bevcalib_benchmark.py \
    --tlc_root "$TLC_ROOT" \
    --ckpt_path "$CKPT_NOJAC" \
    --output_dir "$OUT/nojac_full" \
    --work_dir "$OUT/nojac_full/work" \
    --angle_deg 5 --trans_range 0 --max_frames 200 --batch_size 8 \
    > "$OUT/nojac_full.log" 2>&1

python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime

base = Path('logs/evaluations/tlc_benchmark_cam0')
models = {
    'v54a_lsp_full': base / 'v54a_lsp/tlc_benchmark_results.json',
    'v54a_nolsp': base / 'v54a_nolsp/tlc_benchmark_results.json',
    'v53c': base / 'v53c/tlc_benchmark_results.json',
    'nojac_full (medw ep1)': base / 'nojac_full/tlc_benchmark_results.json',
}

def load_ok(p):
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text()) if 'rig_rot_deg' in r]

lines = [
    '# TLC Benchmark cam0 — 四模型对照',
    '',
    f'- Updated: {datetime.now().isoformat(timespec="seconds")}',
    '- KITTI-360 默认 cam0',
    '- nojac_full 使用 ckpt_best_medw.pth（ep1, MEDW max=0.043°）',
    '',
    '## 全场景 8 scenes',
    '',
    '| Model | Mean Rig Rot (°) | FAST-LIVO2 (°) | KITTI-360 (°) | large_zigzag (°) |',
    '|-------|------------------|----------------|---------------|------------------|',
]

for name, path in models.items():
    ok = load_ok(path)
    if not ok:
        lines.append(f'| {name} | — | — | — | — |')
        continue
    mr = sum(r['rig_rot_deg'] for r in ok) / len(ok)
    fl = [r for r in ok if r['dataset'] == 'FAST-LIVO2']
    k3 = [r for r in ok if r['dataset'] == 'KITTI-360']
    lz = [r for r in ok if r['scene'] == 'large_zigzag']
    fl_m = sum(r['rig_rot_deg'] for r in fl) / len(fl) if fl else float('nan')
    k3_m = sum(r['rig_rot_deg'] for r in k3) / len(k3) if k3 else float('nan')
    lz_m = lz[0]['rig_rot_deg'] if lz else float('nan')
    lines.append(f'| {name} | {mr:.4f} | {fl_m:.4f} | {k3_m:.4f} | {lz_m:.4f} |')

out = base / 'COMPARE_REPORT_4MODEL.md'
out.write_text('\n'.join(lines) + '\n')
print(out)
PY

echo "Done: $OUT/COMPARE_REPORT_4MODEL.md"
