#!/usr/bin/env bash
# Memory footprint test for L20 config
# Tests batch_size=64 with 16384 points before full training
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJFUSION="${PROJFUSION_ROOT:-/mnt/drtraining/user/dahailu/code/ProjFusion}"
PYTHON="${PYTHON:-/opt/conda/envs/bevcalib310/bin/python}"

export PROJFUSION_ROOT="$PROJFUSION"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4

cd "$PROJFUSION"

echo "============================================"
echo "L20 Memory Footprint Test"
echo "============================================"
echo "Testing: batch_size=64, points=16384"
echo "Expected peak memory: ~35-40GB"
echo "Your L20 VRAM: 48GB"
echo "============================================"
echo ""

"$PYTHON" -c "
import torch
import sys
sys.path.insert(0, '$PROJFUSION')

from models.pointgpt.config import get_config
from models.pointgpt.build import build_model_from_cfg

print('Loading model...')
cfg = get_config('cfg/pointgpt/pretrain_fleet_L20.yaml')
model = build_model_from_cfg(cfg.model).cuda()
model.train()

print('Testing forward pass...')
batch_size = 64
npoints = 16384
dummy_input = torch.randn(batch_size, npoints, 3).cuda()

try:
    with torch.cuda.amp.autocast():
        _, loss = model(dummy_input)
    loss.backward()
    
    mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
    
    print('')
    print('✅ Memory test PASSED!')
    print(f'   Peak allocated: {mem_allocated:.2f} GB')
    print(f'   Peak reserved:  {mem_reserved:.2f} GB')
    print(f'   Available margin: {48 - mem_reserved:.2f} GB')
    print('')
    
    if mem_reserved < 40:
        print('💡 You can potentially increase batch_size to 80!')
    elif mem_reserved < 45:
        print('✅ batch_size=64 is optimal for your L20.')
    else:
        print('⚠️  Reduce batch_size to 48 for safety margin.')
    
    sys.exit(0)
    
except RuntimeError as e:
    if 'out of memory' in str(e):
        print('')
        print('❌ OOM! Reduce batch_size to 48 or 32.')
        print(f'   Error: {e}')
        sys.exit(1)
    else:
        raise
" 2>&1 | tee /tmp/l20_memory_test.log

echo ""
echo "============================================"
echo "Test completed. If passed, run:"
echo "  bash tools/run_pretrain_fleet_L20.sh"
echo "============================================"
