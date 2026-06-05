#!/bin/bash
# V40 训练启动前验证：检查参数传递、关键文件路径
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/v40_gmp_p1d.yaml}"
echo "=== V40 启动验证: $CONFIG ==="
echo ""

# 1. 生成 launch 命令
CMD=$(python3 - "$CONFIG" <<'PY'
import sys, yaml, json, shlex

cfg = yaml.safe_load(open(sys.argv[1]))
exp = [e for e in cfg['experiments'] if not e.get('skip', False)][0]
params = {**(cfg.get('defaults') or {}).get('params') or {}, **(exp.get('params') or {})}

issues = []
# pointgpt config 路径
pg_cfg = params.get('native_cross_pointgpt_config', '')
if pg_cfg and not __import__('os').path.isfile(pg_cfg):
    issues.append(f"❌ native_cross_pointgpt_config 不存在: {pg_cfg}")

# vertical fov 引号
vfov = str(params.get('augment_lidar_vertical_fov', ''))
if vfov.startswith('-'):
    quoted = shlex.quote(vfov)
    print(f"augment_lidar_vertical_fov 需引号: {vfov} -> {quoted}")
    if quoted == vfov:
        issues.append(f"❌ vertical_fov 未加引号: {vfov}")

pretrain = params.get('pretrain_ckpt', '')
if pretrain and not __import__('os').path.isfile(pretrain):
    issues.append(f"❌ pretrain_ckpt 不存在: {pretrain}")

# 检查 start_training.sh 是否支持 P2b 参数
import subprocess
r = subprocess.run(
    ['bash', '-c', f'grep -q augment_fov_crop_prob start_training.sh && grep -q augment_lidar_vertical_fov start_training.sh'],
    cwd=sys.argv[2] if len(sys.argv) > 2 else '.',
    capture_output=True,
)
# 简化：直接读文件
st = open('start_training.sh').read()
tu = open('train_universal.sh').read()
for flag in ['augment_fov_crop_prob', 'augment_lidar_vertical_fov', 'correspondence_loss_weight']:
    if flag not in st:
        issues.append(f"❌ start_training.sh 缺少 --{flag}")
    if flag not in tu:
        issues.append(f"❌ train_universal.sh 缺少 --{flag}")

if issues:
    print("\n".join(issues))
    sys.exit(1)

print("✅ 配置检查通过")
print(f"   实验: {exp.get('name')}")
print(f"   correspondence_loss_weight: {params.get('correspondence_loss_weight')}")
print(f"   match_valid_ratio_min: {params.get('match_valid_ratio_min')}")
print(f"   diff_epnp_warmup_epochs: {params.get('diff_epnp_warmup_epochs')}")
print(f"   augment_lidar_vertical_fov: {vfov}")
PY
)
echo "$CMD"
echo ""

# 2. dry-run 命令行
echo "--- batch_train dry-run (关键片段) ---"
DRY_RUN=1 bash batch_train.sh "$CONFIG" 2>&1 | grep -E "augment_lidar_vertical_fov|native_cross_pointgpt_config|correspondence_loss_weight" | head -5

if echo "$(DRY_RUN=1 bash batch_train.sh "$CONFIG" 2>&1)" | grep -q "augment_lidar_vertical_fov -25,15"; then
    echo "❌ 命令行中 vertical_fov 仍未加引号!"
    exit 1
fi
echo "✅ vertical_fov 引号检查通过"
echo ""
echo "可以启动: bash batch_train.sh $CONFIG"
