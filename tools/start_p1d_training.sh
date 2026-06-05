#!/bin/bash
# V40 P1d训练启动脚本

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "============================================"
echo "V40 P1d强化Match路径学习 - 训练启动"
echo "============================================"
echo ""

# 验证配置文件
CONFIG_FILE="configs/v40_gmp_p1d.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误：配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo "✅ 配置文件检查：$CONFIG_FILE"

# 验证YAML语法
echo "检查YAML语法..."
python3 -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ YAML语法检查通过"
else
    echo "❌ YAML语法错误"
    exit 1
fi

# 显示关键配置
echo ""
echo "关键配置确认："
python3 << 'EOF'
import yaml
with open('configs/v40_gmp_p1d.yaml') as f:
    cfg = yaml.safe_load(f)
params = cfg['defaults']['params']
exp = cfg['experiments'][0]
print(f"  实验名称: {exp['name']}")
print(f"  Epoch数: {exp['params']['num_epochs']}")
print(f"  correspondence_loss_weight: {params['correspondence_loss_weight']} (P1c=1.0)")
print(f"  match_valid_ratio_min: {params['match_valid_ratio_min']} (P1c=0.3)")
print(f"  diff_epnp_warmup_epochs: {params['diff_epnp_warmup_epochs']} (P1c=5)")
EOF

# 检查P1c状态
echo ""
echo "检查P1c训练状态..."
P1C_LOG="logs/all_training_data/model_small_10deg_v40_gmp_p1c_main/train.log"
if [ -f "$P1C_LOG" ]; then
    P1C_RUNNING=$(pgrep -f "v40_gmp_p1c_main" | wc -l)
    if [ $P1C_RUNNING -gt 0 ]; then
        echo "⚠️  P1c训练仍在运行"
        echo "    可选择并行运行或先停止P1c"
    else
        echo "✅ P1c训练已停止"
    fi
fi

# 预估时间
echo ""
echo "预估训练时间："
echo "  60 epochs: ~6.9 h"
echo "  预计完成: $(date -d '+7 hours' '+%Y-%m-%d %H:%M')"

# 启动
echo ""
echo "🚀 启动训练..."
echo "执行: bash batch_train.sh $CONFIG_FILE"
echo ""
bash batch_train.sh "$CONFIG_FILE"
