#!/bin/bash
# 演示 defaults 继承机制的效果

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "================================================================"
echo "BEVCalib 配置文件 defaults 继承机制演示"
echo "v2.1 新特性 - 消除重复配置"
echo "================================================================"
echo ""

echo "【演示】配置文件对比"
echo ""
echo "优化前（v2.0）- 每个实验重复所有参数："
echo "  实验1: 18行参数"
echo "  实验2: 18行参数"
echo "  实验3: 18行参数"
echo "  总计: 54行"
echo ""
echo "优化后（v2.1）- 使用 defaults 继承："
echo "  defaults: 12行公共参数"
echo "  实验1: 5行差异参数"
echo "  实验2: 5行差异参数"
echo "  实验3: 5行差异参数"
echo "  总计: 27行 (-50%)"
echo ""
echo "----------------------------------------------------------------"
echo ""

echo "【测试1】验证 batch_train_5deg.yaml（3个实验）"
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml 2>&1 | grep -E "(实验 \[|命令:)" | head -6
echo ""

echo "【测试2】验证 batch_train_lr_ablation.yaml（4个实验，参数覆盖）"
bash batch_train.sh --dry-run configs/batch_train_lr_ablation.yaml 2>&1 | grep -E "(实验 \[|命令:)" | head -8
echo ""

echo "【测试3】验证 batch16_train_multinode.yaml（多机训练）"
bash batch_train.sh --dry-run configs/batch16_train_multinode.yaml 2>&1 | grep -E "(实验 \[|命令:)" | head -4
echo ""

echo "================================================================"
echo "核心优势"
echo "================================================================"
echo ""
echo "✅ 消除重复配置（DRY原则）"
echo "✅ 修改公共参数从改N处 → 改1处"
echo "✅ 实验差异一目了然"
echo "✅ 配置错误显著减少"
echo "✅ 向后兼容（旧格式仍然支持）"
echo ""
echo "统计: 391行 → 167行 (-57%)"
echo ""
echo "详细文档: configs/DEFAULTS_MIGRATION_GUIDE.md"
echo ""
