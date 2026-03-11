#!/bin/bash
# =============================================================================
# 多机训练配置演示脚本
# 展示如何使用配置文件进行多机训练
# =============================================================================

set -e

echo "================================================================"
echo "BEVCalib 多机训练配置演示"
echo "================================================================"
echo ""

# =============================================================================
# 演示1: 查看预配置的多机模板
# =============================================================================

echo "📋 演示1: 查看预配置模板"
echo "----------------------------------------"
echo ""
echo "可用的多机配置模板:"
ls -1 configs/batch_train_multinode*.yaml 2>/dev/null || echo "  (无)"
echo ""
echo "辅助工具:"
ls -1 configs/generate_multinode_configs.sh configs/run_batch_train.slurm 2>/dev/null || echo "  (无)"
echo ""

# =============================================================================
# 演示2: 预览多机配置
# =============================================================================

echo "🔍 演示2: 预览多机配置内容"
echo "----------------------------------------"
echo ""
echo ">>> configs/batch_train_multinode.yaml (前15行):"
head -15 configs/batch_train_multinode.yaml
echo "..."
echo ""

# =============================================================================
# 演示3: 自动生成配置文件
# =============================================================================

echo "🤖 演示3: 自动生成多机配置"
echo "----------------------------------------"
echo ""
echo "执行: bash configs/generate_multinode_configs.sh"
echo ""
cd configs && bash generate_multinode_configs.sh
cd ..
echo ""

# =============================================================================
# 演示4: Dry-run测试生成的配置
# =============================================================================

echo "🧪 演示4: Dry-run验证配置"
echo "----------------------------------------"
echo ""
echo ">>> Master节点配置 (node0.yaml):"
bash batch_train.sh --dry-run configs/batch_train_multinode_node0.yaml 2>&1 | grep -A3 "命令:"
echo ""
echo ">>> Worker节点配置 (node1.yaml):"
bash batch_train.sh --dry-run configs/batch_train_multinode_node1.yaml 2>&1 | grep -A3 "命令:"
echo ""

# =============================================================================
# 演示5: 参数验证
# =============================================================================

echo "✅ 演示5: 验证多机参数"
echo "----------------------------------------"
echo ""
echo "检查生成的命令中的多机参数:"
echo ""
bash batch_train.sh --dry-run configs/batch_train_multinode_node0.yaml 2>&1 | grep "命令:" | grep -o "\-\-nnodes [0-9]*" && echo "  ✓ nnodes 参数存在"
bash batch_train.sh --dry-run configs/batch_train_multinode_node0.yaml 2>&1 | grep "命令:" | grep -o "\-\-node_rank [0-9]*" && echo "  ✓ node_rank 参数存在"
bash batch_train.sh --dry-run configs/batch_train_multinode_node0.yaml 2>&1 | grep "命令:" | grep -o "\-\-master_addr [0-9.]*" && echo "  ✓ master_addr 参数存在"
bash batch_train.sh --dry-run configs/batch_train_multinode_node0.yaml 2>&1 | grep "命令:" | grep -o "\-\-master_port [0-9]*" && echo "  ✓ master_port 参数存在"
echo ""

# =============================================================================
# 演示6: SLURM配置预览
# =============================================================================

echo "🖥️ 演示6: SLURM配置预览"
echo "----------------------------------------"
echo ""
echo "SLURM配置中的多机参数设为null（自动检测）:"
echo ""
bash batch_train.sh --dry-run configs/batch_train_multinode_slurm.yaml 2>&1 | grep -A1 "实验 \[1" | head -5
echo ""
echo "注意: SLURM模式下，多机参数不在命令行中（由环境变量提供）"
echo ""

# =============================================================================
# 总结
# =============================================================================

echo "================================================================"
echo "演示完成！"
echo "================================================================"
echo ""
echo "📚 查看完整文档:"
echo "  - 多机训练指南: cat MULTINODE_TRAINING.md"
echo "  - 快速开始: cat MULTINODE_QUICKSTART.md"
echo "  - 配置文档: cat configs/README.md"
echo ""
echo "🚀 实际使用:"
echo "  1. 自定义参数生成配置:"
echo "     MASTER_IP=<IP> NNODES=<N> bash configs/generate_multinode_configs.sh"
echo ""
echo "  2. 启动训练:"
echo "     Master: bash batch_train.sh configs/batch_train_multinode_node0.yaml"
echo "     Worker: bash batch_train.sh configs/batch_train_multinode_node1.yaml"
echo ""
echo "  3. SLURM提交:"
echo "     sbatch configs/run_batch_train.slurm configs/batch_train_multinode_slurm.yaml"
echo ""
echo "================================================================"
