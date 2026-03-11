#!/bin/bash
# Auto-generate multi-node training configs (with defaults)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MASTER_IP="${MASTER_IP:-192.168.1.100}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
DATASET="${DATASET:-all}"
VERSION="${VERSION:-v1-multinode}"
ANGLE="${ANGLE:-5}"
TRANS="${TRANS:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-null}"

echo "================================================================"
echo "Multi-node Training Config Generator (v2.1 with defaults)"
echo "================================================================"
echo "Master IP:     $MASTER_IP"
echo "Total Nodes:   $NNODES"
echo "GPUs per Node: $GPUS_PER_NODE"
echo "Master Port:   $MASTER_PORT"
echo "Dataset:       $DATASET"
echo "Version:       $VERSION"
echo "Batch Size:    $BATCH_SIZE"
echo "================================================================"
echo ""

for rank in $(seq 0 $((NNODES-1))); do
    ROLE="Worker"
    [ "$rank" -eq 0 ] && ROLE="Master"
    
    OUTPUT_FILE="batch_train_multinode_node${rank}.yaml"
    
    cat > "$OUTPUT_FILE" << YAML_EOF
# Multi-node config for Node ${rank} (${ROLE})
# Generated: $(date)

global:
  dry_run: false
  batch_log_dir: "logs"
  wait_between_experiments: 10

# ============================================================================
# Defaults (using v2.1 inheritance)
# ============================================================================
defaults:
  env: {}
  
  params:
    # 扰动参数
    angle_range_deg: ${ANGLE}
    trans_range: ${TRANS}
    
    # 训练参数
    batch_size: ${BATCH_SIZE}
    learning_rate: ${LEARNING_RATE}
    
    # DDP配置
    use_ddp: true
    ddp_gpus: ${GPUS_PER_NODE}
    
    # 多机DDP配置（节点特定）
    nnodes: ${NNODES}
    node_rank: ${rank}             # Node ${rank} (${ROLE})
    master_addr: "${MASTER_IP}"
    master_port: ${MASTER_PORT}
    
    # 执行模式
    foreground: true
    no_tensorboard: true

# ============================================================================
# Experiments
# ============================================================================
experiments:
  - name: "${ROLE} Node (rank=${rank}/${NNODES})"
    description: "${NNODES} nodes, ${GPUS_PER_NODE} GPUs per node"
    dataset: "${DATASET}"
    version: "${VERSION}"
    env:
      BEV_ZBOUND_STEP: 2.0
    # params 完全继承 defaults
YAML_EOF
    
    echo "✓ Generated: $OUTPUT_FILE (Node ${rank} - ${ROLE})"
done

echo ""
echo "================================================================"
echo "Config files generated (v2.1 format with defaults)!"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Distribute configs to each node:"
for rank in $(seq 0 $((NNODES-1))); do
    echo "     scp batch_train_multinode_node${rank}.yaml node${rank}:/path/to/BEVCalib/configs/"
done
echo ""
echo "  2. Start training on each node:"
echo "     # Master (node0):"
echo "     bash batch_train.sh configs/batch_train_multinode_node0.yaml"
echo ""
echo "     # Workers (within 30 seconds):"
for rank in $(seq 1 $((NNODES-1))); do
    echo "     bash batch_train.sh configs/batch_train_multinode_node${rank}.yaml  # node${rank}"
done
echo ""
echo "================================================================"
