# 多机训练30秒快速开始

## ⚡ 最快的方式（3步）

### 步骤1: 生成配置（10秒）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib/configs
bash generate_multinode_configs.sh
```

### 步骤2: 分发配置（10秒）

```bash
# 假设有2台机器: node0 (Master), node1 (Worker)
scp batch_train_multinode_node0.yaml node0:/path/to/BEVCalib/configs/
scp batch_train_multinode_node1.yaml node1:/path/to/BEVCalib/configs/
```

### 步骤3: 启动训练（10秒）

```bash
# Master节点（先启动）
ssh node0 'cd /path/to/BEVCalib && bash batch_train.sh configs/batch_train_multinode_node0.yaml'

# Worker节点（30秒内启动）
ssh node1 'cd /path/to/BEVCalib && bash batch_train.sh configs/batch_train_multinode_node1.yaml'
```

## 📝 自定义配置（1分钟）

```bash
# 自定义参数生成
MASTER_IP=<你的Master IP> \
NNODES=<机器数> \
GPUS_PER_NODE=<每台GPU数> \
BATCH_SIZE=<总batch size> \
    bash configs/generate_multinode_configs.sh

# 示例: 4机×4GPU，batch=128
MASTER_IP=10.0.0.1 NNODES=4 GPUS_PER_NODE=4 BATCH_SIZE=128 \
    bash configs/generate_multinode_configs.sh
```

## 🔍 验证配置（可选）

```bash
# 预览Master命令
bash batch_train.sh --dry-run configs/batch_train_multinode_node0.yaml

# 预览Worker命令
bash batch_train.sh --dry-run configs/batch_train_multinode_node1.yaml

# 检查输出应包含:
# --nnodes 2 --node_rank 0/1 --master_addr ... --master_port 29500
```

## 🎯 SLURM用户（更快）

```bash
# 1行命令提交
sbatch configs/run_batch_train.slurm configs/batch_train_multinode_slurm.yaml

# 查看状态
squeue -u $USER
```

## 📚 详细文档

完整指南: [MULTINODE_TRAINING.md](MULTINODE_TRAINING.md)

---

**需要时间**: 30秒-1分钟  
**前置条件**: 多台机器 + 网络互通 + 共享存储
