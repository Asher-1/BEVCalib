# BEVCalib 配置文件说明

本目录包含BEVCalib项目的各种配置文件。

## 📁 配置文件分类

### 批量训练配置 (batch_train_*.yaml)

用于 `batch_train.sh` 的批量训练配置，定义多组连续训练实验。

| 配置文件 | 用途 | 实验数 |
| --- | --- | --- |
| `batch_train_5deg.yaml` | 5度扰动 + Z分辨率对比 | 3组 (z=1, z=5, z=10) |
| `batch_train_10deg_rotation.yaml` | 10度rotation-only + Z对比 | 3组 (z=1, z=5, z=10) |
| `batch_train_lr_ablation.yaml` | 学习率消融实验 | 4组 (1e-4, 2e-4, 5e-4, 1e-3) |
| `batch_train_multinode.yaml` | 多机DDP训练示例（手动配置） | 2组 (2机×2GPU) |
| `batch_train_multinode_manual.yaml` | 多机DDP手动配置模板 | 1组 |
| `batch_train_multinode_slurm.yaml` | SLURM集群自动多机训练 | 2组 |

**用法:**
```bash
# 使用指定配置
bash batch_train.sh configs/batch_train_5deg.yaml

# 查看会执行的命令（不实际运行）
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml

# 强制重新训练（忽略已存在的实验目录）
bash batch_train.sh --force configs/batch_train_5deg.yaml

# 跳过名称匹配的实验（grep -E 正则）
bash batch_train.sh --skip-pattern "baseline" configs/batch_train_5deg.yaml
```

## 📝 批量训练配置文件格式

### 基本结构（v2.1+ 支持 defaults 继承）

```yaml
# 全局配置
global:
  dry_run: false                 # 仅打印命令不执行
  force_rerun: false             # 强制重跑（忽略已存在的实验目录）
  batch_log_dir: "logs"          # 批量训练日志目录
  wait_between_experiments: 10   # 实验间等待时间（秒）

# ============================================================================
# 默认参数（所有实验继承，除非实验中显式覆盖）
# ============================================================================
defaults:
  # 默认环境变量
  env: {}
  
  # 默认训练参数
  params:
    angle_range_deg: 5          # 旋转扰动
    trans_range: 0.3            # 平移扰动
    batch_size: 16              # Batch size
    learning_rate: null         # 学习率（null=使用默认）
    use_ddp: true               # 使用DDP
    ddp_gpus: null              # GPU数（null=自动检测）
    use_compile: false          # torch.compile加速
    rotation_only: false        # 仅优化旋转
    foreground: true            # 前台执行
    no_tensorboard: true        # 不启动TensorBoard
    tensorboard_port: null      # TB端口
    nnodes: null                # 多机数（null=1）
    node_rank: null             # 机器编号
    master_addr: null           # master IP
    master_port: null           # master端口

# ============================================================================
# 实验组配置（只需指定与defaults不同的参数）
# ============================================================================
experiments:
  - name: "实验1"                # 实验名称
    description: "实验描述"       # 描述（可选）
    dataset: "B26A"              # 数据集: B26A / all / custom
    version: "v1"                # 版本标签
    env:
      BEV_ZBOUND_STEP: 2.0       # 实验特定环境变量
    # params 继承 defaults（无需重复）
  
  - name: "实验2"
    dataset: "B26A"
    version: "v2"
    env:
      BEV_ZBOUND_STEP: 4.0
    params:
      angle_range_deg: 10        # 覆盖defaults中的5
      batch_size: 32             # 覆盖defaults中的16
```

### 参数继承规则（v2.1+）

**关键特性：**
- `defaults` 区域定义所有实验的公共参数
- 实验配置自动继承 `defaults` 中的所有参数
- 实验中显式指定的参数会覆盖 `defaults`
- 优先级：`experiment.params` > `defaults.params`

**优势：**
- 消除重复配置（DRY原则）
- 易于批量修改公共参数
- 实验配置更简洁易读
- 减少配置错误

**示例：**

```yaml
defaults:
  params:
    batch_size: 16              # 默认batch=16
    learning_rate: null         # 默认使用模型默认值

experiments:
  - name: "实验A"
    dataset: "B26A"
    version: "v1"
    # params完全继承defaults
    # 最终: batch_size=16, learning_rate=null
  
  - name: "实验B"
    dataset: "B26A"
    version: "v2"
    params:
      batch_size: 32            # 覆盖defaults
    # 最终: batch_size=32, learning_rate=null（继承）
  
  - name: "实验C"
    dataset: "B26A"
    version: "v3"
    params:
      learning_rate: 1e-4       # 覆盖defaults
    # 最终: batch_size=16（继承）, learning_rate=1e-4
```

### 参数说明

#### 全局配置 (global)

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dry_run` | bool | false | 仅打印命令不执行（CLI: `--dry-run`） |
| `force_rerun` | bool | false | 强制重跑（CLI: `--force`） |
| `batch_log_dir` | str | "logs" | 批量日志目录 |
| `wait_between_experiments` | int | 10 | 实验间等待时间（秒） |

#### 实验配置 (experiments[])

**必需字段:**
| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `dataset` | str | 数据集名称: B26A / all / custom |
| `version` | str | 版本标签（出现在日志目录名） |

**可选字段:**
| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `name` | str | "exp{i}" | 实验显示名称 |
| `description` | str | "" | 实验描述 |
| `skip` | bool | false | 跳过此实验 |
| `skip_reason` | str | "" | 跳过原因（日志中显示） |

**环境变量 (env):**
| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `BEV_ZBOUND_STEP` | float | BEV Z方向步长 |

**训练参数 (params):**
| 参数 | 类型 | 对应命令行 | 说明 |
| --- | --- | --- | --- |
| `angle_range_deg` | float | `--angle` | 旋转扰动范围（度） |
| `trans_range` | float | `--trans` | 平移扰动范围（米） |
| `batch_size` | int | `--bs` | Batch size |
| `learning_rate` | float | `--lr` | 初始学习率 |
| `use_ddp` | bool | `--ddp` | 使用DDP多GPU |
| `ddp_gpus` | int | `--ddp N` | DDP使用的GPU数 |
| `use_compile` | bool | `--compile` | torch.compile加速 |
| `rotation_only` | bool | `--rotation_only` | 仅优化旋转 |
| `foreground` | bool | `--fg` | 前台执行 |
| `no_tensorboard` | bool | `--no-tb` | 不启动TensorBoard |
| `tensorboard_port` | int | `--tb_port` | TensorBoard端口 |
| `nnodes` | int | `--nnodes` | 多机DDP-总机器数 |
| `node_rank` | int | `--node_rank` | 多机DDP-当前机器编号 |
| `master_addr` | str | `--master_addr` | 多机DDP-master IP |
| `master_port` | int | `--master_port` | 多机DDP-master端口 |

**注意:** 所有 `params` 字段都是可选的，`null` 值表示使用默认值。

## 🖥️ 多机训练配置

### 快速生成多机配置

使用自动生成脚本：

```bash
cd configs/

# 默认配置（2机×2GPU）
bash generate_multinode_configs.sh

# 自定义配置（4机×4GPU，大batch）
MASTER_IP=10.0.0.1 NNODES=4 GPUS_PER_NODE=4 BATCH_SIZE=128 \
    bash generate_multinode_configs.sh

# 生成的文件:
# batch_train_multinode_node0.yaml (Master)
# batch_train_multinode_node1.yaml (Worker)
# batch_train_multinode_node2.yaml (Worker)
# ...
```

### 手动多机训练流程

#### 步骤1: 准备配置文件

**推荐方式：使用自动生成脚本**

```bash
# 自动生成各节点配置
bash configs/generate_multinode_configs.sh \
  configs/batch_train_multinode_manual.yaml 2 192.168.1.100
```

**手动方式：使用 defaults 继承**

```bash
# Master节点配置 (node0.yaml)
cat > configs/batch_train_node0.yaml << 'EOF'
global:
  dry_run: false

defaults:
  params:
    use_ddp: true
    ddp_gpus: 8
    batch_size: 32
    nnodes: 2
    master_addr: "192.168.1.100"
    master_port: 29500
    foreground: true
    no_tensorboard: true
    node_rank: 0               # Master节点

experiments:
  - name: "多机训练"
    dataset: "all"
    version: "v1-multinode"
    env:
      BEV_ZBOUND_STEP: 2.0
EOF

# Worker节点配置 (node1.yaml)
cat > configs/batch_train_node1.yaml << 'EOF'
global:
  dry_run: false

defaults:
  params:
    use_ddp: true
    ddp_gpus: 8
    batch_size: 32
    nnodes: 2
    master_addr: "192.168.1.100"
    master_port: 29500
    foreground: true
    no_tensorboard: true
    node_rank: 1               # Worker节点

experiments:
  - name: "多机训练"
    dataset: "all"
    version: "v1-multinode"
    env:
      BEV_ZBOUND_STEP: 2.0
EOF
```

#### 步骤2: 分发配置到各节点

```bash
# 从主控机器分发
scp configs/batch_train_node0.yaml node0:/path/to/BEVCalib/configs/
scp configs/batch_train_node1.yaml node1:/path/to/BEVCalib/configs/
```

#### 步骤3: 依次启动训练

```bash
# Node 0 (Master) - 先启动
ssh node0 'cd /path/to/BEVCalib && bash batch_train.sh configs/batch_train_node0.yaml'

# Node 1 (Worker) - 30秒内启动
ssh node1 'cd /path/to/BEVCalib && bash batch_train.sh configs/batch_train_node1.yaml'
```

### SLURM集群训练

#### 使用SLURM作业脚本

```bash
# 提交作业（自动多机）
sbatch configs/run_batch_train.slurm configs/batch_train_multinode_slurm.yaml

# 查看作业状态
squeue -u $USER

# 查看日志
tail -f slurm_<job_id>.log

# 取消作业
scancel <job_id>
```

#### SLURM配置文件（使用 defaults 继承）

```yaml
defaults:
  params:
    use_ddp: true
    batch_size: 64
    # 所有多机参数设为null，自动从SLURM环境变量读取
    nnodes: null
    node_rank: null
      master_addr: null
      master_port: null
      foreground: true
      no_tensorboard: true
```

### 多机训练检查清单

**训练前检查**:
- [ ] 所有机器能互相ping通
- [ ] `master_port` (默认29500) 在防火墙中开放
- [ ] 数据集路径在所有机器上一致且可访问
- [ ] 所有机器的 `bevcalib` conda环境已激活
- [ ] 所有机器的GPU数量一致（`ddp_gpus`）
- [ ] Master节点IP地址正确（`master_addr`）
- [ ] 每台机器的配置文件 `node_rank` 正确且唯一

**启动顺序**:
1. 先启动Master节点（rank=0）
2. 在30秒内启动所有Worker节点（rank=1,2,...）
3. 观察Master节点日志，确认所有节点已连接

**监控和调试**:
```bash
# 查看GPU状态
nvidia-smi -l 1

# 查看训练日志（Master节点）
tail -f logs/all/model_*/train.log

# 测试网络连通性
nc -zv 192.168.1.100 29500

# 查看NCCL调试信息（在配置中添加）
env:
  NCCL_DEBUG: "INFO"
  NCCL_DEBUG_SUBSYS: "INIT,COLL"
```

## 使用示例

### 示例1: Z分辨率消融实验

使用 `batch_train_5deg.yaml`:

```bash
# 运行3组实验: z=1, z=5, z=10 (5度扰动)
bash batch_train.sh configs/batch_train_5deg.yaml

# 仅查看会执行的命令
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml
```

### 示例2: Rotation-Only实验

使用 `batch_train_10deg_rotation.yaml`:

```bash
# 运行3组rotation-only实验 (10度扰动)
bash batch_train.sh configs/batch_train_10deg_rotation.yaml
```

### 示例3: 学习率消融实验

使用 `batch_train_lr_ablation.yaml`:

```bash
# 对比4组不同学习率
bash batch_train.sh configs/batch_train_lr_ablation.yaml
```

### 示例4: TensorBoard 监控（v2.1+ 智能管理）

`batch_train.sh` 会自动为每个实验启动 TensorBoard，监控**当前实验的具体目录**：

```bash
# 运行批量训练
bash batch_train.sh configs/batch_train_5deg.yaml

# TensorBoard 自动启动并监控:
#   实验1 (z=1): logs/B26A/model_small_5deg_v4-z1/     ← 只显示实验1
#   实验2 (z=5): logs/B26A/model_small_5deg_v4-z5/     ← 只显示实验2
#   实验3 (z=10): logs/B26A/model_small_5deg_v4-z10/   ← 只显示实验3
```

**优化前** (❌): 监控 `logs/B26A/`，同时显示所有实验，干扰大  
**优化后** (✅): 监控具体实验目录，清晰聚焦当前训练

**禁用 TensorBoard**:
```yaml
defaults:
  params:
    no_tensorboard: true        # 所有实验不启动TensorBoard
```

### 示例5: 创建自定义配置

```bash
# 复制模板
cp configs/batch_train_template.yaml configs/my_experiments.yaml

# 编辑配置
vim configs/my_experiments.yaml

# 运行
bash batch_train.sh configs/my_experiments.yaml
```

## 配置模板

### 最小配置（使用 defaults）

```yaml
global:
  dry_run: false

defaults:
  params:
    use_ddp: true
    foreground: true
    no_tensorboard: true

experiments:
  - dataset: "B26A"
    version: "v1"
    env:
      BEV_ZBOUND_STEP: 4.0
```

### 完整配置（使用 defaults 继承）

```yaml
global:
  dry_run: false
  batch_log_dir: "logs"
  wait_between_experiments: 10

# 定义公共参数
defaults:
  env: {}
  params:
    angle_range_deg: 5
    trans_range: 0.3
    batch_size: 16
    learning_rate: 0.0002
    use_ddp: true
    ddp_gpus: 2
    use_compile: false
    rotation_only: false
    foreground: true
    no_tensorboard: true
    tensorboard_port: 6006

# 实验只需指定差异化参数
experiments:
  - name: "完整配置示例"
    description: "展示所有可用参数"
    dataset: "B26A"
    version: "v1"
    env:
      BEV_ZBOUND_STEP: 2.0
```

### 多机DDP配置（手动，使用 defaults）

```yaml
defaults:
  params:
    use_ddp: true
    ddp_gpus: 8                # 每台机器的GPU数
    batch_size: 32             # 多机可用更大batch
    nnodes: 2                  # 总机器数
    node_rank: 0               # ⚠️ 在不同机器上修改此值
    master_addr: "192.168.1.100"  # Master节点IP
    master_port: 29500         # 通信端口
    foreground: true
    no_tensorboard: true

experiments:
  - name: "多机DDP实验"
    dataset: "all"
    version: "v1_multi"
    env:
      BEV_ZBOUND_STEP: 2.0
```

**使用方式**:
1. 使用 `generate_multinode_configs.sh` 自动生成各节点配置
2. 或手动在每台机器上修改 `defaults.params.node_rank`
3. 先启动Master节点（rank=0），30秒内启动所有Worker

### 多机DDP配置（SLURM自动，使用 defaults）

```yaml
defaults:
  params:
    use_ddp: true
    batch_size: 64
    learning_rate: 4e-4        # 大batch增大学习率
    # SLURM自动检测（设为null）
    nnodes: null               # 从 $SLURM_NNODES 自动检测
    node_rank: null            # 从 $SLURM_NODEID 自动检测
      master_addr: null          # 从 $SLURM_NODELIST 自动检测
      master_port: null          # 自动检测空闲端口
      
      foreground: true
      no_tensorboard: true
```

**使用SLURM**:
```bash
sbatch configs/run_batch_train.slurm configs/batch_train_multinode_slurm.yaml
```

## 常见问题

### Q: 如何添加新实验？

在 `experiments` 列表中添加新条目：

```yaml
experiments:
  - name: "新实验"
    dataset: "B26A"
    version: "v_new"
    env:
      BEV_ZBOUND_STEP: 4.0
    # params 继承 defaults
```

### Q: 如何修改所有实验的某个参数？

**使用 `defaults` 区域**（推荐，v2.1+）：

```yaml
defaults:
  params:
    batch_size: 32              # 修改这里，所有实验都会使用32
    use_ddp: true

experiments:
  - name: "实验1"
    dataset: "B26A"
    version: "v1"
    env:
      BEV_ZBOUND_STEP: 4.0
    # 自动继承 batch_size=32
  
  - name: "实验2"
    dataset: "B26A"
    version: "v2"
    env:
      BEV_ZBOUND_STEP: 2.0
    # 自动继承 batch_size=32
```

### Q: 某个实验需要不同的参数怎么办？

在实验中覆盖 `defaults`：

```yaml
defaults:
  params:
    batch_size: 16              # 默认batch=16

experiments:
  - name: "实验1（使用默认）"
    dataset: "B26A"
    version: "v1"
    env:
      BEV_ZBOUND_STEP: 4.0
    # batch_size=16（继承）
  
  - name: "实验2（覆盖）"
    dataset: "B26A"
    version: "v2"
    env:
      BEV_ZBOUND_STEP: 4.0
    params:
      batch_size: 32            # 覆盖defaults，使用32
```

### Q: 配置文件中的null是什么意思？

`null` 表示使用默认值，等同于不传该参数。

### Q: 如何控制实验的执行顺序？

实验按配置文件中的顺序依次执行。

### Q: 可以并行执行多个实验吗？

当前版本不支持并行，实验串行执行。如需并行，请手动在不同终端运行 `start_training.sh`。

## 依赖要求

- Python 3.7+
- PyYAML: `pip install pyyaml`

配置文件解析会自动检查并安装 PyYAML。

## 版本历史

### v2.1 (2026-03-09)
- ✅ **新增 `defaults` 区域支持参数继承**
- ✅ 消除重复配置，提高可维护性
- ✅ 支持多机DDP参数的默认值配置
- ✅ 所有配置文件已更新为新格式

### v2.0 (2026-03-09)
- ✅ 从硬编码改为配置文件驱动
- ✅ 支持 `start_training.sh` 的所有参数
- ✅ 提供多个场景的预配置模板
- ✅ 更好的可读性和可维护性

### v1.0 (之前)
- 硬编码实验配置在脚本中
- 支持基础参数

---

**维护者**: dahailu  
**最后更新**: 2026-03-09
