#!/bin/bash
# =============================================================================
# BEVCalib 批量训练脚本 (配置文件驱动版本)
#
# 功能: 依次运行多组消融实验，每组完成后自动启动下一组
#       通过YAML配置文件定义实验参数，支持所有start_training.sh选项
#       自动跳过已完成的实验（输出目录已存在train.log）
#
# 用法:
#   bash batch_train.sh [options] [config_file]
#   bash batch_train.sh configs/batch_train_5deg.yaml
#   bash batch_train.sh --dry-run configs/batch_train_5deg.yaml
#   bash batch_train.sh --force configs/batch_train_5deg.yaml
#   bash batch_train.sh --skip-pattern "baseline" configs/batch_train_5deg.yaml
#
# 配置文件:
#   configs/batch_train_5deg.yaml           - 5度扰动Z分辨率对比
#   configs/batch_train_10deg_rotation.yaml - 10度rotation-only实验
#   configs/batch_train_lr_ablation.yaml    - 学习率消融实验
#
# 自定义配置: 复制模板并修改
#   cp configs/batch_train_5deg.yaml configs/my_experiments.yaml
#   bash batch_train.sh configs/my_experiments.yaml
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# 参数解析
# =============================================================================

DRY_RUN=0
FORCE_RERUN=0
SKIP_PATTERN=""
CONFIG_FILE=""

show_help() {
    cat << EOF
BEVCalib 批量训练脚本 (配置文件驱动)

用法:
    bash batch_train.sh [options] [config_file]

选项:
    --dry-run           仅打印命令，不实际执行
    --force             强制重新训练，即使输出目录已存在
    --skip-pattern RE   跳过名称匹配正则表达式的实验 (grep -E 语法)
    -h, --help          显示此帮助信息

配置文件:
    如果不指定，默认使用 configs/batch_train_5deg.yaml

示例:
    # 使用默认配置
    bash batch_train.sh

    # 使用指定配置
    bash batch_train.sh configs/batch_train_10deg_rotation.yaml

    # 仅查看会执行的命令
    bash batch_train.sh --dry-run configs/batch_train_5deg.yaml

    # 强制重新训练（忽略已存在的实验目录）
    bash batch_train.sh --force configs/batch_train_5deg.yaml

    # 跳过名称中包含 baseline 的实验
    bash batch_train.sh --skip-pattern "baseline" configs/batch_train_5deg.yaml

    # 后台运行
    nohup bash batch_train.sh configs/batch_train_5deg.yaml > batch.log 2>&1 &

可用配置文件:
    configs/batch8_train_b26a_optim_v1.yaml           - 5度扰动Z分辨率对比
    configs/batch16_train_32nodes_v1.yaml - 10度rotation-only实验
    configs/batch16_train_32nodes_optim_v3.yaml - 32节点优化实验
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --force)
            FORCE_RERUN=1
            shift
            ;;
        --skip-pattern)
            SKIP_PATTERN="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

# 默认配置文件
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="configs/batch_train_5deg.yaml"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ 配置文件不存在: $CONFIG_FILE"
    echo ""
    show_help
    exit 1
fi

# =============================================================================
# YAML 解析器 (使用 Python)
# =============================================================================

parse_config() {
    python3 - "$CONFIG_FILE" <<'EOF'
import sys
import yaml
import json

config_file = sys.argv[1]
with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 输出为JSON（便于bash解析）
print(json.dumps(config, ensure_ascii=False))
EOF
}

# 检查 Python 和 PyYAML
if ! command -v python3 &>/dev/null; then
    echo "✗ Python3 未安装"
    exit 1
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    echo "✗ PyYAML 未安装，正在安装..."
    pip3 install pyyaml --quiet || {
        echo "✗ PyYAML 安装失败"
        exit 1
    }
fi

# =============================================================================
# 解析配置
# =============================================================================

echo "解析配置文件: $CONFIG_FILE"
CONFIG_JSON=$(parse_config)

if [ -z "$CONFIG_JSON" ]; then
    echo "✗ 配置文件解析失败"
    exit 1
fi

# 提取全局配置
GLOBAL_DRY_RUN=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; c=json.load(sys.stdin); print(c.get('global',{}).get('dry_run', False))" 2>/dev/null)
BATCH_LOG_DIR=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; c=json.load(sys.stdin); print(c.get('global',{}).get('batch_log_dir', 'logs'))" 2>/dev/null)
WAIT_TIME=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; c=json.load(sys.stdin); print(c.get('global',{}).get('wait_between_experiments', 10))" 2>/dev/null)
GLOBAL_FORCE_RERUN=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; c=json.load(sys.stdin); print(c.get('global',{}).get('force_rerun', False))" 2>/dev/null)

# TensorBoard 端口 (从 defaults.params.tensorboard_port 读取)
TB_PORT=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; c=json.load(sys.stdin); p=c.get('defaults',{}).get('params',{}) or {}; v=p.get('tensorboard_port'); print(v if v is not None else 6006)" 2>/dev/null)
TB_PORT=${TB_PORT:-6006}

# Node IP (K8s环境: 宿主机IP, 用于TensorBoard地址显示)
YAML_NODE_IP=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; c=json.load(sys.stdin); p=c.get('defaults',{}).get('params',{}) or {}; v=p.get('node_ip'); print(v if v and v != 'None' else '')" 2>/dev/null)
if [ -n "$YAML_NODE_IP" ]; then
    export NODE_IP="$YAML_NODE_IP"
fi

# 命令行 --dry-run 覆盖配置文件
[ "$DRY_RUN" -eq 1 ] && GLOBAL_DRY_RUN="True"
[ "$GLOBAL_DRY_RUN" == "True" ] && DRY_RUN=1
# 命令行 --force 覆盖配置文件 force_rerun
[ "$FORCE_RERUN" -eq 1 ] && GLOBAL_FORCE_RERUN="True"
[ "$GLOBAL_FORCE_RERUN" == "True" ] && FORCE_RERUN=1
[ "$FORCE_RERUN" -eq 1 ] && export FORCE_RERUN=1

# 提取实验数量
TOTAL=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; c=json.load(sys.stdin); print(len(c.get('experiments',[])))" 2>/dev/null)

if [ "$TOTAL" -eq 0 ]; then
    echo "✗ 配置文件中没有定义实验"
    exit 1
fi

# =============================================================================
# TensorBoard 管理
# =============================================================================

TB_PID=""

stop_tensorboard() {
    if [ -n "$TB_PID" ] && kill -0 "$TB_PID" 2>/dev/null; then
        log "停止上一组 TensorBoard (PID=$TB_PID)..."
        kill "$TB_PID" 2>/dev/null
        wait "$TB_PID" 2>/dev/null
        TB_PID=""
    fi
    pkill -f "tensorboard.*--logdir.*$SCRIPT_DIR/logs" 2>/dev/null || true
    sleep 1
}

detect_local_ip() {
    local ip=""
    # 1. K8s Downward API (recommended: spec.containers[].env with fieldRef status.hostIP)
    if [ -n "${NODE_IP:-}" ]; then ip="$NODE_IP"
    elif [ -n "${MY_NODE_IP:-}" ]; then ip="$MY_NODE_IP"
    elif [ -n "${HOST_IP:-}" ]; then ip="$HOST_IP"
    fi
    # 2. K8s API: query pod's status.hostIP
    if [ -z "$ip" ] && [ -f /var/run/secrets/kubernetes.io/serviceaccount/token ]; then
        local _token _ns _pod
        _token=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token 2>/dev/null)
        _ns=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null)
        _pod=$(hostname)
        if [ -n "$_token" ] && [ -n "$_ns" ] && [ -n "$_pod" ]; then
            ip=$(curl -sk --connect-timeout 2 \
                -H "Authorization: Bearer $_token" \
                "https://kubernetes.default.svc/api/v1/namespaces/$_ns/pods/$_pod" 2>/dev/null \
                | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',{}).get('hostIP',''))" 2>/dev/null)
        fi
    fi
    # 3. Fallback: hostname -I (returns Pod IP in K8s, host IP on bare metal)
    if [ -z "$ip" ]; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    echo "${ip:-127.0.0.1}"
}

detect_public_ip() {
    local ip=""
    ip=$(curl -s --connect-timeout 3 ifconfig.me 2>/dev/null) || \
    ip=$(curl -s --connect-timeout 3 icanhazip.com 2>/dev/null) || \
    ip=""
    echo "${ip:-}"
}

find_free_port() {
    local port=${1:-6006}
    local max_port=$((port + 100))
    while [ "$port" -lt "$max_port" ]; do
        if ! ss -tlnp 2>/dev/null | grep -q ":${port} " && \
           ! lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
    done
    echo "$1"
}

start_tensorboard() {
    local tb_logdir="$1"
    stop_tensorboard

    mkdir -p "$tb_logdir"

    local tb_already_running
    tb_already_running=$(ps aux | grep -E "tensorboard.*--logdir" | grep -v grep | wc -l)

    local tb_startup_log="$tb_logdir/tb_startup.log"
    > "$tb_startup_log"

    tb_msg() {
        echo "$1"
        echo "$1" >> "$tb_startup_log"
    }

    if [ "$tb_already_running" -gt 0 ]; then
        local existing_port
        existing_port=$(ps aux | grep -E "tensorboard.*--logdir" | grep -v grep | grep -oP -- '--port\s+\K[0-9]+' | head -1)
        existing_port=${existing_port:-6006}
        local local_ip; local_ip=$(detect_local_ip)
        local public_ip; public_ip=$(detect_public_ip)
        tb_msg "========================================"
        tb_msg "TensorBoard 已在运行"
        tb_msg "========================================"
        tb_msg "  本机: http://localhost:${existing_port}"
        tb_msg "  内网: http://${local_ip}:${existing_port}"
        [ -n "$public_ip" ] && tb_msg "  公网: http://${public_ip}:${existing_port}"
        tb_msg "  日志目录: $tb_logdir"
        tb_msg "========================================"
        log "TensorBoard 已在运行 (端口 $existing_port)，详见 $tb_startup_log"
        return
    fi

    local tb_port
    tb_port=$(find_free_port "${TB_PORT:-6006}")

    nohup tensorboard --logdir "$tb_logdir" --port "$tb_port" --bind_all \
        > "$tb_logdir/tensorboard.log" 2>&1 &
    TB_PID=$!

    sleep 2
    if kill -0 "$TB_PID" 2>/dev/null; then
        local local_ip; local_ip=$(detect_local_ip)
        local public_ip; public_ip=$(detect_public_ip)
        tb_msg "========================================"
        tb_msg "TensorBoard 已启动"
        tb_msg "========================================"
        tb_msg "  本机: http://localhost:${tb_port}"
        tb_msg "  内网: http://${local_ip}:${tb_port}"
        [ -n "$public_ip" ] && tb_msg "  公网: http://${public_ip}:${tb_port}"
        tb_msg "  日志目录: $tb_logdir"
        tb_msg "  PID: $TB_PID"
        local _pod_ip
        _pod_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        if [ -n "$_pod_ip" ] && [ "$_pod_ip" != "$local_ip" ]; then
            tb_msg "  Pod IP: http://${_pod_ip}:${tb_port} (集群内部)"
        fi
        if [ -f /proc/1/cgroup ] && grep -q "kubepods\|docker" /proc/1/cgroup 2>/dev/null; then
            if [ "$_pod_ip" = "$local_ip" ]; then
                tb_msg "  ⚠️  容器环境: 显示的IP为Pod IP, 外部可能无法访问"
                tb_msg "     设置 NODE_IP 环境变量或 YAML node_ip 字段指定宿主机IP"
            fi
        fi
        tb_msg "========================================"
        log "TensorBoard 已启动: http://${local_ip}:${tb_port} (PID=$TB_PID)"
        log "  详见: $tb_startup_log"
    else
        tb_msg "⚠️  TensorBoard 启动失败, 请手动启动:"
        tb_msg "  tensorboard --logdir $tb_logdir --port $tb_port"
        log "⚠️  TensorBoard 启动失败 (端口$tb_port)"
        TB_PID=""
    fi
}

# =============================================================================
# 日志函数
# =============================================================================

_DETECT_NODE_RANK() {
    # 1. SLURM
    [ -n "$SLURM_NODEID" ] && { echo "$SLURM_NODEID"; return; }
    # 2. 环境变量 NODE_RANK (PyTorch Operator, 手动设置)
    [ -n "$NODE_RANK" ] && { echo "$NODE_RANK"; return; }
    # 3. 常见平台环境变量
    [ -n "$RANK" ] && { echo "$RANK"; return; }
    [ -n "$OMPI_COMM_WORLD_RANK" ] && { echo "$OMPI_COMM_WORLD_RANK"; return; }
    # 4. hostname 模式
    local _hn
    _hn=$(hostname 2>/dev/null || echo "")
    # 4a. K8s: *-master-N / *-worker-N
    if echo "$_hn" | grep -qE -- '-master-[0-9]+$'; then
        echo "0"; return
    elif echo "$_hn" | grep -qE -- '-worker-[0-9]+$'; then
        echo "$_hn" | sed 's/.*worker-//' ; return
    fi
    # 4b. 通用: *-N (如 bev-0, node-3, gpu-12)
    if echo "$_hn" | grep -qE '^[a-zA-Z]+-[0-9]+$'; then
        echo "$_hn" | sed 's/.*-//' ; return
    fi
    # 5. YAML 配置
    local yaml_rank
    yaml_rank=$(echo "$CONFIG_JSON" | python3 -c "
import sys, json
c = json.load(sys.stdin)
d = c.get('defaults',{}).get('params',{}) or {}
e0 = (c.get('experiments') or [{}])[0].get('params',{}) or {}
r = e0.get('node_rank', d.get('node_rank'))
print(r if r is not None else '')
" 2>/dev/null)
    echo "${yaml_rank:-}"
}
_NODE_RANK=$(_DETECT_NODE_RANK)

_IS_MASTER=1
if [ -n "$_NODE_RANK" ] && [ "$_NODE_RANK" != "0" ]; then
    _IS_MASTER=0
fi

mkdir -p "$BATCH_LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
}

trap 'stop_tensorboard; log "批量训练被中断"; exit 130' INT TERM

# =============================================================================
# 构建训练命令
# =============================================================================

build_train_command() {
    local exp_idx=$1
    local config_json="$2"
    
    # 使用Python解析单个实验配置（支持defaults继承）
    python3 - "$config_json" "$exp_idx" <<'PYTHON_EOF'
import sys
import json

config = json.loads(sys.argv[1])
exp_idx = int(sys.argv[2])
exp = config['experiments'][exp_idx]

# 提取基础参数
dataset = exp.get('dataset', 'B26A')
version = exp.get('version', 'v1')

# 提取defaults（如果有，处理None值）
defaults = config.get('defaults', {})
default_env = defaults.get('env') or {}
default_params = defaults.get('params') or {}

# 合并环境变量（实验配置覆盖defaults，处理None值）
env_vars = {}
env_vars.update(default_env)  # 先应用defaults
env_vars.update(exp.get('env') or {})  # 实验配置覆盖

env_cmd = ""
for key, value in env_vars.items():
    if value is not None:
        env_cmd += f"{key}={value} "

# 合并训练参数（实验配置覆盖defaults，处理None值）
params = {}
params.update(default_params)  # 先应用defaults
params.update(exp.get('params') or {})  # 实验配置覆盖

# 构建start_training.sh参数
args = []

# angle_range_deg
if params.get('angle_range_deg') is not None:
    args.append(f"--angle {params['angle_range_deg']}")

# trans_range
if params.get('trans_range') is not None:
    args.append(f"--trans {params['trans_range']}")

# batch_size
if params.get('batch_size') is not None:
    args.append(f"--bs {params['batch_size']}")

# learning_rate
if params.get('learning_rate') is not None:
    args.append(f"--lr {params['learning_rate']}")

# use_ddp
if params.get('use_ddp', False):
    ddp_gpus = params.get('ddp_gpus')
    if ddp_gpus is not None:
        args.append(f"--ddp {ddp_gpus}")
    else:
        args.append("--ddp")

# use_compile
if params.get('use_compile', False):
    args.append("--compile")

# rotation_only
if params.get('rotation_only', False):
    args.append("--rotation_only")

# enable_axis_loss
if params.get('enable_axis_loss', False):
    args.append("--enable_axis_loss")

# weight_axis_rotation
if params.get('weight_axis_rotation') is not None:
    args.append(f"--weight_axis_rotation {params['weight_axis_rotation']}")

# generalization optimization params
OPTIM_PARAMS = [
    ('lr_schedule', '--lr_schedule'),
    ('warmup_epochs', '--warmup_epochs'),
    ('backbone_lr_scale', '--backbone_lr_scale'),
    ('cosine_T0', '--cosine_T0'),
    ('cosine_Tmult', '--cosine_Tmult'),
    ('drop_path_rate', '--drop_path_rate'),
    ('head_dropout', '--head_dropout'),
    ('perturb_distribution', '--perturb_distribution'),
    ('per_axis_prob', '--per_axis_prob'),
    ('per_axis_weights', '--per_axis_weights'),
    ('axis_weights', '--axis_weights'),
    ('augment_pc_jitter', '--augment_pc_jitter'),
    ('augment_pc_dropout', '--augment_pc_dropout'),
    ('augment_color_jitter', '--augment_color_jitter'),
    ('augment_intrinsic', '--augment_intrinsic'),
    ('eval_angle_range_deg', '--eval_angle'),
    ('early_stopping_patience', '--early_stopping_patience'),
    ('seed', '--seed'),
    ('pretrain_ckpt', '--pretrain_ckpt'),
    ('num_epochs', '--num_epochs'),
    ('use_geodesic_loss', '--use_geodesic_loss'),
    ('use_mlp_head', '--use_mlp_head'),
    ('use_deformable', '--use_deformable'),
    ('bev_pool_factor', '--bev_pool_factor'),
    ('use_foundation_depth', '--use_foundation_depth'),
    ('depth_model_type', '--depth_model_type'),
    ('fd_mode', '--fd_mode'),
    ('depth_sup_alpha', '--depth_sup_alpha'),
    ('max_frames_per_seq', '--max_frames_per_seq'),
    ('eval_epoches', '--eval_epoches'),
    ('grad_accum_steps', '--grad_accum_steps'),
]
for yaml_key, cli_flag in OPTIM_PARAMS:
    val = params.get(yaml_key)
    if val is not None and str(val).strip() != '':
        args.append(f"{cli_flag} {val}")

# 批量模式必须前台执行，否则 start_training.sh 会 nohup 后台启动并立即返回，
# 导致多个实验同时抢占 GPU。忽略 YAML 中的 foreground 设置。
args.append("--fg")

# no_tensorboard
if params.get('no_tensorboard', False):
    args.append("--no-tb")

# tensorboard_port
if params.get('tensorboard_port') is not None:
    args.append(f"--tb_port {params['tensorboard_port']}")

# nnodes
if params.get('nnodes') is not None:
    args.append(f"--nnodes {params['nnodes']}")

# node_rank
if params.get('node_rank') is not None:
    args.append(f"--node_rank {params['node_rank']}")

# master_addr
if params.get('master_addr') is not None:
    args.append(f"--master_addr {params['master_addr']}")

# master_port
if params.get('master_port') is not None:
    args.append(f"--master_port {params['master_port']}")

# 组合完整命令（BATCH_MODE=1 跳过交互式确认）
cmd = f"BATCH_MODE=1 {env_cmd}bash start_training.sh {dataset} {version} {' '.join(args)}"
print(cmd)
PYTHON_EOF
}

# =============================================================================
# 执行批量训练
# =============================================================================

log "================================================================"
log "BEVCalib 批量训练启动"
log "配置文件: $CONFIG_FILE"
log "实验总数: $TOTAL"
[ "$FORCE_RERUN" -eq 1 ] && log "强制模式: 已启用 (--force / YAML force_rerun)"
[ -n "$SKIP_PATTERN" ] && log "跳过模式: '$SKIP_PATTERN' (匹配的实验将被跳过)"
[ -n "$_NODE_RANK" ] && log "节点: node_rank=$_NODE_RANK $([ "$_IS_MASTER" -eq 1 ] && echo '(master)' || echo '(worker)')"
log "日志: 每个实验输出到 logs/<dataset>/model_*/train.log"
if [ "$DRY_RUN" -eq 1 ]; then
    log "模式: DRY-RUN（仅打印命令）"
fi
log "================================================================"

SKIPPED=0

for ((i=0; i<TOTAL; i++)); do
    EXP_NUM=$((i + 1))
    
    # 提取实验信息（使用Python解析，支持defaults继承）
    EXPERIMENT_INFO=$(python3 - "$CONFIG_JSON" "$i" <<'PYTHON_INFO'
import sys
import json

config = json.loads(sys.argv[1])
exp_idx = int(sys.argv[2])
exp = config['experiments'][exp_idx]

# 提取defaults（处理None值）
defaults = config.get('defaults', {})
default_params = defaults.get('params') or {}

# 合并params（处理None值）
params = {}
params.update(default_params)
params.update(exp.get('params') or {})

# 提取信息（处理None值）
name = exp.get('name', f'exp{exp_idx}')
desc = exp.get('description', '')
skip = bool(exp.get('skip', False))
skip_reason = exp.get('skip_reason', '')
dataset = exp.get('dataset', 'B26A')
version = exp.get('version', 'v1')
default_env = defaults.get('env') or {}
exp_env = exp.get('env') or {}
merged_env = {**default_env, **exp_env}
zstep = merged_env.get('BEV_ZBOUND_STEP', '')
angle = params.get('angle_range_deg', 5)  # 默认5度

# 映射dataset配置名到实际目录名（与start_training.sh一致）
DATASET_DIR_MAP = {'all': 'all_training_data', 'ALL': 'all_training_data'}
dataset_dir = DATASET_DIR_MAP.get(dataset, dataset)

# 输出为JSON（方便bash解析）
print(json.dumps({
    'name': name,
    'description': desc,
    'skip': skip,
    'skip_reason': skip_reason,
    'dataset': dataset,
    'dataset_dir': dataset_dir,
    'version': version,
    'zstep': zstep,
    'angle': angle
}))
PYTHON_INFO
)
    
    EXP_NAME=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
    EXP_DESC=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['description'])")
    EXP_SKIP=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print('1' if json.load(sys.stdin).get('skip') else '0')")
    EXP_SKIP_REASON=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('skip_reason',''))")
    DATASET=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['dataset'])")
    DATASET_DIR=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['dataset_dir'])")
    VERSION=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
    ZSTEP=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['zstep'])")
    ANGLE=$(echo "$EXPERIMENT_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['angle'])")
    
    log ""
    log "================================================================"
    log "实验 [$EXP_NUM/$TOTAL]: $EXP_NAME"
    if [ -n "$EXP_DESC" ]; then
        log "  描述: $EXP_DESC"
    fi
    log "  数据集: $DATASET"
    log "  版本: $VERSION"
    if [ -n "$ZSTEP" ]; then
        Z_VOXELS=$(python3 -c "print(int(20.0/float('$ZSTEP')))" 2>/dev/null || echo "?")
        log "  BEV_ZBOUND_STEP: $ZSTEP (${Z_VOXELS}个Z体素)"
    fi
    log "================================================================"
    
    # 检查实验级别的 skip 标志
    if [ "$EXP_SKIP" = "1" ]; then
        SKIPPED=$((SKIPPED + 1))
        log "⏭️  跳过实验 [$EXP_NUM/$TOTAL]: $EXP_NAME"
        if [ -n "$EXP_SKIP_REASON" ]; then
            log "  原因: $EXP_SKIP_REASON"
        else
            log "  原因: YAML 配置中 skip: true"
        fi
        continue
    fi
    
    # 构建训练命令
    CMD=$(build_train_command "$i" "$CONFIG_JSON")
    log "命令: $CMD"
    
    if [ "$DRY_RUN" -eq 1 ]; then
        log "[DRY-RUN] 跳过执行"
        continue
    fi
    
    # 推导具体实验的日志目录（与start_training.sh保持一致）
    # 格式: logs/{DATASET_DIR}/model_small_{angle}deg_{VERSION}
    # DATASET_DIR 已做名称映射（如 all -> all_training_data）
    ANGLE_INT=$(python3 -c "print(int(float('$ANGLE')))" 2>/dev/null || echo "5")
    EXPERIMENT_LOG_DIR="$SCRIPT_DIR/logs/$DATASET_DIR/model_small_${ANGLE_INT}deg_${VERSION}"
    
    # 检查是否匹配跳过模式
    if [ -n "$SKIP_PATTERN" ] && echo "$EXP_NAME" | grep -qE "$SKIP_PATTERN"; then
        SKIPPED=$((SKIPPED + 1))
        log "⏭️  跳过实验 [$EXP_NUM/$TOTAL]: $EXP_NAME"
        log "  原因: 名称匹配跳过模式 '$SKIP_PATTERN'"
        continue
    fi
    
    # 检查实验输出目录是否已存在（跳过已完成的实验）
    if [ "$FORCE_RERUN" -eq 0 ] && [ -d "$EXPERIMENT_LOG_DIR" ] && [ -f "$EXPERIMENT_LOG_DIR/train.log" ]; then
        SKIPPED=$((SKIPPED + 1))
        log "⏭️  跳过实验 [$EXP_NUM/$TOTAL]: $EXP_NAME"
        log "  原因: 输出目录已存在且包含训练日志"
        log "  路径: $EXPERIMENT_LOG_DIR"
        log "  如需重新训练，请删除该目录或使用 --force 参数"
        continue
    fi
    
    log "  TensorBoard日志目录: $EXPERIMENT_LOG_DIR"
    
    if [ "$_IS_MASTER" -eq 1 ]; then
        start_tensorboard "$EXPERIMENT_LOG_DIR"
    fi
    
    START_TIME=$(date +%s)
    
    NOISE_FILTER="Grad strides do not match bucket view strides|bucket_view\.sizes\(\)|grad\.sizes\(\) = \[|_execution_engine\.run_backward\(  # Calls"
    
    set +e
    yes y 2>/dev/null | eval "$CMD" 2>&1 | \
        grep --line-buffered -v -E "$NOISE_FILTER"
    EXIT_CODE=${PIPESTATUS[1]}
    set -e
    
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "实验 [$EXP_NUM/$TOTAL] 完成 ✓ (耗时: ${HOURS}h${MINS}m)"
    else
        log "实验 [$EXP_NUM/$TOTAL] 异常退出 (耗时: ${HOURS}h${MINS}m, exit=$EXIT_CODE)"
        log "⚠️  继续执行下一组实验..."
    fi
    
    # 等待GPU资源完全释放
    if [ $EXP_NUM -lt $TOTAL ]; then
        log "等待${WAIT_TIME}秒释放GPU资源..."
        sleep "$WAIT_TIME"
    fi
done

stop_tensorboard

log ""
log "================================================================"
log "批量训练全部完成"
log "  总实验数: $TOTAL"
if [ "$SKIPPED" -gt 0 ]; then
    log "  已跳过: $SKIPPED (输出目录已存在)"
    log "  实际执行: $((TOTAL - SKIPPED))"
fi
log "  配置文件: $CONFIG_FILE"
log "  日志位置: logs/<dataset>/model_*/train.log"
log "================================================================"
if [ "$_IS_MASTER" -eq 1 ]; then
    log ""
    log "下一步建议:"
    log "  1. 查看训练日志: tail -f logs/<dataset>/model_*/train.log"
    log "  2. TensorBoard查看: tensorboard --logdir logs/<dataset>/ --port 6006"
    log "  3. 分析性能: bash utils/scripts/quick_analyze.sh 5deg --skip-test"
fi
log ""
