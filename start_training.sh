#!/bin/bash
# =============================================================================
# BEVCalib 训练启动脚本
# 支持多数据集，自动组织日志结构，支持单机/多机 多GPU DDP并行
#
# ======================== 基本用法 ========================
#
#   bash start_training.sh <dataset> <version> [options]
#
# ======================== 数据集选项 ========================
#
#   B26A         - B26A 数据集 (b26a1_1_training_data, ~1544帧)
#   all          - 全量数据集 (all_training_data, ~405GB)
#   custom       - 自定义数据集 (需设置 CUSTOM_DATASET 环境变量)
#
# ======================== 可选参数 ========================
#
#   --ddp [N]              启用 DDP 多GPU并行 (N=本机GPU数, 默认自动检测)
#   --compile              启用 torch.compile 加速 (需 PyTorch>=2.1 + Python<=3.10)
#   --angle DEG            扰动角度范围 (默认: 5)
#   --trans M              扰动平移范围 (默认: 0.3)
#   --bs N                 batch size (默认: 16)
#   --lr LR                初始学习率 (默认: 使用模型默认值)
#   --fg                   前台阻塞执行 (不使用nohup, 输出到终端, Ctrl+C可停止)
#   --rotation_only        仅优化旋转, 不优化平移 (平移使用设计值)
#   --enable_axis_loss     启用分轴旋转损失 (Roll/Pitch/Yaw 独立监督)
#   --weight_axis_rotation W  分轴旋转损失权重 (默认: 0.3)
#   --lr_schedule TYPE     LR调度器: step 或 cosine_warm_restarts (默认: step)
#   --warmup_epochs N      线性warmup轮数 (默认: 5)
#   --backbone_lr_scale S  预训练backbone学习率倍率 (默认: 0.1)
#   --cosine_T0 N          CosineAnnealingWarmRestarts T_0 (默认: 50)
#   --cosine_Tmult N       CosineAnnealingWarmRestarts T_mult (默认: 2)
#   --drop_path_rate R     Stochastic Depth比率 (默认: 0.1)
#   --head_dropout R       预测头Dropout比率 (默认: 0.1)
#   --perturb_distribution D  扰动分布: uniform 或 truncated_normal (默认: uniform)
#   --per_axis_prob P      单轴扰动概率 (默认: 0.0)
#   --augment_pc_jitter S  点云抖动sigma (默认: 0.0)
#   --augment_pc_dropout R 点云随机丢弃比例 (默认: 0.0)
#   --augment_color_jitter S 图像色彩抖动强度 (默认: 0.0)
#   --augment_intrinsic S  相机内参随机扰动强度 (默认: 0.0, e.g. 0.05=±5%)
#   --augment_pitch_flip_prob P  GT pitch X轴随机扰动概率 (默认: 0.0=禁用)
#   --augment_pitch_flip_max_deg D  pitch扰动最大旋转角 (默认: 2.0°)
#   --augment_pitch_sign_flip_prob P  GT pitch符号翻转概率 (默认: 0.0=禁用, 精确反转pitch角)
#   --backbone_warmup_epochs N 骨干网络LR渐进解冻epoch数 (默认: 0=禁用)
#   --layer_wise_lr_decay F  SwinT层级LR衰减因子 (默认: 1.0=无衰减, 0.65=推荐)
#   --use_pitch_branch 0/1   启用双流Pitch分支 (默认: 0)
#   --pitch_aux_weight W     Pitch辅助损失权重 (默认: 0.3)
#   --sample_step N         采样步长 (每隔N帧取1帧, 与--max_frames_per_seq互斥)
#   --early_stopping_patience N 早停耐心值 (默认: 0=禁用)
#   --seed N               全局随机种子 (默认: 42)
#   --pretrain_ckpt PATH   预训练权重路径 (用于refine/finetune训练)
#   --num_epochs N         训练总epoch数 (默认: 400)
#   --force                强制重新训练，即使输出目录已存在
#   --no-tb                不自动启动 TensorBoard
#   --tb_port PORT         TensorBoard端口 (默认: 自动检测空闲端口, 起始6006)
#
#   多机训练参数 (需配合 --ddp 使用):
#   --nnodes [N]           总机器数 (默认自动检测: SLURM→环境变量→1)
#   --node_rank [R]        当前机器编号 (默认自动检测: SLURM→0)
#   --master_addr [ADDR]   master节点IP (默认自动检测: SLURM→本机IP)
#   --master_port [PORT]   master节点端口 (默认自动检测空闲端口, 起始29500)
#
# ======================== 运行模式 ========================
#
#   1. 经典模式 (默认): 每个GPU独立训练不同扰动设置
#      - GPU 0: 10deg/0.5m  |  GPU 1: 5deg/0.3m
#      - 适合同时探索多组超参数
#
#   2. DDP单机模式 (--ddp): 本机所有GPU协同加速
#      - 数据自动均分到各GPU, 梯度通过 NCCL AllReduce 同步
#      - N个GPU -> 约 N倍训练速度 (NVLink下接近线性)
#
#   3. DDP多机模式 (--ddp --nnodes N): 跨机器分布式训练
#      - 每台机器运行此脚本, 通过 --node_rank 区分角色
#      - master节点(rank=0) 负责 rendezvous 协调
#      - 支持SLURM集群自动检测: nnodes/node_rank/master_addr全自动
#
# ======================== 使用示例 ========================
#
#   # 经典模式: 两个GPU各跑不同扰动 (后台)
#   bash start_training.sh B26A v5
#
#   # 经典模式: 前台执行 (阻塞, 输出到终端)
#   bash start_training.sh B26A v5 --fg
#
#   # DDP单机: 自动检测所有GPU并行训练
#   bash start_training.sh B26A v5 --ddp
#
#   # DDP单机 + 前台执行
#   bash start_training.sh B26A v5 --ddp --fg
#
#   # DDP单机: 指定使用2个GPU
#   bash start_training.sh B26A v5 --ddp 2
#
#   # DDP + 自定义扰动参数
#   bash start_training.sh B26A v5 --ddp --angle 10 --trans 0.5
#
#   # DDP + 自定义学习率
#   bash start_training.sh B26A v5 --ddp --lr 0.0001
#
#   # DDP多机: 手动指定 (2台机器各2GPU)
#   # --- master (192.168.1.100) ---
#   bash start_training.sh B26A v5 --ddp 2 --nnodes 2 --node_rank 0 \
#       --master_addr 192.168.1.100
#   # --- worker ---
#   bash start_training.sh B26A v5 --ddp 2 --nnodes 2 --node_rank 1 \
#       --master_addr 192.168.1.100
#
#   # DDP多机: SLURM集群 (全自动检测)
#   bash start_training.sh B26A v5 --ddp --nnodes --fg
#
#   # 全量数据集DDP多机
#   bash start_training.sh all v1 --bs 24 --ddp --nnodes --fg --angle 10 --trans 0.5
#
#   # 自定义数据集
#   CUSTOM_DATASET=/path/to/my_data bash start_training.sh custom v1 --ddp
#
# ======================== 输出结构 ========================
#
#   logs/
#   └── <dataset_name>/
#       ├── model_small_5deg_v5/          # DDP模式
#       │   └── B26A_scratch/
#       │       ├── train.log
#       │       ├── events.out.tfevents.*  # TensorBoard
#       │       └── checkpoint/
#       ├── model_small_10deg_v5/         # 经典模式 GPU 0
#       └── model_small_5deg_v5/          # 经典模式 GPU 1
#
# ======================== 多机训练前置条件 ========================
#
#   1. 网络: 各机器间能互通 (最好有 InfiniBand/RoCE 高速互联)
#   2. 数据: 各机器能访问相同路径的训练数据 (NFS/共享存储)
#   3. 环境: 各机器 PyTorch/CUDA/NCCL 版本一致
#   4. 防火墙: master_port 需在各机器间开放
#   5. 每台机器的本机GPU数应保持一致
#
# ======================== 常用命令 ========================
#
#   # 查看训练日志
#   tail -f ./logs/B26A/model_small_5deg_v5/train.log
#
#   # 查看GPU状态
#   nvidia-smi
#
#   # 查看TensorBoard
#   tensorboard --logdir ./logs/B26A/ --port 6006
#
#   # 停止所有训练
#   bash stop_training.sh
#
# =============================================================================

set -e

# Quote CLI values that start with '-' (e.g. augment_lidar_vertical_fov=-25,15)
# Comma lists are safe in --flag=value form; only quote < > and other shell metacharacters.
_quote_cli_val() {
    case "$1" in
        *\'*)
            printf '%q' "$1"
            ;;
        -*|*'<'*|*'>'*|*'&'*|*'|'*|*';'*|*[[:space:]]*)
            printf "'%s'" "$1"
            ;;
        *)
            printf '%s' "$1"
            ;;
    esac
}

# argparse/shell: 负值或 P2b 列表参数必须用 --flag=value
_opt_flag_append() {
    local flag="$1"
    local val="$2"
    case "$flag" in
        --augment_lidar_sparse_lines|--augment_lidar_vertical_fov)
            OPTIM_ARGS="$OPTIM_ARGS ${flag}=${val}"
            ;;
        *)
            if [[ "$val" == -* ]]; then
                OPTIM_ARGS="$OPTIM_ARGS ${flag}=${val}"
            else
                OPTIM_ARGS="$OPTIM_ARGS ${flag} $(_quote_cli_val "$val")"
            fi
            ;;
    esac
}

# ============================================================================
# 工具函数
# ============================================================================

find_free_port() {
    local port=${1:-29500}
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
    return 1
}

detect_gpus_per_node() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --list-gpus 2>/dev/null | wc -l
    else
        echo "0"
    fi
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
    ip=$(curl -s --connect-timeout 3 ipinfo.io/ip 2>/dev/null) || \
    ip=""
    echo "${ip:-}"
}

# ============================================================================
# 保存平台环境变量 (Kubernetes/PyTorch Operator 可能预设)
# ============================================================================
_PLATFORM_MASTER_ADDR="${MASTER_ADDR:-}"
_PLATFORM_MASTER_PORT="${MASTER_PORT:-}"
_PLATFORM_WORLD_SIZE="${WORLD_SIZE:-}"
_PLATFORM_RANK="${RANK:-}"
_PLATFORM_NODE_RANK="${NODE_RANK:-}"

# ============================================================================
# 参数解析
# ============================================================================
DATASET_CHOICE=${1:-B26A}
VERSION=${2:-v1}
shift 2 2>/dev/null || true

USE_DDP=0
DDP_NGPUS=""
USE_COMPILE=""
DDP_ANGLE="5"
DDP_TRANS="0.3"
BATCH_SIZE="16"
LEARNING_RATE="1e-4"
FOREGROUND=0
ROTATION_ONLY=""
ENABLE_AXIS_LOSS=""
WEIGHT_AXIS_ROTATION=""
LR_SCHEDULE=""
WARMUP_EPOCHS=""
STEP_SIZE=""
BACKBONE_LR_SCALE=""
COSINE_T0=""
COSINE_TMULT=""
DROP_PATH_RATE=""
HEAD_DROPOUT=""
PERTURB_DISTRIBUTION=""
PER_AXIS_PROB=""
AUGMENT_PC_JITTER=""
AUGMENT_PC_DROPOUT=""
AUGMENT_COLOR_JITTER=""
AUGMENT_INTRINSIC=""
AUGMENT_INTRINSIC_CXCY=""
INTRINSIC_INPUT=""
BACKBONE_WARMUP_EPOCHS=""
LAYER_WISE_LR_DECAY=""
USE_PITCH_BRANCH=""
PITCH_AUX_WEIGHT=""
BEV_INSTANCE_NORM=""
AUGMENT_MOUNT_JITTER_PROB=""
AUGMENT_MOUNT_JITTER_ROT_SIGMA=""
AUGMENT_MOUNT_JITTER_TRANS_SIGMA=""
USE_CONTRASTIVE_EXTRINSIC=""
CONTRASTIVE_WEIGHT=""
EVAL_ANGLE=""
EARLY_STOPPING_PATIENCE=""
SEED=""
PRETRAIN_CKPT=""
NUM_EPOCHS=""
SAVE_CKPT_PER_EPOCHES=""
EVAL_EPOCHES=""
GRAD_ACCUM_STEPS=""
ENABLE_TB=1
TB_PORT=""
NNODES=""
NNODES_AUTO=0
NODE_RANK=""
NODE_RANK_AUTO=0
MASTER_ADDR=""
MASTER_ADDR_AUTO=0
MASTER_PORT=""
MASTER_PORT_AUTO=0
RDZV_TIMEOUT=""
PASSTHROUGH_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ddp)
            USE_DDP=1
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                DDP_NGPUS="$2"
                shift 2
            else
                shift
            fi
            ;;
        --compile)
            USE_COMPILE="--compile"
            shift
            ;;
        --angle)
            DDP_ANGLE="$2"
            shift 2
            ;;
        --trans)
            DDP_TRANS="$2"
            shift 2
            ;;
        --bs|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --fg|--foreground)
            FOREGROUND=1
            shift
            ;;
        --rotation_only)
            ROTATION_ONLY="--rotation_only"
            shift
            ;;
        --enable_axis_loss)
            ENABLE_AXIS_LOSS="--enable_axis_loss"
            shift
            ;;
        --weight_axis_rotation)
            WEIGHT_AXIS_ROTATION="$2"
            shift 2
            ;;
        --lr_schedule)
            LR_SCHEDULE="$2"; shift 2 ;;
        --warmup_epochs)
            WARMUP_EPOCHS="$2"; shift 2 ;;
        --step_size)
            STEP_SIZE="$2"; shift 2 ;;
        --backbone_lr_scale)
            BACKBONE_LR_SCALE="$2"; shift 2 ;;
        --cosine_T0)
            COSINE_T0="$2"; shift 2 ;;
        --cosine_Tmult)
            COSINE_TMULT="$2"; shift 2 ;;
        --drop_path_rate)
            DROP_PATH_RATE="$2"; shift 2 ;;
        --head_dropout)
            HEAD_DROPOUT="$2"; shift 2 ;;
        --perturb_distribution)
            PERTURB_DISTRIBUTION="$2"; shift 2 ;;
        --per_axis_prob)
            PER_AXIS_PROB="$2"; shift 2 ;;
        --per_axis_weights)
            PER_AXIS_WEIGHTS="$2"; shift 2 ;;
        --axis_weights)
            AXIS_WEIGHTS="$2"; shift 2 ;;
        --axis_weights=*)
            AXIS_WEIGHTS="${1#*=}"; shift ;;
        --augment_pc_jitter)
            AUGMENT_PC_JITTER="$2"; shift 2 ;;
        --augment_pc_dropout)
            AUGMENT_PC_DROPOUT="$2"; shift 2 ;;
        --augment_color_jitter)
            AUGMENT_COLOR_JITTER="$2"; shift 2 ;;
        --augment_intrinsic)
            AUGMENT_INTRINSIC="$2"; shift 2 ;;
        --augment_intrinsic_cxcy)
            AUGMENT_INTRINSIC_CXCY="$2"; shift 2 ;;
        --augment_pitch_flip_prob)
            AUGMENT_PITCH_FLIP_PROB="$2"; shift 2 ;;
        --augment_pitch_flip_max_deg)
            AUGMENT_PITCH_FLIP_MAX_DEG="$2"; shift 2 ;;
        --augment_pitch_sign_flip_prob)
            AUGMENT_PITCH_SIGN_FLIP_PROB="$2"; shift 2 ;;
        --backbone_warmup_epochs)
            BACKBONE_WARMUP_EPOCHS="$2"; shift 2 ;;
        --layer_wise_lr_decay)
            LAYER_WISE_LR_DECAY="$2"; shift 2 ;;
        --use_pitch_branch)
            USE_PITCH_BRANCH="$2"; shift 2 ;;
        --pitch_aux_weight)
            PITCH_AUX_WEIGHT="$2"; shift 2 ;;
        --bev_instance_norm)
            BEV_INSTANCE_NORM="$2"; shift 2 ;;
        --augment_mount_jitter_prob)
            AUGMENT_MOUNT_JITTER_PROB="$2"; shift 2 ;;
        --augment_mount_jitter_rot_sigma)
            AUGMENT_MOUNT_JITTER_ROT_SIGMA="$2"; shift 2 ;;
        --augment_mount_jitter_trans_sigma)
            AUGMENT_MOUNT_JITTER_TRANS_SIGMA="$2"; shift 2 ;;
        --use_contrastive_extrinsic)
            USE_CONTRASTIVE_EXTRINSIC="$2"; shift 2 ;;
        --contrastive_weight)
            CONTRASTIVE_WEIGHT="$2"; shift 2 ;;
        --domain_adversarial)
            DOMAIN_ADVERSARIAL="$2"; shift 2 ;;
        --domain_adversarial_weight)
            DOMAIN_ADVERSARIAL_WEIGHT="$2"; shift 2 ;;
        --num_domains)
            NUM_DOMAINS="$2"; shift 2 ;;
        --cam2bev_mode)
            CAM2BEV_MODE="$2"; shift 2 ;;
        --backbone_type)
            BACKBONE_TYPE="$2"; shift 2 ;;
        --backbone_variant)
            BACKBONE_VARIANT="$2"; shift 2 ;;
        --freeze_backbone)
            FREEZE_BACKBONE="$2"; shift 2 ;;
        --backbone_freeze_layers)
            BACKBONE_FREEZE_LAYERS="$2"; shift 2 ;;
        --backbone_weights)
            BACKBONE_WEIGHTS="$2"; shift 2 ;;
        --voxel_mode)
            VOXEL_MODE="$2"; shift 2 ;;
        --to_bev_mode)
            TO_BEV_MODE="$2"; shift 2 ;;
        --scatter_reduce)
            SCATTER_REDUCE="$2"; shift 2 ;;
        --fuser_type)
            FUSER_TYPE="$2"; shift 2 ;;
        --cam_drop_prob)
            CAM_DROP_PROB="$2"; shift 2 ;;
        --cam_drop_mode)
            CAM_DROP_MODE="$2"; shift 2 ;;
        --intrinsic_input)
            INTRINSIC_INPUT="--intrinsic_input"
            shift
            ;;
        --eval_angle)
            EVAL_ANGLE="$2"; shift 2 ;;
        --eval_trans_range)
            EVAL_TRANS_RANGE="$2"; shift 2 ;;
        --early_stopping_patience)
            EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --pretrain_ckpt)
            PRETRAIN_CKPT="$2"; shift 2 ;;
        --pretrain_ckpt_fallback)
            PRETRAIN_CKPT_FALLBACK="$2"; shift 2 ;;
        --resume_ckpt)
            RESUME_CKPT="$2"; shift 2 ;;
        --no_amp)
            NO_AMP="$2"; shift 2 ;;
        --amp_bf16)
            AMP_BF16="$2"; shift 2 ;;
        --num_epochs)
            NUM_EPOCHS="$2"; shift 2 ;;
        --save_ckpt_per_epoches)
            SAVE_CKPT_PER_EPOCHES="$2"; shift 2 ;;
        --use_geodesic_loss)
            USE_GEODESIC_LOSS="$2"; shift 2 ;;
        --use_balanced_axis_loss)
            USE_BALANCED_AXIS_LOSS="$2"; shift 2 ;;
        --use_mlp_head)
            USE_MLP_HEAD="$2"; shift 2 ;;
        --use_deformable)
            USE_DEFORMABLE="$2"; shift 2 ;;
        --bev_pool_factor)
            BEV_POOL_FACTOR="$2"; shift 2 ;;
        --use_foundation_depth)
            USE_FOUNDATION_DEPTH="$2"; shift 2 ;;
        --fd_mode)
            FD_MODE="$2"; shift 2 ;;
        --depth_sup_alpha)
            DEPTH_SUP_ALPHA="$2"; shift 2 ;;
        --depth_model_type)
            DEPTH_MODEL_TYPE="$2"; shift 2 ;;
        --max_frames_per_seq)
            MAX_FRAMES_PER_SEQ="$2"; shift 2 ;;
        --sample_step)
            SAMPLE_STEP="$2"; shift 2 ;;
        --pose_aware_sampling)
            POSE_AWARE_SAMPLING="--pose_aware_sampling"
            shift
            ;;
        --poses_dir)
            POSES_DIR="$2"; shift 2 ;;
        --eval_epoches)
            EVAL_EPOCHES="$2"; shift 2 ;;
        --grad_accum_steps)
            GRAD_ACCUM_STEPS="$2"; shift 2 ;;
        --enable_vis)
            ENABLE_VIS="$2"; shift 2 ;;
        --vis_freq)
            VIS_FREQ="$2"; shift 2 ;;
        --validate_data)
            VALIDATE_DATA="$2"; shift 2 ;;
        --enable_ckpt_eval)
            ENABLE_CKPT_EVAL="$2"; shift 2 ;;
        --enable_medw_eval)
            ENABLE_MEDW_EVAL="$2"; shift 2 ;;
        --enable_jacobian_eval)
            ENABLE_JACOBIAN_EVAL="$2"; shift 2 ;;
        --jacobian_eval_angle_deg)
            JACOBIAN_EVAL_ANGLE_DEG="$2"; shift 2 ;;
        --jacobian_eval_batches)
            JACOBIAN_EVAL_BATCHES="$2"; shift 2 ;;
        --jacobian_eval_n_probes)
            JACOBIAN_EVAL_N_PROBES="$2"; shift 2 ;;
        --jacobian_loss_weight)
            JACOBIAN_LOSS_WEIGHT="$2"; shift 2 ;;
        --jacobian_loss_start_epoch)
            JACOBIAN_LOSS_START_EPOCH="$2"; shift 2 ;;
        --jacobian_loss_probe_deg)
            JACOBIAN_LOSS_PROBE_DEG="$2"; shift 2 ;;
        --jacobian_loss_interval)
            JACOBIAN_LOSS_INTERVAL="$2"; shift 2 ;;
        --medw_eval_max_frames)
            MEDW_EVAL_MAX_FRAMES="$2"; shift 2 ;;
        --enable_dual_gate_ckpt)
            ENABLE_DUAL_GATE_CKPT="$2"; shift 2 ;;
        --dual_gate_jacobian_min)
            DUAL_GATE_JACOBIAN_MIN="$2"; shift 2 ;;
        --dual_gate_medw_max)
            DUAL_GATE_MEDW_MAX="$2"; shift 2 ;;
        --dual_gate_recovery_min)
            DUAL_GATE_RECOVERY_MIN="$2"; shift 2 ;;
        --dual_gate_zd_max)
            DUAL_GATE_ZD_MAX="$2"; shift 2 ;;
        --enable_recovery_gate_ckpt)
            ENABLE_RECOVERY_GATE_CKPT="$2"; shift 2 ;;
        --enable_zd_gate_ckpt)
            ENABLE_ZD_GATE_CKPT="$2"; shift 2 ;;
        --dual_gate_inject_recovery_min)
            DUAL_GATE_INJECT_RECOVERY_MIN="$2"; shift 2 ;;
        --dual_gate_pred_indep_max)
            DUAL_GATE_PRED_INDEP_MAX="$2"; shift 2 ;;
        --enable_inject_recovery_eval)
            ENABLE_INJECT_RECOVERY_EVAL="$2"; shift 2 ;;
        --inject_recovery_eval_deg)
            INJECT_RECOVERY_EVAL_DEG="$2"; shift 2 ;;
        --inject_recovery_eval_batches)
            INJECT_RECOVERY_EVAL_BATCHES="$2"; shift 2 ;;
        --vis_samples)
            VIS_SAMPLES="$2"; shift 2 ;;
        --vis_points)
            VIS_POINTS="$2"; shift 2 ;;
        --vis_point_radius)
            VIS_POINT_RADIUS="$2"; shift 2 ;;
        --validate_sample_ratio)
            VALIDATE_SAMPLE_RATIO="$2"; shift 2 ;;
        --min_point_utilization)
            MIN_POINT_UTILIZATION="$2"; shift 2 ;;
        --min_valid_ratio)
            MIN_VALID_RATIO="$2"; shift 2 ;;
        --ddp_auto_scale)
            DDP_AUTO_SCALE="$2"; shift 2 ;;
        --ddp_reference_gpus)
            DDP_REFERENCE_GPUS="$2"; shift 2 ;;
        --max_scaled_lr)
            MAX_SCALED_LR="$2"; shift 2 ;;
        --data_balance)
            DATA_BALANCE="$2"; shift 2 ;;
        --seq_weight_overrides)
            SEQ_WEIGHT_OVERRIDES="$2"; shift 2 ;;
        --wd)
            WEIGHT_DECAY="$2"; shift 2 ;;
        --target_width)
            TARGET_WIDTH="$2"; shift 2 ;;
        --target_height)
            TARGET_HEIGHT="$2"; shift 2 ;;
        --tinit_dropout_prob)
            TINIT_DROPOUT_PROB="$2"; shift 2 ;;
        --consistency_loss_weight)
            CONSISTENCY_LOSS_WEIGHT="$2"; shift 2 ;;
        --consistency_loss_start_epoch)
            CONSISTENCY_LOSS_START_EPOCH="$2"; shift 2 ;;
        --progressive_angle_start)
            PROGRESSIVE_ANGLE_START="$2"; shift 2 ;;
        --progressive_angle_end)
            PROGRESSIVE_ANGLE_END="$2"; shift 2 ;;
        --progressive_warmup_epochs)
            PROGRESSIVE_WARMUP_EPOCHS="$2"; shift 2 ;;
        --ema_consistency)
            EMA_CONSISTENCY="$2"; shift 2 ;;
        --ema_decay)
            EMA_DECAY="$2"; shift 2 ;;
        --continuous_tinit_noise)
            CONTINUOUS_TINIT_NOISE="$2"; shift 2 ;;
        --continuous_noise_max_deg)
            CONTINUOUS_NOISE_MAX_DEG="$2"; shift 2 ;;
        --zero_perturbation_prob)
            ZERO_PERTURBATION_PROB="$2"; shift 2 ;;
        --zero_drift_loss_weight)
            ZERO_DRIFT_LOSS_WEIGHT="$2"; shift 2 ;;
        --zero_drift_loss_weight_start)
            ZERO_DRIFT_LOSS_WEIGHT_START="$2"; shift 2 ;;
        --zero_drift_loss_ramp_epochs)
            ZERO_DRIFT_LOSS_RAMP_EPOCHS="$2"; shift 2 ;;
        --zero_drift_loss_start_epoch)
            ZERO_DRIFT_LOSS_START_EPOCH="$2"; shift 2 ;;
        --zero_drift_dedicated_ratio)
            ZERO_DRIFT_DEDICATED_RATIO="$2"; shift 2 ;;
        --enable_jacobian_gate_ckpt)
            ENABLE_JACOBIAN_GATE_CKPT="$2"; shift 2 ;;
        --jacobian_early_stop_min)
            JACOBIAN_EARLY_STOP_MIN="$2"; shift 2 ;;
        --jacobian_early_stop_patience)
            JACOBIAN_EARLY_STOP_PATIENCE="$2"; shift 2 ;;
        --inject_recovery_loss_weight)
            INJECT_RECOVERY_LOSS_WEIGHT="$2"; shift 2 ;;
        --inject_recovery_loss_start_epoch)
            INJECT_RECOVERY_LOSS_START_EPOCH="$2"; shift 2 ;;
        --inject_recovery_magnitude_deg)
            INJECT_RECOVERY_MAGNITUDE_DEG="$2"; shift 2 ;;
        --inject_recovery_dedicated_ratio)
            INJECT_RECOVERY_DEDICATED_RATIO="$2"; shift 2 ;;
        --use_mgda)
            USE_MGDA="$2"; shift 2 ;;
        --mgda_start_epoch)
            MGDA_START_EPOCH="$2"; shift 2 ;;
        --mgda_include_inject)
            MGDA_INCLUDE_INJECT="$2"; shift 2 ;;
        --mgda_include_photo)
            MGDA_INCLUDE_PHOTO="$2"; shift 2 ;;
        --use_lsp_loss)
            USE_LSP_LOSS="$2"; shift 2 ;;
        --lsp_weight)
            LSP_WEIGHT="$2"; shift 2 ;;
        --lsp_weight_start)
            LSP_WEIGHT_START="$2"; shift 2 ;;
        --lsp_ramp_epochs)
            LSP_RAMP_EPOCHS="$2"; shift 2 ;;
        --lsp_start_epoch)
            LSP_START_EPOCH="$2"; shift 2 ;;
        --lsp_lambda_ssim)
            LSP_LAMBDA_SSIM="$2"; shift 2 ;;
        --lsp_max_points)
            LSP_MAX_POINTS="$2"; shift 2 ;;
        --lsp_downsample)
            LSP_DOWNSAMPLE="$2"; shift 2 ;;
        --lsp_loss_clip)
            LSP_LOSS_CLIP="$2"; shift 2 ;;
        --lsp_min_valid_ratio)
            LSP_MIN_VALID_RATIO="$2"; shift 2 ;;
        --rig_consistency_weight)
            RIG_CONSISTENCY_WEIGHT="$2"; shift 2 ;;
        --rig_consistency_start_epoch)
            RIG_CONSISTENCY_START_EPOCH="$2"; shift 2 ;;
        --pose_release_epoch)
            POSE_RELEASE_EPOCH="$2"; shift 2 ;;
        --pose_release_joint_epoch)
            POSE_RELEASE_JOINT_EPOCH="$2"; shift 2 ;;
        --pose_release_corr_lr_scale_joint)
            POSE_RELEASE_CORR_LR_SCALE_JOINT="$2"; shift 2 ;;
        --freeze_backbone_epoch)
            FREEZE_BACKBONE_EPOCH="$2"; shift 2 ;;
        --use_dp_head)
            USE_DP_HEAD="$2"; shift 2 ;;
        --dp_gate_deg)
            DP_GATE_DEG="$2"; shift 2 ;;
        --route_loss_weight)
            ROUTE_LOSS_WEIGHT="$2"; shift 2 ;;
        --route_zd_penalty_weight)
            ROUTE_ZD_PENALTY_WEIGHT="$2"; shift 2 ;;
        --use_jacg)
            USE_JACG="$2"; shift 2 ;;
        --jacg_hidden_dim)
            JACG_HIDDEN_DIM="$2"; shift 2 ;;
        --bias_path_in_norm)
            BIAS_PATH_IN_NORM="$2"; shift 2 ;;
        --recovery_path_layer_norm)
            RECOVERY_PATH_LAYER_NORM="$2"; shift 2 ;;
        --use_hard_route_eval)
            USE_HARD_ROUTE_EVAL="$2"; shift 2 ;;
        --use_adir)
            USE_ADIR="$2"; shift 2 ;;
        --adir_steps)
            ADIR_STEPS="$2"; shift 2 ;;
        --adir_max_step_deg)
            ADIR_MAX_STEP_DEG="$2"; shift 2 ;;
        --use_gated_instance_norm)
            USE_GATED_INSTANCE_NORM="$2"; shift 2 ;;
        --gin_init_gate)
            GIN_INIT_GATE="$2"; shift 2 ;;
        --gin_channels)
            GIN_CHANNELS="$2"; shift 2 ;;
        --pitch_vertical_bands)
            PITCH_VERTICAL_BANDS="$2"; shift 2 ;;
        --gin_gate_reg_target)
            GIN_GATE_REG_TARGET="$2"; shift 2 ;;
        --gin_gate_reg_weight)
            GIN_GATE_REG_WEIGHT="$2"; shift 2 ;;
        --multi_scale_perturb)
            MULTI_SCALE_PERTURB="$2"; shift 2 ;;
        --correlation_fusion)
            CORRELATION_FUSION="$2"; shift 2 ;;
        --cross_correlation_fusion)
            CROSS_CORRELATION_FUSION="$2"; shift 2 ;;
        --explicit_tinit)
            EXPLICIT_TINIT="$2"; shift 2 ;;
        --tinit_bev_film)
            TINIT_BEV_FILM="$2"; shift 2 ;;
        --tinit_query_film)
            TINIT_QUERY_FILM="$2"; shift 2 ;;
        --tinit_sensitivity_weight)
            TINIT_SENSITIVITY_WEIGHT="$2"; shift 2 ;;
        --iterative_refine)
            ITERATIVE_REFINE="$2"; shift 2 ;;
        --iterative_refine_weight)
            ITERATIVE_REFINE_WEIGHT="$2"; shift 2 ;;
        --iterative_refine_start_epoch)
            ITERATIVE_REFINE_START_EPOCH="$2"; shift 2 ;;
        --iterative_refine_use_synthetic_init)
            ITERATIVE_REFINE_USE_SYNTHETIC_INIT="$2"; shift 2 ;;
        --iterative_refine_synthetic_residual_deg)
            ITERATIVE_REFINE_SYNTHETIC_RESIDUAL_DEG="$2"; shift 2 ;;
        --native_cross)
            NATIVE_CROSS="$2"; shift 2 ;;
        --native_cross_pc_groups)
            NATIVE_CROSS_PC_GROUPS="$2"; shift 2 ;;
        --native_cross_n_harmonic)
            NATIVE_CROSS_N_HARMONIC="$2"; shift 2 ;;
        --native_cross_n_layers)
            NATIVE_CROSS_N_LAYERS="$2"; shift 2 ;;
        --native_cross_dual_branch)
            NATIVE_CROSS_DUAL_BRANCH="$2"; shift 2 ;;
        --native_cross_knn)
            NATIVE_CROSS_KNN="$2"; shift 2 ;;
        --native_cross_use_fps)
            NATIVE_CROSS_USE_FPS="$2"; shift 2 ;;
        --native_cross_use_pointgpt)
            NATIVE_CROSS_USE_POINTGPT="$2"; shift 2 ;;
        --native_cross_pointgpt_ckpt)
            NATIVE_CROSS_POINTGPT_CKPT="$2"; shift 2 ;;
        --native_cross_pointgpt_config)
            NATIVE_CROSS_POINTGPT_CONFIG="$2"; shift 2 ;;
        --native_cross_pointgpt_max_depth)
            NATIVE_CROSS_POINTGPT_MAX_DEPTH="$2"; shift 2 ;;
        --native_cross_extend_ratio)
            NATIVE_CROSS_EXTEND_RATIO="$2"; shift 2 ;;
        --fusion_backend)
            FUSION_BACKEND="$2"; shift 2 ;;
        --pc_encoder_mode)
            PC_ENCODER_MODE="$2"; shift 2 ;;
        --fusion_variant)
            FUSION_VARIANT="$2"; shift 2 ;;
        --deep_supervision_weight)
            DEEP_SUPERVISION_WEIGHT="$2"; shift 2 ;;
        --gate_entropy_weight)
            GATE_ENTROPY_WEIGHT="$2"; shift 2 ;;
        --appearance_loss_weight)
            APPEARANCE_LOSS_WEIGHT="$2"; shift 2 ;;
        --depth_loss_weight)
            DEPTH_LOSS_WEIGHT="$2"; shift 2 ;;
        --geo_loss_start_epoch)
            GEO_LOSS_START_EPOCH="$2"; shift 2 ;;
        --use_match_head)
            USE_MATCH_HEAD="$2"; shift 2 ;;
        --use_local_correlation)
            USE_LOCAL_CORRELATION="$2"; shift 2 ;;
        --correspondence_loss_weight)
            CORRESPONDENCE_LOSS_WEIGHT="$2"; shift 2 ;;
        --correspondence_loss_start_epoch)
            CORRESPONDENCE_LOSS_START_EPOCH="$2"; shift 2 ;;
        --correspondence_loss_warmup_epochs)
            CORRESPONDENCE_LOSS_WARMUP_EPOCHS="$2"; shift 2 ;;
        --match_disable_fallback)
            MATCH_DISABLE_FALLBACK="$2"; shift 2 ;;
        --match_gate_use_init_ratio)
            MATCH_GATE_USE_INIT_RATIO="$2"; shift 2 ;;
        --match_confidence_threshold)
            MATCH_CONFIDENCE_THRESHOLD="$2"; shift 2 ;;
        --match_corr_validity_mode)
            MATCH_CORR_VALIDITY_MODE="$2"; shift 2 ;;
        --match_epnp_min_points)
            MATCH_EPNP_MIN_POINTS="$2"; shift 2 ;;
        --match_phase_noise_max_deg)
            MATCH_PHASE_NOISE_MAX_DEG="$2"; shift 2 ;;
        --compose_mode)
            COMPOSE_MODE="$2"; shift 2 ;;
        --num_correspondences)
            NUM_CORRESPONDENCES="$2"; shift 2 ;;
        --match_valid_ratio_min)
            MATCH_VALID_RATIO_MIN="$2"; shift 2 ;;
        --correspondence_supervision)
            CORRESPONDENCE_SUPERVISION="$2"; shift 2 ;;
        --differentiable_epnp)
            DIFFERENTIABLE_EPNP="$2"; shift 2 ;;
        --diff_epnp_warmup_epochs)
            DIFF_EPNP_WARMUP_EPOCHS="$2"; shift 2 ;;
        --bev_branch_lr_scale)
            BEV_BRANCH_LR_SCALE="$2"; shift 2 ;;
        --projfusion_image_hw)
            if [[ $# -ge 3 ]]; then
                PROJFUSION_IMAGE_HW="$2 $3"; shift 3
            else
                echo "❌ Error: --projfusion_image_hw requires 2 values (H W)"
                exit 1
            fi ;;
        --augment_fov_crop_prob)
            AUGMENT_FOV_CROP_PROB="$2"; shift 2 ;;
        --augment_fov_crop_ratio_min)
            AUGMENT_FOV_CROP_RATIO_MIN="$2"; shift 2 ;;
        --augment_fov_crop_ratio_max)
            AUGMENT_FOV_CROP_RATIO_MAX="$2"; shift 2 ;;
        --augment_lidar_sparse_prob)
            AUGMENT_LIDAR_SPARSE_PROB="$2"; shift 2 ;;
        --augment_lidar_sparse_lines)
            AUGMENT_LIDAR_SPARSE_LINES="$2"; shift 2 ;;
        --augment_lidar_sparse_lines=*)
            AUGMENT_LIDAR_SPARSE_LINES="${1#*=}"; shift ;;
        --augment_lidar_vertical_fov)
            AUGMENT_LIDAR_VERTICAL_FOV="$2"; shift 2 ;;
        --augment_lidar_vertical_fov=*)
            AUGMENT_LIDAR_VERTICAL_FOV="${1#*=}"; shift ;;
        --force)
            export FORCE_RERUN=1
            shift
            ;;
        --no-tb|--no-tensorboard)
            ENABLE_TB=0
            shift
            ;;
        --tb_port)
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                TB_PORT="$2"
                shift 2
            else
                shift
            fi
            ;;
        --nnodes)
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                NNODES="$2"
                shift 2
            else
                NNODES_AUTO=1
                shift
            fi
            ;;
        --node_rank)
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                NODE_RANK="$2"
                shift 2
            else
                NODE_RANK_AUTO=1
                shift
            fi
            ;;
        --master_addr)
            if [[ $# -ge 2 && "$2" != --* ]]; then
                MASTER_ADDR="$2"
                shift 2
            else
                MASTER_ADDR_AUTO=1
                shift
            fi
            ;;
        --master_port)
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                MASTER_PORT="$2"
                shift 2
            else
                MASTER_PORT_AUTO=1
                shift
            fi
            ;;
        --rdzv_timeout)
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                RDZV_TIMEOUT="$2"
                shift 2
            else
                shift
            fi
            ;;
        *)
            if [[ "$1" == --* ]]; then
                if [[ $# -ge 2 && "$2" != --* ]]; then
                    PASSTHROUGH_ARGS="$PASSTHROUGH_ARGS $1 $(_quote_cli_val "$2")"
                    shift 2
                elif [[ "$1" == *=* ]]; then
                    _pt_val="${1#*=}"
                    if [[ "$_pt_val" == *'<'* || "$_pt_val" == *'>'* ]]; then
                        PASSTHROUGH_ARGS="$PASSTHROUGH_ARGS ${1%%=*}=$(_quote_cli_val "$_pt_val")"
                    else
                        PASSTHROUGH_ARGS="$PASSTHROUGH_ARGS $1"
                    fi
                    shift
                else
                    PASSTHROUGH_ARGS="$PASSTHROUGH_ARGS $1"
                    shift
                fi
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

# ============================================================================
# 自动检测: GPU / 集群环境 / 端口
# ============================================================================

# GPU数量自动检测
if [ "$USE_DDP" -eq 1 ] && [ -z "$DDP_NGPUS" ]; then
    DDP_NGPUS=$(detect_gpus_per_node)
    echo "ℹ️  GPU自动检测: ${DDP_NGPUS} GPUs"
    if [ "$DDP_NGPUS" -lt 1 ]; then
        echo "❌ 错误: 未检测到GPU"
        exit 1
    fi
fi

# SLURM集群自动检测
CLUSTER_DETECTED=""
if [ -n "$SLURM_JOB_ID" ]; then
    CLUSTER_DETECTED="SLURM"
    echo "ℹ️  检测到 SLURM 集群环境 (Job: $SLURM_JOB_ID)"

    if [ -z "$NNODES" ] && [ "$NNODES_AUTO" -eq 1 ]; then
        NNODES=${SLURM_NNODES:-1}
        echo "    SLURM nnodes=$NNODES"
    fi
    if [ -z "$NODE_RANK" ] && [ "$NODE_RANK_AUTO" -eq 1 ]; then
        NODE_RANK=${SLURM_NODEID:-0}
        echo "    SLURM node_rank=$NODE_RANK"
    fi
    if [ -z "$MASTER_ADDR" ] && [ "$MASTER_ADDR_AUTO" -eq 1 ]; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n 1)
        if [ -n "$MASTER_ADDR" ]; then
            echo "    SLURM master_addr=$MASTER_ADDR"
        fi
    fi
    if [ -z "$DDP_NGPUS" ]; then
        DDP_NGPUS=${SLURM_GPUS_ON_NODE:-$(detect_gpus_per_node)}
        echo "    SLURM gpus_per_node=$DDP_NGPUS"
    fi

elif [ -n "$PBS_JOBID" ]; then
    CLUSTER_DETECTED="PBS"
    echo "ℹ️  检测到 PBS/Torque 集群环境"

    if [ -z "$NNODES" ] && [ "$NNODES_AUTO" -eq 1 ]; then
        NNODES=$(wc -l < "$PBS_NODEFILE" 2>/dev/null | awk '{print int($1)}' || echo "1")
    fi
    if [ -z "$NODE_RANK" ] && [ "$NODE_RANK_AUTO" -eq 1 ]; then
        NODE_RANK=0
    fi
    if [ -z "$MASTER_ADDR" ] && [ "$MASTER_ADDR_AUTO" -eq 1 ]; then
        MASTER_ADDR=$(head -1 "$PBS_NODEFILE" 2>/dev/null || detect_local_ip)
    fi
fi

# Kubernetes / Cloud 自动检测
# 支持 hostname 模式: *-master-N / *-worker-N (常见于 K8s PyTorch Operator, 云训练平台)
if [ -z "$CLUSTER_DETECTED" ]; then
    K8S_HOSTNAME=$(hostname 2>/dev/null || echo "")

    # 方式1: PyTorch Operator 环境变量 (MASTER_ADDR, RANK, WORLD_SIZE)
    if [ -n "$_PLATFORM_MASTER_ADDR" ] && [ -n "$_PLATFORM_RANK" ]; then
        CLUSTER_DETECTED="K8s-Operator"
        echo "ℹ️  检测到 PyTorch Operator 环境变量 (hostname: $K8S_HOSTNAME)"
        if [ -z "$MASTER_ADDR" ]; then
            MASTER_ADDR="$_PLATFORM_MASTER_ADDR"
            echo "    master_addr=$MASTER_ADDR (env)"
        fi
        if [ -z "$MASTER_PORT" ] && [ -n "$_PLATFORM_MASTER_PORT" ]; then
            MASTER_PORT="$_PLATFORM_MASTER_PORT"
            echo "    master_port=$MASTER_PORT (env)"
        fi
        if [ -z "$NODE_RANK" ]; then
            NODE_RANK="$_PLATFORM_NODE_RANK"
            [ -z "$NODE_RANK" ] && NODE_RANK="$_PLATFORM_RANK"
            echo "    node_rank=$NODE_RANK (env)"
        fi
        if [ -n "$_PLATFORM_WORLD_SIZE" ] && [ -z "$NNODES" ]; then
            PLATFORM_WS=$_PLATFORM_WORLD_SIZE
            DETECTED_GPUS=$(detect_gpus_per_node)
            if [ "$DETECTED_GPUS" -gt 0 ]; then
                NNODES=$((PLATFORM_WS / DETECTED_GPUS))
                [ "$NNODES" -lt 1 ] && NNODES=1
            else
                NNODES=$PLATFORM_WS
            fi
            echo "    nnodes=$NNODES (WORLD_SIZE=$PLATFORM_WS / ${DETECTED_GPUS}gpus)"
        fi

    # 方式2: hostname 模式 (*-master-N / *-worker-N)
    elif echo "$K8S_HOSTNAME" | grep -qE -- '-(master|worker)-[0-9]+$'; then
        CLUSTER_DETECTED="Kubernetes"
        echo "ℹ️  检测到 Kubernetes/Cloud 环境 (hostname: $K8S_HOSTNAME)"

        if echo "$K8S_HOSTNAME" | grep -qE -- '-master-[0-9]+$'; then
            if [ -z "$NODE_RANK" ]; then
                NODE_RANK=0
                echo "    角色: master, node_rank=0"
            fi
            if [ -z "$MASTER_ADDR" ]; then
                MASTER_ADDR=$(detect_local_ip)
                echo "    master_addr=$MASTER_ADDR (本机IP)"
            fi
        elif echo "$K8S_HOSTNAME" | grep -qE -- '-worker-[0-9]+$'; then
            K8S_WORKER_NUM=$(echo "$K8S_HOSTNAME" | sed 's/.*worker-//')
            if [ -z "$NODE_RANK" ]; then
                NODE_RANK=$((K8S_WORKER_NUM + 1))
                echo "    角色: worker-${K8S_WORKER_NUM}, node_rank=$NODE_RANK"
            fi
            if [ -z "$MASTER_ADDR" ]; then
                MASTER_HOSTNAME=$(echo "$K8S_HOSTNAME" | sed 's/worker-[0-9]*/master-0/')
                RESOLVED_IP=$(getent hosts "$MASTER_HOSTNAME" 2>/dev/null | awk '{print $1}' | head -1)
                if [ -z "$RESOLVED_IP" ]; then
                    RESOLVED_IP=$(ping -c1 -W3 "$MASTER_HOSTNAME" 2>/dev/null | head -1 | grep -oP '\(\K[0-9.]+')
                fi
                if [ -n "$RESOLVED_IP" ]; then
                    MASTER_ADDR="$RESOLVED_IP"
                    echo "    master_addr=$MASTER_ADDR (DNS: $MASTER_HOSTNAME)"
                else
                    echo "    ⚠️  无法解析Master hostname: $MASTER_HOSTNAME"
                    echo "    请手动指定: --master_addr <master_ip>"
                fi
            fi
        fi
    fi
fi

# 通用 hostname 模式: *-N (如 bev-0, node-3, gpu-12)
if [ -z "$CLUSTER_DETECTED" ] && [ -z "$NODE_RANK" ]; then
    _GEN_HOSTNAME=$(hostname 2>/dev/null || echo "")
    if echo "$_GEN_HOSTNAME" | grep -qE '^[a-zA-Z]+-[0-9]+$'; then
        _GEN_RANK=$(echo "$_GEN_HOSTNAME" | sed 's/.*-//')
        if [ -n "$_GEN_RANK" ]; then
            NODE_RANK="$_GEN_RANK"
            CLUSTER_DETECTED="hostname"
            echo "ℹ️  从 hostname 检测到 node_rank=$NODE_RANK (hostname: $_GEN_HOSTNAME)"
        fi
    fi
fi

# 填充未设置的默认值
[ -z "$NNODES" ] && NNODES="1"
[ -z "$NODE_RANK" ] && NODE_RANK="0"

# 多机参数自动检测和校验
if [ "$NNODES" -gt 1 ]; then
    if [ "$USE_DDP" -eq 0 ]; then
        USE_DDP=1
        echo "ℹ️  多机模式自动启用 DDP"
    fi
    if [ -z "$DDP_NGPUS" ]; then
        DDP_NGPUS=$(detect_gpus_per_node)
    fi
    if [ -z "$MASTER_ADDR" ]; then
        if [ "$NODE_RANK" -eq 0 ]; then
            MASTER_ADDR=$(detect_local_ip)
            echo "ℹ️  Master节点IP自动检测: $MASTER_ADDR"
        else
            echo "❌ 错误: Worker节点(rank=$NODE_RANK)必须指定 --master_addr"
            echo "   用法: bash start_training.sh ... --master_addr <master_ip>"
            exit 1
        fi
    fi
fi

# 端口设置
if [ -z "$MASTER_PORT" ]; then
    if [ "$NNODES" -gt 1 ]; then
        # 多机: 使用固定端口，所有节点必须一致
        MASTER_PORT=29500
        echo "ℹ️  多机端口: $MASTER_PORT (所有节点统一)"
    else
        MASTER_PORT=$(find_free_port 29500)
    fi
else
    if [ "$NNODES" -le 1 ]; then
        if ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT} " || \
           lsof -Pi ":$MASTER_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
            OLD_PORT=$MASTER_PORT
            MASTER_PORT=$(find_free_port $MASTER_PORT)
            echo "⚠️  端口 $OLD_PORT 已被占用，自动切换到: $MASTER_PORT"
        fi
    fi
fi

# Rendezvous 超时默认值: 按节点数缩放，单机10分钟
if [ -z "$RDZV_TIMEOUT" ]; then
    if [ "$NNODES" -gt 1 ]; then
        # 基础30分钟 + 每增加4个节点多10分钟，16节点=60分钟
        RDZV_TIMEOUT=$(( 1800 + (NNODES / 4) * 600 ))
    else
        RDZV_TIMEOUT=600
    fi
fi

# DDP GPU数量最终校验
if [ "$USE_DDP" -eq 1 ] && [ -z "$DDP_NGPUS" ]; then
    DDP_NGPUS=$(detect_gpus_per_node)
fi

if [ "$USE_DDP" -eq 1 ] && [ "$DDP_NGPUS" -lt 1 ]; then
    echo "❌ 错误: DDP需要至少1个GPU, 检测到 $DDP_NGPUS"
    exit 1
fi

# ============================================================================
# 数据集配置
# ============================================================================
case $DATASET_CHOICE in
    B26A|b26a)
        DATASET_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/b26a1_1_training_data"
        DATASET_NAME="B26A"
        echo "ℹ️  使用 B26A 数据集"
        ;;
    all|ALL)
        DATASET_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data"
        DATASET_NAME="all_training_data"
        echo "ℹ️  使用全量数据集 (all_training_data)"
        ;;
    all_training_data_c1|c1)
        DATASET_ROOT="/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data_c1"
        DATASET_NAME="all_training_data_c1"
        echo "ℹ️  使用 camera_1 全量数据集 (all_training_data_c1, fisheye FOV~98°)"
        ;;
    custom|CUSTOM)
        if [ -z "$CUSTOM_DATASET" ]; then
            echo "❌ 错误: 使用自定义数据集时，请设置 CUSTOM_DATASET 环境变量"
            echo ""
            echo "示例:"
            echo "  CUSTOM_DATASET=/path/to/dataset bash start_training.sh custom v1"
            exit 1
        fi
        DATASET_ROOT="$CUSTOM_DATASET"
        DATASET_NAME=$(basename "$DATASET_ROOT")
        echo "ℹ️  使用自定义数据集: $DATASET_NAME"
        ;;
    *)
        echo "❌ 错误: 未知的数据集选择 '$DATASET_CHOICE'"
        echo ""
        echo "可用选项:"
        echo "  B26A   - B26A 数据集"
        echo "  all    - 全量数据集 (长焦相机)"
        echo "  c1     - camera_1 全量数据集 (fisheye, FOV~98°)"
        echo "  custom - 自定义数据集 (需设置 CUSTOM_DATASET 环境变量)"
        echo ""
        echo "示例:"
        echo "  bash start_training.sh B26A v1"
        echo "  bash start_training.sh all v1"
        echo "  bash start_training.sh c1 v1"
        echo "  CUSTOM_DATASET=/path bash start_training.sh custom v1"
        exit 1
        ;;
esac

if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ 错误: 数据集不存在: $DATASET_ROOT"
    exit 1
fi

if [ ! -d "$DATASET_ROOT/sequences" ]; then
    echo "❌ 错误: 数据集格式错误，未找到 sequences/ 目录"
    exit 1
fi

# ============================================================================
# 启动信息
# ============================================================================

# 检查是否有正在运行的训练
RUNNING=$(ps aux | grep -E "train_kitti.py" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "⚠️  警告: 检测到 $RUNNING 个正在运行的训练进程"
    if [ "${BATCH_MODE:-0}" -eq 1 ]; then
        echo "BATCH_MODE: 自动继续"
    elif [ "$FOREGROUND" -eq 1 ]; then
        echo "是否继续启动新训练? (y/n)"
        read -t 10 -p "> " CONFIRM || CONFIRM="n"
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            echo "已取消"
            exit 0
        fi
    else
        echo "是否继续启动新训练? (y/n)"
        read -t 10 -p "> " CONFIRM || CONFIRM="n"
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            echo "已取消"
            exit 0
        fi
    fi
fi

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
# conda activate bevcalib

# ============================================================================
# 计算训练日志目录 (TensorBoard 和训练启动共用)
# ============================================================================

if [ "$USE_DDP" -eq 1 ]; then
    LOG_SUFFIX="small_${DDP_ANGLE}deg_${VERSION}"
    TRAIN_LOG_DIR="./logs/${DATASET_NAME}/model_${LOG_SUFFIX}"
else
    # 经典模式有两个独立训练目录，TensorBoard 指向上层以同时显示两个实验
    TRAIN_LOG_DIR="./logs/${DATASET_NAME}"
fi
mkdir -p "$TRAIN_LOG_DIR"

# ============================================================================
# TensorBoard 自动启动 (训练前启动，确保前台模式下也能监控)
# ============================================================================

TB_LOG_DIR="$TRAIN_LOG_DIR"
IS_MASTER=1
if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" -ne 0 ]; then
    IS_MASTER=0
fi

tb_msg() {
    echo "$1"
    echo "$1" >> "$TB_LOG_DIR/tb_startup.log"
}

if [ "$ENABLE_TB" -eq 1 ] && [ "$IS_MASTER" -eq 1 ]; then
    > "$TB_LOG_DIR/tb_startup.log"
    TB_ALREADY_RUNNING=$(ps aux | grep -E "tensorboard.*--logdir" | grep -v grep | wc -l)

    if [ "$TB_ALREADY_RUNNING" -gt 0 ]; then
        EXISTING_TB_PORT=$(ps aux | grep -E "tensorboard.*--logdir" | grep -v grep | grep -oP -- '--port\s+\K[0-9]+' | head -1)
        EXISTING_TB_PORT=${EXISTING_TB_PORT:-6006}
        LOCAL_IP=$(detect_local_ip)
        PUBLIC_IP=$(detect_public_ip)
        tb_msg "========================================"
        tb_msg "TensorBoard 已在运行"
        tb_msg "========================================"
        tb_msg "  本机: http://localhost:${EXISTING_TB_PORT}"
        tb_msg "  内网: http://${LOCAL_IP}:${EXISTING_TB_PORT}"
        if [ -n "$PUBLIC_IP" ]; then
        tb_msg "  公网: http://${PUBLIC_IP}:${EXISTING_TB_PORT}"
        fi
        tb_msg "  日志目录: $TB_LOG_DIR"
        tb_msg "========================================"
    else
        if [ -z "$TB_PORT" ]; then
            TB_PORT=$(find_free_port 6006)
        else
            if ss -tlnp 2>/dev/null | grep -q ":${TB_PORT} " || \
               lsof -Pi ":$TB_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
                OLD_TB_PORT=$TB_PORT
                TB_PORT=$(find_free_port $TB_PORT)
                tb_msg "⚠️  TensorBoard端口 $OLD_TB_PORT 已占用, 使用: $TB_PORT"
            fi
        fi

        nohup tensorboard --logdir "$TB_LOG_DIR" --port $TB_PORT --bind_all \
            > "$TB_LOG_DIR/tensorboard.log" 2>&1 &
        TB_PID=$!

        sleep 2
        if kill -0 $TB_PID 2>/dev/null; then
            LOCAL_IP=$(detect_local_ip)
            PUBLIC_IP=$(detect_public_ip)
            tb_msg "========================================"
            tb_msg "TensorBoard 已启动"
            tb_msg "========================================"
            tb_msg "  本机: http://localhost:${TB_PORT}"
            tb_msg "  内网: http://${LOCAL_IP}:${TB_PORT}"
            if [ -n "$PUBLIC_IP" ]; then
            tb_msg "  公网: http://${PUBLIC_IP}:${TB_PORT}"
            fi
            tb_msg "  日志目录: $TB_LOG_DIR"
            tb_msg "  PID: $TB_PID"
            _POD_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
            if [ -n "$_POD_IP" ] && [ "$_POD_IP" != "$LOCAL_IP" ]; then
                tb_msg "  Pod IP: http://${_POD_IP}:${TB_PORT} (集群内部)"
            fi
            if [ -f /proc/1/cgroup ] && grep -q "kubepods\|docker" /proc/1/cgroup 2>/dev/null; then
                if [ "$_POD_IP" = "$LOCAL_IP" ]; then
                    tb_msg "  ⚠️  容器环境: 显示的IP为Pod IP, 外部可能无法访问"
                    tb_msg "     设置 NODE_IP 环境变量指定宿主机IP"
                fi
            fi
            tb_msg "========================================"
        else
            tb_msg "⚠️  TensorBoard 启动失败, 请手动启动:"
            tb_msg "  tensorboard --logdir $TB_LOG_DIR --port $TB_PORT"
        fi
    fi
    echo ""
fi

echo "启动训练..."
echo ""

# ============================================================================
# 训练启动
# ============================================================================

if [ "$USE_DDP" -eq 1 ]; then
    # ================================================================
    # DDP模式: 单机或多机GPU协同训练
    # ================================================================
    LOG_SUFFIX="small_${DDP_ANGLE}deg_${VERSION}"
    DDP_LOG_DIR="./logs/${DATASET_NAME}/model_${LOG_SUFFIX}"
    
    # 检查实验是否已完成（基于 checkpoint 文件判断，兼容多机 DDP）
    # resume_ckpt 非空时视为断点续训，不因已有 ckpt 跳过
    # Worker 节点永远不跳过 — 必须加入 DDP 集群
    if [ "${FORCE_RERUN:-0}" != "1" ] && [ "${NODE_RANK:-0}" = "0" ]; then
        _CKPT_DIR="$DDP_LOG_DIR/checkpoint"
        _WANTS_RESUME=0
        if [ -n "${RESUME_CKPT:-}" ] && [ "${RESUME_CKPT}" != "null" ]; then
            _WANTS_RESUME=1
        fi
        if [ -d "$_CKPT_DIR" ] && [ "$(find "$_CKPT_DIR" -name '*.pth' -type f 2>/dev/null | wc -l)" -gt 0 ] && [ "$_WANTS_RESUME" -eq 0 ]; then
            _N_CKPT=$(find "$_CKPT_DIR" -name '*.pth' -type f | wc -l)
            echo "⏭️  跳过训练: 检测到已有 checkpoint 文件 (${_N_CKPT}个)"
            echo "  路径: $_CKPT_DIR"
            echo "  如需断点续训，请传入 --resume_ckpt auto"
            echo "  如需从头重训，请删除该目录或使用 --force 参数"
            exit 0
        fi
        if [ -d "$_CKPT_DIR" ] && [ "$_WANTS_RESUME" -eq 1 ]; then
            _N_CKPT=$(find "$_CKPT_DIR" -name '*.pth' -type f 2>/dev/null | wc -l)
            echo "  断点续训: resume_ckpt=$RESUME_CKPT (${_N_CKPT}个 ckpt @ $_CKPT_DIR)"
        fi
    fi
    
    mkdir -p "$DDP_LOG_DIR"

    MULTI_NODE_ARGS=""
    if [ "$NNODES" -gt 1 ]; then
        MULTI_NODE_ARGS="--nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
        echo "[DDP] 多机模式: ${NNODES}机 x ${DDP_NGPUS}GPU, node_rank=${NODE_RANK}"
        echo "[DDP] Master: ${MASTER_ADDR}:${MASTER_PORT}, rendezvous超时: ${RDZV_TIMEOUT}s"
        echo "[DDP] 本机: $(hostname 2>/dev/null), IP=$(hostname -I 2>/dev/null | awk '{print $1}')"

        # DNS 解析验证
        _RESOLVED=$(getent hosts "$MASTER_ADDR" 2>/dev/null | awk '{print $1}' | head -1)
        if [ -n "$_RESOLVED" ]; then
            echo "[DDP] ✓ DNS: $MASTER_ADDR -> $_RESOLVED"
        else
            echo "[DDP] ⚠️  DNS 解析失败: $MASTER_ADDR"
            echo "[DDP]    请确认 K8s headless service 正确创建"
        fi

        if [ "$NODE_RANK" -ne 0 ]; then
            echo "[DDP] Worker节点 → 检查Master连通性..."
            if timeout 10 bash -c "echo >/dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; then
                echo "[DDP] ✓ Master $MASTER_ADDR:$MASTER_PORT 可达"
            else
                echo "[DDP] ⚠️  Master $MASTER_ADDR:$MASTER_PORT 暂不可达 (Master可能尚未启动)"
                echo "[DDP]    torchrun 将等待 ${RDZV_TIMEOUT}s"
            fi
        else
            echo "[DDP] Master节点 → 等待 ${NNODES} 个节点加入 (超时: ${RDZV_TIMEOUT}s)"
        fi
    else
        echo "[DDP] ${DDP_NGPUS}-GPU并行训练 (${DDP_ANGLE}deg, ${DDP_TRANS}m)..."
    fi

    LR_ARG=""
    [ -n "$LEARNING_RATE" ] && LR_ARG="--learning_rate $LEARNING_RATE"

    AXIS_LOSS_ARGS=""
    [ -n "$ENABLE_AXIS_LOSS" ] && AXIS_LOSS_ARGS="--enable_axis_loss"
    [ -n "$WEIGHT_AXIS_ROTATION" ] && AXIS_LOSS_ARGS="$AXIS_LOSS_ARGS --weight_axis_rotation $WEIGHT_AXIS_ROTATION"
    [ -n "$AXIS_WEIGHTS" ] && AXIS_LOSS_ARGS="$AXIS_LOSS_ARGS --axis_weights $AXIS_WEIGHTS"

    OPTIM_ARGS=""
    [ -n "$LR_SCHEDULE" ] && OPTIM_ARGS="$OPTIM_ARGS --lr_schedule $LR_SCHEDULE"
    [ -n "$WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --warmup_epochs $WARMUP_EPOCHS"
    [ -n "$STEP_SIZE" ] && OPTIM_ARGS="$OPTIM_ARGS --step_size $STEP_SIZE"
    [ -n "$BACKBONE_LR_SCALE" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_lr_scale $BACKBONE_LR_SCALE"
    [ -n "$COSINE_T0" ] && OPTIM_ARGS="$OPTIM_ARGS --cosine_T0 $COSINE_T0"
    [ -n "$COSINE_TMULT" ] && OPTIM_ARGS="$OPTIM_ARGS --cosine_Tmult $COSINE_TMULT"
    [ -n "$DROP_PATH_RATE" ] && OPTIM_ARGS="$OPTIM_ARGS --drop_path_rate $DROP_PATH_RATE"
    [ -n "$HEAD_DROPOUT" ] && OPTIM_ARGS="$OPTIM_ARGS --head_dropout $HEAD_DROPOUT"
    [ -n "$PERTURB_DISTRIBUTION" ] && OPTIM_ARGS="$OPTIM_ARGS --perturb_distribution $PERTURB_DISTRIBUTION"
    [ -n "$PER_AXIS_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --per_axis_prob $PER_AXIS_PROB"
    [ -n "$PER_AXIS_WEIGHTS" ] && OPTIM_ARGS="$OPTIM_ARGS --per_axis_weights $PER_AXIS_WEIGHTS"
    [ -n "$AUGMENT_PC_JITTER" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pc_jitter $AUGMENT_PC_JITTER"
    [ -n "$AUGMENT_PC_DROPOUT" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pc_dropout $AUGMENT_PC_DROPOUT"
    [ -n "$AUGMENT_COLOR_JITTER" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_color_jitter $AUGMENT_COLOR_JITTER"
    [ -n "$AUGMENT_INTRINSIC" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_intrinsic $AUGMENT_INTRINSIC"
    [ -n "$AUGMENT_INTRINSIC_CXCY" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_intrinsic_cxcy $AUGMENT_INTRINSIC_CXCY"
    [ -n "$AUGMENT_PITCH_FLIP_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pitch_flip_prob $AUGMENT_PITCH_FLIP_PROB"
    [ -n "$AUGMENT_PITCH_FLIP_MAX_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pitch_flip_max_deg $AUGMENT_PITCH_FLIP_MAX_DEG"
    [ -n "$AUGMENT_PITCH_SIGN_FLIP_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pitch_sign_flip_prob $AUGMENT_PITCH_SIGN_FLIP_PROB"
    [ -n "$EVAL_ANGLE" ] && OPTIM_ARGS="$OPTIM_ARGS --eval_angle_range_deg $EVAL_ANGLE"
    [ -n "$EVAL_TRANS_RANGE" ] && OPTIM_ARGS="$OPTIM_ARGS --eval_trans_range $EVAL_TRANS_RANGE"
    [ -n "$EARLY_STOPPING_PATIENCE" ] && OPTIM_ARGS="$OPTIM_ARGS --early_stopping_patience $EARLY_STOPPING_PATIENCE"
    [ -n "$SEED" ] && OPTIM_ARGS="$OPTIM_ARGS --seed $SEED"
    [ -n "$PRETRAIN_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --pretrain_ckpt $PRETRAIN_CKPT"
    [ -n "$PRETRAIN_CKPT_FALLBACK" ] && OPTIM_ARGS="$OPTIM_ARGS --pretrain_ckpt_fallback $PRETRAIN_CKPT_FALLBACK"
    [ -n "$RESUME_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --resume_ckpt $RESUME_CKPT"
    [ -n "$NO_AMP" ] && OPTIM_ARGS="$OPTIM_ARGS --no_amp $NO_AMP"
    [ -n "$AMP_BF16" ] && OPTIM_ARGS="$OPTIM_ARGS --amp_bf16 $AMP_BF16"
    [ -n "$NUM_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --num_epochs $NUM_EPOCHS"
    [ -n "$SAVE_CKPT_PER_EPOCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --save_ckpt_per_epoches $SAVE_CKPT_PER_EPOCHES"
    [ -n "$USE_GEODESIC_LOSS" ] && OPTIM_ARGS="$OPTIM_ARGS --use_geodesic_loss $USE_GEODESIC_LOSS"
    [ -n "$USE_BALANCED_AXIS_LOSS" ] && OPTIM_ARGS="$OPTIM_ARGS --use_balanced_axis_loss $USE_BALANCED_AXIS_LOSS"
    [ -n "$USE_MLP_HEAD" ] && OPTIM_ARGS="$OPTIM_ARGS --use_mlp_head $USE_MLP_HEAD"
    [ -n "$USE_DEFORMABLE" ] && OPTIM_ARGS="$OPTIM_ARGS --use_deformable $USE_DEFORMABLE"
    [ -n "$BEV_POOL_FACTOR" ] && OPTIM_ARGS="$OPTIM_ARGS --bev_pool_factor $BEV_POOL_FACTOR"
    [ -n "$USE_FOUNDATION_DEPTH" ] && OPTIM_ARGS="$OPTIM_ARGS --use_foundation_depth $USE_FOUNDATION_DEPTH"
    [ -n "$DEPTH_MODEL_TYPE" ] && OPTIM_ARGS="$OPTIM_ARGS --depth_model_type $DEPTH_MODEL_TYPE"
    [ -n "$FD_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --fd_mode $FD_MODE"
    [ -n "$DEPTH_SUP_ALPHA" ] && OPTIM_ARGS="$OPTIM_ARGS --depth_sup_alpha $DEPTH_SUP_ALPHA"
    [ -n "$MAX_FRAMES_PER_SEQ" ] && OPTIM_ARGS="$OPTIM_ARGS --max_frames_per_seq $MAX_FRAMES_PER_SEQ"
    [ -n "$SAMPLE_STEP" ] && OPTIM_ARGS="$OPTIM_ARGS --sample_step $SAMPLE_STEP"
    [ -n "$POSE_AWARE_SAMPLING" ] && OPTIM_ARGS="$OPTIM_ARGS $POSE_AWARE_SAMPLING"
    [ -n "$POSES_DIR" ] && OPTIM_ARGS="$OPTIM_ARGS --poses_dir $POSES_DIR"
    [ -n "$EVAL_EPOCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --eval_epoches $EVAL_EPOCHES"
    [ -n "$GRAD_ACCUM_STEPS" ] && OPTIM_ARGS="$OPTIM_ARGS --grad_accum_steps $GRAD_ACCUM_STEPS"
    [ -n "$VOXEL_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --voxel_mode $VOXEL_MODE"
    [ -n "$TO_BEV_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --to_bev_mode $TO_BEV_MODE"
    [ -n "$SCATTER_REDUCE" ] && OPTIM_ARGS="$OPTIM_ARGS --scatter_reduce $SCATTER_REDUCE"
    [ -n "$FUSER_TYPE" ] && OPTIM_ARGS="$OPTIM_ARGS --fuser_type $FUSER_TYPE"
    [ -n "$CAM_DROP_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --cam_drop_prob $CAM_DROP_PROB"
    [ -n "$CAM_DROP_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --cam_drop_mode $CAM_DROP_MODE"
    [ -n "$INTRINSIC_INPUT" ] && OPTIM_ARGS="$OPTIM_ARGS $INTRINSIC_INPUT"
    [ -n "$ENABLE_VIS" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_vis $ENABLE_VIS"
    [ -n "$VIS_FREQ" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_freq $VIS_FREQ"
    [ -n "$VIS_SAMPLES" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_samples $VIS_SAMPLES"
    [ -n "$VIS_POINTS" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_points $VIS_POINTS"
    [ -n "$VIS_POINT_RADIUS" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_point_radius $VIS_POINT_RADIUS"
    [ -n "$VALIDATE_DATA" ] && OPTIM_ARGS="$OPTIM_ARGS --validate_data $VALIDATE_DATA"
    [ -n "$VALIDATE_SAMPLE_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --validate_sample_ratio $VALIDATE_SAMPLE_RATIO"
    [ -n "$MIN_POINT_UTILIZATION" ] && OPTIM_ARGS="$OPTIM_ARGS --min_point_utilization $MIN_POINT_UTILIZATION"
    [ -n "$MIN_VALID_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --min_valid_ratio $MIN_VALID_RATIO"
    [ -n "$ENABLE_CKPT_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_ckpt_eval $ENABLE_CKPT_EVAL"
    [ -n "$ENABLE_MEDW_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_medw_eval $ENABLE_MEDW_EVAL"
    [ -n "$ENABLE_JACOBIAN_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_jacobian_eval $ENABLE_JACOBIAN_EVAL"
    [ -n "$JACOBIAN_EVAL_ANGLE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_eval_angle_deg $JACOBIAN_EVAL_ANGLE_DEG"
    [ -n "$JACOBIAN_EVAL_BATCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_eval_batches $JACOBIAN_EVAL_BATCHES"
    [ -n "$JACOBIAN_EVAL_N_PROBES" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_eval_n_probes $JACOBIAN_EVAL_N_PROBES"
    [ -n "$JACOBIAN_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_weight $JACOBIAN_LOSS_WEIGHT"
    [ -n "$JACOBIAN_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_start_epoch $JACOBIAN_LOSS_START_EPOCH"
    [ -n "$JACOBIAN_LOSS_PROBE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_probe_deg $JACOBIAN_LOSS_PROBE_DEG"
    [ -n "$JACOBIAN_LOSS_INTERVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_interval $JACOBIAN_LOSS_INTERVAL"
    [ -n "$MEDW_EVAL_MAX_FRAMES" ] && OPTIM_ARGS="$OPTIM_ARGS --medw_eval_max_frames $MEDW_EVAL_MAX_FRAMES"
    [ -n "$ENABLE_DUAL_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_dual_gate_ckpt $ENABLE_DUAL_GATE_CKPT"
    [ -n "$DUAL_GATE_JACOBIAN_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_jacobian_min $DUAL_GATE_JACOBIAN_MIN"
    [ -n "$DUAL_GATE_MEDW_MAX" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_medw_max $DUAL_GATE_MEDW_MAX"
    [ -n "$DUAL_GATE_RECOVERY_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_recovery_min $DUAL_GATE_RECOVERY_MIN"
    [ -n "$DUAL_GATE_ZD_MAX" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_zd_max $DUAL_GATE_ZD_MAX"
    [ -n "$ENABLE_RECOVERY_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_recovery_gate_ckpt $ENABLE_RECOVERY_GATE_CKPT"
    [ -n "$ENABLE_ZD_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_zd_gate_ckpt $ENABLE_ZD_GATE_CKPT"
    [ -n "$DUAL_GATE_INJECT_RECOVERY_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_inject_recovery_min $DUAL_GATE_INJECT_RECOVERY_MIN"
    [ -n "$DUAL_GATE_PRED_INDEP_MAX" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_pred_indep_max $DUAL_GATE_PRED_INDEP_MAX"
    [ -n "$ENABLE_INJECT_RECOVERY_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_inject_recovery_eval $ENABLE_INJECT_RECOVERY_EVAL"
    [ -n "$INJECT_RECOVERY_EVAL_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_eval_deg $INJECT_RECOVERY_EVAL_DEG"
    [ -n "$INJECT_RECOVERY_EVAL_BATCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_eval_batches $INJECT_RECOVERY_EVAL_BATCHES"
    [ -n "$DDP_AUTO_SCALE" ] && OPTIM_ARGS="$OPTIM_ARGS --ddp_auto_scale $DDP_AUTO_SCALE"
    [ -n "$DDP_REFERENCE_GPUS" ] && OPTIM_ARGS="$OPTIM_ARGS --ddp_reference_gpus $DDP_REFERENCE_GPUS"
    [ -n "$MAX_SCALED_LR" ] && OPTIM_ARGS="$OPTIM_ARGS --max_scaled_lr $MAX_SCALED_LR"
    [ -n "$DATA_BALANCE" ] && OPTIM_ARGS="$OPTIM_ARGS --data_balance $DATA_BALANCE"
    [ -n "$SEQ_WEIGHT_OVERRIDES" ] && OPTIM_ARGS="$OPTIM_ARGS --seq_weight_overrides $SEQ_WEIGHT_OVERRIDES"
    [ -n "$WEIGHT_DECAY" ] && OPTIM_ARGS="$OPTIM_ARGS --wd $WEIGHT_DECAY"
    [ -n "$TARGET_WIDTH" ] && OPTIM_ARGS="$OPTIM_ARGS --target_width $TARGET_WIDTH"
    [ -n "$TARGET_HEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --target_height $TARGET_HEIGHT"
    [ -n "$BACKBONE_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_warmup_epochs $BACKBONE_WARMUP_EPOCHS"
    [ -n "$LAYER_WISE_LR_DECAY" ] && OPTIM_ARGS="$OPTIM_ARGS --layer_wise_lr_decay $LAYER_WISE_LR_DECAY"
    [ -n "$USE_PITCH_BRANCH" ] && OPTIM_ARGS="$OPTIM_ARGS --use_pitch_branch $USE_PITCH_BRANCH"
    [ -n "$PITCH_AUX_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --pitch_aux_weight $PITCH_AUX_WEIGHT"
    [ -n "$BEV_INSTANCE_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --bev_instance_norm $BEV_INSTANCE_NORM"
    [ -n "$AUGMENT_MOUNT_JITTER_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_mount_jitter_prob $AUGMENT_MOUNT_JITTER_PROB"
    [ -n "$AUGMENT_MOUNT_JITTER_ROT_SIGMA" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_mount_jitter_rot_sigma $AUGMENT_MOUNT_JITTER_ROT_SIGMA"
    [ -n "$AUGMENT_MOUNT_JITTER_TRANS_SIGMA" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_mount_jitter_trans_sigma $AUGMENT_MOUNT_JITTER_TRANS_SIGMA"
    [ -n "$USE_CONTRASTIVE_EXTRINSIC" ] && OPTIM_ARGS="$OPTIM_ARGS --use_contrastive_extrinsic $USE_CONTRASTIVE_EXTRINSIC"
    [ -n "$CONTRASTIVE_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --contrastive_weight $CONTRASTIVE_WEIGHT"
    [ -n "$DOMAIN_ADVERSARIAL" ] && OPTIM_ARGS="$OPTIM_ARGS --domain_adversarial $DOMAIN_ADVERSARIAL"
    [ -n "$DOMAIN_ADVERSARIAL_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --domain_adversarial_weight $DOMAIN_ADVERSARIAL_WEIGHT"
    [ -n "$NUM_DOMAINS" ] && OPTIM_ARGS="$OPTIM_ARGS --num_domains $NUM_DOMAINS"
    [ -n "$CAM2BEV_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --cam2bev_mode $CAM2BEV_MODE"
    [ -n "$BACKBONE_TYPE" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_type $BACKBONE_TYPE"
    [ -n "$BACKBONE_VARIANT" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_variant $BACKBONE_VARIANT"
    [ -n "$FREEZE_BACKBONE" ] && OPTIM_ARGS="$OPTIM_ARGS --freeze_backbone $FREEZE_BACKBONE"
    [ -n "$BACKBONE_FREEZE_LAYERS" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_freeze_layers $BACKBONE_FREEZE_LAYERS"
    [ -n "$BACKBONE_WEIGHTS" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_weights $BACKBONE_WEIGHTS"
    [ -n "$TINIT_DROPOUT_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_dropout_prob $TINIT_DROPOUT_PROB"
    [ -n "$CONSISTENCY_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --consistency_loss_weight $CONSISTENCY_LOSS_WEIGHT"
    [ -n "$CONSISTENCY_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --consistency_loss_start_epoch $CONSISTENCY_LOSS_START_EPOCH"
    [ -n "$PROGRESSIVE_ANGLE_START" ] && OPTIM_ARGS="$OPTIM_ARGS --progressive_angle_start $PROGRESSIVE_ANGLE_START"
    [ -n "$PROGRESSIVE_ANGLE_END" ] && OPTIM_ARGS="$OPTIM_ARGS --progressive_angle_end $PROGRESSIVE_ANGLE_END"
    [ -n "$PROGRESSIVE_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --progressive_warmup_epochs $PROGRESSIVE_WARMUP_EPOCHS"
    [ -n "$EMA_CONSISTENCY" ] && OPTIM_ARGS="$OPTIM_ARGS --ema_consistency $EMA_CONSISTENCY"
    [ -n "$EMA_DECAY" ] && OPTIM_ARGS="$OPTIM_ARGS --ema_decay $EMA_DECAY"
    [ -n "$CONTINUOUS_TINIT_NOISE" ] && OPTIM_ARGS="$OPTIM_ARGS --continuous_tinit_noise $CONTINUOUS_TINIT_NOISE"
    [ -n "$CONTINUOUS_NOISE_MAX_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --continuous_noise_max_deg $CONTINUOUS_NOISE_MAX_DEG"
    [ -n "$ZERO_PERTURBATION_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_perturbation_prob $ZERO_PERTURBATION_PROB"
    [ -n "$ZERO_DRIFT_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_weight $ZERO_DRIFT_LOSS_WEIGHT"
    [ -n "$ZERO_DRIFT_LOSS_WEIGHT_START" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_weight_start $ZERO_DRIFT_LOSS_WEIGHT_START"
    [ -n "$ZERO_DRIFT_LOSS_RAMP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_ramp_epochs $ZERO_DRIFT_LOSS_RAMP_EPOCHS"
    [ -n "$ZERO_DRIFT_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_start_epoch $ZERO_DRIFT_LOSS_START_EPOCH"
    [ -n "$ZERO_DRIFT_DEDICATED_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_dedicated_ratio $ZERO_DRIFT_DEDICATED_RATIO"
    [ -n "$ENABLE_JACOBIAN_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_jacobian_gate_ckpt $ENABLE_JACOBIAN_GATE_CKPT"
    [ -n "$JACOBIAN_EARLY_STOP_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_early_stop_min $JACOBIAN_EARLY_STOP_MIN"
    [ -n "$JACOBIAN_EARLY_STOP_PATIENCE" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_early_stop_patience $JACOBIAN_EARLY_STOP_PATIENCE"
    [ -n "$INJECT_RECOVERY_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_loss_weight $INJECT_RECOVERY_LOSS_WEIGHT"
    [ -n "$INJECT_RECOVERY_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_loss_start_epoch $INJECT_RECOVERY_LOSS_START_EPOCH"
    [ -n "$INJECT_RECOVERY_MAGNITUDE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_magnitude_deg $INJECT_RECOVERY_MAGNITUDE_DEG"
    [ -n "$INJECT_RECOVERY_DEDICATED_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_dedicated_ratio $INJECT_RECOVERY_DEDICATED_RATIO"
    [ -n "$USE_MGDA" ] && OPTIM_ARGS="$OPTIM_ARGS --use_mgda $USE_MGDA"
    [ -n "$MGDA_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --mgda_start_epoch $MGDA_START_EPOCH"
    [ -n "$MGDA_INCLUDE_INJECT" ] && OPTIM_ARGS="$OPTIM_ARGS --mgda_include_inject $MGDA_INCLUDE_INJECT"
    [ -n "$MGDA_INCLUDE_PHOTO" ] && OPTIM_ARGS="$OPTIM_ARGS --mgda_include_photo $MGDA_INCLUDE_PHOTO"
    [ -n "$USE_LSP_LOSS" ] && OPTIM_ARGS="$OPTIM_ARGS --use_lsp_loss $USE_LSP_LOSS"
    [ -n "$LSP_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_weight $LSP_WEIGHT"
    [ -n "$LSP_WEIGHT_START" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_weight_start $LSP_WEIGHT_START"
    [ -n "$LSP_RAMP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_ramp_epochs $LSP_RAMP_EPOCHS"
    [ -n "$LSP_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_start_epoch $LSP_START_EPOCH"
    [ -n "$LSP_LAMBDA_SSIM" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_lambda_ssim $LSP_LAMBDA_SSIM"
    [ -n "$LSP_MAX_POINTS" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_max_points $LSP_MAX_POINTS"
    [ -n "$LSP_DOWNSAMPLE" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_downsample $LSP_DOWNSAMPLE"
    [ -n "$LSP_LOSS_CLIP" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_loss_clip $LSP_LOSS_CLIP"
    [ -n "$LSP_MIN_VALID_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_min_valid_ratio $LSP_MIN_VALID_RATIO"
    [ -n "$RIG_CONSISTENCY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --rig_consistency_weight $RIG_CONSISTENCY_WEIGHT"
    [ -n "$RIG_CONSISTENCY_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --rig_consistency_start_epoch $RIG_CONSISTENCY_START_EPOCH"
    [ -n "$POSE_RELEASE_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --pose_release_epoch $POSE_RELEASE_EPOCH"
    [ -n "$POSE_RELEASE_JOINT_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --pose_release_joint_epoch $POSE_RELEASE_JOINT_EPOCH"
    [ -n "$POSE_RELEASE_CORR_LR_SCALE_JOINT" ] && OPTIM_ARGS="$OPTIM_ARGS --pose_release_corr_lr_scale_joint $POSE_RELEASE_CORR_LR_SCALE_JOINT"
    [ -n "$FREEZE_BACKBONE_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --freeze_backbone_epoch $FREEZE_BACKBONE_EPOCH"
    [ -n "$USE_DP_HEAD" ] && OPTIM_ARGS="$OPTIM_ARGS --use_dp_head $USE_DP_HEAD"
    [ -n "$DP_GATE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --dp_gate_deg $DP_GATE_DEG"
    [ -n "$ROUTE_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --route_loss_weight $ROUTE_LOSS_WEIGHT"
    [ -n "$ROUTE_ZD_PENALTY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --route_zd_penalty_weight $ROUTE_ZD_PENALTY_WEIGHT"
    [ -n "$USE_JACG" ] && OPTIM_ARGS="$OPTIM_ARGS --use_jacg $USE_JACG"
    [ -n "$JACG_HIDDEN_DIM" ] && OPTIM_ARGS="$OPTIM_ARGS --jacg_hidden_dim $JACG_HIDDEN_DIM"
    [ -n "$BIAS_PATH_IN_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --bias_path_in_norm $BIAS_PATH_IN_NORM"
    [ -n "$RECOVERY_PATH_LAYER_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --recovery_path_layer_norm $RECOVERY_PATH_LAYER_NORM"
    [ -n "$USE_HARD_ROUTE_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --use_hard_route_eval $USE_HARD_ROUTE_EVAL"
    [ -n "$USE_ADIR" ] && OPTIM_ARGS="$OPTIM_ARGS --use_adir $USE_ADIR"
    [ -n "$ADIR_STEPS" ] && OPTIM_ARGS="$OPTIM_ARGS --adir_steps $ADIR_STEPS"
    [ -n "$ADIR_MAX_STEP_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --adir_max_step_deg $ADIR_MAX_STEP_DEG"
    [ -n "$USE_GATED_INSTANCE_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --use_gated_instance_norm $USE_GATED_INSTANCE_NORM"
    [ -n "$GIN_INIT_GATE" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_init_gate $GIN_INIT_GATE"
    [ -n "$GIN_CHANNELS" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_channels $GIN_CHANNELS"
    [ -n "$PITCH_VERTICAL_BANDS" ] && OPTIM_ARGS="$OPTIM_ARGS --pitch_vertical_bands $PITCH_VERTICAL_BANDS"
    [ -n "$GIN_GATE_REG_TARGET" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_gate_reg_target $GIN_GATE_REG_TARGET"
    [ -n "$GIN_GATE_REG_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_gate_reg_weight $GIN_GATE_REG_WEIGHT"
    [ -n "$MULTI_SCALE_PERTURB" ] && OPTIM_ARGS="$OPTIM_ARGS --multi_scale_perturb $MULTI_SCALE_PERTURB"
    [ -n "$CORRELATION_FUSION" ] && OPTIM_ARGS="$OPTIM_ARGS --correlation_fusion $CORRELATION_FUSION"
    [ -n "$CROSS_CORRELATION_FUSION" ] && OPTIM_ARGS="$OPTIM_ARGS --cross_correlation_fusion $CROSS_CORRELATION_FUSION"
    [ -n "$EXPLICIT_TINIT" ] && OPTIM_ARGS="$OPTIM_ARGS --explicit_tinit $EXPLICIT_TINIT"
    [ -n "$TINIT_BEV_FILM" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_bev_film $TINIT_BEV_FILM"
    [ -n "$TINIT_QUERY_FILM" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_query_film $TINIT_QUERY_FILM"
    [ -n "$TINIT_SENSITIVITY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_sensitivity_weight $TINIT_SENSITIVITY_WEIGHT"
    [ -n "$ITERATIVE_REFINE" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine $ITERATIVE_REFINE"
    [ -n "$ITERATIVE_REFINE_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_weight $ITERATIVE_REFINE_WEIGHT"
    [ -n "$ITERATIVE_REFINE_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_start_epoch $ITERATIVE_REFINE_START_EPOCH"
    [ -n "$ITERATIVE_REFINE_USE_SYNTHETIC_INIT" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_use_synthetic_init $ITERATIVE_REFINE_USE_SYNTHETIC_INIT"
    [ -n "$ITERATIVE_REFINE_SYNTHETIC_RESIDUAL_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_synthetic_residual_deg $ITERATIVE_REFINE_SYNTHETIC_RESIDUAL_DEG"
    [ -n "$NATIVE_CROSS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross $NATIVE_CROSS"
    [ -n "$NATIVE_CROSS_PC_GROUPS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pc_groups $NATIVE_CROSS_PC_GROUPS"
    [ -n "$NATIVE_CROSS_N_HARMONIC" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_n_harmonic $NATIVE_CROSS_N_HARMONIC"
    [ -n "$NATIVE_CROSS_N_LAYERS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_n_layers $NATIVE_CROSS_N_LAYERS"
    [ -n "$NATIVE_CROSS_DUAL_BRANCH" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_dual_branch $NATIVE_CROSS_DUAL_BRANCH"
    [ -n "$NATIVE_CROSS_KNN" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_knn $NATIVE_CROSS_KNN"
    [ -n "$NATIVE_CROSS_USE_FPS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_use_fps $NATIVE_CROSS_USE_FPS"
    [ -n "$NATIVE_CROSS_USE_POINTGPT" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_use_pointgpt $NATIVE_CROSS_USE_POINTGPT"
    [ -n "$NATIVE_CROSS_POINTGPT_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pointgpt_ckpt $NATIVE_CROSS_POINTGPT_CKPT"
    [ -n "$NATIVE_CROSS_POINTGPT_CONFIG" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pointgpt_config $NATIVE_CROSS_POINTGPT_CONFIG"
    [ -n "$NATIVE_CROSS_POINTGPT_MAX_DEPTH" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pointgpt_max_depth $NATIVE_CROSS_POINTGPT_MAX_DEPTH"
    [ -n "$NATIVE_CROSS_EXTEND_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_extend_ratio $NATIVE_CROSS_EXTEND_RATIO"
    [ -n "$FUSION_BACKEND" ] && OPTIM_ARGS="$OPTIM_ARGS --fusion_backend $FUSION_BACKEND"
    [ -n "$PC_ENCODER_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --pc_encoder_mode $PC_ENCODER_MODE"
    [ -n "$FUSION_VARIANT" ] && OPTIM_ARGS="$OPTIM_ARGS --fusion_variant $FUSION_VARIANT"
    [ -n "$DEEP_SUPERVISION_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --deep_supervision_weight $DEEP_SUPERVISION_WEIGHT"
    [ -n "$GATE_ENTROPY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --gate_entropy_weight $GATE_ENTROPY_WEIGHT"
    [ -n "$APPEARANCE_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --appearance_loss_weight $APPEARANCE_LOSS_WEIGHT"
    [ -n "$DEPTH_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --depth_loss_weight $DEPTH_LOSS_WEIGHT"
    [ -n "$GEO_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --geo_loss_start_epoch $GEO_LOSS_START_EPOCH"
    [ -n "$USE_MATCH_HEAD" ] && OPTIM_ARGS="$OPTIM_ARGS --use_match_head $USE_MATCH_HEAD"
    [ -n "$USE_LOCAL_CORRELATION" ] && OPTIM_ARGS="$OPTIM_ARGS --use_local_correlation $USE_LOCAL_CORRELATION"
    [ -n "$CORRESPONDENCE_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_loss_weight $CORRESPONDENCE_LOSS_WEIGHT"
    [ -n "$CORRESPONDENCE_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_loss_start_epoch $CORRESPONDENCE_LOSS_START_EPOCH"
    [ -n "$CORRESPONDENCE_LOSS_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_loss_warmup_epochs $CORRESPONDENCE_LOSS_WARMUP_EPOCHS"
    [ -n "$MATCH_DISABLE_FALLBACK" ] && OPTIM_ARGS="$OPTIM_ARGS --match_disable_fallback $MATCH_DISABLE_FALLBACK"
    [ -n "$MATCH_GATE_USE_INIT_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --match_gate_use_init_ratio $MATCH_GATE_USE_INIT_RATIO"
    [ -n "$MATCH_CONFIDENCE_THRESHOLD" ] && OPTIM_ARGS="$OPTIM_ARGS --match_confidence_threshold $MATCH_CONFIDENCE_THRESHOLD"
    [ -n "$MATCH_CORR_VALIDITY_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --match_corr_validity_mode $MATCH_CORR_VALIDITY_MODE"
    [ -n "$MATCH_EPNP_MIN_POINTS" ] && OPTIM_ARGS="$OPTIM_ARGS --match_epnp_min_points $MATCH_EPNP_MIN_POINTS"
    [ -n "$MATCH_PHASE_NOISE_MAX_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --match_phase_noise_max_deg $MATCH_PHASE_NOISE_MAX_DEG"
    [ -n "$COMPOSE_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --compose_mode $COMPOSE_MODE"
    [ -n "$NUM_CORRESPONDENCES" ] && OPTIM_ARGS="$OPTIM_ARGS --num_correspondences $NUM_CORRESPONDENCES"
    [ -n "$MATCH_VALID_RATIO_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --match_valid_ratio_min $MATCH_VALID_RATIO_MIN"
    [ -n "$CORRESPONDENCE_SUPERVISION" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_supervision $CORRESPONDENCE_SUPERVISION"
    [ -n "$DIFFERENTIABLE_EPNP" ] && OPTIM_ARGS="$OPTIM_ARGS --differentiable_epnp $DIFFERENTIABLE_EPNP"
    [ -n "$DIFF_EPNP_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --diff_epnp_warmup_epochs $DIFF_EPNP_WARMUP_EPOCHS"
    [ -n "$BEV_BRANCH_LR_SCALE" ] && OPTIM_ARGS="$OPTIM_ARGS --bev_branch_lr_scale $BEV_BRANCH_LR_SCALE"
    [ -n "$PROJFUSION_IMAGE_HW" ] && OPTIM_ARGS="$OPTIM_ARGS --projfusion_image_hw $PROJFUSION_IMAGE_HW"
    [ -n "$AUGMENT_FOV_CROP_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_fov_crop_prob $(_quote_cli_val "$AUGMENT_FOV_CROP_PROB")"
    [ -n "$AUGMENT_FOV_CROP_RATIO_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_fov_crop_ratio_min $(_quote_cli_val "$AUGMENT_FOV_CROP_RATIO_MIN")"
    [ -n "$AUGMENT_FOV_CROP_RATIO_MAX" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_fov_crop_ratio_max $(_quote_cli_val "$AUGMENT_FOV_CROP_RATIO_MAX")"
    [ -n "$AUGMENT_LIDAR_SPARSE_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_lidar_sparse_prob $(_quote_cli_val "$AUGMENT_LIDAR_SPARSE_PROB")"
    [ -n "$AUGMENT_LIDAR_SPARSE_LINES" ] && _opt_flag_append "--augment_lidar_sparse_lines" "$AUGMENT_LIDAR_SPARSE_LINES"
    [ -n "$AUGMENT_LIDAR_VERTICAL_FOV" ] && _opt_flag_append "--augment_lidar_vertical_fov" "$AUGMENT_LIDAR_VERTICAL_FOV"

    TB_PORT_ARG=""
    [ -n "$TB_PORT" ] && TB_PORT_ARG="--tensorboard_port $TB_PORT"

    TRAIN_CMD="bash train_universal.sh scratch \
        --dataset_root $DATASET_ROOT \
        --dataset_name $DATASET_NAME \
        --ddp $DDP_NGPUS \
        --angle_range_deg $DDP_ANGLE \
        --trans_range $DDP_TRANS \
        --batch_size $BATCH_SIZE \
        --log_suffix ${LOG_SUFFIX} \
        --rdzv_timeout $RDZV_TIMEOUT \
        $MULTI_NODE_ARGS \
        $USE_COMPILE \
        $ROTATION_ONLY \
        $AXIS_LOSS_ARGS \
        $LR_ARG \
        $TB_PORT_ARG \
        $OPTIM_ARGS $PASSTHROUGH_ARGS"

    if [ "$FOREGROUND" -eq 1 ]; then
        echo "  日志: $DDP_LOG_DIR/train.log (+ 终端输出)"
        echo ""
        eval $TRAIN_CMD
    else
        nohup bash -c "$TRAIN_CMD" >> "$DDP_LOG_DIR/train.log" 2>> "$DDP_LOG_DIR/train.stderr.log" &
        PID1=$!
        echo "  PID: $PID1"
        echo "  日志: $DDP_LOG_DIR/train.log"

        echo ""
        echo "========================================"
        if [ "$NNODES" -gt 1 ]; then
            echo "DDP多机训练已启动 (node ${NODE_RANK}/${NNODES})"
        else
            echo "DDP训练已启动 (${DDP_NGPUS} GPUs)"
        fi
        echo "========================================"
        echo ""
        echo "训练进程PID: $PID1"
        if [ "$NNODES" -gt 1 ]; then
            echo "集群: ${NNODES}节点 x ${DDP_NGPUS}GPU = $((NNODES * DDP_NGPUS))GPU"
            echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
        else
            echo "GPU数量: $DDP_NGPUS"
        fi
        echo ""
        echo "日志:"
        echo "  tail -f $DDP_LOG_DIR/train.log"
    fi
else
    # ================================================================
    # 经典模式: 每个GPU独立跑不同扰动
    # ================================================================

    GPU0_LOG_DIR="./logs/${DATASET_NAME}/model_small_10deg_${VERSION}"
    GPU1_LOG_DIR="./logs/${DATASET_NAME}/model_small_5deg_${VERSION}"
    
    # 检查实验是否已完成（基于 checkpoint 文件判断）
    _GPU0_CKPT_DIR="$GPU0_LOG_DIR/checkpoint"
    _GPU1_CKPT_DIR="$GPU1_LOG_DIR/checkpoint"
    _GPU0_HAS_CKPT=0; _GPU1_HAS_CKPT=0
    _WANTS_RESUME=0
    if [ -n "${RESUME_CKPT:-}" ] && [ "${RESUME_CKPT}" != "null" ]; then
        _WANTS_RESUME=1
    fi
    [ -d "$_GPU0_CKPT_DIR" ] && [ "$(find "$_GPU0_CKPT_DIR" -name '*.pth' -type f 2>/dev/null | wc -l)" -gt 0 ] && _GPU0_HAS_CKPT=1
    [ -d "$_GPU1_CKPT_DIR" ] && [ "$(find "$_GPU1_CKPT_DIR" -name '*.pth' -type f 2>/dev/null | wc -l)" -gt 0 ] && _GPU1_HAS_CKPT=1
    if [ "${FORCE_RERUN:-0}" != "1" ] && [ "$_GPU0_HAS_CKPT" -eq 1 ] && [ "$_GPU1_HAS_CKPT" -eq 1 ] && [ "$_WANTS_RESUME" -eq 0 ]; then
        echo "⏭️  跳过训练: 两组实验均已有 checkpoint 文件"
        echo "  GPU0: $_GPU0_CKPT_DIR"
        echo "  GPU1: $_GPU1_CKPT_DIR"
        echo "  如需断点续训，请传入 --resume_ckpt auto"
        echo "  如需从头重训，请删除对应目录或使用 --force 参数"
        exit 0
    fi
    
    mkdir -p "$GPU0_LOG_DIR" "$GPU1_LOG_DIR"

    LR_ARG=""
    [ -n "$LEARNING_RATE" ] && LR_ARG="--learning_rate $LEARNING_RATE"

    AXIS_LOSS_ARGS=""
    [ -n "$ENABLE_AXIS_LOSS" ] && AXIS_LOSS_ARGS="--enable_axis_loss"
    [ -n "$WEIGHT_AXIS_ROTATION" ] && AXIS_LOSS_ARGS="$AXIS_LOSS_ARGS --weight_axis_rotation $WEIGHT_AXIS_ROTATION"
    [ -n "$AXIS_WEIGHTS" ] && AXIS_LOSS_ARGS="$AXIS_LOSS_ARGS --axis_weights $AXIS_WEIGHTS"

    OPTIM_ARGS=""
    [ -n "$LR_SCHEDULE" ] && OPTIM_ARGS="$OPTIM_ARGS --lr_schedule $LR_SCHEDULE"
    [ -n "$WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --warmup_epochs $WARMUP_EPOCHS"
    [ -n "$STEP_SIZE" ] && OPTIM_ARGS="$OPTIM_ARGS --step_size $STEP_SIZE"
    [ -n "$BACKBONE_LR_SCALE" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_lr_scale $BACKBONE_LR_SCALE"
    [ -n "$COSINE_T0" ] && OPTIM_ARGS="$OPTIM_ARGS --cosine_T0 $COSINE_T0"
    [ -n "$COSINE_TMULT" ] && OPTIM_ARGS="$OPTIM_ARGS --cosine_Tmult $COSINE_TMULT"
    [ -n "$DROP_PATH_RATE" ] && OPTIM_ARGS="$OPTIM_ARGS --drop_path_rate $DROP_PATH_RATE"
    [ -n "$HEAD_DROPOUT" ] && OPTIM_ARGS="$OPTIM_ARGS --head_dropout $HEAD_DROPOUT"
    [ -n "$PERTURB_DISTRIBUTION" ] && OPTIM_ARGS="$OPTIM_ARGS --perturb_distribution $PERTURB_DISTRIBUTION"
    [ -n "$PER_AXIS_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --per_axis_prob $PER_AXIS_PROB"
    [ -n "$PER_AXIS_WEIGHTS" ] && OPTIM_ARGS="$OPTIM_ARGS --per_axis_weights $PER_AXIS_WEIGHTS"
    [ -n "$AUGMENT_PC_JITTER" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pc_jitter $AUGMENT_PC_JITTER"
    [ -n "$AUGMENT_PC_DROPOUT" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pc_dropout $AUGMENT_PC_DROPOUT"
    [ -n "$AUGMENT_COLOR_JITTER" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_color_jitter $AUGMENT_COLOR_JITTER"
    [ -n "$AUGMENT_INTRINSIC" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_intrinsic $AUGMENT_INTRINSIC"
    [ -n "$AUGMENT_INTRINSIC_CXCY" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_intrinsic_cxcy $AUGMENT_INTRINSIC_CXCY"
    [ -n "$AUGMENT_PITCH_FLIP_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pitch_flip_prob $AUGMENT_PITCH_FLIP_PROB"
    [ -n "$AUGMENT_PITCH_FLIP_MAX_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pitch_flip_max_deg $AUGMENT_PITCH_FLIP_MAX_DEG"
    [ -n "$AUGMENT_PITCH_SIGN_FLIP_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_pitch_sign_flip_prob $AUGMENT_PITCH_SIGN_FLIP_PROB"
    [ -n "$EVAL_ANGLE" ] && OPTIM_ARGS="$OPTIM_ARGS --eval_angle_range_deg $EVAL_ANGLE"
    [ -n "$EVAL_TRANS_RANGE" ] && OPTIM_ARGS="$OPTIM_ARGS --eval_trans_range $EVAL_TRANS_RANGE"
    [ -n "$EARLY_STOPPING_PATIENCE" ] && OPTIM_ARGS="$OPTIM_ARGS --early_stopping_patience $EARLY_STOPPING_PATIENCE"
    [ -n "$SEED" ] && OPTIM_ARGS="$OPTIM_ARGS --seed $SEED"
    [ -n "$PRETRAIN_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --pretrain_ckpt $PRETRAIN_CKPT"
    [ -n "$PRETRAIN_CKPT_FALLBACK" ] && OPTIM_ARGS="$OPTIM_ARGS --pretrain_ckpt_fallback $PRETRAIN_CKPT_FALLBACK"
    [ -n "$RESUME_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --resume_ckpt $RESUME_CKPT"
    [ -n "$NO_AMP" ] && OPTIM_ARGS="$OPTIM_ARGS --no_amp $NO_AMP"
    [ -n "$AMP_BF16" ] && OPTIM_ARGS="$OPTIM_ARGS --amp_bf16 $AMP_BF16"
    [ -n "$NUM_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --num_epochs $NUM_EPOCHS"
    [ -n "$SAVE_CKPT_PER_EPOCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --save_ckpt_per_epoches $SAVE_CKPT_PER_EPOCHES"
    [ -n "$USE_GEODESIC_LOSS" ] && OPTIM_ARGS="$OPTIM_ARGS --use_geodesic_loss $USE_GEODESIC_LOSS"
    [ -n "$USE_BALANCED_AXIS_LOSS" ] && OPTIM_ARGS="$OPTIM_ARGS --use_balanced_axis_loss $USE_BALANCED_AXIS_LOSS"
    [ -n "$USE_MLP_HEAD" ] && OPTIM_ARGS="$OPTIM_ARGS --use_mlp_head $USE_MLP_HEAD"
    [ -n "$USE_DEFORMABLE" ] && OPTIM_ARGS="$OPTIM_ARGS --use_deformable $USE_DEFORMABLE"
    [ -n "$BEV_POOL_FACTOR" ] && OPTIM_ARGS="$OPTIM_ARGS --bev_pool_factor $BEV_POOL_FACTOR"
    [ -n "$USE_FOUNDATION_DEPTH" ] && OPTIM_ARGS="$OPTIM_ARGS --use_foundation_depth $USE_FOUNDATION_DEPTH"
    [ -n "$DEPTH_MODEL_TYPE" ] && OPTIM_ARGS="$OPTIM_ARGS --depth_model_type $DEPTH_MODEL_TYPE"
    [ -n "$FD_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --fd_mode $FD_MODE"
    [ -n "$DEPTH_SUP_ALPHA" ] && OPTIM_ARGS="$OPTIM_ARGS --depth_sup_alpha $DEPTH_SUP_ALPHA"
    [ -n "$MAX_FRAMES_PER_SEQ" ] && OPTIM_ARGS="$OPTIM_ARGS --max_frames_per_seq $MAX_FRAMES_PER_SEQ"
    [ -n "$SAMPLE_STEP" ] && OPTIM_ARGS="$OPTIM_ARGS --sample_step $SAMPLE_STEP"
    [ -n "$POSE_AWARE_SAMPLING" ] && OPTIM_ARGS="$OPTIM_ARGS $POSE_AWARE_SAMPLING"
    [ -n "$POSES_DIR" ] && OPTIM_ARGS="$OPTIM_ARGS --poses_dir $POSES_DIR"
    [ -n "$EVAL_EPOCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --eval_epoches $EVAL_EPOCHES"
    [ -n "$GRAD_ACCUM_STEPS" ] && OPTIM_ARGS="$OPTIM_ARGS --grad_accum_steps $GRAD_ACCUM_STEPS"
    [ -n "$VOXEL_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --voxel_mode $VOXEL_MODE"
    [ -n "$TO_BEV_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --to_bev_mode $TO_BEV_MODE"
    [ -n "$SCATTER_REDUCE" ] && OPTIM_ARGS="$OPTIM_ARGS --scatter_reduce $SCATTER_REDUCE"
    [ -n "$FUSER_TYPE" ] && OPTIM_ARGS="$OPTIM_ARGS --fuser_type $FUSER_TYPE"
    [ -n "$CAM_DROP_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --cam_drop_prob $CAM_DROP_PROB"
    [ -n "$CAM_DROP_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --cam_drop_mode $CAM_DROP_MODE"
    [ -n "$INTRINSIC_INPUT" ] && OPTIM_ARGS="$OPTIM_ARGS $INTRINSIC_INPUT"
    [ -n "$ENABLE_VIS" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_vis $ENABLE_VIS"
    [ -n "$VIS_FREQ" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_freq $VIS_FREQ"
    [ -n "$VIS_SAMPLES" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_samples $VIS_SAMPLES"
    [ -n "$VIS_POINTS" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_points $VIS_POINTS"
    [ -n "$VIS_POINT_RADIUS" ] && OPTIM_ARGS="$OPTIM_ARGS --vis_point_radius $VIS_POINT_RADIUS"
    [ -n "$VALIDATE_DATA" ] && OPTIM_ARGS="$OPTIM_ARGS --validate_data $VALIDATE_DATA"
    [ -n "$VALIDATE_SAMPLE_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --validate_sample_ratio $VALIDATE_SAMPLE_RATIO"
    [ -n "$MIN_POINT_UTILIZATION" ] && OPTIM_ARGS="$OPTIM_ARGS --min_point_utilization $MIN_POINT_UTILIZATION"
    [ -n "$MIN_VALID_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --min_valid_ratio $MIN_VALID_RATIO"
    [ -n "$ENABLE_CKPT_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_ckpt_eval $ENABLE_CKPT_EVAL"
    [ -n "$ENABLE_MEDW_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_medw_eval $ENABLE_MEDW_EVAL"
    [ -n "$ENABLE_JACOBIAN_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_jacobian_eval $ENABLE_JACOBIAN_EVAL"
    [ -n "$JACOBIAN_EVAL_ANGLE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_eval_angle_deg $JACOBIAN_EVAL_ANGLE_DEG"
    [ -n "$JACOBIAN_EVAL_BATCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_eval_batches $JACOBIAN_EVAL_BATCHES"
    [ -n "$JACOBIAN_EVAL_N_PROBES" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_eval_n_probes $JACOBIAN_EVAL_N_PROBES"
    [ -n "$JACOBIAN_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_weight $JACOBIAN_LOSS_WEIGHT"
    [ -n "$JACOBIAN_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_start_epoch $JACOBIAN_LOSS_START_EPOCH"
    [ -n "$JACOBIAN_LOSS_PROBE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_probe_deg $JACOBIAN_LOSS_PROBE_DEG"
    [ -n "$JACOBIAN_LOSS_INTERVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_loss_interval $JACOBIAN_LOSS_INTERVAL"
    [ -n "$MEDW_EVAL_MAX_FRAMES" ] && OPTIM_ARGS="$OPTIM_ARGS --medw_eval_max_frames $MEDW_EVAL_MAX_FRAMES"
    [ -n "$ENABLE_DUAL_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_dual_gate_ckpt $ENABLE_DUAL_GATE_CKPT"
    [ -n "$DUAL_GATE_JACOBIAN_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_jacobian_min $DUAL_GATE_JACOBIAN_MIN"
    [ -n "$DUAL_GATE_MEDW_MAX" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_medw_max $DUAL_GATE_MEDW_MAX"
    [ -n "$DUAL_GATE_RECOVERY_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_recovery_min $DUAL_GATE_RECOVERY_MIN"
    [ -n "$DUAL_GATE_ZD_MAX" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_zd_max $DUAL_GATE_ZD_MAX"
    [ -n "$ENABLE_RECOVERY_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_recovery_gate_ckpt $ENABLE_RECOVERY_GATE_CKPT"
    [ -n "$ENABLE_ZD_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_zd_gate_ckpt $ENABLE_ZD_GATE_CKPT"
    [ -n "$DUAL_GATE_INJECT_RECOVERY_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_inject_recovery_min $DUAL_GATE_INJECT_RECOVERY_MIN"
    [ -n "$DUAL_GATE_PRED_INDEP_MAX" ] && OPTIM_ARGS="$OPTIM_ARGS --dual_gate_pred_indep_max $DUAL_GATE_PRED_INDEP_MAX"
    [ -n "$ENABLE_INJECT_RECOVERY_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_inject_recovery_eval $ENABLE_INJECT_RECOVERY_EVAL"
    [ -n "$INJECT_RECOVERY_EVAL_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_eval_deg $INJECT_RECOVERY_EVAL_DEG"
    [ -n "$INJECT_RECOVERY_EVAL_BATCHES" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_eval_batches $INJECT_RECOVERY_EVAL_BATCHES"
    [ -n "$DDP_AUTO_SCALE" ] && OPTIM_ARGS="$OPTIM_ARGS --ddp_auto_scale $DDP_AUTO_SCALE"
    [ -n "$DDP_REFERENCE_GPUS" ] && OPTIM_ARGS="$OPTIM_ARGS --ddp_reference_gpus $DDP_REFERENCE_GPUS"
    [ -n "$MAX_SCALED_LR" ] && OPTIM_ARGS="$OPTIM_ARGS --max_scaled_lr $MAX_SCALED_LR"
    [ -n "$DATA_BALANCE" ] && OPTIM_ARGS="$OPTIM_ARGS --data_balance $DATA_BALANCE"
    [ -n "$SEQ_WEIGHT_OVERRIDES" ] && OPTIM_ARGS="$OPTIM_ARGS --seq_weight_overrides $SEQ_WEIGHT_OVERRIDES"
    [ -n "$WEIGHT_DECAY" ] && OPTIM_ARGS="$OPTIM_ARGS --wd $WEIGHT_DECAY"
    [ -n "$TARGET_WIDTH" ] && OPTIM_ARGS="$OPTIM_ARGS --target_width $TARGET_WIDTH"
    [ -n "$TARGET_HEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --target_height $TARGET_HEIGHT"
    [ -n "$BACKBONE_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_warmup_epochs $BACKBONE_WARMUP_EPOCHS"
    [ -n "$LAYER_WISE_LR_DECAY" ] && OPTIM_ARGS="$OPTIM_ARGS --layer_wise_lr_decay $LAYER_WISE_LR_DECAY"
    [ -n "$USE_PITCH_BRANCH" ] && OPTIM_ARGS="$OPTIM_ARGS --use_pitch_branch $USE_PITCH_BRANCH"
    [ -n "$PITCH_AUX_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --pitch_aux_weight $PITCH_AUX_WEIGHT"
    [ -n "$BEV_INSTANCE_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --bev_instance_norm $BEV_INSTANCE_NORM"
    [ -n "$AUGMENT_MOUNT_JITTER_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_mount_jitter_prob $AUGMENT_MOUNT_JITTER_PROB"
    [ -n "$AUGMENT_MOUNT_JITTER_ROT_SIGMA" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_mount_jitter_rot_sigma $AUGMENT_MOUNT_JITTER_ROT_SIGMA"
    [ -n "$AUGMENT_MOUNT_JITTER_TRANS_SIGMA" ] && OPTIM_ARGS="$OPTIM_ARGS --augment_mount_jitter_trans_sigma $AUGMENT_MOUNT_JITTER_TRANS_SIGMA"
    [ -n "$USE_CONTRASTIVE_EXTRINSIC" ] && OPTIM_ARGS="$OPTIM_ARGS --use_contrastive_extrinsic $USE_CONTRASTIVE_EXTRINSIC"
    [ -n "$CONTRASTIVE_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --contrastive_weight $CONTRASTIVE_WEIGHT"
    [ -n "$DOMAIN_ADVERSARIAL" ] && OPTIM_ARGS="$OPTIM_ARGS --domain_adversarial $DOMAIN_ADVERSARIAL"
    [ -n "$DOMAIN_ADVERSARIAL_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --domain_adversarial_weight $DOMAIN_ADVERSARIAL_WEIGHT"
    [ -n "$NUM_DOMAINS" ] && OPTIM_ARGS="$OPTIM_ARGS --num_domains $NUM_DOMAINS"
    [ -n "$CAM2BEV_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --cam2bev_mode $CAM2BEV_MODE"
    [ -n "$BACKBONE_TYPE" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_type $BACKBONE_TYPE"
    [ -n "$BACKBONE_VARIANT" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_variant $BACKBONE_VARIANT"
    [ -n "$FREEZE_BACKBONE" ] && OPTIM_ARGS="$OPTIM_ARGS --freeze_backbone $FREEZE_BACKBONE"
    [ -n "$BACKBONE_FREEZE_LAYERS" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_freeze_layers $BACKBONE_FREEZE_LAYERS"
    [ -n "$BACKBONE_WEIGHTS" ] && OPTIM_ARGS="$OPTIM_ARGS --backbone_weights $BACKBONE_WEIGHTS"
    [ -n "$TINIT_DROPOUT_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_dropout_prob $TINIT_DROPOUT_PROB"
    [ -n "$CONSISTENCY_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --consistency_loss_weight $CONSISTENCY_LOSS_WEIGHT"
    [ -n "$CONSISTENCY_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --consistency_loss_start_epoch $CONSISTENCY_LOSS_START_EPOCH"
    [ -n "$PROGRESSIVE_ANGLE_START" ] && OPTIM_ARGS="$OPTIM_ARGS --progressive_angle_start $PROGRESSIVE_ANGLE_START"
    [ -n "$PROGRESSIVE_ANGLE_END" ] && OPTIM_ARGS="$OPTIM_ARGS --progressive_angle_end $PROGRESSIVE_ANGLE_END"
    [ -n "$PROGRESSIVE_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --progressive_warmup_epochs $PROGRESSIVE_WARMUP_EPOCHS"
    [ -n "$EMA_CONSISTENCY" ] && OPTIM_ARGS="$OPTIM_ARGS --ema_consistency $EMA_CONSISTENCY"
    [ -n "$EMA_DECAY" ] && OPTIM_ARGS="$OPTIM_ARGS --ema_decay $EMA_DECAY"
    [ -n "$CONTINUOUS_TINIT_NOISE" ] && OPTIM_ARGS="$OPTIM_ARGS --continuous_tinit_noise $CONTINUOUS_TINIT_NOISE"
    [ -n "$CONTINUOUS_NOISE_MAX_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --continuous_noise_max_deg $CONTINUOUS_NOISE_MAX_DEG"
    [ -n "$ZERO_PERTURBATION_PROB" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_perturbation_prob $ZERO_PERTURBATION_PROB"
    [ -n "$ZERO_DRIFT_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_weight $ZERO_DRIFT_LOSS_WEIGHT"
    [ -n "$ZERO_DRIFT_LOSS_WEIGHT_START" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_weight_start $ZERO_DRIFT_LOSS_WEIGHT_START"
    [ -n "$ZERO_DRIFT_LOSS_RAMP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_ramp_epochs $ZERO_DRIFT_LOSS_RAMP_EPOCHS"
    [ -n "$ZERO_DRIFT_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_loss_start_epoch $ZERO_DRIFT_LOSS_START_EPOCH"
    [ -n "$ZERO_DRIFT_DEDICATED_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --zero_drift_dedicated_ratio $ZERO_DRIFT_DEDICATED_RATIO"
    [ -n "$ENABLE_JACOBIAN_GATE_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --enable_jacobian_gate_ckpt $ENABLE_JACOBIAN_GATE_CKPT"
    [ -n "$JACOBIAN_EARLY_STOP_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_early_stop_min $JACOBIAN_EARLY_STOP_MIN"
    [ -n "$JACOBIAN_EARLY_STOP_PATIENCE" ] && OPTIM_ARGS="$OPTIM_ARGS --jacobian_early_stop_patience $JACOBIAN_EARLY_STOP_PATIENCE"
    [ -n "$INJECT_RECOVERY_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_loss_weight $INJECT_RECOVERY_LOSS_WEIGHT"
    [ -n "$INJECT_RECOVERY_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_loss_start_epoch $INJECT_RECOVERY_LOSS_START_EPOCH"
    [ -n "$INJECT_RECOVERY_MAGNITUDE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_magnitude_deg $INJECT_RECOVERY_MAGNITUDE_DEG"
    [ -n "$INJECT_RECOVERY_DEDICATED_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --inject_recovery_dedicated_ratio $INJECT_RECOVERY_DEDICATED_RATIO"
    [ -n "$USE_MGDA" ] && OPTIM_ARGS="$OPTIM_ARGS --use_mgda $USE_MGDA"
    [ -n "$MGDA_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --mgda_start_epoch $MGDA_START_EPOCH"
    [ -n "$MGDA_INCLUDE_INJECT" ] && OPTIM_ARGS="$OPTIM_ARGS --mgda_include_inject $MGDA_INCLUDE_INJECT"
    [ -n "$MGDA_INCLUDE_PHOTO" ] && OPTIM_ARGS="$OPTIM_ARGS --mgda_include_photo $MGDA_INCLUDE_PHOTO"
    [ -n "$USE_LSP_LOSS" ] && OPTIM_ARGS="$OPTIM_ARGS --use_lsp_loss $USE_LSP_LOSS"
    [ -n "$LSP_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_weight $LSP_WEIGHT"
    [ -n "$LSP_WEIGHT_START" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_weight_start $LSP_WEIGHT_START"
    [ -n "$LSP_RAMP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_ramp_epochs $LSP_RAMP_EPOCHS"
    [ -n "$LSP_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_start_epoch $LSP_START_EPOCH"
    [ -n "$LSP_LAMBDA_SSIM" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_lambda_ssim $LSP_LAMBDA_SSIM"
    [ -n "$LSP_MAX_POINTS" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_max_points $LSP_MAX_POINTS"
    [ -n "$LSP_DOWNSAMPLE" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_downsample $LSP_DOWNSAMPLE"
    [ -n "$LSP_LOSS_CLIP" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_loss_clip $LSP_LOSS_CLIP"
    [ -n "$LSP_MIN_VALID_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --lsp_min_valid_ratio $LSP_MIN_VALID_RATIO"
    [ -n "$RIG_CONSISTENCY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --rig_consistency_weight $RIG_CONSISTENCY_WEIGHT"
    [ -n "$RIG_CONSISTENCY_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --rig_consistency_start_epoch $RIG_CONSISTENCY_START_EPOCH"
    [ -n "$POSE_RELEASE_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --pose_release_epoch $POSE_RELEASE_EPOCH"
    [ -n "$POSE_RELEASE_JOINT_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --pose_release_joint_epoch $POSE_RELEASE_JOINT_EPOCH"
    [ -n "$POSE_RELEASE_CORR_LR_SCALE_JOINT" ] && OPTIM_ARGS="$OPTIM_ARGS --pose_release_corr_lr_scale_joint $POSE_RELEASE_CORR_LR_SCALE_JOINT"
    [ -n "$FREEZE_BACKBONE_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --freeze_backbone_epoch $FREEZE_BACKBONE_EPOCH"
    [ -n "$USE_DP_HEAD" ] && OPTIM_ARGS="$OPTIM_ARGS --use_dp_head $USE_DP_HEAD"
    [ -n "$DP_GATE_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --dp_gate_deg $DP_GATE_DEG"
    [ -n "$ROUTE_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --route_loss_weight $ROUTE_LOSS_WEIGHT"
    [ -n "$ROUTE_ZD_PENALTY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --route_zd_penalty_weight $ROUTE_ZD_PENALTY_WEIGHT"
    [ -n "$USE_JACG" ] && OPTIM_ARGS="$OPTIM_ARGS --use_jacg $USE_JACG"
    [ -n "$JACG_HIDDEN_DIM" ] && OPTIM_ARGS="$OPTIM_ARGS --jacg_hidden_dim $JACG_HIDDEN_DIM"
    [ -n "$BIAS_PATH_IN_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --bias_path_in_norm $BIAS_PATH_IN_NORM"
    [ -n "$RECOVERY_PATH_LAYER_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --recovery_path_layer_norm $RECOVERY_PATH_LAYER_NORM"
    [ -n "$USE_HARD_ROUTE_EVAL" ] && OPTIM_ARGS="$OPTIM_ARGS --use_hard_route_eval $USE_HARD_ROUTE_EVAL"
    [ -n "$USE_ADIR" ] && OPTIM_ARGS="$OPTIM_ARGS --use_adir $USE_ADIR"
    [ -n "$ADIR_STEPS" ] && OPTIM_ARGS="$OPTIM_ARGS --adir_steps $ADIR_STEPS"
    [ -n "$ADIR_MAX_STEP_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --adir_max_step_deg $ADIR_MAX_STEP_DEG"
    [ -n "$USE_GATED_INSTANCE_NORM" ] && OPTIM_ARGS="$OPTIM_ARGS --use_gated_instance_norm $USE_GATED_INSTANCE_NORM"
    [ -n "$GIN_INIT_GATE" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_init_gate $GIN_INIT_GATE"
    [ -n "$GIN_CHANNELS" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_channels $GIN_CHANNELS"
    [ -n "$PITCH_VERTICAL_BANDS" ] && OPTIM_ARGS="$OPTIM_ARGS --pitch_vertical_bands $PITCH_VERTICAL_BANDS"
    [ -n "$GIN_GATE_REG_TARGET" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_gate_reg_target $GIN_GATE_REG_TARGET"
    [ -n "$GIN_GATE_REG_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --gin_gate_reg_weight $GIN_GATE_REG_WEIGHT"
    [ -n "$MULTI_SCALE_PERTURB" ] && OPTIM_ARGS="$OPTIM_ARGS --multi_scale_perturb $MULTI_SCALE_PERTURB"
    [ -n "$CORRELATION_FUSION" ] && OPTIM_ARGS="$OPTIM_ARGS --correlation_fusion $CORRELATION_FUSION"
    [ -n "$CROSS_CORRELATION_FUSION" ] && OPTIM_ARGS="$OPTIM_ARGS --cross_correlation_fusion $CROSS_CORRELATION_FUSION"
    [ -n "$EXPLICIT_TINIT" ] && OPTIM_ARGS="$OPTIM_ARGS --explicit_tinit $EXPLICIT_TINIT"
    [ -n "$TINIT_BEV_FILM" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_bev_film $TINIT_BEV_FILM"
    [ -n "$TINIT_QUERY_FILM" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_query_film $TINIT_QUERY_FILM"
    [ -n "$TINIT_SENSITIVITY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --tinit_sensitivity_weight $TINIT_SENSITIVITY_WEIGHT"
    [ -n "$ITERATIVE_REFINE" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine $ITERATIVE_REFINE"
    [ -n "$ITERATIVE_REFINE_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_weight $ITERATIVE_REFINE_WEIGHT"
    [ -n "$ITERATIVE_REFINE_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_start_epoch $ITERATIVE_REFINE_START_EPOCH"
    [ -n "$ITERATIVE_REFINE_USE_SYNTHETIC_INIT" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_use_synthetic_init $ITERATIVE_REFINE_USE_SYNTHETIC_INIT"
    [ -n "$ITERATIVE_REFINE_SYNTHETIC_RESIDUAL_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --iterative_refine_synthetic_residual_deg $ITERATIVE_REFINE_SYNTHETIC_RESIDUAL_DEG"
    [ -n "$NATIVE_CROSS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross $NATIVE_CROSS"
    [ -n "$NATIVE_CROSS_PC_GROUPS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pc_groups $NATIVE_CROSS_PC_GROUPS"
    [ -n "$NATIVE_CROSS_N_HARMONIC" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_n_harmonic $NATIVE_CROSS_N_HARMONIC"
    [ -n "$NATIVE_CROSS_N_LAYERS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_n_layers $NATIVE_CROSS_N_LAYERS"
    [ -n "$NATIVE_CROSS_DUAL_BRANCH" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_dual_branch $NATIVE_CROSS_DUAL_BRANCH"
    [ -n "$NATIVE_CROSS_KNN" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_knn $NATIVE_CROSS_KNN"
    [ -n "$NATIVE_CROSS_USE_FPS" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_use_fps $NATIVE_CROSS_USE_FPS"
    [ -n "$NATIVE_CROSS_USE_POINTGPT" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_use_pointgpt $NATIVE_CROSS_USE_POINTGPT"
    [ -n "$NATIVE_CROSS_POINTGPT_CKPT" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pointgpt_ckpt $NATIVE_CROSS_POINTGPT_CKPT"
    [ -n "$NATIVE_CROSS_POINTGPT_CONFIG" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pointgpt_config $NATIVE_CROSS_POINTGPT_CONFIG"
    [ -n "$NATIVE_CROSS_POINTGPT_MAX_DEPTH" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_pointgpt_max_depth $NATIVE_CROSS_POINTGPT_MAX_DEPTH"
    [ -n "$NATIVE_CROSS_EXTEND_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --native_cross_extend_ratio $NATIVE_CROSS_EXTEND_RATIO"
    [ -n "$FUSION_BACKEND" ] && OPTIM_ARGS="$OPTIM_ARGS --fusion_backend $FUSION_BACKEND"
    [ -n "$PC_ENCODER_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --pc_encoder_mode $PC_ENCODER_MODE"
    [ -n "$FUSION_VARIANT" ] && OPTIM_ARGS="$OPTIM_ARGS --fusion_variant $FUSION_VARIANT"
    [ -n "$DEEP_SUPERVISION_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --deep_supervision_weight $DEEP_SUPERVISION_WEIGHT"
    [ -n "$GATE_ENTROPY_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --gate_entropy_weight $GATE_ENTROPY_WEIGHT"
    [ -n "$APPEARANCE_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --appearance_loss_weight $APPEARANCE_LOSS_WEIGHT"
    [ -n "$DEPTH_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --depth_loss_weight $DEPTH_LOSS_WEIGHT"
    [ -n "$GEO_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --geo_loss_start_epoch $GEO_LOSS_START_EPOCH"
    [ -n "$USE_MATCH_HEAD" ] && OPTIM_ARGS="$OPTIM_ARGS --use_match_head $USE_MATCH_HEAD"
    [ -n "$USE_LOCAL_CORRELATION" ] && OPTIM_ARGS="$OPTIM_ARGS --use_local_correlation $USE_LOCAL_CORRELATION"
    [ -n "$CORRESPONDENCE_LOSS_WEIGHT" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_loss_weight $CORRESPONDENCE_LOSS_WEIGHT"
    [ -n "$CORRESPONDENCE_LOSS_START_EPOCH" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_loss_start_epoch $CORRESPONDENCE_LOSS_START_EPOCH"
    [ -n "$CORRESPONDENCE_LOSS_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_loss_warmup_epochs $CORRESPONDENCE_LOSS_WARMUP_EPOCHS"
    [ -n "$MATCH_DISABLE_FALLBACK" ] && OPTIM_ARGS="$OPTIM_ARGS --match_disable_fallback $MATCH_DISABLE_FALLBACK"
    [ -n "$MATCH_GATE_USE_INIT_RATIO" ] && OPTIM_ARGS="$OPTIM_ARGS --match_gate_use_init_ratio $MATCH_GATE_USE_INIT_RATIO"
    [ -n "$MATCH_CONFIDENCE_THRESHOLD" ] && OPTIM_ARGS="$OPTIM_ARGS --match_confidence_threshold $MATCH_CONFIDENCE_THRESHOLD"
    [ -n "$MATCH_CORR_VALIDITY_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --match_corr_validity_mode $MATCH_CORR_VALIDITY_MODE"
    [ -n "$MATCH_EPNP_MIN_POINTS" ] && OPTIM_ARGS="$OPTIM_ARGS --match_epnp_min_points $MATCH_EPNP_MIN_POINTS"
    [ -n "$MATCH_PHASE_NOISE_MAX_DEG" ] && OPTIM_ARGS="$OPTIM_ARGS --match_phase_noise_max_deg $MATCH_PHASE_NOISE_MAX_DEG"
    [ -n "$COMPOSE_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --compose_mode $COMPOSE_MODE"
    [ -n "$NUM_CORRESPONDENCES" ] && OPTIM_ARGS="$OPTIM_ARGS --num_correspondences $NUM_CORRESPONDENCES"
    [ -n "$MATCH_VALID_RATIO_MIN" ] && OPTIM_ARGS="$OPTIM_ARGS --match_valid_ratio_min $MATCH_VALID_RATIO_MIN"
    [ -n "$CORRESPONDENCE_SUPERVISION" ] && OPTIM_ARGS="$OPTIM_ARGS --correspondence_supervision $CORRESPONDENCE_SUPERVISION"
    [ -n "$DIFFERENTIABLE_EPNP" ] && OPTIM_ARGS="$OPTIM_ARGS --differentiable_epnp $DIFFERENTIABLE_EPNP"
    [ -n "$DIFF_EPNP_WARMUP_EPOCHS" ] && OPTIM_ARGS="$OPTIM_ARGS --diff_epnp_warmup_epochs $DIFF_EPNP_WARMUP_EPOCHS"
    [ -n "$BEV_BRANCH_LR_SCALE" ] && OPTIM_ARGS="$OPTIM_ARGS --bev_branch_lr_scale $BEV_BRANCH_LR_SCALE"
    [ -n "$PROJFUSION_IMAGE_HW" ] && OPTIM_ARGS="$OPTIM_ARGS --projfusion_image_hw $PROJFUSION_IMAGE_HW"

    TB_PORT_ARG=""
    [ -n "$TB_PORT" ] && TB_PORT_ARG="--tensorboard_port $TB_PORT"

    CLASSIC_ANGLE_0=${DDP_ANGLE:-10}
    CLASSIC_TRANS_0=${DDP_TRANS:-0.5}
    CLASSIC_ANGLE_1=${DDP_ANGLE:-5}
    CLASSIC_TRANS_1=${DDP_TRANS:-0.3}

    TRAIN_CMD_0="bash train_universal.sh scratch \
        --dataset_root $DATASET_ROOT \
        --dataset_name $DATASET_NAME \
        --cuda_device 0 \
        --angle_range_deg $CLASSIC_ANGLE_0 \
        --trans_range $CLASSIC_TRANS_0 \
        --batch_size $BATCH_SIZE \
        --log_suffix small_${CLASSIC_ANGLE_0}deg_${VERSION} \
        $USE_COMPILE \
        $ROTATION_ONLY \
        $AXIS_LOSS_ARGS \
        $LR_ARG \
        $TB_PORT_ARG \
        $OPTIM_ARGS $PASSTHROUGH_ARGS"

    TRAIN_CMD_1="bash train_universal.sh scratch \
        --dataset_root $DATASET_ROOT \
        --dataset_name $DATASET_NAME \
        --cuda_device 1 \
        --angle_range_deg $CLASSIC_ANGLE_1 \
        --trans_range $CLASSIC_TRANS_1 \
        --batch_size $BATCH_SIZE \
        --log_suffix small_${CLASSIC_ANGLE_1}deg_${VERSION} \
        $USE_COMPILE \
        $ROTATION_ONLY \
        $AXIS_LOSS_ARGS \
        $LR_ARG \
        $TB_PORT_ARG \
        $OPTIM_ARGS $PASSTHROUGH_ARGS"

    if [ "$FOREGROUND" -eq 1 ]; then
        echo "[前台模式] 两个GPU训练并行执行, Ctrl+C 同时停止两个任务"
        echo ""

        echo "[GPU 0] 训练 (${CLASSIC_ANGLE_0}deg, ${CLASSIC_TRANS_0}m)..."
        echo "  日志: $GPU0_LOG_DIR/train.log"
        eval $TRAIN_CMD_0 > /dev/null 2>&1 &
        PID1=$!

        sleep 2

        echo "[GPU 1] 训练 (${CLASSIC_ANGLE_1}deg, ${CLASSIC_TRANS_1}m)..."
        echo "  日志: $GPU1_LOG_DIR/train.log"
        eval $TRAIN_CMD_1 > /dev/null 2>&1 &
        PID2=$!

        echo ""
        echo "训练进程PID: GPU0=$PID1, GPU1=$PID2"
        echo "等待训练完成... (Ctrl+C 停止)"
        echo ""

        # Ctrl+C 时清理子进程
        trap "echo ''; echo '正在停止训练...'; kill $PID1 $PID2 2>/dev/null; wait $PID1 $PID2 2>/dev/null; echo '已停止'; exit 130" INT TERM

        # 实时显示两个任务的日志
        tail -f "$GPU0_LOG_DIR/train.log" "$GPU1_LOG_DIR/train.log" &
        TAIL_PID=$!

        wait $PID1 $PID2 2>/dev/null
        EXIT_CODE=$?
        kill $TAIL_PID 2>/dev/null
        wait $TAIL_PID 2>/dev/null

        echo ""
        echo "========================================"
        echo "训练已完成 (exit=$EXIT_CODE)"
        echo "========================================"
    else
        echo "[GPU 0] 训练 (${CLASSIC_ANGLE_0}deg, ${CLASSIC_TRANS_0}m)..."
        nohup bash -c "$TRAIN_CMD_0" > /dev/null 2>&1 &
        PID1=$!
        echo "  PID: $PID1"
        echo "  日志: $GPU0_LOG_DIR/train.log"

        sleep 2

        echo "[GPU 1] 训练 (${CLASSIC_ANGLE_1}deg, ${CLASSIC_TRANS_1}m)..."
        nohup bash -c "$TRAIN_CMD_1" > /dev/null 2>&1 &
        PID2=$!
        echo "  PID: $PID2"
        echo "  日志: $GPU1_LOG_DIR/train.log"

        echo ""
        echo "========================================"
        echo "所有训练已启动"
        echo "========================================"
        echo ""
        echo "训练进程PID:"
        echo "  GPU 0 (${CLASSIC_ANGLE_0}deg): $PID1"
        echo "  GPU 1 (${CLASSIC_ANGLE_1}deg): $PID2"
        echo ""
        echo "日志:"
        echo "  tail -f $GPU0_LOG_DIR/train.log"
        echo "  tail -f $GPU1_LOG_DIR/train.log"
    fi
fi

if [ "$FOREGROUND" -eq 0 ]; then
    echo ""
    echo "查看训练状态:"
    echo "  nvidia-smi"
    echo "  ps aux | grep train_kitti"
    echo ""
    echo "停止训练:"
    echo "  bash stop_training.sh"
    echo ""
fi
