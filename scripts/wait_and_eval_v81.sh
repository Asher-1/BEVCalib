#!/bin/bash
# wait_and_eval_v81.sh
# 监控 V81 训练完成 → 自动执行泛化评估 → ZD 补偿 → 生成报告
#
# Usage:
#   bash scripts/wait_and_eval_v81.sh

set -euo pipefail

BEVCALIB_ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
TRAIN_LOG="${BEVCALIB_ROOT}/logs/train_v81.log"
TRAIN_DIR="${BEVCALIB_ROOT}/logs/all_training_data_c1/model_small_5deg_c1_v81_iterative_jacobian_S1"
EVAL_CONFIG="${BEVCALIB_ROOT}/configs/c1_retrain/eval_generalization_c1_v81.yaml"
EVAL_LOG="${BEVCALIB_ROOT}/logs/eval_v81_auto.log"
ZD_LOG="${BEVCALIB_ROOT}/logs/zd_compensation_v81.log"
REPORT="${BEVCALIB_ROOT}/logs/p1p2_combined_report.txt"

CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV="bevcalib310"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${EVAL_LOG}"
}

# ============================================================
# Phase 1: 等待训练完成
# ============================================================
log "=========================================="
log "Phase 1: 等待 V81 S1 训练完成"
log "=========================================="

MAX_WAIT_HOURS=48
START_TIME=$(date +%s)
LAST_EPOCH=0

while true; do
    ELAPSED_S=$(( $(date +%s) - START_TIME ))
    ELAPSED_H=$(( ELAPSED_S / 3600 ))

    if [ $ELAPSED_H -ge $MAX_WAIT_HOURS ]; then
        log "⚠️ 超时 (${MAX_WAIT_HOURS}h)，强制进入评估阶段"
        break
    fi

    # 检查训练进程是否存活
    if ! pgrep -f "v81_iterative_jacobian" > /dev/null 2>&1; then
        log "训练进程已退出"
        break
    fi

    # 检查是否完成
    if [ -f "$TRAIN_LOG" ]; then
        if grep -q "Training complete" "$TRAIN_LOG" 2>/dev/null; then
            log "✅ 训练完成!"
            break
        fi
        if grep -q "Early stopping" "$TRAIN_LOG" 2>/dev/null; then
            log "✅ 训练早停!"
            break
        fi

        # 解析当前 epoch
        CUR_EPOCH=$(grep -oP 'Epoch \[\K\d+(?=/200\] completed)' "$TRAIN_LOG" 2>/dev/null | tail -1 || echo "0")
        if [ -n "$CUR_EPOCH" ] && [ "$CUR_EPOCH" -gt "$LAST_EPOCH" ]; then
            LAST_EPOCH=$CUR_EPOCH
            REMAINING=$((200 - CUR_EPOCH))
            ETA_MIN=$(( REMAINING * 4 ))  # ~4 min/epoch
            log "📊 Epoch ${CUR_EPOCH}/200 完成 (预计剩余 ~$(( ETA_MIN / 60 ))h$(( ETA_MIN % 60 ))m)"
        fi

        # 检查是否到达 200 epoch
        if [ "$LAST_EPOCH" -ge 200 ]; then
            log "✅ 达到 200 epochs!"
            break
        fi
    fi

    sleep 180  # 每 3 分钟检查一次
done

# 等待 checkpoint 写入完成
log "等待 checkpoint 写入..."
sleep 30

# 列出可用 checkpoints
log "可用 checkpoints:"
ls -lt "${TRAIN_DIR}/all_training_data_c1_scratch/checkpoint/" 2>/dev/null | tee -a "${EVAL_LOG}"

# ============================================================
# Phase 2: 泛化评估
# ============================================================
log ""
log "=========================================="
log "Phase 2: 启动泛化评估 (V81 vs V66)"
log "=========================================="

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$BEVCALIB_ROOT"

# 2a: 单步评估 (无迭代推理)
log ">>> 2a: 标准泛化评估..."
python run_generalization_eval.py \
    --config "$EVAL_CONFIG" \
    --parallel -1 \
    --eval_max_frames_per_seq 500 \
    2>&1 | tee -a "${EVAL_LOG}"

EVAL_EXIT=$?
log "标准评估完成 (exit code: ${EVAL_EXIT})"

# 2b: 带迭代推理评估 (测量 Recovery 改善)
log ">>> 2b: 迭代推理评估 (iter_steps=2)..."
python run_generalization_eval.py \
    --config "$EVAL_CONFIG" \
    --parallel -1 \
    --cf_bev_r_iter_steps 2 \
    --eval_max_frames_per_seq 500 \
    2>&1 | tee -a "${EVAL_LOG}"

ITER_EXIT=$?
log "迭代推理评估完成 (exit code: ${ITER_EXIT})"

# ============================================================
# Phase 3: ZD 在线补偿
# ============================================================
log ""
log "=========================================="
log "Phase 3: P2 ZD 在线补偿"
log "=========================================="

python eval_p1p2_combined.py \
    --zd_only \
    --ema_alpha 0.15 \
    --max_correction_deg 2.0 \
    --calibration_frames 100 \
    2>&1 | tee -a "${ZD_LOG}"

ZD_EXIT=$?
log "ZD 补偿完成 (exit code: ${ZD_EXIT})"

# ============================================================
# Phase 4: 汇总报告
# ============================================================
log ""
log "=========================================="
log "Phase 4: 汇总报告"
log "=========================================="

{
    echo "================================================================"
    echo "P1+P2 联合评估报告 — V81 (DINOv2 + Iterative + Jacobian + Pitch)"
    echo "================================================================"
    echo ""
    echo "训练配置: configs/c1_retrain/c1_v81_iterative_jacobian.yaml"
    echo "评估配置: configs/c1_retrain/eval_generalization_c1_v81.yaml"
    echo ""

    # 提取各模型的评估结果
    EVAL_BASE="${BEVCALIB_ROOT}/logs/evaluations/generalization_c1_v81"
    for label in c1-v81-best-dual c1-v81-best-medw c1-v81-best-val; do
        EXTRINSICS="${EVAL_BASE}/${label}/extrinsics_and_errors.txt"
        if [ -f "$EXTRINSICS" ]; then
            echo "--- ${label} ---"
            grep -A 20 "EVALUATION STATISTICS" "$EXTRINSICS" 2>/dev/null || echo "(结果解析中)"
            echo ""
        fi
    done

    echo "--- V66 基线 (对比) ---"
    V66_BASE="${BEVCALIB_ROOT}/logs/evaluations/generalization_c1_v66"
    for label in c1-v66-best-dual-baseline c1-v66-best-medw-baseline; do
        EXTRINSICS="${EVAL_BASE}/${label}/extrinsics_and_errors.txt"
        if [ -f "$EXTRINSICS" ]; then
            echo "--- ${label} ---"
            grep -A 20 "EVALUATION STATISTICS" "$EXTRINSICS" 2>/dev/null || echo "(结果解析中)"
            echo ""
        fi
    done

    echo "================================================================"
    echo "目标: Gen BEST < 0.1°"
    echo "================================================================"
} > "$REPORT" 2>&1

log "报告已保存: ${REPORT}"
cat "$REPORT" | tee -a "${EVAL_LOG}"

log ""
log "=========================================="
log "全部完成!"
log "=========================================="
