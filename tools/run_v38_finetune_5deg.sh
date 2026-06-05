#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/v38_finetune_5deg_only.yaml}"
ARM1_CKPT="${ARM1_CKPT:-${ROOT}/logs/all_training_data/model_small_10deg_v38_native_cross_fleet_pointgpt_L20/all_training_data_scratch/checkpoint/ckpt_best_val.pth}"

if [[ ! -f "${ARM1_CKPT}" ]]; then
    echo "Arm1 checkpoint not found (v38 ±10° main must finish first):"
    echo "  ${ARM1_CKPT}"
    echo "Or set ARM1_CKPT=/path/to/ckpt_best_val.pth"
    exit 1
fi

echo "============================================"
echo "V38b: ±5° fine-tune from Arm1 best val"
echo "============================================"
echo "Arm1 ckpt: ${ARM1_CKPT}"
echo "Config   : ${CONFIG}"
echo "============================================"

TMP_CFG="$(mktemp /tmp/v38_finetune_5deg.XXXXXX.yaml)"
sed "s|^      pretrain_ckpt: .*|      pretrain_ckpt: ${ARM1_CKPT}|" "${CONFIG}" > "${TMP_CFG}"

cd "${ROOT}"
bash batch_train.sh "${TMP_CFG}" --force
rm -f "${TMP_CFG}"
