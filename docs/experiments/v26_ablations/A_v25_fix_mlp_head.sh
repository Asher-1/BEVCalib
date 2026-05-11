#!/bin/bash
# v25 Fix: Enable MLP regression head (was accidentally disabled: use_mlp_head=0)
#
# Critical finding: v25 training log shows use_mlp_head=0, meaning the 3-layer
# MLP head was NOT active. This script restarts with --use_mlp_head 1.
#
# Changes from v25_A1_z10:
#   use_mlp_head: 0 → 1  (CRITICAL FIX)
#   axis_weights: 1.0,3.0,1.0 → 1.0,5.0,1.0  (increase Pitch weight)
#
# Expected: MLP head adds capacity for fine-grained angle regression,
# especially Pitch which requires sub-degree precision.

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_A_mlp_fix" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --rotation_only \
    --enable_axis_loss \
    --weight_axis_rotation 0.5 \
    --axis_weights "1.0,5.0,1.0" \
    --backbone_lr_scale 0.5 \
    --warmup_epochs 5 \
    --drop_path_rate 0.1 \
    --head_dropout 0.1 \
    --perturb_distribution truncated_normal \
    --per_axis_prob 0.3 \
    --augment_pc_jitter 0.02 \
    --augment_pc_dropout 0.05 \
    --augment_color_jitter 0.15 \
    --augment_pitch_flip_prob 0.2 \
    --augment_pitch_flip_max_deg 3.0 \
    --augment_pitch_sign_flip_prob 0.15 \
    --early_stopping_patience 30 \
    --num_epochs 400 \
    --save_ckpt_per_epoches 40 \
    --use_mlp_head 1 \
    --ddp 8 \
    "$@"
