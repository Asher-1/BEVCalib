#!/bin/bash
# B1: Full backbone LR (backbone_lr_scale=1.0)
#
# Diagnosis: img_branch gradient is 26x weaker than rotation_pred at Epoch 19.
# With backbone_lr_scale=0.5, effective update ratio is 1/52.
# This experiment removes the backbone LR discount entirely.
#
# Changes from v25:
#   backbone_lr_scale: 0.5 → 1.0
#   use_mlp_head: 0 → 1 (fix)
#   axis_weights: 1.0,3.0,1.0 → 1.0,5.0,1.0
#
# Risk: backbone may overfit if LR too high. Monitor train/val gap.
# Expected: img_branch gradients ≈ 2x improvement, better feature learning.

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_B1_bb_lr_full" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --rotation_only \
    --enable_axis_loss \
    --weight_axis_rotation 0.5 \
    --axis_weights "1.0,5.0,1.0" \
    --backbone_lr_scale 1.0 \
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
