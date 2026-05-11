#!/bin/bash
# E2: Mount Jitter Augmentation (isolated ablation)
#
# Tests mount jitter augmentation WITHOUT contrastive head or BEV instance norm.
# Mount jitter applies random extrinsic rotation/translation offsets to GT
# extrinsics BEFORE perturbation generation, simulating diverse camera
# installations during training.
#
# Parameters: prob=0.3, rot_sigma=0.5°, trans_sigma=0.01m
# These are conservative — enough to simulate slight mounting variations
# without destroying the geometric alignment signal.
#
# Baseline: D1 (layer-wise LR + pitch branch) — adds only mount jitter
# Expected: 10-15% improvement on cross-vehicle sequences

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_E2_mount_jitter" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --rotation_only \
    --enable_axis_loss \
    --weight_axis_rotation 0.5 \
    --axis_weights "1.0,5.0,1.0" \
    --backbone_lr_scale 1.0 \
    --backbone_warmup_epochs 10 \
    --layer_wise_lr_decay 0.65 \
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
    --use_pitch_branch 1 \
    --pitch_aux_weight 0.3 \
    --fuser_type diff \
    --augment_mount_jitter_prob 0.3 \
    --augment_mount_jitter_rot_sigma 0.5 \
    --augment_mount_jitter_trans_sigma 0.01 \
    --ddp 8 \
    "$@"
