#!/bin/bash
# C1: Z-aware Pitch branch + B2 backbone warmup
#
# Enables the FrontViewPitchBranch which uses pre-projection BEV features
# (C*nZ = 1280 dims with 10 Z-voxels) to predict Pitch angles. These Z-rich
# features preserve height distribution information that the main pathway
# compresses via ProjectionHead (1280→128).
#
# Architecture addition:
#   pitch_branch: Linear(1280→128) + GELU + Linear(128→64) + GELU + Linear(64→1)
#   pitch_aux_weight: 0.3 (added to total_loss)
#
# Combined with B2 (full backbone LR + warmup) and MLP head fix.

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_C1_pitch_branch" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --rotation_only \
    --enable_axis_loss \
    --weight_axis_rotation 0.5 \
    --axis_weights "1.0,5.0,1.0" \
    --backbone_lr_scale 1.0 \
    --backbone_warmup_epochs 10 \
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
    --ddp 8 \
    "$@"
