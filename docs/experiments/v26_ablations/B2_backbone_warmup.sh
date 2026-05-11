#!/bin/bash
# B2: Backbone gradual warmup (backbone_lr_scale=1.0 + warmup=10 epochs)
#
# Same as B1 but with 10-epoch warmup: backbone LR starts at 1% and linearly
# increases to full backbone_lr over 10 epochs. This prevents early overfitting
# while still achieving full backbone participation.
#
# Changes from B1:
#   backbone_warmup_epochs: 0 → 10
#
# Expected: safer than B1, gradual unfreezing lets head stabilize first.

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_B2_bb_warmup" \
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
    --ddp 8 \
    "$@"
