#!/bin/bash
# D2: Maximum Pitch optimization — all improvements combined
#
# This is the "kitchen sink" experiment combining ALL Pitch-targeted improvements:
#   1. Dual-flow Pitch branch (Z-aware BEV + 2D front-view features)
#   2. Layer-wise LR decay (0.65) for fine-grained backbone control
#   3. Backbone warmup (10 epochs) for stable gradient flow
#   4. Higher Pitch axis weight (8.0 vs 5.0) for stronger Pitch loss signal
#   5. Higher pitch_aux_weight (0.5 vs 0.3) for auxiliary branch emphasis
#   6. MLP regression head
#
# Target: Pitch error < 0.1 degrees

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_D2_max_pitch" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --rotation_only \
    --enable_axis_loss \
    --weight_axis_rotation 0.5 \
    --axis_weights "1.0,8.0,1.0" \
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
    --pitch_aux_weight 0.5 \
    --ddp 8 \
    "$@"
