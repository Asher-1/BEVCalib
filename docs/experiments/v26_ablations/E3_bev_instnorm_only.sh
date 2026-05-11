#!/bin/bash
# E3: BEV Instance Normalization (isolated ablation)
#
# Tests BEV InstanceNorm2d WITHOUT mount jitter or contrastive head.
# InstanceNorm removes the fixed FOV activation pattern from camera BEV
# features, preserving local spatial variations (alignment signals)
# while discarding the global template that the model memorizes.
#
# Camera BEV features have been shown to vary by <0.04% across scenes,
# indicating the model memorizes a fixed FOV triangle rather than learning
# alignment residuals. InstanceNorm breaks this memorization.
#
# Baseline: D1 (layer-wise LR + pitch branch) — adds only BEV InstanceNorm
# Expected: 8-12% improvement on domain-shifted test sequences

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_E3_bev_instnorm" \
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
    --bev_instance_norm 1 \
    --ddp 8 \
    "$@"
