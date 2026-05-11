#!/bin/bash
# V25 Refactored: All optimizations integrated
#
# This replaces the original v25 experiment with ALL identified fixes:
#
# FIXES from v25 diagnosis:
#   1. use_mlp_head=1       (v25 had Linear head due to use_mlp_head=0 bug)
#   2. backbone_lr_scale=1.0 (v25 used 0.5x, causing backbone undertraining)
#   3. backbone_warmup=10   (gradual unfreezing for stable training)
#
# NEW ARCHITECTURE:
#   4. Dual-flow Pitch branch (Z-aware BEV + 2D front-view gated fusion)
#   5. pitch_aux_weight=0.3  (auxiliary Pitch supervision)
#
# OPTIMIZED TRAINING:
#   6. layer_wise_lr_decay=0.65 (SwinT stage-wise LR: 0.18x→0.27x→0.42x→0.65x)
#   7. axis_weights=1.0,5.0,1.0 (5x Pitch emphasis in axis loss)
#   8. fuser_type=diff       (BEVDiffFuser: best from v24 ablation)
#
# DOMAIN ADAPTATION:
#   9. augment_mount_jitter_prob=0.3 (simulate diverse camera installations)
#      rotation_sigma=0.5°, translation_sigma=0.01m
#  10. bev_instance_norm=1           (remove fixed FOV activation patterns)
#  11. use_contrastive_extrinsic=1   (force BEV diff to encode geometric offset)
#
# PRESERVED from v25 (working well):
#  11. rotation_only, truncated_normal perturbation
#  12. All augmentation (pc jitter, dropout, color jitter, pitch flip/sign)
#  13. drop_path_rate=0.1, head_dropout=0.1
#  14. early_stopping_patience=30
#
# Expected improvement: 0.664° → estimated 0.15-0.25° mean rotation
# Target: Approach <0.1° on best sequences (Seq04, Seq09)

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v25_refactored" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --rotation_only \
    --enable_axis_loss \
    --weight_axis_rotation 0.5 \
    --axis_weights "1.0,5.0,1.0" \
    --fuser_type diff \
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
    --bev_instance_norm 1 \
    --augment_mount_jitter_prob 0.3 \
    --augment_mount_jitter_rot_sigma 0.5 \
    --augment_mount_jitter_trans_sigma 0.01 \
    --use_contrastive_extrinsic 1 \
    --contrastive_weight 0.1 \
    --ddp 8 \
    "$@"
