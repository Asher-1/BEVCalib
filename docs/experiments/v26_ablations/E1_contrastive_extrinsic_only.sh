#!/bin/bash
# E1: Contrastive Extrinsic Embedding (isolated ablation)
#
# Tests the contrastive extrinsic head WITHOUT other domain adaptation features
# (no mount jitter, no BEV instance norm). This isolates the effect of forcing
# the BEV difference map to encode geometric offset via:
#   1. RPY perturbation regression from diff embedding
#   2. In-batch contrastive loss on perturbation similarity
#
# Baseline: D1 (layer-wise LR + pitch branch) — adds only contrastive head
# Expected: diff features become extrinsic-aware, improving generalization 5-10%

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_E1_contrastive" \
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
    --use_contrastive_extrinsic 1 \
    --contrastive_weight 0.1 \
    --ddp 8 \
    "$@"
