#!/bin/bash
# D1: Cosine LR + Layer-wise LR Decay
#
# Combines cosine annealing scheduler with layer-wise learning rate decay
# for SwinT backbone stages. Earlier layers (lower-level features) receive
# progressively smaller LR multipliers:
#   Layer 0 (stem): decay^4 = 0.65^4 ≈ 0.18x
#   Layer 1:        decay^3 = 0.65^3 ≈ 0.27x
#   Layer 2:        decay^2 = 0.65^2 ≈ 0.42x
#   Layer 3:        decay^1 = 0.65^1 ≈ 0.65x
#
# Also includes backbone warmup (10 epochs) and Pitch branch.

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_D1_layerwise_lr" \
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
    --ddp 8 \
    "$@"
