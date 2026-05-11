#!/bin/bash
# E4: Contrastive Weight Sweep — test contrastive_weight=0.3 (vs E1's 0.1)
#
# The contrastive extrinsic loss has two components:
#   - RPY regression loss (forces embedding to predict perturbation offset)
#   - In-batch contrastive loss (pulls similar perturbation embeddings together)
#
# E1 uses weight=0.1 (conservative). This experiment uses weight=0.3 to test
# whether a stronger contrastive signal leads to better domain generalization
# at the cost of potentially hurting main task convergence.
#
# If E4 > E1: contrastive signal is beneficial, push further
# If E4 < E1: weight=0.1 is optimal, contrastive conflicts with main loss

cd "$(dirname "$0")/../.."

FORCE_RERUN=1 bash train_universal.sh scratch \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --log_suffix "small_5deg_v26_E4_contrastive_w03" \
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
    --contrastive_weight 0.3 \
    --ddp 8 \
    "$@"
