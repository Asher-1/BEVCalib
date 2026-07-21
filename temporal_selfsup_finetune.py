"""
Temporal Consistency Self-Supervised Fine-Tuning for BEVCalib

在 test_data 上无 GT 微调模型，利用标定任务的天然自监督信号：
1. 时序一致性: 同 trip 内标定结果应恒定
2. 不动点约束: 迭代推理应收敛
3. 平滑先验: 连续帧预测不应跳变

Usage:
    python temporal_selfsup_finetune.py \
        --model_dir logs/all_training_data_c1/model_small_5deg_c1_v67_quick20_S1/... \
        --checkpoint checkpoint/ckpt_best_dual.pth \
        --test_data /mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1 \
        --train_data /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data_c1 \
        --epochs 5 --lr 1e-5 --finetune_layers last2
"""
import argparse
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))


class TemporalConsistencyLoss(nn.Module):
    """
    Self-supervised loss based on temporal consistency of predictions.

    Within a single trip, the camera-LiDAR calibration is FIXED.
    Therefore, any variance in model predictions across frames in the same trip
    represents either noise or domain bias.

    Loss = Var(predictions within trip) + λ_fp * fixpoint_loss
    """

    def __init__(self, lambda_variance=1.0, lambda_fixpoint=0.5, lambda_smooth=0.2):
        super().__init__()
        self.lambda_variance = lambda_variance
        self.lambda_fixpoint = lambda_fixpoint
        self.lambda_smooth = lambda_smooth

    def forward(self, predictions_euler, frame_indices=None):
        """
        Args:
            predictions_euler: (N, 3) euler angles in radians from same trip
            frame_indices: optional (N,) temporal ordering for smoothness
        """
        losses = {}

        mean_pred = predictions_euler.mean(dim=0, keepdim=True)
        variance = ((predictions_euler - mean_pred) ** 2).mean()
        losses['temporal_variance'] = self.lambda_variance * variance

        target_zero = torch.zeros_like(mean_pred)
        fixpoint = (mean_pred ** 2).mean()
        losses['fixpoint'] = self.lambda_fixpoint * fixpoint

        if frame_indices is not None and len(predictions_euler) > 1:
            sorted_indices = torch.argsort(frame_indices)
            sorted_preds = predictions_euler[sorted_indices]
            diff = sorted_preds[1:] - sorted_preds[:-1]
            smooth = (diff ** 2).mean()
            losses['smoothness'] = self.lambda_smooth * smooth

        total = sum(losses.values())
        losses['total'] = total
        return total, losses


class FixpointLoss(nn.Module):
    """
    Iterative fixpoint constraint:
    If the model's output is applied to the input and the model is run again,
    the second output should be zero (no further correction needed).

    This loss requires a second forward pass.
    """

    def __init__(self, weight=0.3):
        super().__init__()
        self.weight = weight

    def forward(self, model_output_euler, model_fn, current_input):
        """
        model_output_euler: (B, 3) - first pass prediction
        model_fn: callable that takes corrected input and returns new prediction
        current_input: the transformation used as input to first pass
        """
        corrected_euler = model_fn(current_input, model_output_euler)
        return self.weight * (corrected_euler ** 2).mean()


def create_finetune_optimizer(model, args):
    """Create optimizer that only updates specified layers."""
    if args.finetune_layers == 'all':
        params = model.parameters()
    elif args.finetune_layers == 'last2':
        params = []
        for name, param in model.named_parameters():
            if any(k in name for k in ['head', 'decoder.layers.3', 'decoder.layers.2',
                                        'corr_head', 'rocr', 'pitch_branch']):
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False
        print(f"  Fine-tuning {len(params)} parameter groups (last 2 layers + heads)")
    elif args.finetune_layers == 'head_only':
        params = []
        for name, param in model.named_parameters():
            if any(k in name for k in ['head', 'corr_head', 'rocr', 'pitch_branch']):
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False
        print(f"  Fine-tuning {len(params)} parameter groups (heads only)")
    else:
        raise ValueError(f"Unknown finetune_layers: {args.finetune_layers}")

    return optim.AdamW(params, lr=args.lr, weight_decay=1e-4)


def run_selfsup_epoch(model, test_dataloader, criterion, optimizer, device, epoch, args):
    """Run one epoch of self-supervised training on test data."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(test_dataloader):
        if batch_idx >= args.max_batches_per_epoch:
            break

        imgs = batch['image'].to(device)
        pcs = batch['pointcloud'].to(device)
        init_T = batch['init_T'].to(device)
        intrinsic = batch['intrinsic'].to(device)
        masks = batch.get('mask', None)
        if masks is not None:
            masks = masks.to(device)
        frame_ids = batch.get('frame_id', None)

        output = model(imgs, pcs, init_T, intrinsic, masks)
        if isinstance(output, dict):
            quat = output.get('quat', output.get('rotation'))
        else:
            quat = output[0] if isinstance(output, (list, tuple)) else output

        euler = quat_to_euler(quat)

        loss, loss_dict = criterion(euler, frame_ids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            loss_str = ' '.join(f"{k}={v.item():.5f}" for k, v in loss_dict.items() if k != 'total')
            print(f"  [Epoch {epoch}] batch {batch_idx}: total={loss.item():.5f} {loss_str}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def run_supervised_epoch(model, train_dataloader, sup_criterion, optimizer, device, epoch, args):
    """Run one epoch of supervised training on train data (mixed training)."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx >= args.max_batches_per_epoch // 2:
            break

        imgs = batch['image'].to(device)
        pcs = batch['pointcloud'].to(device)
        gt_T = batch['gt_T'].to(device)
        init_T = batch['init_T'].to(device)
        intrinsic = batch['intrinsic'].to(device)

        output = model(imgs, pcs, init_T, intrinsic, None)
        if isinstance(output, dict):
            quat = output.get('quat', output.get('rotation'))
        else:
            quat = output[0] if isinstance(output, (list, tuple)) else output

        loss = sup_criterion(quat, gt_T)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def quat_to_euler(quat):
    """Quaternion (w,x,y,z) to euler (roll, pitch, yaw) in radians."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = torch.clamp(2*(w*y - z*x), -1, 1)
    pitch = torch.asin(sinp)
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return torch.stack([roll, pitch, yaw], dim=1)


def main():
    parser = argparse.ArgumentParser(description='Temporal Self-Supervised Fine-Tuning')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoint/ckpt_best_dual.pth')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--train_data', type=str, default='',
                        help='Optional: supervised data for mixed training')
    parser.add_argument('--target_width', type=int, default=960)
    parser.add_argument('--target_height', type=int, default=540)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--finetune_layers', type=str, default='last2',
                        choices=['all', 'last2', 'head_only'])
    parser.add_argument('--lambda_variance', type=float, default=1.0)
    parser.add_argument('--lambda_fixpoint', type=float, default=0.5)
    parser.add_argument('--lambda_smooth', type=float, default=0.2)
    parser.add_argument('--max_batches_per_epoch', type=int, default=100)
    parser.add_argument('--mixed_training', action='store_true',
                        help='Alternate between supervised (train) and self-sup (test)')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Temporal Consistency Self-Supervised Fine-Tuning")
    print("=" * 60)
    print(f"Model: {args.model_dir}/{args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Fine-tune layers: {args.finetune_layers}")
    print(f"Lambda: var={args.lambda_variance} fp={args.lambda_fixpoint} sm={args.lambda_smooth}")
    print(f"Mixed training: {args.mixed_training}")
    print("=" * 60)

    print("\n[NOTE] Full pipeline requires dataloader integration with the existing")
    print("CustomDataset class. This script demonstrates the loss functions and")
    print("training loop structure.")
    print("\nTo integrate:")
    print("1. Add to run_generalization_eval.py as a --tta_mode=selfsup option")
    print("2. Before evaluation, run N epochs of self-sup on target trip")
    print("3. Then evaluate with the adapted model")
    print("\nExpected workflow:")
    print("  for each test_trip:")
    print("    1. Load base model")
    print("    2. Fine-tune 3-5 epochs on this trip (no GT)")
    print("    3. Evaluate on this trip")
    print("    4. Report improvement over base model")

    criterion = TemporalConsistencyLoss(
        lambda_variance=args.lambda_variance,
        lambda_fixpoint=args.lambda_fixpoint,
        lambda_smooth=args.lambda_smooth
    )

    print(f"\nLoss components:")
    print(f"  - Temporal Variance (×{args.lambda_variance}): min Var(pred) within trip")
    print(f"  - Fixpoint (×{args.lambda_fixpoint}): min |mean(pred)|² → push to zero")
    print(f"  - Smoothness (×{args.lambda_smooth}): min |pred[t] - pred[t-1]|²")

    print(f"\n[READY] Integration with existing training infrastructure needed.")
    print(f"Key functions implemented:")
    print(f"  - TemporalConsistencyLoss: self-supervised loss")
    print(f"  - FixpointLoss: iterative consistency loss")
    print(f"  - create_finetune_optimizer: selective layer unfreezing")
    print(f"  - run_selfsup_epoch: training loop")
    print(f"  - run_supervised_epoch: mixed training support")


if __name__ == '__main__':
    main()
