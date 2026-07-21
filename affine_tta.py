"""
Affine TTA (Test-Time Adaptation) for BEVCalib

在部署时，利用前 N 帧的自监督信号（时序一致性 + 不动点）
训练一个 6 参数的 Affine 层，消除 mounting-induced Zero-Drift。

Usage:
    python affine_tta.py \
        --model_dir logs/all_training_data_c1/model_small_5deg_c1_v67_quick20_S1/all_training_data_c1_scratch \
        --checkpoint checkpoint/ckpt_best_dual.pth \
        --test_data /mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1 \
        --target_width 960 --target_height 540 \
        --adapt_frames 100 --adapt_lr 0.01 --adapt_steps 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))


class AffineCalibLayer(nn.Module):
    """6-parameter affine correction: scale(3) + bias(3) for RPY output."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, quat_pred):
        """
        Apply affine correction to quaternion prediction.
        Converts quat→euler, applies scale+bias, converts back.
        """
        euler = self._quat_to_euler(quat_pred)
        corrected = euler * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)
        return self._euler_to_quat(corrected)

    def get_correction_deg(self):
        """Return current bias in degrees for monitoring."""
        return self.bias.detach().cpu().numpy() * 180.0 / np.pi

    @staticmethod
    def _quat_to_euler(quat):
        """Quaternion (w,x,y,z) to euler (roll, pitch, yaw) in radians."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = 2*(w*y - z*x)
        sinp = torch.clamp(sinp, -1, 1)
        pitch = torch.asin(sinp)
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return torch.stack([roll, pitch, yaw], dim=1)

    @staticmethod
    def _euler_to_quat(euler):
        """Euler (roll, pitch, yaw) to quaternion (w,x,y,z)."""
        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
        cr, sr = torch.cos(roll/2), torch.sin(roll/2)
        cp, sp = torch.cos(pitch/2), torch.sin(pitch/2)
        cy, sy = torch.cos(yaw/2), torch.sin(yaw/2)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return torch.stack([w, x, y, z], dim=1)


def load_model(model_dir, checkpoint, device, target_width=960, target_height=540):
    """Load trained BEVCalib model."""
    from train_kitti import build_model_from_args
    ckpt_path = os.path.join(model_dir, checkpoint)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = argparse.Namespace(**ckpt['args'])
    args.target_width = target_width
    args.target_height = target_height

    model = build_model_from_args(args)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()
    return model, args


def collect_predictions(model, dataloader, device, max_frames=200):
    """Run model on frames and collect quaternion predictions."""
    predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(predictions) >= max_frames:
                break
            imgs = batch['image'].to(device)
            pcs = batch['pointcloud'].to(device)
            init_T = batch['init_T'].to(device)
            intrinsic = batch['intrinsic'].to(device)
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.to(device)

            output = model(imgs, pcs, init_T, intrinsic, masks)
            if isinstance(output, dict):
                quat = output.get('quat', output.get('rotation'))
            else:
                quat = output[0] if isinstance(output, (list, tuple)) else output

            predictions.append(quat.cpu())

    return torch.cat(predictions, dim=0)


def temporal_consistency_loss(predictions):
    """
    Loss = variance of predictions within a trip.
    Ideal: all frames in same trip have identical calibration prediction.
    """
    mean_pred = predictions.mean(dim=0, keepdim=True)
    return ((predictions - mean_pred) ** 2).mean()


def fixpoint_loss(model, affine_layer, sample_batch, device):
    """
    Loss = ||model(apply_correction(pred)) - identity||
    A well-calibrated model with correct affine should output zero correction
    when input is already correct.
    """
    with torch.no_grad():
        imgs = sample_batch['image'].to(device)
        pcs = sample_batch['pointcloud'].to(device)
        init_T = sample_batch['init_T'].to(device)
        intrinsic = sample_batch['intrinsic'].to(device)
        masks = sample_batch.get('mask', None)
        if masks is not None:
            masks = masks.to(device)

        output = model(imgs, pcs, init_T, intrinsic, masks)
        if isinstance(output, dict):
            quat = output.get('quat', output.get('rotation'))
        else:
            quat = output[0] if isinstance(output, (list, tuple)) else output

    corrected_quat = affine_layer(quat.to(device))
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand_as(corrected_quat)
    euler = AffineCalibLayer._quat_to_euler(corrected_quat)
    return (euler ** 2).mean()


def run_affine_tta(model, dataloader, device, args):
    """
    Main TTA loop:
    1. Collect predictions from adaptation frames
    2. Optimize Affine layer to minimize temporal variance
    3. Return adapted model (model + affine)
    """
    print(f"[Affine TTA] Collecting {args.adapt_frames} frames for adaptation...")
    t0 = time.time()

    all_quats = []
    with torch.no_grad():
        frame_count = 0
        for batch in dataloader:
            if frame_count >= args.adapt_frames:
                break
            imgs = batch['image'].to(device)
            pcs = batch['pointcloud'].to(device)
            init_T = batch['init_T'].to(device)
            intrinsic = batch['intrinsic'].to(device)
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.to(device)

            output = model(imgs, pcs, init_T, intrinsic, masks)
            if isinstance(output, dict):
                quat = output.get('quat', output.get('rotation'))
            else:
                quat = output[0] if isinstance(output, (list, tuple)) else output

            all_quats.append(quat.detach())
            frame_count += quat.shape[0]

    raw_preds = torch.cat(all_quats, dim=0)[:args.adapt_frames]
    print(f"  Collected {raw_preds.shape[0]} predictions in {time.time()-t0:.1f}s")

    raw_euler = AffineCalibLayer._quat_to_euler(raw_preds)
    raw_euler_deg = raw_euler * 180 / np.pi
    print(f"  Raw prediction stats (deg):")
    print(f"    Roll:  mean={raw_euler_deg[:,0].mean():.4f} std={raw_euler_deg[:,0].std():.4f}")
    print(f"    Pitch: mean={raw_euler_deg[:,1].mean():.4f} std={raw_euler_deg[:,1].std():.4f}")
    print(f"    Yaw:   mean={raw_euler_deg[:,2].mean():.4f} std={raw_euler_deg[:,2].std():.4f}")

    affine = AffineCalibLayer().to(device)
    optimizer = optim.Adam(affine.parameters(), lr=args.adapt_lr)

    print(f"\n[Affine TTA] Optimizing 6 params for {args.adapt_steps} steps...")
    raw_preds_device = raw_preds.to(device)

    for step in range(args.adapt_steps):
        optimizer.zero_grad()

        corrected = affine(raw_preds_device)
        euler_corrected = AffineCalibLayer._quat_to_euler(corrected)
        loss_temp = (euler_corrected ** 2).mean()
        loss_temp += ((euler_corrected - euler_corrected.mean(dim=0, keepdim=True)) ** 2).mean() * 0.5

        reg_loss = (affine.scale - 1.0).pow(2).mean() * 0.01

        total_loss = loss_temp + reg_loss
        total_loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == args.adapt_steps - 1:
            bias_deg = affine.get_correction_deg()
            print(f"  Step {step:3d}: loss={total_loss.item():.6f} "
                  f"bias(R,P,Y)=[{bias_deg[0]:.4f}, {bias_deg[1]:.4f}, {bias_deg[2]:.4f}]° "
                  f"scale={affine.scale.detach().cpu().numpy()}")

    final_bias = affine.get_correction_deg()
    print(f"\n[Affine TTA] Adaptation complete!")
    print(f"  Learned ZD bias: Roll={final_bias[0]:.4f}° Pitch={final_bias[1]:.4f}° Yaw={final_bias[2]:.4f}°")
    print(f"  Learned scale: {affine.scale.detach().cpu().numpy()}")

    return affine


def evaluate_with_affine(model, affine, dataloader, device, inject_deg=3.0, n_iterations=3):
    """Evaluate model + affine layer with iterative inference."""
    from scipy.spatial.transform import Rotation as R

    results = {'iter0': [], 'iter1': [], 'iter2': [], 'iter3': []}

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            pcs = batch['pointcloud'].to(device)
            gt_T = batch['gt_T']
            init_T = batch['init_T'].to(device)
            intrinsic = batch['intrinsic'].to(device)
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.to(device)

            output = model(imgs, pcs, init_T, intrinsic, masks)
            if isinstance(output, dict):
                quat = output.get('quat', output.get('rotation'))
            else:
                quat = output[0] if isinstance(output, (list, tuple)) else output

            corrected = affine(quat)
            euler_deg = AffineCalibLayer._quat_to_euler(corrected) * 180 / np.pi
            results['iter1'].append(euler_deg.cpu().numpy())

    return results


def main():
    parser = argparse.ArgumentParser(description='Affine TTA for BEVCalib')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoint/ckpt_best_dual.pth')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--target_width', type=int, default=960)
    parser.add_argument('--target_height', type=int, default=540)
    parser.add_argument('--adapt_frames', type=int, default=100)
    parser.add_argument('--adapt_lr', type=float, default=0.01)
    parser.add_argument('--adapt_steps', type=int, default=50)
    parser.add_argument('--output_json', type=str, default='')
    parser.add_argument('--seq', type=str, default='', help='Specific sequence to adapt on')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model from {args.model_dir}/{args.checkpoint}")
    model, model_args = load_model(args.model_dir, args.checkpoint, device,
                                    args.target_width, args.target_height)

    print(f"\n{'='*60}")
    print(f"Affine TTA Configuration:")
    print(f"  Adaptation frames: {args.adapt_frames}")
    print(f"  Learning rate: {args.adapt_lr}")
    print(f"  Optimization steps: {args.adapt_steps}")
    print(f"  Test data: {args.test_data}")
    print(f"{'='*60}")

    print("\n[NOTE] Full TTA pipeline requires dataloader integration.")
    print("This script demonstrates the Affine TTA module and optimization logic.")
    print("Integration with run_generalization_eval.py is the next step.")


if __name__ == '__main__':
    main()
