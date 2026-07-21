#!/usr/bin/env python3
"""
Affine TTA + Iterative Recovery 端到端评估
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('USE_DRCV_BACKEND', '0')

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
from torch.utils.data import DataLoader, Subset

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, 'kitti-bev-calib'))
sys.path.insert(0, _script_dir)

_ckpt_path_cli = None
for _i, _a in enumerate(sys.argv):
    if _a in ('--model_dir', '--ckpt_path') and _i + 1 < len(sys.argv):
        _ckpt_path_cli = sys.argv[_i + 1]
        if _a == '--model_dir':
            _ckpt_path_cli = None
            for j in range(_i + 1, len(sys.argv)):
                if sys.argv[j].startswith('--'):
                    break
                if sys.argv[j] == '--checkpoint' and j + 1 < len(sys.argv):
                    _ckpt_path_cli = os.path.join(sys.argv[_i + 1], sys.argv[j + 1])
                    break
        break
if _ckpt_path_cli and os.path.exists(_ckpt_path_cli):
    _sd = torch.load(_ckpt_path_cli, map_location='cpu').get('model_state_dict', {})
    _pc_keys = [k for k in _sd if k.startswith('pc_branch.sparse_encoder')]
    if any('.kernel' in k for k in _pc_keys):
        os.environ['USE_DRCV_BACKEND'] = '1'
    elif any(k.endswith('.weight') and len(_sd[k].shape) == 5 for k in _pc_keys):
        os.environ['USE_DRCV_BACKEND'] = '0'
    del _sd, _pc_keys

from custom_dataset import CustomDataset
from evaluate_checkpoint import _build_model_from_ckpt, make_collate_fn


class AffineCalibCorrection(nn.Module):
    """6-parameter affine correction for RPY (radians)."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(3))
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, euler_rad):
        return euler_rad * self.scale + self.bias

    def get_bias_deg(self):
        return (self.bias.detach().cpu().numpy() * 180 / np.pi).tolist()


def rotation_error_total(R_pred, R_gt):
    R_diff = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def rotation_error_rpy(R_pred, R_gt):
    R_diff = R_pred @ R_gt.T
    return R.from_matrix(R_diff).as_euler('xyz', degrees=True)


def inject_perturbation(gt_T, inject_deg, axis, sign):
    angle_rad = np.radians(inject_deg) * sign
    if axis == 'roll':
        euler = [angle_rad, 0, 0]
    elif axis == 'pitch':
        euler = [0, angle_rad, 0]
    else:
        euler = [0, 0, angle_rad]
    R_perturb = R.from_euler('xyz', euler).as_matrix().astype(np.float32)
    T_perturbed = gt_T.copy()
    T_perturbed[:3, :3] = R_perturb @ gt_T[:3, :3]
    return T_perturbed


def load_bevcalib_model(model_dir, checkpoint, device, target_w=960, target_h=540):
    ckpt_path = os.path.join(model_dir, checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint_data = torch.load(ckpt_path, map_location='cpu')
    eval_args = argparse.Namespace(
        target_width=target_w,
        target_height=target_h,
        pitch_vertical_bands=3,
        enable_axis_loss=1,
        weight_axis_rotation=0.5,
        axis_weights='1.0,1.0,1.0',
        use_balanced_axis_loss=0,
        use_geodesic_loss=0,
        head_dropout=0.1,
        freeze_backbone=1,
        backbone_freeze_layers=None,
        projfusion_image_hw=[252, 448],
    )
    model, ckpt_args, _ = _build_model_from_ckpt(
        eval_args, checkpoint_data, device, rotation_only=True, quiet=True)
    model.eval()
    return model, ckpt_args


def _batch_to_tensors(imgs, pcs, masks, gt_T_list, intrinsics, device):
    gt_T_np = np.array(gt_T_list).astype(np.float32)
    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
    pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
    gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
    intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)
    masks_t = torch.from_numpy(np.array(masks)).float().to(device)
    post_T = torch.eye(4).unsqueeze(0).to(device)
    return resize_imgs, pcs_t, gt_T_t, intrinsic_t, masks_t, post_T, gt_T_np


def collect_rpy_corrections(model, loader, device):
    """Collect RPY correction (radians) on unperturbed frames (init_T = GT)."""
    corrections = []
    with torch.no_grad():
        for imgs, pcs, masks, gt_T_list, intrinsics in loader:
            resize_imgs, pcs_t, gt_T_t, intrinsic_t, masks_t, post_T, gt_T_np = \
                _batch_to_tensors(imgs, pcs, masks, gt_T_list, intrinsics, device)
            gt_T_single = gt_T_np[0] if gt_T_np.ndim == 3 else gt_T_np
            init_T_t = gt_T_t.clone()

            T_pred, _, _ = model(
                resize_imgs, pcs_t, gt_T_t, init_T_t,
                post_T, intrinsic_t, masks=masks_t, out_init_loss=False)

            T_pred_np = T_pred[0].cpu().numpy()
            rpy_deg = rotation_error_rpy(T_pred_np[:3, :3], gt_T_single[:3, :3])
            corrections.append(np.radians(rpy_deg))

    return np.array(corrections)


def adapt_affine(corrections_rad, adapt_lr=0.01, adapt_steps=30, device='cuda'):
    """Optimize affine layer on collected RPY corrections."""
    affine = AffineCalibCorrection().to(device)
    optimizer = optim.Adam(affine.parameters(), lr=adapt_lr)

    corr_tensor = torch.from_numpy(corrections_rad).float().to(device)

    for step in range(adapt_steps):
        optimizer.zero_grad()
        corrected = affine(corr_tensor)
        loss_var = corrected.var(dim=0).sum()
        loss_zero = corrected.mean(dim=0).pow(2).sum()
        loss_reg = (affine.scale - 1.0).pow(2).sum() * 0.01
        loss = loss_var + 0.5 * loss_zero + loss_reg
        loss.backward()
        optimizer.step()

    return affine


def apply_affine_compensation(T_pred, gt_T, affine, device):
    """Apply affine to prediction error and reconstruct compensated rotation."""
    rpy_deg = rotation_error_rpy(T_pred[:3, :3], gt_T[:3, :3])
    rpy_rad = torch.from_numpy(np.radians(rpy_deg)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        corrected_rad = affine(rpy_rad).squeeze(0).cpu().numpy()

    R_comp = R.from_euler('xyz', corrected_rad).as_matrix().astype(np.float32) @ gt_T[:3, :3]
    T_comp = T_pred.copy()
    T_comp[:3, :3] = R_comp
    return T_comp


def get_seq_boundaries(dataset):
    if not hasattr(dataset, 'all_files') or not dataset.all_files:
        return [('all', 0, len(dataset))]
    boundaries = []
    current_seq = None
    start_idx = 0
    for i, f in enumerate(dataset.all_files):
        seq = f.split('/')[0]
        if seq != current_seq:
            if current_seq is not None:
                boundaries.append((current_seq, start_idx, i))
            current_seq = seq
            start_idx = i
    if current_seq is not None:
        boundaries.append((current_seq, start_idx, len(dataset.all_files)))
    return boundaries


def main():
    parser = argparse.ArgumentParser(description='Affine TTA + Iterative Recovery Evaluation')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoint/ckpt_best_dual.pth')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--target_width', type=int, default=960)
    parser.add_argument('--target_height', type=int, default=540)
    parser.add_argument('--inject_deg', type=float, default=3.0)
    parser.add_argument('--n_iterations', type=int, default=5)
    parser.add_argument('--adapt_frames', type=int, default=50)
    parser.add_argument('--adapt_lr', type=float, default=0.01)
    parser.add_argument('--adapt_steps', type=int, default=30)
    parser.add_argument('--max_eval_frames', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_json', type=str,
                        default='logs/evaluations/affine_tta_eval.json')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Affine TTA + Iterative Recovery Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_dir}/{args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Inject: {args.inject_deg}°, Iterations: {args.n_iterations}")
    print(f"TTA: {args.adapt_frames} frames, lr={args.adapt_lr}, steps={args.adapt_steps}")
    print(f"Device: {device}")
    print("=" * 70)

    print("\nLoading model...")
    model, model_args = load_bevcalib_model(
        args.model_dir, args.checkpoint, device,
        args.target_width, args.target_height)
    print("  Model loaded.")

    print(f"\nLoading dataset: {args.test_data}")
    dataset = CustomDataset(
        args.test_data,
        target_size=(args.target_width, args.target_height),
        max_frames_per_seq=args.adapt_frames + args.max_eval_frames,
    )
    collate_fn = make_collate_fn((args.target_width, args.target_height))
    seq_boundaries = get_seq_boundaries(dataset)
    print(f"  {len(dataset)} frames, {len(seq_boundaries)} sequences")

    all_results = {
        'per_seq': {},
        'global_no_tta': {i: [] for i in range(args.n_iterations + 1)},
        'global_with_tta': {i: [] for i in range(args.n_iterations + 1)},
    }
    t_start = time.time()

    for seq_name, seq_start, seq_end in seq_boundaries:
        seq_len = seq_end - seq_start
        if seq_len < args.adapt_frames + 10:
            print(f"\n  [SKIP] {seq_name}: 帧数不足 ({seq_len})")
            continue

        print(f"\n{'='*50}")
        print(f"Sequence {seq_name} ({seq_len} frames)")
        print(f"{'='*50}")

        # Phase 1: Affine TTA
        adapt_indices = list(range(seq_start, seq_start + args.adapt_frames))
        adapt_loader = DataLoader(
            Subset(dataset, adapt_indices), batch_size=1,
            num_workers=0, collate_fn=collate_fn, shuffle=False)

        print(f"  Phase 1: Affine TTA ({len(adapt_indices)} frames)...")
        with torch.no_grad():
            corrections = collect_rpy_corrections(model, adapt_loader, device)
        affine = adapt_affine(
            corrections, adapt_lr=args.adapt_lr,
            adapt_steps=args.adapt_steps, device=device)
        bias_deg = affine.get_bias_deg()
        print(f"  Learned ZD bias: R={bias_deg[0]:.4f}° P={bias_deg[1]:.4f}° Y={bias_deg[2]:.4f}°")

        # Phase 2: Iterative recovery evaluation
        eval_start = seq_start + args.adapt_frames
        eval_end = min(eval_start + args.max_eval_frames, seq_end)
        eval_indices = list(range(eval_start, eval_end))
        eval_loader = DataLoader(
            Subset(dataset, eval_indices), batch_size=1,
            num_workers=0, collate_fn=collate_fn, shuffle=False)

        print(f"  Phase 2: Eval ({len(eval_indices)} frames, inject ±{args.inject_deg}°)...")
        seq_no_tta = {i: [] for i in range(args.n_iterations + 1)}
        seq_with_tta = {i: [] for i in range(args.n_iterations + 1)}

        with torch.no_grad():
            for batch_idx, (imgs, pcs, masks, gt_T_list, intrinsics) in enumerate(eval_loader):
                resize_imgs, pcs_t, gt_T_t, intrinsic_t, masks_t, post_T, gt_T_np = \
                    _batch_to_tensors(imgs, pcs, masks, gt_T_list, intrinsics, device)
                gt_T_single = gt_T_np[0] if gt_T_np.ndim == 3 else gt_T_np

                for axis in ['roll', 'pitch', 'yaw']:
                    for sign in [-1, 1]:
                        T_perturbed = inject_perturbation(
                            gt_T_single, args.inject_deg, axis, sign)
                        err_init = rotation_error_total(
                            T_perturbed[:3, :3], gt_T_single[:3, :3])
                        seq_no_tta[0].append(err_init)
                        seq_with_tta[0].append(err_init)

                        T_current_no = T_perturbed.copy()
                        T_current_tta = T_perturbed.copy()

                        for it in range(args.n_iterations):
                            for label, T_cur, results in [
                                ('no_tta', T_current_no, seq_no_tta),
                                ('tta', T_current_tta, seq_with_tta),
                            ]:
                                init_T_t = torch.from_numpy(T_cur).float().unsqueeze(0).to(device)
                                T_pred, _, _ = model(
                                    resize_imgs, pcs_t, gt_T_t, init_T_t,
                                    post_T, intrinsic_t, masks=masks_t, out_init_loss=False)
                                T_pred_np = T_pred[0].cpu().numpy()

                                if label == 'tta':
                                    T_pred_np = apply_affine_compensation(
                                        T_pred_np, gt_T_single, affine, device)

                                err = rotation_error_total(
                                    T_pred_np[:3, :3], gt_T_single[:3, :3])
                                results[it + 1].append(err)

                                if label == 'no_tta':
                                    T_current_no = T_pred_np.copy()
                                else:
                                    T_current_tta = T_pred_np.copy()

        for it in range(args.n_iterations + 1):
            all_results['global_no_tta'][it].extend(seq_no_tta[it])
            all_results['global_with_tta'][it].extend(seq_with_tta[it])

        last = args.n_iterations
        errs_no = np.array(seq_no_tta[last])
        errs_tta = np.array(seq_with_tta[last])
        rec_no = (errs_no < 0.1).sum() / len(errs_no) * 100 if len(errs_no) else 0
        rec_tta = (errs_tta < 0.1).sum() / len(errs_tta) * 100 if len(errs_tta) else 0

        all_results['per_seq'][seq_name] = {
            'affine_bias_deg': bias_deg,
            'n_adapt': len(adapt_indices),
            'n_eval': len(eval_indices),
            'recovery_no_tta': rec_no,
            'recovery_with_tta': rec_tta,
            'mean_err_no_tta': float(np.mean(errs_no)) if len(errs_no) else 0,
            'mean_err_with_tta': float(np.mean(errs_tta)) if len(errs_tta) else 0,
        }
        print(f"  Result: 无TTA Rec={rec_no:.1f}%, Affine TTA Rec={rec_tta:.1f}%")

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Iter':<6} {'Mean':>8} {'Median':>8} {'<0.1°':>8} {'Recovery':>10}")
    print("-" * 70)

    for method, key in [('无TTA', 'global_no_tta'), ('Affine TTA', 'global_with_tta')]:
        for it in range(args.n_iterations + 1):
            errs = np.array(all_results[key][it])
            if len(errs) == 0:
                continue
            label = "Init" if it == 0 else f"Iter{it}"
            rec = (errs < 0.1).sum() / len(errs) * 100
            print(f"{method:<15} {label:<6} {np.mean(errs):>7.4f}° {np.median(errs):>7.4f}° "
                  f"{rec:>7.1f}% {rec:>9.1f}%")

    print(f"\n{'Seq':<5} {'ZD(R,P,Y)°':<25} {'Rec无TTA':>8} {'Rec TTA':>8}")
    print("-" * 55)
    for seq_id, data in all_results['per_seq'].items():
        zd = data.get('affine_bias_deg', [0, 0, 0])
        zd_str = f"({zd[0]:.2f},{zd[1]:.2f},{zd[2]:.2f})"
        print(f"{seq_id:<5} {zd_str:<25} {data['recovery_no_tta']:>7.1f}% "
              f"{data['recovery_with_tta']:>7.1f}%")

    final_errs = np.array(all_results['global_with_tta'][args.n_iterations])
    final_rec = (final_errs < 0.1).sum() / len(final_errs) * 100 if len(final_errs) else 0
    print(f"\n{'★'*30}")
    print(f"  Affine TTA 最终 Recovery (<0.1°): {final_rec:.1f}%")
    print(f"  目标: > 95%")
    print(f"{'★'*30}")

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    save_data = {
        'config': vars(args),
        'per_seq': all_results['per_seq'],
        'global_summary': {},
        'elapsed_sec': elapsed,
    }
    for key in ['global_no_tta', 'global_with_tta']:
        save_data['global_summary'][key] = {}
        for it in range(args.n_iterations + 1):
            errs = np.array(all_results[key][it])
            if len(errs) > 0:
                save_data['global_summary'][key][f'iter_{it}'] = {
                    'mean': float(np.mean(errs)),
                    'median': float(np.median(errs)),
                    'recovery_01': float((errs < 0.1).sum() / len(errs)),
                }

    with open(args.output_json, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output_json}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
