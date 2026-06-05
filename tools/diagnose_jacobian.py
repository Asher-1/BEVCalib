#!/usr/bin/env python3
"""
Jacobian Diagnostic for BEVCalib v35/v36 Models.

Measures: J_axis = Δ(correction) / Δ(bias)
- J ≈ 1.0: perfect adaptive response
- J ≈ 0.0: shortcut learning
- J < 0.0: anti-adaptive

Usage:
    python tools/diagnose_jacobian.py --ckpt_path <path> [--n_batches 5]
"""

import torch
import numpy as np
import argparse
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib'))

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('USE_DRCV_BACKEND', '0')


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Euler (degrees) → 3x3 rotation matrix (Rz·Ry·Rx)."""
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R):
    """3x3 rotation matrix → Euler angles (degrees) [roll, pitch, yaw]."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])


def perturb_T_euler(T_base_np, delta_rpy_deg):
    """Apply Euler perturbation to a 4×4 transform. Returns new 4×4."""
    T_out = T_base_np.copy()
    dR = euler_to_rotation_matrix(*delta_rpy_deg)
    T_out[:3, :3] = dR @ T_base_np[:3, :3]
    return T_out


def infer_target_size_from_checkpoint(state_dict, ckpt_args):
    """Infer (height, width) from checkpoint grid_uv or saved args."""
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)
    th = ckpt_args.get('target_height') if ckpt_args else None
    tw = ckpt_args.get('target_width') if ckpt_args else None
    if th and tw:
        return int(th), int(tw)

    grid_key = 'img_branch.cam_pos.grid_uv'
    if grid_key in state_dict:
        fh, fw, _ = state_dict[grid_key].shape
        # Cam2BEVQuery downsamples image by 8× for ray grid
        return int(fh * 8), int(fw * 8)
    return None, None


def build_model(args, ckpt_args, device):
    """Build BEVCalib model from checkpoint args."""
    from bev_calib import BEVCalib

    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)

    native_cross = ckpt_args.get('native_cross', 0)
    if isinstance(native_cross, bool):
        native_cross = int(native_cross)

    model = BEVCalib(
        deformable=ckpt_args.get('deformable', 0),
        bev_encoder=ckpt_args.get('bev_encoder', 1),
        img_shape=(args.target_height, args.target_width),
        rotation_only=ckpt_args.get('rotation_only', True),
        use_mlp_head=ckpt_args.get('use_mlp_head', 0),
        bev_pool_factor=ckpt_args.get('bev_pool_factor', 0),
        use_foundation_depth=ckpt_args.get('use_foundation_depth', 0),
        depth_model_type=ckpt_args.get('depth_model_type', 'midas_small'),
        voxel_mode=ckpt_args.get('voxel_mode', 'hard'),
        to_bev_mode=ckpt_args.get('to_bev_mode', 'concat'),
        scatter_reduce=ckpt_args.get('scatter_reduce', 'sum'),
        fuser_type=ckpt_args.get('fuser_type', 'diff'),
        bev_instance_norm=ckpt_args.get('bev_instance_norm', 0),
        cam2bev_mode=ckpt_args.get('cam2bev_mode', 'query'),
        backbone_type=ckpt_args.get('backbone_type', 'dinov2'),
        backbone_variant=ckpt_args.get('backbone_variant', 'dinov2-small'),
        explicit_tinit=ckpt_args.get('explicit_tinit', 0),
        tinit_sensitivity_weight=ckpt_args.get('tinit_sensitivity_weight', 0.0),
        iterative_refine=ckpt_args.get('iterative_refine', 0),
        native_cross=native_cross > 0,
        native_cross_pc_groups=ckpt_args.get('native_cross_pc_groups', 128),
        native_cross_n_harmonic=ckpt_args.get('native_cross_n_harmonic', 6),
        native_cross_n_layers=ckpt_args.get('native_cross_n_layers', 1),
        native_cross_dual_branch=ckpt_args.get('native_cross_dual_branch', True),
        native_cross_knn=ckpt_args.get('native_cross_knn', 8),
        native_cross_use_fps=ckpt_args.get('native_cross_use_fps', True),
        native_cross_use_pointgpt=ckpt_args.get('native_cross_use_pointgpt', 0) > 0,
        native_cross_pointgpt_ckpt=ckpt_args.get('native_cross_pointgpt_ckpt'),
        native_cross_pointgpt_config=ckpt_args.get('native_cross_pointgpt_config'),
        native_cross_pointgpt_max_depth=ckpt_args.get('native_cross_pointgpt_max_depth', 50.0),
        native_cross_extend_ratio=float(ckpt_args.get('native_cross_extend_ratio', 1.0) or 1.0),
    ).to(device)
    return model


@torch.no_grad()
def compute_jacobian_batch(model, imgs, pcs, masks, gt_T_np, intrinsics,
                           device, angle_range, n_probes=7):
    """Compute per-axis Jacobian for one batch.
    
    Strategy: for each axis, sweep additional bias from -angle_range to +angle_range,
    capture model's quaternion output, convert to Euler, and fit a line.
    """
    from tools import generate_single_perturbation_from_T

    B = imgs.shape[0]
    rotation_only = True

    # Hook to capture rotation output
    captured = {}

    def make_hook(key):
        def fn(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2:
                captured[key] = out[0].detach().cpu()  # quaternion (B, 4)
            elif isinstance(out, torch.Tensor) and out.shape[-1] == 4:
                captured[key] = out.detach().cpu()
        return fn

    # Find the rotation prediction module to hook
    if hasattr(model, 'native_cross') and model.native_cross:
        target_module = model.native_cross_head
    elif hasattr(model, 'rotation_pred'):
        target_module = model.rotation_pred
    else:
        target_module = None

    # Generate a base init_T (small perturbation from GT)
    base_init_T_np, _, _ = generate_single_perturbation_from_T(
        gt_T_np, angle_range_deg=2.0, trans_range=0.0,
        rotation_only=True, distribution='truncated_normal')

    axes = ['roll', 'pitch', 'yaw']
    bias_levels = np.linspace(-angle_range, angle_range, n_probes)
    axis_jacobians = {}

    for ax_idx, ax_name in enumerate(axes):
        corrections_per_bias = []

        for bias_deg in bias_levels:
            # Apply controlled bias on one axis
            biased_init_T_np = base_init_T_np.copy()
            for b in range(B):
                delta = [0.0, 0.0, 0.0]
                delta[ax_idx] = float(bias_deg)
                biased_init_T_np[b] = perturb_T_euler(base_init_T_np[b], delta)

            # Prepare tensors
            init_T = torch.from_numpy(biased_init_T_np).float().to(device)
            gt_T = torch.from_numpy(gt_T_np).float().to(device)
            post_cam2ego = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            cam_K = torch.from_numpy(np.array(intrinsics)).float().to(device)
            masks_t = torch.from_numpy(np.array(masks)).to(device) if masks is not None else None

            # Forward with hook
            hook_key = f"{ax_name}_{bias_deg:.2f}"
            hook = None
            if target_module is not None:
                hook = target_module.register_forward_hook(make_hook(hook_key))

            try:
                _ = model(imgs, pcs, gt_T, init_T, post_cam2ego, cam_K,
                         masks_t, out_init_loss=False)
            except Exception as e:
                print(f"    [WARN] Forward failed at bias={bias_deg:.1f}: {e}")
                if hook:
                    hook.remove()
                continue

            if hook:
                hook.remove()

            if hook_key in captured:
                rot_q = captured[hook_key].numpy()  # (B, 4)
                batch_corrections = []
                for b in range(B):
                    q = rot_q[b]
                    q_norm = q / (np.linalg.norm(q) + 1e-8)
                    w, x, y, z = q_norm
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
                    ])
                    euler = rotation_matrix_to_euler(R)
                    batch_corrections.append(euler[ax_idx])
                mean_correction = np.mean(batch_corrections)
                corrections_per_bias.append((float(bias_deg), mean_correction))
                captured.pop(hook_key, None)

        # Linear regression: correction = J * bias + offset
        if len(corrections_per_bias) >= 3:
            biases = np.array([x[0] for x in corrections_per_bias])
            corrections = np.array([x[1] for x in corrections_per_bias])
            coeffs = np.polyfit(biases, corrections, 1)
            axis_jacobians[ax_name] = float(coeffs[0])
        else:
            axis_jacobians[ax_name] = float('nan')

    return axis_jacobians


def main():
    parser = argparse.ArgumentParser(description="Jacobian Diagnostic for BEVCalib")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_root", type=str,
                        default="/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data")
    parser.add_argument("--target_height", type=int, default=None,
                        help="Image height (auto-detect from ckpt if omitted)")
    parser.add_argument("--target_width", type=int, default=None,
                        help="Image width (auto-detect from ckpt if omitted)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 recommended for diagnostics)")
    parser.add_argument("--angle_range", type=float, default=5.0,
                        help="Bias sweep range (±degrees)")
    parser.add_argument("--n_probes", type=int, default=7,
                        help="Number of bias levels per axis")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_batches", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  BEVCalib Jacobian Diagnostic")
    print("=" * 70)
    print(f"  Checkpoint : {args.ckpt_path}")
    print(f"  Bias sweep : ±{args.angle_range}° ({args.n_probes} probes)")
    print(f"  Averaging  : {args.n_batches} batches × BS={args.batch_size}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"\n  Loading checkpoint...")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    ckpt_args = checkpoint.get('args', {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)

    auto_h, auto_w = infer_target_size_from_checkpoint(state_dict, ckpt_args)
    if args.target_height is None:
        args.target_height = auto_h or 360
    if args.target_width is None:
        args.target_width = auto_w or 640
    print(f"  Image size : {args.target_width}×{args.target_height}")

    # Build model
    print(f"  Building model...")
    model = build_model(args, ckpt_args, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing {len(missing)} keys: {missing[:3]}...")
    model.eval()

    # Load dataset
    print(f"  Loading dataset from {args.data_root}...")
    from custom_dataset import CustomDataset
    target_size = (args.target_width, args.target_height)
    dataset = CustomDataset(
        args.data_root,
        target_size=target_size,
        max_frames_per_seq=50,
        pose_aware_sampling=True,
    )

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        imgs = [item[0] for item in batch]
        gt_T_to_camera = [item[2] for item in batch]
        intrinsics = [item[3] for item in batch]
        pcs, masks = [], []
        max_num_points = max(item[1].shape[0] for item in batch)
        for item in batch:
            pc = item[1]
            masks.append(np.concatenate([np.ones(pc.shape[0]),
                                         np.zeros(max_num_points - pc.shape[0])]))
            if pc.shape[0] < max_num_points:
                pc = np.concatenate([pc, np.full((max_num_points - pc.shape[0],
                                                  pc.shape[1]), 999999)])
            pcs.append(pc)
        return imgs, pcs, masks, gt_T_to_camera, intrinsics

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        collate_fn=collate_fn)

    print(f"  Dataset size: {len(dataset)} frames")

    # Run Jacobian
    all_jacobians = defaultdict(list)

    for i, batch_data in enumerate(loader):
        if i >= args.n_batches:
            break
        if batch_data is None:
            continue

        imgs_raw, pcs_raw, masks_raw, gt_T_raw, intrinsics_raw = batch_data[:5]

        # Prepare tensors like train_kitti.py does
        # imgs_raw is a list of PIL images or numpy arrays
        img_arrays = []
        for im in imgs_raw:
            if hasattr(im, 'convert'):  # PIL Image
                img_arrays.append(np.array(im))
            else:
                img_arrays.append(np.array(im))
        imgs = torch.from_numpy(np.stack(img_arrays)).permute(0, 3, 1, 2).float().to(device)
        pcs_np = np.array(pcs_raw)[:, :, :3]  # xyz_only
        pcs = torch.from_numpy(pcs_np).float().to(device)
        gt_T_np = np.array(gt_T_raw).astype(np.float32)
        masks_np = np.array(masks_raw) if masks_raw is not None else None

        print(f"\n  Batch {i+1}/{args.n_batches}...", end=" ")
        jac = compute_jacobian_batch(
            model, imgs, pcs, masks_np, gt_T_np, intrinsics_raw,
            device, args.angle_range, args.n_probes)

        for ax in ['roll', 'pitch', 'yaw']:
            val = jac.get(ax, float('nan'))
            if not np.isnan(val):
                all_jacobians[ax].append(val)
        print(f"J_roll={jac.get('roll',0):+.3f} J_pitch={jac.get('pitch',0):+.3f} J_yaw={jac.get('yaw',0):+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("  JACOBIAN SUMMARY")
    print("=" * 70)

    summary = {}
    for ax in ['roll', 'pitch', 'yaw']:
        vals = all_jacobians[ax]
        if vals:
            mean_j = float(np.mean(vals))
            std_j = float(np.std(vals))
            summary[ax] = {'mean': mean_j, 'std': std_j, 'n': len(vals)}

            if mean_j > 0.3:
                status = "ADAPTIVE"
            elif mean_j > 0.05:
                status = "WEAK"
            elif mean_j > -0.05:
                status = "SHORTCUT"
            else:
                status = "ANTI-ADAPTIVE"

            print(f"  J_{ax:5s} = {mean_j:+.4f} ± {std_j:.4f}  [{status}]")
        else:
            summary[ax] = {'mean': 0, 'std': 0, 'n': 0}
            print(f"  J_{ax:5s} = N/A")

    overall_j = np.mean([summary[ax]['mean'] for ax in ['roll', 'pitch', 'yaw']])
    print(f"\n  Overall J = {overall_j:+.4f}")

    if overall_j > 0.3:
        verdict = "ADAPTIVE - correction scales with bias (good)"
    elif overall_j > 0.05:
        verdict = "WEAK ADAPTIVE - some sensitivity, needs more training"
    elif overall_j > -0.05:
        verdict = "SHORTCUT LEARNING - constant output regardless of bias"
    else:
        verdict = "ANTI-ADAPTIVE - over-corrects or wrong direction"
    print(f"  Verdict: {verdict}")

    # Save
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt_path)), 'jacobian_diagnostic.json')
    result = {
        'checkpoint': os.path.abspath(args.ckpt_path),
        'config': {'angle_range': args.angle_range, 'n_probes': args.n_probes,
                   'n_batches': args.n_batches, 'batch_size': args.batch_size},
        'jacobians': summary,
        'overall_jacobian': float(overall_j),
        'verdict': verdict,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
