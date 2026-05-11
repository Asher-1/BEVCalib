#!/usr/bin/env python3
"""
Coarse-to-Fine 链式评估脚本

评估流程:
  1. 对每个测试样本施加 ±coarse_angle° 扰动 (模拟真实场景)
  2. Coarse 模型预测: init_T → T_coarse
  3. Fine 模型预测: T_coarse → T_fine (最终输出)
  4. 计算 T_fine vs GT 的误差

对比模式:
  - chain:     coarse → fine (完整链式)
  - coarse:    仅 coarse 模型 (对照)
  - fine-only: 仅 fine 模型 + ±fine_angle° 扰动 (fine 理论上限)

用法:
  python evaluate_coarse_fine.py \
    --coarse_ckpt logs/.../v24_B_diff_only/ckpt_best_val.pth \
    --fine_ckpt logs/.../v25_C1_fine_1deg/ckpt_best_val.pth \
    --coarse_angle 5 --fine_angle 1 \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2

  # 也可用 fine_angle=2 配合 v25_C2_fine_2deg
"""

import torch
import numpy as np
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
from bev_calib import BEVCalib
from tools import generate_single_perturbation_from_T
from visualization import compute_pose_errors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from evaluate_extrinsics import evaluate_sensor_extrinsic


def parse_args():
    parser = argparse.ArgumentParser(description="Coarse-to-Fine chain evaluation")
    parser.add_argument('--coarse_ckpt', type=str, required=True,
                        help="Path to coarse model checkpoint (±5° trained)")
    parser.add_argument('--fine_ckpt', type=str, required=True,
                        help="Path to fine model checkpoint (±1°/±2° trained)")
    parser.add_argument('--coarse_angle', type=float, default=5.0,
                        help="Coarse model perturbation range in degrees")
    parser.add_argument('--fine_angle', type=float, default=1.0,
                        help="Fine model training range in degrees")
    parser.add_argument('--coarse_trans', type=float, default=0.15,
                        help="Coarse model translation perturbation")
    parser.add_argument('--fine_trans', type=float, default=0.03,
                        help="Fine model translation perturbation")
    parser.add_argument('--dataset_root', type=str, required=True,
                        help="Path to test dataset")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory (default: auto-generate)")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_max_frames_per_seq', type=int, default=400)
    parser.add_argument('--target_width', type=int, default=640)
    parser.add_argument('--target_height', type=int, default=384)
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--xyz_only', type=int, default=1)
    parser.add_argument('--rotation_only', type=int, default=-1,
                        help="-1=auto from checkpoint, 0=joint, 1=rotation_only")
    parser.add_argument('--perturb_distribution', type=str, default='truncated_normal')
    parser.add_argument('--per_axis_prob', type=float, default=0.3)
    parser.add_argument('--per_axis_weights', type=str, default=None)
    return parser.parse_args()


def _auto_detect_backend(ckpt_path):
    """Detect spconv vs drcv backend from checkpoint keys."""
    sd = torch.load(ckpt_path, map_location='cpu').get('model_state_dict', {})
    pc_keys = [k for k in sd if k.startswith('pc_branch.sparse_encoder')]
    if any('.kernel' in k for k in pc_keys):
        os.environ['USE_DRCV_BACKEND'] = '1'
    elif any(k.endswith('.weight') and len(sd[k].shape) == 5 for k in pc_keys):
        os.environ.setdefault('USE_DRCV_BACKEND', '0')


def _load_model(ckpt_path, device, args, quiet=False):
    """Load model from checkpoint with auto-detection of all parameters."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

    from evaluate_checkpoint import _build_model_from_ckpt

    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')

    if args.rotation_only == -1:
        if 'rotation_only' in checkpoint:
            rotation_only = checkpoint['rotation_only']
        elif 'optimize_translation' in checkpoint:
            rotation_only = not checkpoint['optimize_translation']
        else:
            rotation_only = True
    else:
        rotation_only = args.rotation_only > 0

    model, ckpt_args, p = _build_model_from_ckpt(args, checkpoint, device, rotation_only, quiet=quiet)
    if not quiet:
        print(f"   Loaded: epoch={epoch}, rotation_only={rotation_only}, fuser={p['fuser_type']}")

    return model, rotation_only, epoch


def make_collate_fn(target_size):
    """Collate function for DataLoader."""
    import cv2
    def collate_fn(batch):
        imgs, pcs, masks, T, K = zip(*batch)
        resized_imgs = []
        for img in imgs:
            if img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            resized_imgs.append(img)
        return resized_imgs, pcs, masks, T, K
    return collate_fn


def evaluate_chain(args):
    """Main evaluation: coarse → fine chain."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Coarse-to-Fine 链式评估")
    print("=" * 80)
    print(f"  Coarse: {args.coarse_ckpt}")
    print(f"  Fine:   {args.fine_ckpt}")
    print(f"  Coarse angle: ±{args.coarse_angle}°, Fine angle: ±{args.fine_angle}°")
    print(f"  Dataset: {args.dataset_root}")
    print()

    _auto_detect_backend(args.coarse_ckpt)

    print("1. Loading coarse model...")
    coarse_model, coarse_rot_only, coarse_epoch = _load_model(
        args.coarse_ckpt, device, args, quiet=False)

    print("2. Loading fine model...")
    fine_model, fine_rot_only, fine_epoch = _load_model(
        args.fine_ckpt, device, args, quiet=False)

    print(f"\n3. Loading dataset: {args.dataset_root}")
    dataset = CustomDataset(
        data_folder=args.dataset_root,
        auto_detect=True,
        max_frames_per_seq=args.eval_max_frames_per_seq,
    )
    print(f"   Total samples: {len(dataset)}")

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                        collate_fn=collate_fn, shuffle=False)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        coarse_name = os.path.basename(os.path.dirname(args.coarse_ckpt))
        fine_name = os.path.basename(os.path.dirname(args.fine_ckpt))
        output_dir = os.path.join("logs/evaluations",
                                  f"coarse_fine_{coarse_name}_to_{fine_name}")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    errors_chain = {'rot_error': [], 'roll_error': [], 'pitch_error': [], 'yaw_error': [],
                    'trans_error': [], 'fwd_error': [], 'lat_error': [], 'ht_error': []}
    errors_coarse = {'rot_error': [], 'roll_error': [], 'pitch_error': [], 'yaw_error': [],
                     'trans_error': [], 'fwd_error': [], 'lat_error': [], 'ht_error': []}

    coarse_residuals = []  # track how much residual error coarse leaves for fine

    print(f"\n4. Running evaluation ({len(loader)} batches)...")
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(loader):
            gt_T_np = np.array(gt_T_to_camera).astype(np.float32)

            _paw = None
            if args.per_axis_weights:
                _paw = tuple(float(x) for x in args.per_axis_weights.split(','))

            init_T_np, _, _ = generate_single_perturbation_from_T(
                gt_T_np,
                angle_range_deg=args.coarse_angle,
                trans_range=args.coarse_trans,
                rotation_only=coarse_rot_only,
                distribution=args.perturb_distribution,
                per_axis_prob=args.per_axis_prob,
                per_axis_weights=_paw,
            )

            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_np = np.array(pcs)[:, :, :3] if args.xyz_only > 0 else np.array(pcs)
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
            init_T_t = torch.from_numpy(init_T_np).float().to(device)
            post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
            intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

            # Stage 1: Coarse prediction
            T_coarse, _, _ = coarse_model(
                resize_imgs, pcs_t, gt_T_t, init_T_t,
                post_cam2ego_T, intrinsic_t, masks=masks, out_init_loss=False)

            # Stage 2: Fine prediction (using coarse output as init)
            T_fine, _, _ = fine_model(
                resize_imgs, pcs_t, gt_T_t, T_coarse.detach(),
                post_cam2ego_T, intrinsic_t, masks=masks, out_init_loss=False)

            T_coarse_np = T_coarse.detach().cpu().numpy()
            T_fine_np = T_fine.detach().cpu().numpy()

            for i in range(len(gt_T_np)):
                err_chain = compute_pose_errors(T_fine_np[i], gt_T_np[i])
                err_coarse = compute_pose_errors(T_coarse_np[i], gt_T_np[i])

                for k in errors_chain:
                    errors_chain[k].append(err_chain[k])
                    errors_coarse[k].append(err_coarse[k])

                coarse_residuals.append(err_coarse['rot_error'])

            sample_count += len(gt_T_np)
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(loader)}, samples: {sample_count}")

    elapsed = time.time() - t0
    print(f"\n5. Evaluation complete ({elapsed:.1f}s, {sample_count} samples)")

    # Compute statistics
    def _stats(errors_dict):
        return {k: {'mean': np.mean(v), 'std': np.std(v),
                    'median': np.median(v), 'p90': np.percentile(v, 90),
                    'p95': np.percentile(v, 95)}
                for k, v in errors_dict.items()}

    stats_chain = _stats(errors_chain)
    stats_coarse = _stats(errors_coarse)

    # Write report
    report_path = os.path.join(output_dir, "coarse_fine_report.md")
    with open(report_path, 'w') as f:
        f.write("# Coarse-to-Fine Evaluation Report\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| Coarse model | `{os.path.basename(os.path.dirname(args.coarse_ckpt))}` |\n")
        f.write(f"| Fine model | `{os.path.basename(os.path.dirname(args.fine_ckpt))}` |\n")
        f.write(f"| Coarse range | ±{args.coarse_angle}° |\n")
        f.write(f"| Fine range | ±{args.fine_angle}° |\n")
        f.write(f"| Dataset | `{os.path.basename(args.dataset_root)}` |\n")
        f.write(f"| Samples | {sample_count} |\n")
        f.write(f"| Eval seed | {args.eval_seed} |\n\n")

        f.write("## Results\n\n")
        f.write("### Mean Rotation Errors (degrees)\n\n")
        f.write("| Metric | Coarse Only | Chain (C→F) | Improvement |\n")
        f.write("|--------|-------------|-----------------|-------------|\n")
        for k in ['rot_error', 'pitch_error', 'roll_error', 'yaw_error']:
            name = k.replace('_error', '').capitalize()
            c = stats_coarse[k]['mean']
            cf = stats_chain[k]['mean']
            imp = (c - cf) / c * 100 if c > 0 else 0
            f.write(f"| {name} | {c:.4f}° | {cf:.4f}° | {imp:+.1f}% |\n")

        f.write(f"\n### Coarse Residual Analysis\n\n")
        cr = np.array(coarse_residuals)
        f.write(f"- Mean coarse residual: {cr.mean():.4f}°\n")
        f.write(f"- Median coarse residual: {np.median(cr):.4f}°\n")
        f.write(f"- P90 coarse residual: {np.percentile(cr, 90):.4f}°\n")
        f.write(f"- Samples within ±{args.fine_angle}°: "
                f"{(cr < args.fine_angle).sum()}/{len(cr)} "
                f"({(cr < args.fine_angle).mean()*100:.1f}%)\n")
        f.write(f"- Samples exceeding fine range: "
                f"{(cr >= args.fine_angle).sum()}/{len(cr)} "
                f"({(cr >= args.fine_angle).mean()*100:.1f}%)\n\n")

        f.write("### Detailed Statistics\n\n")
        f.write("#### Chain (Coarse→Fine)\n\n")
        f.write("| Metric | Mean | Std | Median | P90 | P95 |\n")
        f.write("|--------|------|-----|--------|-----|-----|\n")
        for k in ['rot_error', 'pitch_error', 'roll_error', 'yaw_error']:
            name = k.replace('_error', '').capitalize()
            s = stats_chain[k]
            f.write(f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['median']:.4f} | {s['p90']:.4f} | {s['p95']:.4f} |\n")

        f.write(f"\n#### Coarse Only\n\n")
        f.write("| Metric | Mean | Std | Median | P90 | P95 |\n")
        f.write("|--------|------|-----|--------|-----|-----|\n")
        for k in ['rot_error', 'pitch_error', 'roll_error', 'yaw_error']:
            name = k.replace('_error', '').capitalize()
            s = stats_coarse[k]
            f.write(f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['median']:.4f} | {s['p90']:.4f} | {s['p95']:.4f} |\n")

        f.write(f"\n## Interpretation\n\n")
        chain_rot = stats_chain['rot_error']['mean']
        coarse_rot = stats_coarse['rot_error']['mean']
        if chain_rot < coarse_rot * 0.7:
            f.write("Strong improvement: Chain prediction significantly better than coarse alone.\n")
        elif chain_rot < coarse_rot * 0.9:
            f.write("Moderate improvement: Chain provides meaningful refinement.\n")
        else:
            f.write("Weak/no improvement: Fine model not effectively refining. "
                    "Possible causes:\n"
                    "- Coarse residual exceeds fine model training range\n"
                    "- Fine model overfitting to oracle distribution\n")

        within_range = (cr < args.fine_angle).mean() * 100
        if within_range < 80:
            f.write(f"\nWarning: Only {within_range:.0f}% of coarse residuals fall within "
                    f"fine model range (±{args.fine_angle}°). "
                    f"Consider using ±{args.fine_angle*2}° fine model instead.\n")

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Coarse only:  Mean Rot = {stats_coarse['rot_error']['mean']:.4f}°")
    print(f"  Chain (C→F):  Mean Rot = {stats_chain['rot_error']['mean']:.4f}°")
    imp = (stats_coarse['rot_error']['mean'] - stats_chain['rot_error']['mean']) / \
          stats_coarse['rot_error']['mean'] * 100
    print(f"  Improvement:  {imp:+.1f}%")
    print(f"\n  Coarse residual → fine input:")
    print(f"    Mean: {cr.mean():.4f}°, within ±{args.fine_angle}°: {within_range:.0f}%")
    print(f"\n  Report: {report_path}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == '__main__':
    args = parse_args()
    evaluate_chain(args)
