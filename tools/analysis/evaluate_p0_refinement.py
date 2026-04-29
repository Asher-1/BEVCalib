#!/usr/bin/env python3
"""
P0 推理增强评估：迭代细化 + 多帧时域滤波

基于 LCCNet/CalibRefine 的迭代细化思路和多帧聚合策略，
在不重新训练模型的前提下，通过纯推理时技术降低标定误差。

用法:
    python evaluate_p0_refinement.py \
        --ckpt_path logs/all_training_data/model_small_5deg_v20_v8recipe_pitch_wt3/..../ckpt_best_val.pth \
        --dataset_root /path/to/test_data_v2 \
        --use_full_dataset \
        --angle_range_deg 5.0 \
        --num_iterations 3 \
        --sequence_median
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import sys
import time
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

if '--use_drcv' in sys.argv:
    os.environ['USE_DRCV_BACKEND'] = '1'

if 'USE_DRCV_BACKEND' not in os.environ:
    _ckpt_path = None
    for _i, _a in enumerate(sys.argv):
        if _a == '--ckpt_path' and _i + 1 < len(sys.argv):
            _ckpt_path = sys.argv[_i + 1]
            break
    if _ckpt_path and os.path.exists(_ckpt_path):
        _sd = torch.load(_ckpt_path, map_location='cpu').get('model_state_dict', {})
        _pc_keys = [k for k in _sd if k.startswith('pc_branch.sparse_encoder')]
        if any('.kernel' in k for k in _pc_keys):
            os.environ['USE_DRCV_BACKEND'] = '1'
        elif any(k.endswith('.weight') and len(_sd[k].shape) == 5 for k in _pc_keys):
            os.environ['USE_DRCV_BACKEND'] = '0'
        del _sd, _pc_keys

from custom_dataset import CustomDataset
from bev_calib import BEVCalib
from tools import generate_single_perturbation_from_T
from visualization import compute_pose_errors
from scipy.spatial.transform import Rotation as R


def iterative_refinement_forward(model, imgs, pcs, gt_T, init_T, post_T, intrinsics,
                                 masks, num_iterations, rotation_only):
    """
    迭代细化推理：多次 forward，每次用上一轮的预测结果作为新的 init_T。

    网络预测的是 init_T 到 GT 的残差修正。第一轮修正大部分误差后，
    后续轮次在更小的残差上做精细校正，类似 LCCNet 的 iterative refinement。

    Returns:
        all_pred_Ts: list of (B,4,4) numpy arrays, 每轮迭代的预测结果
    """
    all_pred_Ts = []
    current_init_T = init_T.clone()

    for iteration in range(num_iterations):
        T_pred, _, _ = model(
            imgs, pcs, gt_T, current_init_T,
            post_T, intrinsics, masks=masks, out_init_loss=False
        )
        all_pred_Ts.append(T_pred.detach().cpu().numpy())
        if iteration < num_iterations - 1:
            current_init_T = T_pred.detach().clone()

    return all_pred_Ts


def rotation_matrix_median_svd(Rs):
    """
    Compute median rotation from N rotation matrices using element-wise median
    + SVD projection to nearest SO(3). Avoids Euler angle singularities.

    Args:
        Rs: (N, 3, 3) rotation matrices
    Returns:
        R_median: (3, 3) valid rotation matrix
    """
    R_med_raw = np.median(Rs, axis=0)
    U, _, Vt = np.linalg.svd(R_med_raw)
    R_median = U @ Vt
    if np.linalg.det(R_median) < 0:
        U[:, -1] *= -1
        R_median = U @ Vt
    return R_median


def quaternion_median(quats):
    """
    Median of quaternions with sign alignment to avoid q/-q ambiguity.

    Args:
        quats: (N, 4) quaternions [w, x, y, z]
    Returns:
        q_median: (4,) median quaternion (normalized)
    """
    ref = quats[0]
    aligned = quats.copy()
    for i in range(1, len(aligned)):
        if np.dot(aligned[i], ref) < 0:
            aligned[i] = -aligned[i]
    q_med = np.median(aligned, axis=0)
    q_med /= np.linalg.norm(q_med) + 1e-12
    return q_med


def sequence_median_aggregation(seq_pred_Ts, seq_gt_Ts):
    """
    多帧时域滤波：对同一 sequence 的所有帧预测进行鲁棒聚合。

    使用两种互补方法:
    1. 旋转矩阵元素中位数 + SVD 投影到 SO(3) (避免 Euler 角奇异性)
    2. 平移向量中位数

    同一辆车同一行程的标定参数恒定，因此多帧预测的理想值相同。

    Args:
        seq_pred_Ts: list of (4,4) predicted transforms for frames in one sequence
        seq_gt_Ts: list of (4,4) GT transforms (should be identical within a sequence)

    Returns:
        median_T: (4,4) median-aggregated transform
        per_frame_rotvecs: (N, 3) axis-angle residuals relative to GT (for variance)
    """
    Rs = np.array([T[:3, :3] for T in seq_pred_Ts])
    ts = np.array([T[:3, 3] for T in seq_pred_Ts])

    R_median = rotation_matrix_median_svd(Rs)
    t_median = np.median(ts, axis=0)

    median_T = np.eye(4)
    median_T[:3, :3] = R_median
    median_T[:3, 3] = t_median

    gt_R = seq_gt_Ts[0][:3, :3] if seq_gt_Ts else np.eye(3)
    per_frame_rotvecs = []
    for pred_R in Rs:
        delta_R = pred_R @ gt_R.T
        rv = R.from_matrix(delta_R).as_rotvec()
        per_frame_rotvecs.append(np.degrees(rv))
    per_frame_rotvecs = np.array(per_frame_rotvecs)

    return median_T, per_frame_rotvecs


def evaluate_p0(args):
    """P0 评估主函数"""
    t0 = time.time()

    print("=" * 80)
    print("P0 推理增强评估：迭代细化 + 多帧时域滤波")
    print("=" * 80)
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  Dataset: {args.dataset_root}")
    print(f"  迭代细化轮数: {args.num_iterations}")
    print(f"  序列中位数聚合: {'启用' if args.sequence_median else '禁用'}")
    print(f"  扰动范围: {args.angle_range_deg}° / {args.trans_range}m")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 1. Load checkpoint and model --
    print(f"\n1. 加载模型...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')

    if args.rotation_only == -1:
        rotation_only = checkpoint.get('rotation_only',
                                       not checkpoint.get('optimize_translation', True))
    else:
        rotation_only = args.rotation_only > 0

    state_dict = checkpoint['model_state_dict']
    use_mlp_head = 'rotation_pred.0.weight' in state_dict
    ckpt_args = checkpoint.get('args', {})
    _voxel_mode = args.voxel_mode or ckpt_args.get('voxel_mode', 'hard')
    _scatter_reduce = args.scatter_reduce or ckpt_args.get('scatter_reduce', 'sum')
    _to_bev_mode = args.to_bev_mode or ckpt_args.get('to_bev_mode', 'concat')

    model = BEVCalib(
        deformable=args.deformable > 0,
        bev_encoder=args.bev_encoder > 0,
        img_shape=(args.target_height, args.target_width),
        rotation_only=rotation_only,
        use_mlp_head=use_mlp_head,
        bev_pool_factor=args.bev_pool_factor,
        voxel_mode=_voxel_mode,
        to_bev_mode=_to_bev_mode,
        scatter_reduce=_scatter_reduce,
        intrinsic_input=ckpt_args.get('intrinsic_input', False),
    ).to(device)

    from evaluate_checkpoint import (
        _auto_permute_spconv_weights, _adapt_model_to_checkpoint,
        _resolve_perturbation_from_ckpt,
    )
    _resolve_perturbation_from_ckpt(args, checkpoint)

    state_dict = _auto_permute_spconv_weights(state_dict, model)
    _adapt_model_to_checkpoint(model, state_dict, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"   ✓ Epoch {epoch}, rotation_only={rotation_only}, voxel={_voxel_mode}")

    # -- 2. Load dataset --
    print(f"\n2. 加载数据集...")
    from evaluate_checkpoint import _build_eval_custom_dataset, make_collate_fn
    dataset = _build_eval_custom_dataset(args.dataset_root, args)
    eval_dataset = dataset if args.use_full_dataset else dataset
    print(f"   ✓ {len(eval_dataset)} 个样本")

    # Build sequence mapping
    _eval_idx_to_seq = {}
    seq_boundaries = []
    if hasattr(dataset, 'all_files') and dataset.all_files:
        for loader_idx, fpath in enumerate(dataset.all_files):
            _eval_idx_to_seq[loader_idx] = fpath.split('/')[0]
        cur_seq, cur_start = None, 0
        for loader_idx in range(len(eval_dataset)):
            sid = _eval_idx_to_seq.get(loader_idx, "unknown")
            if sid != cur_seq:
                if cur_seq is not None:
                    seq_boundaries.append((cur_seq, cur_start, loader_idx - 1))
                cur_seq = sid
                cur_start = loader_idx
        if cur_seq is not None:
            seq_boundaries.append((cur_seq, cur_start, len(eval_dataset) - 1))

    print(f"   序列数: {len(seq_boundaries)}")
    for sid, ss, se in seq_boundaries:
        print(f"     Seq {sid}: samples {ss}-{se} ({se - ss + 1} 帧)")

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        num_workers=4, collate_fn=collate_fn, shuffle=False
    )

    # -- 3. Evaluate with iterative refinement --
    print(f"\n3. 开始评估（{args.num_iterations} 轮迭代）...")

    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    errors_per_iter = [defaultdict(list) for _ in range(args.num_iterations)]
    sample_sequences = []
    seq_pred_Ts_per_iter = [defaultdict(list) for _ in range(args.num_iterations)]
    seq_gt_Ts = defaultdict(list)

    sample_count = 0
    max_batches = args.max_batches if args.max_batches > 0 else len(val_loader)
    error_keys = ['rot_error', 'roll_error', 'pitch_error', 'yaw_error',
                  'trans_error', 'fwd_error', 'lat_error', 'ht_error']

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            gt_T_np = np.array(gt_T_to_camera).astype(np.float32)
            _paw = None
            if getattr(args, 'per_axis_weights', '') and args.per_axis_weights:
                _paw = tuple(float(x) for x in args.per_axis_weights.split(','))
            init_T_np, _, _ = generate_single_perturbation_from_T(
                gt_T_np,
                angle_range_deg=args.angle_range_deg,
                trans_range=args.trans_range,
                rotation_only=rotation_only,
                distribution=getattr(args, 'perturb_distribution', 'uniform'),
                per_axis_prob=getattr(args, 'per_axis_prob', 0.0),
                per_axis_weights=_paw,
            )

            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_np = np.array(pcs)[:, :, :3] if args.xyz_only > 0 else np.array(pcs)
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
            init_T_t = torch.from_numpy(init_T_np).float().to(device)
            post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
            intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

            all_iter_preds = iterative_refinement_forward(
                model, resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, intrinsic_t,
                masks=masks, num_iterations=args.num_iterations,
                rotation_only=rotation_only
            )

            B = gt_T_np.shape[0]
            for i in range(B):
                sample_idx = sample_count + i
                seq_id = _eval_idx_to_seq.get(sample_idx, "unknown")
                sample_sequences.append(seq_id)

                seq_gt_Ts[seq_id].append(gt_T_np[i])

                for it_idx, pred_T_batch in enumerate(all_iter_preds):
                    errs = compute_pose_errors(pred_T_batch[i], gt_T_np[i])
                    for key in error_keys:
                        errors_per_iter[it_idx][key].append(errs[key])
                    seq_pred_Ts_per_iter[it_idx][seq_id].append(pred_T_batch[i])

            sample_count += B

            if batch_idx % 20 == 0 or batch_idx == max_batches - 1:
                last_rot = errors_per_iter[-1]['rot_error'][-1] if errors_per_iter[-1]['rot_error'] else 0
                print(f"   Batch {batch_idx + 1}/{max_batches}, "
                      f"samples={sample_count}, last_rot={last_rot:.3f}°")

    # -- 4. Compute per-iteration statistics --
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.ckpt_path), 'p0_eval')
    os.makedirs(output_dir, exist_ok=True)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("P0 推理增强评估报告")
    report_lines.append("=" * 80)
    report_lines.append(f"Checkpoint: {os.path.basename(args.ckpt_path)}")
    report_lines.append(f"Epoch: {epoch}")
    report_lines.append(f"Dataset: {args.dataset_root}")
    report_lines.append(f"总样本数: {sample_count}")
    report_lines.append(f"迭代轮数: {args.num_iterations}")
    report_lines.append(f"序列中位数: {'启用' if args.sequence_median else '禁用'}")
    report_lines.append(f"扰动: {args.angle_range_deg}° / {args.trans_range}m")
    report_lines.append(f"Eval seed: {args.eval_seed}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("一、迭代细化效果（Per-Frame）")
    report_lines.append("=" * 80)

    header = (f"{'Iter':<6} {'Mean Rot°':>10} {'Std':>8} {'Median':>8} "
              f"{'P95':>8} {'Max':>8} {'Roll°':>8} {'Pitch°':>8} {'Yaw°':>8}")
    report_lines.append(header)
    report_lines.append("-" * len(header))

    iter_summary = []
    for it_idx in range(args.num_iterations):
        errs = errors_per_iter[it_idx]
        rot = np.array(errs['rot_error'])
        roll = np.array(errs['roll_error'])
        pitch = np.array(errs['pitch_error'])
        yaw = np.array(errs['yaw_error'])
        summary = {
            'iter': it_idx,
            'rot_mean': np.mean(rot), 'rot_std': np.std(rot),
            'rot_median': np.median(rot), 'rot_p95': np.percentile(rot, 95),
            'rot_max': np.max(rot),
            'roll_mean': np.mean(roll), 'pitch_mean': np.mean(pitch),
            'yaw_mean': np.mean(yaw),
        }
        iter_summary.append(summary)

        label = f"Iter {it_idx}" if it_idx > 0 else "Base"
        report_lines.append(
            f"{label:<6} {summary['rot_mean']:>10.4f} {summary['rot_std']:>8.4f} "
            f"{summary['rot_median']:>8.4f} {summary['rot_p95']:>8.4f} "
            f"{summary['rot_max']:>8.4f} {summary['roll_mean']:>8.4f} "
            f"{summary['pitch_mean']:>8.4f} {summary['yaw_mean']:>8.4f}"
        )

    if args.num_iterations > 1:
        base = iter_summary[0]
        best = iter_summary[-1]
        improvement = (base['rot_mean'] - best['rot_mean']) / base['rot_mean'] * 100
        report_lines.append("")
        report_lines.append(
            f"迭代细化收益: {base['rot_mean']:.4f}° → {best['rot_mean']:.4f}° "
            f"(↓{improvement:.1f}%)")
        report_lines.append(
            f"  Roll:  {base['roll_mean']:.4f}° → {best['roll_mean']:.4f}°")
        report_lines.append(
            f"  Pitch: {base['pitch_mean']:.4f}° → {best['pitch_mean']:.4f}°")
        report_lines.append(
            f"  Yaw:   {base['yaw_mean']:.4f}° → {best['yaw_mean']:.4f}°")

    # -- 5. Sequence median aggregation --
    if args.sequence_median and seq_boundaries:
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("二、多帧中位数聚合效果（Per-Sequence）")
        report_lines.append("=" * 80)
        report_lines.append("")

        best_iter_idx = args.num_iterations - 1

        for agg_iter in [0, best_iter_idx] if best_iter_idx > 0 else [0]:
            label = "Base (无迭代)" if agg_iter == 0 else f"Iter {agg_iter} 后"
            report_lines.append(f"--- {label} + 序列中位数聚合 ---")

            header2 = (f"{'Seq':<6} {'Frames':>6} {'PerFrame°':>10} "
                       f"{'Median°':>10} {'改善%':>8} "
                       f"{'Roll°':>8} {'Pitch°':>8} {'Yaw°':>8} "
                       f"{'RPY Std':>8}")
            report_lines.append(header2)
            report_lines.append("-" * len(header2))

            all_median_errors = defaultdict(list)
            all_perframe_errors = defaultdict(list)
            all_rpy_stds = []

            for seq_id, ss, se in seq_boundaries:
                pred_Ts = seq_pred_Ts_per_iter[agg_iter][seq_id]
                gt_Ts = seq_gt_Ts[seq_id]
                if not pred_Ts:
                    continue

                # Per-frame mean error for this sequence
                frame_errors = []
                for pt, gt in zip(pred_Ts, gt_Ts):
                    frame_errors.append(compute_pose_errors(pt, gt))
                pf_rot = np.mean([e['rot_error'] for e in frame_errors])
                all_perframe_errors['rot'].append(pf_rot)

                # Median aggregation
                median_T, per_frame_rpys = sequence_median_aggregation(pred_Ts, gt_Ts)
                rpy_std = np.mean(np.std(per_frame_rpys, axis=0))
                all_rpy_stds.append(rpy_std)

                median_errs = compute_pose_errors(median_T, gt_Ts[0])
                for key in error_keys:
                    all_median_errors[key].append(median_errs[key])

                improvement_pct = (pf_rot - median_errs['rot_error']) / pf_rot * 100 if pf_rot > 0 else 0

                report_lines.append(
                    f"{seq_id:<6} {len(pred_Ts):>6} {pf_rot:>10.4f} "
                    f"{median_errs['rot_error']:>10.4f} {improvement_pct:>7.1f}% "
                    f"{median_errs['roll_error']:>8.4f} "
                    f"{median_errs['pitch_error']:>8.4f} "
                    f"{median_errs['yaw_error']:>8.4f} "
                    f"{rpy_std:>8.4f}")

            if all_median_errors['rot_error']:
                mean_pf = np.mean(all_perframe_errors['rot'])
                mean_med = np.mean(all_median_errors['rot_error'])
                overall_improve = (mean_pf - mean_med) / mean_pf * 100

                report_lines.append("-" * len(header2))
                report_lines.append(
                    f"{'ALL':<6} {sample_count:>6} {mean_pf:>10.4f} "
                    f"{mean_med:>10.4f} {overall_improve:>7.1f}% "
                    f"{np.mean(all_median_errors['roll_error']):>8.4f} "
                    f"{np.mean(all_median_errors['pitch_error']):>8.4f} "
                    f"{np.mean(all_median_errors['yaw_error']):>8.4f} "
                    f"{np.mean(all_rpy_stds):>8.4f}")

                report_lines.append("")
                report_lines.append(
                    f"中位数聚合收益: PerFrame {mean_pf:.4f}° → Median {mean_med:.4f}° "
                    f"(↓{overall_improve:.1f}%)")
                report_lines.append(
                    f"  平均帧间 RPY 标准差: {np.mean(all_rpy_stds):.4f}°")
            report_lines.append("")

    # -- 6. P80 RPY<0.3° analysis (business target) --
    report_lines.append("=" * 80)
    report_lines.append("三、业务目标分析 (P80 RPY < 0.3°)")
    report_lines.append("=" * 80)

    for it_idx in range(args.num_iterations):
        errs = errors_per_iter[it_idx]
        roll = np.array(errs['roll_error'])
        pitch = np.array(errs['pitch_error'])
        yaw = np.array(errs['yaw_error'])
        n = len(roll)

        rpy_all_ok = np.sum((roll < 0.3) & (pitch < 0.3) & (yaw < 0.3)) / n * 100
        roll_ok = np.sum(roll < 0.3) / n * 100
        pitch_ok = np.sum(pitch < 0.3) / n * 100
        yaw_ok = np.sum(yaw < 0.3) / n * 100

        label = "Base" if it_idx == 0 else f"Iter {it_idx}"
        report_lines.append(
            f"{label}: RPY全<0.3°={rpy_all_ok:.1f}% | "
            f"Roll<0.3°={roll_ok:.1f}% | Pitch<0.3°={pitch_ok:.1f}% | Yaw<0.3°={yaw_ok:.1f}%")

    # -- 7. Per-sequence per-iteration detail --
    if seq_boundaries:
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("四、Per-Sequence 迭代细化详情")
        report_lines.append("=" * 80)

        for it_idx in range(args.num_iterations):
            label = "Base" if it_idx == 0 else f"Iter {it_idx}"
            report_lines.append(f"\n--- {label} ---")

            errs = errors_per_iter[it_idx]
            rot_arr = np.array(errs['rot_error'])
            roll_arr = np.array(errs['roll_error'])
            pitch_arr = np.array(errs['pitch_error'])
            yaw_arr = np.array(errs['yaw_error'])
            seq_arr = np.array(sample_sequences)

            header3 = f"{'Seq':<6} {'N':>5} {'Rot°':>8} {'Roll°':>8} {'Pitch°':>8} {'Yaw°':>8}"
            report_lines.append(header3)
            report_lines.append("-" * len(header3))

            for seq_id, ss, se in seq_boundaries:
                mask = seq_arr == seq_id
                n = int(mask.sum())
                if n == 0:
                    continue
                report_lines.append(
                    f"{seq_id:<6} {n:>5} "
                    f"{np.mean(rot_arr[mask]):>8.4f} "
                    f"{np.mean(roll_arr[mask]):>8.4f} "
                    f"{np.mean(pitch_arr[mask]):>8.4f} "
                    f"{np.mean(yaw_arr[mask]):>8.4f}")

    # -- 8. Write report --
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "p0_refinement_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    # Save detailed JSON
    json_data = {
        'ckpt': os.path.basename(args.ckpt_path),
        'epoch': epoch,
        'dataset': args.dataset_root,
        'num_iterations': args.num_iterations,
        'sequence_median': args.sequence_median,
        'angle_range_deg': args.angle_range_deg,
        'trans_range': args.trans_range,
        'sample_count': sample_count,
        'per_iteration': iter_summary,
    }
    json_path = os.path.join(output_dir, "p0_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)

    # -- 9. Generate comparison chart --
    _generate_p0_charts(errors_per_iter, output_dir, args, seq_boundaries, sample_sequences)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"✓ P0 评估完成！耗时 {elapsed:.0f}s")
    print(f"  报告: {report_path}")
    print(f"  JSON: {json_path}")
    print(f"  图表: {output_dir}/charts/")
    print(f"{'=' * 80}")
    print(f"\n{report_text}")


def _generate_p0_charts(errors_per_iter, output_dir, args, seq_boundaries, sample_sequences):
    """Generate comparison charts for iterative refinement."""
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    num_iter = len(errors_per_iter)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    # Chart 1: Per-iteration error distribution boxplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key, title in zip(axes, ['roll_error', 'pitch_error', 'yaw_error'],
                               ['Roll', 'Pitch', 'Yaw']):
        data = [np.array(errors_per_iter[i][key]) for i in range(num_iter)]
        labels = ['Base'] + [f'Iter {i}' for i in range(1, num_iter)]
        bp = ax.boxplot(data, labels=labels[:num_iter], patch_artist=True,
                        showfliers=False, medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors[:num_iter]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Error (°)')
        ax.set_title(f'{title} Error Distribution')
        ax.grid(True, alpha=0.3)
        for i, d in enumerate(data):
            ax.text(i + 1, np.median(d), f'{np.median(d):.3f}°',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle(f'P0 Iterative Refinement ({num_iter} iterations)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, 'iter_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 2: Mean error convergence across iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    iters = list(range(num_iter))
    for key, label, color in [('rot_error', 'Total Rot', '#FF6B6B'),
                                ('roll_error', 'Roll', '#4ECDC4'),
                                ('pitch_error', 'Pitch', '#45B7D1'),
                                ('yaw_error', 'Yaw', '#96CEB4')]:
        means = [np.mean(errors_per_iter[i][key]) for i in iters]
        ax.plot(iters, means, 'o-', label=label, color=color, linewidth=2, markersize=8)
        for x, y in zip(iters, means):
            ax.annotate(f'{y:.3f}°', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Error (°)')
    ax.set_title('Error Convergence across Iterations')
    ax.set_xticks(iters)
    ax.set_xticklabels(['Base'] + [f'Iter {i}' for i in range(1, num_iter)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, 'iter_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 3: Per-sequence comparison (base vs final iteration)
    if seq_boundaries and num_iter > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        seq_ids = [s[0] for s in seq_boundaries]
        seq_arr = np.array(sample_sequences)
        base_means = []
        final_means = []
        for sid, _, _ in seq_boundaries:
            mask = seq_arr == sid
            base_means.append(np.mean(np.array(errors_per_iter[0]['rot_error'])[mask]))
            final_means.append(np.mean(np.array(errors_per_iter[-1]['rot_error'])[mask]))

        x = np.arange(len(seq_ids))
        width = 0.35
        ax.bar(x - width / 2, base_means, width, label='Base (1x forward)', color='#FF6B6B', alpha=0.8)
        ax.bar(x + width / 2, final_means, width,
               label=f'Iter {num_iter - 1} ({num_iter}x forward)', color='#4ECDC4', alpha=0.8)
        ax.set_xlabel('Sequence')
        ax.set_ylabel('Mean Rotation Error (°)')
        ax.set_title('Per-Sequence: Base vs Iterative Refinement')
        ax.set_xticks(x)
        ax.set_xticklabels(seq_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(os.path.join(charts_dir, 'per_seq_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"   ✓ 图表已保存到 {charts_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0 推理增强评估")

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--num_iterations", type=int, default=3,
                        help="迭代细化轮数 (1=baseline, 3=推荐)")
    parser.add_argument("--sequence_median", action='store_true', default=False,
                        help="启用序列级中位数聚合")

    parser.add_argument("--angle_range_deg", type=float, default=5.0)
    parser.add_argument("--trans_range", type=float, default=0.15)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--target_height", type=int, default=360)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--eval_max_frames_per_seq", type=int, default=None)
    parser.add_argument("--eval_sample_step", type=int, default=None)

    parser.add_argument("--rotation_only", type=int, default=-1)
    parser.add_argument("--deformable", type=int, default=0)
    parser.add_argument("--bev_encoder", type=int, default=1)
    parser.add_argument("--bev_pool_factor", type=int, default=0)
    parser.add_argument("--xyz_only", type=int, default=1)
    parser.add_argument("--use_full_dataset", action='store_true', default=False)
    parser.add_argument("--use_drcv", action='store_true', default=False)
    parser.add_argument("--voxel_mode", type=str, default=None)
    parser.add_argument("--scatter_reduce", type=str, default=None)
    parser.add_argument("--to_bev_mode", type=str, default=None)
    parser.add_argument("--per_axis_weights", type=str, default=None)
    parser.add_argument("--per_axis_prob", type=float, default=0.0)
    parser.add_argument("--perturb_distribution", type=str, default='uniform')
    parser.add_argument("--data_balance", type=int, default=0)

    args = parser.parse_args()
    evaluate_p0(args)
