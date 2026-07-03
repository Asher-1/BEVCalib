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

BEVCALIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BEVCALIB_ROOT)
sys.path.insert(0, os.path.join(BEVCALIB_ROOT, 'kitti-bev-calib'))

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


def _iter_label(it_idx, adaptive_iter2=False):
    if it_idx == 0:
        return "Base"
    return "Adaptive" if adaptive_iter2 else f"Iter {it_idx}"


def _build_seq_boundaries(sample_idx_to_seq, sample_count):
    boundaries = []
    cur_seq, cur_start = None, 0
    for sample_idx in range(sample_count):
        seq_id = sample_idx_to_seq.get(sample_idx, "unknown")
        if seq_id != cur_seq:
            if cur_seq is not None:
                boundaries.append((cur_seq, cur_start, sample_idx - 1))
            cur_seq = seq_id
            cur_start = sample_idx
    if cur_seq is not None and sample_count > 0:
        boundaries.append((cur_seq, cur_start, sample_count - 1))
    return boundaries


def _select_batch_items(obj, pick):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj[pick]
    if isinstance(obj, np.ndarray):
        return obj[pick]
    return [obj[i] for i in pick]


def _sample_uniform_indices(start_idx, end_idx, max_samples):
    n = end_idx - start_idx + 1
    if max_samples <= 0 or n <= max_samples:
        return list(range(start_idx, end_idx + 1))
    return np.linspace(start_idx, end_idx, num=max_samples, dtype=int).tolist()


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


def sequence_median_transform(seq_pred_Ts):
    Rs = np.array([T[:3, :3] for T in seq_pred_Ts])
    ts = np.array([T[:3, 3] for T in seq_pred_Ts])

    R_median = rotation_matrix_median_svd(Rs)
    t_median = np.median(ts, axis=0)

    median_T = np.eye(4, dtype=np.float32)
    median_T[:3, :3] = R_median.astype(np.float32)
    median_T[:3, 3] = t_median.astype(np.float32)
    return median_T


def per_frame_rotvecs_to_reference(seq_pred_Ts, ref_T):
    ref_R = ref_T[:3, :3] if ref_T.shape == (4, 4) else ref_T
    rotvecs = []
    for pred_T in seq_pred_Ts:
        pred_R = pred_T[:3, :3] if pred_T.shape == (4, 4) else pred_T
        delta_R = pred_R @ ref_R.T
        rotvecs.append(np.degrees(R.from_matrix(delta_R).as_rotvec()))
    return np.array(rotvecs, dtype=np.float32)


def transform_rot_delta_deg(T_from, T_to):
    delta_R = T_to[:3, :3] @ T_from[:3, :3].T
    return float(np.degrees(np.linalg.norm(R.from_matrix(delta_R).as_rotvec())))


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
    median_T = sequence_median_transform(seq_pred_Ts)
    gt_ref = seq_gt_Ts[0] if seq_gt_Ts else median_T
    per_frame_rotvecs = per_frame_rotvecs_to_reference(seq_pred_Ts, gt_ref)
    return median_T, per_frame_rotvecs


def _run_single_pass_subset(model, val_loader, device, args, rotation_only,
                            init_T_by_idx, max_batches):
    pred_by_idx = {}
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            gt_T_np_full = np.array(gt_T_to_camera).astype(np.float32)
            batch_size = gt_T_np_full.shape[0]
            batch_global = list(range(sample_offset, sample_offset + batch_size))
            pick_local = [i for i, global_idx in enumerate(batch_global) if global_idx in init_T_by_idx]
            sample_offset += batch_size

            if not pick_local:
                continue

            picked_global = [batch_global[i] for i in pick_local]
            sub_imgs = _select_batch_items(imgs, pick_local)
            sub_pcs = _select_batch_items(pcs, pick_local)
            sub_masks = _select_batch_items(masks, pick_local)
            sub_intrinsics = _select_batch_items(intrinsics, pick_local)

            resize_imgs = torch.from_numpy(np.array(sub_imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_np = np.array(sub_pcs)[:, :, :3] if args.xyz_only > 0 else np.array(sub_pcs)
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np_full[pick_local]).float().to(device)
            init_T_np = np.stack([init_T_by_idx[idx] for idx in picked_global], axis=0).astype(np.float32)
            init_T_t = torch.from_numpy(init_T_np).float().to(device)
            post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
            intrinsic_t = torch.from_numpy(np.array(sub_intrinsics)).float().to(device)

            T_pred, _, _ = model(
                resize_imgs, pcs_t, gt_T_t, init_T_t,
                post_T, intrinsic_t, masks=sub_masks, out_init_loss=False
            )
            pred_np = T_pred.detach().cpu().numpy()
            for local_idx, global_idx in enumerate(picked_global):
                pred_by_idx[global_idx] = pred_np[local_idx]

    return pred_by_idx


def parse_fixed_inject_rpy(raw):
    """Parse fixed R/P/Y degrees for LiDAR-frame right-multiply injection."""
    if raw is None or str(raw).strip() == "":
        return None
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--fixed_inject_rpy must be three comma-separated values, e.g. 2,2,2")
    return vals


def evaluate_p0(args):
    """P0 评估主函数"""
    t0 = time.time()

    fixed_inject_rpy = parse_fixed_inject_rpy(args.fixed_inject_rpy)
    effective_num_iterations = max(1, int(args.num_iterations))
    if args.adaptive_iter2:
        if effective_num_iterations < 2:
            print("[adaptive_iter2] num_iterations<2, 自动提升为 2 (single-pass + conditional second pass)")
        effective_num_iterations = 2

    print("=" * 80)
    print("P0 推理增强评估：迭代细化 + 多帧时域滤波")
    print("=" * 80)
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  Dataset: {args.dataset_root}")
    print(f"  迭代细化轮数: {effective_num_iterations}")
    print(f"  序列中位数聚合: {'启用' if args.sequence_median else '禁用'}")
    print(f"  扰动范围: {args.angle_range_deg}° / {args.trans_range}m")
    if fixed_inject_rpy is not None:
        print(f"  固定注入: R/P/Y={fixed_inject_rpy}° (right-multiply in LiDAR frame)")
    if args.adaptive_iter2:
        print("  Adaptive Iter2: 启用")
        print(f"    second_pass_mode={args.adaptive_second_pass_mode}")
        print(f"    probe_frames={args.adaptive_probe_frames}")
        if args.adaptive_seq_residual_lo_deg > 0:
            print(f"    trigger: probe_med>={args.adaptive_seq_residual_deg:.3f}°"
                  f" OR (probe_med>={args.adaptive_seq_residual_lo_deg:.3f}°"
                  f" and seq_std>={args.adaptive_seq_rpy_std_deg:.3f}°)")
        else:
            print(f"    trigger: probe_med>={args.adaptive_seq_residual_deg:.3f}°")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n1. 加载模型...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')

    if args.rotation_only == -1:
        rotation_only = checkpoint.get('rotation_only',
                                       not checkpoint.get('optimize_translation', True))
    else:
        rotation_only = args.rotation_only > 0

    from evaluate_checkpoint import (
        _build_model_from_ckpt, _resolve_perturbation_from_ckpt,
    )
    _resolve_perturbation_from_ckpt(args, checkpoint)
    model, ckpt_args, model_params = _build_model_from_ckpt(
        args, checkpoint, device, rotation_only)
    print(f"   ✓ Epoch {epoch}, rotation_only={rotation_only}, "
          f"voxel={model_params.get('voxel_mode', ckpt_args.get('voxel_mode', 'hard'))}")

    print(f"\n2. 加载数据集...")
    from evaluate_checkpoint import _build_eval_custom_dataset, make_collate_fn
    dataset = _build_eval_custom_dataset(args.dataset_root, args)
    eval_dataset = dataset if args.use_full_dataset else dataset
    print(f"   ✓ {len(eval_dataset)} 个样本")

    _eval_idx_to_seq = {}
    if hasattr(dataset, 'all_files') and dataset.all_files:
        for loader_idx, fpath in enumerate(dataset.all_files):
            _eval_idx_to_seq[loader_idx] = fpath.split('/')[0]
    seq_boundaries = _build_seq_boundaries(_eval_idx_to_seq, len(eval_dataset))

    print(f"   序列数: {len(seq_boundaries)}")
    for sid, ss, se in seq_boundaries:
        print(f"     Seq {sid}: samples {ss}-{se} ({se - ss + 1} 帧)")

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        num_workers=4, collate_fn=collate_fn, shuffle=False
    )

    print(f"\n3. 开始评估（{effective_num_iterations} 轮迭代）...")

    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    errors_per_iter = [defaultdict(list) for _ in range(effective_num_iterations)]
    sample_sequences = []
    seq_pred_Ts_per_iter = [defaultdict(list) for _ in range(effective_num_iterations)]
    seq_gt_Ts = defaultdict(list)
    seq_init_Ts = defaultdict(list)
    sample_gt_by_idx = {}
    sample_init_by_idx = {}
    sample_pred0_by_idx = {}

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
            if args.use_identity_init:
                init_T_np = gt_T_np.copy()
            elif fixed_inject_rpy is not None:
                dR = R.from_euler('xyz', np.deg2rad(fixed_inject_rpy)).as_matrix().astype(np.float32)
                init_T_np = gt_T_np.copy()
                init_T_np[:, :3, :3] = np.matmul(gt_T_np[:, :3, :3], dR)
            else:
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

            iter_count_this_pass = 1 if args.adaptive_iter2 else effective_num_iterations
            all_iter_preds = iterative_refinement_forward(
                model, resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, intrinsic_t,
                masks=masks, num_iterations=iter_count_this_pass,
                rotation_only=rotation_only
            )

            batch_size = gt_T_np.shape[0]
            for i in range(batch_size):
                sample_idx = sample_count + i
                seq_id = _eval_idx_to_seq.get(sample_idx, "unknown")
                sample_sequences.append(seq_id)
                sample_gt_by_idx[sample_idx] = gt_T_np[i]
                sample_init_by_idx[sample_idx] = init_T_np[i]
                sample_pred0_by_idx[sample_idx] = all_iter_preds[0][i]

                seq_gt_Ts[seq_id].append(gt_T_np[i])
                if fixed_inject_rpy is not None:
                    seq_init_Ts[seq_id].append(init_T_np[i])

                for it_idx, pred_T_batch in enumerate(all_iter_preds):
                    errs = compute_pose_errors(pred_T_batch[i], gt_T_np[i])
                    for key in error_keys:
                        errors_per_iter[it_idx][key].append(errs[key])
                    seq_pred_Ts_per_iter[it_idx][seq_id].append(pred_T_batch[i])

            sample_count += batch_size

            if batch_idx % 20 == 0 or batch_idx == max_batches - 1:
                last_rot = errors_per_iter[0]['rot_error'][-1] if errors_per_iter[0]['rot_error'] else 0.0
                print(f"   Batch {batch_idx + 1}/{max_batches}, "
                      f"samples={sample_count}, last_rot={last_rot:.3f}°")

    seq_boundaries = _build_seq_boundaries(_eval_idx_to_seq, sample_count)
    adaptive_gate_rows = []
    adaptive_triggered_seqs = []
    adaptive_triggered_frames = 0

    if args.adaptive_iter2:
        print("\n4. Adaptive Iter2 Gate: probe sequence-median residual...")
        seq_median_T0 = {}
        seq_consensus_std = {}
        seq_probe_indices = {}
        probe_init_by_idx = {}

        for seq_id, ss, se in seq_boundaries:
            pred_Ts = seq_pred_Ts_per_iter[0][seq_id]
            if not pred_Ts:
                continue
            median_T = sequence_median_transform(pred_Ts)
            seq_median_T0[seq_id] = median_T
            rotvecs_to_med = per_frame_rotvecs_to_reference(pred_Ts, median_T)
            seq_consensus_std[seq_id] = (
                float(np.mean(np.std(rotvecs_to_med, axis=0))) if len(rotvecs_to_med) > 0 else 0.0
            )
            probe_indices = _sample_uniform_indices(ss, se, args.adaptive_probe_frames)
            seq_probe_indices[seq_id] = probe_indices
            for sample_idx in probe_indices:
                probe_init_by_idx[sample_idx] = median_T

        probe_pred_by_idx = _run_single_pass_subset(
            model, val_loader, device, args, rotation_only, probe_init_by_idx, max_batches
        )

        second_pass_init_by_idx = {}
        for seq_id, ss, se in seq_boundaries:
            pred_Ts = seq_pred_Ts_per_iter[0][seq_id]
            if not pred_Ts:
                continue

            probe_indices = [idx for idx in seq_probe_indices.get(seq_id, []) if idx in probe_pred_by_idx]
            probe_corr_vals = [
                transform_rot_delta_deg(seq_median_T0[seq_id], probe_pred_by_idx[idx])
                for idx in probe_indices
            ]
            base_corr_vals = [
                transform_rot_delta_deg(sample_init_by_idx[idx], sample_pred0_by_idx[idx])
                for idx in range(ss, se + 1)
            ]
            probe_corr_med = float(np.median(probe_corr_vals)) if probe_corr_vals else 0.0
            probe_corr_p90 = float(np.percentile(probe_corr_vals, 90)) if probe_corr_vals else 0.0
            base_corr_med = float(np.median(base_corr_vals)) if base_corr_vals else 0.0
            seq_std = float(seq_consensus_std.get(seq_id, 0.0))

            trigger = False
            trigger_reason = "none"
            if probe_corr_med >= args.adaptive_seq_residual_deg:
                trigger = True
                trigger_reason = "probe_med"
            elif (
                args.adaptive_seq_residual_lo_deg > 0
                and probe_corr_med >= args.adaptive_seq_residual_lo_deg
                and seq_std >= args.adaptive_seq_rpy_std_deg
            ):
                trigger = True
                trigger_reason = "probe_lo+std"

            if trigger:
                adaptive_triggered_seqs.append(seq_id)
                for sample_idx in range(ss, se + 1):
                    if args.adaptive_second_pass_mode == 'per_frame_chain':
                        second_pass_init_by_idx[sample_idx] = sample_pred0_by_idx[sample_idx]
                    else:
                        second_pass_init_by_idx[sample_idx] = seq_median_T0[seq_id]
                adaptive_triggered_frames += (se - ss + 1)

            adaptive_gate_rows.append({
                'seq_id': seq_id,
                'frames': se - ss + 1,
                'probe_frames': len(probe_indices),
                'base_corr_med_deg': base_corr_med,
                'probe_corr_med_deg': probe_corr_med,
                'probe_corr_p90_deg': probe_corr_p90,
                'seq_consensus_std_deg': seq_std,
                'trigger': trigger,
                'trigger_reason': trigger_reason,
            })

        if adaptive_gate_rows:
            print(f"   Triggered {len(adaptive_triggered_seqs)}/{len(adaptive_gate_rows)} seqs, "
                  f"{adaptive_triggered_frames}/{sample_count} frames")
            for row in adaptive_gate_rows:
                mark = "TRIGGER" if row['trigger'] else "skip"
                print(f"     Seq {row['seq_id']}: probe_med={row['probe_corr_med_deg']:.3f}° "
                      f"p90={row['probe_corr_p90_deg']:.3f}° std={row['seq_consensus_std_deg']:.3f}° "
                      f"base_corr={row['base_corr_med_deg']:.3f}° -> {mark}")

        second_pred_by_idx = _run_single_pass_subset(
            model, val_loader, device, args, rotation_only, second_pass_init_by_idx, max_batches
        ) if second_pass_init_by_idx else {}

        final_pred_by_idx = dict(sample_pred0_by_idx)
        final_pred_by_idx.update(second_pred_by_idx)
        errors_per_iter[1] = defaultdict(list)
        seq_pred_Ts_per_iter[1] = defaultdict(list)
        for sample_idx in range(sample_count):
            seq_id = _eval_idx_to_seq.get(sample_idx, "unknown")
            pred_T = final_pred_by_idx[sample_idx]
            gt_T = sample_gt_by_idx[sample_idx]
            errs = compute_pose_errors(pred_T, gt_T)
            for key in error_keys:
                errors_per_iter[1][key].append(errs[key])
            seq_pred_Ts_per_iter[1][seq_id].append(pred_T)

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
    report_lines.append(f"迭代轮数: {effective_num_iterations}")
    report_lines.append(f"序列中位数: {'启用' if args.sequence_median else '禁用'}")
    report_lines.append(f"扰动: {args.angle_range_deg}° / {args.trans_range}m")
    if fixed_inject_rpy is not None:
        report_lines.append(f"固定注入 R/P/Y: {fixed_inject_rpy}°")
    report_lines.append(f"Eval seed: {args.eval_seed}")
    if args.adaptive_iter2:
        if args.adaptive_seq_residual_lo_deg > 0:
            report_lines.append(
                f"Adaptive iter2: mode={args.adaptive_second_pass_mode}, "
                f"probe_frames={args.adaptive_probe_frames}, "
                f"med_thr={args.adaptive_seq_residual_deg:.3f}°, "
                f"lo_thr={args.adaptive_seq_residual_lo_deg:.3f}°, "
                f"std_thr={args.adaptive_seq_rpy_std_deg:.3f}°"
            )
        else:
            report_lines.append(
                f"Adaptive iter2: mode={args.adaptive_second_pass_mode}, "
                f"probe_frames={args.adaptive_probe_frames}, "
                f"med_thr={args.adaptive_seq_residual_deg:.3f}°"
            )
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("一、迭代细化效果（Per-Frame）")
    report_lines.append("=" * 80)

    header = (f"{'Iter':<10} {'Mean Rot°':>10} {'Std':>8} {'Median':>8} "
              f"{'P95':>8} {'Max':>8} {'Roll°':>8} {'Pitch°':>8} {'Yaw°':>8}")
    report_lines.append(header)
    report_lines.append("-" * len(header))

    iter_summary = []
    for it_idx in range(effective_num_iterations):
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

        label = _iter_label(it_idx, args.adaptive_iter2)
        report_lines.append(
            f"{label:<10} {summary['rot_mean']:>10.4f} {summary['rot_std']:>8.4f} "
            f"{summary['rot_median']:>8.4f} {summary['rot_p95']:>8.4f} "
            f"{summary['rot_max']:>8.4f} {summary['roll_mean']:>8.4f} "
            f"{summary['pitch_mean']:>8.4f} {summary['yaw_mean']:>8.4f}"
        )

    if effective_num_iterations > 1:
        base = iter_summary[0]
        final = iter_summary[-1]
        improvement = (base['rot_mean'] - final['rot_mean']) / base['rot_mean'] * 100
        report_lines.append("")
        report_lines.append(
            f"迭代细化收益: {base['rot_mean']:.4f}° → {final['rot_mean']:.4f}° "
            f"(↓{improvement:.1f}%)")
        report_lines.append(
            f"  Roll:  {base['roll_mean']:.4f}° → {final['roll_mean']:.4f}°")
        report_lines.append(
            f"  Pitch: {base['pitch_mean']:.4f}° → {final['pitch_mean']:.4f}°")
        report_lines.append(
            f"  Yaw:   {base['yaw_mean']:.4f}° → {final['yaw_mean']:.4f}°")

    if args.adaptive_iter2 and adaptive_gate_rows:
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("二、Adaptive Iter2 Gate")
        report_lines.append("=" * 80)
        if args.adaptive_seq_residual_lo_deg > 0:
            report_lines.append(
                f"模式={args.adaptive_second_pass_mode} | probe_frames={args.adaptive_probe_frames} | "
                f"med_thr={args.adaptive_seq_residual_deg:.3f}° | "
                f"lo_thr={args.adaptive_seq_residual_lo_deg:.3f}° + "
                f"std_thr={args.adaptive_seq_rpy_std_deg:.3f}°"
            )
        else:
            report_lines.append(
                f"模式={args.adaptive_second_pass_mode} | probe_frames={args.adaptive_probe_frames} | "
                f"med_thr={args.adaptive_seq_residual_deg:.3f}°"
            )
        header_gate = (f"{'Seq':<6} {'Frames':>6} {'Probe':>6} {'BaseCorr°':>10} "
                       f"{'ProbeMed°':>10} {'ProbeP90°':>10} {'SeqStd°':>9} {'Gate':>10}")
        report_lines.append(header_gate)
        report_lines.append("-" * len(header_gate))
        for row in adaptive_gate_rows:
            gate_str = row['trigger_reason'] if row['trigger'] else 'skip'
            report_lines.append(
                f"{row['seq_id']:<6} {row['frames']:>6} {row['probe_frames']:>6} "
                f"{row['base_corr_med_deg']:>10.4f} {row['probe_corr_med_deg']:>10.4f} "
                f"{row['probe_corr_p90_deg']:>10.4f} {row['seq_consensus_std_deg']:>9.4f} "
                f"{gate_str:>10}"
            )
        report_lines.append("")
        report_lines.append(
            f"Adaptive触发汇总: {len(adaptive_triggered_seqs)}/{len(adaptive_gate_rows)} seqs, "
            f"{adaptive_triggered_frames}/{sample_count} frames")

    seqmedian_section_title = "三、多帧中位数聚合效果（Per-Sequence）" if args.adaptive_iter2 \
        else "二、多帧中位数聚合效果（Per-Sequence）"
    if args.sequence_median and seq_boundaries:
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(seqmedian_section_title)
        report_lines.append("=" * 80)
        report_lines.append("")

        best_iter_idx = effective_num_iterations - 1
        agg_iters = [0, best_iter_idx] if best_iter_idx > 0 else [0]

        for agg_iter in agg_iters:
            if agg_iter == 0:
                label = "Base (无迭代)"
            elif args.adaptive_iter2:
                label = "Adaptive Iter2 后"
            else:
                label = f"Iter {agg_iter} 后"
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
            fixed_inject_rows = []

            for seq_id, ss, se in seq_boundaries:
                pred_Ts = seq_pred_Ts_per_iter[agg_iter][seq_id]
                gt_Ts = seq_gt_Ts[seq_id]
                if not pred_Ts:
                    continue

                frame_errors = [compute_pose_errors(pt, gt) for pt, gt in zip(pred_Ts, gt_Ts)]
                pf_rot = np.mean([e['rot_error'] for e in frame_errors])
                all_perframe_errors['rot'].append(pf_rot)

                median_T, per_frame_rpys = sequence_median_aggregation(pred_Ts, gt_Ts)
                rpy_std = float(np.mean(np.std(per_frame_rpys, axis=0)))
                all_rpy_stds.append(rpy_std)

                median_errs = compute_pose_errors(median_T, gt_Ts[0])
                for key in error_keys:
                    all_median_errors[key].append(median_errs[key])

                improvement_pct = (pf_rot - median_errs['rot_error']) / pf_rot * 100 if pf_rot > 0 else 0

                if fixed_inject_rpy is not None and seq_init_Ts.get(seq_id):
                    median_init_T, _ = sequence_median_aggregation(seq_init_Ts[seq_id], gt_Ts)
                    inj_errs = compute_pose_errors(median_init_T, gt_Ts[0])
                    injected = inj_errs['rot_error']
                    residual = median_errs['rot_error']
                    recovery = (injected - residual) / injected * 100 if injected > 1e-6 else 100.0
                    fixed_inject_rows.append((seq_id, injected, residual, recovery))

                report_lines.append(
                    f"{seq_id:<6} {len(pred_Ts):>6} {pf_rot:>10.4f} "
                    f"{median_errs['rot_error']:>10.4f} {improvement_pct:>7.1f}% "
                    f"{median_errs['roll_error']:>8.4f} "
                    f"{median_errs['pitch_error']:>8.4f} "
                    f"{median_errs['yaw_error']:>8.4f} "
                    f"{rpy_std:>8.4f}")

            if all_median_errors['rot_error']:
                mean_pf = float(np.mean(all_perframe_errors['rot']))
                mean_med = float(np.mean(all_median_errors['rot_error']))
                overall_improve = (mean_pf - mean_med) / mean_pf * 100 if mean_pf > 1e-9 else 0.0

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

                if fixed_inject_rows:
                    inj_vals = np.array([r[1] for r in fixed_inject_rows], dtype=np.float64)
                    res_vals = np.array([r[2] for r in fixed_inject_rows], dtype=np.float64)
                    rec_vals = np.array([r[3] for r in fixed_inject_rows], dtype=np.float64)
                    report_lines.append("")
                    report_lines.append("--- 固定注入序列中位数恢复诊断 ---")
                    report_lines.append(
                        f"Mean injected={np.mean(inj_vals):.4f}°, "
                        f"Mean residual={np.mean(res_vals):.4f}°, "
                        f"Mean recovery={np.mean(rec_vals):.1f}%, "
                        f"Median recovery={np.median(rec_vals):.1f}%")
                    report_lines.append(f"{'Seq':<6} {'Injected°':>10} {'Residual°':>10} {'Recovery%':>10}")
                    report_lines.append("-" * 42)
                    for sid, injected, residual, recovery in fixed_inject_rows:
                        report_lines.append(
                            f"{sid:<6} {injected:>10.4f} {residual:>10.4f} {recovery:>9.1f}%")
            report_lines.append("")

    rpy_thr = float(getattr(args, 'rpy_threshold_deg', 0.3))
    business_title = f"四、业务目标分析 (RPY < {rpy_thr:.1f}°)" if args.adaptive_iter2 \
        else f"三、业务目标分析 (RPY < {rpy_thr:.1f}°)"
    report_lines.append("=" * 80)
    report_lines.append(business_title)
    report_lines.append("=" * 80)

    for it_idx in range(effective_num_iterations):
        errs = errors_per_iter[it_idx]
        roll = np.array(errs['roll_error'])
        pitch = np.array(errs['pitch_error'])
        yaw = np.array(errs['yaw_error'])
        n = len(roll)

        rpy_all_ok = np.sum((roll < rpy_thr) & (pitch < rpy_thr) & (yaw < rpy_thr)) / n * 100
        roll_ok = np.sum(roll < rpy_thr) / n * 100
        pitch_ok = np.sum(pitch < rpy_thr) / n * 100
        yaw_ok = np.sum(yaw < rpy_thr) / n * 100

        label = _iter_label(it_idx, args.adaptive_iter2)
        report_lines.append(
            f"{label}: RPY全<{rpy_thr:.1f}°={rpy_all_ok:.1f}% | "
            f"Roll<{rpy_thr:.1f}°={roll_ok:.1f}% | "
            f"Pitch<{rpy_thr:.1f}°={pitch_ok:.1f}% | "
            f"Yaw<{rpy_thr:.1f}°={yaw_ok:.1f}%")

    detail_title = "五、Per-Sequence 迭代细化详情" if args.adaptive_iter2 \
        else "四、Per-Sequence 迭代细化详情"
    if seq_boundaries:
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(detail_title)
        report_lines.append("=" * 80)

        seq_arr = np.array(sample_sequences)
        for it_idx in range(effective_num_iterations):
            label = _iter_label(it_idx, args.adaptive_iter2)
            report_lines.append(f"\n--- {label} ---")

            errs = errors_per_iter[it_idx]
            rot_arr = np.array(errs['rot_error'])
            roll_arr = np.array(errs['roll_error'])
            pitch_arr = np.array(errs['pitch_error'])
            yaw_arr = np.array(errs['yaw_error'])

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

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "p0_refinement_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    json_data = {
        'ckpt': os.path.basename(args.ckpt_path),
        'epoch': epoch,
        'dataset': args.dataset_root,
        'num_iterations': effective_num_iterations,
        'sequence_median': args.sequence_median,
        'angle_range_deg': args.angle_range_deg,
        'trans_range': args.trans_range,
        'rpy_threshold_deg': rpy_thr,
        'sample_count': sample_count,
        'adaptive_iter2': bool(args.adaptive_iter2),
        'adaptive_second_pass_mode': args.adaptive_second_pass_mode if args.adaptive_iter2 else None,
        'adaptive_probe_frames': args.adaptive_probe_frames if args.adaptive_iter2 else None,
        'adaptive_seq_residual_deg': args.adaptive_seq_residual_deg if args.adaptive_iter2 else None,
        'adaptive_seq_residual_lo_deg': args.adaptive_seq_residual_lo_deg if args.adaptive_iter2 else None,
        'adaptive_seq_rpy_std_deg': args.adaptive_seq_rpy_std_deg if args.adaptive_iter2 else None,
        'adaptive_gate': adaptive_gate_rows,
        'per_iteration': iter_summary,
    }
    json_path = os.path.join(output_dir, "p0_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)

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
    parser.add_argument("--exclude_seqs", type=str, default=None,
                        help="逗号分隔的序列ID列表，评估时跳过这些序列")
    parser.add_argument("--rpy_threshold_deg", type=float, default=0.3,
                        help="业务统计阈值：RPY三轴均小于该角度才计入全轴通过率")

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
    parser.add_argument("--fixed_inject_rpy", type=str, default=None,
                        help="固定注入 R/P/Y 角度，如 2,2,2；使用 LiDAR-frame right-multiply")
    parser.add_argument("--use_identity_init", action='store_true', default=False,
                        help="使用 GT 作为 init_T，用于 zero-drift 序列聚合诊断")
    parser.add_argument("--adaptive_iter2", action='store_true', default=False,
                        help="默认单轮；按序列 probe 残差/修正幅度触发第二轮")
    parser.add_argument("--adaptive_second_pass_mode", type=str, default='seq_median_anchor',
                        choices=['seq_median_anchor', 'per_frame_chain'],
                        help="Adaptive二轮模式：从首轮序列中位姿出发，或沿每帧pred链式细化")
    parser.add_argument("--adaptive_probe_frames", type=int, default=12,
                        help="每条序列用于估计二轮需求的probe帧数")
    parser.add_argument("--adaptive_seq_residual_deg", type=float, default=0.35,
                        help="probe二轮残余修正幅度中位数阈值（度）；超过则触发二轮")
    parser.add_argument("--adaptive_seq_residual_lo_deg", type=float, default=0.0,
                        help="低阈值：若probe残余修正超过此值且序列分散度也高，则触发二轮")
    parser.add_argument("--adaptive_seq_rpy_std_deg", type=float, default=0.20,
                        help="首轮序列级RPY分散度阈值（度），与lo阈值联动")

    args = parser.parse_args()
    evaluate_p0(args)
