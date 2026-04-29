#!/usr/bin/env python3
"""
TTA (Test-Time Adaptation) 评估：推理时自适应微调

针对跨车标定微差异（贡献46%泛化误差），在推理时对每个新 sequence（新车辆）
做在线自适应，无需 GT 标签。

两种 TTA 策略：
1. TTA-BN: 用测试数据更新 BatchNorm 统计量 (Tent-style, 零梯度)
2. TTA-Head: 冻结backbone, 用时域一致性loss微调回归头 (需梯度)

用法:
    python evaluate_tta.py \
        --ckpt_path logs/.../ckpt_best_val.pth \
        --dataset_root /path/to/test_data_v2 \
        --use_full_dataset \
        --tta_mode head \
        --tta_frames 30 \
        --tta_steps 5 \
        --tta_lr 1e-5 \
        --sequence_median
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import os
import sys
import time
import json
import copy
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
from losses.quat_tools import batch_quat2mat, batch_tvector2mat
from scipy.spatial.transform import Rotation as R


class _FeatureHook:
    """Captures input to the head_drop layer (pooled feature before regression heads)."""
    def __init__(self):
        self.feature = None
    def __call__(self, module, input, output):
        self.feature = input[0].detach()


def tta_head_forward(model, cached_features, init_T, rotation_only):
    """
    Forward through regression head only (with gradients).
    Uses cached backbone features to avoid recomputing the expensive backbone.
    """
    B = cached_features.shape[0]
    x = cached_features.requires_grad_(True)

    if not rotation_only:
        translation = model.translation_pred(x)
    else:
        translation = torch.zeros(B, 3, device=x.device)
    rotation = model.rotation_pred(x)

    T_pred = batch_tvector2mat(translation)
    R_pred = batch_quat2mat(rotation)
    T_pred = torch.bmm(T_pred, R_pred)

    with torch.cuda.amp.autocast(enabled=False):
        pred_T = torch.matmul(
            torch.linalg.inv(T_pred.float()), init_T.float()
        )

    if rotation_only:
        pred_T = pred_T.clone()
        pred_T[:, :3, 3] = init_T[:, :3, 3]

    return pred_T, rotation


def tta_bn_adapt(model, seq_loader, device, num_passes=1):
    """
    TTA-BN: 用测试序列数据更新 BatchNorm 的 running_mean/running_var。

    Tent 风格的自适应：将 BN 层设为 train 模式以收集测试域统计量，
    然后切回 eval 模式使用更新后的统计量做推理。
    """
    bn_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_layers.append((name, m))

    if not bn_layers:
        print("   [TTA-BN] 无 BatchNorm 层, 跳过")
        return

    for _, m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
        m.training = True

    with torch.no_grad():
        for _ in range(num_passes):
            for batch in seq_loader:
                imgs, pcs, masks, gt_T, intrinsics = batch
                gt_T_np = np.array(gt_T).astype(np.float32)
                init_T_np, _, _ = generate_single_perturbation_from_T(
                    gt_T_np, angle_range_deg=5.0, trans_range=0.15, rotation_only=True)

                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
                gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
                init_T_t = torch.from_numpy(init_T_np).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

                model(resize_imgs, pcs_t, gt_T_t, init_T_t,
                      post_T, intrinsic_t, masks=masks, out_init_loss=False)

    for _, m in bn_layers:
        m.training = False

    print(f"   [TTA-BN] 更新了 {len(bn_layers)} 个 BN 层的统计量")


def tta_head_adapt(model, seq_frames, device, args, rotation_only):
    """
    TTA-Head: 冻结 backbone，用时域一致性 loss 微调回归头。

    流程：
    1. 用 hook 缓存 backbone 输出特征（一次性，no_grad）
    2. 多步梯度更新仅通过回归头
    3. 自监督信号：同序列帧的四元数方差最小化 + L2正则防遗忘
    """
    hook = _FeatureHook()
    handle = model.head_drop.register_forward_hook(hook)

    cached_features = []
    cached_init_Ts = []

    with torch.no_grad():
        for batch_data in seq_frames:
            imgs, pcs, init_T, post_T, intrinsics = batch_data
            gt_T_dummy = init_T.clone()
            model(imgs, pcs, gt_T_dummy, init_T, post_T, intrinsics,
                  masks=None, out_init_loss=False)
            cached_features.append(hook.feature.clone())
            cached_init_Ts.append(init_T.clone())

    handle.remove()

    if not cached_features:
        return

    all_feats = torch.cat(cached_features, dim=0)
    all_init_Ts = torch.cat(cached_init_Ts, dim=0)

    original_state = {n: p.clone() for n, p in model.rotation_pred.named_parameters()}

    for p in model.parameters():
        p.requires_grad = False
    for p in model.rotation_pred.parameters():
        p.requires_grad = True
    if not rotation_only:
        for p in model.translation_pred.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.tta_lr, weight_decay=0.0
    )

    for step in range(args.tta_steps):
        _, quats = tta_head_forward(model, all_feats, all_init_Ts, rotation_only)

        ref = quats[0:1].detach()
        dots = (quats * ref).sum(dim=-1)
        signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
        aligned_quats = quats * signs.unsqueeze(-1)

        quat_mean = aligned_quats.mean(dim=0, keepdim=True).detach()
        consistency_loss = ((aligned_quats - quat_mean) ** 2).sum(dim=-1).mean()

        reg_loss = torch.tensor(0.0, device=device)
        for name, p in model.rotation_pred.named_parameters():
            reg_loss = reg_loss + ((p - original_state[name].to(device)) ** 2).sum()
        reg_loss = reg_loss * args.tta_reg_weight

        total_loss = consistency_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step == 0 or step == args.tta_steps - 1:
            print(f"     Step {step}: consistency={consistency_loss.item():.6f}, "
                  f"reg={reg_loss.item():.6f}")

    for p in model.parameters():
        p.requires_grad = False


def rotation_matrix_median_svd(Rs):
    """Element-wise median of rotation matrices + SVD projection to SO(3)."""
    R_med_raw = np.median(Rs, axis=0)
    U, _, Vt = np.linalg.svd(R_med_raw)
    R_median = U @ Vt
    if np.linalg.det(R_median) < 0:
        U[:, -1] *= -1
        R_median = U @ Vt
    return R_median


def evaluate_tta(args):
    """TTA 评估主函数"""
    t0 = time.time()

    print("=" * 80)
    print(f"TTA 评估：{args.tta_mode} 模式")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 1. Load model --
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
        _resolve_perturbation_from_ckpt, _build_eval_custom_dataset, make_collate_fn,
    )
    _resolve_perturbation_from_ckpt(args, checkpoint)
    state_dict = _auto_permute_spconv_weights(state_dict, model)
    _adapt_model_to_checkpoint(model, state_dict, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    base_state = copy.deepcopy(model.state_dict())
    print(f"   ✓ Epoch {epoch}, rotation_only={rotation_only}")

    # -- 2. Load dataset and build per-sequence indices --
    print(f"\n2. 加载数据集...")
    dataset = _build_eval_custom_dataset(args.dataset_root, args)
    eval_dataset = dataset if args.use_full_dataset else dataset

    seq_to_indices = defaultdict(list)
    if hasattr(dataset, 'all_files') and dataset.all_files:
        for idx, fpath in enumerate(dataset.all_files):
            seq_id = fpath.split('/')[0]
            seq_to_indices[seq_id].append(idx)

    seq_ids = sorted(seq_to_indices.keys())
    print(f"   ✓ {len(eval_dataset)} 样本, {len(seq_ids)} 序列")
    for sid in seq_ids:
        print(f"     Seq {sid}: {len(seq_to_indices[sid])} 帧")

    collate_fn = make_collate_fn((args.target_width, args.target_height))

    # -- 3. Per-sequence TTA evaluation --
    print(f"\n3. 开始 TTA 评估 (mode={args.tta_mode}, frames={args.tta_frames}, "
          f"steps={args.tta_steps}, lr={args.tta_lr})...")

    baseline_errors = defaultdict(list)
    tta_errors = defaultdict(list)
    tta_median_errors = defaultdict(list)
    baseline_per_seq = {}
    tta_per_seq = {}
    error_keys = ['rot_error', 'roll_error', 'pitch_error', 'yaw_error',
                  'trans_error', 'fwd_error', 'lat_error', 'ht_error']

    for seq_id in seq_ids:
        indices = seq_to_indices[seq_id]
        print(f"\n   === Seq {seq_id} ({len(indices)} 帧) ===")

        # Restore base model state before each sequence
        model.load_state_dict(base_state)
        model.eval()

        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)

        seq_dataset = Subset(dataset, indices)
        seq_loader = DataLoader(
            seq_dataset, batch_size=args.batch_size,
            num_workers=2, collate_fn=collate_fn, shuffle=False)

        # -- Baseline evaluation (no TTA) --
        seq_baseline = defaultdict(list)
        seq_pred_Ts_baseline = []
        seq_gt_Ts = []

        with torch.no_grad():
            for batch in seq_loader:
                imgs, pcs, masks, gt_T, intrinsics = batch
                gt_T_np = np.array(gt_T).astype(np.float32)
                init_T_np, _, _ = generate_single_perturbation_from_T(
                    gt_T_np, angle_range_deg=args.angle_range_deg,
                    trans_range=args.trans_range, rotation_only=rotation_only)

                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
                gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
                init_T_t = torch.from_numpy(init_T_np).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

                T_pred, _, _ = model(resize_imgs, pcs_t, gt_T_t, init_T_t,
                                     post_T, intrinsic_t, masks=masks, out_init_loss=False)
                T_pred_np = T_pred.detach().cpu().numpy()

                for i in range(len(gt_T_np)):
                    errs = compute_pose_errors(T_pred_np[i], gt_T_np[i])
                    for key in error_keys:
                        seq_baseline[key].append(errs[key])
                        baseline_errors[key].append(errs[key])
                    seq_pred_Ts_baseline.append(T_pred_np[i])
                    seq_gt_Ts.append(gt_T_np[i])

        baseline_rot = np.mean(seq_baseline['rot_error'])
        baseline_per_seq[seq_id] = {k: np.mean(v) for k, v in seq_baseline.items()}
        print(f"     Baseline: Rot={baseline_rot:.4f}°")

        # -- TTA adaptation --
        model.load_state_dict(base_state)
        model.eval()

        np.random.seed(args.eval_seed + 1000)
        torch.manual_seed(args.eval_seed + 1000)

        tta_indices = indices[:args.tta_frames]
        tta_subset = Subset(dataset, tta_indices)
        tta_loader = DataLoader(
            tta_subset, batch_size=min(args.batch_size, len(tta_indices)),
            num_workers=2, collate_fn=collate_fn, shuffle=False)

        if args.tta_mode == 'bn':
            tta_bn_adapt(model, tta_loader, device, num_passes=args.tta_bn_passes)
        elif args.tta_mode == 'head':
            tta_frame_data = []
            for batch in tta_loader:
                imgs, pcs, masks, gt_T, intrinsics = batch
                gt_T_np = np.array(gt_T).astype(np.float32)
                init_T_np, _, _ = generate_single_perturbation_from_T(
                    gt_T_np, angle_range_deg=args.angle_range_deg,
                    trans_range=args.trans_range, rotation_only=rotation_only)
                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
                init_T_t = torch.from_numpy(init_T_np).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).repeat(init_T_t.shape[0], 1, 1).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)
                tta_frame_data.append((resize_imgs, pcs_t, init_T_t, post_T, intrinsic_t))

            tta_head_adapt(model, tta_frame_data, device, args, rotation_only)
        elif args.tta_mode == 'both':
            tta_bn_adapt(model, tta_loader, device, num_passes=args.tta_bn_passes)
            tta_frame_data = []
            for batch in tta_loader:
                imgs, pcs, masks, gt_T, intrinsics = batch
                gt_T_np = np.array(gt_T).astype(np.float32)
                init_T_np, _, _ = generate_single_perturbation_from_T(
                    gt_T_np, angle_range_deg=args.angle_range_deg,
                    trans_range=args.trans_range, rotation_only=rotation_only)
                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
                init_T_t = torch.from_numpy(init_T_np).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).repeat(init_T_t.shape[0], 1, 1).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)
                tta_frame_data.append((resize_imgs, pcs_t, init_T_t, post_T, intrinsic_t))
            tta_head_adapt(model, tta_frame_data, device, args, rotation_only)

        # -- TTA evaluation with same perturbation seed --
        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)

        seq_tta = defaultdict(list)
        seq_pred_Ts_tta = []

        with torch.no_grad():
            for batch in seq_loader:
                imgs, pcs, masks, gt_T, intrinsics = batch
                gt_T_np = np.array(gt_T).astype(np.float32)
                init_T_np, _, _ = generate_single_perturbation_from_T(
                    gt_T_np, angle_range_deg=args.angle_range_deg,
                    trans_range=args.trans_range, rotation_only=rotation_only)

                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
                gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
                init_T_t = torch.from_numpy(init_T_np).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

                T_pred, _, _ = model(resize_imgs, pcs_t, gt_T_t, init_T_t,
                                     post_T, intrinsic_t, masks=masks, out_init_loss=False)
                T_pred_np = T_pred.detach().cpu().numpy()

                for i in range(len(gt_T_np)):
                    errs = compute_pose_errors(T_pred_np[i], gt_T_np[i])
                    for key in error_keys:
                        seq_tta[key].append(errs[key])
                        tta_errors[key].append(errs[key])
                    seq_pred_Ts_tta.append(T_pred_np[i])

        tta_rot = np.mean(seq_tta['rot_error'])
        tta_per_seq[seq_id] = {k: np.mean(v) for k, v in seq_tta.items()}
        delta = (baseline_rot - tta_rot) / baseline_rot * 100
        print(f"     TTA:      Rot={tta_rot:.4f}° ({'+' if delta > 0 else ''}{delta:.1f}%)")

        # Sequence median after TTA
        if args.sequence_median and seq_pred_Ts_tta:
            Rs = np.array([T[:3, :3] for T in seq_pred_Ts_tta])
            ts = np.array([T[:3, 3] for T in seq_pred_Ts_tta])
            R_median = rotation_matrix_median_svd(Rs)
            t_median = np.median(ts, axis=0)
            median_T = np.eye(4)
            median_T[:3, :3] = R_median
            median_T[:3, 3] = t_median
            median_errs = compute_pose_errors(median_T, seq_gt_Ts[0])
            for key in error_keys:
                tta_median_errors[key].append(median_errs[key])
            print(f"     TTA+Med:  Rot={median_errs['rot_error']:.4f}°")

    # -- 4. Generate report --
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.ckpt_path), 'tta_eval')
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append(f"TTA 评估报告 ({args.tta_mode} mode)")
    lines.append("=" * 80)
    lines.append(f"Checkpoint: {os.path.basename(args.ckpt_path)} (Epoch {epoch})")
    lines.append(f"TTA 模式: {args.tta_mode}")
    lines.append(f"TTA 帧数: {args.tta_frames}")
    lines.append(f"TTA 步数: {args.tta_steps}")
    lines.append(f"TTA 学习率: {args.tta_lr}")
    lines.append(f"TTA 正则化: {args.tta_reg_weight}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("一、总体效果")
    lines.append("=" * 80)

    b_rot = np.mean(baseline_errors['rot_error'])
    t_rot = np.mean(tta_errors['rot_error'])
    improve = (b_rot - t_rot) / b_rot * 100

    lines.append(f"Baseline Mean Rot: {b_rot:.4f}°")
    lines.append(f"TTA Mean Rot:      {t_rot:.4f}° ({'+' if improve > 0 else ''}{improve:.1f}%)")
    lines.append(f"  Roll:  {np.mean(baseline_errors['roll_error']):.4f}° → "
                 f"{np.mean(tta_errors['roll_error']):.4f}°")
    lines.append(f"  Pitch: {np.mean(baseline_errors['pitch_error']):.4f}° → "
                 f"{np.mean(tta_errors['pitch_error']):.4f}°")
    lines.append(f"  Yaw:   {np.mean(baseline_errors['yaw_error']):.4f}° → "
                 f"{np.mean(tta_errors['yaw_error']):.4f}°")

    if tta_median_errors['rot_error']:
        tm_rot = np.mean(tta_median_errors['rot_error'])
        tm_improve = (b_rot - tm_rot) / b_rot * 100
        lines.append(f"\nTTA + Median:      {tm_rot:.4f}° "
                     f"({'+' if tm_improve > 0 else ''}{tm_improve:.1f}%)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("二、Per-Sequence 对比")
    lines.append("=" * 80)
    header = f"{'Seq':<6} {'Base Rot°':>10} {'TTA Rot°':>10} {'改善%':>8} {'Base Roll':>10} {'TTA Roll':>10} {'Base Pitch':>11} {'TTA Pitch':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for sid in seq_ids:
        b = baseline_per_seq[sid]
        t = tta_per_seq[sid]
        d = (b['rot_error'] - t['rot_error']) / b['rot_error'] * 100
        lines.append(
            f"{sid:<6} {b['rot_error']:>10.4f} {t['rot_error']:>10.4f} "
            f"{d:>7.1f}% {b['roll_error']:>10.4f} {t['roll_error']:>10.4f} "
            f"{b['pitch_error']:>11.4f} {t['pitch_error']:>10.4f}")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "tta_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"✓ TTA 评估完成！耗时 {elapsed:.0f}s")
    print(f"  报告: {report_path}")
    print(f"{'=' * 80}")
    print(f"\n{report_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA 推理时自适应评估")

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--tta_mode", type=str, default="head",
                        choices=["bn", "head", "both"],
                        help="TTA 模式: bn=BatchNorm适配, head=回归头微调, both=两者组合")
    parser.add_argument("--tta_frames", type=int, default=30,
                        help="TTA 自适应使用的帧数")
    parser.add_argument("--tta_steps", type=int, default=5,
                        help="TTA-Head 梯度步数")
    parser.add_argument("--tta_lr", type=float, default=1e-5,
                        help="TTA-Head 学习率")
    parser.add_argument("--tta_reg_weight", type=float, default=0.1,
                        help="TTA-Head L2正则化权重")
    parser.add_argument("--tta_bn_passes", type=int, default=2,
                        help="TTA-BN 前向传播遍数")
    parser.add_argument("--sequence_median", action='store_true', default=False)

    parser.add_argument("--angle_range_deg", type=float, default=5.0)
    parser.add_argument("--trans_range", type=float, default=0.15)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--target_height", type=int, default=360)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--eval_max_frames_per_seq", type=int, default=None)
    parser.add_argument("--eval_sample_step", type=int, default=None)

    parser.add_argument("--rotation_only", type=int, default=-1)
    parser.add_argument("--deformable", type=int, default=0)
    parser.add_argument("--bev_encoder", type=int, default=1)
    parser.add_argument("--bev_pool_factor", type=int, default=0)
    parser.add_argument("--use_full_dataset", action='store_true', default=False)
    parser.add_argument("--use_drcv", action='store_true', default=False)
    parser.add_argument("--voxel_mode", type=str, default=None)
    parser.add_argument("--scatter_reduce", type=str, default=None)
    parser.add_argument("--to_bev_mode", type=str, default=None)
    parser.add_argument("--data_balance", type=int, default=0)
    parser.add_argument("--per_axis_weights", type=str, default=None)
    parser.add_argument("--per_axis_prob", type=float, default=0.0)
    parser.add_argument("--perturb_distribution", type=str, default='uniform')

    args = parser.parse_args()
    evaluate_tta(args)
