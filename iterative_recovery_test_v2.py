#!/usr/bin/env python3
"""
迭代推理 Recovery 验证脚本 v2

使用模型自带的 iterative_inference 方法进行验证。
对比两种迭代方式:
1. 模型原生 iterative_inference (使用 _core_forward)
2. 使用 training forward (带 loss_fn 的 T_composed)

用法:
  python iterative_recovery_test_v2.py \
    --ckpt_path logs/all_training_data_c1/model_small_5deg_c1_v62_pure_recovery_S1/all_training_data_c1_scratch/checkpoint/ckpt_best_dual.pth \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1 \
    --inject_deg 3.0 \
    --num_iterations 3
"""

import torch
import numpy as np
import os
import sys
import argparse
import json
import time
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, 'kitti-bev-calib'))
sys.path.insert(0, _script_dir)

_ckpt_path_cli = None
for _i, _a in enumerate(sys.argv):
    if _a == '--ckpt_path' and _i + 1 < len(sys.argv):
        _ckpt_path_cli = sys.argv[_i + 1]
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


def rotation_error_total(R_pred, R_gt):
    R_diff = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    return angle


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


def run_iterative_test(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"加载模型: {args.ckpt_path}", flush=True)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')

    model, ckpt_args, _ = _build_model_from_ckpt(args, checkpoint, device, rotation_only=True, quiet=False)
    model.eval()
    print(f"  模型加载完成", flush=True)

    print(f"加载数据集: {args.dataset_root}", flush=True)
    dataset = CustomDataset(
        args.dataset_root,
        target_size=(args.target_width, args.target_height),
        max_frames_per_seq=args.max_frames_per_seq,
    )
    print(f"  总样本数: {len(dataset)}", flush=True)

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)

    inject_deg = args.inject_deg
    num_iters = args.num_iterations
    n_samples = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)

    print(f"\n{'='*70}", flush=True)
    print(f"迭代推理 Recovery 测试 v2 (使用原生 iterative_inference)", flush=True)
    print(f"  注入角度: ±{inject_deg}°", flush=True)
    print(f"  迭代次数: {num_iters}", flush=True)
    print(f"  测试样本: {n_samples}", flush=True)
    print(f"  分辨率: {args.target_width}×{args.target_height}", flush=True)
    print(f"{'='*70}\n", flush=True)

    has_iterative_inference = hasattr(model, 'iterative_inference')
    print(f"  模型是否有 iterative_inference: {has_iterative_inference}", flush=True)

    results_native = {i: [] for i in range(num_iters + 1)}
    results_forward = {i: [] for i in range(num_iters + 1)}
    results_per_axis_native = {axis: {i: [] for i in range(num_iters + 1)} for axis in ['roll', 'pitch', 'yaw']}

    t_start = time.time()
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T_list, intrinsics) in enumerate(loader):
            if sample_idx >= n_samples:
                break

            if sample_idx % 50 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / max(sample_idx, 1) * (n_samples - sample_idx) if sample_idx > 0 else 0
                print(f"  Processing {sample_idx}/{n_samples} ({sample_idx/n_samples*100:.0f}%) "
                      f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]", flush=True)

            gt_T_np = np.array(gt_T_list).astype(np.float32)
            if gt_T_np.ndim == 3:
                gt_T_single = gt_T_np[0]
            else:
                gt_T_single = gt_T_np

            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_np = np.array(pcs)[:, :, :3]
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
            intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)
            masks_t = torch.from_numpy(np.array(masks)).float().to(device)
            post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)

            for axis in ['roll', 'pitch', 'yaw']:
                for sign in [-1, 1]:
                    T_perturbed = inject_perturbation(gt_T_single, inject_deg, axis, sign)

                    err_init = rotation_error_total(T_perturbed[:3, :3], gt_T_single[:3, :3])
                    results_native[0].append(err_init)
                    results_per_axis_native[axis][0].append(err_init)
                    results_forward[0].append(err_init)

                    init_T_t = torch.from_numpy(T_perturbed).float().unsqueeze(0).to(device)

                    # Method 1: native iterative_inference (step by step)
                    if has_iterative_inference:
                        T_current_native = init_T_t.clone()
                        for it in range(num_iters):
                            T_refined = model.iterative_inference(
                                resize_imgs, pcs_t, T_current_native, intrinsic_t,
                                n_iters=1, pcd_mask=masks_t
                            )
                            T_current_native = T_refined
                            err = rotation_error_total(
                                T_current_native[0, :3, :3].cpu().numpy(),
                                gt_T_single[:3, :3]
                            )
                            results_native[it + 1].append(err)
                            results_per_axis_native[axis][it + 1].append(err)

                    # Method 2: training forward (for comparison on first iter)
                    T_composed, _, _ = model(
                        resize_imgs, pcs_t, gt_T_t, init_T_t,
                        post_T, intrinsic_t, masks=masks_t, out_init_loss=False
                    )
                    err_fwd = rotation_error_total(
                        T_composed[0, :3, :3].detach().cpu().numpy(),
                        gt_T_single[:3, :3]
                    )
                    results_forward[1].append(err_fwd)

                    # Method 2: continue iterating with forward
                    T_current_fwd = T_composed.detach()
                    for it in range(1, num_iters):
                        T_composed2, _, _ = model(
                            resize_imgs, pcs_t, gt_T_t, T_current_fwd,
                            post_T, intrinsic_t, masks=masks_t, out_init_loss=False
                        )
                        T_current_fwd = T_composed2.detach()
                        err = rotation_error_total(
                            T_current_fwd[0, :3, :3].cpu().numpy(),
                            gt_T_single[:3, :3]
                        )
                        results_forward[it + 1].append(err)

            sample_idx += 1

    elapsed_total = time.time() - t_start
    print(f"\n评估完成! 耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Recovery 结果 (inject ±{inject_deg}°, {sample_idx} samples × 6 tests)", flush=True)
    print(f"{'='*70}\n", flush=True)

    total_tests = len(results_native[0])
    print(f"总测试数: {total_tests}\n", flush=True)

    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]

    if has_iterative_inference:
        print("=== Method 1: Native iterative_inference ===")
        print(f"{'Iteration':<12} {'Mean':>8} {'Median':>8} {'P90':>8}", end='')
        for t in thresholds:
            print(f" {'<'+str(t)+'°':>8}", end='')
        print()
        print("-" * 80)
        for it in range(num_iters + 1):
            errs = np.array(results_native[it])
            label = "Initial" if it == 0 else f"Iter {it}"
            print(f"{label:<12} {np.mean(errs):>7.4f}° {np.median(errs):>7.4f}° {np.percentile(errs, 90):>7.4f}°", end='')
            for t in thresholds:
                pct = (errs < t).sum() / len(errs) * 100
                print(f" {pct:>7.1f}%", end='')
            print()

    print(f"\n=== Method 2: Training forward (T_composed) ===")
    print(f"{'Iteration':<12} {'Mean':>8} {'Median':>8} {'P90':>8}", end='')
    for t in thresholds:
        print(f" {'<'+str(t)+'°':>8}", end='')
    print()
    print("-" * 80)
    for it in range(num_iters + 1):
        if it not in results_forward or not results_forward[it]:
            continue
        errs = np.array(results_forward[it])
        label = "Initial" if it == 0 else f"Iter {it}"
        print(f"{label:<12} {np.mean(errs):>7.4f}° {np.median(errs):>7.4f}° {np.percentile(errs, 90):>7.4f}°", end='')
        for t in thresholds:
            pct = (errs < t).sum() / len(errs) * 100
            print(f" {pct:>7.1f}%", end='')
        print()

    if has_iterative_inference:
        print(f"\nPer-Axis (Native method):", flush=True)
        print(f"{'Axis':<8}", end='')
        for it in range(1, num_iters + 1):
            print(f" {'Iter'+str(it)+' Mean':>12} {'<0.1°':>8} {'<0.5°':>8}", end='')
        print()
        print("-" * (8 + 28 * num_iters))
        for axis in ['roll', 'pitch', 'yaw']:
            print(f"{axis:<8}", end='')
            for it in range(1, num_iters + 1):
                errs = np.array(results_per_axis_native[axis][it])
                mean_e = np.mean(errs)
                rec01 = (errs < 0.1).sum() / len(errs) * 100
                rec05 = (errs < 0.5).sum() / len(errs) * 100
                print(f" {mean_e:>11.4f}° {rec01:>7.1f}% {rec05:>7.1f}%", end='')
            print()

    output_data = {
        'config': {
            'inject_deg': inject_deg,
            'num_iterations': num_iters,
            'n_samples': sample_idx,
            'total_tests': total_tests,
            'ckpt_path': args.ckpt_path,
        },
        'native_iterative': {},
        'forward_iterative': {},
    }
    for it in range(num_iters + 1):
        if results_native[it]:
            errs = np.array(results_native[it])
            output_data['native_iterative'][f'iter_{it}'] = {
                'mean': float(np.mean(errs)),
                'median': float(np.median(errs)),
                'std': float(np.std(errs)),
                'p90': float(np.percentile(errs, 90)),
                'recovery_01': float((errs < 0.1).sum() / len(errs)),
                'recovery_02': float((errs < 0.2).sum() / len(errs)),
                'recovery_05': float((errs < 0.5).sum() / len(errs)),
                'recovery_10': float((errs < 1.0).sum() / len(errs)),
            }
        if results_forward[it]:
            errs = np.array(results_forward[it])
            output_data['forward_iterative'][f'iter_{it}'] = {
                'mean': float(np.mean(errs)),
                'median': float(np.median(errs)),
                'recovery_01': float((errs < 0.1).sum() / len(errs)),
                'recovery_05': float((errs < 0.5).sum() / len(errs)),
            }

    output_dir = args.output_dir or os.path.dirname(args.ckpt_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'iterative_recovery_v2_inject{inject_deg}deg.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n结果保存到: {output_file}", flush=True)

    if has_iterative_inference:
        final_rec = output_data['native_iterative'].get(f'iter_{num_iters}', {}).get('recovery_01', 0)
    else:
        final_rec = output_data['forward_iterative'].get(f'iter_{num_iters}', {}).get('recovery_01', 0)

    print(f"\n{'★'*30}")
    print(f"  最终 Recovery (<0.1°): {final_rec*100:.1f}%")
    print(f"  目标: > 95%")
    if final_rec > 0.95:
        print(f"  ✓ 达标!")
    else:
        print(f"  ✗ 未达标 (差距: {(0.95 - final_rec)*100:.1f}%)")
        native_iter = output_data.get('native_iterative', {})
        for i in range(1, num_iters + 1):
            r05 = native_iter.get(f'iter_{i}', {}).get('recovery_05', 0)
            r10 = native_iter.get(f'iter_{i}', {}).get('recovery_10', 0)
            if r05 > 0 or r10 > 0:
                print(f"    Iter {i}: <0.5°={r05*100:.1f}%, <1.0°={r10*100:.1f}%")
    print(f"{'★'*30}", flush=True)

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='迭代推理 Recovery 验证 v2')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--inject_deg', type=float, default=3.0)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--target_width', type=int, default=960)
    parser.add_argument('--target_height', type=int, default=540)
    parser.add_argument('--max_samples', type=int, default=200,
                        help='Max samples to test (0=all)')
    parser.add_argument('--max_frames_per_seq', type=int, default=500)
    parser.add_argument('--pitch_vertical_bands', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='')

    args = parser.parse_args()
    run_iterative_test(args)
