#!/usr/bin/env python3
"""
迭代推理 Recovery 验证脚本

对测试集注入固定角度扰动, 使用模型进行 1~N 次迭代推理,
评估每次迭代后的残差, 验证 Recovery > 95% 目标。

用法:
  python iterative_recovery_test.py \
    --ckpt_path logs/all_training_data_c1/model_small_5deg_c1_v62_pure_recovery_S1/all_training_data_c1_scratch/checkpoint/ckpt_best_dual.pth \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1 \
    --inject_deg 3.0 \
    --num_iterations 3 \
    --target_width 960 --target_height 540
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
    """计算总旋转误差 (degrees)"""
    R_diff = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    return angle


def rotation_error_rpy(R_pred, R_gt):
    """计算 RPY 各轴误差 (degrees)"""
    R_diff = R_pred @ R_gt.T
    r = R.from_matrix(R_diff)
    rpy = r.as_euler('xyz', degrees=True)
    return np.abs(rpy)


def inject_perturbation(gt_T, inject_deg, axis, sign):
    """对单个 4x4 矩阵注入固定角度扰动"""
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

    print(f"加载模型: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')

    model, ckpt_args, _ = _build_model_from_ckpt(args, checkpoint, device, rotation_only=True, quiet=False)
    model.eval()
    print(f"  模型加载完成")

    print(f"加载数据集: {args.dataset_root}")
    dataset = CustomDataset(
        args.dataset_root,
        target_size=(args.target_width, args.target_height),
        max_frames_per_seq=args.max_frames_per_seq,
    )
    print(f"  总样本数: {len(dataset)}")

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)

    inject_deg = args.inject_deg
    num_iters = args.num_iterations
    n_samples = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)

    print(f"\n{'='*70}")
    print(f"迭代推理 Recovery 测试")
    print(f"  注入角度: ±{inject_deg}°")
    print(f"  迭代次数: {num_iters}")
    print(f"  测试样本: {n_samples}")
    print(f"  分辨率: {args.target_width}×{args.target_height}")
    print(f"{'='*70}\n")

    results_per_iter = {i: [] for i in range(num_iters + 1)}
    results_per_axis = {axis: {i: [] for i in range(num_iters + 1)} for axis in ['roll', 'pitch', 'yaw']}
    per_sample_details = []

    t_start = time.time()
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T_list, intrinsics) in enumerate(loader):
            if sample_idx >= n_samples:
                break

            if sample_idx % 100 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / max(sample_idx, 1) * (n_samples - sample_idx) if sample_idx > 0 else 0
                print(f"  Processing {sample_idx}/{n_samples} ({sample_idx/n_samples*100:.0f}%) "
                      f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

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
                    results_per_iter[0].append(err_init)
                    results_per_axis[axis][0].append(err_init)

                    iter_errors = [err_init]
                    T_current = T_perturbed.copy()

                    for it in range(num_iters):
                        init_T_t = torch.from_numpy(T_current).float().unsqueeze(0).to(device)

                        T_pred, _, _ = model(
                            resize_imgs, pcs_t, gt_T_t, init_T_t,
                            post_T, intrinsic_t, masks=masks_t, out_init_loss=False
                        )

                        T_current = T_pred.detach().cpu().numpy()[0]

                        err = rotation_error_total(T_current[:3, :3], gt_T_single[:3, :3])
                        results_per_iter[it + 1].append(err)
                        results_per_axis[axis][it + 1].append(err)
                        iter_errors.append(err)

                    per_sample_details.append({
                        'sample_idx': sample_idx,
                        'axis': axis,
                        'sign': sign,
                        'errors': iter_errors,
                    })

            sample_idx += 1

    elapsed_total = time.time() - t_start
    print(f"\n评估完成! 耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    print(f"\n{'='*70}")
    print(f"Recovery 结果 (inject ±{inject_deg}°, {sample_idx} samples × 6 tests/sample)")
    print(f"{'='*70}\n")

    total_tests = len(results_per_iter[0])
    print(f"总测试数: {total_tests}")
    print()

    thresholds = [0.05, 0.1, 0.2, 0.5]
    print(f"{'Iteration':<12} {'Mean':>8} {'Median':>8} {'P90':>8}", end='')
    for t in thresholds:
        print(f" {'<'+str(t)+'°':>8}", end='')
    print(f" {'Recovery':>10}")
    print("-" * 90)

    for it in range(num_iters + 1):
        errs = np.array(results_per_iter[it])
        mean_e = np.mean(errs)
        med = np.median(errs)
        p90 = np.percentile(errs, 90)
        label = "Initial" if it == 0 else f"Iter {it}"
        print(f"{label:<12} {mean_e:>7.4f}° {med:>7.4f}° {p90:>7.4f}°", end='')
        for t in thresholds:
            pct = (errs < t).sum() / len(errs) * 100
            print(f" {pct:>7.1f}%", end='')
        rec01 = (errs < 0.1).sum() / len(errs) * 100
        rec_label = f"{rec01:.1f}%" if it > 0 else "—"
        print(f" {rec_label:>10}")

    print()
    print("Per-Axis Recovery (<0.1°):")
    print(f"{'Axis':<8}", end='')
    for it in range(1, num_iters + 1):
        print(f" {'Iter'+str(it):>10}", end='')
    print()
    print("-" * (8 + 11 * num_iters))
    for axis in ['roll', 'pitch', 'yaw']:
        print(f"{axis:<8}", end='')
        for it in range(1, num_iters + 1):
            errs = np.array(results_per_axis[axis][it])
            rec01 = (errs < 0.1).sum() / len(errs) * 100
            print(f" {rec01:>9.1f}%", end='')
        print()

    output_data = {
        'config': {
            'inject_deg': inject_deg,
            'num_iterations': num_iters,
            'n_samples': sample_idx,
            'total_tests': total_tests,
            'ckpt_path': args.ckpt_path,
            'target_width': args.target_width,
            'target_height': args.target_height,
        },
        'summary': {},
        'per_axis': {},
    }
    for it in range(num_iters + 1):
        errs = np.array(results_per_iter[it])
        output_data['summary'][f'iter_{it}'] = {
            'mean': float(np.mean(errs)),
            'median': float(np.median(errs)),
            'std': float(np.std(errs)),
            'p90': float(np.percentile(errs, 90)),
            'p95': float(np.percentile(errs, 95)),
            'recovery_005': float((errs < 0.05).sum() / len(errs)),
            'recovery_01': float((errs < 0.1).sum() / len(errs)),
            'recovery_02': float((errs < 0.2).sum() / len(errs)),
            'recovery_05': float((errs < 0.5).sum() / len(errs)),
        }
    for axis in ['roll', 'pitch', 'yaw']:
        output_data['per_axis'][axis] = {}
        for it in range(num_iters + 1):
            errs = np.array(results_per_axis[axis][it])
            output_data['per_axis'][axis][f'iter_{it}'] = {
                'mean': float(np.mean(errs)),
                'median': float(np.median(errs)),
                'recovery_01': float((errs < 0.1).sum() / len(errs)),
                'recovery_02': float((errs < 0.2).sum() / len(errs)),
            }

    output_dir = args.output_dir or os.path.dirname(args.ckpt_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'iterative_recovery_inject{inject_deg}deg.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n结果保存到: {output_file}")

    details_file = os.path.join(output_dir, f'iterative_recovery_inject{inject_deg}deg_details.json')
    with open(details_file, 'w') as f:
        json.dump(per_sample_details, f)
    print(f"详细结果保存到: {details_file}")

    final_rec = output_data['summary'][f'iter_{num_iters}']['recovery_01']
    print(f"\n{'★'*30}")
    print(f"  最终 Recovery (<0.1°): {final_rec*100:.1f}%")
    print(f"  目标: > 95%")
    if final_rec > 0.95:
        print(f"  ✓ 达标!")
    else:
        print(f"  ✗ 未达标 (差距: {(0.95 - final_rec)*100:.1f}%)")
        iter2_rec = output_data['summary'].get('iter_2', {}).get('recovery_01', 0)
        if iter2_rec > 0.95:
            print(f"  → Iter 2 已达标 ({iter2_rec*100:.1f}%), 建议部署使用 2 次迭代")
    print(f"{'★'*30}")

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='迭代推理 Recovery 验证')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--inject_deg', type=float, default=3.0)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--target_width', type=int, default=960)
    parser.add_argument('--target_height', type=int, default=540)
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Max samples to test (0=all)')
    parser.add_argument('--max_frames_per_seq', type=int, default=500)
    parser.add_argument('--pitch_vertical_bands', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='')

    args = parser.parse_args()
    run_iterative_test(args)
