#!/usr/bin/env python3
"""
ZD 在线标定补偿验证

模拟真实部署场景：
  - 每个 sequence 代表一段 trip（行程）
  - Trip 前 N 帧用于 ZD 标定（估计模型的系统偏差）
  - 剩余帧用于验证 ZD 补偿后的 Recovery

验证目标：ZD 补偿后 inject 3° 迭代推理 Recovery (<0.1°) > 95%

用法:
  python zd_online_calibration_test.py \
    --ckpt_path logs/.../ckpt_best_dual.pth \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_c1 \
    --inject_deg 3.0 \
    --num_iterations 3 \
    --zd_calib_frames 30
"""

import torch
import numpy as np
import os
import sys
import argparse
import json
import time
from torch.utils.data import DataLoader, Subset
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


def rotation_error_rpy(R_pred, R_gt):
    R_diff = R_pred @ R_gt.T
    r = R.from_matrix(R_diff)
    return r.as_euler('xyz', degrees=True)


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


def compensate_zd(T_pred, zd_rpy_rad):
    """从模型输出中减去 ZD 偏差"""
    R_zd_inv = R.from_euler('xyz', -zd_rpy_rad).as_matrix().astype(np.float32)
    T_comp = T_pred.copy()
    T_comp[:3, :3] = R_zd_inv @ T_pred[:3, :3]
    return T_comp


def run_zd_calibration_test(args):
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

    seq_info = dataset.seq_to_domain_id if hasattr(dataset, 'seq_to_domain_id') else {}
    seq_boundaries = []
    if hasattr(dataset, 'all_files'):
        current_seq = None
        start_idx = 0
        for i, f in enumerate(dataset.all_files):
            seq = f.split('/')[0]
            if seq != current_seq:
                if current_seq is not None:
                    seq_boundaries.append((current_seq, start_idx, i))
                current_seq = seq
                start_idx = i
        if current_seq is not None:
            seq_boundaries.append((current_seq, start_idx, len(dataset.all_files)))
    
    if not seq_boundaries:
        total = len(dataset)
        seq_boundaries = [('all', 0, total)]

    print(f"\n  序列划分: {len(seq_boundaries)} 个 trip", flush=True)
    for seq_name, start, end in seq_boundaries:
        print(f"    {seq_name}: frames {start}-{end} ({end-start} 帧)", flush=True)

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    inject_deg = args.inject_deg
    num_iters = args.num_iterations
    zd_calib_frames = args.zd_calib_frames

    print(f"\n{'='*70}", flush=True)
    print(f"ZD 在线标定补偿验证", flush=True)
    print(f"  注入角度: ±{inject_deg}°", flush=True)
    print(f"  迭代次数: {num_iters}", flush=True)
    print(f"  ZD 标定帧数: {zd_calib_frames} (每 trip 前 N 帧)", flush=True)
    print(f"  分辨率: {args.target_width}×{args.target_height}", flush=True)
    print(f"{'='*70}\n", flush=True)

    all_results = {
        'per_seq': {},
        'global_no_comp': {i: [] for i in range(num_iters + 1)},
        'global_with_comp': {i: [] for i in range(num_iters + 1)},
    }

    t_start = time.time()

    with torch.no_grad():
        for seq_name, seq_start, seq_end in seq_boundaries:
            seq_len = seq_end - seq_start
            if seq_len < zd_calib_frames + 10:
                print(f"\n  [Skip] {seq_name}: 帧数不足 ({seq_len} < {zd_calib_frames + 10})", flush=True)
                continue

            print(f"\n  === Trip: {seq_name} ({seq_len} frames) ===", flush=True)

            # Phase 1: ZD 标定（用前 zd_calib_frames 帧，inject 0° 测量偏差）
            print(f"    Phase 1: ZD 标定 (前 {zd_calib_frames} 帧)...", flush=True)
            zd_rpy_samples = []

            calib_indices = list(range(seq_start, min(seq_start + zd_calib_frames, seq_end)))
            calib_subset = Subset(dataset, calib_indices)
            calib_loader = DataLoader(calib_subset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)

            for batch_idx, (imgs, pcs, masks, gt_T_list, intrinsics) in enumerate(calib_loader):
                gt_T_np = np.array(gt_T_list).astype(np.float32)
                gt_T_single = gt_T_np[0] if gt_T_np.ndim == 3 else gt_T_np

                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_np = np.array(pcs)[:, :, :3]
                pcs_t = torch.from_numpy(pcs_np).float().to(device)
                gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)
                masks_t = torch.from_numpy(np.array(masks)).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).to(device)

                # inject 0°: init_T = GT
                init_T_t = gt_T_t.clone()
                T_pred, _, _ = model(
                    resize_imgs, pcs_t, gt_T_t, init_T_t,
                    post_T, intrinsic_t, masks=masks_t, out_init_loss=False
                )
                T_pred_np = T_pred[0].cpu().numpy()

                rpy_err = rotation_error_rpy(T_pred_np[:3, :3], gt_T_single[:3, :3])
                zd_rpy_samples.append(np.radians(rpy_err))

            zd_rpy_samples = np.array(zd_rpy_samples)
            zd_median = np.median(zd_rpy_samples, axis=0)
            zd_mean = np.mean(zd_rpy_samples, axis=0)
            zd_std = np.std(zd_rpy_samples, axis=0)

            zd_deg = np.degrees(zd_median)
            print(f"    ZD (median, R/P/Y): {zd_deg[0]:.4f}° / {zd_deg[1]:.4f}° / {zd_deg[2]:.4f}°", flush=True)
            print(f"    ZD total: {np.linalg.norm(zd_deg):.4f}°", flush=True)
            print(f"    ZD std:   {np.degrees(zd_std[0]):.4f}° / {np.degrees(zd_std[1]):.4f}° / {np.degrees(zd_std[2]):.4f}°", flush=True)

            # Phase 2: 验证（剩余帧，inject ±deg 迭代推理 + ZD 补偿）
            eval_start = seq_start + zd_calib_frames
            eval_end = min(eval_start + args.max_eval_per_seq, seq_end)
            eval_indices = list(range(eval_start, eval_end))
            eval_subset = Subset(dataset, eval_indices)
            eval_loader = DataLoader(eval_subset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)

            n_eval = len(eval_indices)
            print(f"    Phase 2: 验证 ({n_eval} frames, inject ±{inject_deg}°, {num_iters} iters)...", flush=True)

            seq_results_no_comp = {i: [] for i in range(num_iters + 1)}
            seq_results_with_comp = {i: [] for i in range(num_iters + 1)}

            for batch_idx, (imgs, pcs, masks, gt_T_list, intrinsics) in enumerate(eval_loader):
                gt_T_np = np.array(gt_T_list).astype(np.float32)
                gt_T_single = gt_T_np[0] if gt_T_np.ndim == 3 else gt_T_np

                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_np = np.array(pcs)[:, :, :3]
                pcs_t = torch.from_numpy(pcs_np).float().to(device)
                gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)
                masks_t = torch.from_numpy(np.array(masks)).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).to(device)

                for axis in ['roll', 'pitch', 'yaw']:
                    for sign in [-1, 1]:
                        T_perturbed = inject_perturbation(gt_T_single, inject_deg, axis, sign)
                        err_init = rotation_error_total(T_perturbed[:3, :3], gt_T_single[:3, :3])
                        seq_results_no_comp[0].append(err_init)
                        seq_results_with_comp[0].append(err_init)

                        T_current = T_perturbed.copy()
                        for it in range(num_iters):
                            init_T_t = torch.from_numpy(T_current).float().unsqueeze(0).to(device)
                            T_pred, _, _ = model(
                                resize_imgs, pcs_t, gt_T_t, init_T_t,
                                post_T, intrinsic_t, masks=masks_t, out_init_loss=False
                            )
                            T_pred_np = T_pred[0].cpu().numpy()

                            # Without compensation
                            err_no_comp = rotation_error_total(T_pred_np[:3, :3], gt_T_single[:3, :3])
                            seq_results_no_comp[it + 1].append(err_no_comp)

                            # With ZD compensation
                            T_comp = compensate_zd(T_pred_np, zd_median)
                            err_with_comp = rotation_error_total(T_comp[:3, :3], gt_T_single[:3, :3])
                            seq_results_with_comp[it + 1].append(err_with_comp)

                            # Next iteration uses compensated result
                            T_current = T_comp.copy()

                if (batch_idx + 1) % 50 == 0:
                    print(f"      {batch_idx+1}/{n_eval} frames processed", flush=True)

            # Aggregate per-seq results
            for it in range(num_iters + 1):
                all_results['global_no_comp'][it].extend(seq_results_no_comp[it])
                all_results['global_with_comp'][it].extend(seq_results_with_comp[it])

            last_iter = num_iters
            errs_no = np.array(seq_results_no_comp[last_iter])
            errs_yes = np.array(seq_results_with_comp[last_iter])
            rec_no = (errs_no < 0.1).sum() / len(errs_no) * 100 if len(errs_no) > 0 else 0
            rec_yes = (errs_yes < 0.1).sum() / len(errs_yes) * 100 if len(errs_yes) > 0 else 0

            all_results['per_seq'][seq_name] = {
                'zd_rpy_deg': zd_deg.tolist(),
                'zd_total_deg': float(np.linalg.norm(zd_deg)),
                'zd_std_deg': np.degrees(zd_std).tolist(),
                'n_calib': len(zd_rpy_samples),
                'n_eval': n_eval,
                'recovery_no_comp': rec_no,
                'recovery_with_comp': rec_yes,
                'mean_err_no_comp': float(np.mean(errs_no)),
                'mean_err_with_comp': float(np.mean(errs_yes)),
            }

            print(f"    结果: 无补偿 Rec={rec_no:.1f}%, 有补偿 Rec={rec_yes:.1f}%", flush=True)
            print(f"           无补偿 Mean={np.mean(errs_no):.4f}°, 有补偿 Mean={np.mean(errs_yes):.4f}°", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}", flush=True)
    print(f"总体结果 (inject ±{inject_deg}°, {num_iters} iterations)", flush=True)
    print(f"{'='*70}\n", flush=True)

    print(f"{'Method':<20} {'Iter':<8} {'Mean':>8} {'Median':>8} {'<0.05°':>8} {'<0.1°':>8} {'<0.2°':>8} {'<0.5°':>8}", flush=True)
    print("-" * 80, flush=True)

    for method, results_dict in [('无补偿', all_results['global_no_comp']), ('ZD补偿', all_results['global_with_comp'])]:
        for it in range(num_iters + 1):
            errs = np.array(results_dict[it])
            if len(errs) == 0:
                continue
            label = "Init" if it == 0 else f"Iter{it}"
            mean_e = np.mean(errs)
            med = np.median(errs)
            r005 = (errs < 0.05).sum() / len(errs) * 100
            r01 = (errs < 0.1).sum() / len(errs) * 100
            r02 = (errs < 0.2).sum() / len(errs) * 100
            r05 = (errs < 0.5).sum() / len(errs) * 100
            print(f"{method:<20} {label:<8} {mean_e:>7.4f}° {med:>7.4f}° {r005:>7.1f}% {r01:>7.1f}% {r02:>7.1f}% {r05:>7.1f}%", flush=True)

    print(f"\nPer-Trip ZD 分析:", flush=True)
    print(f"{'Trip':<10} {'ZD(°)':>8} {'Rec无补':>8} {'Rec有补':>8} {'Mean无补':>10} {'Mean有补':>10}", flush=True)
    print("-" * 60, flush=True)
    for seq_name, info in all_results['per_seq'].items():
        print(f"{seq_name:<10} {info['zd_total_deg']:>7.3f}° {info['recovery_no_comp']:>7.1f}% "
              f"{info['recovery_with_comp']:>7.1f}% {info['mean_err_no_comp']:>9.4f}° {info['mean_err_with_comp']:>9.4f}°", flush=True)

    # Final verdict
    final_errs = np.array(all_results['global_with_comp'][num_iters])
    final_rec = (final_errs < 0.1).sum() / len(final_errs) * 100 if len(final_errs) > 0 else 0

    print(f"\n{'★'*30}", flush=True)
    print(f"  ZD补偿后最终 Recovery (<0.1°): {final_rec:.1f}%", flush=True)
    print(f"  目标: > 95%", flush=True)
    if final_rec > 95:
        print(f"  ✓ 达标!", flush=True)
    else:
        print(f"  ✗ 未达标 (差距: {95 - final_rec:.1f}%)", flush=True)
        final_rec_02 = (final_errs < 0.2).sum() / len(final_errs) * 100
        print(f"  Recovery (<0.2°): {final_rec_02:.1f}%", flush=True)
    print(f"{'★'*30}", flush=True)

    # Save results
    output_dir = args.output_dir or 'logs/evaluations/zd_compensation_test'
    os.makedirs(output_dir, exist_ok=True)

    save_data = {
        'config': vars(args),
        'per_seq': all_results['per_seq'],
        'global_summary': {},
    }
    for method_key in ['global_no_comp', 'global_with_comp']:
        save_data['global_summary'][method_key] = {}
        for it in range(num_iters + 1):
            errs = np.array(all_results[method_key][it])
            if len(errs) > 0:
                save_data['global_summary'][method_key][f'iter_{it}'] = {
                    'mean': float(np.mean(errs)),
                    'median': float(np.median(errs)),
                    'std': float(np.std(errs)),
                    'recovery_005': float((errs < 0.05).sum() / len(errs)),
                    'recovery_01': float((errs < 0.1).sum() / len(errs)),
                    'recovery_02': float((errs < 0.2).sum() / len(errs)),
                    'recovery_05': float((errs < 0.5).sum() / len(errs)),
                }

    output_file = os.path.join(output_dir, f'zd_compensation_inject{inject_deg}deg_calib{zd_calib_frames}.json')
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n结果保存到: {output_file}", flush=True)
    print(f"耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)", flush=True)

    return save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZD 在线标定补偿验证')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--inject_deg', type=float, default=3.0)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--target_width', type=int, default=960)
    parser.add_argument('--target_height', type=int, default=540)
    parser.add_argument('--max_frames_per_seq', type=int, default=500)
    parser.add_argument('--max_eval_per_seq', type=int, default=100,
                        help='每个 trip 最多评估帧数')
    parser.add_argument('--zd_calib_frames', type=int, default=30,
                        help='每个 trip 用于 ZD 标定的帧数')
    parser.add_argument('--pitch_vertical_bands', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='')

    args = parser.parse_args()
    run_zd_calibration_test(args)
