#!/usr/bin/env python3
"""
对已保存的 checkpoint 进行评估和可视化
用于对已训练完成的 checkpoint 生成评估报告
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import cv2
import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

from custom_dataset import CustomDataset
from bev_calib import BEVCalib
from tools import generate_single_perturbation_from_T
from visualization import (
    compute_batch_pose_errors,
    visualize_batch_projection,
    compute_pose_errors
)


def _parse_train_log_errors(ckpt_path, target_epoch, existing_train, existing_val):
    """
    从 train.log 中解析指定 epoch 的 train/val 误差。
    搜索路径: checkpoint所在目录向上查找 train.log。
    """
    if not isinstance(target_epoch, int) or target_epoch <= 0:
        return existing_train, existing_val

    search_dir = os.path.dirname(os.path.abspath(ckpt_path))
    log_path = None
    for _ in range(5):
        candidate = os.path.join(search_dir, 'train.log')
        if os.path.isfile(candidate):
            log_path = candidate
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent

    if log_path is None:
        return existing_train, existing_val

    train_errors = existing_train
    val_errors = existing_val

    epoch_tag = f"Epoch [{target_epoch}/"

    # Patterns:
    # Train Pose Error - Trans: 0.0123m (Fwd:0.005m Lat:0.004m Ht:0.003m), Rot: 0.15° (Roll:0.05° ...
    # Train Pose Error - Rot: 0.15° (Roll:0.05° Pitch:0.08° Yaw:0.02°)
    # Val[±10°, ±0.3m] Pose Error - (same as above)
    rot_only_pat = re.compile(
        r'Pose Error - Rot:\s*([\d.]+).*Roll:([\d.]+).*Pitch:([\d.]+).*Yaw:([\d.]+)')
    full_pat = re.compile(
        r'Pose Error - Trans:\s*([\d.]+)m\s*\(Fwd:([\d.]+).*?Lat:([\d.]+).*?Ht:([\d.]+).*?'
        r'Rot:\s*([\d.]+).*?Roll:([\d.]+).*?Pitch:([\d.]+).*?Yaw:([\d.]+)')

    def _build_errors(m, is_full):
        if is_full:
            return {
                'trans_error': float(m.group(1)), 'fwd_error': float(m.group(2)),
                'lat_error': float(m.group(3)), 'ht_error': float(m.group(4)),
                'rot_error': float(m.group(5)), 'roll_error': float(m.group(6)),
                'pitch_error': float(m.group(7)), 'yaw_error': float(m.group(8)),
            }
        return {
            'trans_error': 0.0, 'fwd_error': 0.0, 'lat_error': 0.0, 'ht_error': 0.0,
            'rot_error': float(m.group(1)), 'roll_error': float(m.group(2)),
            'pitch_error': float(m.group(3)), 'yaw_error': float(m.group(4)),
        }

    # Also parse "Checkpoint N Eval Pose Error" lines
    ckpt_eval_pat = re.compile(
        r'Checkpoint\s+(\d+)\s+Eval Pose Error\s*-\s*Rot:\s*([\d.]+).*?'
        r'R:([\d.]+)\s+P:([\d.]+)\s+Y:([\d.]+)')

    # Collect nearest val errors as fallback
    last_val_errors = None
    last_val_epoch = -1

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Match exact epoch
                if epoch_tag in line:
                    is_train = 'Train Pose Error' in line
                    is_val = 'Val' in line and 'Pose Error' in line
                    if is_train or is_val:
                        m_full = full_pat.search(line)
                        m_rot = rot_only_pat.search(line)
                        if m_full:
                            errs = _build_errors(m_full, True)
                        elif m_rot:
                            errs = _build_errors(m_rot, False)
                        else:
                            continue

                        if is_train and train_errors is None:
                            train_errors = errs
                            print(f"   (从 train.log 解析 Epoch {target_epoch} Train errors)")
                        elif is_val and val_errors is None:
                            val_errors = errs
                            print(f"   (从 train.log 解析 Epoch {target_epoch} Val errors)")

                # Match checkpoint eval line
                m_ckpt = ckpt_eval_pat.search(line)
                if m_ckpt and int(m_ckpt.group(1)) == target_epoch and val_errors is None:
                    val_errors = {
                        'trans_error': 0.0, 'fwd_error': 0.0, 'lat_error': 0.0, 'ht_error': 0.0,
                        'rot_error': float(m_ckpt.group(2)), 'roll_error': float(m_ckpt.group(3)),
                        'pitch_error': float(m_ckpt.group(4)), 'yaw_error': float(m_ckpt.group(5)),
                    }
                    print(f"   (从 train.log 解析 Epoch {target_epoch} Checkpoint Eval errors 作为 Val)")

                # Track the latest val error seen (any epoch) as fallback
                if 'Val' in line and 'Pose Error' in line:
                    ep_match = re.search(r'Epoch\s*\[(\d+)/', line)
                    if ep_match:
                        ep_num = int(ep_match.group(1))
                        m_f = full_pat.search(line)
                        m_r = rot_only_pat.search(line)
                        if m_f:
                            last_val_errors = _build_errors(m_f, True)
                            last_val_epoch = ep_num
                        elif m_r:
                            last_val_errors = _build_errors(m_r, False)
                            last_val_epoch = ep_num
    except Exception as e:
        print(f"   [WARN] 解析 train.log 失败: {e}")

    # Fallback: use the latest val errors from any previous epoch
    if val_errors is None and last_val_errors is not None and last_val_epoch > 0:
        val_errors = last_val_errors
        print(f"   (使用最近的 Val 误差: Epoch {last_val_epoch})")

    return train_errors, val_errors

def make_collate_fn(target_size):
    """创建 collate 函数"""
    def crop_and_resize(item, size, intrinsic):
        """
        缩放图像并相应调整内参
        
        ⚠️ 重要修复：图像缩放时必须同步调整内参矩阵！
        """
        img = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        
        # 缩放图像
        resized = cv2.resize(img, size)
        
        # ✅ 修复：调整内参以匹配缩放后的图像
        scale_x = size[0] / w
        scale_y = size[1] / h
        
        new_intrinsic = np.array([
            [intrinsic[0, 0] * scale_x, 0, intrinsic[0, 2] * scale_x],
            [0, intrinsic[1, 1] * scale_y, intrinsic[1, 2] * scale_y],
            [0, 0, 1]
        ])
        
        # 🔍 添加调试输出（只输出第一个样本）
        global _debug_intrinsic_printed
        if not hasattr(crop_and_resize, '_debug_printed'):
            print(f"\n[DEBUG crop_and_resize] 图像尺寸: {w}x{h} → {size[0]}x{size[1]}")
            print(f"[DEBUG crop_and_resize] 缩放比例: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")
            print(f"[DEBUG crop_and_resize] 原始内参K:")
            print(intrinsic)
            print(f"[DEBUG crop_and_resize] 缩放后内参K:")
            print(new_intrinsic)
            crop_and_resize._debug_printed = True
        
        return resized, new_intrinsic
    
    def collate_fn(batch):
        # ✅ 修复：同时处理图像和内参
        processed = [crop_and_resize(item[0], target_size, item[3]) for item in batch]
        imgs = [p[0] for p in processed]
        intrinsics = [p[1] for p in processed]
        gt_T_to_camera = [item[2] for item in batch]
        distortions = [item[4] if len(item) > 4 else None for item in batch]  # 畸变系数
        
        pcs = []
        masks = []
        max_num_points = 0
        for item in batch:
            max_num_points = max(max_num_points, item[1].shape[0])
        for item in batch:
            pc = item[1]
            masks.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(max_num_points - pc.shape[0])], axis=0))
            if pc.shape[0] < max_num_points:
                pc = np.concatenate([pc, np.full((max_num_points - pc.shape[0], pc.shape[1]), 999999)], axis=0)
            pcs.append(pc)

        return imgs, pcs, masks, gt_T_to_camera, intrinsics, distortions
    
    return collate_fn

def evaluate_checkpoint(args):
    """评估指定的 checkpoint"""
    
    print("=" * 80)
    print(f"评估 Checkpoint: {args.ckpt_path}")
    if args.output_dir:
        print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 checkpoint
    print(f"\n1. 加载 checkpoint...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"   Epoch: {epoch}")
    
    # 确定 rotation_only 设置
    if args.rotation_only == -1:
        if 'rotation_only' in checkpoint:
            rotation_only = checkpoint['rotation_only']
        elif 'optimize_translation' in checkpoint:
            rotation_only = not checkpoint['optimize_translation']
        else:
            rotation_only = False
        print(f"   rotation_only: {rotation_only} (从checkpoint读取)")
    else:
        rotation_only = args.rotation_only > 0
    print(f"   优化模式: {'仅旋转 (rotation-only)' if rotation_only else '旋转+平移'}")
    
    # 读取 checkpoint 中保存的 epoch 级别 train/val 误差 (用于可视化信息栏)
    ckpt_train_errors = checkpoint.get('epoch_train_errors', None)
    ckpt_val_errors = checkpoint.get('epoch_val_errors', None)
    ckpt_best_train = checkpoint.get('best_train', None)
    ckpt_best_val = checkpoint.get('best_val', None)
    
    # 优先使用 epoch_train/val_errors，备选 best_train/val
    if ckpt_train_errors is None and ckpt_best_train is not None and ckpt_best_train.get('errors'):
        ckpt_train_errors = ckpt_best_train['errors']
    if ckpt_val_errors is None and ckpt_best_val is not None and ckpt_best_val.get('errors'):
        ckpt_val_errors = ckpt_best_val['errors']
    
    # 回退: 尝试从同目录下的 train.log 解析最后一个 epoch 的 train/val 误差
    if ckpt_train_errors is None or ckpt_val_errors is None:
        ckpt_train_errors, ckpt_val_errors = _parse_train_log_errors(
            args.ckpt_path, epoch, ckpt_train_errors, ckpt_val_errors)
    
    if ckpt_train_errors:
        print(f"   Train errors: Rot {ckpt_train_errors.get('rot_error', -1):.4f}deg")
    if ckpt_val_errors:
        print(f"   Val errors: Rot {ckpt_val_errors.get('rot_error', -1):.4f}deg")
    
    # 创建模型
    print(f"\n2. 初始化模型（图像尺寸: {args.target_width}x{args.target_height}）...")
    model = BEVCalib(
        deformable=args.deformable > 0,
        bev_encoder=args.bev_encoder > 0,
        img_shape=(args.target_height, args.target_width),
        rotation_only=rotation_only,
    ).to(device)
    
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        print(f"   Missing keys: {missing}")
    if unexpected:
        print(f"   Unexpected keys (skipped): {unexpected}")
    model.eval()
    print(f"   ✓ 模型加载完成")
    
    # 加载数据集
    print(f"\n3. 加载数据集...")
    dataset = CustomDataset(
        data_folder=args.dataset_root,
        auto_detect=True
    )
    
    # 根据 use_full_dataset 决定是否使用全量数据
    if args.use_full_dataset:
        eval_dataset = dataset
        print(f"   ✓ 使用全量数据集: {len(eval_dataset)} 个样本（跨数据集泛化测试）")
    else:
        val_size = int(len(dataset) * args.validate_sample_ratio)
        train_size = len(dataset) - val_size
        
        from torch.utils.data import random_split
        import torch as torch_module
        generator = torch_module.Generator().manual_seed(42)
        _, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"   ✓ 验证集: {len(eval_dataset)} 个样本 (ratio={args.validate_sample_ratio})")
    
    collate_fn = make_collate_fn((args.target_width, args.target_height))
    val_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # 创建评估输出目录
    if args.output_dir:
        eval_dir = args.output_dir
    else:
        ckpt_dir = os.path.dirname(args.ckpt_path)
        ckpt_name = os.path.basename(args.ckpt_path).replace('.pth', '')
        eval_dir = os.path.join(ckpt_dir, f"{ckpt_name}_eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"\n4. 开始评估...")
    print(f"   输出目录: {eval_dir}")
    print(f"   扰动参数: {args.angle_range_deg}°, {args.trans_range}m")
    
    # 创建外参结果文件
    extrinsics_file = os.path.join(eval_dir, "extrinsics_and_errors.txt")
    
    # 累积误差统计
    all_errors = {
        'trans_error': [], 'fwd_error': [], 'lat_error': [], 'ht_error': [],
        'rot_error': [], 'roll_error': [], 'pitch_error': [], 'yaw_error': []
    }
    gt_extrinsics_written = False
    
    sample_count = 0
    max_batches = args.max_batches if args.max_batches > 0 else len(val_loader)
    
    with torch.no_grad():
        for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics, distortions) in enumerate(val_loader):
            if batch_index >= max_batches:
                break
            
            gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
            init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(
                gt_T_to_camera_np,
                angle_range_deg=args.angle_range_deg,
                trans_range=args.trans_range,
                rotation_only=rotation_only,
            )
            
            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_np = np.array(pcs)[:, :, :3] if args.xyz_only > 0 else np.array(pcs)
            pcs = torch.from_numpy(pcs_np).float().to(device)
            gt_T_to_camera_torch = torch.from_numpy(gt_T_to_camera_np).float().to(device)
            init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device)
            post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera_torch.shape[0], 1, 1).float().to(device)
            intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)
            
            T_pred, _, _ = model(resize_imgs, pcs, gt_T_to_camera_torch, init_T_to_camera,
                               post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)
            
            imgs_np = np.array(imgs)
            masks_np = np.array(masks)
            T_pred_np = T_pred.detach().cpu().numpy()
            
            # 为每个样本计算误差，按间隔生成可视化
            vis_interval = args.vis_interval
            for i in range(len(imgs_np)):
                sample_idx = sample_count + i
                
                save_vis = (vis_interval > 0 and sample_idx % vis_interval == 0)
                
                if save_vis:
                    print(f"   处理样本 {sample_idx} (含可视化)...", end='', flush=True)
                elif sample_idx % 50 == 0:
                    print(f"   处理样本 {sample_idx}/{max_batches * args.batch_size}...", end='', flush=True)
                
                if save_vis:
                    vis_image = visualize_batch_projection(
                        images=imgs_np[i:i+1],
                        points_batch=pcs_np[i:i+1],
                        init_T_batch=init_T_to_camera_np[i:i+1],
                        gt_T_batch=gt_T_to_camera_np[i:i+1],
                        pred_T_batch=T_pred_np[i:i+1],
                        K_batch=np.array(intrinsics)[i:i+1],
                        masks=masks_np[i:i+1],
                        num_samples=1,
                        max_points=args.vis_points,
                        point_radius=args.vis_point_radius,
                        rotation_only=rotation_only,
                        phase="Eval",
                        epoch=epoch if isinstance(epoch, int) else -1,
                        epoch_train_errors=ckpt_train_errors,
                        epoch_val_errors=ckpt_val_errors,
                    )
                    vis_image_path = os.path.join(eval_dir, f"sample_{sample_idx:04d}_projection.png")
                    cv2.imwrite(vis_image_path, vis_image)
                
                # 计算误差
                errors = compute_pose_errors(T_pred_np[i], gt_T_to_camera_np[i])
                
                # 累积误差
                for key in all_errors:
                    all_errors[key].append(errors[key])
                
                # 保存外参和误差信息到文件
                with open(extrinsics_file, 'a') as f:
                    if not gt_extrinsics_written:
                        eval_mode = "全量数据集泛化测试" if args.use_full_dataset else "验证集评估"
                        f.write(f"Checkpoint: {os.path.basename(args.ckpt_path)}\n")
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Dataset: {args.dataset_root}\n")
                        f.write(f"Mode: {eval_mode}\n")
                        f.write(f"Perturbation: {args.angle_range_deg}deg, {args.trans_range}m\n")
                        f.write(f"="*80 + "\n\n")
                        
                        f.write("Ground Truth Extrinsics (LiDAR → Camera):\n")
                        for row in gt_T_to_camera_np[i]:
                            f.write(f"  {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
                        f.write("\n" + "="*80 + "\n\n")
                        gt_extrinsics_written = True
                    
                    f.write(f"Sample {sample_idx:04d}\n")
                    f.write("-" * 80 + "\n")
                    
                    f.write("\nPredicted Extrinsics (LiDAR → Camera):\n")
                    for row in T_pred_np[i]:
                        f.write(f"  {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
                    
                    if not rotation_only:
                        f.write("\nTranslation Errors (in LiDAR coordinate system):\n")
                        f.write(f"  Total:   {errors['trans_error']:.6f} m\n")
                        f.write(f"  X (Fwd): {errors['fwd_error']:.6f} m\n")
                        f.write(f"  Y (Lat): {errors['lat_error']:.6f} m\n")
                        f.write(f"  Z (Ht):  {errors['ht_error']:.6f} m\n")
                    
                    f.write("\nRotation Errors (axis-angle):\n")
                    f.write(f"  Total:       {errors['rot_error']:.6f} deg\n")
                    f.write(f"  Roll (X):    {errors['roll_error']:.6f} deg\n")
                    f.write(f"  Pitch (Y):   {errors['pitch_error']:.6f} deg\n")
                    f.write(f"  Yaw (Z):     {errors['yaw_error']:.6f} deg\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
                
                if save_vis or sample_idx % 50 == 0:
                    if rotation_only:
                        print(f" ✓ (Rot: {errors['rot_error']:.2f}°)")
                    else:
                        print(f" ✓ (Trans: {errors['trans_error']:.4f}m, Rot: {errors['rot_error']:.2f}°)")
            
            sample_count += len(imgs_np)
    
    # 写入统计摘要（含最大值、最小值、中位数、百分位数）
    with open(extrinsics_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("EVALUATION STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples evaluated: {sample_count}\n\n")

        avg_errors = {k: np.mean(v) for k, v in all_errors.items()}
        std_errors = {k: np.std(v) for k, v in all_errors.items()}
        max_errors = {k: np.max(v) for k, v in all_errors.items()}
        min_errors = {k: np.min(v) for k, v in all_errors.items()}
        med_errors = {k: np.median(v) for k, v in all_errors.items()}
        p90_errors = {k: np.percentile(v, 90) for k, v in all_errors.items()}
        p95_errors = {k: np.percentile(v, 95) for k, v in all_errors.items()}
        p99_errors = {k: np.percentile(v, 99) for k, v in all_errors.items()}

        def write_metric_block(f, label, keys, unit):
            header = f"{'Metric':<14} {'Mean':>10} {'Std':>10} {'Min':>10} {'Median':>10} {'P90':>10} {'P95':>10} {'P99':>10} {'Max':>10}"
            f.write(f"{label} ({unit}):\n")
            f.write(f"  {header}\n")
            f.write(f"  {'-'*len(header)}\n")
            name_map = {
                'trans_error': 'Total', 'fwd_error': 'X (Fwd)', 'lat_error': 'Y (Lat)', 'ht_error': 'Z (Ht)',
                'rot_error': 'Total', 'roll_error': 'Roll (X)', 'pitch_error': 'Pitch (Y)', 'yaw_error': 'Yaw (Z)',
            }
            for k in keys:
                name = name_map.get(k, k)
                f.write(f"  {name:<14} {avg_errors[k]:>10.6f} {std_errors[k]:>10.6f} {min_errors[k]:>10.6f} "
                        f"{med_errors[k]:>10.6f} {p90_errors[k]:>10.6f} {p95_errors[k]:>10.6f} "
                        f"{p99_errors[k]:>10.6f} {max_errors[k]:>10.6f}\n")
            f.write("\n")

        trans_keys = ['trans_error', 'fwd_error', 'lat_error', 'ht_error']
        rot_keys = ['rot_error', 'roll_error', 'pitch_error', 'yaw_error']
        if not rotation_only:
            write_metric_block(f, "Translation Errors", trans_keys, "m")
        write_metric_block(f, "Rotation Errors", rot_keys, "deg")

        f.write("="*80 + "\n")
        f.write("AVERAGE ERRORS ACROSS ALL SAMPLES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples evaluated: {sample_count}\n\n")

        if not rotation_only:
            f.write("Average Translation Errors (in LiDAR coordinate system):\n")
            f.write(f"  Total:   {avg_errors['trans_error']:.6f} ± {std_errors['trans_error']:.6f} m\n")
            f.write(f"  X (Fwd): {avg_errors['fwd_error']:.6f} ± {std_errors['fwd_error']:.6f} m\n")
            f.write(f"  Y (Lat): {avg_errors['lat_error']:.6f} ± {std_errors['lat_error']:.6f} m\n")
            f.write(f"  Z (Ht):  {avg_errors['ht_error']:.6f} ± {std_errors['ht_error']:.6f} m\n")

        f.write("\nAverage Rotation Errors (axis-angle):\n")
        f.write(f"  Total:       {avg_errors['rot_error']:.6f} ± {std_errors['rot_error']:.6f} deg\n")
        f.write(f"  Roll (X):    {avg_errors['roll_error']:.6f} ± {std_errors['roll_error']:.6f} deg\n")
        f.write(f"  Pitch (Y):   {avg_errors['pitch_error']:.6f} ± {std_errors['pitch_error']:.6f} deg\n")
        f.write(f"  Yaw (Z):     {avg_errors['yaw_error']:.6f} ± {std_errors['yaw_error']:.6f} deg\n")
        f.write("\n" + "="*80 + "\n")

    # ========== 生成评估可视化图表 ==========
    print(f"\n5. 生成评估图表...")
    _generate_eval_charts(all_errors, eval_dir, sample_count, args, rotation_only=rotation_only)

    print(f"\n✓ 评估完成！")
    print(f"   - 评估样本数: {sample_count}")
    print(f"   - 输出目录: {eval_dir}")
    print(f"   - 外参文件: {extrinsics_file}")
    print("=" * 80)


def _generate_eval_charts(all_errors, eval_dir, sample_count, args, rotation_only=False):
    """生成评估误差分布可视化图表"""
    charts_dir = os.path.join(eval_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    trans = np.array(all_errors['trans_error'])
    fwd   = np.array(all_errors['fwd_error'])
    lat   = np.array(all_errors['lat_error'])
    ht    = np.array(all_errors['ht_error'])
    rot   = np.array(all_errors['rot_error'])
    roll  = np.array(all_errors['roll_error'])
    pitch = np.array(all_errors['pitch_error'])
    yaw   = np.array(all_errors['yaw_error'])
    indices = np.arange(sample_count)

    ckpt_name = os.path.basename(args.ckpt_path).replace('.pth', '')
    title_suffix = f"({ckpt_name}, {sample_count} samples)"

    def _save(fig, name):
        p = os.path.join(charts_dir, name)
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- Chart 1: Per-sample scatter with mean/P95/max lines ---
    if rotation_only:
        fig, ax2 = plt.subplots(1, 1, figsize=(16, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        ax1.scatter(indices, trans, s=4, alpha=0.4, c='steelblue', label='Per-sample Trans')
        ax1.axhline(np.mean(trans), color='green', ls='--', lw=1.5, label=f'Mean={np.mean(trans):.4f}m')
        ax1.axhline(np.median(trans), color='orange', ls='-.', lw=1.5, label=f'Median={np.median(trans):.4f}m')
        ax1.axhline(np.percentile(trans, 95), color='red', ls=':', lw=1.5, label=f'P95={np.percentile(trans, 95):.4f}m')
        ax1.axhline(np.max(trans), color='darkred', ls='-', lw=1, label=f'Max={np.max(trans):.4f}m')
        ax1.set_ylabel('Translation Error (m)', fontsize=12)
        ax1.set_title(f'Per-Sample Translation Error {title_suffix}', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)

    ax2.scatter(indices, rot, s=4, alpha=0.4, c='coral', label='Per-sample Rot')
    ax2.axhline(np.mean(rot), color='green', ls='--', lw=1.5, label=f'Mean={np.mean(rot):.2f}°')
    ax2.axhline(np.median(rot), color='orange', ls='-.', lw=1.5, label=f'Median={np.median(rot):.2f}°')
    ax2.axhline(np.percentile(rot, 95), color='red', ls=':', lw=1.5, label=f'P95={np.percentile(rot, 95):.2f}°')
    ax2.axhline(np.max(rot), color='darkred', ls='-', lw=1, label=f'Max={np.max(rot):.2f}°')
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Rotation Error (°)', fontsize=12)
    ax2.set_title(f'Per-Sample Rotation Error {title_suffix}', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, 'error_scatter_all_samples.png')
    print(f"   ✓ error_scatter_all_samples.png")

    # --- Chart 2: Sorted error curve (CDF-like) ---
    pct = np.linspace(0, 100, sample_count)
    sorted_rot = np.sort(rot)
    if rotation_only:
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sorted_trans = np.sort(trans)
        ax1.plot(sorted_trans, pct, color='steelblue', lw=2)
        ax1.axvline(np.mean(trans), color='green', ls='--', lw=1.5, label=f'Mean={np.mean(trans):.4f}m')
        ax1.axvline(np.percentile(trans, 95), color='red', ls=':', lw=1.5, label=f'P95={np.percentile(trans, 95):.4f}m')
        ax1.axvline(np.max(trans), color='darkred', ls='-', lw=1, label=f'Max={np.max(trans):.4f}m')
        ax1.set_xlabel('Translation Error (m)', fontsize=12)
        ax1.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax1.set_title('Translation Error CDF', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

    ax2.plot(sorted_rot, pct, color='coral', lw=2)
    ax2.axvline(np.mean(rot), color='green', ls='--', lw=1.5, label=f'Mean={np.mean(rot):.2f}°')
    ax2.axvline(np.percentile(rot, 95), color='red', ls=':', lw=1.5, label=f'P95={np.percentile(rot, 95):.2f}°')
    ax2.axvline(np.max(rot), color='darkred', ls='-', lw=1, label=f'Max={np.max(rot):.2f}°')
    ax2.set_xlabel('Rotation Error (°)', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Rotation Error CDF', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, 'error_cdf.png')
    print(f"   ✓ error_cdf.png")

    # --- Chart 3: Histogram distribution ---
    def _hist(ax, data, title, unit, color):
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(np.mean(data), color='red', ls='--', lw=1.5, label=f'Mean={np.mean(data):.4f}')
        ax.axvline(np.max(data), color='darkred', ls='-', lw=1, label=f'Max={np.max(data):.4f}')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Error ({unit})', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    if rotation_only:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        _hist(axes[0], rot,   'Total Rot',   '°', '#e74c3c')
        _hist(axes[1], roll,  'Roll (X)',    '°', '#1abc9c')
        _hist(axes[2], pitch, 'Pitch (Y)',   '°', '#f39c12')
        _hist(axes[3], yaw,   'Yaw (Z)',    '°', '#8e44ad')
    else:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        _hist(axes[0,0], trans, 'Total Trans', 'm', '#3498db')
        _hist(axes[0,1], fwd,   'Fwd (X)',     'm', '#2ecc71')
        _hist(axes[0,2], lat,   'Lat (Y)',     'm', '#e67e22')
        _hist(axes[0,3], ht,    'Ht (Z)',      'm', '#9b59b6')
        _hist(axes[1,0], rot,   'Total Rot',   '°', '#e74c3c')
        _hist(axes[1,1], roll,  'Roll (X)',    '°', '#1abc9c')
        _hist(axes[1,2], pitch, 'Pitch (Y)',   '°', '#f39c12')
        _hist(axes[1,3], yaw,   'Yaw (Z)',    '°', '#8e44ad')

    fig.suptitle(f'Error Distribution Histograms {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'error_histograms.png')
    print(f"   ✓ error_histograms.png")

    # --- Chart 4: Box plots ---
    if rotation_only:
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        bp1 = ax1.boxplot([trans, fwd, lat, ht], labels=['Total', 'Fwd(X)', 'Lat(Y)', 'Ht(Z)'],
                           patch_artist=True, showmeans=True, meanline=True,
                           meanprops=dict(color='red', ls='--', lw=1.5),
                           medianprops=dict(color='orange', lw=2),
                           flierprops=dict(marker='.', markersize=2, alpha=0.4))
        colors1 = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
        for patch, color in zip(bp1['boxes'], colors1):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax1.set_ylabel('Error (m)', fontsize=12)
        ax1.set_title('Translation Error Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

    bp2 = ax2.boxplot([rot, roll, pitch, yaw], labels=['Total', 'Roll(X)', 'Pitch(Y)', 'Yaw(Z)'],
                       patch_artist=True, showmeans=True, meanline=True,
                       meanprops=dict(color='red', ls='--', lw=1.5),
                       medianprops=dict(color='orange', lw=2),
                       flierprops=dict(marker='.', markersize=2, alpha=0.4))
    colors2 = ['#e74c3c', '#1abc9c', '#f39c12', '#8e44ad']
    for patch, color in zip(bp2['boxes'], colors2):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('Error (°)', fontsize=12)
    ax2.set_title('Rotation Error Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Error Box Plots {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'error_boxplots.png')
    print(f"   ✓ error_boxplots.png")

    # --- Chart 5: Trans vs Rot scatter (correlation) ---
    if not rotation_only:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(trans, rot, s=8, alpha=0.4, c=indices, cmap='viridis')
        ax.axhline(np.mean(rot), color='coral', ls='--', lw=1, alpha=0.6, label=f'Rot Mean={np.mean(rot):.2f}°')
        ax.axvline(np.mean(trans), color='steelblue', ls='--', lw=1, alpha=0.6, label=f'Trans Mean={np.mean(trans):.4f}m')
        ax.set_xlabel('Translation Error (m)', fontsize=13)
        ax.set_ylabel('Rotation Error (°)', fontsize=13)
        ax.set_title(f'Translation vs Rotation Error {title_suffix}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Sample Index', fontsize=11)
        _save(fig, 'error_trans_vs_rot.png')
        print(f"   ✓ error_trans_vs_rot.png")

    # --- Chart 6: Per-sample component stacked area ---
    sorted_idx_r = np.argsort(rot)[::-1]
    if rotation_only:
        fig, ax2 = plt.subplots(1, 1, figsize=(16, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        sorted_idx = np.argsort(trans)[::-1]
        ax1.fill_between(range(len(sorted_idx)), ht[sorted_idx], alpha=0.6, color='#9b59b6', label='Ht(Z)')
        ax1.fill_between(range(len(sorted_idx)), lat[sorted_idx], alpha=0.6, color='#e67e22', label='Lat(Y)')
        ax1.fill_between(range(len(sorted_idx)), fwd[sorted_idx], alpha=0.6, color='#2ecc71', label='Fwd(X)')
        ax1.plot(range(len(sorted_idx)), trans[sorted_idx], color='#2c3e50', lw=1.2, label='Total Trans')
        ax1.axhline(np.mean(trans), color='red', ls='--', lw=1, label=f'Mean={np.mean(trans):.4f}m')
        ax1.set_ylabel('Error (m)', fontsize=12)
        ax1.set_title(f'Translation Components (sorted by total error, descending) {title_suffix}', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim(0, len(sorted_idx)-1)

    ax2.fill_between(range(len(sorted_idx_r)), pitch[sorted_idx_r], alpha=0.6, color='#f39c12', label='Pitch(Y)')
    ax2.fill_between(range(len(sorted_idx_r)), yaw[sorted_idx_r], alpha=0.6, color='#8e44ad', label='Yaw(Z)')
    ax2.fill_between(range(len(sorted_idx_r)), roll[sorted_idx_r], alpha=0.6, color='#1abc9c', label='Roll(X)')
    ax2.plot(range(len(sorted_idx_r)), rot[sorted_idx_r], color='#2c3e50', lw=1.2, label='Total Rot')
    ax2.axhline(np.mean(rot), color='red', ls='--', lw=1, label=f'Mean={np.mean(rot):.2f}°')
    ax2.set_xlabel('Samples (sorted by total error)', fontsize=12)
    ax2.set_ylabel('Error (°)', fontsize=12)
    ax2.set_title(f'Rotation Components (sorted by total error, descending)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, len(sorted_idx_r)-1)

    plt.tight_layout()
    _save(fig, 'error_sorted_components.png')
    print(f"   ✓ error_sorted_components.png")

    print(f"   ✓ 所有图表已保存至: {charts_dir}/")

def main():
    parser = argparse.ArgumentParser(description="评估已保存的 checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint 文件路径")
    parser.add_argument("--dataset_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="评估结果输出目录（默认: checkpoint目录下的 ckpt_xxx_eval/）。"
                            "跨数据集泛化测试时建议指定独立输出目录")
    parser.add_argument("--angle_range_deg", type=float, default=20.0, help="扰动角度范围")
    parser.add_argument("--trans_range", type=float, default=1.5, help="扰动平移范围")
    parser.add_argument("--target_width", type=int, default=640, help="目标图像宽度")
    parser.add_argument("--target_height", type=int, default=360, help="目标图像高度")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=5, help="最多评估的batch数（0表示全部）")
    parser.add_argument("--validate_sample_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--use_full_dataset", action='store_true', default=False,
                       help="使用全量数据集评估（跨数据集泛化测试时使用，忽略 validate_sample_ratio）")
    parser.add_argument("--deformable", type=int, default=0, help="是否使用 deformable attention")
    parser.add_argument("--bev_encoder", type=int, default=1, help="是否使用 BEV encoder")
    parser.add_argument("--xyz_only", type=int, default=1, help="是否只使用 XYZ 坐标")
    parser.add_argument("--vis_points", type=int, default=80000, help="可视化最大点数")
    parser.add_argument("--vis_point_radius", type=int, default=1, help="可视化点半径")
    parser.add_argument("--vis_interval", type=int, default=1,
                       help="每隔N个样本保存一张可视化图（默认1=每帧都保存，"
                            "全量评估时建议设为50-100以减少IO）")
    parser.add_argument("--use_inverse_transform", type=int, default=0, 
                       help="是否对变换矩阵取逆后再使用 (1=是, 0=否) - 用于修复投影高度偏移")
    parser.add_argument("--rotation_only", type=int, default=0,
                       help="仅优化旋转 (1=仅旋转, 0=旋转+平移同时优化). "
                            "设为-1时自动从checkpoint中读取")
    
    args = parser.parse_args()
    evaluate_checkpoint(args)

if __name__ == "__main__":
    main()
