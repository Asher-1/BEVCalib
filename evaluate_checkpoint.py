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
import subprocess
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

if '--use_drcv' in sys.argv:
    os.environ['USE_DRCV_BACKEND'] = '1'

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
        """缩放图像并同步调整内参矩阵"""
        img = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        
        # 缩放图像
        resized = cv2.resize(img, size)
        
        scale_x = size[0] / w
        scale_y = size[1] / h
        
        new_intrinsic = np.array([
            [intrinsic[0, 0] * scale_x, 0, intrinsic[0, 2] * scale_x],
            [0, intrinsic[1, 1] * scale_y, intrinsic[1, 2] * scale_y],
            [0, 0, 1]
        ])
        
        return resized, new_intrinsic
    
    def collate_fn(batch):
        processed = [crop_and_resize(item[0], target_size, item[3]) for item in batch]
        imgs = [p[0] for p in processed]
        intrinsics = [p[1] for p in processed]
        gt_T_to_camera = [item[2] for item in batch]
        
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

        return imgs, pcs, masks, gt_T_to_camera, intrinsics
    
    return collate_fn


def _auto_permute_spconv_weights(state_dict, model):
    """Auto-detect and permute sparse conv weights between spconv v2 and drcv/v1 layouts.
    
    spconv v2: (out_channels, k, k, k, in_channels)  — dim0 is largest (out)
    drcv/v1:   (k, k, k, in_channels, out_channels)  — dim-1 is largest (out)
    
    Detection: find an asymmetric 5D weight (in != out) to determine if ckpt
    uses a different layout than the model. Then permute ALL 5D sparse conv weights.
    """
    model_sd = model.state_dict()
    
    needs_v2_to_v1 = False
    for key in state_dict:
        if key not in model_sd:
            continue
        cs, ms = state_dict[key].shape, model_sd[key].shape
        if len(cs) != 5 or len(ms) != 5:
            continue
        if cs == ms:
            continue
        if sorted(cs) == sorted(ms):
            needs_v2_to_v1 = True
            break
    
    if not needs_v2_to_v1:
        return state_dict
    
    permuted = 0
    for key in list(state_dict.keys()):
        if key not in model_sd:
            continue
        if len(state_dict[key].shape) != 5 or len(model_sd[key].shape) != 5:
            continue
        w = state_dict[key]
        target = model_sd[key].shape
        if w.shape == target:
            pass
        else:
            w_perm = w.permute(1, 2, 3, 4, 0).contiguous()
            if w_perm.shape == target:
                state_dict[key] = w_perm
                permuted += 1
            else:
                w_perm2 = w.permute(4, 0, 1, 2, 3).contiguous()
                if w_perm2.shape == target:
                    state_dict[key] = w_perm2
                    permuted += 1
    
    if permuted > 0:
        print(f"   Auto-permuted {permuted} sparse conv weights (v2↔v1 layout)")
    return state_dict


def _resolve_perturbation_from_ckpt(args, checkpoint):
    """Auto-resolve perturbation params from checkpoint if not set via CLI.
    
    Priority: CLI explicit > checkpoint eval_noise > checkpoint train_noise > fallback defaults.
    Also restores perturb_distribution / per_axis_prob / per_axis_weights from checkpoint args.
    """
    eval_noise = checkpoint.get('eval_noise') or {}
    train_noise = checkpoint.get('train_noise') or {}
    ckpt_args = checkpoint.get('args') or {}
    
    def _pick(key, eval_d, train_d):
        v = eval_d.get(key)
        if v is not None:
            return v
        return train_d.get(key)
    
    if args.angle_range_deg is None:
        ckpt_angle = _pick('angle_range_deg', eval_noise, train_noise)
        if ckpt_angle is not None:
            args.angle_range_deg = float(ckpt_angle)
            print(f"   angle_range_deg={args.angle_range_deg} (从checkpoint恢复)")
        else:
            args.angle_range_deg = 20.0
            print(f"   angle_range_deg={args.angle_range_deg} (checkpoint无记录, 使用默认值)")
    
    if args.trans_range is None:
        ckpt_trans = _pick('trans_range', eval_noise, train_noise)
        if ckpt_trans is not None:
            args.trans_range = float(ckpt_trans)
            print(f"   trans_range={args.trans_range} (从checkpoint恢复)")
        else:
            args.trans_range = 1.5
            print(f"   trans_range={args.trans_range} (checkpoint无记录, 使用默认值)")
    
    if not hasattr(args, 'perturb_distribution') or args.perturb_distribution is None:
        args.perturb_distribution = ckpt_args.get('perturb_distribution', 'uniform')
    if not hasattr(args, 'per_axis_prob') or args.per_axis_prob is None:
        args.per_axis_prob = ckpt_args.get('per_axis_prob', 0.0)
    if not hasattr(args, 'per_axis_weights') or args.per_axis_weights is None:
        args.per_axis_weights = ckpt_args.get('per_axis_weights', '')
    
    if args.perturb_distribution != 'uniform' or args.per_axis_prob > 0:
        print(f"   perturb_distribution={args.perturb_distribution}, "
              f"per_axis_prob={args.per_axis_prob} (从checkpoint恢复)")


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
    
    # 从 checkpoint 自动恢复扰动参数（如果 CLI 未显式指定）
    _resolve_perturbation_from_ckpt(args, checkpoint)
    
    # 创建模型
    state_dict = checkpoint['model_state_dict']
    if args.use_mlp_head >= 0:
        use_mlp_head = args.use_mlp_head > 0
    else:
        use_mlp_head = 'rotation_pred.0.weight' in state_dict
    print(f"\n2. 初始化模型（图像尺寸: {args.target_width}x{args.target_height}）...")
    print(f"   回归头类型: {'MLP (V6)' if use_mlp_head else 'Linear (V5及更早)'}"
          f"{' (auto-detected)' if args.use_mlp_head < 0 else ''}")
    _use_fd = getattr(args, 'use_foundation_depth', 0) > 0
    _fd_mode = getattr(args, 'fd_mode', 'replace') if _use_fd else 'replace'
    model = BEVCalib(
        deformable=args.deformable > 0,
        bev_encoder=args.bev_encoder > 0,
        img_shape=(args.target_height, args.target_width),
        rotation_only=rotation_only,
        use_mlp_head=use_mlp_head,
        bev_pool_factor=args.bev_pool_factor,
        use_foundation_depth=_use_fd,
        depth_model_type=getattr(args, 'depth_model_type', 'midas_small'),
        fd_mode=_fd_mode,
    ).to(device)
    if _use_fd:
        print(f"   Foundation Depth: model={getattr(args, 'depth_model_type', 'midas_small')}, mode={_fd_mode}")
    
    state_dict = _auto_permute_spconv_weights(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
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
        from torch.utils.data import random_split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(114514)
        _, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"   ✓ 验证集: {len(eval_dataset)} 个样本 (80/20划分, seed=114514, 与训练一致)")
    
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
    
    eval_seed = getattr(args, 'eval_seed', 42)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    
    print(f"\n4. 开始评估...")
    print(f"   输出目录: {eval_dir}")
    print(f"   扰动参数: {args.angle_range_deg}°, {args.trans_range}m")
    print(f"   评估seed: {eval_seed} (固定perturbation保证可复现)")
    
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
        for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
            if batch_index >= max_batches:
                break
            
            gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
            _paw = None
            if getattr(args, 'per_axis_weights', '') and args.per_axis_weights:
                _paw = tuple(float(x) for x in args.per_axis_weights.split(','))
            init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(
                gt_T_to_camera_np,
                angle_range_deg=args.angle_range_deg,
                trans_range=args.trans_range,
                rotation_only=rotation_only,
                distribution=getattr(args, 'perturb_distribution', 'uniform'),
                per_axis_prob=getattr(args, 'per_axis_prob', 0.0),
                per_axis_weights=_paw,
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
                    
                    f.write("\nRotation Errors (axis-angle, LiDAR frame: X=Fwd, Y=Left, Z=Up):\n")
                    f.write(f"  Total:            {errors['rot_error']:.6f} deg\n")
                    f.write(f"  Roll  (LiDAR X):  {errors['roll_error']:.6f} deg\n")
                    f.write(f"  Pitch (LiDAR Y):  {errors['pitch_error']:.6f} deg\n")
                    f.write(f"  Yaw   (LiDAR Z):  {errors['yaw_error']:.6f} deg\n")
                    
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
                'rot_error': 'Total', 'roll_error': 'Roll (LiDAR-X)', 'pitch_error': 'Pitch (LiDAR-Y)', 'yaw_error': 'Yaw (LiDAR-Z)',
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

        f.write("\nAverage Rotation Errors (axis-angle, LiDAR frame: X=Fwd, Y=Left, Z=Up):\n")
        f.write(f"  Total:            {avg_errors['rot_error']:.6f} ± {std_errors['rot_error']:.6f} deg\n")
        f.write(f"  Roll  (LiDAR X):  {avg_errors['roll_error']:.6f} ± {std_errors['roll_error']:.6f} deg\n")
        f.write(f"  Pitch (LiDAR Y):  {avg_errors['pitch_error']:.6f} ± {std_errors['pitch_error']:.6f} deg\n")
        f.write(f"  Yaw   (LiDAR Z):  {avg_errors['yaw_error']:.6f} ± {std_errors['yaw_error']:.6f} deg\n")
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
        _hist(axes[0], rot,   'Total Rot',         '°', '#e74c3c')
        _hist(axes[1], roll,  'Roll (LiDAR-X)',   '°', '#1abc9c')
        _hist(axes[2], pitch, 'Pitch (LiDAR-Y)',  '°', '#f39c12')
        _hist(axes[3], yaw,   'Yaw (LiDAR-Z)',   '°', '#8e44ad')
    else:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        _hist(axes[0,0], trans, 'Total Trans', 'm', '#3498db')
        _hist(axes[0,1], fwd,   'Fwd (X)',     'm', '#2ecc71')
        _hist(axes[0,2], lat,   'Lat (Y)',     'm', '#e67e22')
        _hist(axes[0,3], ht,    'Ht (Z)',      'm', '#9b59b6')
        _hist(axes[1,0], rot,   'Total Rot',         '°', '#e74c3c')
        _hist(axes[1,1], roll,  'Roll (LiDAR-X)',   '°', '#1abc9c')
        _hist(axes[1,2], pitch, 'Pitch (LiDAR-Y)',  '°', '#f39c12')
        _hist(axes[1,3], yaw,   'Yaw (LiDAR-Z)',   '°', '#8e44ad')

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

    bp2 = ax2.boxplot([rot, roll, pitch, yaw], labels=['Total', 'Roll(LiDAR-X)', 'Pitch(LiDAR-Y)', 'Yaw(LiDAR-Z)'],
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

    ax2.fill_between(range(len(sorted_idx_r)), pitch[sorted_idx_r], alpha=0.6, color='#f39c12', label='Pitch(LiDAR-Y)')
    ax2.fill_between(range(len(sorted_idx_r)), yaw[sorted_idx_r], alpha=0.6, color='#8e44ad', label='Yaw(LiDAR-Z)')
    ax2.fill_between(range(len(sorted_idx_r)), roll[sorted_idx_r], alpha=0.6, color='#1abc9c', label='Roll(LiDAR-X)')
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

def _load_model_from_ckpt(ckpt_path, device, args, rotation_only):
    """Load a BEVCalib model from checkpoint and return (model, epoch, train_errors, val_errors)."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')

    state_dict = checkpoint['model_state_dict']
    if args.use_mlp_head >= 0:
        use_mlp_head = args.use_mlp_head > 0
    else:
        use_mlp_head = 'rotation_pred.0.weight' in state_dict
    _use_fd = getattr(args, 'use_foundation_depth', 0) > 0
    _fd_mode = getattr(args, 'fd_mode', 'replace') if _use_fd else 'replace'
    model = BEVCalib(
        deformable=args.deformable > 0,
        bev_encoder=args.bev_encoder > 0,
        img_shape=(args.target_height, args.target_width),
        rotation_only=rotation_only,
        use_mlp_head=use_mlp_head,
        bev_pool_factor=args.bev_pool_factor,
        use_foundation_depth=_use_fd,
        depth_model_type=getattr(args, 'depth_model_type', 'midas_small'),
        fd_mode=_fd_mode,
    ).to(device)
    state_dict = _auto_permute_spconv_weights(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    train_err = checkpoint.get('epoch_train_errors', None)
    val_err = checkpoint.get('epoch_val_errors', None)
    best_t = checkpoint.get('best_train', None)
    best_v = checkpoint.get('best_val', None)
    if train_err is None and best_t is not None and best_t.get('errors'):
        train_err = best_t['errors']
    if val_err is None and best_v is not None and best_v.get('errors'):
        val_err = best_v['errors']
    if train_err is None or val_err is None:
        train_err, val_err = _parse_train_log_errors(ckpt_path, epoch, train_err, val_err)

    return model, epoch, train_err, val_err


def compare_checkpoints(args):
    """Compare two checkpoints on the same test data with identical perturbations."""

    label_a = args.label_a
    label_b = args.label_b
    print("=" * 80)
    print(f"模型对比: [{label_a}] vs [{label_b}]")
    print(f"  A: {args.ckpt_path_a}")
    print(f"  B: {args.ckpt_path_b}")
    print(f"  数据集: {args.dataset_root}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect rotation_only from checkpoint A (same logic as evaluate_checkpoint)
    if args.rotation_only == -1:
        ckpt_a = torch.load(args.ckpt_path_a, map_location='cpu')
        if 'rotation_only' in ckpt_a:
            rotation_only = ckpt_a['rotation_only']
        elif 'optimize_translation' in ckpt_a:
            rotation_only = not ckpt_a['optimize_translation']
        else:
            rotation_only = False
        print(f"   rotation_only: {rotation_only} (从checkpoint A读取)")
        _resolve_perturbation_from_ckpt(args, ckpt_a)
        del ckpt_a
    else:
        rotation_only = args.rotation_only > 0
        if args.angle_range_deg is None:
            args.angle_range_deg = 20.0
        if args.trans_range is None:
            args.trans_range = 1.5

    print(f"\n1. 加载模型...")
    model_a, epoch_a, train_err_a, val_err_a = _load_model_from_ckpt(
        args.ckpt_path_a, device, args, rotation_only)
    print(f"   [{label_a}] epoch={epoch_a}")
    model_b, epoch_b, train_err_b, val_err_b = _load_model_from_ckpt(
        args.ckpt_path_b, device, args, rotation_only)
    print(f"   [{label_b}] epoch={epoch_b}")

    print(f"\n2. 加载数据集...")
    dataset = CustomDataset(data_folder=args.dataset_root, auto_detect=True)
    if args.use_full_dataset:
        eval_dataset = dataset
        print(f"   全量: {len(eval_dataset)} 样本")
    else:
        from torch.utils.data import random_split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(114514)
        _, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"   验证集: {len(eval_dataset)} 样本 (80/20划分, seed=114514, 与训练一致)")

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    val_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                            num_workers=4, collate_fn=collate_fn, shuffle=False)

    eval_dir = args.output_dir or f"logs/comparison_{label_a}_vs_{label_b}"
    os.makedirs(eval_dir, exist_ok=True)

    eval_seed = getattr(args, 'eval_seed', 42)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    
    print(f"\n3. 开始对比评估...")
    print(f"   输出目录: {eval_dir}")
    print(f"   扰动: {args.angle_range_deg}°")
    print(f"   评估seed: {eval_seed} (固定perturbation保证可复现)")

    _empty_errors = lambda: {
        'trans_error': [], 'fwd_error': [], 'lat_error': [], 'ht_error': [],
        'rot_error': [], 'roll_error': [], 'pitch_error': [], 'yaw_error': []
    }
    errors_a = _empty_errors()
    errors_b = _empty_errors()

    sample_count = 0
    max_batches = args.max_batches if args.max_batches > 0 else len(val_loader)

    with torch.no_grad():
        for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
            if batch_index >= max_batches:
                break

            gt_T_np = np.array(gt_T_to_camera).astype(np.float32)

            _paw = None
            if getattr(args, 'per_axis_weights', '') and args.per_axis_weights:
                _paw = tuple(float(x) for x in args.per_axis_weights.split(','))
            init_T_np, _, _ = generate_single_perturbation_from_T(
                gt_T_np, angle_range_deg=args.angle_range_deg,
                trans_range=args.trans_range, rotation_only=rotation_only,
                distribution=getattr(args, 'perturb_distribution', 'uniform'),
                per_axis_prob=getattr(args, 'per_axis_prob', 0.0),
                per_axis_weights=_paw)

            imgs_arr = np.array(imgs)
            pcs_np = np.array(pcs)[:, :, :3] if args.xyz_only > 0 else np.array(pcs)
            masks_np = np.array(masks)
            K_np = np.array(intrinsics)

            resize_imgs = torch.from_numpy(imgs_arr).permute(0, 3, 1, 2).float().to(device)
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
            init_T_t = torch.from_numpy(init_T_np).float().to(device)
            post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
            K_t = torch.from_numpy(K_np).float().to(device)

            pred_a, _, _ = model_a(resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t,
                                   masks=masks, out_init_loss=False)
            pred_b, _, _ = model_b(resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t,
                                   masks=masks, out_init_loss=False)

            pred_a_np = pred_a.detach().cpu().numpy()
            pred_b_np = pred_b.detach().cpu().numpy()

            vis_interval = args.vis_interval
            for i in range(len(imgs_arr)):
                sample_idx = sample_count + i
                e_a = compute_pose_errors(pred_a_np[i], gt_T_np[i])
                e_b = compute_pose_errors(pred_b_np[i], gt_T_np[i])
                for k in errors_a:
                    errors_a[k].append(e_a[k])
                    errors_b[k].append(e_b[k])

                save_vis = (vis_interval > 0 and sample_idx % vis_interval == 0)
                if save_vis:
                    vis_a = visualize_batch_projection(
                        images=imgs_arr[i:i+1], points_batch=pcs_np[i:i+1],
                        init_T_batch=init_T_np[i:i+1], gt_T_batch=gt_T_np[i:i+1],
                        pred_T_batch=pred_a_np[i:i+1], K_batch=K_np[i:i+1],
                        masks=masks_np[i:i+1], num_samples=1,
                        max_points=args.vis_points, point_radius=args.vis_point_radius,
                        rotation_only=rotation_only, phase="Eval", epoch=epoch_a if isinstance(epoch_a, int) else -1,
                        epoch_train_errors=train_err_a, epoch_val_errors=val_err_a)
                    vis_b = visualize_batch_projection(
                        images=imgs_arr[i:i+1], points_batch=pcs_np[i:i+1],
                        init_T_batch=init_T_np[i:i+1], gt_T_batch=gt_T_np[i:i+1],
                        pred_T_batch=pred_b_np[i:i+1], K_batch=K_np[i:i+1],
                        masks=masks_np[i:i+1], num_samples=1,
                        max_points=args.vis_points, point_radius=args.vis_point_radius,
                        rotation_only=rotation_only, phase="Eval", epoch=epoch_b if isinstance(epoch_b, int) else -1,
                        epoch_train_errors=train_err_b, epoch_val_errors=val_err_b)

                    ha, wa = vis_a.shape[:2]
                    hb, wb = vis_b.shape[:2]
                    max_w = max(wa, wb)
                    if wa < max_w:
                        vis_a = np.hstack([vis_a, np.zeros((ha, max_w - wa, 3), dtype=np.uint8)])
                    if wb < max_w:
                        vis_b = np.hstack([vis_b, np.zeros((hb, max_w - wb, 3), dtype=np.uint8)])

                    label_bar_h = 36
                    def _make_label_bar(text, w, bg_color):
                        bar = np.full((label_bar_h, w, 3), bg_color, dtype=np.uint8)
                        cv2.putText(bar, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                        return bar

                    bar_a = _make_label_bar(
                        f"[{label_a}] Rot: {e_a['rot_error']:.2f} deg  (R:{e_a['roll_error']:.2f} P:{e_a['pitch_error']:.2f} Y:{e_a['yaw_error']:.2f})",
                        max_w, (180, 60, 0))
                    bar_b = _make_label_bar(
                        f"[{label_b}] Rot: {e_b['rot_error']:.2f} deg  (R:{e_b['roll_error']:.2f} P:{e_b['pitch_error']:.2f} Y:{e_b['yaw_error']:.2f})",
                        max_w, (0, 100, 180))

                    winner = label_a if e_a['rot_error'] < e_b['rot_error'] else label_b
                    diff = abs(e_a['rot_error'] - e_b['rot_error'])
                    bar_summary = _make_label_bar(
                        f"Sample {sample_idx:04d} | Winner: [{winner}] by {diff:.2f} deg | Perturbation: {args.angle_range_deg} deg",
                        max_w, (40, 40, 40))

                    combined = np.vstack([bar_summary, bar_a, vis_a, bar_b, vis_b])
                    cv2.imwrite(os.path.join(eval_dir, f"compare_{sample_idx:04d}.png"), combined)
                    print(f"   样本 {sample_idx}: [{label_a}] {e_a['rot_error']:.2f}° vs [{label_b}] {e_b['rot_error']:.2f}° -> {winner}")

                elif sample_idx % 50 == 0:
                    print(f"   样本 {sample_idx}/{max_batches * args.batch_size}...")

            sample_count += len(imgs_arr)

    # Write comparison summary text
    summary_path = os.path.join(eval_dir, "comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write(f"MODEL COMPARISON: [{label_a}] vs [{label_b}]\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Model A [{label_a}]: {args.ckpt_path_a}\n")
        f.write(f"Model B [{label_b}]: {args.ckpt_path_b}\n")
        f.write(f"Dataset: {args.dataset_root}\n")
        f.write(f"Perturbation: {args.angle_range_deg} deg\n")
        f.write(f"Samples: {sample_count}\n\n")

        rot_a = np.array(errors_a['rot_error'])
        rot_b = np.array(errors_b['rot_error'])
        wins_a = int(np.sum(rot_a < rot_b))
        wins_b = int(np.sum(rot_b < rot_a))
        ties = sample_count - wins_a - wins_b

        def _stats(arr):
            return (np.mean(arr), np.median(arr), np.std(arr),
                    np.percentile(arr, 90), np.percentile(arr, 95), np.max(arr))

        header = f"{'Metric':<14} {'Mean':>8} {'Median':>8} {'Std':>8} {'P90':>8} {'P95':>8} {'Max':>8}"
        for lbl, errs in [(label_a, errors_a), (label_b, errors_b)]:
            f.write(f"\n[{lbl}] Rotation Errors (deg):\n")
            f.write(f"  {header}\n")
            f.write(f"  {'-' * len(header)}\n")
            name_map = {'rot_error': 'Total', 'roll_error': 'Roll(LiDAR-X)',
                        'pitch_error': 'Pitch(LiDAR-Y)', 'yaw_error': 'Yaw(LiDAR-Z)'}
            for k in ['rot_error', 'roll_error', 'pitch_error', 'yaw_error']:
                s = _stats(np.array(errs[k]))
                f.write(f"  {name_map[k]:<14} {s[0]:>8.4f} {s[1]:>8.4f} {s[2]:>8.4f} "
                        f"{s[3]:>8.4f} {s[4]:>8.4f} {s[5]:>8.4f}\n")

        f.write(f"\nWin/Loss/Tie (by total rotation error):\n")
        f.write(f"  [{label_a}] wins: {wins_a}  |  [{label_b}] wins: {wins_b}  |  ties: {ties}\n")
        f.write(f"  [{label_a}] win rate: {100*wins_a/max(sample_count,1):.1f}%\n")
        f.write(f"  [{label_b}] win rate: {100*wins_b/max(sample_count,1):.1f}%\n")

    print(f"\n4. 生成对比图表...")
    _generate_comparison_charts(errors_a, errors_b, label_a, label_b,
                                eval_dir, sample_count, args, rotation_only)

    print(f"\n{'='*80}")
    print(f"对比完成! {sample_count} 样本")
    print(f"  输出: {eval_dir}")
    print(f"  摘要: {summary_path}")
    print(f"{'='*80}")


def _generate_comparison_charts(errors_a, errors_b, label_a, label_b,
                                eval_dir, sample_count, args, rotation_only):
    """Generate comparison charts overlaying two models' error distributions."""
    charts_dir = os.path.join(eval_dir, "comparison_charts")
    os.makedirs(charts_dir, exist_ok=True)

    def _save(fig, name):
        fig.savefig(os.path.join(charts_dir, name), dpi=150, bbox_inches='tight')
        plt.close(fig)

    rot_a = np.array(errors_a['rot_error'])
    rot_b = np.array(errors_b['rot_error'])
    roll_a, roll_b = np.array(errors_a['roll_error']), np.array(errors_b['roll_error'])
    pitch_a, pitch_b = np.array(errors_a['pitch_error']), np.array(errors_b['pitch_error'])
    yaw_a, yaw_b = np.array(errors_a['yaw_error']), np.array(errors_b['yaw_error'])
    indices = np.arange(sample_count)

    COLOR_A = '#2196F3'
    COLOR_B = '#FF5722'

    # --- Chart 1: CDF overlay ---
    pct = np.linspace(0, 100, sample_count)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, da, db, title in zip(
            axes,
            [rot_a, roll_a, pitch_a, yaw_a],
            [rot_b, roll_b, pitch_b, yaw_b],
            ['Total Rotation', 'Roll (LiDAR-X)', 'Pitch (LiDAR-Y)', 'Yaw (LiDAR-Z)']):
        ax.plot(np.sort(da), pct, color=COLOR_A, lw=2, label=f'{label_a} (Mean={np.mean(da):.2f})')
        ax.plot(np.sort(db), pct, color=COLOR_B, lw=2, label=f'{label_b} (Mean={np.mean(db):.2f})')
        ax.set_xlabel('Error (deg)', fontsize=11)
        ax.set_ylabel('Cumulative %', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Rotation Error CDF: [{label_a}] vs [{label_b}] ({sample_count} samples)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'cdf_overlay.png')
    print(f"   cdf_overlay.png")

    # --- Chart 2: Paired scatter ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(rot_a, rot_b, s=10, alpha=0.5, c='#333', edgecolors='none')
    lim = max(rot_a.max(), rot_b.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='Equal performance')
    ax.set_xlabel(f'[{label_a}] Rotation Error (deg)', fontsize=12)
    ax.set_ylabel(f'[{label_b}] Rotation Error (deg)', fontsize=12)
    wins_a = int(np.sum(rot_a < rot_b))
    wins_b = int(np.sum(rot_b < rot_a))
    ax.set_title(f'Paired Sample Comparison\n[{label_a}] wins {wins_a}, [{label_b}] wins {wins_b}',
                 fontsize=13, fontweight='bold')
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.06, color=COLOR_A,
                    label=f'{label_a} better (below diagonal)')
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.06, color=COLOR_B,
                    label=f'{label_b} better (above diagonal)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    _save(fig, 'paired_scatter.png')
    print(f"   paired_scatter.png")

    # --- Chart 3: Side-by-side box plots ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, da, db, title in zip(
            axes,
            [rot_a, roll_a, pitch_a, yaw_a],
            [rot_b, roll_b, pitch_b, yaw_b],
            ['Total Rot', 'Roll (LiDAR-X)', 'Pitch (LiDAR-Y)', 'Yaw (LiDAR-Z)']):
        bp = ax.boxplot([da, db], tick_labels=[label_a, label_b], patch_artist=True,
                        showmeans=True, meanline=True,
                        meanprops=dict(color='red', ls='--', lw=1.5),
                        medianprops=dict(color='orange', lw=2),
                        flierprops=dict(marker='.', markersize=3, alpha=0.4))
        bp['boxes'][0].set_facecolor(COLOR_A)
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor(COLOR_B)
        bp['boxes'][1].set_alpha(0.5)
        ax.set_ylabel('Error (deg)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'Error Box Plots: [{label_a}] vs [{label_b}]', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'boxplot_comparison.png')
    print(f"   boxplot_comparison.png")

    # --- Chart 4: Win/loss bar ---
    ties = sample_count - wins_a - wins_b
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([label_a, label_b, 'Tie'], [wins_a, wins_b, ties],
                  color=[COLOR_A, COLOR_B, '#888'], edgecolor='white', linewidth=1.5)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + sample_count*0.01, f'{int(h)}\n({100*h/max(sample_count,1):.1f}%)',
                ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title(f'Win/Loss by Rotation Error ({sample_count} samples)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    _save(fig, 'win_loss_bar.png')
    print(f"   win_loss_bar.png")

    # --- Chart 5: Per-sample error difference ---
    diff = rot_a - rot_b
    sorted_idx = np.argsort(diff)
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = [COLOR_A if d < 0 else COLOR_B for d in diff[sorted_idx]]
    ax.bar(range(sample_count), diff[sorted_idx], color=colors, width=1.0, edgecolor='none')
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(np.mean(diff), color='green', ls='--', lw=1.5,
               label=f'Mean diff: {np.mean(diff):.3f} deg (negative = {label_a} better)')
    ax.set_xlabel('Samples (sorted by error difference)', fontsize=12)
    ax.set_ylabel(f'Error diff: [{label_a}] - [{label_b}] (deg)', fontsize=12)
    ax.set_title(f'Per-Sample Error Difference', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_xlim(0, sample_count - 1)
    _save(fig, 'error_difference.png')
    print(f"   error_difference.png")

    # --- Chart 6: Summary table ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    stats = {}
    for lbl, errs in [(label_a, errors_a), (label_b, errors_b)]:
        r = np.array(errs['rot_error'])
        stats[lbl] = [f'{np.mean(r):.3f}', f'{np.median(r):.3f}', f'{np.std(r):.3f}',
                       f'{np.percentile(r, 90):.3f}', f'{np.percentile(r, 95):.3f}', f'{np.max(r):.3f}']
    col_labels = ['Mean', 'Median', 'Std', 'P90', 'P95', 'Max']
    row_labels = [f'[{label_a}]', f'[{label_b}]']
    cell_text = [stats[label_a], stats[label_b]]
    table = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(fontweight='bold')
        elif c == -1:
            cell.set_facecolor('#F0F0F0')
            cell.set_text_props(fontweight='bold')
    ax.set_title(f'Rotation Error Summary (deg) - {sample_count} samples',
                 fontsize=14, fontweight='bold', pad=20)
    _save(fig, 'summary_table.png')
    print(f"   summary_table.png")
    print(f"   All charts saved to: {charts_dir}/")


MULTI_EVAL_MODELS = [
    {
        "label": "10deg-v2-z5",
        "dir_name": "model_small_10deg_v2_z5",
        "ckpt": "ckpt_240.pth",
        "bev_zbound_step": "4.0",
        "rotation_only": 0,
    },
    {
        "label": "10deg-v3-z10",
        "dir_name": "model_small_10deg_v3_z10",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "2.0",
        "rotation_only": 0,
    },
    {
        "label": "10deg-v4-z10",
        "dir_name": "model_small_10deg_v4_z10_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "2.0",
        "rotation_only": 1,
    },
    {
        "label": "10deg-v4-z1",
        "dir_name": "model_small_10deg_v4_z1_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "20.0",
        "rotation_only": 1,
    },
    {
        "label": "10deg-v4-z5",
        "dir_name": "model_small_10deg_v4_z5_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "4.0",
        "rotation_only": 1,
    },
    {
        "label": "5deg-v4-z1",
        "dir_name": "model_small_5deg_v4_z1_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "20.0",
        "rotation_only": 1,
    },
    {
        "label": "5deg-v4-z5",
        "dir_name": "model_small_5deg_v4_z5_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "4.0",
        "rotation_only": 1,
    },
]


def _parse_eval_stats(extrinsics_path):
    """Parse the EVALUATION STATISTICS block from extrinsics_and_errors.txt."""
    result = {}
    if not os.path.isfile(extrinsics_path):
        return None
    with open(extrinsics_path, 'r') as f:
        text = f.read()
    stats_start = text.find("EVALUATION STATISTICS")
    if stats_start < 0:
        return None
    block = text[stats_start:]

    m = re.search(r'Total samples evaluated:\s*(\d+)', block)
    result['samples'] = int(m.group(1)) if m else 0

    rot_names = {
        'Total': 'rot_error', 'Roll (X)': 'roll_error', 'Roll (LiDAR-X)': 'roll_error',
        'Pitch (Y)': 'pitch_error', 'Pitch (LiDAR-Y)': 'pitch_error',
        'Yaw (Z)': 'yaw_error', 'Yaw (LiDAR-Z)': 'yaw_error',
    }
    trans_names = {
        'Total': 'trans_error', 'X (Fwd)': 'fwd_error',
        'Y (Lat)': 'lat_error', 'Z (Ht)': 'ht_error',
    }

    in_rot_section = False
    in_trans_section = False
    for line in block.split('\n'):
        stripped = line.strip()
        if 'Rotation Errors' in stripped:
            in_rot_section = True
            in_trans_section = False
            continue
        elif 'Translation Errors' in stripped:
            in_trans_section = True
            in_rot_section = False
            continue
        elif stripped.startswith('===') or stripped.startswith('AVERAGE'):
            in_rot_section = False
            in_trans_section = False
            continue

        active_map = None
        if in_rot_section:
            active_map = rot_names
        elif in_trans_section:
            active_map = trans_names
        if active_map is None:
            continue

        for display_name, key in active_map.items():
            if stripped.startswith(display_name):
                parts = stripped.split()
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        continue
                if len(nums) >= 8:
                    try:
                        result[f'{key}_mean'] = nums[0]
                        result[f'{key}_std'] = nums[1]
                        result[f'{key}_min'] = nums[2]
                        result[f'{key}_median'] = nums[3]
                        result[f'{key}_p90'] = nums[4]
                        result[f'{key}_p95'] = nums[5]
                        result[f'{key}_p99'] = nums[6]
                        result[f'{key}_max'] = nums[7]
                    except IndexError:
                        pass
    return result if 'rot_error_mean' in result else None


def _parse_per_sample_errors(extrinsics_path):
    """Parse per-sample rotation errors from extrinsics_and_errors.txt."""
    errors = []
    if not os.path.isfile(extrinsics_path):
        return errors
    with open(extrinsics_path, 'r') as f:
        for line in f:
            m = re.match(r'\s*Total:\s+([\d.]+)\s+deg', line)
            if m:
                errors.append(float(m.group(1)))
    return errors


def _generate_multi_charts(all_stats, output_dir):
    """Generate comparison charts for multi-model evaluation."""
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    labels = [s['label'] for s in all_stats]
    n = len(labels)

    z_color_map = {'z1': '#e74c3c', 'z5': '#e67e22', 'z10': '#2196F3'}
    def _color(label):
        for zk, c in z_color_map.items():
            if zk in label:
                return c
        return '#888'
    colors = [_color(l) for l in labels]

    def _save(fig, name):
        fig.savefig(os.path.join(charts_dir, name), dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)

    means = [s.get('rot_error_mean', 0) for s in all_stats]
    p95s = [s.get('rot_error_p95', 0) for s in all_stats]
    maxs = [s.get('rot_error_max', 0) for s in all_stats]

    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(12, n * 1.5), 6))
    ax.bar(x - w, means, w, label='Mean', color=[c for c in colors], alpha=0.9,
           edgecolor='white', linewidth=1)
    ax.bar(x, p95s, w, label='P95', color=[c for c in colors], alpha=0.6,
           edgecolor='white', linewidth=1)
    ax.bar(x + w, maxs, w, label='Max', color=[c for c in colors], alpha=0.35,
           edgecolor='white', linewidth=1)
    for i, v in enumerate(means):
        ax.text(i - w, v + 0.1, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Rotation Error (deg)', fontsize=12)
    ax.set_title('Rotation Error: Mean / P95 / Max', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    _save(fig, 'rotation_error_bar.png')
    print(f"   rotation_error_bar.png")

    rolls = [s.get('roll_error_mean', 0) for s in all_stats]
    pitches = [s.get('pitch_error_mean', 0) for s in all_stats]
    yaws = [s.get('yaw_error_mean', 0) for s in all_stats]
    fig, ax = plt.subplots(figsize=(max(12, n * 1.5), 6))
    ax.bar(x, rolls, 0.6, label='Roll (LiDAR-X)', color='#1abc9c')
    ax.bar(x, pitches, 0.6, bottom=rolls, label='Pitch (LiDAR-Y)', color='#f39c12')
    bottoms = [r + p for r, p in zip(rolls, pitches)]
    ax.bar(x, yaws, 0.6, bottom=bottoms, label='Yaw (LiDAR-Z)', color='#8e44ad')
    for i, v in enumerate(means):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Mean Rotation Error (deg)', fontsize=12)
    ax.set_title('Rotation Error Component Breakdown (Roll + Pitch + Yaw)', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    _save(fig, 'rotation_components_bar.png')
    print(f"   rotation_components_bar.png")

    sorted_idx = np.argsort(means)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.6)))
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_means = [means[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    bars = ax.barh(range(n), sorted_means, color=sorted_colors, edgecolor='white',
                   linewidth=1.5, height=0.6)
    for i, (bar, v) in enumerate(zip(bars, sorted_means)):
        ax.text(v + 0.1, i, f'{v:.2f} deg', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_labels, fontsize=11)
    ax.set_xlabel('Mean Rotation Error (deg)', fontsize=12)
    ax.set_title('Model Ranking (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, 'model_ranking.png')
    print(f"   model_ranking.png")

    print(f"   All charts saved to: {charts_dir}/")


def _generate_feishu_report(all_stats, output_dir, args):
    """Generate Feishu-compatible markdown report."""
    report_path = os.path.join(output_dir, "generalization_report.md")
    lines = []

    lines.append("BEVCalib 多模型泛化性能对比报告\n")
    lines.append(f"数据集: test_data ({all_stats[0].get('samples', '?')} samples) | "
                 f"扰动范围: +/-{args.angle_range_deg} deg | "
                 f"评估日期: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("\n一、实验概况\n")
    lines.append("| 模型标签 | 训练角度 | Z体素 | 架构 | 优化模式 | Checkpoint |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for m in MULTI_EVAL_MODELS:
        angle = "10deg" if "10deg" in m["label"] else "5deg"
        z = m["label"].split("-")[-1]
        ver = m["label"].split("-")[1] if "-" in m["label"] else "?"
        mode = "rotation_only" if m["rotation_only"] else "rotation+translation"
        lines.append(f"| {m['label']} | {angle} | {z} | {ver} | {mode} | {m['ckpt']} |")

    lines.append("\n\n二、旋转误差总览 (Total Rotation Error, deg)\n")
    lines.append("| 模型 | Mean | Std | Median | P90 | P95 | Max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in sorted(all_stats, key=lambda x: x.get('rot_error_mean', 999)):
        lines.append(
            f"| {s['label']} "
            f"| {s.get('rot_error_mean', -1):.3f} "
            f"| {s.get('rot_error_std', -1):.3f} "
            f"| {s.get('rot_error_median', -1):.3f} "
            f"| {s.get('rot_error_p90', -1):.3f} "
            f"| {s.get('rot_error_p95', -1):.3f} "
            f"| {s.get('rot_error_max', -1):.3f} |"
        )

    lines.append("\n![Rotation Error Bar Chart](charts/rotation_error_bar.png)\n")

    lines.append("\n三、旋转分量分析 (Roll / Pitch / Yaw Mean, deg)\n")
    lines.append("| 模型 | Roll(LiDAR-X) | Pitch(LiDAR-Y) | Yaw(LiDAR-Z) | Total |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for s in sorted(all_stats, key=lambda x: x.get('rot_error_mean', 999)):
        lines.append(
            f"| {s['label']} "
            f"| {s.get('roll_error_mean', -1):.3f} "
            f"| {s.get('pitch_error_mean', -1):.3f} "
            f"| {s.get('yaw_error_mean', -1):.3f} "
            f"| {s.get('rot_error_mean', -1):.3f} |"
        )

    lines.append("\n![Rotation Components](charts/rotation_components_bar.png)\n")

    has_trans = [s for s in all_stats if s.get('trans_error_mean') is not None
                 and s.get('trans_error_mean', 0) > 0]
    if has_trans:
        lines.append("\n四、平移误差对比 (仅 rotation+translation 模型, m)\n")
        lines.append("| 模型 | Trans Mean | Fwd(X) | Lat(Y) | Ht(Z) | Trans P95 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for s in has_trans:
            lines.append(
                f"| {s['label']} "
                f"| {s.get('trans_error_mean', -1):.4f} "
                f"| {s.get('fwd_error_mean', -1):.4f} "
                f"| {s.get('lat_error_mean', -1):.4f} "
                f"| {s.get('ht_error_mean', -1):.4f} "
                f"| {s.get('trans_error_p95', -1):.4f} |"
            )

    lines.append("\n\n五、模型排名 (按 Mean Rotation Error)\n")
    lines.append("| 排名 | 模型 | Mean Rot(deg) | P95 Rot(deg) | Max Rot(deg) |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for rank, s in enumerate(
            sorted(all_stats, key=lambda x: x.get('rot_error_mean', 999)), 1):
        lines.append(
            f"| {rank} | {s['label']} "
            f"| {s.get('rot_error_mean', -1):.3f} "
            f"| {s.get('rot_error_p95', -1):.3f} "
            f"| {s.get('rot_error_max', -1):.3f} |"
        )

    lines.append("\n![Model Ranking](charts/model_ranking.png)\n")

    lines.append("\n六、结论与建议\n")
    best = min(all_stats, key=lambda x: x.get('rot_error_mean', 999))
    worst = max(all_stats, key=lambda x: x.get('rot_error_mean', 999))
    lines.append(f"- 最佳泛化模型: {best['label']} (Mean Rot: {best.get('rot_error_mean', -1):.3f} deg)")
    lines.append(f"- 最差泛化模型: {worst['label']} (Mean Rot: {worst.get('rot_error_mean', -1):.3f} deg)")
    ratio = worst.get('rot_error_mean', 1) / max(best.get('rot_error_mean', 1), 0.001)
    lines.append(f"- 最差/最佳比值: {ratio:.1f}x")

    report_text = "\n".join(lines) + "\n"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n   Report saved to: {report_path}")
    return report_path


def multi_eval_and_report(args):
    """Evaluate all models on test data via subprocesses and generate report."""
    models_dir = args.models_dir
    output_dir = args.output_dir or "logs/multi_eval_test_data"
    os.makedirs(output_dir, exist_ok=True)

    angle_str = f"{args.angle_range_deg} deg" if args.angle_range_deg is not None else "auto (from checkpoint)"
    print("=" * 80)
    print("多模型泛化性能评估")
    print(f"  模型目录: {models_dir}")
    print(f"  数据集: {args.dataset_root}")
    print(f"  扰动: {angle_str}")
    print(f"  输出: {output_dir}")
    print(f"  模型数: {len(MULTI_EVAL_MODELS)}")
    print("=" * 80)

    script_path = os.path.abspath(__file__)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    all_stats = []
    for idx, mcfg in enumerate(MULTI_EVAL_MODELS):
        label = mcfg["label"]
        ckpt_dir_candidates = [
            os.path.join(models_dir, mcfg["dir_name"], "all_training_data_scratch", "checkpoint"),
            os.path.join(models_dir, mcfg["dir_name"], "B26A_scratch", "checkpoint"),
        ]
        ckpt_path = None
        for cdir in ckpt_dir_candidates:
            candidate = os.path.join(cdir, mcfg["ckpt"])
            if os.path.isfile(candidate):
                ckpt_path = candidate
                break
        if ckpt_path is None:
            print(f"\n[SKIP] {label}: checkpoint not found in {ckpt_dir_candidates}")
            continue

        per_model_dir = os.path.join(output_dir, label)
        extrinsics_path = os.path.join(per_model_dir, "extrinsics_and_errors.txt")

        if os.path.isfile(extrinsics_path) and "EVALUATION STATISTICS" in open(extrinsics_path).read():
            print(f"\n[{idx+1}/{len(MULTI_EVAL_MODELS)}] {label}: 已有结果，跳过评估")
        else:
            print(f"\n[{idx+1}/{len(MULTI_EVAL_MODELS)}] {label}: 开始评估...")
            print(f"   ckpt: {ckpt_path}")
            print(f"   BEV_ZBOUND_STEP={mcfg['bev_zbound_step']}, rotation_only={mcfg['rotation_only']}")

            env = os.environ.copy()
            env["BEV_ZBOUND_STEP"] = mcfg["bev_zbound_step"]

            python_bin = sys.executable
            cmd = [
                python_bin, script_path,
                "--mode", "eval",
                "--ckpt_path", ckpt_path,
                "--dataset_root", args.dataset_root,
                "--output_dir", per_model_dir,
                "--use_full_dataset",
                "--max_batches", str(args.max_batches),
                "--rotation_only", str(mcfg["rotation_only"]),
                "--vis_interval", str(args.vis_interval),
                "--batch_size", str(args.batch_size),
                "--deformable", str(args.deformable),
                "--bev_encoder", str(args.bev_encoder),
                "--target_width", str(args.target_width),
                "--target_height", str(args.target_height),
                "--use_mlp_head", str(args.use_mlp_head),
            ]
            if args.angle_range_deg is not None:
                cmd += ["--angle_range_deg", str(args.angle_range_deg)]
            if args.trans_range is not None:
                cmd += ["--trans_range", str(args.trans_range)]

            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0:
                print(f"   [ERROR] {label} 评估失败 (exit {proc.returncode})")
                if proc.stderr:
                    for errline in proc.stderr.strip().split('\n')[-10:]:
                        print(f"      {errline}")
                continue
            print(f"   [OK] {label} 评估完成")

        stats = _parse_eval_stats(extrinsics_path)
        if stats:
            stats['label'] = label
            all_stats.append(stats)
            rot_m = stats.get('rot_error_mean', -1)
            rot_p95 = stats.get('rot_error_p95', -1)
            print(f"   Rot: Mean={rot_m:.3f} P95={rot_p95:.3f} deg ({stats.get('samples', '?')} samples)")
        else:
            print(f"   [WARN] {label}: 无法解析评估结果")

    if not all_stats:
        print("\n[ERROR] 没有可用的评估结果，无法生成报告")
        return

    print(f"\n{'='*80}")
    print(f"生成对比报告和图表 ({len(all_stats)} 个模型)...")

    _generate_multi_charts(all_stats, output_dir)
    report_path = _generate_feishu_report(all_stats, output_dir, args)

    print(f"\n{'='*80}")
    print(f"多模型泛化评估完成!")
    print(f"  评估模型数: {len(all_stats)}/{len(MULTI_EVAL_MODELS)}")
    print(f"  报告: {report_path}")
    print(f"  图表: {os.path.join(output_dir, 'charts')}/")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="评估已保存的 checkpoint")
    parser.add_argument("--mode", type=str, default="eval",
                       choices=["eval", "compare", "multi_eval"],
                       help="运行模式: eval=单模型评估, compare=两模型对比, multi_eval=多模型泛化评估")

    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint 文件路径 (eval模式)")
    parser.add_argument("--ckpt_path_a", type=str, default=None, help="模型A checkpoint (compare模式)")
    parser.add_argument("--ckpt_path_b", type=str, default=None, help="模型B checkpoint (compare模式)")
    parser.add_argument("--label_a", type=str, default="ModelA", help="模型A标签 (compare模式)")
    parser.add_argument("--label_b", type=str, default="ModelB", help="模型B标签 (compare模式)")
    parser.add_argument("--models_dir", type=str, default="logs/all_training_data",
                       help="模型根目录 (multi_eval模式)")

    parser.add_argument("--dataset_root", type=str, required=False, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="评估结果输出目录（默认: checkpoint目录下的 ckpt_xxx_eval/）。"
                            "跨数据集泛化测试时建议指定独立输出目录")
    parser.add_argument("--angle_range_deg", type=float, default=None,
                       help="扰动角度范围（默认从checkpoint的eval_noise/train_noise读取，若无则20.0）")
    parser.add_argument("--trans_range", type=float, default=None,
                       help="扰动平移范围（默认从checkpoint的eval_noise/train_noise读取，若无则1.5）")
    parser.add_argument("--target_width", type=int, default=640, help="目标图像宽度")
    parser.add_argument("--target_height", type=int, default=360, help="目标图像高度")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=0, help="最多评估的batch数（0表示全部）")
    parser.add_argument("--validate_sample_ratio", type=float, default=0.2,
                       help="验证集比例（默认0.2，与训练时80/20划分一致）")
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
    parser.add_argument("--rotation_only", type=int, default=-1,
                       help="仅优化旋转 (1=仅旋转, 0=旋转+平移同时优化). "
                            "设为-1时自动从checkpoint中读取（默认-1）")
    parser.add_argument("--use_mlp_head", type=int, default=-1,
                       help="回归头类型 (1=MLP, 0=Linear, -1=自动从checkpoint检测)")
    parser.add_argument("--bev_pool_factor", type=int, default=0,
                       help="BEV spatial avg-pool factor (0=disabled, 2=2x2 pool)")
    parser.add_argument("--use_drcv", action='store_true', default=False,
                       help="Use drcv sparse conv backend (env var handled before import)")
    parser.add_argument("--eval_seed", type=int, default=42,
                       help="Fixed seed for perturbation generation, ensuring reproducible evaluation (default: 42)")
    parser.add_argument("--use_foundation_depth", type=int, default=0,
                       help="Use Foundation Depth (MiDaS) for LSS (1=enable, 0=disable)")
    parser.add_argument("--depth_model_type", type=str, default="midas_small",
                       help="Depth model type for Foundation Depth (default: midas_small)")
    parser.add_argument("--fd_mode", type=str, default="replace",
                       choices=["replace", "replace_v1", "replace_v2", "dual_path", "supervision"],
                       help="Foundation Depth mode (default: replace). "
                            "replace_v2=SpatialAligner, dual_path=fusion, supervision=aux loss")
    
    args = parser.parse_args()

    if args.mode == "multi_eval":
        if not args.dataset_root:
            parser.error("multi_eval模式需要 --dataset_root")
        multi_eval_and_report(args)
    elif args.mode == "compare":
        if not args.ckpt_path_a or not args.ckpt_path_b:
            parser.error("compare模式需要 --ckpt_path_a 和 --ckpt_path_b")
        compare_checkpoints(args)
    else:
        if not args.ckpt_path:
            parser.error("eval模式需要 --ckpt_path")
        evaluate_checkpoint(args)

if __name__ == "__main__":
    main()
