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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

from custom_dataset import CustomDataset
from bev_calib import BEVCalib
from tools import generate_single_perturbation_from_T
from visualization import (
    compute_batch_pose_errors,
    visualize_batch_projection,
    compute_pose_errors
)

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
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 checkpoint
    print(f"\n1. 加载 checkpoint...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"   Epoch: {epoch}")
    
    # 创建模型
    print(f"\n2. 初始化模型（图像尺寸: {args.target_width}x{args.target_height}）...")
    model = BEVCalib(
        deformable=args.deformable > 0,
        bev_encoder=args.bev_encoder > 0,
        img_shape=(args.target_height, args.target_width)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ 模型加载完成")
    
    # 加载数据集
    print(f"\n3. 加载数据集...")
    dataset = CustomDataset(
        data_folder=args.dataset_root,
        auto_detect=True
    )
    
    # 划分训练集和验证集（使用相同的划分方式）
    val_size = int(len(dataset) * args.validate_sample_ratio)
    train_size = len(dataset) - val_size
    
    from torch.utils.data import random_split
    import torch as torch_module
    generator = torch_module.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    collate_fn = make_collate_fn((args.target_width, args.target_height))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    print(f"   ✓ 验证集: {len(val_dataset)} 个样本")
    
    # 创建评估输出目录
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
                trans_range=args.trans_range
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
            
            # 为每个样本生成可视化和保存外参
            for i in range(len(imgs_np)):
                sample_idx = sample_count + i
                
                print(f"   处理样本 {sample_idx}...", end='', flush=True)
                
                # 🔍 调试：检查D是否被传递
                if sample_idx == 0:
                    D_to_pass = [distortions[i]] if distortions is not None and i < len(distortions) else None
                    print(f"\n[DEBUG evaluation] distortions变量: {distortions is not None}")
                    if distortions is not None and i < len(distortions):
                        print(f"[DEBUG evaluation] D = {distortions[i]}")
                        print(f"[DEBUG evaluation] D传递到visualization: {D_to_pass}")
                    else:
                        print(f"[DEBUG evaluation] ⚠️  D未传递！distortions={distortions}")
                
                # 生成单样本可视化
                vis_image = visualize_batch_projection(
                    images=imgs_np[i:i+1],
                    points_batch=pcs_np[i:i+1],
                    init_T_batch=init_T_to_camera_np[i:i+1],
                    gt_T_batch=gt_T_to_camera_np[i:i+1],
                    pred_T_batch=T_pred_np[i:i+1],
                    K_batch=np.array(intrinsics)[i:i+1],
                    D_batch=[distortions[i]] if distortions is not None and i < len(distortions) else None,
                    camera_model='pinhole',  # 从calib.txt读取，默认pinhole
                    masks=masks_np[i:i+1],
                    num_samples=1,
                    max_points=args.vis_points,
                    point_radius=args.vis_point_radius,
                    use_inverse_transform=args.use_inverse_transform > 0
                )
                
                # 保存可视化图像
                vis_image_path = os.path.join(eval_dir, f"sample_{sample_idx:04d}_projection.png")
                cv2.imwrite(vis_image_path, vis_image)
                
                # 计算误差
                errors = compute_pose_errors(T_pred_np[i], gt_T_to_camera_np[i])
                
                # 累积误差
                for key in all_errors:
                    all_errors[key].append(errors[key])
                
                # 保存外参和误差信息
                with open(extrinsics_file, 'a') as f:
                    # 只在第一次写入文件头和 GT
                    if not gt_extrinsics_written:
                        f.write(f"Checkpoint: {os.path.basename(args.ckpt_path)}\n")
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Evaluation on validation set (perturbation: {args.angle_range_deg}deg, {args.trans_range}m)\n")
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
                
                print(f" ✓ (Trans: {errors['trans_error']:.4f}m, Rot: {errors['rot_error']:.2f}°)")
            
            sample_count += len(imgs_np)
    
    # 写入平均误差统计
    with open(extrinsics_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("AVERAGE ERRORS ACROSS ALL SAMPLES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples evaluated: {sample_count}\n\n")
        
        avg_errors = {key: np.mean(values) for key, values in all_errors.items()}
        std_errors = {key: np.std(values) for key, values in all_errors.items()}
        
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
    
    print(f"\n✓ 评估完成！")
    print(f"   - 评估样本数: {sample_count}")
    print(f"   - 输出目录: {eval_dir}")
    print(f"   - 外参文件: {extrinsics_file}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="评估已保存的 checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint 文件路径")
    parser.add_argument("--dataset_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--angle_range_deg", type=float, default=20.0, help="扰动角度范围")
    parser.add_argument("--trans_range", type=float, default=1.5, help="扰动平移范围")
    parser.add_argument("--target_width", type=int, default=640, help="目标图像宽度")
    parser.add_argument("--target_height", type=int, default=360, help="目标图像高度")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=5, help="最多评估的batch数（0表示全部）")
    parser.add_argument("--validate_sample_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--deformable", type=int, default=0, help="是否使用 deformable attention")
    parser.add_argument("--bev_encoder", type=int, default=1, help="是否使用 BEV encoder")
    parser.add_argument("--xyz_only", type=int, default=1, help="是否只使用 XYZ 坐标")
    parser.add_argument("--vis_points", type=int, default=80000, help="可视化最大点数")
    parser.add_argument("--vis_point_radius", type=int, default=1, help="可视化点半径")
    parser.add_argument("--use_inverse_transform", type=int, default=0, 
                       help="是否对变换矩阵取逆后再使用 (1=是, 0=否) - 用于修复投影高度偏移")
    
    args = parser.parse_args()
    evaluate_checkpoint(args)

if __name__ == "__main__":
    main()
