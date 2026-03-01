#!/usr/bin/env python3
"""
批量生成点云投影可视化图像
用于快速验证大量帧的投影效果
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import sys


def load_calib(calib_file):
    """加载标定文件"""
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, values = line.split(':', 1)
            key = key.strip()
            values = values.strip()
            
            try:
                calib[key] = np.array([float(x) for x in values.split()])
            except ValueError:
                calib[key] = values
    return calib


def load_pointcloud(bin_file):
    """加载点云文件"""
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape(-1, 4)
    return points


def project_points(points, calib, img_shape):
    """投影点云到图像"""
    Tr_3x4 = calib['Tr'].reshape(3, 4)
    Tr = np.vstack([Tr_3x4, [0, 0, 0, 1]])
    P2 = calib['P2'].reshape(3, 4)
    
    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    
    Tr_inv = np.linalg.inv(Tr)
    points_cam = (Tr_inv @ points_hom.T).T[:, :3]
    
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    depths = points_cam[:, 2].copy()
    
    if len(points_cam) == 0:
        return None, None, 0
    
    points_cam_hom = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
    points_img = (P2 @ points_cam_hom.T).T
    points_img = points_img[:, :2] / points_img[:, 2:3]
    
    H, W = img_shape[:2]
    mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < W) & \
           (points_img[:, 1] >= 0) & (points_img[:, 1] < H)
    
    points_img = points_img[mask]
    depths = depths[mask]
    
    return points_img, depths, mask.sum()


def generate_projection(dataset_root, sequence, frame, output_file, verbose=False):
    """生成单帧投影图像"""
    dataset_root = Path(dataset_root)
    seq_dir = dataset_root / 'sequences' / sequence
    
    img_file = seq_dir / 'image_2' / f'{frame:06d}.png'
    pc_file = seq_dir / 'velodyne' / f'{frame:06d}.bin'
    calib_file = seq_dir / 'calib.txt'
    
    if not img_file.exists() or not pc_file.exists():
        if verbose:
            print(f"跳过帧 {frame}: 文件不存在")
        return False, 0
    
    try:
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points = load_pointcloud(pc_file)
        calib = load_calib(calib_file)
        
        points_img, depths, num_visible = project_points(points, calib, img.shape)
        
        if num_visible == 0:
            if verbose:
                print(f"跳过帧 {frame}: 没有投影点")
            return False, 0
        
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
        ax.imshow(img)
        
        scatter = ax.scatter(points_img[:, 0], points_img[:, 1],
                            c=depths, cmap='jet', s=1, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Depth (m)')
        
        ax.set_title(f'Sequence {sequence} - Frame {frame:06d} | '
                    f'Projected: {num_visible}/{points.shape[0]}',
                    fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True, num_visible
        
    except Exception as e:
        if verbose:
            print(f"错误处理帧 {frame}: {e}")
        return False, 0


def main():
    parser = argparse.ArgumentParser(description='批量生成点云投影可视化图像')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100,
                       help='生成样本数量（默认: 100）')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='起始帧（默认: 0）')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='结束帧（默认: None，自动检测）')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检测总帧数
    seq_dir = dataset_root / 'sequences' / args.sequence
    velodyne_files = sorted(list((seq_dir / 'velodyne').glob('*.bin')))
    total_frames = len(velodyne_files)
    
    if total_frames == 0:
        print(f"错误: Sequence {args.sequence} 中没有找到点云文件")
        return
    
    print(f"\n{'='*80}")
    print(f"批量生成投影可视化图像")
    print(f"{'='*80}")
    print(f"数据集: {dataset_root}")
    print(f"Sequence: {args.sequence}")
    print(f"总帧数: {total_frames}")
    print(f"输出目录: {output_dir}")
    print(f"目标样本数: {args.num_samples}")
    
    # 确定帧范围
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else total_frames - 1
    
    # 均匀采样
    frame_indices = np.linspace(start_frame, end_frame, 
                               min(args.num_samples, end_frame - start_frame + 1),
                               dtype=int)
    
    print(f"\n采样策略:")
    print(f"  起始帧: {frame_indices[0]}")
    print(f"  结束帧: {frame_indices[-1]}")
    print(f"  实际采样: {len(frame_indices)} 帧")
    print(f"  采样间隔: 约每 {(end_frame - start_frame) / len(frame_indices):.1f} 帧")
    
    print(f"\n开始生成投影图像...")
    
    success_count = 0
    total_projection_points = 0
    
    for frame in tqdm(frame_indices, desc="生成投影图像"):
        output_file = output_dir / f'seq{args.sequence}_frame{frame:06d}.png'
        success, num_points = generate_projection(
            args.dataset_root, args.sequence, frame, output_file
        )
        
        if success:
            success_count += 1
            total_projection_points += num_points
    
    print(f"\n{'='*80}")
    print(f"完成！")
    print(f"{'='*80}")
    print(f"成功生成: {success_count}/{len(frame_indices)} 张图像")
    print(f"平均投影点数: {total_projection_points/max(success_count,1):.0f} 点/帧")
    print(f"输出目录: {output_dir}")
    print(f"\n查看图像:")
    print(f"  cd {output_dir}")
    print(f"  ls -lh")


if __name__ == '__main__':
    main()
