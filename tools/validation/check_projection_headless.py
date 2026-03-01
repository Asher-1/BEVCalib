#!/usr/bin/env python3
"""
无头模式点云投影验证工具
用于无图形界面环境验证点云投影效果
"""

import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


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
            
            # 跳过非数字行（如 camera_model）
            try:
                calib[key] = np.array([float(x) for x in values.split()])
            except ValueError:
                calib[key] = values  # 保存为字符串
    return calib


def load_pointcloud(bin_file):
    """加载点云文件"""
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape(-1, 4)
    return points


def project_points(points, calib, img_shape):
    """投影点云到图像
    
    注意：KITTI标准格式中 Tr = Camera→Sensing/Velodyne
         投影时需要使用 inv(Tr) = Sensing/Velodyne→Camera
    """
    # 获取标定参数
    Tr_3x4 = calib['Tr'].reshape(3, 4)
    # 扩展为4x4矩阵
    Tr = np.vstack([Tr_3x4, [0, 0, 0, 1]])
    P2 = calib['P2'].reshape(3, 4)
    
    # 转换为齐次坐标
    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    
    # 变换到相机坐标系（使用inv(Tr)！）
    Tr_inv = np.linalg.inv(Tr)  # Sensing/Velodyne → Camera
    points_cam = (Tr_inv @ points_hom.T).T[:, :3]
    
    # 过滤掉相机后面的点
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    depths = points_cam[:, 2].copy()
    
    if len(points_cam) == 0:
        return None, None, 0
    
    # 投影到图像
    points_cam_hom = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
    points_img = (P2 @ points_cam_hom.T).T
    points_img = points_img[:, :2] / points_img[:, 2:3]
    
    # 过滤图像内的点
    H, W = img_shape[:2]
    mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < W) & \
           (points_img[:, 1] >= 0) & (points_img[:, 1] < H)
    
    points_img = points_img[mask]
    depths = depths[mask]
    
    return points_img, depths, mask.sum()


def visualize_projection(dataset_root, sequence, frame, output_file):
    """可视化点云投影"""
    dataset_root = Path(dataset_root)
    seq_dir = dataset_root / 'sequences' / sequence
    
    # 加载数据
    print(f"\n处理 Sequence {sequence}, Frame {frame}")
    print("="*60)
    
    img_file = seq_dir / 'image_2' / f'{frame:06d}.png'
    pc_file = seq_dir / 'velodyne' / f'{frame:06d}.bin'
    calib_file = seq_dir / 'calib.txt'
    
    print(f"图像: {img_file}")
    print(f"点云: {pc_file}")
    print(f"标定: {calib_file}")
    
    # 加载数据
    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = load_pointcloud(pc_file)
    calib = load_calib(calib_file)
    
    print(f"\n✓ 图像尺寸: {img.shape}")
    print(f"✓ 点云数量: {points.shape[0]} 点")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # 投影
    points_img, depths, num_visible = project_points(points, calib, img.shape)
    
    print(f"\n投影结果:")
    print(f"  相机前方点数: {(points[:, 0] > 0).sum()} / {points.shape[0]}")
    print(f"  图像内点数: {num_visible}")
    
    if num_visible == 0:
        print("  ❌ 没有点被投影到图像上！")
        print("\n可能的原因：")
        print("  1. 标定矩阵不正确")
        print("  2. 点云坐标系不匹配")
        print("  3. 点云在相机视野外")
        return False
    
    # 创建可视化
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.imshow(img)
    
    # 深度着色
    scatter = ax.scatter(points_img[:, 0], points_img[:, 1],
                        c=depths, cmap='jet', s=1, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    
    ax.set_title(f'Sequence {sequence} - Frame {frame:06d}\n'
                f'投影点数: {num_visible}/{points.shape[0]}')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 投影图像已保存: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='无头模式点云投影验证')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    success = visualize_projection(
        args.dataset_root,
        args.sequence,
        args.frame,
        args.output
    )
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()
