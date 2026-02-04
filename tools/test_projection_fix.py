#!/usr/bin/env python3
"""
测试投影修复效果

对比修复前后的投影效果，验证是否与C++版本一致
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from visualize_projection import (
    load_calib, 
    project_and_render,
    get_camera_fov
)


def test_projection(dataset_root: str, frame_idx: int = 0):
    """测试投影效果"""
    dataset_root = Path(dataset_root)
    seq_dir = dataset_root / 'sequences' / '00'
    
    # 加载数据
    image_path = seq_dir / 'image_2' / f'{frame_idx:06d}.png'
    pcd_path = seq_dir / 'velodyne' / f'{frame_idx:06d}.bin'
    calib_path = seq_dir / 'calib.txt'
    
    if not all(p.exists() for p in [image_path, pcd_path, calib_path]):
        print(f"❌ 文件不存在")
        return
    
    # 加载标定
    Tr, K, D, camera_model = load_calib(str(calib_path))
    
    # 加载图像
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    
    # 加载点云
    pcd = np.fromfile(str(pcd_path), dtype=np.float32).reshape(-1, 4)
    
    print("=" * 60)
    print("投影测试")
    print("=" * 60)
    print(f"\n数据集: {dataset_root}")
    print(f"帧索引: {frame_idx}")
    print(f"图像尺寸: {w}x{h}")
    print(f"点云数量: {len(pcd)}")
    print(f"相机模型: {camera_model}")
    
    # 打印点云范围
    print(f"\n点云范围:")
    print(f"  X: [{pcd[:, 0].min():.2f}, {pcd[:, 0].max():.2f}]")
    print(f"  Y: [{pcd[:, 1].min():.2f}, {pcd[:, 1].max():.2f}]")
    print(f"  Z: [{pcd[:, 2].min():.2f}, {pcd[:, 2].max():.2f}]")
    
    # 计算FOV
    fov_rad = get_camera_fov(K, D, (h, w))
    print(f"\n相机FOV: {np.degrees(fov_rad):.2f}°")
    
    # 打印Tr矩阵
    print(f"\nTr矩阵 (LiDAR → Camera):")
    print(Tr)
    
    # 测试1: 使用修复后的投影（无距离过滤）
    print("\n" + "=" * 60)
    print("测试1: 修复后的投影（无距离过滤，对齐C++）")
    print("=" * 60)
    
    img_fixed, num_points_fixed = project_and_render(
        pcd, image.copy(), K, Tr, D, camera_model,
        use_fov_filter=True,
        use_distance_filter=False,  # 关闭距离过滤
        point_radius=2,
        unit_depth=2.0,
        verbose=True
    )
    
    # 测试2: 使用旧的投影（有距离过滤）
    print("\n" + "=" * 60)
    print("测试2: 旧的投影（有距离过滤）")
    print("=" * 60)
    
    img_old, num_points_old = project_and_render(
        pcd, image.copy(), K, Tr, D, camera_model,
        use_fov_filter=True,
        use_distance_filter=True,  # 开启距离过滤
        max_depth=100.0,
        point_radius=2,
        unit_depth=2.0,
        verbose=True
    )
    
    # 保存结果
    output_dir = Path('./projection_test_results')
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / f'frame_{frame_idx:06d}_fixed.jpg'), img_fixed)
    cv2.imwrite(str(output_dir / f'frame_{frame_idx:06d}_old.jpg'), img_old)
    cv2.imwrite(str(output_dir / f'frame_{frame_idx:06d}_original.jpg'), image)
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"  - frame_{frame_idx:06d}_fixed.jpg (修复后，{num_points_fixed}点)")
    print(f"  - frame_{frame_idx:06d}_old.jpg (旧版本，{num_points_old}点)")
    print(f"  - frame_{frame_idx:06d}_original.jpg (原始图像)")
    
    # 对比
    print(f"\n对比:")
    print(f"  修复后点数: {num_points_fixed}")
    print(f"  旧版本点数: {num_points_old}")
    print(f"  增加: {num_points_fixed - num_points_old} 点 ({(num_points_fixed - num_points_old) / max(num_points_old, 1) * 100:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='测试投影修复效果')
    parser.add_argument('--dataset_root', type=str, 
                       default='/home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data',
                       help='数据集根目录')
    parser.add_argument('--frame', type=int, default=0, help='帧索引')
    args = parser.parse_args()
    
    test_projection(args.dataset_root, args.frame)


if __name__ == '__main__':
    main()
