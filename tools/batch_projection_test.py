#!/usr/bin/env python3
"""
批量点云投影测试脚本
将投影结果保存到数据集目录的 projection_results 文件夹

用法:
    python tools/batch_projection_test.py --dataset_root <path> --frames 0 100 500 1000
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from visualize_projection import (
    load_calib, project_and_render, get_camera_fov
)


def batch_projection_test(dataset_root: Path, 
                         frames: list,
                         output_subdir: str = "projection_results"):
    """
    批量投影测试
    
    Args:
        dataset_root: 数据集根目录
        frames: 要测试的帧列表
        output_subdir: 输出子目录名
    """
    seq_dir = dataset_root / 'sequences' / '00'
    output_dir = dataset_root / output_subdir
    output_dir.mkdir(exist_ok=True)
    
    # 创建带时间戳的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"批量点云投影测试")
    print(f"{'='*70}")
    print(f"数据集: {dataset_root}")
    print(f"输出目录: {run_dir}")
    print(f"测试帧: {frames}")
    print(f"{'='*70}\n")
    
    # 加载标定参数（只需加载一次）
    calib_path = seq_dir / 'calib.txt'
    if not calib_path.exists():
        print(f"❌ 标定文件不存在: {calib_path}")
        return
    
    Tr, K, D, camera_model = load_calib(str(calib_path))
    
    # 计算相机FOV
    h, w = 2160, 3840  # 假设图像尺寸
    fov_rad = get_camera_fov(K, D, (h, w))
    fov_deg = np.degrees(fov_rad)
    
    print(f"标定参数:")
    print(f"  相机模型: {camera_model}")
    print(f"  FOV: {fov_deg:.2f}°")
    print(f"  K:\n{K}")
    print(f"  D: {D}")
    print(f"  Tr:\n{Tr}")
    print()
    
    # 统计信息
    results = []
    
    for frame_idx in frames:
        image_path = seq_dir / 'image_2' / f'{frame_idx:06d}.png'
        pc_path = seq_dir / 'velodyne' / f'{frame_idx:06d}.bin'
        
        if not image_path.exists():
            print(f"⚠️ 帧 {frame_idx}: 图像不存在")
            continue
        if not pc_path.exists():
            print(f"⚠️ 帧 {frame_idx}: 点云不存在")
            continue
        
        # 加载数据
        img = cv2.imread(str(image_path))
        points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)
        
        # 检查点云范围
        x_range = (points[:, 0].min(), points[:, 0].max())
        y_range = (points[:, 1].min(), points[:, 1].max())
        z_range = (points[:, 2].min(), points[:, 2].max())
        
        # 投影
        img_with_points, num_valid = project_and_render(
            points, img, K, Tr, D,
            camera_model=camera_model,
            min_depth=0.0,
            max_depth=200.0,
            use_fov_filter=True,
            point_radius=3,
            unit_depth=2.0,
            verbose=False
        )
        
        # 添加信息到图像
        info_text = [
            f"Frame: {frame_idx:06d}",
            f"Total points: {len(points)}",
            f"Projected: {num_valid}",
            f"X: [{x_range[0]:.1f}, {x_range[1]:.1f}]",
            f"Y: [{y_range[0]:.1f}, {y_range[1]:.1f}]",
            f"Z: [{z_range[0]:.1f}, {z_range[1]:.1f}]",
        ]
        
        y_offset = 50
        for text in info_text:
            cv2.putText(img_with_points, text, (30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            y_offset += 50
        
        # 保存结果
        output_path = run_dir / f'projection_{frame_idx:06d}.png'
        cv2.imwrite(str(output_path), img_with_points)
        
        # 记录结果
        result = {
            'frame': frame_idx,
            'total_points': len(points),
            'projected_points': num_valid,
            'projection_rate': num_valid / len(points) * 100 if len(points) > 0 else 0,
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range,
        }
        results.append(result)
        
        print(f"✓ 帧 {frame_idx:06d}: {num_valid}/{len(points)} 点 ({result['projection_rate']:.1f}%) -> {output_path.name}")
    
    # 保存统计报告
    report_path = run_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write(f"批量投影测试报告\n")
        f.write(f"{'='*60}\n")
        f.write(f"数据集: {dataset_root}\n")
        f.write(f"时间: {timestamp}\n")
        f.write(f"相机模型: {camera_model}\n")
        f.write(f"FOV: {fov_deg:.2f}°\n\n")
        
        f.write(f"帧统计:\n")
        f.write(f"{'-'*60}\n")
        for r in results:
            f.write(f"帧 {r['frame']:06d}:\n")
            f.write(f"  总点数: {r['total_points']}\n")
            f.write(f"  投影点数: {r['projected_points']}\n")
            f.write(f"  投影率: {r['projection_rate']:.1f}%\n")
            f.write(f"  X范围: [{r['x_range'][0]:.2f}, {r['x_range'][1]:.2f}]\n")
            f.write(f"  Y范围: [{r['y_range'][0]:.2f}, {r['y_range'][1]:.2f}]\n")
            f.write(f"  Z范围: [{r['z_range'][0]:.2f}, {r['z_range'][1]:.2f}]\n\n")
        
        # 汇总统计
        if results:
            avg_rate = np.mean([r['projection_rate'] for r in results])
            f.write(f"\n汇总:\n")
            f.write(f"  测试帧数: {len(results)}\n")
            f.write(f"  平均投影率: {avg_rate:.1f}%\n")
    
    print(f"\n{'='*70}")
    print(f"测试完成！")
    print(f"  结果保存到: {run_dir}")
    print(f"  报告: {report_path}")
    print(f"{'='*70}\n")
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='批量点云投影测试')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--frames', type=int, nargs='+', 
                       default=[0, 50, 100, 200, 500, 1000],
                       help='要测试的帧索引列表')
    parser.add_argument('--output_subdir', type=str, default='projection_results',
                       help='输出子目录名')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ 数据集不存在: {dataset_root}")
        return
    
    batch_projection_test(dataset_root, args.frames, args.output_subdir)


if __name__ == '__main__':
    main()
