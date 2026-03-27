#!/usr/bin/env python3
"""
全面的点云投影验证工具
对每个序列采样多个关键帧进行投影验证，并按序列分类存储
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime


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
    """投影LiDAR点云到图像
    
    Tr = Camera→LiDAR (KITTI标准), inv(Tr) = LiDAR→Camera
    """
    Tr_3x4 = calib['Tr'].reshape(3, 4)
    Tr = np.vstack([Tr_3x4, [0, 0, 0, 1]])
    P2 = calib['P2'].reshape(3, 4)
    
    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    
    # LiDAR→Camera
    Tr_inv = np.linalg.inv(Tr)
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
    img_file = seq_dir / 'image_2' / f'{frame:06d}.png'
    pc_file = seq_dir / 'velodyne' / f'{frame:06d}.bin'
    calib_file = seq_dir / 'calib.txt'
    
    if not img_file.exists() or not pc_file.exists() or not calib_file.exists():
        print(f"  ⚠️ 文件不存在，跳过帧 {frame}")
        return None
    
    # 加载数据
    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = load_pointcloud(pc_file)
    calib = load_calib(calib_file)
    
    # 投影
    points_img, depths, num_visible = project_points(points, calib, img.shape)
    
    if num_visible == 0:
        print(f"  ❌ 帧 {frame}: 没有可见点")
        return None
    
    # 创建可视化
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.imshow(img)
    
    # 深度着色
    scatter = ax.scatter(points_img[:, 0], points_img[:, 1],
                        c=depths, cmap='jet', s=1, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    
    ax.set_title(f'Sequence {sequence} - Frame {frame:06d}\n'
                f'Visible Points: {num_visible}/{points.shape[0]} ({100*num_visible/points.shape[0]:.1f}%)',
                fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 返回统计信息
    return {
        'frame': frame,
        'total_points': int(points.shape[0]),
        'visible_points': int(num_visible),
        'visible_ratio': float(num_visible / points.shape[0]),
        'depth_min': float(depths.min()),
        'depth_max': float(depths.max()),
        'depth_mean': float(depths.mean()),
        'output_file': str(output_file)
    }


def get_sample_frames(num_frames, num_samples=10):
    """获取采样帧索引"""
    if num_frames <= num_samples:
        return list(range(num_frames))
    
    # 采样策略：均匀分布10个采样点
    # 包含开始和结束，中间8个点均匀分布
    indices = []
    for i in range(num_samples):
        idx = int(i * (num_frames - 1) / (num_samples - 1))
        indices.append(idx)
    
    return sorted(list(set(indices)))  # 去重并排序


def analyze_camera_direction(calib):
    """分析相机朝向（基于Tr矩阵）
    
    Tr = Camera→LiDAR, 其旋转矩阵R的第3列 = 相机Z轴(前向)在LiDAR系中的方向
    """
    Tr_3x4 = calib['Tr'].reshape(3, 4)
    R = Tr_3x4[:3, :3]
    
    cam_forward_in_lidar = R[:, 2]
    
    axis_labels = {0: 'X', 1: 'Y', 2: 'Z'}
    dominant_idx = int(np.argmax(np.abs(cam_forward_in_lidar)))
    dominant_sign = '+' if cam_forward_in_lidar[dominant_idx] > 0 else '-'
    dominant_axis = axis_labels[dominant_idx]
    
    direction_map = {
        (0, '+'): '前方(+X)', (0, '-'): '后方(-X)',
        (1, '+'): '左侧(+Y)', (1, '-'): '右侧(-Y)',
        (2, '+'): '上方(+Z)', (2, '-'): '下方(-Z)',
    }
    direction = direction_map.get((dominant_idx, dominant_sign), '未知')
    
    return {
        'cam_forward_in_lidar': cam_forward_in_lidar.tolist(),
        'dominant_axis': f'{dominant_sign}{dominant_axis}',
        'direction': direction,
    }


def validate_sequence(dataset_root, sequence, output_base_dir):
    """验证单个序列的投影效果"""
    dataset_root = Path(dataset_root)
    seq_dir = dataset_root / 'sequences' / sequence
    image_dir = seq_dir / 'image_2'
    
    if not image_dir.exists():
        print(f"  ❌ 序列 {sequence}: 图像目录不存在")
        return None
    
    images = sorted(image_dir.glob('*.png'))
    num_frames = len(images)
    
    if num_frames == 0:
        print(f"  ❌ 序列 {sequence}: 没有图像文件")
        return None
    
    print(f"\n{'='*80}")
    print(f"验证序列 {sequence} ({num_frames} 帧)")
    print(f"{'='*80}")
    
    seq_output_dir = output_base_dir / f'sequence_{sequence}'
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    
    calib_file = seq_dir / 'calib.txt'
    calib = load_calib(calib_file) if calib_file.exists() else {}
    
    cam_info = analyze_camera_direction(calib) if 'Tr' in calib else None
    if cam_info:
        print(f"  相机朝向(LiDAR系): {cam_info['direction']} ({cam_info['dominant_axis']})")
    
    sample_frames = get_sample_frames(num_frames, num_samples=10)
    print(f"采样帧: {sample_frames}")
    print(f"输出目录: {seq_output_dir}")
    
    results = []
    zero_visible_count = 0
    for frame_idx in sample_frames:
        output_file = seq_output_dir / f'frame_{frame_idx:06d}.png'
        
        print(f"  处理帧 {frame_idx:06d}...", end=' ')
        
        result = visualize_projection(dataset_root, sequence, frame_idx, output_file)
        
        if result:
            print(f"✓ ({result['visible_points']}/{result['total_points']} 点, {result['visible_ratio']*100:.1f}%)")
            results.append(result)
        else:
            zero_visible_count += 1
            print("跳过 (0 可见点)")
    
    stats = {
        'sequence': sequence,
        'num_frames': num_frames,
        'num_samples': len(results),
        'num_attempted': len(sample_frames),
        'num_zero_visible': zero_visible_count,
        'sample_frames': sample_frames,
        'results': results,
        'camera_direction': cam_info,
    }
    
    if results:
        stats['summary'] = {
            'avg_visible_ratio': sum(r['visible_ratio'] for r in results) / len(results),
            'avg_depth': sum(r['depth_mean'] for r in results) / len(results),
            'min_depth': min(r['depth_min'] for r in results),
            'max_depth': max(r['depth_max'] for r in results),
            'status': 'ok',
        }
        print(f"\n  ✅ 序列 {sequence} 验证完成: {len(results)}/{len(sample_frames)} 帧成功")
        print(f"  平均可见率: {stats['summary']['avg_visible_ratio']*100:.1f}%")
    else:
        stats['summary'] = {
            'avg_visible_ratio': 0.0,
            'status': 'no_visible_points',
            'reason': f"相机朝向LiDAR系{cam_info['direction']}，与前向点云不匹配" if cam_info else "所有采样帧均无可见点",
        }
        print(f"\n  ⚠️  序列 {sequence}: 所有{len(sample_frames)}帧均无可见点投影")
        if cam_info:
            print(f"     原因: 相机朝向为{cam_info['direction']}，与前向LiDAR点云不在同一方向")
    
    stats_file = seq_output_dir / 'statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return stats


def generate_overview_report(all_results, output_dir, dataset_root=""):
    """生成总览报告"""
    report_file = output_dir / 'PROJECTION_VALIDATION_REPORT.md'
    
    ok_results = [r for r in all_results if r['summary'].get('status') == 'ok']
    fail_results = [r for r in all_results if r['summary'].get('status') != 'ok']
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 点云投影验证报告\n\n")
        f.write(f"**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 总体概况\n\n")
        f.write(f"- 验证序列数: {len(all_results)} (成功: {len(ok_results)}, 无可见点: {len(fail_results)})\n")
        total_samples = sum(r['num_samples'] for r in all_results)
        f.write(f"- 总采样帧数: {total_samples}\n")
        f.write(f"- 每序列采样: 10 帧 (均匀分布)\n\n")
        
        f.write("---\n\n")
        f.write("## 各序列投影效果\n\n")
        f.write("| 序列 | 总帧数 | 采样数 | 平均可见率 | 深度范围 | 相机朝向 | 状态 |\n")
        f.write("|------|--------|--------|-----------|----------|----------|------|\n")
        
        for result in sorted(all_results, key=lambda x: x['sequence']):
            seq = result['sequence']
            num_frames = result['num_frames']
            cam_dir = result.get('camera_direction', {})
            cam_label = cam_dir.get('direction', '未知') if cam_dir else '未知'
            
            if result['summary'].get('status') == 'ok':
                num_samples = result['num_samples']
                avg_ratio = result['summary']['avg_visible_ratio'] * 100
                min_depth = result['summary']['min_depth']
                max_depth = result['summary']['max_depth']
                f.write(f"| {seq} | {num_frames:,} | {num_samples} | {avg_ratio:.1f}% | "
                       f"{min_depth:.1f}-{max_depth:.1f}m | {cam_label} | ✅ |\n")
            else:
                reason = result['summary'].get('reason', '无可见点')
                f.write(f"| {seq} | {num_frames:,} | 0 | 0.0% | N/A | {cam_label} | ⚠️ |\n")
        
        if fail_results:
            f.write("\n**⚠️ 无可见点序列说明**:\n\n")
            for result in fail_results:
                reason = result['summary'].get('reason', '所有采样帧均无可见点')
                f.write(f"- **序列 {result['sequence']}**: {reason}\n")
            f.write("\n")
        
        f.write("\n---\n\n")
        f.write("## 详细结果\n\n")
        
        for result in sorted(all_results, key=lambda x: x['sequence']):
            seq = result['sequence']
            f.write(f"### 序列 {seq}\n\n")
            f.write(f"- **总帧数**: {result['num_frames']:,}\n")
            
            cam_dir = result.get('camera_direction', {})
            if cam_dir:
                f.write(f"- **相机朝向**: {cam_dir.get('direction', '未知')} (LiDAR系 {cam_dir.get('dominant_axis', '')})\n")
            
            if result['summary'].get('status') == 'ok':
                f.write(f"- **采样帧**: {result['sample_frames']}\n")
                f.write(f"- **平均可见率**: {result['summary']['avg_visible_ratio']*100:.1f}%\n")
                f.write(f"- **深度范围**: {result['summary']['min_depth']:.1f}m - {result['summary']['max_depth']:.1f}m\n")
                f.write(f"- **图像位置**: `sequence_{seq}/`\n\n")
                
                f.write("| 帧 | 总点数 | 可见点 | 可见率 | 深度范围 | 图像 |\n")
                f.write("|----|--------|--------|--------|----------|------|\n")
                
                for r in result['results']:
                    frame = r['frame']
                    total = r['total_points']
                    visible = r['visible_points']
                    ratio = r['visible_ratio'] * 100
                    depth_min = r['depth_min']
                    depth_max = r['depth_max']
                    f.write(f"| {frame:06d} | {total:,} | {visible:,} | {ratio:.1f}% | "
                           f"{depth_min:.1f}-{depth_max:.1f}m | `frame_{frame:06d}.png` |\n")
            else:
                reason = result['summary'].get('reason', '所有采样帧均无可见点')
                f.write(f"- **状态**: ⚠️ 无可见投影点\n")
                f.write(f"- **原因**: {reason}\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 数据来源确认\n\n")
        f.write("✅ **所有投影使用的数据均来自**:\n\n")
        f.write("- **图像**: `sequences/{seq_id}/image_2/{frame:06d}.png`\n")
        f.write("- **点云**: `sequences/{seq_id}/velodyne/{frame:06d}.bin`\n")
        f.write("- **标定**: `sequences/{seq_id}/calib.txt`\n\n")
        if dataset_root:
            f.write(f"数据源位置: `{dataset_root}`\n\n")
    
    print(f"\n✅ 总览报告已生成: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='全面的点云投影验证')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--sequences', type=str, nargs='+',
                       help='指定要验证的序列 (默认: 全部)')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取要验证的序列
    sequences_dir = dataset_root / 'sequences'
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    
    print(f"\n{'='*80}")
    print(f"📊 全面点云投影验证")
    print(f"{'='*80}")
    print(f"数据集: {dataset_root}")
    print(f"序列数: {len(sequences)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")
    
    all_results = []
    for seq in sequences:
        result = validate_sequence(dataset_root, seq, output_dir)
        if result:
            all_results.append(result)
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset_root': str(dataset_root),
            'total_sequences': len(all_results),
            'sequences': all_results
        }, f, indent=2, ensure_ascii=False)
    
    generate_overview_report(all_results, output_dir, dataset_root=str(dataset_root))
    
    print(f"\n{'='*80}")
    print(f"✅ 验证完成!")
    print(f"{'='*80}")
    print(f"验证序列: {len(all_results)}/{len(sequences)}")
    print(f"输出目录: {output_dir}")
    print(f"查看报告: {output_dir}/PROJECTION_VALIDATION_REPORT.md")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
