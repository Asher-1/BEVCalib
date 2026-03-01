#!/usr/bin/env python3
"""
数据集统计摘要工具
快速显示数据集的关键信息和统计
"""

import argparse
from pathlib import Path
import numpy as np


def load_calib_tr(calib_file):
    """加载Tr矩阵"""
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('Tr:'):
                values = [float(x) for x in line.strip().split()[1:]]
                if len(values) == 12:
                    return np.array(values).reshape(3, 4)
    return None


def show_dataset_summary(dataset_root):
    """显示数据集摘要"""
    dataset_root = Path(dataset_root)
    sequences_dir = dataset_root / 'sequences'
    poses_dir = dataset_root / 'poses'
    
    if not sequences_dir.exists():
        print(f"❌ 数据集不存在: {dataset_root}")
        return
    
    sequences = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    
    print("\n" + "="*80)
    print("📊 BEVCalib 数据集统计摘要")
    print("="*80)
    print(f"数据集路径: {dataset_root}")
    print(f"序列数量: {len(sequences)}")
    print()
    
    # 统计信息
    total_frames = 0
    seq_info = []
    
    print("序列详情:")
    print("-" * 80)
    print(f"{'序列':<6} {'帧数':<8} {'图像':<6} {'点云':<6} {'Poses':<6} {'Tr矩阵':<8} {'状态':<6}")
    print("-" * 80)
    
    for seq in sequences:
        seq_dir = sequences_dir / seq
        image_dir = seq_dir / 'image_2'
        velodyne_dir = seq_dir / 'velodyne'
        calib_file = seq_dir / 'calib.txt'
        poses_file = poses_dir / f'{seq}.txt'
        
        # 统计帧数
        num_images = len(list(image_dir.glob('*.png'))) if image_dir.exists() else 0
        num_velodyne = len(list(velodyne_dir.glob('*.bin'))) if velodyne_dir.exists() else 0
        num_poses = 0
        if poses_file.exists():
            with open(poses_file) as f:
                num_poses = len(f.readlines())
        
        # 检查Tr矩阵
        tr_status = "❌"
        if calib_file.exists():
            tr = load_calib_tr(calib_file)
            if tr is not None:
                # 检查行列式
                R = tr[:3, :3]
                det = np.linalg.det(R)
                if 0.99 < det < 1.01:
                    tr_status = "✓"
        
        # 对齐性检查
        aligned = num_images == num_velodyne == num_poses
        status = "✅" if aligned and tr_status == "✓" else "⚠️"
        
        print(f"{seq:<6} {num_images:<8} {num_images:<6} {num_velodyne:<6} {num_poses:<6} {tr_status:<8} {status:<6}")
        
        total_frames += num_images
        seq_info.append({
            'seq': seq,
            'frames': num_images,
            'aligned': aligned,
            'tr_ok': tr_status == "✓"
        })
    
    print("-" * 80)
    print(f"总计   {total_frames:<8}")
    print("="*80)
    
    # 统计摘要
    all_aligned = all(info['aligned'] for info in seq_info)
    all_tr_ok = all(info['tr_ok'] for info in seq_info)
    
    print("\n验证摘要:")
    print(f"  - 总帧数: {total_frames:,}")
    print(f"  - 数据对齐: {'✅ 全部对齐' if all_aligned else '⚠️ 部分不对齐'}")
    print(f"  - Tr矩阵: {'✅ 全部正确' if all_tr_ok else '⚠️ 部分错误'}")
    
    if all_aligned and all_tr_ok:
        print("\n🎉 数据集状态良好，可以用于训练！")
    else:
        print("\n⚠️ 数据集存在问题，建议运行完整验证:")
        print("   python tools/validate_all_sequences.py \\")
        print(f"       --dataset_root {dataset_root} \\")
        print("       --output_dir validation_results/")
    
    # 规模分布
    print("\n序列规模分布:")
    frames_list = [info['frames'] for info in seq_info]
    print(f"  - 最大序列: {max(frames_list):,} 帧 (Seq {seq_info[frames_list.index(max(frames_list))]['seq']})")
    print(f"  - 最小序列: {min(frames_list):,} 帧 (Seq {seq_info[frames_list.index(min(frames_list))]['seq']})")
    print(f"  - 平均规模: {sum(frames_list)//len(frames_list):,} 帧")
    
    print("\n" + "="*80)
    print()


def main():
    parser = argparse.ArgumentParser(description='显示数据集统计摘要')
    parser.add_argument('dataset_root', type=str, help='数据集根目录')
    
    args = parser.parse_args()
    show_dataset_summary(args.dataset_root)


if __name__ == '__main__':
    main()
