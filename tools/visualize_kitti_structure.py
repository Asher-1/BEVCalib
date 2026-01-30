#!/usr/bin/env python3
"""
KITTI-Odometry 数据集结构可视化工具

用法:
    python visualize_kitti_structure.py /path/to/kitti-odometry
    python visualize_kitti_structure.py /path/to/kitti-odometry --sequence 00
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def analyze_sequence(dataset_root: str, sequence: str):
    """分析单个序列的结构"""
    seq_path = Path(dataset_root) / 'sequences' / sequence
    
    if not seq_path.exists():
        print(f"❌ 序列不存在: {seq_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"序列 {sequence} 分析")
    print(f"{'='*60}")
    print(f"路径: {seq_path}\n")
    
    info = {
        'sequence': sequence,
        'path': str(seq_path),
        'has_image_2': False,
        'has_image_3': False,
        'has_velodyne': False,
        'has_calib': False,
        'has_times': False,
        'image_count': 0,
        'velodyne_count': 0,
    }
    
    # 检查各个目录和文件
    image_2_dir = seq_path / 'image_2'
    if image_2_dir.exists():
        info['has_image_2'] = True
        info['image_count'] = len(list(image_2_dir.glob('*.png')))
        print(f"✓ image_2/     : {info['image_count']} 张图像")
        
        # 检查图像尺寸
        if info['image_count'] > 0:
            from PIL import Image
            first_img = list(image_2_dir.glob('*.png'))[0]
            img = Image.open(first_img)
            print(f"  └─ 图像尺寸   : {img.size[0]} × {img.size[1]}")
    else:
        print(f"✗ image_2/     : 不存在")
    
    image_3_dir = seq_path / 'image_3'
    if image_3_dir.exists():
        info['has_image_3'] = True
        count = len(list(image_3_dir.glob('*.png')))
        print(f"✓ image_3/     : {count} 张图像")
    else:
        print(f"✗ image_3/     : 不存在")
    
    velodyne_dir = seq_path / 'velodyne'
    if velodyne_dir.exists():
        info['has_velodyne'] = True
        info['velodyne_count'] = len(list(velodyne_dir.glob('*.bin')))
        print(f"✓ velodyne/    : {info['velodyne_count']} 个点云文件")
        
        # 检查点云统计
        if info['velodyne_count'] > 0:
            first_bin = list(velodyne_dir.glob('*.bin'))[0]
            points = np.fromfile(str(first_bin), dtype=np.float32).reshape(-1, 4)
            print(f"  └─ 点数范围   : {points.shape[0]} 点 (第一帧)")
            print(f"     X 范围    : [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] 米")
            print(f"     Y 范围    : [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] 米")
            print(f"     Z 范围    : [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] 米")
    else:
        print(f"✗ velodyne/    : 不存在")
    
    calib_file = seq_path / 'calib.txt'
    if calib_file.exists():
        info['has_calib'] = True
        print(f"✓ calib.txt    : 存在")
        
        # 解析标定文件
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    try:
                        calib[key.strip()] = [float(x) for x in value.split()]
                    except ValueError:
                        # 跳过非数值行（如 camera_model: pinhole）
                        continue
        
        print(f"  └─ 投影矩阵P2:")
        P2 = np.array(calib['P2']).reshape(3, 4)
        print(f"     fx = {P2[0, 0]:.2f}, fy = {P2[1, 1]:.2f}")
        print(f"     cx = {P2[0, 2]:.2f}, cy = {P2[1, 2]:.2f}")
        
        print(f"  └─ Tr (Velo→Cam0):")
        Tr = np.array(calib['Tr']).reshape(3, 4)
        print(f"     平移: [{Tr[0, 3]:.3f}, {Tr[1, 3]:.3f}, {Tr[2, 3]:.3f}]")
    else:
        print(f"✗ calib.txt    : 不存在")
    
    times_file = seq_path / 'times.txt'
    if times_file.exists():
        info['has_times'] = True
        with open(times_file, 'r') as f:
            times = [float(line.strip()) for line in f]
        print(f"✓ times.txt    : {len(times)} 个时间戳")
        if len(times) > 1:
            print(f"  └─ 时长       : {times[-1]:.1f} 秒")
            print(f"     平均FPS    : {len(times) / times[-1]:.1f}")
    else:
        print(f"✗ times.txt    : 不存在")
    
    # 检查位姿文件
    pose_file = Path(dataset_root) / 'poses' / f'{sequence}.txt'
    if pose_file.exists():
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        print(f"✓ 位姿文件     : {len(poses)} 个位姿")
        info['has_poses'] = True
    else:
        print(f"✗ 位姿文件     : 不存在（测试集无位姿）")
        info['has_poses'] = False
    
    # 数据完整性检查
    print(f"\n数据完整性:")
    if info['has_image_2'] and info['has_velodyne'] and info['has_calib']:
        if info['image_count'] == info['velodyne_count']:
            print(f"✓ 图像与点云数量匹配")
        else:
            print(f"⚠ 图像({info['image_count']})与点云({info['velodyne_count']})数量不匹配")
        
        if info['has_times'] and len(times) == info['image_count']:
            print(f"✓ 时间戳数量匹配")
        
        print(f"✓ 数据集完整，可用于训练/测试")
    else:
        print(f"✗ 数据集不完整")
    
    return info


def analyze_dataset(dataset_root: str):
    """分析整个数据集"""
    root = Path(dataset_root)
    
    print(f"\n{'='*60}")
    print(f"KITTI-Odometry 数据集分析")
    print(f"{'='*60}")
    print(f"数据集路径: {dataset_root}\n")
    
    # 检查主目录结构
    sequences_dir = root / 'sequences'
    poses_dir = root / 'poses'
    
    if not sequences_dir.exists():
        print(f"❌ 未找到 sequences 目录")
        return
    
    print(f"✓ sequences/   : 存在")
    
    if poses_dir.exists():
        pose_files = list(poses_dir.glob('*.txt'))
        print(f"✓ poses/       : {len(pose_files)} 个位姿文件")
    else:
        print(f"⚠ poses/       : 不存在")
    
    # 列出所有序列
    sequences = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    
    print(f"\n找到 {len(sequences)} 个序列:")
    
    train_sequences = []
    test_sequences = []
    
    for seq in sequences:
        seq_info = analyze_sequence_brief(dataset_root, seq)
        
        # 判断训练/测试集
        pose_file = poses_dir / f'{seq}.txt'
        if pose_file.exists():
            train_sequences.append(seq)
            seq_type = "训练"
        else:
            test_sequences.append(seq)
            seq_type = "测试"
        
        status = "✓" if seq_info['complete'] else "✗"
        print(f"  {status} {seq}: {seq_info['frame_count']:4d} 帧 | {seq_type}集")
    
    # 统计
    print(f"\n{'='*60}")
    print(f"数据集统计:")
    print(f"{'='*60}")
    print(f"训练集: {len(train_sequences)} 个序列 ({', '.join(train_sequences)})")
    print(f"测试集: {len(test_sequences)} 个序列 ({', '.join(test_sequences)})")
    
    total_frames = sum([analyze_sequence_brief(dataset_root, seq)['frame_count'] 
                       for seq in sequences])
    print(f"总帧数: {total_frames:,}")
    
    print(f"\nBEVCalib 使用的数据:")
    print(f"  - image_2/    (左相机图像)")
    print(f"  - velodyne/   (点云数据)")
    print(f"  - calib.txt   (标定参数: P2, Tr)")


def analyze_sequence_brief(dataset_root: str, sequence: str):
    """简要分析序列（不打印详细信息）"""
    seq_path = Path(dataset_root) / 'sequences' / sequence
    
    info = {
        'sequence': sequence,
        'frame_count': 0,
        'complete': False
    }
    
    image_2_dir = seq_path / 'image_2'
    velodyne_dir = seq_path / 'velodyne'
    calib_file = seq_path / 'calib.txt'
    
    if image_2_dir.exists():
        info['frame_count'] = len(list(image_2_dir.glob('*.png')))
    
    info['complete'] = (
        image_2_dir.exists() and 
        velodyne_dir.exists() and 
        calib_file.exists()
    )
    
    return info


def main():
    parser = argparse.ArgumentParser(
        description='KITTI-Odometry 数据集结构可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 分析整个数据集
    python visualize_kitti_structure.py /path/to/kitti-odometry
    
    # 分析特定序列
    python visualize_kitti_structure.py /path/to/kitti-odometry --sequence 00
    
    # 分析多个序列
    python visualize_kitti_structure.py /path/to/kitti-odometry --sequence 00 01 02
        """
    )
    parser.add_argument('dataset_root', help='KITTI-Odometry 数据集根目录')
    parser.add_argument('--sequence', '-s', nargs='+', help='指定序列（可选）')
    
    args = parser.parse_args()
    
    if not Path(args.dataset_root).exists():
        print(f"❌ 数据集路径不存在: {args.dataset_root}")
        sys.exit(1)
    
    if args.sequence:
        # 分析指定序列
        for seq in args.sequence:
            analyze_sequence(args.dataset_root, seq)
    else:
        # 分析整个数据集
        analyze_dataset(args.dataset_root)
    
    print(f"\n{'='*60}")
    print(f"📖 详细文档: KITTI_ODOMETRY_STRUCTURE.md")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
