#!/usr/bin/env python3
"""
批量处理 trips 目录下的所有行程数据
生成 KITTI-Odometry 格式的训练数据集

每个 trip 对应一个 sequence ID（00, 01, 02, ...）

注意：prepare_custom_dataset.py 已经实现了只读取主 lidar (frame_id: "atx_202") 的功能，
     因此这里直接使用原始配置目录即可。
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def find_bag_directory(trip_dir):
    """
    查找 trip 目录下的 bag 文件目录
    
    Args:
        trip_dir: trip 目录路径
        
    Returns:
        bag 目录路径
    """
    trip_dir = Path(trip_dir)
    
    # 查找 bags/unimportant 或 bags/important
    for bag_subdir in ['bags/unimportant', 'bags/important']:
        bag_dir = trip_dir / bag_subdir
        if bag_dir.exists():
            return bag_dir
    
    # 如果没找到，直接返回 bags 目录
    bag_dir = trip_dir / 'bags'
    if bag_dir.exists():
        return bag_dir
    
    raise ValueError(f"在 {trip_dir} 中未找到 bag 目录")


def process_trip(trip_dir, sequence_id, output_base_dir, log_file, camera_name='traffic_2', target_fps=10.0):
    """
    处理单个 trip 行程
    
    Args:
        trip_dir: trip 目录路径
        sequence_id: 序列ID（如 '00', '01'）
        output_base_dir: 输出基础目录
        log_file: 日志文件句柄
        camera_name: 相机名称
        target_fps: 目标帧率
    """
    trip_dir = Path(trip_dir)
    trip_name = trip_dir.name
    
    log_msg = f"\n{'='*80}\n"
    log_msg += f"处理 Trip: {trip_name}\n"
    log_msg += f"Sequence ID: {sequence_id}\n"
    log_msg += f"{'='*80}\n"
    print(log_msg)
    log_file.write(log_msg)
    log_file.flush()
    
    try:
        # 1. 查找 bag 目录
        bag_dir = find_bag_directory(trip_dir)
        log_msg = f"Bag 目录: {bag_dir}\n"
        print(log_msg)
        log_file.write(log_msg)
        log_file.flush()
        
        # 2. 查找配置目录
        config_dir = trip_dir / 'configs'
        if not config_dir.exists():
            raise ValueError(f"配置目录不存在: {config_dir}")
        
        log_msg = f"配置目录: {config_dir}\n"
        print(log_msg)
        log_file.write(log_msg)
        log_file.flush()
        
        # 3. 准备输出目录
        output_dir = Path(output_base_dir)
        
        # 4. 调用 prepare_custom_dataset.py
        # 注意：prepare_custom_dataset.py 已经内置了只读取主 lidar (frame_id: "atx_202") 的功能
        script_dir = Path(__file__).parent
        prepare_script = script_dir / 'prepare_custom_dataset.py'
        
        cmd = [
            'python', str(prepare_script),
            '--bag_dir', str(bag_dir),
            '--config_dir', str(config_dir),  # 直接使用原始配置目录
            '--output_dir', str(output_dir),
            '--camera_name', camera_name,
            '--target_fps', str(target_fps),
            '--sequence_id', sequence_id,
            '--num_workers', '32',
            '--batch_size', '800',
            '--save_debug_samples', '20',
        ]
        
        log_msg = f"\n执行命令:\n{' '.join(cmd)}\n\n"
        print(log_msg)
        log_file.write(log_msg)
        log_file.flush()
        
        # 执行命令并实时输出日志
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时读取输出
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"处理失败，返回码: {process.returncode}")
        
        log_msg = f"\n✓ Trip {trip_name} 处理成功\n"
        print(log_msg)
        log_file.write(log_msg)
        log_file.flush()
        
    except Exception as e:
        log_msg = f"\n✗ Trip {trip_name} 处理失败: {str(e)}\n"
        print(log_msg)
        log_file.write(log_msg)
        log_file.flush()
        raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理 trips 目录下的所有行程数据')
    parser.add_argument('--trips_dir', type=str, required=True,
                       help='trips 根目录（包含所有行程的目录）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录（生成 bevcalib_training_data）')
    parser.add_argument('--camera_name', type=str, default='traffic_2',
                       help='相机名称（默认: traffic_2）')
    parser.add_argument('--target_fps', type=float, default=10.0,
                       help='目标帧率（默认: 10.0）')
    parser.add_argument('--start_sequence', type=int, default=0,
                       help='起始 sequence ID（默认: 0）')
    
    args = parser.parse_args()
    
    trips_dir = Path(args.trips_dir)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件
    log_file_path = output_dir / f'batch_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    print(f"\n{'='*80}")
    print(f"批量处理 Trips 数据集")
    print(f"{'='*80}")
    print(f"\nTrips 目录: {trips_dir}")
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file_path}")
    print(f"相机名称: {args.camera_name}")
    print(f"目标帧率: {args.target_fps} fps")
    print(f"起始 Sequence ID: {args.start_sequence}")
    
    # 获取所有 trip 目录（按名称排序以保证一致性）
    trip_dirs = sorted([d for d in trips_dir.iterdir() if d.is_dir()])
    
    print(f"\n找到 {len(trip_dirs)} 个 trip 目录:")
    for i, trip_dir in enumerate(trip_dirs):
        seq_id = f"{args.start_sequence + i:02d}"
        print(f"  {seq_id}: {trip_dir.name}")
    
    # 打开日志文件
    with open(log_file_path, 'w') as log_file:
        start_time = datetime.now()
        log_file.write(f"批量处理开始时间: {start_time}\n")
        log_file.write(f"Trips 目录: {trips_dir}\n")
        log_file.write(f"输出目录: {output_dir}\n")
        log_file.write(f"相机名称: {args.camera_name}\n")
        log_file.write(f"目标帧率: {args.target_fps} fps\n")
        log_file.write(f"起始 Sequence ID: {args.start_sequence}\n")
        log_file.write(f"\n找到 {len(trip_dirs)} 个 trip 目录\n")
        log_file.flush()
        
        # 处理每个 trip
        success_count = 0
        failed_trips = []
        
        for i, trip_dir in enumerate(trip_dirs):
            seq_id = f"{args.start_sequence + i:02d}"
            
            try:
                process_trip(
                    trip_dir=trip_dir,
                    sequence_id=seq_id,
                    output_base_dir=output_dir,
                    log_file=log_file,
                    camera_name=args.camera_name,
                    target_fps=args.target_fps
                )
                success_count += 1
            except Exception as e:
                failed_trips.append((trip_dir.name, str(e)))
                print(f"\n跳过失败的 trip，继续处理下一个...\n")
        
        # 输出总结
        end_time = datetime.now()
        duration = end_time - start_time
        
        summary = f"\n{'='*80}\n"
        summary += f"批量处理完成\n"
        summary += f"{'='*80}\n"
        summary += f"开始时间: {start_time}\n"
        summary += f"结束时间: {end_time}\n"
        summary += f"总用时: {duration}\n"
        summary += f"\n处理结果:\n"
        summary += f"  总计: {len(trip_dirs)} 个 trips\n"
        summary += f"  成功: {success_count} 个\n"
        summary += f"  失败: {len(failed_trips)} 个\n"
        
        if failed_trips:
            summary += f"\n失败的 trips:\n"
            for trip_name, error in failed_trips:
                summary += f"  - {trip_name}: {error}\n"
        
        summary += f"\n输出目录: {output_dir}\n"
        summary += f"日志文件: {log_file_path}\n"
        
        print(summary)
        log_file.write(summary)
    
    print(f"\n完整日志已保存到: {log_file_path}")


if __name__ == '__main__':
    main()
