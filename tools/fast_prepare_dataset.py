#!/usr/bin/env python3
"""快速数据集准备脚本 - 针对B26A数据集优化

特点：
1. 直接处理，不使用复杂的并行逻辑
2. 跳过去畸变（原始点云已在世界坐标系，需要转换回传感器坐标系）
3. 优化内存使用
"""

import os
import sys
import argparse
import numpy as np
import cv2
import struct
import yaml
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import gc

def load_config(config_dir: str, camera_name: str):
    """加载配置文件"""
    # 加载相机配置
    camera_cfg = Path(config_dir) / 'cameras.cfg'
    with open(camera_cfg, 'r') as f:
        cameras = yaml.safe_load(f)
    
    camera_config = cameras.get(camera_name)
    if not camera_config:
        raise ValueError(f"Camera {camera_name} not found")
    
    # 加载外参
    extrinsics_file = Path(config_dir) / 'extrinsics.yaml'
    with open(extrinsics_file, 'r') as f:
        extrinsics = yaml.safe_load(f)
    
    # 获取相机外参 (Camera → Sensing)
    camera_ext = extrinsics.get(camera_name, {})
    
    return camera_config, camera_ext

def extract_string_msg(rawdata):
    """提取String消息"""
    try:
        if len(rawdata) < 4:
            return None
        length = struct.unpack('<I', rawdata[:4])[0]
        if 0 < length < len(rawdata):
            return rawdata[4:4+length]
        return rawdata
    except:
        return rawdata

def decode_image(data):
    """解码图像"""
    jpeg_start = data.find(b'\xff\xd8')
    if jpeg_start != -1:
        jpeg_end = data.rfind(b'\xff\xd9')
        if jpeg_end > jpeg_start:
            img_array = np.frombuffer(data[jpeg_start:jpeg_end+2], np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
    return None

def extract_header_timestamp(data):
    """从protobuf数据中提取header时间戳"""
    try:
        # 跳过DeepRoute header
        if data[:4] == b'$$$$':
            data = data[8:]
        
        # 查找timestamp字段 (field 2, wire type 1 = fixed64)
        i = 0
        while i < len(data) - 9:
            tag = data[i]
            field_number = tag >> 3
            wire_type = tag & 0x7
            
            if wire_type == 1 and field_number == 2:  # Fixed64
                val = struct.unpack_from('<q', data, i+1)[0]
                if 1.5e15 < val < 2e15:  # 合理的微秒时间戳范围
                    return val / 1e6  # 转换为秒
            i += 1
    except:
        pass
    return None

def parse_pointcloud(data):
    """解析点云数据"""
    try:
        # 跳过DeepRoute header
        if data[:4] == b'$$$$':
            data = data[8:]
        
        # 查找points字段 (通常是field 4或5)
        i = 0
        while i < len(data) - 100:
            tag = data[i]
            field_number = tag >> 3
            wire_type = tag & 0x7
            
            if wire_type == 2:  # Length-delimited
                i += 1
                # 读取长度
                length = 0
                shift = 0
                while i < len(data):
                    byte = data[i]
                    i += 1
                    length |= (byte & 0x7F) << shift
                    if (byte & 0x80) == 0:
                        break
                    shift += 7
                
                # 检查是否是点云数据 (大块浮点数据)
                if length > 1000 and i + length <= len(data):
                    payload = data[i:i+length]
                    
                    # 尝试解析为float32数组
                    if len(payload) % 4 == 0:
                        arr = np.frombuffer(payload, dtype=np.float32)
                        
                        # 检查是否可以reshape为点云
                        for cols in [5, 4, 3]:
                            if len(arr) % cols == 0:
                                points = arr.reshape(-1, cols)
                                if points.shape[0] > 100:
                                    # 验证是否是合理的点云数据
                                    if np.abs(points[:, :3]).max() < 1000:
                                        return points
                
                i += length
            else:
                i += 1
    except:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description='快速数据集准备')
    parser.add_argument('--bag_dir', required=True, help='Bag文件目录')
    parser.add_argument('--config_dir', required=True, help='配置文件目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--camera_name', default='traffic_2', help='相机名称')
    parser.add_argument('--target_fps', type=float, default=10.0, help='目标帧率')
    args = parser.parse_args()
    
    print("=" * 60)
    print("快速数据集准备工具")
    print("=" * 60)
    
    # 加载配置
    camera_config, camera_ext = load_config(args.config_dir, args.camera_name)
    print(f"相机: {args.camera_name}")
    print(f"图像尺寸: {camera_config['resolution']['width']}x{camera_config['resolution']['height']}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    seq_dir = output_dir / 'sequences' / '00'
    image_dir = seq_dir / 'image_2'
    velodyne_dir = seq_dir / 'velodyne'
    image_dir.mkdir(parents=True, exist_ok=True)
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找bag文件
    bag_dir = Path(args.bag_dir)
    
    # 分类bag文件
    heavy_bags = sorted(bag_dir.glob('**/Heavy_Topic_Group/*.bag'))
    medium_bags = sorted(bag_dir.glob('**/Medium_Topic_Group/*.bag'))
    light_bags = sorted(bag_dir.glob('**/Light_Topic_Group/*.bag'))
    
    print(f"\n找到bag文件:")
    print(f"  Heavy (图像): {len(heavy_bags)}")
    print(f"  Medium (点云): {len(medium_bags)}")
    print(f"  Light (位姿): {len(light_bags)}")
    
    # 确定topic名称
    image_topic = f'/sensors/camera/{args.camera_name}_raw_data/compressed_proto'
    lidar_topic = '/sensing/lidar/combined_proto'
    pose_topic = '/localization/pose'
    
    from rosbags.rosbag1 import Reader
    
    # 提取数据
    images = []  # (timestamp, image_path)
    pointclouds = []  # (timestamp, points)
    poses = []  # (timestamp, R, t)
    
    # 1. 提取图像
    print(f"\n提取图像...")
    img_count = 0
    for bag_file in tqdm(heavy_bags, desc="处理Heavy bags"):
        try:
            with Reader(str(bag_file)) as reader:
                if image_topic not in reader.topics:
                    continue
                
                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic == image_topic:
                        data = extract_string_msg(rawdata)
                        if data:
                            ts = extract_header_timestamp(data) or (timestamp / 1e9)
                            image = decode_image(data)
                            if image is not None:
                                # 保存图像
                                img_path = image_dir / f"{img_count:06d}.png"
                                cv2.imwrite(str(img_path), image)
                                images.append((ts, str(img_path)))
                                img_count += 1
                                del image
        except Exception as e:
            print(f"  错误: {bag_file.name}: {e}")
        gc.collect()
    
    print(f"  提取了 {len(images)} 张图像")
    
    # 2. 提取点云
    print(f"\n提取点云...")
    pc_count = 0
    for bag_file in tqdm(medium_bags, desc="处理Medium bags"):
        try:
            with Reader(str(bag_file)) as reader:
                if lidar_topic not in reader.topics:
                    continue
                
                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic == lidar_topic:
                        data = extract_string_msg(rawdata)
                        if data:
                            ts = extract_header_timestamp(data) or (timestamp / 1e9)
                            points = parse_pointcloud(data)
                            if points is not None and points.shape[0] > 100:
                                pointclouds.append((ts, points))
                                pc_count += 1
        except Exception as e:
            print(f"  错误: {bag_file.name}: {e}")
        gc.collect()
    
    print(f"  提取了 {len(pointclouds)} 帧点云")
    
    # 3. 提取位姿
    print(f"\n提取位姿...")
    for bag_file in tqdm(light_bags, desc="处理Light bags"):
        try:
            with Reader(str(bag_file)) as reader:
                if pose_topic not in reader.topics:
                    continue
                
                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic == pose_topic:
                        data = extract_string_msg(rawdata)
                        if data:
                            # 简化的位姿解析
                            ts = timestamp / 1e9
                            # 尝试解析位姿...
                            # 这里简化处理，假设位姿数据格式已知
                            pass
        except Exception as e:
            pass
    
    print(f"  提取了 {len(poses)} 个位姿")
    
    # 4. 同步数据
    print(f"\n同步数据...")
    images.sort(key=lambda x: x[0])
    pointclouds.sort(key=lambda x: x[0])
    
    # 简单的时间戳匹配
    synced_pairs = []
    max_time_diff = 0.055  # 55ms
    
    pc_idx = 0
    for img_ts, img_path in images:
        # 找最近的点云
        while pc_idx < len(pointclouds) - 1:
            if pointclouds[pc_idx + 1][0] <= img_ts:
                pc_idx += 1
            else:
                break
        
        if pc_idx < len(pointclouds):
            pc_ts, points = pointclouds[pc_idx]
            if abs(pc_ts - img_ts) < max_time_diff:
                synced_pairs.append((img_path, points, img_ts))
    
    print(f"  同步了 {len(synced_pairs)} 对数据")
    
    # 5. 降采样到目标帧率
    if args.target_fps > 0 and len(synced_pairs) > 0:
        target_interval = 1.0 / args.target_fps
        filtered_pairs = [synced_pairs[0]]
        last_ts = synced_pairs[0][2]
        
        for pair in synced_pairs[1:]:
            if pair[2] - last_ts >= target_interval:
                filtered_pairs.append(pair)
                last_ts = pair[2]
        
        synced_pairs = filtered_pairs
        print(f"  降采样后: {len(synced_pairs)} 帧 ({args.target_fps} fps)")
    
    # 6. 保存最终数据
    print(f"\n保存数据...")
    times = []
    
    for idx, (img_path, points, ts) in enumerate(tqdm(synced_pairs, desc="保存")):
        # 重命名图像
        new_img_path = image_dir / f"{idx:06d}.png"
        if Path(img_path) != new_img_path:
            os.rename(img_path, new_img_path)
        
        # 保存点云 (只保存前4列: x, y, z, intensity)
        pc_path = velodyne_dir / f"{idx:06d}.bin"
        points[:, :4].astype(np.float32).tofile(str(pc_path))
        
        times.append(ts)
    
    # 保存times.txt
    times_file = seq_dir / 'times.txt'
    with open(times_file, 'w') as f:
        for ts in times:
            f.write(f"{ts:.6f}\n")
    
    # 保存calib.txt
    calib_file = seq_dir / 'calib.txt'
    
    # 构建内参矩阵
    intrinsic = camera_config['intrinsic']
    K = np.array([
        [intrinsic['fx'], 0, intrinsic['cx'], 0],
        [0, intrinsic['fy'], intrinsic['cy'], 0],
        [0, 0, 1, 0]
    ])
    
    # 构建外参矩阵 (Sensing → Camera)
    pos = camera_ext.get('position', [0, 0, 0])
    quat = camera_ext.get('orientation', [0, 0, 0, 1])
    r = R.from_quat(quat)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = pos
    T_sensing_to_camera = np.linalg.inv(T)
    
    with open(calib_file, 'w') as f:
        for i in range(4):
            f.write(f"P{i}: " + " ".join([f"{v:.12e}" for v in K.flatten()]) + "\n")
        f.write("Tr: " + " ".join([f"{v:.12e}" for v in T_sensing_to_camera[:3].flatten()]) + "\n")
        
        # 畸变系数
        dist = camera_config.get('distortion', {})
        D = [dist.get('k1', 0), dist.get('k2', 0), dist.get('p1', 0), dist.get('p2', 0), dist.get('k3', 0)]
        f.write("D: " + " ".join([f"{v:.12e}" for v in D]) + "\n")
        f.write(f"camera_model: {dist.get('model_type', 'pinhole')}\n")
    
    # 清理多余的图像文件
    for img_file in image_dir.glob('*.png'):
        idx = int(img_file.stem)
        if idx >= len(synced_pairs):
            img_file.unlink()
    
    print(f"\n完成!")
    print(f"  输出目录: {seq_dir}")
    print(f"  图像: {len(synced_pairs)} 张")
    print(f"  点云: {len(synced_pairs)} 帧")

if __name__ == '__main__':
    main()
