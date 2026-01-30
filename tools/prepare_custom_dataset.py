#!/usr/bin/env python3
"""
BEVCalib 自定义数据集准备脚本（流式处理版本）

从 rosbag 和配置文件提取数据，转换为 BEVCalib 训练格式。
采用流式处理，减少内存占用。

使用方法:
    python prepare_custom_dataset.py \\
        --bag_dir /path/to/bags \\
        --config_dir /path/to/config \\
        --output_dir /path/to/output \\
        --camera_name traffic_2 \\
        --target_fps 10.0
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp
import struct
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time
from datetime import timedelta


@dataclass
class ImageMetadata:
    """图像元数据（不包含图像数据）"""
    timestamp: float
    file_path: str


@dataclass
class PointCloudMetadata:
    """点云元数据（不包含点云数据）"""
    timestamp: float
    file_path: str


@dataclass
class PoseMetadata:
    """位姿元数据（Sensing系在World系中的位姿）
    
    注意：与C++对齐，存储的是 Sensing→World 变换
    - 含义：Sensing系在World系中的位姿
    - 作用：将Sensing系的点变换到World系
    
    C++参考：lidar_online_calibrator.cpp:849-852
        sensing_pose.second = vehicle_pose * iso_vehicle_sensing_;
        其中：vehicle_pose = Vehicle→World
              iso_vehicle_sensing_ = Sensing→Vehicle
              结果：sensing_pose = Sensing→World
    """
    timestamp: float
    position: np.ndarray  # (3,) xyz - Sensing在World系中的位置
    orientation: np.ndarray  # (4,) quaternion (x, y, z, w) - Sensing在World系中的姿态


class UndistortionUtils:
    """点云去畸变工具类"""
    
    @staticmethod
    def quat_to_matrix(q):
        """四元数转旋转矩阵 (x, y, z, w)"""
        x, y, z, w = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    @staticmethod
    def isometry_to_vec6(R_mat, t):
        """将旋转矩阵和平移向量转换为6D向量 [rx, ry, rz, tx, ty, tz]"""
        # 使用Rodrigues公式：R -> angle-axis
        r = R.from_matrix(R_mat).as_rotvec()
        return np.concatenate([r, t])
    
    @staticmethod
    def vec6_to_isometry(v):
        """将6D向量转换为旋转矩阵和平移向量"""
        r_vec = v[:3]
        t = v[3:6]
        R_mat = R.from_rotvec(r_vec).as_matrix()
        return R_mat, t
    
    @staticmethod
    def motion_interpolate(poses: List[PoseMetadata], timestamp: float):
        """位姿插值（严格参考C++的motion_interpolate实现）
        
        Args:
            poses: 位姿列表（按时间戳排序）
            timestamp: 目标时间戳（秒）
            
        Returns:
            (R, t): 旋转矩阵和平移向量，如果失败返回None
            
        Note:
            参考 math_utils.cpp:313-346
            C++版本只支持插值，不支持外推。时间戳超出范围直接返回false。
        """
        if not poses or len(poses) < 2:
            return None
        
        # 查找timestamp前后的两个位姿进行插值（严格按照C++逻辑）
        for i in range(len(poses) - 1):
            t1 = poses[i].timestamp
            t2 = poses[i + 1].timestamp
            
            # 只在范围内插值（对应C++的 if (t1 <= t && t2 >= t)）
            if t1 <= timestamp <= t2:
                # 计算插值系数
                alpha = (timestamp - t1) / (t2 - t1)
                break
        else:
            # 时间戳不在任何区间内，返回None（对应C++的return false）
            return None
        
        # 位姿1
        R1 = UndistortionUtils.quat_to_matrix(poses[i].orientation)
        t1_vec = poses[i].position
        
        # 位姿2
        R2 = UndistortionUtils.quat_to_matrix(poses[i + 1].orientation)
        t2_vec = poses[i + 1].position
        
        # 计算相对运动
        R_delta = R1.T @ R2
        t_delta = R1.T @ (t2_vec - t1_vec)
        
        # 转换为6D向量并插值/外推
        v_delta = UndistortionUtils.isometry_to_vec6(R_delta, t_delta)
        v_interp = v_delta * alpha
        
        # 转换回矩阵
        R_interp, t_interp = UndistortionUtils.vec6_to_isometry(v_interp)
        
        # 应用到pose1
        R_result = R1 @ R_interp
        t_result = t1_vec + R1 @ t_interp
        
        return R_result, t_result
    
    @staticmethod
    def motion_extrapolate(poses: List[PoseMetadata], timestamp: float):
        """位姿外推（用于时间戳超出范围的情况）
        
        Args:
            poses: 位姿列表（按时间戳排序）
            timestamp: 目标时间戳（秒）
            
        Returns:
            (R, t): 旋转矩阵和平移向量，如果失败返回None
            
        Note:
            当timestamp在pose范围外时，使用最近的两个pose进行外推
            外推使用与插值相同的数学公式，但alpha会超出[0,1]范围
        """
        if not poses or len(poses) < 2:
            return None
        
        # 确定外推方向和使用的pose对
        if timestamp < poses[0].timestamp:
            # 向前外推：使用前两个pose
            i = 0
            t1 = poses[0].timestamp
            t2 = poses[1].timestamp
        elif timestamp > poses[-1].timestamp:
            # 向后外推：使用后两个pose
            i = len(poses) - 2
            t1 = poses[i].timestamp
            t2 = poses[i + 1].timestamp
        else:
            # 在范围内，应该使用插值
            return None
        
        # 计算外推系数（会超出[0,1]范围）
        if t2 == t1:
            return None
        alpha = (timestamp - t1) / (t2 - t1)
        
        # 位姿1
        R1 = UndistortionUtils.quat_to_matrix(poses[i].orientation)
        t1_vec = poses[i].position
        
        # 位姿2
        R2 = UndistortionUtils.quat_to_matrix(poses[i + 1].orientation)
        t2_vec = poses[i + 1].position
        
        # 计算相对运动
        R_delta = R1.T @ R2
        t_delta = R1.T @ (t2_vec - t1_vec)
        
        # 转换为6D向量并外推
        v_delta = UndistortionUtils.isometry_to_vec6(R_delta, t_delta)
        v_extrap = v_delta * alpha  # alpha可以<0或>1
        
        # 转换回矩阵
        R_extrap, t_extrap = UndistortionUtils.vec6_to_isometry(v_extrap)
        
        # 应用到pose1
        R_result = R1 @ R_extrap
        t_result = t1_vec + R1 @ t_extrap
        
        return R_result, t_result
    
    @staticmethod
    def undistort_pointcloud(points_raw: np.ndarray,
                            cloud_timestamp: float,
                            target_timestamp: float,
                            poses: List[PoseMetadata],
                            debug: bool = False) -> np.ndarray:
        """点云去畸变（完全对齐C++实现）
        
        参考: ~/codetree/repo/calibration/modules/calib_utils/src/math_utils.cpp:169-252
        
        ✅ 关键假设（与C++一致）：
            1. 输入poses是 Sensing→World 变换（Sensing在World系中的位姿）
            2. LiDAR系 = Sensing系（iso_vehicle_lidar = Identity）
        
        🎯 去畸变原理：
            - 将点云从激光雷达扫描时刻(cloud_timestamp)转换到目标时刻(target_timestamp)
            - 目的：消除车辆运动造成的点云畸变，使点云与图像在空间上对齐
            - 输出的点云在逻辑上对应于target_timestamp时刻的LiDAR坐标系
        
        Args:
            points_raw: 原始点云 (N, 5): x, y, z, intensity, timestamp (LiDAR坐标系)
                       其中timestamp单位是2微秒
            cloud_timestamp: 点云扫描开始时刻（秒）
            target_timestamp: 目标时刻（通常是图像曝光时刻，秒）
            poses: GNSS位姿列表（Sensing→World，已转换！）
            debug: 是否打印调试信息
            
        Returns:
            去畸变后的点云 (N, 4): x, y, z, intensity
            ✅ 空间位置对应于target_timestamp时刻的LiDAR坐标系
            ✅ 与该时刻的图像完全对齐（时间和空间双重对齐）
            ✅ 不包含timestamp（所有点已统一到target_timestamp，原始时间戳已无意义）
            ✅ 符合KITTI标准格式
        
        C++参考：
            - math_utils.cpp:184-188: lidar_pose_data = vehicle_poses * iso_vehicle_lidar
            - math_utils.cpp:200: delta_stamp = max_inner_stamp * 2 (单位:2us)
            - math_utils.cpp:215: dt = timestamp * 2.0e-6 (单位:2us)
        """
        if points_raw.shape[1] < 5:
            # 没有timestamp，无法去畸变，直接返回前4列
            return points_raw[:, :4]
        
        if not poses or len(poses) == 0:
            # 没有位姿数据，无法去畸变
            return points_raw[:, :4]
        
        # 找到点云扫描的最大内部时间戳（单位：2微秒）
        # C++参考：math_utils.cpp:190-200
        max_inner_timestamp_2us = points_raw[:, 4].max()
        delta_time_us = max_inner_timestamp_2us * 2  # 转换为微秒
        end_timestamp = cloud_timestamp + delta_time_us * 1e-6  # 转换为秒
        
        # 插值获取三个关键时刻的位姿（Sensing→World变换）
        # C++参考：math_utils.cpp:198-205
        start_pose = UndistortionUtils.motion_interpolate(poses, cloud_timestamp)
        end_pose = UndistortionUtils.motion_interpolate(poses, end_timestamp)
        target_pose = UndistortionUtils.motion_interpolate(poses, target_timestamp)
        
        if start_pose is None or end_pose is None or target_pose is None:
            # 插值失败，返回原始点云（去掉timestamp）
            # C++参考：math_utils.cpp:247-250
            print(f"⚠️  去畸变失败：插值返回None")
            print(f"  cloud_ts={cloud_timestamp:.6f}, end_ts={end_timestamp:.6f}, target_ts={target_timestamp:.6f}")
            print(f"  start_pose={'有效' if start_pose else 'None'}, end_pose={'有效' if end_pose else 'None'}, target_pose={'有效' if target_pose else 'None'}")
            return points_raw[:, :4]
        
        R_start, t_start = start_pose  # Sensing→World（已转换！）
        R_end, t_end = end_pose
        R_target, t_target = target_pose
        
        # 🔍 DEBUG: 打印pose信息
        if debug:
            print(f"\n🔍 去畸变调试信息 (完全对齐C++):")
            print(f"  cloud_ts={cloud_timestamp:.6f}, target_ts={target_timestamp:.6f}, delta={end_timestamp-cloud_timestamp:.6f}s")
            print(f"  假设：LiDAR系 = Sensing系 (iso_vehicle_lidar = Identity)")
            print(f"  start_pose (Sensing→World): t=[{t_start[0]:.2f}, {t_start[1]:.2f}, {t_start[2]:.2f}]")
            print(f"  end_pose (Sensing→World): t=[{t_end[0]:.2f}, {t_end[1]:.2f}, {t_end[2]:.2f}]")
            print(f"  target_pose (Sensing→World): t=[{t_target[0]:.2f}, {t_target[1]:.2f}, {t_target[2]:.2f}]")
        
        R_lidar_start = R_start
        t_lidar_start = t_start
        R_lidar_end = R_end
        t_lidar_end = t_end
        R_lidar_target = R_target
        t_lidar_target = t_target
        
        # 计算运动增量（LiDAR坐标系）
        # ✅ 最终修复：去畸变变换（对齐C++逻辑）
        # 
        # C++中pose的含义：
        # - lidar_pose = (Vehicle→World) @ (LiDAR→Vehicle) = LiDAR→World
        # 
        # 去畸变流程：
        # 1. 点p在(start+dt)时刻的LiDAR系
        # 2. 变换到World系：P_world = LiDAR→World(start+dt) * p
        # 3. 变换到target时刻的LiDAR系：pu = World→LiDAR(target) * P_world
        #                              = World→LiDAR(target) * LiDAR→World(start+dt) * p
        # 
        # 其中：LiDAR→World(start+dt) ≈ LiDAR→World(start) * exp(v*dt)
        # 
        # C++实现：
        # - delta = inv(start) * end  （从start到end的运动）
        # - iso_target_start = inv(target) * start  （从start的LiDAR到target的LiDAR）
        # - delta2 = iso_target_start * delta2  （组合变换）
        # - pu = delta2 * p  （应用到点）
        
        # delta = inv(start) * end （从start到end的运动）
        R_delta = R_lidar_start.T @ R_lidar_end
        t_delta = R_lidar_start.T @ (t_lidar_end - t_lidar_start)
        v_full = UndistortionUtils.isometry_to_vec6(R_delta, t_delta)
        v_per_second = v_full / (delta_time_us * 1e-6)
        
        # iso_target_start = inv(target) * start （从start的LiDAR到target的LiDAR）
        R_target_start = R_lidar_target.T @ R_lidar_start  # ✅ 恢复：target^{-1} * start
        t_target_start = R_lidar_target.T @ (t_lidar_start - t_lidar_target)  # ✅ 恢复
        
        # 向量化处理所有点
        xyz = points_raw[:, :3]  # (N, 3) LiDAR坐标系
        intensity = points_raw[:, 3:4]  # (N, 1)
        ts_us = points_raw[:, 4]  # (N,)
        
        # 计算每个点的时间偏移（秒）
        dt = ts_us * 2.0e-6  # (N,) uint: 2us
        
        # 计算每个点的运动增量 v2 = v * dt
        v_points = v_per_second[np.newaxis, :] * dt[:, np.newaxis]  # (N, 6)
        r_vecs = v_points[:, :3]  # (N, 3)
        t_vecs = v_points[:, 3:6]  # (N, 3)
        
        # 批量计算delta2 (对应C++的Vec2Isometry)
        R_delta2 = R.from_rotvec(r_vecs).as_matrix()  # (N, 3, 3)
        
        # ✅ 最终修复：delta2 = iso_target_start * delta2（完全对齐C++）
        # 
        # 关键理解：delta2不是LiDAR系中的增量，而是pose的右乘增量
        # 即：pose(start+dt) = pose(start) * delta2
        # 其中pose表示LiDAR→World的变换
        # 
        # 所以最终变换：
        # delta2_final = iso_target_start * delta2
        #              = [inv(target) * start] * delta2
        #              = inv(target) * [start * delta2]
        #              = inv(target) * (start+dt)
        #              = World→LiDAR(target) * LiDAR→World(start+dt)
        # 
        # 应用到点：
        # pu = delta2_final * p
        #    = World→LiDAR(target) * LiDAR→World(start+dt) * p
        # 
        # 物理含义：点p从(start+dt)时刻的LiDAR系变换到target时刻的LiDAR系
        
        # 齐次变换矩阵乘法：iso_target_start * delta2
        R_combined = np.einsum('ij,njk->nik', R_target_start, R_delta2)  # (N, 3, 3)
        t_combined = np.einsum('ij,nj->ni', R_target_start, t_vecs) + t_target_start  # (N, 3)
        
        # 应用到点：pu = delta2 * p （对应C++第225行）
        xyz_undistorted = np.einsum('nij,nj->ni', R_combined, xyz) + t_combined  # (N, 3)
        
        # 拼接强度（不保留timestamp）
        # 注意：去畸变后所有点都在target_timestamp时刻，原始timestamp已无意义
        # 输出格式：(N, 4) = [x, y, z, intensity]（符合KITTI标准）
        points_undistorted = np.hstack([xyz_undistorted, intensity])  # (N, 4)
        
        # 🔍 DEBUG: 打印结果统计
        if debug:
            print(f"  Raw cloud:")
            print(f"    X: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}], mean={xyz[:, 0].mean():.2f}")
            print(f"    Y: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}], mean={xyz[:, 1].mean():.2f}")
            print(f"    Z: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}], mean={xyz[:, 2].mean():.2f}")
            print(f"  Undistorted cloud:")
            print(f"    X: [{xyz_undistorted[:, 0].min():.2f}, {xyz_undistorted[:, 0].max():.2f}], mean={xyz_undistorted[:, 0].mean():.2f}")
            print(f"    Y: [{xyz_undistorted[:, 1].min():.2f}, {xyz_undistorted[:, 1].max():.2f}], mean={xyz_undistorted[:, 1].mean():.2f}")
            print(f"    Z: [{xyz_undistorted[:, 2].min():.2f}, {xyz_undistorted[:, 2].max():.2f}], mean={xyz_undistorted[:, 2].mean():.2f}")
            diff = xyz_undistorted - xyz
            print(f"  Difference (mean ± std):")
            print(f"    X: {diff[:, 0].mean():.3f} ± {diff[:, 0].std():.3f}m")
            print(f"    Y: {diff[:, 1].mean():.3f} ± {diff[:, 1].std():.3f}m")
            print(f"    Z: {diff[:, 2].mean():.3f} ± {diff[:, 2].std():.3f}m\n")
        
        return points_undistorted.astype(np.float32)


class ConfigParser:
    """配置文件解析器"""
    
    @staticmethod
    def parse_cameras_cfg(filepath: str) -> Dict:
        """解析 cameras.cfg 文件"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        cameras = {}
        config_pattern = r'config\s*\{(.*?)\n\}'
        config_blocks = re.findall(config_pattern, content, re.DOTALL)
        
        for block in config_blocks:
            camera = ConfigParser._parse_camera_block(block)
            if camera:
                cameras[camera['camera_dev']] = camera
        
        return cameras
    
    @staticmethod
    def _parse_camera_block(block: str) -> Optional[Dict]:
        """解析单个相机配置块"""
        
        def get_value(pattern, text, default=None):
            match = re.search(pattern, text)
            return match.group(1).strip().strip('"') if match else default
        
        def get_float(pattern, text, default=0.0):
            val = get_value(pattern, text)
            return float(val) if val is not None else default
        
        def get_int(pattern, text, default=0):
            val = get_value(pattern, text)
            return int(val) if val is not None else default
        
        camera_dev = get_value(r'camera_dev:\s*"([^"]+)"', block)
        if not camera_dev:
            return None
        
        pos_x = get_float(r'position\s*\{[^}]*x:\s*([-\d.e]+)', block)
        pos_y = get_float(r'position\s*\{[^}]*y:\s*([-\d.e]+)', block)
        pos_z = get_float(r'position\s*\{[^}]*z:\s*([-\d.e]+)', block)
        
        ori_qx = get_float(r'orientation\s*\{[^}]*qx:\s*([-\d.e]+)', block)
        ori_qy = get_float(r'orientation\s*\{[^}]*qy:\s*([-\d.e]+)', block)
        ori_qz = get_float(r'orientation\s*\{[^}]*qz:\s*([-\d.e]+)', block)
        ori_qw = get_float(r'orientation\s*\{[^}]*qw:\s*([-\d.e]+)', block)
        
        intrinsic = {
            'img_width': get_int(r'img_width:\s*(\d+)', block, 1920),
            'img_height': get_int(r'img_height:\s*(\d+)', block, 1080),
            'f_x': get_float(r'f_x:\s*([-\d.e]+)', block),
            'f_y': get_float(r'f_y:\s*([-\d.e]+)', block),
            'o_x': get_float(r'o_x:\s*([-\d.e]+)', block),
            'o_y': get_float(r'o_y:\s*([-\d.e]+)', block),
        }
        
        # 解析畸变系数（支持多种格式）
        # 实际配置格式都是带下划线的：k_1, k_2, p_1, p_2
        # 根据model_type区分：
        # - PINHOLE: k_1, k_2, p_1, p_2 (针孔相机，有径向和切向畸变)
        # - KANNALA_BRANDT: k_1, k_2, k_3, k_4 (鱼眼相机，只有径向畸变)
        
        # 提取模型类型
        model_type_raw = get_value(r'model_type:\s*(\w+)', block, 'PINHOLE')
        
        # 提取畸变系数（带下划线）
        k_1 = get_float(r'k_1:\s*([-\d.e]+)', block, None)
        k_2 = get_float(r'k_2:\s*([-\d.e]+)', block, None)
        p_1 = get_float(r'p_1:\s*([-\d.e]+)', block, None)
        p_2 = get_float(r'p_2:\s*([-\d.e]+)', block, None)
        k_3 = get_float(r'k_3:\s*([-\d.e]+)', block, None)
        k_4 = get_float(r'k_4:\s*([-\d.e]+)', block, None)
        
        if model_type_raw == 'KANNALA_BRANDT':
            # 鱼眼模型：只有径向畸变 k_1, k_2, k_3, k_4
            distortion = {
                'k1': k_1 if k_1 is not None else 0.0,
                'k2': k_2 if k_2 is not None else 0.0,
                'k3': k_3 if k_3 is not None else 0.0,
                'k4': k_4 if k_4 is not None else 0.0,
                'model_type': 'fisheye',
            }
        else:
            # PINHOLE针孔模型：径向畸变 k_1, k_2 + 切向畸变 p_1, p_2
            distortion = {
                'k1': k_1 if k_1 is not None else 0.0,
                'k2': k_2 if k_2 is not None else 0.0,
                'p1': p_1 if p_1 is not None else 0.0,
                'p2': p_2 if p_2 is not None else 0.0,
                'k3': 0.0,  # PINHOLE通常只用k1, k2
                'model_type': 'pinhole',
            }
        
        return {
            'camera_dev': camera_dev,
            'position': np.array([pos_x, pos_y, pos_z]),
            'orientation': np.array([ori_qx, ori_qy, ori_qz, ori_qw]),
            'intrinsic': intrinsic,
            'distortion': distortion
        }
    
    @staticmethod
    def parse_lidars_cfg(filepath: str) -> Dict:
        """解析 lidars.cfg 文件（包含vehicle_to_sensing和sensor_to_lidar）"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        def find_block(text, name):
            m = re.search(rf'{name}\s*\{{', text)
            if not m:
                return None
            start = m.end() - 1
            depth, i = 0, start
            while i < len(text):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start + 1 : i]
                i += 1
            return None
        
        result = {}
        
        # 1. 解析vehicle_to_sensing（在根级别）
        v2s_blk = find_block(content, 'vehicle_to_sensing')
        if v2s_blk:
            pos = find_block(v2s_blk, 'position')
            v2s_position = None
            if pos:
                x = re.search(r'x:\s*([-\d.e]+)', pos)
                y = re.search(r'y:\s*([-\d.e]+)', pos)
                z = re.search(r'z:\s*([-\d.e]+)', pos)
                if x and y and z:
                    v2s_position = np.array([float(x.group(1)), float(y.group(1)), float(z.group(1))])
            
            ori = find_block(v2s_blk, 'orientation')
            v2s_orientation = None
            if ori:
                qx = re.search(r'qx:\s*([-\d.e]+)', ori)
                qy = re.search(r'qy:\s*([-\d.e]+)', ori)
                qz = re.search(r'qz:\s*([-\d.e]+)', ori)
                qw = re.search(r'qw:\s*([-\d.e]+)', ori)
                if qx and qy and qz and qw:
                    v2s_orientation = [float(qx.group(1)), float(qy.group(1)), float(qz.group(1)), float(qw.group(1))]
            
            if v2s_position is not None and v2s_orientation is not None:
                result['vehicle_to_sensing'] = {
                    'position': v2s_position,
                    'orientation': v2s_orientation  # [qx, qy, qz, qw]
                }
        
        # 2. 解析sensor_to_lidar
        blk = find_block(content, 'sensor_to_lidar')
        if not blk:
            config_blk = find_block(content, 'config')
            if config_blk:
                blk = find_block(config_blk, 'sensor_to_lidar')
            if not blk:
                raise ValueError(f'在 {filepath} 中未找到 sensor_to_lidar')
        
        pos = find_block(blk, 'position')
        position = None
        if pos:
            x = re.search(r'x:\s*([-\d.e]+)', pos)
            y = re.search(r'y:\s*([-\d.e]+)', pos)
            z = re.search(r'z:\s*([-\d.e]+)', pos)
            if x and y and z:
                position = np.array([float(x.group(1)), float(y.group(1)), float(z.group(1))])
        
        ori = find_block(blk, 'orientation')
        orientation = None
        if ori:
            qx = re.search(r'qx:\s*([-\d.e]+)', ori)
            qy = re.search(r'qy:\s*([-\d.e]+)', ori)
            qz = re.search(r'qz:\s*([-\d.e]+)', ori)
            qw = re.search(r'qw:\s*([-\d.e]+)', ori)
            if qx and qy and qz and qw:
                orientation = np.array([
                    float(qx.group(1)), float(qy.group(1)),
                    float(qz.group(1)), float(qw.group(1))
                ])
        
        if position is None or orientation is None:
            raise ValueError(f'sensor_to_lidar 的 position 或 orientation 不完整: {filepath}')
        
        # 将sensor_to_lidar的position和orientation添加到result中
        result['position'] = position
        result['orientation'] = orientation
        
        return result


class PointCloudIO:
    """点云文件 I/O 工具"""
    
    @staticmethod
    def save_ply(points: np.ndarray, filepath: str):
        """保存点云为 PLY 格式（ASCII，方便查看和可视化）
        
        Args:
            points: (N, 3) 或 (N, 4) 的点云数据
            filepath: 保存路径
        """
        if points.shape[1] == 3:
            # 添加强度通道
            intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack([points, intensity])
        
        with open(filepath, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("end_header\n")
            
            # 点云数据
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f}\n")
    
    @staticmethod
    def load_ply(filepath: str) -> np.ndarray:
        """从 PLY 文件加载点云
        
        Returns:
            (N, 4) 的点云数据 [x, y, z, intensity]
        """
        with open(filepath, 'r') as f:
            # 读取 header
            line = f.readline()
            if not line.startswith('ply'):
                raise ValueError(f"不是有效的 PLY 文件: {filepath}")
            
            # 跳过 header，找到 vertex 数量
            num_vertices = 0
            while True:
                line = f.readline()
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line.startswith('end_header'):
                    break
            
            # 读取点云数据
            points = []
            for _ in range(num_vertices):
                line = f.readline().strip()
                if line:
                    values = [float(v) for v in line.split()]
                    points.append(values[:4])  # x, y, z, intensity
            
            return np.array(points, dtype=np.float32)
    
    @staticmethod
    def ply_to_bin(ply_path: str, bin_path: str):
        """将 PLY 转换为 BIN 格式"""
        points = PointCloudIO.load_ply(ply_path)
        points.astype(np.float32).tofile(bin_path)


class ProtobufUtils:
    """Protobuf消息处理工具"""
    
    @staticmethod
    def extract_header_timestamp(data: bytes) -> Optional[float]:
        """从protobuf消息中提取header.timestamp_sec()
        
        参考C++:
          image_msg.header().timestamp_sec()
          cloud_msg->header().timestamp_sec()
        
        Protobuf wire format:
          - field 1 (header): nested message
            - field 1 (timestamp_sec): double (64-bit)
        
        Args:
            data: protobuf消息的bytes数据
            
        Returns:
            timestamp_sec (float)，如果解析失败返回None
        """
        # 跳过自定义header ($$$$开头)
        pos = 0
        if data.startswith(b'$$$$'):
            if len(data) < 8:
                return None
            # 读取header长度（4字节，小端）
            header_len = struct.unpack('<I', data[4:8])[0]
            # 跳过 "$$$$" + 长度字段 + header内容
            pos = 8 + header_len
            if pos >= len(data):
                return None
        while pos < len(data):
            # 解析tag
            if pos >= len(data):
                break
            tag = data[pos]
            pos += 1
            
            field_num = tag >> 3
            wire_type = tag & 7
            
            # field 1 是 header (wire_type=2: length-delimited)
            if field_num == 1 and wire_type == 2:
                # 读取长度
                length, pos_new = ProtobufUtils._decode_varint(data, pos)
                if pos_new >= len(data):
                    break
                pos = pos_new
                
                # 读取header内容
                header_data = data[pos:pos+length]
                pos += length
                
                # 在header中查找field 1 (timestamp_sec, wire_type=1: 64-bit)
                h_pos = 0
                while h_pos < len(header_data):
                    if h_pos >= len(header_data):
                        break
                    h_tag = header_data[h_pos]
                    h_pos += 1
                    
                    h_field_num = h_tag >> 3
                    h_wire_type = h_tag & 7
                    
                    if h_field_num == 1 and h_wire_type == 1:  # timestamp_sec (double)
                        if h_pos + 8 <= len(header_data):
                            timestamp_sec = struct.unpack('<d', header_data[h_pos:h_pos+8])[0]
                            return timestamp_sec
                        break
                    elif h_wire_type == 0:  # varint
                        _, h_pos = ProtobufUtils._decode_varint(header_data, h_pos)
                    elif h_wire_type == 1:  # 64-bit
                        h_pos += 8
                    elif h_wire_type == 2:  # length-delimited
                        l, h_pos = ProtobufUtils._decode_varint(header_data, h_pos)
                        h_pos += l
                    elif h_wire_type == 5:  # 32-bit
                        h_pos += 4
                    else:
                        break
                break
            elif wire_type == 0:  # varint
                _, pos = ProtobufUtils._decode_varint(data, pos)
            elif wire_type == 1:  # 64-bit
                pos += 8
            elif wire_type == 2:  # length-delimited
                length, pos = ProtobufUtils._decode_varint(data, pos)
                pos += length
            elif wire_type == 5:  # 32-bit
                pos += 4
            else:
                break
        
        return None
    
    @staticmethod
    def _decode_varint(buf: bytes, pos: int):
        """解析protobuf varint"""
        n, sh = 0, 0
        while pos < len(buf):
            b = buf[pos]
            pos += 1
            n |= (b & 0x7F) << sh
            if not (b & 0x80):
                return n, pos
            sh += 7
            if sh >= 35:
                break
        return n, pos


class PointCloudParser:
    """点云解析器（proto 格式）- 完全对齐Self-Cali-GS实现"""
    
    # DataType 常量（与 PointCloud2.proto 一致）
    _DT_INT8 = 1
    _DT_UINT8 = 2
    _DT_INT16 = 3
    _DT_UINT16 = 4
    _DT_INT32 = 5
    _DT_UINT32 = 6
    _DT_FLOAT32 = 7
    _DT_FLOAT64 = 8
    
    @staticmethod
    def _decode_varint(buf: bytes, pos: int):
        """解析 protobuf varint"""
        n, sh = 0, 0
        while pos < len(buf):
            b = buf[pos]
            pos += 1
            n |= (b & 0x7F) << sh
            if not (b & 0x80):
                return n, pos
            sh += 7
            if sh >= 35:
                break
        return n, pos
    
    @staticmethod
    def _parse_pointcloud2_wire(data: bytes):
        """手动解析 PointCloud2 protobuf wire format
        
        返回: (point_step, data_blob, fields_list)
          - point_step: 每个点的字节数
          - data_blob: 点云数据
          - fields_list: [(name, offset, datatype), ...]
        """
        point_step = None
        data_blob = None
        fields_list = []
        pos = 0
        
        while pos < len(data):
            tag, pos = PointCloudParser._decode_varint(data, pos)
            if pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 0:  # Varint
                v, pos = PointCloudParser._decode_varint(data, pos)
                if field_num == 6:  # point_step
                    point_step = v
            elif wire == 2:  # Length-delimited
                L, pos = PointCloudParser._decode_varint(data, pos)
                if pos + L > len(data):
                    break
                chunk = data[pos:pos + L]
                pos += L
                
                if field_num == 4:  # fields (repeated PointField)
                    # 解析 PointField 消息
                    i = 0
                    while i < len(chunk):
                        start_i = i
                        name, offset, datatype = None, 0, PointCloudParser._DT_FLOAT32
                        
                        # 解析单个 PointField
                        while i < len(chunk):
                            t, i = PointCloudParser._decode_varint(chunk, i)
                            if i >= len(chunk):
                                break
                            fn, w = t >> 3, t & 7
                            if w == 0:  # Varint
                                v, i = PointCloudParser._decode_varint(chunk, i)
                                if fn == 2:  # offset
                                    offset = v
                                elif fn == 3:  # datatype
                                    datatype = v
                            elif w == 2:  # Length-delimited (string)
                                ln, i = PointCloudParser._decode_varint(chunk, i)
                                if fn == 1 and i + ln <= len(chunk):  # name
                                    name = chunk[i:i + ln].decode('utf-8', errors='ignore').strip().lower()
                                i += ln
                            elif w == 5:  # Fixed32
                                if i + 4 > len(chunk):
                                    break
                                i += 4
                            elif w == 1:  # Fixed64
                                if i + 8 > len(chunk):
                                    break
                                i += 8
                            else:
                                break
                        
                        # 保存成功解析的 PointField
                        if name and name in ('x', 'y', 'z'):
                            fields_list.append((name, offset, datatype))
                        
                        # 避免死循环
                        if i == start_i:
                            i += 1
                elif field_num == 8:  # data (bytes)
                    data_blob = chunk
        
        # 验证解析结果
        if point_step and point_step > 0 and data_blob is not None and len(data_blob) >= point_step:
            return (point_step, data_blob, fields_list)
        return None
    
    @staticmethod
    def _datatype_to_fmt_scale(dt: int):
        """将 DataType 转换为 struct format 和 scale"""
        if dt in (PointCloudParser._DT_INT16, PointCloudParser._DT_UINT16):
            return ('h' if dt == PointCloudParser._DT_INT16 else 'H', 0.01)
        if dt in (PointCloudParser._DT_INT32, PointCloudParser._DT_UINT32):
            return ('i' if dt == PointCloudParser._DT_INT32 else 'I', 1.0)
        if dt == PointCloudParser._DT_FLOAT32:
            return ('f', 1.0)
        if dt == PointCloudParser._DT_FLOAT64:
            return ('d', 1.0)
        return ('f', 1.0)
    
    @staticmethod
    def _read_xyz_from_fields(buf: bytes, offset: int, step: int, fields_map: dict):
        """从 buffer 中读取 x, y, z"""
        try:
            x_info = fields_map['x']
            y_info = fields_map['y']
            z_info = fields_map['z']
        except KeyError:
            return None
        
        need = max(x_info[0], y_info[0], z_info[0]) + 8
        if offset + need > len(buf):
            return None
        
        def _read_value(info):
            off, fmt, scale = info
            if fmt == 'h':
                return struct.unpack_from('<h', buf, offset + off)[0] * scale
            elif fmt == 'H':
                return struct.unpack_from('<H', buf, offset + off)[0] * scale
            elif fmt == 'i':
                return struct.unpack_from('<i', buf, offset + off)[0] * scale
            elif fmt == 'I':
                return struct.unpack_from('<I', buf, offset + off)[0] * scale
            elif fmt == 'f':
                return struct.unpack_from('<f', buf, offset + off)[0] * scale
            elif fmt == 'd':
                return struct.unpack_from('<d', buf, offset + off)[0] * scale
            return struct.unpack_from('<f', buf, offset + off)[0] * scale
        
        return (_read_value(x_info), _read_value(y_info), _read_value(z_info))
    
    @staticmethod
    def parse_proto_pointcloud2(data: bytes) -> Optional[np.ndarray]:
        """解析 PointCloud2 proto 数据
        
        参考: ~/develop/code/github/Self-Cali-GS/surround_calibration/data/lidar_utils.py
        """
        if data is None or len(data) < 16:
            return None
        
        # 尝试多个候选位置（可能有不同的前缀）
        candidates = [data]
        if len(data) > 4:
            candidates.append(data[4:])
        if len(data) > 8:
            candidates.append(data[8:])
        
        # 使用 wire format 解析
        for to_parse in candidates:
            result = PointCloudParser._parse_pointcloud2_wire(to_parse)
            if result is None:
                continue
            
            step, raw, flist = result
            if not step or step <= 0:
                continue
            
            n = len(raw) // step
            if n < 50:
                continue
            
            # 构建 fields_map
            fields_map = {}
            for name, off, dt in flist:
                fmt, scale = PointCloudParser._datatype_to_fmt_scale(dt)
                fields_map[name] = (off, fmt, scale)
            
            # 如果没有找到完整的 x, y, z，使用默认映射（INT16）
            if len(fields_map) != 3:
                fields_map = {'x': (0, 'h', 0.01), 'y': (2, 'h', 0.01), 'z': (4, 'h', 0.01)}
            
            # 提取所有点（包含intensity和timestamp）
            # CompressedPoint 格式: x,y,z(int16), intensity(uint8), ring(uint8), timestamp(uint16)
            # offset: 0-1(x), 2-3(y), 4-5(z), 6(intensity), 7(ring), 8-9(timestamp)
            pts = []
            for i in range(min(n, 500000)):  # 限制最大点数
                base = i * step
                xyz = PointCloudParser._read_xyz_from_fields(raw, base, step, fields_map)
                if xyz is None:
                    continue
                x, y, z = xyz
                
                # 验证点坐标
                if (np.isnan(x) or np.isnan(y) or np.isnan(z) or 
                    np.isinf(x) or np.isinf(y) or np.isinf(z) or
                    abs(x) > 500 or abs(y) > 500 or abs(z) > 100):
                    continue
                
                # 读取intensity（offset 6，uint8）
                intensity = 0.0
                if base + 6 < len(raw):
                    intensity = float(raw[base + 6])
                
                # 读取timestamp（offset 8-9，uint16）
                timestamp_raw = 0
                if base + 9 < len(raw):
                    timestamp_raw = struct.unpack_from('<H', raw, base + 8)[0]
                
                # 返回 (x, y, z, intensity, timestamp)
                # timestamp单位是2微秒（根据C++代码）
                pts.append([x, y, z, intensity, float(timestamp_raw)])
            
            if len(pts) >= 50:
                return np.array(pts, dtype=np.float32)
        
        # Fallback: 尝试直接按float格式读取（如果wire format失败）
        # 参考Self-Cali-GS: 尝试不同的起始偏移量
        for offset_start in [0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500]:
            if len(data) < offset_start + 16:
                continue
            rem = len(data) - offset_start
            if rem < 16 or rem % 16 != 0:
                continue
            n = rem // 16
            pts, ok = [], 0
            for i in range(min(2000, n)):
                o = offset_start + i * 16
                try:
                    x, y, z, _ = struct.unpack_from('ffff', data, o)
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)) and abs(x) < 5000 and abs(y) < 5000 and abs(z) < 500:
                        pts.append([x, y, z, 0.0, 0.0])  # 添加dummy intensity和timestamp
                        ok += 1
                except Exception:
                    continue
            if ok > 50:
                # 如果找到足够的有效点，处理剩余的点
                for i in range(min(2000, n), n):
                    o = offset_start + i * 16
                    try:
                        x, y, z, _ = struct.unpack_from('ffff', data, o)
                        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)) and abs(x) < 5000 and abs(y) < 5000 and abs(z) < 500:
                            pts.append([x, y, z, 0.0, 0.0])
                    except Exception:
                        continue
                return np.array(pts, dtype=np.float32) if pts else None
        
        return None


class BEVCalibDatasetPreparer:
    """BEVCalib 数据集准备器（流式处理版本）"""
    
    # 定位 topics (按照用户要求，只使用/localization/pose)
    LOCALIZATION_TOPICS = [
        '/localization/pose',  # 优先使用这个topic，与C++参考代码一致
        # '/localiztaion/gnss/calibration_pose',  # 注意拼写错误是原始数据中的
        # '/sensors/gnss/pose'
    ]
    
    def __init__(
        self,
        bag_path: str,
        config_dir: str,
        output_dir: str,
        camera_name: str = "traffic_2",
        target_fps: float = 10.0,
        max_time_diff: float = 0.055,  # 55ms，参考C++: kMaxLidarCameraDelta = 55000us
        lidar_topic: str = '/sensors/lidar/combined_point_cloud_proto',
        pose_topic: str = None,  # 改为可选，如果为None则自动检测
        batch_size: int = 500,  # 每批处理的帧数（增加默认值以提升速度）
        num_workers: int = 4,  # 并行处理的工作线程数
        max_frames: int = None,  # 最大处理帧数（用于测试）
    ):
        self.bag_path = Path(bag_path)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.camera_name = camera_name
        self.target_fps = target_fps
        self.max_time_diff = max_time_diff
        self.lidar_topic = lidar_topic
        self.pose_topic = pose_topic  # 如果为None，则自动检测
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_frames = max_frames  # 最大处理帧数（用于测试）
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / 'temp'
        self.temp_dir.mkdir(exist_ok=True)
        self.temp_image_dir = self.temp_dir / 'images'
        self.temp_pc_dir = self.temp_dir / 'pointclouds'
        self.temp_image_dir.mkdir(exist_ok=True)
        self.temp_pc_dir.mkdir(exist_ok=True)
        
        # 元数据列表（不存储实际数据）
        self.image_metadata: List[ImageMetadata] = []
        self.pc_metadata: List[PointCloudMetadata] = []
        self.pose_metadata: List[PoseMetadata] = []
        
        # 从bag中提取的 Sensing→Vehicle 变换（优先级高于本地配置）
        # C++命名：T_vehicle_to_sensing（从右往左读：Sensing→Vehicle）
        # 注意：必须在_compute_transforms()之前初始化
        self.vehicle_to_sensing_from_bag: Optional[np.ndarray] = None
        
        # 计数器
        self.image_counter = 0
        self.pc_counter = 0
        
        # 调试选项
        self.verbose = True
        
        # 加载配置并计算变换矩阵
        self._load_configs()
        self._compute_transforms()
    
    def _load_configs(self):
        """加载配置文件"""
        cameras_cfg = self.config_dir / 'cameras.cfg'
        lidars_cfg = self.config_dir / 'lidars.cfg'
        
        if not cameras_cfg.exists():
            raise FileNotFoundError(f"未找到 cameras.cfg: {cameras_cfg}")
        if not lidars_cfg.exists():
            raise FileNotFoundError(f"未找到 lidars.cfg: {lidars_cfg}")
        
        cameras = ConfigParser.parse_cameras_cfg(str(cameras_cfg))
        if self.camera_name not in cameras:
            raise ValueError(f"未找到相机 {self.camera_name}，可用: {list(cameras.keys())}")
        
        self.camera_config = cameras[self.camera_name]
        self.lidar_config = ConfigParser.parse_lidars_cfg(str(lidars_cfg))
        
        print(f"已加载配置:")
        print(f"  相机: {self.camera_name}")
        print(f"  图像尺寸: {self.camera_config['intrinsic']['img_width']}x{self.camera_config['intrinsic']['img_height']}")
        print(f"  内参 fx={self.camera_config['intrinsic']['f_x']:.2f}, fy={self.camera_config['intrinsic']['f_y']:.2f}")
    
    def _extract_vehicle_to_sensing_from_pc_msg(self, data: bytes) -> Optional[np.ndarray]:
        """从点云protobuf消息中提取 vehicle_to_sensing (Sensing→Vehicle) 变换
        
        **命名约定**: vehicle_to_sensing = Sensing→Vehicle
        
        参考：lidar_online_calibrator.cpp:1038-1054 + proto_utils.cpp:569-588
        
        Returns:
            T_sensing_to_vehicle: Sensing→Vehicle的4x4变换矩阵，如果提取失败返回None
        """
        try:
            # 手动解析protobuf wire format以查找lidar_configs字段
            # 寻找vehicle_to_sensing的position和orientation
            
            # 尝试使用protobuf反序列化（如果消息包含lidar_configs）
            # 注意：我们不需要完整的proto定义，只需要提取特定字段
            
            # 简化方案：尝试搜索特定的字节模式
            # vehicle_to_sensing包含: position(x,y,z) + orientation(qw,qx,qy,qz)
            
            # 暂时返回None，使用fallback逻辑
            return None
            
        except Exception as e:
            return None
    
    def _get_vehicle_to_sensing_transform(self) -> np.ndarray:
        """获取 Sensing→Vehicle 的变换
        
        参考：proto_utils.cpp:569-588
        
        **C++命名约定**（从右往左读）：
        - vehicle_to_sensing = Sensing→Vehicle (Sensing在Vehicle系中的位姿)
        - T_vehicle_to_sensing表示将Vehicle系变换到Sensing系 = Sensing→Vehicle
        
        优先级：
        1. 从bag中的点云消息提取（self.vehicle_to_sensing_from_bag）
        2. 从本地lidars.cfg的vehicle_to_sensing字段读取
        3. 默认单位矩阵（Vehicle == Sensing）
        
        Returns:
            T_vehicle_to_sensing: Sensing→Vehicle的4x4变换矩阵（从右往左读）
        """
        # 优先使用从bag中提取的配置
        if self.vehicle_to_sensing_from_bag is not None:
            return self.vehicle_to_sensing_from_bag
        
        # 尝试从本地lidars.cfg读取vehicle_to_sensing = Sensing→Vehicle
        # 按C++规范命名：T_vehicle_to_sensing（从右往左读：Sensing→Vehicle）
        T_vehicle_to_sensing = np.eye(4)
        
        # 检查lidar_config是否包含vehicle_to_sensing
        if isinstance(self.lidar_config, dict) and 'vehicle_to_sensing' in self.lidar_config:
            v2s = self.lidar_config['vehicle_to_sensing']
            if 'position' in v2s and 'orientation' in v2s:
                pos = v2s['position']
                ori = v2s['orientation']  # [qx, qy, qz, qw]
                
                # 构建变换矩阵：Sensing→Vehicle
                r = R.from_quat(ori)
                T_vehicle_to_sensing[:3, :3] = r.as_matrix()
                T_vehicle_to_sensing[:3, 3] = pos
                
                print(f"✓ 从lidars.cfg读取 vehicle_to_sensing")
                print(f"  (C++命名: T_vehicle_to_sensing = Sensing→Vehicle)")
                print(f"  position: {pos}")
                print(f"  orientation: {ori}")
                return T_vehicle_to_sensing
        
        # 默认：Vehicle == Sensing（单位矩阵）
        print(f"ℹ️  使用默认假设: Vehicle == Sensing (单位矩阵)")
        return T_vehicle_to_sensing
    
    def _convert_poses_to_sensing_frame(self):
        """将Vehicle系的pose转换为Sensing系（对齐C++实现）
        
        C++参考：lidar_online_calibrator.cpp:849-852
            sensing_pose.second = vehicle_pose * iso_vehicle_sensing_;
            lidar_pose_data_.push_back(sensing_pose);
        
        变换理解：
        - 输入：vehicle_pose = Vehicle→World（Vehicle在World系中的位姿）
        - 中间：T_vehicle_to_sensing = Sensing→Vehicle（从config读取）
        - 输出：sensing_pose = (Vehicle→World) @ (Sensing→Vehicle) = Sensing→World
        
        含义：Sensing在World系中的位姿
        """
        print(f"  转换位姿到Sensing坐标系...")
        
        for i, pose in enumerate(self.pose_metadata):
            # 构建Vehicle系的pose（Vehicle→World）
            R_vehicle = UndistortionUtils.quat_to_matrix(pose.orientation)
            t_vehicle = pose.position
            
            iso_vehicle = np.eye(4)
            iso_vehicle[:3, :3] = R_vehicle
            iso_vehicle[:3, 3] = t_vehicle
            
            # 转换到Sensing系：sensing_pose = vehicle_pose * T_vehicle_to_sensing
            # 其中T_vehicle_to_sensing = Sensing→Vehicle（C++命名约定）
            iso_sensing = iso_vehicle @ self.T_vehicle_to_sensing
            
            # 更新pose（现在是Sensing系）
            R_sensing = iso_sensing[:3, :3]
            t_sensing = iso_sensing[:3, 3]
            
            # 转换回四元数
            r = R.from_matrix(R_sensing)
            pose.orientation = r.as_quat()  # [x, y, z, w]
            pose.position = t_sensing
        
        print(f"  ✓ 已转换 {len(self.pose_metadata)} 个位姿到Sensing系")
    
    def _compute_transforms(self):
        """计算变换矩阵
        
        参考：proto_utils.cpp:193-201, 274-323 + 坐标系文档
        
        **C++命名约定**（从右往左读）：
        - T_A_to_B 表示将A系坐标变换到B系 = B→A 的变换 = B在A系中的位姿
        - 例如：T_sensing_to_camera = Camera→Sensing (将Sensing系变换到Camera系)
        
        **config文件含义**：
        - cameras.cfg: sensor_to_cam = Camera→Sensing (Camera在Sensing系中的位姿)
        - lidars.cfg: sensor_to_lidar = LiDAR→Sensing (LiDAR在Sensing系中的位姿)
        - lidars.cfg: vehicle_to_sensing = Sensing→Vehicle (Sensing在Vehicle系中的位姿)
        
        **坐标系定义**：
        - Vehicle系：后轴中心，前左上=XYZ
        - Sensing系：虚拟系，前左上=XYZ
        - LiDAR系：前左上=XYZ（与KITTI Velodyne一致）
        - Camera系：光轴=Z，右下前=XYZ（OpenCV标准）
        """
        # 1. 从config读取 sensor_to_cam = Camera→Sensing
        # 按C++规范命名：T_sensing_to_camera（从右往左读：Camera→Sensing）
        T_sensing_to_camera = np.eye(4)
        r = R.from_quat(self.camera_config['orientation'])
        T_sensing_to_camera[:3, :3] = r.as_matrix()
        T_sensing_to_camera[:3, 3] = self.camera_config['position']
        
        # 2. 取逆得到 Sensing→Camera (从右往左读)
        T_camera_to_sensing = np.linalg.inv(T_sensing_to_camera)
        
        # 3. 从config读取 sensor_to_lidar = LiDAR→Sensing
        # 按C++规范命名：T_sensing_to_lidar（从右往左读：LiDAR→Sensing）
        T_sensing_to_lidar = np.eye(4)
        r = R.from_quat(self.lidar_config['orientation'])
        T_sensing_to_lidar[:3, :3] = r.as_matrix()
        T_sensing_to_lidar[:3, 3] = self.lidar_config['position']
        
        # 4. 取逆得到 Sensing→LiDAR (从右往左读)
        T_lidar_to_sensing = np.linalg.inv(T_sensing_to_lidar)
        
        # 5. 计算 LiDAR→Camera (KITTI标准Tr矩阵)
        # 变换链：LiDAR → Sensing → Camera
        # T_camera_to_lidar = T_camera_to_sensing @ T_sensing_to_lidar
        #                   = (Sensing→Camera) @ (LiDAR→Sensing)
        #                   = LiDAR → Camera
        self.T_camera_to_lidar = T_camera_to_sensing @ T_sensing_to_lidar
        
        # 6. 取逆得到 Camera→LiDAR (从右往左读)
        self.T_lidar_to_camera = np.linalg.inv(self.T_camera_to_lidar)
        
        # 7. 保存用于去畸变和投影的变换
        self.T_sensing_to_lidar = T_sensing_to_lidar    # LiDAR → Sensing (从右往左读)
        self.T_lidar_to_sensing = T_lidar_to_sensing    # Sensing → LiDAR (从右往左读)
        self.T_sensing_to_camera = T_sensing_to_camera  # Camera → Sensing (从右往左读)
        self.T_camera_to_sensing = T_camera_to_sensing  # Sensing → Camera (从右往左读)
        
        # 8. 获取并缓存 Vehicle→Sensing 变换（频繁使用，提前计算）
        self.T_vehicle_to_sensing = self._get_vehicle_to_sensing_transform()
        
        # 9. 内参矩阵
        intrinsic = self.camera_config['intrinsic']
        self.K = np.array([
            [intrinsic['f_x'], 0, intrinsic['o_x']],
            [0, intrinsic['f_y'], intrinsic['o_y']],
            [0, 0, 1]
        ])
        
        print(f"✓ 变换矩阵已计算 (C++命名约定，从右往左读):")
        print(f"  T_camera_to_lidar: LiDAR → Camera (KITTI Tr矩阵)")
        print(f"  T_lidar_to_camera: Camera → LiDAR")
        print(f"  T_lidar_to_sensing: Sensing → LiDAR (去畸变用)")
        print(f"  T_vehicle_to_sensing: Sensing → Vehicle (已缓存)")
    
    def extract_data_from_bag(self):
        """从 rosbag 提取数据（流式处理+并行加速）"""
        print(f"\n{'='*80}")
        print(f"阶段 1/3: 从 rosbag 提取数据")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        bag_files = self._find_bag_files()
        if not bag_files:
            raise ValueError(f"未找到 bag 文件: {self.bag_path}")
        
        print(f"找到 {len(bag_files)} 个 bag 文件")
        
        try:
            from rosbags.rosbag1 import Reader
            use_rosbags = True
        except ImportError:
            try:
                import rosbag
                use_rosbags = False
            except ImportError:
                raise ImportError("需要安装 rosbag 或 rosbags: pip install rosbags")
        
        # 确定图像 topic
        possible_topics = [
            f'/sensors/camera/{self.camera_name}_raw_data/compressed_proto',
            f'/sensors/camera/{self.camera_name}/compressed_proto',
        ]
        
        image_topic = None
        detected_pose_topic = None
        
        # 关键修复：扫描所有bag文件以收集所有可用topics
        # （因为不同的Topic Group有不同的topics！）
        print(f"\n扫描所有bag文件以检测topics...")
        all_available_topics = set()
        
        # 🔍 优化：按文件名分组，确保每个Topic Group都被扫描
        from collections import defaultdict
        bags_by_group = defaultdict(list)
        for bf in bag_files:
            # 提取Topic Group名称（如 Heavy_Topic_Group, Light_Topic_Group等）
            parts = bf.parts
            group_name = 'default'
            for i, p in enumerate(parts):
                if 'Topic_Group' in p:
                    group_name = p
                    break
            bags_by_group[group_name].append(bf)
        
        # 从每个Group扫描至少1个bag
        bags_to_scan = []
        for group, blist in bags_by_group.items():
            bags_to_scan.append(blist[0])  # 每组扫描第一个
        
        print(f"  发现 {len(bags_by_group)} 个Topic Groups: {list(bags_by_group.keys())}")
        print(f"  扫描 {len(bags_to_scan)} 个代表性bag文件...")
        
        if use_rosbags:
            for bag_file in bags_to_scan:
                try:
                    with Reader(str(bag_file)) as reader:
                        all_available_topics.update(reader.topics.keys())
                except Exception as e:
                    print(f"  警告: 无法读取 {bag_file.name}: {e}")
        else:
            import rosbag as rb
            for bag_file in bags_to_scan:
                try:
                    with rb.Bag(str(bag_file), 'r') as bag:
                        info = bag.get_type_and_topic_info()
                        all_available_topics.update(info[1].keys())
                except Exception as e:
                    print(f"  警告: 无法读取 {bag_file.name}: {e}")
        
        print(f"  ✓ 扫描完成，共找到 {len(all_available_topics)} 个不同的topics")
        
        # 查找图像topic
        for t in possible_topics:
            if t in all_available_topics:
                image_topic = t
                print(f"  ✓ 找到图像 topic: {t}")
                break
        
        # 查找位姿topic（如果没有指定）
        if self.pose_topic is None:
            for t in self.LOCALIZATION_TOPICS:
                if t in all_available_topics:
                    detected_pose_topic = t
                    print(f"  ✓ 找到位姿 topic: {t}")
                    break
            if detected_pose_topic is None:
                print(f"  ⚠️  警告: 未找到任何位姿topic，将跳过点云去畸变")
                print(f"     尝试的topics: {self.LOCALIZATION_TOPICS}")
        else:
            if self.pose_topic in all_available_topics:
                detected_pose_topic = self.pose_topic
                print(f"  ✓ 使用指定位姿 topic: {self.pose_topic}")
        
        # 使用检测到的topic或指定的topic
        self.active_pose_topic = detected_pose_topic or self.pose_topic
        
        # 如果设置了max_frames，强制串行提取以便提前终止
        if self.max_frames is not None:
            print(f"\n⚠️  设置了max_frames={self.max_frames}，使用串行提取以便提前终止")
            force_serial = True
        else:
            force_serial = False
        
        # 并行提取（如果有多个bag文件且未设置max_frames）
        if len(bag_files) > 1 and self.num_workers > 1 and not force_serial:
            print(f"\n使用 {self.num_workers} 个线程并行处理 bag 文件...")
            self._extract_parallel(bag_files, image_topic, possible_topics, use_rosbags)
        else:
            # 串行提取
            # 关键修复：对于max_frames模式，优先处理Heavy bags（图像），快速达到阈值！
            if self.max_frames is not None:
                # 快速模式：优先处理Heavy bags（图像），然后是Light/Medium/Tiny（点云+位姿）
                sorted_bags = sorted(bag_files, key=lambda x: (
                    1 if 'Heavy_Topic_Group' not in x.name else 0,
                    x.name
                ))
                print(f"  🚀 快速模式（max_frames={self.max_frames}）：优先处理Heavy bags（图像），快速收集数据")
            else:
                # 正常模式：优先处理Light/Medium/Tiny bags（点云+位姿），最后处理Heavy bags（图像）
                sorted_bags = sorted(bag_files, key=lambda x: (
                    0 if 'Heavy_Topic_Group' not in x.name else 1,
                    x.name
                ))
                print(f"  优化处理顺序：优先处理Light/Medium/Tiny bags（点云+位姿），最后处理Heavy bags（图像）")
            
            for bag_file in sorted_bags:
                # 提前终止检查（仅针对图像和点云，位姿需要完整）
                if self.max_frames is not None:
                    # 修正：使用1.1倍作为缓冲，而不是1.5倍
                    img_count = len(self.image_metadata)
                    pc_count = len(self.pc_metadata)
                    threshold = self.max_frames * 1.1
                    
                    # 关键修复：如果图像和点云都达到阈值，则完全停止提取（只提取位姿）
                    if img_count >= threshold and pc_count >= threshold:
                        # 注意：位姿不限制，继续提取以确保时间覆盖
                        print(f"\n✓ 图像和点云已足够（图像:{img_count}, 点云:{pc_count}）")
                        print(f"   继续提取位姿数据以确保时间范围完整...")
                        # 切换到仅提取位姿模式
                        try:
                            if use_rosbags:
                                self._extract_poses_only(bag_file, use_rosbags=True)
                            else:
                                self._extract_poses_only(bag_file, use_rosbags=False)
                        except Exception as e:
                            print(f"  警告: {e}")
                        continue
                    # 关键修复：如果任一数据类型已达到阈值，跳过主要包含该类型的bags
                    elif img_count >= threshold and 'Heavy_Topic_Group' in bag_file.name:
                        print(f"\n⏭️  跳过 {bag_file.name}（图像数已足够:{img_count}）")
                        continue
                    elif pc_count >= threshold and 'Heavy_Topic_Group' not in bag_file.name:
                        print(f"\n⏭️  跳过 {bag_file.name}（点云数已足够:{pc_count}）")
                        # 但仍需要处理位姿
                        try:
                            if use_rosbags:
                                self._extract_poses_only(bag_file, use_rosbags=True)
                            else:
                                self._extract_poses_only(bag_file, use_rosbags=False)
                        except Exception as e:
                            print(f"  警告: {e}")
                        continue
                
                print(f"\n处理: {bag_file.name}")
                try:
                    if use_rosbags:
                        self._extract_streaming_rosbags(bag_file, image_topic, possible_topics)
                    else:
                        self._extract_streaming_rosbag(bag_file, image_topic, possible_topics)
                except Exception as e:
                    print(f"  警告: {e}")
        
        # 排序元数据
        self.image_metadata.sort(key=lambda x: x.timestamp)
        self.pc_metadata.sort(key=lambda x: x.timestamp)
        self.pose_metadata.sort(key=lambda x: x.timestamp)
        
        # ✅ 关键修复：将Vehicle系的pose转换为Sensing系（对齐C++实现）
        # C++参考：lidar_online_calibrator.cpp:849-852
        #   sensing_pose.second = vehicle_pose * iso_vehicle_sensing_;
        if len(self.pose_metadata) > 0:
            self._convert_poses_to_sensing_frame()
        
        extract_time = time.time() - start_time
        
        print(f"\n✓ 数据提取完成:")
        print(f"  图像: {len(self.image_metadata)} 帧")
        print(f"  点云: {len(self.pc_metadata)} 帧")
        print(f"  位姿: {len(self.pose_metadata)} 个")
        if len(self.pose_metadata) == 0:
            print(f"  ⚠️  警告：未提取到位姿数据！点云去畸变将跳过！")
            print(f"    检测到的位姿topic: {self.active_pose_topic or '未找到'}")
            print(f"    可能的原因：1) bag中没有位姿数据；2) topic名称不匹配")
        print(f"  耗时: {timedelta(seconds=int(extract_time))}")
        print(f"  速度: {len(self.image_metadata) / extract_time:.2f} 帧/秒")
        
        # 保存提取阶段的耗时
        self._extract_time = extract_time
    
    def _extract_parallel(self, bag_files: List[Path], image_topic: Optional[str],
                         possible_topics: List[str], use_rosbags: bool):
        """并行处理多个bag文件"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # 线程锁保护元数据列表
        lock = threading.Lock()
        
        def process_single_bag(bag_file: Path):
            """处理单个bag文件"""
            try:
                if use_rosbags:
                    # 创建临时元数据列表
                    temp_images = []
                    temp_pcs = []
                    temp_poses = []
                    
                    # 提取数据
                    self._extract_streaming_rosbags_to_lists(
                        bag_file, image_topic, possible_topics,
                        temp_images, temp_pcs, temp_poses
                    )
                    
                    # 合并到主列表（需要加锁）
                    with lock:
                        self.image_metadata.extend(temp_images)
                        self.pc_metadata.extend(temp_pcs)
                        self.pose_metadata.extend(temp_poses)
                    
                    return len(temp_images), len(temp_pcs), len(temp_poses)
                else:
                    # rosbag库暂不支持并行（GIL限制）
                    return 0, 0, 0
            except Exception as e:
                print(f"  错误处理 {bag_file.name}: {e}")
                return 0, 0, 0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_single_bag, bf): bf for bf in bag_files}
            
            for future in tqdm(as_completed(futures), total=len(bag_files),
                             desc="  并行处理bag文件", unit="bag"):
                bag_file = futures[future]
                try:
                    n_images, n_pcs, n_poses = future.result()
                    print(f"  ✓ {bag_file.name}: {n_images} 图像, {n_pcs} 点云, {n_poses} 位姿")
                except Exception as e:
                    print(f"  ✗ {bag_file.name}: 错误 - {e}")
    
    def _find_bag_files(self) -> List[Path]:
        """查找 bag 文件"""
        if self.bag_path.is_file():
            return [self.bag_path]
        elif self.bag_path.is_dir():
            return sorted(self.bag_path.glob('**/*.bag'))
        return []
    
    def _extract_poses_only(self, bag_file: Path, use_rosbags: bool = True):
        """只提取位姿数据（确保时间范围完整）
        
        参考C++实现，位姿数据需要覆盖所有图像/点云的时间范围
        """
        pose_count_before = len(self.pose_metadata)
        
        if use_rosbags:
            from rosbags.rosbag1 import Reader
            
            with Reader(str(bag_file)) as reader:
                # 检查位姿topic是否在当前bag中
                if self.active_pose_topic and self.active_pose_topic not in reader.topics:
                    return
                
                # 只关注位姿topic
                for connection, timestamp, rawdata in reader.messages():
                    if self.active_pose_topic and connection.topic == self.active_pose_topic:
                        try:
                            data = self._extract_string_msg(rawdata)
                            if data:
                                # 创建临时msg对象
                                class TempMsg:
                                    def __init__(self, d):
                                        self.data = d
                                ts_sec = timestamp / 1e9
                                pose_data = self._decode_pose_msg(TempMsg(data), fallback_timestamp=ts_sec)
                                if pose_data:
                                    self.pose_metadata.append(pose_data)
                        except Exception as e:
                            pass
                    elif not self.active_pose_topic and connection.topic in self.LOCALIZATION_TOPICS:
                        # 自动检测位姿topic
                        try:
                            data = self._extract_string_msg(rawdata)
                            if data:
                                class TempMsg:
                                    def __init__(self, d):
                                        self.data = d
                                ts_sec = timestamp / 1e9
                                pose_data = self._decode_pose_msg(TempMsg(data), fallback_timestamp=ts_sec)
                                if pose_data:
                                    self.active_pose_topic = connection.topic
                                    self.pose_metadata.append(pose_data)
                        except Exception as e:
                            pass
        else:
            import rosbag as rb
            with rb.Bag(str(bag_file), 'r') as bag:
                for topic, msg, t in bag.read_messages(topics=self.LOCALIZATION_TOPICS):
                    try:
                        pose_data = self._decode_pose_msg(msg)
                        if pose_data:
                            if not self.active_pose_topic:
                                self.active_pose_topic = topic
                            self.pose_metadata.append(pose_data)
                    except Exception as e:
                        pass
        
        pose_count_after = len(self.pose_metadata)
        if pose_count_after > pose_count_before:
            print(f"    提取 {pose_count_after - pose_count_before} 个位姿")
    
    def _extract_streaming_rosbags(self, bag_file: Path, image_topic: Optional[str],
                                   possible_topics: List[str]):
        """使用 rosbags 流式提取"""
        temp_images = []
        temp_pcs = []
        temp_poses = []
        
        self._extract_streaming_rosbags_to_lists(
            bag_file, image_topic, possible_topics,
            temp_images, temp_pcs, temp_poses
        )
        
        # 合并到主列表
        self.image_metadata.extend(temp_images)
        self.pc_metadata.extend(temp_pcs)
        self.pose_metadata.extend(temp_poses)
    
    def _extract_streaming_rosbags_to_lists(self, bag_file: Path, image_topic: Optional[str],
                                           possible_topics: List[str],
                                           out_images: list, out_pcs: list, out_poses: list):
        """使用 rosbags 流式提取（输出到指定列表）"""
        from rosbags.rosbag1 import Reader
        
        image_buffer = []
        pc_buffer = []
        
        topics_to_check = [image_topic] if image_topic else possible_topics
        
        with Reader(str(bag_file)) as reader:
            available = list(reader.topics.keys())
            topics_to_check = [t for t in topics_to_check if t in available]
            
            # 检查位姿topic是否可用
            pose_topic_available = self.active_pose_topic and self.active_pose_topic in available
            
            msg_count = 0
            for connection, timestamp, rawdata in tqdm(reader.messages(), 
                                                      desc=f"  提取数据",
                                                      unit="msg"):
                # 🔥 提前终止检查：每10条消息检查一次（快速模式用更细粒度）
                msg_count += 1
                if self.max_frames is not None and msg_count % 10 == 0:
                    img_count = len(self.image_metadata) + len(image_buffer)
                    pc_count = len(self.pc_metadata) + len(pc_buffer)
                    if img_count >= self.max_frames * 1.1 and pc_count >= self.max_frames * 1.1:
                        print(f"\n  ⚡ 提前终止：已收集足够数据 (图像:{img_count}, 点云:{pc_count})")
                        break
                
                # bag_record_time用作fallback（当header时间戳提取失败时）
                bag_record_time = timestamp / 1e9
                
                # 提取图像
                if connection.topic in topics_to_check:
                    data = self._extract_string_msg(rawdata)
                    if data:
                        # ✅ 关键修复：从header提取时间戳（参考C++: image_msg.header().timestamp_sec()）
                        header_ts = ProtobufUtils.extract_header_timestamp(data)
                        ts_sec = header_ts if header_ts is not None else bag_record_time
                        
                        image = self._decode_image_msg_from_bytes(data)
                        if image is not None:
                            image_buffer.append((ts_sec, image))
                            
                            # 批量保存
                            if len(image_buffer) >= self.batch_size:
                                self._save_image_batch(image_buffer)
                                image_buffer.clear()
                
                # 提取点云
                elif connection.topic == self.lidar_topic:
                    data = self._extract_string_msg(rawdata)
                    if data:
                        # ✅ 关键修复：从header提取时间戳（参考C++: cloud_msg->header().timestamp_sec()）
                        header_ts = ProtobufUtils.extract_header_timestamp(data)
                        ts_sec = header_ts if header_ts is not None else bag_record_time
                        
                        # 尝试从首个点云消息提取 vehicle_to_sensing (Sensing→Vehicle)
                        if self.vehicle_to_sensing_from_bag is None:
                            v2s = self._extract_vehicle_to_sensing_from_pc_msg(data)
                            if v2s is not None:
                                self.vehicle_to_sensing_from_bag = v2s
                                print(f"✓ 从bag点云消息中提取到 vehicle_to_sensing")
                                print(f"   (C++命名: T_vehicle_to_sensing = Sensing→Vehicle)")
                        
                        points = PointCloudParser.parse_proto_pointcloud2(data)
                        if points is not None and points.shape[0] >= 50:
                            pc_buffer.append((ts_sec, points))
                            
                            # 批量保存
                            if len(pc_buffer) >= self.batch_size:
                                self._save_pc_batch(pc_buffer)
                                pc_buffer.clear()
                
                # 提取位姿
                elif pose_topic_available and connection.topic == self.active_pose_topic:
                    data = self._extract_string_msg(rawdata)
                    if data:
                        # 创建一个临时msg对象以重用_decode_pose_msg
                        class TempMsg:
                            def __init__(self, d):
                                self.data = d
                        pose = self._decode_pose_msg(TempMsg(data), fallback_timestamp=bag_record_time)
                        if pose:
                            out_poses.append(pose)
        
        # 保存剩余的数据
        if image_buffer:
            self._save_image_batch(image_buffer)
        if pc_buffer:
            self._save_pc_batch(pc_buffer)
    
    def _extract_streaming_rosbag(self, bag_file: Path, image_topic: Optional[str],
                                  possible_topics: List[str]):
        """使用 rosbag 流式提取"""
        import rosbag
        
        image_buffer = []
        pc_buffer = []
        
        # 检查可用topics
        with rosbag.Bag(str(bag_file), 'r') as bag:
            info = bag.get_type_and_topic_info()[1]
            available_topics = list(info.keys())
            pose_topic_available = self.active_pose_topic and self.active_pose_topic in available_topics
        
        with rosbag.Bag(str(bag_file), 'r') as bag:
            msg_count = 0
            for topic, msg, t in tqdm(bag.read_messages(),
                                     desc=f"  提取数据",
                                     unit="msg"):
                # 🔥 提前终止检查：每10条消息检查一次（快速模式用更细粒度）
                msg_count += 1
                if self.max_frames is not None and msg_count % 10 == 0:
                    img_count = len(self.image_metadata) + len(image_buffer)
                    pc_count = len(self.pc_metadata) + len(pc_buffer)
                    if img_count >= self.max_frames * 1.1 and pc_count >= self.max_frames * 1.1:
                        print(f"\n  ⚡ 提前终止：已收集足够数据 (图像:{img_count}, 点云:{pc_count})")
                        break
                
                # ⚠️ 注意：t.to_sec()是bag记录时间，不是传感器数据时间戳
                bag_record_time = t.to_sec()  # 用作fallback
                
                # 提取图像
                if (image_topic and topic == image_topic) or topic in possible_topics:
                    if hasattr(msg, 'data'):
                        data = msg.data
                        if isinstance(data, str):
                            data = data.encode('latin-1')
                        elif not isinstance(data, bytes):
                            data = None
                        
                        if data:
                            # ✅ 关键修复：从header提取时间戳（参考C++: image_msg.header().timestamp_sec()）
                            header_ts = ProtobufUtils.extract_header_timestamp(data)
                            ts_sec = header_ts if header_ts is not None else bag_record_time
                            
                            image = self._decode_image_msg_from_bytes(data)
                            if image is not None:
                                image_buffer.append((ts_sec, image))
                                if len(image_buffer) >= self.batch_size:
                                    self._save_image_batch(image_buffer)
                                    image_buffer.clear()
                
                # 提取点云
                elif topic == self.lidar_topic:
                    if hasattr(msg, 'data'):
                        data = msg.data
                        if isinstance(data, str):
                            data = data.encode('latin-1')
                        elif not isinstance(data, bytes):
                            continue
                        
                        # ✅ 关键修复：从header提取时间戳（参考C++: cloud_msg->header().timestamp_sec()）
                        header_ts = ProtobufUtils.extract_header_timestamp(data)
                        ts_sec = header_ts if header_ts is not None else bag_record_time
                        
                        # 尝试从首个点云消息提取 vehicle_to_sensing (Sensing→Vehicle)
                        if self.vehicle_to_sensing_from_bag is None:
                            v2s = self._extract_vehicle_to_sensing_from_pc_msg(data)
                            if v2s is not None:
                                self.vehicle_to_sensing_from_bag = v2s
                                print(f"✓ 从bag点云消息中提取到 vehicle_to_sensing")
                                print(f"   (C++命名: T_vehicle_to_sensing = Sensing→Vehicle)")
                        
                        points = PointCloudParser.parse_proto_pointcloud2(data)
                        if points is not None and points.shape[0] >= 50:
                            pc_buffer.append((ts_sec, points))
                            if len(pc_buffer) >= self.batch_size:
                                self._save_pc_batch(pc_buffer)
                                pc_buffer.clear()
                
                # 提取位姿
                elif pose_topic_available and topic == self.active_pose_topic:
                    pose = self._decode_pose_msg(msg, fallback_timestamp=bag_record_time)
                    if pose is not None:
                        self.pose_metadata.append(pose)
        
        if image_buffer:
            self._save_image_batch(image_buffer)
        if pc_buffer:
            self._save_pc_batch(pc_buffer)
    
    def _save_image_batch(self, batch: List[Tuple[float, np.ndarray]]):
        """保存一批图像到临时目录"""
        for ts, image in batch:
            filename = f"{self.image_counter:06d}.jpg"
            filepath = self.temp_image_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            self.image_metadata.append(ImageMetadata(
                timestamp=ts,
                file_path=str(filepath)
            ))
            self.image_counter += 1
            
            del image  # 立即释放
    
    def _save_pc_batch(self, batch: List[Tuple[float, np.ndarray]]):
        """保存一批点云到临时目录（BIN 格式，保留timestamp用于去畸变）"""
        for ts, points in batch:
            filename = f"{self.pc_counter:06d}.bin"
            filepath = self.temp_pc_dir / filename
            
            # 保存点云（可能是(N,4)或(N,5)，保持原样）
            # (N,5)格式：x, y, z, intensity, timestamp（用于去畸变）
            # (N,4)格式：x, y, z, intensity（已经去畸变或无timestamp）
            if points.shape[1] == 3:
                # 添加强度通道（全0）
                intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, intensity])
            
            # 保存为 BIN 格式
            points.astype(np.float32).tofile(str(filepath))
            
            self.pc_metadata.append(PointCloudMetadata(
                timestamp=ts,
                file_path=str(filepath)
            ))
            self.pc_counter += 1
            
            del points  # 立即释放
    
    def _extract_string_msg(self, rawdata: bytes) -> Optional[bytes]:
        """提取 std_msgs/String"""
        try:
            if len(rawdata) < 4:
                return None
            length = struct.unpack('<I', rawdata[:4])[0]
            if 0 < length < len(rawdata):
                return rawdata[4:4+length]
            return rawdata
        except:
            return rawdata
    
    def _decode_pose_msg(self, msg, fallback_timestamp: float = None) -> Optional[PoseMetadata]:
        """解码位姿消息（protobuf 格式）- 参考Self-Cali-GS实现"""
        try:
            if hasattr(msg, 'data'):
                data = msg.data
                if isinstance(data, str):
                    data = data.encode('latin-1')
                elif not isinstance(data, bytes):
                    return None
                
                # 提取 String 包装
                data = self._extract_string_msg(data)
                if not data or len(data) < 48:  # 至少需要6个double (position+euler)
                    return None
                
                # 尝试wire format解析 (参考Self-Cali-GS)
                position, euler_angles, timestamp = self._parse_proto_localization_wire(data)
                
                if position is not None and euler_angles is not None:
                    # ✅ 验证position不是全零（参考Self-Cali-GS）
                    if np.max(np.abs(position)) < 1e-6:
                        return None  # position全零，无效数据
                    
                    # 从欧拉角计算四元数
                    r = R.from_euler('xyz', euler_angles, degrees=False)
                    orientation = r.as_quat()  # [x, y, z, w]
                    
                    # 如果没有提取到时间戳，使用fallback
                    if timestamp is None:
                        timestamp = fallback_timestamp
                    
                    if timestamp is None:
                        return None
                    
                    return PoseMetadata(
                        timestamp=timestamp,
                        position=position,
                        orientation=orientation
                    )
            
            return None
        except Exception as e:
            return None
    
    def _parse_proto_localization_wire(self, data: bytes) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """按照wire format解析定位proto（完全对齐Self-Cali-GS实现）
        
        Returns:
            (position, euler_angles, timestamp)
        """
        position = None
        euler_angles = None
        timestamp = None
        
        try:
            # 跳过DeepRoute的额外header: $$$$ + <4 bytes length>
            if data[:4] == b'$$$$':
                if len(data) < 8:
                    return None, None, None
                data = data[8:]  # 跳过4字节marker + 4字节length
            # 方法1: 尝试JSON格式（某些bag使用JSON）
            try:
                text = data.decode('utf-8', errors='ignore').strip()
                if text.startswith('{'):
                    import json
                    j = json.loads(text)
                    
                    # 提取timestamp
                    if 'measurement_time' in j:
                        timestamp = float(j['measurement_time']) / 1e6
                    
                    # 提取position
                    if 'position' in j:
                        p = j['position']
                        if isinstance(p, dict):
                            position = np.array([float(p.get('x', 0)), float(p.get('y', 0)), float(p.get('z', 0))])
                        elif isinstance(p, (list, tuple)):
                            position = np.array([float(p[0]), float(p[1]), float(p[2])])
                    
                    # 提取euler_angles
                    if 'euler_angles' in j:
                        e = j['euler_angles']
                        if isinstance(e, dict):
                            euler_angles = np.array([float(e.get('x', 0)), float(e.get('y', 0)), float(e.get('z', 0))])
                        else:
                            euler_angles = np.array([float(e[0]), float(e[1]), float(e[2])])
                    
                    if position is not None and euler_angles is not None:
                        return position, euler_angles, timestamp
            except:
                pass
            
            # 方法2: Wire format解析
            i = 0
            while i < len(data) - 1:
                tag = data[i]
                i += 1
                
                field_number = tag >> 3
                wire_type = tag & 0x7
                
                if wire_type == 0:  # Varint
                    while i < len(data):
                        byte = data[i]
                        i += 1
                        if (byte & 0x80) == 0:
                            break
                elif wire_type == 1:  # Fixed64 (sfixed64, double)
                    if i + 8 > len(data):
                        break
                    val = struct.unpack_from('<q', data, i)[0]  # int64
                    i += 8
                    # field 2 通常是 measurement_time (微秒)
                    if field_number == 2 and 1.5e15 < val < 2e15:
                        timestamp = float(val) / 1e6  # 微秒 → 秒（显式转换）
                elif wire_type == 2:  # Length-delimited
                    if i >= len(data):
                        break
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
                    
                    if i + length > len(data):
                        break
                    
                    payload = data[i:i+length]
                    i += length
                    
                    # Point3D 可能是24 bytes (3个纯double) 或 27 bytes (带tag的double)
                    if length == 24:
                        try:
                            coords = struct.unpack_from('<ddd', payload, 0)
                            if field_number == 5:  # position
                                position = np.array(coords, dtype=float)
                            elif field_number == 6:  # euler_angles
                                euler_angles = np.array(coords, dtype=float)
                        except:
                            pass
                    elif length == 27:
                        # DeepRoute格式: 每个double前有tag (field + wire_type)
                        # tag (1 byte) + double (8 bytes) = 9 bytes per value
                        try:
                            j = 0
                            coords = []
                            while j < len(payload) - 8:
                                if (payload[j] & 0x7) == 1:  # Fixed64 (double)
                                    val = struct.unpack_from('<d', payload, j+1)[0]
                                    coords.append(val)
                                    j += 9
                                else:
                                    j += 1
                            
                            if len(coords) == 3:
                                if field_number == 5:  # position
                                    position = np.array(coords, dtype=float)
                                elif field_number == 6:  # euler_angles
                                    euler_angles = np.array(coords, dtype=float)
                        except:
                            pass
                elif wire_type == 5:  # Fixed32
                    if i + 4 > len(data):
                        break
                    i += 4
                else:
                    break
            
            # 方法3: Fallback启发式搜索 - 搜索连续6个double (position+euler)
            # 这是最鲁棒的方法，对齐Self-Cali-GS
            if position is None or euler_angles is None:
                for i in range(0, len(data) - 47, 1):
                    try:
                        d = [struct.unpack_from('<d', data, i + k * 8)[0] for k in range(6)]
                        # Self-Cali-GS的验证条件（加强版）：
                        # 1. position在合理范围内（< 1e6米，即1000公里）
                        # 2. euler角度应在 [-4, 4] 范围内（~±229度）
                        if not all(abs(v) < 1e6 and not (np.isnan(v) or np.isinf(v)) for v in d[0:3]):
                            continue
                        if not all(abs(v) < 4 for v in d[3:6]):  # euler更严格
                            continue
                        
                        position = np.array(d[0:3], dtype=float)
                        euler_angles = np.array(d[3:6], dtype=float)
                        
                        # 搜索timestamp（在找到pos+euler之后）
                        if timestamp is None:
                            for j in range(0, min(i, len(data) - 8)):
                                try:
                                    v = struct.unpack_from('<q', data, j)[0]  # int64
                                    if 1.5e15 < v < 2e15:
                                        timestamp = float(v) / 1e6  # 微秒 → 秒（显式转换）
                                        break
                                except:
                                    continue
                        break
                    except:
                        continue
            
            # 方法4: 只搜索euler_angles (3个double)，作为最后的fallback
            if position is None and euler_angles is None:
                for i in range(0, len(data) - 23, 1):
                    try:
                        d = [struct.unpack_from('<d', data, i + k * 8)[0] for k in range(3)]
                        if all(abs(v) < 4 for v in d):  # euler范围检查
                            euler_angles = np.array(d, dtype=float)
                            # 如果只有euler，position设为零（虽然不理想）
                            break
                    except:
                        continue
            
            # 最后搜索时间戳（如果还没找到）
            if timestamp is None:
                for i in range(0, len(data) - 8, 1):
                    try:
                        val = struct.unpack_from('<q', data, i)[0]  # int64
                        if 1.5e15 < val < 2e15:  # 微秒范围验证
                            timestamp = float(val) / 1e6  # 微秒 → 秒（显式转换）
                            break
                    except:
                        continue
            
            return position, euler_angles, timestamp
            
        except Exception:
            return None, None, None
    
    def _extract_proto_timestamp(self, data: bytes) -> Optional[float]:
        """从 protobuf 中提取时间戳"""
        try:
            # 查找时间戳（通常是 int64 或 double）
            # Protobuf tag for timestamp 通常是 field 3 或 4
            for i in range(min(200, len(data) - 8)):
                try:
                    # 尝试读取为 uint64 (nanoseconds since epoch)
                    ts_ns = struct.unpack('<Q', data[i:i+8])[0]
                    # 检查是否是合理的时间戳（2020-2030年）
                    if 1577836800e9 < ts_ns < 1893456000e9:
                        return ts_ns / 1e9
                    
                    # 尝试读取为 double (seconds since epoch)
                    ts_sec = struct.unpack('<d', data[i:i+8])[0]
                    if 1577836800 < ts_sec < 1893456000:
                        return ts_sec
                except:
                    pass
            
            return None
        except:
            return None
    
    def _decode_image(self, rawdata: bytes) -> Optional[np.ndarray]:
        """解码图像（rosbags）"""
        try:
            data = self._extract_string_msg(rawdata)
            if not data:
                return None
            
            # JPEG
            jpeg_start = data.find(b'\xff\xd8')
            if jpeg_start != -1:
                jpeg_end = data.rfind(b'\xff\xd9')
                if jpeg_end > jpeg_start:
                    img_array = np.frombuffer(data[jpeg_start:jpeg_end+2], np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if image is not None:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PNG
            png_start = data.find(b'\x89PNG')
            if png_start != -1:
                img_array = np.frombuffer(data[png_start:], np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
        return None
    
    def _decode_image_msg_from_bytes(self, data: bytes) -> Optional[np.ndarray]:
        """从bytes解码图像"""
        try:
            jpeg_start = data.find(b'\xff\xd8')
            if jpeg_start != -1:
                jpeg_end = data.rfind(b'\xff\xd9')
                if jpeg_end > jpeg_start:
                    img_array = np.frombuffer(data[jpeg_start:jpeg_end+2], np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if image is not None:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
        return None
    
    def _decode_image_msg(self, msg) -> Optional[np.ndarray]:
        """解码图像（rosbag）"""
        try:
            if hasattr(msg, 'data'):
                data = msg.data
                if isinstance(data, str):
                    data = data.encode('latin-1')
                elif not isinstance(data, bytes):
                    return None
                return self._decode_image_msg_from_bytes(data)
        except:
            pass
        return None
    
    def _process_single_frame(self, args):
        """处理单帧数据（用于并行处理）"""
        idx, img_idx, pc_idx, image_dir, velodyne_dir = args
        
        # 复制图像
        src_img = Path(self.image_metadata[img_idx].file_path)
        dst_img = image_dir / f"{idx:06d}.png"
        img = Image.open(src_img)
        img.save(dst_img)
        
        # 读取并去畸变点云
        src_pc = Path(self.pc_metadata[pc_idx].file_path)
        dst_pc = velodyne_dir / f"{idx:06d}.bin"
        
        # 读取原始点云
        points_data = np.fromfile(str(src_pc), dtype=np.float32)
        
        # 判断是否有timestamp（5列）
        if len(points_data) % 5 == 0:
            # (N, 5): x, y, z, intensity, timestamp
            points_raw = points_data.reshape(-1, 5)
            
            if self.pose_metadata and len(self.pose_metadata) > 0:
                # 有pose数据，应用去畸变
                # ✅ 去畸变原理：将点云从激光雷达扫描时刻(cloud_ts)转换到图像时刻(target_ts)
                # ✅ 去畸变后的点云在空间上对应于图像时刻，消除了运动畸变
                cloud_ts = self.pc_metadata[pc_idx].timestamp  # LiDAR扫描开始时刻
                target_ts = self.image_metadata[img_idx].timestamp  # 图像曝光时刻（目标对齐时刻）
                
                # 🔍 添加调试信息
                if idx == 0:  # 只在第一帧打印
                    print(f"\n🔍 去畸变调试信息（Frame {idx}） - 完全对齐C++:")
                    print(f"  cloud_ts: {cloud_ts:.6f}")
                    print(f"  target_ts: {target_ts:.6f}")
                    print(f"  时间差: {abs(target_ts - cloud_ts)*1000:.1f}ms")
                    print(f"  假设：LiDAR系 = Sensing系 (iso_vehicle_lidar = Identity)")
                    print(f"  poses是Sensing→World变换（Sensing在World系中的位姿）")
                
                # 点云去畸变（poses已经是Sensing系，LiDAR系=Sensing系）
                points_undistorted = UndistortionUtils.undistort_pointcloud(
                    points_raw, cloud_ts, target_ts, self.pose_metadata,
                    debug=(idx == 0)  # 只在第一帧启用详细调试
                )
                
                # 🔍 打印去畸变前后的点云范围
                if idx == 0:
                    print(f"\n  去畸变前点云范围:")
                    print(f"    X: [{points_raw[:, 0].min():.2f}, {points_raw[:, 0].max():.2f}]")
                    print(f"    Y: [{points_raw[:, 1].min():.2f}, {points_raw[:, 1].max():.2f}]")
                    print(f"    Z: [{points_raw[:, 2].min():.2f}, {points_raw[:, 2].max():.2f}]")
                    print(f"  去畸变后点云范围:")
                    print(f"    X: [{points_undistorted[:, 0].min():.2f}, {points_undistorted[:, 0].max():.2f}]")
                    print(f"    Y: [{points_undistorted[:, 1].min():.2f}, {points_undistorted[:, 1].max():.2f}]")
                    print(f"    Z: [{points_undistorted[:, 2].min():.2f}, {points_undistorted[:, 2].max():.2f}]")
                    
                    # 检查点云是否有异常大的变化
                    max_shift = np.abs(points_undistorted[:, :3] - points_raw[:, :3]).max()
                    print(f"  最大位移: {max_shift:.2f}m")
                    if max_shift > 10.0:
                        print(f"  ⚠️  警告：位移过大（>{max_shift:.1f}m），可能存在问题！")
            else:
                # 没有pose数据，直接使用前4列
                points_undistorted = points_raw[:, :4]
                print(f"  ⚠️  警告：没有pose数据, 数据异常，直接使用前4列")
        elif len(points_data) % 4 == 0:
            # (N, 4): x, y, z, intensity，无timestamp
            points_raw = points_data.reshape(-1, 4)
            points_undistorted = points_raw
        else:
            return False  # 格式异常
        
        # ✅ 关键修复：将点云从Sensing系转换到LiDAR系
        # KITTI-Odometry标准要求点云在LiDAR系，这样投影时可以直接使用Tr（LiDAR→Camera）
        # 变换链：Sensing系 → LiDAR系
        # P_lidar = T_lidar_to_sensing @ P_sensing
        
        # 转换点云到LiDAR系（使用C++命名约定）
        points_homo = np.hstack([points_undistorted[:, :3], np.ones((points_undistorted.shape[0], 1))])
        points_lidar = (self.T_lidar_to_sensing @ points_homo.T).T  # (N, 4)
        
        # 组合为最终点云：[x, y, z, intensity]（LiDAR系）
        points_final = np.hstack([points_lidar[:, :3], points_undistorted[:, 3:4]])
        
        if idx == 0 and self.verbose:
            print(f"\n  转换到LiDAR系:")
            print(f"    去畸变后（Sensing系）范围:")
            print(f"      X: [{points_undistorted[:, 0].min():.2f}, {points_undistorted[:, 0].max():.2f}]")
            print(f"      Y: [{points_undistorted[:, 1].min():.2f}, {points_undistorted[:, 1].max():.2f}]")
            print(f"      Z: [{points_undistorted[:, 2].min():.2f}, {points_undistorted[:, 2].max():.2f}]")
            print(f"    转换后（LiDAR系）范围:")
            print(f"      X: [{points_final[:, 0].min():.2f}, {points_final[:, 0].max():.2f}]")
            print(f"      Y: [{points_final[:, 1].min():.2f}, {points_final[:, 1].max():.2f}]")
            print(f"      Z: [{points_final[:, 2].min():.2f}, {points_final[:, 2].max():.2f}]")
        
        # 保存点云（LiDAR系，符合KITTI标准）
        # 格式：(N, 4) = [x, y, z, intensity]
        points_final.astype(np.float32).tofile(str(dst_pc))
        return True
    
    def sync_and_save(self, sequence_id: str = "00"):
        """同步并生成最终数据集（参考C++的GetSyncCalibrationData逻辑）
        
        参考：lidar_online_calibrator.cpp:525-744
        - 严格使用motion_interpolate进行位姿插值
        - 如果插值失败，跳过该帧（不做时间戳校正）
        - 使用时间差阈值进行数据对齐（kMaxLidarCameraDelta=55ms）
        
        🎯 时间对齐原理：
            1. 每一帧数据的逻辑时刻 = 图像曝光时刻 (image_timestamp)
            2. 点云通过去畸变从LiDAR扫描时刻转换到图像时刻
            3. 位姿使用motion_interpolate插值到图像时刻
            4. 最终：图像、点云、位姿 在图像时刻完全对齐
        """
        print(f"\n{'='*80}")
        print(f"阶段 2/3: 同步数据")
        print(f"{'='*80}")
        
        sync_start_time = time.time()
        
        if not self.image_metadata or not self.pc_metadata:
            raise ValueError("图像或点云数据为空")
        
        # 打印时间戳范围用于诊断
        if self.pose_metadata:
            data_ts_min = min(self.image_metadata[0].timestamp, self.pc_metadata[0].timestamp)
            data_ts_max = max(self.image_metadata[-1].timestamp, self.pc_metadata[-1].timestamp)
            pose_ts_min = self.pose_metadata[0].timestamp
            pose_ts_max = self.pose_metadata[-1].timestamp
            
            print(f"\n时间戳范围:")
            print(f"  数据: [{data_ts_min:.3f}, {data_ts_max:.3f}]")
            print(f"  位姿: [{pose_ts_min:.3f}, {pose_ts_max:.3f}]")
            
            if data_ts_min < pose_ts_min or data_ts_max > pose_ts_max:
                print(f"  ⚠️  警告: 数据时间戳超出位姿范围，这些帧将跳过去畸变")
                print(f"     参考C++实现，motion_interpolate失败时会跳过该帧")
        
        # 创建最终目录
        seq_dir = self.output_dir / 'sequences' / sequence_id
        seq_dir.mkdir(parents=True, exist_ok=True)
        image_dir = seq_dir / 'image_2'
        velodyne_dir = seq_dir / 'velodyne'
        image_dir.mkdir(exist_ok=True)
        velodyne_dir.mkdir(exist_ok=True)
        
        # ========================================================================
        # 同步图像和点云 - 完全对齐C++ get_sync_obstacles_inner实现
        # ========================================================================
        synced_pairs = []
        rejected_pairs = 0  # 统计因时间差过大而被拒绝的对数
        rejected_details = []  # 存储拒绝配对的详细信息
        
        img_idx = 0
        pc_idx = 0
        
        print(f"\n开始同步（对齐C++实现）:")
        print(f"  图像数量: {len(self.image_metadata)}")
        print(f"  点云数量: {len(self.pc_metadata)}")
        print(f"  同步阈值: {self.max_time_diff*1000:.0f}ms (kMaxLidarCameraDelta)")
        
        with tqdm(total=min(len(self.image_metadata), len(self.pc_metadata)),
                 desc="同步数据") as pbar:
            while img_idx < len(self.image_metadata) and pc_idx < len(self.pc_metadata):
                # 获取当前图像和点云的时间戳（微秒）
                image_time = self.image_metadata[img_idx].timestamp
                cloud_time = self.pc_metadata[pc_idx].timestamp
                
                # ✅ C++逻辑：找到最大时间戳作为参考点
                max_stamp = max(image_time, cloud_time)
                stamp_lower_bound = max_stamp - self.max_time_diff
                
                # ✅ C++逻辑：移除过旧的数据（时间戳 < max_stamp - kMaxLidarCameraDelta）
                is_data_sync = True
                
                if image_time < stamp_lower_bound:
                    # 图像过旧，跳过
                    img_idx += 1
                    is_data_sync = False
                    pbar.update(1)
                    continue
                
                if cloud_time < stamp_lower_bound:
                    # 点云过旧，跳过
                    pc_idx += 1
                    is_data_sync = False
                    pbar.update(1)
                    continue
                
                # ✅ C++逻辑：检查图像和点云的时间差
                lidar_camera_delta = abs(image_time - cloud_time)
                
                if lidar_camera_delta <= self.max_time_diff:
                    # 时间差在阈值内，配对成功
                    synced_pairs.append((img_idx, pc_idx))
                    img_idx += 1
                    pc_idx += 1
                    pbar.update(1)
                else:
                    # 时间差过大，移除时间戳较小的那个
                    rejected_pairs += 1
                    delta_ms = lidar_camera_delta * 1000
                    exceed_ms = delta_ms - self.max_time_diff * 1000
                    rejected_details.append((img_idx, pc_idx, delta_ms, exceed_ms))
                    
                    if image_time < cloud_time:
                        img_idx += 1
                    else:
                        pc_idx += 1
                    pbar.update(1)
        
        # 关键：过滤掉时间戳在pose范围之外的帧（参考您的指示）
        if self.pose_metadata:
            pose_ts_min = self.pose_metadata[0].timestamp
            pose_ts_max = self.pose_metadata[-1].timestamp
            
            synced_pairs_filtered = []
            for img_idx, pc_idx in synced_pairs:
                pc_ts = self.pc_metadata[pc_idx].timestamp
                # 检查点云时间戳是否在pose范围内
                # 注意：还要考虑点云扫描的end_timestamp
                temp_pc_file = self.temp_dir / 'pointclouds' / f'{pc_idx:06d}.bin'
                if temp_pc_file.exists():
                    points_raw = np.fromfile(temp_pc_file, dtype=np.float32).reshape(-1, 5)
                    if points_raw.shape[0] > 0 and points_raw.shape[1] >= 5:
                        max_inner_ts_us = points_raw[:, 4].max()
                        delta_time_us = max_inner_ts_us * 2
                        end_ts = pc_ts + delta_time_us * 1e-6
                        
                        # start, end, target 都需要在pose范围内
                        img_ts = self.image_metadata[img_idx].timestamp
                        if (pc_ts >= pose_ts_min and pc_ts <= pose_ts_max and
                            end_ts >= pose_ts_min and end_ts <= pose_ts_max and
                            img_ts >= pose_ts_min and img_ts <= pose_ts_max):
                            synced_pairs_filtered.append((img_idx, pc_idx))
            
            if len(synced_pairs_filtered) < len(synced_pairs):
                print(f"\n  ⚠️  过滤时间戳超出pose范围的帧: {len(synced_pairs)} → {len(synced_pairs_filtered)}")
            synced_pairs = synced_pairs_filtered
        
        # ✅ 可选：基于target_fps进行降采样（如果同步帧数过多）
        if len(synced_pairs) > 0 and self.target_fps is not None:
            # 计算原始帧率
            if len(synced_pairs) >= 2:
                first_img_idx, first_pc_idx = synced_pairs[0]
                last_img_idx, last_pc_idx = synced_pairs[-1]
                # 时间跨度（last_idx >= first_idx 总是成立，所以不需要abs）
                time_span = (self.image_metadata[last_img_idx].timestamp - 
                            self.image_metadata[first_img_idx].timestamp)
                if time_span > 0:  # 防止除零（极端情况：所有帧时间戳相同）
                    original_fps = (len(synced_pairs) - 1) / time_span
                    
                    # 🔍 调试信息：打印降采样判断的详细信息
                    print(f"\n  降采样判断:")
                    print(f"    同步帧数: {len(synced_pairs)}")
                    print(f"    时间跨度: {time_span:.3f}s")
                    print(f"    原始帧率: {original_fps:.2f} fps")
                    print(f"    目标帧率: {self.target_fps:.1f} fps")
                    
                    # 如果原始帧率 > target_fps，进行降采样
                    if original_fps > self.target_fps:
                        target_interval = 1.0 / self.target_fps
                        downsampled_pairs = []
                        last_selected_time = -float('inf')
                        
                        for img_idx, pc_idx in synced_pairs:
                            current_time = self.image_metadata[img_idx].timestamp
                            if current_time - last_selected_time >= target_interval:
                                downsampled_pairs.append((img_idx, pc_idx))
                                last_selected_time = current_time
                        
                        if len(downsampled_pairs) < len(synced_pairs):
                            print(f"\n  📉 降采样: {len(synced_pairs)} → {len(downsampled_pairs)} 帧 (目标: {self.target_fps:.1f} fps)")
                            synced_pairs = downsampled_pairs
                    else:
                        print(f"    ✓ 无需降采样（原始帧率 <= 目标帧率）")
        
        # 打印同步统计（参考C++ kMaxLidarCameraDelta）
        print(f"\n同步统计:")
        print(f"  成功配对: {len(synced_pairs)} 帧")
        if rejected_pairs > 0:
            print(f"  拒绝配对: {rejected_pairs} 对（lidar-camera时间差 > {self.max_time_diff*1000:.0f}ms）")
            # 打印每个被拒绝配对的详细信息
            for img_idx, pc_idx, delta_ms, exceed_ms in rejected_details[:10]:  # 最多显示前10个
                print(f"    - 图像#{img_idx:04d} & 点云#{pc_idx:04d}: Δt={delta_ms:.1f}ms (超出{exceed_ms:.1f}ms)")
            if len(rejected_details) > 10:
                print(f"    ... 还有 {len(rejected_details)-10} 个被拒绝的配对")
        
        # 限制处理帧数（用于测试）
        if self.max_frames is not None and len(synced_pairs) > self.max_frames:
            print(f"  ⚠️  限制处理帧数: {len(synced_pairs)} → {self.max_frames}")
            synced_pairs = synced_pairs[:self.max_frames]
        
        sync_time = time.time() - sync_start_time
        
        print(f"\n✓ 同步完成:")
        print(f"  同步对数: {len(synced_pairs)}")
        print(f"  耗时: {timedelta(seconds=int(sync_time))}")
        print(f"  速度: {len(synced_pairs) / sync_time:.2f} 对/秒")
        
        # 保存最终数据集（并行处理）
        print(f"\n{'='*80}")
        print(f"阶段 3/3: 去畸变并保存 (使用 {self.num_workers} 个线程)")
        print(f"{'='*80}")
        
        save_start_time = time.time()
        
        # 准备并行任务参数
        tasks = [
            (idx, img_idx, pc_idx, image_dir, velodyne_dir)
            for idx, (img_idx, pc_idx) in enumerate(synced_pairs)
        ]
        
        # 使用线程池并行处理（I/O密集型任务，线程池更合适）
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_single_frame, task) for task in tasks]
            for future in tqdm(futures, desc="  处理进度", total=len(tasks)):
                results.append(future.result())
        
        save_time = time.time() - save_start_time
        
        failed_count = sum(1 for r in results if not r)
        
        print(f"\n✓ 去畸变和保存完成:")
        print(f"  成功: {len(results) - failed_count} 帧")
        if failed_count > 0:
            print(f"  失败: {failed_count} 帧")
        print(f"  耗时: {timedelta(seconds=int(save_time))}")
        print(f"  速度: {len(results) / save_time:.2f} 帧/秒")
        print(f"  平均: {save_time / len(results):.2f} 秒/帧")
        
        # 保存标定文件
        self._save_calib_file(seq_dir)
        
        # 保存位姿文件
        if self.pose_metadata:
            self._save_poses_file(synced_pairs, sequence_id)
        else:
            print("⚠️  警告: 未找到位姿数据，跳过 poses 文件生成")
        
        # 保存时间戳文件
        self._save_times_file(synced_pairs, seq_dir)
        
        # 清理临时文件（可选）
        # 注意：如果需要验证去畸变效果，请保留temp目录
        # temp/pointclouds/ 中存储的是去畸变前的原始点云
        print(f"\n💡 提示: temp目录包含去畸变前的原始点云")
        print(f"   位置: {self.temp_dir}")
        print(f"   如果需要验证去畸变效果，请保留此目录")
        print(f"   如果不需要，可以手动删除以节省空间")
        # import shutil
        # shutil.rmtree(self.temp_dir)  # 注释掉自动删除
        
        # 计算总耗时
        total_time = getattr(self, '_extract_time', 0) + sync_time + save_time
        
        print(f"\n{'='*80}")
        print(f"数据集生成完成！")
        print(f"{'='*80}")
        print(f"\n📁 输出位置:")
        print(f"  数据集: {seq_dir}")
        print(f"  临时文件: {self.temp_dir} (已保留)")
        
        print(f"\n📊 统计信息:")
        print(f"  图像: {len(synced_pairs)} 张")
        print(f"  点云: {len(synced_pairs)} 帧")
        if self.pose_metadata:
            print(f"  位姿: {len(self.pose_metadata)} 个")
        
        print(f"\n⏱️  耗时统计:")
        if hasattr(self, '_extract_time'):
            print(f"  1. 数据提取: {timedelta(seconds=int(self._extract_time))} ({self._extract_time/total_time*100:.1f}%)")
        print(f"  2. 数据同步: {timedelta(seconds=int(sync_time))} ({sync_time/total_time*100:.1f}%)")
        print(f"  3. 去畸变保存: {timedelta(seconds=int(save_time))} ({save_time/total_time*100:.1f}%)")
        print(f"  {'─'*40}")
        print(f"  总计: {timedelta(seconds=int(total_time))}")
        print(f"  平均速度: {len(synced_pairs) / total_time:.2f} 帧/秒")
        
        print(f"\n🚀 性能指标:")
        print(f"  线程数: {self.num_workers}")
        print(f"  批次大小: {self.batch_size}")
        if hasattr(self, '_extract_time') and len(synced_pairs) > 0:
            print(f"  提取效率: {len(self.image_metadata) / self._extract_time:.2f} 帧/秒")
        print(f"  保存效率: {len(synced_pairs) / save_time:.2f} 帧/秒")
    
    def _save_calib_file(self, seq_dir: Path):
        """保存标定文件（KITTI-Odometry标准格式）
        
        参考：
        - manual_sensor_calib.cpp:1585-1626
        - 坐标系文档：P_A = T^A_B * P_B
        
        **KITTI标准**：
        - Velodyne系：x=forward, y=left, z=up（前左上，与我们的LiDAR系一致）
        - Camera系：x=right, y=down, z=forward（右下前，通用定义）
        - Tr矩阵：**LiDAR → Camera**（将Velodyne点云变换到Camera坐标系）
        
        **坐标变换链**：
        - 点云在LiDAR系：P_lidar（去畸变后已转换到LiDAR系）
        - 变换到Camera系：P_camera = T_lidar_to_camera * P_lidar
        - 投影到图像：p = K * P_camera（P_camera.z > 0时可见）
        
        **关键修复**：
        - 去畸变后的点云已从Sensing系转换到LiDAR系
        - 符合KITTI标准，投影时直接使用Tr（LiDAR→Camera）
        - 不需要额外的扩展字段
        """
        calib_path = seq_dir / 'calib.txt'
        
        # P2: 左侧彩色相机的投影矩阵 (3x4)
        P2 = np.zeros((3, 4))
        P2[:3, :3] = self.K
        
        # Tr: Velodyne到cam0的变换矩阵（LiDAR → Camera）
        # C++命名约定：T_camera_to_lidar（从右往左读：LiDAR → Camera）
        T_velo_to_cam0 = self.T_camera_to_lidar  # LiDAR → Camera（KITTI标准）
        T_velo_to_cam0_3x4 = T_velo_to_cam0[:3, :]  # 只取前3行（KITTI标准：3x4矩阵）
        
        # D: 畸变系数（参考C++实现，投影时需要）
        # 支持两种模型：
        # 1. pinhole: k1, k2, p1, p2, k3 (OpenCV标准，5个参数)
        # 2. fisheye: k1, k2, k3, k4 (KANNALA_BRANDT鱼眼模型，4个参数)
        distortion = self.camera_config.get('distortion', {})
        model_type = distortion.get('model_type', 'pinhole')
        
        if model_type == 'fisheye':
            # 鱼眼模型：k1, k2, k3, k4（无切向畸变）
            D = np.array([
                distortion.get('k1', 0.0),
                distortion.get('k2', 0.0),
                distortion.get('k3', 0.0),
                distortion.get('k4', 0.0)
            ])
        else:
            # Pinhole模型：k1, k2, p1, p2, k3
            D = np.array([
                distortion.get('k1', 0.0),
                distortion.get('k2', 0.0),
                distortion.get('p1', 0.0),
                distortion.get('p2', 0.0),
                distortion.get('k3', 0.0)
            ])
        
        with open(calib_path, 'w') as f:
            # 完整的KITTI格式包含P0-P3，但BEVCalib只需要P2
            # 为了兼容性，我们也写入P0, P1, P3（使用相同的P2）
            for i in range(4):
                f.write(f"P{i}: ")
                f.write(" ".join([f"{val:.12e}" for val in P2.flatten()]))
                f.write("\n")
            
            # Tr: 3x4矩阵（12个数）
            f.write("Tr: ")
            f.write(" ".join([f"{val:.12e}" for val in T_velo_to_cam0_3x4.flatten()]))
            f.write("\n")
            
            # D: 畸变系数
            # pinhole模型：k1, k2, p1, p2, k3 (5个参数)
            # fisheye模型：k1, k2, k3, k4 (4个参数)
            # C++参考：lidar_cam_fusion_manual需要畸变系数进行去畸变
            f.write("D: ")
            f.write(" ".join([f"{val:.12e}" for val in D]))
            f.write("\n")
            
            # 保存相机模型类型（用于投影时选择正确的去畸变方法）
            f.write(f"camera_model: {model_type}\n")
        
        print(f"标定文件已保存: {calib_path} (KITTI-Odometry格式 + 畸变系数)")
        print(f"  ✓ 点云坐标系: LiDAR系（已从Sensing系转换）")
        print(f"  ✓ 投影变换: Tr (LiDAR→Camera)")
        
        # 根据模型类型打印畸变系数
        if model_type == 'fisheye':
            print(f"  ✓ 畸变模型: {model_type} (KANNALA_BRANDT)")
            print(f"  ✓ 畸变系数: D (k1={D[0]:.6f}, k2={D[1]:.6f}, k3={D[2]:.6f}, k4={D[3]:.6f})")
        else:
            print(f"  ✓ 畸变模型: {model_type}")
            print(f"  ✓ 畸变系数: D (k1={D[0]:.6f}, k2={D[1]:.6f}, p1={D[2]:.6f}, p2={D[3]:.6f}, k3={D[4]:.6f})")
    
    def _save_poses_file(self, synced_pairs: List[Tuple[int, int]], sequence_id: str):
        """保存位姿文件（KITTI-Odometry 格式）
        
        KITTI格式要求：
        1. 每行是从第i帧到第0帧的变换矩阵 T_0_to_i
        2. P_0 = T_0_to_i @ P_i （将第i帧的点投影到第0帧）
        3. 使用线性插值获得图像时刻的精确位姿
        """
        poses_dir = self.output_dir / 'poses'
        poses_dir.mkdir(exist_ok=True)
        poses_file = poses_dir / f'{sequence_id}.txt'
        
        print(f"生成位姿文件...")
        
        # ✅ 复用已有的插值/外推实现
        # 1. 优先使用 motion_interpolate（李代数插值）
        # 2. 插值失败时使用 motion_extrapolate（外推）
        # 3. 都失败时使用最近邻（极端情况）
        poses_world = []
        interpolated_count = 0
        extrapolated_count = 0
        nearest_count = 0
        
        for idx, (img_idx, _) in enumerate(synced_pairs):
            img_ts = self.image_metadata[img_idx].timestamp
            
            # 使用已有的motion_interpolate方法（严格参考C++实现）
            result = UndistortionUtils.motion_interpolate(self.pose_metadata, img_ts)
            
            if result is not None:
                # 插值成功
                R_mat, t_vec = result
                T_world = np.eye(4)
                T_world[:3, :3] = R_mat
                T_world[:3, 3] = t_vec
                poses_world.append(T_world)
                interpolated_count += 1
            else:
                # 插值失败（时间戳超出范围），尝试外推
                result_extrap = UndistortionUtils.motion_extrapolate(self.pose_metadata, img_ts)
                
                if result_extrap is not None:
                    # 外推成功
                    R_mat, t_vec = result_extrap
                    T_world = np.eye(4)
                    T_world[:3, :3] = R_mat
                    T_world[:3, 3] = t_vec
                    poses_world.append(T_world)
                    extrapolated_count += 1
                else:
                    # 外推也失败（极端情况：pose数量<2），使用最近邻
                    best_pose = None
                    best_diff = float('inf')
                    for pose in self.pose_metadata:
                        diff = abs(pose.timestamp - img_ts)
                        if diff < best_diff:
                            best_diff = diff
                            best_pose = pose
                    
                    if best_pose is not None:
                        T_world = np.eye(4)
                        T_world[:3, :3] = R.from_quat(best_pose.orientation).as_matrix()
                        T_world[:3, 3] = best_pose.position
                        poses_world.append(T_world)
                        nearest_count += 1
                    else:
                        # 没有任何pose，使用单位矩阵或上一帧
                        if poses_world:
                            poses_world.append(poses_world[-1].copy())
                        else:
                            poses_world.append(np.eye(4))
        
        # ✅ 修复2：转换为相对于第0帧的位姿
        # KITTI格式：T_0_to_i = inv(T_0_to_world) @ T_i_to_world
        # 
        # C++命名约定（从右往左读）：
        #   T_0_to_world = 第0帧→World（第0帧在World系中的位姿）
        #   T_i_to_world = 第i帧→World（第i帧在World系中的位姿）
        #   T_world_to_0 = inv(T_0_to_world) = World→第0帧
        #   T_0_to_i = World→第0帧 @ 第i帧→World = 第i帧→第0帧
        if len(poses_world) == 0:
            print("警告: 没有有效的位姿数据")
            return
        
        T_0_to_world = poses_world[0]  # 第0帧→World（C++命名约定）
        T_world_to_0 = np.linalg.inv(T_0_to_world)  # World→第0帧
        
        poses_relative = []
        for T_i_to_world in poses_world:  # 第i帧→World
            # 从第i帧到第0帧的变换
            T_0_to_i = T_world_to_0 @ T_i_to_world  # 第i帧→第0帧
            poses_relative.append(T_0_to_i)
        
        # 写入文件（KITTI 格式：每行12个数，代表 3x4 变换矩阵）
        with open(poses_file, 'w') as f:
            for T in poses_relative:
                # 只保存前3行（3x4 矩阵）
                pose_line = T[:3, :].flatten()
                f.write(' '.join([f"{val:.12e}" for val in pose_line]))
                f.write('\n')
        
        print(f"位姿文件已保存: {poses_file}")
        print(f"  总帧数: {len(poses_relative)}")
        print(f"  插值帧数: {interpolated_count} (李代数插值)")
        if extrapolated_count > 0:
            print(f"  外推帧数: {extrapolated_count} (李代数外推)")
        if nearest_count > 0:
            print(f"  最近邻: {nearest_count}")
        
        # 验证第0帧应该是单位矩阵
        if not np.allclose(poses_relative[0], np.eye(4), atol=1e-6):
            print(f"  ⚠️  警告: 第0帧位姿不是单位矩阵（预期行为）")
        else:
            print(f"  ✓ 第0帧位姿为单位矩阵（参考坐标系）")
    
    def _save_times_file(self, synced_pairs: List[Tuple[int, int]], seq_dir: Path):
        """保存时间戳文件（KITTI-Odometry 格式）
        
        KITTI格式要求：
        1. 每行一个浮点数，表示该帧的时间戳（秒）
        2. 时间戳相对于第一帧（第0帧时间戳为0）
        3. 高精度浮点数格式（保留足够精度）
        
        times.txt 技术特点：
        - 确保不同传感器数据之间的时间同步
        - 用于轨迹插值和运动补偿
        - 支持时序数据的精确对齐
        """
        times_file = seq_dir / 'times.txt'
        
        print(f"\n生成时间戳文件...")
        
        if not synced_pairs:
            print("  ⚠️  警告: 没有同步的帧，跳过 times.txt 生成")
            return
        
        # 提取所有图像的时间戳
        timestamps = []
        for img_idx, _ in synced_pairs:
            img_ts = self.image_metadata[img_idx].timestamp
            timestamps.append(img_ts)
        
        # 转换为相对第一帧的时间（KITTI格式）
        first_timestamp = timestamps[0]
        relative_timestamps = [ts - first_timestamp for ts in timestamps]
        
        # 写入文件
        with open(times_file, 'w') as f:
            for ts in relative_timestamps:
                # 使用足够的精度（6位小数，足以表示毫秒级精度）
                f.write(f"{ts:.6f}\n")
        
        print(f"时间戳文件已保存: {times_file}")
        print(f"  总帧数: {len(relative_timestamps)}")
        print(f"  时间范围: {relative_timestamps[0]:.3f}s ~ {relative_timestamps[-1]:.3f}s")
        print(f"  总时长: {relative_timestamps[-1]:.3f}s")
        
        # 计算平均帧率
        if len(relative_timestamps) > 1:
            total_duration = relative_timestamps[-1] - relative_timestamps[0]
            if total_duration > 0:
                avg_fps = (len(relative_timestamps) - 1) / total_duration
                print(f"  平均帧率: {avg_fps:.2f} fps")
        
        # 验证第0帧时间戳应该是0
        if abs(relative_timestamps[0]) > 1e-9:
            print(f"  ⚠️  警告: 第0帧时间戳不是0 ({relative_timestamps[0]:.9f})")
        else:
            print(f"  ✓ 第0帧时间戳为0（参考时刻）")


def main():
    parser = argparse.ArgumentParser(description='准备 BEVCalib 数据集（流式处理）')
    parser.add_argument('--bag_dir', type=str, required=True)
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--camera_name', type=str, default='traffic_2')
    parser.add_argument('--target_fps', type=float, default=10.0)
    parser.add_argument('--max_time_diff', type=float, default=0.055,
                       help='最大时间差阈值（秒）。参考C++: kMaxLidarCameraDelta=55ms。'
                            '超过此阈值的lidar-图像对将被丢弃。')
    parser.add_argument('--lidar_topic', type=str,
                       default='/sensors/lidar/combined_point_cloud_proto')
    parser.add_argument('--pose_topic', type=str,
                       default="/localization/pose",
                       help='位姿 topic（默认: /localization/pose）')
    parser.add_argument('--sequence_id', type=str, default='00')
    parser.add_argument('--batch_size', type=int, default=500,
                       help='批处理大小（默认: 500，增大可提升速度）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行处理的线程数（默认: 4）')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='最大处理帧数，用于测试（默认: None，处理所有帧）')
    args = parser.parse_args()
    
    # 打印配置信息
    print(f"\n{'='*80}")
    print(f"BEVCalib 数据集生成工具")
    print(f"{'='*80}")
    print(f"\n⚙️  配置:")
    print(f"  Bag目录: {args.bag_dir}")
    print(f"  配置目录: {args.config_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  相机: {args.camera_name}")
    print(f"  目标帧率: {args.target_fps} fps")
    print(f"  批次大小: {args.batch_size}")
    print(f"  线程数: {args.num_workers}")
    
    total_start_time = time.time()
    
    preparer = BEVCalibDatasetPreparer(
        bag_path=args.bag_dir,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        camera_name=args.camera_name,
        target_fps=args.target_fps,
        max_time_diff=args.max_time_diff,
        lidar_topic=args.lidar_topic,
        pose_topic=args.pose_topic,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
    )
    
    preparer.extract_data_from_bag()
    preparer.sync_and_save(sequence_id=args.sequence_id)
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"✓ 全部完成！")
    print(f"{'='*80}")
    print(f"\n总用时: {timedelta(seconds=int(total_time))}")
    print(f"\n📚 下一步:")
    print(f"  1. 验证数据集:")
    print(f"     python validate_kitti_odometry.py --dataset_root {args.output_dir}")
    print(f"\n  2. 可视化去畸变效果:")
    print(f"     python visualize_undistortion.py {args.output_dir} --frame 0")
    print(f"\n  3. 投影点云到图像:")
    print(f"     python project_points_to_image.py --dataset_root {args.output_dir} --frame 0")
    print(f"\n  4. 开始训练:")
    print(f"     python kitti-bev-calib/train_kitti.py --dataset_root {args.output_dir}")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
