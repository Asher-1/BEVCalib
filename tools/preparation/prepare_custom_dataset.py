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
import matplotlib
matplotlib.use('Agg')  # 使用无头后端，避免GUI依赖
import matplotlib.pyplot as plt
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
    def motion_interpolate(poses: List[PoseMetadata], timestamp: float, max_gap: float = 1.0):
        """位姿插值（严格参考C++的motion_interpolate实现）
        
        Args:
            poses: 位姿列表（按时间戳排序）
            timestamp: 目标时间戳（秒）
            max_gap: 允许的最大pose间隔（秒），超过此间隔认为数据不连续
            
        Returns:
            (R, t): 旋转矩阵和平移向量，如果失败返回None
            
        Note:
            参考 math_utils.cpp:313-346
            C++版本只支持插值，不支持外推。时间戳超出范围直接返回false。
            当pose间隔超过max_gap时，认为数据不连续，不进行插值。
            精度优先：无法精确插值时返回None，让调用方跳过该帧。
        """
        if not poses or len(poses) < 2:
            return None
        
        # 查找timestamp前后的两个位姿进行插值（严格按照C++逻辑）
        for i in range(len(poses) - 1):
            t1 = poses[i].timestamp
            t2 = poses[i + 1].timestamp
            
            if t1 <= timestamp <= t2:
                if (t2 - t1) > max_gap:
                    return None
                
                alpha = (timestamp - t1) / (t2 - t1)
                break
        else:
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
    def can_interpolate(poses: List[PoseMetadata], timestamp: float, max_gap: float = 1.0) -> bool:
        """检查是否可以对给定时间戳进行位姿插值
        
        Args:
            poses: 位姿列表（按时间戳排序）
            timestamp: 目标时间戳（秒）
            max_gap: 允许的最大pose间隔（秒）
            
        Returns:
            True 如果可以插值，False 否则
            
        Note:
            用于快速检查，避免重复计算插值结果
        """
        if not poses or len(poses) < 2:
            return False
        
        for i in range(len(poses) - 1):
            t1 = poses[i].timestamp
            t2 = poses[i + 1].timestamp
            
            if t1 <= timestamp <= t2:
                # 检查pose间隔是否在允许范围内
                return (t2 - t1) <= max_gap
        
        return False
    
    @staticmethod
    def find_nearest_pose(poses: List[PoseMetadata], timestamp: float) -> Tuple[Optional[int], float]:
        """找到最近的pose（参考C++ manual_sensor_calib.cpp的min_delta逻辑）
        
        Args:
            poses: 位姿列表（按时间戳排序）
            timestamp: 目标时间戳（秒）
            
        Returns:
            (index, delta): 最近pose的索引和时间差（秒），如果没有pose返回(None, inf)
        """
        if not poses:
            return None, float('inf')
        
        min_delta = float('inf')
        best_idx = None
        
        for i, pose in enumerate(poses):
            delta = abs(pose.timestamp - timestamp)
            if delta < min_delta:
                min_delta = delta
                best_idx = i
        
        return best_idx, min_delta
    
    @staticmethod
    def can_interpolate_nearest(poses: List[PoseMetadata], timestamp: float, 
                                max_delta: float = 0.1) -> bool:
        """使用最近邻方式检查是否可以进行位姿插值
        
        参考 C++ manual_sensor_calib.cpp 的 min_delta 逻辑：
        - 找到最近的pose
        - 如果时间差 < max_delta，则认为可以插值
        
        这种方式更适合处理不连续的bag数据，因为：
        1. 不要求pose严格包围目标时间戳
        2. 只要有足够近的pose就可以使用
        
        Args:
            poses: 位姿列表
            timestamp: 目标时间戳（秒）
            max_delta: 最大允许的时间差（秒），参考C++的0.1e9 ns = 100ms
            
        Returns:
            True 如果可以插值
        """
        _, min_delta = UndistortionUtils.find_nearest_pose(poses, timestamp)
        return min_delta <= max_delta
    
    @staticmethod
    def motion_extrapolate(poses: List[PoseMetadata], timestamp: float,
                           max_extrap_time: float = 1.0):
        """位姿外推（用于时间戳超出范围的情况）
        
        Args:
            poses: 位姿列表（按时间戳排序）
            timestamp: 目标时间戳（秒）
            max_extrap_time: 允许的最大外推时间（秒），超过则拒绝外推
            
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
            extrap_dt = t1 - timestamp
        elif timestamp > poses[-1].timestamp:
            # 向后外推：使用后两个pose
            i = len(poses) - 2
            t1 = poses[i].timestamp
            t2 = poses[i + 1].timestamp
            extrap_dt = timestamp - t2
        else:
            # 在范围内，应该使用插值
            return None
        
        if extrap_dt > max_extrap_time:
            return None
        
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
                            debug: bool = False,
                            frame_idx: int = -1) -> np.ndarray:
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
            frame_idx: 帧索引（用于日志）
            
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
            
        ⚠️ 与C++时间戳单位对比：
            - C++: cloud_stamp/target_stamp 是微秒(int64_t)，pose时间戳也是微秒
            - C++: 插值时 pose_data[i].first * 1e-6 转换为秒
            - Python: 所有时间戳已经是秒，无需转换
        """
        if points_raw.shape[1] < 5:
            # 没有timestamp，无法去畸变，直接返回前4列
            return points_raw[:, :4]
        
        if not poses or len(poses) == 0:
            # 没有位姿数据，无法去畸变
            return points_raw[:, :4]
        
        # ✅ 检测点云是否已经在世界坐标系
        # 传感器坐标系的点云范围通常在 ±200m 以内
        # 世界坐标系的点云坐标可能非常大（取决于车辆位置）
        xyz = points_raw[:, :3]
        xyz_range = np.abs(xyz).max()
        
        if xyz_range > 500.0:
            print(f"⚠️  [frame={frame_idx}] 点云坐标范围异常（{xyz_range:.1f}m > 500m），可能解析错误，跳过")
            return None
        
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
            # 插值失败，返回 None 表示该帧应该被跳过
            # C++参考：math_utils.cpp:247-250 - 失败时跳过该帧
            # 注意：不应该返回原始点云，因为没有正确的位姿插值会导致坐标错误
            return None
        
        R_start, t_start = start_pose  # Sensing→World（已转换！）
        R_end, t_end = end_pose
        R_target, t_target = target_pose
        
        # ✅ 位姿合理性检查：检测位姿突变
        # 正常情况下，start/end/target三个位姿应该非常接近（时间跨度通常<100ms）
        # 如果位姿差异过大，说明数据有问题（如位姿跳变、时间戳错误等）
        MAX_POSE_DISPLACEMENT = 5.0  # 最大允许位移（米）- 对应50m/s车速下的100ms
        MAX_POSE_ROTATION = 0.5  # 最大允许旋转（弧度）- 约30度
        
        # 检查start到end的位移
        displacement_se = np.linalg.norm(t_end - t_start)
        # 检查start到target的位移
        displacement_st = np.linalg.norm(t_target - t_start)
        
        # 检查旋转变化（使用旋转向量的模）
        R_delta_se = R_start.T @ R_end
        rotation_se = np.linalg.norm(R.from_matrix(R_delta_se).as_rotvec())
        R_delta_st = R_start.T @ R_target
        rotation_st = np.linalg.norm(R.from_matrix(R_delta_st).as_rotvec())
        
        if displacement_se > MAX_POSE_DISPLACEMENT or displacement_st > MAX_POSE_DISPLACEMENT:
            if debug:
                print(f"⚠️  位姿位移异常！start-end: {displacement_se:.2f}m, start-target: {displacement_st:.2f}m")
            return None
        
        if rotation_se > MAX_POSE_ROTATION or rotation_st > MAX_POSE_ROTATION:
            if debug:
                print(f"⚠️  位姿旋转异常！start-end: {np.degrees(rotation_se):.1f}°, start-target: {np.degrees(rotation_st):.1f}°")
            return None
        
        # 🔍 DEBUG: 打印pose信息（对齐C++日志格式，便于对比）
        if debug:
            print(f"\n{'='*60}")
            print(f"🔍 去畸变调试信息 [frame_idx={frame_idx}]")
            print(f"{'='*60}")
            print(f"  === 时间戳信息 (Python用秒，C++用微秒) ===")
            print(f"  cloud_stamp(s): {cloud_timestamp:.6f}")
            print(f"  cloud_stamp(us): {int(cloud_timestamp * 1e6)}")  # 便于与C++对比
            print(f"  target_stamp(s): {target_timestamp:.6f}")
            print(f"  target_stamp(us): {int(target_timestamp * 1e6)}")
            print(f"  lidar_camera_delta(ms): {(target_timestamp - cloud_timestamp)*1000:.2f}")
            print(f"  ")
            print(f"  === 点云内部时间戳 ===")
            print(f"  max_inner_stamp(2us): {max_inner_timestamp_2us:.0f}")
            print(f"  delta_stamp(us): {delta_time_us:.0f}")
            print(f"  end_stamp(s): {end_timestamp:.6f}")
            print(f"  scan_duration(ms): {delta_time_us * 1e-3:.2f}")
            print(f"  ")
            print(f"  === 位姿范围 ===")
            print(f"  poses.size(): {len(poses)}")
            print(f"  pose_range(s): [{poses[0].timestamp:.6f}, {poses[-1].timestamp:.6f}]")
            print(f"  ")
            print(f"  === 插值位姿 (Sensing→World) ===")
            print(f"  start_pose.t: [{t_start[0]:.4f}, {t_start[1]:.4f}, {t_start[2]:.4f}]")
            print(f"  end_pose.t: [{t_end[0]:.4f}, {t_end[1]:.4f}, {t_end[2]:.4f}]")
            print(f"  target_pose.t: [{t_target[0]:.4f}, {t_target[1]:.4f}, {t_target[2]:.4f}]")
        
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
        
        # ✅ 结果范围验证：检测去畸变异常
        # 合理的传感器数据范围：
        # - 激光雷达通常有效范围 200m
        # - 去畸变不应该显著改变点云的位置，只做微小调整
        # - 异常情况：位姿插值外推时可能产生极端变换
        MAX_REASONABLE_RANGE = 350.0  # 最大合理距离 (米)，长距lidar可达300m+
        MAX_REASONABLE_HEIGHT = 50.0  # 最大合理高度 (米)
        
        x_min, x_max = xyz_undistorted[:, 0].min(), xyz_undistorted[:, 0].max()
        y_min, y_max = xyz_undistorted[:, 1].min(), xyz_undistorted[:, 1].max()
        z_min, z_max = xyz_undistorted[:, 2].min(), xyz_undistorted[:, 2].max()
        
        is_abnormal = (
            x_min < -MAX_REASONABLE_RANGE or x_max > MAX_REASONABLE_RANGE or
            y_min < -MAX_REASONABLE_RANGE or y_max > MAX_REASONABLE_RANGE or
            z_min < -MAX_REASONABLE_HEIGHT or z_max > MAX_REASONABLE_HEIGHT
        )
        
        if is_abnormal:
            print(f"⚠️  去畸变结果异常 [frame={frame_idx}]: "
                  f"X:[{x_min:.1f},{x_max:.1f}] Y:[{y_min:.1f},{y_max:.1f}] Z:[{z_min:.1f},{z_max:.1f}]")
            return None
        
        # 🔍 DEBUG: 打印结果统计（格式对齐C++ MLOG输出，便于逐行对比）
        if debug:
            print(f"  ")
            print(f"  === 运动增量 (对齐C++: delta = inv(start) * end) ===")
            print(f"  delta.translation: [{t_delta[0]:.6f}, {t_delta[1]:.6f}, {t_delta[2]:.6f}]")
            print(f"  v(6d): {v_full}")
            print(f"  v_per_sec(trans): [{v_per_second[3]:.6f}, {v_per_second[4]:.6f}, {v_per_second[5]:.6f}] m/s")
            print(f"  v_per_sec(rot): [{v_per_second[0]:.6f}, {v_per_second[1]:.6f}, {v_per_second[2]:.6f}] rad/s")
            print(f"  ")
            print(f"  === iso_target_start (对齐C++: inv(target) * start) ===")
            print(f"  iso_target_start.translation: [{t_target_start[0]:.6f}, {t_target_start[1]:.6f}, {t_target_start[2]:.6f}]")
            iso_ts_rotvec = R.from_matrix(R_target_start).as_rotvec()
            print(f"  iso_target_start.axis*angle: [{iso_ts_rotvec[0]:.6f}, {iso_ts_rotvec[1]:.6f}, {iso_ts_rotvec[2]:.6f}]")
            print(f"  ")
            
            # ✅ 关键对比：前10个点的逐点去畸变过程（与C++完全对齐）
            print(f"  === First 10 points undistortion detail ===")
            for i in range(min(10, len(xyz))):
                dt_i = ts_us[i] * 2.0e-6  # 秒
                v2_i = v_per_second * dt_i
                R_delta2_i = R.from_rotvec(v2_i[:3]).as_matrix()
                t_delta2_i = v2_i[3:6]
                
                # delta2 = iso_target_start * delta2
                R_comb_i = R_target_start @ R_delta2_i
                t_comb_i = R_target_start @ t_delta2_i + t_target_start
                
                p_raw = xyz[i]
                p_undist = R_comb_i @ p_raw + t_comb_i
                
                print(f"  Point[{i}] ts_2us={ts_us[i]:.0f}"
                      f" dt_sec={dt_i:.6f}"
                      f" raw=[{p_raw[0]:.4f}, {p_raw[1]:.4f}, {p_raw[2]:.4f}]"
                      f" undist=[{p_undist[0]:.4f}, {p_undist[1]:.4f}, {p_undist[2]:.4f}]")
            
            print(f"  ")
            print(f"  === 点云范围 ===")
            print(f"  cloud_raw range: X=[{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]"
                  f" Y=[{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]"
                  f" Z=[{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
            print(f"  cloud_undistorted range: X=[{xyz_undistorted[:, 0].min():.2f}, {xyz_undistorted[:, 0].max():.2f}]"
                  f" Y=[{xyz_undistorted[:, 1].min():.2f}, {xyz_undistorted[:, 1].max():.2f}]"
                  f" Z=[{xyz_undistorted[:, 2].min():.2f}, {xyz_undistorted[:, 2].max():.2f}]")
            
            diff = xyz_undistorted - xyz
            print(f"  ")
            print(f"  === 去畸变位移统计 ===")
            print(f"  Difference (mean ± std):")
            print(f"    X: {diff[:, 0].mean():.6f} ± {diff[:, 0].std():.6f}m, max={np.abs(diff[:, 0]).max():.6f}m")
            print(f"    Y: {diff[:, 1].mean():.6f} ± {diff[:, 1].std():.6f}m, max={np.abs(diff[:, 1]).max():.6f}m")
            print(f"    Z: {diff[:, 2].mean():.6f} ± {diff[:, 2].std():.6f}m, max={np.abs(diff[:, 2]).max():.6f}m")
            max_displacement = np.sqrt((diff**2).sum(axis=1)).max()
            mean_displacement = np.sqrt((diff**2).sum(axis=1)).mean()
            print(f"  最大位移: {max_displacement:.6f}m, 平均位移: {mean_displacement:.6f}m")
            if max_displacement > 1.0:
                print(f"  ⚠️ 警告: 最大位移超过1米，检查是否正常！")
            print(f"{'='*60}\n")
        
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
        """解析 lidars.cfg 文件
        
        cfg字段名 → 内部key名 → 含义:
        - vehicle_to_sensing → 'sensing_to_vehicle' → Sensing→Vehicle
        - sensor_to_lidar   → 'lidar_to_sensing'   → LiDAR→Sensing
        """
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
        
        # 1. 解析vehicle_to_sensing (Sensing→Vehicle)
        s2v_blk = find_block(content, 'vehicle_to_sensing')  # config file field name
        if s2v_blk:
            pos = find_block(s2v_blk, 'position')
            s2v_position = None
            if pos:
                x = re.search(r'x:\s*([-\d.e]+)', pos)
                y = re.search(r'y:\s*([-\d.e]+)', pos)
                z = re.search(r'z:\s*([-\d.e]+)', pos)
                if x and y and z:
                    s2v_position = np.array([float(x.group(1)), float(y.group(1)), float(z.group(1))])
            
            ori = find_block(s2v_blk, 'orientation')
            s2v_orientation = None
            if ori:
                qx = re.search(r'qx:\s*([-\d.e]+)', ori)
                qy = re.search(r'qy:\s*([-\d.e]+)', ori)
                qz = re.search(r'qz:\s*([-\d.e]+)', ori)
                qw = re.search(r'qw:\s*([-\d.e]+)', ori)
                if qx and qy and qz and qw:
                    s2v_orientation = [float(qx.group(1)), float(qy.group(1)), float(qz.group(1)), float(qw.group(1))]
            
            if s2v_position is not None and s2v_orientation is not None:
                result['sensing_to_vehicle'] = {
                    'position': s2v_position,
                    'orientation': s2v_orientation  # [qx, qy, qz, qw]
                }
        
        # 2. 解析sensor_to_lidar (LiDAR→Sensing)
        blk = find_block(content, 'sensor_to_lidar')  # config file field name
        if not blk:
            config_blk = find_block(content, 'config')
            if config_blk:
                blk = find_block(config_blk, 'sensor_to_lidar')  # config file field name
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
        
        # 将LiDAR→Sensing的position和orientation添加到result中
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
        """解析protobuf varint (支持最大10字节/64位)"""
        n, sh = 0, 0
        while pos < len(buf):
            b = buf[pos]
            pos += 1
            n |= (b & 0x7F) << sh
            if not (b & 0x80):
                return n, pos
            sh += 7
            if sh >= 64:
                break
        return n, pos


class PointCloudParser:
    """点云解析器（proto 格式）- 完全对齐Self-Cali-GS实现
    
    ✅ 支持lidar_configs解析，提取BAG中的lidar外参和header frame_id
    """
    
    # DataType 常量（与 PointCloud2.proto 一致）
    _DT_INT8 = 1
    _DT_UINT8 = 2
    _DT_INT16 = 3
    _DT_UINT16 = 4
    _DT_INT32 = 5
    _DT_UINT32 = 6
    _DT_FLOAT32 = 7
    _DT_FLOAT64 = 8
    
    # 静态缓存：避免重复打印lidar_configs日志
    _lidar_configs_logged = False
    
    # 从BAG中提取的首个有效lidar外参（用于点云Sensing→LiDAR坐标转换）
    _lidar_to_sensing_from_bag: Optional[np.ndarray] = None
    _sensing_to_vehicle_from_bag_proto: Optional[np.ndarray] = None
    _bag_lidar_frame_id: Optional[str] = None  # lidar_configs中的frame_id (如 "atx_202")
    
    # BAG消息header中的frame_id（指示点云所在坐标系）
    # == "lidar": 点云在Sensing系（combined）→ 需要Sensing→LiDAR转换
    # != "lidar": 点云已在LiDAR系 → 直接保存
    _header_frame_id: Optional[str] = None
    
    @staticmethod
    def _decode_varint(buf: bytes, pos: int):
        """解析 protobuf varint (支持最大10字节/64位)"""
        n, sh = 0, 0
        while pos < len(buf):
            b = buf[pos]
            pos += 1
            n |= (b & 0x7F) << sh
            if not (b & 0x80):
                return n, pos
            sh += 7
            if sh >= 64:
                break
        return n, pos
    
    @staticmethod
    def _parse_header(data: bytes) -> dict:
        """解析 Header 消息
        
        Proto定义 (header.proto):
          - timestamp_sec: field 1 (double)
          - frame_id: field 9 (string)
        """
        result = {'timestamp_sec': None, 'frame_id': None}
        pos = 0
        
        while pos < len(data):
            tag, pos = PointCloudParser._decode_varint(data, pos)
            if pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 1:  # Fixed64 (double)
                if pos + 8 > len(data):
                    break
                if field_num == 1:  # timestamp_sec
                    result['timestamp_sec'] = struct.unpack_from('<d', data, pos)[0]
                pos += 8
            elif wire == 2:  # Length-delimited
                L, pos = PointCloudParser._decode_varint(data, pos)
                if pos + L > len(data):
                    break
                if field_num == 9:  # frame_id
                    result['frame_id'] = data[pos:pos + L].decode('utf-8', errors='ignore')
                pos += L
            elif wire == 0:  # Varint
                _, pos = PointCloudParser._decode_varint(data, pos)
            elif wire == 5:  # Fixed32
                pos += 4
            else:
                break
        
        return result
    
    @staticmethod
    def _parse_vector3(data: bytes) -> Optional[np.ndarray]:
        """解析 Vector3 消息 (float版本)
        
        Proto定义 (geometry.proto):
          - x: field 1 (float)
          - y: field 2 (float)
          - z: field 3 (float)
        """
        x, y, z = 0.0, 0.0, 0.0
        pos = 0
        
        while pos < len(data):
            tag, pos = PointCloudParser._decode_varint(data, pos)
            if pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 5:  # Fixed32 (float)
                if pos + 4 > len(data):
                    break
                val = struct.unpack_from('<f', data, pos)[0]
                if field_num == 1:
                    x = val
                elif field_num == 2:
                    y = val
                elif field_num == 3:
                    z = val
                pos += 4
            elif wire == 0:  # Varint
                _, pos = PointCloudParser._decode_varint(data, pos)
            elif wire == 2:  # Length-delimited
                L, pos = PointCloudParser._decode_varint(data, pos)
                pos += L
            else:
                break
        
        return np.array([x, y, z], dtype=np.float32)
    
    @staticmethod
    def _parse_quaternion_f(data: bytes) -> Optional[np.ndarray]:
        """解析 Quaternion_f 消息 (float版本)
        
        Proto定义 (geometry.proto):
          - qx: field 1 (float)
          - qy: field 2 (float)
          - qz: field 3 (float)
          - qw: field 4 (float)
        """
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        pos = 0
        
        while pos < len(data):
            tag, pos = PointCloudParser._decode_varint(data, pos)
            if pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 5:  # Fixed32 (float)
                if pos + 4 > len(data):
                    break
                val = struct.unpack_from('<f', data, pos)[0]
                if field_num == 1:
                    qx = val
                elif field_num == 2:
                    qy = val
                elif field_num == 3:
                    qz = val
                elif field_num == 4:
                    qw = val
                pos += 4
            elif wire == 0:  # Varint
                _, pos = PointCloudParser._decode_varint(data, pos)
            elif wire == 2:  # Length-delimited
                L, pos = PointCloudParser._decode_varint(data, pos)
                pos += L
            else:
                break
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    @staticmethod
    def _parse_transformation3(data: bytes) -> Optional[np.ndarray]:
        """解析 Transformation3 消息，返回 4x4 变换矩阵
        
        Proto定义 (geometry.proto):
          - position: field 1 (Vector3)
          - orientation: field 2 (Quaternion_f)
        """
        position = np.zeros(3, dtype=np.float32)
        orientation = np.array([0, 0, 0, 1], dtype=np.float32)  # [qx, qy, qz, qw]
        pos = 0
        
        while pos < len(data):
            tag, pos = PointCloudParser._decode_varint(data, pos)
            if pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 2:  # Length-delimited
                L, pos = PointCloudParser._decode_varint(data, pos)
                if pos + L > len(data):
                    break
                chunk = data[pos:pos + L]
                pos += L
                
                if field_num == 1:  # position
                    position = PointCloudParser._parse_vector3(chunk)
                elif field_num == 2:  # orientation
                    orientation = PointCloudParser._parse_quaternion_f(chunk)
            elif wire == 0:  # Varint
                _, pos = PointCloudParser._decode_varint(data, pos)
            elif wire == 5:  # Fixed32
                pos += 4
            elif wire == 1:  # Fixed64
                pos += 8
            else:
                break
        
        # 构建 4x4 变换矩阵
        T = np.eye(4, dtype=np.float64)
        r = R.from_quat(orientation)  # [qx, qy, qz, qw]
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = position
        
        return T
    
    @staticmethod
    def _parse_lidar_config_single(data: bytes) -> dict:
        """解析单个 LidarConfig.Config 消息
        
        Proto定义 (config.proto):
          - frame_id: field 1 (string)
          - ring_id_start: field 27 (int32)
          - ring_id_end: field 28 (int32)
          - sensor_to_lidar: field 26 (repeated Transformation3)
        """
        result = {
            'frame_id': None,
            'ring_id_start': 0,
            'ring_id_end': 255,
            'lidar_to_sensing': None  # Transformation3
        }
        pos = 0
        
        while pos < len(data):
            tag, pos = PointCloudParser._decode_varint(data, pos)
            if pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 0:  # Varint
                val, pos = PointCloudParser._decode_varint(data, pos)
                if field_num == 27:
                    result['ring_id_start'] = val
                elif field_num == 28:
                    result['ring_id_end'] = val
            elif wire == 2:  # Length-delimited
                L, pos = PointCloudParser._decode_varint(data, pos)
                if pos + L > len(data):
                    break
                chunk = data[pos:pos + L]
                pos += L
                
                if field_num == 1:  # frame_id (string)
                    try:
                        result['frame_id'] = chunk.decode('utf-8', errors='ignore')
                    except:
                        pass
                elif field_num == 26:  # proto: sensor_to_lidar (LiDAR→Sensing)
                    if result['lidar_to_sensing'] is None:
                        result['lidar_to_sensing'] = PointCloudParser._parse_transformation3(chunk)
            elif wire == 5:  # Fixed32
                pos += 4
            elif wire == 1:  # Fixed64
                pos += 8
            else:
                break
        
        return result
    
    @staticmethod
    def _is_main_lidar(frame_id: str, main_lidar_suffix: str = "_202") -> bool:
        """判断是否是主lidar
        
        Args:
            frame_id: lidar的frame_id (如 "atx_202", "hesai_202", "robosense_202")
            main_lidar_suffix: 主lidar的后缀标识（默认"_202"）
        
        Returns:
            bool: 如果frame_id以main_lidar_suffix结尾，返回True
        
        示例:
            _is_main_lidar("atx_202")        → True
            _is_main_lidar("hesai_202")      → True
            _is_main_lidar("robosense_202")  → True
            _is_main_lidar("atx_203")        → False
        """
        return frame_id is not None and frame_id.endswith(main_lidar_suffix)
    
    @staticmethod
    def _parse_lidar_configs(data: bytes, main_lidar_frame_id: str = "atx_202") -> dict:
        """解析 LidarConfig 消息
        
        Proto定义 (config.proto):
          - vehicle_to_sensing: field 1 (Transformation3)
          - config: field 2 (repeated Config)
        
        Args:
            data: protobuf消息bytes数据
            main_lidar_frame_id: 主lidar的标识，支持两种模式：
                               - 精确匹配: "atx_202" (完整frame_id)
                               - 后缀匹配: "_202" (以_开头，匹配所有_202结尾的lidar)
                               默认"atx_202"（兼容旧版本）
        
        返回:
          {
            'sensing_to_vehicle': 4x4 ndarray (Sensing→Vehicle),
            'configs': [
              {'frame_id': str, 'ring_id_start': int, 'ring_id_end': int, 'lidar_to_sensing': 4x4 ndarray},
              ...
            ]
          }
        
        注意：
          如果存在多个lidar，只保留主lidar的配置
          - 后缀模式: main_lidar_frame_id="_202" → 保留所有以_202结尾的lidar
          - 精确模式: main_lidar_frame_id="atx_202" → 只保留atx_202
        """
        result = {
            'sensing_to_vehicle': None,
            'configs': []
        }
        pos = 0
        
        # 判断是后缀匹配还是精确匹配
        is_suffix_mode = main_lidar_frame_id is not None and main_lidar_frame_id.startswith("_")
        
        while pos < len(data):
            tag, pos = PointCloudParser._decode_varint(data, pos)
            if pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 2:  # Length-delimited
                L, pos = PointCloudParser._decode_varint(data, pos)
                if pos + L > len(data):
                    break
                chunk = data[pos:pos + L]
                pos += L
                
                if field_num == 1:  # proto: vehicle_to_sensing (Sensing→Vehicle)
                    result['sensing_to_vehicle'] = PointCloudParser._parse_transformation3(chunk)
                elif field_num == 2:  # config (repeated)
                    config = PointCloudParser._parse_lidar_config_single(chunk)
                    frame_id = config.get('frame_id')
                    
                    # ✅ 过滤：只保留主lidar的配置
                    should_keep = False
                    if main_lidar_frame_id is None:
                        should_keep = True
                    elif is_suffix_mode:
                        # 后缀匹配模式: frame_id 以指定后缀结尾
                        should_keep = PointCloudParser._is_main_lidar(frame_id, main_lidar_frame_id)
                    else:
                        # 精确匹配模式: frame_id 完全相同
                        should_keep = (frame_id == main_lidar_frame_id)
                    
                    if should_keep:
                        result['configs'].append(config)
            elif wire == 0:  # Varint
                _, pos = PointCloudParser._decode_varint(data, pos)
            elif wire == 5:  # Fixed32
                pos += 4
            elif wire == 1:  # Fixed64
                pos += 8
            else:
                break
        
        return result
    
    @staticmethod
    def _extract_frame_id_and_lidar_configs(data: bytes, main_lidar_frame_id: str = "atx_202") -> Tuple[Optional[str], Optional[dict]]:
        """从 PointCloud2 消息中提取 frame_id 和 lidar_configs
        
        Proto定义 (pointcloud2.proto):
          - header: field 1 (Header)
          - lidar_configs: field 12 (LidarConfig)
        
        Args:
            data: protobuf消息bytes数据
            main_lidar_frame_id: 主lidar的标识，支持两种模式：
                               - 精确匹配: "atx_202" (完整frame_id)
                               - 后缀匹配: "_202" (以_开头，匹配所有_202结尾的lidar)
                               默认"atx_202"（兼容旧版本）
        
        ⚠️ 重要：lidar_configs (field 12) 通常在消息末尾（在data blob之后）
        
        返回: (frame_id, lidar_configs)
        
        注意：如果存在多个lidar，只保留主lidar的配置
          - 后缀模式: main_lidar_frame_id="_202" → 保留所有以_202结尾的lidar
          - 精确模式: main_lidar_frame_id="atx_202" → 只保留atx_202
        """
        frame_id = None
        lidar_configs = None
        
        # 策略1：从开头解析找 frame_id（在header中，通常在前100字节）
        pos = 0
        max_header_search = min(len(data), 500)  # 只搜索前500字节找header
        
        while pos < max_header_search:
            try:
                tag, new_pos = PointCloudParser._decode_varint(data, pos)
            except:
                pos += 1
                continue
            if new_pos >= len(data):
                break
            field_num, wire = tag >> 3, tag & 7
            
            if wire == 2:  # Length-delimited
                L, new_pos = PointCloudParser._decode_varint(data, new_pos)
                if new_pos + L > len(data):
                    break
                chunk = data[new_pos:new_pos + L]
                
                if field_num == 1:  # header
                    header = PointCloudParser._parse_header(chunk)
                    frame_id = header.get('frame_id')
                    if frame_id:
                        break  # 找到frame_id后停止
                pos = new_pos + L
            elif wire == 0:  # Varint
                _, pos = PointCloudParser._decode_varint(data, new_pos)
            elif wire == 5:  # Fixed32
                pos = new_pos + 4
            elif wire == 1:  # Fixed64
                pos = new_pos + 8
            else:
                pos += 1  # 跳过无效字节继续搜索
        
        # 策略2：从末尾搜索找 lidar_configs
        # field 12, wire type 2 的 tag 是 (12 << 3) | 2 = 98 = 0x62
        # lidar_configs 通常在消息最后 200 字节内
        search_start = max(0, len(data) - 500)  # 从末尾500字节开始搜索
        
        for i in range(search_start, len(data) - 10):
            if data[i] == 0x62:  # 可能是 field 12 tag
                try:
                    # 验证这是否是有效的 field 12
                    length, next_pos = PointCloudParser._decode_varint(data, i + 1)
                    # lidar_configs 长度通常在 50-500 字节
                    if 20 < length < 1000 and next_pos + length <= len(data):
                        chunk = data[next_pos:next_pos + length]
                        # 尝试解析为 lidar_configs（只保留主lidar配置）
                        parsed = PointCloudParser._parse_lidar_configs(chunk, main_lidar_frame_id)
                        # 验证解析结果是否有效
                        if parsed and (parsed.get('sensing_to_vehicle') is not None or parsed.get('configs')):
                            lidar_configs = parsed
                            break
                except:
                    continue
        
        return frame_id, lidar_configs
    
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
        """从 buffer 中读取 x, y, z（单点版本，保留兼容性）"""
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
    def _parse_points_fast_numpy(raw: bytes, step: int, fields_map: dict, max_points: int = 500000) -> Optional[np.ndarray]:
        """使用NumPy向量化操作快速解析点云数据
        
        ✅ 性能关键函数：使用numpy structured array和向量化操作
        比Python循环快10-100倍
        """
        n = len(raw) // step
        if n < 50:
            return None
        
        n = min(n, max_points)
        data_len = n * step
        
        # 获取字段信息
        try:
            x_off, x_fmt, x_scale = fields_map['x']
            y_off, y_fmt, y_scale = fields_map['y']
            z_off, z_fmt, z_scale = fields_map['z']
        except KeyError:
            return None
        
        # 格式到numpy dtype的映射
        fmt_to_dtype = {
            'h': np.int16,   # signed short (2 bytes)
            'H': np.uint16,  # unsigned short (2 bytes)
            'i': np.int32,   # signed int (4 bytes)
            'I': np.uint32,  # unsigned int (4 bytes)
            'f': np.float32, # float (4 bytes)
            'd': np.float64, # double (8 bytes)
        }
        
        dtype_sizes = {'h': 2, 'H': 2, 'i': 4, 'I': 4, 'f': 4, 'd': 8}
        
        try:
            # 将原始数据转换为numpy数组
            raw_array = np.frombuffer(raw[:data_len], dtype=np.uint8)
            
            # 重塑为 (n, step) 的2D数组，每行一个点
            if len(raw_array) < n * step:
                return None
            points_raw = raw_array[:n * step].reshape(n, step)
            
            # 提取x, y, z（使用视图避免复制）
            x_dtype = fmt_to_dtype.get(x_fmt, np.float32)
            y_dtype = fmt_to_dtype.get(y_fmt, np.float32)
            z_dtype = fmt_to_dtype.get(z_fmt, np.float32)
            x_size = dtype_sizes.get(x_fmt, 4)
            y_size = dtype_sizes.get(y_fmt, 4)
            z_size = dtype_sizes.get(z_fmt, 4)
            
            # 使用contiguous array来提取数据（关键性能优化）
            x_bytes = np.ascontiguousarray(points_raw[:, x_off:x_off+x_size])
            y_bytes = np.ascontiguousarray(points_raw[:, y_off:y_off+y_size])
            z_bytes = np.ascontiguousarray(points_raw[:, z_off:z_off+z_size])
            
            x_data = x_bytes.view(x_dtype).flatten().astype(np.float32) * x_scale
            y_data = y_bytes.view(y_dtype).flatten().astype(np.float32) * y_scale
            z_data = z_bytes.view(z_dtype).flatten().astype(np.float32) * z_scale
            
            # ✅ 关键修复：对齐C++ PointXYZIBT/PointXYZIRT 结构体
            # 参考 /home/ludahai/codetree/repo/common/common/point.h:
            # struct PointXYZIBT {
            #   float x, y, z;           // 12 bytes (offset 0, 4, 8)
            #   uint8_t intensity;       // 1 byte (offset 12)
            #   uint8_t ring;            // 1 byte (offset 13)
            #   uint16_t timestamp;      // 2 bytes (offset 14) - 单位: 2us
            # };
            # typedef PointXYZIBT PointXYZIRT;
            # 总共 16 bytes
            
            # 提取intensity (offset 12, uint8)
            intensity_data = np.zeros(n, dtype=np.float32)
            if step >= 13:  # 确保有intensity字段
                intensity_data = points_raw[:, 12].astype(np.float32)
            elif step > 6:  # 兼容旧格式
                intensity_data = points_raw[:, 6].astype(np.float32)
            
            # 提取timestamp (offset 14-15, uint16, 单位: 2us)
            timestamp_data = np.zeros(n, dtype=np.float32)
            if step >= 16:  # PointXYZIBT/PointXYZIRT格式 (16 bytes)
                ts_bytes = np.ascontiguousarray(points_raw[:, 14:16])
                timestamp_data = ts_bytes.view(np.uint16).flatten().astype(np.float32)
            elif step >= 10:  # 兼容旧格式
                ts_bytes = np.ascontiguousarray(points_raw[:, 8:10])
                timestamp_data = ts_bytes.view(np.uint16).flatten().astype(np.float32)
            
            # 过滤无效点（向量化操作）
            valid_mask = (
                np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data) &
                (np.abs(x_data) <= 500) & (np.abs(y_data) <= 500) & (np.abs(z_data) <= 100)
            )
            
            valid_count = valid_mask.sum()
            if valid_count < 50:
                return None
            
            # 组合结果（向量化）
            points = np.column_stack([
                x_data[valid_mask],
                y_data[valid_mask],
                z_data[valid_mask],
                intensity_data[valid_mask],
                timestamp_data[valid_mask]
            ])
            
            return points.astype(np.float32)
            
        except Exception as e:
            return None
    
    @staticmethod
    def parse_proto_pointcloud2(data: bytes, main_lidar_frame_id: str = "atx_202",
                                config_for_comparison: Optional[dict] = None) -> Optional[np.ndarray]:
        """解析 PointCloud2 proto 数据
        
        参考: ~/develop/code/github/Self-Cali-GS/surround_calibration/data/lidar_utils.py
        
        点云坐标系取决于BAG header frame_id：
        - "lidar": 点云在Sensing系（后续由_process_single_frame做Sensing→LiDAR转换）
        - 其他值: 点云已在LiDAR系
        
        Args:
            data: protobuf消息bytes数据
            main_lidar_frame_id: 主lidar的标识，支持两种模式：
                               - 精确匹配: "atx_202" (完整frame_id)
                               - 后缀匹配: "_202" (以_开头，匹配所有_202结尾的lidar，如atx_202, hesai_202等)
                               默认"atx_202"（兼容旧版本，精确匹配）
                               ⚠️ 如果存在多个lidar，只提取和使用主lidar的外参
                               ✅ 设置为None则保留所有lidar配置
            config_for_comparison: 从lidars.cfg读取的配置，用于与bag提取的配置进行对比
        """
        if data is None or len(data) < 16:
            return None
        
        # 提取frame_id和lidar_configs（对齐C++）
        # 优化：_header_frame_id 已缓存后复用，无需每条消息重复提取
        frame_id = PointCloudParser._header_frame_id
        lidar_configs = None
        need_extract = (frame_id is None)
        
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
            
            # 仅首次需要从proto中提取header frame_id和lidar_configs
            if need_extract:
                frame_id, lidar_configs = PointCloudParser._extract_frame_id_and_lidar_configs(to_parse, main_lidar_frame_id)
                if frame_id is not None:
                    PointCloudParser._header_frame_id = frame_id
                    print(f"  📌 BAG header frame_id: '{frame_id}' "
                          f"({'点云在Sensing系，需要转换到LiDAR系' if frame_id == 'lidar' else '点云已在LiDAR系'})")
                    need_extract = False
                else:
                    continue
            
            # 构建 fields_map
            fields_map = {}
            for name, off, dt in flist:
                fmt, scale = PointCloudParser._datatype_to_fmt_scale(dt)
                fields_map[name] = (off, fmt, scale)
            
            # 如果proto字段解析不完整，根据point_step推断正确的格式
            if 'x' not in fields_map or 'y' not in fields_map or 'z' not in fields_map:
                if step == 16:
                    # 16字节标准格式: x(f32,0), y(f32,4), z(f32,8), intensity(u8,12), ring(u8,13), ts(u16,14)
                    fields_map = {'x': (0, 'f', 1.0), 'y': (4, 'f', 1.0), 'z': (8, 'f', 1.0)}
                elif step == 10:
                    fields_map = {'x': (0, 'h', 0.01), 'y': (2, 'h', 0.01), 'z': (4, 'h', 0.01)}
                else:
                    fields_map = {'x': (0, 'h', 0.01), 'y': (2, 'h', 0.01), 'z': (4, 'h', 0.01)}
                if not hasattr(PointCloudParser, '_fallback_warned'):
                    PointCloudParser._fallback_warned = True
                    print(f"  ⚠️  proto字段解析不完整(got {len(flist)} fields)，"
                          f"使用point_step={step}推断格式: "
                          f"{'FLOAT32' if step == 16 else 'INT16'}")
            
            # ✅ 使用向量化快速解析（性能关键优化）
            points = PointCloudParser._parse_points_fast_numpy(raw, step, fields_map)
            if points is not None and len(points) >= 50:
                # 首次解析时打印点云格式诊断信息
                if not hasattr(PointCloudParser, '_format_logged'):
                    PointCloudParser._format_logged = True
                    print(f"\n📊 点云格式诊断:")
                    print(f"  point_step: {step} bytes")
                    print(f"  fields_map: {fields_map}")
                    print(f"  point_count: {len(points)}")
                    if step == 16:
                        print(f"  ✅ 匹配 C++ PointXYZIBT/PointXYZIRT 格式 (16 bytes)")
                        print(f"     structure: x(0-3), y(4-7), z(8-11), intensity(12), ring(13), timestamp(14-15)")
                    elif step == 10:
                        print(f"  ⚠️  旧格式点云 (10 bytes)，timestamp位置可能不同")
                    else:
                        print(f"  ⚠️  非标准点云格式 ({step} bytes)")
                    
                    # 打印前几个点的timestamp值用于验证
                    if points.shape[1] >= 5:
                        ts_sample = points[:5, 4]
                        print(f"  timestamp样本(前5点): {ts_sample}")
                        print(f"  timestamp范围: [{points[:, 4].min():.0f}, {points[:, 4].max():.0f}] (单位: 2us)")
                        print(f"  扫描时长约: {points[:, 4].max() * 2 / 1000:.1f} ms")
                
                # ✅ 新增：打印frame_id和lidar_configs信息（对齐C++日志）
                if not PointCloudParser._lidar_configs_logged:
                    PointCloudParser._lidar_configs_logged = True
                    print(f"\n=== PYTHON_CPP_COMPARE: PointCloud lidar_configs ===")
                    print(f"  frame_id: {frame_id}")
                    print(f"  has_lidar_configs: {'YES' if lidar_configs else 'NO'}")
                    
                    if lidar_configs:
                        s2v_bag = lidar_configs.get('sensing_to_vehicle')
                        if s2v_bag is not None:
                            print(f"\n  【从BAG提取】Sensing→Vehicle:")
                            print(f"    position: [{s2v_bag[0, 3]:.6f}, {s2v_bag[1, 3]:.6f}, {s2v_bag[2, 3]:.6f}]")
                            print(f"    rotation (3x3):")
                            for row_idx in range(3):
                                print(f"      [{s2v_bag[row_idx, 0]:9.6f}, {s2v_bag[row_idx, 1]:9.6f}, {s2v_bag[row_idx, 2]:9.6f}]")
                        
                        # 打印从bag提取的 LiDAR→Sensing
                        configs = lidar_configs.get('configs', [])
                        print(f"\n  config_size: {len(configs)} (filtered: only main_lidar_frame_id='{main_lidar_frame_id}')")
                        for i, cfg in enumerate(configs):
                            l2s_bag = cfg.get('lidar_to_sensing')
                            cfg_frame_id = cfg.get('frame_id', 'unknown')
                            print(f"\n    【从BAG提取】config[{i}]: frame_id='{cfg_frame_id}', ring=[{cfg['ring_id_start']}, {cfg['ring_id_end']})")
                            if l2s_bag is not None:
                                print(f"      LiDAR→Sensing translation: [{l2s_bag[0, 3]:.6f}, {l2s_bag[1, 3]:.6f}, {l2s_bag[2, 3]:.6f}]")
                                print(f"      LiDAR→Sensing rotation (3x3):")
                                for row_idx in range(3):
                                    print(f"        [{l2s_bag[row_idx, 0]:9.6f}, {l2s_bag[row_idx, 1]:9.6f}, {l2s_bag[row_idx, 2]:9.6f}]")
                        
                        # 如果提供了lidars.cfg配置，进行对比
                        if config_for_comparison is not None:
                            print(f"\n=== 对比 lidars.cfg 配置 ===")
                            
                            # 对比 Sensing→Vehicle
                            if 'sensing_to_vehicle' in config_for_comparison:
                                s2v_cfg = config_for_comparison['sensing_to_vehicle']
                                if 'position' in s2v_cfg and 'orientation' in s2v_cfg:
                                    from scipy.spatial.transform import Rotation as R
                                    pos_cfg = np.array(s2v_cfg['position'])
                                    ori_cfg = np.array(s2v_cfg['orientation'])  # [qx, qy, qz, qw]
                                    T_cfg = np.eye(4)
                                    T_cfg[:3, :3] = R.from_quat(ori_cfg).as_matrix()
                                    T_cfg[:3, 3] = pos_cfg
                                    
                                    print(f"\n  【从LIDARS.CFG】Sensing→Vehicle:")
                                    print(f"    position: [{T_cfg[0, 3]:.6f}, {T_cfg[1, 3]:.6f}, {T_cfg[2, 3]:.6f}]")
                                    print(f"    rotation (3x3):")
                                    for row_idx in range(3):
                                        print(f"      [{T_cfg[row_idx, 0]:9.6f}, {T_cfg[row_idx, 1]:9.6f}, {T_cfg[row_idx, 2]:9.6f}]")
                                    
                                    # 计算差异
                                    if s2v_bag is not None:
                                        pos_diff = np.linalg.norm(T_cfg[:3, 3] - s2v_bag[:3, 3])
                                        rot_diff = np.linalg.norm(T_cfg[:3, :3] - s2v_bag[:3, :3], 'fro')
                                        print(f"\n  【GAP】Sensing→Vehicle:")
                                        print(f"    位置差异 (L2 norm): {pos_diff:.6f} m")
                                        print(f"    旋转差异 (Frobenius norm): {rot_diff:.6f}")
                                        print(f"    位置分量差异: [{T_cfg[0,3]-s2v_bag[0,3]:.6f}, {T_cfg[1,3]-s2v_bag[1,3]:.6f}, {T_cfg[2,3]-s2v_bag[2,3]:.6f}]")
                            
                            # 对比 LiDAR→Sensing
                            if 'position' in config_for_comparison and 'orientation' in config_for_comparison:
                                from scipy.spatial.transform import Rotation as R
                                pos_cfg = np.array(config_for_comparison['position'])
                                ori_cfg = np.array(config_for_comparison['orientation'])  # [qx, qy, qz, qw]
                                T_l2s_cfg = np.eye(4)
                                T_l2s_cfg[:3, :3] = R.from_quat(ori_cfg).as_matrix()
                                T_l2s_cfg[:3, 3] = pos_cfg
                                
                                print(f"\n  【从LIDARS.CFG】LiDAR→Sensing:")
                                print(f"    position: [{T_l2s_cfg[0, 3]:.6f}, {T_l2s_cfg[1, 3]:.6f}, {T_l2s_cfg[2, 3]:.6f}]")
                                print(f"    rotation (3x3):")
                                for row_idx in range(3):
                                    print(f"      [{T_l2s_cfg[row_idx, 0]:9.6f}, {T_l2s_cfg[row_idx, 1]:9.6f}, {T_l2s_cfg[row_idx, 2]:9.6f}]")
                                
                                # 计算差异
                                if configs and l2s_bag is not None:
                                    pos_diff = np.linalg.norm(T_l2s_cfg[:3, 3] - l2s_bag[:3, 3])
                                    rot_diff = np.linalg.norm(T_l2s_cfg[:3, :3] - l2s_bag[:3, :3], 'fro')
                                    print(f"\n  【GAP】LiDAR→Sensing:")
                                    print(f"    位置差异 (L2 norm): {pos_diff:.6f} m")
                                    print(f"    旋转差异 (Frobenius norm): {rot_diff:.6f}")
                                    print(f"    位置分量差异: [{T_l2s_cfg[0,3]-l2s_bag[0,3]:.6f}, {T_l2s_cfg[1,3]-l2s_bag[1,3]:.6f}, {T_l2s_cfg[2,3]-l2s_bag[2,3]:.6f}]")
                
                # 缓存首个有效的BAG lidar外参（供后续Tr矩阵决策使用）
                if PointCloudParser._lidar_to_sensing_from_bag is None and lidar_configs:
                    configs = lidar_configs.get('configs', [])
                    for cfg in configs:
                        l2s = cfg.get('lidar_to_sensing')
                        if l2s is not None:
                            PointCloudParser._lidar_to_sensing_from_bag = l2s.copy()
                            PointCloudParser._bag_lidar_frame_id = cfg.get('frame_id')
                            break
                    s2v = lidar_configs.get('sensing_to_vehicle')
                    if s2v is not None:
                        PointCloudParser._sensing_to_vehicle_from_bag_proto = s2v.copy()
                
                return points
        
        # frame_id始终为None → bag数据无效，不做fallback解析
        if frame_id is None:
            if not hasattr(PointCloudParser, '_invalid_bag_warned'):
                PointCloudParser._invalid_bag_warned = True
                print(f"  ❌ BAG无效: 所有候选偏移均无法提取frame_id，跳过此bag的点云数据")
            return None
        
        # Fallback: 尝试直接按float格式读取（如果wire format失败）
        # 尝试不同的起始偏移量
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
        save_debug_samples: int = 0,  # 保存调试样本数量（未去畸变点云）
        max_pose_gap: float = 0.5,  # 最大允许的pose间隔（秒），用于处理不连续bag数据
        force_config: bool = False,  # 强制使用lidars.cfg外参替代bag外参
        sequence_id: str = "00",  # sequence ID，用于隔离临时目录（并行安全）
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
        self.save_debug_samples = save_debug_samples  # 保存调试样本数量
        self.max_pose_gap = max_pose_gap  # 最大允许的pose间隔
        self.force_config = force_config
        self.sequence_id = sequence_id
        
        # 点云坐标系转换状态（由_update_transforms_from_bag设置）
        self._need_sensing_to_lidar = False  # 是否需要Sensing→LiDAR转换
        self._sensing_to_lidar_transform = None  # Sensing→LiDAR的4x4变换矩阵
        
        # 创建输出目录（temp目录按sequence_id隔离，支持并行处理多个trip）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / f'temp_seq{sequence_id}'
        self.temp_dir.mkdir(exist_ok=True)
        self.temp_image_dir = self.temp_dir / 'images'
        self.temp_pc_dir = self.temp_dir / 'pointclouds'
        self.temp_image_dir.mkdir(exist_ok=True)
        self.temp_pc_dir.mkdir(exist_ok=True)
        
        # 元数据列表（不存储实际数据）
        self.image_metadata: List[ImageMetadata] = []
        self.pc_metadata: List[PointCloudMetadata] = []
        self.pose_metadata: List[PoseMetadata] = []
        
        # bag文件按时间连续性分组的时间段 [(start_ts, end_ts), ...]
        self.bag_segments: List[Tuple[float, float]] = []
        
        # 从bag中提取的 Sensing→Vehicle 变换（优先级高于本地配置）
        # 从BAG提取的 Sensing→Vehicle 变换 (cfg字段: vehicle_to_sensing)
        # 注意：必须在_compute_transforms()之前初始化
        self.sensing_to_vehicle_from_bag: Optional[np.ndarray] = None
        
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
    
    def _extract_sensing_to_vehicle_from_pc_msg(self, data: bytes) -> Optional[np.ndarray]:
        """从点云protobuf消息中提取 Sensing→Vehicle 变换
        
        proto字段名: vehicle_to_sensing
        
        参考：lidar_online_calibrator.cpp:1038-1054 + proto_utils.cpp:569-588
        
        Returns:
            T_sensing_to_vehicle: Sensing→Vehicle的4x4变换矩阵，如果提取失败返回None
        """
        try:
            # 手动解析protobuf wire format以查找lidar_configs字段
            # 寻找 Sensing→Vehicle 的position和orientation
            
            # 尝试使用protobuf反序列化（如果消息包含lidar_configs）
            # 注意：我们不需要完整的proto定义，只需要提取特定字段
            
            # 简化方案：尝试搜索特定的字节模式
            # Sensing→Vehicle 包含: position(x,y,z) + orientation(qw,qx,qy,qz)
            
            # 暂时返回None，使用fallback逻辑
            return None
            
        except Exception as e:
            return None
    
    def _get_sensing_to_vehicle_transform(self) -> np.ndarray:
        """获取 Sensing→Vehicle 的变换矩阵
        
        参考：proto_utils.cpp:569-588
        
        cfg字段: vehicle_to_sensing → 内部key: 'sensing_to_vehicle'
        
        优先级：
        1. 从bag中的点云消息提取（self.sensing_to_vehicle_from_bag）
        2. 从本地lidars.cfg的 'sensing_to_vehicle' 读取
        3. 默认单位矩阵（Vehicle == Sensing）
        
        Returns:
            T_sensing_to_vehicle: Sensing→Vehicle的4x4变换矩阵
        """
        if self.sensing_to_vehicle_from_bag is not None and not self.force_config:
            return self.sensing_to_vehicle_from_bag
        
        T_sensing_to_vehicle = np.eye(4)
        
        if isinstance(self.lidar_config, dict) and 'sensing_to_vehicle' in self.lidar_config:
            s2v = self.lidar_config['sensing_to_vehicle']
            if 'position' in s2v and 'orientation' in s2v:
                pos = s2v['position']
                ori = s2v['orientation']  # [qx, qy, qz, qw]
                
                r = R.from_quat(ori)
                T_sensing_to_vehicle[:3, :3] = r.as_matrix()
                T_sensing_to_vehicle[:3, 3] = pos
                
                print(f"✓ 从lidars.cfg读取 Sensing→Vehicle")
                print(f"  position: {pos}")
                print(f"  orientation: {ori}")
                return T_sensing_to_vehicle
        
        print(f"ℹ️  使用默认假设: Vehicle == Sensing (单位矩阵)")
        return T_sensing_to_vehicle
    
    def _update_transforms_from_bag(self):
        """BAG提取后，根据header frame_id决定是否需要点云坐标转换
        
        **架构说明**：
        - calib.txt的Tr始终由lidars.cfg + cameras.cfg合成，不受BAG外参影响
        - BAG的 LiDAR→Sensing 仅用于：当header frame_id == "lidar"时，
          将Sensing系点云转换到LiDAR系
        
        **header frame_id含义**：
        - == "lidar": 点云在Sensing系（combined）→ 需要Sensing→LiDAR转换
        - != "lidar": 点云已在LiDAR系 → 直接保存
        
        **--force-config作用**：
        - 当需要Sensing→LiDAR转换时，控制使用哪个 LiDAR→Sensing 外参：
          - 默认: BAG中的（录制时实际使用的外参）
          - --force-config: lidars.cfg中的
        """
        header_fid = PointCloudParser._header_frame_id
        l2s_bag = PointCloudParser._lidar_to_sensing_from_bag
        bag_cfg_fid = PointCloudParser._bag_lidar_frame_id
        
        self._need_sensing_to_lidar = False
        self._sensing_to_lidar_transform = None
        
        print(f"\n{'─'*60}")
        print(f"  坐标系分析:")
        print(f"    BAG header frame_id: '{header_fid}'")
        print(f"    Tr矩阵: 始终由lidars.cfg + cameras.cfg合成 (Camera→LiDAR)")
        
        if header_fid == "lidar":
            # 点云在Sensing系，需要转换到LiDAR系
            self._need_sensing_to_lidar = True
            
            if self.force_config or l2s_bag is None:
                # 使用lidars.cfg的 LiDAR→Sensing 进行转换
                self._sensing_to_lidar_transform = self.T_sensing_to_lidar.copy()
                source = "lidars.cfg"
                if self.force_config:
                    print(f"    ⚠️  --force-config: 点云Sensing→LiDAR转换使用lidars.cfg外参")
                else:
                    print(f"    ℹ️  BAG中未提取到LiDAR→Sensing外参，使用lidars.cfg进行转换")
            else:
                # 使用BAG的 LiDAR→Sensing 进行转换（录制时实际使用的外参）
                self._sensing_to_lidar_transform = np.linalg.inv(l2s_bag)
                source = "BAG"
                print(f"    ✓ 点云Sensing→LiDAR转换使用BAG外参 (config frame_id='{bag_cfg_fid}')")
            
            print(f"    转换矩阵来源: {source}")
            t = self._sensing_to_lidar_transform[:3, 3]
            print(f"    Sensing→LiDAR平移: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        else:
            print(f"    ✓ 点云已在LiDAR系，无需坐标转换")
        
        # 对比BAG和config的 LiDAR→Sensing（诊断信息）
        if l2s_bag is not None:
            cfg_l2s = self.T_lidar_to_sensing
            pos_diff = np.linalg.norm(cfg_l2s[:3, 3] - l2s_bag[:3, 3])
            rot_diff = np.linalg.norm(cfg_l2s[:3, :3] - l2s_bag[:3, :3], 'fro')
            print(f"    [诊断] BAG vs lidars.cfg LiDAR→Sensing差异: "
                  f"位置={pos_diff:.6f}m, 旋转={rot_diff:.6f}")
        
        # 更新 Sensing→Vehicle（如果BAG中有且合格，用于位姿转换）
        s2v_bag = PointCloudParser._sensing_to_vehicle_from_bag_proto
        if s2v_bag is not None:
            s2v_rot_diff = np.linalg.norm(s2v_bag[:3, :3] - np.eye(3), 'fro')
            s2v_trans_norm = np.linalg.norm(s2v_bag[:3, 3])
            if s2v_rot_diff > 0.001 or s2v_trans_norm > 0.001:
                self.sensing_to_vehicle_from_bag = s2v_bag
                self.T_sensing_to_vehicle = s2v_bag
                print(f"    ✓ 更新 Sensing→Vehicle (从BAG): "
                      f"translation=[{s2v_bag[0, 3]:.6f}, {s2v_bag[1, 3]:.6f}, {s2v_bag[2, 3]:.6f}]")
        print(f"{'─'*60}")
    
    def _convert_poses_to_sensing_frame(self):
        """将Vehicle系的pose转换为Sensing系（对齐C++实现）
        
        C++参考：lidar_online_calibrator.cpp:849-852
            sensing_pose.second = vehicle_pose * iso_vehicle_sensing_;
            lidar_pose_data_.push_back(sensing_pose);
        
        变换理解：
        - 输入：vehicle_pose = Vehicle→World（Vehicle在World系中的位姿）
        - 中间：T_sensing_to_vehicle = Sensing→Vehicle
        - 输出：sensing_pose = (Vehicle→World) @ (Sensing→Vehicle) = Sensing→World
        
        含义：Sensing在World系中的位姿
        
        ⚠️ C++中时间戳单位：
        - C++: pose时间戳是微秒(int64_t)，measurement_time()返回微秒
        - Python: 这里已经转换为秒
        """
        print(f"\n  转换位姿到Sensing坐标系...")
        print(f"  === 位姿坐标系转换 (对齐C++) ===")
        print(f"  T_sensing_to_vehicle (Sensing→Vehicle):")
        print(f"    旋转:\n{self.T_sensing_to_vehicle[:3, :3]}")
        print(f"    平移: {self.T_sensing_to_vehicle[:3, 3]}")
        
        if self.pose_metadata:
            pose0 = self.pose_metadata[0]
            print(f"  转换前(Vehicle系)第一个pose:")
            print(f"    timestamp(s): {pose0.timestamp:.6f}")
            print(f"    timestamp(us): {int(pose0.timestamp * 1e6)}")
            print(f"    position: {pose0.position}")
            print(f"    orientation(quat): {pose0.orientation}")
        
        for i, pose in enumerate(self.pose_metadata):
            R_vehicle = UndistortionUtils.quat_to_matrix(pose.orientation)
            t_vehicle = pose.position
            
            iso_vehicle = np.eye(4)
            iso_vehicle[:3, :3] = R_vehicle
            iso_vehicle[:3, 3] = t_vehicle
            
            # sensing_pose = vehicle_pose @ T_sensing_to_vehicle = Sensing→World
            iso_sensing = iso_vehicle @ self.T_sensing_to_vehicle
            
            # 更新pose（现在是Sensing系）
            R_sensing = iso_sensing[:3, :3]
            t_sensing = iso_sensing[:3, 3]
            
            # 转换回四元数
            r = R.from_matrix(R_sensing)
            pose.orientation = r.as_quat()  # [x, y, z, w]
            pose.position = t_sensing
        
        # 打印转换后的第一个pose
        if self.pose_metadata:
            pose0 = self.pose_metadata[0]
            print(f"  转换后(Sensing系)第一个pose:")
            print(f"    position: {pose0.position}")
            print(f"    orientation(quat): {pose0.orientation}")
        
        print(f"  ✓ 已转换 {len(self.pose_metadata)} 个位姿到Sensing系")
    
    def _generate_projection_visualization(self, points: np.ndarray, image: np.ndarray, 
                                          output_path: Path, frame_idx: int, 
                                          is_undistorted: bool = True) -> bool:
        """生成点云投影到图像的可视化图
        
        Args:
            points: (N, 4+) 点云数据 [x, y, z, intensity, ...]
                    注意：此函数在sync_and_save的debug阶段调用，
                    此时点云可能仍在Sensing系（未经Sensing→LiDAR转换），
                    函数内部会按需转换到LiDAR系后再投影
            image: (H, W, 3) RGB图像
            output_path: 输出文件路径
            frame_idx: 帧索引
            is_undistorted: 是否使用去畸变后的点云
            
        Returns:
            bool: 是否成功生成
        """
        try:
            fx = self.camera_config['intrinsic']['f_x']
            fy = self.camera_config['intrinsic']['f_y']
            cx = self.camera_config['intrinsic']['o_x']
            cy = self.camera_config['intrinsic']['o_y']
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0,  0,  1]])
            
            points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
            
            # 如果点云在Sensing系，先转换到LiDAR系
            if self._need_sensing_to_lidar and self._sensing_to_lidar_transform is not None:
                xyz_lidar = (self._sensing_to_lidar_transform @ points_hom.T).T[:, :3]
                points_hom = np.hstack([xyz_lidar, np.ones((len(xyz_lidar), 1))])
            
            # LiDAR→Camera投影
            points_cam = (self.T_lidar_to_cam @ points_hom.T).T[:, :3]
            
            # 过滤相机前方的点
            mask = points_cam[:, 2] > 0
            points_cam = points_cam[mask]
            depths = points_cam[:, 2].copy()
            
            if len(points_cam) == 0:
                return False
            
            # 投影到图像平面
            points_img = (K @ points_cam.T).T
            points_img = points_img[:, :2] / points_img[:, 2:3]
            
            # 过滤图像范围内的点
            H, W = image.shape[:2]
            mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < W) & \
                   (points_img[:, 1] >= 0) & (points_img[:, 1] < H)
            points_img = points_img[mask]
            depths = depths[mask]
            
            if len(points_img) == 0:
                return False
            
            # 生成可视化图
            fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
            ax.imshow(image)
            
            scatter = ax.scatter(points_img[:, 0], points_img[:, 1],
                                c=depths, cmap='jet', s=1, alpha=0.5)
            plt.colorbar(scatter, ax=ax, label='Depth (m)')
            
            # 标注点云状态
            status_text = "✓ Undistorted" if is_undistorted else "⚠ Raw (Not Undistorted)"
            status_color = 'green' if is_undistorted else 'orange'
            
            title_text = f'Frame {frame_idx:06d} | Projected: {len(points_img)}/{points.shape[0]} points\n{status_text}'
            ax.set_title(title_text, fontsize=16, color='black')
            
            # 在图像右上角添加状态标签
            ax.text(0.98, 0.02, status_text, 
                   transform=ax.transAxes,
                   fontsize=14, fontweight='bold',
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8),
                   color='white')
            
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  投影可视化失败 (Frame {frame_idx}): {e}")
            return False
    
    def _compute_transforms(self):
        """计算变换矩阵
        
        参考：proto_utils.cpp:193-201, 274-323 + 坐标系文档
        
        **命名约定**（从左往右读，与训练代码一致）：
        - T_A_to_B 表示 A→B 的变换，将A系坐标变换到B系
        - 例如：T_cam_to_sensing = Camera→Sensing
        
        **config文件字段** (cfg字段名 → 内部key名)：
        - cameras.cfg: sensor_to_cam → Camera→Sensing
        - lidars.cfg: sensor_to_lidar → 'lidar_to_sensing' → LiDAR→Sensing
        - lidars.cfg: vehicle_to_sensing → 'sensing_to_vehicle' → Sensing→Vehicle
        
        **坐标系定义**：
        - Vehicle系：后轴中心，前左上=XYZ
        - Sensing系：虚拟系，前左上=XYZ
        - LiDAR系：前左上=XYZ（与KITTI Velodyne一致）
        - Camera系：光轴=Z，右下前=XYZ（OpenCV标准）
        """
        # 1. 从config读取 Camera→Sensing (config字段: sensor_to_cam)
        T_cam_to_sensing = np.eye(4)
        r = R.from_quat(self.camera_config['orientation'])
        T_cam_to_sensing[:3, :3] = r.as_matrix()
        T_cam_to_sensing[:3, 3] = self.camera_config['position']
        
        # 2. 取逆得到 Sensing→Camera
        T_sensing_to_cam = np.linalg.inv(T_cam_to_sensing)
        
        # 3. 从config读取 LiDAR→Sensing (cfg字段: sensor_to_lidar)
        T_lidar_to_sensing = np.eye(4)
        r = R.from_quat(self.lidar_config['orientation'])
        T_lidar_to_sensing[:3, :3] = r.as_matrix()
        T_lidar_to_sensing[:3, 3] = self.lidar_config['position']
        
        # 4. 取逆得到 Sensing→LiDAR
        T_sensing_to_lidar = np.linalg.inv(T_lidar_to_sensing)
        
        # 5. 合成 LiDAR→Camera (KITTI标准Tr矩阵的逆)
        # 变换链：LiDAR→Sensing→Camera
        # T_lidar_to_cam = T_sensing_to_cam @ T_lidar_to_sensing
        self.T_lidar_to_cam = T_sensing_to_cam @ T_lidar_to_sensing
        
        # 6. 取逆得到 Camera→LiDAR (calib.txt的Tr)
        self.T_cam_to_lidar = np.linalg.inv(self.T_lidar_to_cam)
        
        # 7. 保存用于去畸变和投影的变换
        self.T_lidar_to_sensing = T_lidar_to_sensing  # LiDAR→Sensing
        self.T_sensing_to_lidar = T_sensing_to_lidar  # Sensing→LiDAR
        self.T_cam_to_sensing = T_cam_to_sensing      # Camera→Sensing
        self.T_sensing_to_cam = T_sensing_to_cam      # Sensing→Camera
        
        # 8. 获取并缓存 Sensing→Vehicle 变换（频繁使用，提前计算）
        self.T_sensing_to_vehicle = self._get_sensing_to_vehicle_transform()
        
        # 9. 内参矩阵
        intrinsic = self.camera_config['intrinsic']
        self.K = np.array([
            [intrinsic['f_x'], 0, intrinsic['o_x']],
            [0, intrinsic['f_y'], intrinsic['o_y']],
            [0, 0, 1]
        ])
        
        # 10. 初始化图像去畸变（更新self.K为去畸变后的新内参）
        self._init_undistortion_maps()
        
        print(f"\n✓ 变换矩阵已计算 (从左往右读: T_A_to_B = A→B):")
        print(f"  === 坐标系变换矩阵 ===")
        print(f"  T_cam_to_sensing (Camera→Sensing, cameras.cfg):")
        print(f"    旋转:\n{T_cam_to_sensing[:3, :3]}")
        print(f"    平移: {T_cam_to_sensing[:3, 3]}")
        print(f"  T_lidar_to_sensing (LiDAR→Sensing, lidars.cfg):")
        print(f"    旋转:\n{T_lidar_to_sensing[:3, :3]}")
        print(f"    平移: {T_lidar_to_sensing[:3, 3]}")
        print(f"  T_lidar_to_cam (LiDAR→Camera, 合成):")
        print(f"    平移: {self.T_lidar_to_cam[:3, 3]}")
        print(f"  T_sensing_to_vehicle (Sensing→Vehicle):")
        print(f"    平移: {self.T_sensing_to_vehicle[:3, 3]}")
        print(f"  ")
        print(f"  === 关键说明 ===")
        print(f"  1. 点云保存在LiDAR系 (KITTI-Odometry标准)")
        print(f"  2. calib.txt Tr = Camera→LiDAR (由lidars.cfg + cameras.cfg合成)")
        print(f"  3. calib.txt T_cam2sensing = Camera→Sensing (cameras.cfg相机外参)")
        if self.force_config:
            print(f"  ⚠️  --force-config: 点云坐标转换强制使用lidars.cfg外参")
    
    def extract_data_from_bag(self):
        """从 rosbag 提取数据（流式处理+并行加速）"""
        # 重置PointCloudParser的静态缓存（避免多次运行时残留）
        PointCloudParser._lidar_configs_logged = False
        PointCloudParser._lidar_to_sensing_from_bag = None
        PointCloudParser._sensing_to_vehicle_from_bag_proto = None
        PointCloudParser._bag_lidar_frame_id = None
        PointCloudParser._header_frame_id = None
        if hasattr(PointCloudParser, '_invalid_bag_warned'):
            del PointCloudParser._invalid_bag_warned
        if hasattr(PointCloudParser, '_format_logged'):
            del PointCloudParser._format_logged
        
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
        
        # 第一步：扫描所有bag文件的时间范围并按连续性分组
        print(f"\n扫描bag文件时间范围并分组...")
        bag_ranges = self._scan_bag_time_ranges(bag_files, use_rosbags)
        bag_groups = self._group_bags_by_continuity(bag_ranges, max_gap=1.0)
        
        # 计算每组的时间边界，存储到 self.bag_segments 供后续使用
        self.bag_segments = []
        for group in bag_groups:
            group_start = group[0][1]   # 第一个bag的start_ts
            group_end = group[-1][2]    # 最后一个bag的end_ts
            self.bag_segments.append((group_start, group_end))
        
        print(f"  ✓ {len(bag_ranges)} 个bag → {len(bag_groups)} 个连续组（间隔阈值: 1.0s）")
        if len(bag_groups) > 1:
            for i, group in enumerate(bag_groups):
                grp_start = group[0][1]
                grp_end = group[-1][2]
                duration = grp_end - grp_start
                bag_names = [g[0].name for g in group]
                print(f"    组 {i+1}: {len(group)} bags, {duration:.1f}s "
                      f"[{bag_names[0]} ~ {bag_names[-1]}]")
            gaps = []
            for i in range(1, len(bag_groups)):
                gap = bag_groups[i][0][1] - bag_groups[i-1][-1][2]
                gaps.append(gap)
            print(f"    组间间隔: {', '.join(f'{g:.0f}s' for g in gaps)}")
        
        # 确定图像 topic
        possible_topics = [
            f'/sensors/camera/{self.camera_name}_raw_data/compressed_proto',
            f'/sensors/camera/{self.camera_name}/compressed_proto',
        ]
        
        image_topic = None
        detected_pose_topic = None
        
        # 扫描所有bag文件以收集所有可用topics
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
        
        # 线程锁保护元数据列表和计数器
        lock = threading.Lock()
        counter_lock = threading.Lock()  # 专门用于计数器的锁
        
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
                        temp_images, temp_pcs, temp_poses,
                        counter_lock=counter_lock
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
        
        # ✅ 优化：使用较少的并行度避免I/O竞争
        # 点云解析是CPU密集型，但文件I/O是瓶颈
        import gc
        actual_workers = min(self.num_workers, 4)
        
        # 使用线程池并行处理（I/O密集型任务）
        # 注意：点云解析是CPU密集型，但由于GIL，线程池效率有限
        # 但进程池会导致内存问题，所以保持线程池
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(process_single_bag, bf): bf for bf in bag_files}
            
            completed = 0
            for future in tqdm(as_completed(futures), total=len(bag_files),
                             desc="  并行处理bag文件", unit="bag"):
                bag_file = futures[future]
                try:
                    n_images, n_pcs, n_poses = future.result()
                    print(f"  ✓ {bag_file.name}: {n_images} 图像, {n_pcs} 点云, {n_poses} 位姿")
                except Exception as e:
                    print(f"  ✗ {bag_file.name}: 错误 - {e}")
                
                # 每处理10个bag文件，强制GC
                completed += 1
                if completed % 10 == 0:
                    gc.collect()
    
    def _find_bag_files(self) -> List[Path]:
        """查找 bag 文件"""
        if self.bag_path.is_file():
            return [self.bag_path]
        elif self.bag_path.is_dir():
            return sorted(self.bag_path.glob('**/*.bag'))
        return []
    
    def _scan_bag_time_ranges(self, bag_files: List[Path], use_rosbags: bool = True) -> List[Tuple[Path, float, float]]:
        """扫描所有bag文件，提取每个bag的起止时间戳
        
        只读取bag的索引/元数据，不解析消息内容，非常快速。
        
        Returns:
            [(bag_path, start_ts_sec, end_ts_sec), ...] 按start_ts排序
        """
        bag_ranges = []
        for bf in bag_files:
            try:
                if use_rosbags:
                    from rosbags.rosbag1 import Reader
                    with Reader(str(bf)) as reader:
                        start_ns = reader.start_time
                        end_ns = reader.end_time
                        bag_ranges.append((bf, start_ns / 1e9, end_ns / 1e9))
                else:
                    import rosbag as rb
                    with rb.Bag(str(bf), 'r') as bag:
                        start_t = bag.get_start_time()
                        end_t = bag.get_end_time()
                        bag_ranges.append((bf, start_t, end_t))
            except Exception as e:
                pass
        
        bag_ranges.sort(key=lambda x: x[1])
        return bag_ranges
    
    def _group_bags_by_continuity(self, bag_ranges: List[Tuple[Path, float, float]], 
                                   max_gap: float = 1.0) -> List[List[Tuple[Path, float, float]]]:
        """按时间连续性将bag文件分组
        
        如果当前bag的开始时间与前一个bag的结束时间之差 < max_gap，
        则认为连续，合并到同一组；否则另起新组。
        
        Args:
            bag_ranges: [(bag_path, start_ts, end_ts), ...] 已按start_ts排序
            max_gap: 最大允许间隔（秒），默认1.0s
            
        Returns:
            groups: [[(path, start, end), ...], ...] 每组是一系列连续的bag
        """
        if not bag_ranges:
            return []
        
        groups = [[bag_ranges[0]]]
        
        for i in range(1, len(bag_ranges)):
            _, cur_start, _ = bag_ranges[i]
            _, _, prev_end = bag_ranges[i - 1]
            
            gap = cur_start - prev_end
            if gap < max_gap:
                groups[-1].append(bag_ranges[i])
            else:
                groups.append([bag_ranges[i]])
        
        return groups
    
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
                                           out_images: list, out_pcs: list, out_poses: list,
                                           counter_lock=None):
        """使用 rosbags 流式提取（输出到指定列表）
        
        注意：此函数直接将数据添加到 out_images, out_pcs, out_poses 列表。
        使用bag文件名+局部计数器作为文件名，避免多线程竞争。
        """
        from rosbags.rosbag1 import Reader
        import hashlib
        
        topics_to_check = [image_topic] if image_topic else possible_topics
        
        local_img_count = 0
        local_pc_count = 0
        local_pose_count = 0
        lidar_consecutive_fail = 0
        bag_invalid = False
        
        # 使用bag文件名的hash作为前缀，避免文件名冲突
        bag_hash = hashlib.md5(bag_file.name.encode()).hexdigest()[:8]
        
        with Reader(str(bag_file)) as reader:
            available = list(reader.topics.keys())
            topics_to_check = [t for t in topics_to_check if t in available]
            
            # 检查位姿topic是否可用
            pose_topic_available = self.active_pose_topic and self.active_pose_topic in available
            # 检查lidar topic是否可用
            lidar_topic_available = self.lidar_topic in available
            
            msg_count = 0
            for connection, timestamp, rawdata in tqdm(reader.messages(), 
                                                      desc=f"  提取数据",
                                                      unit="msg",
                                                      leave=False):  # 不保留进度条
                if bag_invalid:
                    break
                
                msg_count += 1
                
                # bag_record_time用作fallback（当header时间戳提取失败时）
                bag_record_time = timestamp / 1e9
                
                # 提取图像
                if connection.topic in topics_to_check:
                    data = self._extract_string_msg(rawdata)
                    if data:
                        header_ts = ProtobufUtils.extract_header_timestamp(data)
                        ts_sec = header_ts if header_ts is not None else bag_record_time
                        
                        image = self._decode_image_msg_from_bytes(data)
                        if image is not None:
                            image = self._undistort_image(image)
                            filename = f"{bag_hash}_{local_img_count:06d}.jpg"
                            filepath = self.temp_image_dir / filename
                            cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                            
                            out_images.append(ImageMetadata(
                                timestamp=ts_sec,
                                file_path=str(filepath)
                            ))
                            local_img_count += 1
                            del image  # 立即释放
                
                # 提取点云
                elif lidar_topic_available and connection.topic == self.lidar_topic:
                    data = self._extract_string_msg(rawdata)
                    if data:
                        header_ts = ProtobufUtils.extract_header_timestamp(data)
                        ts_sec = header_ts if header_ts is not None else bag_record_time
                        
                        # 尝试从首个点云消息提取 Sensing→Vehicle
                        if self.sensing_to_vehicle_from_bag is None:
                            s2v = self._extract_sensing_to_vehicle_from_pc_msg(data)
                            if s2v is not None:
                                self.sensing_to_vehicle_from_bag = s2v
                                print(f"✓ 从bag点云消息中提取到 Sensing→Vehicle")
                        
                        points = PointCloudParser.parse_proto_pointcloud2(data, main_lidar_frame_id="_202", config_for_comparison=self.lidar_config)
                        if points is not None and points.shape[0] >= 50:
                            lidar_consecutive_fail = 0
                            # 使用bag_hash+局部计数器作为文件名，避免冲突
                            filename = f"{bag_hash}_{local_pc_count:06d}.bin"
                            filepath = self.temp_pc_dir / filename
                            
                            if points.shape[1] == 3:
                                intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
                                points = np.hstack([points, intensity])
                            
                            points.astype(np.float32).tofile(str(filepath))
                            
                            out_pcs.append(PointCloudMetadata(
                                timestamp=ts_sec,
                                file_path=str(filepath)
                            ))
                            local_pc_count += 1
                            del points  # 立即释放
                        else:
                            lidar_consecutive_fail += 1
                            if lidar_consecutive_fail >= 3 and local_pc_count == 0:
                                print(f"\n  ❌ BAG无效: 连续{lidar_consecutive_fail}次点云解析失败"
                                      f"(frame_id无法提取)，跳过此bag及其图像/位姿数据")
                                bag_invalid = True
                                # 清理已收集的图像文件和元数据
                                for img_meta in out_images:
                                    try:
                                        Path(img_meta.file_path).unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                out_images.clear()
                                out_pcs.clear()
                                out_poses.clear()
                                break
                
                # 提取位姿
                elif pose_topic_available and connection.topic == self.active_pose_topic:
                    data = self._extract_string_msg(rawdata)
                    if data:
                        class TempMsg:
                            def __init__(self, d):
                                self.data = d
                        pose = self._decode_pose_msg(TempMsg(data), fallback_timestamp=bag_record_time)
                        if pose:
                            out_poses.append(pose)
                            local_pose_count += 1
    
    def _extract_streaming_rosbag(self, bag_file: Path, image_topic: Optional[str],
                                  possible_topics: List[str]):
        """使用 rosbag 流式提取"""
        import rosbag
        
        image_buffer = []
        pc_buffer = []
        lidar_consecutive_fail = 0
        local_pc_count = 0
        bag_invalid = False
        
        # 记录进入此bag前的元数据数量，用于回滚
        img_meta_before = len(self.image_metadata)
        pose_meta_before = len(self.pose_metadata)
        
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
                if bag_invalid:
                    break
                
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
                                # ✅ 内存优化：降低批量大小
                                if len(image_buffer) >= min(self.batch_size, 50):
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
                        
                        # 尝试从首个点云消息提取 Sensing→Vehicle
                        if self.sensing_to_vehicle_from_bag is None:
                            s2v = self._extract_sensing_to_vehicle_from_pc_msg(data)
                            if s2v is not None:
                                self.sensing_to_vehicle_from_bag = s2v
                                print(f"✓ 从bag点云消息中提取到 Sensing→Vehicle")
                        
                        points = PointCloudParser.parse_proto_pointcloud2(data, main_lidar_frame_id="_202", config_for_comparison=self.lidar_config)
                        if points is not None and points.shape[0] >= 50:
                            lidar_consecutive_fail = 0
                            local_pc_count += 1
                            pc_buffer.append((ts_sec, points))
                            if len(pc_buffer) >= min(self.batch_size, 20):
                                self._save_pc_batch(pc_buffer)
                                pc_buffer.clear()
                        else:
                            lidar_consecutive_fail += 1
                            if lidar_consecutive_fail >= 3 and local_pc_count == 0:
                                print(f"\n  ❌ BAG无效: 连续{lidar_consecutive_fail}次点云解析失败"
                                      f"(frame_id无法提取)，跳过此bag及其图像/位姿数据")
                                bag_invalid = True
                                break
                
                # 提取位姿
                elif pose_topic_available and topic == self.active_pose_topic:
                    pose = self._decode_pose_msg(msg, fallback_timestamp=bag_record_time)
                    if pose is not None:
                        self.pose_metadata.append(pose)
        
        if bag_invalid:
            # 回滚此bag已写入的图像和位姿元数据
            rollback_imgs = self.image_metadata[img_meta_before:]
            for img_meta in rollback_imgs:
                try:
                    Path(img_meta.file_path).unlink(missing_ok=True)
                except Exception:
                    pass
            del self.image_metadata[img_meta_before:]
            del self.pose_metadata[pose_meta_before:]
            image_buffer.clear()
            pc_buffer.clear()
            return
        
        if image_buffer:
            self._save_image_batch(image_buffer)
        if pc_buffer:
            self._save_pc_batch(pc_buffer)
    
    def _save_image_batch(self, batch: List[Tuple[float, np.ndarray]]):
        """保存一批图像到临时目录（已去畸变）"""
        for ts, image in batch:
            image = self._undistort_image(image)
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
            # 跳过DeepRoute的额外header: $$$$ + <4 bytes length> + <header content>
            if data[:4] == b'$$$$':
                if len(data) < 8:
                    return None, None, None
                header_len = struct.unpack('<I', data[4:8])[0]
                if len(data) < 8 + header_len:
                    return None, None, None
                data = data[8 + header_len:]  # 跳过4字节marker + 4字节length + header内容
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
    
    def _init_undistortion_maps(self):
        """预计算图像去畸变映射表，并更新self.K为去畸变后的新内参。
        
        根据cameras.cfg中的畸变系数，预计算cv2.remap所需的映射表。
        去畸变后图像尺寸不变，但内参会更新以反映去畸变后的成像模型。
        符合KITTI-Odometry的要求：保存的图像应已去畸变。
        """
        distortion = self.camera_config.get('distortion', {})
        model_type = distortion.get('model_type', 'pinhole')
        
        intrinsic = self.camera_config['intrinsic']
        w = intrinsic['img_width']
        h = intrinsic['img_height']
        
        if model_type == 'fisheye':
            D = np.array([distortion.get('k1', 0.0),
                          distortion.get('k2', 0.0),
                          distortion.get('k3', 0.0),
                          distortion.get('k4', 0.0)])
        else:
            D = np.array([distortion.get('k1', 0.0),
                          distortion.get('k2', 0.0),
                          distortion.get('p1', 0.0),
                          distortion.get('p2', 0.0),
                          distortion.get('k3', 0.0)])
        
        if np.allclose(D, 0, atol=1e-10):
            self._undist_map1 = None
            self._undist_map2 = None
            print(f"  ℹ️ 畸变系数全为0，跳过去畸变初始化")
            return
        
        K_orig = self.K.copy()
        
        if model_type == 'fisheye':
            D_col = D.reshape(4, 1)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectifyMap(
                K_orig, D_col, (w, h), np.eye(3), balance=0, new_size=(w, h))
            self._undist_map1, self._undist_map2 = cv2.fisheye.initUndistortRectifyMap(
                K_orig, D_col, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        else:
            new_K, _roi = cv2.getOptimalNewCameraMatrix(
                K_orig, D, (w, h), alpha=0, newImgSize=(w, h))
            self._undist_map1, self._undist_map2 = cv2.initUndistortRectifyMap(
                K_orig, D, None, new_K, (w, h), cv2.CV_16SC2)
        
        self.K = new_K
        
        print(f"\n  ✓ 图像去畸变初始化完成 (模型: {model_type})")
        print(f"    原始内参: fx={K_orig[0,0]:.2f}, fy={K_orig[1,1]:.2f}, "
              f"cx={K_orig[0,2]:.2f}, cy={K_orig[1,2]:.2f}")
        print(f"    新内参:   fx={new_K[0,0]:.2f}, fy={new_K[1,1]:.2f}, "
              f"cx={new_K[0,2]:.2f}, cy={new_K[1,2]:.2f}")
        if model_type == 'fisheye':
            print(f"    畸变系数: k1={D[0]:.6f}, k2={D[1]:.6f}, k3={D[2]:.6f}, k4={D[3]:.6f}")
        else:
            print(f"    畸变系数: k1={D[0]:.6f}, k2={D[1]:.6f}, p1={D[2]:.6f}, p2={D[3]:.6f}, k3={D[4]:.6f}")
    
    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        """使用预计算的映射表对图像去畸变。"""
        if self._undist_map1 is None:
            return image
        return cv2.remap(image, self._undist_map1, self._undist_map2,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
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
        
        # 复制图像（使用shutil.copy更快）
        src_img = Path(self.image_metadata[img_idx].file_path)
        dst_img = image_dir / f"{idx:06d}.png"
        
        try:
            # 如果源文件是PNG，直接复制；否则转换
            if src_img.suffix.lower() == '.png':
                import shutil
                shutil.copy2(str(src_img), str(dst_img))
            else:
                img = Image.open(src_img)
                img.save(dst_img)
        except Exception as e:
            # 图像读取/保存失败，跳过该帧
            if idx == 0:
                print(f"⚠️  图像处理失败: {e}")
            return None
        
        # 读取并去畸变点云
        src_pc = Path(self.pc_metadata[pc_idx].file_path)
        dst_pc = velodyne_dir / f"{idx:06d}.bin"
        
        # 读取原始点云
        try:
            points_data = np.fromfile(str(src_pc), dtype=np.float32)
        except Exception as e:
            if dst_img.exists():
                dst_img.unlink()
            return False
        
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
                
                # 点云去畸变（poses已经是Sensing系，LiDAR系=Sensing系）
                # 注：调试信息只在verbose模式且第一帧时打印
                points_undistorted = UndistortionUtils.undistort_pointcloud(
                    points_raw, cloud_ts, target_ts, self.pose_metadata,
                    debug=(idx == 0 and self.verbose),
                    frame_idx=idx
                )
                
                # ✅ 关键修复：如果去畸变失败（返回None），跳过该帧
                if points_undistorted is None:
                    # 删除已保存的图像
                    if dst_img.exists():
                        dst_img.unlink()
                    return None  # 返回 None 表示该帧被跳过
                
                # 🔍 打印去畸变前后的点云范围（仅verbose模式）
                if idx == 0 and self.verbose:
                    print(f"\n  去畸变前点云范围:")
                    print(f"    X: [{points_raw[:, 0].min():.2f}, {points_raw[:, 0].max():.2f}]")
                    print(f"    Y: [{points_raw[:, 1].min():.2f}, {points_raw[:, 1].max():.2f}]")
                    print(f"    Z: [{points_raw[:, 2].min():.2f}, {points_raw[:, 2].max():.2f}]")
                    print(f"  去畸变后点云范围:")
                    print(f"    X: [{points_undistorted[:, 0].min():.2f}, {points_undistorted[:, 0].max():.2f}]")
                    print(f"    Y: [{points_undistorted[:, 1].min():.2f}, {points_undistorted[:, 1].max():.2f}]")
                    print(f"    Z: [{points_undistorted[:, 2].min():.2f}, {points_undistorted[:, 2].max():.2f}]")
            else:
                # 没有pose数据，直接使用前4列
                points_undistorted = points_raw[:, :4]
                if idx == 0 and self.verbose:
                    print(f"  ⚠️  警告：没有pose数据, 直接使用前4列")
        elif len(points_data) % 4 == 0:
            # (N, 4): x, y, z, intensity，无timestamp
            points_raw = points_data.reshape(-1, 4)
            points_undistorted = points_raw
        else:
            return False  # 格式异常
        
        # 点云坐标系处理：确保保存到velodyne/的点云在LiDAR系（KITTI-Odometry标准）
        #
        # 去畸变后点云所在坐标系取决于BAG header frame_id：
        # - "lidar": 点云在Sensing系 → 需要Sensing→LiDAR转换
        # - 其他值: 点云已在LiDAR系 → 直接保存
        points_final = points_undistorted
        
        if self._need_sensing_to_lidar and self._sensing_to_lidar_transform is not None:
            xyz_hom = np.hstack([points_final[:, :3], np.ones((len(points_final), 1))])
            xyz_lidar = (self._sensing_to_lidar_transform @ xyz_hom.T).T[:, :3]
            points_final = np.hstack([xyz_lidar, points_final[:, 3:4]])
            if idx == 0 and self.verbose:
                print(f"  ✓ 点云已从Sensing系转换到LiDAR系")
                print(f"  Sensing→LiDAR转换后点云范围:")
                print(f"    X: [{points_final[:, 0].min():.2f}, {points_final[:, 0].max():.2f}]")
                print(f"    Y: [{points_final[:, 1].min():.2f}, {points_final[:, 1].max():.2f}]")
                print(f"    Z: [{points_final[:, 2].min():.2f}, {points_final[:, 2].max():.2f}]")
        
        # 保存点云（LiDAR系，KITTI-Odometry标准）
        # 格式：(N, 4) = [x, y, z, intensity]
        points_final.astype(np.float32).tofile(str(dst_pc))
        
        # ✅ 内存优化：显式删除大数组
        del points_data, points_raw, points_undistorted, points_final
        
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
        
        🔧 改进：基于bag文件分组处理不连续数据
            - 在提取阶段预先扫描bag时间范围，按1s间隔阈值分组
            - 同步阶段使用bag分组的时间段，严格过滤跨段帧
            - 避免跨gap插值导致的失败
        """
        print(f"\n{'='*80}")
        print(f"阶段 2/3: 同步数据")
        print(f"{'='*80}")
        
        # 分析BAG点云坐标系，决定是否需要Sensing→LiDAR转换
        self._update_transforms_from_bag()
        
        sync_start_time = time.time()
        
        if not self.image_metadata or not self.pc_metadata:
            raise ValueError("图像或点云数据为空")
        
        # 使用bag文件分组确定的时间段（在extract阶段已计算）
        segment_boundaries = []
        if hasattr(self, 'bag_segments') and self.bag_segments:
            segment_boundaries = self.bag_segments  # [(start_ts, end_ts), ...]
            if len(segment_boundaries) > 1:
                print(f"\n🔍 基于bag文件分组，共 {len(segment_boundaries)} 个连续时间段:")
                for i, (seg_start, seg_end) in enumerate(segment_boundaries):
                    duration = seg_end - seg_start
                    print(f"  段 {i+1}: [{seg_start:.3f}, {seg_end:.3f}] ({duration:.1f}s)")
                print(f"  💡 将按段独立处理，只保留时间戳完全落在某一段内的帧")
        
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
        
        # 关键：过滤掉无法进行位姿插值的帧
        # 🔧 改进：支持不连续的bag数据（线上根据车速/场景过滤后的数据）
        # 参考 C++ manual_sensor_calib.cpp 的 min_delta 逻辑
        if self.pose_metadata:
            pose_ts_min = self.pose_metadata[0].timestamp
            pose_ts_max = self.pose_metadata[-1].timestamp
            
            MAX_POSE_GAP = self.max_pose_gap
            
            synced_pairs_filtered = []
            skipped_reasons = {'no_file': 0, 'no_pose_coverage': 0, 'pose_too_far': 0, 'cross_segment': 0}
            
            print(f"\n  检查pose覆盖（严格插值模式，精度优先）:")
            print(f"    最大pose间隔: {MAX_POSE_GAP:.1f}s")
            if len(segment_boundaries) > 1:
                print(f"    bag分组: {len(segment_boundaries)} 个连续段，过滤跨段帧")
            
            for img_idx, pc_idx in synced_pairs:
                pc_meta = self.pc_metadata[pc_idx]
                pc_ts = pc_meta.timestamp
                img_ts = self.image_metadata[img_idx].timestamp
                
                temp_pc_file = Path(pc_meta.file_path)
                
                if not temp_pc_file.exists():
                    skipped_reasons['no_file'] += 1
                    continue
                
                try:
                    points_raw = np.fromfile(str(temp_pc_file), dtype=np.float32)
                    if len(points_raw) % 5 == 0:
                        points_raw = points_raw.reshape(-1, 5)
                    elif len(points_raw) % 4 == 0:
                        points_raw = points_raw.reshape(-1, 4)
                        end_ts = pc_ts + 0.1
                    else:
                        skipped_reasons['no_file'] += 1
                        continue
                    
                    if points_raw.shape[1] >= 5:
                        max_inner_ts_us = points_raw[:, 4].max()
                        delta_time_us = max_inner_ts_us * 2
                        end_ts = pc_ts + delta_time_us * 1e-6
                    else:
                        end_ts = pc_ts + 0.1
                    
                    # 检查帧是否完全落在某个bag分组的时间段内
                    crosses_segment = False
                    if len(segment_boundaries) > 1:
                        found_segment = False
                        BOUNDARY_TOLERANCE = 0.2
                        for seg_min, seg_max in segment_boundaries:
                            if (seg_min - BOUNDARY_TOLERANCE <= pc_ts <= seg_max + BOUNDARY_TOLERANCE and 
                                seg_min - BOUNDARY_TOLERANCE <= end_ts <= seg_max + BOUNDARY_TOLERANCE and
                                seg_min - BOUNDARY_TOLERANCE <= img_ts <= seg_max + BOUNDARY_TOLERANCE):
                                found_segment = True
                                break
                        
                        if not found_segment:
                            crosses_segment = True
                    
                    if crosses_segment:
                        skipped_reasons['cross_segment'] += 1
                        continue
                    
                    # 严格插值检查：三个时间戳都必须可以精确插值
                    can_interp_pc_start = UndistortionUtils.can_interpolate(
                        self.pose_metadata, pc_ts, MAX_POSE_GAP)
                    can_interp_pc_end = UndistortionUtils.can_interpolate(
                        self.pose_metadata, end_ts, MAX_POSE_GAP)
                    can_interp_img = UndistortionUtils.can_interpolate(
                        self.pose_metadata, img_ts, MAX_POSE_GAP)
                    
                    can_process = (can_interp_pc_start and can_interp_pc_end and can_interp_img)
                    
                    if can_process:
                        synced_pairs_filtered.append((img_idx, pc_idx))
                    else:
                        if (pc_ts < pose_ts_min or end_ts > pose_ts_max or 
                            img_ts < pose_ts_min or img_ts > pose_ts_max):
                            skipped_reasons['no_pose_coverage'] += 1
                        else:
                            skipped_reasons['pose_too_far'] += 1
                        
                except Exception as e:
                    skipped_reasons['no_file'] += 1
                    continue
            
            if len(synced_pairs_filtered) < len(synced_pairs):
                print(f"\n  ⚠️  过滤无法插值的帧: {len(synced_pairs)} → {len(synced_pairs_filtered)}")
                if skipped_reasons['no_file'] > 0:
                    print(f"      - 文件不存在/读取失败: {skipped_reasons['no_file']}")
                if skipped_reasons['cross_segment'] > 0:
                    print(f"      - 不属于任何bag连续组: {skipped_reasons['cross_segment']}")
                    print(f"        💡 这些帧的时间戳落在bag组之间的gap区域")
                if skipped_reasons['no_pose_coverage'] > 0:
                    print(f"      - 超出pose时间范围: {skipped_reasons['no_pose_coverage']}")
                if skipped_reasons['pose_too_far'] > 0:
                    print(f"      - 相邻pose间隔>{MAX_POSE_GAP:.1f}s无法插值: {skipped_reasons['pose_too_far']}")
                    print(f"        💡 提示: 这些帧处于pose数据不连续的gap区域")
            else:
                print(f"    ✓ 所有 {len(synced_pairs)} 帧都有pose覆盖")
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
        
        # ========================================================================
        # 保存调试样本（未去畸变的点云，用于可视化对比）
        # ========================================================================
        if self.save_debug_samples > 0 and len(synced_pairs) > 0:
            debug_dir = seq_dir / 'debug_raw_pointclouds'
            debug_dir.mkdir(exist_ok=True)
            
            # 均匀采样
            sample_interval = max(1, len(synced_pairs) // self.save_debug_samples)
            sample_indices = list(range(0, len(synced_pairs), sample_interval))[:self.save_debug_samples]
            
            print(f"\n📸 保存调试样本（未去畸变点云 + 投影可视化）:")
            print(f"  样本数量: {len(sample_indices)}")
            print(f"  采样间隔: 每 {sample_interval} 帧")
            print(f"  保存位置: {debug_dir}")
            
            projection_success_count = 0
            undistorted_count = 0
            raw_count = 0
            
            for sample_idx, pair_idx in enumerate(tqdm(sample_indices, desc="  保存调试样本")):
                img_idx, pc_idx = synced_pairs[pair_idx]
                pc_meta = self.pc_metadata[pc_idx]
                img_meta = self.image_metadata[img_idx]
                
                # 复制原始点云
                # ⚠️ 关键修复：使用 pair_idx（帧索引）而不是 sample_idx 作为文件名
                # 这样 debug_raw_pointclouds/000005_raw.bin 直接对应 velodyne/000005.bin
                src_path = Path(pc_meta.file_path)
                if src_path.exists():
                    dst_path = debug_dir / f"{pair_idx:06d}_raw.bin"  # 使用 pair_idx
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    
                    # 同时保存对应的图像
                    img_src = Path(img_meta.file_path)
                    if img_src.exists():
                        img_dst = debug_dir / f"{pair_idx:06d}_image.jpg"  # 使用 pair_idx
                        shutil.copy2(img_src, img_dst)
                        
                        # 生成点云投影可视化图（使用去畸变后的点云）
                        try:
                            # 读取原始点云
                            points_raw = np.fromfile(src_path, dtype=np.float32)
                            if len(points_raw) >= 4:
                                points_raw = points_raw.reshape(-1, 4)  # [x, y, z, intensity]
                                
                                # ⚠️ 关键：对点云进行去畸变处理
                                # 获取时间戳
                                pc_ts = pc_meta.timestamp
                                img_ts = img_meta.timestamp
                                
                                # 计算点云扫描时间（参考prepare_custom_dataset.py逻辑）
                                if points_raw.shape[1] >= 5:
                                    # 如果有timestamp列，使用timestamp范围
                                    timestamps = points_raw[:, 4]
                                    start_ts = pc_ts + timestamps.min() * 2e-6  # timestamp单位是2us
                                    end_ts = pc_ts + timestamps.max() * 2e-6
                                else:
                                    # 否则假设100ms扫描时间
                                    start_ts = pc_ts
                                    end_ts = pc_ts + 0.1
                                
                                # 去畸变到图像时刻
                                points_undistorted = UndistortionUtils.undistort_pointcloud(
                                    points_raw, pc_ts, img_ts, self.pose_metadata,
                                    debug=False, frame_idx=pair_idx
                                )
                                
                                # 判断是否成功去畸变
                                is_undistorted = (points_undistorted is not None)
                                
                                # 如果去畸变成功，使用去畸变后的点云；否则使用原始点云
                                points_to_project = points_undistorted if is_undistorted else points_raw
                                
                                # 统计
                                if is_undistorted:
                                    undistorted_count += 1
                                else:
                                    raw_count += 1
                                
                                # 读取图像
                                image = cv2.imread(str(img_src))
                                if image is not None:
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    
                                    # 生成投影可视化，传入去畸变状态
                                    proj_output = debug_dir / f"{pair_idx:06d}_projection.png"
                                    if self._generate_projection_visualization(points_to_project, image, proj_output, pair_idx, is_undistorted):
                                        projection_success_count += 1
                        except Exception as e:
                            pass  # 静默失败，不影响主流程
            
            print(f"  ✓ 已保存 {len(sample_indices)} 个调试样本")
            print(f"  ✓ 生成 {projection_success_count}/{len(sample_indices)} 个投影可视化图")
            print(f"     - 使用去畸变点云: {undistorted_count} 张 ✓")
            if raw_count > 0:
                print(f"     - 使用原始点云: {raw_count} 张 ⚠️ (去畸变失败，投影可能不准确)")
            print(f"  💡 提示: ")
            print(f"     - 原始点云: debug_raw_pointclouds/*_raw.bin")
            print(f"     - 去畸变点云: velodyne/*.bin")
            print(f"     - 投影可视化: debug_raw_pointclouds/*_projection.png")
            print(f"       * 绿色标签 '✓ Undistorted': 使用去畸变点云，投影准确")
            print(f"       * 橙色标签 '⚠ Raw': 使用原始点云，投影可能不准确")
            print(f"     可对比查看去畸变效果和投影质量")
        
        # 保存最终数据集（并行处理）
        print(f"\n{'='*80}")
        print(f"阶段 3/3: 去畸变并保存 (使用 {self.num_workers} 个线程)")
        print(f"{'='*80}")
        
        save_start_time = time.time()
        
        # ✅ 优化：直接在保存时处理去畸变，跳过预验证步骤
        # 预验证太慢，改为在保存时检测并跳过失败帧
        valid_pairs = synced_pairs  # 直接使用所有配对，在保存时过滤
        skipped_count = 0  # 将在保存阶段统计
        
        # 使用原始配对，在保存时过滤
        synced_pairs = valid_pairs
        
        # 保存帧（去畸变和保存合并处理）
        print(f"\n  去畸变并保存...")
        
        # ✅ 内存优化：限制并行度，避免OOM
        # 每个点云约 200MB (100万点 * 5 * 4字节 * 10倍处理开销)
        # 8个并行 = 1.6GB 内存使用，安全阈值
        actual_workers = min(self.num_workers, 8)
        print(f"  使用 {actual_workers} 个并行工作线程 (内存安全模式)")
        
        # 准备任务（使用临时索引，后续重新编号）
        tasks = [
            (tmp_idx, img_idx, pc_idx, image_dir, velodyne_dir)
            for tmp_idx, (img_idx, pc_idx) in enumerate(synced_pairs)
        ]
        
        # ✅ 内存优化：分批处理，每批处理后强制GC
        import gc
        batch_size = 200  # 每批200帧
        results = []
        
        for batch_start in range(0, len(tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            
            # 使用线程池并行处理当前批次
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = [executor.submit(self._process_single_frame, task) for task in batch_tasks]
                for future in tqdm(futures, desc=f"  批次 {batch_start//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}", 
                                   total=len(batch_tasks), leave=False):
                    result = future.result()
                    results.append(result)
            
            # 强制垃圾回收
            gc.collect()
        
        # 检查是否有跳过的帧，如果有则需要重新编号
        skipped_indices = [i for i, r in enumerate(results) if r is None or r is False]
        if skipped_indices:
            print(f"\n  重新编号（跳过 {len(skipped_indices)} 帧）...")
            # 收集成功的帧
            success_indices = [i for i, r in enumerate(results) if r is True]
            
            # ✅ 修复：使用临时目录避免文件覆盖问题
            import tempfile
            import shutil
            
            # 创建临时目录
            temp_rename_dir = self.temp_dir / 'rename_temp'
            temp_rename_dir.mkdir(exist_ok=True)
            temp_img_dir = temp_rename_dir / 'images'
            temp_pc_dir = temp_rename_dir / 'pointclouds'
            temp_img_dir.mkdir(exist_ok=True)
            temp_pc_dir.mkdir(exist_ok=True)
            
            # 第一步：将成功的帧移动到临时目录并重新编号
            for new_idx, old_idx in enumerate(tqdm(success_indices, desc="  重编号(1/2)")):
                old_img = image_dir / f"{old_idx:06d}.png"
                old_pc = velodyne_dir / f"{old_idx:06d}.bin"
                new_img = temp_img_dir / f"{new_idx:06d}.png"
                new_pc = temp_pc_dir / f"{new_idx:06d}.bin"
                
                if old_img.exists():
                    shutil.move(str(old_img), str(new_img))
                if old_pc.exists():
                    shutil.move(str(old_pc), str(new_pc))
            
            # 第二步：清空原目录中的残留文件
            for f in image_dir.glob('*.png'):
                f.unlink()
            for f in velodyne_dir.glob('*.bin'):
                f.unlink()
            
            # 第三步：将重新编号的文件移回原目录
            for f in tqdm(list(temp_img_dir.glob('*.png')), desc="  重编号(2/2)", leave=False):
                shutil.move(str(f), str(image_dir / f.name))
            for f in temp_pc_dir.glob('*.bin'):
                shutil.move(str(f), str(velodyne_dir / f.name))
            
            # 清理临时目录
            shutil.rmtree(str(temp_rename_dir))
            
            # 更新 synced_pairs 为只包含成功的帧
            synced_pairs = [synced_pairs[i] for i in success_indices]
            
            print(f"  ✓ 重新编号完成: {len(success_indices)} 帧")
        
        save_time = time.time() - save_start_time
        
        # 统计结果
        success_count = sum(1 for r in results if r is True)
        failed_count = sum(1 for r in results if r is False)
        skipped_in_save = sum(1 for r in results if r is None)
        
        print(f"\n✓ 去畸变和保存完成:")
        print(f"  成功: {success_count} 帧")
        if failed_count > 0:
            print(f"  格式错误: {failed_count} 帧")
        if skipped_in_save > 0:
            print(f"  去畸变失败跳过: {skipped_in_save} 帧")
        print(f"  耗时: {timedelta(seconds=int(save_time))}")
        if len(results) > 0:
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
        
        # 清理临时文件（自动删除以节省存储空间）
        # temp/ 目录存储的是从bag提取的原始数据，处理完后可以删除
        # 最终数据已保存在 sequences/ 目录，debug样本在 debug_raw_pointclouds/
        print(f"\n🗑️  清理临时文件...")
        print(f"   位置: {self.temp_dir}")
        import shutil
        try:
            # 获取 temp 目录大小
            import subprocess
            size_output = subprocess.run(['du', '-sh', str(self.temp_dir)], 
                                       capture_output=True, text=True, timeout=30)
            temp_size = size_output.stdout.split()[0] if size_output.returncode == 0 else "未知"
            
            shutil.rmtree(self.temp_dir)
            print(f"   ✓ 已删除临时目录，节省了约 {temp_size} 存储空间")
        except Exception as e:
            print(f"   ⚠️  删除临时目录失败: {e}")
            print(f"   💡 您可以手动删除以节省空间: rm -rf {self.temp_dir}")
        
        # 计算总耗时
        total_time = getattr(self, '_extract_time', 0) + sync_time + save_time
        
        print(f"\n{'='*80}")
        print(f"数据集生成完成！")
        print(f"{'='*80}")
        print(f"\n📁 输出位置:")
        print(f"  数据集: {seq_dir}")
        if self.save_debug_samples > 0:
            print(f"  调试样本: {seq_dir / 'debug_raw_pointclouds'}")
        
        print(f"\n📊 统计信息:")
        print(f"  有效图像: {success_count} 张")
        print(f"  有效点云: {success_count} 帧")
        if skipped_count > 0:
            print(f"  去畸变失败跳过: {skipped_count} 帧 (不保存)")
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
        """保存标定文件（符合KITTI-Odometry标准格式）
        
        参考：KITTI Odometry 数据集格式规范
        
        **KITTI标准格式**：
        - 点云在LiDAR系（velodyne文件夹）
        - Tr矩阵：**Camera → LiDAR**（Velodyne→Camera 的逆）
        
        **坐标变换链**：
        - 点云在LiDAR系：P_lidar
        - Tr = Camera → LiDAR
        - 使用时取逆：P_camera = inv(Tr) * P_lidar
        - 投影到图像：p = K * P_camera（P_camera.z > 0时可见）
        
        **扩展字段**：
        - T_cam2sensing: Camera→Sensing（cameras.cfg相机外参，系统已知量）
        - D: 畸变系数
        - camera_model: 相机模型类型
        
        **Tr合成方式**：
        - 使用lidars.cfg的 LiDAR→Sensing + cameras.cfg的 Camera→Sensing 合成
        - 不使用BAG中的 LiDAR→Sensing（BAG的仅用于点云坐标转换）
        """
        calib_path = seq_dir / 'calib.txt'
        
        P2 = np.zeros((3, 4))
        P2[:3, :3] = self.K
        
        # Tr: Camera→LiDAR (KITTI标准: Velodyne→Camera的逆)
        # self.T_cam_to_lidar = Camera→LiDAR = inv(T_lidar_to_cam)
        # 由lidars.cfg LiDAR→Sensing + cameras.cfg Camera→Sensing 合成
        Tr_3x4 = self.T_cam_to_lidar[:3, :]
        
        # T_cam2sensing: Camera→Sensing（cameras.cfg相机外参，系统已知量）
        T_cam2sensing_3x4 = self.T_cam_to_sensing[:3, :]
        
        # D: 畸变系数 — 图像在prepare阶段已完成去畸变，此处写零
        distortion = self.camera_config.get('distortion', {})
        model_type = distortion.get('model_type', 'pinhole')
        
        if model_type == 'fisheye':
            D = np.zeros(4)
        else:
            D = np.zeros(5)
        
        with open(calib_path, 'w') as f:
            # 完整的KITTI格式包含P0-P3，但BEVCalib只需要P2
            # 为了兼容性，我们也写入P0, P1, P3（使用相同的P2）
            for i in range(4):
                f.write(f"P{i}: ")
                f.write(" ".join([f"{val:.12e}" for val in P2.flatten()]))
                f.write("\n")
            
            # Tr: Camera→LiDAR (KITTI标准: 3x4, 12个数)
            f.write("Tr: ")
            f.write(" ".join([f"{val:.12e}" for val in Tr_3x4.flatten()]))
            f.write("\n")
            
            # T_cam2sensing: Camera→Sensing (系统已知相机外参, 3x4, 12个数)
            f.write("T_cam2sensing: ")
            f.write(" ".join([f"{val:.12e}" for val in T_cam2sensing_3x4.flatten()]))
            f.write("\n")
            
            # D: 畸变系数
            f.write("D: ")
            f.write(" ".join([f"{val:.12e}" for val in D]))
            f.write("\n")
            
            f.write(f"camera_model: {model_type}\n")
        
        print(f"标定文件已保存: {calib_path}")
        print(f"  ✓ 点云坐标系: LiDAR系 (KITTI-Odometry标准)")
        print(f"  ✓ Tr: Camera→LiDAR (由lidars.cfg + cameras.cfg合成)")
        print(f"  ✓ T_cam2sensing: Camera→Sensing (cameras.cfg相机外参)")
        
        print(f"  ✓ 图像已在prepare阶段去畸变，D=0")
    
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
        
        # 转换为相对于第0帧的位姿 (KITTI格式)
        # T_i_to_0 = T_world_to_0 @ T_i_to_world = framei→frame0
        if len(poses_world) == 0:
            print("警告: 没有有效的位姿数据")
            return
        
        T_0_to_world = poses_world[0]                # frame0→World
        T_world_to_0 = np.linalg.inv(T_0_to_world)  # World→frame0
        
        poses_relative = []
        for T_i_to_world in poses_world:             # framei→World
            T_i_to_0 = T_world_to_0 @ T_i_to_world  # framei→frame0
            poses_relative.append(T_i_to_0)
        
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
    parser.add_argument('--save_debug_samples', type=int, default=0,
                       help='保存用于调试的未去畸变点云样本数量（默认: 0，不保存）。'
                            '设置为10-20可保存均匀采样的样本用于可视化对比。')
    parser.add_argument('--max_pose_gap', type=float, default=0.5,
                       help='最大允许的pose间隔（秒）。用于处理不连续的bag数据。'
                            '超过此间隔的时间段将被认为数据不连续，相关帧会被跳过。'
                            '对于连续数据，默认0.5秒足够；对于不连续数据，可适当增大。')
    parser.add_argument('--force-config', action='store_true', default=False,
                       help='强制使用lidars.cfg中的lidar外参替代bag中的外参。'
                            '默认行为：优先使用bag中提取的合格外参生成calib.txt，'
                            '不合格则fallback到lidars.cfg。'
                            '加此选项后，所有环节强制使用lidars.cfg中的参数。')
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
    print(f"  最大pose间隔: {args.max_pose_gap}s（用于处理不连续bag数据）")
    if args.force_config:
        print(f"  ⚠️  强制使用lidars.cfg外参（忽略bag中的lidar外参）")
    
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
        save_debug_samples=args.save_debug_samples,
        max_pose_gap=args.max_pose_gap,
        force_config=args.force_config,
        sequence_id=args.sequence_id,
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
    print(f"\n  4. 开始训练:")
    print(f"     python kitti-bev-calib/train_kitti.py --dataset_root {args.output_dir}")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
