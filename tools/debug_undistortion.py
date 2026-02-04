#!/usr/bin/env python3
"""调试点云去畸变算法，对比C++和Python实现"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def vec6_to_isometry(v):
    """将6D向量转换为旋转矩阵和平移向量"""
    r_vec = v[:3]
    t = v[3:6]
    R_mat = R.from_rotvec(r_vec).as_matrix()
    return R_mat, t

def isometry_to_vec6(R_mat, t):
    """将旋转矩阵和平移向量转换为6D向量"""
    r = R.from_matrix(R_mat).as_rotvec()
    return np.concatenate([r, t])

def isometry_multiply(R1, t1, R2, t2):
    """齐次矩阵乘法: [R1|t1] * [R2|t2] = [R1*R2 | R1*t2 + t1]"""
    R_out = R1 @ R2
    t_out = R1 @ t2 + t1
    return R_out, t_out

def undistort_cpp_style(points, cloud_ts, target_ts, poses):
    """完全模拟C++实现的去畸变"""
    # 1. 找到三个时刻的位姿
    def interpolate(ts):
        for i in range(len(poses) - 1):
            t1, R1, pos1 = poses[i]
            t2, R2, pos2 = poses[i + 1]
            if t1 <= ts <= t2:
                alpha = (ts - t1) / (t2 - t1)
                # delta = inv(iso1) * iso2
                R_delta = R1.T @ R2
                t_delta = R1.T @ (pos2 - pos1)
                # 转为vec6并插值
                v = isometry_to_vec6(R_delta, t_delta)
                v2 = v * alpha
                R_interp, t_interp = vec6_to_isometry(v2)
                # iso = iso1 * delta_interp
                R_out = R1 @ R_interp
                t_out = pos1 + R1 @ t_interp
                return R_out, t_out
        return None
    
    start_pose = interpolate(cloud_ts)
    
    # 计算end_stamp
    max_inner_ts = points[:, 4].max()
    delta_stamp_us = max_inner_ts * 2
    end_ts = cloud_ts + delta_stamp_us * 1e-6
    
    end_pose = interpolate(end_ts)
    target_pose = interpolate(target_ts)
    
    if start_pose is None or end_pose is None or target_pose is None:
        print(f"插值失败: start={start_pose is not None}, end={end_pose is not None}, target={target_pose is not None}")
        return None
    
    R_start, t_start = start_pose
    R_end, t_end = end_pose
    R_target, t_target = target_pose
    
    print(f"start_pose: t=[{t_start[0]:.2f}, {t_start[1]:.2f}, {t_start[2]:.2f}]")
    print(f"end_pose: t=[{t_end[0]:.2f}, {t_end[1]:.2f}, {t_end[2]:.2f}]")
    print(f"target_pose: t=[{t_target[0]:.2f}, {t_target[1]:.2f}, {t_target[2]:.2f}]")
    
    # 2. 计算 delta = inv(start) * end
    R_delta = R_start.T @ R_end
    t_delta = R_start.T @ (t_end - t_start)
    v = isometry_to_vec6(R_delta, t_delta)
    v_per_second = v / (delta_stamp_us * 1e-6)
    
    print(f"v_per_second: {v_per_second}")
    print(f"  rotation rate: {np.linalg.norm(v_per_second[:3]) * 180/np.pi:.2f} deg/s")
    print(f"  translation rate: {np.linalg.norm(v_per_second[3:]):.2f} m/s")
    
    # 3. 计算 iso_target_start = inv(target) * start
    R_target_start = R_target.T @ R_start
    t_target_start = R_target.T @ (t_start - t_target)
    
    print(f"iso_target_start: t=[{t_target_start[0]:.2f}, {t_target_start[1]:.2f}, {t_target_start[2]:.2f}]")
    
    # 4. 对每个点进行去畸变
    xyz = points[:, :3]
    intensity = points[:, 3:4]
    ts_us = points[:, 4]
    
    # 逐点计算（模拟C++循环）
    xyz_undistorted = np.zeros_like(xyz)
    for i in range(len(xyz)):
        dt = ts_us[i] * 2.0e-6
        v2 = v_per_second * dt
        R_delta2, t_delta2 = vec6_to_isometry(v2)
        
        # delta2 = iso_target_start * delta2
        R_combined, t_combined = isometry_multiply(R_target_start, t_target_start, R_delta2, t_delta2)
        
        # pu = delta2 * p
        p = xyz[i]
        pu = R_combined @ p + t_combined
        xyz_undistorted[i] = pu
    
    return np.hstack([xyz_undistorted, intensity])

def undistort_python_vectorized(points, cloud_ts, target_ts, poses):
    """Python向量化实现的去畸变"""
    # 1. 找到三个时刻的位姿
    def interpolate(ts):
        for i in range(len(poses) - 1):
            t1, R1, pos1 = poses[i]
            t2, R2, pos2 = poses[i + 1]
            if t1 <= ts <= t2:
                alpha = (ts - t1) / (t2 - t1)
                R_delta = R1.T @ R2
                t_delta = R1.T @ (pos2 - pos1)
                v = isometry_to_vec6(R_delta, t_delta)
                v2 = v * alpha
                R_interp, t_interp = vec6_to_isometry(v2)
                R_out = R1 @ R_interp
                t_out = pos1 + R1 @ t_interp
                return R_out, t_out
        return None
    
    start_pose = interpolate(cloud_ts)
    max_inner_ts = points[:, 4].max()
    delta_stamp_us = max_inner_ts * 2
    end_ts = cloud_ts + delta_stamp_us * 1e-6
    end_pose = interpolate(end_ts)
    target_pose = interpolate(target_ts)
    
    if start_pose is None or end_pose is None or target_pose is None:
        return None
    
    R_start, t_start = start_pose
    R_end, t_end = end_pose
    R_target, t_target = target_pose
    
    # 2. 计算速度
    R_delta = R_start.T @ R_end
    t_delta = R_start.T @ (t_end - t_start)
    v = isometry_to_vec6(R_delta, t_delta)
    v_per_second = v / (delta_stamp_us * 1e-6)
    
    # 3. 计算 iso_target_start
    R_target_start = R_target.T @ R_start
    t_target_start = R_target.T @ (t_start - t_target)
    
    # 4. 向量化计算
    xyz = points[:, :3]
    intensity = points[:, 3:4]
    ts_us = points[:, 4]
    
    dt = ts_us * 2.0e-6
    v_points = v_per_second[np.newaxis, :] * dt[:, np.newaxis]
    r_vecs = v_points[:, :3]
    t_vecs = v_points[:, 3:6]
    
    R_delta2 = R.from_rotvec(r_vecs).as_matrix()
    
    # iso_target_start * delta2
    R_combined = np.einsum('ij,njk->nik', R_target_start, R_delta2)
    t_combined = np.einsum('ij,nj->ni', R_target_start, t_vecs) + t_target_start
    
    xyz_undistorted = np.einsum('nij,nj->ni', R_combined, xyz) + t_combined
    
    return np.hstack([xyz_undistorted, intensity])

# 测试
if __name__ == "__main__":
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟位姿序列（车辆沿X轴移动）
    poses = []
    for i in range(10):
        t = 100.0 + i * 0.1  # 时间戳（秒）
        # 车辆沿X轴以10m/s移动
        pos = np.array([i * 1.0, 0.0, 0.0])  # 每0.1秒移动1米
        # 无旋转
        R_mat = np.eye(3)
        poses.append((t, R_mat, pos))
    
    # 模拟点云
    N = 100
    points = np.zeros((N, 5))
    points[:, 0] = np.random.uniform(5, 50, N)  # X
    points[:, 1] = np.random.uniform(-10, 10, N)  # Y
    points[:, 2] = np.random.uniform(-2, 2, N)  # Z
    points[:, 3] = np.random.uniform(0, 1, N)  # intensity
    points[:, 4] = np.random.uniform(0, 50000, N)  # timestamp (单位: 2us)
    
    cloud_ts = 100.0
    target_ts = 100.05  # 50ms后
    
    print("=" * 60)
    print("测试1: 简单直线运动")
    print("=" * 60)
    
    print("\n--- C++风格逐点计算 ---")
    result_cpp = undistort_cpp_style(points.copy(), cloud_ts, target_ts, poses)
    
    print("\n--- Python向量化计算 ---")
    result_py = undistort_python_vectorized(points.copy(), cloud_ts, target_ts, poses)
    
    if result_cpp is not None and result_py is not None:
        diff = np.abs(result_cpp - result_py)
        print(f"\n差异统计:")
        print(f"  最大差异: {diff.max():.6e}")
        print(f"  平均差异: {diff.mean():.6e}")
        
        print(f"\n原始点云范围:")
        print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        print(f"\n去畸变后点云范围 (C++):")
        print(f"  X: [{result_cpp[:, 0].min():.2f}, {result_cpp[:, 0].max():.2f}]")
        print(f"  Y: [{result_cpp[:, 1].min():.2f}, {result_cpp[:, 1].max():.2f}]")
        print(f"  Z: [{result_cpp[:, 2].min():.2f}, {result_cpp[:, 2].max():.2f}]")
        
        print(f"\n去畸变后点云范围 (Python):")
        print(f"  X: [{result_py[:, 0].min():.2f}, {result_py[:, 0].max():.2f}]")
        print(f"  Y: [{result_py[:, 1].min():.2f}, {result_py[:, 1].max():.2f}]")
        print(f"  Z: [{result_py[:, 2].min():.2f}, {result_py[:, 2].max():.2f}]")
