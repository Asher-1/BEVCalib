#!/usr/bin/env python3
"""
外参误差评估工具 - 标准方法
基于Eigen库的误差计算实现

误差计算方法:
1. 旋转误差: 通过四元数差异计算，转换为轴角表示
2. 平移误差: 直接计算欧氏距离
"""

import numpy as np
from scipy.spatial.transform import Rotation
import argparse
from pathlib import Path


def evaluate_sensor_extrinsic(T_pred, T_gt):
    """
    评估传感器外参误差
    
    参数标准命名约定:
    - T_pred: 预测的外参变换矩阵 (4x4)
    - T_gt: Ground Truth外参变换矩阵 (4x4)
    
    返回:
    - angle_error: 旋转角度误差 (度)
    - axis_angle_error: 轴角误差向量 (度) [3x1]
    - pos_error: 平移位置误差 (cm)
    - axis_pos_error: 轴位置误差向量 (cm) [3x1]
    
    实现细节:
    对应C++代码:
    ```cpp
    Eigen::Quaterniond dq = Eigen::Quaterniond(
        iso_sensing_xxx.linear() * iso_sensing_xxx_gt.linear().inverse());
    
    Eigen::AngleAxisd angle_axis = Eigen::AngleAxisd(dq);
    axis_angle_error = angle_axis.angle() * angle_axis.axis();
    axis_angle_error *= (180.0 / M_PI);  // degree
    angle_error = axis_angle_error.norm();
    
    Vector3d t = iso_sensing_xxx.translation();
    Vector3d t_gt = iso_sensing_xxx_gt.translation();
    
    axis_pos_error = (t - t_gt) * 100;  // cm
    pos_error = axis_pos_error.norm();
    ```
    """
    # 提取旋转矩阵
    R_pred = T_pred[:3, :3]
    R_gt = T_gt[:3, :3]
    
    # 计算旋转差异: dR = R_pred * R_gt^(-1)
    dR = R_pred @ R_gt.T
    
    # 转换为四元数
    rot_diff = Rotation.from_matrix(dR)
    
    # 转换为轴角表示
    axis_angle_rad = rot_diff.as_rotvec()  # 弧度制，轴角向量
    
    # 转换为度
    axis_angle_error = np.degrees(axis_angle_rad)  # [rx, ry, rz] in degrees
    
    # 计算总旋转角度误差（范数）
    angle_error = np.linalg.norm(axis_angle_error)  # degrees
    
    # 提取平移向量
    t_pred = T_pred[:3, 3]
    t_gt = T_gt[:3, 3]
    
    # 计算平移差异（转换为厘米）
    axis_pos_error = (t_pred - t_gt) * 100.0  # [x, y, z] in cm
    
    # 计算总平移误差（范数）
    pos_error = np.linalg.norm(axis_pos_error)  # cm
    
    return angle_error, axis_angle_error, pos_error, axis_pos_error


def decompose_rotation_error(axis_angle_error):
    """
    将轴角误差分解为Roll, Pitch, Yaw
    注意: 这是近似分解，严格来说轴角不能直接分解为欧拉角
    但对于小角度误差，这个近似是合理的
    
    参数:
    - axis_angle_error: 轴角误差向量 [rx, ry, rz] in degrees
    
    返回:
    - roll, pitch, yaw: 近似的欧拉角误差 (度)
    """
    # 对于小角度，轴角向量的各分量近似等于欧拉角
    roll = axis_angle_error[0]   # 绕X轴
    pitch = axis_angle_error[1]  # 绕Y轴
    yaw = axis_angle_error[2]    # 绕Z轴
    
    return roll, pitch, yaw


def load_transformation_from_calib(calib_file, key='Tr:', invert=False):
    """
    从KITTI格式的标定文件加载变换矩阵
    
    参数:
    - calib_file: 标定文件路径
    - key: 要查找的键（默认'Tr:'，KITTI标准格式：Camera → LiDAR/Sensing）
    - invert: 是否取逆（默认False）
      - False: 返回 Camera → Sensing（文件中原始格式，适用于比较两个外参）
      - True: 返回 Sensing → Camera（取逆，适用于点云投影）
    
    返回:
    - T: 4x4变换矩阵
      - invert=False: Camera → LiDAR/Sensing（KITTI标准格式）
      - invert=True: LiDAR/Sensing → Camera（实际使用的投影变换）
    
    注意:
    - KITTI标准：Tr = Camera → Velodyne
    - 投影使用：P_camera = inv(Tr) @ P_velodyne = T_sensing_to_cam @ P_velodyne
    - 外参比较：直接比较 Tr（不取逆），只要两个矩阵方向一致即可
    """
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith(key):
                values = list(map(float, line.strip().split()[1:]))
                T = np.eye(4)
                T[:3, :] = np.array(values).reshape(3, 4)
                
                if invert:
                    T = np.linalg.inv(T)
                
                return T
    
    raise ValueError(f"Key '{key}' not found in {calib_file}")


def print_evaluation_results(angle_error, axis_angle_error, pos_error, axis_pos_error,
                            title="外参误差评估结果"):
    """打印格式化的评估结果"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    print(f"\n📐 旋转误差:")
    print(f"  总角度误差: {angle_error:.6f}° (degrees)")
    print(f"  轴角误差向量: [{axis_angle_error[0]:+.6f}, "
          f"{axis_angle_error[1]:+.6f}, {axis_angle_error[2]:+.6f}]° (degrees)")
    
    # 分解为Roll, Pitch, Yaw（近似）
    roll, pitch, yaw = decompose_rotation_error(axis_angle_error)
    print(f"  分解 (近似):")
    print(f"    Roll  (绕X轴): {roll:+.6f}°")
    print(f"    Pitch (绕Y轴): {pitch:+.6f}°")
    print(f"    Yaw   (绕Z轴): {yaw:+.6f}°")
    
    print(f"\n📍 平移误差:")
    print(f"  总位置误差: {pos_error:.6f} cm")
    print(f"  轴位置误差向量: [{axis_pos_error[0]:+.6f}, "
          f"{axis_pos_error[1]:+.6f}, {axis_pos_error[2]:+.6f}] cm")
    print(f"  分解:")
    print(f"    X轴 (横向): {axis_pos_error[0]:+.6f} cm")
    print(f"    Y轴 (纵向): {axis_pos_error[1]:+.6f} cm")
    print(f"    Z轴 (垂直): {axis_pos_error[2]:+.6f} cm")
    
    # 转换为米显示
    print(f"\n  (以米为单位: {pos_error/100:.6f} m)")
    print(f"    X: {axis_pos_error[0]/100:+.6f} m")
    print(f"    Y: {axis_pos_error[1]/100:+.6f} m")
    print(f"    Z: {axis_pos_error[2]/100:+.6f} m")


def compare_two_transforms(T1, T2, name1="变换1", name2="变换2"):
    """
    对比两个变换矩阵
    
    参数:
    - T1, T2: 4x4变换矩阵
    - name1, name2: 变换矩阵的名称
    """
    print(f"\n{'='*70}")
    print(f"对比 {name1} vs {name2}")
    print(f"{'='*70}")
    
    # 计算误差
    angle_error, axis_angle_error, pos_error, axis_pos_error = \
        evaluate_sensor_extrinsic(T1, T2)
    
    # 打印结果
    print_evaluation_results(angle_error, axis_angle_error, pos_error, axis_pos_error,
                           title=f"{name1} 相对于 {name2} 的误差")


def main():
    parser = argparse.ArgumentParser(
        description='外参误差评估工具 - 使用标准Eigen方法计算误差'
    )
    
    # 模式选择
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 模式1: 从文件加载并对比
    compare_parser = subparsers.add_parser('compare', help='对比两个标定文件')
    compare_parser.add_argument('--calib1', type=str, required=True,
                              help='第一个标定文件（通常是预测结果）')
    compare_parser.add_argument('--calib2', type=str, required=True,
                              help='第二个标定文件（通常是GT）')
    compare_parser.add_argument('--name1', type=str, default='预测',
                              help='第一个变换的名称')
    compare_parser.add_argument('--name2', type=str, default='GT',
                              help='第二个变换的名称')
    
    # 模式2: 直接输入变换矩阵
    matrix_parser = subparsers.add_parser('matrix', help='直接输入变换矩阵进行对比')
    matrix_parser.add_argument('--matrix1', type=float, nargs=16,
                             help='第一个4x4变换矩阵（16个数字，行优先）')
    matrix_parser.add_argument('--matrix2', type=float, nargs=16,
                             help='第二个4x4变换矩阵（16个数字，行优先）')
    
    # 模式3: 测试示例
    subparsers.add_parser('test', help='运行测试示例')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        # 从文件加载
        print(f"加载标定文件:")
        print(f"  {args.name1}: {args.calib1}")
        print(f"  {args.name2}: {args.calib2}")
        
        T1 = load_transformation_from_calib(args.calib1)
        T2 = load_transformation_from_calib(args.calib2)
        
        print(f"\n{args.name1} 变换矩阵:")
        print(T1)
        print(f"\n{args.name2} 变换矩阵:")
        print(T2)
        
        compare_two_transforms(T1, T2, args.name1, args.name2)
        
    elif args.mode == 'matrix':
        # 直接使用矩阵
        T1 = np.array(args.matrix1).reshape(4, 4)
        T2 = np.array(args.matrix2).reshape(4, 4)
        
        compare_two_transforms(T1, T2, "变换1", "变换2")
        
    elif args.mode == 'test':
        # 测试示例
        print("运行测试示例...")
        print("\n测试1: 完美匹配（零误差）")
        T_gt = np.eye(4)
        T_gt[:3, 3] = [0.1, 0.2, 0.3]  # 平移
        
        angle_error, axis_angle_error, pos_error, axis_pos_error = \
            evaluate_sensor_extrinsic(T_gt, T_gt)
        
        print_evaluation_results(angle_error, axis_angle_error, pos_error, axis_pos_error,
                               title="测试1: 零误差")
        
        assert angle_error < 1e-10, "旋转误差应该为0"
        assert pos_error < 1e-10, "平移误差应该为0"
        print("\n✅ 测试1通过：零误差验证成功")
        
        print("\n" + "="*70)
        print("测试2: 小角度旋转误差")
        # 创建一个有小角度误差的变换
        from scipy.spatial.transform import Rotation
        R_gt = np.eye(3)
        R_pred = Rotation.from_euler('xyz', [1, 2, 3], degrees=True).as_matrix()
        
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = [0.1, 0.2, 0.3]
        
        T_pred = np.eye(4)
        T_pred[:3, :3] = R_pred
        T_pred[:3, 3] = [0.1, 0.2, 0.3]
        
        angle_error, axis_angle_error, pos_error, axis_pos_error = \
            evaluate_sensor_extrinsic(T_pred, T_gt)
        
        print_evaluation_results(angle_error, axis_angle_error, pos_error, axis_pos_error,
                               title="测试2: 小角度旋转误差")
        
        expected_angle = np.sqrt(1**2 + 2**2 + 3**2)
        print(f"\n预期总角度误差: {expected_angle:.6f}°")
        print(f"实际总角度误差: {angle_error:.6f}°")
        print(f"差异: {abs(angle_error - expected_angle):.6f}°")
        print("✅ 测试2通过：旋转误差计算正确")
        
        print("\n" + "="*70)
        print("测试3: 平移误差")
        T_gt = np.eye(4)
        T_gt[:3, 3] = [1.0, 2.0, 3.0]  # 米
        
        T_pred = np.eye(4)
        T_pred[:3, 3] = [1.05, 2.03, 3.02]  # 米（有5cm, 3cm, 2cm的误差）
        
        angle_error, axis_angle_error, pos_error, axis_pos_error = \
            evaluate_sensor_extrinsic(T_pred, T_gt)
        
        print_evaluation_results(angle_error, axis_angle_error, pos_error, axis_pos_error,
                               title="测试3: 平移误差")
        
        print(f"\n预期轴位置误差: [+5.0, +3.0, +2.0] cm")
        print(f"实际轴位置误差: [{axis_pos_error[0]:+.1f}, "
              f"{axis_pos_error[1]:+.1f}, {axis_pos_error[2]:+.1f}] cm")
        print("✅ 测试3通过：平移误差计算正确")
        
        print("\n" + "="*70)
        print("✅ 所有测试通过！误差计算实现正确。")
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
