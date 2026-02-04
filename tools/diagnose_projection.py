#!/usr/bin/env python3
"""
点云投影诊断工具

用于诊断Python投影与C++投影差异的根本原因。

功能：
1. 验证外参变换方向
2. 检查点云坐标系
3. 对比FOV过滤效果
4. 生成详细的诊断报告
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import Tuple, Optional, Dict
import json


def load_calib(calib_file: str) -> Dict:
    """
    加载标定文件，返回所有参数
    """
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            
            key = parts[0].rstrip(':')
            
            if key == 'camera_model':
                calib['camera_model'] = parts[1] if len(parts) > 1 else 'pinhole'
                continue
            
            try:
                values = np.array([float(v) for v in parts[1:]])
            except ValueError:
                continue
            
            if key == 'P2':
                P2 = values.reshape(3, 4)
                calib['K'] = P2[:3, :3]
                calib['P2'] = P2
            elif key == 'Tr':
                if len(values) == 12:
                    Tr_3x4 = values.reshape(3, 4)
                    calib['Tr'] = np.vstack([Tr_3x4, [0, 0, 0, 1]])
                elif len(values) == 16:
                    calib['Tr'] = values.reshape(4, 4)
            elif key == 'D':
                calib['D'] = values
    
    return calib


def analyze_transform_matrix(T: np.ndarray, name: str) -> Dict:
    """
    分析变换矩阵的特性
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    # 检查是否是有效的旋转矩阵
    det = np.linalg.det(R)
    is_rotation = np.allclose(det, 1.0, atol=1e-6)
    is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    
    # 提取旋转角度（欧拉角）
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    euler_deg = r.as_euler('xyz', degrees=True)
    
    # 分析变换方向
    # 如果t主要在某个方向，说明变换的方向
    analysis = {
        'name': name,
        'translation': t.tolist(),
        'euler_angles_deg': euler_deg.tolist(),
        'determinant': det,
        'is_valid_rotation': is_rotation and is_orthogonal,
        'translation_magnitude': np.linalg.norm(t),
    }
    
    # 判断主要变换方向
    if abs(t[0]) > abs(t[1]) and abs(t[0]) > abs(t[2]):
        analysis['primary_translation_axis'] = 'X'
    elif abs(t[1]) > abs(t[0]) and abs(t[1]) > abs(t[2]):
        analysis['primary_translation_axis'] = 'Y'
    else:
        analysis['primary_translation_axis'] = 'Z'
    
    return analysis


def analyze_pointcloud(points: np.ndarray) -> Dict:
    """
    分析点云的特性
    """
    xyz = points[:, :3]
    
    analysis = {
        'num_points': len(points),
        'x_range': [float(xyz[:, 0].min()), float(xyz[:, 0].max())],
        'y_range': [float(xyz[:, 1].min()), float(xyz[:, 1].max())],
        'z_range': [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
        'x_mean': float(xyz[:, 0].mean()),
        'y_mean': float(xyz[:, 1].mean()),
        'z_mean': float(xyz[:, 2].mean()),
        'centroid': xyz.mean(axis=0).tolist(),
    }
    
    # 判断坐标系类型
    # KITTI LiDAR: X前, Y左, Z上
    # Camera: X右, Y下, Z前
    
    x_range = analysis['x_range'][1] - analysis['x_range'][0]
    y_range = analysis['y_range'][1] - analysis['y_range'][0]
    z_range = analysis['z_range'][1] - analysis['z_range'][0]
    
    # 如果X范围最大且为正，可能是LiDAR坐标系（前向）
    if x_range > y_range and x_range > z_range and analysis['x_mean'] > 0:
        analysis['likely_coordinate_system'] = 'LiDAR (X-forward)'
    # 如果Z范围最大且为正，可能是Camera坐标系（前向）
    elif z_range > x_range and z_range > y_range and analysis['z_mean'] > 0:
        analysis['likely_coordinate_system'] = 'Camera (Z-forward)'
    else:
        analysis['likely_coordinate_system'] = 'Unknown'
    
    return analysis


def project_and_analyze(points: np.ndarray, 
                        Tr: np.ndarray, 
                        K: np.ndarray, 
                        D: np.ndarray,
                        camera_model: str,
                        image_size: Tuple[int, int]) -> Dict:
    """
    投影点云并分析每个步骤
    """
    h, w = image_size
    
    # 步骤1: 转换到相机坐标系
    pts_3d = points[:, :3]
    pts_homo = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    pts_cam = (Tr @ pts_homo.T).T[:, :3]
    
    analysis = {
        'step1_lidar_to_camera': {
            'input_points': len(pts_3d),
            'pts_cam_x_range': [float(pts_cam[:, 0].min()), float(pts_cam[:, 0].max())],
            'pts_cam_y_range': [float(pts_cam[:, 1].min()), float(pts_cam[:, 1].max())],
            'pts_cam_z_range': [float(pts_cam[:, 2].min()), float(pts_cam[:, 2].max())],
        }
    }
    
    # 步骤2: 过滤相机后方的点
    mask_front = pts_cam[:, 2] > 0
    pts_cam_front = pts_cam[mask_front]
    
    analysis['step2_filter_behind'] = {
        'points_in_front': int(mask_front.sum()),
        'points_behind': int((~mask_front).sum()),
        'percentage_in_front': float(mask_front.sum() / len(mask_front) * 100),
    }
    
    # 步骤3: FOV过滤
    # 计算FOV
    vertex = np.array([
        [0, 0], [0, h], [w, h], [w, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    if D is not None and len(D) >= 4:
        if len(D) == 4:
            vertex_undist = cv2.fisheye.undistortPoints(vertex, K.astype(np.float32), D.astype(np.float32))
        else:
            vertex_undist = cv2.undistortPoints(vertex, K.astype(np.float32), D.astype(np.float32))
    else:
        K_inv = np.linalg.inv(K)
        vertex_undist = []
        for point in vertex[:, 0, :]:
            p_homo = np.array([point[0], point[1], 1.0])
            p_norm = K_inv @ p_homo
            vertex_undist.append(p_norm[:2])
        vertex_undist = np.array(vertex_undist).reshape(-1, 1, 2)
    
    angles = [np.arctan(np.linalg.norm(p)) for p in vertex_undist[:, 0, :]]
    fov_rad = max(angles) * 2.0
    
    # 计算每个点的角度
    xy_norm = np.linalg.norm(pts_cam_front[:, :2], axis=1)
    theta = np.abs(np.arctan2(xy_norm, pts_cam_front[:, 2]))
    mask_fov = theta < 0.5 * fov_rad
    
    analysis['step3_fov_filter'] = {
        'fov_degrees': float(np.degrees(fov_rad)),
        'points_in_fov': int(mask_fov.sum()),
        'points_outside_fov': int((~mask_fov).sum()),
        'percentage_in_fov': float(mask_fov.sum() / len(mask_fov) * 100) if len(mask_fov) > 0 else 0,
        'theta_range_deg': [float(np.degrees(theta.min())), float(np.degrees(theta.max()))] if len(theta) > 0 else [0, 0],
    }
    
    pts_cam_fov = pts_cam_front[mask_fov]
    
    # 步骤4: 投影到图像
    if len(pts_cam_fov) > 0:
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        pts_3d_cam = pts_cam_fov.reshape(-1, 1, 3).astype(np.float32)
        
        if camera_model == 'fisheye' and D is not None and len(D) == 4:
            pts_2d, _ = cv2.fisheye.projectPoints(pts_3d_cam, rvec, tvec, K.astype(np.float32), D.astype(np.float32))
        else:
            pts_2d, _ = cv2.projectPoints(pts_3d_cam, rvec, tvec, K.astype(np.float32), D.astype(np.float32) if D is not None else np.zeros(5))
        
        pts_2d = pts_2d.reshape(-1, 2)
        
        # 步骤5: 过滤图像边界
        mask_img = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
                   (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        
        analysis['step4_project_to_image'] = {
            'points_projected': len(pts_2d),
            'u_range': [float(pts_2d[:, 0].min()), float(pts_2d[:, 0].max())],
            'v_range': [float(pts_2d[:, 1].min()), float(pts_2d[:, 1].max())],
        }
        
        analysis['step5_image_bounds'] = {
            'points_in_image': int(mask_img.sum()),
            'points_outside_image': int((~mask_img).sum()),
            'percentage_in_image': float(mask_img.sum() / len(mask_img) * 100) if len(mask_img) > 0 else 0,
        }
        
        # 分析深度分布
        depths = pts_cam_fov[mask_img, 2]
        if len(depths) > 0:
            analysis['depth_distribution'] = {
                'min': float(depths.min()),
                'max': float(depths.max()),
                'mean': float(depths.mean()),
                'std': float(depths.std()),
                'percentiles': {
                    '10%': float(np.percentile(depths, 10)),
                    '50%': float(np.percentile(depths, 50)),
                    '90%': float(np.percentile(depths, 90)),
                }
            }
    else:
        analysis['step4_project_to_image'] = {'points_projected': 0}
        analysis['step5_image_bounds'] = {'points_in_image': 0}
    
    return analysis


def diagnose_frame(dataset_root: str, sequence: str, frame_idx: int, output_dir: str = None):
    """
    诊断单帧的投影问题
    """
    dataset_root = Path(dataset_root)
    seq_dir = dataset_root / 'sequences' / sequence
    
    # 加载数据
    image_path = seq_dir / 'image_2' / f'{frame_idx:06d}.png'
    pc_path = seq_dir / 'velodyne' / f'{frame_idx:06d}.bin'
    calib_path = seq_dir / 'calib.txt'
    
    if not all(p.exists() for p in [image_path, pc_path, calib_path]):
        print(f"❌ 文件不存在")
        return None
    
    # 加载图像
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    
    # 加载点云
    points = np.fromfile(str(pc_path), dtype=np.float32)
    if len(points) % 4 == 0:
        points = points.reshape(-1, 4)
    elif len(points) % 3 == 0:
        points = points.reshape(-1, 3)
    else:
        print(f"❌ 点云格式错误")
        return None
    
    # 加载标定
    calib = load_calib(str(calib_path))
    
    print("=" * 70)
    print("点云投影诊断报告")
    print("=" * 70)
    print(f"\n数据集: {dataset_root}")
    print(f"序列: {sequence}")
    print(f"帧: {frame_idx}")
    print(f"图像尺寸: {w}x{h}")
    
    # 1. 分析Tr矩阵
    print("\n" + "-" * 70)
    print("1. Tr矩阵分析 (LiDAR → Camera)")
    print("-" * 70)
    
    Tr = calib['Tr']
    tr_analysis = analyze_transform_matrix(Tr, 'Tr')
    
    print(f"  平移向量: [{tr_analysis['translation'][0]:.4f}, {tr_analysis['translation'][1]:.4f}, {tr_analysis['translation'][2]:.4f}]")
    print(f"  欧拉角(度): [{tr_analysis['euler_angles_deg'][0]:.2f}, {tr_analysis['euler_angles_deg'][1]:.2f}, {tr_analysis['euler_angles_deg'][2]:.2f}]")
    print(f"  平移幅度: {tr_analysis['translation_magnitude']:.4f}m")
    print(f"  主要平移方向: {tr_analysis['primary_translation_axis']}")
    print(f"  是否有效旋转矩阵: {tr_analysis['is_valid_rotation']}")
    
    # 分析Tr的物理意义
    R = Tr[:3, :3]
    t = Tr[:3, 3]
    
    # 检查X轴变换（LiDAR的X应该变成Camera的Z）
    x_axis_lidar = np.array([1, 0, 0])
    x_in_cam = R @ x_axis_lidar
    print(f"\n  LiDAR X轴 [1,0,0] 在Camera系中: [{x_in_cam[0]:.3f}, {x_in_cam[1]:.3f}, {x_in_cam[2]:.3f}]")
    
    # 检查Y轴变换（LiDAR的Y应该变成Camera的-X）
    y_axis_lidar = np.array([0, 1, 0])
    y_in_cam = R @ y_axis_lidar
    print(f"  LiDAR Y轴 [0,1,0] 在Camera系中: [{y_in_cam[0]:.3f}, {y_in_cam[1]:.3f}, {y_in_cam[2]:.3f}]")
    
    # 检查Z轴变换（LiDAR的Z应该变成Camera的-Y）
    z_axis_lidar = np.array([0, 0, 1])
    z_in_cam = R @ z_axis_lidar
    print(f"  LiDAR Z轴 [0,0,1] 在Camera系中: [{z_in_cam[0]:.3f}, {z_in_cam[1]:.3f}, {z_in_cam[2]:.3f}]")
    
    # 判断坐标系变换是否正确
    # 标准KITTI: LiDAR(X前,Y左,Z上) → Camera(X右,Y下,Z前)
    # 期望: LiDAR_X → Camera_Z, LiDAR_Y → Camera_-X, LiDAR_Z → Camera_-Y
    expected_correct = (
        abs(x_in_cam[2]) > 0.9 and  # LiDAR X → Camera Z
        abs(y_in_cam[0]) > 0.9 and  # LiDAR Y → Camera X (or -X)
        abs(z_in_cam[1]) > 0.9      # LiDAR Z → Camera Y (or -Y)
    )
    print(f"\n  坐标系变换是否符合KITTI标准: {'✓ 是' if expected_correct else '✗ 否'}")
    
    # 2. 分析点云
    print("\n" + "-" * 70)
    print("2. 点云分析")
    print("-" * 70)
    
    pc_analysis = analyze_pointcloud(points)
    
    print(f"  点数: {pc_analysis['num_points']}")
    print(f"  X范围: [{pc_analysis['x_range'][0]:.2f}, {pc_analysis['x_range'][1]:.2f}], 均值: {pc_analysis['x_mean']:.2f}")
    print(f"  Y范围: [{pc_analysis['y_range'][0]:.2f}, {pc_analysis['y_range'][1]:.2f}], 均值: {pc_analysis['y_mean']:.2f}")
    print(f"  Z范围: [{pc_analysis['z_range'][0]:.2f}, {pc_analysis['z_range'][1]:.2f}], 均值: {pc_analysis['z_mean']:.2f}")
    print(f"  质心: [{pc_analysis['centroid'][0]:.2f}, {pc_analysis['centroid'][1]:.2f}, {pc_analysis['centroid'][2]:.2f}]")
    print(f"  推测坐标系: {pc_analysis['likely_coordinate_system']}")
    
    # 3. 投影分析
    print("\n" + "-" * 70)
    print("3. 投影流程分析")
    print("-" * 70)
    
    K = calib['K']
    D = calib.get('D', np.zeros(5))
    camera_model = calib.get('camera_model', 'pinhole')
    
    proj_analysis = project_and_analyze(points, Tr, K, D, camera_model, (h, w))
    
    print(f"\n  步骤1: LiDAR → Camera 变换")
    step1 = proj_analysis['step1_lidar_to_camera']
    print(f"    输入点数: {step1['input_points']}")
    print(f"    Camera X范围: [{step1['pts_cam_x_range'][0]:.2f}, {step1['pts_cam_x_range'][1]:.2f}]")
    print(f"    Camera Y范围: [{step1['pts_cam_y_range'][0]:.2f}, {step1['pts_cam_y_range'][1]:.2f}]")
    print(f"    Camera Z范围: [{step1['pts_cam_z_range'][0]:.2f}, {step1['pts_cam_z_range'][1]:.2f}]")
    
    print(f"\n  步骤2: 过滤相机后方的点 (Z > 0)")
    step2 = proj_analysis['step2_filter_behind']
    print(f"    相机前方点数: {step2['points_in_front']} ({step2['percentage_in_front']:.1f}%)")
    print(f"    相机后方点数: {step2['points_behind']}")
    
    print(f"\n  步骤3: FOV过滤")
    step3 = proj_analysis['step3_fov_filter']
    print(f"    相机FOV: {step3['fov_degrees']:.2f}°")
    print(f"    FOV内点数: {step3['points_in_fov']} ({step3['percentage_in_fov']:.1f}%)")
    print(f"    FOV外点数: {step3['points_outside_fov']}")
    print(f"    角度范围: [{step3['theta_range_deg'][0]:.2f}°, {step3['theta_range_deg'][1]:.2f}°]")
    
    if 'step4_project_to_image' in proj_analysis:
        print(f"\n  步骤4: 投影到图像平面")
        step4 = proj_analysis['step4_project_to_image']
        print(f"    投影点数: {step4['points_projected']}")
        if step4['points_projected'] > 0:
            print(f"    U范围: [{step4['u_range'][0]:.1f}, {step4['u_range'][1]:.1f}]")
            print(f"    V范围: [{step4['v_range'][0]:.1f}, {step4['v_range'][1]:.1f}]")
    
    if 'step5_image_bounds' in proj_analysis:
        print(f"\n  步骤5: 图像边界过滤")
        step5 = proj_analysis['step5_image_bounds']
        print(f"    图像内点数: {step5['points_in_image']} ({step5['percentage_in_image']:.1f}%)")
        print(f"    图像外点数: {step5['points_outside_image']}")
    
    if 'depth_distribution' in proj_analysis:
        print(f"\n  深度分布 (最终投影点)")
        depth = proj_analysis['depth_distribution']
        print(f"    范围: [{depth['min']:.2f}m, {depth['max']:.2f}m]")
        print(f"    均值: {depth['mean']:.2f}m, 标准差: {depth['std']:.2f}m")
        print(f"    百分位: 10%={depth['percentiles']['10%']:.2f}m, 50%={depth['percentiles']['50%']:.2f}m, 90%={depth['percentiles']['90%']:.2f}m")
    
    # 4. 问题诊断
    print("\n" + "-" * 70)
    print("4. 问题诊断")
    print("-" * 70)
    
    issues = []
    
    # 检查坐标系
    if not expected_correct:
        issues.append("⚠️  Tr矩阵的坐标系变换不符合KITTI标准")
    
    # 检查点云坐标系
    if pc_analysis['likely_coordinate_system'] != 'LiDAR (X-forward)':
        issues.append(f"⚠️  点云坐标系可能不是标准LiDAR坐标系: {pc_analysis['likely_coordinate_system']}")
    
    # 检查Camera Z范围
    if step1['pts_cam_z_range'][0] < 0:
        issues.append(f"⚠️  变换后有点在相机后方 (Z < 0): {step2['points_behind']} 点")
    
    # 检查FOV过滤
    if step3['percentage_in_fov'] < 50:
        issues.append(f"⚠️  FOV过滤后点数较少: 只有 {step3['percentage_in_fov']:.1f}% 的点在FOV内")
    
    # 检查最终投影点数
    if 'step5_image_bounds' in proj_analysis:
        final_points = proj_analysis['step5_image_bounds']['points_in_image']
        if final_points < 1000:
            issues.append(f"⚠️  最终投影点数较少: {final_points} 点")
    
    if issues:
        print("\n  发现以下问题:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("\n  ✓ 未发现明显问题")
    
    # 保存报告
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'dataset_root': str(dataset_root),
            'sequence': sequence,
            'frame_idx': frame_idx,
            'image_size': [w, h],
            'tr_analysis': tr_analysis,
            'pointcloud_analysis': pc_analysis,
            'projection_analysis': proj_analysis,
            'issues': issues,
        }
        
        report_path = output_dir / f'diagnosis_{sequence}_{frame_idx:06d}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ 报告已保存: {report_path}")
    
    print("\n" + "=" * 70)
    
    return {
        'tr_analysis': tr_analysis,
        'pointcloud_analysis': pc_analysis,
        'projection_analysis': proj_analysis,
        'issues': issues,
    }


def main():
    parser = argparse.ArgumentParser(description='点云投影诊断工具')
    parser.add_argument('--dataset_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--sequence', type=str, default='00', help='序列ID')
    parser.add_argument('--frame', type=int, default=0, help='帧索引')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    diagnose_frame(args.dataset_root, args.sequence, args.frame, args.output_dir)


if __name__ == '__main__':
    main()
