#!/usr/bin/env python3
"""
点云投影可视化工具

支持两种模式：
1. project: 单纯的点云投影到图像
2. compare: 对比去畸变前后的效果
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import Tuple, Optional


# ============================================================
# 投影工具函数
# ============================================================

def get_camera_fov(K: np.ndarray, D: np.ndarray, image_size: tuple) -> float:
    """
    计算相机FOV（完全对齐C++实现）
    
    参考: C++ manual_sensor_calib.cpp: get_camera_fov()
    
    C++实现逻辑：
    1. 获取图像4个角点
    2. 使用cv::undistortPoints去畸变（得到归一化相机坐标）
    3. 计算每个角点的角度：angle = atan(norm(x, y))
    4. 取最大角度并乘以2
    
    Args:
        K: (3, 3) 相机内参矩阵
        D: (N,) 畸变系数
        image_size: (height, width) 图像尺寸
    
    Returns:
        fov: 相机FOV（弧度）
    """
    h, w = image_size
    
    # C++参考: 获取图像的4个角点
    vertex = np.array([
        [0, 0],
        [0, h],
        [w, h],
        [w, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)  # OpenCV要求的格式
    
    # ✅ 使用cv::undistortPoints去畸变（对齐C++）
    if D is not None and len(D) >= 4:
        if len(D) == 4:
            # fisheye model
            vertex_undist = cv2.fisheye.undistortPoints(vertex, K, D)
        else:
            # pinhole model (5 or 8 parameters)
            vertex_undist = cv2.undistortPoints(vertex, K, D)
    else:
        # 无畸变，直接使用K_inv转换
        K_inv = np.linalg.inv(K)
        vertex_undist = []
        for point in vertex[:, 0, :]:
            p_homo = np.array([point[0], point[1], 1.0])
            p_norm = K_inv @ p_homo
            vertex_undist.append(p_norm[:2])
        vertex_undist = np.array(vertex_undist).reshape(-1, 1, 2)
    
    # C++参考: 计算每个角点的角度
    angles = []
    for point in vertex_undist[:, 0, :]:
        # C++参考: double angle = atan(p.norm())
        # 其中p是归一化相机坐标(x, y)
        angle = np.arctan(np.linalg.norm(point))
        angles.append(angle)
    
    # C++参考: fov = max_angle * 2.0
    fov = max(angles) * 2.0
    
    return fov


def load_calib(calib_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    加载KITTI格式的标定文件
    
    Args:
        calib_file: 标定文件路径
    
    Returns:
        Tr: (4, 4) LiDAR→Camera变换矩阵
        K: (3, 3) 相机内参矩阵
        D: (N,) 畸变系数 (pinhole: 5个, fisheye: 4个)
        camera_model: 相机模型 ('pinhole' 或 'fisheye')
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
            
            # 读取非数值字段（如 camera_model）
            if key == 'camera_model':
                calib['camera_model'] = parts[1] if len(parts) > 1 else 'pinhole'
                continue
            
            try:
                values = np.array([float(v) for v in parts[1:]])
            except ValueError:
                # 跳过无法转换为浮点数的行
                continue
            
            if key == 'P2':
                P2 = values.reshape(3, 4)
                calib['K'] = P2[:3, :3]  # 提取内参矩阵
            elif key == 'Tr':
                if len(values) == 12:
                    Tr_3x4 = values.reshape(3, 4)
                    calib['Tr'] = np.vstack([Tr_3x4, [0, 0, 0, 1]])  # 扩展为4x4
                elif len(values) == 16:
                    calib['Tr'] = values.reshape(4, 4)
            elif key == 'D':
                # 畸变系数：pinhole: k1, k2, p1, p2, k3; fisheye: k1, k2, k3, k4
                calib['D'] = values
    
    K = calib.get('K')
    Tr = calib.get('Tr')
    D = calib.get('D', np.zeros(5))
    camera_model = calib.get('camera_model', 'pinhole')
    
    if K is None or Tr is None:
        raise ValueError(f"无法从 {calib_file} 加载P2和Tr")
    
    return Tr, K, D, camera_model


def project_points_to_camera(points: np.ndarray, 
                             Tr: np.ndarray,
                             min_depth: float = 0.0,  # 对齐C++：不过滤近点
                             max_depth: float = 100.0,  # 对齐C++ distance_filter_threshold_
                             use_fov_filter: bool = True,  # ✅ 改为True，对齐C++！
                             K: Optional[np.ndarray] = None,
                             D: Optional[np.ndarray] = None,
                             image_size: Optional[tuple] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将LiDAR点云转换到相机坐标系并过滤（完全对齐C++实现）
    
    参考: C++ manual_sensor_calib.cpp: lidar_cam_fusion_manual()
          - 步骤1: 3D距离过滤
            double distance = sqrt(point.x^2 + point.y^2 + point.z^2);
            if (distance <= max_distance) { ... }
          - 步骤2: FOV过滤
            double theta = abs(atan2(pc.segment(0, 2).norm(), pc(2)));
            if (pc(2) > 0 && theta < 0.5 * fov) { ... }
    
    Args:
        points: (N, 3) 或 (N, 4) LiDAR坐标系下的点云 [x, y, z] 或 [x, y, z, intensity]
        Tr: (4, 4) LiDAR→Camera变换矩阵
        min_depth: 最小3D距离（默认0.0，不过滤近点）
        max_depth: 最大3D距离（默认100.0，对齐C++）
                  注意：这是3D欧几里得距离，不是Z深度！
        use_fov_filter: 是否使用FOV过滤（默认True，对齐C++）
        K: (3, 3) 内参矩阵（FOV过滤时需要）
        D: (N,) 畸变系数（FOV过滤时需要）
        image_size: (height, width) 图像尺寸（FOV过滤时需要）
    
    Returns:
        pts_cam: (M, 3) 相机坐标系下的点云 [x, y, z]
        depths: (M,) Z深度值（用于颜色映射）
        valid_mask: (N,) 有效点的mask
    """
    # 提取xyz
    if points.shape[1] >= 3:
        pts_3d = points[:, :3]
    else:
        raise ValueError("点云至少需要3列(x,y,z)")
    
    # 转换为齐次坐标 (N, 4)
    pts_3d_homo = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
    
    # ✅ 步骤1: LiDAR坐标系 → 相机坐标系
    # C++参考: pc = rot * p + trans
    pts_cam = (Tr @ pts_3d_homo.T).T  # (N, 4)
    
    # ✅ 步骤2: 过滤相机后方的点 + 距离过滤（完全对齐C++）
    # C++参考: filter_pointcloud_by_distance()
    #   double distance = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    #   if (distance <= max_distance) { ... }
    # C++默认: distance_filter_threshold_ = 100.0 (line 424)
    
    # 计算3D欧几里得距离（对齐C++）
    distances_3d = np.sqrt(pts_cam[:, 0]**2 + pts_cam[:, 1]**2 + pts_cam[:, 2]**2)
    
    # 过滤：相机前方 + 3D距离范围内
    valid_mask = (pts_cam[:, 2] > 0) & (distances_3d <= max_depth)
    
    if min_depth > 0:
        valid_mask = valid_mask & (distances_3d >= min_depth)
    
    if use_fov_filter and K is not None and D is not None and image_size is not None:
        # 计算FOV角度（对齐C++实现）
        fov_rad = get_camera_fov(K, D, image_size)
        
        # 计算每个点相对于Z轴的角度
        # C++参考: double theta = abs(atan2(pc.segment(0, 2).norm(), pc(2)))
        xy_norm = np.linalg.norm(pts_cam[:, :2], axis=1)
        theta = np.abs(np.arctan2(xy_norm, pts_cam[:, 2]))
        
        # FOV过滤
        valid_mask = valid_mask & (theta < 0.5 * fov_rad)
    
    pts_cam_valid = pts_cam[valid_mask, :3]  # (M, 3)
    depths = pts_cam[valid_mask, 2]  # (M,)
    
    return pts_cam_valid, depths, valid_mask


def project_camera_to_image(pts_cam: np.ndarray,
                            K: np.ndarray,
                            D: np.ndarray,
                            camera_model: str,
                            image_size: tuple) -> np.ndarray:
    """
    将相机坐标系的点投影到图像平面（完全对齐C++实现）
    
    参考: C++ cv::projectPoints() / cv::fisheye::projectPoints()
    
    Args:
        pts_cam: (N, 3) 相机坐标系下的点云 [x, y, z]
        K: (3, 3) 内参矩阵
        D: (M,) 畸变系数
        camera_model: 相机模型 ('pinhole' 或 'fisheye')
        image_size: (height, width) 图像尺寸
    
    Returns:
        pts_2d: (N, 2) 图像坐标 [u, v]
    """
    # 准备投影参数
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)
    
    if np.any(np.abs(D) > 1e-6):
        # 有畸变，根据相机模型选择投影函数
        pts_3d_cam = pts_cam.reshape(-1, 1, 3).astype(np.float32)
        
        if camera_model == 'fisheye':
            # 使用鱼眼相机投影（对齐C++）
            try:
                pts_2d, _ = cv2.fisheye.projectPoints(
                    pts_3d_cam, rvec, tvec,
                    K.astype(np.float32),
                    D.astype(np.float32)
                )
                pts_2d = pts_2d.reshape(-1, 2)
            except cv2.error as e:
                # 如果fisheye投影失败，回退到普通投影
                print(f"⚠️  鱼眼投影失败，回退到普通投影: {e}")
                pts_2d, _ = cv2.projectPoints(
                    pts_3d_cam, rvec, tvec,
                    K.astype(np.float32),
                    D.astype(np.float32)
                )
                pts_2d = pts_2d.reshape(-1, 2)
        else:
            # 使用针孔相机投影（对齐C++）
            pts_2d, _ = cv2.projectPoints(
                pts_3d_cam, rvec, tvec,
                K.astype(np.float32),
                D.astype(np.float32)
            )
            pts_2d = pts_2d.reshape(-1, 2)
    else:
        # 如果没有畸变，使用矩阵投影
        pts_2d_homo = (K @ pts_cam.T).T  # (N, 3)
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]
    
    return pts_2d


def filter_image_bounds(pts_2d: np.ndarray, 
                        depths: np.ndarray,
                        image_size: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    过滤图像边界外的点
    
    Args:
        pts_2d: (N, 2) 图像坐标
        depths: (N,) 深度值
        image_size: (height, width) 图像尺寸
    
    Returns:
        pts_2d_valid: (M, 2) 有效的图像坐标
        depths_valid: (M,) 有效的深度
        img_mask: (N,) 有效点的mask
    """
    h, w = image_size
    img_mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    
    return pts_2d[img_mask], depths[img_mask], img_mask


def color_from_depth(depth: float, unit_depth: float = 2.0) -> tuple:
    """
    根据深度值生成颜色（完全对齐C++实现）
    
    参考: C++ manual_sensor_calib.cpp: color_from_depth()
    
    Args:
        depth: 深度值（米）
        unit_depth: 单位深度（默认2.0米，对齐C++）
    
    Returns:
        (b, g, r): BGR颜色元组
    """
    # C++: 6色表（注意：C++是BGR格式，OpenCV也是BGR）
    color_table = np.array([
        [255, 0, 0],     # 蓝色
        [0, 255, 0],     # 绿色
        [0, 0, 255],     # 红色
        [255, 255, 0],   # 青色
        [0, 255, 255],   # 黄色
        [255, 0, 255]    # 品红
    ], dtype=np.float32)
    
    color_table_size = 6
    
    # C++逻辑
    depth_scale = depth / unit_depth
    idx = int(np.floor(depth_scale))
    scale = (depth - idx * unit_depth) / unit_depth  # 归一化到[0,1]
    idx = idx % color_table_size
    
    left_color = color_table[idx]
    right_color = color_table[(idx + 1) % color_table_size]
    
    # 线性插值
    color = left_color + scale * (right_color - left_color)
    
    return tuple(color.astype(np.uint8))


def render_points_on_image(image: np.ndarray,
                           pts_2d: np.ndarray,
                           depths: np.ndarray,
                           max_depth: float = 100.0,
                           min_depth: float = 0.0,
                           point_radius: int = 3,
                           unit_depth: float = 2.0) -> np.ndarray:
    """
    在图像上绘制深度着色的点（完全对齐C++实现）
    
    参考: C++ manual_sensor_calib.cpp: lidar_cam_fusion_manual()
          - color_from_depth(depth, unit_depth)
          - circle(img, image_points[i], radius, color, -1)
    
    Args:
        image: 输入图像
        pts_2d: (N, 2) 图像坐标
        depths: (N,) 深度值（Z坐标）
        max_depth: 最大深度（用于过滤，不用于颜色映射）
        min_depth: 最小深度
        point_radius: 点的半径（对齐C++）
        unit_depth: 单位深度（默认2.0米，对齐C++）
    
    Returns:
        img_with_points: 绘制了点的图像
    """
    img_copy = image.copy()
    
    # C++参考: circle(img, image_points[i], radius, color, -1)
    for i in range(len(pts_2d)):
        u, v = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        depth = depths[i]
        
        # 使用C++的颜色映射（6色表 + unit_depth=2.0）
        b, g, r = color_from_depth(depth, unit_depth)
        
        # C++参考: circle(img, image_points[i], radius, color, -1)
        # OpenCV需要int类型的BGR元组
        cv2.circle(img_copy, (u, v), point_radius, (int(b), int(g), int(r)), -1)
    
    return img_copy


def project_and_render(points: np.ndarray,
                      image: np.ndarray,
                      K: np.ndarray,
                      Tr: np.ndarray,
                      D: np.ndarray,
                      camera_model: str = 'pinhole',
                      min_depth: float = 0.0,
                      max_depth: float = 100.0,
                      use_fov_filter: bool = True,  # ✅ 改为True，对齐C++
                      point_radius: int = 3,
                      unit_depth: float = 2.0,  # ✅ 新增：对齐C++的颜色映射
                      verbose: bool = True) -> Tuple[np.ndarray, int]:
    """
    完整的投影和渲染流程（一站式接口，完全对齐C++）
    
    参考: C++ manual_sensor_calib.cpp: lidar_cam_fusion_manual()
    
    Args:
        points: (N, 3) 或 (N, 4) LiDAR点云
        image: 输入图像
        K: (3, 3) 相机内参矩阵
        Tr: (4, 4) LiDAR→Camera变换矩阵
        D: (M,) 畸变系数
        camera_model: 相机模型 ('pinhole' 或 'fisheye')
        min_depth: 最小3D距离（默认0.0，对齐C++）
        max_depth: 最大3D距离（默认100.0，对齐C++）
        use_fov_filter: 是否使用FOV过滤（默认True，对齐C++）
        point_radius: 点的半径
        unit_depth: 颜色映射的单位深度（默认2.0米，对齐C++）
        verbose: 是否打印信息
    
    Returns:
        img_with_points: 绘制了点的图像
        num_valid_points: 有效投影点数
    """
    h, w = image.shape[:2]
    
    # 步骤1: 转换到相机坐标系
    pts_cam, depths, valid_mask = project_points_to_camera(
        points, Tr, min_depth, max_depth, use_fov_filter, K, D, (h, w)
    )
    
    if len(pts_cam) == 0:
        if verbose:
            print("⚠️  没有点在相机前方或FOV内")
        return image.copy(), 0
    
    # 步骤2: 投影到图像平面
    pts_2d = project_camera_to_image(pts_cam, K, D, camera_model, (h, w))
    
    # 步骤3: 过滤图像边界
    pts_2d_valid, depths_valid, img_mask = filter_image_bounds(pts_2d, depths, (h, w))
    
    if verbose:
        print(f"  有效投影点数: {len(pts_2d_valid)} / {len(points)}")
    
    # 步骤4: 渲染到图像（对齐C++的颜色映射）
    img_with_points = render_points_on_image(
        image, pts_2d_valid, depths_valid, max_depth, min_depth, point_radius, unit_depth
    )
    
    return img_with_points, len(pts_2d_valid)


# ============================================================
# 用户功能函数
# ============================================================

def visualize_single_projection(dataset_root: Path, 
                                sequence_id: str, 
                                frame_idx: int,
                                output_path: Optional[Path] = None):
    """
    单纯的点云投影可视化
    
    Args:
        dataset_root: 数据集根目录
        sequence_id: 序列ID
        frame_idx: 帧索引
        output_path: 输出路径（可选）
    """
    seq_dir = dataset_root / 'sequences' / sequence_id
    
    # 构建文件路径
    image_path = seq_dir / 'image_2' / f'{frame_idx:06d}.png'
    pc_path = seq_dir / 'velodyne' / f'{frame_idx:06d}.bin'
    calib_path = seq_dir / 'calib.txt'
    
    # 检查文件存在
    if not image_path.exists():
        print(f"❌ 图像不存在: {image_path}")
        return
    if not pc_path.exists():
        print(f"❌ 点云不存在: {pc_path}")
        return
    if not calib_path.exists():
        print(f"❌ 标定文件不存在: {calib_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"点云投影可视化 - 帧 {frame_idx:06d}")
    print(f"{'='*60}\n")
    
    # 1. 加载图像
    img = cv2.imread(str(image_path))
    print(f"✓ 加载图像: {img.shape}")
    
    # 2. 加载点云
    points = np.fromfile(str(pc_path), dtype=np.float32)
    
    if len(points) % 4 == 0:
        points = points.reshape(-1, 4)
    elif len(points) % 3 == 0:
        points = points.reshape(-1, 3)
    else:
        raise ValueError(f"点云格式错误: {len(points)} 不是3或4的倍数")
    
    print(f"✓ 加载点云: {points.shape}")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # 3. 加载标定参数
    Tr, K, D, camera_model = load_calib(str(calib_path))
    
    print(f"✓ 加载标定参数")
    print(f"  Tr shape: {Tr.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  相机模型: {camera_model}")
    if len(D) == 4:
        print(f"  D (畸变系数/鱼眼): k1={D[0]:.6f}, k2={D[1]:.6f}, k3={D[2]:.6f}, k4={D[3]:.6f}")
    elif len(D) >= 5:
        print(f"  D (畸变系数/针孔): k1={D[0]:.6f}, k2={D[1]:.6f}, p1={D[2]:.6f}, p2={D[3]:.6f}, k3={D[4]:.6f}")
    else:
        print(f"  D (畸变系数): {D}")
    print(f"  点云坐标系: LiDAR系（KITTI标准）")
    
    # 4. 投影点云到图像（完全对齐C++）
    img_with_points, num_valid = project_and_render(
        points, img, K, Tr, D,
        camera_model=camera_model,
        min_depth=0.0,        # 对齐C++：不过滤近点
        max_depth=200.0,      # 用户设置：200米
        use_fov_filter=True,  # ✅ 对齐C++：使用FOV过滤
        point_radius=3,
        unit_depth=2.0,       # ✅ 对齐C++：6色表颜色映射
        verbose=False
    )
    
    print(f"✓ 投影点云: {num_valid}/{len(points)} 个点在图像内")
    
    if num_valid == 0:
        print("❌ 没有点被投影到图像上！")
        return
    
    # 5. 添加信息文本
    text_lines = [
        f"Total points: {len(points)}",
        f"Projected points: {num_valid}",
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(img_with_points, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    # 6. 显示和保存
    if output_path is None:
        output_path = dataset_root / f'projection_{sequence_id}_{frame_idx:06d}.jpg'
    
    cv2.imwrite(str(output_path), img_with_points)
    print(f"✓ 保存结果: {output_path}")
    
    # 显示（缩小以适应屏幕）
    scale = 0.6
    h, w = img_with_points.shape[:2]
    img_display = cv2.resize(img_with_points, (int(w * scale), int(h * scale)))
    
    cv2.imshow('Point Cloud Projection', img_display)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_undistortion(dataset_root: Path,
                         sequence_id: str,
                         frame_idx: int,
                         temp_dir: Optional[Path] = None) -> bool:
    """
    对比去畸变前后的效果
    
    Args:
        dataset_root: 数据集根目录
        sequence_id: 序列ID
        frame_idx: 帧索引
        temp_dir: 临时目录（存储去畸变前的点云）
    
    Returns:
        bool: True表示继续，False表示退出
    """
    print(f"\n{'='*80}")
    print(f"对比帧 {frame_idx:06d} 的去畸变效果")
    print(f"{'='*80}\n")
    
    seq_dir = dataset_root / 'sequences' / sequence_id
    
    # 1. 加载标定参数
    calib_path = seq_dir / 'calib.txt'
    Tr, K, D, camera_model = load_calib(str(calib_path))
    
    print(f"✓ 加载标定参数")
    print(f"  相机模型: {camera_model}")
    
    # 2. 加载图像
    image_path = seq_dir / 'image_2' / f'{frame_idx:06d}.png'
    if not image_path.exists():
        print(f"❌ 图像不存在: {image_path}")
        return False
    
    image = cv2.imread(str(image_path))
    print(f"✓ 加载图像: {image.shape}")
    
    # 3. 加载去畸变后的点云（最终数据）
    pc_after_path = seq_dir / 'velodyne' / f'{frame_idx:06d}.bin'
    if not pc_after_path.exists():
        print(f"❌ 去畸变后点云不存在: {pc_after_path}")
        return False
    
    pc_after = np.fromfile(str(pc_after_path), dtype=np.float32).reshape(-1, 4)
    print(f"✓ 加载去畸变后点云: {pc_after.shape}")
    print(f"  X: [{pc_after[:, 0].min():.2f}, {pc_after[:, 0].max():.2f}]")
    print(f"  Y: [{pc_after[:, 1].min():.2f}, {pc_after[:, 1].max():.2f}]")
    print(f"  Z: [{pc_after[:, 2].min():.2f}, {pc_after[:, 2].max():.2f}]")
    
    # 4. 尝试加载去畸变前的点云（临时文件）
    pc_before = None
    if temp_dir is None:
        temp_dir = dataset_root / 'temp'
    
    if temp_dir.exists():
        temp_pc_files = sorted((temp_dir / 'pointclouds').glob('*.bin'))
        if frame_idx < len(temp_pc_files):
            pc_before_path = temp_pc_files[frame_idx]
            pc_before_data = np.fromfile(str(pc_before_path), dtype=np.float32)
            
            # 判断格式
            if len(pc_before_data) % 5 == 0:
                pc_before = pc_before_data.reshape(-1, 5)[:, :4]
                print(f"✓ 加载去畸变前点云: {pc_before.shape} (从 N×5 格式)")
            elif len(pc_before_data) % 4 == 0:
                pc_before = pc_before_data.reshape(-1, 4)
                print(f"✓ 加载去畸变前点云: {pc_before.shape}")
            
            if pc_before is not None:
                print(f"  X: [{pc_before[:, 0].min():.2f}, {pc_before[:, 0].max():.2f}]")
                print(f"  Y: [{pc_before[:, 1].min():.2f}, {pc_before[:, 1].max():.2f}]")
                print(f"  Z: [{pc_before[:, 2].min():.2f}, {pc_before[:, 2].max():.2f}]")
    
    if pc_before is None:
        print(f"⚠️  未找到去畸变前的点云，只显示去畸变后的结果")
    
    # 5. 投影点云到图像（完全对齐C++）
    print(f"\n投影点云到图像...")
    
    # 去畸变后
    print(f"  [去畸变后]")
    img_after, _ = project_and_render(
        pc_after, image, K, Tr, D,
        camera_model=camera_model,
        min_depth=0.0, max_depth=200.0, use_fov_filter=True,  # ✅ FOV过滤
        point_radius=4, unit_depth=2.0, verbose=True
    )
    
    # 去畸变前
    if pc_before is not None:
        print(f"  [去畸变前]")
        img_before, _ = project_and_render(
            pc_before, image, K, Tr, D,
            camera_model=camera_model,
            min_depth=0.0, max_depth=200.0, use_fov_filter=True,  # ✅ FOV过滤
            point_radius=4, unit_depth=2.0, verbose=True
        )
    else:
        img_before = image.copy()
        cv2.putText(img_before, "No undistorted data", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 6. 并排显示
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_before, "Before Undistortion", (20, 40), font, 1.2, (0, 255, 0), 2)
    cv2.putText(img_after, "After Undistortion", (20, 40), font, 1.2, (0, 255, 0), 2)
    
    # 缩小图像以便并排显示
    scale = 0.5
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_before_small = cv2.resize(img_before, (new_w, new_h))
    img_after_small = cv2.resize(img_after, (new_w, new_h))
    
    # 横向拼接
    comparison = np.hstack([img_before_small, img_after_small])
    
    # 7. 显示和保存
    output_path = dataset_root / f'undistortion_comparison_{frame_idx:06d}.jpg'
    cv2.imwrite(str(output_path), comparison)
    print(f"\n✓ 对比图已保存: {output_path}")
    
    # 显示窗口
    cv2.namedWindow('Undistortion Comparison', cv2.WINDOW_NORMAL)
    cv2.imshow('Undistortion Comparison', comparison)
    print(f"\n按任意键继续，ESC退出...")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return key != 27  # ESC键退出


def batch_compare(dataset_root: Path,
                 sequence_id: str,
                 start_idx: int = 0,
                 num_frames: int = 5,
                 temp_dir: Optional[Path] = None):
    """
    批量对比多帧
    
    Args:
        dataset_root: 数据集根目录
        sequence_id: 序列ID
        start_idx: 起始帧索引
        num_frames: 对比帧数
        temp_dir: 临时目录
    """
    print(f"\n批量对比 {num_frames} 帧，从索引 {start_idx} 开始\n")
    
    for i in range(num_frames):
        frame_idx = start_idx + i
        
        if not compare_undistortion(dataset_root, sequence_id, frame_idx, temp_dir):
            print("用户取消")
            break
        
        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='点云投影可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明:
  project   - 单纯的点云投影到图像
  compare   - 对比去畸变前后的效果

示例:
  # 投影单帧点云
  %(prog)s --mode project --dataset_root /path/to/data --frame 0
  
  # 对比去畸变效果
  %(prog)s --mode compare --dataset_root /path/to/data --frame 0
  
  # 批量对比多帧
  %(prog)s --mode compare --dataset_root /path/to/data --frame 0 --num_frames 5
        """
    )
    
    parser.add_argument('--mode', type=str, required=True, choices=['project', 'compare'],
                       help='运行模式: project=投影, compare=对比去畸变')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--sequence', type=str, default='00',
                       help='序列ID (默认: 00)')
    parser.add_argument('--frame', type=int, default=0,
                       help='帧索引 (默认: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径 (仅project模式，可选)')
    parser.add_argument('--num_frames', type=int, default=1,
                       help='对比帧数 (仅compare模式，默认: 1)')
    parser.add_argument('--temp_dir', type=str, default=None,
                       help='临时目录 (仅compare模式，存储去畸变前的点云)')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    if args.mode == 'project':
        # 投影模式
        output_path = Path(args.output) if args.output else None
        visualize_single_projection(dataset_root, args.sequence, args.frame, output_path)
    elif args.mode == 'compare':
        # 对比模式
        temp_dir = Path(args.temp_dir) if args.temp_dir else None
        if args.num_frames == 1:
            compare_undistortion(dataset_root, args.sequence, args.frame, temp_dir)
        else:
            batch_compare(dataset_root, args.sequence, args.frame, args.num_frames, temp_dir)
    
    print("\n✓ 完成")


if __name__ == '__main__':
    main()
