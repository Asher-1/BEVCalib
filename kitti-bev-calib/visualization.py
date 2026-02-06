"""
训练可视化工具
用于点云投影可视化、RPY误差计算、TensorBoard图像渲染
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import math


def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为欧拉角 (Roll, Pitch, Yaw)
    使用 ZYX 顺序 (yaw-pitch-roll)
    
    Args:
        R: (3, 3) 旋转矩阵
    
    Returns:
        angles: (3,) [roll, pitch, yaw] in radians
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0
    
    return np.array([roll, pitch, yaw])


def compute_pose_errors(pred_T: np.ndarray, gt_T: np.ndarray) -> Dict[str, float]:
    """
    计算预测变换矩阵与真值之间的误差
    
    Args:
        pred_T: (4, 4) 预测的变换矩阵
        gt_T: (4, 4) 真值变换矩阵
    
    Returns:
        dict: 包含各项误差指标
            - trans_error: 平移总误差 (m)
            - x_error, y_error, z_error: 各轴平移误差 (m)
            - rot_error: 旋转总误差 (deg)
            - roll_error, pitch_error, yaw_error: 各轴旋转误差 (deg)
    """
    # 提取平移向量
    pred_t = pred_T[:3, 3]
    gt_t = gt_T[:3, 3]
    
    # 平移误差
    trans_diff = pred_t - gt_t
    x_error = abs(trans_diff[0])
    y_error = abs(trans_diff[1])
    z_error = abs(trans_diff[2])
    trans_error = np.linalg.norm(trans_diff)
    
    # 提取旋转矩阵
    pred_R = pred_T[:3, :3]
    gt_R = gt_T[:3, :3]
    
    # 计算相对旋转矩阵 (误差旋转矩阵)
    # R_diff 表示从 gt_R 到 pred_R 的旋转差异
    R_diff = pred_R @ gt_R.T
    
    # 总旋转误差 (使用旋转矩阵的迹计算角度-轴表示的角度)
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1.0, 3.0)
    rot_error = np.rad2deg(np.arccos((trace - 1) / 2))
    
    # 将误差旋转矩阵转换为RPY欧拉角
    # 这样得到的才是真正的RPY误差分量
    euler_error = rotation_matrix_to_euler_angles(R_diff)
    
    roll_error = np.abs(np.rad2deg(euler_error[0]))
    pitch_error = np.abs(np.rad2deg(euler_error[1]))
    yaw_error = np.abs(np.rad2deg(euler_error[2]))
    
    return {
        'trans_error': trans_error,
        'x_error': x_error,
        'y_error': y_error,
        'z_error': z_error,
        'rot_error': rot_error,
        'roll_error': roll_error,
        'pitch_error': pitch_error,
        'yaw_error': yaw_error,
    }


def compute_batch_pose_errors(pred_T_batch: torch.Tensor, gt_T_batch: torch.Tensor) -> Dict[str, float]:
    """
    计算一个batch的平均误差
    
    Args:
        pred_T_batch: (B, 4, 4) 预测的变换矩阵
        gt_T_batch: (B, 4, 4) 真值变换矩阵
    
    Returns:
        dict: 平均误差指标
    """
    B = pred_T_batch.shape[0]
    
    errors = {
        'trans_error': 0, 'x_error': 0, 'y_error': 0, 'z_error': 0,
        'rot_error': 0, 'roll_error': 0, 'pitch_error': 0, 'yaw_error': 0,
    }
    
    pred_np = pred_T_batch.detach().cpu().numpy()
    gt_np = gt_T_batch.detach().cpu().numpy()
    
    for i in range(B):
        sample_errors = compute_pose_errors(pred_np[i], gt_np[i])
        for key in errors:
            errors[key] += sample_errors[key]
    
    for key in errors:
        errors[key] /= B
    
    return errors


def project_points_to_image(
    points: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int],
    min_depth: float = 0.1,
    max_depth: float = 200.0,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将点云投影到图像平面
    
    Args:
        points: (N, 3) 或 (N, 4) LiDAR坐标系下的点云
        T: (4, 4) LiDAR→Camera变换矩阵
        K: (3, 3) 相机内参矩阵
        image_size: (height, width)
        min_depth: 最小深度
        max_depth: 最大深度
        debug: 是否输出调试信息
    
    Returns:
        pts_2d: (M, 2) 图像坐标
        depths: (M,) 深度值
        valid_mask: (N,) 有效点的mask
    """
    if len(points) == 0:
        return np.array([]), np.array([]), np.array([], dtype=bool)
    
    # 提取xyz
    pts_3d = points[:, :3]
    
    if debug:
        print(f"[DEBUG] Input points: {len(pts_3d)}, range X:[{pts_3d[:,0].min():.2f},{pts_3d[:,0].max():.2f}], "
              f"Y:[{pts_3d[:,1].min():.2f},{pts_3d[:,1].max():.2f}], Z:[{pts_3d[:,2].min():.2f},{pts_3d[:,2].max():.2f}]")
    
    # 转换为齐次坐标
    pts_3d_homo = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
    
    # LiDAR坐标系 → 相机坐标系
    pts_cam = (T @ pts_3d_homo.T).T  # (N, 4)
    
    if debug:
        print(f"[DEBUG] After transform: range X:[{pts_cam[:,0].min():.2f},{pts_cam[:,0].max():.2f}], "
              f"Y:[{pts_cam[:,1].min():.2f},{pts_cam[:,1].max():.2f}], Z(depth):[{pts_cam[:,2].min():.2f},{pts_cam[:,2].max():.2f}]")
    
    # 过滤相机后方的点和超出深度范围的点
    valid_mask = (pts_cam[:, 2] > min_depth) & (pts_cam[:, 2] < max_depth)
    
    if debug:
        print(f"[DEBUG] Points with valid depth ({min_depth}<z<{max_depth}): {valid_mask.sum()}/{len(pts_cam)}")
    
    pts_cam_valid = pts_cam[valid_mask, :3]
    
    if len(pts_cam_valid) == 0:
        return np.array([]), np.array([]), valid_mask
    
    # 投影到图像平面
    pts_2d_homo = (K @ pts_cam_valid.T).T
    pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]
    
    depths = pts_cam_valid[:, 2]
    
    # 过滤图像边界外的点
    h, w = image_size
    in_bounds = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    
    if debug:
        print(f"[DEBUG] Points in image bounds ({w}x{h}): {in_bounds.sum()}/{len(pts_2d)}")
        if in_bounds.sum() > 0:
            print(f"[DEBUG] 2D coords range: u:[{pts_2d[in_bounds,0].min():.1f},{pts_2d[in_bounds,0].max():.1f}], "
                  f"v:[{pts_2d[in_bounds,1].min():.1f},{pts_2d[in_bounds,1].max():.1f}]")
    
    pts_2d = pts_2d[in_bounds]
    depths = depths[in_bounds]
    
    # 更新有效掩码
    valid_indices = np.where(valid_mask)[0]
    valid_indices = valid_indices[in_bounds]
    full_valid_mask = np.zeros(len(points), dtype=bool)
    full_valid_mask[valid_indices] = True
    
    return pts_2d, depths, full_valid_mask


def render_projected_points(
    image: np.ndarray,
    pts_2d: np.ndarray,
    depths: np.ndarray,
    color_mode: str = 'depth',  # 'depth', 'fixed_green', 'fixed_red', 'fixed_blue'
    point_radius: int = 1,
    max_depth: float = 100.0,
    alpha: float = 1.0
) -> np.ndarray:
    """
    在图像上渲染投影的点云
    
    Args:
        image: (H, W, 3) BGR图像
        pts_2d: (N, 2) 投影点坐标
        depths: (N,) 深度值
        color_mode: 颜色模式
        point_radius: 点半径
        max_depth: 最大深度（用于颜色映射）
        alpha: 透明度
    
    Returns:
        rendered_image: (H, W, 3) 渲染后的图像
    """
    if len(pts_2d) == 0:
        return image.copy()
    
    img_copy = image.copy()
    
    # 按深度排序，远处的点先绘制，近处的点后绘制（覆盖）
    if len(depths) > 0:
        sorted_indices = np.argsort(-depths)  # 从远到近
        pts_2d = pts_2d[sorted_indices]
        depths = depths[sorted_indices]
    
    for i, (pt, depth) in enumerate(zip(pts_2d, depths)):
        u, v = int(pt[0]), int(pt[1])
        
        if color_mode == 'depth':
            # 根据深度映射颜色 - 使用JET colormap风格
            # 近=红色(hot), 远=蓝色(cold)
            normalized_depth = np.clip(depth / max_depth, 0.0, 1.0)
            # 使用更鲜艳的颜色映射
            if normalized_depth < 0.25:
                # 红色到橙色
                r = 255
                g = int(normalized_depth * 4 * 255)
                b = 0
            elif normalized_depth < 0.5:
                # 橙色到黄色到绿色
                r = int((0.5 - normalized_depth) * 4 * 255)
                g = 255
                b = 0
            elif normalized_depth < 0.75:
                # 绿色到青色
                r = 0
                g = 255
                b = int((normalized_depth - 0.5) * 4 * 255)
            else:
                # 青色到蓝色
                r = 0
                g = int((1.0 - normalized_depth) * 4 * 255)
                b = 255
            color = (b, g, r)  # BGR
        elif color_mode == 'fixed_green':
            color = (0, 255, 0)
        elif color_mode == 'fixed_red':
            color = (0, 0, 255)
        elif color_mode == 'fixed_blue':
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        
        cv2.circle(img_copy, (u, v), point_radius, color, -1)
    
    if alpha < 1.0:
        img_copy = cv2.addWeighted(image, 1 - alpha, img_copy, alpha, 0)
    
    return img_copy


def create_projection_comparison(
    image: np.ndarray,
    points: np.ndarray,
    gt_T: np.ndarray,
    pred_T: np.ndarray,
    K: np.ndarray,
    pose_errors: Dict[str, float],
    max_points: int = 5000,
    point_radius: int = 1
) -> np.ndarray:
    """
    创建GT与预测投影对比可视化图像
    
    Args:
        image: (H, W, 3) BGR图像
        points: (N, 3) 点云
        gt_T: (4, 4) 真值变换矩阵
        pred_T: (4, 4) 预测变换矩阵
        K: (3, 3) 相机内参矩阵
        pose_errors: 姿态误差字典
        max_points: 最大投影点数
        point_radius: 点半径
    
    Returns:
        comparison_image: (H, W*2, 3) 对比图像 (左=GT绿色, 右=Pred红色)
    """
    h, w = image.shape[:2]
    
    # 随机采样点云（如果点数过多）
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sampled = points[indices]
    else:
        points_sampled = points
    
    # 投影GT
    gt_pts_2d, gt_depths, _ = project_points_to_image(
        points_sampled, gt_T, K, (h, w)
    )
    
    # 投影预测
    pred_pts_2d, pred_depths, _ = project_points_to_image(
        points_sampled, pred_T, K, (h, w)
    )
    
    # 渲染GT (绿色)
    gt_image = render_projected_points(
        image, gt_pts_2d, gt_depths, 
        color_mode='fixed_green', point_radius=point_radius
    )
    
    # 渲染Pred (红色)
    pred_image = render_projected_points(
        image, pred_pts_2d, pred_depths,
        color_mode='fixed_red', point_radius=point_radius
    )
    
    # 添加文字标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # GT图像标注
    cv2.putText(gt_image, "GT (Green)", (10, 20), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(gt_image, f"Points: {len(gt_pts_2d)}", (10, 40), font, font_scale, (255, 255, 255), font_thickness)
    
    # Pred图像标注
    cv2.putText(pred_image, "Pred (Red)", (10, 20), font, font_scale, (0, 0, 255), font_thickness)
    cv2.putText(pred_image, f"Points: {len(pred_pts_2d)}", (10, 40), font, font_scale, (255, 255, 255), font_thickness)
    
    # 添加误差信息
    error_texts = [
        f"Trans: {pose_errors['trans_error']:.3f}m",
        f"X: {pose_errors['x_error']:.3f}m Y: {pose_errors['y_error']:.3f}m Z: {pose_errors['z_error']:.3f}m",
        f"Rot: {pose_errors['rot_error']:.2f}deg",
        f"R: {pose_errors['roll_error']:.2f} P: {pose_errors['pitch_error']:.2f} Y: {pose_errors['yaw_error']:.2f}deg",
    ]
    
    for i, text in enumerate(error_texts):
        cv2.putText(pred_image, text, (10, 60 + i * 18), font, font_scale, (0, 255, 255), font_thickness)
    
    # 拼接图像
    comparison = np.hstack([gt_image, pred_image])
    
    return comparison


def create_overlay_comparison(
    image: np.ndarray,
    points: np.ndarray,
    gt_T: np.ndarray,
    pred_T: np.ndarray,
    K: np.ndarray,
    pose_errors: Dict[str, float],
    max_points: int = 5000,
    point_radius: int = 1
) -> np.ndarray:
    """
    创建GT与预测投影叠加可视化图像
    GT用绿色，Pred用红色，重叠区域表示误差小
    
    Args:
        image: (H, W, 3) BGR图像
        points: (N, 3) 点云
        gt_T: (4, 4) 真值变换矩阵
        pred_T: (4, 4) 预测变换矩阵
        K: (3, 3) 相机内参矩阵
        pose_errors: 姿态误差字典
        max_points: 最大投影点数
        point_radius: 点半径
    
    Returns:
        overlay_image: (H, W, 3) 叠加图像
    """
    h, w = image.shape[:2]
    
    # 随机采样点云
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sampled = points[indices]
    else:
        points_sampled = points
    
    # 投影GT
    gt_pts_2d, gt_depths, _ = project_points_to_image(
        points_sampled, gt_T, K, (h, w)
    )
    
    # 投影预测
    pred_pts_2d, pred_depths, _ = project_points_to_image(
        points_sampled, pred_T, K, (h, w)
    )
    
    # 创建叠加图像
    overlay = image.copy()
    
    # 先绘制GT (绿色)
    overlay = render_projected_points(
        overlay, gt_pts_2d, gt_depths,
        color_mode='fixed_green', point_radius=point_radius
    )
    
    # 再绘制Pred (红色，较小的点)
    overlay = render_projected_points(
        overlay, pred_pts_2d, pred_depths,
        color_mode='fixed_red', point_radius=max(1, point_radius - 1)
    )
    
    # 添加图例和误差信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    cv2.putText(overlay, "Overlay: GT(Green) + Pred(Red)", (10, 20), font, font_scale, (255, 255, 255), font_thickness)
    
    error_texts = [
        f"Trans Error: {pose_errors['trans_error']:.3f}m | X:{pose_errors['x_error']:.3f} Y:{pose_errors['y_error']:.3f} Z:{pose_errors['z_error']:.3f}",
        f"Rot Error: {pose_errors['rot_error']:.2f}deg | R:{pose_errors['roll_error']:.2f} P:{pose_errors['pitch_error']:.2f} Y:{pose_errors['yaw_error']:.2f}",
    ]
    
    for i, text in enumerate(error_texts):
        cv2.putText(overlay, text, (10, 40 + i * 18), font, font_scale, (0, 255, 255), font_thickness)
    
    return overlay


def create_init_gt_pred_comparison(
    image: np.ndarray,
    points: np.ndarray,
    init_T: np.ndarray,
    gt_T: np.ndarray,
    pred_T: np.ndarray,
    K: np.ndarray,
    max_points: int = 5000,
    point_radius: int = 1
) -> np.ndarray:
    """
    创建初始值、GT、预测值三路对比图像
    
    Args:
        image: (H, W, 3) BGR图像
        points: (N, 3) 点云
        init_T: (4, 4) 初始变换矩阵
        gt_T: (4, 4) 真值变换矩阵
        pred_T: (4, 4) 预测变换矩阵
        K: (3, 3) 相机内参矩阵
        max_points: 最大投影点数
        point_radius: 点半径
    
    Returns:
        comparison_image: (H, W*3, 3) 三路对比图像
    """
    h, w = image.shape[:2]
    
    # 随机采样点云
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sampled = points[indices]
    else:
        points_sampled = points
    
    # 计算各项误差
    init_errors = compute_pose_errors(init_T, gt_T)
    pred_errors = compute_pose_errors(pred_T, gt_T)
    
    # 投影初始值 (使用深度着色)
    init_pts_2d, init_depths, _ = project_points_to_image(points_sampled, init_T, K, (h, w), min_depth=0.1, max_depth=200.0, debug=False)
    init_image = render_projected_points(image, init_pts_2d, init_depths, color_mode='depth', point_radius=point_radius, max_depth=100.0)
    
    # 投影GT (使用深度着色) - 首次调用时输出调试信息
    gt_pts_2d, gt_depths, _ = project_points_to_image(points_sampled, gt_T, K, (h, w), min_depth=0.1, max_depth=200.0, debug=False)
    gt_image = render_projected_points(image, gt_pts_2d, gt_depths, color_mode='depth', point_radius=point_radius, max_depth=100.0)
    
    # 投影预测 (使用深度着色)
    pred_pts_2d, pred_depths, _ = project_points_to_image(points_sampled, pred_T, K, (h, w), min_depth=0.1, max_depth=200.0, debug=False)
    pred_image = render_projected_points(image, pred_pts_2d, pred_depths, color_mode='depth', point_radius=point_radius, max_depth=100.0)
    
    # 如果没有有效投影点，在图像上添加警告
    if len(gt_pts_2d) == 0:
        cv2.putText(gt_image, "WARNING: No valid projection!", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 添加文字标注 - 使用黑色背景增加可读性
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    line_height = 16
    
    def put_text_with_bg(img, text, pos, color):
        """带黑色背景的文字"""
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(img, (pos[0]-2, pos[1]-text_h-2), (pos[0]+text_w+2, pos[1]+4), (0, 0, 0), -1)
        cv2.putText(img, text, pos, font, font_scale, color, font_thickness)
    
    # 初始值图像 - 详细误差
    y = 16
    put_text_with_bg(init_image, f"Init | Pts:{len(init_pts_2d)}", (5, y), (100, 200, 255))
    y += line_height
    put_text_with_bg(init_image, f"Trans:{init_errors['trans_error']:.3f}m", (5, y), (255, 255, 255))
    y += line_height
    put_text_with_bg(init_image, f"X:{init_errors['x_error']:.3f} Y:{init_errors['y_error']:.3f} Z:{init_errors['z_error']:.3f}m", (5, y), (0, 255, 255))
    y += line_height
    put_text_with_bg(init_image, f"Rot:{init_errors['rot_error']:.2f}deg", (5, y), (255, 255, 255))
    y += line_height
    put_text_with_bg(init_image, f"R:{init_errors['roll_error']:.2f} P:{init_errors['pitch_error']:.2f} Y:{init_errors['yaw_error']:.2f}", (5, y), (0, 255, 255))
    
    # GT图像
    y = 16
    put_text_with_bg(gt_image, f"GT (Reference) | Pts:{len(gt_pts_2d)}", (5, y), (0, 255, 0))
    y += line_height
    put_text_with_bg(gt_image, "Ground Truth", (5, y), (255, 255, 255))
    
    # 预测图像 - 详细误差
    y = 16
    put_text_with_bg(pred_image, f"Pred | Pts:{len(pred_pts_2d)}", (5, y), (100, 100, 255))
    y += line_height
    put_text_with_bg(pred_image, f"Trans:{pred_errors['trans_error']:.3f}m", (5, y), (255, 255, 255))
    y += line_height
    put_text_with_bg(pred_image, f"X:{pred_errors['x_error']:.3f} Y:{pred_errors['y_error']:.3f} Z:{pred_errors['z_error']:.3f}m", (5, y), (0, 255, 255))
    y += line_height
    put_text_with_bg(pred_image, f"Rot:{pred_errors['rot_error']:.2f}deg", (5, y), (255, 255, 255))
    y += line_height
    put_text_with_bg(pred_image, f"R:{pred_errors['roll_error']:.2f} P:{pred_errors['pitch_error']:.2f} Y:{pred_errors['yaw_error']:.2f}", (5, y), (0, 255, 255))
    
    # 拼接图像
    comparison = np.hstack([init_image, gt_image, pred_image])
    
    return comparison


def visualize_batch_projection(
    images: np.ndarray,
    points_batch: np.ndarray,
    init_T_batch: np.ndarray,
    gt_T_batch: np.ndarray,
    pred_T_batch: np.ndarray,
    K_batch: np.ndarray,
    masks: Optional[np.ndarray] = None,
    num_samples: int = 2,
    max_points: int = 8000,
    point_radius: int = 1,
    debug: bool = False
) -> np.ndarray:
    """
    可视化一个batch中的多个样本
    
    Args:
        images: (B, H, W, 3) BGR图像
        points_batch: (B, N, 3) 点云
        init_T_batch: (B, 4, 4) 初始变换矩阵
        gt_T_batch: (B, 4, 4) 真值变换矩阵
        pred_T_batch: (B, 4, 4) 预测变换矩阵
        K_batch: (B, 3, 3) 相机内参矩阵
        masks: (B, N) 点云有效掩码
        num_samples: 可视化的样本数
        max_points: 每个样本最大点数
        point_radius: 点半径
        debug: 是否输出调试信息
    
    Returns:
        vis_image: 拼接的可视化图像
    """
    B = min(images.shape[0], num_samples)
    vis_list = []
    
    for i in range(B):
        img = images[i]
        points = points_batch[i]
        
        if debug and i == 0:
            print(f"\n[VIS DEBUG] Sample {i}: image shape={img.shape}, points shape={points.shape}")
        
        # 应用mask过滤无效点
        if masks is not None:
            valid_mask = masks[i] == 1
            points = points[valid_mask]
            if debug and i == 0:
                print(f"[VIS DEBUG] After mask: {len(points)} points")
        
        # 过滤填充的点 (padding的点是999999)
        valid_points = points[np.all(np.abs(points) < 999998, axis=1)]
        
        if debug and i == 0:
            print(f"[VIS DEBUG] After padding filter: {len(valid_points)} points")
        
        # 额外过滤：去除原点附近的点
        if len(valid_points) > 0:
            distance_from_origin = np.linalg.norm(valid_points[:, :3], axis=1)
            valid_points = valid_points[distance_from_origin > 1.0]
        
        if debug and i == 0:
            print(f"[VIS DEBUG] After origin filter: {len(valid_points)} points")
            if len(valid_points) > 0:
                print(f"[VIS DEBUG] Point range: X[{valid_points[:,0].min():.2f}, {valid_points[:,0].max():.2f}], "
                      f"Y[{valid_points[:,1].min():.2f}, {valid_points[:,1].max():.2f}], "
                      f"Z[{valid_points[:,2].min():.2f}, {valid_points[:,2].max():.2f}]")
        
        init_T = init_T_batch[i]
        gt_T = gt_T_batch[i]
        pred_T = pred_T_batch[i]
        K = K_batch[i]
        
        if debug and i == 0:
            print(f"[VIS DEBUG] GT T matrix:\n{gt_T}")
            print(f"[VIS DEBUG] K matrix:\n{K}")
        
        # 创建三路对比图
        comparison = create_init_gt_pred_comparison(
            img, valid_points, init_T, gt_T, pred_T, K,
            max_points=max_points, point_radius=point_radius
        )
        
        vis_list.append(comparison)
    
    # 垂直拼接多个样本
    if len(vis_list) > 1:
        vis_image = np.vstack(vis_list)
    else:
        vis_image = vis_list[0]
    
    return vis_image


def prepare_image_for_tensorboard(image_bgr: np.ndarray) -> np.ndarray:
    """
    将BGR图像转换为TensorBoard格式 (RGB, CHW)
    
    Args:
        image_bgr: (H, W, 3) BGR图像
    
    Returns:
        image_rgb_chw: (3, H, W) RGB图像
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb_chw = image_rgb.transpose(2, 0, 1)
    return image_rgb_chw

