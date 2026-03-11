"""
训练可视化工具
用于点云投影可视化、RPY误差计算、TensorBoard图像渲染
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import math
import sys
from pathlib import Path

# 导入标准外参评估工具
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.evaluate_extrinsics import evaluate_sensor_extrinsic


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
    计算预测变换矩阵与真值之间的误差 (使用标准 evaluate_sensor_extrinsic 方法)
    
    ⚠️ 坐标系说明:
    - T矩阵: LiDAR→Camera 变换 (4x4)
    - T[:3, 3]: Camera在LiDAR坐标系中的位置
    - T[:3, :3]: LiDAR→Camera 的旋转矩阵
    
    evaluate_sensor_extrinsic 直接计算误差，无需额外坐标系转换:
    - axis_pos_error = (t_pred - t_gt) * 100  # 已经在LiDAR坐标系
    - axis_angle_error: 轴角误差向量 (度)
    
    Args:
        pred_T: (4, 4) 预测的变换矩阵 (LiDAR → Camera)
        gt_T: (4, 4) 真值变换矩阵 (LiDAR → Camera)
    
    Returns:
        dict: 包含各项误差指标 (LiDAR坐标系)
            - trans_error: 平移总误差 (m)
            - fwd_error, lat_error, ht_error: LiDAR X/Y/Z 方向平移误差 (m)
            - rot_error: 旋转总误差 (deg)
            - roll_error, pitch_error, yaw_error: 轴角误差分量 (deg)
    
    ✅ 与 C++ Eigen 实现完全一致
    ✅ 与 evaluate_extrinsics.py 标准方法一致
    """
    # 使用标准方法计算误差
    angle_error, axis_angle_error, pos_error, axis_pos_error = \
        evaluate_sensor_extrinsic(pred_T, gt_T)
    
    # 返回结果，保持与原接口兼容
    # 注意: axis_pos_error 已经在 LiDAR 坐标系中，单位为 cm
    return {
        # 平移误差 (转换为米)
        'trans_error': pos_error / 100.0,  # cm → m
        'fwd_error': abs(axis_pos_error[0]) / 100.0,   # LiDAR X (前向) cm→m
        'lat_error': abs(axis_pos_error[1]) / 100.0,   # LiDAR Y (横向) cm→m
        'ht_error': abs(axis_pos_error[2]) / 100.0,    # LiDAR Z (高度) cm→m
        
        # 旋转误差 (度)
        'rot_error': angle_error,  # 总旋转误差
        'roll_error': abs(axis_angle_error[0]),   # 轴角误差 X 分量
        'pitch_error': abs(axis_angle_error[1]),  # 轴角误差 Y 分量
        'yaw_error': abs(axis_angle_error[2]),    # 轴角误差 Z 分量
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
        'trans_error': 0, 'fwd_error': 0, 'lat_error': 0, 'ht_error': 0,
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
    将点云投影到图像平面 (纯针孔模型，需要输入已去畸变的图像)
    
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
    H, W = img_copy.shape[:2]

    sorted_indices = np.argsort(-depths)
    pts_2d = pts_2d[sorted_indices]
    depths = depths[sorted_indices]

    u = pts_2d[:, 0].astype(np.int32)
    v = pts_2d[:, 1].astype(np.int32)

    if color_mode == 'depth':
        nd = np.clip(depths / max_depth, 0.0, 1.0)
        colors = np.zeros((len(nd), 3), dtype=np.uint8)
        m0 = nd < 0.25
        m1 = (nd >= 0.25) & (nd < 0.5)
        m2 = (nd >= 0.5) & (nd < 0.75)
        m3 = nd >= 0.75
        colors[m0, 2] = 255
        colors[m0, 1] = (nd[m0] * 4 * 255).astype(np.uint8)
        colors[m1, 2] = ((0.5 - nd[m1]) * 4 * 255).astype(np.uint8)
        colors[m1, 1] = 255
        colors[m2, 1] = 255
        colors[m2, 0] = ((nd[m2] - 0.5) * 4 * 255).astype(np.uint8)
        colors[m3, 1] = ((1.0 - nd[m3]) * 4 * 255).astype(np.uint8)
        colors[m3, 0] = 255
    elif color_mode == 'fixed_green':
        colors = np.full((len(u), 3), (0, 255, 0), dtype=np.uint8)
    elif color_mode == 'fixed_red':
        colors = np.full((len(u), 3), (0, 0, 255), dtype=np.uint8)
    elif color_mode == 'fixed_blue':
        colors = np.full((len(u), 3), (255, 0, 0), dtype=np.uint8)
    else:
        colors = np.full((len(u), 3), (0, 255, 0), dtype=np.uint8)

    if point_radius <= 1:
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        img_copy[v[valid], u[valid]] = colors[valid]
    else:
        for du in range(-point_radius, point_radius + 1):
            for dv in range(-point_radius, point_radius + 1):
                if du * du + dv * dv <= point_radius * point_radius:
                    uc = np.clip(u + du, 0, W - 1)
                    vc = np.clip(v + dv, 0, H - 1)
                    img_copy[vc, uc] = colors

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
    
    cv2.putText(gt_image, "GT (Green)", (10, 20), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(gt_image, f"Points: {len(gt_pts_2d)}", (10, 40), font, font_scale, (255, 255, 255), font_thickness)
    
    cv2.putText(pred_image, "Pred (Red)", (10, 20), font, font_scale, (0, 0, 255), font_thickness)
    cv2.putText(pred_image, f"Points: {len(pred_pts_2d)}", (10, 40), font, font_scale, (255, 255, 255), font_thickness)
    
    error_texts = [
        f"Trans: {pose_errors['trans_error']:.3f}m",
        f"Fwd:{pose_errors['fwd_error']:.3f} Lat:{pose_errors['lat_error']:.3f} Ht:{pose_errors['ht_error']:.3f}m",
        f"Rot: {pose_errors['rot_error']:.2f}deg",
        f"R:{pose_errors['roll_error']:.2f} P:{pose_errors['pitch_error']:.2f} Y:{pose_errors['yaw_error']:.2f}deg",
    ]
    
    for i, text in enumerate(error_texts):
        cv2.putText(pred_image, text, (10, 60 + i * 18), font, font_scale, (0, 255, 255), font_thickness)
    
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
    """
    h, w = image.shape[:2]
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sampled = points[indices]
    else:
        points_sampled = points
    
    gt_pts_2d, gt_depths, _ = project_points_to_image(
        points_sampled, gt_T, K, (h, w)
    )
    pred_pts_2d, pred_depths, _ = project_points_to_image(
        points_sampled, pred_T, K, (h, w)
    )
    
    overlay = image.copy()
    overlay = render_projected_points(
        overlay, gt_pts_2d, gt_depths,
        color_mode='fixed_green', point_radius=point_radius
    )
    overlay = render_projected_points(
        overlay, pred_pts_2d, pred_depths,
        color_mode='fixed_red', point_radius=max(1, point_radius - 1)
    )
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    cv2.putText(overlay, "Overlay: GT(Green) + Pred(Red)", (10, 20), font, font_scale, (255, 255, 255), font_thickness)
    
    error_texts = [
        f"Trans Error: {pose_errors['trans_error']:.3f}m | Fwd:{pose_errors['fwd_error']:.3f} Lat:{pose_errors['lat_error']:.3f} Ht:{pose_errors['ht_error']:.3f}",
        f"Rot Error: {pose_errors['rot_error']:.2f}deg | R:{pose_errors['roll_error']:.2f} P:{pose_errors['pitch_error']:.2f} Y:{pose_errors['yaw_error']:.2f}",
    ]
    
    for i, text in enumerate(error_texts):
        cv2.putText(overlay, text, (10, 40 + i * 18), font, font_scale, (0, 255, 255), font_thickness)
    
    return overlay


def _build_green_yellow_red_lut() -> np.ndarray:
    """Build a 256-entry BGR LUT: green(0) → yellow(128) → red(255)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            s = t / 0.5
            lut[i] = [0, 255, int(255 * s)]           # green → yellow (BGR)
        else:
            s = (t - 0.5) / 0.5
            lut[i] = [0, int(255 * (1 - s)), 255]     # yellow → red (BGR)
    return lut


_GYR_LUT = _build_green_yellow_red_lut()


def _draw_colorbar(img: np.ndarray, x: int, y: int, bar_h: int, bar_w: int,
                   max_val: float, label: str = "px") -> None:
    """Draw a compact vertical colorbar legend (green→yellow→red)."""
    indices = np.linspace(255, 0, bar_h).astype(np.uint8)
    bar_color = _GYR_LUT[indices]
    bar_color = np.repeat(bar_color[:, np.newaxis, :], bar_w, axis=1)

    img[y:y+bar_h, x:x+bar_w] = bar_color

    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (200, 200, 200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.3, 1
    cv2.putText(img, f"{max_val:.0f}{label}", (x + bar_w + 2, y + 10), font, fs, (255, 255, 255), ft)
    cv2.putText(img, f"0{label}", (x + bar_w + 2, y + bar_h), font, fs, (255, 255, 255), ft)


def create_error_analysis_panel(
    image: np.ndarray,
    points: np.ndarray,
    gt_T: np.ndarray,
    pred_T: np.ndarray,
    K: np.ndarray,
    pred_errors: Dict[str, float],
    max_points: int = 5000,
    arrow_sample: int = 60,
    point_radius: int = 1,
    rotation_only: bool = False,
) -> np.ndarray:
    """
    Error analysis panel using displacement-magnitude heatmap + sparse guide arrows.

    Color scheme: green (small error / good alignment) → yellow → red (large error).

    Visualization layers:
      1. Points colored by displacement magnitude (green→yellow→red)
      2. Sparse thin guide arrows (1px) showing displacement direction
      3. Mean displacement indicator arrow
      4. Colorbar legend + compact statistics
    """
    h, w = image.shape[:2]

    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sampled = points[indices]
    else:
        points_sampled = points

    N = len(points_sampled)
    if N == 0:
        panel = image.copy()
        cv2.putText(panel, "No valid points", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return panel

    pts_homo = np.hstack([points_sampled[:, :3], np.ones((N, 1))])
    gt_cam = (gt_T @ pts_homo.T).T
    pred_cam = (pred_T @ pts_homo.T).T

    both_depth_valid = (gt_cam[:, 2] > 0.1) & (pred_cam[:, 2] > 0.1)
    gt_cam_v = gt_cam[both_depth_valid, :3]
    pred_cam_v = pred_cam[both_depth_valid, :3]

    if len(gt_cam_v) == 0:
        panel = image.copy()
        cv2.putText(panel, "No overlapping projections", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return panel

    gt_2d_h = (K @ gt_cam_v.T).T
    gt_2d = gt_2d_h[:, :2] / gt_2d_h[:, 2:3]
    pred_2d_h = (K @ pred_cam_v.T).T
    pred_2d = pred_2d_h[:, :2] / pred_2d_h[:, 2:3]

    in_bounds = (
        (gt_2d[:, 0] >= 0) & (gt_2d[:, 0] < w) &
        (gt_2d[:, 1] >= 0) & (gt_2d[:, 1] < h) &
        (pred_2d[:, 0] >= -w * 0.5) & (pred_2d[:, 0] < w * 1.5) &
        (pred_2d[:, 1] >= -h * 0.5) & (pred_2d[:, 1] < h * 1.5)
    )
    gt_matched = gt_2d[in_bounds]
    pred_matched = pred_2d[in_bounds]

    if len(gt_matched) == 0:
        panel = image.copy()
        cv2.putText(panel, "No in-bounds matches", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return panel

    displacements = pred_matched - gt_matched
    magnitudes = np.linalg.norm(displacements, axis=1)
    p95_disp = np.percentile(magnitudes, 95) if len(magnitudes) > 10 else magnitudes.max()
    max_disp = max(p95_disp, 2.0)

    # ---- Layer 1: dim background ----
    panel = (image.astype(np.float32) * 0.4).astype(np.uint8)

    # ---- Layer 2: points colored by displacement magnitude (plasma colormap) ----
    norm_mag = np.clip(magnitudes / max_disp, 0.0, 1.0)
    color_indices = (norm_mag * 255).astype(np.uint8)
    pt_colors = _GYR_LUT[color_indices]

    u = np.clip(gt_matched[:, 0].astype(int), 0, w - 1)
    v = np.clip(gt_matched[:, 1].astype(int), 0, h - 1)

    sort_idx = np.argsort(-magnitudes)
    pr = max(1, point_radius)
    if pr <= 1:
        panel[v[sort_idx], u[sort_idx]] = pt_colors[sort_idx]
    else:
        for si in sort_idx:
            cv2.circle(panel, (int(u[si]), int(v[si])), pr,
                       pt_colors[si].tolist(), -1, cv2.LINE_AA)

    # ---- Layer 3: sparse thin guide arrows ----
    n_arrows = min(arrow_sample, len(gt_matched))
    above_thresh = np.where(magnitudes > max(1.0, max_disp * 0.05))[0]
    if len(above_thresh) > n_arrows:
        arrow_idx = above_thresh[np.round(np.linspace(0, len(above_thresh) - 1, n_arrows)).astype(int)]
    else:
        arrow_idx = above_thresh

    for idx in arrow_idx:
        gp = gt_matched[idx].astype(int)
        pp = pred_matched[idx].astype(int)
        c = pt_colors[idx].tolist()
        cv2.arrowedLine(panel, tuple(gp), tuple(pp), c, thickness=1, tipLength=0.3,
                        line_type=cv2.LINE_AA)

    # ---- Layer 4: mean displacement indicator ----
    mean_disp = np.mean(displacements, axis=0) if len(displacements) > 0 else np.zeros(2)
    mean_mag = np.linalg.norm(mean_disp)

    if mean_mag > 0.3:
        scale = min(40.0 / max(mean_mag, 1e-6), 5.0)
        scale = max(scale, 1.0)
        cx, cy = w // 2, h // 2
        ex = int(np.clip(cx + mean_disp[0] * scale, 5, w - 5))
        ey = int(np.clip(cy + mean_disp[1] * scale, 5, h - 5))
        cv2.arrowedLine(panel, (cx, cy), (ex, ey), (255, 255, 255), 2,
                        tipLength=0.25, line_type=cv2.LINE_AA)

    # ---- Colorbar legend ----
    bar_h = max(40, h // 6)
    bar_w = 8
    _draw_colorbar(panel, w - bar_w - 35, h - bar_h - 10, bar_h, bar_w, max_disp)

    # ---- Compute per-axis statistics ----
    abs_dx = np.abs(displacements[:, 0])
    abs_dy = np.abs(displacements[:, 1])
    p95_dx = np.percentile(abs_dx, 95) if len(abs_dx) > 10 else abs_dx.max()
    p95_dy = np.percentile(abs_dy, 95) if len(abs_dy) > 10 else abs_dy.max()
    max_dx = abs_dx.max()
    max_dy = abs_dy.max()
    mean_mag_all = np.mean(magnitudes)

    # ---- Quality rating ----
    if mean_mag_all < 1.0:
        quality_label, quality_color = "Excellent (<1px)", (0, 255, 0)
    elif mean_mag_all < 5.0:
        quality_label, quality_color = "Good (<5px)", (255, 255, 0)
    elif mean_mag_all < 15.0:
        quality_label, quality_color = "Fair (<15px)", (0, 180, 255)
    else:
        quality_label, quality_color = "Poor (>15px)", (0, 0, 255)

    # ---- Compact text overlay ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, lh = 0.35, 1, 14

    def _txt(img, text, pos, color):
        (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
        cv2.rectangle(img, (pos[0]-1, pos[1]-th-1), (pos[0]+tw+1, pos[1]+3), (0, 0, 0), -1)
        cv2.putText(img, text, pos, font, fs, color, ft, cv2.LINE_AA)

    y = 12
    _txt(panel, f"Error Analysis | {quality_label}", (4, y), quality_color)
    y += lh
    _txt(panel, f"P95:({p95_dx:.1f},{p95_dy:.1f})px |{p95_disp:.1f}|  Max:({max_dx:.1f},{max_dy:.1f})px |{magnitudes.max():.1f}|", (4, y), (0, 230, 255))
    y += lh
    dx, dy = mean_disp[0], mean_disp[1]
    _txt(panel, f"Mean:({dx:+.1f},{dy:+.1f})px |{mean_mag:.1f}|", (4, y), (200, 200, 255))
    y += lh
    if not rotation_only:
        _txt(panel, f"T:{pred_errors['trans_error']:.3f}m R:{pred_errors['rot_error']:.2f}deg", (4, y), (255, 255, 255))
    else:
        _txt(panel, f"Rot:{pred_errors['rot_error']:.2f}deg", (4, y), (255, 255, 255))
    y += lh
    _txt(panel, f"R:{pred_errors['roll_error']:.2f} P:{pred_errors['pitch_error']:.2f} Y:{pred_errors['yaw_error']:.2f}", (4, y), (0, 255, 255))

    return panel


def _render_info_bar(
    width: int,
    bar_height: int,
    phase: str,
    epoch: int,
    epoch_train_errors: Optional[Dict[str, float]],
    epoch_val_errors: Optional[Dict[str, float]],
    rotation_only: bool,
) -> np.ndarray:
    """Render a thin info bar showing phase, epoch, and train vs val comparison."""
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.38, 1

    def _put(text, x, y, color):
        cv2.putText(bar, text, (x, y), font, fs, color, ft, cv2.LINE_AA)

    x = 8
    y_text = bar_height - 6

    phase_colors = {"Train": (255, 200, 100), "Val": (100, 255, 100), "Eval": (180, 180, 255)}
    pc = phase_colors.get(phase, (220, 220, 220))

    parts = []
    if phase:
        parts.append(f"[{phase}]")
    if epoch >= 0:
        parts.append(f"Epoch {epoch}")
    header = " ".join(parts)
    _put(header, x, y_text, pc)
    x += cv2.getTextSize(header + "  ", font, fs, ft)[0][0]

    if epoch_train_errors:
        rot_t = epoch_train_errors.get('rot_error', -1)
        if rotation_only:
            t_str = f"Train Rot:{rot_t:.2f}deg"
        else:
            trans_t = epoch_train_errors.get('trans_error', -1)
            t_str = f"Train T:{trans_t:.3f}m R:{rot_t:.2f}deg"
        _put(t_str, x, y_text, (255, 200, 100))
        x += cv2.getTextSize(t_str + "   ", font, fs, ft)[0][0]

    if epoch_val_errors:
        rot_v = epoch_val_errors.get('rot_error', -1)
        if rotation_only:
            v_str = f"Val Rot:{rot_v:.2f}deg"
        else:
            trans_v = epoch_val_errors.get('trans_error', -1)
            v_str = f"Val T:{trans_v:.3f}m R:{rot_v:.2f}deg"
        _put(v_str, x, y_text, (100, 255, 100))
        x += cv2.getTextSize(v_str + "   ", font, fs, ft)[0][0]

    if epoch_train_errors and epoch_val_errors:
        rot_t = epoch_train_errors.get('rot_error', 0)
        rot_v = epoch_val_errors.get('rot_error', 0)
        gap = rot_v - rot_t
        gap_str = f"Gap:{gap:+.2f}deg"
        gap_color = (0, 200, 255) if gap < 0.5 else (0, 128, 255) if gap < 2.0 else (0, 0, 255)
        _put(gap_str, x, y_text, gap_color)

    cv2.line(bar, (0, 0), (width, 0), (80, 80, 80), 1)
    return bar


def create_init_gt_pred_comparison(
    image: np.ndarray,
    points: np.ndarray,
    init_T: np.ndarray,
    gt_T: np.ndarray,
    pred_T: np.ndarray,
    K: np.ndarray,
    max_points: int = 5000,
    point_radius: int = 1,
    rotation_only: bool = False,
    phase: str = "",
    epoch: int = -1,
    epoch_train_errors: Optional[Dict[str, float]] = None,
    epoch_val_errors: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    创建 2x2 网格可视化面板 (含可选底部信息栏)

    Layout:
        [Init (depth)       | GT (depth)          ]
        [Pred (depth)       | Error Analysis       ]
        [--- Train vs Val info bar (optional) ---  ]

    Args:
        image: (H, W, 3) BGR图像 (已去畸变)
        points: (N, 3) 点云
        init_T: (4, 4) 初始变换矩阵
        gt_T: (4, 4) 真值变换矩阵
        pred_T: (4, 4) 预测变换矩阵
        K: (3, 3) 相机内参矩阵
        max_points: 最大投影点数
        point_radius: 点半径
        rotation_only: 是否仅优化旋转 (跳过平移相关文字)
        phase: 阶段标签 ("Train" / "Val" / "Eval")
        epoch: 当前训练轮次 (-1 表示不显示)
        epoch_train_errors: 当前 epoch 训练集平均误差
        epoch_val_errors: 当前 epoch 验证集平均误差

    Returns:
        comparison_image: 2x2 网格图像 (可选附加底部信息栏)
    """
    h, w = image.shape[:2]
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sampled = points[indices]
    else:
        points_sampled = points
    
    init_errors = compute_pose_errors(init_T, gt_T)
    pred_errors = compute_pose_errors(pred_T, gt_T)
    
    gt_trans = gt_T[:3, 3]
    gt_euler = rotation_matrix_to_euler_angles(gt_T[:3, :3])
    gt_roll_deg = np.rad2deg(gt_euler[0])
    gt_pitch_deg = np.rad2deg(gt_euler[1])
    gt_yaw_deg = np.rad2deg(gt_euler[2])
    
    # 投影
    init_pts_2d, init_depths, _ = project_points_to_image(points_sampled, init_T, K, (h, w), min_depth=0.1, max_depth=200.0, debug=False)
    init_image = render_projected_points(image, init_pts_2d, init_depths, color_mode='depth', point_radius=point_radius, max_depth=100.0)
    
    gt_pts_2d, gt_depths, _ = project_points_to_image(points_sampled, gt_T, K, (h, w), min_depth=0.1, max_depth=200.0, debug=False)
    gt_image = render_projected_points(image, gt_pts_2d, gt_depths, color_mode='depth', point_radius=point_radius, max_depth=100.0)
    
    pred_pts_2d, pred_depths, _ = project_points_to_image(points_sampled, pred_T, K, (h, w), min_depth=0.1, max_depth=200.0, debug=False)
    pred_image = render_projected_points(image, pred_pts_2d, pred_depths, color_mode='depth', point_radius=point_radius, max_depth=100.0)
    
    if len(gt_pts_2d) == 0:
        cv2.putText(gt_image, "WARNING: No valid projection!", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # --- Text helper ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    line_height = 16
    
    def put_text_with_bg(img, text, pos, color):
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(img, (pos[0]-2, pos[1]-text_h-2), (pos[0]+text_w+2, pos[1]+4), (0, 0, 0), -1)
        cv2.putText(img, text, pos, font, font_scale, color, font_thickness)
    
    # --- Init panel ---
    y = 16
    put_text_with_bg(init_image, f"Init | Pts:{len(init_pts_2d)}", (5, y), (100, 200, 255))
    y += line_height
    if not rotation_only:
        put_text_with_bg(init_image, f"Trans:{init_errors['trans_error']:.3f}m", (5, y), (255, 255, 255))
        y += line_height
        put_text_with_bg(init_image, f"Fwd:{init_errors['fwd_error']:.3f} Lat:{init_errors['lat_error']:.3f} Ht:{init_errors['ht_error']:.3f}m", (5, y), (0, 255, 255))
        y += line_height
    put_text_with_bg(init_image, f"Rot:{init_errors['rot_error']:.2f}deg", (5, y), (255, 255, 255))
    y += line_height
    put_text_with_bg(init_image, f"R:{init_errors['roll_error']:.2f} P:{init_errors['pitch_error']:.2f} Y:{init_errors['yaw_error']:.2f}", (5, y), (0, 255, 255))
    
    # --- GT panel (Zero Error Reference) ---
    y = 16
    put_text_with_bg(gt_image, f"GT (Zero Error Reference) | Pts:{len(gt_pts_2d)}", (5, y), (0, 255, 0))
    y += line_height
    put_text_with_bg(gt_image, f"RPY: {gt_roll_deg:.2f} {gt_pitch_deg:.2f} {gt_yaw_deg:.2f}deg", (5, y), (140, 200, 140))
    if not rotation_only:
        y += line_height
        put_text_with_bg(gt_image, f"XYZ: {gt_trans[0]:.3f} {gt_trans[1]:.3f} {gt_trans[2]:.3f}m", (5, y), (140, 200, 140))
    
    # --- Pred panel (with improvement metric) ---
    y = 16
    put_text_with_bg(pred_image, f"Pred | Pts:{len(pred_pts_2d)}", (5, y), (100, 100, 255))
    y += line_height

    init_rot = init_errors['rot_error']
    pred_rot = pred_errors['rot_error']
    rot_pct = ((init_rot - pred_rot) / init_rot * 100) if init_rot > 1e-6 else 0.0
    rot_improve_str = f"Rot: {init_rot:.2f} -> {pred_rot:.2f}deg"
    if rot_pct > 0:
        rot_improve_str += f" (v{rot_pct:.1f}%)"
    else:
        rot_improve_str += f" (^{abs(rot_pct):.1f}%)"
    improve_color = (0, 255, 0) if rot_pct > 0 else (0, 0, 255)
    put_text_with_bg(pred_image, rot_improve_str, (5, y), improve_color)
    y += line_height
    put_text_with_bg(pred_image, f"R:{pred_errors['roll_error']:.2f} P:{pred_errors['pitch_error']:.2f} Y:{pred_errors['yaw_error']:.2f}", (5, y), (0, 255, 255))

    if not rotation_only:
        y += line_height
        init_trans = init_errors['trans_error']
        pred_trans = pred_errors['trans_error']
        trans_pct = ((init_trans - pred_trans) / init_trans * 100) if init_trans > 1e-6 else 0.0
        trans_improve_str = f"Trans: {init_trans:.3f} -> {pred_trans:.3f}m"
        if trans_pct > 0:
            trans_improve_str += f" (v{trans_pct:.1f}%)"
        else:
            trans_improve_str += f" (^{abs(trans_pct):.1f}%)"
        t_color = (0, 255, 0) if trans_pct > 0 else (0, 0, 255)
        put_text_with_bg(pred_image, trans_improve_str, (5, y), t_color)
        y += line_height
        put_text_with_bg(pred_image, f"Fwd:{pred_errors['fwd_error']:.3f} Lat:{pred_errors['lat_error']:.3f} Ht:{pred_errors['ht_error']:.3f}m", (5, y), (0, 255, 255))
    
    # --- Error analysis panel ---
    error_panel = create_error_analysis_panel(
        image, points_sampled, gt_T, pred_T, K, pred_errors,
        max_points=max_points, arrow_sample=60,
        point_radius=point_radius, rotation_only=rotation_only,
    )

    # --- 2x2 grid ---
    top_row = np.hstack([init_image, gt_image])
    bot_row = np.hstack([pred_image, error_panel])
    comparison = np.vstack([top_row, bot_row])

    # --- Optional info bar (train vs val) ---
    show_bar = phase or epoch >= 0 or epoch_train_errors or epoch_val_errors
    if show_bar:
        bar = _render_info_bar(
            width=comparison.shape[1],
            bar_height=24,
            phase=phase,
            epoch=epoch,
            epoch_train_errors=epoch_train_errors,
            epoch_val_errors=epoch_val_errors,
            rotation_only=rotation_only,
        )
        comparison = np.vstack([comparison, bar])

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
    debug: bool = False,
    rotation_only: bool = False,
    phase: str = "",
    epoch: int = -1,
    epoch_train_errors: Optional[Dict[str, float]] = None,
    epoch_val_errors: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    可视化一个batch中的多个样本
    
    Args:
        images: (B, H, W, 3) BGR图像 (已去畸变)
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
        rotation_only: 是否仅旋转优化
        phase: "Train" / "Val" / "Eval"
        epoch: 训练轮次
        epoch_train_errors: 当前 epoch 训练集平均误差
        epoch_val_errors: 当前 epoch 验证集平均误差
    
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
        
        if masks is not None:
            valid_mask = masks[i] == 1
            points = points[valid_mask]
            if debug and i == 0:
                print(f"[VIS DEBUG] After mask: {len(points)} points")
        
        valid_points = points[np.all(np.abs(points) < 999998, axis=1)]
        
        if debug and i == 0:
            print(f"[VIS DEBUG] After padding filter: {len(valid_points)} points")
        
        if len(valid_points) > 0:
            distance_from_origin = np.linalg.norm(valid_points[:, :3], axis=1)
            valid_points = valid_points[distance_from_origin > 1.0]
        
        if debug and i == 0:
            print(f"[VIS DEBUG] After origin filter: {len(valid_points)} points")
            if len(valid_points) > 0:
                print(f"[VIS DEBUG] Point range: X[{valid_points[:,0].min():.2f}, {valid_points[:,0].max():.2f}], "
                      f"Y[{valid_points[:,1].min():.2f}, {valid_points[:,1].max():.2f}], "
                      f"Z[{valid_points[:,2].min():.2f}, {valid_points[:,2].max():.2f}]")
        
        if len(valid_points) < 100:
            raw_count = len(points_batch[i])
            mask_count = int((masks[i] == 1).sum()) if masks is not None else raw_count
            print(f"[VIS WARNING] Sample {i}: only {len(valid_points)} valid points "
                  f"(raw={raw_count}, after_mask={mask_count}). "
                  f"Possible data pipeline issue -- expected thousands of points per sample.")
        
        init_T = init_T_batch[i]
        gt_T = gt_T_batch[i]
        pred_T = pred_T_batch[i]
        K = K_batch[i]
        
        if debug and i == 0:
            print(f"[VIS DEBUG] GT T matrix:\n{gt_T}")
            print(f"[VIS DEBUG] K matrix:\n{K}")
        
        comparison = create_init_gt_pred_comparison(
            img, valid_points, init_T, gt_T, pred_T, K,
            max_points=max_points, point_radius=point_radius,
            rotation_only=rotation_only,
            phase=phase, epoch=epoch,
            epoch_train_errors=epoch_train_errors,
            epoch_val_errors=epoch_val_errors,
        )
        
        vis_list.append(comparison)
    
    if len(vis_list) > 1:
        max_w = max(v.shape[1] for v in vis_list)
        padded = []
        for v in vis_list:
            if v.shape[1] < max_w:
                pad = np.zeros((v.shape[0], max_w - v.shape[1], 3), dtype=np.uint8)
                v = np.hstack([v, pad])
            padded.append(v)
        vis_image = np.vstack(padded)
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
