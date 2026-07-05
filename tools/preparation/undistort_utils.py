"""Undistortion helpers: OpenCV fisheye vs C++ EquidistantCamera alignment."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def get_cpp_focal_scale(camera_name: str) -> float:
    """Match camera_online_calibrator.cpp get_rectified_camera focal_scale rules."""
    if camera_name.startswith('tra'):
        return -1.0
    if camera_name.startswith('pan'):
        return -0.7 * 2.0 / 3.0
    return -0.7


def equidistant_r(theta: np.ndarray, k2: float, k3: float, k4: float, k5: float) -> np.ndarray:
    """C++ EquidistantCamera::r — theta + k2*θ³ + k3*θ⁵ + k4*θ⁷ + k5*θ⁹."""
    t2 = theta * theta
    t3 = t2 * theta
    t5 = t2 * t3
    t7 = t2 * t5
    t9 = t2 * t7
    return theta + k2 * t3 + k3 * t5 + k4 * t7 + k5 * t9


def build_cpp_undistort_rectify_maps(
    mu: float,
    mv: float,
    u0: float,
    v0: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    width: int,
    height: int,
    focal_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C++ EquidistantCamera::initUndistortRectifyMap (R = I, cx/cy = image center).

    Returns (map1, map2, K_rect) compatible with cv2.remap.
    """
    fx_abs = mu * abs(focal_scale)
    fy_abs = mv * abs(focal_scale)
    K_rect = np.array(
        [[fx_abs, 0.0, width / 2.0], [0.0, fy_abs, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    K_inv = np.linalg.inv(K_rect)

    u_grid, v_grid = np.meshgrid(np.arange(width, dtype=np.float64),
                                  np.arange(height, dtype=np.float64))
    ones = np.ones_like(u_grid)
    xo = np.stack([u_grid.ravel(), v_grid.ravel(), ones.ravel()], axis=0)

    uo = K_inv @ xo
    norm = np.linalg.norm(uo, axis=0)
    theta = np.arccos(np.clip(uo[2] / np.maximum(norm, 1e-12), -1.0, 1.0))
    phi = np.arctan2(uo[1], uo[0])

    r_val = equidistant_r(theta, k1, k2, k3, k4)
    px = mu * r_val * np.cos(phi) + u0
    py = mv * r_val * np.sin(phi) + v0

    map_x = px.reshape(height, width).astype(np.float32)
    map_y = py.reshape(height, width).astype(np.float32)
    map1, map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    return map1, map2, K_rect


def build_opencv_fisheye_undistort_maps(
    K: np.ndarray,
    D: np.ndarray,
    width: int,
    height: int,
    balance: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV fisheye undistort (current prepare_custom_dataset default)."""
    D_col = np.asarray(D, dtype=np.float64).reshape(4, 1)
    fisheye_est = getattr(
        cv2.fisheye,
        'estimateNewCameraMatrixForUndistortRectifyMap',
        cv2.fisheye.estimateNewCameraMatrixForUndistortRectify,
    )
    new_K = fisheye_est(K, D_col, (width, height), np.eye(3), balance=balance, new_size=(width, height))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D_col, np.eye(3), new_K, (width, height), cv2.CV_16SC2)
    return map1, map2, new_K


def scale_intrinsics(K: np.ndarray, out_w: int, out_h: int, in_w: int, in_h: int) -> np.ndarray:
    """Scale pinhole K from (in_w,in_h) to (out_w,out_h)."""
    if (in_w, in_h) == (out_w, out_h):
        return K.copy()
    sx, sy = out_w / in_w, out_h / in_h
    K_out = K.copy()
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out


def undistort_image(
    image: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Remap then optionally resize."""
    out = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if output_size is None:
        return out
    ow, oh = output_size
    h, w = out.shape[:2]
    if (w, h) == (ow, oh):
        return out
    return cv2.resize(out, (ow, oh), interpolation=cv2.INTER_LINEAR)
