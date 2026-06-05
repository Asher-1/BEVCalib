#!/usr/bin/env python3
"""Unit tests for V40 M-1 camera geometry (K scaling + projection)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera_geometry import (
    project_cam_to_pixel,
    scale_intrinsics_for_resize,
    transform_points_se3,
)


def test_scale_intrinsics_aspect_preserving():
    k = torch.tensor([[[800.0, 0.0, 320.0], [0.0, 800.0, 180.0], [0.0, 0.0, 1.0]]])
    k_vit = scale_intrinsics_for_resize(k, src_h=360, src_w=640, dst_h=252, dst_w=448)
    assert abs(k_vit[0, 0, 0].item() - 800.0 * 448 / 640) < 1e-4
    assert abs(k_vit[0, 1, 1].item() - 800.0 * 252 / 360) < 1e-4
    assert abs(k_vit[0, 0, 2].item() - 320.0 * 448 / 640) < 1e-4
    assert abs(k_vit[0, 1, 2].item() - 180.0 * 252 / 360) < 1e-4


def test_projection_roundtrip_consistency():
    """T_gt projection at K640 and K_vit should match after scale."""
    k640 = torch.tensor([[[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]]])
    k_vit = scale_intrinsics_for_resize(k640, 360, 640, 252, 448)
    t = torch.eye(4).unsqueeze(0)
    t[0, 2, 3] = 1.5
    pts = torch.tensor([[[10.0, 0.5, 30.0]]])
    cam = transform_points_se3(pts, t)
    u640, v640, _ = project_cam_to_pixel(cam, k640)
    u_vit, v_vit, _ = project_cam_to_pixel(cam, k_vit)
    u640_s = u640 * (448 / 640)
    v640_s = v640 * (252 / 360)
    assert abs(u640_s.item() - u_vit.item()) < 0.05
    assert abs(v640_s.item() - v_vit.item()) < 0.05


if __name__ == '__main__':
    test_scale_intrinsics_aspect_preserving()
    test_projection_roundtrip_consistency()
    print('OK: test_camera_geometry')
