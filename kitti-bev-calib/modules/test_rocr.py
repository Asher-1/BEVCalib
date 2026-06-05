"""Unit tests for RoCR (Rotation-only Correlation Refine).

Tests:
1. Identity case: zero-offset corr → R_geo ≈ Identity
2. Known rotation: GT-derived corr peak → R_geo ≈ R_gt
3. SVD determinant correction (reflection rejection)
4. RoCR dropout (training anti-shortcut AS4)
5. Low valid_ratio fallback to identity
6. Center negative bias shifts peak away from centre
7. Gradient flow through soft-argmax → SVD
"""

import math
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rocr_refine import RoCR


def _make_intrinsic(fx=500.0, fy=500.0, cx=320.0, cy=180.0, B=2):
    K = torch.zeros(B, 3, 3)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0
    return K


def _axis_angle_to_matrix(axis, angle_deg):
    """Single rotation matrix from axis-angle."""
    angle = math.radians(angle_deg)
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()
    K = torch.zeros(3, 3)
    K[0, 1] = -axis[2]; K[0, 2] = axis[1]
    K[1, 0] = axis[2];  K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]; K[2, 1] = axis[0]
    return torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _project_groups(xyz, K, R):
    """Project xyz through R and K, return pixel coords."""
    xyz_cam = (R @ xyz.unsqueeze(-1)).squeeze(-1)
    depth = xyz_cam[..., 2].clamp(min=0.1)
    u = K[0, 0] * xyz_cam[..., 0] / depth + K[0, 2]
    v = K[1, 1] * xyz_cam[..., 1] / depth + K[1, 2]
    return torch.stack([u, v], dim=-1)


def _build_corr_from_offset(delta_uv_patches, G, radius):
    """Build a corr_map with peak at the specified offset (in patch units)."""
    W = 2 * radius + 1
    corr = torch.zeros(G, W * W)
    for g in range(G):
        dx = int(round(delta_uv_patches[g, 0].item()))
        dy = int(round(delta_uv_patches[g, 1].item()))
        dx = max(-radius, min(radius, dx))
        dy = max(-radius, min(radius, dy))
        idx = (dy + radius) * W + (dx + radius)
        corr[g, idx] = 10.0
    return corr


def test_identity_rotation():
    """When corr peak is at window centre and uv_init is the identity projection,
    R_geo should be close to identity."""
    B, G, radius = 2, 64, 4
    W = 2 * radius + 1
    patch_size = 4.0

    rocr = RoCR(dropout_prob=0.0, center_neg_bias=0.0, temperature=0.1)
    rocr.eval()

    corr_map = torch.zeros(B, G, W * W)
    center_idx = W * W // 2
    corr_map[:, :, center_idx] = 10.0

    xyz = torch.randn(B, G, 3)
    xyz[..., 2] = xyz[..., 2].abs() + 10.0

    K = _make_intrinsic(B=B)
    K_single = K[0]

    uv_pixel = torch.zeros(B, G, 2)
    for b in range(B):
        uv_pixel[b] = _project_groups(xyz[b], K_single, torch.eye(3))
    uv_feat = uv_pixel / patch_size

    valid = (
        (uv_feat[..., 0] >= 0) & (uv_feat[..., 0] < 160)
        & (uv_feat[..., 1] >= 0) & (uv_feat[..., 1] < 90)
    )

    with torch.no_grad():
        out = rocr(corr_map, xyz, uv_feat, K, valid, radius, patch_size, 90, 160)

    R = out['R_geo']
    eye = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    err = (R - eye).norm(dim=(1, 2))
    print(f"[test_identity] R-I Frobenius norm: {err.tolist()}")
    assert err.max() < 0.15, f"Identity test failed: err={err.max():.4f}"
    print("  PASSED")


def test_known_rotation():
    """When corr peaks match a known rotation, R_geo should recover it.

    Uses continuous (non-quantized) corr map to avoid quantization noise,
    and validates the SVD produces R close to R_gt.
    """
    B, G, radius = 1, 128, 8
    W = 2 * radius + 1
    patch_size = 4.0

    rocr = RoCR(dropout_prob=0.0, center_neg_bias=0.0, temperature=0.05)
    rocr.eval()

    R_gt = _axis_angle_to_matrix([0, 1, 0], 3.0)  # 3° yaw
    K_single = _make_intrinsic(B=1)[0]

    torch.manual_seed(123)
    xyz = torch.randn(G, 3) * 3.0
    xyz[:, 2] = xyz[:, 2].abs() + 15.0
    xyz[:, 0] = xyz[:, 0].clamp(-5, 5)
    xyz[:, 1] = xyz[:, 1].clamp(-3, 3)

    R_init = torch.eye(3)
    uv_init_px = _project_groups(xyz, K_single, R_init)
    uv_gt_px = _project_groups(xyz, K_single, R_gt)

    uv_init_feat = uv_init_px / patch_size
    delta_uv_feat = (uv_gt_px - uv_init_px) / patch_size

    corr_map = torch.zeros(1, G, W * W)
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    dy_grid, dx_grid = torch.meshgrid(offsets, offsets, indexing='ij')
    dx_flat = dx_grid.reshape(-1)
    dy_flat = dy_grid.reshape(-1)

    for g in range(G):
        target_dx = delta_uv_feat[g, 0]
        target_dy = delta_uv_feat[g, 1]
        dist_sq = (dx_flat - target_dx) ** 2 + (dy_flat - target_dy) ** 2
        corr_map[0, g] = torch.exp(-dist_sq / 0.3)

    valid_mask = (
        (uv_init_feat[:, 0] >= 0) & (uv_init_feat[:, 0] < 160)
        & (uv_init_feat[:, 1] >= 0) & (uv_init_feat[:, 1] < 90)
        & (delta_uv_feat[:, 0].abs() < radius)
        & (delta_uv_feat[:, 1].abs() < radius)
    )

    xyz_b = xyz.unsqueeze(0)
    uv_init_b = uv_init_feat.unsqueeze(0)
    K_b = K_single.unsqueeze(0)
    valid = valid_mask.unsqueeze(0)

    n_valid = valid.sum().item()
    print(f"[test_known_rotation] {n_valid}/{G} points valid, "
          f"delta_uv range: dx=[{delta_uv_feat[:,0].min():.1f}, {delta_uv_feat[:,0].max():.1f}] "
          f"dy=[{delta_uv_feat[:,1].min():.1f}, {delta_uv_feat[:,1].max():.1f}]")

    with torch.no_grad():
        out = rocr(corr_map, xyz_b, uv_init_b, K_b, valid, radius, patch_size, 90, 160)

    R_pred = out['R_geo'][0]

    R_rel = R_pred @ R_gt.T
    tr = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    cos_angle = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    angular_err = torch.acos(cos_angle).item() * 180.0 / math.pi
    print(f"[test_known_rotation] R_rel trace={tr.item():.6f}, cos={cos_angle.item():.6f}")
    print(f"  Angular error: {angular_err:.2f}° (GT=3° yaw)")
    print(f"  R_pred:\n{R_pred}")
    print(f"  R_gt:\n{R_gt}")
    print(f"  R_rel:\n{R_rel}")
    assert angular_err < 3.0, f"Known rotation test failed: err={angular_err:.2f}°"
    print("  PASSED")


def test_dropout():
    """With dropout_prob=1.0, RoCR should always return identity."""
    B, G, radius = 4, 16, 4
    W = 2 * radius + 1

    rocr = RoCR(dropout_prob=1.0, center_neg_bias=0.0)
    rocr.train()

    corr_map = torch.randn(B, G, W * W)
    xyz = torch.randn(B, G, 3); xyz[..., 2] = xyz[..., 2].abs() + 5
    K = _make_intrinsic(B=B)
    uv = torch.rand(B, G, 2) * 100
    valid = torch.ones(B, G, dtype=torch.bool)

    out = rocr(corr_map, xyz, uv, K, valid, radius, 4.0, 90, 160)
    assert out['skipped'].all(), "Dropout=1.0 should skip all"

    eye = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(out['R_geo'], eye, atol=1e-6), "Skipped should return identity"
    print("[test_dropout] PASSED")


def test_low_valid_ratio():
    """When few groups are valid, RoCR should fall back to identity."""
    B, G, radius = 2, 32, 4
    W = 2 * radius + 1

    rocr = RoCR(min_valid_ratio=0.5, dropout_prob=0.0, center_neg_bias=0.0)
    rocr.eval()

    corr_map = torch.randn(B, G, W * W)
    xyz = torch.randn(B, G, 3); xyz[..., 2] = xyz[..., 2].abs() + 5
    K = _make_intrinsic(B=B)
    uv = torch.rand(B, G, 2) * 100

    valid = torch.zeros(B, G, dtype=torch.bool)
    valid[:, :5] = True  # only 5/32 = 15.6% < 50%

    with torch.no_grad():
        out = rocr(corr_map, xyz, uv, K, valid, radius, 4.0, 90, 160)

    assert out['skipped'].all(), f"Low valid should skip, got {out['skipped']}"
    eye = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(out['R_geo'], eye, atol=1e-6)
    print("[test_low_valid_ratio] PASSED")


def test_center_neg_bias():
    """Center neg bias should shift the soft-argmax away from (0,0)."""
    G, radius = 16, 4
    W = 2 * radius + 1
    center_idx = W * W // 2

    logits_flat = torch.zeros(1, G, W * W)
    logits_flat[:, :, center_idx] = 5.0
    logits_flat[:, :, center_idx + 1] = 4.8

    offset_no_bias = RoCR._soft_argmax_offset(logits_flat / 0.1, radius, W)
    logits_biased = logits_flat.clone()
    logits_biased[:, :, center_idx] -= 1.0
    offset_biased = RoCR._soft_argmax_offset(logits_biased / 0.1, radius, W)

    shift = (offset_biased - offset_no_bias).abs().sum()
    print(f"[test_center_neg_bias] Offset shift: {shift:.4f}")
    assert shift > 0.01, f"Bias should shift peak, but shift={shift:.6f}"
    print("  PASSED")


def test_gradient_flow():
    """Gradients should flow through soft-argmax and SVD back to corr_map."""
    B, G, radius = 1, 32, 4
    W = 2 * radius + 1

    rocr = RoCR(dropout_prob=0.0, center_neg_bias=0.0, temperature=1.0)
    rocr.train()

    corr_map = torch.randn(B, G, W * W, requires_grad=True)
    xyz = torch.randn(B, G, 3); xyz[..., 2] = xyz[..., 2].abs() + 5
    K = _make_intrinsic(B=B)
    uv = torch.rand(B, G, 2) * 100
    valid = torch.ones(B, G, dtype=torch.bool)

    out = rocr(corr_map, xyz, uv, K, valid, radius, 4.0, 90, 160)

    loss = out['R_geo'].sum()
    loss.backward()

    assert corr_map.grad is not None, "No gradient on corr_map!"
    grad_norm = corr_map.grad.norm().item()
    print(f"[test_gradient_flow] Grad norm on corr_map: {grad_norm:.6f}")
    assert grad_norm > 1e-10, f"Gradient too small: {grad_norm}"
    print("  PASSED")


def test_svd_determinant():
    """SVD should produce proper rotations (det=+1), not reflections."""
    B, G = 4, 64

    rocr = RoCR()
    cam_dirs = F.normalize(torch.randn(B, G, 3), dim=-1)
    R_true = _axis_angle_to_matrix([1, 1, 1], 5.0).unsqueeze(0).expand(B, -1, -1)
    lidar_dirs = torch.bmm(cam_dirs, R_true)
    valid = torch.ones(B, G, dtype=torch.bool)

    R_pred = RoCR._batch_rotation_svd(cam_dirs, lidar_dirs, valid)
    det = torch.det(R_pred)
    print(f"[test_svd_determinant] det(R): {det.tolist()}")
    assert torch.allclose(det, torch.ones(B), atol=1e-4), f"det should be +1, got {det}"

    orth_err = (R_pred @ R_pred.transpose(1, 2) - torch.eye(3).unsqueeze(0)).norm(dim=(1, 2))
    assert orth_err.max() < 1e-4, f"Not orthogonal: {orth_err}"
    print("  PASSED")


if __name__ == '__main__':
    torch.manual_seed(42)
    test_identity_rotation()
    test_known_rotation()
    test_dropout()
    test_low_valid_ratio()
    test_center_neg_bias()
    test_gradient_flow()
    test_svd_determinant()
    print("\n=== ALL RoCR TESTS PASSED ===")
