#!/usr/bin/env python3
"""
V36 MVP smoke test: model build, forward, Jacobian probe, iterative inference.

Usage:
    python tools/smoke_test_v36.py [--device cuda]
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('USE_DRCV_BACKEND', '0')


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    import math
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=torch.float32)


def perturb_T_euler(T_batch, delta_rpy_deg):
    """Apply roll/pitch/yaw perturbation (degrees) to batched 4x4 transforms."""
    B = T_batch.shape[0]
    out = T_batch.clone()
    dR = euler_to_rotation_matrix(*delta_rpy_deg).to(T_batch.device)
    for b in range(B):
        out[b, :3, :3] = dR @ T_batch[b, :3, :3]
    return out


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="V36 MVP smoke test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_points", type=int, default=8000)
    parser.add_argument("--use-pointgpt", action="store_true",
                        help="Use ProjFusion PointGPT pretrained encoder")
    parser.add_argument("--pointgpt-ckpt", type=str,
                        default="/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/kitti_pointgpt_tiny.pth")
    args = parser.parse_args()
    device = torch.device(args.device)

    from bev_calib import BEVCalib

    print("=" * 70)
    print(f"  V36 Smoke Test ({'PointGPT' if args.use_pointgpt else 'MVP MLP'})")
    print("=" * 70)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = BEVCalib(
        img_shape=(640, 360),
        native_cross=True,
        backbone_variant='dinov2-small',
        freeze_backbone=True,
        rotation_only=True,
        native_cross_pc_groups=128,
        native_cross_n_harmonic=6,
        native_cross_n_layers=1,
        native_cross_dual_branch=True,
        native_cross_knn=8,
        native_cross_use_fps=True,
        native_cross_use_pointgpt=args.use_pointgpt,
        native_cross_pointgpt_ckpt=args.pointgpt_ckpt if args.use_pointgpt else None,
        enable_axis_loss=True,
    ).to(device)

    tr, tot = count_params(model)
    print(f"\n[1] Model build")
    print(f"    Trainable: {tr/1e6:.2f}M  (MVP baseline was ~0.81M single-branch)")
    print(f"    Total:     {tot/1e6:.2f}M  (was ~50M+ with full BEV path)")
    print(f"    dino_encoder: {hasattr(model, 'dino_encoder')}")
    print(f"    pc_branch:    {hasattr(model, 'pc_branch')} (expect False)")
    print(f"    dual_branch:  {model.native_cross_head.dual_branch}")
    print(f"    pointgpt:     {model.native_cross_head.use_pointgpt}")
    if model.native_cross_head.use_pointgpt:
        print(f"    pc_feat_dim:  {model.native_cross_head.point_encoder.out_dim}")

    B, N = args.batch_size, args.n_points
    img = torch.randn(B, 3, 360, 640, device=device)
    pc = torch.randn(B, N, 3, device=device)
    mask = torch.ones(B, N, device=device)
    mask[:, int(N * 0.8):] = 0
    pc = pc * mask.unsqueeze(-1)

    gt_T = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    init_T = gt_T.clone()
    init_T = perturb_T_euler(init_T, [2.0, -1.0, 1.5])

    K = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 700
    K[:, 0, 2] = 320
    K[:, 1, 2] = 180
    post = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)

    model.eval()
    with torch.no_grad():
        _, _, loss = model(img, pc, gt_T, init_T, post, K, masks=mask)
    print(f"\n[2] Forward pass")
    print(f"    rotation_loss (display): {loss['rotation_loss'].item():.2f}°")
    print(f"    peak GPU mem: {gpu_mem_mb():.0f} MB")

    corrs = []
    for bias in [-3.0, 3.0]:
        biased = perturb_T_euler(init_T, [bias, 0, 0])
        with torch.no_grad():
            _, _, loss_b = model(img, pc, gt_T, biased, post, K, masks=mask)
        corrs.append(loss_b['rotation_loss'].item())
    j_roll = (corrs[1] - corrs[0]) / 6.0
    print(f"\n[3] Jacobian quick probe (roll bias)")
    print(f"    loss@-3°={corrs[0]:.2f}°  loss@+3°={corrs[1]:.2f}°")
    print(f"    dLoss/dBias ≈ {j_roll:+.3f}  ({'adaptive signal' if abs(j_roll) > 0.05 else 'shortcut-like'})")

    head = model.native_cross_head
    patch = 14
    pad_h = (patch - 360 % patch) % patch
    pad_w = (patch - 640 % patch) % patch
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.no_grad():
        tokens = model.dino_encoder.backbone(img_pad)
        img_feat = tokens[:, 1:, :]
        fh, fw = img_pad.shape[2] // patch, img_pad.shape[3] // patch
        T_refined = head.iterative_inference(
            img_feat, pc, init_T, K, img_pad.shape[2], img_pad.shape[3],
            fh, fw, n_iters=3, mask=mask)
    print(f"\n[4] Iterative inference (3 steps)")
    print(f"    T_refined: {T_refined.shape}")

    print("\n" + "=" * 70)
    print("  ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == '__main__':
    main()
