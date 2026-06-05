#!/usr/bin/env python3
"""Smoke backward test: differentiable_epnp=1 must not produce NaN grads in match_head."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "kitti-bev-calib"
sys.path.insert(0, str(ROOT))

from gmp.hybrid_pose_head import HybridPoseHead


def _random_se3(b, device, dtype):
    q = torch.randn(b, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    from losses.quat_tools import batch_quat2mat, batch_tvector2mat
    t = torch.zeros(b, 3, device=device, dtype=dtype)
    return torch.bmm(batch_tvector2mat(t), batch_quat2mat(q))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, g, d = 4, 128, 384
    torch.manual_seed(42)

    head = HybridPoseHead(
        proj_dim=d,
        use_match_head=True,
        use_local_correlation=True,
        differentiable_epnp=True,
        diff_epnp_warmup_epochs=0,
        token_dim=d,
    ).to(device)
    head.train()
    head.set_training_epoch(10)

    f_proj = torch.randn(b, d, device=device, requires_grad=True)
    vit_h, vit_w = 252, 448
    gh, gw = vit_h // 14, vit_w // 14
    p = gh * gw
    cache = {
        "xyz": torch.rand(b, g, 3, device=device),
        "feat_3d": torch.randn(b, g, d, device=device),
        "feat_2d": torch.randn(b, p, d, device=device),
    }
    K = torch.eye(3, device=device).unsqueeze(0).expand(b, -1, -1).clone()
    K[:, 0, 0] = 400
    K[:, 1, 1] = 400
    K[:, 0, 2] = 224
    K[:, 1, 2] = 126
    T_init = _random_se3(b, device, torch.float32)
    T_gt = _random_se3(b, device, torch.float32)

    q_pred, meta = head(f_proj, cache, 60.0, K, T_init, T_gt=T_gt, training=True)
    loss = q_pred.norm(dim=-1).mean()
    if "correspondence_loss" in meta:
        loss = loss + meta["correspondence_loss"]
    loss.backward()

    bad = []
    for name, p in head.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            bad.append(name)
    if bad:
        print("FAIL: NaN/Inf gradients in:", bad)
        sys.exit(1)
    print("PASS: differentiable_epnp=1 backward stable, loss=", float(loss.detach()))
    print("  match_valid_ratio_gt=", float(meta["match_valid_ratio_gt"].mean()))
    print("  match_valid_ratio_init=", float(meta["match_valid_ratio_init"].mean()))
    print("  epnp_insufficient_ratio=", float(meta["epnp_insufficient_ratio"]))
    print("  epnp_mean_effective_points=", float(meta["epnp_mean_effective_points"]))


if __name__ == "__main__":
    main()
