#!/usr/bin/env python3
"""Verify GMP monitoring patch: match ratios, EPnP meta, fallback gate, backward."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "kitti-bev-calib"
sys.path.insert(0, str(ROOT))

from gmp.diff_epnp import DifferentiableEPnP
from gmp.hybrid_pose_head import HybridPoseHead
from gmp.match_head import CorrespondenceHead


def _random_se3(b, device, dtype):
    q = torch.randn(b, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    from losses.quat_tools import batch_quat2mat, batch_tvector2mat
    t = torch.zeros(b, 3, device=device, dtype=dtype)
    return torch.bmm(batch_tvector2mat(t), batch_quat2mat(q))


def _make_cache(b, g, d, device):
    vit_h, vit_w = 252, 448
    gh, gw = vit_h // 14, vit_w // 14
    p = gh * gw
    return {
        "xyz": torch.rand(b, g, 3, device=device),
        "feat_3d": torch.randn(b, g, d, device=device),
        "feat_2d": torch.randn(b, p, d, device=device),
    }, vit_h, vit_w


def test_diff_epnp_returns_meta():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, k = 4, 16
    epnp = DifferentiableEPnP(detach_rotation_grad=False).to(device)
    xyz = torch.randn(b, k, 3, device=device)
    uv = torch.rand(b, k, 2, device=device) * 400
    K = torch.eye(3, device=device).unsqueeze(0).expand(b, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 400
    weights = torch.zeros(b, k, device=device)
    weights[:, :8] = 1.0
    T = torch.eye(4, device=device).unsqueeze(0).expand(b, -1, -1)
    r, meta = epnp(xyz, uv, K, weights, T, force_detach=False)
    assert r.shape == (b, 3, 3), r.shape
    assert "epnp_insufficient_ratio" in meta
    assert "epnp_mean_effective_points" in meta
    assert float(meta["epnp_insufficient_ratio"]) == 0.0
    assert float(meta["epnp_mean_effective_points"]) == 8.0
    weights2 = torch.zeros(b, k, device=device)
    weights2[:, :2] = 1.0
    _, meta2 = epnp(xyz, uv, K, weights2, T, force_detach=False)
    assert float(meta2["epnp_insufficient_ratio"]) == 1.0
    assert float(meta2["epnp_mean_effective_points"]) == 2.0
    print("PASS test_diff_epnp_returns_meta")


def test_match_head_ratios():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, g, d = 2, 32, 384
    head = CorrespondenceHead(token_dim=d, num_points=8).to(device)
    pc = torch.randn(b, g, 3, device=device)
    tok = torch.randn(b, g, d, device=device)
    K = torch.eye(3, device=device).unsqueeze(0).expand(b, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 400
    K[:, 0, 2], K[:, 1, 2] = 224, 126
    T_init = _random_se3(b, device, torch.float32)
    T_gt = _random_se3(b, device, torch.float32)
    out = head(pc, tok, T_init, K, 252, 448, T_gt=T_gt)
    assert out["match_valid_ratio_init"].shape == (b,)
    assert out["match_valid_ratio_gt"].shape == (b,)
    assert torch.all(out["match_valid_ratio_gt"] <= out["match_valid_ratio_init"] + 1e-6)
    assert torch.allclose(out["match_valid_ratio"], out["match_valid_ratio_gt"])
    out_inf = head(pc, tok, T_init, K, 252, 448, T_gt=None)
    assert torch.allclose(
        out_inf["match_valid_ratio_gt"], out_inf["match_valid_ratio_init"])
    print("PASS test_match_head_ratios")


def test_hybrid_fallback_uses_gt_ratio():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, g, d = 4, 128, 384
    head = HybridPoseHead(
        proj_dim=d,
        use_match_head=True,
        use_local_correlation=False,
        differentiable_epnp=True,
        diff_epnp_warmup_epochs=0,
        match_valid_ratio_min=0.99,
        token_dim=d,
    ).to(device)
    head.eval()
    cache, _, _ = _make_cache(b, g, d, device)
    f_proj = torch.randn(b, d, device=device)
    K = torch.eye(3, device=device).unsqueeze(0).expand(b, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 400
    T_init = _random_se3(b, device, torch.float32)
    T_gt = _random_se3(b, device, torch.float32)
    with torch.no_grad():
        _, meta = head(f_proj, cache, 60.0, K, T_init, T_gt=T_gt, training=False)
    assert "match_valid_ratio_init" in meta
    assert "match_valid_ratio_gt" in meta
    assert "epnp_insufficient_ratio" in meta
    assert float(meta["match_fallback_ratio"]) == 1.0
    print("PASS test_hybrid_inference_meta")


def test_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, g, d = 4, 128, 384
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
    cache, vit_h, vit_w = _make_cache(b, g, d, device)
    K = torch.eye(3, device=device).unsqueeze(0).expand(b, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 400
    T_init = _random_se3(b, device, torch.float32)
    T_gt = _random_se3(b, device, torch.float32)
    q_pred, meta = head(f_proj, cache, 60.0, K, T_init, T_gt=T_gt, training=True)
    loss = q_pred.norm(dim=-1).mean()
    if "correspondence_loss" in meta:
        loss = loss + meta["correspondence_loss"]
    loss.backward()
    for name, p in head.named_parameters():
        if p.grad is None:
            continue
        assert not torch.isnan(p.grad).any(), name
        assert not torch.isinf(p.grad).any(), name
    print("PASS test_backward")


def main():
    test_diff_epnp_returns_meta()
    test_match_head_ratios()
    test_hybrid_fallback_uses_gt_ratio()
    test_backward()
    print("ALL OK")


if __name__ == "__main__":
    main()
