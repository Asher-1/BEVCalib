#!/usr/bin/env python3
"""
Verify that:
  A) int()-cast changes on shape/nx values produce bit-exact output vs tensor originals
  B) scatter_add bev_pool matches the CUDA kernel numerically
"""
import os, sys
import torch
import numpy as np

os.environ["BEV_ZBOUND_STEP"] = "2.0"

_KITTI_DIR = os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib')
if _KITTI_DIR not in sys.path:
    sys.path.insert(0, _KITTI_DIR)

from img_branch.bev_pool.bev_pool import (
    bev_pool, _bev_pool_scatter_add, set_use_scatter_add
)


def make_test_inputs(B, D, H, W, C, num_points, device="cuda"):
    """Generate realistic bev_pool inputs."""
    feats = torch.randn(num_points, C, device=device, dtype=torch.float32)
    coords = torch.stack([
        torch.randint(0, H, (num_points,), device=device),   # coord 0: height
        torch.randint(0, W, (num_points,), device=device),   # coord 1: width
        torch.randint(0, D, (num_points,), device=device),   # coord 2: depth
        torch.randint(0, B, (num_points,), device=device),   # coord 3: batch
    ], dim=1).long()
    return feats, coords


def test_A_int_cast():
    """Verify int()-cast vs tensor-scalar arguments produce identical output."""
    print("=" * 60)
    print("TEST A: int()-cast vs tensor-scalar bev_pool arguments")
    print("=" * 60)

    device = "cuda"
    set_use_scatter_add(False)

    B, D, H, W, C = 2, 10, 100, 100, 64
    num_points = 50000
    feats, coords = make_test_inputs(B, D, H, W, C, num_points, device)

    nx = torch.tensor([H, W, D], dtype=torch.long, device=device)

    out_int = bev_pool(feats, coords, int(B), int(nx[2]), int(nx[0]), int(nx[1]))
    out_tensor = bev_pool(feats, coords, B, nx[2], nx[0], nx[1])

    match = torch.equal(out_int, out_tensor)
    max_diff = (out_int - out_tensor).abs().max().item()

    print(f"  Output shape      : {out_int.shape}")
    print(f"  Bit-exact match   : {match}")
    print(f"  Max abs difference: {max_diff}")

    if match:
        print("  >> PASS: int()-cast does NOT change bev_pool output\n")
    else:
        print(f"  >> FAIL: outputs differ! max_diff={max_diff}\n")
    return match


def test_A_autograd():
    """Verify gradients are identical with int()-cast vs tensor-scalar args."""
    print("=" * 60)
    print("TEST A.grad: int()-cast gradient consistency")
    print("=" * 60)

    device = "cuda"
    set_use_scatter_add(False)

    B, D, H, W, C = 2, 10, 100, 100, 64
    num_points = 50000
    feats_base, coords = make_test_inputs(B, D, H, W, C, num_points, device)

    nx = torch.tensor([H, W, D], dtype=torch.long, device=device)

    feats1 = feats_base.clone().requires_grad_(True)
    out1 = bev_pool(feats1, coords, int(B), int(nx[2]), int(nx[0]), int(nx[1]))
    loss1 = out1.sum()
    loss1.backward()
    grad1 = feats1.grad.clone()

    feats2 = feats_base.clone().requires_grad_(True)
    out2 = bev_pool(feats2, coords, B, nx[2], nx[0], nx[1])
    loss2 = out2.sum()
    loss2.backward()
    grad2 = feats2.grad.clone()

    match = torch.equal(grad1, grad2)
    max_diff = (grad1 - grad2).abs().max().item()

    print(f"  Grad shape        : {grad1.shape}")
    print(f"  Bit-exact match   : {match}")
    print(f"  Max abs difference: {max_diff}")

    if match:
        print("  >> PASS: int()-cast does NOT change gradients\n")
    else:
        print(f"  >> FAIL: gradients differ! max_diff={max_diff}\n")
    return match


def test_B_scatter_add():
    """Verify scatter_add bev_pool matches CUDA kernel numerically."""
    print("=" * 60)
    print("TEST B: scatter_add vs CUDA kernel numerical consistency")
    print("=" * 60)

    device = "cuda"
    B, D, H, W, C = 2, 10, 100, 100, 64
    num_points = 50000
    feats, coords = make_test_inputs(B, D, H, W, C, num_points, device)

    set_use_scatter_add(False)
    out_cuda = bev_pool(feats, coords, B, D, H, W)

    set_use_scatter_add(True)
    out_scatter = bev_pool(feats, coords, B, D, H, W)

    set_use_scatter_add(False)

    match_exact = torch.equal(out_cuda, out_scatter)
    max_diff = (out_cuda - out_scatter).abs().max().item()
    rel_diff = ((out_cuda - out_scatter).abs() /
                (out_cuda.abs().clamp(min=1e-8))).max().item()

    print(f"  Output shape       : {out_cuda.shape}")
    print(f"  CUDA  sum/mean     : {out_cuda.sum().item():.4f} / {out_cuda.mean().item():.6f}")
    print(f"  Scatter sum/mean   : {out_scatter.sum().item():.4f} / {out_scatter.mean().item():.6f}")
    print(f"  Bit-exact match    : {match_exact}")
    print(f"  Max abs difference : {max_diff}")
    print(f"  Max rel difference : {rel_diff}")

    tol = 1e-5
    close = max_diff < tol
    if close:
        print(f"  >> PASS: scatter_add matches CUDA kernel (max_diff < {tol})\n")
    else:
        print(f"  >> FAIL: max_diff={max_diff} exceeds tolerance {tol}\n")

    return close


def test_B_scatter_add_gradient():
    """Verify scatter_add gradient via float64 finite differences (proper methodology)."""
    print("=" * 60)
    print("TEST B.grad: scatter_add gradient correctness")
    print("=" * 60)

    device = "cuda"
    B, D, H, W, C = 1, 3, 20, 20, 8
    num_points = 500

    feats_base = torch.randn(num_points, C, device=device, dtype=torch.float64)
    coords = torch.stack([
        torch.randint(0, H, (num_points,), device=device),
        torch.randint(0, W, (num_points,), device=device),
        torch.randint(0, D, (num_points,), device=device),
        torch.randint(0, B, (num_points,), device=device),
    ], dim=1).long()

    set_use_scatter_add(True)

    eps = 1e-6
    max_err = 0.0
    test_indices = [0, 42, 100, 250, 499]

    for idx in test_indices:
        for ch in [0, C - 1]:
            feats = feats_base.clone().requires_grad_(True)
            out = _bev_pool_scatter_add(feats, coords, B, D, H, W)
            out.sum().backward()
            analytic = feats.grad[idx, ch].item()

            feats_p = feats_base.clone()
            feats_p[idx, ch] += eps
            out_p = _bev_pool_scatter_add(feats_p, coords, B, D, H, W).sum().item()

            feats_m = feats_base.clone()
            feats_m[idx, ch] -= eps
            out_m = _bev_pool_scatter_add(feats_m, coords, B, D, H, W).sum().item()

            fd = (out_p - out_m) / (2 * eps)
            err = abs(fd - analytic)
            max_err = max(max_err, err)

    print(f"  Tested {len(test_indices) * 2} gradient entries (float64, eps={eps})")
    print(f"  Max |analytic - finite_diff| : {max_err:.2e}")

    ok = max_err < 1e-4
    if ok:
        print("  >> PASS: scatter_add gradient is correct\n")
    else:
        print(f"  >> FAIL: gradient error {max_err:.2e} > 1e-4\n")

    set_use_scatter_add(False)
    return ok


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    results = {}
    results["A: int-cast forward"] = test_A_int_cast()
    results["A: int-cast gradient"] = test_A_autograd()
    results["B: scatter_add forward"] = test_B_scatter_add()
    results["B: scatter_add gradient"] = test_B_scatter_add_gradient()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<30s} : {status}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("All tests PASSED - changes are safe for training.")
    else:
        print("Some tests FAILED - investigate before training!")
    sys.exit(0 if all_pass else 1)
