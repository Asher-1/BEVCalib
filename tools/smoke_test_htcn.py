#!/usr/bin/env python3
"""Smoke test for HybridTripleCalib (forward + backward)."""

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'kitti-bev-calib'))
os.chdir(os.path.join(ROOT, 'kitti-bev-calib'))

from hybrid_triple_calib import HybridTripleCalib


def run_one(backend, pc_mode='pointgpt2bev', device='cuda'):
    print(f"\n=== smoke: backend={backend}, pc={pc_mode} ===")
    model = HybridTripleCalib(
        img_shape=(640, 360),
        fusion_backend=backend,
        pc_encoder_mode=pc_mode,
        fusion_variant='gated',
        pointgpt_ckpt='/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_L20.pth',
        pointgpt_config='/mnt/drtraining/user/dahailu/code/ProjFusion/cfg/pointgpt/finetune_fleet_L20.yaml',
        pointgpt_max_depth=60.0,
        deep_supervision_weight=0.2,
        iterative_refine=0,
        projfusion_margin=2.5,
    ).to(device)
    model.train()
    B, N = 2, 4096
    img = torch.randn(B, 3, 360, 640, device=device)
    pc = torch.randn(B, N, 3, device=device) * 10
    gt_T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    init_T = gt_T.clone()
    post_T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    K = torch.tensor([[[700., 0., 320.], [0., 700., 180.], [0., 0., 1.]]], device=device).repeat(B, 1, 1)
    # PointGPT2BEV: bev_mask must not be all-ones on sparse scatter
    if backend == 'bev_only' and pc_mode == 'pointgpt2bev':
        with torch.no_grad():
            xyz_g, feat_g = model.pointgpt_encoder(pc)
            _, bev_mask = model.pc_branch(xyz_g, feat_g)
            fill_ratio = bev_mask.mean().item()
            assert fill_ratio < 0.95, f"bev_mask too dense ({fill_ratio:.2%}), mask bug?"
            print(f"  bev_mask fill_ratio={fill_ratio:.2%} (OK)")

    t_expected, _, loss = model(img, pc, gt_T, init_T, post_T, K)
    loss['total_loss'].backward()
    print(f"  total_loss={loss['total_loss'].item():.4f}, t_expected shape={tuple(t_expected.shape)}")
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"  params_with_grad={n_grad}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-proj', action='store_true')
    args = parser.parse_args()
    run_one('bev_only', pc_mode='pointgpt2bev', device=args.device)
    if not args.skip_proj:
        run_one('hybrid_triple', pc_mode='pointgpt2bev', device=args.device)
        model = HybridTripleCalib(
            img_shape=(640, 360),
            fusion_backend='hybrid_triple',
            pc_encoder_mode='pointgpt2bev',
            fusion_variant='gated',
            pointgpt_ckpt='/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_L20.pth',
            pointgpt_config='/mnt/drtraining/user/dahailu/code/ProjFusion/cfg/pointgpt/finetune_fleet_L20.yaml',
            pointgpt_max_depth=60.0,
            iterative_refine=0,
            projfusion_margin=2.5,
        )
        assert model.proj_branch.encoder.fnet_3d is None
        assert model.pointgpt_encoder is not None
        print("  shared PointGPT: skip load at init, single encoder OK")
    print("\nSmoke test PASSED")


if __name__ == '__main__':
    main()
