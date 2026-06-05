#!/usr/bin/env python3
"""Smoke test inline Jacobian + MEDW helpers on HTCN (no full training)."""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'kitti-bev-calib'))
os.chdir(os.path.join(ROOT, 'kitti-bev-calib'))

from hybrid_triple_calib import HybridTripleCalib
from custom_dataset import CustomDataset
from train_kitti import (
    PreprocessedDataset,
    collate_fn,
    get_target_size,
    stratified_split_by_sequence,
    _build_val_idx_to_seq,
    _compute_medw_from_val_accum,
    _run_jacobian_eval_inprocess,
    generate_single_perturbation_from_T,
)


class _Args:
    enable_medw_eval = 1
    medw_eval_max_frames = 200
    jacobian_eval_angle_deg = 10.0
    jacobian_eval_batches = 1
    jacobian_eval_n_probes = 3
    perturb_distribution = 'truncated_normal'
    per_axis_prob = 0.3


def main():
    if not torch.cuda.is_available():
        print('SKIP: CUDA not available')
        return 0
    device = torch.device('cuda')
    target_size = get_target_size(use_custom_dataset=True, target_width=640, target_height=360)
    args = _Args()

    print('=== Build HTCN ===')
    model = HybridTripleCalib(
        img_shape=(target_size[1], target_size[0]),
        fusion_backend='hybrid_triple',
        pc_encoder_mode='pointgpt2bev',
        fusion_variant='gated',
        pointgpt_ckpt='/mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_L20.pth',
        pointgpt_config='/mnt/drtraining/user/dahailu/code/ProjFusion/cfg/pointgpt/finetune_fleet_L20.yaml',
        pointgpt_max_depth=60.0,
        iterative_refine=0,
        projfusion_margin=2.5,
    ).to(device)
    assert model.proj_branch.encoder.fnet_3d is None, 'Proj PointGPT should be stripped'
    model.eval()

    train_root = '/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data'
    print('=== Val split loader (Jacobian + MEDW reuse) ===')
    ds = CustomDataset(train_root, target_size=target_size, max_frames_per_seq=50, auto_detect=True)
    _, val_subset, _ = stratified_split_by_sequence(ds, train_ratio=0.8, seed=114514)
    val_ds = PreprocessedDataset(val_subset, target_size, crop=False)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=False)
    val_idx_to_seq = _build_val_idx_to_seq(val_ds, ds)
    identity = torch.eye(4, device=device)
    eval_noise = {'angle_range_deg': 5.0, 'trans_range': 0.15}

    print('=== Jacobian smoke ===')
    jac = _run_jacobian_eval_inprocess(
        model, val_loader, device, args, False, torch.float32, identity, True,
    )
    assert jac is not None, 'Jacobian returned None'
    print(f"  overall={jac['overall']:.3f} verdict={jac['verdict']}")

    print('=== MEDW from val forward reuse (max 2 batches) ===')
    medw_T_pred, medw_T_gt, medw_seqs = [], [], []
    sample_idx = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if batch_idx >= 2 or batch_data is None:
                break
            imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data[:5]
            gt_T_np = np.array(gt_T_to_camera).astype(np.float32)
            init_T_np, _, _ = generate_single_perturbation_from_T(
                gt_T_np, angle_range_deg=eval_noise['angle_range_deg'],
                trans_range=eval_noise['trans_range'], rotation_only=True,
                distribution=args.perturb_distribution, per_axis_prob=args.per_axis_prob,
            )
            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np).to(device)
            init_T_t = torch.from_numpy(init_T_np.astype(np.float32)).to(device)
            B = gt_T_t.shape[0]
            post_T = identity.unsqueeze(0).expand(B, -1, -1)
            K = torch.from_numpy(np.array(intrinsics, dtype=np.float32)).to(device)
            T_pred, _, _ = model(resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, K, masks=masks)
            T_pred_np = T_pred.detach().cpu().numpy()
            for i in range(B):
                medw_seqs.append(val_idx_to_seq.get(sample_idx + i, 'unknown'))
                medw_T_pred.append(T_pred_np[i].copy())
                medw_T_gt.append(gt_T_np[i].copy())
            sample_idx += B

    medw = _compute_medw_from_val_accum(
        medw_T_pred, medw_T_gt, medw_seqs, window=args.medw_eval_max_frames)
    assert medw is not None, 'MEDW returned None'
    print(f"  MEDW200 rot={medw['rot']:.4f}° (from {len(medw_T_pred)} val frames)")
    print('\nEval metrics smoke PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
