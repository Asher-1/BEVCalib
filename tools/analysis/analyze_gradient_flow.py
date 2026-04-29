#!/usr/bin/env python3
"""
梯度流向分析: 测量 backbone vs head 的梯度范数比值。

加载一个 checkpoint，跑几个 batch 的 forward+backward，
统计每个参数组的梯度范数，判断 backbone_lr_scale=0.1 是否导致 backbone 欠训练。

用法:
  python analyze_gradient_flow.py \
    --ckpt_path logs/all_training_data/v21/model_small_5deg_v21_generalize_full/all_training_data_scratch/checkpoint/ckpt_best_val.pth \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --num_batches 5
"""
import os
import sys
import argparse
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

def analyze_gradients(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["BEV_ZBOUND_STEP"] = str(args.bev_zbound_step)

    from bev_calib import BEVCalib
    from custom_dataset import CustomDataset
    from torch.utils.data import DataLoader
    from tools import generate_single_perturbation_from_T
    import cv2
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_shape = (360, 640)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get('args', {})
    model = BEVCalib(
        deformable=False,
        bev_encoder=True,
        img_shape=img_shape,
        rotation_only=args.rotation_only,
        enable_axis_loss=True,
        weight_axis_rotation=0.5,
        axis_weights=(1.0, 3.0, 1.0),
        use_geodesic_loss=False,
        use_mlp_head=False,
        voxel_mode=args.voxel_mode,
        scatter_reduce=args.scatter_reduce,
        intrinsic_input=ckpt_args.get('intrinsic_input', False),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.train()

    target_size = (640, 360)

    def crop_and_resize(item, size, intrinsics, crop=False):
        img = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        if crop:
            mid_width = w // 2
            start_x = (w - mid_width) // 2
            cropped = img[:, start_x:start_x + mid_width]
            resized = cv2.resize(cropped, size)
        else:
            resized = cv2.resize(img, size)
        if crop:
            new_cx = intrinsics[0, 2] - start_x
            scale_x = size[0] / mid_width
        else:
            new_cx = intrinsics[0, 2]
            scale_x = size[0] / w
        scale_y = size[1] / h
        new_intrinsics = np.array([
            [intrinsics[0, 0] * scale_x, 0, new_cx * scale_x],
            [0, intrinsics[1, 1] * scale_y, intrinsics[1, 2] * scale_y],
            [0, 0, 1]
        ])
        return resized, new_intrinsics

    dataset = CustomDataset(
        data_folder=args.dataset_root,
        max_frames_per_seq=500, sample_step=None,
    )

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        processed = [crop_and_resize(item[0], target_size, item[3], False) for item in batch]
        imgs = [item[0] for item in processed]
        intrinsics = [item[1] for item in processed]
        gt_T_to_camera = [item[2] for item in batch]
        pcs, masks = [], []
        max_num_points = max(item[1].shape[0] for item in batch)
        for item in batch:
            pc = item[1]
            masks.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(max_num_points - pc.shape[0])], axis=0))
            if pc.shape[0] < max_num_points:
                pc = np.concatenate([pc, np.full((max_num_points - pc.shape[0], pc.shape[1]), 999999)], axis=0)
            pcs.append(pc)
        return imgs, pcs, masks, gt_T_to_camera, intrinsics

    loader = DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=collate_fn, shuffle=True, drop_last=True)

    backbone_names = set()
    head_names = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'img_branch' in name or 'pc_branch' in name:
            backbone_names.add(name)
        else:
            head_names.add(name)

    print(f"Backbone params: {len(backbone_names)}, Head params: {len(head_names)}")
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"{'='*80}")

    grad_stats = defaultdict(lambda: {'norms': [], 'max': [], 'mean': []})

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        imgs, pcs, masks, gt_T, intrinsics = batch_data
        gt_T = np.array(gt_T).astype(np.float32)
        init_T, _, _ = generate_single_perturbation_from_T(
            gt_T, angle_range_deg=5.0, trans_range=0.15)

        resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
        pcs_np = np.array(pcs)[:, :, :3]
        pcs_t = torch.from_numpy(pcs_np).float().to(device)
        gt_T_t = torch.from_numpy(gt_T).float().to(device)
        init_T_t = torch.from_numpy(init_T).float().to(device)
        post_T = torch.eye(4).unsqueeze(0).repeat(gt_T.shape[0], 1, 1).float().to(device)
        intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

        model.zero_grad()
        _, _, loss = model(
            resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, intrinsic_t,
            masks=masks, out_init_loss=False)
        loss["total_loss"].backward()

        bb_norms, hd_norms = [], []
        per_module = defaultdict(list)

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            gnorm = param.grad.data.norm(2).item()
            gmax = param.grad.data.abs().max().item()
            gmean = param.grad.data.abs().mean().item()

            if name in backbone_names:
                group = 'backbone'
                bb_norms.append(gnorm)
            else:
                group = 'head'
                hd_norms.append(gnorm)

            module = name.split('.')[0]
            per_module[module].append(gnorm)

            grad_stats[name]['norms'].append(gnorm)
            grad_stats[name]['max'].append(gmax)
            grad_stats[name]['mean'].append(gmean)

        bb_total = np.sqrt(sum(n**2 for n in bb_norms)) if bb_norms else 0
        hd_total = np.sqrt(sum(n**2 for n in hd_norms)) if hd_norms else 0

        print(f"\nBatch {batch_idx+1}/{args.num_batches}:")
        print(f"  Loss: {loss['total_loss'].item():.4f}")
        print(f"  Backbone grad L2 norm: {bb_total:.6f}")
        print(f"  Head     grad L2 norm: {hd_total:.6f}")
        ratio = hd_total / bb_total if bb_total > 0 else float('inf')
        print(f"  Head/Backbone ratio:   {ratio:.2f}x")

        print(f"  Per-module grad norms:")
        for mod, norms in sorted(per_module.items()):
            mod_total = np.sqrt(sum(n**2 for n in norms))
            print(f"    {mod:30s}: L2={mod_total:.6f} (params={len(norms)})")

    print(f"\n{'='*80}")
    print("SUMMARY: Effective parameter update magnitude")
    print(f"{'='*80}")

    bb_avg_norms = []
    hd_avg_norms = []
    for name, stats in grad_stats.items():
        avg_norm = np.mean(stats['norms'])
        if name in backbone_names:
            bb_avg_norms.append(avg_norm)
        else:
            hd_avg_norms.append(avg_norm)

    bb_avg = np.mean(bb_avg_norms) if bb_avg_norms else 0
    hd_avg = np.mean(hd_avg_norms) if hd_avg_norms else 0

    print(f"\nAverage per-param gradient norm:")
    print(f"  Backbone: {bb_avg:.8f}")
    print(f"  Head:     {hd_avg:.8f}")
    print(f"  Ratio:    {hd_avg/bb_avg:.2f}x" if bb_avg > 0 else "  Ratio:    inf")

    bb_lr = args.lr * args.backbone_lr_scale
    hd_lr = args.lr

    bb_update = bb_avg * bb_lr
    hd_update = hd_avg * hd_lr

    print(f"\nEffective update magnitude (grad_norm × lr):")
    print(f"  Backbone: {bb_avg:.8f} × {bb_lr:.2e} = {bb_update:.10f}")
    print(f"  Head:     {hd_avg:.8f} × {hd_lr:.2e} = {hd_update:.10f}")
    total_ratio = hd_update / bb_update if bb_update > 0 else float('inf')
    print(f"  Head/Backbone effective update ratio: {total_ratio:.1f}x")

    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    if total_ratio > 50:
        print(f"  [SEVERE] Head updates {total_ratio:.0f}x faster than backbone.")
        print(f"  Backbone is likely frozen/under-trained. Consider backbone_lr_scale >= 0.3")
    elif total_ratio > 10:
        print(f"  [WARNING] Head updates {total_ratio:.0f}x faster than backbone.")
        print(f"  Backbone may be under-trained. Consider backbone_lr_scale = 0.3~0.5")
    else:
        print(f"  [OK] Head/backbone update ratio = {total_ratio:.0f}x. Seems reasonable.")

    print(f"\nTop-10 backbone params with LARGEST gradients:")
    bb_sorted = sorted(
        [(name, np.mean(stats['norms'])) for name, stats in grad_stats.items() if name in backbone_names],
        key=lambda x: -x[1])[:10]
    for name, norm in bb_sorted:
        print(f"  {norm:.8f}  {name}")

    print(f"\nTop-10 head params with LARGEST gradients:")
    hd_sorted = sorted(
        [(name, np.mean(stats['norms'])) for name, stats in grad_stats.items() if name in head_names],
        key=lambda x: -x[1])[:10]
    for name, norm in hd_sorted:
        print(f"  {norm:.8f}  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str,
                        default="/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data")
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--bev_zbound_step", type=float, default=4.0)
    parser.add_argument("--rotation_only", type=int, default=1)
    parser.add_argument("--voxel_mode", type=str, default="hard")
    parser.add_argument("--scatter_reduce", type=str, default="sum")
    args = parser.parse_args()
    analyze_gradients(args)
