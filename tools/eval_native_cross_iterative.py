#!/usr/bin/env python3
"""Compare single-step forward vs native_cross iterative_inference on validation set."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'kitti-bev-calib'))
sys.path.insert(0, ROOT)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('USE_DRCV_BACKEND', '0')

from tools import generate_single_perturbation_from_T  # noqa: E402
from visualization import compute_pose_errors  # noqa: E402
from evaluate_checkpoint import (  # noqa: E402
    _build_eval_custom_dataset,
    _build_model_from_ckpt,
    _resolve_perturbation_from_ckpt,
    make_collate_fn,
)


def _extract_img_feat(model, imgs):
    """DINOv2 patch tokens for native_cross iterative path."""
    patch_size = 14
    B, _, img_h, img_w = imgs.shape
    pad_h = (patch_size - img_h % patch_size) % patch_size
    pad_w = (patch_size - img_w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='reflect')
    img_h_pad, img_w_pad = imgs.shape[2], imgs.shape[3]
    feat_h = img_h_pad // patch_size
    feat_w = img_w_pad // patch_size

    backbone = model.dino_encoder.backbone
    backbone.eval()
    with torch.no_grad():
        tokens = backbone(imgs)
        img_feat = tokens[:, 1:, :] if tokens.shape[1] == feat_h * feat_w + 1 else tokens[:, 1:, :]
    return img_feat, imgs.shape[2], imgs.shape[3], feat_h, feat_w


@torch.no_grad()
def forward_iterative(model, imgs, pcs, init_T, K, masks, n_iters):
    img_feat, img_h, img_w, feat_h, feat_w = _extract_img_feat(model, imgs)
    if K.dim() == 4:
        K = K.squeeze(1)
    if init_T.dim() == 4:
        init_T = init_T.squeeze(1)
    mask_t = None
    if masks is not None:
        mask_t = torch.as_tensor(masks, device=imgs.device)
        if mask_t.dtype != torch.bool:
            mask_t = mask_t.bool()
    return model.native_cross_head.iterative_inference(
        img_feat, pcs, init_T, K, img_h, img_w, feat_h, feat_w,
        n_iters=n_iters, mask=mask_t,
    )


def _summarize_errors(errors_list):
    """Aggregate per-frame error dicts into summary stats."""
    if not errors_list:
        return {}
    keys = ['rot_error', 'roll_error', 'pitch_error', 'yaw_error']
    out = {}
    for k in keys:
        vals = np.array([e[k] for e in errors_list], dtype=np.float64)
        out[k] = {
            'mean': float(vals.mean()),
            'median': float(np.median(vals)),
            'p90': float(np.percentile(vals, 90)),
            'p95': float(np.percentile(vals, 95)),
            'max': float(vals.max()),
        }
    rot = np.array([e['rot_error'] for e in errors_list])
    roll = np.array([e['roll_error'] for e in errors_list])
    pitch = np.array([e['pitch_error'] for e in errors_list])
    yaw = np.array([e['yaw_error'] for e in errors_list])
    out['pct_rot_lt_0p1'] = float((rot < 0.1).mean() * 100)
    out['pct_all_rpy_lt_0p1'] = float(((roll < 0.1) & (pitch < 0.1) & (yaw < 0.1)).mean() * 100)
    out['pct_rot_lt_0p5'] = float((rot < 0.5).mean() * 100)
    out['n_frames'] = len(errors_list)
    return out


def main():
    parser = argparse.ArgumentParser(description='Native cross iterative inference eval')
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--target_height', type=int, default=360)
    parser.add_argument('--target_width', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--angle_range_deg', type=float, default=5.0)
    parser.add_argument('--trans_range', type=float, default=0.15)
    parser.add_argument('--validate_sample_ratio', type=float, default=0.2)
    parser.add_argument('--eval_max_frames_per_seq', type=int, default=500)
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--n_iters_list', type=str, default='1,2,3,5',
                        help='Comma-separated iterative_inference step counts')
    parser.add_argument('--compare_single', action='store_true', default=True,
                        help='Also run standard model forward (single-step training path)')
    parser.add_argument('--pointgpt_ckpt', type=str, default=None,
                        help='Override PointGPT checkpoint path (A/B pretrain comparison)')
    parser.add_argument('--pointgpt_config', type=str, default=None,
                        help='Override PointGPT config yaml')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Shard index for multi-GPU parallel eval (0-based)')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of eval shards')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='CUDA device id for this shard worker')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('[FATAL] CUDA required')
        sys.exit(1)
    if args.gpu_id >= torch.cuda.device_count():
        print(f'[FATAL] gpu_id={args.gpu_id} unavailable (count={torch.cuda.device_count()})')
        sys.exit(1)
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    rotation_only = ckpt.get('rotation_only', True)
    _resolve_perturbation_from_ckpt(args, ckpt)

    print('=' * 72)
    print(f'Checkpoint: {args.ckpt_path}')
    print(f'Epoch: {ckpt.get("epoch", "?")}')
    print(f'Dataset: {args.dataset_root}')
    print(f'Perturbation: {args.angle_range_deg}°, {args.trans_range}m')
    print('=' * 72)

    model, _, p = _build_model_from_ckpt(args, ckpt, device, rotation_only, quiet=True)
    if args.pointgpt_ckpt or args.pointgpt_config:
        from pointgpt_wrapper import PointGPTEncoder
        pg_cfg = args.pointgpt_config or p.get('native_cross_pointgpt_config')
        pg_ckpt = args.pointgpt_ckpt or p.get('native_cross_pointgpt_ckpt')
        max_depth = p.get('native_cross_pointgpt_max_depth', 50.0)
        print(f'[A/B] Override PointGPT: ckpt={pg_ckpt}')
        model.native_cross_head.point_encoder = PointGPTEncoder(
            config_path=pg_cfg,
            checkpoint_path=pg_ckpt,
            max_depth=max_depth,
            freeze=True,
        ).to(device)
    if not getattr(model, 'native_cross', False):
        print('[FATAL] Checkpoint is not native_cross; use standard evaluate_checkpoint.py')
        sys.exit(1)
    model.eval()

    dataset = _build_eval_custom_dataset(args.dataset_root, args)
    val_size = max(1, int(len(dataset) * args.validate_sample_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(114514)
    _, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    if args.num_shards < 1:
        print('[FATAL] num_shards must be >= 1')
        sys.exit(1)
    if not (0 <= args.shard_id < args.num_shards):
        print(f'[FATAL] shard_id={args.shard_id} out of range for num_shards={args.num_shards}')
        sys.exit(1)
    if args.num_shards > 1:
        shard_indices = list(range(args.shard_id, len(eval_dataset), args.num_shards))
        eval_dataset = torch.utils.data.Subset(eval_dataset, shard_indices)
    print(f'Val samples: {len(eval_dataset)} (ratio={args.validate_sample_ratio}, seed=114514, '
          f'shard={args.shard_id}/{args.num_shards}, gpu={args.gpu_id})')

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )

    n_iters_list = [int(x) for x in args.n_iters_list.split(',') if x.strip()]
    modes = []
    if args.compare_single:
        modes.append(('single_forward', None))
    for n in n_iters_list:
        modes.append((f'iterative_{n}', n))

    errors_by_mode = {name: [] for name, _ in modes}
    t0 = time.time()
    max_batches = args.max_batches if args.max_batches > 0 else len(loader)

    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T, intrinsics) in enumerate(loader):
            if batch_idx >= max_batches:
                break

            gt_np = np.array(gt_T).astype(np.float32)
            init_np, _, _ = generate_single_perturbation_from_T(
                gt_np,
                angle_range_deg=args.angle_range_deg,
                trans_range=args.trans_range,
                rotation_only=rotation_only,
            )

            imgs_t = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_np = np.array(pcs)[:, :, :3]
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_t = torch.from_numpy(gt_np).float().to(device)
            init_t = torch.from_numpy(init_np).float().to(device)
            post = torch.eye(4, device=device).unsqueeze(0).expand(gt_t.shape[0], -1, -1)
            K_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

            preds = {}
            if args.compare_single:
                T_single, _, _ = model(
                    imgs_t, pcs_t, gt_t, init_t, post, K_t, masks=masks, out_init_loss=False)
                preds['single_forward'] = T_single.detach().cpu().numpy()

            for name, n_iters in modes:
                if n_iters is None:
                    continue
                T_iter = forward_iterative(model, imgs_t, pcs_t, init_t, K_t, masks, n_iters)
                preds[name] = T_iter.detach().cpu().numpy()

            for i in range(gt_np.shape[0]):
                for name, _ in modes:
                    err = compute_pose_errors(preds[name][i], gt_np[i])
                    errors_by_mode[name].append(err)

            if batch_idx % 20 == 0:
                print(f'  batch {batch_idx}/{max_batches} ...', flush=True)

    elapsed = time.time() - t0
    summary = {
        'checkpoint': args.ckpt_path,
        'epoch': ckpt.get('epoch'),
        'dataset': args.dataset_root,
        'perturbation_deg': args.angle_range_deg,
        'n_val_frames': len(errors_by_mode[modes[0][0]]),
        'elapsed_sec': round(elapsed, 1),
        'shard_id': args.shard_id,
        'num_shards': args.num_shards,
        'gpu_id': args.gpu_id,
        'pointgpt_ckpt': args.pointgpt_ckpt,
        'pointgpt_config': args.pointgpt_config,
        'modes': {},
        'raw_errors_by_mode': errors_by_mode,
    }

    print('\n' + '=' * 72)
    print(f'Results ({summary["n_val_frames"]} val frames, {elapsed:.1f}s)')
    print('=' * 72)
    print(f'{"Mode":<20} {"Rot mean":>9} {"Rot med":>9} {"R/P/Y mean":>22} '
          f'{"<0.1° rot":>10} {"<0.1° RPY":>10}')
    print('-' * 72)

    baseline_rot = None
    for name, _ in modes:
        s = _summarize_errors(errors_by_mode[name])
        summary['modes'][name] = s
        rpy = f"{s['roll_error']['mean']:.2f}/{s['pitch_error']['mean']:.2f}/{s['yaw_error']['mean']:.2f}"
        print(f"{name:<20} {s['rot_error']['mean']:9.3f} {s['rot_error']['median']:9.3f} "
              f"{rpy:>22} {s['pct_rot_lt_0p1']:9.1f}% {s['pct_all_rpy_lt_0p1']:9.1f}%")
        if baseline_rot is None:
            baseline_rot = s['rot_error']['mean']
        elif name.startswith('iterative'):
            gain = (baseline_rot - s['rot_error']['mean']) / baseline_rot * 100
            print(f'  → vs single: {gain:+.1f}% rot mean reduction')

    out_path = args.output_json
    if out_path is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        out_path = os.path.join(ckpt_dir, 'iterative_eval.json')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
