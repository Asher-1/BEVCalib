"""
DST-Calib 假说验证: 屏蔽相机图像后模型表现是否不变？

方法: Hook model.img_branch 的 forward，替换输入图像为 zeros/random。
"""

import sys, os, argparse, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

from evaluate_checkpoint import (
    _auto_permute_spconv_weights,
    _adapt_model_to_checkpoint,
    _build_eval_custom_dataset,
    make_collate_fn,
)
from bev_calib import BEVCalib
from visualization import compute_pose_errors

BYPASS_MODE = 'normal'


class _ImgBypassHook:
    """Hook to replace camera images in img_branch.forward."""

    def __init__(self, mode='normal'):
        self.mode = mode

    def __call__(self, module, args, kwargs):
        if self.mode == 'normal':
            return args, kwargs
        for k, v in kwargs.items():
            if k == 'imgs' and torch.is_tensor(v):
                if self.mode == 'zero':
                    kwargs[k] = torch.zeros_like(v)
                elif self.mode == 'random':
                    kwargs[k] = torch.randn_like(v) * 0.5
        if len(args) > 0:
            new_args = list(args)
            for i, a in enumerate(new_args):
                if torch.is_tensor(a) and a.ndim == 4 and a.shape[1] == 3:
                    if self.mode == 'zero':
                        new_args[i] = torch.zeros_like(a)
                    elif self.mode == 'random':
                        new_args[i] = torch.randn_like(a) * 0.5
            args = tuple(new_args)
        return args, kwargs


def build_model_and_data(args):
    ckpt_data = torch.load(args.ckpt_path, map_location='cpu')
    ckpt_args = ckpt_data.get('args', {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)

    rotation_only = ckpt_args.get('rotation_only', True)
    use_mlp_head = ckpt_args.get('use_mlp_head', 0)
    bev_encoder = ckpt_args.get('bev_encoder', 1)

    model = BEVCalib(
        rotation_only=rotation_only,
        use_mlp_head=use_mlp_head,
        drop_path_rate=0.0,
        head_dropout=0.0,
        num_heads=8,
        bev_encoder=bev_encoder > 0 if isinstance(bev_encoder, int) else bev_encoder,
        img_shape=(360, 640),
        voxel_mode=args.voxel_mode,
        scatter_reduce=args.scatter_reduce,
        to_bev_mode="concat",
    )

    state_dict = ckpt_data.get('model_state_dict', ckpt_data.get('state_dict', ckpt_data))
    state_dict = _auto_permute_spconv_weights(state_dict, model)
    _adapt_model_to_checkpoint(model, state_dict, 'cuda')
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    fake_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        angle_range_deg=args.angle_range_deg,
        trans_range=0.15,
        use_full_dataset=True,
        max_batches=0,
        eval_sample_step=None,
        eval_max_frames_per_seq=args.max_frames,
    )
    ds = _build_eval_custom_dataset(args.dataset_root, fake_args)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=make_collate_fn((640, 360)),
    )
    return model, loader, rotation_only


def run_eval(model, loader, rotation_only, max_samples=200):
    errors = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_samples:
                break
            items = [b.cuda() if torch.is_tensor(b) else b for b in batch]
            imgs, pcs, gt_T, init_T, post_T, intrinsics = items[0], items[1], items[2], items[3], items[4], items[5]

            gt_T_dummy = init_T.clone()
            result = model(imgs, pcs, gt_T_dummy, init_T, post_T, intrinsics,
                           masks=None, out_init_loss=False)
            pred_T = result[0] if isinstance(result, tuple) else result

            for b in range(pred_T.shape[0]):
                errs = compute_pose_errors(
                    pred_T[b].cpu().numpy(),
                    gt_T[b].cpu().numpy(),
                    rotation_only=rotation_only,
                )
                errors.append(errs)

    rot_errors = [e['total_rotation_error'] for e in errors]
    roll_errors = [e['roll_error'] for e in errors]
    pitch_errors = [e['pitch_error'] for e in errors]
    yaw_errors = [e['yaw_error'] for e in errors]

    return {
        'n': len(rot_errors),
        'rot_mean': np.mean(rot_errors),
        'rot_std': np.std(rot_errors),
        'roll_mean': np.mean(roll_errors),
        'pitch_mean': np.mean(pitch_errors),
        'yaw_mean': np.mean(yaw_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--angle_range_deg', type=float, default=5.0)
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--voxel_mode', default='scatter')
    parser.add_argument('--scatter_reduce', default='mean')
    args = parser.parse_args()

    print(f"Loading model: {os.path.basename(args.ckpt_path)}")
    model, loader, rotation_only = build_model_and_data(args)
    print(f"Dataset: {len(loader.dataset)} samples, evaluating first {args.max_frames}")

    modes = ['normal', 'zero', 'random']
    results = {}

    hook = _ImgBypassHook('normal')

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()} images")
        print(f"{'='*60}")

        hook.mode = mode

        if mode != 'normal':
            original_forward = model.forward

            def make_bypass_forward(bypass_mode):
                def bypass_forward(img, pc, gt_T, init_T, post_T, cam_intrinsic, **kwargs):
                    if bypass_mode == 'zero':
                        img = torch.zeros_like(img)
                    elif bypass_mode == 'random':
                        img = torch.randn_like(img) * 0.5
                    return original_forward(img, pc, gt_T, init_T, post_T, cam_intrinsic, **kwargs)
                return bypass_forward

            model.forward = make_bypass_forward(mode)

        t0 = time.time()
        res = run_eval(model, loader, rotation_only, max_samples=args.max_frames)
        elapsed = time.time() - t0
        results[mode] = res

        if mode != 'normal':
            model.forward = original_forward

        print(f"  Samples: {res['n']}")
        print(f"  Total Rot: {res['rot_mean']:.4f}° ± {res['rot_std']:.4f}°")
        print(f"  Roll:  {res['roll_mean']:.4f}°  Pitch: {res['pitch_mean']:.4f}°  Yaw: {res['yaw_mean']:.4f}°")
        print(f"  Time: {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("CAMERA BYPASS VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Mode':<10s} {'Rot Mean':>10s} {'Roll':>8s} {'Pitch':>8s} {'Yaw':>8s} {'vs Normal':>12s}")
    print(f"{'-'*58}")

    normal_rot = results['normal']['rot_mean']
    for mode in modes:
        r = results[mode]
        delta = (r['rot_mean'] - normal_rot) / normal_rot * 100 if normal_rot > 0 else 0
        sign = '+' if delta >= 0 else ''
        print(f"{mode:<10s} {r['rot_mean']:>9.4f}° {r['roll_mean']:>7.4f}° "
              f"{r['pitch_mean']:>7.4f}° {r['yaw_mean']:>7.4f}° {sign}{delta:>10.1f}%")

    zero_delta = abs(results['zero']['rot_mean'] - normal_rot) / normal_rot * 100
    rand_delta = abs(results['random']['rot_mean'] - normal_rot) / normal_rot * 100

    print(f"\nCamera impact (zero): {zero_delta:.1f}%")
    print(f"Camera impact (random): {rand_delta:.1f}%")
    print()

    if zero_delta < 10 and rand_delta < 15:
        print("=" * 60)
        print("CONCLUSION: DST-Calib 假说 **已验证**")
        print("  模型几乎完全忽略相机图像 (<10% 影响)")
        print("  这解释了为什么增加 dropout/augmentation 无法突破 0.65° 地板")
        print("  根因: 模型仅从 LiDAR 投影模式学习，未建立跨模态对应关系")
        print("=" * 60)
    elif zero_delta < 30:
        print("=" * 60)
        print("CONCLUSION: 相机贡献较弱 (10-30%)")
        print("  模型部分使用相机，但 LiDAR 仍然是主要信息源")
        print("=" * 60)
    else:
        print("=" * 60)
        print("CONCLUSION: 相机贡献显著 (>30%)")
        print("  DST-Calib 假说在本模型上 **未确认**")
        print("  泛化问题可能有其他原因")
        print("=" * 60)


if __name__ == '__main__':
    main()
