"""
分析 cam_bev_feats vs pc_bev_feats 特征图。
Hook conv_fuser 的输入，可视化和对比两个分支的特征活跃度。
"""

import sys, os, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

from evaluate_checkpoint import (
    _auto_permute_spconv_weights,
    _adapt_model_to_checkpoint,
    _build_eval_custom_dataset,
    make_collate_fn,
)
from bev_calib import BEVCalib
from tools import generate_single_perturbation_from_T


class FeatureCapture:
    def __init__(self):
        self.cam_feats = None
        self.pc_feats = None
        self.fused_feats = None

    def fuser_hook(self, module, args, output):
        self.cam_feats = args[0].detach().cpu()
        self.pc_feats = args[1].detach().cpu()
        self.fused_feats = output.detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--output_dir', default='logs/evaluations/feature_analysis')
    parser.add_argument('--angle_range_deg', type=float, default=5.0)
    parser.add_argument('--num_samples', type=int, default=6)
    parser.add_argument('--voxel_mode', default='scatter')
    parser.add_argument('--scatter_reduce', default='mean')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    capture = FeatureCapture()

    def _pre_hook(module, args):
        capture.cam_feats = args[0].detach().cpu()
        capture.pc_feats = args[1].detach().cpu()
        return args

    def _post_hook(module, inp, out):
        capture.fused_feats = out.detach().cpu()

    model.conv_fuser.register_forward_pre_hook(_pre_hook)
    handle = model.conv_fuser.register_forward_hook(_post_hook)

    fake_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        angle_range_deg=args.angle_range_deg,
        trans_range=0.15,
        use_full_dataset=True,
        max_batches=0,
        eval_sample_step=None,
        eval_max_frames_per_seq=50,
    )
    ds = _build_eval_custom_dataset(args.dataset_root, fake_args)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=make_collate_fn((640, 360)),
    )

    all_cam_stats = []
    all_pc_stats = []
    device = 'cuda'

    sample_indices = np.linspace(0, len(ds)-1, args.num_samples, dtype=int)

    print(f"Analyzing {args.num_samples} samples from {len(ds)} total")

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T_list, intrinsics) in enumerate(loader):
            if batch_idx >= max(sample_indices) + 1:
                break

            gt_T_np = np.array(gt_T_list).astype(np.float32)
            init_T_np, _, _ = generate_single_perturbation_from_T(
                gt_T_np, angle_range_deg=args.angle_range_deg,
                trans_range=0.15, rotation_only=rotation_only,
            )

            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            pcs_np = np.array(pcs)[:, :, :3]
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
            init_T_t = torch.from_numpy(init_T_np).float().to(device)
            post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
            K_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

            _ = model(resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t, masks=masks, out_init_loss=False)

            cam_f = capture.cam_feats[0]
            pc_f = capture.pc_feats[0]

            cam_mag = cam_f.abs().mean(dim=0).numpy()
            pc_mag = pc_f.abs().mean(dim=0).numpy()
            cam_std = cam_f.std(dim=0).numpy()
            pc_std = pc_f.std(dim=0).numpy()

            cam_stat = {
                'mean_activation': cam_f.abs().mean().item(),
                'max_activation': cam_f.abs().max().item(),
                'std_across_channels': cam_f.std(dim=0).mean().item(),
                'sparsity': (cam_f.abs() < 0.01).float().mean().item(),
                'l2_norm': cam_f.norm().item(),
                'num_channels': cam_f.shape[0],
            }
            pc_stat = {
                'mean_activation': pc_f.abs().mean().item(),
                'max_activation': pc_f.abs().max().item(),
                'std_across_channels': pc_f.std(dim=0).mean().item(),
                'sparsity': (pc_f.abs() < 0.01).float().mean().item(),
                'l2_norm': pc_f.norm().item(),
                'num_channels': pc_f.shape[0],
            }
            all_cam_stats.append(cam_stat)
            all_pc_stats.append(pc_stat)

            if batch_idx in sample_indices:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f'Sample {batch_idx}: Feature Map Analysis', fontsize=14)

                axes[0, 0].imshow(cam_mag, cmap='hot', aspect='auto')
                axes[0, 0].set_title(f'Camera BEV (mean |activation|)\n'
                                     f'mean={cam_stat["mean_activation"]:.4f}, '
                                     f'L2={cam_stat["l2_norm"]:.1f}')
                axes[0, 0].set_ylabel('Camera Branch')

                axes[0, 1].imshow(cam_std, cmap='viridis', aspect='auto')
                axes[0, 1].set_title(f'Camera BEV (channel std)\n'
                                     f'sparsity={cam_stat["sparsity"]:.1%}')

                cam_hist = cam_f.flatten().numpy()
                axes[0, 2].hist(cam_hist, bins=100, alpha=0.7, color='red', density=True)
                axes[0, 2].set_title(f'Camera activation distribution\n{cam_f.shape[0]}ch')
                axes[0, 2].set_xlim(-2, 2)

                axes[1, 0].imshow(pc_mag, cmap='hot', aspect='auto')
                axes[1, 0].set_title(f'LiDAR BEV (mean |activation|)\n'
                                     f'mean={pc_stat["mean_activation"]:.4f}, '
                                     f'L2={pc_stat["l2_norm"]:.1f}')
                axes[1, 0].set_ylabel('LiDAR Branch')

                axes[1, 1].imshow(pc_std, cmap='viridis', aspect='auto')
                axes[1, 1].set_title(f'LiDAR BEV (channel std)\n'
                                     f'sparsity={pc_stat["sparsity"]:.1%}')

                pc_hist = pc_f.flatten().numpy()
                axes[1, 2].hist(pc_hist, bins=100, alpha=0.7, color='blue', density=True)
                axes[1, 2].set_title(f'LiDAR activation distribution\n{pc_f.shape[0]}ch')
                axes[1, 2].set_xlim(-2, 2)

                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'feature_map_sample_{batch_idx:04d}.png'), dpi=150)
                plt.close()
                print(f"  Sample {batch_idx}: cam_act={cam_stat['mean_activation']:.4f} "
                      f"pc_act={pc_stat['mean_activation']:.4f} "
                      f"ratio={pc_stat['mean_activation']/max(cam_stat['mean_activation'],1e-8):.2f}x")

    print(f"\n{'='*60}")
    print("FEATURE ACTIVITY SUMMARY (averaged over all batches)")
    print(f"{'='*60}")

    n = len(all_cam_stats)
    metrics = ['mean_activation', 'max_activation', 'std_across_channels', 'sparsity', 'l2_norm']
    print(f"\n{'Metric':<25s} {'Camera':>12s} {'LiDAR':>12s} {'Ratio(L/C)':>12s}")
    print(f"{'-'*61}")

    for m in metrics:
        cam_v = np.mean([s[m] for s in all_cam_stats])
        pc_v = np.mean([s[m] for s in all_pc_stats])
        ratio = pc_v / max(cam_v, 1e-8)
        print(f"{m:<25s} {cam_v:>12.4f} {pc_v:>12.4f} {ratio:>11.2f}x")

    cam_act = np.mean([s['mean_activation'] for s in all_cam_stats])
    pc_act = np.mean([s['mean_activation'] for s in all_pc_stats])
    cam_sp = np.mean([s['sparsity'] for s in all_cam_stats])
    pc_sp = np.mean([s['sparsity'] for s in all_pc_stats])

    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")

    if pc_act / max(cam_act, 1e-8) > 5:
        print(f"  LiDAR activation 比 Camera 高 {pc_act/max(cam_act,1e-8):.0f}x")
        print(f"  >>> Camera 分支严重不活跃，features 接近死亡")
    elif pc_act / max(cam_act, 1e-8) > 2:
        print(f"  LiDAR activation 比 Camera 高 {pc_act/max(cam_act,1e-8):.1f}x")
        print(f"  >>> Camera 分支活跃度明显低于 LiDAR")
    else:
        print(f"  两个分支活跃度相近 (ratio={pc_act/max(cam_act,1e-8):.2f}x)")

    if cam_sp > 0.8:
        print(f"  Camera sparsity={cam_sp:.1%} — 超过 80% 的 camera 特征近零")
        print(f"  >>> Camera 分支实质性退化为空操作")
    elif cam_sp > 0.5:
        print(f"  Camera sparsity={cam_sp:.1%} — 超过半数特征近零")

    print(f"\n  Camera channels: {all_cam_stats[0]['num_channels']}")
    print(f"  LiDAR channels:  {all_pc_stats[0]['num_channels']}")
    print(f"  Feature maps saved to: {args.output_dir}")

    handle.remove()


if __name__ == '__main__':
    main()
