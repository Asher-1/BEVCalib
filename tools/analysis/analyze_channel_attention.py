"""
B: Camera vs LiDAR channel importance (gradient-based)
C: Transformer attention pattern analysis

在 conv_fuser 层分析两个分支的梯度重要性，
在 Transformer attention 层分析空间注意力模式。
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


def load_model_and_data(args):
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
        intrinsic_input=ckpt_args.get('intrinsic_input', False),
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
        eval_max_frames_per_seq=30,
    )
    ds = _build_eval_custom_dataset(args.dataset_root, fake_args)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=make_collate_fn((640, 360)),
    )
    return model, loader, rotation_only


def prepare_batch(batch, rotation_only, device='cuda'):
    imgs, pcs, masks, gt_T_list, intrinsics = batch

    gt_T_np = np.array(gt_T_list).astype(np.float32)
    init_T_np, _, _ = generate_single_perturbation_from_T(
        gt_T_np, angle_range_deg=5.0, trans_range=0.15, rotation_only=rotation_only,
    )
    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
    pcs_np = np.array(pcs)[:, :, :3]
    pcs_t = torch.from_numpy(pcs_np).float().to(device)
    gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
    init_T_t = torch.from_numpy(init_T_np).float().to(device)
    post_T = torch.eye(4).unsqueeze(0).repeat(gt_T_t.shape[0], 1, 1).float().to(device)
    K_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

    return resize_imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t, masks


def analyze_channel_gradient_importance(model, loader, rotation_only, num_samples, output_dir):
    """B: Gradient-based channel importance for Camera vs LiDAR."""
    print("\n" + "="*60)
    print("B: CHANNEL GRADIENT IMPORTANCE ANALYSIS")
    print("="*60)

    cam_ch_grads = []
    pc_ch_grads = []
    cam_feats_holder = [None]
    pc_feats_holder = [None]

    def fuser_pre_hook(module, args):
        cam_feats_holder[0] = args[0]
        pc_feats_holder[0] = args[1]
        return args

    h = model.conv_fuser.register_forward_pre_hook(fuser_pre_hook)

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t, masks = prepare_batch(
            batch, rotation_only)

        model.zero_grad()
        imgs.requires_grad_(True)
        pred_T, init_loss, loss_dict = model(imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t,
                                             masks=masks, out_init_loss=False)

        total_loss = loss_dict["total_loss"] if isinstance(loss_dict, dict) else loss_dict
        total_loss.backward()

        cam_f = cam_feats_holder[0]
        pc_f = pc_feats_holder[0]

        if cam_f.grad is not None:
            cam_grad = cam_f.grad.abs().mean(dim=(0, 2, 3))
            cam_ch_grads.append(cam_grad.detach().cpu().numpy())
        if pc_f.grad is not None:
            pc_grad = pc_f.grad.abs().mean(dim=(0, 2, 3))
            pc_ch_grads.append(pc_grad.detach().cpu().numpy())

        if i % 10 == 0:
            print(f"  Sample {i}/{num_samples}")

    h.remove()

    if not cam_ch_grads and not pc_ch_grads:
        print("  WARNING: No gradients captured on branch features.")
        print("  Using alternative method: 1x1 conv weight analysis...")
        analyze_fuser_weights(model, output_dir)
        return

    cam_importance = np.mean(cam_ch_grads, axis=0) if cam_ch_grads else np.zeros(128)
    pc_importance = np.mean(pc_ch_grads, axis=0) if pc_ch_grads else np.zeros(128)

    cam_total = cam_importance.sum()
    pc_total = pc_importance.sum()
    print(f"\n  Camera gradient importance: {cam_total:.4f}")
    print(f"  LiDAR gradient importance:  {pc_total:.4f}")
    print(f"  Ratio (LiDAR/Camera): {pc_total/max(cam_total,1e-8):.2f}x")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(range(len(cam_importance)), cam_importance, alpha=0.7, color='red', label='Camera')
    axes[0].set_title(f'Camera Channel Gradient Importance\nTotal={cam_total:.4f}')
    axes[0].set_xlabel('Channel')
    axes[1].bar(range(len(pc_importance)), pc_importance, alpha=0.7, color='blue', label='LiDAR')
    axes[1].set_title(f'LiDAR Channel Gradient Importance\nTotal={pc_total:.4f}')
    axes[1].set_xlabel('Channel')

    top_k = 20
    cam_top = np.argsort(cam_importance)[-top_k:][::-1]
    pc_top = np.argsort(pc_importance)[-top_k:][::-1]
    x = range(top_k)
    axes[2].barh([f'C-{c}' for c in cam_top], cam_importance[cam_top], alpha=0.7, color='red', label='Camera')
    axes[2].barh([f'L-{c}' for c in pc_top], pc_importance[pc_top], alpha=0.7, color='blue', label='LiDAR')
    axes[2].set_title(f'Top-{top_k} Channels')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_gradient_importance.png'), dpi=150)
    plt.close()


def analyze_fuser_weights(model, output_dir):
    """Fallback: Analyze conv_fuser 1x1 conv weights to see camera vs LiDAR importance."""
    conv = model.conv_fuser[0]
    W = conv.weight.detach().cpu()
    n_cam = model.img_branch.out_channels
    n_pc = model.pc_branch.out_channels

    cam_weight = W[:, :n_cam, :, :].abs().mean(dim=(0, 2, 3)).numpy()
    pc_weight = W[:, n_cam:n_cam+n_pc, :, :].abs().mean(dim=(0, 2, 3)).numpy()

    cam_total = cam_weight.sum()
    pc_total = pc_weight.sum()

    print(f"\n  Conv1x1 Weight Analysis:")
    print(f"  Camera input weight magnitude: {cam_total:.4f} (mean/ch: {cam_weight.mean():.4f})")
    print(f"  LiDAR input weight magnitude:  {pc_total:.4f} (mean/ch: {pc_weight.mean():.4f})")
    print(f"  Ratio (LiDAR/Camera): {pc_total/max(cam_total,1e-8):.2f}x")

    cam_out_importance = W[:, :n_cam, :, :].abs().mean(dim=(1, 2, 3)).numpy()
    pc_out_importance = W[:, n_cam:n_cam+n_pc, :, :].abs().mean(dim=(1, 2, 3)).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Conv Fuser 1x1 Weight Analysis: Camera vs LiDAR Input Importance', fontsize=13)

    axes[0].bar(range(n_cam), cam_weight, alpha=0.7, color='red')
    axes[0].set_title(f'Camera Input Channels\nMean={cam_weight.mean():.4f}')
    axes[0].set_xlabel('Input Channel')
    axes[0].set_ylabel('|Weight| mean over output')

    axes[1].bar(range(n_pc), pc_weight, alpha=0.7, color='blue')
    axes[1].set_title(f'LiDAR Input Channels\nMean={pc_weight.mean():.4f}')
    axes[1].set_xlabel('Input Channel')

    axes[2].bar(range(len(cam_out_importance)), cam_out_importance, alpha=0.6, color='red', label=f'Camera ({n_cam}ch)')
    axes[2].bar(range(len(pc_out_importance)), pc_out_importance, alpha=0.6, color='blue', label=f'LiDAR ({n_pc}ch)')
    axes[2].set_title('Per-Output Channel: Camera vs LiDAR Contribution')
    axes[2].set_xlabel('Output Channel')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fuser_weight_analysis.png'), dpi=150)
    plt.close()


def analyze_attention_patterns(model, loader, rotation_only, num_samples, output_dir):
    """C: Extract and visualize transformer attention patterns."""
    print("\n" + "="*60)
    print("C: TRANSFORMER ATTENTION PATTERN ANALYSIS")
    print("="*60)

    attn_weights_all = []
    bev_masks_all = []

    attn_hooks = []
    for layer_idx, layer in enumerate(model.transformer.layers):
        mha = layer.self_attn

        def make_hook(idx):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_w = output[1]
                    if attn_w is not None:
                        attn_weights_all.append((idx, attn_w.detach().cpu()))
            return hook

        h = mha.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
        attn_hooks.append(h)

    bev_shape_holder = [None]

    def fuser_post_hook(module, inp, out):
        bev_shape_holder[0] = out.shape

    fh = model.conv_fuser.register_forward_hook(fuser_post_hook)

    need_attn = hasattr(model.transformer.layers[0].self_attn, 'need_weights')
    original_forwards = []
    for layer in model.transformer.layers:
        orig = layer.self_attn.forward
        original_forwards.append(orig)

    spatial_attn_maps = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t, masks = prepare_batch(
                batch, rotation_only)
            attn_weights_all.clear()
            _ = model(imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t, masks=masks, out_init_loss=False)

            if bev_shape_holder[0] is not None:
                bev_H = bev_shape_holder[0][2]
                bev_W = bev_shape_holder[0][3]
            else:
                bev_H, bev_W = 100, 100

            if i % 10 == 0:
                print(f"  Sample {i}/{num_samples}, captured {len(attn_weights_all)} attention tensors")

    for h in attn_hooks:
        h.remove()
    fh.remove()

    print(f"\n  BEV shape: {bev_H}x{bev_W}")

    if not attn_weights_all:
        print("  No attention weights captured (PyTorch TransformerEncoder doesn't expose them by default)")
        print("  Falling back to fuser weight analysis + spatial feature importance...")
        analyze_spatial_feature_importance(model, loader, rotation_only, num_samples, output_dir)
        return

    print(f"  Captured {len(attn_weights_all)} attention weight tensors")


def analyze_spatial_feature_importance(model, loader, rotation_only, num_samples, output_dir):
    """Fallback attention analysis: spatial feature importance via cam_bev_mask."""
    print("\n  SPATIAL FEATURE IMPORTANCE (cam_bev_mask analysis)")

    cam_masks_all = []
    cam_feats_all = []
    pc_feats_all = []

    def pre_hook(module, args):
        cam_feats_all.append(args[0].detach().cpu())
        pc_feats_all.append(args[1].detach().cpu())
        return args

    h = model.conv_fuser.register_forward_pre_hook(pre_hook)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= min(num_samples, 20):
                break
            imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t, masks = prepare_batch(
                batch, rotation_only)
            _ = model(imgs, pcs_t, gt_T_t, init_T_t, post_T, K_t, masks=masks, out_init_loss=False)

    h.remove()

    if not cam_feats_all:
        print("  No features captured")
        return

    n = len(cam_feats_all)
    cam_spatial = torch.stack([f[0].abs().mean(dim=0) for f in cam_feats_all]).mean(dim=0).numpy()
    pc_spatial = torch.stack([f[0].abs().mean(dim=0) for f in pc_feats_all]).mean(dim=0).numpy()

    diff_spatial = cam_spatial - pc_spatial
    ratio_spatial = cam_spatial / np.maximum(pc_spatial, 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Spatial Feature Importance (averaged over {n} samples)', fontsize=14)

    im0 = axes[0, 0].imshow(cam_spatial, cmap='hot', aspect='auto')
    axes[0, 0].set_title('Camera BEV\n(mean |activation|)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(pc_spatial, cmap='hot', aspect='auto')
    axes[0, 1].set_title('LiDAR BEV\n(mean |activation|)')
    plt.colorbar(im1, ax=axes[0, 1])

    vmax = max(abs(diff_spatial.min()), abs(diff_spatial.max()))
    im2 = axes[0, 2].imshow(diff_spatial, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[0, 2].set_title('Cam - LiDAR\n(red=Camera dominant)')
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(np.clip(ratio_spatial, 0, 3), cmap='coolwarm', vmin=0, vmax=3, aspect='auto')
    axes[1, 0].set_title('Cam/LiDAR ratio\n(>1 = camera dominant)')
    plt.colorbar(im3, ax=axes[1, 0])

    cam_channel_var = torch.stack([f[0].std(dim=0) for f in cam_feats_all]).mean(dim=0).numpy()
    pc_channel_var = torch.stack([f[0].std(dim=0) for f in pc_feats_all]).mean(dim=0).numpy()

    im4 = axes[1, 1].imshow(cam_channel_var, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Camera channel variance\n(high=diverse features)')
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(pc_channel_var, cmap='viridis', aspect='auto')
    axes[1, 2].set_title('LiDAR channel variance\n(high=diverse features)')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_feature_importance.png'), dpi=150)
    plt.close()

    cam_center = cam_spatial[30:70, 30:70].mean()
    cam_edge = np.concatenate([cam_spatial[:10, :].flatten(), cam_spatial[-10:, :].flatten(),
                                cam_spatial[:, :10].flatten(), cam_spatial[:, -10:].flatten()]).mean()
    pc_center = pc_spatial[30:70, 30:70].mean()
    pc_edge = np.concatenate([pc_spatial[:10, :].flatten(), pc_spatial[-10:, :].flatten(),
                               pc_spatial[:, :10].flatten(), pc_spatial[:, -10:].flatten()]).mean()

    print(f"\n  Spatial Distribution:")
    print(f"  {'Region':<15s} {'Camera':>10s} {'LiDAR':>10s} {'Ratio(C/L)':>12s}")
    print(f"  {'-'*47}")
    print(f"  {'Center':15s} {cam_center:10.4f} {pc_center:10.4f} {cam_center/max(pc_center,1e-8):11.2f}x")
    print(f"  {'Edge':15s} {cam_edge:10.4f} {pc_edge:10.4f} {cam_edge/max(pc_edge,1e-8):11.2f}x")
    print(f"  {'Center/Edge':15s} {cam_center/max(cam_edge,1e-8):10.2f}x {pc_center/max(pc_edge,1e-8):10.2f}x")

    cam_cov = np.sum(cam_spatial > 0.1) / cam_spatial.size
    pc_cov = np.sum(pc_spatial > 0.1) / pc_spatial.size
    print(f"\n  Coverage (|act| > 0.1):")
    print(f"    Camera: {cam_cov:.1%}")
    print(f"    LiDAR:  {pc_cov:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--output_dir', default='logs/evaluations/channel_attention_analysis')
    parser.add_argument('--angle_range_deg', type=float, default=5.0)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--voxel_mode', default='scatter')
    parser.add_argument('--scatter_reduce', default='mean')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model, loader, rotation_only = load_model_and_data(args)

    analyze_fuser_weights(model, args.output_dir)
    analyze_spatial_feature_importance(model, loader, rotation_only, args.num_samples, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
