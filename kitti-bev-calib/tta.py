"""Test-Time Adaptation (TTA) for BEVCalib.

Adapts a pre-trained BEVCalib model to a new vehicle/installation at inference
time using self-supervised consistency losses. No ground-truth extrinsics needed.

Algorithm:
  Phase 1 — Initial inference on all test frames → raw predictions
  Phase 2 — Robust consensus: SVD-mean across frames → reference extrinsic R̄
  Phase 3 — Consistency fine-tuning: freeze backbone, update only last
            transformer layer + head using L_consistency = geodesic(R̂ᵢ, R̄)
  Phase 4 — Re-inference with adapted model → refined predictions

Usage:
  from tta import test_time_adapt
  adapted_preds = test_time_adapt(model, frames_data, device, tta_config)
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F


def _rotation_matrix_to_quaternion(R):
    """Batch rotation matrix (B,3,3) → quaternion (B,4) [w,x,y,z]."""
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    s = torch.sqrt(torch.clamp(tr + 1.0, min=1e-10)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s

    mask = tr <= 0
    if mask.any():
        diag = torch.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], dim=-1)
        max_idx = diag.argmax(dim=-1)
        for b in range(B):
            if not mask[b]:
                continue
            i = max_idx[b].item()
            j = (i + 1) % 3
            k = (i + 2) % 3
            s_val = torch.sqrt(torch.clamp(
                1.0 + R[b, i, i] - R[b, j, j] - R[b, k, k], min=1e-10)) * 2
            q[b, 0] = (R[b, k, j] - R[b, j, k]) / s_val
            q[b, i + 1] = 0.25 * s_val
            q[b, j + 1] = (R[b, j, i] + R[b, i, j]) / s_val
            q[b, k + 1] = (R[b, k, i] + R[b, i, k]) / s_val
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _geodesic_loss(R_pred, R_target):
    """Geodesic distance between rotation matrices: arccos((tr(R₁ᵀR₂) - 1) / 2)."""
    R_rel = R_pred @ R_target.transpose(-2, -1)
    tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = (tr - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).mean()


def _svd_mean_rotation(rotations):
    """Compute SVD-based mean rotation from (N, 3, 3) rotation matrices."""
    M = rotations.mean(dim=0)
    U, _, Vt = torch.linalg.svd(M)
    R_mean = U @ Vt
    if torch.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_mean


def _robust_median_rotation(rotations):
    """Find the rotation closest to all others (geometric median approximation)."""
    N = rotations.shape[0]
    dists = torch.zeros(N, device=rotations.device)
    for i in range(N):
        R_rel = rotations[i].unsqueeze(0) @ rotations.transpose(-2, -1)
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        dists[i] = torch.acos(cos_angle).sum()
    return rotations[dists.argmin()]


def test_time_adapt(model, seq_data, device, config=None):
    """Adapt model to a test sequence using self-supervised consistency.

    Args:
        model: Pre-trained BEVCalib model (will be deep-copied, original unchanged).
        seq_data: List of (img_tensor, pc_tensor, init_T, intrinsic) per frame.
                  img_tensor: (3, H, W), pc_tensor: (N, 3/4), init_T: (4, 4),
                  intrinsic: (3, 3).
        device: torch device.
        config: dict with TTA hyperparameters:
            - tta_steps: number of fine-tuning steps (default: 10)
            - tta_lr: learning rate for adaptation (default: 1e-5)
            - tta_batch_size: frames per adaptation step (default: 4)
            - tta_perturb_sigma: perturbation sigma in degrees (default: 3.0)
            - tta_adapt_layers: 'head' or 'head+transformer' (default: 'head')

    Returns:
        R_adapted: (3, 3) adapted rotation matrix (consensus extrinsic).
        stats: dict with adaptation statistics.
    """
    if config is None:
        config = {}
    tta_steps = config.get('tta_steps', 10)
    tta_lr = config.get('tta_lr', 1e-5)
    tta_batch_size = config.get('tta_batch_size', 4)
    tta_perturb_sigma = config.get('tta_perturb_sigma', 3.0)
    adapt_layers = config.get('tta_adapt_layers', 'head')

    adapted_model = copy.deepcopy(model)

    N = len(seq_data)
    if N < 2:
        return None, {'status': 'too_few_frames'}

    # === Phase 1: Initial inference ===
    adapted_model.eval()
    raw_rotations = []
    with torch.no_grad():
        for img_t, pc_t, init_T, intrinsic in seq_data:
            img_b = img_t.unsqueeze(0).to(device)
            pc_b = pc_t.unsqueeze(0).to(device)
            gt_T_b = init_T.unsqueeze(0).to(device)
            init_T_b = init_T.unsqueeze(0).to(device)
            post_T = torch.eye(4, device=device).unsqueeze(0)
            K_b = intrinsic.unsqueeze(0).to(device)

            T_pred, _, _ = adapted_model(
                img_b, pc_b, gt_T_b, init_T_b, post_T, K_b, out_init_loss=False)
            raw_rotations.append(T_pred[0, :3, :3].cpu())

    all_R = torch.stack(raw_rotations, dim=0).to(device)

    # === Phase 2: Robust consensus ===
    R_ref = _svd_mean_rotation(all_R)
    R_median = _robust_median_rotation(all_R)

    dists_mean = []
    dists_med = []
    for i in range(N):
        R_rel = all_R[i] @ R_ref.T
        tr = (R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]).clamp(-1 + 1e-7, 3 - 1e-7)
        dists_mean.append(torch.acos((tr - 1) / 2).item())
        R_rel2 = all_R[i] @ R_median.T
        tr2 = (R_rel2[0, 0] + R_rel2[1, 1] + R_rel2[2, 2]).clamp(-1 + 1e-7, 3 - 1e-7)
        dists_med.append(torch.acos((tr2 - 1) / 2).item())

    initial_spread = np.degrees(np.mean(dists_mean))

    if initial_spread < 0.05:
        return R_ref.cpu(), {
            'status': 'skip_low_variance',
            'initial_spread_deg': initial_spread,
            'tta_steps': 0,
        }

    # === Phase 3: Consistency fine-tuning ===
    adapted_model.train()
    for p in adapted_model.parameters():
        p.requires_grad_(False)

    if adapt_layers == 'head+transformer':
        for p in adapted_model.deformable_transformer[-1].parameters():
            p.requires_grad_(True)
    for p in adapted_model.rotation_pred.parameters():
        p.requires_grad_(True)
    if hasattr(adapted_model, 'head_drop'):
        adapted_model.head_drop.eval()

    trainable = [p for p in adapted_model.parameters() if p.requires_grad]
    if not trainable:
        adapted_model.eval()
        return R_ref.cpu(), {
            'status': 'no_trainable_params',
            'initial_spread_deg': initial_spread,
        }

    optimizer = torch.optim.Adam(trainable, lr=tta_lr)
    R_target = R_ref.detach()
    T_target = torch.eye(4, device=device)
    T_target[:3, :3] = R_target

    losses_history = []
    for step in range(tta_steps):
        indices = torch.randperm(N)[:tta_batch_size].tolist()
        batch_loss = torch.tensor(0.0, device=device)
        count = 0

        for idx in indices:
            img_t, pc_t, init_T, intrinsic = seq_data[idx]
            img_b = img_t.unsqueeze(0).to(device)
            pc_b = pc_t.unsqueeze(0).to(device)
            init_T_b = init_T.unsqueeze(0).to(device)
            gt_T_b = T_target.unsqueeze(0)
            post_T = torch.eye(4, device=device).unsqueeze(0)
            K_b = intrinsic.unsqueeze(0).to(device)

            T_pred, _, loss = adapted_model(
                img_b, pc_b, gt_T_b, init_T_b, post_T, K_b, out_init_loss=False)
            R_pred = T_pred[0, :3, :3]
            consistency_loss = _geodesic_loss(
                R_pred.unsqueeze(0), R_target.unsqueeze(0))
            batch_loss = batch_loss + consistency_loss
            count += 1

        if count > 0:
            avg_loss = batch_loss / count
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses_history.append(avg_loss.item())

    # === Phase 4: Re-inference with adapted model ===
    adapted_model.eval()
    adapted_rotations = []
    with torch.no_grad():
        for img_t, pc_t, init_T, intrinsic in seq_data:
            img_b = img_t.unsqueeze(0).to(device)
            pc_b = pc_t.unsqueeze(0).to(device)
            gt_T_b = T_target.unsqueeze(0)
            init_T_b = init_T.unsqueeze(0).to(device)
            post_T = torch.eye(4, device=device).unsqueeze(0)
            K_b = intrinsic.unsqueeze(0).to(device)

            T_pred, _, _ = adapted_model(
                img_b, pc_b, gt_T_b, init_T_b, post_T, K_b, out_init_loss=False)
            adapted_rotations.append(T_pred[0, :3, :3].cpu())

    all_R_adapted = torch.stack(adapted_rotations, dim=0).to(device)
    R_final = _svd_mean_rotation(all_R_adapted)

    final_dists = []
    for i in range(N):
        R_rel = all_R_adapted[i] @ R_final.T
        tr = (R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]).clamp(-1 + 1e-7, 3 - 1e-7)
        final_dists.append(torch.acos((tr - 1) / 2).item())
    final_spread = np.degrees(np.mean(final_dists))

    del adapted_model
    torch.cuda.empty_cache()

    return R_final.cpu(), {
        'status': 'adapted',
        'initial_spread_deg': initial_spread,
        'final_spread_deg': final_spread,
        'spread_reduction': (initial_spread - final_spread) / initial_spread * 100
        if initial_spread > 1e-6 else 0.0,
        'tta_steps': tta_steps,
        'losses': losses_history,
        'final_loss': losses_history[-1] if losses_history else None,
    }
