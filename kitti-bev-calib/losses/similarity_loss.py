"""V60: Similarity Loss for cross-attention supervision.

Implements L_sim (paper Eq. 9): cross-entropy between predicted similarity
matrix from SimCrossAttention and GT point-pixel correspondence matrix.

Also provides GT correspondence matrix construction using known extrinsics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.corr_alignment_loss import compute_projection_v42


def build_gt_correspondence(
    xyz_groups: torch.Tensor,
    T_gt: torch.Tensor,
    cam_intrinsic: torch.Tensor,
    feat_h: int,
    feat_w: int,
    patch_size: float,
    use_registry_token: bool = True,
    soft_radius: float = 0.0,
) -> tuple:
    """Build ground-truth point-pixel correspondence matrix.

    Uses GT extrinsic to project 3D points onto the image and find
    which feature-map token each point corresponds to.

    Args:
        xyz_groups: (B, G, 3) 3D point group centers in LiDAR frame.
        T_gt: (B, 4, 4) ground-truth extrinsic (LiDAR → camera).
        cam_intrinsic: (B, 3, 3) camera intrinsic matrix.
        feat_h, feat_w: Feature map spatial dimensions.
        patch_size: Stride from original image to feature map.
        use_registry_token: If True, adds registry column for OOV points.
        soft_radius: If > 0, use soft Gaussian target instead of one-hot.

    Returns:
        gt_corr: (B, G, N_tokens) GT correspondence matrix.
            N_tokens = feat_h * feat_w [+ 1 if registry_token].
        fov_mask: (B, G) boolean mask indicating points within camera FOV.
    """
    B, G, _ = xyz_groups.shape
    device = xyz_groups.device

    uv_gt = compute_projection_v42(xyz_groups, T_gt, cam_intrinsic)
    uv_feat = uv_gt / patch_size

    in_fov = (
        (uv_feat[..., 0] >= 0) & (uv_feat[..., 0] < feat_w)
        & (uv_feat[..., 1] >= 0) & (uv_feat[..., 1] < feat_h)
    )

    u_idx = uv_feat[..., 0].long().clamp(0, feat_w - 1)
    v_idx = uv_feat[..., 1].long().clamp(0, feat_h - 1)
    token_idx = v_idx * feat_w + u_idx

    n_img_tokens = feat_h * feat_w
    n_tokens = n_img_tokens + (1 if use_registry_token else 0)

    oov_mask = ~in_fov  # (B, G) True for out-of-FOV points

    if soft_radius > 0:
        grid_u = torch.arange(feat_w, device=device).float()
        grid_v = torch.arange(feat_h, device=device).float()
        gv, gu = torch.meshgrid(grid_v, grid_u, indexing='ij')
        grid_coords = torch.stack([gu.flatten(), gv.flatten()], dim=-1)

        uv_feat_clamped = uv_feat.clone()
        uv_feat_clamped[..., 0] = uv_feat_clamped[..., 0].clamp(0, feat_w - 1)
        uv_feat_clamped[..., 1] = uv_feat_clamped[..., 1].clamp(0, feat_h - 1)

        diff = grid_coords.unsqueeze(0).unsqueeze(0) - uv_feat_clamped.unsqueeze(2)
        dist_sq = (diff ** 2).sum(-1)
        gt_corr_img = torch.exp(-dist_sq / (2 * soft_radius ** 2))
        gt_corr_img = gt_corr_img / gt_corr_img.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if use_registry_token:
            reg_col = torch.zeros(B, G, 1, device=device)
            gt_corr = torch.cat([gt_corr_img, reg_col], dim=-1)
        else:
            gt_corr = gt_corr_img

        # Zero out OOV rows, then set registry token to 1 for OOV points
        gt_corr = gt_corr.masked_fill(oov_mask.unsqueeze(-1), 0.0)
        if use_registry_token and oov_mask.any():
            # Set last column (registry) to 1.0 for OOV points
            reg_fill = torch.zeros(B, G, device=device)
            reg_fill[oov_mask] = 1.0
            gt_corr[:, :, -1] = gt_corr[:, :, -1] + reg_fill
    else:
        gt_corr = torch.zeros(B, G, n_tokens, device=device)
        token_idx_safe = token_idx.clamp(0, n_img_tokens - 1)
        gt_corr.scatter_(2, token_idx_safe.unsqueeze(-1), 1.0)

        # Zero out OOV rows, then set registry token to 1 for OOV points
        gt_corr = gt_corr.masked_fill(oov_mask.unsqueeze(-1), 0.0)
        if use_registry_token and oov_mask.any():
            reg_fill = torch.zeros(B, G, device=device)
            reg_fill[oov_mask] = 1.0
            gt_corr[:, :, -1] = gt_corr[:, :, -1] + reg_fill
        elif oov_mask.any():
            gt_corr[oov_mask, 0] = 1.0

    return gt_corr, in_fov


def similarity_loss(
    sim_matrices: list,
    gt_corr: torch.Tensor,
    fov_mask: torch.Tensor = None,
    label_smoothing: float = 0.01,
) -> torch.Tensor:
    """Compute L_sim: cross-entropy between attention similarity and GT correspondence.

    Averaged over all decoder layers (paper Eq. 9).

    Args:
        sim_matrices: List of (B, G, N_kv) predicted similarity per layer.
        gt_corr: (B, G, N_kv) GT correspondence matrix.
        fov_mask: (B, G) optional mask (True = in FOV, contributes more).
        label_smoothing: Smoothing factor for numerical stability.

    Returns:
        Scalar loss value.
    """
    if not sim_matrices:
        return torch.tensor(0.0, device=gt_corr.device, requires_grad=True)

    B, G, N_kv = gt_corr.shape
    total_loss = torch.tensor(0.0, device=gt_corr.device)

    gt_flat = gt_corr.reshape(B * G, N_kv)

    if label_smoothing > 0:
        gt_flat = gt_flat * (1.0 - label_smoothing) + label_smoothing / N_kv

    for sim_matrix in sim_matrices:
        pred = sim_matrix.reshape(B * G, N_kv).clamp(min=1e-8)
        log_pred = pred.log()
        layer_loss = -(gt_flat * log_pred).sum(dim=-1)

        if fov_mask is not None:
            weights = fov_mask.float().reshape(B * G)
            weights = weights * 0.7 + 0.3
            layer_loss = (layer_loss * weights).sum() / weights.sum().clamp(min=1.0)
        else:
            layer_loss = layer_loss.mean()

        total_loss = total_loss + layer_loss

    return total_loss / len(sim_matrices)


def fov_classification_loss(
    fov_logits: torch.Tensor,
    gt_fov_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy for FOV classification auxiliary task (paper Eq. 10).

    Args:
        fov_logits: (B, G) or (B, G, 1) predicted FOV membership logits.
        gt_fov_mask: (B, G) boolean GT mask.

    Returns:
        Scalar BCE loss.
    """
    if fov_logits.dim() == 3:
        fov_logits = fov_logits.squeeze(-1)
    return F.binary_cross_entropy_with_logits(
        fov_logits, gt_fov_mask.float())
