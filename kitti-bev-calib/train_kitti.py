import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from kitti_dataset import KittiDataset
from custom_dataset import CustomDataset
from bev_calib import BEVCalib
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime, timedelta
from torch.utils.data import random_split, Subset
from collections import defaultdict
import numpy as np
import random
from pathlib import Path
from tools import generate_single_perturbation_from_T, augment_gt_pitch_flip, augment_mount_jitter
import shutil
import cv2
import os
import time
import json
from contextlib import nullcontext


_MGDA_MULTITASK_BACKWARD = None
_MGDA_LOAD_PATH = None


def _get_mgda_multitask_backward():
    """Load MGDA helper from a path relative to this script (avoids utils package shadowing)."""
    global _MGDA_MULTITASK_BACKWARD, _MGDA_LOAD_PATH
    if _MGDA_MULTITASK_BACKWARD is not None:
        return _MGDA_MULTITASK_BACKWARD, _MGDA_LOAD_PATH

    import importlib.util

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / 'utils' / 'mgda.py',
        script_dir / 'utils' / 'mgda.py',
        script_dir / 'mgda.py',
    ]
    last_err = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location('_bevcalib_mgda', path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, 'mgda_multitask_backward', None)
            if fn is None:
                raise AttributeError(f"mgda_multitask_backward missing in {path}")
            _MGDA_MULTITASK_BACKWARD = fn
            _MGDA_LOAD_PATH = str(path)
            return fn, _MGDA_LOAD_PATH
        except Exception as exc:
            last_err = exc

    tried = ", ".join(str(p) for p in candidates)
    raise ModuleNotFoundError(
        f"Cannot load MGDA module (tried: {tried}): {last_err}"
    )


def _get_mgda_weight_params(raw_model):
    """Small bottleneck params for MGDA weight estimation (not full backbone)."""
    params = []
    seen = set()
    corr_head = getattr(raw_model, 'corr_head', None)
    if corr_head is not None:
        for p in corr_head.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    pqi = getattr(raw_model, 'pose_query_init', None)
    if pqi is not None:
        for p in pqi.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


def _mgda_build_weighted_loss(task_losses, grad_accum_steps, weight_params):
    """MGDA weights on bottleneck params → single scalar loss for one backward()."""
    import importlib.util

    script_dir = Path(__file__).resolve().parent
    mgda_path = script_dir.parent / 'utils' / 'mgda.py'
    spec = importlib.util.spec_from_file_location('_bevcalib_mgda', mgda_path)
    mgda_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mgda_mod)

    div = float(grad_accum_steps) if grad_accum_steps > 1 else 1.0
    scaled = {k: v / div for k, v in task_losses.items()}
    weighted, meta = mgda_mod.mgda_weighted_loss(scaled, weight_params)
    return weighted, meta


def _accumulate_fusion_gate_stats(raw_model, accum):
    """Accumulate gated-fusion weights for collapse monitoring."""
    meta = getattr(raw_model, '_last_fusion_meta', None)
    if not meta or 'gate_bev' not in meta:
        return
    gb = meta['gate_bev'].detach().float()
    gp = meta['gate_proj'].detach().float()
    accum['gate_bev_sum'] += gb.sum().item()
    accum['gate_proj_sum'] += gp.sum().item()
    n = gb.numel()
    accum['gate_count'] += n
    w = torch.stack([gb, gp], dim=-1).clamp(min=1e-8)
    ent = -(w * w.log()).sum(dim=-1) / np.log(2)
    accum['gate_entropy_sum'] += ent.sum().item()


def _loss_scalar(loss, key, default=None):
    if key not in loss:
        return default
    v = loss[key]
    return float(v.item()) if hasattr(v, 'item') else float(v)


def _format_gmp_step_suffix(loss, correspondence_loss_weight=0.0):
    """GMP step log: match/EPnP + geo metrics (when present in loss dict)."""
    chunks = []
    match_parts = []
    corr_px = _loss_scalar(loss, 'correspondence_loss')
    if corr_px is not None:
        corr_w = float(correspondence_loss_weight or 0.0)
        if corr_w > 0:
            match_parts.append(
                f"corr={corr_px:.1f}px×{corr_w:g}={corr_px * corr_w:.0f}")
        else:
            match_parts.append(f"corr={corr_px:.1f}px")
    v_gt = _loss_scalar(loss, 'match_valid_ratio_gt')
    v_init = _loss_scalar(loss, 'match_valid_ratio_init')
    if v_gt is not None:
        match_parts.append(f"valid_gt={v_gt:.3f}")
    if v_init is not None:
        match_parts.append(f"valid_init={v_init:.3f}")
    elif _loss_scalar(loss, 'match_valid_ratio') is not None:
        match_parts.append(f"valid={_loss_scalar(loss, 'match_valid_ratio'):.3f}")
    if _loss_scalar(loss, 'match_fallback_ratio') is not None:
        match_parts.append(f"fb={_loss_scalar(loss, 'match_fallback_ratio'):.3f}")
    if _loss_scalar(loss, 'epnp_insufficient_ratio') is not None:
        match_parts.append(
            f"epnp_fail={_loss_scalar(loss, 'epnp_insufficient_ratio'):.3f}")
    if _loss_scalar(loss, 'epnp_mean_effective_points') is not None:
        match_parts.append(
            f"epnp_pts={_loss_scalar(loss, 'epnp_mean_effective_points'):.1f}")
    if _loss_scalar(loss, 'epnp_grad_detached') is not None:
        match_parts.append(f"epnp_grad={int(_loss_scalar(loss, 'epnp_grad_detached'))}")
    if _loss_scalar(loss, 'corr_valid_ratio') is not None:
        match_parts.append(f"corr_win={_loss_scalar(loss, 'corr_valid_ratio'):.3f}")
    if match_parts:
        chunks.append(f"match[{' '.join(match_parts)}]")
    if _loss_scalar(loss, 'appearance_loss') is not None:
        geo = f"app={_loss_scalar(loss, 'appearance_loss'):.4f}"
        if _loss_scalar(loss, 'depth_loss') is not None:
            geo += f" dep={_loss_scalar(loss, 'depth_loss'):.4f}"
        if _loss_scalar(loss, 'geo_valid_ratio') is not None:
            geo += f" gvalid={_loss_scalar(loss, 'geo_valid_ratio'):.3f}"
        chunks.append(f"geo[{geo}]")
    return f" | {' | '.join(chunks)}" if chunks else ""


def _format_step_loss_head(total_loss, loss, batch_errors, rotation_only,
                           correspondence_loss_weight=0.0):
    """Human-readable step loss: separate weighted total from pose / aux terms."""
    tl = float(total_loss.item() if hasattr(total_loss, 'item') else total_loss)
    rot_err = batch_errors['rot_error']
    rot_loss_deg = _loss_scalar(loss, 'rotation_loss')
    parts = [f"total={tl:.2f} (w-sum)"]
    if rot_loss_deg is not None:
        parts.append(f"pose_L={rot_loss_deg:.2f}°")
    parts.append(f"Rot err={rot_err:.2f}°")
    corr_px = _loss_scalar(loss, 'correspondence_loss')
    corr_w = float(correspondence_loss_weight or 0.0)
    if corr_px is not None and corr_w > 0:
        parts.append(f"corr×{corr_w:g}={corr_px * corr_w:.0f}")
    elif corr_px is not None:
        parts.append(f"corr={corr_px:.1f}px")
    cons = _loss_scalar(loss, 'v32_consistency_loss')
    if cons is not None:
        parts.append(f"cons={cons:.4f}")
    jac_w = _loss_scalar(loss, 'jacobian_weighted')
    if jac_w is not None:
        parts.append(f"jac={jac_w:.4f}")
    mag_p = _loss_scalar(loss, 'magnitude_pred_deg')
    mag_g = _loss_scalar(loss, 'magnitude_gt_deg')
    if mag_p is not None and mag_g is not None:
        parts.append(f"mag={mag_p:.2f}/{mag_g:.2f}°")
    overcorr = _loss_scalar(loss, 'overcorr_ratio')
    if overcorr is not None:
        parts.append(f"oc={overcorr:.2f}")
    head = "Loss: " + ", ".join(parts)
    if not rotation_only:
        head += (f", Trans err={batch_errors['trans_error']:.4f}m "
                 f"(Fwd:{batch_errors['fwd_error']:.4f} Lat:{batch_errors['lat_error']:.4f} "
                 f"Ht:{batch_errors['ht_error']:.4f})")
    return head


def _log_gmp_step_scalars(writer, loss, global_step):
    if writer is None:
        return
    for key, tag in (
        ('correspondence_loss', 'GMP/train/correspondence_loss_px'),
        ('match_valid_ratio', 'GMP/train/match_valid_ratio'),
        ('match_valid_ratio_init', 'GMP/train/match_valid_ratio_init'),
        ('match_valid_ratio_gt', 'GMP/train/match_valid_ratio_gt'),
        ('match_fallback_ratio', 'GMP/train/match_fallback_ratio'),
        ('epnp_insufficient_ratio', 'GMP/train/epnp_insufficient_ratio'),
        ('epnp_mean_effective_points', 'GMP/train/epnp_mean_effective_points'),
        ('epnp_grad_detached', 'GMP/train/epnp_grad_detached'),
        ('corr_valid_ratio', 'GMP/train/corr_valid_ratio'),
        ('appearance_loss', 'GMP/train/appearance_loss'),
        ('depth_loss', 'GMP/train/depth_loss'),
        ('geo_valid_ratio', 'GMP/train/geo_valid_ratio'),
        ('magnitude_loss', 'V46/train/magnitude_loss'),
        ('magnitude_pred_deg', 'V46/train/magnitude_pred_deg'),
        ('magnitude_gt_deg', 'V46/train/magnitude_gt_deg'),
        ('overcorr_ratio', 'V46/train/overcorr_ratio'),
        ('overcorr_scale', 'V46/train/overcorr_scale'),
    ):
        v = _loss_scalar(loss, key)
        if v is not None:
            writer.add_scalar(tag, v, global_step)


def set_seed(seed, rank=0):
    """Fix all random seeds for reproducible training.
    
    Each DDP rank gets a unique but deterministic seed (base_seed + rank)
    so that data augmentation differs across GPUs while remaining reproducible.
    cuDNN benchmark stays on for speed — its ~1e-6 level non-determinism
    is negligible compared to the config-level differences we care about.
    """
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stratified_split_by_sequence(dataset, train_ratio=0.8, seed=114514):
    """Split dataset into train/val ensuring each sequence contributes proportionally.
    
    Unlike random_split which can leave some sequences under-represented in training,
    this splits within each sequence independently, guaranteeing every sequence has
    ~train_ratio of its samples in training and ~(1-train_ratio) in validation.
    
    Returns (train_subset, val_subset, split_stats) where split_stats is a dict
    mapping seq_id -> (train_count, val_count, total).
    """
    rng = np.random.RandomState(seed)
    
    seq_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        entry = dataset.all_files[idx]
        seq_id = entry.split('/')[0]
        seq_to_indices[seq_id].append(idx)
    
    train_indices = []
    val_indices = []
    split_stats = {}
    
    for seq_id in sorted(seq_to_indices.keys()):
        indices = np.array(seq_to_indices[seq_id])
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        n_train = max(1, n_train)
        
        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:].tolist())
        split_stats[seq_id] = (n_train, len(indices) - n_train, len(indices))
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), split_stats


def parse_seq_weight_overrides(spec):
    """Parse sequence weighting overrides like '02:1.5,03:2.0'."""
    overrides = {}
    if not spec:
        return overrides
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Invalid seq_weight_overrides item '{item}', expected SEQ:WEIGHT")
        seq_id, value = item.split(':', 1)
        seq_id = seq_id.strip()
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"seq_weight_overrides weight must be > 0, got {weight} for seq {seq_id}")
        overrides[seq_id] = weight
    return overrides


def build_balanced_weights(subset, dataset, mode=1, seq_weight_overrides=None):
    """Build per-sample weights for balanced sampling across sequences.
    
    mode=1 (full): each sequence gets equal total weight 1/N.
      → weight_i = 1 / (N * count_of_seq(i))
    mode=2 (sqrt): softer balance using sqrt(1/count).
      → weight_i = sqrt(1 / count_of_seq(i)) / Z   (Z = normalization constant)
    """
    seq_weight_overrides = seq_weight_overrides or {}
    all_files = dataset.all_files
    seq_counts = defaultdict(int)
    sample_seqs = []
    
    indices = subset.indices if hasattr(subset, 'indices') else range(len(subset))
    for idx in indices:
        real_idx = idx
        if hasattr(subset, 'dataset') and hasattr(subset.dataset, 'indices'):
            real_idx = subset.dataset.indices[idx]
        entry = all_files[real_idx]
        seq_id = entry.split('/')[0]
        seq_counts[seq_id] += 1
        sample_seqs.append(seq_id)
    
    num_seqs = len(seq_counts)
    weights = []
    if mode == 2:
        raw = {
            s: math.sqrt(1.0 / c) * seq_weight_overrides.get(s, 1.0)
            for s, c in seq_counts.items()
        }
        z = sum(raw[s] * seq_counts[s] for s in seq_counts)
        for seq_id in sample_seqs:
            weights.append(raw[seq_id] / z)
    else:
        raw = {s: seq_weight_overrides.get(s, 1.0) / (num_seqs * c) for s, c in seq_counts.items()}
        z = sum(raw[s] * seq_counts[s] for s in seq_counts)
        for seq_id in sample_seqs:
            weights.append(raw[seq_id] / z)
    
    return weights, seq_counts


class DistributedBalancedSampler(torch.utils.data.Sampler):
    """Distributed sampler with per-sequence balanced (weighted) sampling.
    
    Combines DistributedSampler's shard logic with WeightedRandomSampler's
    weighted sampling so each GPU sees a balanced, non-overlapping subset.
    
    Algorithm per epoch:
      1. Build weighted sample pool (all ranks use same seed → same order)
      2. Shard the pool: rank k takes indices k, k+world_size, k+2*world_size, ...
    """

    def __init__(self, weights, num_samples, num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.total_size = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples_per_replica = int(math.ceil(self.total_size / self.num_replicas))
        self.padded_total = self.num_samples_per_replica * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.padded_total, replacement=True, generator=g).tolist()
        assert len(indices) == self.padded_total
        per_rank = indices[self.rank::self.num_replicas]
        return iter(per_rank[:self.num_samples_per_replica])

    def __len__(self):
        return self.num_samples_per_replica


from visualization import (
    compute_batch_pose_errors,
    visualize_batch_projection,
    prepare_image_for_tensorboard,
    compute_pose_errors
)


def _compute_medw_deploy(T_pred_arr, T_gt_arr, seq_arr, unique_seqs, window=200):
    """MEDW{N}: uniform sample N frames/seq, median axis-angle aggregate, compare to GT."""
    from scipy.spatial.transform import Rotation as ScipyRot

    per_seq_errs = {'rot_error': [], 'roll_error': [], 'pitch_error': [], 'yaw_error': []}
    seq_aa_cache = {}
    seq_t_cache = {}
    for sid in unique_seqs:
        mask = seq_arr == sid
        seq_Rs = T_pred_arr[mask, :3, :3]
        seq_ts = T_pred_arr[mask, :3, 3]
        if len(seq_Rs) == 0:
            continue
        seq_aa_cache[sid] = ScipyRot.from_matrix(seq_Rs).as_rotvec()
        seq_t_cache[sid] = seq_ts

    for sid in unique_seqs:
        if sid not in seq_aa_cache:
            continue
        mask = seq_arr == sid
        gt_T = T_gt_arr[mask][0]
        seq_aa_full = seq_aa_cache[sid]
        seq_ts_full = seq_t_cache[sid]
        n = len(seq_aa_full)
        ns = min(window, n)
        indices = np.linspace(0, n - 1, num=ns, dtype=int)
        aa_sub = seq_aa_full[indices]
        ts_sub = seq_ts_full[indices]
        aa_med = np.median(aa_sub, axis=0)
        R_avg = ScipyRot.from_rotvec(aa_med).as_matrix()
        t_avg = np.mean(ts_sub, axis=0)
        T_agg = np.eye(4, dtype=np.float64)
        T_agg[:3, :3] = R_avg
        T_agg[:3, 3] = t_avg
        errs = compute_pose_errors(T_agg, gt_T)
        for k in per_seq_errs:
            per_seq_errs[k].append(errs[k])

    if not per_seq_errs['rot_error']:
        return None
    return {
        'rot': float(np.mean(per_seq_errs['rot_error'])),
        'roll': float(np.mean(per_seq_errs['roll_error'])),
        'pitch': float(np.mean(per_seq_errs['pitch_error'])),
        'yaw': float(np.mean(per_seq_errs['yaw_error'])),
    }


def _euler_perturb_T(T_base_np, delta_rpy_deg):
    """Apply Euler perturbation (degrees) to 4×4 transform rotation part."""
    r, p, y = np.deg2rad(delta_rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    dR = Rz @ Ry @ Rx
    T_out = T_base_np.copy()
    T_out[:3, :3] = dR @ T_base_np[:3, :3]
    return T_out


def _rotation_matrix_to_euler(R):
    """3×3 rotation → Euler degrees [roll, pitch, yaw] (LiDAR convention)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])


def _perturbation_euler_from_T_pair(gt_T, init_T):
    """Applied rotation perturbation (deg) from gt→init: R_init = delta_R @ R_gt."""
    R_gt = gt_T[:3, :3]
    R_init = init_T[:3, :3]
    delta_R = R_init @ R_gt.T
    return _rotation_matrix_to_euler(delta_R)


def _axis_error_and_correction_euler(gt_T, init_T, out_T):
    """Per-axis init error and correction toward GT (degrees).

    init_err = euler(R_init @ R_gt^T)
    out_err  = euler(R_out  @ R_gt^T)
    correction = init_err - out_err   # ideal adaptive: d(correction)/d(bias) ≈ 1
    """
    init_err = _perturbation_euler_from_T_pair(gt_T, init_T)
    out_err = _perturbation_euler_from_T_pair(gt_T, out_T)
    correction = init_err - out_err
    return init_err, out_err, correction


def _euler_from_delta_R_torch(R_delta):
    """(B,3,3) delta rotation → (B,3) roll/pitch/yaw degrees (matches numpy path)."""
    sy = torch.sqrt(R_delta[:, 0, 0] ** 2 + R_delta[:, 1, 0] ** 2 + 1e-8)
    roll = torch.atan2(R_delta[:, 2, 1], R_delta[:, 2, 2])
    pitch = torch.atan2(-R_delta[:, 2, 0], sy)
    yaw = torch.atan2(R_delta[:, 1, 0], R_delta[:, 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / math.pi)


def _axis_correction_torch(R_gt, R_init, R_out):
    """Differentiable per-axis correction = init_err - out_err (degrees)."""
    R_gt_f = R_gt.float()
    R_init_f = R_init.float()
    R_out_f = R_out.float()
    R_err_init = torch.bmm(R_init_f, R_gt_f.transpose(1, 2))
    R_err_out = torch.bmm(R_out_f, R_gt_f.transpose(1, 2))
    return _euler_from_delta_R_torch(R_err_init) - _euler_from_delta_R_torch(R_err_out)


def _zero_drift_axis_error_deg(R_gt, R_pred):
    """Per-axis signed R_pred vs R_gt error in degrees (euler of R_pred @ R_gt^T)."""
    R_err = torch.bmm(R_pred.float(), R_gt.float().transpose(1, 2))
    return _euler_from_delta_R_torch(R_err)


def _apply_fixed_inject_batch(gt_T_np, inject_deg):
    """Fixed-Inject (gdiag): init_R = gt_R @ dR, all RPY = inject_deg."""
    mag = float(inject_deg)
    delta = [mag, mag, mag]
    init_T = gt_T_np.copy()
    for bi in range(init_T.shape[0]):
        init_T[bi] = _euler_perturb_T(gt_T_np[bi], delta)
    return init_T


def _effective_lsp_weight(args, epoch):
    """Ramp LSP weight from start to final over ramp_epochs (V54)."""
    w_max = float(getattr(args, 'lsp_weight', 0.0))
    if w_max <= 0:
        return 0.0
    start = int(getattr(args, 'lsp_start_epoch', 10))
    if epoch < start:
        return 0.0
    w_min = float(getattr(args, 'lsp_weight_start', 0.0))
    ramp = int(getattr(args, 'lsp_ramp_epochs', 0))
    if ramp <= 0:
        return w_max
    ep_in = epoch - start + 1
    alpha = min(1.0, ep_in / float(ramp))
    return w_min + alpha * (w_max - w_min)


def _prs_lr_scales(args, epoch):
    """Pose Release Scheduler LR multipliers (corr_path, pose_head)."""
    rel = int(getattr(args, 'pose_release_epoch', 0))
    if rel <= 0:
        return 1.0, 1.0
    if epoch < rel:
        return 1.0, 0.0
    joint_ep = int(getattr(args, 'pose_release_joint_epoch', 50))
    corr_joint = float(getattr(args, 'pose_release_corr_lr_scale_joint', 0.5))
    if epoch >= joint_ep:
        return corr_joint, 1.0
    return 1.0, 1.0


def _is_v54_pose_head_param(name: str) -> bool:
    prefixes = ('corr_head.', 'dp_head.', 'adir_refiner.', 'pitch_branch.', 'pitch_conf_net.')
    return any(name.startswith(p) for p in prefixes)


def _effective_zero_drift_loss_weight(args, epoch):
    """Linear ramp: weight_start → weight over ramp_epochs after start_epoch."""
    w_max = float(getattr(args, 'zero_drift_loss_weight', 0.0))
    if w_max <= 0:
        return 0.0
    start = int(getattr(args, 'zero_drift_loss_start_epoch', 5))
    if epoch < start:
        return 0.0
    w_min = float(getattr(args, 'zero_drift_loss_weight_start', 0.0))
    ramp = int(getattr(args, 'zero_drift_loss_ramp_epochs', 0))
    if w_min <= 0 or ramp <= 0 or w_min >= w_max:
        return w_max
    t = min(1.0, (epoch - start + 1) / float(ramp))
    return w_min + (w_max - w_min) * t


def _finalize_jacobian_axis_accum(axis_accum):
    """Mean per-axis J values collected from one or more val batches."""
    if not axis_accum or not any(axis_accum.get(k) for k in ('roll', 'pitch', 'yaw')):
        return None
    result = {}
    for key in ('roll', 'pitch', 'yaw'):
        vals = [float(v) for v in axis_accum.get(key, []) if v == v]
        result[key] = float(np.mean(vals)) if vals else float('nan')
    finite = [v for v in result.values() if v == v]
    if not finite:
        return None
    result['overall'] = float(np.mean(finite))
    result['verdict'] = 'ADAPTIVE' if result['overall'] > 0.85 else (
        'WEAK' if result['overall'] < 0.3 else 'MODERATE')
    return result


def _merge_jacobian_results(results):
    """Average Jacobian dicts gathered from DDP ranks."""
    valid = [r for r in results if r]
    if not valid:
        return None
    merged = {}
    for key in ('roll', 'pitch', 'yaw'):
        vals = [float(r[key]) for r in valid if key in r and r[key] == r[key]]
        merged[key] = float(np.mean(vals)) if vals else float('nan')
    finite = [v for v in merged.values() if v == v]
    if not finite:
        return None
    merged['overall'] = float(np.mean(finite))
    merged['verdict'] = 'ADAPTIVE' if merged['overall'] > 0.85 else (
        'WEAK' if merged['overall'] < 0.3 else 'MODERATE')
    return merged


def _medw_max_rpy(medw_result):
    """Max per-axis MEDW error (Roll/Pitch/Yaw)."""
    if medw_result is None:
        return float('inf')
    return max(float(medw_result['roll']), float(medw_result['pitch']),
               float(medw_result['yaw']))


def _medw_axis_pass(medw_result, threshold_deg):
    """True when max(R,P,Y) MEDW < threshold."""
    if medw_result is None:
        return False
    return _medw_max_rpy(medw_result) < float(threshold_deg)


def _jacobian_min_axis(jac_result):
    """Minimum Jacobian across roll/pitch/yaw (ignore NaN)."""
    if jac_result is None:
        return float('-inf')
    vals = [float(jac_result[k]) for k in ('roll', 'pitch', 'yaw')
            if k in jac_result and jac_result[k] == jac_result[k]]
    return min(vals) if vals else float(jac_result.get('overall', float('-inf')))


def _jacobian_pass(jac_result, j_min=0.85):
    """True when overall and all axis Jacobians exceed threshold."""
    if jac_result is None:
        return False
    j_min = float(j_min)
    if jac_result.get('overall') != jac_result.get('overall'):
        return False
    if float(jac_result['overall']) <= j_min:
        return False
    for key in ('roll', 'pitch', 'yaw'):
        v = jac_result.get(key)
        if v != v or float(v) <= j_min:
            return False
    return True


def _dual_gate_pass(medw_result, jac_result, medw_max_deg, jac_min):
    return _medw_axis_pass(medw_result, medw_max_deg) and _jacobian_pass(jac_result, jac_min)


def _format_dual_gate_status(medw_result, jac_result, medw_max_deg, jac_min):
    """Human-readable dual-gate pass/fail breakdown."""
    medw_max = _medw_max_rpy(medw_result)
    medw_ok = _medw_axis_pass(medw_result, medw_max_deg)
    jac_ok = _jacobian_pass(jac_result, jac_min)
    medw_detail = "N/A"
    if medw_result is not None:
        medw_detail = (f"max(R,P,Y)={medw_max:.4f}° "
                       f"(R={medw_result['roll']:.4f} P={medw_result['pitch']:.4f} "
                       f"Y={medw_result['yaw']:.4f}) thr<{medw_max_deg:.2f}° "
                       f"{'PASS' if medw_ok else 'FAIL'}")
    jac_detail = "N/A"
    if jac_result is not None:
        jac_min_ax = _jacobian_min_axis(jac_result)
        jac_detail = (f"Jac overall={jac_result.get('overall', float('nan')):.3f} "
                      f"(R={jac_result.get('roll', float('nan')):.3f} "
                      f"P={jac_result.get('pitch', float('nan')):.3f} "
                      f"Y={jac_result.get('yaw', float('nan')):.3f}) "
                      f"min_axis={jac_min_ax:.3f} thr>{jac_min:.2f} "
                      f"{'PASS' if jac_ok else 'FAIL'}")
    verdict = "PASS" if (medw_ok and jac_ok) else "FAIL"
    return verdict, medw_detail, jac_detail


def _write_convergence_report(log_dir, ckpt_save_dir, args, best_medw, best_dual, kpi_history):
    """Write CONVERGENCE_REPORT.md + convergence_report.json after training."""
    medw_thr = float(args.dual_gate_medw_max)
    jac_thr = float(args.dual_gate_jacobian_min)
    converged = best_dual.get('epoch', -1) > 0

    if converged:
        verdict = "CONVERGED"
        verdict_cn = "收敛达标"
        detail = (f"Epoch {best_dual['epoch']} 通过 dual gate，"
                  f"ckpt: {os.path.join(ckpt_save_dir, 'ckpt_best_dual.pth')}")
    elif kpi_history:
        last = kpi_history[-1]
        _, medw_d, jac_d = _format_dual_gate_status(
            last.get('medw'), last.get('jacobian'), medw_thr, jac_thr)
        verdict = "NOT_CONVERGED"
        verdict_cn = "未收敛"
        detail = f"末次 eval (ep{last['epoch']}): {medw_d}; {jac_d}"
    else:
        verdict = "NO_KPI_EVAL"
        verdict_cn = "无 KPI 评估"
        detail = "未启用 MEDW/Jacobian eval 或无 eval 记录"

    lines = [
        "# BEVCalib 收敛报告 / Convergence Report",
        "",
        f"**Verdict: {verdict} ({verdict_cn})**",
        "",
        "## 验收标准",
        f"- MEDW{args.medw_eval_max_frames}: max(Roll, Pitch, Yaw) < **{medw_thr:.2f}°**",
        f"- Jacobian@±{args.jacobian_eval_angle_deg}°: overall 及 R/P/Y 均 > **{jac_thr:.2f}**",
        "",
        "## 结果摘要",
        detail,
        "",
    ]

    if best_dual.get('epoch', -1) > 0:
        lines.extend([
            "## Best Dual Gate Checkpoint",
            f"- Epoch: {best_dual['epoch']}",
            f"- max(R,P,Y): {best_dual.get('medw_max_rpy', best_dual.get('medw', float('nan'))):.4f}°",
            f"- MEDW R/P/Y: {best_dual.get('medw_roll', float('nan')):.4f} / "
            f"{best_dual.get('medw_pitch', float('nan')):.4f} / "
            f"{best_dual.get('medw_yaw', float('nan')):.4f}°",
            f"- Jacobian overall: {best_dual.get('jacobian', float('nan')):.3f} "
            f"(R={best_dual.get('jacobian_roll', float('nan')):.3f} "
            f"P={best_dual.get('jacobian_pitch', float('nan')):.3f} "
            f"Y={best_dual.get('jacobian_yaw', float('nan')):.3f})",
            "",
        ])
    elif best_medw.get('epoch', -1) > 0:
        bm = _medw_max_rpy(best_medw)
        lines.extend([
            "## Best MEDW (dual gate 未通过)",
            f"- Epoch: {best_medw['epoch']}",
            f"- max(R,P,Y): {bm:.4f}° "
            f"(R={best_medw['roll']:.4f} P={best_medw['pitch']:.4f} Y={best_medw['yaw']:.4f})",
            "",
        ])

    if kpi_history:
        lines.extend([
            "## KPI 评估历史 (每 eval epoch)",
            "",
            "| Epoch | max(R,P,Y)° | R | P | Y | Jac overall | Jac R | P | Y | Dual |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for rec in kpi_history:
            m = rec.get('medw')
            j = rec.get('jacobian')
            if m is None:
                lines.append(f"| {rec['epoch']} | - | - | - | - | - | - | - | - | - |")
                continue
            mx = _medw_max_rpy(m)
            jo = j.get('overall', float('nan')) if j else float('nan')
            jr = j.get('roll', float('nan')) if j else float('nan')
            jp = j.get('pitch', float('nan')) if j else float('nan')
            jy = j.get('yaw', float('nan')) if j else float('nan')
            dg = "PASS" if rec.get('dual_pass') else "FAIL"
            lines.append(
                f"| {rec['epoch']} | {mx:.4f} | {m['roll']:.4f} | {m['pitch']:.4f} | "
                f"{m['yaw']:.4f} | {jo:.3f} | {jr:.3f} | {jp:.3f} | {jy:.3f} | {dg} |")
        lines.append("")

    report_md = os.path.join(log_dir, "CONVERGENCE_REPORT.md")
    with open(report_md, 'w') as f:
        f.write("\n".join(lines) + "\n")

    report_json = {
        'verdict': verdict,
        'verdict_cn': verdict_cn,
        'converged': converged,
        'criteria': {
            'medw_max_rpy_deg': medw_thr,
            'jacobian_min': jac_thr,
            'medw_window': args.medw_eval_max_frames,
            'jacobian_angle_deg': args.jacobian_eval_angle_deg,
        },
        'best_dual': best_dual,
        'best_medw': best_medw,
        'kpi_history': kpi_history,
        'report_md': report_md,
    }
    json_path = os.path.join(log_dir, "convergence_report.json")
    with open(json_path, 'w') as jf:
        json.dump(report_json, jf, indent=2)
    return report_md, json_path, verdict


def _clear_projfusion_encoder_buffer(model):
    """Reset ProjFusion feat_buffer between multiple forwards in one step."""
    raw = model.module if hasattr(model, 'module') else model
    enc = getattr(getattr(raw, 'proj_branch', None), 'encoder', None)
    if enc is not None and hasattr(enc, 'clear_buffer'):
        enc.clear_buffer()


def _get_gmp_model(model):
    """GeoMatchProjCalib (GMP) wrapper, or None for legacy backends."""
    raw = model.module if hasattr(model, 'module') else model
    if hasattr(raw, 'forward_pose_from_cache') and hasattr(raw, '_stash_proj_cache_for_jacobian'):
        return raw
    return None


def _compute_jacobian_supervision_loss(
        model, resize_imgs, pcs_t, gt_T_t, init_T_np, post_cam2ego_T,
        intrinsic_matrix, masks_t, probe_deg, use_amp, amp_dtype, domain_ids_t=None,
        max_samples=4, T_pred_center=None, axis_weights=None):
    """Train-time loss: encourage d(correction)/d(bias) ≈ 1 on a random axis.

    GMP: reuses encoder cache from main forward via forward_pose_from_cache (no
    DINOv2/PointGPT re-encode). Must run after main loss backward() so stash is
    populated and main graph is released. Non-GMP: full model forward fallback.
    """
    B = gt_T_t.shape[0]
    if B < 1 or probe_deg <= 0:
        return None
    n_sub = min(max_samples, B)
    sub_idx = np.random.choice(B, n_sub, replace=False)
    axis_probs = None
    if axis_weights is not None:
        axis_probs = np.asarray(axis_weights, dtype=np.float64)
        if axis_probs.shape[0] == 3 and np.isfinite(axis_probs).all() and axis_probs.sum() > 0:
            axis_probs = axis_probs / axis_probs.sum()
        else:
            axis_probs = None
    ax_idx = int(np.random.choice(3, p=axis_probs)) if axis_probs is not None else int(np.random.randint(0, 3))
    probe = float(probe_deg)

    init_probe_np = init_T_np.copy()
    delta = [0.0, 0.0, 0.0]
    delta[ax_idx] = probe
    for i in sub_idx:
        init_probe_np[i] = _euler_perturb_T(init_T_np[i], delta)

    sub_t = torch.as_tensor(sub_idx, device=gt_T_t.device, dtype=torch.long)
    imgs_s = resize_imgs.index_select(0, sub_t)
    pcs_s = pcs_t.index_select(0, sub_t)
    gt_s = gt_T_t.index_select(0, sub_t)
    post_s = post_cam2ego_T.index_select(0, sub_t)
    K_s = intrinsic_matrix.index_select(0, sub_t)
    masks_s = masks_t.index_select(0, sub_t) if masks_t is not None else None
    dom_s = domain_ids_t.index_select(0, sub_t) if domain_ids_t is not None else None
    init_base_t = torch.from_numpy(init_T_np[sub_idx].astype(np.float32)).to(gt_T_t.device)
    init_probe_t = torch.from_numpy(init_probe_np[sub_idx].astype(np.float32)).to(gt_T_t.device)

    R_gt = gt_s[:, :3, :3]
    gmp = _get_gmp_model(model)
    use_cache_probe = (
        gmp is not None and getattr(gmp, '_jac_stash', None) is not None)

    if T_pred_center is not None:
        T0 = T_pred_center.index_select(0, sub_t)
        corr_0 = _axis_correction_torch(
            R_gt, init_base_t[:, :3, :3], T0[:, :3, :3]).detach()
    elif use_cache_probe:
        with torch.no_grad():
            T0 = gmp.forward_pose_from_cache(
                init_base_t, gt_s, pcs_s, masks=masks_s, batch_indices=sub_t)
        corr_0 = _axis_correction_torch(
            R_gt, init_base_t[:, :3, :3], T0[:, :3, :3])
    else:
        with torch.no_grad():
            _clear_projfusion_encoder_buffer(model)
            T0, _, _ = model(
                imgs_s, pcs_s, gt_s, init_base_t, post_s, K_s,
                masks=masks_s, out_init_loss=False, domain_ids=dom_s)
            corr_0 = _axis_correction_torch(
                R_gt, init_base_t[:, :3, :3], T0[:, :3, :3])

    with autocast(enabled=use_amp, dtype=amp_dtype):
        if use_cache_probe:
            T_out_probe = gmp.forward_pose_from_cache(
                init_probe_t, gt_s, pcs_s, masks=masks_s, batch_indices=sub_t)
        else:
            _clear_projfusion_encoder_buffer(model)
            T_out_probe, _, _ = model(
                imgs_s, pcs_s, gt_s, init_probe_t, post_s, K_s,
                masks=masks_s, out_init_loss=False, domain_ids=dom_s)

    corr_probe = _axis_correction_torch(
        R_gt, init_probe_t[:, :3, :3], T_out_probe[:, :3, :3])
    j_est = (corr_probe[:, ax_idx] - corr_0[:, ax_idx]) / probe
    j_raw = torch.nn.functional.smooth_l1_loss(
        j_est, torch.ones_like(j_est), beta=0.5, reduction='mean')
    return j_raw, j_est.detach().mean()


def _batch_data_to_numpy(data, xyz_only=False):
    """Convert dataloader batch (list/ndarray/torch.Tensor) to host numpy."""
    if isinstance(data, torch.Tensor):
        out = data.detach().cpu().numpy()
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            out = np.asarray(data)
        elif isinstance(data[0], torch.Tensor):
            out = np.stack([x.detach().cpu().numpy() for x in data])
        else:
            out = np.asarray(data)
    else:
        out = np.asarray(data)
    if xyz_only and out.ndim >= 3 and out.shape[-1] > 3:
        out = out[..., :3]
    return out


def _compute_jacobian_one_batch(raw_model, imgs, pcs, masks, gt_T_np, intrinsics, device,
                                angle_range, n_probes, use_amp, amp_dtype, identity_4x4,
                                xyz_only_choise):
    """Per-axis J = d(correction)/d(init_bias) with controlled single-axis sweep."""
    B = len(imgs) if not isinstance(imgs, torch.Tensor) else imgs.shape[0]
    base_init_T_np, _, _ = generate_single_perturbation_from_T(
        gt_T_np, angle_range_deg=2.0, trans_range=0.0, rotation_only=True,
        distribution='truncated_normal')
    bias_levels = np.linspace(-angle_range, angle_range, n_probes)
    pcs_np = _batch_data_to_numpy(pcs, xyz_only=xyz_only_choise)
    pcs_t = torch.from_numpy(pcs_np).float().to(device, non_blocking=True)
    gt_T_t = torch.from_numpy(gt_T_np.astype(np.float32)).to(device, non_blocking=True)
    imgs_np = _batch_data_to_numpy(imgs)
    resize_imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float().to(device, non_blocking=True)
    post_cam2ego_T = identity_4x4.unsqueeze(0).expand(B, -1, -1)
    intrinsic_matrix = torch.from_numpy(np.array(intrinsics, dtype=np.float32)).to(device, non_blocking=True)
    masks_t = torch.from_numpy(np.array(masks)).float().to(device, non_blocking=True) if masks is not None else None

    axis_jacobians = {}
    for ax_idx, ax_name in enumerate(['roll', 'pitch', 'yaw']):
        corrections_per_bias = []
        for bias_deg in bias_levels:
            biased_init_np = base_init_T_np.copy()
            delta = [0.0, 0.0, 0.0]
            delta[ax_idx] = float(bias_deg)
            for b in range(B):
                biased_init_np[b] = _euler_perturb_T(base_init_T_np[b], delta)
            init_T_t = torch.from_numpy(biased_init_np.astype(np.float32)).to(device, non_blocking=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                T_pred, _, _ = raw_model(
                    resize_imgs, pcs_t, gt_T_t, init_T_t, post_cam2ego_T,
                    intrinsic_matrix, masks=masks_t, out_init_loss=False,
                )
            T_pred_np = T_pred.detach().cpu().numpy()
            batch_corr = []
            for b in range(B):
                _, _, corr = _axis_error_and_correction_euler(
                    gt_T_np[b], biased_init_np[b], T_pred_np[b])
                batch_corr.append(corr[ax_idx])
            corrections_per_bias.append((float(bias_deg), float(np.mean(batch_corr))))
        if len(corrections_per_bias) >= 3:
            biases = np.array([x[0] for x in corrections_per_bias])
            corrections = np.array([x[1] for x in corrections_per_bias])
            axis_jacobians[ax_name] = float(np.polyfit(biases, corrections, 1)[0])
        else:
            axis_jacobians[ax_name] = float('nan')
    return axis_jacobians


def _run_jacobian_eval_inprocess(raw_model, val_loader, device, args, use_amp, amp_dtype,
                                 identity_4x4, xyz_only_choise):
    """Standalone Jacobian sweep (re-reads val_loader). Prefer in-val-loop hook below."""
    n_batches = args.jacobian_eval_batches
    angle_range = args.jacobian_eval_angle_deg
    n_probes = args.jacobian_eval_n_probes
    accum = {'roll': [], 'pitch': [], 'yaw': []}
    raw_model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if batch_idx >= n_batches or batch_data is None:
                break
            imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data[:5]
            gt_T_np = np.array(gt_T_to_camera).astype(np.float32)
            j = _compute_jacobian_one_batch(
                raw_model, imgs, pcs, masks, gt_T_np, intrinsics, device,
                angle_range, n_probes, use_amp, amp_dtype, identity_4x4, xyz_only_choise,
            )
            for k, v in j.items():
                if v == v:
                    accum[k].append(v)
    return _finalize_jacobian_axis_accum(accum)


def _build_val_idx_to_seq(val_preprocessed, custom_dataset):
    """Map val DataLoader sample index -> sequence id (for MEDW aggregation)."""
    idx_to_seq = {}
    subset = val_preprocessed.dataset if hasattr(val_preprocessed, 'dataset') else val_preprocessed
    all_files = getattr(custom_dataset, 'all_files', None)
    if hasattr(subset, 'indices') and all_files is not None:
        for loader_idx, global_idx in enumerate(subset.indices):
            idx_to_seq[loader_idx] = all_files[global_idx].split('/')[0]
    elif all_files is not None:
        for loader_idx, fpath in enumerate(all_files):
            idx_to_seq[loader_idx] = fpath.split('/')[0]
    return idx_to_seq


def _compute_medw_from_val_accum(all_T_pred, all_T_gt, sample_sequences, window=200):
    """MEDW from val-eval forward results (zero extra inference)."""
    if not all_T_pred:
        return None
    T_pred_arr = np.array(all_T_pred)
    T_gt_arr = np.array(all_T_gt)
    seq_arr = np.array(sample_sequences)
    unique_seqs = sorted(set(sample_sequences), key=lambda x: (str(type(x)), str(x)))
    return _compute_medw_deploy(T_pred_arr, T_gt_arr, seq_arr, unique_seqs, window=window)


def _ddp_val_rank_indices(dataset_len, rank, world_size, drop_last=False):
    """Mirror DistributedSampler(shuffle=False) index assignment per rank."""
    if drop_last:
        num_samples = dataset_len // world_size
    else:
        num_samples = math.ceil(dataset_len / world_size)
    total_size = num_samples * world_size
    indices = list(range(dataset_len))
    if total_size > len(indices):
        indices += indices[:(total_size - len(indices))]
    return indices[rank:total_size:world_size]


def _ddp_all_reduce_scalar(value, device):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def _ddp_all_reduce_dict_sum(d, device):
    if not d:
        return d
    return {k: _ddp_all_reduce_scalar(v, device) for k, v in d.items()}


def _ddp_gather_object(local_obj):
    if not (dist.is_available() and dist.is_initialized()):
        return [local_obj]
    world_size = dist.get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_obj)
    return gathered

import sys
import io
import math
from contextlib import contextmanager

_tprint_log_file = None
_original_stderr = None

class _StderrTee:
    """Tee stderr to both original stderr and train.log file."""
    def __init__(self, orig, log_file):
        self._orig = orig
        self._log_file = log_file
    def write(self, s):
        self._orig.write(s)
        if self._log_file is not None and not self._log_file.closed:
            try:
                self._log_file.write(s)
                self._log_file.flush()
            except (ValueError, OSError):
                pass
    def flush(self):
        self._orig.flush()
    def fileno(self):
        return self._orig.fileno()
    def isatty(self):
        return self._orig.isatty()

def _cleanup_log():
    global _tprint_log_file, _original_stderr
    if _original_stderr is not None:
        sys.stderr = _original_stderr
        _original_stderr = None
    if _tprint_log_file is not None and not _tprint_log_file.closed:
        _tprint_log_file.flush()
        _tprint_log_file.close()
        _tprint_log_file = None

def tprint_setup(log_dir):
    """Setup tprint to also write to train.log in log_dir.
    Also installs a stderr tee so warnings/errors go to train.log."""
    global _tprint_log_file, _original_stderr
    if _tprint_log_file is not None:
        return
    import atexit
    log_path = os.path.join(log_dir, "train.log")
    _tprint_log_file = open(log_path, "a", buffering=1)
    _original_stderr = sys.stderr
    sys.stderr = _StderrTee(sys.stderr, _tprint_log_file)
    atexit.register(_cleanup_log)

def tprint(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] " + " ".join(str(a) for a in args)
    print(msg, flush=True)
    if _tprint_log_file is not None:
        _tprint_log_file.write(msg + "\n")
        _tprint_log_file.flush()

@contextmanager
def capture_prints(is_main):
    """Capture stdout+stderr from third-party modules (Dataset, Model init).
    On master: tee stdout to terminal+train.log; stderr already handled by _StderrTee.
    On workers: suppress entirely to avoid N-fold duplication in torchrun."""
    if not is_main:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return

    buf = io.StringIO()

    class TeeWriter:
        def __init__(self, orig, buf):
            self._orig = orig
            self._buf = buf
        def write(self, s):
            self._orig.write(s)
            self._buf.write(s)
        def flush(self):
            self._orig.flush()
        def fileno(self):
            return self._orig.fileno()
        def isatty(self):
            return getattr(self._orig, 'isatty', lambda: False)()

    old_stdout = sys.stdout
    sys.stdout = TeeWriter(old_stdout, buf)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        captured = buf.getvalue()
        if captured and _tprint_log_file is not None:
            _tprint_log_file.write(captured)
            _tprint_log_file.flush()


def _subsample_pcd(pcs_np, masks, max_pts):
    """Subsample point clouds to max_pts per sample. Returns (pcs_np, masks)."""
    B = pcs_np.shape[0]
    new_pcs, new_masks = [], []
    for bi in range(B):
        m = np.asarray(masks[bi])
        valid_idx = np.where(m == 1)[0]
        n_valid = len(valid_idx)
        if n_valid > max_pts:
            chosen = np.random.choice(n_valid, max_pts, replace=False)
            chosen.sort()
            sel_idx = valid_idx[chosen]
            new_pcs.append(pcs_np[bi, sel_idx])
            new_masks.append(np.ones(max_pts, dtype=np.float32))
        elif n_valid > 0:
            new_pcs.append(pcs_np[bi, valid_idx])
            new_masks.append(np.ones(n_valid, dtype=np.float32))
        else:
            new_pcs.append(pcs_np[bi, :1])
            new_masks.append(np.ones(1, dtype=np.float32))
    max_n = max(p.shape[0] for p in new_pcs)
    padded_pcs = np.zeros((B, max_n, pcs_np.shape[2]), dtype=np.float32)
    padded_masks = np.zeros((B, max_n), dtype=np.float32)
    for bi in range(B):
        n = new_pcs[bi].shape[0]
        padded_pcs[bi, :n] = new_pcs[bi]
        padded_masks[bi, :n] = new_masks[bi]
    return padded_pcs, padded_masks


def _apply_color_jitter(imgs_tensor, strength):
    """Apply random color jitter to a batch of images (B, C, H, W) in [0, 255]."""
    B = imgs_tensor.shape[0]
    for i in range(B):
        img = imgs_tensor[i]  # (C, H, W)
        brightness = 1.0 + (torch.rand(1).item() * 2 - 1) * strength * 0.3
        img = img * brightness
        contrast = 1.0 + (torch.rand(1).item() * 2 - 1) * strength * 0.3
        mean = img.mean()
        img = (img - mean) * contrast + mean
        saturation = 1.0 + (torch.rand(1).item() * 2 - 1) * strength * 0.3
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        img = img * saturation + gray.unsqueeze(0) * (1 - saturation)
        imgs_tensor[i] = img.clamp(0, 255)
    return imgs_tensor


def _augment_intrinsics(intrinsic_matrix, strength, cx_cy_strength=None):
    """Randomly perturb camera intrinsic matrix to improve robustness to unseen cameras.

    Augmentation strategy:
      - fx, fy scaled by same random factor (preserves aspect ratio)
      - cx, cy independently offset (separate strength to match actual cross-vehicle variation)
      - K[2,2] = 1 is preserved

    The frustum geometry in LSS's get_geometry() depends on inv(K), so varying K
    during training forces the model to generalize across different focal lengths.

    Args:
        intrinsic_matrix: (B, 3, 3) tensor on device
        strength: max relative deviation for fx/fy, e.g. 0.05 means ±5%
        cx_cy_strength: max relative deviation for cx/cy (default: same as strength)
    Returns:
        augmented (B, 3, 3) tensor (same device, same dtype)
    """
    if cx_cy_strength is None:
        cx_cy_strength = strength

    B = intrinsic_matrix.shape[0]
    dev = intrinsic_matrix.device
    K = intrinsic_matrix.clone()

    focal_scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * strength
    cx_scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * cx_cy_strength
    cy_scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * cx_cy_strength

    K[:, 0, 0] *= focal_scale       # fx
    K[:, 1, 1] *= focal_scale       # fy
    K[:, 0, 2] *= cx_scale          # cx
    K[:, 1, 2] *= cy_scale          # cy
    return K


def _apply_fov_crop(imgs_tensor, intrinsics, crop_ratio_min=0.75, crop_ratio_max=0.95):
    """
    Apply random center crop to simulate different FOV/mounting height.
    Simulates various camera installations with different vertical FOV coverage.
    
    Args:
        imgs_tensor: (B, 3, H, W) torch.Tensor in [0, 255]
        intrinsics: (B, 3, 3) numpy array
        crop_ratio_min/max: range of crop ratios (e.g. 0.75-0.95 keeps center 75%-95%)
    
    Returns:
        cropped_imgs: (B, 3, H, W) torch.Tensor (resized back to original size)
        updated_intrinsics: (B, 3, 3) numpy array
    """
    B, C, H, W = imgs_tensor.shape
    device = imgs_tensor.device
    
    # Convert to numpy for cropping
    imgs_np = imgs_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
    
    cropped_imgs = []
    updated_intrinsics = []
    
    for b in range(B):
        img = imgs_np[b]
        K = intrinsics[b].copy()
        
        # Random crop ratio (independent for H and W to simulate different aspects)
        crop_ratio_h = random.uniform(crop_ratio_min, crop_ratio_max)
        crop_ratio_w = random.uniform(crop_ratio_min, crop_ratio_max)
        
        H_crop = int(H * crop_ratio_h)
        W_crop = int(W * crop_ratio_w)
        
        # Center crop
        y_start = (H - H_crop) // 2
        x_start = (W - W_crop) // 2
        img_cropped = img[y_start:y_start+H_crop, x_start:x_start+W_crop]
        
        # Resize back to original size
        img_resized = cv2.resize(img_cropped, (W, H))
        
        # Update intrinsics
        # Step 1: adjust for crop offset
        K[0, 2] -= x_start  # cx
        K[1, 2] -= y_start  # cy
        
        # Step 2: adjust for resize scale
        scale_x = W / W_crop
        scale_y = H / H_crop
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        
        cropped_imgs.append(img_resized)
        updated_intrinsics.append(K)
    
    # Convert back to tensor
    cropped_imgs = np.stack(cropped_imgs)  # (B, H, W, 3)
    cropped_imgs_tensor = torch.from_numpy(cropped_imgs).permute(0, 3, 1, 2).to(device).float()
    updated_intrinsics = np.stack(updated_intrinsics)
    
    return cropped_imgs_tensor, updated_intrinsics


def _parse_csv_cli_list(s, cast=float):
    """Parse comma-separated CLI values (tolerate bash printf %q artifacts like '16\\')."""
    cleaned = str(s).replace('\\', '')
    return [cast(x.strip()) for x in cleaned.split(',') if x.strip()]


def _apply_lidar_sparsification(pcs_np, masks, target_lines=32, 
                                  vertical_fov=(-25, 15), original_lines=128):
    """
    Simulate sparse LiDAR by vertical angle binning (no ring id needed).
    Models different LiDAR configurations (16/32/64 vs 128 lines).
    
    Uses vertical angle to create pseudo ring bins, then samples uniformly
    to simulate lower vertical resolution LiDAR.
    
    Args:
        pcs_np: (B, N, 4) numpy array [x, y, z, intensity]
        masks: list of (N,) masks
        target_lines: target number of lines (16/32/64)
        vertical_fov: (min_deg, max_deg) vertical FOV range
        original_lines: original resolution (e.g. 128)
    
    Returns:
        sparse_pcs: (B, N_sparse, 4) padded array
        sparse_masks: list of (N_sparse,) masks
    """
    B = pcs_np.shape[0]
    v_min, v_max = np.deg2rad(vertical_fov[0]), np.deg2rad(vertical_fov[1])
    
    sparse_pcs = []
    sparse_masks = []
    
    for b in range(B):
        pc = pcs_np[b]  # (N, 4)
        mask = np.asarray(masks[b])
        valid_idx = np.where(mask == 1)[0]
        
        if len(valid_idx) == 0:
            sparse_pcs.append(pc)
            sparse_masks.append(mask)
            continue
        
        # Get valid points
        pc_valid = pc[valid_idx]  # (N_valid, 4)
        x, y, z = pc_valid[:, 0], pc_valid[:, 1], pc_valid[:, 2]
        
        # Compute vertical angle for each point
        distance_xy = np.sqrt(x**2 + y**2)
        distance_xy = np.maximum(distance_xy, 1e-6)  # avoid division by zero
        vertical_angle = np.arctan2(z, distance_xy)  # radians
        
        # Assign to pseudo ring bins
        # Normalize angle to [0, 1]
        angle_normalized = (vertical_angle - v_min) / (v_max - v_min)
        angle_normalized = np.clip(angle_normalized, 0, 1)
        
        # Map to original_lines bins
        pseudo_ring_id = (angle_normalized * original_lines).astype(int)
        pseudo_ring_id = np.clip(pseudo_ring_id, 0, original_lines - 1)
        
        # Select target_lines uniformly from original_lines
        selected_rings = np.linspace(0, original_lines - 1, target_lines).astype(int)
        
        # Keep only points from selected rings
        keep_mask = np.isin(pseudo_ring_id, selected_rings)
        pc_sparse = pc_valid[keep_mask]
        
        if len(pc_sparse) == 0:
            # Fallback: keep at least 1 point
            pc_sparse = pc_valid[:1]
        
        sparse_pcs.append(pc_sparse)
        sparse_masks.append(np.ones(len(pc_sparse)))
    
    # Pad to same length
    max_pts = max(pc.shape[0] for pc in sparse_pcs)
    padded_pcs = np.full((B, max_pts, pcs_np.shape[2]), 999999, dtype=np.float32)
    padded_masks = []
    
    for b in range(B):
        n = sparse_pcs[b].shape[0]
        padded_pcs[b, :n, :] = sparse_pcs[b]
        padded_masks.append(np.concatenate([sparse_masks[b], np.zeros(max_pts - n)]))
    
    return padded_pcs, padded_masks


def setup_ddp():
    """Auto-detect and initialize DDP when launched via torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        timeout_minutes = int(os.environ.get('DDP_TIMEOUT_MINUTES', '30'))
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=timeout_minutes))
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--dataset_root", type=str, default="YOUR_PATH_TO_KITTI/kitti-odemetry")
    parser.add_argument("--log_dir", type=str, default="./logs/kitti_default")
    parser.add_argument("--save_ckpt_per_epoches", type=int, default=-1)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--angle_range_deg", type=float, default=None)
    parser.add_argument("--trans_range", type=float, default=None)
    parser.add_argument("--eval_angle_range_deg", type=float, default=None)
    parser.add_argument("--eval_trans_range", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--eval_epoches", type=int, default=50)
    parser.add_argument("--deformable", type=int, default=-1)
    parser.add_argument("--bev_encoder", type=int, default=1)
    parser.add_argument("--xyz_only", type=int, default=1)
    parser.add_argument("--max_pcd_points", type=int, default=0,
                        help="Max points per sample (0=unlimited). For V42 PointEncoder, 16384 recommended.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--scheduler", type=int, default=-1)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume training from a full checkpoint (restores epoch, optimizer, scheduler, scaler)")
    parser.add_argument("--use_custom_dataset", type=int, default=0, help="使用 CustomDataset (1) 还是 KittiDataset (0)")
    # 图像尺寸参数
    parser.add_argument("--target_width", type=int, default=None, help="目标图像宽度 (默认: KITTI=704, 自定义4K=640)")
    parser.add_argument("--target_height", type=int, default=None, help="目标图像高度 (默认: KITTI=256, 自定义4K=360)")
    # 数据利用率校验参数
    parser.add_argument("--validate_data", type=int, default=0, help="是否在训练前验证数据利用率 (1=启用, 0=禁用)")
    parser.add_argument("--validate_sample_ratio", type=float, default=0.1, help="数据验证采样比例 (0.0-1.0)")
    parser.add_argument("--min_point_utilization", type=float, default=0.5, help="最低点云利用率阈值 (0.0-1.0)")
    parser.add_argument("--min_valid_ratio", type=float, default=0.9, help="最低有效帧比例阈值 (0.0-1.0)")
    parser.add_argument("--max_frames_per_seq", type=int, default=None, help="每个序列最大帧数 (均匀采样), None=使用全部帧")
    parser.add_argument("--sample_step", type=int, default=None, help="采样步长 (每隔N帧取1帧), 与max_frames_per_seq互斥")
    parser.add_argument("--pose_aware_sampling", action="store_true", help="启用基于Pose的智能去冗余采样")
    parser.add_argument("--poses_dir", type=str, default="", help="Pose文件目录 (KITTI格式)")
    # 可视化参数
    parser.add_argument("--vis_freq", type=int, default=40, help="训练可视化频率 (每多少个batch可视化一次)")
    parser.add_argument("--vis_samples", type=int, default=3, help="每次可视化的样本数")
    parser.add_argument("--vis_points", type=int, default=80000, help="每个样本最大可视化点数")
    parser.add_argument("--vis_point_radius", type=int, default=1, help="可视化点的半径")
    parser.add_argument("--enable_vis", type=int, default=1, help="是否启用点云投影可视化 (1=启用, 0=禁用)")
    parser.add_argument("--enable_ckpt_eval", type=int, default=1, help="是否在保存checkpoint时进行评估 (1=启用, 0=禁用)")
    parser.add_argument("--enable_medw_eval", type=int, default=0,
                        help="eval epoch 时在 val 划分上统计 MEDW (复用 val forward, 1=启用)")
    parser.add_argument("--medw_eval_max_frames", type=int, default=200,
                        help="MEDW eval 每序列均匀采样帧数 (默认 200，复用 val forward)")
    parser.add_argument("--enable_dual_gate_ckpt", type=int, default=0,
                        help="Save ckpt_best_dual.pth only when MEDW and Jacobian both pass gates (1=on)")
    parser.add_argument("--dual_gate_jacobian_min", type=float, default=0.85,
                        help="Dual gate: min Jacobian on overall AND each axis (roll/pitch/yaw)")
    parser.add_argument("--dual_gate_medw_max", type=float, default=0.30,
                        help="Dual gate: max(R,P,Y) MEDW error (deg) on val split must be below this")
    parser.add_argument("--enable_jacobian_gate_ckpt", type=int, default=0,
                        help="Save ckpt_best_jacobian.pth when val Jacobian overall improves (1=on)")
    parser.add_argument("--jacobian_early_stop_min", type=float, default=0.0,
                        help="V52 S2: stop if val Jacobian overall below this for patience evals (0=off)")
    parser.add_argument("--jacobian_early_stop_patience", type=int, default=2,
                        help="V52 S2: consecutive eval cycles below jacobian_early_stop_min before stop")
    parser.add_argument("--enable_jacobian_eval", type=int, default=0,
                        help="eval epoch 时在 val 上跑 lightweight Jacobian (1=启用)")
    parser.add_argument("--jacobian_eval_angle_deg", type=float, default=3.0,
                        help="Jacobian bias sweep ±degrees (部署±3°对齐; 训练仍可±5°)")
    parser.add_argument("--jacobian_eval_batches", type=int, default=1,
                        help="Jacobian sweep 复用 val 前 N 个 batch (额外 forward, 不二次读数据)")
    parser.add_argument("--jacobian_eval_n_probes", type=int, default=3,
                        help="每轴 bias 采样点数 (≥3; 默认 3 平衡速度与 polyfit)")
    parser.add_argument("--jacobian_loss_weight", type=float, default=0.0,
                        help="Train-time Jacobian supervision: penalize (d(correction)/d(bias)-1)^2 (0=off)")
    parser.add_argument("--jacobian_loss_start_epoch", type=int, default=10,
                        help="Epoch to start jacobian_loss (after main pose loss stabilizes)")
    parser.add_argument("--jacobian_loss_probe_deg", type=float, default=2.0,
                        help="±degrees for two-probe Jacobian loss (init±probe on one axis)")
    parser.add_argument("--jacobian_loss_interval", type=int, default=4,
                        help="Run Jacobian supervision every N train batches (1=every batch)")
    parser.add_argument("--jacobian_loss_axis_weights", type=str, default="",
                        help="Optional R,P,Y sampling weights for train-time Jacobian loss, e.g. '1,1,2'")
    parser.add_argument("--compile", type=int, default=0, help="使用 torch.compile 加速模型 (1=启用, 0=禁用)")
    parser.add_argument("--no_amp", type=int, default=0, help="禁用 AMP 混合精度训练 (1=禁用FP16, 用FP32; 0=默认FP16)")
    parser.add_argument("--amp_bf16", type=int, default=0, help="AMP 使用 bfloat16 替代 float16 (减少溢出风险, 需GPU支持)")
    parser.add_argument("--rotation_only", type=int, default=0,
                        help="仅优化旋转 (1=仅旋转, 0=旋转+平移同时优化)")
    parser.add_argument("--enable_axis_loss", type=int, default=0,
                        help="启用分轴旋转损失 (1=启用, 0=禁用)")
    parser.add_argument("--weight_axis_rotation", type=float, default=0.3,
                        help="分轴旋转损失权重 (默认: 0.3)")
    parser.add_argument("--axis_weights", type=str, default="3.0,1.5,1.0",
                        help="Roll,Pitch,Yaw weights for axis loss (default: 3.0,1.5,1.0)")
    parser.add_argument("--use_geodesic_loss", type=int, default=0,
                        help="Use SO(3) geodesic loss instead of quaternion distance (1=enable)")
    parser.add_argument("--use_balanced_axis_loss", type=int, default=0,
                        help="Use balanced Huber axis loss instead of weighted axis loss (1=enable)")
    parser.add_argument("--use_mlp_head", type=int, default=1,
                        help="Use 3-layer MLP regression head (1=MLP, 0=single Linear)")
    parser.add_argument("--use_pitch_branch", type=int, default=0,
                        help="Enable front-view Pitch branch for Z-aware Pitch prediction (1=enable)")
    parser.add_argument("--pitch_aux_weight", type=float, default=0.3,
                        help="Auxiliary pitch loss weight (only when use_pitch_branch=1)")
    parser.add_argument("--use_dla", type=int, default=0,
                        help="V44: Enable DLA multi-scale aggregation (1=enable)")
    parser.add_argument("--use_pitch_fusion", type=int, default=0,
                        help="V44: Enable Pitch branch inference fusion (1=enable)")
    parser.add_argument("--use_instance_norm", type=int, default=0,
                        help="V44: Enable Instance Normalization for domain alignment (1=enable)")
    parser.add_argument("--use_gated_instance_norm", type=int, default=0,
                        help="V45: Enable Gated Instance Norm (1=enable, overrides use_instance_norm)")
    parser.add_argument("--gin_init_gate", type=float, default=0.5,
                        help="V45: GIN initial gate value (0.5=balanced IN/bypass)")
    parser.add_argument("--gin_channels", type=int, default=0,
                        help="V48: Partial GIN — channels to apply GIN to (0=all, <feat_dim=partial)")
    parser.add_argument("--pitch_vertical_bands", type=int, default=3,
                       help="V48: Number of vertical bands for pitch branch (1=GAP, 3=top/mid/bot)")
    parser.add_argument("--gin_gate_reg_target", type=float, default=0.0,
                        help="V48: GIN gate regularization target (0=disabled, 0.5=balanced)")
    parser.add_argument("--gin_gate_reg_weight", type=float, default=0.0,
                        help="V48: weight for GIN gate regularization loss (0=disabled)")
    parser.add_argument("--drop_path_rate", type=float, default=0.1,
                        help="Stochastic depth rate for transformer layers")
    parser.add_argument("--head_dropout", type=float, default=0.1,
                        help="Dropout rate before prediction heads")
    parser.add_argument("--bev_pool_factor", type=int, default=0,
                        help="Spatial avg-pool factor before transformer (0=disabled, 2=2x2 pool)")
    parser.add_argument("--use_foundation_depth", type=int, default=0,
                        help="Replace LSS depth head with frozen foundation depth model (1=enable)")
    parser.add_argument("--depth_model_type", type=str, default="midas_small",
                        choices=["midas_small", "dpt_swin2_t", "dpt_beit_l"],
                        help="Foundation depth model type (default: midas_small)")
    parser.add_argument("--fd_mode", type=str, default="replace",
                        choices=["replace", "replace_v1", "replace_v2", "dual_path", "supervision"],
                        help="Foundation depth mode: replace(v1 compat), replace_v2(fixed), dual_path(fusion), supervision(aux loss)")
    parser.add_argument("--depth_sup_alpha", type=float, default=0.5,
                        help="Depth supervision loss weight (only for fd_mode=supervision)")
    parser.add_argument("--voxel_mode", type=str, default="hard",
                        choices=["hard", "scatter"],
                        help="Voxelization: hard(CUDA hard_voxelize+sum) or scatter(torch.unique+scatter_add_, drinfer-trace compatible)")
    parser.add_argument("--to_bev_mode", type=str, default="concat",
                        choices=["concat", "learned", "sum"],
                        help="Sparse-to-BEV: concat(z-plane concat), learned(per-z kernel), sum(scatter-add)")
    parser.add_argument("--scatter_reduce", type=str, default="sum",
                        choices=["sum", "mean"],
                        help="Scatter voxelization reduce: sum(match hard) or mean(dr_voxelization style)")
    parser.add_argument("--fuser_type", type=str, default="concat",
                        choices=["concat", "diff", "diff_v2"],
                        help="BEV fuser: concat(ConvFuser) / diff(BEVDiffFuser) / diff_v2(BEVDiffFuser with cam-drop-aware dual path)")
    parser.add_argument("--correlation_fusion", type=int, default=0,
                        help="V33-B: Use spatial correlation fusion instead of concat+transformer+global_pool. "
                             "Preserves spatial alignment info for true calibration ability.")
    parser.add_argument("--cross_correlation_fusion", type=int, default=0,
                        help="V34: Use cost-volume cross-correlation fusion for explicit spatial offset detection.")
    parser.add_argument("--explicit_tinit", type=int, default=0,
                        help="V35: Explicit T_init RPY encoding injected before prediction head.")
    parser.add_argument("--tinit_bev_film", type=int, default=0,
                        help="V80: Apply T_init-conditioned FiLM adapter to BEV feature map before pooling.")
    parser.add_argument("--tinit_query_film", type=int, default=0,
                        help="V81: Apply T_init-conditioned FiLM adapter to Cam2BEV query tokens.")
    parser.add_argument("--tinit_sensitivity_weight", type=float, default=0.0,
                        help="V35 Phase2: Weight for T_init sensitivity contrastive loss (0=disabled).")
    parser.add_argument("--correction_quat_loss_weight", type=float, default=0.0,
                        help="Direct raw correction quaternion supervision weight (0=disabled).")
    parser.add_argument("--iterative_refine", type=int, default=0,
                        help="V35 Phase3: Number of iterative refinement steps (0=disabled, 3=recommended).")
    parser.add_argument("--native_cross", type=int, default=0,
                        help="V36: Native-domain cross-attention mode (0=disabled, 1=enabled).")
    parser.add_argument("--native_cross_pc_groups", type=int, default=128,
                        help="V36: Number of point cloud groups for cross-attention.")
    parser.add_argument("--native_cross_n_harmonic", type=int, default=6,
                        help="V36: Number of harmonic functions for positional embedding.")
    parser.add_argument("--native_cross_n_layers", type=int, default=1,
                        help="V36: Number of stacked cross-attention layers (1=MVP, 2+=deeper).")
    parser.add_argument("--native_cross_dual_branch", type=int, default=1,
                        help="V36: Dual independent cross-attention branches (ProjFusion-style).")
    parser.add_argument("--native_cross_knn", type=int, default=8,
                        help="V36: kNN neighbors for local point geometry encoding.")
    parser.add_argument("--native_cross_use_fps", type=int, default=1,
                        help="V36: Use farthest-point sampling for point groups.")
    parser.add_argument("--native_cross_use_pointgpt", type=int, default=0,
                        help="V36: Use ProjFusion PointGPT pretrained encoder (requires ckpt).")
    parser.add_argument("--native_cross_pointgpt_ckpt", type=str, default=None,
                        help="V36: Path to kitti_pointgpt_tiny.pth")
    parser.add_argument("--native_cross_pointgpt_config", type=str, default=None,
                        help="V36: PointGPT yaml config (default: ProjFusion finetune_kitti_tiny.yaml)")
    parser.add_argument("--native_cross_pointgpt_max_depth", type=float, default=50.0,
                        help="V36: Point cloud depth normalization for PointGPT.")
    parser.add_argument("--native_cross_extend_ratio", type=float, default=1.0,
                        help="V36/V37: ProjFusion-style projection canvas expansion (2.5 recommended)")
    parser.add_argument("--fusion_backend", type=str, default="bev",
                        choices=["bev", "bev_only", "proj_only", "hybrid_dual", "hybrid_triple",
                                 "geo_match_proj", "cf_bev_r"],
                        help="Fusion backend: HTCN variants, V40 geo_match_proj (GMP), or V42 cf_bev_r.")
    # --- CF-BEV-R (V42) parameters ---
    parser.add_argument("--cf_feat_dim", type=int, default=256)
    parser.add_argument("--cf_n_groups", type=int, default=128)
    parser.add_argument("--cf_knn", type=int, default=8)
    parser.add_argument("--cf_corr_heads", type=int, default=4)
    parser.add_argument("--cf_corr_radius", type=int, default=4)
    parser.add_argument("--cf_num_queries", type=int, default=6)
    parser.add_argument("--cf_encoder_layers", type=int, default=2)
    parser.add_argument("--cf_decoder_layers", type=int, default=4)
    parser.add_argument("--use_rocr", type=int, default=1)
    parser.add_argument("--rocr_dropout", type=float, default=0.3)
    parser.add_argument("--rocr_center_bias", type=float, default=0.5)
    parser.add_argument("--rocr_detach_epochs", type=int, default=0,
                        help="Detach RoCR gradient for first N epochs (V42 S1)")
    parser.add_argument("--quat_norm_weight", type=float, default=0.5,
                        help="Quaternion normalization loss weight (default: 0.5)")
    parser.add_argument("--corr_alignment_weight", type=float, default=0.0)
    parser.add_argument("--corr_alignment_warmup", type=int, default=20)
    parser.add_argument("--seq_consistency_weight", type=float, default=0.0)
    parser.add_argument("--seq_consistency_start_epoch", type=int, default=999)
    parser.add_argument("--corr_window_mode", type=str, default="fixed",
                        choices=["fixed", "adaptive"])
    parser.add_argument("--pc_encoder_mode", type=str, default="pointgpt2bev",
                        choices=["spconv", "pointgpt2bev"],
                        help="HTCN BEV branch point cloud encoder.")
    parser.add_argument("--fusion_variant", type=str, default="gated",
                        choices=["gated", "cascade", "residual"],
                        help="HTCN fusion head variant.")
    parser.add_argument("--deep_supervision_weight", type=float, default=0.2,
                        help="HTCN auxiliary branch loss weight.")
    parser.add_argument("--gate_entropy_weight", type=float, default=0.0,
                        help="Weight for gate entropy regularization (prevent gate collapse, e.g. 0.05)")
    parser.add_argument("--projfusion_image_hw", type=int, nargs=2, default=[224, 448],
                        help="ViT input size for ProjFusion branch (H W). GMP default: 252 448.")
    # V40 GMP (geo_match_proj)
    parser.add_argument("--appearance_loss_weight", type=float, default=0.0,
                        help="V40 GeoConsistency photometric weight (P0a).")
    parser.add_argument("--depth_loss_weight", type=float, default=0.0,
                        help="V40 GeoConsistency depth weight (P0a).")
    parser.add_argument("--geo_loss_start_epoch", type=int, default=5,
                        help="V40: start GeoConsistency after this epoch (pose-only warmup).")
    parser.add_argument("--use_match_head", type=int, default=0,
                        help="V40 P1b: CorrespondenceHead + EPnP (0=off).")
    parser.add_argument("--use_local_correlation", type=int, default=0,
                        help="V40 P1a: local multi-head correlation (0=off).")
    parser.add_argument("--correspondence_loss_weight", type=float, default=0.0,
                        help="V40 P1b: L_corr weight.")
    parser.add_argument("--correspondence_loss_start_epoch", type=int, default=0,
                        help="Start adding L_corr to total_loss after this epoch (0-indexed).")
    parser.add_argument("--correspondence_loss_warmup_epochs", type=int, default=0,
                        help="Linearly ramp correspondence_loss_weight over N epochs after start.")
    parser.add_argument("--compose_mode", type=str, default="match_then_refine",
                        choices=["refine_only", "match_only", "match_then_refine"],
                        help="V40 P1: pose composition mode.")
    parser.add_argument("--num_correspondences", type=int, default=64,
                        help="V40 P1b: number of sparse correspondences K.")
    parser.add_argument("--match_valid_ratio_min", type=float, default=0.3,
                        help="V40 P1b: fallback to refine when valid_ratio below this.")
    parser.add_argument("--match_disable_fallback", type=int, default=0,
                        help="V41: 1=never zero R_match via fallback gate (force match path).")
    parser.add_argument("--match_gate_use_init_ratio", type=int, default=0,
                        help="V41: 1=gate fallback on match_valid_ratio_init not GT-masked ratio.")
    parser.add_argument("--match_confidence_threshold", type=float, default=0.2,
                        help="CorrespondenceHead min confidence for valid match.")
    parser.add_argument("--match_corr_validity_mode", type=str, default='gt',
                        choices=['gt', 'init'],
                        help="L_corr mask: gt=intersect GT proj (strict); init=T_init visible points.")
    parser.add_argument("--match_epnp_min_points", type=int, default=4,
                        help="EPnP min weighted points before identity R.")
    parser.add_argument("--match_phase_noise_max_deg", type=float, default=0.0,
                        help="Cap V32 continuous init noise (deg) when match head on; 0=use global max.")
    parser.add_argument("--correspondence_supervision", type=int, default=1,
                        help="V40 P1b: use T_gt pseudo uv labels for L_corr (1=on).")
    parser.add_argument("--differentiable_epnp", type=int, default=0,
                        help="V40 P1b: enable gradient flow through EPnP (0=detach/stable, 1=diff).")
    parser.add_argument("--diff_epnp_warmup_epochs", type=int, default=5,
                        help="V40 P1b: detach EPnP pose gradients for first N epochs when differentiable_epnp=1.")
    parser.add_argument("--cam_drop_prob", type=float, default=0.0,
                        help="Camera branch dropout probability during training (0=disabled)")
    parser.add_argument("--cam_drop_mode", type=str, default="zero",
                        choices=["zero", "noise"],
                        help="Camera dropout mode: zero(hard zeros) / noise(Gaussian noise at 0.1*std)")
    parser.add_argument("--intrinsic_input", action="store_true", default=False,
                        help="Feed normalized intrinsics (fx,fy,cx,cy) to prediction head")
    parser.add_argument("--lr_schedule", type=str, default="step",
                        choices=["step", "cosine_warm_restarts"],
                        help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                        help="LR multiplier for pretrained backbone (SwinT)")
    parser.add_argument("--bev_branch_lr_scale", type=float, default=None,
                        help="LR multiplier for HTCN BEV trainable modules (fuser/transformer/pc_branch). "
                             "Default: same as backbone_lr_scale")
    parser.add_argument("--backbone_warmup_epochs", type=int, default=0,
                        help="Gradually increase backbone LR from 1%% to 100%% over N epochs (0=disabled)")
    
    # P2b: FOV crop augmentation
    parser.add_argument("--augment_fov_crop_prob", type=float, default=0.0,
                        help="P2b: Probability of random FOV center crop (0.0-1.0, 0=disabled). "
                             "Simulates different mounting heights / camera FOV configurations.")
    parser.add_argument("--augment_fov_crop_ratio_min", type=float, default=0.75,
                        help="P2b: Min FOV crop ratio (e.g. 0.75=keep center 75%, default: 0.75)")
    parser.add_argument("--augment_fov_crop_ratio_max", type=float, default=0.95,
                        help="P2b: Max FOV crop ratio (e.g. 0.95=keep center 95%, default: 0.95)")
    
    # P2b: LiDAR sparsification augmentation
    parser.add_argument("--augment_lidar_sparse_prob", type=float, default=0.0,
                        help="P2b: Probability of simulating sparse LiDAR (0.0-1.0, 0=disabled). "
                             "Models 16/32/64-line LiDAR by vertical angle binning.")
    parser.add_argument("--augment_lidar_sparse_lines", type=str, default="16,32,64",
                        help="P2b: Comma-separated target line numbers (e.g. '16,32,64'). "
                             "Randomly picks one to simulate lower vertical resolution.")
    parser.add_argument("--augment_lidar_vertical_fov", type=str, default="-25,15",
                        help="P2b: Vertical FOV range in degrees (e.g. '-25,15' for -25° to +15°). "
                             "Used for vertical angle binning. Adjust per dataset.")
    
    parser.add_argument("--augment_mount_jitter_prob", type=float, default=0.0,
                        help="Probability of applying mount jitter to GT extrinsics (0=disabled). "
                             "Simulates diverse camera installations for domain generalization.")
    parser.add_argument("--augment_mount_jitter_rot_sigma", type=float, default=0.5,
                        help="Std of mount jitter rotation per axis in degrees (default: 0.5)")
    parser.add_argument("--augment_mount_jitter_trans_sigma", type=float, default=0.01,
                        help="Std of mount jitter translation per axis in meters (default: 0.01)")
    parser.add_argument("--use_contrastive_extrinsic", type=int, default=0,
                        help="Enable contrastive extrinsic embedding head. "
                             "Forces BEV diff map to encode geometric offset, not scene content.")
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="Weight of contrastive extrinsic loss (default: 0.1)")
    parser.add_argument("--bev_instance_norm", type=int, default=0,
                        help="Apply InstanceNorm2d to camera BEV features before fusion "
                             "(removes fixed FOV activation pattern, improves domain generalization)")
    parser.add_argument("--domain_adversarial", type=int, default=0,
                        help="Enable Domain Adversarial Training (DANN) to learn "
                             "domain-invariant BEV features (1=enable)")
    parser.add_argument("--domain_adversarial_weight", type=float, default=0.1,
                        help="Weight of DANN domain classification loss (default: 0.1)")
    parser.add_argument("--num_domains", type=int, default=0,
                        help="Number of domain classes for DANN (0=auto-detect from dataset)")
    parser.add_argument("--cam2bev_mode", type=str, default="lss", choices=["lss", "query"],
                        help="Camera-to-BEV mode: 'lss' (LSS depth lifting, default) or "
                             "'query' (BEVFormer-style deformable cross-attention, no depth)")
    parser.add_argument("--backbone_type", type=str, default="swin", choices=["swin", "dinov2"],
                        help="Image backbone: 'swin' (Swin-Tiny, default) or 'dinov2' (DINOv2)")
    parser.add_argument("--backbone_variant", type=str, default="dinov2-small",
                        choices=["dinov2-small", "dinov2-base"],
                        help="DINOv2 variant (only used when --backbone_type=dinov2)")
    parser.add_argument("--freeze_backbone", type=int, default=0,
                        help="Freeze backbone weights, only train BEV layers (0/1)")
    parser.add_argument("--backbone_freeze_layers", type=str, default=None,
                        help="Partial freeze: e.g. '0:-2' freezes all blocks except last 2. "
                             "Only effective when --freeze_backbone=1 and --backbone_type=dinov2")
    parser.add_argument("--backbone_weights", type=str, default=None,
                        help="Explicit path to backbone pretrained weights (.pth). "
                             "If not set, auto-searches ckpt/checkpoints/ and torch.hub cache")
    parser.add_argument("--layer_wise_lr_decay", type=float, default=1.0,
                        help="Layer-wise LR decay factor for SwinT backbone (1.0=no decay, "
                             "0.75=each deeper layer gets 0.75x more LR). Applied multiplicatively "
                             "on top of backbone_lr_scale. E.g. 0.75 with 3 stages: "
                             "stage0=0.56x, stage1=0.75x, stage2=1.0x backbone_lr")
    parser.add_argument("--cosine_T0", type=int, default=50,
                        help="CosineAnnealingWarmRestarts T_0 period")
    parser.add_argument("--cosine_Tmult", type=int, default=2,
                        help="CosineAnnealingWarmRestarts T_mult")
    parser.add_argument("--perturb_distribution", type=str, default="uniform",
                        choices=["uniform", "truncated_normal", "magnitude_balanced"],
                        help="Perturbation angle distribution")
    parser.add_argument("--multi_scale_perturb", type=str, default="",
                        help="V48: Multi-scale perturbation ranges with probabilities, "
                             "e.g. '0.5:0.1,1.0:0.1,2.0:0.1' (remaining prob uses angle_range_deg)")
    parser.add_argument("--per_axis_prob", type=float, default=0.0,
                        help="Probability of single-axis perturbation (0=disabled)")
    parser.add_argument("--per_axis_weights", type=str, default="",
                        help="Roll,Pitch,Yaw sampling weights for per-axis mode (e.g. 0.5,0.3,0.2). Empty=uniform")
    parser.add_argument("--symmetric_perturb", type=int, default=0,
                        help="Force symmetric positive/negative perturbation within each batch (0=disabled, 1=enabled)")
    parser.add_argument("--multi_range_prob", type=float, default=0.0,
                        help="V46: probability of using wide-range perturbation (experience replay) to prevent generalization collapse (0=disabled)")
    parser.add_argument("--multi_range_angle", type=float, default=5.0,
                        help="V46: wide-range angle for experience replay (default: 5.0°)")
    parser.add_argument("--zero_perturbation_prob", type=float, default=0.0,
                        help="V43: probability of using T_init=T_gt (zero perturbation) to teach model identity mapping")
    parser.add_argument("--zero_drift_loss_weight", type=float, default=0.0,
                        help="V51: extra loss on init=gt batches penalizing R_pred deviation from R_gt (0=disabled)")
    parser.add_argument("--zero_drift_loss_weight_start", type=float, default=0.0,
                        help="V52 S2: ramp start weight; 0=use weight fixed. Linear ramp to zero_drift_loss_weight")
    parser.add_argument("--zero_drift_loss_ramp_epochs", type=int, default=0,
                        help="V52 S2: epochs to ramp zero_drift weight from start to final (0=no ramp)")
    parser.add_argument("--zero_drift_loss_start_epoch", type=int, default=5,
                        help="V51: epoch to start zero_drift_loss")
    parser.add_argument("--zero_drift_dedicated_ratio", type=float, default=0.0,
                        help="V51: force init=gt on this fraction of batches (in addition to zero_perturbation_prob)")
    parser.add_argument("--inject_recovery_loss_weight", type=float, default=0.0,
                        help="V52: extra loss on Fixed-Inject batches (all RPY inject), "
                             "penalize out_err vs GT (aligns with gdiag Genuine Recovery)")
    parser.add_argument("--inject_recovery_loss_start_epoch", type=int, default=15,
                        help="V52: epoch to start inject_recovery_loss")
    parser.add_argument("--inject_recovery_magnitude_deg", type=float, default=2.0,
                        help="V52: Fixed-Inject magnitude on all RPY axes (match gdiag inject_deg)")
    parser.add_argument("--inject_recovery_dedicated_ratio", type=float, default=0.0,
                        help="V52: fraction of batches with gt + fixed all-axis inject")
    parser.add_argument("--use_mgda", type=int, default=0,
                        help="V52d/V53e: MGDA minimum-norm grad for pose+zd(+inject) tasks")
    parser.add_argument("--mgda_start_epoch", type=int, default=5,
                        help="Epoch to start MGDA multi-task backward")
    parser.add_argument("--mgda_include_inject", type=int, default=1,
                        help="Include inject_recovery as MGDA task (else fold into pose)")
    parser.add_argument("--mgda_include_photo", type=int, default=1,
                        help="V54: include LSP as MGDA photo task")
    # === V54: TLC-inspired Photo-Geometric Joint Alignment ===
    parser.add_argument("--use_lsp_loss", type=int, default=0,
                        help="V54: LiDAR Splat Photo Loss (train-only)")
    parser.add_argument("--lsp_weight", type=float, default=0.35,
                        help="V54: final LSP loss weight")
    parser.add_argument("--lsp_weight_start", type=float, default=0.0,
                        help="V54: LSP weight at lsp_start_epoch")
    parser.add_argument("--lsp_ramp_epochs", type=int, default=15,
                        help="V54: epochs to ramp LSP weight to final")
    parser.add_argument("--lsp_start_epoch", type=int, default=10,
                        help="V54: epoch to enable LSP (aligns with pose release)")
    parser.add_argument("--lsp_lambda_ssim", type=float, default=0.2,
                        help="V54: SSIM fraction in LSP (TLC lambda_dssim)")
    parser.add_argument("--lsp_max_points", type=int, default=4096,
                        help="V54: max LiDAR points for LSP per sample")
    parser.add_argument("--lsp_downsample", type=int, default=4,
                        help="V54: image downsample factor for LSP splat")
    parser.add_argument("--lsp_loss_clip", type=float, default=2.0,
                        help="V54: clip LSP total loss to this max (0=disable)")
    parser.add_argument("--lsp_min_valid_ratio", type=float, default=0.02,
                        help="V54: skip LSP when valid point ratio below this threshold")
    parser.add_argument("--rig_consistency_weight", type=float, default=0.0,
                        help="V54: rig consistency loss weight (TLC use_rig)")
    parser.add_argument("--rig_consistency_start_epoch", type=int, default=5,
                        help="V54: epoch to start rig consistency loss")
    parser.add_argument("--pose_release_epoch", type=int, default=0,
                        help="V54 PRS: epoch to release pose-head LR (0=disabled)")
    parser.add_argument("--pose_release_joint_epoch", type=int, default=50,
                        help="V54 PRS: epoch to reduce corr-path LR in joint phase")
    parser.add_argument("--pose_release_corr_lr_scale_joint", type=float, default=0.5,
                        help="V54 PRS: corr-path LR scale after joint_epoch")
    parser.add_argument("--freeze_backbone_epoch", type=int, default=999,
                        help="V54b refine: freeze backbone LR from this epoch (999=off)")
    # === V53: Dual-Path Pose Head ===
    parser.add_argument("--use_dp_head", type=int, default=0,
                        help="V53: enable Dual-Path Pose Head (Bias + Recovery + Router)")
    parser.add_argument("--dp_gate_deg", type=float, default=1.5,
                        help="V53: router target threshold (degrees) for recovery path")
    parser.add_argument("--route_loss_weight", type=float, default=0.0,
                        help="V53: BCE weight for magnitude router supervision")
    parser.add_argument("--route_zd_penalty_weight", type=float, default=0.0,
                        help="V54d: penalize high route_w when init_err <= dp_gate_deg (anti-collapse)")
    parser.add_argument("--dp_route_input_mode", type=str, default="gt_or_pred",
                        help="V71: router input mode: gt_or_pred (legacy) or mag_only (deploy-consistent)")
    parser.add_argument("--dp_jacg_input_mode", type=str, default="gt_or_pred",
                        help="V71: JACG input mode: gt_or_pred (legacy) or mag_only")
    parser.add_argument("--dp_recovery_quat_loss_weight", type=float, default=0.0,
                        help="V71: direct correction-quaternion supervision weight for DP recovery branch")
    parser.add_argument("--dp_train_hard_route", type=int, default=0,
                        help="V71b: use GT hard gate for DP output during training while router still learns BCE")
    parser.add_argument("--use_jacg", type=int, default=1,
                        help="V53: Jacobian-Aware Correction Gain on recovery path")
    parser.add_argument("--jacg_hidden_dim", type=int, default=64,
                        help="V53: hidden dim for router / JACG MLPs")
    parser.add_argument("--bias_path_in_norm", type=int, default=1,
                        help="V53: LayerNorm on bias path tokens")
    parser.add_argument("--recovery_path_layer_norm", type=int, default=1,
                        help="V53: LayerNorm on recovery path tokens (reserved)")
    parser.add_argument("--use_hard_route_eval", type=int, default=0,
                        help="V55: use init_err hard gate at inference for DP-Head "
                             "(init<=gate走BiasPath, init>gate走RecoveryPath)")
    parser.add_argument("--use_adir", type=int, default=0,
                        help="V53b: Axis-Decoupled Iterative Refinement after pose head")
    parser.add_argument("--adir_steps", type=int, default=2,
                        help="V53b: ADIR refinement iterations")
    parser.add_argument("--adir_max_step_deg", type=float, default=1.0,
                        help="V53b: max per-axis correction per ADIR step (degrees)")
    parser.add_argument("--adir_train_only", type=int, default=0,
                        help="V72: freeze pretrained base model and train only the identity-initialized ADIR adapter")
    parser.add_argument("--trainable_prefixes", type=str, default="",
                        help="V73: comma-separated module/name prefixes to keep trainable after loading "
                             "pretrain, e.g. 'corr_head,pose_query_init,adir_refiner'. Empty=default.")
    parser.add_argument("--overcorrection_penalty", type=float, default=0.0,
                        help="V47: extra weight multiplier on loss when model overcorrects "
                             "(prediction error > initial perturbation). 0=disabled, 2.0=recommended")
    parser.add_argument("--use_magnitude_head", type=int, default=0,
                        help="V46: enable magnitude estimation head in PoseQueryDecoder (0=disabled, 1=enabled)")
    parser.add_argument("--magnitude_loss_weight", type=float, default=0.3,
                        help="V46: weight for magnitude estimation auxiliary loss (default: 0.3)")
    parser.add_argument("--augment_pc_jitter", type=float, default=0.0,
                        help="Point cloud Gaussian jitter sigma in meters (0=disabled)")
    parser.add_argument("--augment_pc_dropout", type=float, default=0.0,
                        help="Point cloud random dropout ratio (0=disabled)")
    parser.add_argument("--augment_color_jitter", type=float, default=0.0,
                        help="Image color jitter strength (0=disabled)")
    parser.add_argument("--augment_intrinsic", type=float, default=0.0,
                        help="Camera intrinsic augmentation strength: max relative deviation for fx/fy (e.g. 0.05=±5%%, 0=disabled)")
    parser.add_argument("--augment_intrinsic_cxcy", type=float, default=0.0,
                        help="Separate cx/cy augmentation strength (0=use same as --augment_intrinsic)")
    parser.add_argument("--augment_pitch_flip_prob", type=float, default=0.0,
                        help="GT pitch perturbation probability (0=disabled). "
                             "Random Y-axis (pitch) rotation in [-max_deg, +max_deg].")
    parser.add_argument("--augment_pitch_flip_max_deg", type=float, default=2.0,
                        help="Max rotation angle (degrees) for pitch perturbation")
    parser.add_argument("--augment_pitch_sign_flip_prob", type=float, default=0.0,
                        help="GT pitch sign flip probability (0=disabled). "
                             "Precisely negates the pitch angle to simulate "
                             "reversed camera mounting (e.g. Seq02/06).")
    # === v32: T_init Invariance Training ===
    parser.add_argument("--tinit_dropout_prob", type=float, default=0.0,
                        help="v32: Probability of replacing T_init with random rotation (0=disabled). "
                             "Forces model to rely on visual features instead of T_init shortcut.")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.0,
                        help="v32: Weight for consistency loss (0=disabled). Penalizes prediction "
                             "difference when same scene gets two different T_init perturbations.")
    parser.add_argument("--consistency_loss_start_epoch", type=int, default=0,
                        help="v32: Epoch to start applying consistency loss.")
    parser.add_argument("--conditional_consistency_threshold", type=float, default=0.0,
                        help="v49: Only apply consistency loss when angular distance between "
                             "two T_init paths is below this threshold (degrees). "
                             "0=disabled (always apply). Recommended: 1.0-2.0")
    parser.add_argument("--quat_bias_reset_alpha", type=float, default=0.0,
                        help="v49: At each cosine restart cycle boundary, blend quat_head bias "
                             "toward identity [1,0,0,0] by this fraction. "
                             "0=disabled, 0.5=half-reset, 1.0=full-reset. Recommended: 0.3-0.5")
    parser.add_argument("--decoder_pool_mode", type=str, default="mean",
                        choices=["mean", "attention"],
                        help="v49: Query aggregation in PoseQueryDecoder. "
                             "'mean'=simple average (v48 default), "
                             "'attention'=learned attention pooling (v49 enhancement)")
    parser.add_argument("--progressive_angle_start", type=float, default=0.0,
                        help="v32: Starting angle_range for progressive curriculum (0=disabled, use fixed angle_range_deg).")
    parser.add_argument("--progressive_angle_end", type=float, default=0.0,
                        help="v32: Ending angle_range for progressive curriculum.")
    parser.add_argument("--progressive_warmup_epochs", type=int, default=100,
                        help="v32: Number of epochs to ramp from progressive_angle_start to progressive_angle_end.")
    parser.add_argument("--ema_consistency", type=int, default=0,
                        help="v32.1: Use EMA target network for consistency loss (0=disabled, 1=enabled)")
    parser.add_argument("--ema_decay", type=float, default=0.996,
                        help="v32.1: EMA decay rate for target network (default: 0.996)")
    parser.add_argument("--continuous_tinit_noise", type=int, default=0,
                        help="v32.1: Replace binary T_init dropout with continuous noise schedule "
                             "(0=disabled/use binary dropout, 1=enabled)")
    parser.add_argument("--continuous_noise_max_deg", type=float, default=30.0,
                        help="v32.1: Max extra rotation noise in degrees for continuous schedule (default: 30)")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Early stopping patience in epochs (0=disabled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility (default: 42)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (1=disabled, >1=accumulate N micro-batches per optimizer step)")
    parser.add_argument("--ddp_auto_scale", type=int, default=1,
                        help="Auto-scale hyperparams for large-scale DDP. "
                             "0=disabled, 1=speedup (scale LR+warmup, keep epochs → actual speedup), "
                             "2=preserve_steps (scale epochs to match total steps, no speedup)")
    parser.add_argument("--ddp_reference_gpus", type=int, default=8,
                        help="Reference GPU count the hyperparameters were tuned for (default: 8)")
    parser.add_argument("--max_scaled_lr", type=float, default=4e-4,
                        help="Max learning rate after DDP auto-scaling (0=unlimited). "
                             "Recommended: 4e-4 based on V16 LR ablation (2e-4@8GPU ≈ 4e-4@128GPU)")
    parser.add_argument("--data_balance", type=int, default=0,
                        help="Per-sequence balanced sampling mode (0=off, 1=full, 2=sqrt). "
                             "1: each sequence equal probability (1/N). "
                             "2: softer sqrt(1/count) balance, less oversampling of small seqs.")
    parser.add_argument("--seq_weight_overrides", type=str, default="",
                        help="Optional per-sequence sampler multipliers, e.g. '02:1.4,03:2.2'. "
                             "Applied on top of --data_balance and re-normalized.")
    return parser.parse_args()

def crop_and_resize(item, size, intrinsics, crop=True):
    """
    图像预处理: 缩放 → 更新内参
    
    Args:
        item: PIL Image 或 numpy array
        size: (width, height) 目标尺寸
        intrinsics: (3, 3) 原始相机内参矩阵
        crop: 是否裁剪中间区域
    
    Returns:
        resized: (H, W, 3) BGR图像
        new_intrinsics: (3, 3) 调整后的内参矩阵
    """
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


def get_target_size(use_custom_dataset, target_width=None, target_height=None):
    """
    根据数据集类型和参数获取目标图像尺寸
    
    Args:
        use_custom_dataset: 是否使用自定义数据集
        target_width: 用户指定的宽度 (可选)
        target_height: 用户指定的高度 (可选)
    
    Returns:
        (width, height) 元组
    
    预设尺寸说明:
        - KITTI (1242x375, 宽高比3.31): 704x256 (宽高比2.75)
        - 自定义4K (3840x2160, 宽高比1.78): 640x360 (宽高比1.78, 保持16:9)
        
    注意: 尺寸应为8的倍数，以匹配模型的下采样率
    """
    if target_width is not None and target_height is not None:
        # 用户显式指定尺寸
        return (target_width, target_height)
    
    if use_custom_dataset:
        # 自定义数据集 (如 B26A 4K: 3840x2160)
        # 保持 16:9 宽高比，使用 640x360
        default_width = 640
        default_height = 360
    else:
        # KITTI 数据集 (1242x375)
        # 原始配置
        default_width = 704
        default_height = 256
    
    return (target_width or default_width, target_height or default_height)


class PreprocessedDataset(Dataset):
    """Wraps a dataset to perform image preprocessing (resize) in worker processes."""
    def __init__(self, dataset, target_size, crop=False):
        self.dataset = dataset
        self.target_size = target_size
        self.crop = crop

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        result = self.dataset[idx]
        if result is None:
            return None
        img, pcd, gt_transform, intrinsic = result[0], result[1], result[2], result[3]
        extra = result[4:] if len(result) > 4 else ()
        if isinstance(img, np.ndarray) and img.shape[:2] == (self.target_size[1], self.target_size[0]):
            return (img, pcd, gt_transform, intrinsic) + extra
        resized_img, new_intrinsic = crop_and_resize(img, self.target_size, intrinsic, self.crop)
        return (resized_img, pcd, gt_transform, new_intrinsic) + extra


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    imgs = [item[0] for item in batch]
    gt_T_to_camera = [item[2] for item in batch]
    intrinsics = [item[3] for item in batch]

    pcs = []
    masks = []
    max_num_points = max(item[1].shape[0] for item in batch)
    for item in batch:
        pc = item[1]
        masks.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(max_num_points - pc.shape[0])], axis=0))
        if pc.shape[0] < max_num_points:
            pc = np.concatenate([pc, np.full((max_num_points - pc.shape[0], pc.shape[1]), 999999)], axis=0)
        pcs.append(pc)

    has_domain_ids = len(batch[0]) > 4
    if has_domain_ids:
        domain_ids = [item[4] for item in batch]
        return imgs, pcs, masks, gt_T_to_camera, intrinsics, domain_ids
    return imgs, pcs, masks, gt_T_to_camera, intrinsics

def _build_ckpt_metadata(model, args):
    """Build model_config + env metadata for checkpoint (checkpoint-first eval)."""
    raw_model = model.module if hasattr(model, 'module') else model
    meta = {
        'model_config': {
            'rotation_only': raw_model.rotation_only,
            'intrinsic_input': getattr(raw_model, 'intrinsic_input', False),
            'deformable': getattr(raw_model, 'deformable', False),
            'bev_encoder_use': getattr(raw_model, 'bev_encoder_use', True),
            'bev_pool_factor': getattr(raw_model, 'bev_pool_factor', 0),
        },
        'env': {
            'torch_version': torch.__version__,
            'cuda_version': getattr(torch.version, 'cuda', 'N/A'),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        },
    }
    return meta


def main():
    args = parse_args()
    
    use_ddp, rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    set_seed(args.seed, rank=rank)

    if is_main:
        os.makedirs(args.log_dir, exist_ok=True)
        tprint_setup(args.log_dir)
        tprint(f"训练启动, 参数配置:")
        tprint(args)
        from bev_settings import xbound, ybound, zbound, sparse_shape, DATASET_TYPE
        _nx_z = int((zbound[1] - zbound[0]) / zbound[2])
        tprint(f"[BEV Settings] 数据集类型: {DATASET_TYPE}")
        tprint(f"[BEV Settings] xbound: {xbound}, ybound: {ybound}")
        tprint(f"[BEV Settings] zbound: {zbound} → {_nx_z}个Z体素 (步长{zbound[2]}m)")
        tprint(f"[BEV Settings] sparse_shape: {sparse_shape}")
    if use_ddp and is_main:
        tprint(f"DDP enabled: {world_size} GPUs, rank={rank}, local_rank={local_rank}")

    if not is_main:
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.WARNING)
    
    num_epochs = args.num_epochs
    dataset_root = args.dataset_root
    log_dir = args.log_dir
    if args.label is not None:
        log_dir = os.path.join(log_dir, args.label)
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{log_dir}/{current_time}"
    ckpt_save_dir = os.path.join(log_dir, "checkpoint")
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_save_dir, exist_ok=True)
    if use_ddp:
        dist.barrier()
    
    if is_main:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        bev_calib_dir = os.path.join(parent_dir, 'kitti-bev-calib')
        dest_dir = os.path.join(log_dir, 'kitti-bev-calib')
        try:
            shutil.copytree(bev_calib_dir, dest_dir, dirs_exist_ok=True, 
                           ignore=shutil.ignore_patterns('logs', '__pycache__', '*.pyc', '.git*'))
        except Exception as e:
            tprint(f"警告: 复制源代码失败: {e}")
    
    writer = SummaryWriter(log_dir) if is_main else None
    
    # 预先计算目标图像尺寸（供 CustomDataset 查找预处理图像）
    target_size = get_target_size(
        use_custom_dataset=args.use_custom_dataset > 0,
        target_width=args.target_width,
        target_height=args.target_height
    )
    
    # 选择数据集类型 (capture_prints suppresses worker stdout, writes master output to train.log)
    with capture_prints(is_main):
        if args.use_custom_dataset:
            dataset = CustomDataset(dataset_root, target_size=target_size,
                                    max_frames_per_seq=args.max_frames_per_seq,
                                    sample_step=args.sample_step,
                                    pose_aware_sampling=args.pose_aware_sampling,
                                    poses_dir=args.poses_dir or None,
                                    return_seq_id=(
                                        args.domain_adversarial > 0
                                        or getattr(args, 'rig_consistency_weight', 0.0) > 0))
        else:
            if is_main:
                print("使用 KittiDataset")
            dataset = KittiDataset(dataset_root)

    # 数据利用率校验 (only on rank 0)
    if args.validate_data > 0 and is_main:
        tprint("=" * 60)
        tprint("开始数据利用率校验...")
        tprint("=" * 60)
        
        with capture_prints(is_main):
            validation_result = dataset.validate_data_utilization(
                sample_ratio=args.validate_sample_ratio,
                min_utilization=args.min_point_utilization,
                min_valid_ratio=args.min_valid_ratio,
                verbose=True
            )
        
        validation_log_path = os.path.join(log_dir, "data_validation.txt")
        with open(validation_log_path, 'w') as f:
            f.write("数据利用率校验结果\n")
            f.write("="*60 + "\n")
            for key, value in validation_result.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        tprint(f"验证结果已保存到: {validation_log_path}")
        
        if not validation_result['passed']:
            tprint("错误: 数据利用率验证未通过，退出训练！")
            tprint("   可以通过以下方式解决：")
            tprint("   1. 检查 bev_settings.py 中的体素化范围配置是否与数据集匹配")
            tprint("   2. 调整 --min_point_utilization 或 --min_valid_ratio 阈值")
            tprint("   4. 使用 --validate_data=0 跳过验证（不推荐）")
            exit(1)
    elif is_main:
        tprint("跳过数据利用率校验 (--validate_data=0)")
    if use_ddp:
        dist.barrier()

    if is_main:
        tprint(f"目标图像尺寸: {target_size[0]}x{target_size[1]} (宽x高)")
        if args.use_custom_dataset > 0:
            tprint("   (自定义数据集模式,保持16:9宽高比)")
        else:
            tprint("   (KITTI数据集模式)")
    
    if args.use_custom_dataset and hasattr(dataset, 'all_files'):
        train_dataset, val_dataset, split_stats = stratified_split_by_sequence(
            dataset, train_ratio=0.8, seed=114514
        )
        if is_main:
            tprint("数据集按Sequence分层划分 (每个Sequence独立80/20):")
            tprint(f"  {'Seq':<6} {'Train':>7} {'Val':>7} {'Total':>7}  {'Train%':>6}")
            tprint(f"  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
            for seq_id, (n_tr, n_va, n_tot) in split_stats.items():
                pct = n_tr / n_tot * 100 if n_tot > 0 else 0
                tprint(f"  {seq_id:<6} {n_tr:>7} {n_va:>7} {n_tot:>7}  {pct:>5.1f}%")
            tprint(f"  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
            total_tr = sum(s[0] for s in split_stats.values())
            total_va = sum(s[1] for s in split_stats.values())
            total_all = sum(s[2] for s in split_stats.values())
            tprint(f"  {'合计':<5} {total_tr:>7} {total_va:>7} {total_all:>7}  {total_tr/total_all*100:>5.1f}%")
    else:
        generator = torch.Generator().manual_seed(114514)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=generator
        )

    train_dataset = PreprocessedDataset(train_dataset, target_size, crop=False)
    val_dataset = PreprocessedDataset(val_dataset, target_size, crop=False)

    # ── DDP Auto-Scaling ──────────────────────────────────────────────────
    # Mode 0: disabled — no scaling, use original hyperparams as-is
    # Mode 1: speedup (DEFAULT) — scale LR + warmup only, keep epochs same → ACTUAL SPEEDUP
    #         Based on large-batch training theory (Goyal et al. 2017):
    #         larger batch → more stable gradients → can use higher LR → same epochs, fewer steps
    # Mode 2: preserve_steps — scale epochs to match total steps exactly (no speedup)
    ddp_scaled = False
    if use_ddp and args.ddp_auto_scale > 0:
        ref_gpus = args.ddp_reference_gpus
        if world_size > ref_gpus:
            train_size = len(train_dataset)
            ref_steps = train_size // (args.batch_size * ref_gpus)
            actual_steps = train_size // (args.batch_size * world_size)
            actual_steps = max(actual_steps, 1)

            if actual_steps < ref_steps:
                ddp_scaled = True
                step_ratio = ref_steps / actual_steps
                gpu_ratio = world_size / ref_gpus
                scale_mode = args.ddp_auto_scale

                orig = {
                    'num_epochs': num_epochs,
                    'lr': args.lr,
                    'warmup_epochs': args.warmup_epochs,
                    'step_size': args.step_size,
                    'cosine_T0': args.cosine_T0,
                    'early_stopping_patience': args.early_stopping_patience,
                    'save_ckpt_per_epoches': args.save_ckpt_per_epoches,
                    'eval_epoches': args.eval_epoches,
                }

                lr_mult = math.sqrt(gpu_ratio)
                max_lr = getattr(args, 'max_scaled_lr', 0)

                if scale_mode == 1:
                    mode_name = "⚡ 加速模式 (speedup)"
                    warmup_scaled = max(orig['warmup_epochs'],
                                        min(int(orig['warmup_epochs'] * math.sqrt(step_ratio)),
                                            int(num_epochs * 0.1)))
                    scaled_lr = orig['lr'] * lr_mult
                    if max_lr > 0 and scaled_lr > max_lr:
                        lr_mult = max_lr / orig['lr']
                        if is_main:
                            tprint(f"   ⚠️  LR cap: {scaled_lr:.2e} → {max_lr:.2e} (max_scaled_lr={max_lr:.0e})")
                        scaled_lr = max_lr
                    args.lr = scaled_lr
                    args.warmup_epochs = warmup_scaled
                    # step_size, cosine_T0, patience, save_ckpt, eval: keep same (epoch-based schedule)

                elif scale_mode == 2:
                    mode_name = "🔒 保步模式 (preserve_steps)"
                    num_epochs = int(num_epochs * step_ratio)
                    scaled_lr = orig['lr'] * lr_mult
                    if max_lr > 0 and scaled_lr > max_lr:
                        if is_main:
                            tprint(f"   ⚠️  LR cap: {scaled_lr:.2e} → {max_lr:.2e} (max_scaled_lr={max_lr:.0e})")
                        scaled_lr = max_lr
                    args.lr = scaled_lr
                    args.warmup_epochs = max(1, int(orig['warmup_epochs'] * step_ratio))
                    args.step_size = max(1, int(orig['step_size'] * step_ratio))
                    args.cosine_T0 = max(1, int(orig['cosine_T0'] * step_ratio))
                    if orig['early_stopping_patience'] > 0:
                        args.early_stopping_patience = max(1, int(orig['early_stopping_patience'] * step_ratio))
                    if orig['save_ckpt_per_epoches'] > 0:
                        args.save_ckpt_per_epoches = max(1, int(orig['save_ckpt_per_epoches'] * step_ratio))
                    args.eval_epoches = max(1, int(orig['eval_epoches'] * step_ratio))

                if is_main:
                    scaled_total = num_epochs * actual_steps
                    ref_total = orig['num_epochs'] * ref_steps
                    speedup = ref_total / scaled_total if scaled_total > 0 else 0
                    tprint("=" * 80)
                    tprint(f"🔄 DDP Auto-Scaling: {mode_name}")
                    tprint(f"   {ref_gpus} GPUs → {world_size} GPUs (x{gpu_ratio:.0f})")
                    tprint(f"   训练集: {train_size}, batch/GPU: {args.batch_size}, "
                           f"全局batch: {args.batch_size * world_size}")
                    tprint(f"   步数/epoch: {ref_steps} ({ref_gpus}GPU) → {actual_steps} ({world_size}GPU)")
                    tprint(f"   {'参数':<25} {'原始':>12} {'缩放后':>12}")
                    tprint(f"   {'─'*25} {'─'*12} {'─'*12}")
                    tprint(f"   {'Epochs':<25} {orig['num_epochs']:>12} {num_epochs:>12}")
                    lr_note = f"√{gpu_ratio:.0f}x"
                    if max_lr > 0 and orig['lr'] * math.sqrt(gpu_ratio) > max_lr:
                        lr_note += f", capped at {max_lr:.0e}"
                    tprint(f"   {'Learning Rate':<25} {orig['lr']:>12.2e} {args.lr:>12.2e} ({lr_note})")
                    tprint(f"   {'Warmup Epochs':<25} {orig['warmup_epochs']:>12} {args.warmup_epochs:>12}")
                    if args.step_size != orig['step_size']:
                        tprint(f"   {'LR Step Size':<25} {orig['step_size']:>12} {args.step_size:>12}")
                    if num_epochs != orig['num_epochs']:
                        tprint(f"   总步数: {scaled_total} (参考: {ref_total}, 偏差: "
                               f"{abs(scaled_total-ref_total)/ref_total*100:.1f}%)")
                    else:
                        tprint(f"   总步数: {scaled_total} (参考: {ref_total})")
                        tprint(f"   ⚡ 预期加速: ~{speedup:.1f}x "
                               f"(每epoch {actual_steps}步×{num_epochs}ep, 大batch梯度更稳定)")
                    if actual_steps <= 2:
                        tprint(f"   ⚠️  每epoch仅{actual_steps}步, 建议减少到 "
                               f"{ref_gpus}-{min(world_size, ref_gpus*4)} GPUs")
                    tprint(f"   💡 --ddp_auto_scale 0=禁用, 1=加速(默认), 2=保步")
                    tprint("=" * 80)

    num_workers = min(16, os.cpu_count() or 4)
    if is_main:
        tprint(f"DataLoader: num_workers={num_workers}, pin_memory=True, persistent_workers=True")

    balance_mode = args.data_balance > 0
    train_sampler = None
    _balance_active = False

    raw_ds = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    orig_ds = raw_ds.dataset if hasattr(raw_ds, 'dataset') else raw_ds
    can_balance = balance_mode and hasattr(orig_ds, 'all_files')

    if can_balance:
        balance_mode_int = args.data_balance
        seq_weight_overrides = parse_seq_weight_overrides(getattr(args, 'seq_weight_overrides', ''))
        weights, bal_counts = build_balanced_weights(
            raw_ds, orig_ds, mode=balance_mode_int, seq_weight_overrides=seq_weight_overrides)
        if use_ddp:
            train_sampler = DistributedBalancedSampler(
                weights, num_samples=len(weights), seed=args.seed)
        else:
            train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        _balance_active = True
        mode_name = {1: "full(1/N)", 2: "sqrt(1/√count)"}.get(balance_mode_int, "unknown")
        if is_main:
            tprint(f"📊 数据均衡采样已启用: mode={mode_name}, {len(bal_counts)} seqs, {'DDP' if use_ddp else '单机'}")
            if seq_weight_overrides:
                tprint(f"   seq_weight_overrides: {seq_weight_overrides}")
            offset = 0
            for sid, cnt in sorted(bal_counts.items()):
                w_sample = weights[offset]
                effective_ratio = w_sample * cnt * len(weights) * 100
                tprint(f"   {sid}: {cnt} samples, eff_ratio={effective_ratio:.1f}%")
                offset += cnt
    elif use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    if is_main and balance_mode:
        if _balance_active:
            sampler_type = "DistributedBalancedSampler" if use_ddp else "WeightedRandomSampler"
            tprint(f"   均衡采样: 已启用 ({sampler_type})")
        else:
            tprint(f"   均衡采样: 数据集无 all_files 属性, 降级为默认采样")
    
    val_sampler = None
    if use_ddp:
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_worker_init_fn,
    )

    val_idx_to_seq = None
    if args.enable_medw_eval > 0:
        val_idx_to_seq = _build_val_idx_to_seq(val_dataset, dataset)
        if is_main:
            tprint(f"MEDW eval: reuse val split ({len(val_idx_to_seq)} samples), "
                   f"window={args.medw_eval_max_frames} (zero extra forward)")
            if use_ddp:
                tprint(f"  Val DDP: {dist.get_world_size()} GPUs shard val forward (~{dist.get_world_size()}× faster)")

    deformable_choise = args.deformable > 0
    bev_encoder_choise = args.bev_encoder > 0
    xyz_only_choise = args.xyz_only > 0
    rotation_only = args.rotation_only > 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    img_shape = (target_size[1], target_size[0])
    if is_main:
        tprint(f"网络输入尺寸 (H, W): {img_shape}")
        tprint(f"优化模式: {'仅旋转 (rotation only)' if rotation_only else '旋转+平移 (translation+rotation)'}")
    
    enable_axis_loss = args.enable_axis_loss > 0
    use_geodesic_loss = args.use_geodesic_loss > 0
    if args.use_balanced_axis_loss > 0 and not enable_axis_loss:
        if is_main:
            tprint("WARNING: --use_balanced_axis_loss requires --enable_axis_loss; "
                   "forcing enable_axis_loss=True")
        enable_axis_loss = True
    use_mlp_head = args.use_mlp_head > 0
    axis_weights_tuple = tuple(float(x) for x in args.axis_weights.split(','))
    _num_domains = args.num_domains
    if args.domain_adversarial > 0 and _num_domains == 0:
        raw_ds = dataset
        if hasattr(raw_ds, 'num_domains'):
            _num_domains = raw_ds.num_domains
        else:
            _num_domains = 21
        if is_main:
            tprint(f"DANN: auto-detected {_num_domains} domains from dataset")
    use_foundation_depth = args.use_foundation_depth > 0
    fd_mode = args.fd_mode if use_foundation_depth else "replace"
    if is_main and use_foundation_depth:
        tprint(f"Foundation Depth: 启用 (model={args.depth_model_type}, mode={fd_mode})")
    with capture_prints(is_main):
        from hybrid_triple_calib import build_calib_model
        model = build_calib_model(
            args, device, img_shape, rotation_only, is_main=is_main, tprint=tprint)

    if args.pretrain_ckpt is not None:
        state_dict = torch.load(args.pretrain_ckpt, map_location=device)
        ckpt_sd = state_dict['model_state_dict']
        model_sd = model.state_dict()
        # V40 HybridPoseHead: V39 ckpt uses fusion_head.head.*, P1c uses refine_head.head.*
        remapped_sd = {}
        remap_count = 0
        for k, v in ckpt_sd.items():
            if k.startswith('fusion_head.head.') and k not in model_sd:
                alt = k.replace('fusion_head.head.', 'fusion_head.refine_head.head.', 1)
                if alt in model_sd and model_sd[alt].shape == v.shape:
                    remapped_sd[alt] = v
                    remap_count += 1
                    continue
            remapped_sd[k] = v
        if remap_count and is_main:
            tprint(f"  Pretrain remap: fusion_head.head.* -> refine_head.head.* ({remap_count} keys)")
        ckpt_sd = remapped_sd
        filtered_sd = {}
        skipped_shape = []
        partial_loaded = []
        for k, v in ckpt_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered_sd[k] = v
            elif k in model_sd:
                m_shape = model_sd[k].shape
                if len(v.shape) == 1 and len(m_shape) == 1 and m_shape[0] < v.shape[0] and 'feat_in' in k:
                    filtered_sd[k] = v[:m_shape[0]].clone()
                    partial_loaded.append(f"{k}: ckpt[:{m_shape[0]}]/{v.shape[0]}")
                elif len(v.shape) == 2 and len(m_shape) == 2 and m_shape[0] < v.shape[0] and v.shape[1] == m_shape[1] and 'feat_in' in k:
                    filtered_sd[k] = v[:m_shape[0]].clone()
                    partial_loaded.append(f"{k}: ckpt[:{m_shape[0]},:]{v.shape}")
                elif (len(v.shape) == 2 and len(m_shape) == 2
                      and m_shape[0] == v.shape[0] and m_shape[1] > v.shape[1]
                      and (k.endswith('rotation_pred.weight') or k.endswith('translation_pred.weight'))):
                    expanded = model_sd[k].clone()
                    expanded[:, :v.shape[1]] = v
                    expanded[:, v.shape[1]:] = 0
                    filtered_sd[k] = expanded
                    partial_loaded.append(
                        f"{k}: expand input {list(v.shape)}->{list(m_shape)} (new cols zero)")
                elif len(v.shape) == 2 and len(m_shape) == 2 and 'pitch_branch.img_encoder' in k and m_shape[1] > v.shape[1] and m_shape[1] % v.shape[1] == 0:
                    n_rep = m_shape[1] // v.shape[1]
                    filtered_sd[k] = v.repeat(1, n_rep)[:m_shape[0], :m_shape[1]].clone()
                    partial_loaded.append(f"{k}: tile {n_rep}x ({list(v.shape)}->{list(m_shape)})")
                else:
                    skipped_shape.append(f"{k}: ckpt={list(v.shape)} vs model={list(m_shape)}")
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        if is_main:
            tprint(f"Load pretrain model from {args.pretrain_ckpt}")
            tprint(f"  Loaded {len(filtered_sd)}/{len(model_sd)} model keys from ckpt")
            if partial_loaded:
                tprint(f"  Partial GIN weight transfer: {partial_loaded}")
            if skipped_shape:
                tprint(f"  Shape mismatch (skipped): {skipped_shape}")
            if missing:
                tprint(f"  Missing keys (new layers): {len(missing)} keys")
            if unexpected:
                tprint(f"  Unexpected keys (skipped): {unexpected}")
    
    if args.compile > 0:
        try:
            model = torch.compile(model)
            if is_main:
                tprint("torch.compile enabled")
        except Exception as e:
            if is_main:
                tprint(f"torch.compile failed, falling back to eager mode: {e}")
    
    trainable_prefixes = [
        p.strip() for p in str(getattr(args, 'trainable_prefixes', '') or '').split(',')
        if p.strip()
    ]
    if trainable_prefixes:
        trainable, frozen = 0, 0
        matched_prefixes = set()
        for name, param in model.named_parameters():
            matched = False
            for prefix in trainable_prefixes:
                if name == prefix or name.startswith(prefix + '.'):
                    matched = True
                    matched_prefixes.add(prefix)
                    break
            param.requires_grad_(matched)
            if matched:
                trainable += param.numel()
            else:
                frozen += param.numel()
        if is_main:
            tprint(f"V73 selective training: prefixes={trainable_prefixes}, "
                   f"trainable={trainable:,} params, frozen={frozen:,} params")
            missing_prefixes = sorted(set(trainable_prefixes) - matched_prefixes)
            if missing_prefixes:
                tprint(f"  WARNING: trainable_prefixes not matched: {missing_prefixes}")

    if getattr(args, 'adir_train_only', 0) > 0:
        trainable, frozen = 0, 0
        for name, param in model.named_parameters():
            is_adir = name.startswith('adir_refiner.')
            param.requires_grad_(is_adir)
            if is_adir:
                trainable += param.numel()
            else:
                frozen += param.numel()
        if is_main:
            tprint(f"V72 ADIR adapter-only training: trainable={trainable:,} params, "
                   f"frozen={frozen:,} params")

    if use_ddp:
        need_find_unused = (getattr(args, 'cam_drop_prob', 0) > 0
                            or getattr(args, 'use_pitch_branch', 0) > 0
                            or getattr(args, 'use_contrastive_extrinsic', 0) > 0
                            or getattr(args, 'domain_adversarial', 0) > 0
                            or getattr(args, 'cam2bev_mode', 'lss') == 'query'
                            or getattr(args, 'backbone_type', 'swin') == 'dinov2'
                            or getattr(args, 'fusion_backend', 'bev') == 'geo_match_proj'
                            or getattr(args, 'fusion_backend', 'bev') == 'cf_bev_r')
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=need_find_unused)
        if is_main:
            tprint(f"Model wrapped with DistributedDataParallel on {world_size} GPUs "
                   f"(find_unused_parameters={need_find_unused})")
    
    raw_model = model.module if use_ddp else model

    if is_main:
        tprint(f"The weight decay is: {args.wd}")
        tprint(f"The initial learning rate is: {args.lr}")

    backbone_params = []
    bev_branch_params = []
    head_params = []
    corr_path_params = []
    pose_head_params = []
    _use_prs = int(getattr(args, 'pose_release_epoch', 0)) > 0
    _backbone_param_set = set()
    _bev_branch_param_set = set()
    _corr_path_param_set = set()
    _pose_head_param_set = set()
    _module_param_map = {}
    _layer_wise_groups = {}
    _HTCN_BEV_MODULES = ('conv_fuser', 'transformer', 'bev_encoder', 'pose_embed', 'pc_branch')
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(m + '.') or name == m for m in _HTCN_BEV_MODULES):
            bev_branch_params.append(param)
            _bev_branch_param_set.add(id(param))
        elif 'img_branch' in name or name.startswith('img_encoder.'):
            backbone_params.append(param)
            _backbone_param_set.add(id(param))
            if args.layer_wise_lr_decay < 1.0:
                import re
                m_layer = re.search(r'(?:CamEncode|img_encoder)\.model\.encoder\.layers\.(\d+)', name)
                if m_layer:
                    layer_idx = int(m_layer.group(1))
                    _layer_wise_groups.setdefault(layer_idx, []).append(param)
                else:
                    _layer_wise_groups.setdefault(-1, []).append(param)
        elif _use_prs and _is_v54_pose_head_param(name):
            pose_head_params.append(param)
            head_params.append(param)
            _pose_head_param_set.add(id(param))
        else:
            head_params.append(param)
            if _use_prs:
                corr_path_params.append(param)
                _corr_path_param_set.add(id(param))
        mod = name.split('.')[0]
        if mod not in _module_param_map:
            _module_param_map[mod] = []
        _module_param_map[mod].append(param)

    def _compute_grad_norms():
        """Compute per-group gradient L2 norms (backbone, bev_branch, head, per-module)."""
        bb_sq, bev_sq, hd_sq = 0.0, 0.0, 0.0
        mod_sq = {m: 0.0 for m in _module_param_map}
        for mod, params in _module_param_map.items():
            for p in params:
                if p.grad is None:
                    continue
                g2 = p.grad.data.norm(2).item() ** 2
                mod_sq[mod] += g2
                if id(p) in _backbone_param_set:
                    bb_sq += g2
                elif id(p) in _bev_branch_param_set:
                    bev_sq += g2
                else:
                    hd_sq += g2
        return bb_sq ** 0.5, bev_sq ** 0.5, hd_sq ** 0.5, {m: v ** 0.5 for m, v in mod_sq.items()}

    backbone_lr = args.lr * args.backbone_lr_scale
    bev_branch_lr_scale = (args.bev_branch_lr_scale if args.bev_branch_lr_scale is not None
                           else args.backbone_lr_scale)
    bev_branch_lr = args.lr * bev_branch_lr_scale

    if args.layer_wise_lr_decay < 1.0 and _layer_wise_groups:
        max_layer = max(k for k in _layer_wise_groups if k >= 0) if any(k >= 0 for k in _layer_wise_groups) else 0
        param_groups = []
        for layer_idx in sorted(_layer_wise_groups.keys()):
            if layer_idx < 0:
                depth = 0
            else:
                depth = max_layer - layer_idx
            layer_lr = backbone_lr * (args.layer_wise_lr_decay ** depth)
            param_groups.append({'params': _layer_wise_groups[layer_idx], 'lr': layer_lr})
        if _use_prs and pose_head_params:
            param_groups.append({'params': corr_path_params, 'lr': args.lr, 'name': 'corr_path'})
            param_groups.append({'params': pose_head_params, 'lr': args.lr, 'name': 'pose_head'})
        else:
            param_groups.append({'params': head_params, 'lr': args.lr, 'name': 'head'})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)
        _backbone_group_count = len(param_groups) - (2 if _use_prs and pose_head_params else 1)
    else:
        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'})
        if bev_branch_params:
            param_groups.append({'params': bev_branch_params, 'lr': bev_branch_lr, 'name': 'bev_branch'})
        if _use_prs and pose_head_params:
            param_groups.append({'params': corr_path_params, 'lr': args.lr, 'name': 'corr_path'})
            param_groups.append({'params': pose_head_params, 'lr': args.lr, 'name': 'pose_head'})
        else:
            param_groups.append({'params': head_params, 'lr': args.lr, 'name': 'head'})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)
        _backbone_group_count = max(len(param_groups) - (2 if _use_prs and pose_head_params else 1), 1)

    _prs_corr_pg = next((i for i, pg in enumerate(optimizer.param_groups)
                         if pg.get('name') == 'corr_path'), None)
    _prs_pose_pg = next((i for i, pg in enumerate(optimizer.param_groups)
                         if pg.get('name') == 'pose_head'), None)
    _backbone_pg = next((i for i, pg in enumerate(optimizer.param_groups)
                           if pg.get('name') == 'backbone'), 0)

    if is_main:
        tprint(f"Differential LR: backbone={backbone_lr:.2e} ({len(backbone_params)} params), "
               f"bev_branch={bev_branch_lr:.2e} ({len(bev_branch_params)} params), "
               f"heads={args.lr:.2e} ({len(head_params)} params)")
        if _use_prs and pose_head_params:
            tprint(f"  V54 PRS: corr_path={len(corr_path_params)} params, "
                   f"pose_head={len(pose_head_params)} params, release_ep={args.pose_release_epoch}")
        if args.layer_wise_lr_decay < 1.0 and _layer_wise_groups:
            for i, pg in enumerate(optimizer.param_groups[:-1]):
                tprint(f"  Layer group {i}: lr={pg['lr']:.2e} ({len(pg['params'])} params)")
        if args.backbone_warmup_epochs > 0:
            tprint(f"Backbone warmup: {args.backbone_warmup_epochs} epochs "
                   f"(1%→100% of backbone_lr={backbone_lr:.2e})")

    scheduler = None
    scheduler_choice = args.scheduler > 0
    if scheduler_choice:
        if args.lr_schedule == "cosine_warm_restarts":
            cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult)
            if args.warmup_epochs > 0:
                warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
                scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                                         milestones=[args.warmup_epochs])
                if is_main:
                    tprint(f"LR Schedule: LinearWarmup({args.warmup_epochs}ep) -> "
                           f"CosineWarmRestarts(T0={args.cosine_T0}, Tmult={args.cosine_Tmult})")
            else:
                scheduler = cosine_sched
                if is_main:
                    tprint(f"LR Schedule: CosineWarmRestarts(T0={args.cosine_T0}, Tmult={args.cosine_Tmult})")
        else:
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)
            if is_main:
                tprint(f"LR Schedule: StepLR(step={args.step_size}, gamma=0.5)")

    if is_main:
        tprint(f"Random seed: {args.seed} (cudnn.benchmark=True)")

    use_amp = torch.cuda.is_available() and not args.no_amp
    amp_dtype = torch.float16
    if args.amp_bf16 and torch.cuda.is_bf16_supported():
        try:
            _t1 = torch.randn(1, 1, 4, 4, dtype=torch.bfloat16, device='cuda')
            torch.nn.functional.interpolate(_t1, size=(8, 8), mode='bilinear', align_corners=False)
            torch.nn.functional.interpolate(_t1, scale_factor=2, mode='bilinear', align_corners=False)
            amp_dtype = torch.bfloat16
            del _t1
        except RuntimeError:
            if is_main:
                tprint("WARNING: --amp_bf16 requested but F.interpolate(bf16) not supported, falling back to fp16")
    _init_scale_exp = max(8, 14 - int(np.log2(max(world_size, 1))))
    _init_scale = 2 ** _init_scale_exp
    scaler = GradScaler(enabled=use_amp, init_scale=_init_scale,
                        growth_factor=1.5, growth_interval=1000,
                        backoff_factor=0.5)
    if is_main:
        if use_amp:
            tprint(f"AMP enabled with {amp_dtype}, GradScaler(init=2^{_init_scale_exp}, grow=1.5x/1000steps)")
        else:
            tprint(f"AMP disabled (--no_amp=1), training in FP32")

    grad_accum_steps = max(1, args.grad_accum_steps)
    if grad_accum_steps > 1 and is_main:
        effective_bs = args.batch_size * world_size * grad_accum_steps
        tprint(f"Gradient Accumulation: {grad_accum_steps} steps, "
               f"effective batch size = {args.batch_size}×{world_size}GPU×{grad_accum_steps}accum = {effective_bs}")
    
    if is_main and args.augment_intrinsic > 0:
        _cxcy_str = getattr(args, 'augment_intrinsic_cxcy', 0.0)
        if _cxcy_str > 0:
            tprint(f"Intrinsic augmentation: fx/fy ±{args.augment_intrinsic*100:.0f}%, cx/cy ±{_cxcy_str*100:.0f}%")
        else:
            tprint(f"Intrinsic augmentation: fx/fy ±{args.augment_intrinsic*100:.0f}%, cx/cy ±{args.augment_intrinsic*100:.0f}%")

    # === v32: T_init Invariance Config ===
    _v32_tinit_dropout = getattr(args, 'tinit_dropout_prob', 0.0)
    _v32_consistency_w = getattr(args, 'consistency_loss_weight', 0.0)
    _v32_consistency_start = getattr(args, 'consistency_loss_start_epoch', 0)
    _v32_prog_start = getattr(args, 'progressive_angle_start', 0.0)
    _v32_prog_end = getattr(args, 'progressive_angle_end', 0.0)
    _v32_prog_warmup = getattr(args, 'progressive_warmup_epochs', 100)
    _v32_enabled = _v32_tinit_dropout > 0 or _v32_consistency_w > 0 or _v32_prog_start > 0

    # === v32.1: EMA Target Network ===
    _v321_ema_enabled = getattr(args, 'ema_consistency', 0) > 0 and _v32_consistency_w > 0
    _v321_ema_decay = getattr(args, 'ema_decay', 0.996)
    _v321_ema_model = None
    if _v321_ema_enabled:
        import copy
        _v321_ema_model = copy.deepcopy(raw_model)
        _v321_ema_model.eval()
        for p in _v321_ema_model.parameters():
            p.requires_grad_(False)

    # === v32.1: Continuous T_init Noise ===
    _v321_continuous_noise = getattr(args, 'continuous_tinit_noise', 0) > 0
    _v321_noise_max_deg = getattr(args, 'continuous_noise_max_deg', 30.0)
    _match_phase_noise_max = float(getattr(args, 'match_phase_noise_max_deg', 0.0) or 0.0)
    _use_match_head = getattr(args, 'use_match_head', 0) > 0

    if is_main and _v32_enabled:
        tprint("=" * 60)
        tprint("v32 T_init Invariance Training ENABLED:")
        if _v321_continuous_noise:
            tprint(f"  [v32.1] Continuous T_init noise: 0°-{_v321_noise_max_deg}° (replaces binary dropout)")
        elif _v32_tinit_dropout > 0:
            tprint(f"  T_init dropout: {_v32_tinit_dropout*100:.0f}% probability")
        if _v32_consistency_w > 0:
            _cons_mode = "EMA target" if _v321_ema_enabled else "stop-gradient"
            tprint(f"  Consistency loss: weight={_v32_consistency_w}, start_epoch={_v32_consistency_start}, mode={_cons_mode}")
            if _v321_ema_enabled:
                tprint(f"  EMA decay: {_v321_ema_decay}")
        if _v32_prog_start > 0:
            tprint(f"  Progressive angle: {_v32_prog_start}° → {_v32_prog_end}° over {_v32_prog_warmup} epochs")
        tprint("=" * 60)

    if is_main and _use_match_head:
        tprint("=" * 60)
        tprint("GMP Match/Corr training:")
        tprint(f"  L_corr weight={getattr(args, 'correspondence_loss_weight', 0)}, "
               f"start_ep={getattr(args, 'correspondence_loss_start_epoch', 0)}, "
               f"warmup_ep={getattr(args, 'correspondence_loss_warmup_epochs', 0)}")
        tprint(f"  fallback: disable={getattr(args, 'match_disable_fallback', 0)}, "
               f"gate_init_ratio={getattr(args, 'match_gate_use_init_ratio', 0)}, "
               f"valid_min={getattr(args, 'match_valid_ratio_min', 0.3)}")
        tprint(f"  corr_validity={getattr(args, 'match_corr_validity_mode', 'gt')}, "
               f"conf_thr={getattr(args, 'match_confidence_threshold', 0.2)}, "
               f"epnp_min_pts={getattr(args, 'match_epnp_min_points', 4)}")
        if _match_phase_noise_max > 0:
            tprint(f"  match_phase_noise cap: {_match_phase_noise_max}° "
                   f"(global continuous max {_v321_noise_max_deg}°)")
        tprint("=" * 60)

    # === V42 CF-BEV-R specific training config ===
    _v42_enabled = getattr(args, 'fusion_backend', '') == 'cf_bev_r'
    _v42_rocr_detach_epochs = getattr(args, 'rocr_detach_epochs', 0)
    _v42_corr_alignment_w = getattr(args, 'corr_alignment_weight', 0.0)
    _v42_corr_alignment_warmup = getattr(args, 'corr_alignment_warmup', 20)
    _v42_seq_consistency_w = getattr(args, 'seq_consistency_weight', 0.0)
    _v42_seq_consistency_start = getattr(args, 'seq_consistency_start_epoch', 999)

    _v42_corr_loss = None
    _v42_seq_loss = None
    if _v42_enabled:
        if _v42_corr_alignment_w > 0:
            from losses.corr_alignment_loss import CorrelationAlignmentLoss
            _v42_corr_loss = CorrelationAlignmentLoss(
                weight=_v42_corr_alignment_w,
                warmup_epochs=_v42_corr_alignment_warmup,
            )
        if _v42_seq_consistency_w > 0:
            from losses.corr_alignment_loss import SequenceConsistencyLoss
            _v42_seq_loss = SequenceConsistencyLoss(weight=_v42_seq_consistency_w)
        if is_main:
            tprint("=" * 60)
            tprint("V42 CF-BEV-R Training Config:")
            tprint(f"  RoCR detach epochs: {_v42_rocr_detach_epochs}")
            tprint(f"  Corr alignment loss: w={_v42_corr_alignment_w}, warmup={_v42_corr_alignment_warmup}")
            tprint(f"  Seq consistency loss: w={_v42_seq_consistency_w}, start_epoch={_v42_seq_consistency_start}")
            tprint("=" * 60)

    _v54_lsp_fn = None
    _v54_rig_fn = None
    if getattr(args, 'use_lsp_loss', 0) > 0:
        from losses.lidar_splat_photo_loss import LiDARSplatPhotoLoss
        _v54_lsp_fn = LiDARSplatPhotoLoss(
            lambda_ssim=getattr(args, 'lsp_lambda_ssim', 0.2),
            max_points=getattr(args, 'lsp_max_points', 4096),
            downsample=getattr(args, 'lsp_downsample', 4),
            loss_clip=getattr(args, 'lsp_loss_clip', 2.0),
            min_valid_ratio=getattr(args, 'lsp_min_valid_ratio', 0.02),
        )
    if getattr(args, 'rig_consistency_weight', 0.0) > 0:
        from losses.rig_consistency_loss import RigConsistencyLoss
        _v54_rig_fn = RigConsistencyLoss(weight=args.rig_consistency_weight)

    if is_main and (getattr(args, 'use_lsp_loss', 0) > 0
                    or getattr(args, 'rig_consistency_weight', 0.0) > 0
                    or int(getattr(args, 'pose_release_epoch', 0)) > 0):
        tprint("=" * 60)
        tprint("V54 Photo-Geometric Joint Alignment:")
        if getattr(args, 'use_lsp_loss', 0) > 0:
            tprint(f"  LSP: w ramp {args.lsp_weight_start}→{args.lsp_weight} over "
                   f"{args.lsp_ramp_epochs}ep, start_ep={args.lsp_start_epoch}, "
                   f"λ_ssim={args.lsp_lambda_ssim}, clip={args.lsp_loss_clip}, "
                   f"min_valid={args.lsp_min_valid_ratio}")
        if getattr(args, 'rig_consistency_weight', 0.0) > 0:
            tprint(f"  RigC: w={args.rig_consistency_weight}, start_ep={args.rig_consistency_start_epoch}")
        if int(getattr(args, 'pose_release_epoch', 0)) > 0:
            tprint(f"  PRS: pose release ep={args.pose_release_epoch}, "
                   f"joint ep={args.pose_release_joint_epoch} "
                   f"(corr×{args.pose_release_corr_lr_scale_joint})")
        if int(getattr(args, 'freeze_backbone_epoch', 999)) < 999:
            tprint(f"  Refine: freeze backbone from ep={args.freeze_backbone_epoch}")
        tprint("=" * 60)

    def _v32_get_angle_range(epoch):
        """Progressive angle curriculum for v32."""
        if _v32_prog_start <= 0:
            return train_noise["angle_range_deg"]
        if epoch >= _v32_prog_warmup:
            return _v32_prog_end
        alpha = epoch / max(1, _v32_prog_warmup)
        return _v32_prog_start + alpha * (_v32_prog_end - _v32_prog_start)

    def _v32_apply_tinit_dropout(init_T_np, gt_T_np, prob):
        """Replace T_init with random rotation to break shortcut.
        Uses random perturbation far from GT (15-30 deg) to force visual learning."""
        if prob <= 0:
            return init_T_np
        B = init_T_np.shape[0]
        mask = np.random.random(B) < prob
        if not mask.any():
            return init_T_np
        from scipy.spatial.transform import Rotation as R_sp
        for i in range(B):
            if mask[i]:
                rand_rv = np.random.randn(3)
                rand_rv = rand_rv / (np.linalg.norm(rand_rv) + 1e-8)
                rand_angle = np.random.uniform(15, 30) * np.pi / 180
                dR = R_sp.from_rotvec(rand_rv * rand_angle).as_matrix()
                init_T_np[i, :3, :3] = dR @ gt_T_np[i, :3, :3]
        return init_T_np

    def _v321_apply_continuous_noise(init_T_np, max_deg, epoch=0):
        """v32.1: Apply continuous random rotation noise to ALL T_init samples.
        Noise magnitude ramps up with progressive curriculum to avoid
        overwhelming the model in early training. At epoch 0, max noise = max_deg * alpha
        where alpha tracks the progressive curriculum fraction (0→1)."""
        if _v32_prog_warmup > 0:
            alpha = min(1.0, epoch / max(1, _v32_prog_warmup))
        else:
            alpha = 1.0
        effective_max = max_deg * max(0.1, alpha)
        if _use_match_head and _match_phase_noise_max > 0:
            effective_max = min(effective_max, _match_phase_noise_max)
        from scipy.spatial.transform import Rotation as R_sp
        B = init_T_np.shape[0]
        for i in range(B):
            noise_deg = np.random.uniform(0, effective_max)
            rand_rv = np.random.randn(3)
            rand_rv = rand_rv / (np.linalg.norm(rand_rv) + 1e-8)
            noise_rad = noise_deg * np.pi / 180
            dR = R_sp.from_rotvec(rand_rv * noise_rad).as_matrix()
            init_T_np[i, :3, :3] = dR @ init_T_np[i, :3, :3]
        return init_T_np

    if is_main and args.augment_pitch_flip_prob > 0:
        tprint(f"GT pitch perturbation (Y-axis): prob={args.augment_pitch_flip_prob}, "
               f"max_deg={args.augment_pitch_flip_max_deg}°")
    if is_main and getattr(args, 'augment_pitch_sign_flip_prob', 0) > 0:
        tprint(f"GT pitch sign flip: prob={args.augment_pitch_sign_flip_prob} "
               f"(precisely negate pitch angle)")

    _identity_4x4 = torch.eye(4, device=device)

    train_noise = {
        "angle_range_deg": args.angle_range_deg if args.angle_range_deg is not None else 20,
        "trans_range": args.trans_range if args.trans_range is not None else 1.5,
    }

    eval_noise = {
        "angle_range_deg": args.eval_angle_range_deg if args.eval_angle_range_deg is not None else train_noise["angle_range_deg"],
        "trans_range": args.eval_trans_range if args.eval_trans_range is not None else train_noise["trans_range"],
    }

    per_axis_weights_parsed = None
    if args.per_axis_weights:
        per_axis_weights_parsed = tuple(float(x) for x in args.per_axis_weights.split(','))
    jacobian_loss_axis_weights_parsed = None
    if getattr(args, 'jacobian_loss_axis_weights', ''):
        jacobian_loss_axis_weights_parsed = tuple(
            float(x) for x in args.jacobian_loss_axis_weights.split(','))
        if len(jacobian_loss_axis_weights_parsed) != 3:
            raise ValueError("--jacobian_loss_axis_weights expects three comma-separated values")

    global_step = 0
    
    # 累积误差统计
    epoch_pose_errors = {
        'trans_error': 0, 'fwd_error': 0, 'lat_error': 0, 'ht_error': 0,
        'rot_error': 0, 'roll_error': 0, 'pitch_error': 0, 'yaw_error': 0,
    }
    
    best_train = {'epoch': -1, 'loss': float('inf'), 'trans': float('inf'), 'rot': float('inf'), 'errors': None}
    best_val = {'epoch': -1, 'loss': float('inf'), 'trans': float('inf'), 'rot': float('inf'), 'errors': None}
    best_medw = {'epoch': -1, 'rot': float('inf'), 'roll': float('inf'), 'pitch': float('inf'),
                 'yaw': float('inf'), 'max_rpy': float('inf')}
    best_dual = {'epoch': -1, 'score': float('inf'), 'medw_max_rpy': float('inf'),
                 'medw_roll': float('inf'), 'medw_pitch': float('inf'), 'medw_yaw': float('inf'),
                 'jacobian': float('-inf'), 'jacobian_roll': float('-inf'),
                 'jacobian_pitch': float('-inf'), 'jacobian_yaw': float('-inf')}
    best_jacobian = {'epoch': -1, 'overall': float('-inf'),
                     'roll': float('-inf'), 'pitch': float('-inf'), 'yaw': float('-inf')}
    kpi_history = []
    last_epoch_train_errors = None
    last_epoch_val_errors = None
    checkpoint_records = []
    early_stop_counter = 0
    jacobian_early_stop_counter = 0
    early_stop_triggered = False
    
    if is_main:
        tprint("=" * 80)
        tprint(f"Regression Head: {'MLP (3-layer)' if use_mlp_head else 'Linear (single layer)'}")
        tprint("Loss Calculation Formula:")
        rot_label = "geodesic_loss(radians)" if use_geodesic_loss else "rotation_loss(radians)"
        formula = f"  total_loss = w_rot * {rot_label} + w_pc * PC_reproj_loss + w_quat * quat_norm_loss"
        if enable_axis_loss:
            formula += f" + {args.weight_axis_rotation} * axis_rotation_loss"
        tprint(formula)
        if enable_axis_loss:
            tprint(f"  Axis weights (R/P/Y): {args.axis_weights}")
        if rotation_only:
            tprint("  Default weights (rotation_only): w_rot=1.0, w_pc=1.0, w_trans=0.0, w_quat=0.5")
        else:
            tprint("  Default weights (full-pose): w_rot=0.5, w_pc=0.5, w_trans=1.0, w_quat=0.5")
        tprint("")
        tprint("Log Terminology:")
        tprint("  • rotation_loss / pose_L: displayed in degrees (°); total_loss (w-sum) uses radians + all weighted terms")
        tprint("  • Step Loss total (w-sum): weighted sum incl. corr×weight, geo, cons, jac; NOT comparable to A-only ~3–4 scale")
        tprint("  • correspondence_loss: raw px; step log shows corr×weight contribution when match head enabled")
        tprint("  • PC_reproj_loss: point cloud reprojection error")
        tprint("  • quat_norm_loss: quaternion normalization penalty")
        tprint("  • Pose Error - Rot: rotation error with Roll/Pitch/Yaw breakdown")
        tprint("  • Pose Error - Trans: translation error with Forward/Lateral/Height breakdown")
        if args.enable_medw_eval > 0:
            tprint(f"  • MEDW eval: MEDW{args.medw_eval_max_frames} on val split "
                   f"(reuse val forward) every {args.eval_epoches} epochs")
        if args.enable_jacobian_eval > 0:
            _jac_extra = (args.jacobian_eval_batches * 3 * args.jacobian_eval_n_probes)
            tprint(f"  • Jacobian eval: ±{args.jacobian_eval_angle_deg}° sweep on val batch[0:"
                   f"{args.jacobian_eval_batches}] (+{_jac_extra} extra forwards/eval, "
                   f"correction=init_err-out_err)")
        if getattr(args, 'jacobian_loss_weight', 0.0) > 0:
            tprint(f"  • Jacobian loss: weight={args.jacobian_loss_weight}, "
                   f"start_ep={args.jacobian_loss_start_epoch}, "
                   f"interval={getattr(args, 'jacobian_loss_interval', 4)} batches, "
                   f"probe=±{args.jacobian_loss_probe_deg}°")
            if jacobian_loss_axis_weights_parsed is not None:
                tprint(f"    axis sampling weights R/P/Y={jacobian_loss_axis_weights_parsed}")
        if getattr(args, 'zero_drift_loss_weight', 0.0) > 0:
            _zd_ramp = int(getattr(args, 'zero_drift_loss_ramp_epochs', 0))
            _zd_w0 = float(getattr(args, 'zero_drift_loss_weight_start', 0.0))
            _zd_w_msg = (f"ramp {_zd_w0}→{args.zero_drift_loss_weight} over {_zd_ramp}ep"
                         if _zd_ramp > 0 and _zd_w0 > 0 else f"weight={args.zero_drift_loss_weight}")
            tprint(f"  • Zero-drift loss: {_zd_w_msg}, "
                   f"start_ep={args.zero_drift_loss_start_epoch}, "
                   f"dedicated_ratio={getattr(args, 'zero_drift_dedicated_ratio', 0.0)}")
        if getattr(args, 'use_mgda', 0) > 0:
            _inj_mgda = "yes" if getattr(args, 'mgda_include_inject', 1) else "no"
            _photo_mgda = "yes" if getattr(args, 'mgda_include_photo', 1) else "no"
            tprint(f"  • MGDA: pose+zd(+inject={_inj_mgda},+photo={_photo_mgda}) weighted-loss, "
                   f"start_ep={getattr(args, 'mgda_start_epoch', 5)}")
        if getattr(args, 'use_lsp_loss', 0) > 0:
            tprint(f"  • LSP: ramp {args.lsp_weight_start}→{args.lsp_weight}, start_ep={args.lsp_start_epoch}")
        if getattr(args, 'rig_consistency_weight', 0.0) > 0:
            tprint(f"  • Rig consistency: w={args.rig_consistency_weight}, "
                   f"start_ep={args.rig_consistency_start_epoch}")
        if int(getattr(args, 'pose_release_epoch', 0)) > 0:
            tprint(f"  • Pose release (PRS): ep={args.pose_release_epoch}")
        if getattr(args, 'jacobian_early_stop_min', 0.0) > 0:
            tprint(f"  • Jacobian early-stop: overall < {args.jacobian_early_stop_min} "
                   f"for {args.jacobian_early_stop_patience} eval cycles → stop")
        if args.enable_jacobian_gate_ckpt > 0:
            tprint("  • Jacobian gate ckpt: best val Jacobian overall → ckpt_best_jacobian.pth")
        if getattr(args, 'inject_recovery_loss_weight', 0.0) > 0:
            tprint(f"  • Inject-recovery loss: weight={args.inject_recovery_loss_weight}, "
                   f"start_ep={args.inject_recovery_loss_start_epoch}, "
                   f"inject={args.inject_recovery_magnitude_deg}° all-RPY, "
                   f"dedicated_ratio={getattr(args, 'inject_recovery_dedicated_ratio', 0.0)}")
        if getattr(args, 'correction_quat_loss_weight', 0.0) > 0:
            tprint(f"  • Direct correction quaternion loss: "
                   f"weight={args.correction_quat_loss_weight}")
        if getattr(args, 'use_dp_head', 0) > 0:
            tprint(f"  • V53 DP-Head: gate={args.dp_gate_deg}°, route_loss={args.route_loss_weight}, "
                   f"route_zd_penalty={getattr(args, 'route_zd_penalty_weight', 0.0)}, "
                   f"route_input={getattr(args, 'dp_route_input_mode', 'gt_or_pred')}, "
                   f"jacg_input={getattr(args, 'dp_jacg_input_mode', 'gt_or_pred')}, "
                   f"rec_q={getattr(args, 'dp_recovery_quat_loss_weight', 0.0)}, "
                   f"train_hard={getattr(args, 'dp_train_hard_route', 0)}, "
                   f"jacg={getattr(args, 'use_jacg', 1)}, "
                   f"hard_eval={getattr(args, 'use_hard_route_eval', 0)}")
        if getattr(args, 'use_adir', 0) > 0:
            tprint(f"  • V53b ADIR: steps={args.adir_steps}, max_step={args.adir_max_step_deg}°, "
                   f"adapter_only={getattr(args, 'adir_train_only', 0)}")
        if args.enable_dual_gate_ckpt > 0:
            tprint(f"  • Dual gate ckpt: max(MEDW{args.medw_eval_max_frames} R,P,Y) < {args.dual_gate_medw_max}° "
                   f"AND Jacobian R/P/Y/overall > {args.dual_gate_jacobian_min} → ckpt_best_dual.pth")
        if getattr(args, 'fusion_backend', 'bev') == 'geo_match_proj':
            tprint("  • match[...]: L_corr(px), valid_init/valid_gt, fb, epnp_fail, epnp_pts, epnp_grad")
            tprint("  • geo[...]: appearance/depth consistency loss + geo valid ratio")
        tprint("=" * 80)
    
    start_epoch = 0
    if args.resume_ckpt is not None:
        resume_path = args.resume_ckpt
        if resume_path == "auto":
            _candidates = []
            for _f in sorted(os.listdir(ckpt_save_dir)):
                if _f.startswith("ckpt_") and _f.endswith(".pth") and _f not in ("ckpt_best_val.pth",):
                    try:
                        _ep = int(_f.replace("ckpt_", "").replace(".pth", "").replace("emergency_ep", ""))
                        _candidates.append((_ep, os.path.join(ckpt_save_dir, _f)))
                    except ValueError:
                        continue
            if _candidates:
                _candidates.sort(key=lambda x: x[0], reverse=True)
                resume_path = _candidates[0][1]
                if is_main:
                    tprint(f"Auto-resume: found {resume_path} (epoch {_candidates[0][0]})")
            else:
                resume_path = None
                if is_main:
                    tprint(f"Auto-resume: no checkpoint found in {ckpt_save_dir}, starting from scratch")
        if resume_path is not None and os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            model_to_load = model.module if use_ddp else model
            model_to_load.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except (ValueError, RuntimeError) as _opt_err:
                    if is_main:
                        tprint(f"WARNING: optimizer state incompatible ({_opt_err}); "
                               f"using freshly built optimizer (model weights still loaded)")
            if 'scheduler_state_dict' in ckpt and scheduler is not None and ckpt['scheduler_state_dict'] is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except (ValueError, RuntimeError) as _sched_err:
                    if is_main:
                        tprint(f"WARNING: scheduler state incompatible ({_sched_err}); "
                               f"rebuilding scheduler from epoch 0")
                    if 'epoch' in ckpt:
                        for _ in range(ckpt['epoch']):
                            scheduler.step()
            elif scheduler is not None and 'epoch' in ckpt:
                for _ in range(ckpt['epoch']):
                    scheduler.step()
            if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
                try:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                except (ValueError, RuntimeError) as _sc_err:
                    if is_main:
                        tprint(f"WARNING: GradScaler state incompatible ({_sc_err}); using fresh scaler")
            start_epoch = ckpt.get('epoch', 0)
            if 'best_train' in ckpt and ckpt['best_train'] is not None:
                best_train = ckpt['best_train']
            if 'best_val' in ckpt and ckpt['best_val'] is not None:
                best_val = ckpt['best_val']
            if 'best_medw' in ckpt and ckpt['best_medw'] is not None:
                best_medw = ckpt['best_medw']
            if 'best_dual' in ckpt and ckpt['best_dual'] is not None:
                best_dual = ckpt['best_dual']
            if 'best_jacobian' in ckpt and ckpt['best_jacobian'] is not None:
                best_jacobian = ckpt['best_jacobian']
            if 'kpi_history' in ckpt and ckpt['kpi_history']:
                kpi_history = ckpt['kpi_history']
            if 'early_stop_counter' in ckpt:
                early_stop_counter = int(ckpt['early_stop_counter'])
            if 'jacobian_early_stop_counter' in ckpt:
                jacobian_early_stop_counter = int(ckpt['jacobian_early_stop_counter'])
            if _v321_ema_enabled and _v321_ema_model is not None and 'ema_state_dict' in ckpt:
                _v321_ema_model.load_state_dict(ckpt['ema_state_dict'])
                if is_main:
                    tprint(f"  EMA model restored from checkpoint")
            elif _v321_ema_enabled and _v321_ema_model is not None:
                import copy as _copy_mod
                _v321_ema_model.load_state_dict(model_to_load.state_dict())
                if is_main:
                    tprint(f"  EMA model re-initialized from online model (no ema_state_dict in checkpoint)")
            if is_main:
                tprint(f"Resumed from {resume_path}: epoch={start_epoch}, "
                       f"best_val_rot={best_val.get('rot', 'N/A')}")
                if start_epoch >= num_epochs:
                    tprint(f"WARNING: resume epoch ({start_epoch}) >= num_epochs ({num_epochs}); "
                           f"training loop will not run. Use --pretrain_ckpt for refine, "
                           f"or set num_epochs > {start_epoch}.")
        if use_ddp:
            dist.barrier()

    training_start_time = time.time()
    epoch_times = []

    def _emergency_save(epoch_num, reason=""):
        """Save checkpoint on crash so --resume_ckpt auto can recover."""
        if not is_main:
            return
        try:
            epath = os.path.join(ckpt_save_dir, f"ckpt_emergency_ep{epoch_num}.pth")
            model_to_save = model.module if use_ddp else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_num,
                'best_val': best_val,
                'emergency': True,
                'reason': reason,
            }, epath)
            tprint(f"Emergency checkpoint saved: {epath}")
        except Exception as e2:
            tprint(f"Failed to save emergency checkpoint: {e2}")

    _current_epoch = [start_epoch]

    def _sigterm_handler(sig, frame):
        _emergency_save(_current_epoch[0], f"Signal {sig} (peer crash)")
        import sys
        sys.exit(128 + sig)

    import signal
    signal.signal(signal.SIGTERM, _sigterm_handler)

    _qbr_alpha = getattr(args, 'quat_bias_reset_alpha', 0.0)
    _qbr_T0 = getattr(args, 'cosine_T0', 40)
    _qbr_Tmult = getattr(args, 'cosine_Tmult', 2)

    _mgda_weight_params = _get_mgda_weight_params(raw_model)
    for epoch in range(start_epoch, num_epochs):
        _current_epoch[0] = epoch
        main._cuda_error_count = 0
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        if (is_main and getattr(args, 'use_mgda', 0) > 0
                and epoch == getattr(args, 'mgda_start_epoch', 5)):
            tprint(f"  [MGDA] epoch {epoch + 1} start: weighted-loss mode, "
                   f"bottleneck={len(_mgda_weight_params)} tensors")
        model.train()
        if hasattr(raw_model, 'set_training_epoch'):
            raw_model.set_training_epoch(epoch)

        if _qbr_alpha > 0 and epoch > 0:
            _is_cycle_boundary = False
            _cycle_start = 0
            _T = _qbr_T0
            while _cycle_start + _T <= epoch:
                _cycle_start += _T
                _T = int(_T * _qbr_Tmult)
            if epoch == _cycle_start:
                _is_cycle_boundary = True
            if _is_cycle_boundary and is_main:
                _qh = getattr(raw_model, 'corr_head', None)
                if _qh and hasattr(_qh, 'quat_head'):
                    _final = _qh.quat_head[-1]
                    _id_bias = torch.tensor([1.0, 0.0, 0.0, 0.0], device=_final.bias.device)
                    with torch.no_grad():
                        _final.bias.copy_((1 - _qbr_alpha) * _final.bias + _qbr_alpha * _id_bias)
                    tprint(f"  [v49] Quat bias reset (alpha={_qbr_alpha}) at cycle boundary ep{epoch}: "
                           f"new bias={_final.bias.data.cpu().tolist()}")
        if args.domain_adversarial > 0:
            raw_model._dann_epoch_ratio = epoch / max(num_epochs - 1, 1)

        if args.backbone_warmup_epochs > 0:
            if epoch < args.backbone_warmup_epochs:
                warmup_denom = max(1, args.backbone_warmup_epochs - 1)
                warmup_factor = 0.01 + 0.99 * (epoch / warmup_denom)
            else:
                warmup_factor = 1.0
            for gi in range(_backbone_group_count):
                base_lr = optimizer.param_groups[gi].get('_base_lr',
                    optimizer.param_groups[gi]['lr'] if epoch == 0 else
                    optimizer.param_groups[gi].get('_base_lr', backbone_lr))
                if epoch == 0:
                    optimizer.param_groups[gi]['_base_lr'] = optimizer.param_groups[gi]['lr']
                    base_lr = optimizer.param_groups[gi]['lr']
                optimizer.param_groups[gi]['lr'] = base_lr * warmup_factor
            if is_main and (epoch == 0 or epoch == args.backbone_warmup_epochs - 1
                            or epoch == args.backbone_warmup_epochs):
                tprint(f"  Backbone warmup: factor={warmup_factor:.3f}, "
                       f"backbone_lr={optimizer.param_groups[0]['lr']:.2e}")

        _prs_corr_scale, _prs_pose_scale = _prs_lr_scales(args, epoch)
        if _prs_corr_pg is not None:
            _base = optimizer.param_groups[_prs_corr_pg].get('_base_lr', args.lr)
            if '_base_lr' not in optimizer.param_groups[_prs_corr_pg]:
                optimizer.param_groups[_prs_corr_pg]['_base_lr'] = _base
            optimizer.param_groups[_prs_corr_pg]['lr'] = _base * _prs_corr_scale
        if _prs_pose_pg is not None:
            _base = optimizer.param_groups[_prs_pose_pg].get('_base_lr', args.lr)
            if '_base_lr' not in optimizer.param_groups[_prs_pose_pg]:
                optimizer.param_groups[_prs_pose_pg]['_base_lr'] = _base
            optimizer.param_groups[_prs_pose_pg]['lr'] = _base * _prs_pose_scale
        _freeze_bb = int(getattr(args, 'freeze_backbone_epoch', 999))
        if _freeze_bb < 999 and epoch >= _freeze_bb:
            for _pg in optimizer.param_groups:
                if _pg.get('name') in ('backbone', 'bev_branch'):
                    _pg['lr'] = 0.0
                elif _pg.get('name') is None and optimizer.param_groups.index(_pg) < _backbone_group_count:
                    _pg['lr'] = 0.0
        if is_main and int(getattr(args, 'pose_release_epoch', 0)) > 0:
            if epoch == args.pose_release_epoch:
                tprint(f"  [V54 PRS] Pose released at epoch {epoch + 1}")
            if (_prs_corr_pg is not None and epoch == args.pose_release_joint_epoch):
                tprint(f"  [V54 PRS] Joint phase: corr_path lr ×{args.pose_release_corr_lr_scale_joint}")
        if is_main and (_freeze_bb < 999 and epoch == _freeze_bb):
            tprint(f"  [V54 Refine] Backbone frozen from epoch {epoch + 1}")

        train_loss = {}
        for key in epoch_pose_errors:
            epoch_pose_errors[key] = 0

        if epoch == 0 and hasattr(raw_model, 'get_module_profile'):
            raw_model._profile_modules = True
        elif epoch == 1:
            if hasattr(raw_model, '_profile_modules'):
                raw_model._profile_modules = False

        epoch_start = time.time()
        out_init_loss_choice = epoch < 5
        t_data_total, t_prep_total, t_compute_total, t_vis_total = 0.0, 0.0, 0.0, 0.0
        vis_count = 0
        _epoch_grad_accum = {'bb': [], 'bev': [], 'hd': [], 'mod': {}}
        _profile_on = getattr(raw_model, '_profile_modules', False)
        _bwd_profile_events = [] if _profile_on else None
        _do_detailed_profile = _profile_on
        t_h2d_total = 0.0
        t_cpu_aug_total = 0.0
        processed_batches = 0
        _gate_stats = {'gate_bev_sum': 0.0, 'gate_proj_sum': 0.0, 'gate_entropy_sum': 0.0, 'gate_count': 0}
        t_iter_start = time.time()
        for batch_index, batch_data in enumerate(train_loader):
            t_data_end = time.time()
            t_data_total += t_data_end - t_iter_start

            if batch_data is None:
                t_iter_start = time.time()
                continue
            if len(batch_data) == 6:
                imgs, pcs, masks, gt_T_to_camera, intrinsics, domain_ids_list = batch_data
            else:
                imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data
                domain_ids_list = None

            t_prep_start = time.time()
            gt_T_to_camera_np = np.array(gt_T_to_camera, dtype=np.float32)
            if args.augment_mount_jitter_prob > 0:
                gt_T_to_camera_np = augment_mount_jitter(
                    gt_T_to_camera_np,
                    prob=args.augment_mount_jitter_prob,
                    rotation_sigma_deg=args.augment_mount_jitter_rot_sigma,
                    translation_sigma_m=args.augment_mount_jitter_trans_sigma,
                )
            _sign_flip_p = getattr(args, 'augment_pitch_sign_flip_prob', 0.0)
            if args.augment_pitch_flip_prob > 0 or _sign_flip_p > 0:
                gt_T_to_camera_np = augment_gt_pitch_flip(
                    gt_T_to_camera_np,
                    prob=args.augment_pitch_flip_prob,
                    max_deg=args.augment_pitch_flip_max_deg,
                    sign_flip_prob=_sign_flip_p,
                )
            _current_angle_range = _v32_get_angle_range(epoch) if _v32_enabled else train_noise["angle_range_deg"]
            _mr_prob = getattr(args, 'multi_range_prob', 0.0)
            _mr_angle = getattr(args, 'multi_range_angle', 5.0)
            if _mr_prob > 0 and np.random.rand() < _mr_prob:
                _current_angle_range = _mr_angle
            _ms_spec = getattr(args, 'multi_scale_perturb', '')
            if _ms_spec:
                _ms_roll = np.random.rand()
                _ms_cum = 0.0
                for _ms_pair in _ms_spec.split(','):
                    _ms_parts = _ms_pair.strip().split(':')
                    if len(_ms_parts) == 2:
                        _ms_angle, _ms_p = float(_ms_parts[0]), float(_ms_parts[1])
                        _ms_cum += _ms_p
                        if _ms_roll < _ms_cum:
                            _current_angle_range = _ms_angle
                            break
            _use_zero_perturb = False
            _use_inject_recovery = False
            _inj_dedicated = float(getattr(args, 'inject_recovery_dedicated_ratio', 0.0))
            _inj_loss_w_cfg = float(getattr(args, 'inject_recovery_loss_weight', 0.0))
            if (_inj_dedicated > 0 and _inj_loss_w_cfg > 0
                    and np.random.rand() < _inj_dedicated):
                _use_inject_recovery = True
                init_T_to_camera_np = _apply_fixed_inject_batch(
                    gt_T_to_camera_np, getattr(args, 'inject_recovery_magnitude_deg', 2.0))
            else:
                _zd_dedicated = float(getattr(args, 'zero_drift_dedicated_ratio', 0.0))
                if _zd_dedicated > 0 and np.random.rand() < _zd_dedicated:
                    _use_zero_perturb = True
                elif (args.zero_perturbation_prob > 0 and
                      np.random.rand() < args.zero_perturbation_prob):
                    _use_zero_perturb = True
            if _use_inject_recovery:
                pass  # init already set above
            elif _use_zero_perturb:
                init_T_to_camera_np = gt_T_to_camera_np.copy()
            else:
                init_T_to_camera_np, _, _ = generate_single_perturbation_from_T(
                    gt_T_to_camera_np,
                    angle_range_deg=_current_angle_range,
                    trans_range=train_noise["trans_range"],
                    rotation_only=rotation_only,
                    distribution=args.perturb_distribution,
                    per_axis_prob=args.per_axis_prob,
                    per_axis_weights=per_axis_weights_parsed,
                    symmetric_perturb=bool(getattr(args, 'symmetric_perturb', 0)),
                )
                if _v321_continuous_noise:
                    init_T_to_camera_np = _v321_apply_continuous_noise(
                        init_T_to_camera_np, _v321_noise_max_deg, epoch=epoch)
                elif _v32_tinit_dropout > 0:
                    init_T_to_camera_np = _v32_apply_tinit_dropout(
                        init_T_to_camera_np, gt_T_to_camera_np, _v32_tinit_dropout)

            _v32_init_T_alt = None
            if _v32_consistency_w > 0 and epoch >= _v32_consistency_start:
                _v32_init_T_alt, _, _ = generate_single_perturbation_from_T(
                    gt_T_to_camera_np,
                    angle_range_deg=_current_angle_range,
                    trans_range=train_noise["trans_range"],
                    rotation_only=rotation_only,
                    distribution=args.perturb_distribution,
                    per_axis_prob=args.per_axis_prob,
                    per_axis_weights=per_axis_weights_parsed,
                )
                if _v321_continuous_noise:
                    _v32_init_T_alt = _v321_apply_continuous_noise(
                        _v32_init_T_alt, _v321_noise_max_deg, epoch=epoch)
                elif _v32_tinit_dropout > 0:
                    _v32_init_T_alt = _v32_apply_tinit_dropout(
                        _v32_init_T_alt, gt_T_to_camera_np, _v32_tinit_dropout)

            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float()
            if args.augment_color_jitter > 0:
                resize_imgs = _apply_color_jitter(resize_imgs, args.augment_color_jitter)
            
            # P2b: FOV crop augmentation (before intrinsic augmentation)
            if args.augment_fov_crop_prob > 0 and random.random() < args.augment_fov_crop_prob:
                resize_imgs, intrinsics = _apply_fov_crop(
                    resize_imgs, intrinsics,
                    crop_ratio_min=args.augment_fov_crop_ratio_min,
                    crop_ratio_max=args.augment_fov_crop_ratio_max
                )
            
            if xyz_only_choise:
                pcs_np = np.array(pcs)[:, :, :3]
            else:
                pcs_np = np.array(pcs)
            if args.max_pcd_points > 0:
                pcs_np, masks = _subsample_pcd(pcs_np, masks, args.max_pcd_points)
            if args.augment_pc_jitter > 0:
                pcs_np = pcs_np + np.random.normal(0, args.augment_pc_jitter, pcs_np.shape).astype(np.float32)
            if args.augment_pc_dropout > 0:
                keep_ratio = 1.0 - np.random.uniform(0, args.augment_pc_dropout)
                B_pc = pcs_np.shape[0]
                new_pcs = []
                new_masks = []
                for b in range(B_pc):
                    mask_b = np.asarray(masks[b])
                    valid_idx = np.where(mask_b == 1)[0]
                    n_valid = len(valid_idx)
                    if n_valid <= 1:
                        new_pcs.append(pcs_np[b])
                        new_masks.append(mask_b)
                        continue
                    n_keep = max(1, int(n_valid * keep_ratio))
                    chosen = np.random.choice(n_valid, n_keep, replace=False)
                    keep_idx = valid_idx[chosen]
                    keep_idx.sort()
                    new_pc = pcs_np[b, keep_idx, :]
                    new_mask = np.ones(n_keep)
                    new_pcs.append(new_pc)
                    new_masks.append(new_mask)
                max_pts = max(pc.shape[0] for pc in new_pcs)
                padded_pcs = np.full((B_pc, max_pts, pcs_np.shape[2]), 999999, dtype=np.float32)
                padded_masks = []
                for b in range(B_pc):
                    n = new_pcs[b].shape[0]
                    padded_pcs[b, :n, :] = new_pcs[b]
                    padded_masks.append(np.concatenate([new_masks[b], np.zeros(max_pts - n)]))
                pcs_np = padded_pcs
                masks = padded_masks
            
            # P2b: LiDAR sparsification augmentation
            if args.augment_lidar_sparse_prob > 0 and random.random() < args.augment_lidar_sparse_prob:
                target_lines_choices = _parse_csv_cli_list(args.augment_lidar_sparse_lines, cast=int)
                target_lines = random.choice(target_lines_choices)
                v_fov = _parse_csv_cli_list(args.augment_lidar_vertical_fov, cast=float)
                pcs_np, masks = _apply_lidar_sparsification(
                    pcs_np, masks,
                    target_lines=target_lines,
                    vertical_fov=tuple(v_fov),
                    original_lines=128
                )
            
            if _do_detailed_profile:
                t_cpu_aug_total += time.time() - t_prep_start
                t_h2d_start = time.time()
            resize_imgs = resize_imgs.to(device, non_blocking=True)
            pcs_t = torch.from_numpy(pcs_np).float().to(device, non_blocking=True)
            masks_t = torch.from_numpy(np.array(masks)).float().to(device, non_blocking=True) if masks is not None else None
            gt_T_to_camera_t = torch.from_numpy(gt_T_to_camera_np).to(device, non_blocking=True)
            init_T_to_camera_t = torch.from_numpy(init_T_to_camera_np.astype(np.float32)).to(device, non_blocking=True)
            B_cur = gt_T_to_camera_t.shape[0]
            post_cam2ego_T = _identity_4x4.unsqueeze(0).expand(B_cur, -1, -1)
            intrinsic_matrix = torch.from_numpy(np.array(intrinsics, dtype=np.float32)).to(device, non_blocking=True)
            domain_ids_t = None
            if domain_ids_list is not None:
                domain_ids_t = torch.tensor(domain_ids_list, dtype=torch.long, device=device)
            if args.augment_intrinsic > 0:
                _cxcy = getattr(args, 'augment_intrinsic_cxcy', 0.0)
                intrinsic_matrix = _augment_intrinsics(
                    intrinsic_matrix, args.augment_intrinsic,
                    cx_cy_strength=_cxcy if _cxcy > 0 else None
                )
            if _do_detailed_profile:
                t_h2d_total += time.time() - t_h2d_start
            t_prep_total += time.time() - t_prep_start

            t_compute_start = time.time()
            is_accum_step = (batch_index + 1) % grad_accum_steps != 0 and (batch_index + 1) < len(train_loader)
            if batch_index % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            sync_ctx = model.no_sync() if (use_ddp and is_accum_step) else nullcontext()
            _bwd_ev = None
            _jac_loss_w = getattr(args, 'jacobian_loss_weight', 0.0)
            _jac_loss_start = getattr(args, 'jacobian_loss_start_epoch', 10)
            _jac_interval = max(1, int(getattr(args, 'jacobian_loss_interval', 4)))
            _jac_do_this_batch = (
                _jac_loss_w > 0 and epoch >= _jac_loss_start
                and batch_index % _jac_interval == 0)
            B_cur = resize_imgs.shape[0]
            _use_mgda = (
                getattr(args, 'use_mgda', 0) > 0
                and epoch >= getattr(args, 'mgda_start_epoch', 5))
            _mgda_tasks = {}
            # V32 consistency: split-forward approach (memory-safe).
            # Run main batch with gradients, then alt batch with no_grad for consistency loss.
            # The old 2B-concat approach OOMs on L20 (46GB) when B=16 → 2B=32.
            _cons_need_alt_forward = (_v32_init_T_alt is not None and not _v321_ema_enabled)
            if _cons_need_alt_forward:
                init_T_alt_t = torch.from_numpy(
                    _v32_init_T_alt.astype(np.float32)).to(device, non_blocking=True)
            _fwd_imgs = resize_imgs
            _fwd_pcs = pcs_t
            _fwd_gt = gt_T_to_camera_t
            _fwd_init = init_T_to_camera_t
            _fwd_post = post_cam2ego_T
            _fwd_K = intrinsic_matrix
            _fwd_masks = masks_t
            _fwd_dom = domain_ids_t
            _cons_batched = False
            if _bwd_profile_events is not None:
                _bwd_ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            with sync_ctx:
                try:
                    with autocast(enabled=use_amp, dtype=amp_dtype):
                        _v42_kwargs = {}
                        if _v42_enabled:
                            _v42_kwargs['rocr_detach'] = (epoch < _v42_rocr_detach_epochs)
                        T_pred_all, init_loss, loss = model(
                            _fwd_imgs, _fwd_pcs, _fwd_gt, _fwd_init, _fwd_post, _fwd_K,
                            masks=_fwd_masks, out_init_loss=out_init_loss_choice,
                            domain_ids=_fwd_dom, **_v42_kwargs)
                        T_pred = T_pred_all[:B_cur]
                        total_loss = loss["total_loss"]

                        # V42: corr_alignment_loss (slice to B_cur for consistency-doubled batches)
                        if _v42_corr_loss is not None and 'v42_delta_uv' in loss:
                            _ca_loss = _v42_corr_loss(
                                delta_uv_pred=loss['v42_delta_uv'][:B_cur],
                                T_gt=_fwd_gt[:B_cur],
                                T_init=_fwd_init[:B_cur],
                                xyz_groups=loss['v42_xyz_groups'][:B_cur],
                                cam_intrinsic=_fwd_K[:B_cur],
                                valid_mask=loss['v42_valid_mask'][:B_cur],
                                patch_size=loss.get('v42_patch_size', 4.0),
                                current_epoch=epoch,
                                corr_radius=getattr(args, 'cf_corr_radius', 4),
                            )
                            _ca_loss_raw = _ca_loss.item()
                            _ca_cap = 10.0
                            if _ca_loss_raw > _ca_cap:
                                _ca_loss = _ca_loss * (_ca_cap / _ca_loss_raw)
                            total_loss = total_loss + _ca_loss
                            loss['corr_alignment_loss'] = _ca_loss_raw

                        # V42: seq_consistency_loss (slice to B_cur for consistency-doubled batches)
                        if (_v42_seq_loss is not None
                                and epoch >= _v42_seq_consistency_start
                                and 'v42_rotation' in loss):
                            _rot_q = loss['v42_rotation'][:B_cur]
                            _q_list = list(_rot_q.unbind(0))
                            _sc_loss = _v42_seq_loss(_q_list)
                            total_loss = total_loss + _sc_loss
                            loss['seq_consistency_loss'] = _sc_loss.item()

                        if fd_mode == "supervision" and use_foundation_depth:
                            ds_loss = raw_model.img_branch.get_depth_supervision_loss(alpha=args.depth_sup_alpha)
                            total_loss = total_loss + ds_loss
                            loss["depth_sup_loss"] = ds_loss

                        if _cons_need_alt_forward:
                            with torch.no_grad():
                                T_pred_alt_all, _, _ = raw_model(
                                    resize_imgs, pcs_t, gt_T_to_camera_t,
                                    init_T_alt_t, post_cam2ego_T, intrinsic_matrix,
                                    masks=masks_t, out_init_loss=False)
                            T_pred_alt = T_pred_alt_all.detach()
                            R_diff = torch.bmm(
                                T_pred[:, :3, :3], T_pred_alt[:, :3, :3].transpose(1, 2))
                            trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
                            per_sample_cons = 1.0 - trace / 3.0

                            _cond_thresh_deg = getattr(args, 'conditional_consistency_threshold', 0.0)
                            if _cond_thresh_deg > 0:
                                _R_init_main = _fwd_init[:B_cur, :3, :3]
                                _R_init_alt = init_T_alt_t[:B_cur, :3, :3]
                                _R_pair_diff = torch.bmm(_R_init_main, _R_init_alt.transpose(1, 2))
                                _pair_trace = _R_pair_diff[:, 0, 0] + _R_pair_diff[:, 1, 1] + _R_pair_diff[:, 2, 2]
                                _pair_angle = torch.acos(torch.clamp((_pair_trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
                                _thresh_rad = _cond_thresh_deg * 3.14159265 / 180.0
                                _cons_mask = (_pair_angle < _thresh_rad).float()
                                if _cons_mask.sum() > 0:
                                    consistency_loss = (per_sample_cons * _cons_mask).sum() / _cons_mask.sum()
                                else:
                                    consistency_loss = torch.tensor(0.0, device=per_sample_cons.device)
                                loss["cons_mask_ratio"] = _cons_mask.mean().item()
                            else:
                                consistency_loss = per_sample_cons.mean()

                            total_loss = total_loss + _v32_consistency_w * consistency_loss
                            loss["v32_consistency_loss"] = consistency_loss.item()
                        elif _v32_init_T_alt is not None and _v321_ema_enabled:
                            init_T_alt_t = torch.from_numpy(
                                _v32_init_T_alt.astype(np.float32)).to(device, non_blocking=True)
                            with torch.no_grad():
                                T_pred_alt, _, _ = _v321_ema_model(
                                    resize_imgs, pcs_t, gt_T_to_camera_t, init_T_alt_t,
                                    post_cam2ego_T, intrinsic_matrix, masks=masks_t,
                                    out_init_loss=False, domain_ids=domain_ids_t)
                            R_diff = torch.bmm(
                                T_pred[:, :3, :3], T_pred_alt[:, :3, :3].detach().transpose(1, 2))
                            trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
                            consistency_loss = (1.0 - trace / 3.0).mean()
                            total_loss = total_loss + _v32_consistency_w * consistency_loss
                            loss["v32_consistency_loss"] = consistency_loss.item()

                        # V47: overcorrection penalty
                        _overcorr_w = getattr(args, 'overcorrection_penalty', 0.0)
                        if _overcorr_w > 0 and not _use_zero_perturb:
                            with torch.no_grad():
                                _R_pred = T_pred[:B_cur, :3, :3]
                                _R_gt = _fwd_gt[:B_cur, :3, :3]
                                _R_init = _fwd_init[:B_cur, :3, :3]
                                _tr_pred = (_R_pred @ _R_gt.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1)
                                _tr_init = (_R_init @ _R_gt.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1)
                                _ang_pred = torch.acos(torch.clamp((_tr_pred - 1) / 2, -1 + 1e-7, 1 - 1e-7))
                                _ang_init = torch.acos(torch.clamp((_tr_init - 1) / 2, -1 + 1e-7, 1 - 1e-7))
                                _overcorr_ratio = (_ang_pred > _ang_init).float().mean()
                            _overcorr_scale = 1.0 + _overcorr_w * _overcorr_ratio
                            total_loss = total_loss * _overcorr_scale
                            loss['overcorr_ratio'] = _overcorr_ratio.item()
                            loss['overcorr_scale'] = _overcorr_scale.item()

                        _zd_loss_w = _effective_zero_drift_loss_weight(args, epoch)
                        _zd_loss_start = getattr(args, 'zero_drift_loss_start_epoch', 5)
                        _task_zd = None
                        if _zd_loss_w > 0 and epoch >= _zd_loss_start and _use_zero_perturb:
                            _R_gt_zd = _fwd_gt[:B_cur, :3, :3]
                            _R_pred_zd = T_pred[:B_cur, :3, :3]
                            _zd_rpy = _zero_drift_axis_error_deg(_R_gt_zd, _R_pred_zd)
                            _zd_loss = torch.nn.functional.smooth_l1_loss(
                                _zd_rpy, torch.zeros_like(_zd_rpy), beta=0.1, reduction='mean')
                            _task_zd = _zd_loss_w * _zd_loss
                            loss['zero_drift_loss'] = _zd_loss.item()
                            loss['zero_drift_loss_weight_eff'] = _zd_loss_w

                        _inj_loss_w = getattr(args, 'inject_recovery_loss_weight', 0.0)
                        _inj_loss_start = getattr(args, 'inject_recovery_loss_start_epoch', 15)
                        _task_inj = None
                        if (_inj_loss_w > 0 and epoch >= _inj_loss_start
                                and _use_inject_recovery):
                            _R_gt_inj = _fwd_gt[:B_cur, :3, :3]
                            _R_pred_inj = T_pred[:B_cur, :3, :3]
                            _out_err_inj = _zero_drift_axis_error_deg(_R_gt_inj, _R_pred_inj)
                            _inj_loss = torch.nn.functional.smooth_l1_loss(
                                _out_err_inj, torch.zeros_like(_out_err_inj),
                                beta=0.1, reduction='mean')
                            _task_inj = _inj_loss_w * _inj_loss
                            loss['inject_recovery_loss'] = _inj_loss.item()

                        _task_photo = None
                        _lsp_w_eff = _effective_lsp_weight(args, epoch)
                        if (_v54_lsp_fn is not None and _lsp_w_eff > 0
                                and epoch >= getattr(args, 'lsp_start_epoch', 10)):
                            _lsp_out = _v54_lsp_fn(
                                _fwd_imgs[:B_cur], _fwd_pcs[:B_cur],
                                T_pred[:B_cur], _fwd_gt[:B_cur], _fwd_K[:B_cur],
                                mask=_fwd_masks[:B_cur] if _fwd_masks is not None else None)
                            _lsp_vr = float(_lsp_out['lsp_valid_ratio'].item())
                            _lsp_skipped = bool(_lsp_out.get('lsp_skipped', False))
                            loss['lsp_valid_ratio'] = _lsp_vr
                            loss['lsp_loss_weight_eff'] = _lsp_w_eff
                            if (not _lsp_skipped
                                    and torch.isfinite(_lsp_out['lsp_loss'])
                                    and _lsp_out['lsp_loss'].item() > 0):
                                _task_photo = _lsp_w_eff * _lsp_out['lsp_loss']
                                loss['lsp_loss'] = _lsp_out['lsp_loss'].item()
                            else:
                                loss['lsp_skipped'] = 1
                                loss['lsp_loss'] = 0.0

                        _task_rig = None
                        _rig_start = getattr(args, 'rig_consistency_start_epoch', 5)
                        if (_v54_rig_fn is not None and epoch >= _rig_start
                                and _use_zero_perturb
                                and domain_ids_list is not None and 'v42_rotation' in loss):
                            _seq_t = torch.tensor(
                                domain_ids_list[:B_cur], dtype=torch.long, device=device)
                            _zp_t = torch.tensor(
                                [_use_zero_perturb] * B_cur, dtype=torch.float32, device=device)
                            _rig_raw = _v54_rig_fn(
                                loss['v42_rotation'][:B_cur], _seq_t, _zp_t)
                            _task_rig = _rig_raw
                            loss['rig_consistency_loss'] = _rig_raw.item()

                        _mag_w = getattr(args, 'magnitude_loss_weight', 0.3)
                        if 'magnitude_loss' in loss and _mag_w > 0:
                            _mag_loss = loss['magnitude_loss']
                            total_loss = total_loss + _mag_w * _mag_loss
                            loss['magnitude_loss_weighted'] = (_mag_w * _mag_loss).item()

                        _gin_reg_w = getattr(args, 'gin_gate_reg_weight', 0.0)
                        _raw = model.module if hasattr(model, 'module') else model
                        if _gin_reg_w > 0 and hasattr(_raw, 'feat_in') and _raw.feat_in is not None and hasattr(_raw.feat_in, 'gate_reg_loss'):
                            _gin_reg = _raw.feat_in.gate_reg_loss()
                            total_loss = total_loss + _gin_reg_w * _gin_reg
                            loss['gin_gate_reg'] = _gin_reg.item()

                        _mgda_tasks = {'pose': total_loss}
                        if _task_zd is not None or _task_rig is not None:
                            _zd_combined = (
                                (_task_zd if _task_zd is not None else total_loss.new_tensor(0.0))
                                + (_task_rig if _task_rig is not None else total_loss.new_tensor(0.0))
                            )
                            _mgda_tasks['zd'] = _zd_combined
                        if _task_inj is not None:
                            if getattr(args, 'mgda_include_inject', 1) > 0:
                                _mgda_tasks['inject'] = _task_inj
                            else:
                                _mgda_tasks['pose'] = _mgda_tasks['pose'] + _task_inj
                        if (_task_photo is not None
                                and getattr(args, 'mgda_include_photo', 1) > 0):
                            _mgda_tasks['photo'] = _task_photo
                        elif _task_photo is not None:
                            _mgda_tasks['pose'] = _mgda_tasks['pose'] + _task_photo
                        if not _use_mgda:
                            if _task_zd is not None:
                                total_loss = total_loss + _task_zd
                            if _task_rig is not None:
                                total_loss = total_loss + _task_rig
                            if _task_inj is not None:
                                total_loss = total_loss + _task_inj
                            if _task_photo is not None:
                                total_loss = total_loss + _task_photo
                        else:
                            total_loss = sum(_mgda_tasks.values())
                            loss['mgda_n_tasks'] = len(_mgda_tasks)

                        if grad_accum_steps > 1:
                            total_loss = total_loss / grad_accum_steps
                            if _use_mgda and len(_mgda_tasks) >= 2:
                                _mgda_tasks = {k: v / grad_accum_steps for k, v in _mgda_tasks.items()}
                except RuntimeError as _fwd_err:
                    if "CUDA" in str(_fwd_err) or "illegal" in str(_fwd_err):
                        if is_main:
                            tprint(f"  [CUDA GUARD] forward() failed at epoch {epoch+1} batch {batch_index}: {_fwd_err}")
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        t_iter_start = time.time()
                        _cuda_error_count = getattr(main, '_cuda_error_count', 0) + 1
                        main._cuda_error_count = _cuda_error_count
                        if is_main:
                            tprint(f"    Cumulative CUDA errors: {_cuda_error_count}")
                        if _cuda_error_count >= 5:
                            if is_main:
                                tprint(f"  [CUDA GUARD] Too many CUDA errors ({_cuda_error_count}), saving emergency checkpoint and stopping...")
                                _emer_path = os.path.join(ckpt_save_dir, f"ckpt_emergency_ep{epoch+1}.pth")
                                try:
                                    model_to_save = model.module if use_ddp else model
                                    torch.save({
                                        'epoch': epoch + 1,
                                        'model_state_dict': model_to_save.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                                        'scaler_state_dict': scaler.state_dict(),
                                        'train_loss': train_loss,
                                        'train_noise': train_noise,
                                        'eval_noise': eval_noise,
                                        'rotation_only': rotation_only,
                                        'best_train': best_train,
                                        'best_val': best_val,
                                        'args': vars(args),
                                    }, _emer_path)
                                    tprint(f"    Emergency checkpoint saved to {_emer_path}")
                                except Exception:
                                    tprint(f"    Failed to save emergency checkpoint")
                            raise
                        continue
                    raise
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if is_main:
                        _nan_parts = []
                        for k, v in loss.items():
                            if k == "total_loss" or v is None:
                                continue
                            if torch.is_tensor(v) and v.dim() == 0 and not (torch.isnan(v) or torch.isinf(v)):
                                _nan_parts.append(f"{k}={v.item():.4f}")
                            elif not torch.is_tensor(v):
                                _nan_parts.append(f"{k}={float(v):.4f}")
                        tprint(f"  [NaN GUARD] Zeroing batch {batch_index}: " + ", ".join(_nan_parts))
                    total_loss = total_loss * 0.0
                if _bwd_ev is not None:
                    _bwd_ev[0].record()
                try:
                    if _use_mgda and len(_mgda_tasks) >= 2:
                        if is_main and not getattr(main, '_mgda_load_logged', False):
                            _, _mgda_path = _get_mgda_multitask_backward()
                            tprint(f"  [MGDA] loaded from {_mgda_path}")
                            main._mgda_load_logged = True
                        _mgda_log_batch = is_main and (batch_index == 0 or batch_index % 100 == 0)
                        if _mgda_log_batch:
                            _mgda_task_str = ", ".join(
                                f"{k}={v.item():.4f}" for k, v in _mgda_tasks.items())
                            tprint(
                                f"  [MGDA] backward start ep={epoch + 1} batch={batch_index + 1} "
                                f"n_tasks={len(_mgda_tasks)} [{_mgda_task_str}]")
                        _mgda_t0 = time.time()
                        try:
                            _mgda_loss, _mgda_meta = _mgda_build_weighted_loss(
                                _mgda_tasks, grad_accum_steps=1,
                                weight_params=_mgda_weight_params)
                            if _mgda_log_batch and 'mgda_alpha' in _mgda_meta:
                                _alpha_str = ", ".join(
                                    f"{k}={v:.3f}" for k, v in _mgda_meta['mgda_alpha'].items())
                                tprint(f"  [MGDA] weights ep={epoch + 1} batch={batch_index + 1}: {_alpha_str}")
                            scaler.scale(_mgda_loss).backward()
                        except Exception as _mgda_err:
                            if is_main:
                                tprint(
                                    f"  [MGDA] backward FAILED ep={epoch + 1} "
                                    f"batch={batch_index + 1} after {time.time() - _mgda_t0:.2f}s: "
                                    f"{type(_mgda_err).__name__}: {_mgda_err}")
                            raise
                        else:
                            if _mgda_log_batch:
                                tprint(
                                    f"  [MGDA] backward done ep={epoch + 1} batch={batch_index + 1} "
                                    f"({time.time() - _mgda_t0:.2f}s)")
                    else:
                        scaler.scale(total_loss).backward()
                except RuntimeError as _bwd_err:
                    if "CUDA" in str(_bwd_err) or "illegal" in str(_bwd_err):
                        _cuda_error_count = getattr(main, '_cuda_error_count', 0) + 1
                        main._cuda_error_count = _cuda_error_count
                        if is_main:
                            tprint(f"  [CUDA GUARD] backward() failed at batch {batch_index}: {_bwd_err}")
                            try:
                                tprint(f"    loss={total_loss.item():.4f}, scale={scaler.get_scale():.0f}")
                            except RuntimeError:
                                tprint(f"    (unable to read loss/scale after CUDA error)")
                            tprint(f"    Cumulative CUDA errors: {_cuda_error_count}")
                        optimizer.zero_grad(set_to_none=True)
                        try:
                            scaler.unscale_(optimizer)
                        except RuntimeError:
                            pass
                        try:
                            scaler.update()
                        except (AssertionError, RuntimeError):
                            if is_main:
                                tprint(f"    scaler.update() failed, manually halving scale")
                            _new_scale = max(scaler.get_scale() * 0.5, 1.0)
                            scaler._scale = torch.tensor(_new_scale).to(scaler._scale.device)
                            scaler._growth_tracker = torch.tensor(0).to(scaler._growth_tracker.device)
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                        if _cuda_error_count >= 3:
                            if is_main:
                                tprint(f"  [CUDA GUARD] Too many CUDA errors ({_cuda_error_count}), saving emergency checkpoint and stopping...")
                                _emer_path = os.path.join(ckpt_save_dir, f"ckpt_emergency_ep{epoch+1}.pth")
                                try:
                                    model_to_save = model.module if use_ddp else model
                                    torch.save({
                                        'epoch': epoch + 1,
                                        'model_state_dict': model_to_save.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                                        'scaler_state_dict': scaler.state_dict(),
                                        'best_train': best_train,
                                        'best_val': best_val,
                                        'args': vars(args),
                                    }, _emer_path)
                                    tprint(f"    Emergency checkpoint saved to {_emer_path}")
                                except Exception:
                                    tprint(f"    Failed to save emergency checkpoint")
                            raise
                        global_step += 1
                        t_iter_start = time.time()
                        continue
                    raise
                if _jac_do_this_batch:
                    # After main backward: ProjFusion clear_buffer in jacobian probe
                    # must not run while the consistency 2B forward graph is live.
                    try:
                        jac_out = _compute_jacobian_supervision_loss(
                            model, resize_imgs, pcs_t, gt_T_to_camera_t,
                            init_T_to_camera_np, post_cam2ego_T, intrinsic_matrix,
                            masks_t, args.jacobian_loss_probe_deg, use_amp, amp_dtype,
                            domain_ids_t=domain_ids_t, T_pred_center=T_pred.detach(),
                            axis_weights=jacobian_loss_axis_weights_parsed)
                        if jac_out is not None:
                            jac_sup, j_est_mean = jac_out
                            _jac_term = _jac_loss_w * jac_sup
                            if grad_accum_steps > 1:
                                _jac_term = _jac_term / grad_accum_steps
                            loss["jacobian_supervision_loss"] = jac_sup.item()
                            loss["jacobian_j_est"] = j_est_mean.item()
                            loss["jacobian_weighted"] = (_jac_loss_w * jac_sup).item()
                            scaler.scale(_jac_term).backward()
                    except RuntimeError as _jac_err:
                        if is_main:
                            tprint(f"  [JAC GUARD] jacobian backward failed at batch {batch_index}: {_jac_err}")
                        optimizer.zero_grad(set_to_none=True)
                        try:
                            scaler.unscale_(optimizer)
                        except RuntimeError:
                            pass
                        try:
                            scaler.update()
                        except (AssertionError, RuntimeError):
                            pass
                        global_step += 1
                        t_iter_start = time.time()
                        continue
                if _bwd_ev is not None:
                    _bwd_ev[1].record()
            if not is_accum_step:
                scaler.unscale_(optimizer)
                _found_inf = sum(
                    torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item()
                    for p in model.parameters() if p.grad is not None
                )
                if _found_inf > 0:
                    if is_main:
                        tprint(f"  [NaN GUARD] Inf/NaN in gradients at batch {batch_index}, "
                               f"affected params: {_found_inf}. Skipping optimizer step.")
                    if not hasattr(main, '_nan_consec_count'):
                        main._nan_consec_count = 0
                    main._nan_consec_count += 1
                    _NAN_RECOVERY_THRESH = 30
                    if main._nan_consec_count >= _NAN_RECOVERY_THRESH:
                        if is_main:
                            _ckpt_dir = os.path.join(args.log_dir, args.label, 'checkpoint')
                            _best_ckpt = os.path.join(_ckpt_dir, 'ckpt_best_val.pth')
                            if os.path.exists(_best_ckpt):
                                tprint(f"  [NaN RECOVERY] {main._nan_consec_count} consecutive NaN batches! "
                                       f"Rolling back to {_best_ckpt}")
                                _ckpt = torch.load(_best_ckpt, map_location=device)
                                raw_model.load_state_dict(_ckpt['model_state_dict'], strict=False)
                                if 'optimizer_state_dict' in _ckpt:
                                    optimizer.load_state_dict(_ckpt['optimizer_state_dict'])
                                main._nan_consec_count = 0
                                tprint(f"  [NaN RECOVERY] Model restored. Reducing LR by 0.5x.")
                                for _pg in optimizer.param_groups:
                                    _pg['lr'] *= 0.5
                            else:
                                tprint(f"  [NaN RECOVERY] {main._nan_consec_count} consecutive NaN! "
                                       f"No best_val checkpoint found, zeroing grads and continuing.")
                                main._nan_consec_count = 0
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    global_step += 1
                    t_iter_start = time.time()
                    continue
                else:
                    if hasattr(main, '_nan_consec_count'):
                        main._nan_consec_count = 0
                if is_main and batch_index % 50 == 0:
                    bb_gn, bev_gn, hd_gn, mod_gn = _compute_grad_norms()
                    _epoch_grad_accum['bb'].append(bb_gn)
                    _epoch_grad_accum['bev'].append(bev_gn)
                    _epoch_grad_accum['hd'].append(hd_gn)
                    for m, v in mod_gn.items():
                        _epoch_grad_accum['mod'].setdefault(m, []).append(v)
                    if writer is not None:
                        writer.add_scalar('GradNorm/backbone', bb_gn, global_step)
                        writer.add_scalar('GradNorm/bev_branch', bev_gn, global_step)
                        writer.add_scalar('GradNorm/head', hd_gn, global_step)
                        writer.add_scalar('GradNorm/ratio_hd_bb', hd_gn / max(bb_gn, 1e-10), global_step)
                if _v42_enabled:
                    _corr_params = [p for n, p in raw_model.named_parameters()
                                    if 'corr' in n and p.grad is not None]
                    if _corr_params:
                        torch.nn.utils.clip_grad_norm_(_corr_params, max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                if _bwd_ev is not None:
                    _bwd_ev[2].record()
                scaler.step(optimizer)
                scaler.update()
                if _v321_ema_enabled and _v321_ema_model is not None:
                    _tau = _v321_ema_decay
                    with torch.no_grad():
                        for p_online, p_ema in zip(raw_model.parameters(), _v321_ema_model.parameters()):
                            p_ema.data.mul_(_tau).add_(p_online.data, alpha=1 - _tau)
                        for b_online, b_ema in zip(raw_model.buffers(), _v321_ema_model.buffers()):
                            b_ema.data.copy_(b_online.data)
                if _bwd_ev is not None:
                    _bwd_ev[3].record()

            if _bwd_ev is not None:
                _bwd_profile_events.append((_bwd_ev, not is_accum_step))
            t_compute_total += time.time() - t_compute_start
            
            with torch.no_grad():
                batch_errors = compute_batch_pose_errors(T_pred, gt_T_to_camera_t)
                for key in epoch_pose_errors:
                    epoch_pose_errors[key] += batch_errors[key]
                if is_main:
                    _accumulate_fusion_gate_stats(raw_model, _gate_stats)
            
            for key in loss.keys():
                _val = loss[key]
                if _val is None:
                    continue
                if hasattr(_val, 'dim') and _val.dim() > 0:
                    continue
                _lv = _val.item() if hasattr(_val, 'item') else float(_val)
                if key not in train_loss:
                    train_loss[key] = _lv
                else:
                    train_loss[key] += _lv
            
            if init_loss is not None:
                for key in init_loss.keys():
                    train_key = f"init_{key}"
                    if train_key not in train_loss:
                        train_loss[train_key] = init_loss[key].item()
                    else:
                        train_loss[train_key] += init_loss[key].item()

            if batch_index % 10 == 0 and is_main:
                _v32_suffix = ""
                if _v32_enabled:
                    _v32_parts = [f"ang={_current_angle_range:.1f}°"]
                    _v32_suffix = f" | v32[{' '.join(_v32_parts)}]"
                _corr_w = float(getattr(args, 'correspondence_loss_weight', 0.0) or 0.0)
                _loss_head = _format_step_loss_head(
                    total_loss, loss, batch_errors, rotation_only, _corr_w)
                _gmp_suffix = _format_gmp_step_suffix(loss, _corr_w)
                tprint(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index+1}/{len(train_loader)}], "
                       f"{_loss_head}{_v32_suffix}{_gmp_suffix}")
                if writer is not None:
                    _log_gmp_step_scalars(writer, loss, global_step)
            
            if args.enable_vis > 0 and batch_index % args.vis_freq == 0 and is_main:
                t_vis_start = time.time()
                with torch.no_grad():
                    imgs_np = np.array(imgs)
                    masks_np = np.array(masks)
                    T_pred_np = T_pred.detach().cpu().numpy()
                    debug_vis = (epoch == 0 and batch_index == 0)
                    
                    vis_image = visualize_batch_projection(
                        images=imgs_np,
                        points_batch=pcs_np,
                        init_T_batch=init_T_to_camera_np,
                        gt_T_batch=gt_T_to_camera_np,
                        pred_T_batch=T_pred_np,
                        K_batch=np.array(intrinsics),
                        masks=masks_np,
                        num_samples=args.vis_samples,
                        max_points=args.vis_points,
                        point_radius=args.vis_point_radius,
                        debug=debug_vis,
                        rotation_only=rotation_only,
                        phase="Train",
                        epoch=epoch + 1,
                        epoch_train_errors=last_epoch_train_errors,
                        epoch_val_errors=last_epoch_val_errors,
                    )
                    
                    vis_image_tb = prepare_image_for_tensorboard(vis_image)
                    writer.add_image('Train/Projection', vis_image_tb, global_step)
                    
                    if not rotation_only:
                        writer.add_scalar('Train/PoseError/trans_error_m', batch_errors['trans_error'], global_step)
                        writer.add_scalar('Train/PoseError/fwd_error_m', batch_errors['fwd_error'], global_step)
                        writer.add_scalar('Train/PoseError/lat_error_m', batch_errors['lat_error'], global_step)
                        writer.add_scalar('Train/PoseError/ht_error_m', batch_errors['ht_error'], global_step)
                    writer.add_scalar('Train/PoseError/rot_error_deg', batch_errors['rot_error'], global_step)
                    writer.add_scalar('Train/PoseError/roll_error_deg', batch_errors['roll_error'], global_step)
                    writer.add_scalar('Train/PoseError/pitch_error_deg', batch_errors['pitch_error'], global_step)
                    writer.add_scalar('Train/PoseError/yaw_error_deg', batch_errors['yaw_error'], global_step)
                t_vis_total += time.time() - t_vis_start
                vis_count += 1
            
            processed_batches += 1
            global_step += 1
            t_iter_start = time.time()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        if is_main:
            steps_per_sec = len(train_loader) / epoch_time
            t_other = epoch_time - t_data_total - t_prep_total - t_compute_total - t_vis_total
            elapsed_total = time.time() - training_start_time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            eta_s = avg_epoch_time * (num_epochs - epoch - 1)
            tprint(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.1f}s ({epoch_time/60:.1f}min), {steps_per_sec:.2f} steps/s, {epoch_time/len(train_loader):.2f}s/step"
                   f"  [elapsed={elapsed_total/3600:.1f}h, avg={avg_epoch_time:.1f}s/ep, ETA={eta_s/3600:.1f}h]")
            tprint(f"  Profiling: data_load={t_data_total:.1f}s({t_data_total/epoch_time*100:.1f}%), "
                   f"prep={t_prep_total:.1f}s({t_prep_total/epoch_time*100:.1f}%), "
                   f"compute={t_compute_total:.1f}s({t_compute_total/epoch_time*100:.1f}%), "
                   f"vis={t_vis_total:.1f}s({t_vis_total/epoch_time*100:.1f}%, {vis_count}calls, {t_vis_total/max(vis_count,1):.1f}s/call), "
                   f"other={t_other:.1f}s({t_other/epoch_time*100:.1f}%)")
            if _do_detailed_profile:
                tprint(f"  Prep detail: cpu_aug={t_cpu_aug_total:.1f}s({t_cpu_aug_total/epoch_time*100:.1f}%), "
                       f"h2d_transfer={t_h2d_total:.1f}s({t_h2d_total/epoch_time*100:.1f}%)")
            mod_prof = raw_model.get_module_profile(reset=True) if hasattr(raw_model, 'get_module_profile') else {}
            if mod_prof:
                total_ms = mod_prof.get("total", 1)
                parts = " | ".join(f"{k}={v:.1f}ms({v/total_ms*100:.0f}%)" for k, v in mod_prof.items() if k != "total")
                tprint(f"  Module forward: {parts} | total={total_ms:.1f}ms")
            if _bwd_profile_events:
                torch.cuda.synchronize()
                bwd_ms_sum, clip_ms_sum, optim_ms_sum = 0.0, 0.0, 0.0
                bwd_count, optim_count = 0, 0
                for ev_list, has_optim_step in _bwd_profile_events:
                    bwd_ms_sum += ev_list[0].elapsed_time(ev_list[1])
                    bwd_count += 1
                    if has_optim_step:
                        clip_ms_sum += ev_list[1].elapsed_time(ev_list[2])
                        optim_ms_sum += ev_list[2].elapsed_time(ev_list[3])
                        optim_count += 1
                bwd_avg = bwd_ms_sum / max(bwd_count, 1)
                clip_avg = clip_ms_sum / max(optim_count, 1)
                optim_avg = optim_ms_sum / max(optim_count, 1)
                fwd_avg = mod_prof.get("total", 0) if mod_prof else 0
                total_step = fwd_avg + bwd_avg + clip_avg + optim_avg
                tprint(f"  Module backward: bwd={bwd_avg:.1f}ms | grad_clip={clip_avg:.1f}ms | optim_step={optim_avg:.1f}ms")
                if total_step > 0:
                    tprint(f"  Compute breakdown: fwd={fwd_avg:.1f}ms({fwd_avg/total_step*100:.0f}%) | "
                           f"bwd={bwd_avg:.1f}ms({bwd_avg/total_step*100:.0f}%) | "
                           f"clip+optim={clip_avg+optim_avg:.1f}ms({(clip_avg+optim_avg)/total_step*100:.0f}%) | "
                           f"total={total_step:.1f}ms/step")
                _bwd_profile_events = None
            if _epoch_grad_accum['bb']:
                ga = _epoch_grad_accum
                avg_bb = sum(ga['bb']) / len(ga['bb'])
                avg_hd = sum(ga['hd']) / len(ga['hd'])
                ratio = avg_hd / max(avg_bb, 1e-10)
                eff_ratio = ratio / args.backbone_lr_scale if args.backbone_lr_scale > 0 else float('inf')
                tprint(f"  Grad norms (pre-clip avg): backbone={avg_bb:.4f}, head={avg_hd:.4f}, "
                       f"ratio={ratio:.1f}x, eff_update_ratio={eff_ratio:.1f}x")
                top_mods = sorted(ga['mod'].items(), key=lambda x: -sum(x[1])/len(x[1]))[:5]
                mod_str = " | ".join(f"{m}={sum(v)/len(v):.2f}" for m, v in top_mods)
                tprint(f"  Per-module grad: {mod_str}")

        if scheduler_choice:   
            scheduler.step()    
        
        if is_main:
            effective_batches = max(1, processed_batches)
            for key in train_loss.keys():
                train_loss[key] /= effective_batches
                if rotation_only and 'translation' in key:
                    continue
                # 为各个损失添加单位说明
                if key == "total_loss":
                    unit_str = " (weighted sum)"
                elif key == "rotation_loss":
                    unit_str = "° (for display; total_loss uses radians)"
                elif key == "geodesic_loss":
                    unit_str = "° (geodesic; drives training)"
                elif key == "PC_reproj_loss":
                    unit_str = " (point cloud reprojection)"
                elif key == "quat_norm_loss":
                    unit_str = " (quaternion normalization)"
                elif key == "translation_loss":
                    unit_str = "m"
                elif key == "correspondence_loss":
                    unit_str = "px (L_corr)"
                elif key == "match_valid_ratio":
                    unit_str = " (match inlier ratio, GT-masked alias)"
                elif key == "match_valid_ratio_init":
                    unit_str = " (match inlier ratio, T_init only)"
                elif key == "match_valid_ratio_gt":
                    unit_str = " (match inlier ratio, GT-masked, gates fallback)"
                elif key == "match_fallback_ratio":
                    unit_str = " (EPnP fallback to refine-only, uses valid_gt)"
                elif key == "epnp_insufficient_ratio":
                    unit_str = " (EPnP returned identity: weight count < min_points)"
                elif key == "epnp_mean_effective_points":
                    unit_str = " (mean EPnP points with weight > 1e-4)"
                elif key == "epnp_grad_detached":
                    unit_str = " (1=warmup, pose grad off EPnP)"
                elif key == "corr_valid_ratio":
                    unit_str = " (local corr window valid)"
                elif key == "geo_valid_ratio":
                    unit_str = " (geo consistency valid)"
                else:
                    unit_str = ""
                
                tprint(f"Epoch [{epoch+1}/{num_epochs}], Train Loss {key}: {train_loss[key]:.4f}{unit_str}")
                writer.add_scalar(f"Loss/train/{key}", train_loss[key], epoch)
            
            for key in epoch_pose_errors:
                epoch_pose_errors[key] /= effective_batches
            last_epoch_train_errors = dict(epoch_pose_errors)
            
            if rotation_only:
                tprint(f"Epoch [{epoch+1}/{num_epochs}], Train Pose Error - "
                       f"Rot: {epoch_pose_errors['rot_error']:.2f}° "
                       f"(Roll:{epoch_pose_errors['roll_error']:.2f}° Pitch:{epoch_pose_errors['pitch_error']:.2f}° Yaw:{epoch_pose_errors['yaw_error']:.2f}°)")
            else:
                tprint(f"Epoch [{epoch+1}/{num_epochs}], Train Pose Error - "
                       f"Trans: {epoch_pose_errors['trans_error']:.4f}m "
                       f"(Fwd:{epoch_pose_errors['fwd_error']:.4f}m Lat:{epoch_pose_errors['lat_error']:.4f}m Ht:{epoch_pose_errors['ht_error']:.4f}m), "
                       f"Rot: {epoch_pose_errors['rot_error']:.2f}° "
                       f"(Roll:{epoch_pose_errors['roll_error']:.2f}° Pitch:{epoch_pose_errors['pitch_error']:.2f}° Yaw:{epoch_pose_errors['yaw_error']:.2f}°)")
            
            if not rotation_only:
                writer.add_scalar('Epoch/train/trans_error_m', epoch_pose_errors['trans_error'], epoch)
                writer.add_scalar('Epoch/train/fwd_error_m', epoch_pose_errors['fwd_error'], epoch)
                writer.add_scalar('Epoch/train/lat_error_m', epoch_pose_errors['lat_error'], epoch)
                writer.add_scalar('Epoch/train/ht_error_m', epoch_pose_errors['ht_error'], epoch)
            writer.add_scalar('Epoch/train/rot_error_deg', epoch_pose_errors['rot_error'], epoch)
            writer.add_scalar('Epoch/train/roll_error_deg', epoch_pose_errors['roll_error'], epoch)
            writer.add_scalar('Epoch/train/pitch_error_deg', epoch_pose_errors['pitch_error'], epoch)
            writer.add_scalar('Epoch/train/yaw_error_deg', epoch_pose_errors['yaw_error'], epoch)

            if _gate_stats['gate_count'] > 0:
                _gn = _gate_stats['gate_count']
                _mean_bev = _gate_stats['gate_bev_sum'] / _gn
                _mean_proj = _gate_stats['gate_proj_sum'] / _gn
                _mean_ent = _gate_stats['gate_entropy_sum'] / _gn
                writer.add_scalar('Epoch/train/gate_bev_mean', _mean_bev, epoch)
                writer.add_scalar('Epoch/train/gate_proj_mean', _mean_proj, epoch)
                writer.add_scalar('Epoch/train/gate_entropy', _mean_ent, epoch)
                tprint(f"Epoch [{epoch+1}/{num_epochs}], Gate mean: "
                       f"bev={_mean_bev:.3f} proj={_mean_proj:.3f} entropy={_mean_ent:.3f} "
                       f"(collapse if entropy→0 or weight→1)")
            
            cur_train_loss = train_loss.get('total_loss', float('inf')) if train_loss else float('inf')
            if rotation_only:
                cur_score = epoch_pose_errors['rot_error']
                best_score = best_train['rot']
            else:
                cur_score = epoch_pose_errors['trans_error'] + epoch_pose_errors['rot_error'] * 0.1
                best_score = best_train['trans'] + best_train['rot'] * 0.1
            if cur_score < best_score:
                best_train.update({
                    'epoch': epoch + 1,
                    'loss': cur_train_loss,
                    'trans': epoch_pose_errors['trans_error'],
                    'rot': epoch_pose_errors['rot_error'],
                    'errors': dict(epoch_pose_errors),
                })
        
        if is_main and (epoch == num_epochs - 1 or (args.save_ckpt_per_epoches > 0 and (epoch + 1) % args.save_ckpt_per_epoches == 0)):
            ckpt_path = os.path.join(ckpt_save_dir, f"ckpt_{epoch+1}.pth")
            model_to_save = model.module if use_ddp else model
            _ckpt_data = {
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'train_noise': train_noise,
                'eval_noise': eval_noise,
                'rotation_only': rotation_only,
                'epoch_train_errors': last_epoch_train_errors,
                'epoch_val_errors': last_epoch_val_errors,
                'best_train': best_train,
                'best_val': best_val,
                'best_medw': best_medw,
                'best_dual': best_dual,
                'best_jacobian': best_jacobian,
                'kpi_history': kpi_history,
                'early_stop_counter': early_stop_counter,
                'jacobian_early_stop_counter': jacobian_early_stop_counter,
                'args': vars(args),
            }
            if _v321_ema_enabled and _v321_ema_model is not None:
                _ckpt_data['ema_state_dict'] = _v321_ema_model.state_dict()
            _ckpt_data.update(_build_ckpt_metadata(model, args))
            torch.save(_ckpt_data, ckpt_path)
            tprint(f"Checkpoint saved to {ckpt_path}")
            
            if args.enable_ckpt_eval > 0 and is_main:
                ckpt_eval_dir = os.path.join(ckpt_save_dir, f"ckpt_{epoch+1}_eval")
                os.makedirs(ckpt_eval_dir, exist_ok=True)
                
                tprint(f"Evaluating checkpoint {epoch+1} on validation set and saving visualization to {ckpt_eval_dir}...")
                raw_model.eval()
                
                # 使用验证集的噪声范围
                eval_trans_range = eval_noise["trans_range"]
                eval_angle_range = eval_noise["angle_range_deg"]
                
                # 创建外参结果文件
                extrinsics_file = os.path.join(ckpt_eval_dir, "extrinsics_and_errors.txt")
                
                # 累积误差统计
                all_errors = {
                    'trans_error': [], 'fwd_error': [], 'lat_error': [], 'ht_error': [],
                    'rot_error': [], 'roll_error': [], 'pitch_error': [], 'yaw_error': []
                }
                gt_extrinsics_written = False
                
                sample_count = 0
                with torch.no_grad():
                    for batch_index, batch_data in enumerate(val_loader):
                        if batch_index >= 5 or batch_data is None:
                            break
                        imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data[:5]
                        
                        gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
                        init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(
                            gt_T_to_camera_np, 
                            angle_range_deg=eval_angle_range, 
                            trans_range=eval_trans_range,
                            rotation_only=rotation_only,
                            distribution=args.perturb_distribution,
                            per_axis_prob=args.per_axis_prob,
                            per_axis_weights=per_axis_weights_parsed,
                        )
                        
                        resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device, non_blocking=True)
                        if xyz_only_choise:
                            pcs_np = np.array(pcs)[:, :, :3]
                        else:
                            pcs_np = np.array(pcs)
                        if args.max_pcd_points > 0:
                            pcs_np, masks = _subsample_pcd(pcs_np, masks, args.max_pcd_points)
                        pcs = torch.from_numpy(pcs_np).float().to(device, non_blocking=True)
                        gt_T_to_camera = torch.from_numpy(gt_T_to_camera_np).float().to(device, non_blocking=True)
                        init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device, non_blocking=True)
                        B_cur = gt_T_to_camera.shape[0]
                        post_cam2ego_T = _identity_4x4.unsqueeze(0).expand(B_cur, -1, -1)
                        intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device, non_blocking=True)
                        masks_t_vis = torch.from_numpy(np.array(masks)).float().to(device, non_blocking=True) if masks is not None else None
                        
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            T_pred, _, _ = raw_model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks_t_vis, out_init_loss=False)
                        
                        imgs_np = np.array(imgs)
                        masks_np = np.array(masks)
                        T_pred_np = T_pred.detach().cpu().numpy()
                        
                        # 为每个样本生成可视化和保存外参
                        for i in range(len(imgs_np)):
                            sample_idx = sample_count + i
                            
                            vis_image = visualize_batch_projection(
                                images=imgs_np[i:i+1],
                                points_batch=pcs_np[i:i+1],
                                init_T_batch=init_T_to_camera_np[i:i+1],
                                gt_T_batch=gt_T_to_camera_np[i:i+1],
                                pred_T_batch=T_pred_np[i:i+1],
                                K_batch=np.array(intrinsics)[i:i+1],
                                masks=masks_np[i:i+1],
                                num_samples=1,
                                max_points=args.vis_points,
                                point_radius=args.vis_point_radius,
                                rotation_only=rotation_only,
                                phase="Eval",
                                epoch=epoch + 1,
                                epoch_train_errors=last_epoch_train_errors,
                                epoch_val_errors=last_epoch_val_errors,
                            )
                            
                            # 保存可视化图像
                            vis_image_path = os.path.join(ckpt_eval_dir, f"sample_{sample_idx:04d}_projection.png")
                            cv2.imwrite(vis_image_path, vis_image)
                            
                            # 计算误差
                            errors = compute_pose_errors(T_pred_np[i], gt_T_to_camera_np[i])
                            
                            # 累积误差
                            for key in all_errors:
                                all_errors[key].append(errors[key])
                            
                            # 保存外参和误差信息
                            with open(extrinsics_file, 'a') as f:
                                # 只在第一次写入文件头和 GT
                                if not gt_extrinsics_written:
                                    f.write(f"Checkpoint: epoch_{epoch+1}\n")
                                    f.write(f"Evaluation on validation set (perturbation: {eval_angle_range}deg, {eval_trans_range}m)\n")
                                    f.write(f"="*80 + "\n\n")
                                    
                                    f.write("Ground Truth Extrinsics (LiDAR → Camera):\n")
                                    for row in gt_T_to_camera_np[i]:
                                        f.write(f"  {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
                                    f.write("\n" + "="*80 + "\n\n")
                                    gt_extrinsics_written = True
                                
                                f.write(f"Sample {sample_idx:04d}\n")
                                f.write("-" * 80 + "\n")
                                
                                f.write("\nPredicted Extrinsics (LiDAR → Camera):\n")
                                for row in T_pred_np[i]:
                                    f.write(f"  {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
                                
                                if not rotation_only:
                                    f.write("\nTranslation Errors (in LiDAR coordinate system):\n")
                                    f.write(f"  Total:   {errors['trans_error']:.6f} m\n")
                                    f.write(f"  X (Fwd): {errors['fwd_error']:.6f} m\n")
                                    f.write(f"  Y (Lat): {errors['lat_error']:.6f} m\n")
                                    f.write(f"  Z (Ht):  {errors['ht_error']:.6f} m\n")
                                
                                f.write("\nRotation Errors (axis-angle, LiDAR frame: X=Fwd, Y=Left, Z=Up):\n")
                                f.write(f"  Total:            {errors['rot_error']:.6f} deg\n")
                                f.write(f"  Roll  (LiDAR X):  {errors['roll_error']:.6f} deg\n")
                                f.write(f"  Pitch (LiDAR Y):  {errors['pitch_error']:.6f} deg\n")
                                f.write(f"  Yaw   (LiDAR Z):  {errors['yaw_error']:.6f} deg\n")
                                
                                f.write("\n" + "="*80 + "\n\n")
                        
                        sample_count += len(imgs_np)
                
                # 写入平均误差统计
                with open(extrinsics_file, 'a') as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write("AVERAGE ERRORS ACROSS ALL SAMPLES\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Total samples evaluated: {sample_count}\n\n")
                    
                    avg_errors = {key: np.mean(values) for key, values in all_errors.items()}
                    std_errors = {key: np.std(values) for key, values in all_errors.items()}
                    
                    if not rotation_only:
                        f.write("Average Translation Errors (in LiDAR coordinate system):\n")
                        f.write(f"  Total:   {avg_errors['trans_error']:.6f} ± {std_errors['trans_error']:.6f} m\n")
                        f.write(f"  X (Fwd): {avg_errors['fwd_error']:.6f} ± {std_errors['fwd_error']:.6f} m\n")
                        f.write(f"  Y (Lat): {avg_errors['lat_error']:.6f} ± {std_errors['lat_error']:.6f} m\n")
                        f.write(f"  Z (Ht):  {avg_errors['ht_error']:.6f} ± {std_errors['ht_error']:.6f} m\n")
                    
                    f.write("\nAverage Rotation Errors (axis-angle, LiDAR frame: X=Fwd, Y=Left, Z=Up):\n")
                    f.write(f"  Total:            {avg_errors['rot_error']:.6f} ± {std_errors['rot_error']:.6f} deg\n")
                    f.write(f"  Roll  (LiDAR X):  {avg_errors['roll_error']:.6f} ± {std_errors['roll_error']:.6f} deg\n")
                    f.write(f"  Pitch (LiDAR Y):  {avg_errors['pitch_error']:.6f} ± {std_errors['pitch_error']:.6f} deg\n")
                    f.write(f"  Yaw   (LiDAR Z):  {avg_errors['yaw_error']:.6f} ± {std_errors['yaw_error']:.6f} deg\n")
                    f.write("\n" + "="*80 + "\n")
                
                avg_eval_errors = {key: np.mean(values) for key, values in all_errors.items()}
                if rotation_only:
                    tprint(f"Checkpoint {epoch+1} Eval Pose Error - "
                           f"Rot: {avg_eval_errors['rot_error']:.2f}° "
                           f"(R:{avg_eval_errors['roll_error']:.2f} P:{avg_eval_errors['pitch_error']:.2f} Y:{avg_eval_errors['yaw_error']:.2f})")
                else:
                    tprint(f"Checkpoint {epoch+1} Eval Pose Error - "
                           f"Trans: {avg_eval_errors['trans_error']:.4f}m "
                           f"(Fwd:{avg_eval_errors['fwd_error']:.4f} Lat:{avg_eval_errors['lat_error']:.4f} Ht:{avg_eval_errors['ht_error']:.4f}), "
                           f"Rot: {avg_eval_errors['rot_error']:.2f}° "
                           f"(R:{avg_eval_errors['roll_error']:.2f} P:{avg_eval_errors['pitch_error']:.2f} Y:{avg_eval_errors['yaw_error']:.2f})")
                tprint(f"Checkpoint evaluation complete: {sample_count} samples saved to {ckpt_eval_dir}")
                
                checkpoint_records.append({
                    'epoch': epoch + 1,
                    'train': dict(epoch_pose_errors),
                    'eval': dict(avg_eval_errors),
                })
            else:
                checkpoint_records.append({
                    'epoch': epoch + 1,
                    'train': dict(epoch_pose_errors),
                    'eval': None,
                })

        train_loss = None
        init_loss = None
        loss = None

        if use_ddp:
            dist.barrier()

        if epoch % args.eval_epoches == 0:
            torch.cuda.empty_cache()
            eval_trans_range = eval_noise["trans_range"]
            eval_angle_range = eval_noise["angle_range_deg"]
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            raw_model.eval()
            val_loss = {}
            val_pose_errors = {
                'trans_error': 0, 'fwd_error': 0, 'lat_error': 0, 'ht_error': 0,
                'rot_error': 0, 'roll_error': 0, 'pitch_error': 0, 'yaw_error': 0,
            }
            val_processed_batches = 0
            medw_T_pred, medw_T_gt, medw_seqs = [], [], []
            medw_local_idx = 0
            if use_ddp:
                rank_val_indices = _ddp_val_rank_indices(
                    len(val_dataset), dist.get_rank(), dist.get_world_size())
            else:
                rank_val_indices = list(range(len(val_dataset)))

            jacobian_axis_accum = (
                {'roll': [], 'pitch': [], 'yaw': []}
                if args.enable_jacobian_eval > 0 else None
            )

            with torch.no_grad():
                for batch_index, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data[:5]
                    gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
                    init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(
                        gt_T_to_camera_np, angle_range_deg=eval_angle_range, trans_range=eval_trans_range,
                        rotation_only=rotation_only, distribution=args.perturb_distribution,
                        per_axis_prob=args.per_axis_prob, per_axis_weights=per_axis_weights_parsed)
                    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device, non_blocking=True)
                    if xyz_only_choise:
                        pcs_np = np.array(pcs)[:, :, :3]
                    else:
                        pcs_np = np.array(pcs)
                    if args.max_pcd_points > 0:
                        pcs_np, masks = _subsample_pcd(pcs_np, masks, args.max_pcd_points)
                    pcs = torch.from_numpy(pcs_np).float().to(device, non_blocking=True)
                    gt_T_to_camera = torch.from_numpy(gt_T_to_camera_np).float().to(device, non_blocking=True)
                    init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device, non_blocking=True)
                    B_cur = gt_T_to_camera.shape[0]
                    post_cam2ego_T = _identity_4x4.unsqueeze(0).expand(B_cur, -1, -1)
                    intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device, non_blocking=True)
                    masks_t_val = torch.from_numpy(np.array(masks)).float().to(device, non_blocking=True) if masks is not None else None
                    with autocast(enabled=use_amp, dtype=amp_dtype):
                        T_pred, init_loss, loss = raw_model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks_t_val, out_init_loss=False)

                    batch_errors = compute_batch_pose_errors(T_pred, gt_T_to_camera)
                    for key in val_pose_errors:
                        val_pose_errors[key] += batch_errors[key]

                    # Jacobian: piggyback on already-loaded val batch (no 2nd dataloader pass).
                    # Extra cost = jacobian_eval_batches × 3 axes × n_probes forwards per eval epoch.
                    if (jacobian_axis_accum is not None
                            and batch_index < args.jacobian_eval_batches):
                        j_batch = _compute_jacobian_one_batch(
                            raw_model, imgs, pcs_np, masks, gt_T_to_camera_np, intrinsics,
                            device, args.jacobian_eval_angle_deg, args.jacobian_eval_n_probes,
                            use_amp, amp_dtype, _identity_4x4, xyz_only_choise)
                        for k, v in j_batch.items():
                            if v == v:
                                jacobian_axis_accum[k].append(v)

                    if args.enable_medw_eval > 0 and val_idx_to_seq is not None:
                        T_pred_np_medw = T_pred.detach().cpu().numpy()
                        for i in range(B_cur):
                            global_val_idx = rank_val_indices[medw_local_idx + i]
                            medw_seqs.append(val_idx_to_seq.get(global_val_idx, 'unknown'))
                            medw_T_pred.append(T_pred_np_medw[i].copy())
                            medw_T_gt.append(gt_T_to_camera_np[i].copy())
                        medw_local_idx += B_cur

                    for key in loss.keys():
                        _val = loss[key]
                        if _val is None:
                            continue
                        if hasattr(_val, 'dim') and _val.dim() > 0:
                            continue
                        val_key = key
                        _lv = _val.item() if hasattr(_val, 'item') else float(_val)
                        if val_key not in val_loss.keys():
                            val_loss[val_key] = _lv
                        else:
                            val_loss[val_key] += _lv
                    if init_loss is not None:
                        for key in init_loss.keys():
                            val_key = f"init_{key}"
                            if val_key not in val_loss.keys():
                                val_loss[val_key] = init_loss[key].item()
                            else:
                                val_loss[val_key] += init_loss[key].item()

                    val_processed_batches += 1

                    if is_main and args.enable_vis > 0 and batch_index == 0:
                        imgs_np = np.array(imgs)
                        masks_np = np.array(masks)
                        T_pred_np = T_pred.detach().cpu().numpy()

                        vis_image = visualize_batch_projection(
                            images=imgs_np,
                            points_batch=pcs_np,
                            init_T_batch=init_T_to_camera_np,
                            gt_T_batch=gt_T_to_camera_np,
                            pred_T_batch=T_pred_np,
                            K_batch=np.array(intrinsics),
                            masks=masks_np,
                            num_samples=args.vis_samples,
                            max_points=args.vis_points,
                            point_radius=args.vis_point_radius,
                            rotation_only=rotation_only,
                            phase="Val",
                            epoch=epoch + 1,
                            epoch_train_errors=last_epoch_train_errors,
                            epoch_val_errors=last_epoch_val_errors,
                        )

                        vis_image_tb = prepare_image_for_tensorboard(vis_image)
                        writer.add_image('Val/Projection', vis_image_tb, epoch)

            # DDP: 聚合各 rank 的 val 统计
            if use_ddp:
                val_processed_batches = int(_ddp_all_reduce_scalar(val_processed_batches, device))
                val_loss = _ddp_all_reduce_dict_sum(val_loss, device)
                val_pose_errors = _ddp_all_reduce_dict_sum(val_pose_errors, device)
                if args.enable_medw_eval > 0 and medw_T_pred:
                    gathered_medw = _ddp_gather_object((medw_T_pred, medw_T_gt, medw_seqs))
                    if is_main:
                        medw_T_pred, medw_T_gt, medw_seqs = [], [], []
                        for pred, gt, seqs in gathered_medw:
                            medw_T_pred.extend(pred)
                            medw_T_gt.extend(gt)
                            medw_seqs.extend(seqs)

            jac_result = None
            if jacobian_axis_accum is not None:
                if use_ddp:
                    _jac_gather = _ddp_gather_object(jacobian_axis_accum)
                    if is_main:
                        _merged_acc = {'roll': [], 'pitch': [], 'yaw': []}
                        for acc in _jac_gather:
                            for k in _merged_acc:
                                _merged_acc[k].extend(acc.get(k, []))
                        jac_result = _finalize_jacobian_axis_accum(_merged_acc)
                else:
                    jac_result = _finalize_jacobian_axis_accum(jacobian_axis_accum)

            raw_model.train()

            if is_main:
                if rotation_only:
                    val_label = f"Val[±{eval_angle_range}°]"
                else:
                    val_label = f"Val[±{eval_angle_range}°, ±{eval_trans_range}m]"

                effective_val_batches = max(1, val_processed_batches)
                for key in val_loss.keys():
                    val_loss[key] /= effective_val_batches
                    if rotation_only and 'translation' in key:
                        continue
                    if key == "total_loss":
                        unit_str = " (weighted sum)"
                    elif key == "rotation_loss":
                        unit_str = "° (for display; total_loss uses radians)"
                    elif key == "geodesic_loss":
                        unit_str = "° (geodesic; drives training)"
                    elif key == "PC_reproj_loss":
                        unit_str = " (point cloud reprojection)"
                    elif key == "quat_norm_loss":
                        unit_str = " (quaternion normalization)"
                    elif key == "translation_loss":
                        unit_str = "m"
                    elif key == "correspondence_loss":
                        unit_str = "px (L_corr)"
                    elif key == "match_valid_ratio":
                        unit_str = " (match inlier ratio, GT-masked alias)"
                    elif key == "match_valid_ratio_init":
                        unit_str = " (match inlier ratio, T_init only)"
                    elif key == "match_valid_ratio_gt":
                        unit_str = " (match inlier ratio, GT-masked, gates fallback)"
                    elif key == "match_fallback_ratio":
                        unit_str = " (EPnP fallback to refine-only, uses valid_gt)"
                    elif key == "epnp_insufficient_ratio":
                        unit_str = " (EPnP returned identity: weight count < min_points)"
                    elif key == "epnp_mean_effective_points":
                        unit_str = " (mean EPnP points with weight > 1e-4)"
                    elif key == "epnp_grad_detached":
                        unit_str = " (1=warmup, pose grad off EPnP)"
                    elif key == "corr_valid_ratio":
                        unit_str = " (local corr window valid)"
                    elif key == "geo_valid_ratio":
                        unit_str = " (geo consistency valid)"
                    else:
                        unit_str = ""

                    tprint(f"Epoch [{epoch+1}/{num_epochs}], {val_label} Loss {key}: {val_loss[key]:.4f}{unit_str}")
                    writer.add_scalar(f"Loss/val/{key}", val_loss[key], epoch)

                for key in val_pose_errors:
                    val_pose_errors[key] /= effective_val_batches
                last_epoch_val_errors = dict(val_pose_errors)

                if rotation_only:
                    tprint(f"Epoch [{epoch+1}/{num_epochs}], {val_label} Pose Error - "
                           f"Rot: {val_pose_errors['rot_error']:.2f}° "
                           f"(Roll:{val_pose_errors['roll_error']:.2f}° Pitch:{val_pose_errors['pitch_error']:.2f}° Yaw:{val_pose_errors['yaw_error']:.2f}°)")
                else:
                    tprint(f"Epoch [{epoch+1}/{num_epochs}], {val_label} Pose Error - "
                           f"Trans: {val_pose_errors['trans_error']:.4f}m "
                           f"(Fwd:{val_pose_errors['fwd_error']:.4f}m Lat:{val_pose_errors['lat_error']:.4f}m Ht:{val_pose_errors['ht_error']:.4f}m), "
                           f"Rot: {val_pose_errors['rot_error']:.2f}° "
                           f"(Roll:{val_pose_errors['roll_error']:.2f}° Pitch:{val_pose_errors['pitch_error']:.2f}° Yaw:{val_pose_errors['yaw_error']:.2f}°)")

                if not rotation_only:
                    writer.add_scalar('Epoch/val/trans_error_m', val_pose_errors['trans_error'], epoch)
                    writer.add_scalar('Epoch/val/fwd_error_m', val_pose_errors['fwd_error'], epoch)
                    writer.add_scalar('Epoch/val/lat_error_m', val_pose_errors['lat_error'], epoch)
                    writer.add_scalar('Epoch/val/ht_error_m', val_pose_errors['ht_error'], epoch)
                writer.add_scalar('Epoch/val/rot_error_deg', val_pose_errors['rot_error'], epoch)
                writer.add_scalar('Epoch/val/roll_error_deg', val_pose_errors['roll_error'], epoch)
                writer.add_scalar('Epoch/val/pitch_error_deg', val_pose_errors['pitch_error'], epoch)
                writer.add_scalar('Epoch/val/yaw_error_deg', val_pose_errors['yaw_error'], epoch)

                medw_result = None
                if args.enable_medw_eval > 0 and medw_T_pred:
                    medw_result = _compute_medw_from_val_accum(
                        medw_T_pred, medw_T_gt, medw_seqs, window=args.medw_eval_max_frames)
                    if medw_result is not None:
                        _medw_mx = _medw_max_rpy(medw_result)
                        writer.add_scalar('Epoch/medw/rot_error_deg', medw_result['rot'], epoch)
                        writer.add_scalar('Epoch/medw/roll_error_deg', medw_result['roll'], epoch)
                        writer.add_scalar('Epoch/medw/pitch_error_deg', medw_result['pitch'], epoch)
                        writer.add_scalar('Epoch/medw/yaw_error_deg', medw_result['yaw'], epoch)
                        writer.add_scalar('Epoch/medw/max_rpy_error_deg', _medw_mx, epoch)
                        tprint(f"Epoch [{epoch+1}/{num_epochs}], MEDW{args.medw_eval_max_frames} "
                               f"(val reuse): max(R,P,Y)={_medw_mx:.4f}° "
                               f"(R:{medw_result['roll']:.4f} P:{medw_result['pitch']:.4f} "
                               f"Y:{medw_result['yaw']:.4f}, rot={medw_result['rot']:.4f}°)")
                    else:
                        tprint(f"  [MEDW WARN] MEDW{args.medw_eval_max_frames} eval returned no result")

                if rotation_only:
                    cur_val_score = val_pose_errors['rot_error']
                    best_val_score = best_val['rot']
                else:
                    cur_val_score = val_pose_errors['trans_error'] + val_pose_errors['rot_error'] * 0.1
                    best_val_score = best_val['trans'] + best_val['rot'] * 0.1
                if cur_val_score < best_val_score:
                    cur_val_loss = val_loss.get('total_loss', float('inf')) if val_loss else float('inf')
                    best_val.update({
                        'epoch': epoch + 1,
                        'loss': cur_val_loss,
                        'trans': val_pose_errors['trans_error'],
                        'rot': val_pose_errors['rot_error'],
                        'errors': dict(val_pose_errors),
                    })
                    early_stop_counter = 0
                    best_ckpt_path = os.path.join(ckpt_save_dir, "ckpt_best_val.pth")
                    model_to_save = model.module if use_ddp else model
                    _best_data = {
                        'epoch': epoch + 1,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'scaler_state_dict': scaler.state_dict(),
                        'train_noise': train_noise,
                        'eval_noise': eval_noise,
                        'rotation_only': rotation_only,
                        'epoch_train_errors': last_epoch_train_errors,
                        'epoch_val_errors': dict(val_pose_errors),
                        'best_train': best_train,
                        'best_val': best_val,
                        'args': vars(args),
                    }
                    _best_data.update(_build_ckpt_metadata(model, args))
                    if _v321_ema_enabled and _v321_ema_model is not None:
                        _best_data['ema_state_dict'] = _v321_ema_model.state_dict()
                    torch.save(_best_data, best_ckpt_path)
                    tprint(f"Best val model saved to {best_ckpt_path} "
                           f"(val_rot={val_pose_errors['rot_error']:.4f}°)")
                else:
                    early_stop_counter += 1
                    if args.early_stopping_patience > 0:
                        tprint(f"Val did not improve for {early_stop_counter} eval cycles "
                               f"(patience={args.early_stopping_patience})")

                if args.early_stopping_patience > 0 and early_stop_counter >= args.early_stopping_patience:
                    tprint(f"Early stopping triggered at epoch {epoch+1} "
                           f"(no val improvement for {early_stop_counter} eval cycles)")
                    early_stop_triggered = True

                if args.enable_jacobian_eval > 0 and jac_result is not None:
                    writer.add_scalar('Epoch/jacobian/overall', jac_result['overall'], epoch)
                    writer.add_scalar('Epoch/jacobian/roll', jac_result['roll'], epoch)
                    writer.add_scalar('Epoch/jacobian/pitch', jac_result['pitch'], epoch)
                    writer.add_scalar('Epoch/jacobian/yaw', jac_result['yaw'], epoch)
                    tprint(f"Epoch [{epoch+1}/{num_epochs}], Jacobian ±{args.jacobian_eval_angle_deg}° "
                           f"(controlled sweep, correction=init_err-out_err): "
                           f"Overall={jac_result['overall']:.3f} "
                           f"(R:{jac_result['roll']:.3f} P:{jac_result['pitch']:.3f} "
                           f"Y:{jac_result['yaw']:.3f}) [{jac_result['verdict']}]")
                    _jac_es_min = float(getattr(args, 'jacobian_early_stop_min', 0.0))
                    _jac_es_pat = int(getattr(args, 'jacobian_early_stop_patience', 2))
                    if _jac_es_min > 0:
                        if float(jac_result['overall']) < _jac_es_min:
                            jacobian_early_stop_counter += 1
                            tprint(f"  Jacobian early-stop: overall={jac_result['overall']:.3f} "
                                   f"< {_jac_es_min} ({jacobian_early_stop_counter}/{_jac_es_pat} eval cycles)")
                        else:
                            jacobian_early_stop_counter = 0
                        if jacobian_early_stop_counter >= _jac_es_pat:
                            tprint(f"Early stopping triggered at epoch {epoch+1} "
                                   f"(Jacobian overall < {_jac_es_min} for {_jac_es_pat} eval cycles)")
                            early_stop_triggered = True
                    if args.enable_jacobian_gate_ckpt > 0:
                        _jac_ov = float(jac_result['overall'])
                        if _jac_ov > best_jacobian.get('overall', float('-inf')):
                            best_jacobian.update({
                                'epoch': epoch + 1,
                                'overall': _jac_ov,
                                'roll': float(jac_result['roll']),
                                'pitch': float(jac_result['pitch']),
                                'yaw': float(jac_result['yaw']),
                            })
                            model_to_save = model.module if use_ddp else model
                            jac_path = os.path.join(ckpt_save_dir, 'ckpt_best_jacobian.pth')
                            _jac_data = {
                                'epoch': epoch + 1,
                                'model_state_dict': model_to_save.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                                'scaler_state_dict': scaler.state_dict(),
                                'train_noise': train_noise,
                                'eval_noise': eval_noise,
                                'rotation_only': rotation_only,
                                'epoch_train_errors': last_epoch_train_errors,
                                'epoch_val_errors': last_epoch_val_errors,
                                'best_train': best_train,
                                'best_val': best_val,
                                'best_medw': best_medw,
                                'best_dual': best_dual,
                                'best_jacobian': best_jacobian,
                                'jacobian_eval': jac_result,
                                'args': vars(args),
                            }
                            _jac_data.update(_build_ckpt_metadata(model, args))
                            if _v321_ema_enabled and _v321_ema_model is not None:
                                _jac_data['ema_state_dict'] = _v321_ema_model.state_dict()
                            torch.save(_jac_data, jac_path)
                            tprint(f"  ★ New best Jacobian → {jac_path} (overall={_jac_ov:.3f})")
                elif args.enable_jacobian_eval > 0:
                    tprint("  [JAC WARN] Insufficient Jacobian data from val sweep")

                if args.enable_medw_eval > 0 and medw_result is not None:
                    medw_max_rpy = _medw_max_rpy(medw_result)
                    if medw_max_rpy < best_medw.get('max_rpy', float('inf')):
                        best_medw.update({
                            'epoch': epoch + 1,
                            'rot': medw_result['rot'],
                            'roll': medw_result['roll'],
                            'pitch': medw_result['pitch'],
                            'yaw': medw_result['yaw'],
                            'max_rpy': medw_max_rpy,
                        })
                        model_to_save = model.module if use_ddp else model
                        best_medw_path = os.path.join(ckpt_save_dir, 'ckpt_best_medw.pth')
                        _medw_best_data = {
                            'epoch': epoch + 1,
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'scaler_state_dict': scaler.state_dict(),
                            'train_noise': train_noise,
                            'eval_noise': eval_noise,
                            'rotation_only': rotation_only,
                            'epoch_train_errors': last_epoch_train_errors,
                            'epoch_val_errors': last_epoch_val_errors,
                            'best_train': best_train,
                            'best_val': best_val,
                            'best_medw': best_medw,
                            'medw_eval': medw_result,
                            'args': vars(args),
                        }
                        _medw_best_data.update(_build_ckpt_metadata(model, args))
                        if _v321_ema_enabled and _v321_ema_model is not None:
                            _medw_best_data['ema_state_dict'] = _v321_ema_model.state_dict()
                        torch.save(_medw_best_data, best_medw_path)
                        medw_log_path = os.path.join(ckpt_save_dir, 'medw_eval_summary.json')
                        with open(medw_log_path, 'w') as jf:
                            json.dump({
                                'best_medw_rot': medw_result['rot'],
                                'best_medw_max_rpy': medw_max_rpy,
                                'best_medw_roll': medw_result['roll'],
                                'best_medw_pitch': medw_result['pitch'],
                                'best_medw_yaw': medw_result['yaw'],
                                'best_epoch': epoch + 1,
                                'best_ckpt_path': best_medw_path,
                                'medw_window': args.medw_eval_max_frames,
                                'medw_source': 'val_split_reuse',
                            }, jf, indent=2)
                        tprint(f"  ★ New best MEDW → {best_medw_path} "
                               f"(max_rpy={medw_max_rpy:.4f}° R={medw_result['roll']:.4f} "
                               f"P={medw_result['pitch']:.4f} Y={medw_result['yaw']:.4f})")

                _dual_pass = False
                if args.enable_dual_gate_ckpt > 0 and medw_result is not None and jac_result is not None:
                    _dual_pass = _dual_gate_pass(
                        medw_result, jac_result,
                        args.dual_gate_medw_max, args.dual_gate_jacobian_min)
                    _dg_verdict, _dg_medw, _dg_jac = _format_dual_gate_status(
                        medw_result, jac_result,
                        args.dual_gate_medw_max, args.dual_gate_jacobian_min)
                    if _dual_pass:
                        medw_max_rpy = _medw_max_rpy(medw_result)
                        jac_ov = float(jac_result['overall'])
                        dual_score = medw_max_rpy - 0.05 * _jacobian_min_axis(jac_result)
                        if dual_score < best_dual['score']:
                            best_dual.update({
                                'epoch': epoch + 1,
                                'score': dual_score,
                                'medw_max_rpy': medw_max_rpy,
                                'medw_roll': medw_result['roll'],
                                'medw_pitch': medw_result['pitch'],
                                'medw_yaw': medw_result['yaw'],
                                'jacobian': jac_ov,
                                'jacobian_roll': float(jac_result['roll']),
                                'jacobian_pitch': float(jac_result['pitch']),
                                'jacobian_yaw': float(jac_result['yaw']),
                            })
                            model_to_save = model.module if use_ddp else model
                            dual_path = os.path.join(ckpt_save_dir, 'ckpt_best_dual.pth')
                            _dual_data = {
                                'epoch': epoch + 1,
                                'model_state_dict': model_to_save.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                                'scaler_state_dict': scaler.state_dict(),
                                'train_noise': train_noise,
                                'eval_noise': eval_noise,
                                'rotation_only': rotation_only,
                                'epoch_train_errors': last_epoch_train_errors,
                                'epoch_val_errors': last_epoch_val_errors,
                                'best_train': best_train,
                                'best_val': best_val,
                                'best_medw': best_medw,
                                'best_dual': best_dual,
                                'medw_eval': medw_result,
                                'jacobian_eval': jac_result,
                                'args': vars(args),
                            }
                            _dual_data.update(_build_ckpt_metadata(model, args))
                            if _v321_ema_enabled and _v321_ema_model is not None:
                                _dual_data['ema_state_dict'] = _v321_ema_model.state_dict()
                            torch.save(_dual_data, dual_path)
                            tprint(f"  ★ Dual gate PASS → {dual_path} "
                                   f"({_dg_medw}; {_dg_jac}; score={dual_score:.4f})")
                    else:
                        tprint(f"  ○ Dual gate FAIL (ep {epoch+1}): {_dg_medw}; {_dg_jac}")

                if (args.enable_medw_eval > 0 or args.enable_jacobian_eval > 0) and is_main:
                    kpi_history.append({
                        'epoch': epoch + 1,
                        'medw': medw_result,
                        'jacobian': jac_result,
                        'dual_pass': _dual_pass,
                    })

            val_loss = None
            loss = None

        if use_ddp:
            # All ranks participate in val; sync before early-stop broadcast.
            dist.barrier()

        if use_ddp:
            stop_tensor = torch.tensor([1 if early_stop_triggered else 0],
                                       device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            early_stop_triggered = stop_tensor.item() == 1
            dist.barrier()

        if early_stop_triggered:
            if is_main:
                tprint("All ranks stopping due to early stopping.")
            break

    if is_main:
        def _fmt_err(e):
            if rotation_only:
                return (f"Rot: {e['rot_error']:.2f}° (R:{e['roll_error']:.2f} P:{e['pitch_error']:.2f} Y:{e['yaw_error']:.2f})")
            return (f"{e['trans_error']:.4f}m (Fwd:{e['fwd_error']:.4f} Lat:{e['lat_error']:.4f} Ht:{e['ht_error']:.4f}), "
                    f"Rot: {e['rot_error']:.2f}° (R:{e['roll_error']:.2f} P:{e['pitch_error']:.2f} Y:{e['yaw_error']:.2f})")

        has_eval = checkpoint_records and any(r['eval'] is not None for r in checkpoint_records)
        best_eval_idx = -1
        if has_eval:
            eval_scores = [(i, r['eval']['rot_error'] if rotation_only else r['eval']['trans_error'] + r['eval']['rot_error'] * 0.1)
                           for i, r in enumerate(checkpoint_records) if r['eval'] is not None]
            if eval_scores:
                best_eval_idx = min(eval_scores, key=lambda x: x[1])[0]

        md_lines = []
        md_lines.append("=" * 80)
        md_lines.append("训练完成总结 / Training Summary")
        md_lines.append("=" * 80)
        md_lines.append("")

        total_training_time = time.time() - training_start_time
        completed_epochs = len(epoch_times)
        avg_ep = sum(epoch_times) / max(completed_epochs, 1)
        md_lines.append(f"Training Time: {total_training_time/3600:.2f}h ({total_training_time:.0f}s), "
                        f"{completed_epochs} epochs, avg {avg_ep:.1f}s/epoch ({avg_ep/60:.1f}min/epoch)")
        md_lines.append("")

        if best_train['epoch'] > 0 and best_train['errors']:
            e = best_train['errors']
            md_lines.append(f"Best Train  (Epoch {best_train['epoch']}):")
            md_lines.append(f"  Pose Error - {_fmt_err(e)}")
            md_lines.append("")

        if best_val['epoch'] > 0 and best_val['errors']:
            e = best_val['errors']
            md_lines.append(f"Best Val    (Epoch {best_val['epoch']}):")
            md_lines.append(f"  Pose Error - {_fmt_err(e)}")
            md_lines.append("")

        if best_medw['epoch'] > 0:
            md_lines.append(f"Best MEDW{args.medw_eval_max_frames} (Epoch {best_medw['epoch']}):")
            md_lines.append(f"  max(R,P,Y): {best_medw.get('max_rpy', _medw_max_rpy(best_medw)):.4f}° "
                            f"(R:{best_medw['roll']:.4f} P:{best_medw['pitch']:.4f} "
                            f"Y:{best_medw['yaw']:.4f}, rot={best_medw['rot']:.4f}°)")
            md_lines.append("")

        if best_dual['epoch'] > 0:
            md_lines.append(f"Best Dual Gate (Epoch {best_dual['epoch']}) — CONVERGED:")
            md_lines.append(f"  max(R,P,Y): {best_dual.get('medw_max_rpy', float('nan')):.4f}° "
                            f"(R:{best_dual.get('medw_roll', float('nan')):.4f} "
                            f"P:{best_dual.get('medw_pitch', float('nan')):.4f} "
                            f"Y:{best_dual.get('medw_yaw', float('nan')):.4f})")
            md_lines.append(f"  Jacobian: {best_dual.get('jacobian', float('nan')):.3f} "
                            f"(R:{best_dual.get('jacobian_roll', float('nan')):.3f} "
                            f"P:{best_dual.get('jacobian_pitch', float('nan')):.3f} "
                            f"Y:{best_dual.get('jacobian_yaw', float('nan')):.3f})")
            md_lines.append(f"  ckpt: {os.path.join(ckpt_save_dir, 'ckpt_best_dual.pth')}")
            md_lines.append("")
        elif args.enable_dual_gate_ckpt > 0 and kpi_history:
            md_lines.append("Dual Gate: NOT CONVERGED (never passed max(R,P,Y) + Jacobian axes gate)")
            md_lines.append("")

        if best_jacobian.get('epoch', -1) > 0:
            md_lines.append(f"Best Jacobian (Epoch {best_jacobian['epoch']}):")
            md_lines.append(f"  overall: {best_jacobian.get('overall', float('nan')):.3f} "
                            f"(R:{best_jacobian.get('roll', float('nan')):.3f} "
                            f"P:{best_jacobian.get('pitch', float('nan')):.3f} "
                            f"Y:{best_jacobian.get('yaw', float('nan')):.3f})")
            md_lines.append(f"  ckpt: {os.path.join(ckpt_save_dir, 'ckpt_best_jacobian.pth')}")
            md_lines.append("")

        if checkpoint_records:
            md_lines.append(f"Checkpoint Performance Table ({len(checkpoint_records)} checkpoints)")
            md_lines.append("")

            if rotation_only:
                hdr = "| Epoch | Rot(°) | Roll(LiDAR-X) | Pitch(LiDAR-Y) | Yaw(LiDAR-Z) |"
                sep = "| ---: | ---: | ---: | ---: | ---: |"
            else:
                hdr = "| Epoch | Trans(m) | Fwd(LiDAR-X) | Lat(LiDAR-Y) | Ht(LiDAR-Z) | Rot(°) | Roll(LiDAR-X) | Pitch(LiDAR-Y) | Yaw(LiDAR-Z) |"
                sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

            md_lines.append("--- Train ---")
            md_lines.append("")
            md_lines.append(hdr)
            md_lines.append(sep)
            for rec in checkpoint_records:
                t = rec['train']
                if rotation_only:
                    md_lines.append(f"| {rec['epoch']} | {t['rot_error']:.2f} | {t['roll_error']:.2f} | {t['pitch_error']:.2f} | {t['yaw_error']:.2f} |")
                else:
                    md_lines.append(f"| {rec['epoch']} | {t['trans_error']:.4f} | {t['fwd_error']:.4f} | {t['lat_error']:.4f} | {t['ht_error']:.4f}"
                                    f" | {t['rot_error']:.2f} | {t['roll_error']:.2f} | {t['pitch_error']:.2f} | {t['yaw_error']:.2f} |")
            md_lines.append("")

            if has_eval:
                md_lines.append("--- Eval (Validation) ---")
                md_lines.append("")
                if rotation_only:
                    md_lines.append(hdr)
                    md_lines.append(sep)
                    for i, rec in enumerate(checkpoint_records):
                        if rec['eval'] is not None:
                            e = rec['eval']
                            best_mark = " *" if i == best_eval_idx else ""
                            md_lines.append(f"| {rec['epoch']} | {e['rot_error']:.2f} | {e['roll_error']:.2f} | {e['pitch_error']:.2f} | {e['yaw_error']:.2f}{best_mark} |")
                        else:
                            md_lines.append(f"| {rec['epoch']} | N/A | | | |")
                else:
                    md_lines.append(hdr.rstrip(" |") + " | ΔTrans |")
                    md_lines.append(sep.rstrip(" |") + " | ---: |")
                    for i, rec in enumerate(checkpoint_records):
                        if rec['eval'] is not None:
                            e = rec['eval']
                            t = rec['train']
                            gap = e['trans_error'] - t['trans_error']
                            best_mark = " *" if i == best_eval_idx else ""
                            md_lines.append(f"| {rec['epoch']} | {e['trans_error']:.4f} | {e['fwd_error']:.4f} | {e['lat_error']:.4f} | {e['ht_error']:.4f}"
                                            f" | {e['rot_error']:.2f} | {e['roll_error']:.2f} | {e['pitch_error']:.2f} | {e['yaw_error']:.2f}"
                                            f" | {gap:+.4f}{best_mark} |")
                        else:
                            md_lines.append(f"| {rec['epoch']} | N/A | | | | N/A | | | | |")
                md_lines.append("")
                if best_eval_idx >= 0:
                    md_lines.append(f"* = Best eval checkpoint (Epoch {checkpoint_records[best_eval_idx]['epoch']})")
                if not rotation_only:
                    md_lines.append("ΔTrans = Eval Trans - Train Trans (泛化差距, 越小越好)")

        summary_text = "\n".join(md_lines)
        tprint(summary_text)

        summary_md_path = os.path.join(log_dir, "training_summary.md")
        with open(summary_md_path, 'w') as f:
            f.write(summary_text + "\n")

        tprint(f"训练总结已保存到: {summary_md_path}")

        if args.enable_medw_eval > 0 or args.enable_jacobian_eval > 0:
            conv_md, conv_json, conv_verdict = _write_convergence_report(
                log_dir, ckpt_save_dir, args, best_medw, best_dual, kpi_history)
            tprint(f"收敛报告: {conv_verdict} → {conv_md}")
            tprint(f"  JSON: {conv_json}")

        tprint("=" * 80)
    
    if is_main and writer is not None:
        writer.close()
        tprint(f"Logs are saved at {log_dir}")
    _cleanup_log()
    cleanup_ddp()


if __name__ == "__main__":
    main()
