import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
from tools import generate_single_perturbation_from_T, augment_gt_pitch_flip
import shutil
import cv2
import os
import time
from contextlib import nullcontext


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
from visualization import (
    compute_batch_pose_errors,
    visualize_batch_projection,
    prepare_image_for_tensorboard,
    compute_pose_errors
)

import sys
import io
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


def _augment_intrinsics(intrinsic_matrix, strength):
    """Randomly perturb camera intrinsic matrix to improve robustness to unseen cameras.

    Augmentation strategy:
      - fx, fy scaled by same random factor (preserves aspect ratio)
      - cx, cy independently offset by a smaller factor
      - K[2,2] = 1 is preserved

    The frustum geometry in LSS's get_geometry() depends on inv(K), so varying K
    during training forces the model to generalize across different focal lengths.

    Args:
        intrinsic_matrix: (B, 3, 3) tensor on device
        strength: max relative deviation, e.g. 0.05 means ±5%
    Returns:
        augmented (B, 3, 3) tensor (same device, same dtype)
    """
    B = intrinsic_matrix.shape[0]
    dev = intrinsic_matrix.device
    K = intrinsic_matrix.clone()

    focal_scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * strength
    cx_scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * strength * 0.5
    cy_scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * strength * 0.5

    K[:, 0, 0] *= focal_scale       # fx
    K[:, 1, 1] *= focal_scale       # fy
    K[:, 0, 2] *= cx_scale          # cx
    K[:, 1, 2] *= cy_scale          # cy
    return K

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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--scheduler", type=int, default=-1)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
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
    # 可视化参数
    parser.add_argument("--vis_freq", type=int, default=40, help="训练可视化频率 (每多少个batch可视化一次)")
    parser.add_argument("--vis_samples", type=int, default=3, help="每次可视化的样本数")
    parser.add_argument("--vis_points", type=int, default=80000, help="每个样本最大可视化点数")
    parser.add_argument("--vis_point_radius", type=int, default=1, help="可视化点的半径")
    parser.add_argument("--enable_vis", type=int, default=1, help="是否启用点云投影可视化 (1=启用, 0=禁用)")
    parser.add_argument("--enable_ckpt_eval", type=int, default=1, help="是否在保存checkpoint时进行评估 (1=启用, 0=禁用)")
    parser.add_argument("--compile", type=int, default=0, help="使用 torch.compile 加速模型 (1=启用, 0=禁用)")
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
    parser.add_argument("--use_mlp_head", type=int, default=1,
                        help="Use 3-layer MLP regression head (1=MLP, 0=single Linear)")
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
    parser.add_argument("--lr_schedule", type=str, default="step",
                        choices=["step", "cosine_warm_restarts"],
                        help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                        help="LR multiplier for pretrained backbone (SwinT)")
    parser.add_argument("--cosine_T0", type=int, default=50,
                        help="CosineAnnealingWarmRestarts T_0 period")
    parser.add_argument("--cosine_Tmult", type=int, default=2,
                        help="CosineAnnealingWarmRestarts T_mult")
    parser.add_argument("--perturb_distribution", type=str, default="uniform",
                        choices=["uniform", "truncated_normal"],
                        help="Perturbation angle distribution")
    parser.add_argument("--per_axis_prob", type=float, default=0.0,
                        help="Probability of single-axis perturbation (0=disabled)")
    parser.add_argument("--per_axis_weights", type=str, default="",
                        help="Roll,Pitch,Yaw sampling weights for per-axis mode (e.g. 0.5,0.3,0.2). Empty=uniform")
    parser.add_argument("--augment_pc_jitter", type=float, default=0.0,
                        help="Point cloud Gaussian jitter sigma in meters (0=disabled)")
    parser.add_argument("--augment_pc_dropout", type=float, default=0.0,
                        help="Point cloud random dropout ratio (0=disabled)")
    parser.add_argument("--augment_color_jitter", type=float, default=0.0,
                        help="Image color jitter strength (0=disabled)")
    parser.add_argument("--augment_intrinsic", type=float, default=0.0,
                        help="Camera intrinsic augmentation strength: max relative deviation for fx/fy/cx/cy (e.g. 0.05=±5%%, 0=disabled)")
    parser.add_argument("--augment_pitch_flip_prob", type=float, default=0.0,
                        help="GT pitch flip augmentation probability (0=disabled). "
                             "When > 0, randomly rotates GT around LiDAR Y-axis to "
                             "balance the positive/negative pitch distribution.")
    parser.add_argument("--augment_pitch_flip_max_deg", type=float, default=6.0,
                        help="Max rotation angle (degrees) for GT pitch flip augmentation")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Early stopping patience in epochs (0=disabled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility (default: 42)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (1=disabled, >1=accumulate N micro-batches per optimizer step)")
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
        img, pcd, gt_transform, intrinsic = result
        if isinstance(img, np.ndarray) and img.shape[:2] == (self.target_size[1], self.target_size[0]):
            return img, pcd, gt_transform, intrinsic
        resized_img, new_intrinsic = crop_and_resize(img, self.target_size, intrinsic, self.crop)
        return resized_img, pcd, gt_transform, new_intrinsic


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

    return imgs, pcs, masks, gt_T_to_camera, intrinsics

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
                                    max_frames_per_seq=args.max_frames_per_seq)
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

    num_workers = min(16, os.cpu_count() or 4)
    if is_main:
        tprint(f"DataLoader: num_workers={num_workers}, pin_memory=True, persistent_workers=True")

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if use_ddp else None

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
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=_worker_init_fn,
    )

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
    use_mlp_head = args.use_mlp_head > 0
    axis_weights_tuple = tuple(float(x) for x in args.axis_weights.split(','))
    use_foundation_depth = args.use_foundation_depth > 0
    fd_mode = args.fd_mode if use_foundation_depth else "replace"
    if is_main and use_foundation_depth:
        tprint(f"Foundation Depth: 启用 (model={args.depth_model_type}, mode={fd_mode})")
    with capture_prints(is_main):
        model = BEVCalib(
            deformable=deformable_choise,
            bev_encoder=bev_encoder_choise,
            img_shape=img_shape,
            rotation_only=rotation_only,
            enable_axis_loss=enable_axis_loss,
            weight_axis_rotation=args.weight_axis_rotation,
            axis_weights=axis_weights_tuple,
            drop_path_rate=args.drop_path_rate,
            head_dropout=args.head_dropout,
            use_geodesic_loss=use_geodesic_loss,
            use_mlp_head=use_mlp_head,
            bev_pool_factor=args.bev_pool_factor,
            use_foundation_depth=use_foundation_depth,
            depth_model_type=args.depth_model_type,
            fd_mode=fd_mode,
            voxel_mode=args.voxel_mode,
            to_bev_mode=args.to_bev_mode,
            scatter_reduce=args.scatter_reduce,
        ).to(device)

    if args.pretrain_ckpt is not None:
        state_dict = torch.load(args.pretrain_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict['model_state_dict'], strict=False)
        if is_main:
            tprint(f"Load pretrain model from {args.pretrain_ckpt}")
            if missing:
                tprint(f"  Missing keys (new layers): {missing}")
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
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if is_main:
            tprint(f"Model wrapped with DistributedDataParallel on {world_size} GPUs "
                   f"(find_unused_parameters=False)")
    
    raw_model = model.module if use_ddp else model

    if is_main:
        tprint(f"The weight decay is: {args.wd}")
        tprint(f"The initial learning rate is: {args.lr}")

    backbone_params = []
    head_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'img_branch' in name or 'pc_branch' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = args.lr * args.backbone_lr_scale
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.wd)

    if is_main:
        tprint(f"Differential LR: backbone={backbone_lr:.2e} ({len(backbone_params)} params), "
               f"heads={args.lr:.2e} ({len(head_params)} params)")

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

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.float16
    scaler = GradScaler(enabled=use_amp)
    if use_amp and is_main:
        tprint(f"AMP enabled with {amp_dtype}, GradScaler=on")

    grad_accum_steps = max(1, args.grad_accum_steps)
    if grad_accum_steps > 1 and is_main:
        effective_bs = args.batch_size * world_size * grad_accum_steps
        tprint(f"Gradient Accumulation: {grad_accum_steps} steps, "
               f"effective batch size = {args.batch_size}×{world_size}GPU×{grad_accum_steps}accum = {effective_bs}")
    
    if is_main and args.augment_intrinsic > 0:
        tprint(f"Intrinsic augmentation: ±{args.augment_intrinsic*100:.0f}% (fx/fy coupled, cx/cy ±{args.augment_intrinsic*50:.0f}%)")

    if is_main and args.augment_pitch_flip_prob > 0:
        tprint(f"GT pitch flip augmentation: prob={args.augment_pitch_flip_prob}, "
               f"max_deg={args.augment_pitch_flip_max_deg}°")

    _identity_4x4 = torch.eye(4, device=device)

    train_noise = {
        "angle_range_deg": args.angle_range_deg if args.angle_range_deg is not None else 20,
        "trans_range": args.trans_range if args.trans_range is not None else 1.5,
    }

    eval_noise = {
        "angle_range_deg": args.eval_angle_range_deg if args.eval_angle_range_deg is not None else train_noise["angle_range_deg"],
        "trans_range": args.eval_trans_range if args.eval_trans_range is not None else train_noise["trans_range"],
    }

    global_step = 0
    
    # 累积误差统计
    epoch_pose_errors = {
        'trans_error': 0, 'fwd_error': 0, 'lat_error': 0, 'ht_error': 0,
        'rot_error': 0, 'roll_error': 0, 'pitch_error': 0, 'yaw_error': 0,
    }
    
    best_train = {'epoch': -1, 'loss': float('inf'), 'trans': float('inf'), 'rot': float('inf'), 'errors': None}
    best_val = {'epoch': -1, 'loss': float('inf'), 'trans': float('inf'), 'rot': float('inf'), 'errors': None}
    last_epoch_train_errors = None
    last_epoch_val_errors = None
    checkpoint_records = []
    early_stop_counter = 0
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
        tprint("  • rotation_loss: displayed in degrees (°) for readability, but total_loss uses radians")
        tprint("  • PC_reproj_loss: point cloud reprojection error")
        tprint("  • quat_norm_loss: quaternion normalization penalty")
        tprint("  • Pose Error - Rot: rotation error with Roll/Pitch/Yaw breakdown")
        tprint("  • Pose Error - Trans: translation error with Forward/Lateral/Height breakdown")
        tprint("=" * 80)
    
    training_start_time = time.time()
    epoch_times = []

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = {}
        for key in epoch_pose_errors:
            epoch_pose_errors[key] = 0

        if epoch == 0:
            raw_model._profile_modules = True
        elif epoch == 1:
            raw_model._profile_modules = False

        epoch_start = time.time()
        out_init_loss_choice = epoch < 5
        t_data_total, t_prep_total, t_compute_total, t_vis_total = 0.0, 0.0, 0.0, 0.0
        vis_count = 0
        _bwd_profile_events = [] if raw_model._profile_modules else None
        _do_detailed_profile = raw_model._profile_modules
        t_h2d_total = 0.0
        t_cpu_aug_total = 0.0
        processed_batches = 0
        t_iter_start = time.time()
        for batch_index, batch_data in enumerate(train_loader):
            t_data_end = time.time()
            t_data_total += t_data_end - t_iter_start

            if batch_data is None:
                t_iter_start = time.time()
                continue
            imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data

            t_prep_start = time.time()
            gt_T_to_camera_np = np.array(gt_T_to_camera, dtype=np.float32)
            if args.augment_pitch_flip_prob > 0:
                gt_T_to_camera_np = augment_gt_pitch_flip(
                    gt_T_to_camera_np,
                    prob=args.augment_pitch_flip_prob,
                    max_deg=args.augment_pitch_flip_max_deg,
                )
            per_axis_weights_parsed = None
            if args.per_axis_weights:
                per_axis_weights_parsed = tuple(float(x) for x in args.per_axis_weights.split(','))
            init_T_to_camera_np, _, _ = generate_single_perturbation_from_T(
                gt_T_to_camera_np,
                angle_range_deg=train_noise["angle_range_deg"],
                trans_range=train_noise["trans_range"],
                rotation_only=rotation_only,
                distribution=args.perturb_distribution,
                per_axis_prob=args.per_axis_prob,
                per_axis_weights=per_axis_weights_parsed,
            )
            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float()
            if args.augment_color_jitter > 0:
                resize_imgs = _apply_color_jitter(resize_imgs, args.augment_color_jitter)
            if xyz_only_choise:
                pcs_np = np.array(pcs)[:, :, :3]
            else:
                pcs_np = np.array(pcs)
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
            if args.augment_intrinsic > 0:
                intrinsic_matrix = _augment_intrinsics(intrinsic_matrix, args.augment_intrinsic)
            if _do_detailed_profile:
                t_h2d_total += time.time() - t_h2d_start
            t_prep_total += time.time() - t_prep_start

            t_compute_start = time.time()
            is_accum_step = (batch_index + 1) % grad_accum_steps != 0 and (batch_index + 1) < len(train_loader)
            if batch_index % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            sync_ctx = model.no_sync() if (use_ddp and is_accum_step) else nullcontext()
            _bwd_ev = None
            if _bwd_profile_events is not None:
                _bwd_ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            with sync_ctx:
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    T_pred, init_loss, loss = model(resize_imgs, pcs_t, gt_T_to_camera_t, init_T_to_camera_t, post_cam2ego_T, intrinsic_matrix, masks=masks_t, out_init_loss=out_init_loss_choice)
                    total_loss = loss["total_loss"]
                    if fd_mode == "supervision" and use_foundation_depth:
                        ds_loss = raw_model.img_branch.get_depth_supervision_loss(alpha=args.depth_sup_alpha)
                        total_loss = total_loss + ds_loss
                        loss["depth_sup_loss"] = ds_loss.item()
                    if grad_accum_steps > 1:
                        total_loss = total_loss / grad_accum_steps
                if _bwd_ev is not None:
                    _bwd_ev[0].record()
                scaler.scale(total_loss).backward()
                if _bwd_ev is not None:
                    _bwd_ev[1].record()
            if not is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                if _bwd_ev is not None:
                    _bwd_ev[2].record()
                scaler.step(optimizer)
                scaler.update()
                if _bwd_ev is not None:
                    _bwd_ev[3].record()
            if _bwd_ev is not None:
                _bwd_profile_events.append((_bwd_ev, not is_accum_step))
            t_compute_total += time.time() - t_compute_start
            
            with torch.no_grad():
                batch_errors = compute_batch_pose_errors(T_pred, gt_T_to_camera_t)
                for key in epoch_pose_errors:
                    epoch_pose_errors[key] += batch_errors[key]
            
            for key in loss.keys():
                if key not in train_loss:
                    train_loss[key] = loss[key].item()
                else:
                    train_loss[key] += loss[key].item()
            
            if init_loss is not None:
                for key in init_loss.keys():
                    train_key = f"init_{key}"
                    if train_key not in train_loss:
                        train_loss[train_key] = init_loss[key].item()
                    else:
                        train_loss[train_key] += init_loss[key].item()

            if batch_index % 10 == 0 and is_main:
                if rotation_only:
                    tprint(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index+1}/{len(train_loader)}], "
                           f"Loss: {total_loss.item():.4f} (total), Rot: {batch_errors['rot_error']:.2f}°")
                else:
                    tprint(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index+1}/{len(train_loader)}], "
                           f"Loss: {total_loss.item():.4f} (total), Trans: {batch_errors['trans_error']:.4f}m "
                           f"(Fwd:{batch_errors['fwd_error']:.4f}m Lat:{batch_errors['lat_error']:.4f}m Ht:{batch_errors['ht_error']:.4f}m), "
                           f"Rot: {batch_errors['rot_error']:.2f}°")
            
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
            mod_prof = raw_model.get_module_profile(reset=True)
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
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_noise': train_noise,
                'eval_noise': eval_noise,
                'rotation_only': rotation_only,
                'epoch_train_errors': last_epoch_train_errors,
                'epoch_val_errors': last_epoch_val_errors,
                'best_train': best_train,
                'best_val': best_val,
                'args': vars(args) 
            }, ckpt_path)
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
                        imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data
                        
                        gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
                        init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(
                            gt_T_to_camera_np, 
                            angle_range_deg=eval_angle_range, 
                            trans_range=eval_trans_range,
                            rotation_only=rotation_only,
                        )
                        
                        resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device, non_blocking=True)
                        if xyz_only_choise:
                            pcs_np = np.array(pcs)[:, :, :3]
                        else:
                            pcs_np = np.array(pcs)
                        pcs = torch.from_numpy(pcs_np).float().to(device, non_blocking=True)
                        gt_T_to_camera = torch.from_numpy(gt_T_to_camera_np).float().to(device, non_blocking=True)
                        init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device, non_blocking=True)
                        B_cur = gt_T_to_camera.shape[0]
                        post_cam2ego_T = _identity_4x4.unsqueeze(0).expand(B_cur, -1, -1)
                        intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device, non_blocking=True)
                        
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            T_pred, _, _ = raw_model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)
                        
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

        if epoch % args.eval_epoches == 0 and is_main:
            eval_trans_range = eval_noise["trans_range"]
            eval_angle_range = eval_noise["angle_range_deg"]
            raw_model.eval()
            val_loss = {}
            val_pose_errors = {
                'trans_error': 0, 'fwd_error': 0, 'lat_error': 0, 'ht_error': 0,
                'rot_error': 0, 'roll_error': 0, 'pitch_error': 0, 'yaw_error': 0,
            }
            val_processed_batches = 0
            
            with torch.no_grad():
                for batch_index, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data
                    gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
                    init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(gt_T_to_camera_np, angle_range_deg=eval_angle_range, trans_range=eval_trans_range, rotation_only=rotation_only)
                    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device, non_blocking=True)
                    if xyz_only_choise:
                        pcs_np = np.array(pcs)[:, :, :3]
                    else:
                        pcs_np = np.array(pcs)
                    pcs = torch.from_numpy(pcs_np).float().to(device, non_blocking=True)
                    gt_T_to_camera = torch.from_numpy(gt_T_to_camera_np).float().to(device, non_blocking=True)
                    init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device, non_blocking=True)
                    B_cur = gt_T_to_camera.shape[0]
                    post_cam2ego_T = _identity_4x4.unsqueeze(0).expand(B_cur, -1, -1)
                    intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device, non_blocking=True)
                    with autocast(enabled=use_amp, dtype=amp_dtype):
                        T_pred, init_loss, loss = raw_model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)

                    # 计算姿态误差
                    batch_errors = compute_batch_pose_errors(T_pred, gt_T_to_camera)
                    for key in val_pose_errors:
                        val_pose_errors[key] += batch_errors[key]

                    for key in loss.keys():
                        val_key = key
                        if val_key not in val_loss.keys():
                            val_loss[val_key] = loss[key].item()
                        else:
                            val_loss[val_key] += loss[key].item()
                    if init_loss is not None:
                        for key in init_loss.keys():
                            val_key = f"init_{key}"
                            if val_key not in val_loss.keys():
                                val_loss[val_key] = init_loss[key].item()
                            else:
                                val_loss[val_key] += init_loss[key].item()
                    
                    val_processed_batches += 1
                    
                    # 验证集可视化 (每个epoch只可视化第一个batch)
                    if args.enable_vis > 0 and batch_index == 0:
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

            # 构建验证集标签（清晰标注扰动范围）
            if rotation_only:
                val_label = f"Val[±{eval_angle_range}°]"
            else:
                val_label = f"Val[±{eval_angle_range}°, ±{eval_trans_range}m]"
            
            effective_val_batches = max(1, val_processed_batches)
            for key in val_loss.keys():
                val_loss[key] /= effective_val_batches
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
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_noise': train_noise,
                    'eval_noise': eval_noise,
                    'rotation_only': rotation_only,
                    'epoch_train_errors': last_epoch_train_errors,
                    'epoch_val_errors': dict(val_pose_errors),
                    'best_train': best_train,
                    'best_val': best_val,
                    'args': vars(args),
                }, best_ckpt_path)
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

            val_loss = None
            loss = None

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
        tprint("=" * 80)
    
    if is_main and writer is not None:
        writer.close()
        tprint(f"Logs are saved at {log_dir}")
    _cleanup_log()
    cleanup_ddp()


if __name__ == "__main__":
    main()
