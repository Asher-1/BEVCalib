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
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime, timedelta
from torch.utils.data import random_split
import numpy as np
from pathlib import Path
from tools import generate_single_perturbation_from_T
import shutil
import cv2
import os
import time
from visualization import (
    compute_batch_pose_errors,
    visualize_batch_projection,
    prepare_image_for_tensorboard,
    compute_pose_errors
)

def tprint(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]", *args, **kwargs)

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
    parser.add_argument("--eval_epoches", type=int, default=20)
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
    parser.add_argument("--validate_data", type=int, default=1, help="是否在训练前验证数据利用率 (1=启用, 0=禁用)")
    parser.add_argument("--validate_sample_ratio", type=float, default=0.1, help="数据验证采样比例 (0.0-1.0)")
    parser.add_argument("--min_point_utilization", type=float, default=0.5, help="最低点云利用率阈值 (0.0-1.0)")
    parser.add_argument("--min_valid_ratio", type=float, default=0.9, help="最低有效帧比例阈值 (0.0-1.0)")
    # 可视化参数
    parser.add_argument("--vis_freq", type=int, default=40, help="训练可视化频率 (每多少个batch可视化一次)")
    parser.add_argument("--vis_samples", type=int, default=3, help="每次可视化的样本数")
    parser.add_argument("--vis_points", type=int, default=80000, help="每个样本最大可视化点数")
    parser.add_argument("--vis_point_radius", type=int, default=1, help="可视化点的半径")
    parser.add_argument("--enable_vis", type=int, default=1, help="是否启用点云投影可视化 (1=启用, 0=禁用)")
    parser.add_argument("--enable_ckpt_eval", type=int, default=1, help="是否在保存checkpoint时进行评估 (1=启用, 0=禁用)")
    parser.add_argument("--compile", type=int, default=0, help="使用 torch.compile 加速模型 (1=启用, 0=禁用)")
    return parser.parse_args()

def crop_and_resize(item, size, intrinsics, crop=True, distortion=None):
    """
    图像预处理: 缩放 → 更新内参
    
    Args:
        item: PIL Image 或 numpy array
        size: (width, height) 目标尺寸
        intrinsics: (3, 3) 原始相机内参矩阵
        crop: 是否裁剪中间区域
        distortion: 畸变系数 (保留参数但不使用)
    
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
        img, pcd, gt_transform, intrinsic, distortion = result
        if isinstance(img, np.ndarray) and img.shape[:2] == (self.target_size[1], self.target_size[0]):
            return img, pcd, gt_transform, intrinsic
        resized_img, new_intrinsic = crop_and_resize(img, self.target_size, intrinsic, self.crop, distortion)
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
    
    if is_main:
        tprint(f"训练启动, 参数配置:")
        tprint(args)
    if use_ddp and is_main:
        tprint(f"DDP enabled: {world_size} GPUs, rank={rank}, local_rank={local_rank}")
    
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
            print(f"警告: 复制源代码失败: {e}")
    
    writer = SummaryWriter(log_dir) if is_main else None
    
    # 预先计算目标图像尺寸（供 CustomDataset 查找预处理图像）
    target_size = get_target_size(
        use_custom_dataset=args.use_custom_dataset > 0,
        target_width=args.target_width,
        target_height=args.target_height
    )
    
    # 选择数据集类型
    if args.use_custom_dataset:
        dataset = CustomDataset(dataset_root, target_size=target_size)
    else:
        print("使用 KittiDataset")
        dataset = KittiDataset(dataset_root)

    # 数据利用率校验 (only on rank 0)
    if args.validate_data > 0 and is_main:
        print("\n" + "="*60)
        print("开始数据利用率校验...")
        print("="*60)
        
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
        print(f"验证结果已保存到: {validation_log_path}")
        
        if not validation_result['passed']:
            print("\n错误: 数据利用率验证未通过，退出训练！")
            print("   可以通过以下方式解决：")
            print("   1. 检查 bev_settings.py 中的体素化范围配置是否与数据集匹配")
            print("   2. 调整 --min_point_utilization 或 --min_valid_ratio 阈值")
            print("   4. 使用 --validate_data=0 跳过验证（不推荐）")
            exit(1)
    elif is_main:
        print("\n跳过数据利用率校验 (--validate_data=0)")
    if use_ddp:
        dist.barrier()

    if is_main:
        tprint(f"目标图像尺寸: {target_size[0]}x{target_size[1]} (宽x高)")
        if args.use_custom_dataset > 0:
            print(f"   (自定义数据集模式，保持16:9宽高比)")
        else:
            print(f"   (KITTI数据集模式)")
    
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None

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
    )

    deformable_choise = args.deformable > 0
    bev_encoder_choise = args.bev_encoder > 0
    xyz_only_choise = args.xyz_only > 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    
    img_shape = (target_size[1], target_size[0])
    if is_main:
        tprint(f"网络输入尺寸 (H, W): {img_shape}")
    
    model = BEVCalib(
        deformable=deformable_choise,
        bev_encoder=bev_encoder_choise,
        img_shape=img_shape
    ).to(device)

    if args.pretrain_ckpt is not None:
        state_dict = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
        if is_main:
            tprint(f"Load pretrain model from {args.pretrain_ckpt}")
    
    if args.compile > 0:
        try:
            model = torch.compile(model)
            if is_main:
                tprint("torch.compile enabled")
        except Exception as e:
            if is_main:
                tprint(f"torch.compile failed, falling back to eager mode: {e}")
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if is_main:
            tprint(f"Model wrapped with DistributedDataParallel on {world_size} GPUs")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler_choice = args.scheduler > 0
    if scheduler_choice:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.float16
    scaler = GradScaler(enabled=use_amp)
    if use_amp and is_main:
        tprint(f"AMP enabled with {amp_dtype}, GradScaler=on")
    
    raw_model = model.module if use_ddp else model
    
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
    checkpoint_records = []
    
    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = {}
        for key in epoch_pose_errors:
            epoch_pose_errors[key] = 0
        
        epoch_start = time.time()
        out_init_loss_choice = epoch < 5
        t_data_total, t_prep_total, t_compute_total, t_vis_total = 0.0, 0.0, 0.0, 0.0
        vis_count = 0
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
            init_T_to_camera_np, _, _ = generate_single_perturbation_from_T(gt_T_to_camera_np, angle_range_deg=train_noise["angle_range_deg"], trans_range=train_noise["trans_range"])
            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device, non_blocking=True)
            if xyz_only_choise:
                pcs_np = np.array(pcs)[:, :, :3]
            else:
                pcs_np = np.array(pcs)
            pcs_t = torch.from_numpy(pcs_np).float().to(device, non_blocking=True)
            gt_T_to_camera_t = torch.from_numpy(gt_T_to_camera_np).to(device, non_blocking=True)
            init_T_to_camera_t = torch.from_numpy(init_T_to_camera_np.astype(np.float32)).to(device, non_blocking=True)
            B_cur = gt_T_to_camera_t.shape[0]
            post_cam2ego_T = _identity_4x4.unsqueeze(0).expand(B_cur, -1, -1)
            intrinsic_matrix = torch.from_numpy(np.array(intrinsics, dtype=np.float32)).to(device, non_blocking=True)
            t_prep_total += time.time() - t_prep_start

            t_compute_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                T_pred, init_loss, loss = model(resize_imgs, pcs_t, gt_T_to_camera_t, init_T_to_camera_t, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=out_init_loss_choice)
                total_loss = loss["total_loss"]
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            scaler.step(optimizer)
            scaler.update()
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
                tprint(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}, "
                       f"Trans: {batch_errors['trans_error']:.4f}m (Fwd:{batch_errors['fwd_error']:.4f} Lat:{batch_errors['lat_error']:.4f} Ht:{batch_errors['ht_error']:.4f}), "
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
                        debug=debug_vis
                    )
                    
                    vis_image_tb = prepare_image_for_tensorboard(vis_image)
                    writer.add_image('Train/Projection', vis_image_tb, global_step)
                    
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
            
            global_step += 1
            t_iter_start = time.time()
        
        epoch_time = time.time() - epoch_start
        if is_main:
            steps_per_sec = len(train_loader) / epoch_time
            t_other = epoch_time - t_data_total - t_prep_total - t_compute_total - t_vis_total
            tprint(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.1f}s ({epoch_time/60:.1f}min), {steps_per_sec:.2f} steps/s, {epoch_time/len(train_loader):.2f}s/step")
            tprint(f"  Profiling: data_load={t_data_total:.1f}s({t_data_total/epoch_time*100:.1f}%), "
                   f"prep={t_prep_total:.1f}s({t_prep_total/epoch_time*100:.1f}%), "
                   f"compute={t_compute_total:.1f}s({t_compute_total/epoch_time*100:.1f}%), "
                   f"vis={t_vis_total:.1f}s({t_vis_total/epoch_time*100:.1f}%, {vis_count}calls, {t_vis_total/max(vis_count,1):.1f}s/call), "
                   f"other={t_other:.1f}s({t_other/epoch_time*100:.1f}%)")

        if scheduler_choice:   
            scheduler.step()    
        
        if is_main:
            for key in train_loss.keys():
                train_loss[key] /= len(train_loader)
                tprint(f"Epoch [{epoch+1}/{num_epochs}], Train Loss {key}: {train_loss[key]:.4f}")
                writer.add_scalar(f"Loss/train/{key}", train_loss[key], epoch)
            
            for key in epoch_pose_errors:
                epoch_pose_errors[key] /= len(train_loader)
            
            tprint(f"Epoch [{epoch+1}/{num_epochs}], Train Pose Error - "
                   f"Trans: {epoch_pose_errors['trans_error']:.4f}m "
                   f"(Fwd:{epoch_pose_errors['fwd_error']:.4f} Lat:{epoch_pose_errors['lat_error']:.4f} Ht:{epoch_pose_errors['ht_error']:.4f}), "
                   f"Rot: {epoch_pose_errors['rot_error']:.2f}° "
                   f"(R:{epoch_pose_errors['roll_error']:.2f} P:{epoch_pose_errors['pitch_error']:.2f} Y:{epoch_pose_errors['yaw_error']:.2f})")
            
            writer.add_scalar('Epoch/train/trans_error_m', epoch_pose_errors['trans_error'], epoch)
            writer.add_scalar('Epoch/train/fwd_error_m', epoch_pose_errors['fwd_error'], epoch)
            writer.add_scalar('Epoch/train/lat_error_m', epoch_pose_errors['lat_error'], epoch)
            writer.add_scalar('Epoch/train/ht_error_m', epoch_pose_errors['ht_error'], epoch)
            writer.add_scalar('Epoch/train/rot_error_deg', epoch_pose_errors['rot_error'], epoch)
            writer.add_scalar('Epoch/train/roll_error_deg', epoch_pose_errors['roll_error'], epoch)
            writer.add_scalar('Epoch/train/pitch_error_deg', epoch_pose_errors['pitch_error'], epoch)
            writer.add_scalar('Epoch/train/yaw_error_deg', epoch_pose_errors['yaw_error'], epoch)
            
            cur_train_loss = train_loss.get('total_loss', float('inf')) if train_loss else float('inf')
            if epoch_pose_errors['trans_error'] + epoch_pose_errors['rot_error'] * 0.1 < best_train['trans'] + best_train['rot'] * 0.1:
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
                            trans_range=eval_trans_range
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
                            
                            # 生成单样本可视化
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
                                point_radius=args.vis_point_radius
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
                                
                                f.write("\nTranslation Errors (in LiDAR coordinate system):\n")
                                f.write(f"  Total:   {errors['trans_error']:.6f} m\n")
                                f.write(f"  X (Fwd): {errors['fwd_error']:.6f} m\n")
                                f.write(f"  Y (Lat): {errors['lat_error']:.6f} m\n")
                                f.write(f"  Z (Ht):  {errors['ht_error']:.6f} m\n")
                                
                                f.write("\nRotation Errors (axis-angle):\n")
                                f.write(f"  Total:       {errors['rot_error']:.6f} deg\n")
                                f.write(f"  Roll (X):    {errors['roll_error']:.6f} deg\n")
                                f.write(f"  Pitch (Y):   {errors['pitch_error']:.6f} deg\n")
                                f.write(f"  Yaw (Z):     {errors['yaw_error']:.6f} deg\n")
                                
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
                    
                    f.write("Average Translation Errors (in LiDAR coordinate system):\n")
                    f.write(f"  Total:   {avg_errors['trans_error']:.6f} ± {std_errors['trans_error']:.6f} m\n")
                    f.write(f"  X (Fwd): {avg_errors['fwd_error']:.6f} ± {std_errors['fwd_error']:.6f} m\n")
                    f.write(f"  Y (Lat): {avg_errors['lat_error']:.6f} ± {std_errors['lat_error']:.6f} m\n")
                    f.write(f"  Z (Ht):  {avg_errors['ht_error']:.6f} ± {std_errors['ht_error']:.6f} m\n")
                    
                    f.write("\nAverage Rotation Errors (axis-angle):\n")
                    f.write(f"  Total:       {avg_errors['rot_error']:.6f} ± {std_errors['rot_error']:.6f} deg\n")
                    f.write(f"  Roll (X):    {avg_errors['roll_error']:.6f} ± {std_errors['roll_error']:.6f} deg\n")
                    f.write(f"  Pitch (Y):   {avg_errors['pitch_error']:.6f} ± {std_errors['pitch_error']:.6f} deg\n")
                    f.write(f"  Yaw (Z):     {avg_errors['yaw_error']:.6f} ± {std_errors['yaw_error']:.6f} deg\n")
                    f.write("\n" + "="*80 + "\n")
                
                avg_eval_errors = {key: np.mean(values) for key, values in all_errors.items()}
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
            
            with torch.no_grad():
                for batch_index, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    imgs, pcs, masks, gt_T_to_camera, intrinsics = batch_data
                    gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
                    init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(gt_T_to_camera_np, angle_range_deg=eval_angle_range, trans_range=eval_trans_range)
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
                            point_radius=args.vis_point_radius
                        )
                        
                        vis_image_tb = prepare_image_for_tensorboard(vis_image)
                        writer.add_image('Val/Projection', vis_image_tb, epoch)

            for key in val_loss.keys():
                val_loss[key] /= len(val_loader)
                tprint(f"Epoch [{epoch+1}/{num_epochs}], {eval_angle_range}_{eval_trans_range} Validation Loss {key}: {val_loss[key]:.4f}")
                writer.add_scalar(f"Loss/val/{key}", val_loss[key], epoch)
            
            # 记录验证集姿态误差
            for key in val_pose_errors:
                val_pose_errors[key] /= len(val_loader)
            
            tprint(f"Epoch [{epoch+1}/{num_epochs}], Val Pose Error - "
                   f"Trans: {val_pose_errors['trans_error']:.4f}m "
                   f"(Fwd:{val_pose_errors['fwd_error']:.4f} Lat:{val_pose_errors['lat_error']:.4f} Ht:{val_pose_errors['ht_error']:.4f}), "
                   f"Rot: {val_pose_errors['rot_error']:.2f}° "
                   f"(R:{val_pose_errors['roll_error']:.2f} P:{val_pose_errors['pitch_error']:.2f} Y:{val_pose_errors['yaw_error']:.2f})")
            
            writer.add_scalar('Epoch/val/trans_error_m', val_pose_errors['trans_error'], epoch)
            writer.add_scalar('Epoch/val/fwd_error_m', val_pose_errors['fwd_error'], epoch)
            writer.add_scalar('Epoch/val/lat_error_m', val_pose_errors['lat_error'], epoch)
            writer.add_scalar('Epoch/val/ht_error_m', val_pose_errors['ht_error'], epoch)
            writer.add_scalar('Epoch/val/rot_error_deg', val_pose_errors['rot_error'], epoch)
            writer.add_scalar('Epoch/val/roll_error_deg', val_pose_errors['roll_error'], epoch)
            writer.add_scalar('Epoch/val/pitch_error_deg', val_pose_errors['pitch_error'], epoch)
            writer.add_scalar('Epoch/val/yaw_error_deg', val_pose_errors['yaw_error'], epoch)
            
            if val_pose_errors['trans_error'] + val_pose_errors['rot_error'] * 0.1 < best_val['trans'] + best_val['rot'] * 0.1:
                cur_val_loss = val_loss.get('total_loss', float('inf')) if val_loss else float('inf')
                best_val.update({
                    'epoch': epoch + 1,
                    'loss': cur_val_loss,
                    'trans': val_pose_errors['trans_error'],
                    'rot': val_pose_errors['rot_error'],
                    'errors': dict(val_pose_errors),
                })

            val_loss = None
            loss = None

        if use_ddp:
            dist.barrier()

    if is_main:
        print("\n" + "=" * 80)
        print("训练完成总结 / Training Summary")
        print("=" * 80)
        
        if best_train['epoch'] > 0 and best_train['errors']:
            e = best_train['errors']
            print(f"\n  Best Train  (Epoch {best_train['epoch']}):")
            print(f"    Pose Error - Trans: {e['trans_error']:.4f}m "
                  f"(Fwd:{e['fwd_error']:.4f} Lat:{e['lat_error']:.4f} Ht:{e['ht_error']:.4f}), "
                  f"Rot: {e['rot_error']:.2f}° "
                  f"(R:{e['roll_error']:.2f} P:{e['pitch_error']:.2f} Y:{e['yaw_error']:.2f})")
        
        if best_val['epoch'] > 0 and best_val['errors']:
            e = best_val['errors']
            print(f"\n  Best Val    (Epoch {best_val['epoch']}):")
            print(f"    Pose Error - Trans: {e['trans_error']:.4f}m "
                  f"(Fwd:{e['fwd_error']:.4f} Lat:{e['lat_error']:.4f} Ht:{e['ht_error']:.4f}), "
                  f"Rot: {e['rot_error']:.2f}° "
                  f"(R:{e['roll_error']:.2f} P:{e['pitch_error']:.2f} Y:{e['yaw_error']:.2f})")
        
        if checkpoint_records:
            has_eval = any(r['eval'] is not None for r in checkpoint_records)
            
            print(f"\n  Checkpoint Performance Table ({len(checkpoint_records)} checkpoints)")
            print("  " + "-" * 150)
            if has_eval:
                print(f"  {'Ckpt':>6} | {'--- Train ---':^50s} | {'--- Eval (Validation) ---':^50s} | {'Gap':>6}")
                print(f"  {'Epoch':>6} | {'Trans(m)':>8} {'Fwd':>6} {'Lat':>6} {'Ht':>6} | {'Rot(°)':>6} {'R':>5} {'P':>5} {'Y':>5}"
                      f" | {'Trans(m)':>8} {'Fwd':>6} {'Lat':>6} {'Ht':>6} | {'Rot(°)':>6} {'R':>5} {'P':>5} {'Y':>5}"
                      f" | {'ΔTrans':>6}")
            else:
                print(f"  {'Ckpt':>6} | {'--- Train ---':^50s}")
                print(f"  {'Epoch':>6} | {'Trans(m)':>8} {'Fwd':>6} {'Lat':>6} {'Ht':>6} | {'Rot(°)':>6} {'R':>5} {'P':>5} {'Y':>5}")
            print("  " + "-" * 150)
            
            best_eval_idx = -1
            if has_eval:
                eval_scores = [(i, r['eval']['trans_error'] + r['eval']['rot_error'] * 0.1) 
                               for i, r in enumerate(checkpoint_records) if r['eval'] is not None]
                if eval_scores:
                    best_eval_idx = min(eval_scores, key=lambda x: x[1])[0]
            
            for i, rec in enumerate(checkpoint_records):
                t = rec['train']
                marker = " *" if i == best_eval_idx else "  "
                line = (f"  {rec['epoch']:>6} | {t['trans_error']:>8.4f} {t['fwd_error']:>6.4f} {t['lat_error']:>6.4f} {t['ht_error']:>6.4f}"
                        f" | {t['rot_error']:>6.2f} {t['roll_error']:>5.2f} {t['pitch_error']:>5.2f} {t['yaw_error']:>5.2f}")
                if has_eval:
                    if rec['eval'] is not None:
                        e = rec['eval']
                        gap = e['trans_error'] - t['trans_error']
                        line += (f" | {e['trans_error']:>8.4f} {e['fwd_error']:>6.4f} {e['lat_error']:>6.4f} {e['ht_error']:>6.4f}"
                                 f" | {e['rot_error']:>6.2f} {e['roll_error']:>5.2f} {e['pitch_error']:>5.2f} {e['yaw_error']:>5.2f}"
                                 f" | {gap:>+6.4f}{marker}")
                    else:
                        line += f" | {'N/A':>8} {'':>6} {'':>6} {'':>6} | {'N/A':>6} {'':>5} {'':>5} {'':>5} | {'':>6}{marker}"
                print(line)
            
            print("  " + "-" * 150)
            if best_eval_idx >= 0:
                print(f"  * = Best eval checkpoint (Epoch {checkpoint_records[best_eval_idx]['epoch']})")
            print(f"  ΔTrans = Eval Trans - Train Trans (泛化差距, 越小越好)")
        
        summary_path = os.path.join(log_dir, "training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Training Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Epochs: {num_epochs}\n")
            f.write(f"Train Noise: angle={train_noise['angle_range_deg']}°, trans={train_noise['trans_range']}m\n")
            f.write(f"Eval  Noise: angle={eval_noise['angle_range_deg']}°, trans={eval_noise['trans_range']}m\n\n")
            
            if best_train['epoch'] > 0 and best_train['errors']:
                e = best_train['errors']
                f.write(f"Best Train (Epoch {best_train['epoch']}):\n")
                f.write(f"  Trans: {e['trans_error']:.4f}m (Fwd:{e['fwd_error']:.4f} Lat:{e['lat_error']:.4f} Ht:{e['ht_error']:.4f})\n")
                f.write(f"  Rot:   {e['rot_error']:.2f}° (R:{e['roll_error']:.2f} P:{e['pitch_error']:.2f} Y:{e['yaw_error']:.2f})\n\n")
            
            if best_val['epoch'] > 0 and best_val['errors']:
                e = best_val['errors']
                f.write(f"Best Val (Epoch {best_val['epoch']}):\n")
                f.write(f"  Trans: {e['trans_error']:.4f}m (Fwd:{e['fwd_error']:.4f} Lat:{e['lat_error']:.4f} Ht:{e['ht_error']:.4f})\n")
                f.write(f"  Rot:   {e['rot_error']:.2f}° (R:{e['roll_error']:.2f} P:{e['pitch_error']:.2f} Y:{e['yaw_error']:.2f})\n\n")
            
            if checkpoint_records:
                has_eval = any(r['eval'] is not None for r in checkpoint_records)
                f.write(f"Checkpoint Performance Table ({len(checkpoint_records)} checkpoints)\n")
                f.write("-" * 150 + "\n")
                if has_eval:
                    f.write(f"{'Ckpt':>6} | {'--- Train ---':^50s} | {'--- Eval (Validation) ---':^50s} | {'Gap':>6}\n")
                    f.write(f"{'Epoch':>6} | {'Trans(m)':>8} {'Fwd':>6} {'Lat':>6} {'Ht':>6} | {'Rot(°)':>6} {'R':>5} {'P':>5} {'Y':>5}"
                            f" | {'Trans(m)':>8} {'Fwd':>6} {'Lat':>6} {'Ht':>6} | {'Rot(°)':>6} {'R':>5} {'P':>5} {'Y':>5}"
                            f" | {'ΔTrans':>6}\n")
                else:
                    f.write(f"{'Ckpt':>6} | {'--- Train ---':^50s}\n")
                    f.write(f"{'Epoch':>6} | {'Trans(m)':>8} {'Fwd':>6} {'Lat':>6} {'Ht':>6} | {'Rot(°)':>6} {'R':>5} {'P':>5} {'Y':>5}\n")
                f.write("-" * 150 + "\n")
                
                best_eval_idx = -1
                if has_eval:
                    eval_scores = [(i, r['eval']['trans_error'] + r['eval']['rot_error'] * 0.1) 
                                   for i, r in enumerate(checkpoint_records) if r['eval'] is not None]
                    if eval_scores:
                        best_eval_idx = min(eval_scores, key=lambda x: x[1])[0]
                
                for i, rec in enumerate(checkpoint_records):
                    t = rec['train']
                    marker = " *" if i == best_eval_idx else "  "
                    line = (f"{rec['epoch']:>6} | {t['trans_error']:>8.4f} {t['fwd_error']:>6.4f} {t['lat_error']:>6.4f} {t['ht_error']:>6.4f}"
                            f" | {t['rot_error']:>6.2f} {t['roll_error']:>5.2f} {t['pitch_error']:>5.2f} {t['yaw_error']:>5.2f}")
                    if has_eval:
                        if rec['eval'] is not None:
                            e = rec['eval']
                            gap = e['trans_error'] - t['trans_error']
                            line += (f" | {e['trans_error']:>8.4f} {e['fwd_error']:>6.4f} {e['lat_error']:>6.4f} {e['ht_error']:>6.4f}"
                                     f" | {e['rot_error']:>6.2f} {e['roll_error']:>5.2f} {e['pitch_error']:>5.2f} {e['yaw_error']:>5.2f}"
                                     f" | {gap:>+6.4f}{marker}")
                        else:
                            line += f" | {'N/A':>8} {'':>6} {'':>6} {'':>6} | {'N/A':>6} {'':>5} {'':>5} {'':>5} | {'':>6}{marker}"
                    f.write(line + "\n")
                
                f.write("-" * 150 + "\n")
                if best_eval_idx >= 0:
                    f.write(f"* = Best eval checkpoint (Epoch {checkpoint_records[best_eval_idx]['epoch']})\n")
                f.write(f"ΔTrans = Eval Trans - Train Trans (generalization gap)\n")
        
        print(f"\n  训练总结已保存到: {summary_path}")
        print("=" * 80)
    
    if is_main and writer is not None:
        writer.close()
        tprint(f"Logs are saved at {log_dir}")
    cleanup_ddp()


if __name__ == "__main__":
    main()
