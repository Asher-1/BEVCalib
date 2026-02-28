import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kitti_dataset import KittiDataset
from custom_dataset import CustomDataset
from bev_calib import BEVCalib
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from torch.utils.data import random_split
import numpy as np
from pathlib import Path
from tools import generate_single_perturbation_from_T
import shutil
import cv2
import os
from visualization import (
    compute_batch_pose_errors,
    visualize_batch_projection,
    prepare_image_for_tensorboard,
    compute_pose_errors
)

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
    parser.add_argument("--eval_epoches", type=int, default=4)
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
    return parser.parse_args()

def crop_and_resize(item, size, intrinsics, crop=True, distortion=None):
    """
    图像预处理: 缩放 → 更新内参
    
    注意: 不再应用cv2.undistort，因为B26A等相机管线输出的图像已经是去畸变的。
    诊断验证: 对已去畸变图像再次undistort会引入最大~9px偏移(640x360)，
    导致GT投影与图像不对齐。
    
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
    
    # 注意: 不对图像做去畸变处理
    # B26A相机管线输出的图像已经过去畸变校正，calib.txt中的D系数是原始镜头畸变参数(仅供参考)
    # 诊断结果: 二次undistort/一次undistort差异比=0.971，证实图像已去畸变
    
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


def make_collate_fn(target_size):
    """
    创建带有指定 target_size 的 collate_fn
    
    Args:
        target_size: (width, height) 目标图像尺寸
    
    Returns:
        collate_fn 函数
    
    Note:
        数据集返回: (img, pcd, gt_transform, intrinsic, distortion)
        distortion 用于去畸变图像
    """
    def collate_fn(batch):
        # item结构: (img, pcd, gt_transform, intrinsic, distortion)
        # 去畸变 + 缩放在 crop_and_resize 中完成
        processed_data = [crop_and_resize(item[0], target_size, item[3], False, item[4]) for item in batch]
        imgs = [item[0] for item in processed_data]
        intrinsics = [item[1] for item in processed_data]

        gt_T_to_camera = [item[2] for item in batch]
        
        pcs = []
        masks = []
        max_num_points = 0
        for item in batch:
            max_num_points = max(max_num_points, item[1].shape[0])
        for item in batch:
            pc = item[1]
            masks.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(max_num_points - pc.shape[0])], axis=0))
            if pc.shape[0] < max_num_points:
                pc = np.concatenate([pc, np.full((max_num_points - pc.shape[0], pc.shape[1]), 999999)], axis=0)
            pcs.append(pc)

        return imgs, pcs, masks, gt_T_to_camera, intrinsics
    
    return collate_fn

def main():
    args = parse_args()
    print(args)
    num_epochs = args.num_epochs
    dataset_root = args.dataset_root
    log_dir = args.log_dir
    if args.label is not None:
        log_dir = os.path.join(log_dir, args.label)
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{log_dir}/{current_time}"
    ckpt_save_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)
    
    # 复制源代码到日志目录（排除logs目录避免无限递归）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    bev_calib_dir = os.path.join(parent_dir, 'kitti-bev-calib')
    dest_dir = os.path.join(log_dir, 'kitti-bev-calib')
    
    # 使用ignore参数排除logs目录
    def ignore_logs(directory, files):
        return ['logs', '__pycache__', '.git'] if 'logs' in files or '__pycache__' in files else []
    
    try:
        shutil.copytree(bev_calib_dir, dest_dir, dirs_exist_ok=True, 
                       ignore=shutil.ignore_patterns('logs', '__pycache__', '*.pyc', '.git*'))
    except Exception as e:
        print(f"警告: 复制源代码失败: {e}")
    
    writer = SummaryWriter(log_dir)
    
    # 选择数据集类型
    if args.use_custom_dataset:
        dataset = CustomDataset(dataset_root)
    else:
        print("使用 KittiDataset")
        dataset = KittiDataset(dataset_root)

    # 数据利用率校验
    if args.validate_data > 0:
        print("\n" + "="*60)
        print("开始数据利用率校验...")
        print("="*60)
        
        validation_result = dataset.validate_data_utilization(
            sample_ratio=args.validate_sample_ratio,
            min_utilization=args.min_point_utilization,
            min_valid_ratio=args.min_valid_ratio,
            verbose=True
        )
        
        # 将验证结果保存到日志
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
            print("\n❌ 错误: 数据利用率验证未通过，退出训练！")
            print("   可以通过以下方式解决：")
            print("   1. 检查 bev_settings.py 中的体素化范围配置是否与数据集匹配")
            print("   2. 调整 --min_point_utilization 或 --min_valid_ratio 阈值")
            print("   4. 使用 --validate_data=0 跳过验证（不推荐）")
            exit(1)
    else:
        print("\n⚠️ 跳过数据利用率校验 (--validate_data=0)")

    # 获取目标图像尺寸
    target_size = get_target_size(
        use_custom_dataset=args.use_custom_dataset > 0,
        target_width=args.target_width,
        target_height=args.target_height
    )
    print(f"\n📐 目标图像尺寸: {target_size[0]}x{target_size[1]} (宽x高)")
    if args.use_custom_dataset > 0:
        print(f"   (自定义数据集模式，保持16:9宽高比)")
    else:
        print(f"   (KITTI数据集模式)")
    
    # 创建 collate_fn
    collate_fn = make_collate_fn(target_size)
    
    generator = torch.Generator().manual_seed(114514)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        shuffle=False
    )

    deformable_choise = args.deformable > 0
    bev_encoder_choise = args.bev_encoder > 0
    xyz_only_choise = args.xyz_only > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # img_shape 格式为 (H, W)，而 target_size 是 (W, H)
    img_shape = (target_size[1], target_size[0])
    print(f"🔧 网络输入尺寸 (H, W): {img_shape}")
    
    model = BEVCalib(
        deformable=deformable_choise,
        bev_encoder=bev_encoder_choise,
        img_shape=img_shape
    ).to(device)

    if args.pretrain_ckpt is not None:
        state_dict = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
        print(f"Load pretrain model from {args.pretrain_ckpt}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler_choice = args.scheduler > 0
    if scheduler_choice:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    train_noise = {
        "angle_range_deg": args.angle_range_deg if args.angle_range_deg is not None else 20,
        "trans_range": args.trans_range if args.trans_range is not None else 1.5,
    }

    eval_noise = {
        "angle_range_deg": args.eval_angle_range_deg if args.eval_angle_range_deg is not None else train_noise["angle_range_deg"],
        "trans_range": args.eval_trans_range if args.eval_trans_range is not None else train_noise["trans_range"],
    }

    # 全局步数计数器
    global_step = 0
    
    # 累积误差统计
    epoch_pose_errors = {
        'trans_error': 0, 'fwd_error': 0, 'lat_error': 0, 'ht_error': 0,
        'rot_error': 0, 'roll_error': 0, 'pitch_error': 0, 'yaw_error': 0,
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = {}
        # 重置epoch误差统计
        for key in epoch_pose_errors:
            epoch_pose_errors[key] = 0
        
        out_init_loss_choice = False
        if epoch < 5:
            out_init_loss_choice = True # Output initial loss for the first 5 epochs
        for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(train_loader):
            gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
            init_T_to_camera_np, _, _ = generate_single_perturbation_from_T(gt_T_to_camera_np, angle_range_deg=train_noise["angle_range_deg"], trans_range=train_noise["trans_range"])
            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            if xyz_only_choise:
                pcs_np = np.array(pcs)[:, :, :3]
            else:
                pcs_np = np.array(pcs)
            pcs = torch.from_numpy(pcs_np).float().to(device)
            gt_T_to_camera = torch.from_numpy(gt_T_to_camera_np).float().to(device)
            init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device)
            post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
            intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)

            optimizer.zero_grad()
            # img, pc, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, cam_intrinsic
            T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=out_init_loss_choice)
            total_loss = loss["total_loss"]
            total_loss.backward()
            optimizer.step()
            
            # 计算详细的姿态误差
            with torch.no_grad():
                batch_errors = compute_batch_pose_errors(T_pred, gt_T_to_camera)
                for key in epoch_pose_errors:
                    epoch_pose_errors[key] += batch_errors[key]
            
            for key in loss.keys():
                if key not in train_loss.keys():
                    train_loss[key] = loss[key].item()
                else:
                    train_loss[key] += loss[key].item()
            
            if init_loss is not None:
                for key in init_loss.keys():
                    train_key = f"init_{key}"
                    if train_key not in train_loss.keys():
                        train_loss[train_key] = init_loss[key].item()
                    else:
                        train_loss[train_key] += init_loss[key].item()

            if batch_index % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}, "
                      f"Trans: {batch_errors['trans_error']:.4f}m (Fwd:{batch_errors['fwd_error']:.4f} Lat:{batch_errors['lat_error']:.4f} Ht:{batch_errors['ht_error']:.4f}), "
                      f"Rot: {batch_errors['rot_error']:.2f}°")
            
            # TensorBoard 可视化
            if args.enable_vis > 0 and batch_index % args.vis_freq == 0:
                with torch.no_grad():
                    # 准备可视化数据
                    imgs_np = np.array(imgs)  # (B, H, W, 3) BGR
                    masks_np = np.array(masks)
                    T_pred_np = T_pred.detach().cpu().numpy()
                    
                    # 首次可视化时输出调试信息
                    debug_vis = (epoch == 0 and batch_index == 0)
                    
                    # 创建可视化图像
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
                    
                    # 转换为TensorBoard格式并记录
                    vis_image_tb = prepare_image_for_tensorboard(vis_image)
                    writer.add_image('Train/Projection', vis_image_tb, global_step)
                    
                    # 记录当前batch的详细误差
                    writer.add_scalar('Train/PoseError/trans_error_m', batch_errors['trans_error'], global_step)
                    writer.add_scalar('Train/PoseError/fwd_error_m', batch_errors['fwd_error'], global_step)
                    writer.add_scalar('Train/PoseError/lat_error_m', batch_errors['lat_error'], global_step)
                    writer.add_scalar('Train/PoseError/ht_error_m', batch_errors['ht_error'], global_step)
                    writer.add_scalar('Train/PoseError/rot_error_deg', batch_errors['rot_error'], global_step)
                    writer.add_scalar('Train/PoseError/roll_error_deg', batch_errors['roll_error'], global_step)
                    writer.add_scalar('Train/PoseError/pitch_error_deg', batch_errors['pitch_error'], global_step)
                    writer.add_scalar('Train/PoseError/yaw_error_deg', batch_errors['yaw_error'], global_step)
            
            global_step += 1

        if scheduler_choice:   
            scheduler.step()    
        
        for key in train_loss.keys():
            train_loss[key] /= len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss {key}: {train_loss[key]:.4f}")
            writer.add_scalar(f"Loss/train/{key}", train_loss[key], epoch)
        
        # 记录epoch平均姿态误差
        for key in epoch_pose_errors:
            epoch_pose_errors[key] /= len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Pose Error - "
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
        
        if epoch == num_epochs - 1 or (args.save_ckpt_per_epoches > 0 and (epoch + 1) % args.save_ckpt_per_epoches == 0):
            ckpt_path = os.path.join(ckpt_save_dir, f"ckpt_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_noise': train_noise,
                'eval_noise': eval_noise,
                'args': vars(args) 
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            
            # 保存checkpoint对应的推理结果和可视化（使用验证集）
            if args.enable_ckpt_eval > 0:
                ckpt_eval_dir = os.path.join(ckpt_save_dir, f"ckpt_{epoch+1}_eval")
                os.makedirs(ckpt_eval_dir, exist_ok=True)
                
                print(f"Evaluating checkpoint {epoch+1} on validation set and saving visualization to {ckpt_eval_dir}...")
                model.eval()
                
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
                    for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
                        # 只评估前几个batch（避免太多样本）
                        if batch_index >= 5:  # 最多评估5个batch
                            break
                        
                        gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
                        init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(
                            gt_T_to_camera_np, 
                            angle_range_deg=eval_angle_range, 
                            trans_range=eval_trans_range
                        )
                        
                        resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                        if xyz_only_choise:
                            pcs_np = np.array(pcs)[:, :, :3]
                        else:
                            pcs_np = np.array(pcs)
                        pcs = torch.from_numpy(pcs_np).float().to(device)
                        gt_T_to_camera = torch.from_numpy(gt_T_to_camera_np).float().to(device)
                        init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device)
                        post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
                        intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)
                        
                        T_pred, _, _ = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)
                        
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
                
                print(f"Checkpoint evaluation complete: {sample_count} samples saved to {ckpt_eval_dir}")

        train_loss = None
        init_loss = None
        loss = None

        if epoch % args.eval_epoches == 0:
            eval_trans_range = eval_noise["trans_range"]
            eval_angle_range = eval_noise["angle_range_deg"]
            model.eval()
            val_loss = {}
            val_pose_errors = {
                'trans_error': 0, 'fwd_error': 0, 'lat_error': 0, 'ht_error': 0,
                'rot_error': 0, 'roll_error': 0, 'pitch_error': 0, 'yaw_error': 0,
            }
            
            with torch.no_grad():
                for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
                    # img, pc, depth_img, gt_T_to_camera, init_T_to_camera
                    gt_T_to_camera_np = np.array(gt_T_to_camera).astype(np.float32)
                    init_T_to_camera_np, ang_err, trans_err = generate_single_perturbation_from_T(gt_T_to_camera_np, angle_range_deg=eval_angle_range, trans_range=eval_trans_range)
                    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                    if xyz_only_choise:
                        pcs_np = np.array(pcs)[:, :, :3]
                    else:
                        pcs_np = np.array(pcs)
                    pcs = torch.from_numpy(pcs_np).float().to(device)
                    gt_T_to_camera = torch.from_numpy(gt_T_to_camera_np).float().to(device)
                    init_T_to_camera = torch.from_numpy(init_T_to_camera_np).float().to(device)
                    post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
                    intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)
                    T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)

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
                print(f"Epoch [{epoch+1}/{num_epochs}], {eval_angle_range}_{eval_trans_range} Validation Loss {key}: {val_loss[key]:.4f}")
                writer.add_scalar(f"Loss/val/{key}", val_loss[key], epoch)
            
            # 记录验证集姿态误差
            for key in val_pose_errors:
                val_pose_errors[key] /= len(val_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Pose Error - "
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

            val_loss = None
            loss = None

    writer.close()
    print(f"Logs are saved at {log_dir}")


if __name__ == "__main__":
    main()
