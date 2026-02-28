import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kitti_dataset import KittiDataset
from custom_dataset import CustomDataset
from bev_calib import BEVCalib
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from torch.utils.data import random_split
import numpy as np
from tools import generate_single_perturbation_from_T
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser("Run inference / evaluation")
    parser.add_argument("--dataset_root", type=str, default="/data/HangQiu/data/kitti-odemetry")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the saved .pth checkpoint")
    parser.add_argument("--log_dir", type=str, default="./logs/inference")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--xyz_only", type=int, default=1)
    parser.add_argument("--angle_range_deg", type=float, default=20.0)
    parser.add_argument("--trans_range", type=float, default=1.5)
    # 图像尺寸参数
    parser.add_argument("--use_custom_dataset", type=int, default=0, help="使用自定义数据集模式 (1=是, 0=否)")
    parser.add_argument("--target_width", type=int, default=None, help="目标图像宽度")
    parser.add_argument("--target_height", type=int, default=None, help="目标图像高度")
    return parser.parse_args()


def get_target_size(use_custom_dataset, target_width=None, target_height=None):
    """根据数据集类型获取目标图像尺寸"""
    if target_width is not None and target_height is not None:
        return (target_width, target_height)
    
    if use_custom_dataset:
        # 自定义4K数据集: 640x360 (16:9)
        return (target_width or 640, target_height or 360)
    else:
        # KITTI: 704x256
        return (target_width or 704, target_height or 256)

def make_collate_fn(target_size):
    """创建带有指定 target_size 的 collate_fn"""
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

def crop_and_resize(item, size, intrinsics, crop=True, distortion=None):
    """
    图像预处理: 缩放 → 更新内参
    注意: 不再应用cv2.undistort (B26A相机输出已去畸变图像)
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

def rotation_matrix_to_euler_xyz(R):
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
    singular = sy < 1e-6

    roll = torch.where(
        ~singular,
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
        torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    )

    pitch = torch.atan2(-R[:, 2, 0], sy)

    yaw = torch.where(
        ~singular,  
        torch.atan2(R[:, 1, 0], R[:, 0, 0]),
        torch.zeros_like(roll)
    )

    return roll * 180.0 / torch.pi, pitch * 180.0 / torch.pi, yaw * 180.0 / torch.pi

def main():
    args = parse_args()
    xyz_only_choise = args.xyz_only > 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 获取目标图像尺寸
    target_size = get_target_size(
        use_custom_dataset=args.use_custom_dataset > 0,
        target_width=args.target_width,
        target_height=args.target_height
    )
    print(f"📐 目标图像尺寸: {target_size[0]}x{target_size[1]} (宽x高)")

    # 创建 collate_fn
    collate_fn = make_collate_fn(target_size)

    # 选择数据集类型
    if args.use_custom_dataset > 0:
        dataset = CustomDataset(args.dataset_root)
    else:
        dataset = KittiDataset(args.dataset_root)
    
    gen = torch.Generator().manual_seed(114514)
    split_size = int(0.8 * len(dataset))
    _, val_dataset = random_split(dataset, [split_size, len(dataset) - split_size],
                                  generator=gen)

    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # img_shape 格式为 (H, W)，而 target_size 是 (W, H)
    img_shape = (target_size[1], target_size[0])
    print(f"🔧 网络输入尺寸 (H, W): {img_shape}")
    
    model = BEVCalib(
        deformable=False,      
        bev_encoder=True,
        img_shape=img_shape
    ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_losses = []
    translation_losses = []
    rotation_losses = []
    quant_losses = []
    reproj_losses = []

    translation_errors = []
    rotation_errors = []

    eval_angle = np.array([args.angle_range_deg])
    eval_trans_range = np.array([args.trans_range])

    for angle, trans in zip(eval_angle, eval_trans_range):
        print(f"\nEvaluating perturb   angle {angle},  trans {trans}")
        step = 0

        with torch.no_grad():
            for b_idx, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(loader):
                gt_T_to_camera = np.array(gt_T_to_camera).astype(np.float32)
                init_T_to_camera, ang_err, trans_err = generate_single_perturbation_from_T(gt_T_to_camera, angle_range_deg=eval_angle, trans_range=eval_trans_range)
                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                if xyz_only_choise:
                    pcs = np.array(pcs)[:, :, :3]
                pcs = torch.from_numpy(np.array(pcs)).float().to(device)
                gt_T_to_camera = torch.from_numpy(gt_T_to_camera).float().to(device)
                init_T_to_camera = torch.from_numpy(init_T_to_camera).float().to(device)
                post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
                intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)
                T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)

                metrics = {k: v.item() for k, v in loss.items()}
                if init_loss:
                    metrics.update({f"init_{k}": v.item() for k, v in init_loss.items()})

                total_losses.append(metrics["total_loss"])
                translation_losses.append(metrics["translation_loss"])
                rotation_losses.append(metrics["rotation_loss"])
                quant_losses.append(metrics["quat_norm_loss"])
                reproj_losses.append(metrics["PC_reproj_loss"])

                # calculate the error 
                batch_size = T_pred.shape[0]
                translation_error = torch.abs((T_pred[:, :3, 3] - gt_T_to_camera[:, :3, 3]).reshape(batch_size, 3))
                rotation_error = torch.abs(torch.stack(rotation_matrix_to_euler_xyz(T_pred[:, :3, :3] @ gt_T_to_camera[:, :3, :3].transpose(-2, -1)), dim=0).reshape(batch_size, 3))

                translation_errors.append(translation_error)
                rotation_errors.append(rotation_error)

                # TensorBoard
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{angle}_{trans}/{k}", v, step)

                print(f"Batch {b_idx:04d} | " +
                      " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()]))
                step += 1

    writer.close()

    print("\nInference finished. Logs: ", log_dir)
    print("Average losses:")
    print(len(total_losses))
    print(f"Total loss: {np.mean(total_losses):.6f}")
    print(f"Translation loss: {np.mean(translation_losses):.6f}")
    print(f"Rotation loss: {np.mean(rotation_losses):.6f}")
    print(f"Quantization loss: {np.mean(quant_losses):.6f}")
    print(f"Reprojection loss: {np.mean(reproj_losses):.6f}")

    print("STD losses:")
    print(f"Total loss: {np.std(total_losses):.6f}")
    print(f"Translation loss: {np.std(translation_losses):.6f}")    
    print(f"Rotation loss: {np.std(rotation_losses):.6f}")
    print(f"Quantization loss: {np.std(quant_losses):.6f}")
    print(f"Reprojection loss: {np.std(reproj_losses):.6f}")

    print("\n")
    print("=" * 50)
    print("Errors")
    print("=" * 50)

    translation_errors = torch.cat(translation_errors, dim=0).cpu().numpy()
    rotation_errors = torch.cat(rotation_errors, dim=0).cpu().numpy()

    print("Average translation xyz error: ", np.mean(translation_errors, axis=0))
    print("Average rotation ypr error: ", np.mean(rotation_errors, axis=0))

    print("STD of translation xyz error: ", np.std(translation_errors, axis=0))
    print("STD of rotation ypr error: ", np.std(rotation_errors, axis=0))

if __name__ == "__main__":
    main()
