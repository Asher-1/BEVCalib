#!/usr/bin/env python3
"""
BEVCalib 部署推理接口

提供:
1. 单帧推理 (single_frame_inference)
2. 迭代推理 (iterative_inference)
3. ZD 在线标定 (calibrate_zero_drift)
4. ZD 补偿推理 (inference_with_zd_compensation)

用法:
    from deploy.inference_api import BEVCalibInference
    
    engine = BEVCalibInference(
        ckpt_path="logs/.../ckpt_best_dual.pth",
        target_size=(960, 540),
    )
    
    # 单帧推理
    T_pred = engine.predict(image, pointcloud, init_T, intrinsic)
    
    # 迭代推理
    T_pred = engine.predict_iterative(image, pointcloud, init_T, intrinsic, n_iters=3)
    
    # ZD 标定 (需要连续帧)
    engine.calibrate_zd(frames_list)
    T_pred = engine.predict_with_zd_compensation(image, pointcloud, init_T, intrinsic)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import json
import cv2
from typing import Optional, Tuple, List, Dict

_deploy_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_deploy_dir)
sys.path.insert(0, os.path.join(_project_dir, 'kitti-bev-calib'))
sys.path.insert(0, _project_dir)


class BEVCalibInference:
    """BEVCalib 部署推理引擎"""
    
    def __init__(
        self,
        ckpt_path: str,
        target_size: Tuple[int, int] = (960, 540),
        device: str = 'cuda:0',
        zd_calibration_frames: int = 20,
    ):
        self.target_size = target_size
        self.target_width, self.target_height = target_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.zd_calibration_frames = zd_calibration_frames
        
        self.zd_offset = None  # (3,) RPY offset in radians
        self.zd_calibrated = False
        
        self._load_model(ckpt_path)
    
    def _load_model(self, ckpt_path: str):
        """加载 checkpoint 并初始化模型"""
        from evaluate_checkpoint import _build_model_from_ckpt
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        args = argparse.Namespace(
            target_width=self.target_width,
            target_height=self.target_height,
            pitch_vertical_bands=3,
        )
        
        self.model, self.ckpt_args, _ = _build_model_from_ckpt(
            args, checkpoint, self.device, rotation_only=True, quiet=True
        )
        self.model.eval()
        
        self.ckpt_path = ckpt_path
        print(f"[BEVCalibInference] 模型加载完成: {os.path.basename(ckpt_path)}")
        print(f"  分辨率: {self.target_width}x{self.target_height}")
        print(f"  设备: {self.device}")
    
    def preprocess_image(self, image: np.ndarray, intrinsic: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """图像预处理 + 内参缩放"""
        h, w = image.shape[:2]
        scale_x = self.target_width / w
        scale_y = self.target_height / h
        
        resized = cv2.resize(image, (self.target_width, self.target_height))
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = resized[:, :, :3]
        
        K_scaled = intrinsic.copy()
        K_scaled[0, :] *= scale_x
        K_scaled[1, :] *= scale_y
        
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        K_tensor = torch.from_numpy(K_scaled.astype(np.float32)).unsqueeze(0).to(self.device)
        
        return img_tensor, K_tensor
    
    def preprocess_pointcloud(self, pointcloud: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """点云预处理 + padding/mask"""
        pc = pointcloud[:, :3].astype(np.float32)
        mask = np.ones(pc.shape[0], dtype=np.float32)
        
        pc_tensor = torch.from_numpy(pc).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        
        return pc_tensor, mask_tensor
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        pointcloud: np.ndarray,
        init_T: np.ndarray,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        """单帧推理
        
        Args:
            image: (H, W, 3) uint8 或 float
            pointcloud: (N, 3+) 点云
            init_T: (4, 4) 初始外参矩阵
            intrinsic: (3, 3) 相机内参
            
        Returns:
            T_pred: (4, 4) 矫正后的外参矩阵
        """
        img_t, K_t = self.preprocess_image(image, intrinsic)
        pc_t, mask_t = self.preprocess_pointcloud(pointcloud)
        
        init_T_t = torch.from_numpy(init_T.astype(np.float32)).unsqueeze(0).to(self.device)
        gt_T_t = init_T_t.clone()
        post_T = torch.eye(4).unsqueeze(0).to(self.device)
        
        T_pred, _, _ = self.model(
            img_t, pc_t, gt_T_t, init_T_t,
            post_T, K_t, masks=mask_t, out_init_loss=False
        )
        
        return T_pred[0].cpu().numpy()
    
    @torch.no_grad()
    def predict_iterative(
        self,
        image: np.ndarray,
        pointcloud: np.ndarray,
        init_T: np.ndarray,
        intrinsic: np.ndarray,
        n_iters: int = 3,
    ) -> np.ndarray:
        """迭代推理
        
        Args:
            n_iters: 迭代次数 (推荐 2-3)
            
        Returns:
            T_pred: (4, 4) 矫正后的外参矩阵
        """
        img_t, K_t = self.preprocess_image(image, intrinsic)
        pc_t, mask_t = self.preprocess_pointcloud(pointcloud)
        
        init_T_t = torch.from_numpy(init_T.astype(np.float32)).unsqueeze(0).to(self.device)
        
        if hasattr(self.model, 'iterative_inference'):
            T_pred = self.model.iterative_inference(
                img_t, pc_t, init_T_t, K_t,
                n_iters=n_iters, pcd_mask=mask_t
            )
        else:
            T_current = init_T_t
            post_T = torch.eye(4).unsqueeze(0).to(self.device)
            for _ in range(n_iters):
                T_pred, _, _ = self.model(
                    img_t, pc_t, T_current, T_current,
                    post_T, K_t, masks=mask_t, out_init_loss=False
                )
                T_current = T_pred.detach()
        
        return T_pred[0].cpu().numpy()
    
    @torch.no_grad()
    def calibrate_zd(
        self,
        frames: List[Dict],
        method: str = 'median',
    ) -> np.ndarray:
        """ZD 在线标定
        
        在已知 GT 或稳定运行条件下，估计模型的 Zero-Drift 偏差。
        
        Args:
            frames: [{'image': np.ndarray, 'pointcloud': np.ndarray, 
                     'gt_T': np.ndarray, 'intrinsic': np.ndarray}, ...]
            method: 'median' 或 'mean'
            
        Returns:
            zd_offset_deg: (3,) RPY ZD偏差 (degrees)
        """
        from scipy.spatial.transform import Rotation as R
        
        rpy_errors = []
        
        for frame in frames[:self.zd_calibration_frames]:
            T_pred = self.predict(
                frame['image'], frame['pointcloud'],
                frame['gt_T'], frame['intrinsic']
            )
            
            R_pred = T_pred[:3, :3]
            R_gt = frame['gt_T'][:3, :3]
            R_diff = R_pred @ R_gt.T
            
            r = R.from_matrix(R_diff)
            rpy = r.as_euler('xyz', degrees=False)
            rpy_errors.append(rpy)
        
        rpy_errors = np.array(rpy_errors)
        
        if method == 'median':
            self.zd_offset = np.median(rpy_errors, axis=0)
        else:
            self.zd_offset = np.mean(rpy_errors, axis=0)
        
        self.zd_calibrated = True
        zd_deg = np.degrees(self.zd_offset)
        
        print(f"[BEVCalibInference] ZD 标定完成 ({len(rpy_errors)} frames, {method})")
        print(f"  ZD offset (R/P/Y): {zd_deg[0]:.4f}° / {zd_deg[1]:.4f}° / {zd_deg[2]:.4f}°")
        print(f"  Total ZD: {np.linalg.norm(zd_deg):.4f}°")
        
        return zd_deg
    
    @torch.no_grad()
    def predict_with_zd_compensation(
        self,
        image: np.ndarray,
        pointcloud: np.ndarray,
        init_T: np.ndarray,
        intrinsic: np.ndarray,
        n_iters: int = 1,
    ) -> np.ndarray:
        """带 ZD 补偿的推理
        
        先进行推理，然后减去标定的 ZD 偏差。
        
        Returns:
            T_pred_compensated: (4, 4) ZD 补偿后的外参矩阵
        """
        from scipy.spatial.transform import Rotation as R
        
        if not self.zd_calibrated:
            raise RuntimeError("ZD 未标定! 请先调用 calibrate_zd()")
        
        if n_iters > 1:
            T_pred = self.predict_iterative(image, pointcloud, init_T, intrinsic, n_iters)
        else:
            T_pred = self.predict(image, pointcloud, init_T, intrinsic)
        
        R_zd = R.from_euler('xyz', -self.zd_offset).as_matrix().astype(np.float32)
        T_compensated = T_pred.copy()
        T_compensated[:3, :3] = R_zd @ T_pred[:3, :3]
        
        return T_compensated
    
    def export_config(self) -> Dict:
        """导出部署配置"""
        config = {
            'ckpt_path': self.ckpt_path,
            'target_size': list(self.target_size),
            'zd_calibrated': self.zd_calibrated,
            'zd_offset_rad': self.zd_offset.tolist() if self.zd_offset is not None else None,
            'zd_offset_deg': np.degrees(self.zd_offset).tolist() if self.zd_offset is not None else None,
        }
        return config
    
    def save_config(self, path: str):
        """保存部署配置到 JSON"""
        config = self.export_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[BEVCalibInference] 配置保存到: {path}")


def main():
    """CLI 入口: 单帧推理 demo"""
    parser = argparse.ArgumentParser(description='BEVCalib Inference API')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--target_width', type=int, default=960)
    parser.add_argument('--target_height', type=int, default=540)
    parser.add_argument('--n_iters', type=int, default=3)
    args = parser.parse_args()
    
    engine = BEVCalibInference(
        ckpt_path=args.ckpt_path,
        target_size=(args.target_width, args.target_height),
    )
    
    print(f"\nBEVCalib Inference Engine 就绪")
    print(f"  使用方法:")
    print(f"    T_pred = engine.predict(image, pointcloud, init_T, intrinsic)")
    print(f"    T_pred = engine.predict_iterative(..., n_iters={args.n_iters})")
    print(f"    engine.calibrate_zd(frames)")
    print(f"    T_pred = engine.predict_with_zd_compensation(...)")


if __name__ == '__main__':
    main()
