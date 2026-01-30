"""
Custom Dataset Loader for BEVCalib
专门用于加载自定义数据集（如 B26A），处理非标准 KITTI 格式的数据

主要特性:
1. 自动检测点云坐标系（LiDAR vs Camera）
2. 自动转换到 BEV 坐标系
3. 支持更大的探测范围（可配置）
4. 兼容 KITTI-Odometry 目录结构
"""

from pykitti import odometry
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, 
                 data_folder='./data/custom-dataset', 
                 suf='.png',
                 sequences=None,
                 auto_detect=True,
                 max_range=90.0,
                 detect_coordinate_system=True):
        """
        自定义数据集加载器
        
        Args:
            data_folder: 数据集根目录
            suf: 图像文件后缀
            sequences: 指定序列列表，如 ['00', '01']。如果为 None，则自动检测
            auto_detect: 是否自动检测可用序列（默认 True）
            max_range: 点云最大范围（米），超出此范围的点将被裁剪。默认 90.0
            detect_coordinate_system: 是否自动检测并转换坐标系（默认 True）
        """
        self.all_files = []
        self.dataset_root = data_folder
        self.K = {}
        self.T = {}
        self.max_range = max_range
        self.detect_coordinate_system = detect_coordinate_system
        
        # 确定要使用的序列
        if sequences is not None:
            self.sequences = sequences
        elif auto_detect:
            self.sequences = self._detect_sequences()
            if not self.sequences:
                raise ValueError(f"在 {data_folder} 中未找到任何有效序列")
        else:
            raise ValueError("必须指定 sequences 或启用 auto_detect")
        
        print(f"[CustomDataset] 使用序列: {self.sequences}")
        print(f"[CustomDataset] 最大探测范围: {self.max_range}m")
        print(f"[CustomDataset] 坐标系自动检测: {self.detect_coordinate_system}")
        
        loaded_sequences = []
        for seq in self.sequences:
            try:
                odom = odometry(data_folder, seq)
                calib = odom.calib
                T_cam02_velo_np = calib.T_cam2_velo
                T_velo2_cam0_np = np.linalg.inv(T_cam02_velo_np)
                self.K[seq] = calib.K_cam2
                self.T[seq] = T_velo2_cam0_np
                
                image_list = os.listdir(os.path.join(self.dataset_root, 'sequences', seq, 'image_2'))
                image_list.sort()

                frame_count = 0
                for image_name in image_list:
                    base_name = image_name.split('.')[0]
                    if not os.path.exists(os.path.join(self.dataset_root, 'sequences', seq, 'velodyne', base_name + '.bin')):
                        continue
                    if not os.path.exists(os.path.join(self.dataset_root, 'sequences', seq, 'image_2', base_name + suf)):
                        continue

                    self.all_files.append(os.path.join(seq, base_name))
                    frame_count += 1
                
                if frame_count > 0:
                    loaded_sequences.append(seq)
                    print(f"  ✓ 序列 {seq}: {frame_count} 帧")
                else:
                    print(f"  ⚠ 序列 {seq}: 未找到有效帧")
                    
            except Exception as e:
                print(f"  ✗ 序列 {seq}: 加载失败 ({e})")
                continue
        
        if not self.all_files:
            raise ValueError(f"未找到任何有效数据！检查路径: {data_folder}")
        
        print(f"\n[CustomDataset] 总计: {len(self.all_files)} 帧来自 {len(loaded_sequences)} 个序列")
    
    def _detect_sequences(self):
        """自动检测数据集中存在的序列"""
        sequences = []
        sequences_dir = os.path.join(self.dataset_root, 'sequences')
        
        if not os.path.exists(sequences_dir):
            return sequences
        
        for item in os.listdir(sequences_dir):
            seq_path = os.path.join(sequences_dir, item)
            
            if not os.path.isdir(seq_path):
                continue
            
            image_dir = os.path.join(seq_path, 'image_2')
            velodyne_dir = os.path.join(seq_path, 'velodyne')
            calib_file = os.path.join(seq_path, 'calib.txt')
            
            if os.path.exists(image_dir) and os.path.exists(velodyne_dir) and os.path.exists(calib_file):
                sequences.append(item)
        
        sequences.sort()
        return sequences

    def _transform_to_bev_coordinates(self, pcd, seq):
        """
        将点云转换到 BEV 坐标系
        
        坐标系定义:
        - LiDAR 坐标系: X向前, Y向左, Z向上
        - 相机坐标系: X向右, Y向下, Z向前  
        - BEV 坐标系: X向前, Y向左, Z向上
        
        Args:
            pcd: 点云数据 (N, 4)，最后一列是强度
            seq: 序列ID
            
        Returns:
            转换后的点云 (N, 4)
        """
        # 检测坐标系：如果点云主要在X>0方向且中心远离原点，说明是LiDAR坐标系
        x_center = (pcd[:, 0].min() + pcd[:, 0].max()) / 2
        
        if abs(x_center) > 50:  # 点云中心距离原点超过50米
            # 判断是哪种坐标系
            if x_center > 50:
                # LiDAR 坐标系 (X向前，点云主要在正X方向)
                # 需要转换: LiDAR -> Camera -> BEV
                
                # 步骤1: LiDAR -> Camera
                gt_transform = self.T[seq]  # Tr的逆（相机到LiDAR）
                T_lidar_to_cam = np.linalg.inv(gt_transform)  # LiDAR到相机
                
                xyz = pcd[:, :3]
                ones = np.ones((xyz.shape[0], 1))
                xyz_hom = np.hstack([xyz, ones])  # (N, 4)
                xyz_cam = (T_lidar_to_cam @ xyz_hom.T).T  # (N, 4)
                
                # 步骤2: Camera -> BEV
                # Camera: X_right, Y_down, Z_forward -> BEV: X_forward, Y_left, Z_up
                xyz_bev = np.zeros_like(xyz_cam[:, :3])
                xyz_bev[:, 0] = xyz_cam[:, 2]   # BEV_X = Camera_Z (forward)
                xyz_bev[:, 1] = -xyz_cam[:, 0]  # BEV_Y = -Camera_X (left)
                xyz_bev[:, 2] = -xyz_cam[:, 1]  # BEV_Z = -Camera_Y (up)
                
                pcd[:, :3] = xyz_bev
            else:
                # 可能是其他坐标系，需要根据实际情况处理
                print(f"  ⚠️ 警告: 检测到异常坐标系 (X_center={x_center:.2f}m)，保持原样")
        
        return pcd

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        seq = self.all_files[idx].split('/')[0]
        id = self.all_files[idx].split('/')[1]
        img_path = os.path.join(self.dataset_root, 'sequences', seq, 'image_2', id + '.png')
        pcd_path = os.path.join(self.dataset_root, 'sequences', seq, 'velodyne', id + '.bin')
        
        if not os.path.exists(img_path) or not os.path.exists(pcd_path):
            raise FileNotFoundError(f"文件不存在: img={img_path}, pcd={pcd_path}")
        
        img = Image.open(img_path)
        img = img.resize((1242, 375))
        
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        
        # 自动检测并转换坐标系
        if self.detect_coordinate_system:
            pcd = self._transform_to_bev_coordinates(pcd, seq)
        
        # 裁剪到指定范围内（避免超出体素化范围）
        # 注意：这里使用 max_range 而不是硬编码的 90
        in_range = (pcd[:, 0] >= -self.max_range) & (pcd[:, 0] <= self.max_range) & \
                   (pcd[:, 1] >= -self.max_range) & (pcd[:, 1] <= self.max_range) & \
                   (pcd[:, 2] >= -10) & (pcd[:, 2] <= 10)
        pcd = pcd[in_range, :]
        
        # 过滤车身附近的点（去除车体自身）
        valid_ind = pcd[:, 0] < -3.
        valid_ind = valid_ind | (pcd[:, 0] > 3.)
        valid_ind = valid_ind | (pcd[:, 1] < -3.)
        valid_ind = valid_ind | (pcd[:, 1] > 3.)
        pcd = pcd[valid_ind, :]
        
        gt_transform = self.T[seq]
        intrinsic = self.K[seq]
        
        return img, pcd, gt_transform, intrinsic


if __name__ == "__main__":
    # 测试
    dataset_root = '/home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data'
    
    print("=" * 60)
    print("测试 CustomDataset")
    print("=" * 60)
    
    dataset = CustomDataset(
        data_folder=dataset_root,
        max_range=90.0,
        detect_coordinate_system=True
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试加载第一个样本
    print("\n测试加载样本...")
    img, pcd, gt_transform, intrinsic = dataset[0]
    
    print(f"图像尺寸: {img.size}")
    print(f"点云形状: {pcd.shape}")
    print(f"点云范围:")
    print(f"  X: [{pcd[:, 0].min():.2f}, {pcd[:, 0].max():.2f}]")
    print(f"  Y: [{pcd[:, 1].min():.2f}, {pcd[:, 1].max():.2f}]")
    print(f"  Z: [{pcd[:, 2].min():.2f}, {pcd[:, 2].max():.2f}]")
    print(f"变换矩阵形状: {gt_transform.shape}")
    print(f"内参矩阵形状: {intrinsic.shape}")
    
    print("\n✅ CustomDataset 测试通过！")
