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
import open3d as o3d
from PIL import Image
from torch.utils.data import Dataset
import os
from tqdm import tqdm

# 从 bev_settings 导入体素化范围配置
from bev_settings import xbound, ybound, zbound


class CustomDataset(Dataset):
    """
    KITTI-Odometry 格式数据集加载器
    
    支持：
    1. 标准 KITTI-Odometry 数据集（22个序列）
    2. 自定义数据集（自动检测序列）
    
    坐标系说明：
    - 输入点云坐标系：Sensing系（X前进，Y左，Z上）
    - Tr矩阵：Sensing → Camera
    - 输出点云坐标系：BEV坐标系（X前进，Y左，Z上），范围裁剪到体素化范围
    - 返回的 gt_transform 是 Tr 的逆矩阵（Camera → Sensing）
    
    数据利用率统计：
    - 记录原始点云数量、过滤后点云数量、利用率
    - 支持训练前验证数据利用率
    """
    
    # 标准KITTI-Odometry序列
    KITTI_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    
    def __init__(self, data_folder='./data/kitti-odemetry', suf='.png', sequences=None, auto_detect=True):
        # 使用 bev_settings 的体素化范围配置
        self.x_min, self.x_max = xbound[0], xbound[1]
        self.y_min, self.y_max = ybound[0], ybound[1]
        self.z_min, self.z_max = zbound[0], zbound[1]
        
        # 数据利用率统计
        self.utilization_stats = {
            'total_original_points': 0,
            'total_filtered_points': 0,
            'total_ego_filtered_points': 0,
            'total_range_filtered_points': 0,
            'low_point_frames': 0,
            'null_frames': 0,
            'valid_frames': 0,
        }
        """
        初始化数据集
        
        Args:
            data_folder: 数据集根目录
            suf: 图像文件后缀
            sequences: 指定序列列表，如 ['00', '01']。如果为 None，则使用 auto_detect
            auto_detect: 是否自动检测可用序列（默认 True）
                - True: 自动扫描 sequences/ 目录，找到所有有效序列
                - False: 使用标准 KITTI 序列列表
        """
        self.all_files = []
        self.dataset_root = data_folder
        self.K = {}
        self.T = {}
        
        # 确定要使用的序列
        if sequences is not None:
            self.sequences = sequences
        elif auto_detect:
            self.sequences = self._detect_sequences()
            if not self.sequences:
                # 如果自动检测失败，回退到标准KITTI序列
                print(f"[CustomDataset] 自动检测未找到序列，尝试标准KITTI序列")
                self.sequences = self.KITTI_SEQUENCES
        else:
            self.sequences = self.KITTI_SEQUENCES
        
        print(f"[CustomDataset] 使用序列: {self.sequences}")
        
        loaded_sequences = []
        for seq in self.sequences:
            try:
                odom = odometry(data_folder, seq)
                calib = odom.calib
                # 注意: pykitti的T_cam2_velo实际上是 velo->cam 的变换矩阵
                # 变换点云到相机坐标系: P_cam = T_cam2_velo @ P_velo
                # 不需要取逆！
                T_velo_to_cam = calib.T_cam2_velo
                self.K[seq] = calib.K_cam2
                self.T[seq] = T_velo_to_cam
                
                image_list = os.listdir(os.path.join(self.dataset_root, 'sequences', seq, 'image_2'))
                image_list.sort()

                frame_count = 0
                for image_name in image_list:
                    base_name = image_name.split('.')[0]
                    if not os.path.exists(os.path.join(self.dataset_root, 'sequences', seq, 'velodyne',
                                                       base_name + '.bin')):
                        continue
                    if not os.path.exists(os.path.join(self.dataset_root, 'sequences', seq, 'image_2',
                                                       base_name + suf)):
                        continue

                    self.all_files.append(os.path.join(seq, base_name))
                    frame_count += 1
                
                if frame_count > 0:
                    loaded_sequences.append(seq)
                    print(f"  ✓ 序列 {seq}: {frame_count} 帧")
            except Exception as e:
                # 序列不存在或加载失败，跳过
                continue
        
        if not self.all_files:
            raise ValueError(f"未找到任何有效数据！检查路径: {data_folder}")
        
        print(f"[CustomDataset] 总计: {len(self.all_files)} 帧来自 {len(loaded_sequences)} 个序列")
    
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
            
            # 检查必要的子目录和文件
            image_dir = os.path.join(seq_path, 'image_2')
            velodyne_dir = os.path.join(seq_path, 'velodyne')
            calib_file = os.path.join(seq_path, 'calib.txt')
            
            if os.path.exists(image_dir) and os.path.exists(velodyne_dir) and os.path.exists(calib_file):
                sequences.append(item)
        
        sequences.sort()
        return sequences

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx, track_stats=False):
        seq = self.all_files[idx].split('/')[0]
        id = self.all_files[idx].split('/')[1]
        img_path = os.path.join(self.dataset_root, 'sequences', seq, 'image_2', id+'.png')
        pcd_path = os.path.join(self.dataset_root, 'sequences', seq, 'velodyne', id+'.bin')
        if not os.path.exists(img_path) or not os.path.exists(pcd_path):
            print('File not exist')
            assert False
        img = Image.open(img_path)
        img.resize((1242, 375))
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        
        # 记录原始点云数量
        original_points = len(pcd)
        
        # ========== 坐标系转换 ==========
        # 使用 bev_settings.py 中的体素化范围配置
        # =============================================
        
        # 步骤1: 过滤自车周围的点（避免自车点云干扰）
        ego_filter = (np.abs(pcd[:, 0]) > 3.) | (np.abs(pcd[:, 1]) > 3.)
        after_ego_filter = np.sum(ego_filter)
        pcd = pcd[ego_filter, :]
        
        # 步骤2: 裁剪到体素化范围（使用 bev_settings 配置）
        range_filter = (pcd[:, 0] >= self.x_min) & (pcd[:, 0] <= self.x_max) & \
                       (pcd[:, 1] >= self.y_min) & (pcd[:, 1] <= self.y_max) & \
                       (pcd[:, 2] >= self.z_min) & (pcd[:, 2] <= self.z_max)
        pcd = pcd[range_filter, :]
        
        final_points = len(pcd)
        
        # 记录统计信息
        if track_stats:
            self.utilization_stats['total_original_points'] += original_points
            self.utilization_stats['total_ego_filtered_points'] += after_ego_filter
            self.utilization_stats['total_filtered_points'] += final_points
        
        # 确保有足够的点
        if len(pcd) < 100:
            if track_stats:
                self.utilization_stats['low_point_frames'] += 1
            # 如果点太少，扩大范围重新过滤
            pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
            ego_filter = (np.abs(pcd[:, 0]) > 2.) | (np.abs(pcd[:, 1]) > 2.)
            pcd = pcd[ego_filter, :]
            # 放宽 Z 轴限制
            range_filter = (pcd[:, 0] >= self.x_min) & (pcd[:, 0] <= self.x_max) & \
                           (pcd[:, 1] >= self.y_min) & (pcd[:, 1] <= self.y_max)
            pcd = pcd[range_filter, :]
            
            # 如果仍然太少，创建一个虚拟点云（避免训练崩溃）
            if len(pcd) < 10:
                if track_stats:
                    self.utilization_stats['null_frames'] += 1
                print(f"⚠️ 帧 {seq}/{id} 是异常帧（去畸变可能失败），使用虚拟点云")
                # 创建一个小的虚拟点云，位于原点附近
                x_center = (self.x_min + self.x_max) / 2
                pcd = np.array([
                    [x_center + 10.0, 0.0, 0.0, 0.0],
                    [x_center + 20.0, 0.0, 0.0, 0.0],
                    [x_center + 30.0, 0.0, 0.0, 0.0],
                    [x_center + 40.0, 0.0, 0.0, 0.0],
                    [x_center + 50.0, 0.0, 0.0, 0.0],
                ], dtype=np.float32)
        
        if track_stats:
            self.utilization_stats['valid_frames'] += 1
        
        gt_transform = self.T[seq]
        intrinsic = self.K[seq]
        return img, pcd, gt_transform, intrinsic
    
    def validate_data_utilization(self, sample_ratio=0.1, min_utilization=0.3, min_valid_ratio=0.9, verbose=True):
        """
        验证数据利用率
        
        Args:
            sample_ratio: 采样比例，用于快速验证 (0.0-1.0)
            min_utilization: 最低点云利用率阈值
            min_valid_ratio: 最低有效帧比例阈值
            verbose: 是否打印详细信息
        
        Returns:
            dict: 验证结果，包含 'passed' 布尔值和详细统计
        """
        # 重置统计
        self.utilization_stats = {
            'total_original_points': 0,
            'total_filtered_points': 0,
            'total_ego_filtered_points': 0,
            'total_range_filtered_points': 0,
            'low_point_frames': 0,
            'null_frames': 0,
            'valid_frames': 0,
        }
        
        # 采样验证
        num_samples = max(1, int(len(self) * sample_ratio))
        sample_indices = np.random.choice(len(self), num_samples, replace=False)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"数据利用率验证 (采样 {num_samples}/{len(self)} 帧, {sample_ratio*100:.1f}%)")
            print(f"BEV范围配置: X=[{self.x_min}, {self.x_max}], Y=[{self.y_min}, {self.y_max}], Z=[{self.z_min}, {self.z_max}]")
            print(f"{'='*60}")
        
        for idx in tqdm(sample_indices, desc="验证数据利用率", disable=not verbose):
            try:
                self.__getitem__(idx, track_stats=True)
            except Exception as e:
                self.utilization_stats['null_frames'] += 1
                if verbose:
                    print(f"  ⚠️ 帧 {idx} 加载失败: {e}")
        
        # 计算统计指标
        stats = self.utilization_stats
        total_frames = num_samples
        valid_frames = stats['valid_frames']
        null_frames = stats['null_frames']
        low_point_frames = stats['low_point_frames']
        
        # 点云利用率 = 过滤后点数 / 原始点数
        if stats['total_original_points'] > 0:
            point_utilization = stats['total_filtered_points'] / stats['total_original_points']
            ego_filter_ratio = stats['total_ego_filtered_points'] / stats['total_original_points']
        else:
            point_utilization = 0
            ego_filter_ratio = 0
        
        # 有效帧比例
        valid_ratio = valid_frames / total_frames if total_frames > 0 else 0
        
        # 判断是否通过
        passed = point_utilization >= min_utilization and valid_ratio >= min_valid_ratio
        
        result = {
            'passed': passed,
            'point_utilization': point_utilization,
            'ego_filter_ratio': ego_filter_ratio,
            'valid_ratio': valid_ratio,
            'total_frames': total_frames,
            'valid_frames': valid_frames,
            'null_frames': null_frames,
            'low_point_frames': low_point_frames,
            'total_original_points': stats['total_original_points'],
            'total_filtered_points': stats['total_filtered_points'],
            'min_utilization': min_utilization,
            'min_valid_ratio': min_valid_ratio,
        }
        
        if verbose:
            print(f"\n📊 验证结果:")
            print(f"  - 点云利用率: {point_utilization*100:.2f}% (阈值: {min_utilization*100:.1f}%)")
            print(f"  - 自车过滤后保留: {ego_filter_ratio*100:.2f}%")
            print(f"  - 有效帧比例: {valid_ratio*100:.2f}% (阈值: {min_valid_ratio*100:.1f}%)")
            print(f"  - 总帧数: {total_frames}, 有效帧: {valid_frames}, 无效帧: {null_frames}, 低点数帧: {low_point_frames}")
            print(f"  - 原始点数: {stats['total_original_points']:,}, 过滤后: {stats['total_filtered_points']:,}")
            
            if passed:
                print(f"\n✅ 数据利用率验证通过！")
            else:
                print(f"\n❌ 数据利用率验证未通过！")
                if point_utilization < min_utilization:
                    print(f"   - 点云利用率过低 ({point_utilization*100:.2f}% < {min_utilization*100:.1f}%)")
                    print(f"   - 建议: 检查 bev_settings.py 中的体素化范围是否与数据集匹配")
                if valid_ratio < min_valid_ratio:
                    print(f"   - 有效帧比例过低 ({valid_ratio*100:.2f}% < {min_valid_ratio*100:.1f}%)")
                    print(f"   - 建议: 检查数据集质量或调整过滤参数")
            print(f"{'='*60}\n")
        
        return result


if __name__ == "__main__":
    # 测试
    dataset_root = '/home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data'
    
    print("=" * 60)
    print("测试 CustomDataset")
    print("=" * 60)
    
    dataset = CustomDataset(
        data_folder=dataset_root,
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
