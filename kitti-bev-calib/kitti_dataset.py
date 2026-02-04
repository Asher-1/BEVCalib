from pykitti import odometry
import numpy as np
import open3d as o3d
from PIL import Image
from torch.utils.data import Dataset
import os

class KittiDataset(Dataset):
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
    
    ⚠️ 关键修复：
    - 点云需要从Sensing系转换到Camera系，再转换到BEV坐标系
    - BEV坐标系：X前进(Camera Z), Y左(-Camera X), Z上(-Camera Y)
    - 体素化范围：X: [-90, 90], Y: [-90, 90], Z: [-10, 10]
    """
    
    # 标准KITTI-Odometry序列
    KITTI_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    
    def __init__(self, data_folder='./data/kitti-odemetry', suf='.png', sequences=None, auto_detect=True):
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
                print(f"[KittiDataset] 自动检测未找到序列，尝试标准KITTI序列")
                self.sequences = self.KITTI_SEQUENCES
        else:
            self.sequences = self.KITTI_SEQUENCES
        
        print(f"[KittiDataset] 使用序列: {self.sequences}")
        
        loaded_sequences = []
        for seq in self.sequences:
            try:
                odom = odometry(data_folder, seq)
                calib = odom.calib
                T_cam02_velo_np = calib.T_cam2_velo
                # retore Tr from velo to cam
                T_velo2_cam0_np = np.linalg.inv(T_cam02_velo_np)
                self.K[seq] = calib.K_cam2
                self.T[seq] = T_velo2_cam0_np
                
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
        
        print(f"[KittiDataset] 总计: {len(self.all_files)} 帧来自 {len(loaded_sequences)} 个序列")
    
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
    
    def __getitem__(self, idx):
        seq = self.all_files[idx].split('/')[0]
        id = self.all_files[idx].split('/')[1]
        img_path = os.path.join(self.dataset_root, 'sequences', seq, 'image_2', id+'.png')
        pcd_path = os.path.join(self.dataset_root, 'sequences', seq, 'velodyne', id+'.bin')
        if not os.path.exists(img_path) or not os.path.exists(pcd_path):
            print('File not exist')
            assert False
        img = Image.open(img_path)
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        
        # ========== 坐标系转换（关键修复） ==========
        # 
        # 输入点云：Sensing系 (X前进, Y左, Z上)
        # 目标：BEV/Voxel坐标系，与体素化范围匹配
        # 
        # 体素化范围（bev_settings.py）：
        #   X: [-90, 90]  (前后方向)
        #   Y: [-90, 90]  (左右方向)
        #   Z: [-10, 10]  (高度方向)
        #
        # Sensing系已经是 (X前进, Y左, Z上)，与BEV坐标系一致
        # 只需要进行范围裁剪
        # =============================================
        
        # 步骤1: 过滤自车周围的点（避免自车点云干扰）
        ego_filter = (np.abs(pcd[:, 0]) > 3.) | (np.abs(pcd[:, 1]) > 3.)
        pcd = pcd[ego_filter, :]
        
        # 步骤2: 裁剪到体素化范围
        # X: [-90, 90] (前后方向)
        # Y: [-90, 90] (左右方向)  
        # Z: [-10, 10] (高度方向)
        range_filter = (pcd[:, 0] >= -90) & (pcd[:, 0] <= 90) & \
                       (pcd[:, 1] >= -90) & (pcd[:, 1] <= 90) & \
                       (pcd[:, 2] >= -10) & (pcd[:, 2] <= 10)
        pcd = pcd[range_filter, :]
        
        # 确保有足够的点
        if len(pcd) < 100:
            print(f"⚠️ 帧 {seq}/{id} 点云数量过少: {len(pcd)}")
            # 如果点太少，扩大范围重新过滤
            pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
            ego_filter = (np.abs(pcd[:, 0]) > 2.) | (np.abs(pcd[:, 1]) > 2.)
            pcd = pcd[ego_filter, :]
            range_filter = (pcd[:, 0] >= -90) & (pcd[:, 0] <= 90) & \
                           (pcd[:, 1] >= -90) & (pcd[:, 1] <= 90)
            pcd = pcd[range_filter, :]
            
            # 如果仍然太少，return None
            if len(pcd) < 10:
                return None
        
        gt_transform = self.T[seq]
        intrinsic = self.K[seq]
        return img, pcd, gt_transform, intrinsic
        
if __name__ == "__main__":
    dataset_root = './data/kitti-odemetry'
    train_dataset = KittiDataset(data_folder=dataset_root, split='train')
    print(len(train_dataset))
    val_dataset = KittiDataset(data_folder=dataset_root, split='val')
    print(len(val_dataset))
    print(train_dataset.all_files[-2])
    all_size = []
    for i in range(len(val_dataset)):
        img = val_dataset[i][0]
        # print(f"img size: {img.size}")
        if img.size not in all_size:
            all_size.append(img.size)
    print(all_size)
    # for i in range(len(train_dataset)):
    #     train_dataset[i]

    