# BEVCalib 自定义数据集制作Pipeline

## 1. 概述

本文档详细说明如何为BEVCalib准备自定义数据集。BEVCalib需要同步的LiDAR点云和相机图像数据，以及它们之间的标定参数（变换矩阵和相机内参）。

## 2. 数据格式要求

### 2.1 目录结构

自定义数据集应遵循以下目录结构：

```
your-dataset/
├── sequences/
│   ├── 00/                    # 序列00
│   │   ├── image_2/           # 相机图像目录
│   │   │   ├── 000000.png     # 图像文件（PNG格式）
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── velodyne/          # 点云数据目录
│   │   │   ├── 000000.bin     # 点云文件（BIN格式）
│   │   │   ├── 000001.bin
│   │   │   └── ...
│   │   └── calib.txt          # 标定文件（必需）
│   ├── 01/                    # 序列01
│   │   └── ...
│   └── ...
```

**关键要求**：
- 每个序列必须有独立的目录（00, 01, 02, ...）
- 图像和点云文件名必须匹配（相同的前缀，如 `000000.png` 和 `000000.bin`）
- 每个序列目录下必须有 `calib.txt` 文件

### 2.2 图像数据格式

**格式要求**：
- **文件格式**：PNG（推荐）或JPG
- **命名规则**：6位数字 + 扩展名（如 `000000.png`, `000001.png`）
- **图像内容**：RGB彩色图像
- **尺寸要求**：无严格限制，但建议宽度在1000-2000像素之间，训练时会自动resize到(704, 256)

**示例代码**：
```python
from PIL import Image
import numpy as np

# 读取图像
img = Image.open('sequences/00/image_2/000000.png')
print(f"Image size: {img.size}")  # (width, height)
print(f"Image mode: {img.mode}")  # 应为 'RGB'
```

### 2.3 点云数据格式

**格式要求**：
- **文件格式**：BIN文件（二进制numpy数组）
- **数据类型**：`np.float32`
- **数据形状**：`(N, 4)` 或 `(N, 3)`
  - `(N, 4)`：包含 x, y, z, intensity（强度）
  - `(N, 3)`：仅包含 x, y, z（训练时使用 `--xyz_only 1`）
- **坐标系**：LiDAR坐标系（通常x向前，y向左，z向上）
- **单位**：米（meters）

**点云文件生成示例**：
```python
import numpy as np

# 假设你有点云数据（N个点，每个点有x, y, z, intensity）
points = np.array([
    [x1, y1, z1, intensity1],
    [x2, y2, z2, intensity2],
    ...
], dtype=np.float32)

# 保存为BIN文件
points.tofile('sequences/00/velodyne/000000.bin')

# 读取验证
loaded_points = np.fromfile('sequences/00/velodyne/000000.bin', dtype=np.float32)
loaded_points = loaded_points.reshape(-1, 4)  # 或 reshape(-1, 3) 如果只有xyz
print(f"Point cloud shape: {loaded_points.shape}")
```

**从其他格式转换**：
```python
# 从PCD格式转换
import open3d as o3d

pcd = o3d.io.read_point_cloud("pointcloud.pcd")
points = np.asarray(pcd.points)  # (N, 3)
colors = np.asarray(pcd.colors)  # (N, 3) 可选

# 如果有强度信息，可以添加
if hasattr(pcd, 'intensity'):
    intensity = np.asarray(pcd.intensity)
    points = np.column_stack([points, intensity])
else:
    # 如果没有强度，可以设置为0或使用颜色信息
    intensity = np.zeros((points.shape[0], 1))
    points = np.column_stack([points, intensity])

points = points.astype(np.float32)
points.tofile('sequences/00/velodyne/000000.bin')
```

### 2.4 标定文件格式（calib.txt）

**格式要求**：
标定文件必须包含相机内参矩阵 `K_cam2` 和从相机到LiDAR的变换矩阵 `T_cam2_velo`。

**KITTI格式示例**：
```
P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P3: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
Tr: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 0.000000000000e+00 0.000000000000e+00 9.999421000000e-01 -1.078540000000e-02 0.000000000000e+00 7.402527000000e-03 1.080804000000e-02 9.999455000000e-01 0.000000000000e+00 -2.721546000000e-02 -5.677670000000e-01 -7.630879000000e-01 1.000000000000e+00
```

**关键信息**：
- `P2`：相机2（通常是左相机）的投影矩阵，前3x3部分是内参矩阵 `K_cam2`
- `Tr`：从Velodyne（LiDAR）到相机0的变换矩阵，需要转换为相机2到Velodyne的变换

**标定参数说明**：

1. **相机内参矩阵 K**（3x3）：
   ```
   K = [[fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]]
   ```
   - `fx, fy`：焦距（像素单位）
   - `cx, cy`：主点坐标（像素单位）

2. **变换矩阵 T_cam2_velo**（4x4齐次变换矩阵）：
   ```
   T = [[R11, R12, R13, tx],
        [R21, R22, R23, ty],
        [R31, R32, R33, tz],
        [  0,   0,   0,  1]]
   ```
   - 前3x3：旋转矩阵R（从LiDAR坐标系到相机坐标系）
   - 前3个元素第4列：平移向量t（从LiDAR坐标系到相机坐标系）

**生成标定文件示例**：
```python
import numpy as np

# 相机内参
fx, fy = 718.856, 718.856  # 焦距
cx, cy = 607.1928, 185.2157  # 主点

# 构建内参矩阵
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])

# LiDAR到相机的变换矩阵（4x4）
# 这是一个示例，需要根据实际标定结果填写
T_velo_to_cam = np.array([
    [0.9999,  0.0098, -0.0074, -0.0272],
    [0.0000,  0.9999, -0.0108, -0.5678],
    [0.0074,  0.0108,  0.9999, -0.7631],
    [0.0000,  0.0000,  0.0000,  1.0000]
])

# 相机到LiDAR的变换矩阵（取逆）
T_cam_to_velo = np.linalg.inv(T_velo_to_cam)

# 生成KITTI格式的calib.txt
def write_calib_file(output_path, K, T_cam_to_velo):
    """
    写入KITTI格式的标定文件
    
    Args:
        output_path: 输出文件路径
        K: 相机内参矩阵 (3x3)
        T_cam_to_velo: 相机到LiDAR的变换矩阵 (4x4)
    """
    with open(output_path, 'w') as f:
        # P0, P1, P2, P3 是投影矩阵（4x3），这里我们只关心P2
        # P2的前3x3是内参，第4列是0
        P2 = np.zeros((3, 4))
        P2[:3, :3] = K
        
        # Tr是从Velodyne到相机0的变换，但我们需要相机2到Velodyne
        # 如果使用相机2，需要根据实际情况调整
        T_velo_to_cam0 = np.linalg.inv(T_cam_to_velo)
        
        # 写入P2（相机2的投影矩阵）
        f.write("P2: ")
        f.write(" ".join([f"{val:.12e}" for val in P2.flatten()]))
        f.write("\n")
        
        # 写入Tr（Velodyne到相机0的变换）
        f.write("Tr: ")
        f.write(" ".join([f"{val:.12e}" for val in T_velo_to_cam0.flatten()]))
        f.write("\n")

# 使用示例
write_calib_file('sequences/00/calib.txt', K, T_cam_to_velo)
```

## 3. 数据准备Pipeline

### 3.1 完整的数据准备脚本

以下是一个完整的数据准备脚本示例：

```python
#!/usr/bin/env python3
"""
自定义数据集准备脚本
将原始数据转换为BEVCalib所需的格式
"""

import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

def prepare_images(source_dir, target_dir, sequence_id, start_idx=0):
    """
    准备图像数据
    
    Args:
        source_dir: 源图像目录
        target_dir: 目标数据集根目录
        sequence_id: 序列ID（如 '00', '01'）
        start_idx: 起始索引
    """
    target_img_dir = os.path.join(target_dir, 'sequences', sequence_id, 'image_2')
    os.makedirs(target_img_dir, exist_ok=True)
    
    # 获取所有图像文件
    img_files = sorted([f for f in os.listdir(source_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for idx, img_file in enumerate(img_files):
        # 读取并转换为RGB
        img_path = os.path.join(source_dir, img_file)
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 保存为PNG格式，使用6位数字命名
        target_name = f"{start_idx + idx:06d}.png"
        target_path = os.path.join(target_img_dir, target_name)
        img.save(target_path)
        print(f"Saved: {target_path}")

def prepare_pointclouds(source_dir, target_dir, sequence_id, start_idx=0, 
                       has_intensity=True):
    """
    准备点云数据
    
    Args:
        source_dir: 源点云目录
        target_dir: 目标数据集根目录
        sequence_id: 序列ID
        start_idx: 起始索引
        has_intensity: 是否包含强度信息
    """
    target_pc_dir = os.path.join(target_dir, 'sequences', sequence_id, 'velodyne')
    os.makedirs(target_pc_dir, exist_ok=True)
    
    # 获取所有点云文件
    pc_files = sorted([f for f in os.listdir(source_dir) 
                      if f.lower().endswith(('.bin', '.pcd', '.ply', '.las'))])
    
    for idx, pc_file in enumerate(pc_files):
        pc_path = os.path.join(source_dir, pc_file)
        
        # 根据文件格式读取点云
        if pc_file.endswith('.bin'):
            # 已经是BIN格式
            points = np.fromfile(pc_path, dtype=np.float32)
            if has_intensity:
                points = points.reshape(-1, 4)
            else:
                points = points.reshape(-1, 3)
        elif pc_file.endswith('.pcd'):
            # PCD格式
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(pc_path)
            points = np.asarray(pcd.points)
            if has_intensity and len(pcd.colors) > 0:
                # 使用颜色作为强度（简化处理）
                intensity = np.mean(np.asarray(pcd.colors), axis=1, keepdims=True)
                points = np.column_stack([points, intensity])
            elif has_intensity:
                intensity = np.zeros((points.shape[0], 1))
                points = np.column_stack([points, intensity])
        else:
            # 其他格式，需要根据实际情况处理
            print(f"Warning: Unsupported format for {pc_file}")
            continue
        
        # 确保数据类型为float32
        points = points.astype(np.float32)
        
        # 保存为BIN格式
        target_name = f"{start_idx + idx:06d}.bin"
        target_path = os.path.join(target_pc_dir, target_name)
        points.tofile(target_path)
        print(f"Saved: {target_path}")

def write_calib_file(output_path, K, T_cam_to_velo):
    """
    写入KITTI格式的标定文件
    """
    with open(output_path, 'w') as f:
        # P2: 相机2的投影矩阵（3x4）
        P2 = np.zeros((3, 4))
        P2[:3, :3] = K
        
        f.write("P2: ")
        f.write(" ".join([f"{val:.12e}" for val in P2.flatten()]))
        f.write("\n")
        
        # Tr: Velodyne到相机0的变换矩阵（4x4）
        T_velo_to_cam0 = np.linalg.inv(T_cam_to_velo)
        
        f.write("Tr: ")
        f.write(" ".join([f"{val:.12e}" for val in T_velo_to_cam0.flatten()]))
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='准备自定义数据集')
    parser.add_argument('--source_images', type=str, required=True,
                       help='源图像目录')
    parser.add_argument('--source_pointclouds', type=str, required=True,
                       help='源点云目录')
    parser.add_argument('--target_dir', type=str, required=True,
                       help='目标数据集根目录')
    parser.add_argument('--sequence_id', type=str, default='00',
                       help='序列ID（如 00, 01）')
    parser.add_argument('--fx', type=float, required=True,
                       help='相机焦距fx（像素）')
    parser.add_argument('--fy', type=float, required=True,
                       help='相机焦距fy（像素）')
    parser.add_argument('--cx', type=float, required=True,
                       help='相机主点cx（像素）')
    parser.add_argument('--cy', type=float, required=True,
                       help='相机主点cy（像素）')
    parser.add_argument('--T_cam_to_velo', type=str, required=True,
                       help='相机到LiDAR的变换矩阵文件路径（4x4 numpy数组，.npy格式）')
    parser.add_argument('--has_intensity', action='store_true',
                       help='点云是否包含强度信息')
    
    args = parser.parse_args()
    
    # 创建目录结构
    os.makedirs(os.path.join(args.target_dir, 'sequences', args.sequence_id), 
                exist_ok=True)
    
    # 准备图像
    print("Preparing images...")
    prepare_images(args.source_images, args.target_dir, args.sequence_id)
    
    # 准备点云
    print("Preparing point clouds...")
    prepare_pointclouds(args.source_pointclouds, args.target_dir, 
                       args.sequence_id, has_intensity=args.has_intensity)
    
    # 准备标定文件
    print("Preparing calibration file...")
    K = np.array([
        [args.fx,  0, args.cx],
        [ 0, args.fy, args.cy],
        [ 0,  0,  1]
    ])
    T_cam_to_velo = np.load(args.T_cam_to_velo)
    
    calib_path = os.path.join(args.target_dir, 'sequences', 
                             args.sequence_id, 'calib.txt')
    write_calib_file(calib_path, K, T_cam_to_velo)
    print(f"Calibration file saved to: {calib_path}")
    
    print("Data preparation completed!")

if __name__ == '__main__':
    main()
```

### 3.2 使用示例

```bash
# 1. 准备变换矩阵文件（T_cam_to_velo.npy）
# 这是一个4x4的numpy数组，保存相机到LiDAR的变换矩阵
python -c "import numpy as np; np.save('T_cam_to_velo.npy', your_4x4_matrix)"

# 2. 运行数据准备脚本
python prepare_custom_dataset.py \
    --source_images /path/to/your/images \
    --source_pointclouds /path/to/your/pointclouds \
    --target_dir /path/to/output/dataset \
    --sequence_id 00 \
    --fx 718.856 \
    --fy 718.856 \
    --cx 607.1928 \
    --cy 185.2157 \
    --T_cam_to_velo T_cam_to_velo.npy \
    --has_intensity
```

## 4. 修改数据集类以支持自定义数据

### 4.1 创建自定义数据集类

如果您的数据格式与KITTI不完全一致，可以创建自定义数据集类：

```python
# custom_dataset.py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import json

class CustomDataset(Dataset):
    """
    自定义数据集类
    适用于非KITTI格式的数据
    """
    def __init__(self, data_folder='./data/custom', sequences=None):
        self.all_files = []
        self.dataset_root = data_folder
        self.K = {}  # 存储每个序列的相机内参
        self.T = {}  # 存储每个序列的变换矩阵（LiDAR到相机）
        
        # 如果没有指定序列，自动发现
        if sequences is None:
            sequences_dir = os.path.join(data_folder, 'sequences')
            if os.path.exists(sequences_dir):
                sequences = sorted([d for d in os.listdir(sequences_dir) 
                                  if os.path.isdir(os.path.join(sequences_dir, d))])
            else:
                sequences = []
        
        for seq in sequences:
            seq_dir = os.path.join(self.dataset_root, 'sequences', seq)
            
            # 加载标定信息
            calib_path = os.path.join(seq_dir, 'calib.txt')
            if os.path.exists(calib_path):
                K, T = self.load_calib(calib_path)
                self.K[seq] = K
                self.T[seq] = T
            else:
                # 或者从JSON文件加载
                calib_json = os.path.join(seq_dir, 'calib.json')
                if os.path.exists(calib_json):
                    K, T = self.load_calib_json(calib_json)
                    self.K[seq] = K
                    self.T[seq] = T
                else:
                    print(f"Warning: No calibration file found for sequence {seq}")
                    continue
            
            # 获取所有图像文件
            img_dir = os.path.join(seq_dir, 'image_2')
            if not os.path.exists(img_dir):
                continue
                
            image_list = sorted([f for f in os.listdir(img_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            for image_name in image_list:
                base_name = os.path.splitext(image_name)[0]
                pcd_path = os.path.join(seq_dir, 'velodyne', base_name + '.bin')
                
                # 检查对应的点云文件是否存在
                if os.path.exists(pcd_path):
                    self.all_files.append(os.path.join(seq, base_name))
    
    def load_calib(self, calib_path):
        """
        从KITTI格式的calib.txt加载标定信息
        """
        # 这里可以使用pykitti，或者自己解析
        from pykitti import odometry
        # 简化版本：直接解析文件
        K = None
        T_velo_to_cam = None
        
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('P2:'):
                    # 解析P2矩阵
                    values = [float(x) for x in line.split()[1:]]
                    P2 = np.array(values).reshape(3, 4)
                    K = P2[:3, :3]
                elif line.startswith('Tr:'):
                    # 解析Tr矩阵
                    values = [float(x) for x in line.split()[1:]]
                    T_velo_to_cam = np.array(values).reshape(4, 4)
        
        if K is None or T_velo_to_cam is None:
            raise ValueError(f"Failed to parse calibration file: {calib_path}")
        
        # 转换为LiDAR到相机的变换矩阵
        T_velo_to_cam0 = T_velo_to_cam
        return K, T_velo_to_cam0
    
    def load_calib_json(self, calib_json):
        """
        从JSON文件加载标定信息
        JSON格式示例:
        {
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "T_velo_to_cam": [[...], [...], [...], [...]]
        }
        """
        with open(calib_json, 'r') as f:
            calib_data = json.load(f)
        
        K = np.array(calib_data['K'], dtype=np.float32)
        T_velo_to_cam = np.array(calib_data['T_velo_to_cam'], dtype=np.float32)
        
        return K, T_velo_to_cam
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        seq = self.all_files[idx].split('/')[0]
        id = self.all_files[idx].split('/')[1]
        
        img_path = os.path.join(self.dataset_root, 'sequences', seq, 
                               'image_2', id + '.png')
        pcd_path = os.path.join(self.dataset_root, 'sequences', seq, 
                               'velodyne', id + '.bin')
        
        if not os.path.exists(img_path) or not os.path.exists(pcd_path):
            raise FileNotFoundError(f"File not found: {img_path} or {pcd_path}")
        
        # 加载图像
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 加载点云
        pcd = np.fromfile(pcd_path, dtype=np.float32)
        # 尝试不同的形状
        if len(pcd) % 4 == 0:
            pcd = pcd.reshape(-1, 4)
        elif len(pcd) % 3 == 0:
            pcd = pcd.reshape(-1, 3)
        else:
            raise ValueError(f"Invalid point cloud format: {pcd_path}")
        
        # 点云过滤（可选，与KITTI数据集保持一致）
        if pcd.shape[1] >= 3:
            valid_ind = pcd[:, 0] < -3.
            valid_ind = valid_ind | (pcd[:, 0] > 3.)
            valid_ind = valid_ind | (pcd[:, 1] < -3.)
            valid_ind = valid_ind | (pcd[:, 1] > 3.)
            pcd = pcd[valid_ind, :]
        
        gt_transform = self.T[seq]
        intrinsic = self.K[seq]
        
        return img, pcd, gt_transform, intrinsic

if __name__ == "__main__":
    # 测试数据集
    dataset = CustomDataset(data_folder='./data/custom')
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        img, pcd, gt_transform, intrinsic = dataset[0]
        print(f"Image size: {img.size}")
        print(f"Point cloud shape: {pcd.shape}")
        print(f"GT transform shape: {gt_transform.shape}")
        print(f"Intrinsic shape: {intrinsic.shape}")
```

### 4.2 在训练脚本中使用自定义数据集

修改 `train_kitti.py` 以支持自定义数据集：

```python
# 在train_kitti.py中
from custom_dataset import CustomDataset  # 导入自定义数据集类

# 替换原来的KittiDataset
# dataset = KittiDataset(dataset_root)
dataset = CustomDataset(data_folder=dataset_root)
```

或者创建一个通用的数据集加载函数：

```python
def load_dataset(dataset_root, dataset_type='kitti'):
    """
    通用的数据集加载函数
    
    Args:
        dataset_root: 数据集根目录
        dataset_type: 数据集类型 ('kitti' 或 'custom')
    """
    if dataset_type == 'kitti':
        from kitti_dataset import KittiDataset
        return KittiDataset(data_folder=dataset_root)
    elif dataset_type == 'custom':
        from custom_dataset import CustomDataset
        return CustomDataset(data_folder=dataset_root)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# 使用
dataset = load_dataset(dataset_root, dataset_type='custom')
```

## 5. 数据验证

### 5.1 验证脚本

创建验证脚本来检查数据集的完整性：

```python
#!/usr/bin/env python3
"""
数据集验证脚本
检查数据集的完整性和格式正确性
"""

import os
import numpy as np
from PIL import Image
import argparse

def validate_dataset(dataset_root):
    """
    验证数据集
    """
    sequences_dir = os.path.join(dataset_root, 'sequences')
    
    if not os.path.exists(sequences_dir):
        print(f"Error: Sequences directory not found: {sequences_dir}")
        return False
    
    sequences = sorted([d for d in os.listdir(sequences_dir) 
                      if os.path.isdir(os.path.join(sequences_dir, d))])
    
    if len(sequences) == 0:
        print("Error: No sequences found")
        return False
    
    print(f"Found {len(sequences)} sequences: {sequences}")
    
    all_valid = True
    
    for seq in sequences:
        print(f"\nValidating sequence {seq}...")
        seq_dir = os.path.join(sequences_dir, seq)
        
        # 检查标定文件
        calib_path = os.path.join(seq_dir, 'calib.txt')
        if not os.path.exists(calib_path):
            print(f"  Error: Calibration file not found: {calib_path}")
            all_valid = False
            continue
        
        # 检查图像目录
        img_dir = os.path.join(seq_dir, 'image_2')
        if not os.path.exists(img_dir):
            print(f"  Error: Image directory not found: {img_dir}")
            all_valid = False
            continue
        
        # 检查点云目录
        pc_dir = os.path.join(seq_dir, 'velodyne')
        if not os.path.exists(pc_dir):
            print(f"  Error: Point cloud directory not found: {pc_dir}")
            all_valid = False
            continue
        
        # 获取图像和点云文件列表
        img_files = sorted([f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.png', '.jpg'))])
        pc_files = sorted([f for f in os.listdir(pc_dir) 
                          if f.endswith('.bin')])
        
        print(f"  Found {len(img_files)} images and {len(pc_files)} point clouds")
        
        # 检查文件匹配
        img_basenames = {os.path.splitext(f)[0] for f in img_files}
        pc_basenames = {os.path.splitext(f)[0] for f in pc_files}
        
        matched = img_basenames & pc_basenames
        img_only = img_basenames - pc_basenames
        pc_only = pc_basenames - img_basenames
        
        print(f"  Matched pairs: {len(matched)}")
        if img_only:
            print(f"  Warning: {len(img_only)} images without matching point clouds")
        if pc_only:
            print(f"  Warning: {len(pc_only)} point clouds without matching images")
        
        # 验证几个样本
        sample_count = min(5, len(matched))
        print(f"  Validating {sample_count} samples...")
        
        for i, basename in enumerate(list(matched)[:sample_count]):
            img_path = os.path.join(img_dir, basename + '.png')
            pc_path = os.path.join(pc_dir, basename + '.bin')
            
            # 验证图像
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    print(f"    Warning: Image {basename} is not RGB mode")
            except Exception as e:
                print(f"    Error: Failed to load image {basename}: {e}")
                all_valid = False
            
            # 验证点云
            try:
                pc = np.fromfile(pc_path, dtype=np.float32)
                if len(pc) == 0:
                    print(f"    Error: Point cloud {basename} is empty")
                    all_valid = False
                elif len(pc) % 3 != 0 and len(pc) % 4 != 0:
                    print(f"    Error: Point cloud {basename} has invalid size: {len(pc)}")
                    all_valid = False
            except Exception as e:
                print(f"    Error: Failed to load point cloud {basename}: {e}")
                all_valid = False
    
    if all_valid:
        print("\n✓ Dataset validation passed!")
    else:
        print("\n✗ Dataset validation failed!")
    
    return all_valid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='验证数据集')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    args = parser.parse_args()
    
    validate_dataset(args.dataset_root)
```

### 5.2 运行验证

```bash
python validate_dataset.py --dataset_root /path/to/your/dataset
```

## 6. 常见问题与解决方案

### 6.1 标定参数获取

**问题**：如何获取相机内参和LiDAR-相机变换矩阵？

**解决方案**：
1. **使用标定工具**：
   - 使用OpenCV的相机标定工具获取相机内参
   - 使用Autoware、ROS等工具进行LiDAR-相机联合标定
   
2. **从现有标定文件提取**：
   - 如果已有其他格式的标定文件，可以转换为KITTI格式

3. **手动标定**：
   - 使用标定板进行手动标定（不推荐，精度较低）

### 6.2 坐标系转换

**问题**：如何确保坐标系正确？

**解决方案**：
- **KITTI坐标系约定**：
  - 相机坐标系：x向右，y向下，z向前
  - LiDAR坐标系：x向前，y向左，z向上
- 如果您的数据使用不同的坐标系，需要进行转换

### 6.3 点云格式转换

**问题**：如何从其他点云格式转换？

**解决方案**：
- 使用Open3D库读取各种格式（PCD, PLY, LAS等）
- 转换为numpy数组后保存为BIN格式

### 6.4 图像和点云同步

**问题**：如何确保图像和点云时间同步？

**解决方案**：
- 确保图像和点云文件名匹配
- 如果时间戳不同，需要先进行时间对齐
- 可以使用文件名中的时间戳进行匹配

## 7. 总结

制作自定义数据集的步骤：

1. **准备数据**：
   - 收集同步的图像和点云数据
   - 获取相机内参和LiDAR-相机变换矩阵

2. **数据转换**：
   - 将图像转换为PNG格式
   - 将点云转换为BIN格式（float32）
   - 创建标定文件（calib.txt）

3. **组织目录结构**：
   - 按照KITTI格式组织数据
   - 确保文件名匹配

4. **验证数据**：
   - 运行验证脚本检查数据完整性
   - 测试数据集加载

5. **训练/测试**：
   - 使用自定义数据集类或修改现有代码
   - 开始训练或推理

## 8. 参考资源

- [KITTI数据集格式说明](https://www.cvlibs.net/datasets/kitti/)
- [Open3D文档](http://www.open3d.org/docs/)
- [OpenCV相机标定教程](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
- [pykitti库文档](https://github.com/utiasSTARS/pykitti)

---

**注意**：本文档基于BEVCalib代码库的分析编写。如果遇到问题，请参考源代码或提交Issue。
