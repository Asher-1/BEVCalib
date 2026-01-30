#!/usr/bin/env python3
"""
点云查看工具

支持查看 PLY 和 BIN 格式的点云文件。

使用方法:
    # 查看 PLY 格式点云
    python view_pointcloud.py temp/pointclouds/000000.ply
    
    # 查看 BIN 格式点云
    python view_pointcloud.py sequences/velodyne/000000.bin
    
    # 查看多个点云（可视化比较）
    python view_pointcloud.py temp/pointclouds/000000.ply temp/pointclouds/000001.ply
    
    # 混合格式也支持
    python view_pointcloud.py temp/pointclouds/000000.ply sequences/velodyne/000001.bin
    
    # 打印点云统计信息
    python view_pointcloud.py temp/pointclouds/000000.ply --info
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def load_ply(filepath: str) -> np.ndarray:
    """加载 PLY 文件
    
    Returns:
        (N, 4) 的点云数据 [x, y, z, intensity]
    """
    with open(filepath, 'r') as f:
        # 读取 header
        line = f.readline()
        if not line.startswith('ply'):
            raise ValueError(f"不是有效的 PLY 文件: {filepath}")
        
        # 跳过 header，找到 vertex 数量
        num_vertices = 0
        while True:
            line = f.readline()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('end_header'):
                break
        
        # 读取点云数据
        points = []
        for _ in range(num_vertices):
            line = f.readline().strip()
            if line:
                values = [float(v) for v in line.split()]
                points.append(values[:4])  # x, y, z, intensity
        
        return np.array(points, dtype=np.float32)


def load_bin(filepath: str) -> np.ndarray:
    """加载 BIN 文件（KITTI 格式）
    
    BIN 格式：每个点包含 4 个 float32 值 [x, y, z, intensity]
    
    Returns:
        (N, 4) 的点云数据 [x, y, z, intensity]
    """
    points = np.fromfile(filepath, dtype=np.float32)
    
    # 支持两种格式：
    # - KITTI 标准格式：每个点 4 个 float32 [x, y, z, intensity]
    # - 带timestamp格式：每个点 5 个 float32 [x, y, z, intensity, timestamp]
    if points.shape[0] % 5 == 0:
        # (N, 5) 格式，包含timestamp
        points = points.reshape(-1, 5)
        # 只取前4列用于可视化
        points = points[:, :4]
    elif points.shape[0] % 4 == 0:
        # (N, 4) 格式，标准KITTI格式
        points = points.reshape(-1, 4)
    else:
        raise ValueError(f"BIN 文件格式错误: {filepath} (点数据既不是4的倍数也不是5的倍数)")
    
    return points


def load_pointcloud(filepath: str) -> np.ndarray:
    """自动检测格式并加载点云
    
    支持格式:
        - .ply: PLY 格式
        - .bin: KITTI BIN 格式
    
    Returns:
        (N, 4) 的点云数据 [x, y, z, intensity]
    """
    filepath_lower = filepath.lower()
    
    if filepath_lower.endswith('.ply'):
        return load_ply(filepath)
    elif filepath_lower.endswith('.bin'):
        return load_bin(filepath)
    else:
        # 尝试根据内容判断
        try:
            return load_ply(filepath)
        except:
            try:
                return load_bin(filepath)
            except:
                raise ValueError(f"无法识别的点云格式: {filepath}\n支持的格式: .ply, .bin")


def print_pointcloud_info(points: np.ndarray, filepath: str):
    """打印点云统计信息"""
    file_format = "PLY" if filepath.lower().endswith('.ply') else "BIN" if filepath.lower().endswith('.bin') else "Unknown"
    file_size = Path(filepath).stat().st_size / 1024 / 1024  # MB
    
    print(f"\n{'='*60}")
    print(f"点云文件: {filepath}")
    print(f"格式: {file_format} | 大小: {file_size:.2f} MB")
    print(f"{'='*60}")
    print(f"点数: {points.shape[0]:,}")
    print(f"\nX 范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] 米")
    print(f"Y 范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] 米")
    print(f"Z 范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] 米")
    print(f"Intensity 范围: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")
    
    # 点密度信息
    x_span = points[:, 0].max() - points[:, 0].min()
    y_span = points[:, 1].max() - points[:, 1].min()
    z_span = points[:, 2].max() - points[:, 2].min()
    volume = x_span * y_span * z_span
    if volume > 0:
        density = points.shape[0] / volume
        print(f"\n空间范围: {x_span:.2f} × {y_span:.2f} × {z_span:.2f} 米³")
        print(f"点密度: {density:.2f} 点/米³")
    
    # 点分布
    print(f"\n点云中心: ({points[:, 0].mean():.2f}, {points[:, 1].mean():.2f}, {points[:, 2].mean():.2f})")
    print(f"{'='*60}\n")


def visualize_with_open3d(filepaths: list):
    """使用 Open3D 可视化点云"""
    try:
        import open3d as o3d
    except ImportError:
        print("错误: 需要安装 open3d")
        print("安装命令: pip install open3d")
        sys.exit(1)
    
    point_clouds = []
    colors_palette = [
        [1, 0, 0],  # 红色
        [0, 1, 0],  # 绿色
        [0, 0, 1],  # 蓝色
        [1, 1, 0],  # 黄色
        [1, 0, 1],  # 品红
        [0, 1, 1],  # 青色
    ]
    
    for idx, filepath in enumerate(filepaths):
        points = load_pointcloud(filepath)
        
        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # 如果有多个点云，使用不同颜色
        if len(filepaths) > 1:
            color = colors_palette[idx % len(colors_palette)]
            colors = np.tile(color, (points.shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # 单个点云，使用强度着色
            if points[:, 3].max() > 0:
                intensity_norm = points[:, 3] / points[:, 3].max()
            else:
                intensity_norm = np.zeros(points.shape[0])
            colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        point_clouds.append(pcd)
        print(f"✓ 加载: {filepath} ({points.shape[0]:,} 点)")
    
    print("\n启动 Open3D 可视化...")
    print("  - 鼠标左键：旋转")
    print("  - 鼠标滚轮：缩放")
    print("  - 鼠标右键：平移")
    print("  - 按 'H' 键：显示帮助")
    print("  - 按 'Q' 键：退出\n")
    
    o3d.visualization.draw_geometries(
        point_clouds,
        window_name="点云查看器",
        width=1280,
        height=720,
        left=50,
        top=50
    )


def visualize_with_matplotlib(filepaths: list):
    """使用 Matplotlib 可视化点云（备选方案）"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("错误: 需要安装 matplotlib")
        print("安装命令: pip install matplotlib")
        sys.exit(1)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors_palette = ['r', 'g', 'b', 'y', 'm', 'c']
    
    for idx, filepath in enumerate(filepaths):
        points = load_pointcloud(filepath)
        
        # 为了显示速度，限制显示点数
        max_display_points = 10000
        if points.shape[0] > max_display_points:
            indices = np.random.choice(points.shape[0], max_display_points, replace=False)
            display_points = points[indices]
        else:
            display_points = points
        
        color = colors_palette[idx % len(colors_palette)]
        label = Path(filepath).name
        
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            c=color,
            marker='.',
            s=1,
            label=label,
            alpha=0.6
        )
        
        print(f"✓ 加载: {filepath} ({points.shape[0]:,} 点)")
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_zlabel('Z (米)')
    ax.set_title('点云可视化')
    if len(filepaths) > 1:
        ax.legend()
    
    print("\n显示 Matplotlib 可视化...")
    print("  - 鼠标拖动：旋转视角")
    print("  - 关闭窗口：退出\n")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='点云查看工具 - 支持 PLY 和 BIN 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看 PLY 格式点云
  python view_pointcloud.py temp/pointclouds/000000.ply
  
  # 查看 BIN 格式点云
  python view_pointcloud.py sequences/velodyne/000000.bin
  
  # 查看多个点云（支持混合格式）
  python view_pointcloud.py temp/pointclouds/000000.ply sequences/velodyne/000001.bin
  
  # 只显示统计信息
  python view_pointcloud.py temp/pointclouds/000000.ply --info
  
  # 使用 Matplotlib（如果 Open3D 不可用）
  python view_pointcloud.py temp/pointclouds/000000.ply --backend matplotlib

支持的格式:
  - .ply: PLY 格式（ASCII）
  - .bin: KITTI BIN 格式（每个点 4 个 float32: x,y,z,intensity）
        """
    )
    parser.add_argument('files', nargs='+', help='点云文件路径（支持 .ply 和 .bin）')
    parser.add_argument('--info', action='store_true', help='只显示统计信息，不可视化')
    parser.add_argument('--backend', choices=['auto', 'open3d', 'matplotlib'], 
                       default='auto', help='可视化后端')
    
    args = parser.parse_args()
    
    # 验证文件存在
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"错误: 文件不存在: {filepath}")
            sys.exit(1)
    
    # 打印统计信息
    for filepath in args.files:
        try:
            points = load_pointcloud(filepath)
            print_pointcloud_info(points, filepath)
        except Exception as e:
            print(f"错误: 无法读取 {filepath}: {e}")
            sys.exit(1)
    
    # 可视化
    if not args.info:
        if args.backend == 'auto':
            try:
                import open3d as o3d
                visualize_with_open3d(args.files)
            except ImportError:
                print("Open3D 不可用，使用 Matplotlib...")
                visualize_with_matplotlib(args.files)
        elif args.backend == 'open3d':
            visualize_with_open3d(args.files)
        elif args.backend == 'matplotlib':
            visualize_with_matplotlib(args.files)


if __name__ == '__main__':
    main()
