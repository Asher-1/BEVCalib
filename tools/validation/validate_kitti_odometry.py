#!/usr/bin/env python3
"""
验证数据集是否符合 KITTI-Odometry 标准格式
"""

import numpy as np
from pathlib import Path
import argparse

from validation_utils import list_sequence_images


class KITTIOdometryValidator:
    """KITTI-Odometry 格式验证器"""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.errors = []
        self.warnings = []
        self.passed = []
    
    def validate(self, sequence_id: str = "00"):
        """验证数据集"""
        print(f"🔍 验证 KITTI-Odometry 数据集: {self.dataset_root}")
        print(f"   序列: {sequence_id}\n")
        
        # 1. 验证目录结构
        self._validate_directory_structure(sequence_id)
        
        # 2. 验证 calib.txt
        self._validate_calib(sequence_id)
        
        # 3. 验证 poses 文件
        self._validate_poses(sequence_id)
        
        # 4. 验证图像文件
        self._validate_images(sequence_id)
        
        # 5. 验证点云文件
        self._validate_velodyne(sequence_id)
        
        # 6. 验证数据对齐
        self._validate_alignment(sequence_id)
        
        # 输出报告
        self._print_report()
    
    def _validate_directory_structure(self, sequence_id: str):
        """验证目录结构"""
        seq_dir = self.dataset_root / 'sequences' / sequence_id
        
        if not seq_dir.exists():
            self.errors.append(f"❌ 序列目录不存在: {seq_dir}")
            return
        self.passed.append(f"✓ 序列目录存在: sequences/{sequence_id}/")
        
        # 检查必需的子目录
        required_dirs = ['image_2', 'velodyne']
        for dir_name in required_dirs:
            dir_path = seq_dir / dir_name
            if not dir_path.exists():
                self.errors.append(f"❌ 缺少目录: sequences/{sequence_id}/{dir_name}/")
            else:
                self.passed.append(f"✓ 目录存在: sequences/{sequence_id}/{dir_name}/")
        
        # 检查 calib.txt
        calib_file = seq_dir / 'calib.txt'
        if not calib_file.exists():
            self.errors.append(f"❌ 缺少文件: sequences/{sequence_id}/calib.txt")
        else:
            self.passed.append(f"✓ 文件存在: sequences/{sequence_id}/calib.txt")
    
    def _validate_calib(self, sequence_id: str):
        """验证 calib.txt 格式"""
        calib_file = self.dataset_root / 'sequences' / sequence_id / 'calib.txt'
        
        if not calib_file.exists():
            return
        
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        # 解析 calib.txt
        calib_data = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            
            key = parts[0].rstrip(':')
            
            # 跳过非数值行（如 camera_model: pinhole）
            try:
                values = [float(v) for v in parts[1:]]
                calib_data[key] = values
            except ValueError:
                # 跳过无法转换为浮点数的行
                continue
        
        # 验证 P0-P3
        for i in range(4):
            key = f'P{i}'
            if key in calib_data:
                if len(calib_data[key]) == 12:
                    self.passed.append(f"✓ {key}: 3×4 投影矩阵 (12个数) ✓")
                else:
                    self.warnings.append(f"⚠️  {key}: 数值个数 {len(calib_data[key])} (应该是12)")
            else:
                self.warnings.append(f"⚠️  缺少 {key} (非强制，但推荐)")
        
        # 验证 Tr
        if 'Tr' in calib_data:
            num_values = len(calib_data['Tr'])
            if num_values == 12:
                self.passed.append(f"✓ Tr: 3×4 变换矩阵 (12个数) ✓")
                
                # 验证矩阵是否合理（旋转矩阵的行列式应该接近1）
                Tr = np.array(calib_data['Tr']).reshape(3, 4)
                R = Tr[:3, :3]
                det = np.linalg.det(R)
                if 0.99 < det < 1.01:
                    self.passed.append(f"✓ Tr旋转矩阵行列式: {det:.6f} (接近1) ✓")
                else:
                    self.warnings.append(f"⚠️  Tr旋转矩阵行列式: {det:.6f} (应该接近1)")
            elif num_values == 16:
                self.errors.append(f"❌ Tr: 4×4 矩阵 (16个数)，应该是 3×4 (12个数)")
                self.warnings.append(f"   修复建议: 只保存Tr矩阵的前3行")
            else:
                self.errors.append(f"❌ Tr: 数值个数 {num_values} (应该是12)")
        else:
            self.errors.append(f"❌ 缺少 Tr 变换矩阵")
    
    def _validate_poses(self, sequence_id: str):
        """验证 poses 文件"""
        poses_file = self.dataset_root / 'poses' / f'{sequence_id}.txt'
        
        if not poses_file.exists():
            self.warnings.append(f"⚠️  缺少 poses/{sequence_id}.txt (非强制)")
            return
        
        self.passed.append(f"✓ 文件存在: poses/{sequence_id}.txt")
        
        with open(poses_file, 'r') as f:
            lines = f.readlines()
        
        # 验证每一行
        invalid_lines = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 12:
                invalid_lines.append((i, len(parts)))
        
        if not invalid_lines:
            self.passed.append(f"✓ poses文件格式正确: {len(lines)}行，每行12个数 ✓")
        else:
            for line_num, num_values in invalid_lines[:5]:  # 只显示前5个错误
                self.errors.append(f"❌ poses文件第{line_num+1}行: {num_values}个数 (应该是12)")
            if len(invalid_lines) > 5:
                self.errors.append(f"   ... 还有 {len(invalid_lines)-5} 行格式错误")
    
    def _validate_images(self, sequence_id: str):
        """验证图像文件"""
        image_dir = self.dataset_root / 'sequences' / sequence_id / 'image_2'
        
        if not image_dir.exists():
            return
        
        images = list_sequence_images(image_dir)
        
        if not images:
            self.errors.append(f"❌ image_2/ 目录为空")
            return
        
        self.passed.append(f"✓ 图像数量: {len(images)} 张")
        
        # 验证命名格式 (000000.jpg/png, 000001.jpg/png, ...)
        ext = images[0].suffix
        expected_names = [f"{i:06d}{ext}" for i in range(len(images))]
        actual_names = [img.name for img in images]
        
        if actual_names == expected_names:
            self.passed.append(f"✓ 图像命名格式正确 (6位补零) ✓")
        else:
            mismatches = [i for i, (e, a) in enumerate(zip(expected_names, actual_names)) if e != a]
            if mismatches:
                self.warnings.append(f"⚠️  图像命名不连续，从索引 {mismatches[0]} 开始")
    
    def _validate_velodyne(self, sequence_id: str):
        """验证点云文件"""
        velodyne_dir = self.dataset_root / 'sequences' / sequence_id / 'velodyne'
        
        if not velodyne_dir.exists():
            return
        
        clouds = sorted(velodyne_dir.glob('*.bin'))
        
        if not clouds:
            self.errors.append(f"❌ velodyne/ 目录为空")
            return
        
        self.passed.append(f"✓ 点云数量: {len(clouds)} 帧")
        
        # 验证命名格式
        expected_names = [f"{i:06d}.bin" for i in range(len(clouds))]
        actual_names = [pc.name for pc in clouds]
        
        if actual_names == expected_names:
            self.passed.append(f"✓ 点云命名格式正确 (6位补零) ✓")
        else:
            mismatches = [i for i, (e, a) in enumerate(zip(expected_names, actual_names)) if e != a]
            if mismatches:
                self.warnings.append(f"⚠️  点云命名不连续，从索引 {mismatches[0]} 开始")
        
        # 验证点云格式（检查第一个文件）
        if clouds:
            first_cloud = clouds[0]
            data = np.fromfile(str(first_cloud), dtype=np.float32)
            
            if len(data) % 4 == 0:
                num_points = len(data) // 4
                self.passed.append(f"✓ 点云格式: Float32, 每点4个值 (N={num_points}) ✓")
                
                # 检查坐标范围是否合理
                points = data.reshape(-1, 4)
                x_range = (points[:, 0].min(), points[:, 0].max())
                y_range = (points[:, 1].min(), points[:, 1].max())
                z_range = (points[:, 2].min(), points[:, 2].max())
                
                self.passed.append(f"✓ 坐标范围:")
                self.passed.append(f"   X: [{x_range[0]:.2f}, {x_range[1]:.2f}] m")
                self.passed.append(f"   Y: [{y_range[0]:.2f}, {y_range[1]:.2f}] m")
                self.passed.append(f"   Z: [{z_range[0]:.2f}, {z_range[1]:.2f}] m")
                
                # 检查是否有异常值
                if abs(x_range[0]) > 1000 or abs(x_range[1]) > 1000:
                    self.warnings.append(f"⚠️  X坐标范围异常 (>1000m)")
                if abs(y_range[0]) > 1000 or abs(y_range[1]) > 1000:
                    self.warnings.append(f"⚠️  Y坐标范围异常 (>1000m)")
                if abs(z_range[0]) > 100 or abs(z_range[1]) > 100:
                    self.warnings.append(f"⚠️  Z坐标范围异常 (>100m)")
            else:
                self.errors.append(f"❌ 点云格式错误: 数据长度 {len(data)} 不是4的倍数")
    
    def _validate_alignment(self, sequence_id: str):
        """验证数据对齐（图像、点云、位姿数量是否一致）"""
        image_dir = self.dataset_root / 'sequences' / sequence_id / 'image_2'
        velodyne_dir = self.dataset_root / 'sequences' / sequence_id / 'velodyne'
        poses_file = self.dataset_root / 'poses' / f'{sequence_id}.txt'
        
        counts = {}
        
        if image_dir.exists():
            counts['images'] = len(list_sequence_images(image_dir))
        
        if velodyne_dir.exists():
            counts['velodyne'] = len(list(velodyne_dir.glob('*.bin')))
        
        if poses_file.exists():
            with open(poses_file, 'r') as f:
                counts['poses'] = len(f.readlines())
        
        if len(set(counts.values())) == 1:
            self.passed.append(f"✓ 数据对齐: 图像、点云、位姿数量一致 ({counts.get('images', 0)}) ✓")
        else:
            self.warnings.append(f"⚠️  数据数量不一致:")
            for key, count in counts.items():
                self.warnings.append(f"   {key}: {count}")
    
    def _print_report(self):
        """输出验证报告"""
        print("\n" + "="*80)
        print("📊 验证报告")
        print("="*80)
        
        if self.passed:
            print("\n✅ 通过的检查项:")
            for item in self.passed:
                print(f"   {item}")
        
        if self.warnings:
            print("\n⚠️  警告:")
            for item in self.warnings:
                print(f"   {item}")
        
        if self.errors:
            print("\n❌ 错误:")
            for item in self.errors:
                print(f"   {item}")
        
        print("\n" + "="*80)
        print(f"总结: {len(self.passed)} 项通过, {len(self.warnings)} 项警告, {len(self.errors)} 项错误")
        print("="*80)
        
        if not self.errors:
            print("\n🎉 数据集格式验证通过！可以用于训练。")
            return 0
        else:
            print("\n⚠️  发现错误，建议修复后再进行训练。")
            return 1


def main():
    parser = argparse.ArgumentParser(description='验证 KITTI-Odometry 数据集格式')
    parser.add_argument('dataset_root', type=str, help='数据集根目录')
    parser.add_argument('--sequence', type=str, default='00', help='序列ID (默认: 00)')
    
    args = parser.parse_args()
    
    validator = KITTIOdometryValidator(args.dataset_root)
    exit_code = validator.validate(args.sequence)
    
    return exit_code


if __name__ == '__main__':
    exit(main())
