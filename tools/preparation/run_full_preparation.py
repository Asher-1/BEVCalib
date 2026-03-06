#!/usr/bin/env python3
"""
完整数据准备流程 - 主控制脚本

功能：
  1. 运行 batch_prepare_trips.py（或 prepare_custom_dataset.py）生成原始数据
  2. 自动运行 resize_images.py 生成训练用的 resize 图像
  
用法：
  # 方式1: 批量处理（推荐）
  python run_full_preparation.py --mode batch \
      --trips_file trips_list.txt \
      --output_dir /path/to/output \
      --resize_width 640 --resize_height 360
  
  # 方式2: 单个 trip 处理
  python run_full_preparation.py --mode single \
      --trip_dir /path/to/trip \
      --output_dir /path/to/output \
      --sequence_id 00 \
      --resize_width 640 --resize_height 360
  
  # 方式3: 只运行 resize（数据已生成）
  python run_full_preparation.py --mode resize-only \
      --dataset_root /path/to/output \
      --resize_width 640 --resize_height 360

作者: AI Assistant
日期: 2026-02-28
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class FullPreparationPipeline:
    """完整的数据准备流水线"""
    
    def __init__(self, args):
        self.args = args
        self.script_dir = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 脚本路径
        self.batch_prepare_script = self.script_dir / "batch_prepare_trips.py"
        self.prepare_single_script = self.script_dir / "prepare_custom_dataset.py"
        self.resize_script = self.script_dir / "resize_images.py"
        
        # 验证脚本存在
        self._validate_scripts()
    
    def _validate_scripts(self):
        """验证必要的脚本是否存在"""
        if self.args.mode in ['batch', 'single']:
            if self.args.mode == 'batch' and not self.batch_prepare_script.exists():
                print(f"❌ 错误: {self.batch_prepare_script} 不存在")
                sys.exit(1)
            if self.args.mode == 'single' and not self.prepare_single_script.exists():
                print(f"❌ 错误: {self.prepare_single_script} 不存在")
                sys.exit(1)
        
        if not self.resize_script.exists():
            print(f"❌ 错误: {self.resize_script} 不存在")
            sys.exit(1)
    
    def run(self):
        """运行完整流程"""
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 25 + "完整数据准备流程" + " " * 32 + "║")
        print("╚" + "=" * 78 + "╝")
        print(f"\n开始时间: {self.timestamp}")
        print(f"模式: {self.args.mode}")
        print(f"输出目录: {self.args.output_dir}")
        print(f"目标分辨率: {self.args.resize_width}×{self.args.resize_height}")
        if self.args.force_config:
            print(f"⚠️  强制使用lidars.cfg外参（忽略bag中的lidar外参）")
        print()
        
        # 步骤1: 数据准备
        if self.args.mode == 'batch':
            success = self._run_batch_prepare()
        elif self.args.mode == 'single':
            success = self._run_single_prepare()
        elif self.args.mode == 'resize-only':
            print("⏭️  跳过数据准备步骤（resize-only 模式）")
            success = True
        else:
            print(f"❌ 未知模式: {self.args.mode}")
            return 1
        
        if not success:
            print("\n❌ 数据准备步骤失败，停止流程")
            return 1
        
        # 步骤2: 图像 resize
        if self.args.skip_resize:
            print("\n⏭️  跳过图像 resize 步骤（--skip-resize）")
        else:
            success = self._run_resize()
            if not success:
                print("\n❌ 图像 resize 步骤失败")
                return 1
        
        # 完成
        print("\n" + "=" * 80)
        print("✅ 完整数据准备流程完成！")
        print("=" * 80)
        print(f"\n数据集位置: {self.args.output_dir}")
        print(f"原始图像: sequences/*/image_2/")
        print(f"Resize图像: sequences/*/image_2_{self.args.resize_width}x{self.args.resize_height}/")
        print(f"\n下一步: 使用以下配置训练模型")
        print(f"  --data_root {self.args.output_dir}")
        print(f"  --img_H {self.args.resize_height}")
        print(f"  --img_W {self.args.resize_width}")
        
        return 0
    
    def _run_batch_prepare(self):
        """运行批量数据准备"""
        print("=" * 80)
        print("步骤 1/2: 批量数据准备（生成原始图像和点云）")
        print("=" * 80)
        
        cmd = [
            sys.executable,
            str(self.batch_prepare_script),
            "--trips_file", self.args.trips_file,
            "--output_base_dir", self.args.output_dir,
        ]
        
        # 可选参数
        if self.args.camera_name:
            cmd.extend(["--camera_name", self.args.camera_name])
        if self.args.target_fps:
            cmd.extend(["--target_fps", str(self.args.target_fps)])
        if self.args.start_sequence_id:
            cmd.extend(["--start_sequence_id", self.args.start_sequence_id])
        if self.args.force_config:
            cmd.append("--force-config")
        
        print(f"\n执行命令:\n{' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            print("\n✅ 批量数据准备完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 批量数据准备失败: {e}")
            return False
    
    def _run_single_prepare(self):
        """运行单个 trip 数据准备"""
        print("=" * 80)
        print("步骤 1/2: 单个 trip 数据准备（生成原始图像和点云）")
        print("=" * 80)
        
        print("⚠️  注意: 单个 trip 模式需要手动调用 prepare_custom_dataset.py")
        print(f"    请运行: python {self.prepare_single_script} ...")
        print("\n    或者使用 --mode batch 进行批量处理")
        
        # 检查输出目录是否已存在数据
        output_dir = Path(self.args.output_dir)
        seq_dir = output_dir / "sequences"
        if seq_dir.exists() and any(seq_dir.iterdir()):
            print(f"\n✓ 检测到已存在的数据: {seq_dir}")
            print("  假设数据准备已完成，继续 resize 步骤")
            return True
        else:
            print(f"\n❌ 输出目录中没有找到数据: {seq_dir}")
            print("    请先运行数据准备脚本")
            return False
    
    def _run_resize(self):
        """运行图像 resize"""
        print("\n" + "=" * 80)
        print("步骤 2/2: 图像 Resize（生成训练用的 resize 图像）")
        print("=" * 80)
        
        # 检查数据是否存在
        dataset_root = Path(self.args.output_dir if self.args.mode == 'resize-only' else self.args.output_dir)
        seq_dir = dataset_root / "sequences"
        
        if not seq_dir.exists():
            print(f"❌ 未找到 sequences 目录: {seq_dir}")
            print("   请先运行数据准备步骤")
            return False
        
        # 统计需要 resize 的图像数量
        image_count = 0
        for seq in seq_dir.iterdir():
            if seq.is_dir():
                image_2_dir = seq / "image_2"
                if image_2_dir.exists():
                    image_count += len(list(image_2_dir.glob("*.png")))
        
        if image_count == 0:
            print(f"❌ 未找到需要 resize 的图像")
            return False
        
        print(f"\n找到 {image_count} 张图像需要 resize")
        print(f"目标尺寸: {self.args.resize_width}×{self.args.resize_height}")
        print(f"并行工作进程: {self.args.resize_workers}\n")
        
        cmd = [
            sys.executable,
            str(self.resize_script),
            "--dataset_root", str(dataset_root),
            "--width", str(self.args.resize_width),
            "--height", str(self.args.resize_height),
            "--workers", str(self.args.resize_workers),
            "--quality", str(self.args.resize_quality),
        ]
        
        print(f"执行命令:\n{' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            print("\n✅ 图像 resize 完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 图像 resize 失败: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="完整数据准备流程（数据提取 + 图像 resize）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

1. 批量处理（推荐）:
   python run_full_preparation.py --mode batch \\
       --trips_file trips_list.txt \\
       --output_dir /data/bevcalib_training_data \\
       --resize_width 640 --resize_height 360

2. 只运行 resize（数据已生成）:
   python run_full_preparation.py --mode resize-only \\
       --dataset_root /data/bevcalib_training_data \\
       --resize_width 640 --resize_height 360

3. 跳过 resize:
   python run_full_preparation.py --mode batch \\
       --trips_file trips_list.txt \\
       --output_dir /data/bevcalib_training_data \\
       --skip-resize

输出结构:
  output_dir/
    ├── sequences/
    │   ├── 00/
    │   │   ├── image_2/          ← 原始PNG (由prepare生成)
    │   │   ├── image_2_640x360/  ← resize后的JPEG (由resize生成)
    │   │   ├── velodyne/         ← 点云
    │   │   └── calib.txt
    │   ├── 01/
    │   └── ...
    └── image_2_640x360_meta.json
        """
    )
    
    # ========== 模式选择 ==========
    parser.add_argument("--mode", type=str, required=True, 
                        choices=['batch', 'single', 'resize-only'],
                        help="运行模式: batch(批量处理), single(单个trip), resize-only(仅resize)")
    
    # ========== 通用参数 ==========
    parser.add_argument("--output_dir", "--output_base_dir", "--dataset_root", 
                        type=str, dest='output_dir',
                        help="输出目录（数据集根目录）")
    
    # ========== 数据准备参数 (batch 模式) ==========
    parser.add_argument("--trips_file", type=str,
                        help="[batch模式] trips 列表文件")
    parser.add_argument("--camera_name", type=str, default="traffic_2",
                        help="[batch模式] 相机名称（默认: traffic_2）")
    parser.add_argument("--target_fps", type=float, default=10.0,
                        help="[batch模式] 目标帧率（默认: 10.0）")
    parser.add_argument("--start_sequence_id", type=str,
                        help="[batch模式] 起始序列ID（如 '00'）")
    
    # ========== 数据准备参数 (single 模式) ==========
    parser.add_argument("--trip_dir", type=str,
                        help="[single模式] 单个 trip 目录")
    parser.add_argument("--sequence_id", type=str,
                        help="[single模式] 序列ID（如 '00'）")
    
    # ========== Resize 参数 ==========
    parser.add_argument("--resize_width", "--width", type=int, default=640, dest='resize_width',
                        help="Resize 目标宽度（默认: 640）")
    parser.add_argument("--resize_height", "--height", type=int, default=360, dest='resize_height',
                        help="Resize 目标高度（默认: 360）")
    parser.add_argument("--resize_workers", type=int, default=32,
                        help="Resize 并行工作进程数（默认: 32）")
    parser.add_argument("--resize_quality", type=int, default=95,
                        help="JPEG 质量（0-100，默认: 95）")
    
    # ========== 外参控制参数 ==========
    parser.add_argument("--force-config", action="store_true", default=False,
                        help="强制使用lidars.cfg中的lidar外参替代bag中的外参。"
                             "默认行为：优先使用bag中的合格外参，不合格则fallback到lidars.cfg。")
    
    # ========== 控制参数 ==========
    parser.add_argument("--skip-resize", action="store_true",
                        help="跳过 resize 步骤，只运行数据准备")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="跳过数据准备步骤，只运行 resize")
    
    args = parser.parse_args()
    
    # ========== 参数验证 ==========
    if not args.output_dir:
        print("❌ 错误: 必须指定 --output_dir")
        parser.print_help()
        sys.exit(1)
    
    if args.mode == 'batch' and not args.trips_file:
        print("❌ 错误: batch 模式需要 --trips_file")
        parser.print_help()
        sys.exit(1)
    
    if args.mode == 'single' and (not args.trip_dir or not args.sequence_id):
        print("❌ 错误: single 模式需要 --trip_dir 和 --sequence_id")
        parser.print_help()
        sys.exit(1)
    
    if args.skip_resize and args.skip_prepare:
        print("❌ 错误: 不能同时跳过两个步骤")
        sys.exit(1)
    
    # 强制 resize-only 模式跳过准备步骤
    if args.mode == 'resize-only':
        args.skip_prepare = True
    
    # ========== 运行流程 ==========
    pipeline = FullPreparationPipeline(args)
    exit_code = pipeline.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
