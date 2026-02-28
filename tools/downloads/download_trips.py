#!/usr/bin/env python3
"""
Trip 数据批量下载脚本

功能：从 drfile trip 空间下载行程数据到本地目录
默认只下载每个 trip 中的 bags/important 和 configs 目录

用法：
    python download_trips.py <trip_name1> [trip_name2] [trip_name3] ...
    python download_trips.py --file trips.txt
    python download_trips.py --full <trip_name>  # 下载完整 trip
    python download_trips.py --interactive
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import logging


class TripDownloader:
    """Trip 数据下载器"""
    
    def __init__(self, output_dir: str = None, selective_download: bool = True):
        """
        初始化下载器
        
        Args:
            output_dir: 输出目录，默认为当前目录下的 trips
            selective_download: 是否只下载特定目录（bags/important 和 configs）
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trips')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selective_download = selective_download
        
        # 配置日志
        log_file = self.output_dir.parent / f'download_trips_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")
        
        if self.selective_download:
            self.logger.info("选择性下载模式：只下载 bags/important 和 configs 目录")
        
    def check_drfile_installed(self) -> bool:
        """检查 drfile 是否已安装"""
        try:
            result = subprocess.run(
                ['which', 'drfile'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False
            )
            if result.returncode == 0:
                self.logger.info(f"drfile 已安装: {result.stdout.strip()}")
                
                # 获取版本信息
                version_result = subprocess.run(
                    ['drfile', '--version'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=False
                )
                if version_result.returncode == 0:
                    self.logger.info(f"drfile 版本: {version_result.stdout.strip()}")
                return True
            else:
                self.logger.error("drfile 未安装，请先安装 drfile 工具")
                return False
        except Exception as e:
            self.logger.error(f"检查 drfile 安装状态失败: {e}")
            return False
    
    def check_trip_exists(self, trip_name: str) -> bool:
        """
        检查 trip 是否存在于 drfile 中
        
        Args:
            trip_name: trip 名称
            
        Returns:
            bool: trip 是否存在
        """
        try:
            trip_path = f"trip:/{trip_name}"
            result = subprocess.run(
                ['drfile', 'head', trip_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False
            )
            
            if result.returncode == 0:
                self.logger.info(f"✓ Trip 存在: {trip_name}")
                return True
            else:
                self.logger.warning(f"✗ Trip 不存在或无法访问: {trip_name}")
                self.logger.debug(f"错误信息: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"检查 trip 存在性失败: {e}")
            return False
    
    def download_subdirectory(self, trip_name: str, remote_subdir: str, local_subdir: str, local_base_path: Path, flatten: bool = False) -> bool:
        """
        下载 trip 中的单个子目录
        
        Args:
            trip_name: trip 名称
            remote_subdir: 远程子目录路径（相对于 trip 根目录）
            local_subdir: 本地子目录路径（相对于 trip 根目录）
            local_base_path: 本地基础路径
            flatten: 是否扁平化（将子目录的内容直接放到目标目录）
            
        Returns:
            bool: 下载是否成功
        """
        remote_path = f"trip:/{trip_name}/{remote_subdir}"
        
        self.logger.info(f"  下载: {remote_subdir} -> {local_subdir}")
        
        try:
            if flatten:
                # 扁平化模式：下载到临时目录，然后移动内容
                temp_dir = local_base_path / f"_temp_{local_subdir.replace('/', '_')}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                cmd = ['drfile', 'download', remote_path, str(temp_dir)]
                self.logger.debug(f"  执行命令（临时）: {' '.join(cmd)}")
                
                # 使用 Popen 以便实时显示输出
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时输出日志
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self.logger.info(f"    {line}")
                
                process.wait()
                
                if process.returncode == 0:
                    # 查找下载的实际目录（drfile 会创建与源同名的目录）
                    remote_dir_name = os.path.basename(remote_subdir)
                    downloaded_dir = temp_dir / remote_dir_name
                    
                    if downloaded_dir.exists() and downloaded_dir.is_dir():
                        # 移动内容到目标位置
                        target_dir = local_base_path / local_subdir
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        self.logger.info(f"  移动内容: {downloaded_dir} -> {target_dir}")
                        
                        # 移动所有内容
                        for item in downloaded_dir.iterdir():
                            dest = target_dir / item.name
                            if dest.exists():
                                if dest.is_dir():
                                    shutil.rmtree(dest)
                                else:
                                    dest.unlink()
                            shutil.move(str(item), str(dest))
                        
                        # 删除临时目录
                        shutil.rmtree(temp_dir)
                        
                        # 统计文件数量
                        file_count = sum(1 for _ in target_dir.rglob('*') if _.is_file())
                        self.logger.info(f"  ✓ {local_subdir}: {file_count} 个文件")
                        return True
                    else:
                        self.logger.error(f"  ✗ 未找到下载的目录: {downloaded_dir}")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return False
                else:
                    self.logger.warning(f"  ✗ {remote_subdir} 下载失败或不存在 (退出码: {process.returncode})")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return False
            else:
                # 普通模式：直接下载
                local_path = local_base_path / local_subdir
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                cmd = ['drfile', 'download', remote_path, str(local_path)]
                self.logger.debug(f"  执行命令: {' '.join(cmd)}")
                
                # 使用 Popen 以便实时显示输出
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时输出日志
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self.logger.info(f"    {line}")
                
                process.wait()
                
                if process.returncode == 0:
                    # 统计文件数量
                    if local_path.exists():
                        file_count = sum(1 for _ in local_path.rglob('*') if _.is_file())
                        self.logger.info(f"  ✓ {local_subdir}: {file_count} 个文件")
                    return True
                else:
                    self.logger.warning(f"  ✗ {remote_subdir} 下载失败或不存在 (退出码: {process.returncode})")
                    return False
                
        except Exception as e:
            self.logger.error(f"  ✗ 下载 {remote_subdir} 时发生异常: {e}")
            return False
    
    def download_trip(self, trip_name: str) -> bool:
        """
        下载单个 trip
        
        Args:
            trip_name: trip 名称
            
        Returns:
            bool: 下载是否成功
        """
        trip_name = trip_name.strip().strip('/')
        
        if not trip_name:
            self.logger.warning("Trip 名称为空，跳过")
            return False
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"开始下载 trip: {trip_name}")
        
        # 检查 trip 是否存在
        if not self.check_trip_exists(trip_name):
            return False
        
        # 构建本地路径
        local_path = self.output_dir / trip_name
        
        # 如果目录已存在
        if local_path.exists():
            self.logger.warning(f"目标目录已存在: {local_path}")
            self.logger.info(f"将继续下载（drfile 会自动处理重复文件）")
        
        # 创建目标目录
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.selective_download:
                # 选择性下载特定目录
                # 格式：(远程路径, 本地路径, 是否扁平化)
                # 注意：drfile download trip:/xxx/DIR PATH 会在 PATH 下创建 DIR/ 子目录
                # 所以要下载到父目录，让 drfile 自动创建子目录
                # 下载顺序：先下载小文件 configs，再下载大文件 bags/important
                self.logger.info(f"选择性下载模式：")
                download_map = [
                    ('configs', '.', False),              # 1. 先下载 configs 到根目录，drfile 创建 ./configs/
                    ('bags/important', 'bags', False),    # 2. 再下载 bags/important 到 bags/，drfile 创建 bags/important/
                ]
                
                success_count = 0
                for remote_subdir, local_subdir, flatten in download_map:
                    if self.download_subdirectory(trip_name, remote_subdir, local_subdir, local_path, flatten):
                        success_count += 1
                
                if success_count > 0:
                    self.logger.info(f"✓ 下载成功: {trip_name} ({success_count}/{len(download_map)} 个目录)")
                    
                    # 统计总文件数
                    file_count = sum(1 for _ in local_path.rglob('*') if _.is_file())
                    dir_count = sum(1 for _ in local_path.rglob('*') if _.is_dir())
                    self.logger.info(f"  总计 - 文件数: {file_count}, 目录数: {dir_count}")
                    
                    return True
                else:
                    self.logger.error(f"✗ 下载失败: {trip_name} (所有子目录都失败)")
                    return False
                    
            else:
                # 下载整个 trip
                trip_path = f"trip:/{trip_name}"
                self.logger.info(f"下载完整 trip: {trip_path} -> {local_path}")
                
                cmd = ['drfile', 'download', trip_path, str(local_path)]
                self.logger.debug(f"执行命令: {' '.join(cmd)}")
                
                # 使用 Popen 以便实时显示输出
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时输出日志
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self.logger.info(f"  {line}")
                
                process.wait()
                
                if process.returncode == 0:
                    self.logger.info(f"✓ 下载成功: {trip_name}")
                    
                    # 检查下载的文件数量
                    file_count = sum(1 for _ in local_path.rglob('*') if _.is_file())
                    dir_count = sum(1 for _ in local_path.rglob('*') if _.is_dir())
                    self.logger.info(f"  文件数: {file_count}, 目录数: {dir_count}")
                    
                    return True
                else:
                    self.logger.error(f"✗ 下载失败: {trip_name} (退出码: {process.returncode})")
                    return False
                
        except Exception as e:
            self.logger.error(f"✗ 下载 {trip_name} 时发生异常: {e}")
            return False
    
    def download_trips(self, trip_names: list) -> dict:
        """
        批量下载 trips
        
        Args:
            trip_names: trip 名称列表
            
        Returns:
            dict: 下载结果统计
        """
        if not trip_names:
            self.logger.warning("没有指定要下载的 trip")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        self.logger.info(f"开始批量下载，共 {len(trip_names)} 个 trip")
        self.logger.info(f"输出目录: {self.output_dir}")
        
        success_count = 0
        failed_count = 0
        failed_trips = []
        
        for idx, trip_name in enumerate(trip_names, 1):
            self.logger.info(f"\n进度: [{idx}/{len(trip_names)}]")
            
            if self.download_trip(trip_name):
                success_count += 1
            else:
                failed_count += 1
                failed_trips.append(trip_name)
        
        # 输出统计信息
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"下载完成！")
        self.logger.info(f"成功: {success_count}/{len(trip_names)}")
        self.logger.info(f"失败: {failed_count}/{len(trip_names)}")
        
        if failed_trips:
            self.logger.warning(f"\n失败的 trips:")
            for trip in failed_trips:
                self.logger.warning(f"  - {trip}")
        
        return {
            'success': success_count,
            'failed': failed_count,
            'total': len(trip_names),
            'failed_trips': failed_trips
        }


def read_trips_from_file(file_path: str) -> list:
    """
    从文件读取 trip 列表
    
    Args:
        file_path: 文件路径
        
    Returns:
        list: trip 名称列表
    """
    trips = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释行
                if line and not line.startswith('#'):
                    trips.append(line)
        return trips
    except FileNotFoundError:
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件失败: {e}")
        sys.exit(1)


def interactive_mode():
    """交互模式：让用户逐个输入 trip 名称"""
    print("交互模式：请输入 trip 名称（每行一个，输入空行结束）")
    trips = []
    while True:
        try:
            trip = input(f"Trip {len(trips)+1}: ").strip()
            if not trip:
                break
            trips.append(trip)
        except KeyboardInterrupt:
            print("\n已取消")
            sys.exit(0)
    return trips


def main():
    parser = argparse.ArgumentParser(
        description='批量下载 trip 行程数据（默认只下载 bags/important 和 configs 目录）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  # 下载单个 trip（只下载 bags/important 和 configs）
  %(prog)s YR-P789-19_20260213_012754
  
  # 下载多个 trips
  %(prog)s YR-P789-19_20260213_012754 YR-B26A1-1_20251117_031232
  
  # 下载完整的 trip（所有数据）
  %(prog)s --full YR-P789-19_20260213_012754
  
  # 从文件读取 trip 列表
  %(prog)s --file trips.txt
  
  # 交互模式
  %(prog)s --interactive
  
  # 指定输出目录
  %(prog)s --output /path/to/output YR-P789-19_20260213_012754
        '''
    )
    
    parser.add_argument(
        'trips',
        nargs='*',
        help='trip 名称列表'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='从文件读取 trip 列表（每行一个 trip 名称）'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='交互模式：逐个输入 trip 名称'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出目录（默认为脚本所在目录的 trips 子目录）'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='仅检查 trips 是否存在，不执行下载'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='下载完整的 trip（默认只下载 bags/important 和 configs）'
    )
    
    args = parser.parse_args()
    
    # 获取 trip 列表
    trips = []
    if args.file:
        trips = read_trips_from_file(args.file)
    elif args.interactive:
        trips = interactive_mode()
    elif args.trips:
        trips = args.trips
    else:
        parser.print_help()
        sys.exit(1)
    
    if not trips:
        print("错误: 没有指定要下载的 trip")
        sys.exit(1)
    
    # 创建下载器（默认选择性下载，除非指定 --full）
    downloader = TripDownloader(
        output_dir=args.output,
        selective_download=not args.full
    )
    
    # 检查 drfile 是否安装
    if not downloader.check_drfile_installed():
        sys.exit(1)
    
    # 仅检查模式
    if args.check_only:
        downloader.logger.info("检查模式：仅检查 trips 是否存在")
        for trip in trips:
            downloader.check_trip_exists(trip)
        sys.exit(0)
    
    # 执行下载
    result = downloader.download_trips(trips)
    
    # 返回适当的退出码
    sys.exit(0 if result['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
