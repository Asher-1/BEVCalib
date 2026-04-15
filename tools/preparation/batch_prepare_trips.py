#!/usr/bin/env python3
"""
批量处理 trips 目录下的所有行程数据
生成 KITTI-Odometry 格式的训练数据集

每个 trip 对应一个 sequence ID（00, 01, 02, ...）

支持:
  - 串行处理（默认，日志实时输出）
  - 并行处理（-j N，多个trip同时处理，大幅加速）

注意：prepare_custom_dataset.py 已经实现了只读取主 lidar (frame_id: "atx_202") 的功能，
     因此这里直接使用原始配置目录即可。
"""

import os
import sys
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_bag_directory(trip_dir):
    """查找 trip 目录下的 bag 文件目录"""
    trip_dir = Path(trip_dir)
    for bag_subdir in ['bags/unimportant', 'bags/important']:
        bag_dir = trip_dir / bag_subdir
        if bag_dir.exists():
            return bag_dir
    bag_dir = trip_dir / 'bags'
    if bag_dir.exists():
        return bag_dir
    raise ValueError(f"在 {trip_dir} 中未找到 bag 目录")


def _build_prepare_cmd(trip_dir, sequence_id, output_base_dir, camera_name, target_fps,
                       force_config, num_workers, batch_size):
    """构建 prepare_custom_dataset.py 命令"""
    trip_dir = Path(trip_dir)
    bag_dir = find_bag_directory(trip_dir)
    config_dir = trip_dir / 'configs'
    if not config_dir.exists():
        raise ValueError(f"配置目录不存在: {config_dir}")

    script_dir = Path(__file__).parent
    prepare_script = script_dir / 'prepare_custom_dataset.py'

    cmd = [
        sys.executable, str(prepare_script),
        '--bag_dir', str(bag_dir),
        '--config_dir', str(config_dir),
        '--output_dir', str(output_base_dir),
        '--camera_name', camera_name,
        '--target_fps', str(target_fps),
        '--sequence_id', sequence_id,
        '--num_workers', str(num_workers),
        '--batch_size', str(batch_size),
        # '--save_debug_samples', '20',
    ]
    if force_config:
        cmd.append('--force-config')
    return cmd, bag_dir, config_dir


def process_trip_serial(trip_dir, sequence_id, output_base_dir, log_file,
                        camera_name='traffic_2', target_fps=10.0, force_config=False,
                        num_workers=32, batch_size=800):
    """串行模式：处理单个 trip，实时输出日志（带 trip 前缀标识）"""
    trip_dir = Path(trip_dir)
    trip_name = trip_dir.name
    tag = f"[seq{sequence_id}|{trip_name}]"

    def _log(msg, end='\n'):
        line = f"{tag} {msg}{end}" if msg.strip() else f"{end}"
        print(line, end='')
        log_file.write(line)
        log_file.flush()

    _log("")
    _log(f"{'█'*80}")
    _log(f"  开始处理 Trip: {trip_name}")
    _log(f"  Sequence ID: {sequence_id}")
    _log(f"{'█'*80}")

    try:
        cmd, bag_dir, config_dir = _build_prepare_cmd(
            trip_dir, sequence_id, output_base_dir, camera_name, target_fps,
            force_config, num_workers, batch_size)

        _log(f"Bag 目录: {bag_dir}")
        _log(f"配置目录: {config_dir}")
        _log(f"执行命令: {' '.join(cmd)}")
        _log("")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1)

        for line in process.stdout:
            tagged = f"{tag} {line}" if line.strip() else line
            print(tagged, end='')
            log_file.write(tagged)
            log_file.flush()

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"处理失败，返回码: {process.returncode}")

        _log(f"{'█'*80}")
        _log(f"  ✓ Trip {trip_name} (seq{sequence_id}) 处理成功")
        _log(f"{'█'*80}")
        return True, trip_name, sequence_id, None

    except Exception as e:
        _log(f"  ✗ Trip {trip_name} 处理失败: {e}")
        return False, trip_name, sequence_id, str(e)


import re

_ALWAYS_PRINT_KEYWORDS = [
    '阶段 ', '去畸变并保存', '同步对数:', '有效图像:', '有效点云:',
    '✓ 全部完成', '✓ 同步完成', '重新编号', '📊 统计信息', '⚠️', '❌',
    '位姿:', '🗑️',
]

_TQDM_PERCENT_RE = re.compile(r'(\d+)%\|')


def _is_progress_line(line: str, last_pct: dict) -> bool:
    """判断是否是用户关心的进度行（减少 tqdm 刷屏）

    对于 tqdm 进度条（含 %| 的行），只在 0%, 25%, 50%, 75%, 100% 时输出。
    对于关键事件行（阶段切换、统计、错误等），始终输出。
    last_pct 是调用者维护的 {bar_key: last_printed_pct} 缓存。
    """
    stripped = line.strip()
    if not stripped:
        return False
    for kw in _ALWAYS_PRINT_KEYWORDS:
        if kw in stripped:
            return True

    m = _TQDM_PERCENT_RE.search(stripped)
    if m:
        pct = int(m.group(1))
        prefix = stripped[:m.start()].strip()
        bar_key = prefix if prefix else '_default_'
        prev = last_pct.get(bar_key, -1)
        milestone = (pct // 25) * 25
        if milestone > prev or pct == 100:
            last_pct[bar_key] = milestone
            return True
        return False

    return False


def _process_trip_worker(args):
    """并行模式的 worker 函数（在子进程中运行）

    关键进度行会实时打印到终端（带 trip 标识前缀），
    完整日志写入 per_trip_log_path。
    """
    trip_dir, sequence_id, output_base_dir, per_trip_log_path, \
        camera_name, target_fps, force_config, num_workers, batch_size = args

    trip_dir = Path(trip_dir)
    trip_name = trip_dir.name
    tag = f"[seq{sequence_id}|{trip_name}]"
    start = datetime.now()

    try:
        cmd, bag_dir, config_dir = _build_prepare_cmd(
            trip_dir, sequence_id, output_base_dir, camera_name, target_fps,
            force_config, num_workers, batch_size)

        with open(per_trip_log_path, 'w') as lf:
            lf.write(f"Trip: {trip_name}\n")
            lf.write(f"Sequence ID: {sequence_id}\n")
            lf.write(f"开始时间: {start}\n")
            lf.write(f"Bag 目录: {bag_dir}\n")
            lf.write(f"配置目录: {config_dir}\n")
            lf.write(f"命令: {' '.join(cmd)}\n")
            lf.write(f"{'='*80}\n\n")
            lf.flush()

            print(f"{tag} 🚀 开始处理 (workers={num_workers})", flush=True)

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1)
            tqdm_pct_cache = {}
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                if _is_progress_line(line, tqdm_pct_cache):
                    clean = line.rstrip('\n\r')
                    clean = clean.replace('\r', '').replace('\x1b[A', '')
                    if clean.strip():
                        print(f"{tag} {clean.strip()}", flush=True)
            proc.wait()

            elapsed = datetime.now() - start
            lf.write(f"\n{'='*80}\n")
            if proc.returncode == 0:
                lf.write(f"✓ Trip {trip_name} (seq{sequence_id}) 处理成功  耗时: {elapsed}\n")
                print(f"{tag} ✅ 处理成功  耗时: {elapsed}", flush=True)
            else:
                lf.write(f"✗ Trip {trip_name} (seq{sequence_id}) 处理失败  返回码: {proc.returncode}\n")
                print(f"{tag} ❌ 处理失败  返回码: {proc.returncode}", flush=True)

        if proc.returncode != 0:
            return False, trip_name, sequence_id, f"返回码: {proc.returncode}", str(elapsed)
        return True, trip_name, sequence_id, None, str(elapsed)

    except Exception as e:
        print(f"{tag} ❌ 异常: {e}", flush=True)
        return False, trip_name, sequence_id, str(e), str(datetime.now() - start)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量处理 trips 目录下的所有行程数据')
    parser.add_argument('--trips_dir', type=str, required=True,
                       help='trips 根目录（包含所有行程的目录）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录（生成 bevcalib_training_data）')
    parser.add_argument('--camera_name', type=str, default='traffic_2',
                       help='相机名称（默认: traffic_2）')
    parser.add_argument('--target_fps', type=float, default=10.0,
                       help='目标帧率（默认: 10.0）')
    parser.add_argument('--start_sequence', type=int, default=0,
                       help='起始 sequence ID（默认: 0）')
    parser.add_argument('--force-config', action='store_true', default=False,
                       help='强制使用lidars.cfg中的lidar外参替代bag中的外参')
    parser.add_argument('-j', '--parallel-trips', type=int, default=1,
                       help='并行处理的trip数量（默认: 1=串行）。'
                            '建议值: 2~4，每个trip内部也有多线程，注意CPU总负载')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='每个trip内部的工作线程数（默认: 串行32, 并行自动计算）')
    parser.add_argument('--batch_size', type=int, default=800,
                       help='点云处理批次大小（默认: 800）')

    args = parser.parse_args()

    trips_dir = Path(args.trips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 trip 目录
    trip_dirs = sorted([d for d in trips_dir.iterdir() if d.is_dir()])
    if not trip_dirs:
        print(f"❌ 在 {trips_dir} 中未找到任何 trip 目录")
        sys.exit(1)

    n_trips = len(trip_dirs)
    parallel = min(args.parallel_trips, n_trips)
    is_parallel = parallel > 1

    # 计算每个trip的内部线程数
    total_cpus = multiprocessing.cpu_count()
    if args.num_workers is not None:
        workers_per_trip = args.num_workers
    elif is_parallel:
        workers_per_trip = max(4, total_cpus // parallel)
    else:
        workers_per_trip = 32

    log_file_path = output_dir / f'batch_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # 打印配置
    print(f"\n{'='*80}")
    print(f"批量处理 Trips 数据集")
    print(f"{'='*80}")
    print(f"\nTrips 目录: {trips_dir}")
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file_path}")
    print(f"相机名称: {args.camera_name}")
    print(f"目标帧率: {args.target_fps} fps")
    print(f"起始 Sequence ID: {args.start_sequence}")
    print(f"处理模式: {'并行 (jobs=' + str(parallel) + ')' if is_parallel else '串行'}")
    print(f"每trip线程数: {workers_per_trip}")
    if args.force_config:
        print(f"⚠️  强制使用lidars.cfg外参（忽略bag中的lidar外参）")

    print(f"\n找到 {n_trips} 个 trip 目录:")
    trip_plan = []
    for i, td in enumerate(trip_dirs):
        seq_id = f"{args.start_sequence + i:02d}"
        print(f"  {seq_id}: {td.name}")
        trip_plan.append((td, seq_id))

    # 打开日志
    with open(log_file_path, 'w') as log_file:
        start_time = datetime.now()
        header = (
            f"批量处理开始时间: {start_time}\n"
            f"Trips 目录: {trips_dir}\n"
            f"输出目录: {output_dir}\n"
            f"相机名称: {args.camera_name}\n"
            f"目标帧率: {args.target_fps} fps\n"
            f"起始 Sequence ID: {args.start_sequence}\n"
            f"处理模式: {'并行 (jobs=' + str(parallel) + ')' if is_parallel else '串行'}\n"
            f"每trip线程数: {workers_per_trip}\n"
        )
        if args.force_config:
            header += f"⚠️  强制使用lidars.cfg外参\n"
        header += f"\n找到 {n_trips} 个 trip 目录\n"
        log_file.write(header)
        log_file.flush()

        results = []  # (success, trip_name, seq_id, error, elapsed)

        if is_parallel:
            # ===== 并行模式 =====
            print(f"\n🚀 并行模式: {parallel} 个trip同时处理，每trip {workers_per_trip} 线程")
            print(f"   每个trip的日志单独存储，处理完成后合并到主日志\n")
            log_file.write(f"\n🚀 并行模式: {parallel} 个trip同时处理\n\n")
            log_file.flush()

            per_trip_logs = {}
            worker_args = []
            for td, seq_id in trip_plan:
                per_log = output_dir / f'_trip_log_seq{seq_id}_{td.name}.log'
                per_trip_logs[seq_id] = per_log
                worker_args.append((
                    str(td), seq_id, str(output_dir), str(per_log),
                    args.camera_name, args.target_fps, args.force_config,
                    workers_per_trip, args.batch_size
                ))

            with ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(_process_trip_worker, wa): wa[1]  # seq_id
                           for wa in worker_args}
                for future in as_completed(futures):
                    seq_id = futures[future]
                    result = future.result()
                    success, trip_name, sid, error, elapsed = result
                    results.append(result)
                    status = "✓ 成功" if success else f"✗ 失败: {error}"
                    msg = f"  seq {sid} ({trip_name}): {status}  [{elapsed}]"
                    print(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

            # 合并各trip日志到主日志（按seq_id排序）
            log_file.write(f"\n\n{'='*80}\n")
            log_file.write(f"各 Trip 详细日志\n")
            log_file.write(f"{'='*80}\n\n")
            for td, seq_id in trip_plan:
                per_log = per_trip_logs[seq_id]
                log_file.write(f"\n{'█'*80}\n")
                log_file.write(f"  seq {seq_id} | {td.name}\n")
                log_file.write(f"{'█'*80}\n\n")
                if per_log.exists():
                    log_file.write(per_log.read_text())
                    per_log.unlink()  # 清理临时日志
                else:
                    log_file.write(f"  (日志文件未生成)\n")
                log_file.write("\n")
            log_file.flush()

        else:
            # ===== 串行模式 =====
            for td, seq_id in trip_plan:
                success, trip_name, sid, error = process_trip_serial(
                    trip_dir=td,
                    sequence_id=seq_id,
                    output_base_dir=output_dir,
                    log_file=log_file,
                    camera_name=args.camera_name,
                    target_fps=args.target_fps,
                    force_config=args.force_config,
                    num_workers=workers_per_trip,
                    batch_size=args.batch_size,
                )
                elapsed = ""  # serial mode doesn't track per-trip time separately
                results.append((success, trip_name, sid, error, elapsed))
                if not success:
                    print(f"\n跳过失败的 trip，继续处理下一个...\n")

        # ===== 输出总结 =====
        end_time = datetime.now()
        duration = end_time - start_time
        success_count = sum(1 for r in results if r[0])
        failed = [(r[1], r[3]) for r in results if not r[0]]

        summary = f"\n{'='*80}\n"
        summary += f"批量处理完成\n"
        summary += f"{'='*80}\n"
        summary += f"开始时间: {start_time}\n"
        summary += f"结束时间: {end_time}\n"
        summary += f"总用时: {duration}\n"
        summary += f"处理模式: {'并行 (' + str(parallel) + ' jobs)' if is_parallel else '串行'}\n"
        summary += f"\n处理结果:\n"
        summary += f"  总计: {n_trips} 个 trips\n"
        summary += f"  成功: {success_count} 个\n"
        summary += f"  失败: {len(failed)} 个\n"

        if failed:
            summary += f"\n失败的 trips:\n"
            for trip_name, error in failed:
                summary += f"  - {trip_name}: {error}\n"

        failed_names = {t[0] for t in failed}
        summary += f"\nTrip 名称与 Sequence ID 对应关系:\n"
        summary += f"{'─'*72}\n"
        summary += f"  {'Seq':<6}{'Trip Name':<40}{'Samples':>8}  {'Status':<8}\n"
        summary += f"  {'─'*4}  {'─'*38}  {'─'*6}  {'─'*6}\n"
        total_samples = 0
        for td, seq_id in trip_plan:
            status = "FAILED" if td.name in failed_names else "OK"
            seq_img_dir = output_dir / 'sequences' / seq_id / 'image_2'
            if seq_img_dir.exists():
                sample_count = len(list(seq_img_dir.glob('*.png')))
            else:
                sample_count = 0
            total_samples += sample_count
            summary += f"  {seq_id:<6}{td.name:<40}{sample_count:>8}  {status:<8}\n"
        summary += f"  {'─'*4}  {'─'*38}  {'─'*6}  {'─'*6}\n"
        summary += f"  {'':6}{'Total':<40}{total_samples:>8}\n"
        summary += f"{'─'*72}\n"
        summary += f"\n输出目录: {output_dir}\n"
        summary += f"日志文件: {log_file_path}\n"

        print(summary)
        log_file.write(summary)

        # 下一步提示
        script_dir = Path(__file__).parent
        next_steps = f"\n{'='*80}\n"
        next_steps += f"📋 下一步操作\n"
        next_steps += f"{'='*80}\n\n"
        next_steps += f"原始数据准备完成！现在需要 resize 图像用于训练。\n\n"
        next_steps += f"方式1: 使用快捷脚本（推荐）\n"
        next_steps += f"  cd {script_dir}\n"
        next_steps += f"  ./run_resize_only.sh {output_dir} 640 360\n\n"
        next_steps += f"方式2: 使用完整流水线脚本\n"
        next_steps += f"  cd {script_dir}\n"
        next_steps += f"  ./run_preparation_pipeline.sh full {trips_dir} {output_dir} 640 360\n\n"
        next_steps += f"方式3: 直接调用 Python 脚本\n"
        next_steps += f"  python {script_dir}/resize_images.py \\\n"
        next_steps += f"    --dataset_root {output_dir} \\\n"
        next_steps += f"    --width 640 --height 360 --workers 32\n\n"
        next_steps += f"{'='*80}\n"

        print(next_steps)
        log_file.write(next_steps)

    print(f"\n完整日志已保存到: {log_file_path}")


if __name__ == '__main__':
    main()
