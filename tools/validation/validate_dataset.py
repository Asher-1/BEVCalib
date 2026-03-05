#!/usr/bin/env python3
"""
BEVCalib 数据集统一验证工具

整合所有验证功能的统一入口：
  summary          数据集摘要
  format           KITTI格式验证
  tr               Tr矩阵验证
  projection       单帧投影测试
  projection-full  多序列投影验证
  quick            快速验证（~17秒，前3序列，各1帧）
  full             完整验证（~15分钟，所有序列+完整投影）

用法：
    python tools/validate_dataset.py summary /path/to/dataset
    python tools/validate_dataset.py format /path/to/dataset --sequence 00
    python tools/validate_dataset.py projection /path/to/dataset --sequence 00 --frame 0
    python tools/validate_dataset.py quick /path/to/dataset --output-dir results/
    python tools/validate_dataset.py full /path/to/dataset --output-dir results/
"""

import sys
import argparse
from pathlib import Path

# 添加tools目录到路径
TOOLS_DIR = Path(__file__).parent.parent  # tools/ 根目录
VALIDATION_DIR = Path(__file__).parent     # tools/validation/
sys.path.insert(0, str(VALIDATION_DIR))


def run_summary(args):
    """运行快速摘要"""
    from show_dataset_summary import show_dataset_summary
    show_dataset_summary(args.dataset_root)


def run_format_validation(args):
    """运行格式验证"""
    from validate_kitti_odometry import KITTIOdometryValidator
    
    validator = KITTIOdometryValidator(args.dataset_root)
    
    if args.all_sequences:
        # 验证所有序列
        sequences_dir = Path(args.dataset_root) / 'sequences'
        sequences = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
        
        all_passed = True
        for seq in sequences:
            print(f"\n{'='*80}")
            print(f"验证序列 {seq}")
            print('='*80)
            exit_code = validator.validate(seq)
            if exit_code != 0:
                all_passed = False
        
        return 0 if all_passed else 1
    else:
        # 验证单个序列
        return validator.validate(args.sequence)


def run_tr_validation(args):
    """运行Tr矩阵验证"""
    import subprocess
    
    cmd = [
        sys.executable,
        str(VALIDATION_DIR / 'verify_dataset_tr_fix.py'),
        '--dataset_root', args.dataset_root
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_projection_test(args):
    """运行投影测试"""
    import subprocess
    
    if args.output is None:
        args.output = f"projection_seq{args.sequence}_frame{args.frame:06d}.png"
    
    cmd = [
        sys.executable,
        str(VALIDATION_DIR / 'check_projection_headless.py'),
        '--dataset_root', args.dataset_root,
        '--sequence', args.sequence,
        '--frame', str(args.frame),
        '--output', args.output
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_comprehensive_projection(args):
    """运行完整投影验证"""
    import subprocess
    
    cmd = [
        sys.executable,
        str(VALIDATION_DIR / 'comprehensive_projection_validation.py'),
        '--dataset_root', args.dataset_root,
        '--output_dir', args.output_dir
    ]
    
    if args.sequences:
        cmd.extend(['--sequences'] + args.sequences)
    
    result = subprocess.run(cmd)
    return result.returncode


def _collect_dataset_statistics(dataset_root):
    """收集数据集的详细统计信息"""
    import os
    import re
    
    dataset_root = Path(dataset_root)
    sequences_dir = dataset_root / 'sequences'
    poses_dir = dataset_root / 'poses'
    
    if not sequences_dir.exists():
        return None
    
    sequences = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    
    seq_stats = []
    total_frames = 0
    total_velodyne_bytes = 0
    total_image_bytes = 0
    
    for seq in sequences:
        seq_dir = sequences_dir / seq
        image_dir = seq_dir / 'image_2'
        velodyne_dir = seq_dir / 'velodyne'
        calib_file = seq_dir / 'calib.txt'
        times_file = seq_dir / 'times.txt'
        poses_file = poses_dir / f'{seq}.txt'
        
        num_images = len(list(image_dir.glob('*.png'))) if image_dir.exists() else 0
        num_velodyne = len(list(velodyne_dir.glob('*.bin'))) if velodyne_dir.exists() else 0
        num_poses = 0
        if poses_file.exists():
            with open(poses_file) as f:
                num_poses = sum(1 for _ in f)
        
        vel_bytes = sum(f.stat().st_size for f in velodyne_dir.glob('*.bin')) if velodyne_dir.exists() else 0
        img_bytes = sum(f.stat().st_size for f in image_dir.glob('*.png')) if image_dir.exists() else 0
        
        duration = 0.0
        fps = 0.0
        if times_file.exists():
            with open(times_file) as f:
                times = [float(line.strip()) for line in f if line.strip()]
            if len(times) >= 2:
                duration = times[-1] - times[0]
                diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
                median_dt = sorted(diffs)[len(diffs)//2]
                fps = 1.0 / median_dt if median_dt > 0 else 0.0
        
        calib_info = {}
        cam_direction = None
        if calib_file.exists():
            import numpy as np
            calib = {}
            with open(calib_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    key, values = line.split(':', 1)
                    key, values = key.strip(), values.strip()
                    try:
                        calib[key] = np.array([float(x) for x in values.split()])
                    except ValueError:
                        calib[key] = values
            
            if 'P2' in calib:
                P2 = calib['P2'].reshape(3, 4)
                calib_info['fx'] = float(P2[0, 0])
                calib_info['fy'] = float(P2[1, 1])
                calib_info['cx'] = float(P2[0, 2])
                calib_info['cy'] = float(P2[1, 2])
            if 'camera_model' in calib:
                calib_info['camera_model'] = str(calib['camera_model'])
            if 'D' in calib:
                calib_info['has_distortion'] = True
            
            if 'Tr' in calib:
                Tr_3x4 = calib['Tr'].reshape(3, 4)
                R = Tr_3x4[:3, :3]
                cam_fwd = R[:, 2]
                dominant_idx = int(np.argmax(np.abs(cam_fwd)))
                dominant_sign = '+' if cam_fwd[dominant_idx] > 0 else '-'
                axis_labels = {0: 'X', 1: 'Y', 2: 'Z'}
                direction_map = {
                    (0, '+'): '前方(+X)', (0, '-'): '后方(-X)',
                    (1, '+'): '左侧(+Y)', (1, '-'): '右侧(-Y)',
                    (2, '+'): '上方(+Z)', (2, '-'): '下方(-Z)',
                }
                cam_direction = direction_map.get((dominant_idx, dominant_sign), '未知')
                calib_info['camera_direction'] = cam_direction
                R = Tr_3x4[:3, :3]
                calib_info['tr_det'] = float(np.linalg.det(R))
        
        img_resolution = None
        if image_dir.exists():
            sample_imgs = sorted(image_dir.glob('*.png'))[:1]
            if sample_imgs:
                import cv2
                img = cv2.imread(str(sample_imgs[0]))
                if img is not None:
                    img_resolution = (img.shape[1], img.shape[0])
        
        resized_dir = seq_dir.parent.parent / 'image_2_640x360_meta.json'
        has_resized = False
        for d in seq_dir.parent.parent.iterdir():
            if d.name.startswith('image_2_') and d.name != 'image_2' and d.is_dir():
                has_resized = True
                break
        
        info = {
            'sequence': seq,
            'num_images': num_images,
            'num_velodyne': num_velodyne,
            'num_poses': num_poses,
            'aligned': num_images == num_velodyne,
            'velodyne_size_bytes': vel_bytes,
            'image_size_bytes': img_bytes,
            'total_size_bytes': vel_bytes + img_bytes,
            'duration_sec': duration,
            'fps': fps,
            'calib_info': calib_info,
            'camera_direction': cam_direction,
            'image_resolution': img_resolution,
        }
        seq_stats.append(info)
        
        total_frames += num_images
        total_velodyne_bytes += vel_bytes
        total_image_bytes += img_bytes
    
    trip_mapping = _extract_trip_mapping(dataset_root)
    
    return {
        'dataset_root': str(dataset_root),
        'num_sequences': len(sequences),
        'total_frames': total_frames,
        'total_velodyne_bytes': total_velodyne_bytes,
        'total_image_bytes': total_image_bytes,
        'total_size_bytes': total_velodyne_bytes + total_image_bytes,
        'sequences': seq_stats,
        'trip_mapping': trip_mapping,
    }


def _extract_trip_mapping(dataset_root):
    """从 batch processing log 中提取 trip 名称与 sequence ID 的映射"""
    import re
    dataset_root = Path(dataset_root)
    mapping = {}
    
    for log_file in sorted(dataset_root.glob('batch_processing_*.log'), reverse=True):
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    m = re.match(r'\s*seq\s+(\d+)\s+<->\s+(\S+)', line)
                    if m:
                        mapping[m.group(1)] = m.group(2)
        except Exception:
            pass
        if mapping:
            break
    
    return mapping


def _format_bytes(num_bytes):
    """格式化字节数为可读字符串"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024**2):.1f} MB"
    else:
        return f"{num_bytes / (1024**3):.2f} GB"


def _format_duration(seconds):
    """格式化秒数为可读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def run_full_validation(args, thorough=False):
    """运行验证流程

    Args:
        thorough: False=快速模式(前3序列,各1帧), True=完整模式(所有序列+完整投影)
    """
    import subprocess
    import json
    from datetime import datetime
    
    mode_label = "完整模式" if thorough else "快速模式"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_root': args.dataset_root,
        'mode': 'full' if thorough else 'quick',
        'validations': {}
    }
    
    print("\n" + "="*80)
    print(f"BEVCalib 数据集验证 [{mode_label}]")
    print("="*80)
    print(f"数据集:   {args.dataset_root}")
    print(f"输出目录: {output_dir}")
    if thorough:
        print(f"范围:     所有序列 + 完整投影（每序列采样10帧）")
    else:
        print(f"范围:     前3个序列 + 采样投影（各1帧）")
    print("="*80 + "\n")
    
    # 0. 收集数据集统计
    print("\n步骤 0/4: 收集数据集统计信息")
    print("-" * 80)
    dataset_stats = _collect_dataset_statistics(args.dataset_root)
    if dataset_stats:
        results['dataset_stats'] = dataset_stats
        print(f"  序列数: {dataset_stats['num_sequences']}")
        print(f"  总帧数: {dataset_stats['total_frames']:,}")
        print(f"  总数据量: {_format_bytes(dataset_stats['total_size_bytes'])}")
        print(f"    点云: {_format_bytes(dataset_stats['total_velodyne_bytes'])}")
        print(f"    图像: {_format_bytes(dataset_stats['total_image_bytes'])}")
        for s in dataset_stats['sequences']:
            trip = dataset_stats['trip_mapping'].get(s['sequence'], '')
            trip_label = f"  ({trip})" if trip else ""
            print(f"  seq {s['sequence']}: {s['num_images']:>6,} 帧, "
                  f"{_format_bytes(s['total_size_bytes']):>10}, "
                  f"{_format_duration(s['duration_sec']):>8}, "
                  f"cam={s.get('camera_direction', '?')}{trip_label}")
    
    # 1. 快速摘要
    print("\n步骤 1/4: 数据集摘要")
    print("-" * 80)
    try:
        from show_dataset_summary import show_dataset_summary
        show_dataset_summary(args.dataset_root)
        results['validations']['summary'] = {'status': 'success'}
    except Exception as e:
        print(f"错误: {e}")
        results['validations']['summary'] = {'status': 'failed', 'error': str(e)}
    
    # 2. Tr矩阵验证
    print("\n步骤 2/4: Tr矩阵验证")
    print("-" * 80)
    cmd = [sys.executable, str(VALIDATION_DIR / 'verify_dataset_tr_fix.py'),
           '--dataset_root', args.dataset_root]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    log_file = output_dir / 'tr_matrix_validation.log'
    with open(log_file, 'w') as f:
        f.write(result.stdout)
    
    results['validations']['tr_matrix'] = {
        'status': 'success' if result.returncode == 0 else 'failed',
        'log_file': str(log_file)
    }
    
    # 3. KITTI格式验证
    print("\n步骤 3/4: KITTI格式验证")
    print("-" * 80)
    
    sequences_dir = Path(args.dataset_root) / 'sequences'
    sequences = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    validate_sequences = sequences if thorough else sequences[:3]
    
    format_results = []
    for seq in validate_sequences:
        print(f"  验证序列 {seq}...")
        cmd = [sys.executable, str(VALIDATION_DIR / 'validate_kitti_odometry.py'),
               args.dataset_root, '--sequence', seq]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        log_file = output_dir / f'format_seq{seq}.log'
        with open(log_file, 'w') as f:
            f.write(result.stdout)
        
        format_results.append({
            'sequence': seq,
            'status': 'success' if result.returncode == 0 else 'failed',
            'log_file': str(log_file)
        })
    
    results['validations']['format'] = format_results
    
    # 4. 投影验证
    if thorough:
        print("\n步骤 4/4: 完整投影验证（每序列采样10帧）")
        print("-" * 80)
        print("  正在运行完整投影验证...")
        
        projection_dir = output_dir / 'projection_validation'
        cmd = [sys.executable, str(VALIDATION_DIR / 'comprehensive_projection_validation.py'),
               '--dataset_root', args.dataset_root,
               '--output_dir', str(projection_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        proj_summary = None
        proj_summary_file = projection_dir / 'summary.json'
        if proj_summary_file.exists():
            try:
                import json as _json
                with open(proj_summary_file) as f:
                    proj_summary = _json.load(f)
            except Exception:
                pass
        
        results['validations']['projection'] = {
            'type': 'comprehensive',
            'status': 'success' if result.returncode == 0 else 'failed',
            'output_dir': str(projection_dir),
            'description': '每序列采样10帧（均匀分布）',
            'summary': proj_summary,
        }
        
        if result.returncode == 0:
            print("  完整投影验证完成")
            print(f"  报告: {projection_dir}/PROJECTION_VALIDATION_REPORT.md")
        else:
            print("  投影验证出现问题（详见报告）")
    else:
        print("\n步骤 4/4: 快速投影验证（采样）")
        print("-" * 80)
        
        projection_dir = output_dir / 'sample_projections'
        projection_dir.mkdir(exist_ok=True)
        
        projection_results = []
        for seq in sequences[:3]:
            output_file = projection_dir / f'seq{seq}_frame000000.png'
            print(f"  测试序列 {seq} 第0帧...")
            
            cmd = [sys.executable, str(VALIDATION_DIR / 'check_projection_headless.py'),
                   '--dataset_root', args.dataset_root,
                   '--sequence', seq,
                   '--frame', '0',
                   '--output', str(output_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            projection_results.append({
                'sequence': seq,
                'frame': 0,
                'status': 'success' if result.returncode == 0 else 'failed',
                'output_file': str(output_file)
            })
        
        results['validations']['projection'] = {
            'type': 'sampled',
            'samples': projection_results,
            'description': '前3序列各1帧（第0帧）'
        }
    
    # 保存结果JSON
    results_file = output_dir / 'validation_summary.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成报告
    _write_report(results, output_dir, thorough)
    
    report_file = output_dir / 'VALIDATION_SUMMARY.md'
    print("\n" + "="*80)
    print(f"验证完成！[{mode_label}]")
    print(f"报告: {report_file}")
    print(f"JSON: {results_file}")
    print("="*80 + "\n")
    
    return 0


def _write_report(results, output_dir, thorough):
    """生成 Markdown 验证报告"""
    report_file = output_dir / 'VALIDATION_SUMMARY.md'
    results_file = output_dir / 'validation_summary.json'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        mode_label = "完整验证" if thorough else "快速验证"
        f.write(f"# 数据集验证报告（{mode_label}）\n\n")
        f.write(f"**验证时间**: {results['timestamp']}\n\n")
        f.write(f"**数据集**: `{results['dataset_root']}`\n\n")
        f.write(f"**模式**: {mode_label}\n\n")
        
        # ============================
        # Section 1: 数据集总览
        # ============================
        ds = results.get('dataset_stats')
        if ds:
            f.write("---\n\n")
            f.write("## 1. 数据集总览\n\n")
            
            total_duration = sum(s['duration_sec'] for s in ds['sequences'])
            
            f.write("### 基本信息\n\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|----|\n")
            f.write(f"| 序列数 | {ds['num_sequences']} |\n")
            f.write(f"| 总帧数 | {ds['total_frames']:,} |\n")
            f.write(f"| 总时长 | {_format_duration(total_duration)} |\n")
            f.write(f"| 总数据量 | {_format_bytes(ds['total_size_bytes'])} |\n")
            f.write(f"| 点云数据 | {_format_bytes(ds['total_velodyne_bytes'])} |\n")
            f.write(f"| 图像数据 | {_format_bytes(ds['total_image_bytes'])} |\n")
            f.write(f"\n")
            
            # Per-sequence breakdown
            f.write("### 各序列明细\n\n")
            f.write("| 序列 | Trip 名称 | 帧数 | 时长 | 帧率 | 数据量 | 图像分辨率 | 相机朝向 | 相机模型 |\n")
            f.write("|------|-----------|------|------|------|--------|-----------|----------|----------|\n")
            
            for s in ds['sequences']:
                trip = ds['trip_mapping'].get(s['sequence'], '-')
                ci = s.get('calib_info', {})
                res_str = f"{s['image_resolution'][0]}x{s['image_resolution'][1]}" if s.get('image_resolution') else '-'
                cam_dir = s.get('camera_direction', '-') or '-'
                cam_model = ci.get('camera_model', '-')
                f.write(f"| {s['sequence']} | {trip} | {s['num_images']:,} | "
                       f"{_format_duration(s['duration_sec'])} | {s['fps']:.1f} | "
                       f"{_format_bytes(s['total_size_bytes'])} | "
                       f"{res_str} | {cam_dir} | {cam_model} |\n")
            f.write("\n")
            
            # Data distribution chart (text)
            f.write("### 数据分布\n\n")
            f.write("```\n")
            max_frames = max(s['num_images'] for s in ds['sequences']) if ds['sequences'] else 1
            bar_width = 40
            for s in ds['sequences']:
                bar_len = int(s['num_images'] / max_frames * bar_width) if max_frames > 0 else 0
                bar = '#' * bar_len + '.' * (bar_width - bar_len)
                pct = s['num_images'] / ds['total_frames'] * 100 if ds['total_frames'] > 0 else 0
                f.write(f"  seq {s['sequence']}  |{bar}| {s['num_images']:>6,} 帧 ({pct:5.1f}%)\n")
            f.write("```\n\n")
            
            # Train/Val split
            f.write("### 训练/验证集划分建议 (80/20)\n\n")
            train_ratio = 0.8
            total = ds['total_frames']
            train_target = int(total * train_ratio)
            val_target = total - train_target
            
            f.write(f"| 划分 | 帧数 | 占比 |\n")
            f.write(f"|------|------|------|\n")
            f.write(f"| 训练集 | {train_target:,} | {train_ratio*100:.0f}% |\n")
            f.write(f"| 验证集 | {val_target:,} | {(1-train_ratio)*100:.0f}% |\n\n")
            
            # Per-sequence split suggestion
            f.write("**逐序列划分明细** (每个序列内部按帧顺序前80%训练、后20%验证):\n\n")
            f.write("| 序列 | 总帧数 | 训练帧 | 验证帧 |\n")
            f.write("|------|--------|--------|--------|\n")
            total_train = 0
            total_val = 0
            for s in ds['sequences']:
                n = s['num_images']
                n_train = int(n * train_ratio)
                n_val = n - n_train
                total_train += n_train
                total_val += n_val
                f.write(f"| {s['sequence']} | {n:,} | {n_train:,} | {n_val:,} |\n")
            f.write(f"| **合计** | **{total:,}** | **{total_train:,}** | **{total_val:,}** |\n\n")
            
            # Camera intrinsics comparison
            f.write("### 相机内参对比\n\n")
            f.write("| 序列 | fx | fy | cx | cy |\n")
            f.write("|------|-----|-----|-----|-----|\n")
            for s in ds['sequences']:
                ci = s.get('calib_info', {})
                f.write(f"| {s['sequence']} | "
                       f"{ci.get('fx', 0):.1f} | {ci.get('fy', 0):.1f} | "
                       f"{ci.get('cx', 0):.1f} | {ci.get('cy', 0):.1f} |\n")
            f.write("\n")
        
        # ============================
        # Section 2: 验证结果
        # ============================
        f.write("---\n\n")
        f.write("## 2. 验证结果\n\n")
        
        # Summary
        f.write("### 2.1 数据集格式摘要\n")
        summary_ok = results['validations']['summary']['status'] == 'success'
        f.write(f"状态: {'PASS' if summary_ok else 'FAIL'}\n\n")
        
        # Tr
        f.write("### 2.2 Tr矩阵验证\n")
        tr_ok = results['validations']['tr_matrix']['status'] == 'success'
        f.write(f"状态: {'PASS' if tr_ok else 'FAIL'}\n")
        f.write(f"日志: `{results['validations']['tr_matrix']['log_file']}`\n\n")
        
        # Format
        f.write("### 2.3 KITTI格式验证\n\n")
        f.write("| 序列 | 状态 | 日志文件 |\n")
        f.write("|------|------|----------|\n")
        for fmt in results['validations']['format']:
            status = 'PASS' if fmt['status'] == 'success' else 'FAIL'
            f.write(f"| {fmt['sequence']} | {status} | `{fmt['log_file']}` |\n")
        f.write("\n")
        
        # Projection
        proj = results['validations']['projection']
        if proj['type'] == 'sampled':
            f.write("### 2.4 投影验证（采样）\n\n")
            f.write(f"说明: {proj['description']}\n\n")
            f.write("| 序列 | 帧 | 状态 | 输出文件 |\n")
            f.write("|------|-----|------|----------|\n")
            for s in proj['samples']:
                status = 'PASS' if s['status'] == 'success' else 'FAIL'
                f.write(f"| {s['sequence']} | {s['frame']:06d} | {status} | `{s['output_file']}` |\n")
            f.write("\n")
        else:
            f.write("### 2.4 完整投影验证\n\n")
            f.write(f"说明: {proj['description']}\n\n")
            status = 'PASS' if proj['status'] == 'success' else 'FAIL'
            f.write(f"状态: {status}\n")
            f.write(f"输出目录: `{proj['output_dir']}`\n\n")
            
            proj_summary = proj.get('summary')
            if proj_summary and 'sequences' in proj_summary:
                f.write("| 序列 | 帧数 | 采样数 | 平均可见率 | 相机朝向 | 状态 |\n")
                f.write("|------|------|--------|-----------|----------|------|\n")
                for ps in proj_summary['sequences']:
                    cam = ps.get('camera_direction', {})
                    cam_label = cam.get('direction', '-') if cam else '-'
                    sm = ps.get('summary', {})
                    if sm.get('status') == 'ok':
                        avg_r = sm.get('avg_visible_ratio', 0) * 100
                        f.write(f"| {ps['sequence']} | {ps['num_frames']:,} | {ps['num_samples']} | "
                               f"{avg_r:.1f}% | {cam_label} | PASS |\n")
                    else:
                        reason = sm.get('reason', '无可见点')
                        f.write(f"| {ps['sequence']} | {ps['num_frames']:,} | 0 | "
                               f"0.0% | {cam_label} | WARN |\n")
                f.write("\n")
                
                fail_seqs = [ps for ps in proj_summary['sequences']
                           if ps.get('summary', {}).get('status') != 'ok']
                if fail_seqs:
                    f.write("**投影异常序列说明**:\n\n")
                    for ps in fail_seqs:
                        reason = ps.get('summary', {}).get('reason', '所有采样帧均无可见点')
                        f.write(f"- seq {ps['sequence']}: {reason}\n")
                    f.write("\n")
        
        # ============================
        # Section 3: 文件位置
        # ============================
        f.write("---\n\n")
        f.write("## 3. 文件位置\n\n")
        f.write(f"- 验证报告: `{report_file}`\n")
        f.write(f"- JSON结果: `{results_file}`\n")
        f.write(f"- 日志目录: `{output_dir}/`\n")
        if proj.get('output_dir'):
            f.write(f"- 投影验证: `{proj['output_dir']}/PROJECTION_VALIDATION_REPORT.md`\n")


def main():
    parser = argparse.ArgumentParser(
        description='BEVCalib 数据集统一验证工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 数据集摘要
  python validate_dataset.py summary /path/to/dataset

  # 快速验证 (~17秒，前3序列，各1帧投影)
  python validate_dataset.py quick /path/to/dataset --output-dir results/

  # 完整验证 (~15分钟，所有序列 + 完整投影)
  python validate_dataset.py full /path/to/dataset --output-dir results/

  # 格式验证（单序列 / 全部）
  python validate_dataset.py format /path/to/dataset --sequence 00
  python validate_dataset.py format /path/to/dataset --all

  # Tr矩阵验证
  python validate_dataset.py tr /path/to/dataset

  # 单帧投影测试
  python validate_dataset.py projection /path/to/dataset --sequence 00 --frame 0

  # 多序列投影验证
  python validate_dataset.py projection-full /path/to/dataset --output-dir proj/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='验证命令')
    
    # --- 快速验证 ---
    quick_parser = subparsers.add_parser(
        'quick', help='快速验证（~17秒，前3序列，各1帧投影）')
    quick_parser.add_argument('dataset_root', help='数据集根目录')
    quick_parser.add_argument('--output-dir', required=True, help='验证结果输出目录')
    
    # --- 完整验证 ---
    full_parser = subparsers.add_parser(
        'full', help='完整验证（~15分钟，所有序列 + 完整投影）')
    full_parser.add_argument('dataset_root', help='数据集根目录')
    full_parser.add_argument('--output-dir', required=True, help='验证结果输出目录')
    
    # --- 摘要 ---
    summary_parser = subparsers.add_parser('summary', help='显示数据集摘要')
    summary_parser.add_argument('dataset_root', help='数据集根目录')
    
    # --- 格式验证 ---
    format_parser = subparsers.add_parser('format', help='验证KITTI格式')
    format_parser.add_argument('dataset_root', help='数据集根目录')
    format_parser.add_argument('--sequence', default='00', help='序列ID (默认: 00)')
    format_parser.add_argument('--all', dest='all_sequences', action='store_true',
                              help='验证所有序列')
    
    # --- Tr矩阵验证 ---
    tr_parser = subparsers.add_parser('tr', help='验证Tr矩阵')
    tr_parser.add_argument('dataset_root', help='数据集根目录')
    
    # --- 单帧投影 ---
    projection_parser = subparsers.add_parser('projection', help='测试单帧投影')
    projection_parser.add_argument('dataset_root', help='数据集根目录')
    projection_parser.add_argument('--sequence', default='00', help='序列ID')
    projection_parser.add_argument('--frame', type=int, default=0, help='帧号')
    projection_parser.add_argument('--output', help='输出文件路径')
    
    # --- 多序列投影验证 ---
    proj_full_parser = subparsers.add_parser('projection-full',
                                             help='多序列投影验证（每序列多帧采样）')
    proj_full_parser.add_argument('dataset_root', help='数据集根目录')
    proj_full_parser.add_argument('--output-dir', required=True, help='输出目录')
    proj_full_parser.add_argument('--sequences', nargs='+', help='指定序列（默认全部）')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'summary':
            return run_summary(args)
        elif args.command == 'format':
            return run_format_validation(args)
        elif args.command == 'tr':
            return run_tr_validation(args)
        elif args.command == 'projection':
            return run_projection_test(args)
        elif args.command == 'projection-full':
            return run_comprehensive_projection(args)
        elif args.command == 'quick':
            return run_full_validation(args, thorough=False)
        elif args.command == 'full':
            return run_full_validation(args, thorough=True)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 130
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
