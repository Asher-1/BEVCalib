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
        print(f"范围:     所有序列 + 完整投影（每序列采样5帧）")
    else:
        print(f"范围:     前3个序列 + 采样投影（各1帧）")
    print("="*80 + "\n")
    
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
        print("\n步骤 4/4: 完整投影验证（每序列采样5帧）")
        print("-" * 80)
        print("  正在运行完整投影验证...")
        print("  预计耗时约10-15分钟...")
        
        projection_dir = output_dir / 'projection_validation'
        cmd = [sys.executable, str(VALIDATION_DIR / 'comprehensive_projection_validation.py'),
               '--dataset_root', args.dataset_root,
               '--output_dir', str(projection_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        results['validations']['projection'] = {
            'type': 'comprehensive',
            'status': 'success' if result.returncode == 0 else 'failed',
            'output_dir': str(projection_dir),
            'description': '每序列采样5帧（开始、1/4、中间、3/4、结束）'
        }
        
        if result.returncode == 0:
            print("  完整投影验证完成")
            print(f"  报告: {projection_dir}/PROJECTION_VALIDATION_REPORT.md")
        else:
            print("  投影验证失败")
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
        f.write(f"# 数据集验证摘要（{mode_label}）\n\n")
        f.write(f"**验证时间**: {results['timestamp']}\n\n")
        f.write(f"**数据集**: `{results['dataset_root']}`\n\n")
        f.write(f"**模式**: {mode_label}\n\n")
        f.write("## 验证结果\n\n")
        
        # 摘要
        f.write("### 1. 数据集摘要\n")
        summary_ok = results['validations']['summary']['status'] == 'success'
        f.write(f"状态: {'PASS' if summary_ok else 'FAIL'}\n\n")
        
        # Tr矩阵
        f.write("### 2. Tr矩阵验证\n")
        tr_ok = results['validations']['tr_matrix']['status'] == 'success'
        f.write(f"状态: {'PASS' if tr_ok else 'FAIL'}\n")
        f.write(f"日志: `{results['validations']['tr_matrix']['log_file']}`\n\n")
        
        # 格式验证
        f.write("### 3. KITTI格式验证\n\n")
        f.write("| 序列 | 状态 | 日志文件 |\n")
        f.write("|------|------|----------|\n")
        for fmt in results['validations']['format']:
            status = 'PASS' if fmt['status'] == 'success' else 'FAIL'
            f.write(f"| {fmt['sequence']} | {status} | `{fmt['log_file']}` |\n")
        f.write("\n")
        
        # 投影验证
        proj = results['validations']['projection']
        if proj['type'] == 'sampled':
            f.write("### 4. 投影验证（采样）\n\n")
            f.write(f"说明: {proj['description']}\n\n")
            f.write("| 序列 | 帧 | 状态 | 输出文件 |\n")
            f.write("|------|-----|------|----------|\n")
            for s in proj['samples']:
                status = 'PASS' if s['status'] == 'success' else 'FAIL'
                f.write(f"| {s['sequence']} | {s['frame']:06d} | {status} | `{s['output_file']}` |\n")
            f.write("\n")
        else:
            f.write("### 4. 完整投影验证\n\n")
            f.write(f"说明: {proj['description']}\n\n")
            status = 'PASS' if proj['status'] == 'success' else 'FAIL'
            f.write(f"状态: {status}\n")
            f.write(f"输出目录: `{proj['output_dir']}`\n\n")
        
        f.write("## 文件位置\n\n")
        f.write(f"- JSON结果: `{results_file}`\n")
        f.write(f"- 验证报告: `{report_file}`\n")
        f.write(f"- 日志目录: `{output_dir}/`\n")


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
