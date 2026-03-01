#!/usr/bin/env python3
"""
修复已生成数据集中的 Tr 矩阵

问题：
在修复代码之前生成的数据集，calib.txt 中的 Tr 矩阵格式为 Sensing → Camera（旧格式）
需要取逆转换为 Camera → Sensing（KITTI 标准格式）

用法：
    python tools/fix_calib_tr_inversion.py --dataset_root /path/to/dataset
    
功能：
    1. 扫描数据集中所有 sequences/*/calib.txt 文件
    2. 读取 Tr 矩阵（3x4）
    3. 转换为 4x4 齐次矩阵
    4. 取逆：Camera → Sensing = inv(Sensing → Camera)
    5. 备份原文件为 calib.txt.backup
    6. 保存修复后的 calib.txt
"""

import numpy as np
import argparse
import os
from pathlib import Path
import shutil


def parse_calib_line(line):
    """解析 calib.txt 中的一行"""
    parts = line.strip().split()
    label = parts[0].rstrip(':')
    
    # 尝试转换为 float，如果失败则保持原始字符串
    values = []
    for x in parts[1:]:
        try:
            values.append(float(x))
        except ValueError:
            # 非数值字段（如 camera_model: pinhole），保持原样
            return label, parts[1:]
    
    return label, values


def write_calib_line(label, values):
    """格式化输出 calib.txt 的一行"""
    # 判断是否全是数值
    if all(isinstance(v, (int, float)) for v in values):
        values_str = ' '.join([f'{v:e}' for v in values])
    else:
        # 非数值字段（如 camera_model），直接拼接
        values_str = ' '.join([str(v) for v in values])
    return f'{label}: {values_str}\n'


def invert_tr_matrix(tr_3x4):
    """
    将 3x4 的 Tr 矩阵取逆
    
    输入：
        tr_3x4: (3, 4) numpy array, 旧格式 Tr = Sensing → Camera
        
    输出：
        tr_3x4_inv: (3, 4) numpy array, KITTI 标准 Tr = Camera → Sensing
    """
    # 转换为 4x4 齐次矩阵
    tr_4x4 = np.vstack([tr_3x4, [0, 0, 0, 1]])
    
    # 取逆
    tr_4x4_inv = np.linalg.inv(tr_4x4)
    
    # 转回 3x4
    tr_3x4_inv = tr_4x4_inv[:3, :]
    
    return tr_3x4_inv


def fix_calib_file(calib_path, backup=True, dry_run=False):
    """
    修复单个 calib.txt 文件中的 Tr 矩阵
    
    参数：
        calib_path: calib.txt 文件路径
        backup: 是否备份原文件
        dry_run: 是否只预览不实际修改
        
    返回：
        success: 是否成功修复
        old_tr: 旧的 Tr 矩阵（3x4）
        new_tr: 新的 Tr 矩阵（3x4）
    """
    print(f'\n{"="*80}')
    print(f'处理文件: {calib_path}')
    print(f'{"="*80}')
    
    if not os.path.exists(calib_path):
        print(f'❌ 文件不存在: {calib_path}')
        return False, None, None
    
    # 读取文件
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # 解析所有行
    calib_data = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            label, values = parse_calib_line(line)
            calib_data[label] = values
    
    # 检查 Tr 是否存在
    if 'Tr' not in calib_data:
        print('❌ 未找到 Tr 矩阵')
        return False, None, None
    
    # 获取旧的 Tr 矩阵
    old_tr_flat = calib_data['Tr']
    if len(old_tr_flat) != 12:
        print(f'❌ Tr 矩阵格式错误，应为 12 个值，实际为 {len(old_tr_flat)} 个')
        return False, None, None
    
    old_tr = np.array(old_tr_flat).reshape(3, 4)
    
    print('\n旧 Tr 矩阵 (Sensing → Camera):')
    print(old_tr)
    
    # 取逆得到新的 Tr 矩阵
    new_tr = invert_tr_matrix(old_tr)
    
    print('\n新 Tr 矩阵 (Camera → Sensing, KITTI 标准):')
    print(new_tr)
    
    # 验证逆矩阵
    old_tr_4x4 = np.vstack([old_tr, [0, 0, 0, 1]])
    new_tr_4x4 = np.vstack([new_tr, [0, 0, 0, 1]])
    identity = old_tr_4x4 @ new_tr_4x4
    is_valid = np.allclose(identity, np.eye(4), atol=1e-6)
    
    print(f'\n验证: old_Tr @ new_Tr = I? {is_valid}')
    if is_valid:
        print('✓ 逆矩阵验证通过')
    else:
        print('❌ 逆矩阵验证失败')
        print('Identity matrix:')
        print(identity)
        return False, old_tr, new_tr
    
    if dry_run:
        print('\n[DRY RUN] 预览模式，不实际修改文件')
        return True, old_tr, new_tr
    
    # 备份原文件
    if backup:
        backup_path = str(calib_path) + '.backup'
        shutil.copy2(str(calib_path), backup_path)
        print(f'\n✓ 已备份原文件: {backup_path}')
    
    # 更新 Tr 矩阵
    calib_data['Tr'] = new_tr.flatten().tolist()
    
    # 写回文件
    with open(calib_path, 'w') as f:
        # 按顺序写入（保持原有顺序）
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                f.write(line)
                continue
            
            if ':' not in line_stripped:
                f.write(line)
                continue
            
            label = line_stripped.split(':')[0]
            if label in calib_data:
                f.write(write_calib_line(label, calib_data[label]))
            else:
                f.write(line)
    
    print(f'\n✓ 已保存修复后的文件: {calib_path}')
    
    return True, old_tr, new_tr


def main():
    parser = argparse.ArgumentParser(
        description='修复数据集中的 Tr 矩阵（从 Sensing→Camera 转换为 Camera→Sensing）'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='数据集根目录路径'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不备份原文件'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，不实际修改文件'
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f'❌ 数据集根目录不存在: {dataset_root}')
        return
    
    # 查找所有 calib.txt 文件
    sequences_dir = dataset_root / 'sequences'
    if not sequences_dir.exists():
        print(f'❌ 未找到 sequences 目录: {sequences_dir}')
        return
    
    calib_files = sorted(sequences_dir.glob('*/calib.txt'))
    
    if not calib_files:
        print(f'❌ 未找到任何 calib.txt 文件')
        return
    
    print(f'\n{"="*80}')
    print(f'数据集根目录: {dataset_root}')
    print(f'找到 {len(calib_files)} 个 calib.txt 文件')
    print(f'备份原文件: {"否" if args.no_backup else "是"}')
    print(f'预览模式: {"是" if args.dry_run else "否"}')
    print(f'{"="*80}')
    
    # 统计
    success_count = 0
    failed_count = 0
    
    # 处理每个文件
    for calib_file in calib_files:
        success, old_tr, new_tr = fix_calib_file(
            calib_file,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # 输出统计
    print(f'\n{"="*80}')
    print('修复完成')
    print(f'{"="*80}')
    print(f'总文件数: {len(calib_files)}')
    print(f'成功修复: {success_count}')
    print(f'失败: {failed_count}')
    
    if args.dry_run:
        print('\n[DRY RUN] 这是预览模式，未实际修改文件')
        print('如需真正修改，请移除 --dry-run 参数重新运行')
    else:
        print('\n✓ 所有文件已修复')
        print(f'✓ 备份文件: {dataset_root}/sequences/*/calib.txt.backup')
        print('\n📌 重要提示:')
        print('   - 旧格式: Tr = Sensing → Camera')
        print('   - 新格式: Tr = Camera → Sensing (KITTI 标准)')
        print('   - 数据加载器会自动取逆: inv(Tr) = Sensing → Camera (用于投影)')


if __name__ == '__main__':
    main()
