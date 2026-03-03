#!/usr/bin/env python3
"""
验证数据集 Tr 矩阵格式

检查项：
1. Tr 矩阵是否存在
2. 旋转矩阵正交性、行列式
3. 平移向量合理性

用法：
    python tools/verify_dataset_tr_fix.py --dataset_root /path/to/dataset
"""

import numpy as np
import argparse
from pathlib import Path


def load_calib_tr(calib_file):
    """加载 calib.txt 中的 Tr 矩阵"""
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('Tr:'):
                values = [float(x) for x in line.strip().split()[1:]]
                if len(values) == 12:
                    return np.array(values).reshape(3, 4)
    return None


def verify_tr_format(tr_3x4):
    """验证 Tr 矩阵格式"""
    R = tr_3x4[:3, :3]
    t = tr_3x4[:3, 3]
    
    RTR = R.T @ R
    is_orthogonal = np.allclose(RTR, np.eye(3), atol=1e-3)
    
    det_R = np.linalg.det(R)
    is_proper_rotation = np.isclose(det_R, 1.0, atol=1e-3)
    
    trans_norm = np.linalg.norm(t)
    is_reasonable_translation = 0.1 < trans_norm < 10.0

    return {
        'is_orthogonal': is_orthogonal,
        'is_proper_rotation': is_proper_rotation,
        'is_reasonable_translation': is_reasonable_translation,
        'translation_norm': trans_norm,
        'det_R': det_R,
        'all_checks_passed': is_orthogonal and is_proper_rotation and is_reasonable_translation
    }


def main():
    parser = argparse.ArgumentParser(description='验证数据集 Tr 矩阵格式')
    parser.add_argument('--dataset_root', type=str, required=True, help='数据集根目录')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    print(f'\n{"="*80}')
    print(f'验证数据集: {dataset_root}')
    print(f'{"="*80}\n')

    sequences_dir = dataset_root / 'sequences'
    if not sequences_dir.exists():
        print(f'未找到 sequences 目录: {sequences_dir}')
        return 1

    calib_files = sorted(sequences_dir.glob('*/calib.txt'))
    if not calib_files:
        print(f'未找到任何 calib.txt 文件')
        return 1

    print(f'找到 {len(calib_files)} 个 calib.txt 文件\n')

    all_passed = True

    for calib_file in calib_files:
        seq_name = calib_file.parent.name

        print(f'{"="*80}')
        print(f'序列: {seq_name}')
        print(f'{"="*80}')

        tr = load_calib_tr(calib_file)
        if tr is None:
            print(f'  未找到 Tr 矩阵')
            all_passed = False
            continue

        print(f'\nTr 矩阵 (Camera → Sensing, KITTI 标准):')
        print(tr)

        print(f'\n验证 Tr 矩阵格式...')
        checks = verify_tr_format(tr)

        print(f'  - 旋转矩阵正交性: {"PASS" if checks["is_orthogonal"] else "FAIL"}')
        print(f'  - 旋转矩阵行列式: {checks["det_R"]:.6f} (应接近 1.0)')
        print(f'  - 平移向量合理性: {"PASS" if checks["is_reasonable_translation"] else "FAIL"}')
        print(f'  - 平移向量模: {checks["translation_norm"]:.4f} m')

        if checks['all_checks_passed']:
            print(f'\n  Tr 矩阵格式检查通过')
        else:
            print(f'\n  Tr 矩阵格式检查失败')
            all_passed = False

        print()

    # 总结
    print(f'{"="*80}')
    print(f'验证总结')
    print(f'{"="*80}\n')

    if all_passed:
        print(f'所有检查通过！数据集 Tr 矩阵格式正确。')
    else:
        print(f'部分检查失败，请检查上述错误信息。')

    print()
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
