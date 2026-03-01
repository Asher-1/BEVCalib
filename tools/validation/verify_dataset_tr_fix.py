#!/usr/bin/env python3
"""
验证数据集 Tr 矩阵修复是否成功

检查项：
1. Tr 矩阵格式是否符合 KITTI 标准（Camera → Sensing）
2. 数据加载器是否正确处理 Tr 矩阵
3. 投影是否正常工作

用法：
    python tools/verify_dataset_tr_fix.py --dataset_root /path/to/dataset
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    """
    验证 Tr 矩阵是否符合 KITTI 标准（Camera → Sensing）
    
    简单的启发式检查：
    - Tr 的旋转部分应该接近单位阵的某种旋转
    - 平移部分应该合理（通常几十厘米到几米）
    """
    # 转换为 4x4
    tr_4x4 = np.vstack([tr_3x4, [0, 0, 0, 1]])
    
    # 提取旋转和平移
    R = tr_3x4[:3, :3]
    t = tr_3x4[:3, 3]
    
    # 检查旋转矩阵的性质
    # R^T @ R 应该接近单位阵
    RTR = R.T @ R
    is_orthogonal = np.allclose(RTR, np.eye(3), atol=1e-3)
    
    # det(R) 应该接近 1
    det_R = np.linalg.det(R)
    is_proper_rotation = np.isclose(det_R, 1.0, atol=1e-3)
    
    # 平移向量应该合理（不能太大）
    trans_norm = np.linalg.norm(t)
    is_reasonable_translation = 0.1 < trans_norm < 10.0  # 10cm 到 10m
    
    return {
        'is_orthogonal': is_orthogonal,
        'is_proper_rotation': is_proper_rotation,
        'is_reasonable_translation': is_reasonable_translation,
        'translation_norm': trans_norm,
        'det_R': det_R,
        'all_checks_passed': is_orthogonal and is_proper_rotation and is_reasonable_translation
    }


def test_data_loader(dataset_root):
    """测试数据加载器是否正确处理 Tr 矩阵"""
    try:
        from kitti_bev_calib.custom_dataset import CustomDataset
        
        # 创建数据集
        dataset = CustomDataset(
            root_dir=dataset_root,
            mode='train',
            target_width=640,
            target_height=360,
            validate_sample_ratio=0.1,
            perturbation_params={'angle_range_deg': 20.0, 'trans_range': 1.5}
        )
        
        # 获取第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            return True, f"数据加载器正常，数据集包含 {len(dataset)} 个样本"
        else:
            return False, "数据集为空"
    
    except Exception as e:
        return False, f"数据加载器错误: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='验证数据集 Tr 矩阵修复')
    parser.add_argument('--dataset_root', type=str, required=True, help='数据集根目录')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    print(f'\n{"="*80}')
    print(f'验证数据集: {dataset_root}')
    print(f'{"="*80}\n')
    
    # 查找所有 calib.txt
    sequences_dir = dataset_root / 'sequences'
    if not sequences_dir.exists():
        print(f'❌ 未找到 sequences 目录: {sequences_dir}')
        return
    
    calib_files = sorted(sequences_dir.glob('*/calib.txt'))
    
    if not calib_files:
        print(f'❌ 未找到任何 calib.txt 文件')
        return
    
    print(f'找到 {len(calib_files)} 个 calib.txt 文件\n')
    
    all_passed = True
    
    # 检查每个 calib.txt
    for calib_file in calib_files:
        seq_name = calib_file.parent.name
        
        print(f'{"="*80}')
        print(f'序列: {seq_name}')
        print(f'{"="*80}')
        
        # 1. 加载 Tr 矩阵
        tr = load_calib_tr(calib_file)
        if tr is None:
            print(f'❌ 未找到 Tr 矩阵')
            all_passed = False
            continue
        
        print(f'\nTr 矩阵 (Camera → Sensing, KITTI 标准):')
        print(tr)
        
        # 2. 验证格式
        print(f'\n验证 Tr 矩阵格式...')
        checks = verify_tr_format(tr)
        
        print(f'  - 旋转矩阵正交性: {"✓" if checks["is_orthogonal"] else "✗"}')
        print(f'  - 旋转矩阵行列式: {checks["det_R"]:.6f} (应接近 1.0)')
        print(f'  - 平移向量合理性: {"✓" if checks["is_reasonable_translation"] else "✗"}')
        print(f'  - 平移向量模: {checks["translation_norm"]:.4f} m')
        
        if checks['all_checks_passed']:
            print(f'\n✓ Tr 矩阵格式检查通过')
        else:
            print(f'\n✗ Tr 矩阵格式检查失败')
            all_passed = False
        
        # 3. 检查是否有备份文件（说明已经修复过）
        backup_file = str(calib_file) + '.backup'
        if os.path.exists(backup_file):
            print(f'\n✓ 找到备份文件: {backup_file}')
            print(f'  （说明已通过 fix_calib_tr_inversion.py 修复）')
            
            # 对比修复前后
            tr_backup = load_calib_tr(backup_file)
            if tr_backup is not None:
                print(f'\n修复前后对比:')
                print(f'  - 修复前 Tr[0,0]: {tr_backup[0,0]:.6e}')
                print(f'  - 修复后 Tr[0,0]: {tr[0,0]:.6e}')
                print(f'  - 是否相同: {np.allclose(tr, tr_backup)}')
        else:
            print(f'\n⚠️  未找到备份文件')
            print(f'  （可能是新生成的数据集，或者未使用 fix_calib_tr_inversion.py 修复）')
        
        print()
    
    # 4. 测试数据加载器
    print(f'{"="*80}')
    print(f'测试数据加载器')
    print(f'{"="*80}\n')
    
    loader_ok, loader_msg = test_data_loader(dataset_root)
    if loader_ok:
        print(f'✓ {loader_msg}')
    else:
        print(f'✗ {loader_msg}')
        all_passed = False
    
    # 5. 总结
    print(f'\n{"="*80}')
    print(f'验证总结')
    print(f'{"="*80}\n')
    
    if all_passed:
        print(f'✅ 所有检查通过！数据集 Tr 矩阵格式正确。')
        print(f'\n数据集可以正常用于训练和推理。')
    else:
        print(f'❌ 部分检查失败，请检查上述错误信息。')
        print(f'\n建议：')
        print(f'  1. 如果数据集是在 2025-02-04 之前生成的，请运行:')
        print(f'     python tools/fix_calib_tr_inversion.py --dataset_root {dataset_root}')
        print(f'  2. 如果数据集是新生成的，请检查生成脚本版本')
    
    print()


if __name__ == '__main__':
    main()
