"""
BEVCalib 公共工具模块

提供标准的外参误差评估工具，与 C++ Eigen 实现完全一致。
"""

from .evaluate_extrinsics import (
    evaluate_sensor_extrinsic,
    decompose_rotation_error,
    print_evaluation_results,
    load_transformation_from_calib,
    compare_two_transforms
)

__all__ = [
    'evaluate_sensor_extrinsic',
    'decompose_rotation_error',
    'print_evaluation_results',
    'load_transformation_from_calib',
    'compare_two_transforms',
]

__version__ = '1.0.0'
