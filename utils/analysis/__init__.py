"""
BEVCalib Training Analysis Toolkit

通用的训练模型性能分析和泛化能力评估工具集
"""

from .training_analyzer import TrainingAnalyzer
from .test_evaluator import TestEvaluator
from .report_generator import ReportGenerator
from .visualizer import Visualizer

__all__ = [
    'TrainingAnalyzer',
    'TestEvaluator',
    'ReportGenerator',
    'Visualizer',
]

__version__ = '1.0.0'
