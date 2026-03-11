#!/usr/bin/env python3
"""
测试数据评估模块
运行模型在测试数据上的评估并解析结果
"""
import os
import re
import subprocess
from typing import Dict, Optional, List


class TestEvaluator:
    """测试数据评估器"""
    
    def __init__(self, bevcalib_root: str, test_data_root: str):
        """
        初始化评估器
        
        Args:
            bevcalib_root: BEVCalib项目根目录
            test_data_root: 测试数据根目录
        """
        self.bevcalib_root = bevcalib_root
        self.test_data_root = test_data_root
    
    def evaluate_checkpoint(
        self,
        ckpt_path: str,
        output_dir: str,
        zbound_step: float,
        angle_range: float = 5.0,
        trans_range: float = 0.3,
        batch_size: int = 8,
        rotation_only: bool = True,
        vis_interval: int = 100,
        use_conda: bool = True,
        conda_env: str = "bevcalib"
    ) -> bool:
        """
        评估单个checkpoint
        
        Args:
            ckpt_path: checkpoint文件路径
            output_dir: 输出目录
            zbound_step: BEV Z方向步长
            angle_range: 角度扰动范围
            trans_range: 平移扰动范围
            batch_size: 批大小
            rotation_only: 是否仅评估旋转
            vis_interval: 可视化间隔
            use_conda: 是否使用conda环境
            conda_env: conda环境名称
        
        Returns:
            是否成功
        """
        if not os.path.exists(ckpt_path):
            print(f"✗ Checkpoint not found: {ckpt_path}")
            return False
        
        eval_script = os.path.join(self.bevcalib_root, "evaluate_checkpoint.py")
        if not os.path.exists(eval_script):
            print(f"✗ Evaluation script not found: {eval_script}")
            return False
        
        # 构建评估命令
        cmd = [
            "python", eval_script,
            "--ckpt_path", ckpt_path,
            "--dataset_root", self.test_data_root,
            "--output_dir", output_dir,
            "--angle_range_deg", str(angle_range),
            "--trans_range", str(trans_range),
            "--batch_size", str(batch_size),
            "--vis_interval", str(vis_interval),
            "--use_full_dataset",
            "--max_batches", "0",
            "--rotation_only", "1" if rotation_only else "0",
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        env["BEV_ZBOUND_STEP"] = str(zbound_step)
        
        # 使用conda环境
        if use_conda:
            cmd = ["conda", "run", "-n", conda_env] + cmd
        
        print(f"Running evaluation...")
        print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
        print(f"  BEV_ZBOUND_STEP: {zbound_step}")
        print(f"  Output: {output_dir}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.bevcalib_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"  ✓ Evaluation complete")
                return True
            else:
                print(f"  ✗ Evaluation failed")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}")
                return False
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ Evaluation timed out")
            return False
        except Exception as e:
            print(f"  ✗ Evaluation error: {e}")
            return False
    
    def parse_evaluation_results(self, output_dir: str) -> Optional[Dict]:
        """
        解析评估结果
        
        Args:
            output_dir: 评估输出目录
        
        Returns:
            评估结果字典，失败则返回None
        """
        errors_file = os.path.join(output_dir, "extrinsics_and_errors.txt")
        
        if not os.path.exists(errors_file):
            return None
        
        with open(errors_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 提取平均误差
        avg_match = re.search(
            r'Average Rotation Errors.*?Total:\s+([\d.]+)\s+±\s+([\d.]+)\s+deg',
            content,
            re.DOTALL
        )
        
        if not avg_match:
            return None
        
        results = {
            "total": {
                "mean": float(avg_match.group(1)),
                "std": float(avg_match.group(2))
            }
        }
        
        # 提取分量误差
        components = [
            ('roll', r'Roll \(X\):\s+([\d.]+)\s+±\s+([\d.]+)'),
            ('pitch', r'Pitch \(Y\):\s+([\d.]+)\s+±\s+([\d.]+)'),
            ('yaw', r'Yaw \(Z\):\s+([\d.]+)\s+±\s+([\d.]+)')
        ]
        
        for comp_name, pattern in components:
            match = re.search(pattern, content)
            if match:
                results[comp_name] = {
                    "mean": float(match.group(1)),
                    "std": float(match.group(2))
                }
        
        # 提取统计信息
        stats_match = re.search(
            r'Total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            content
        )
        if stats_match:
            results["stats"] = {
                "mean": float(stats_match.group(1)),
                "std": float(stats_match.group(2)),
                "min": float(stats_match.group(3)),
                "median": float(stats_match.group(4)),
                "p90": float(stats_match.group(5)),
                "p95": float(stats_match.group(6)),
                "p99": float(stats_match.group(7)),
                "max": float(stats_match.group(8))
            }
        
        return results


def batch_evaluate_experiments(
    evaluator: TestEvaluator,
    experiments: Dict[str, Dict],
    parallel: bool = False
) -> Dict[str, Optional[Dict]]:
    """
    批量评估多个实验
    
    Args:
        evaluator: TestEvaluator实例
        experiments: {实验名称: 配置字典} 的字典
            配置字典应包含: ckpt_path, output_dir, zbound_step等
        parallel: 是否并行评估
    
    Returns:
        {实验名称: 评估结果} 的字典
    """
    results = {}
    
    if parallel:
        # 并行评估 (可以使用ProcessPoolExecutor)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        def eval_single(name, config):
            success = evaluator.evaluate_checkpoint(**config)
            if success:
                return name, evaluator.parse_evaluation_results(config['output_dir'])
            return name, None
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(eval_single, name, config): name 
                for name, config in experiments.items()
            }
            
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
    else:
        # 串行评估
        for name, config in experiments.items():
            print(f"\n{'='*80}")
            print(f"Evaluating: {name}")
            print(f"{'='*80}\n")
            
            success = evaluator.evaluate_checkpoint(**config)
            if success:
                results[name] = evaluator.parse_evaluation_results(config['output_dir'])
            else:
                results[name] = None
    
    return results
