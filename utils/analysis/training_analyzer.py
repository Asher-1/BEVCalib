#!/usr/bin/env python3
"""
训练日志分析模块
解析训练日志，提取性能指标
"""
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional


class TrainingAnalyzer:
    """训练日志分析器"""
    
    def __init__(self, log_path: str):
        """
        初始化分析器
        
        Args:
            log_path: 训练日志文件路径
        """
        self.log_path = log_path
        self.train_data = defaultdict(list)
        self.val_data = defaultdict(list)
        
    def parse_log(self) -> Dict:
        """
        解析训练日志
        
        Returns:
            包含train和val数据的字典
        """
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
        
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self._parse_train_line(line)
                self._parse_val_line(line)
        
        return {
            'train': dict(self.train_data),
            'val': dict(self.val_data)
        }
    
    def _parse_train_line(self, line: str):
        """解析训练误差行"""
        # 提取Epoch（从任何Train Loss行）
        if "Train Loss" in line and "Epoch [" in line and 'epoch' not in self.train_data or len(self.train_data.get('epoch', [])) < 400:
            epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                # 避免重复添加同一个epoch
                if not self.train_data['epoch'] or epoch != self.train_data['epoch'][-1]:
                    # Translation Loss (可能不存在，rotation-only模式)
                    if "Train Loss translation_loss:" in line:
                        trans_match = re.search(r'Train Loss translation_loss: ([\d.]+)', line)
                        if trans_match:
                            self.train_data['epoch'].append(epoch)
                            self.train_data['trans_loss'].append(float(trans_match.group(1)))
                    # 或者从total_loss提取epoch（rotation-only模式）
                    elif "Train Loss total_loss:" in line and (not self.train_data['epoch'] or epoch != self.train_data['epoch'][-1]):
                        self.train_data['epoch'].append(epoch)
        
        # Rotation Loss
        if "Train Loss rotation_loss:" in line:
            match = re.search(r'Train Loss rotation_loss: ([\d.]+)', line)
            if match:
                self.train_data['rot_loss'].append(float(match.group(1)))
        
        # Pose Error
        if "Train Pose Error" in line:
            # Translation components
            trans_match = re.search(
                r'Trans: ([\d.]+)m \((?:Fwd|Forward):([\d.]+)m? (?:Lat|Lateral):([\d.]+)m? (?:Ht|Height):([\d.]+)m?\)',
                line
            )
            if trans_match:
                self.train_data['trans_error'].append(float(trans_match.group(1)))
                self.train_data['fwd'].append(float(trans_match.group(2)))
                self.train_data['lat'].append(float(trans_match.group(3)))
                self.train_data['ht'].append(float(trans_match.group(4)))
            
            # Rotation components
            rot_match = re.search(
                r'Rot: ([\d.]+)° \((?:R|Roll):([\d.]+)°? (?:P|Pitch):([\d.]+)°? (?:Y|Yaw):([\d.]+)°?\)',
                line
            )
            if rot_match:
                self.train_data['rot_error'].append(float(rot_match.group(1)))
                self.train_data['roll'].append(float(rot_match.group(2)))
                self.train_data['pitch'].append(float(rot_match.group(3)))
                self.train_data['yaw'].append(float(rot_match.group(4)))
    
    def _parse_val_line(self, line: str):
        """解析验证误差行"""
        if "Val" not in line and "Validation" not in line:
            return
        
        if "Pose Error" in line:
            # Translation components
            trans_match = re.search(
                r'Trans: ([\d.]+)m \((?:Fwd|Forward):([\d.]+)m? (?:Lat|Lateral):([\d.]+)m? (?:Ht|Height):([\d.]+)m?\)',
                line
            )
            if trans_match:
                self.val_data['trans_error'].append(float(trans_match.group(1)))
                self.val_data['fwd'].append(float(trans_match.group(2)))
                self.val_data['lat'].append(float(trans_match.group(3)))
                self.val_data['ht'].append(float(trans_match.group(4)))
            
            # Rotation components
            rot_match = re.search(
                r'Rot: ([\d.]+)° \((?:R|Roll):([\d.]+)°? (?:P|Pitch):([\d.]+)°? (?:Y|Yaw):([\d.]+)°?\)',
                line
            )
            if rot_match:
                self.val_data['rot_error'].append(float(rot_match.group(1)))
                self.val_data['roll'].append(float(rot_match.group(2)))
                self.val_data['pitch'].append(float(rot_match.group(3)))
                self.val_data['yaw'].append(float(rot_match.group(4)))
    
    def get_final_metrics(self) -> Optional[Dict]:
        """
        获取最终epoch的指标
        
        Returns:
            最终指标字典，如果没有数据则返回None
        """
        if not self.train_data or 'epoch' not in self.train_data or not self.train_data['epoch']:
            return None
        
        metrics = {}
        for key in ['trans_error', 'fwd', 'lat', 'ht', 'rot_error', 'roll', 'pitch', 'yaw']:
            if key in self.train_data and self.train_data[key]:
                metrics[key] = self.train_data[key][-1]
            else:
                # 对于rotation-only模式，平移误差可能不存在
                metrics[key] = 0.0 if key in ['trans_error', 'fwd', 'lat', 'ht'] else None
        
        # 至少要有旋转误差
        if metrics.get('rot_error') is None:
            return None
        
        return metrics
    
    def get_convergence_data(self, milestones: Optional[List[int]] = None) -> Dict:
        """
        获取关键里程碑的收敛数据
        
        Args:
            milestones: 关键epoch列表，默认为[20, 40, 80, 120, 200, 400]
        
        Returns:
            里程碑数据字典
        """
        if milestones is None:
            milestones = [20, 40, 80, 120, 200, 400]
        
        convergence = {}
        epochs = self.train_data.get('epoch', [])
        
        for milestone in milestones:
            if milestone in epochs:
                idx = epochs.index(milestone)
                convergence[milestone] = {
                    'trans': self.train_data.get('trans_error', [None])[idx] if idx < len(self.train_data.get('trans_error', [])) else None,
                    'lat': self.train_data.get('lat', [None])[idx] if idx < len(self.train_data.get('lat', [])) else None,
                    'rot': self.train_data.get('rot_error', [None])[idx] if idx < len(self.train_data.get('rot_error', [])) else None,
                }
        
        return convergence


def analyze_multiple_experiments(log_paths: Dict[str, str]) -> Dict:
    """
    批量分析多个实验
    
    Args:
        log_paths: {实验名称: 日志路径} 的字典
    
    Returns:
        {实验名称: 分析结果} 的字典
    """
    results = {}
    
    for name, path in log_paths.items():
        print(f"Analyzing {name}: {path}")
        try:
            analyzer = TrainingAnalyzer(path)
            data = analyzer.parse_log()
            results[name] = {
                'data': data,
                'final_metrics': analyzer.get_final_metrics(),
                'convergence': analyzer.get_convergence_data()
            }
            
            if data['train']['epoch']:
                print(f"  ✓ Extracted {len(data['train']['epoch'])} training epochs")
                final = analyzer.get_final_metrics()
                if final:
                    print(f"    Final: Trans={final.get('trans_error', 0):.4f}m, Rot={final.get('rot_error', 0):.2f}°")
            else:
                print(f"  ⚠ No training data found")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[name] = None
    
    return results
