#!/usr/bin/env python3
"""
可视化模块
生成各种对比分析图表
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Optional, Tuple


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, output_dir: str, style: str = 'default'):
        """
        初始化可视化器
        
        Args:
            output_dir: 图表输出目录
            style: 绘图风格
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图风格
        if style != 'default':
            plt.style.use(style)
        
        # 默认颜色和标记配置
        self.colors = [
            "#e74c3c", "#f39c12", "#3498db", "#2ecc71", 
            "#9b59b6", "#1abc9c", "#34495e", "#e67e22"
        ]
        self.markers = ["o", "s", "D", "^", "v", "<", ">", "p"]
    
    def plot_convergence_curves(
        self,
        experiments_data: Dict[str, Dict],
        output_filename: str = "convergence_curves.png",
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        绘制收敛曲线
        
        Args:
            experiments_data: {实验名称: 数据字典} 的字典
            output_filename: 输出文件名
            figsize: 图表大小
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Convergence Comparison', fontsize=16, fontweight='bold')
        
        # 训练平移误差
        ax = axes[0, 0]
        for idx, (name, data) in enumerate(experiments_data.items()):
            train_data = data.get('data', {}).get('train', {})
            if 'trans_error' in train_data and train_data['trans_error']:
                epochs = train_data.get('epoch', list(range(len(train_data['trans_error']))))
                ax.plot(epochs, train_data['trans_error'],
                       label=f'{name} (final: {train_data["trans_error"][-1]:.4f}m)',
                       color=self.colors[idx % len(self.colors)],
                       marker=self.markers[idx % len(self.markers)],
                       markersize=4, markevery=max(len(epochs)//20, 1), linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Translation Error (m)', fontsize=12)
        ax.set_title('Train Translation Error', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 训练旋转误差
        ax = axes[0, 1]
        for idx, (name, data) in enumerate(experiments_data.items()):
            train_data = data.get('data', {}).get('train', {})
            if 'rot_error' in train_data and train_data['rot_error']:
                epochs = train_data.get('epoch', list(range(len(train_data['rot_error']))))
                ax.plot(epochs, train_data['rot_error'],
                       label=f'{name} (final: {train_data["rot_error"][-1]:.2f}°)',
                       color=self.colors[idx % len(self.colors)],
                       marker=self.markers[idx % len(self.markers)],
                       markersize=4, markevery=max(len(epochs)//20, 1), linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Rotation Error (°)', fontsize=12)
        ax.set_title('Train Rotation Error', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Lateral误差（关键指标）
        ax = axes[1, 0]
        for idx, (name, data) in enumerate(experiments_data.items()):
            train_data = data.get('data', {}).get('train', {})
            if 'lat' in train_data and train_data['lat']:
                epochs = train_data.get('epoch', list(range(len(train_data['lat']))))
                ax.plot(epochs, train_data['lat'],
                       label=f'{name} (final: {train_data["lat"][-1]:.4f}m)',
                       color=self.colors[idx % len(self.colors)],
                       marker=self.markers[idx % len(self.markers)],
                       markersize=4, markevery=max(len(epochs)//20, 1), linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Lateral Error (m)', fontsize=12)
        ax.set_title('Lateral (Y) Error - Key Metric', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # 所有误差分量对比
        ax = axes[1, 1]
        exp_names = list(experiments_data.keys())
        x = np.arange(len(exp_names))
        width = 0.15
        
        for i, metric in enumerate(['fwd', 'lat', 'ht', 'roll', 'pitch', 'yaw']):
            values = []
            for name in exp_names:
                train_data = experiments_data[name].get('data', {}).get('train', {})
                if metric in train_data and train_data[metric]:
                    values.append(train_data[metric][-1])
                else:
                    values.append(0)
            
            offset = (i - 2.5) * width
            ax.bar(x + offset, values, width, label=metric.upper(),
                  alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('Final Error Components', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=15, ha='right')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved convergence curves: {output_path}")
        plt.close()
    
    def plot_generalization_comparison(
        self,
        experiments_data: Dict[str, Dict],
        test_results: Dict[str, Optional[Dict]],
        output_filename: str = "generalization_comparison.png",
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        绘制泛化能力对比
        
        Args:
            experiments_data: 训练数据
            test_results: 测试结果
            output_filename: 输出文件名
            figsize: 图表大小
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Generalization Performance: Training vs Test', 
                    fontsize=16, fontweight='bold')
        
        exp_names = []
        train_vals = []
        test_vals = []
        test_stds = []
        degradation = []
        
        for name in experiments_data.keys():
            if name not in test_results or test_results[name] is None:
                continue
            
            # 训练误差
            final_metrics = experiments_data[name].get('final_metrics', {})
            train_rot = final_metrics.get('rot_error', 0)
            
            # 测试误差
            test_rot = test_results[name]['total']['mean']
            test_std = test_results[name]['total']['std']
            
            exp_names.append(name)
            train_vals.append(train_rot)
            test_vals.append(test_rot)
            test_stds.append(test_std)
            degradation.append(test_rot / train_rot if train_rot > 0 else 0)
        
        if not exp_names:
            print("⚠ No test results available for plotting")
            return
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        # 左图: 训练 vs 测试误差
        ax = axes[0]
        bars1 = ax.bar(x - width/2, train_vals, width, label='Train',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, test_vals, width, label='Test',
                      yerr=test_stds, capsize=5,
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rotation Error (°)', fontsize=12, fontweight='bold')
        ax.set_title('Training vs Test Error', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 右图: 泛化衰退倍数
        ax = axes[1]
        colors = ['#2ecc71' if d < 5 else '#f39c12' if d < 7 else '#e74c3c' 
                 for d in degradation]
        bars = ax.bar(x, degradation, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        for bar, deg in zip(bars, degradation):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{deg:.2f}x', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
        
        ax.axhline(y=5, color='orange', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Threshold: 5x')
        ax.axhline(y=7, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Threshold: 7x')
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Degradation (Test/Train)', fontsize=12, fontweight='bold')
        ax.set_title('Generalization Degradation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved generalization comparison: {output_path}")
        plt.close()
    
    def plot_component_breakdown(
        self,
        experiments_data: Dict[str, Dict],
        output_filename: str = "component_breakdown.png",
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        绘制误差分量详细分解
        
        Args:
            experiments_data: 实验数据
            output_filename: 输出文件名
            figsize: 图表大小
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Error Component Breakdown', fontsize=16, fontweight='bold')
        
        exp_names = list(experiments_data.keys())
        x = np.arange(len(exp_names))
        width = 0.2
        
        # 平移误差分量
        ax = axes[0]
        trans_components = ['fwd', 'lat', 'ht']
        comp_labels = ['Forward (X)', 'Lateral (Y)', 'Height (Z)']
        comp_colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (comp, label, color) in enumerate(zip(trans_components, comp_labels, comp_colors)):
            values = []
            for name in exp_names:
                train_data = experiments_data[name].get('data', {}).get('train', {})
                values.append(train_data.get(comp, [0])[-1] if train_data.get(comp) else 0)
            
            bars = ax.bar(x + (i-1)*width, values, width, label=label,
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error (m)', fontsize=12, fontweight='bold')
        ax.set_title('Translation Components', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 旋转误差分量
        ax = axes[1]
        rot_components = ['roll', 'pitch', 'yaw']
        comp_labels = ['Roll', 'Pitch', 'Yaw']
        comp_colors = ['#9b59b6', '#f39c12', '#1abc9c']
        
        for i, (comp, label, color) in enumerate(zip(rot_components, comp_labels, comp_colors)):
            values = []
            for name in exp_names:
                train_data = experiments_data[name].get('data', {}).get('train', {})
                values.append(train_data.get(comp, [0])[-1] if train_data.get(comp) else 0)
            
            bars = ax.bar(x + (i-1)*width, values, width, label=label,
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error (°)', fontsize=12, fontweight='bold')
        ax.set_title('Rotation Components', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved component breakdown: {output_path}")
        plt.close()
