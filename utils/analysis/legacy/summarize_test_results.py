#!/usr/bin/env python3
"""
汇总测试数据评估结果并生成对比可视化
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BASE_DIR = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/B26A"
OUTPUT_DIR = "/mnt/drtraining/user/dahailu/code/BEVCalib/analysis_results"

MODELS = {
    "z=1": "model_small_5deg_v4-z1",
    "z=5": "model_small_5deg_v4-z5",
    "z=10": "model_small_5deg_v4-z10",
}

TRAIN_FINAL = {
    "z=1": {"rot": 0.15},
    "z=5": {"rot": 0.13},
    "z=10": {"rot": 0.11},
}

def parse_test_results(eval_dir):
    """解析测试评估结果"""
    errors_file = os.path.join(eval_dir, "extrinsics_and_errors.txt")
    
    if not os.path.exists(errors_file):
        return None
    
    with open(errors_file, 'r') as f:
        content = f.read()
    
    # 提取平均误差
    avg_match = re.search(r'Average Rotation Errors.*?Total:\s+([\d.]+)\s+±\s+([\d.]+)\s+deg', content, re.DOTALL)
    if not avg_match:
        return None
    
    total_mean = float(avg_match.group(1))
    total_std = float(avg_match.group(2))
    
    # 提取分量误差
    roll_match = re.search(r'Roll \(X\):\s+([\d.]+)\s+±\s+([\d.]+)', content)
    pitch_match = re.search(r'Pitch \(Y\):\s+([\d.]+)\s+±\s+([\d.]+)', content)
    yaw_match = re.search(r'Yaw \(Z\):\s+([\d.]+)\s+±\s+([\d.]+)', content)
    
    return {
        "total": {"mean": total_mean, "std": total_std},
        "roll": {"mean": float(roll_match.group(1)), "std": float(roll_match.group(2))} if roll_match else None,
        "pitch": {"mean": float(pitch_match.group(1)), "std": float(pitch_match.group(2))} if pitch_match else None,
        "yaw": {"mean": float(yaw_match.group(1)), "std": float(yaw_match.group(2))} if yaw_match else None,
    }


def plot_generalization_comparison(test_results, output_path):
    """绘制泛化能力对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Generalization Performance: Training vs Test Data', fontsize=16, fontweight='bold')
    
    models = []
    train_vals = []
    test_vals = []
    test_stds = []
    degradation = []
    
    for model_name in ["z=1", "z=5", "z=10"]:
        if model_name in test_results and test_results[model_name] is not None:
            models.append(model_name)
            train_vals.append(TRAIN_FINAL[model_name]["rot"])
            test_vals.append(test_results[model_name]["total"]["mean"])
            test_stds.append(test_results[model_name]["total"]["std"])
            deg = test_results[model_name]["total"]["mean"] / TRAIN_FINAL[model_name]["rot"]
            degradation.append(deg)
    
    if not models:
        print("⚠ No test results available yet")
        return
    
    x = np.arange(len(models))
    width = 0.35
    
    # 左图: 训练 vs 测试误差
    ax = axes[0]
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train (B26A)', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test (test_data)', 
                   yerr=test_stds, capsize=5,
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rotation Error (°)', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Test Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 右图: 泛化衰退倍数
    ax = axes[1]
    bars = ax.bar(x, degradation, color=['#e74c3c' if d > 7 else '#f39c12' if d > 5 else '#2ecc71' for d in degradation],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (bar, deg) in enumerate(zip(bars, degradation)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{deg:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 5x')
    ax.axhline(y=7, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 7x')
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degradation Factor (Test/Train)', fontsize=12, fontweight='bold')
    ax.set_title('Generalization Degradation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加注释
    ax.text(0.02, 0.98, 'Lower is better\n<5x: Excellent\n5-7x: Good\n>7x: Needs improvement', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved generalization comparison: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("Summarizing Test Data Evaluation Results")
    print("=" * 80)
    print()
    
    test_results = {}
    
    for model_name, model_dir in MODELS.items():
        eval_dir = os.path.join(BASE_DIR, model_dir, "test_data_eval")
        print(f"Checking {model_name}: {eval_dir}")
        
        results = parse_test_results(eval_dir)
        if results:
            test_results[model_name] = results
            print(f"  ✓ Total Rot: {results['total']['mean']:.3f}° ± {results['total']['std']:.3f}°")
            print(f"    Train Rot: {TRAIN_FINAL[model_name]['rot']:.2f}°")
            print(f"    Degradation: {results['total']['mean'] / TRAIN_FINAL[model_name]['rot']:.2f}x")
        else:
            print(f"  ⚠ No results found (evaluation not complete or failed)")
        print()
    
    if test_results:
        print("Generating generalization comparison plot...")
        plot_generalization_comparison(test_results, 
                                      os.path.join(OUTPUT_DIR, "generalization_comparison.png"))
        
        # 生成摘要表格
        print()
        print("=" * 80)
        print("Test Data Results Summary")
        print("=" * 80)
        print()
        print("| 模型 | 训练误差(°) | 测试误差(°) | 泛化衰退 | 评估 |")
        print("| --- | ---: | ---: | ---: | --- |")
        
        for model_name in ["z=1", "z=5", "z=10"]:
            train_err = TRAIN_FINAL[model_name]["rot"]
            if model_name in test_results:
                test_err = test_results[model_name]["total"]["mean"]
                test_std = test_results[model_name]["total"]["std"]
                deg = test_err / train_err
                status = "✓优秀" if deg < 5 else "✓良好" if deg < 7 else "⚠需改进"
                print(f"| {model_name} | {train_err:.2f} | {test_err:.3f} ± {test_std:.3f} | {deg:.2f}x | {status} |")
            else:
                print(f"| {model_name} | {train_err:.2f} | - | - | 评估中 |")
    
    print()
    print("=" * 80)
    print(f"✓ Summary complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
