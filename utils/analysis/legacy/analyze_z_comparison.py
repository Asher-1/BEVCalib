#!/usr/bin/env python3
"""
分析并可视化 z=1, z=5, z=10 三组模型的训练性能和泛化能力
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

# 配置
BASE_DIR = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/B26A"
MODELS = {
    "z=1": "model_small_5deg_v4-z1",
    "z=5": "model_small_5deg_v4-z5",
    "z=10": "model_small_5deg_v4-z10",
}
OUTPUT_DIR = "/mnt/drtraining/user/dahailu/code/BEVCalib/analysis_results"

# 颜色和标记配置
COLORS = {"z=1": "#e74c3c", "z=5": "#f39c12", "z=10": "#3498db"}
MARKERS = {"z=1": "o", "z=5": "s", "z=10": "D"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_train_log(log_path):
    """解析训练日志，提取每个epoch的训练和验证指标"""
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    
    with open(log_path, 'r') as f:
        for line in f:
            # 训练Loss: Epoch [N/400], Train Loss translation_loss: X
            if "Train Loss translation_loss:" in line:
                match = re.search(r'Epoch \[(\d+)/400\], Train Loss translation_loss: ([\d.]+)', line)
                if match:
                    epoch, trans = int(match.group(1)), float(match.group(2))
                    train_data['epoch'].append(epoch)
                    train_data['trans'].append(trans)
            
            # 训练Loss: rotation
            if "Train Loss rotation_loss:" in line:
                match = re.search(r'Train Loss rotation_loss: ([\d.]+)', line)
                if match:
                    train_data['rot'].append(float(match.group(1)))
            
            # 训练误差: Train Pose Error - Trans: 0.0562m (Fwd:0.0202 Lat:0.0431 Ht:0.0161), Rot: 0.15° (R:0.07 P:0.07 Y:0.08)
            if "Train Pose Error" in line:
                trans_match = re.search(r'Trans: ([\d.]+)m \(Fwd:([\d.]+) Lat:([\d.]+) Ht:([\d.]+)\)', line)
                if trans_match:
                    train_data['trans_error'].append(float(trans_match.group(1)))
                    train_data['fwd'].append(float(trans_match.group(2)))
                    train_data['lat'].append(float(trans_match.group(3)))
                    train_data['ht'].append(float(trans_match.group(4)))
                
                rot_match = re.search(r'Rot: ([\d.]+)° \(R[ol]*:([\d.]+) P[itch]*:([\d.]+) Y[aw]*:([\d.]+)', line)
                if rot_match:
                    train_data['rot_error'].append(float(rot_match.group(1)))
                    train_data['roll'].append(float(rot_match.group(2)))
                    train_data['pitch'].append(float(rot_match.group(3)))
                    train_data['yaw'].append(float(rot_match.group(4)))
            
            # 验证误差: Val Pose Error (注意旧日志可能没有Val标签)
            if "Val Pose Error" in line or ("5.0_0.3 Validation" in line and "Pose Error" in line):
                trans_match = re.search(r'Trans: ([\d.]+)m \(Fwd:([\d.]+) Lat:([\d.]+) Ht:([\d.]+)\)', line)
                if trans_match:
                    val_data['trans_error'].append(float(trans_match.group(1)))
                    val_data['fwd'].append(float(trans_match.group(2)))
                    val_data['lat'].append(float(trans_match.group(3)))
                    val_data['ht'].append(float(trans_match.group(4)))
                
                rot_match = re.search(r'Rot: ([\d.]+)° \(R[ol]*:([\d.]+) P[itch]*:([\d.]+) Y[aw]*:([\d.]+)', line)
                if rot_match:
                    val_data['rot_error'].append(float(rot_match.group(1)))
                    val_data['roll'].append(float(rot_match.group(2)))
                    val_data['pitch'].append(float(rot_match.group(3)))
                    val_data['yaw'].append(float(rot_match.group(4)))
    
    return dict(train_data), dict(val_data)


def plot_convergence_curves(all_data, output_path):
    """绘制收敛曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Convergence Comparison (z=1 vs z=5 vs z=10)', fontsize=16, fontweight='bold')
    
    # 训练平移误差
    ax = axes[0, 0]
    for model_name, data in all_data.items():
        train_data = data['train']
        if 'trans_error' in train_data and train_data['trans_error']:
            epochs = train_data['epoch']
            ax.plot(epochs, train_data['trans_error'], 
                   label=model_name, color=COLORS[model_name], 
                   marker=MARKERS[model_name], markersize=4, markevery=20, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Translation Error (m)', fontsize=12)
    ax.set_title('Train Translation Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 训练旋转误差
    ax = axes[0, 1]
    for model_name, data in all_data.items():
        train_data = data['train']
        if 'rot_error' in train_data and train_data['rot_error']:
            epochs = train_data['epoch']
            ax.plot(epochs, train_data['rot_error'], 
                   label=model_name, color=COLORS[model_name], 
                   marker=MARKERS[model_name], markersize=4, markevery=20, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rotation Error (°)', fontsize=12)
    ax.set_title('Train Rotation Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 验证平移误差
    ax = axes[1, 0]
    for model_name, data in all_data.items():
        val_data = data['val']
        if 'trans_error' in val_data and val_data['trans_error']:
            # 验证数据可能不是每个epoch都有
            val_epochs = list(range(len(val_data['trans_error'])))
            ax.plot(val_epochs, val_data['trans_error'], 
                   label=model_name, color=COLORS[model_name], 
                   marker=MARKERS[model_name], markersize=5, linewidth=2)
    ax.set_xlabel('Validation Steps', fontsize=12)
    ax.set_ylabel('Translation Error (m)', fontsize=12)
    ax.set_title('Validation Translation Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 验证旋转误差
    ax = axes[1, 1]
    for model_name, data in all_data.items():
        val_data = data['val']
        if 'rot_error' in val_data and val_data['rot_error']:
            val_epochs = list(range(len(val_data['rot_error'])))
            ax.plot(val_epochs, val_data['rot_error'], 
                   label=model_name, color=COLORS[model_name], 
                   marker=MARKERS[model_name], markersize=5, linewidth=2)
    ax.set_xlabel('Validation Steps', fontsize=12)
    ax.set_ylabel('Rotation Error (°)', fontsize=12)
    ax.set_title('Validation Rotation Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved convergence curves: {output_path}")
    plt.close()


def plot_component_breakdown(all_data, output_path):
    """绘制误差分量对比（最终epoch）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Error Component Breakdown at Epoch 400', fontsize=16, fontweight='bold')
    
    models = list(all_data.keys())
    x = np.arange(len(models))
    width = 0.2
    
    # 平移误差分量
    ax = axes[0]
    fwd_vals = [all_data[m]['train']['fwd'][-1] if all_data[m]['train']['fwd'] else 0 for m in models]
    lat_vals = [all_data[m]['train']['lat'][-1] if all_data[m]['train']['lat'] else 0 for m in models]
    ht_vals = [all_data[m]['train']['ht'][-1] if all_data[m]['train']['ht'] else 0 for m in models]
    
    ax.bar(x - width, fwd_vals, width, label='Forward (X)', color='#3498db', alpha=0.8)
    ax.bar(x, lat_vals, width, label='Lateral (Y)', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, ht_vals, width, label='Height (Z)', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('Translation Error Components', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 旋转误差分量
    ax = axes[1]
    roll_vals = [all_data[m]['train']['roll'][-1] if all_data[m]['train']['roll'] else 0 for m in models]
    pitch_vals = [all_data[m]['train']['pitch'][-1] if all_data[m]['train']['pitch'] else 0 for m in models]
    yaw_vals = [all_data[m]['train']['yaw'][-1] if all_data[m]['train']['yaw'] else 0 for m in models]
    
    ax.bar(x - width, roll_vals, width, label='Roll', color='#9b59b6', alpha=0.8)
    ax.bar(x, pitch_vals, width, label='Pitch', color='#f39c12', alpha=0.8)
    ax.bar(x + width, yaw_vals, width, label='Yaw', color='#1abc9c', alpha=0.8)
    
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel('Error (°)', fontsize=12)
    ax.set_title('Rotation Error Components', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved component breakdown: {output_path}")
    plt.close()


def generate_summary_table(all_data):
    """生成Feishu兼容的总结表格"""
    summary = []
    summary.append("一、训练性能总结 (Epoch 400)\n")
    summary.append("| 模型 | Trans(m) | Fwd(m) | Lat(m) | Ht(m) | Rot(°) | Roll(°) | Pitch(°) | Yaw(°) |")
    summary.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    
    for model_name in ["z=1", "z=5", "z=10"]:
        data = all_data[model_name]['train']
        if data['trans_error']:
            summary.append(
                f"| {model_name} | {data['trans_error'][-1]:.4f} | {data['fwd'][-1]:.4f} | "
                f"{data['lat'][-1]:.4f} | {data['ht'][-1]:.4f} | {data['rot_error'][-1]:.2f} | "
                f"{data['roll'][-1]:.2f} | {data['pitch'][-1]:.2f} | {data['yaw'][-1]:.2f} |"
            )
    
    return "\n".join(summary)


def main():
    """主函数"""
    print("=" * 80)
    print("BEVCalib Z-Ablation Analysis: z=1 vs z=5 vs z=10")
    print("=" * 80)
    print()
    
    all_data = {}
    
    # 解析所有训练日志
    for model_name, model_dir in MODELS.items():
        log_path = os.path.join(BASE_DIR, model_dir, "train.log")
        print(f"Parsing {model_name}: {log_path}")
        
        if not os.path.exists(log_path):
            print(f"  ⚠ Warning: Log file not found, skipping...")
            continue
        
        train_data, val_data = parse_train_log(log_path)
        all_data[model_name] = {'train': train_data, 'val': val_data}
        
        if train_data['epoch']:
            print(f"  ✓ Extracted {len(train_data['epoch'])} training epochs")
            print(f"    Final Train Trans: {train_data['trans_error'][-1]:.4f}m, Rot: {train_data['rot_error'][-1]:.2f}°")
            print(f"    Final components - Fwd:{train_data['fwd'][-1]:.4f}m, Lat:{train_data['lat'][-1]:.4f}m, Ht:{train_data['ht'][-1]:.4f}m")
    
    print()
    
    # 生成可视化
    if all_data:
        print("Generating visualizations...")
        plot_convergence_curves(all_data, os.path.join(OUTPUT_DIR, "z_comparison_convergence.png"))
        plot_component_breakdown(all_data, os.path.join(OUTPUT_DIR, "z_comparison_components.png"))
        
        # 生成表格总结
        summary_text = generate_summary_table(all_data)
        summary_path = os.path.join(OUTPUT_DIR, "training_summary.md")
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"✓ Saved summary table: {summary_path}")
        print()
        print(summary_text)
    
    print()
    print("=" * 80)
    print("✓ Training analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
