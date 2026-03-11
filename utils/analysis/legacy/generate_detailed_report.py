#!/usr/bin/env python3
"""
生成详细的Feishu兼容分析报告
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
COLORS = {"z=1": "#e74c3c", "z=5": "#f39c12", "z=10": "#3498db"}
MARKERS = {"z=1": "o", "z=5": "s", "z=10": "D"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_train_log(log_path):
    """解析训练日志"""
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    
    with open(log_path, 'r') as f:
        for line in f:
            if "Train Loss translation_loss:" in line:
                match = re.search(r'Epoch \[(\d+)/400\], Train Loss translation_loss: ([\d.]+)', line)
                if match:
                    epoch, trans = int(match.group(1)), float(match.group(2))
                    train_data['epoch'].append(epoch)
                    train_data['trans'].append(trans)
            
            if "Train Loss rotation_loss:" in line:
                match = re.search(r'Train Loss rotation_loss: ([\d.]+)', line)
                if match:
                    train_data['rot'].append(float(match.group(1)))
            
            if "Train Pose Error" in line:
                trans_match = re.search(r'Trans: ([\d.]+)m \(Fwd:([\d.]+) Lat:([\d.]+) Ht:([\d.]+)\)', line)
                rot_match = re.search(r'Rot: ([\d.]+)° \((?:R|Roll):([\d.]+) (?:P|Pitch):([\d.]+) (?:Y|Yaw):([\d.]+)', line)
                
                if trans_match:
                    train_data['trans_error'].append(float(trans_match.group(1)))
                    train_data['fwd'].append(float(trans_match.group(2)))
                    train_data['lat'].append(float(trans_match.group(3)))
                    train_data['ht'].append(float(trans_match.group(4)))
                
                if rot_match:
                    train_data['rot_error'].append(float(rot_match.group(1)))
                    train_data['roll'].append(float(rot_match.group(2)))
                    train_data['pitch'].append(float(rot_match.group(3)))
                    train_data['yaw'].append(float(rot_match.group(4)))
    
    return dict(train_data), dict(val_data)


def plot_lateral_focus(all_data, output_path):
    """重点关注Lateral(Y)误差的收敛"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    for model_name, data in all_data.items():
        train_data = data['train']
        if 'lat' in train_data and train_data['lat']:
            epochs = train_data['epoch']
            ax.plot(epochs, train_data['lat'], 
                   label=f'{model_name} (final: {train_data["lat"][-1]:.4f}m)', 
                   color=COLORS[model_name], 
                   marker=MARKERS[model_name], markersize=4, markevery=20, linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Lateral Error (m)', fontsize=14, fontweight='bold')
    ax.set_title('Lateral (Y-axis) Error Convergence - Key Differentiator', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 添加标注
    ax.text(0.02, 0.98, 'Lower is better\nLateral error is the main\nbenefit of higher Z resolution', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved lateral focus plot: {output_path}")
    plt.close()


def plot_performance_comparison(all_data, output_path):
    """绘制最终性能对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Final Performance Comparison (Epoch 400)', fontsize=16, fontweight='bold')
    
    models = list(all_data.keys())
    x = np.arange(len(models))
    width = 0.25
    
    # 平移误差总量和分量
    ax = axes[0]
    trans_total = [all_data[m]['train']['trans_error'][-1] if all_data[m]['train']['trans_error'] else 0 for m in models]
    fwd_vals = [all_data[m]['train']['fwd'][-1] if all_data[m]['train']['fwd'] else 0 for m in models]
    lat_vals = [all_data[m]['train']['lat'][-1] if all_data[m]['train']['lat'] else 0 for m in models]
    ht_vals = [all_data[m]['train']['ht'][-1] if all_data[m]['train']['ht'] else 0 for m in models]
    
    # 总误差
    bars1 = ax.bar(x - width*1.5, trans_total, width, label='Total Trans', 
                   color='#34495e', alpha=0.9, edgecolor='black', linewidth=1.5)
    # 分量
    bars2 = ax.bar(x - width*0.5, fwd_vals, width, label='Forward (X)', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width*0.5, lat_vals, width, label='Lateral (Y)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax.bar(x + width*1.5, ht_vals, width, label='Height (Z)', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (m)', fontsize=12, fontweight='bold')
    ax.set_title('Translation Errors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 旋转误差
    ax = axes[1]
    rot_total = [all_data[m]['train']['rot_error'][-1] if all_data[m]['train']['rot_error'] else 0 for m in models]
    roll_vals = [all_data[m]['train']['roll'][-1] if all_data[m]['train']['roll'] else 0 for m in models]
    pitch_vals = [all_data[m]['train']['pitch'][-1] if all_data[m]['train']['pitch'] else 0 for m in models]
    yaw_vals = [all_data[m]['train']['yaw'][-1] if all_data[m]['train']['yaw'] else 0 for m in models]
    
    bars1 = ax.bar(x - width*1.5, rot_total, width, label='Total Rot', 
                   color='#34495e', alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x - width*0.5, roll_vals, width, label='Roll', 
                   color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width*0.5, pitch_vals, width, label='Pitch', 
                   color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax.bar(x + width*1.5, yaw_vals, width, label='Yaw', 
                   color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=1)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (°)', fontsize=12, fontweight='bold')
    ax.set_title('Rotation Errors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved performance comparison: {output_path}")
    plt.close()


def generate_feishu_report(all_data, output_path):
    """生成Feishu兼容的Markdown报告"""
    lines = []
    
    lines.append("BEVCalib Z-Ablation 实验分析报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append("实验配置: z=1, z=5, z=10 (BEV高度体素分辨率消融实验)")
    lines.append("")
    lines.append("一、训练性能总结")
    lines.append("")
    lines.append("1.1 最终训练精度 (Epoch 400)")
    lines.append("")
    lines.append("| 模型 | Trans(m) | Fwd(m) | Lat(m) | Ht(m) | Rot(°) | Roll(°) | Pitch(°) | Yaw(°) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    
    for model_name in ["z=1", "z=5", "z=10"]:
        data = all_data[model_name]['train']
        if data['trans_error']:
            lines.append(
                f"| {model_name} | {data['trans_error'][-1]:.4f} | {data['fwd'][-1]:.4f} | "
                f"{data['lat'][-1]:.4f} | {data['ht'][-1]:.4f} | {data['rot_error'][-1]:.2f} | "
                f"{data['roll'][-1]:.2f} | {data['pitch'][-1]:.2f} | {data['yaw'][-1]:.2f} |"
            )
    
    lines.append("")
    lines.append("1.2 关键发现")
    lines.append("")
    
    # 计算改进百分比
    lat_z1 = all_data["z=1"]['train']['lat'][-1]
    lat_z5 = all_data["z=5"]['train']['lat'][-1]
    lat_z10 = all_data["z=10"]['train']['lat'][-1]
    
    improv_5_vs_1 = (lat_z1 - lat_z5) / lat_z1 * 100
    improv_10_vs_1 = (lat_z1 - lat_z10) / lat_z1 * 100
    improv_10_vs_5 = (lat_z5 - lat_z10) / lat_z5 * 100
    
    lines.append(f"Lateral(Y)误差对比:")
    lines.append(f"  • z=1: {lat_z1:.4f}m (基线)")
    lines.append(f"  • z=5: {lat_z5:.4f}m (相比z=1改进 {improv_5_vs_1:.1f}%)")
    lines.append(f"  • z=10: {lat_z10:.4f}m (相比z=1改进 {improv_10_vs_1:.1f}%, 相比z=5改进 {improv_10_vs_5:.1f}%)")
    lines.append("")
    lines.append("核心结论: 增加BEV高度体素数量(Z分辨率)主要改善Lateral(Y轴)标定精度")
    lines.append("原因: 更多高度层提供了更丰富的视差线索,帮助模型更准确地推断横向偏移")
    lines.append("")
    
    lines.append("1.3 收敛速度对比")
    lines.append("")
    
    # 检查特定epoch的误差
    for epoch_idx in [19, 39, 79, 119, 199, 399]:  # epoch 20, 40, 80, 120, 200, 400
        if epoch_idx < len(all_data["z=1"]['train']['trans_error']):
            lines.append(f"Epoch {epoch_idx+1}:")
            for model_name in ["z=1", "z=5", "z=10"]:
                trans_err = all_data[model_name]['train']['trans_error'][epoch_idx]
                lat_err = all_data[model_name]['train']['lat'][epoch_idx]
                lines.append(f"  {model_name}: Trans={trans_err:.4f}m, Lat={lat_err:.4f}m")
    
    lines.append("")
    lines.append("二、可视化图表说明")
    lines.append("")
    lines.append("![收敛曲线](z_comparison_convergence.png)")
    lines.append("")
    lines.append("图1: 训练和验证误差收敛曲线")
    lines.append("")
    lines.append("![Lateral误差对比](z_comparison_lateral_focus.png)")
    lines.append("")
    lines.append("图2: Lateral(Y)误差收敛曲线 - Z分辨率的关键影响")
    lines.append("")
    lines.append("![性能对比](z_comparison_performance.png)")
    lines.append("")
    lines.append("图3: 最终性能对比 (Epoch 400)")
    lines.append("")
    
    lines.append("三、泛化能力评估")
    lines.append("")
    lines.append("测试数据集: test_data (3个sequences, 约1446帧)")
    lines.append("")
    lines.append("评估中... 结果将在评估完成后更新")
    lines.append("")
    
    report_text = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Saved Feishu report: {output_path}")
    return report_text


def main():
    """主函数"""
    print("=" * 80)
    print("Generating Detailed Analysis Report")
    print("=" * 80)
    print()
    
    all_data = {}
    
    # 解析训练日志
    for model_name, model_dir in MODELS.items():
        log_path = os.path.join(BASE_DIR, model_dir, "train.log")
        print(f"Parsing {model_name}: {log_path}")
        
        if not os.path.exists(log_path):
            print(f"  ⚠ Warning: Log file not found")
            continue
        
        train_data, val_data = parse_train_log(log_path)
        all_data[model_name] = {'train': train_data, 'val': val_data}
        
        if train_data['epoch']:
            print(f"  ✓ Extracted {len(train_data['epoch'])} training epochs")
    
    print()
    
    # 生成可视化
    if all_data:
        print("Generating visualizations...")
        plot_lateral_focus(all_data, os.path.join(OUTPUT_DIR, "z_comparison_lateral_focus.png"))
        plot_performance_comparison(all_data, os.path.join(OUTPUT_DIR, "z_comparison_performance.png"))
        
        # 生成报告
        report_text = generate_feishu_report(all_data, os.path.join(OUTPUT_DIR, "detailed_report.md"))
        print()
        print("=" * 80)
        print("Report Preview:")
        print("=" * 80)
        print(report_text[:1500])
        print("...")
        print("=" * 80)
    
    print()
    print(f"✓ Analysis complete! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
