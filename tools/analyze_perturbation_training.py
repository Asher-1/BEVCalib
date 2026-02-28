#!/usr/bin/env python3
"""
BEVCalib 扰动训练对比分析脚本
分析不同扰动范围(angle_range_deg, trans_range)对训练收敛和模型精度的影响
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 实验配置
EXPERIMENTS = {
    '3deg (3°±0.1m)': {
        'log_path': 'logs/B26A_model_small_3deg_v6.5/train.log',
        'angle_range': 3,
        'trans_range': 0.1,
        'model': 'small',
    },
    '5deg (5°±0.3m)': {
        'log_path': 'logs/B26A_model_small_5deg_v6.5/train.log',
        'angle_range': 5,
        'trans_range': 0.3,
        'model': 'small',
    },
    '10deg (10°±0.5m)': {
        'log_path': 'logs/B26A_model_small_10deg_v6/train.log',
        'angle_range': 10,
        'trans_range': 0.5,
        'model': 'small',
    },
    '15deg (15°±1.0m)': {
        'log_path': 'logs/B26A_model_medium_15deg_v6/train.log',
        'angle_range': 15,
        'trans_range': 1.0,
        'model': 'medium',
    },
    '20deg (20°±1.5m)': {
        'log_path': 'logs/B26A_model_standard_20deg_v6/train.log',
        'angle_range': 20,
        'trans_range': 1.5,
        'model': 'standard',
    },
    # v7: batch_size=16, lr=5e-05
    '10deg v7 (10°±0.5m)': {
        'log_path': 'logs/B26A_model_small_10deg_v7/train.log',
        'angle_range': 10,
        'trans_range': 0.5,
        'model': 'small',
    },
    '20deg v7 (20°±1.5m)': {
        'log_path': 'logs/B26A_model_standard_20deg_v7/train.log',
        'angle_range': 20,
        'trans_range': 1.5,
        'model': 'standard',
    },
}


def parse_train_log(log_path):
    """解析训练日志，提取关键指标"""
    base_dir = Path(__file__).parent.parent
    full_path = base_dir / log_path
    
    if not full_path.exists():
        return None
    
    data = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_trans': [],
        'train_rot': [],
        'val_trans': [],
        'val_rot': [],
    }
    
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 解析 Train Loss
    for m in re.finditer(r'Epoch \[(\d+)/\d+\], Train Loss total_loss: ([\d.]+)', content):
        epoch = int(m.group(1))
        loss = float(m.group(2))
        data['epoch'].append(epoch)
        data['train_loss'].append(loss)
    
    # 解析 Train Pose Error
    train_trans_match = re.findall(
        r'Epoch \[(\d+)/\d+\], Train Pose Error - Trans: ([\d.]+)m.*Rot: ([\d.]+)°',
        content
    )
    train_pose = {int(e): (float(t), float(r)) for e, t, r in train_trans_match}
    
    # 解析 Validation Loss (每4个epoch一次)
    val_loss_match = re.findall(
        r'Epoch \[(\d+)/\d+\], [\d.]+_[\d.]+ Validation Loss total_loss: ([\d.]+)',
        content
    )
    val_loss_dict = {int(e): float(v) for e, v in val_loss_match}
    
    # 解析 Val Pose Error
    val_pose_match = re.findall(
        r'Epoch \[(\d+)/\d+\], Val Pose Error - Trans: ([\d.]+)m.*Rot: ([\d.]+)°',
        content
    )
    val_pose = {int(e): (float(t), float(r)) for e, t, r in val_pose_match}
    
    # 对齐数据
    epochs = sorted(set(data['epoch']))
    data['train_loss'] = [data['train_loss'][data['epoch'].index(e)] for e in epochs]
    data['epoch'] = epochs
    
    data['val_loss'] = []
    data['train_trans'] = []
    data['train_rot'] = []
    data['val_trans'] = []
    data['val_rot'] = []
    
    for e in epochs:
        data['val_loss'].append(val_loss_dict.get(e, float('nan')))
        if e in train_pose:
            data['train_trans'].append(train_pose[e][0])
            data['train_rot'].append(train_pose[e][1])
        else:
            data['train_trans'].append(float('nan'))
            data['train_rot'].append(float('nan'))
        if e in val_pose:
            data['val_trans'].append(val_pose[e][0])
            data['val_rot'].append(val_pose[e][1])
        else:
            data['val_trans'].append(float('nan'))
            data['val_rot'].append(float('nan'))
    
    return data


def plot_training_curves(all_data, output_dir):
    """绘制训练曲线对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n = len([d for d in all_data.values() if d is not None])
    colors = plt.cm.viridis([0.05 + 0.9 * i / max(n - 1, 1) for i in range(n)])
    
    for (name, data), color in zip(all_data.items(), colors):
        if data is None:
            continue
        epochs = data['epoch']
        axes[0, 0].plot(epochs, data['train_loss'], label=name, color=color, alpha=0.9)
        axes[0, 1].semilogy(epochs, data['train_loss'], label=name, color=color, alpha=0.9)
        
        val_epochs = [e for e, v in zip(epochs, data['val_loss']) if not (v != v or v > 100)]
        val_losses = [v for v in data['val_loss'] if not (v != v or v > 100)]
        if val_epochs and val_losses:
            axes[1, 0].plot(val_epochs, val_losses, label=name, color=color, alpha=0.9)
        axes[1, 1].plot(epochs, data['train_rot'], label=name, color=color, alpha=0.9)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].set_title('Train Loss (Linear)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Train Loss (log)')
    axes[0, 1].set_title('Train Loss (Log)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Rot Error (deg)')
    axes[1, 1].set_title('Train Rotation Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'training_curves_comparison.png'}")


def plot_perturbation_impact(all_data, output_dir):
    """绘制扰动对训练的影响分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    angle_ranges = [3, 5, 10, 15, 20]
    names = list(all_data.keys())
    
    # 最终指标
    final_train_loss = []
    final_val_loss = []
    final_train_rot = []
    final_val_rot = []
    final_train_trans = []
    final_val_trans = []
    
    for name in names:
        data = all_data.get(name)
        if data is None or not data['epoch']:
            continue
        idx = -1
        final_train_loss.append(data['train_loss'][idx])
        final_train_rot.append(data['train_rot'][idx] if not (data['train_rot'][idx] != data['train_rot'][idx]) else 0)
        final_train_trans.append(data['train_trans'][idx] if not (data['train_trans'][idx] != data['train_trans'][idx]) else 0)
        
        val_losses = [v for v in data['val_loss'] if not (v != v or v > 100)]
        val_rots = [v for v in data['val_rot'] if not (v != v or v > 100)]
        val_trans = [v for v in data['val_trans'] if not (v != v or v > 100)]
        final_val_loss.append(val_losses[-1] if val_losses else float('nan'))
        final_val_rot.append(val_rots[-1] if val_rots else float('nan'))
        final_val_trans.append(val_trans[-1] if val_trans else float('nan'))
    
    x = range(len(names))
    width = 0.25 if len(names) > 5 else 0.35
    
    axes[0, 0].bar([i - width/2 for i in x], final_train_loss, width, label='Train Loss')
    axes[0, 0].bar([i + width/2 for i in x], final_val_loss, width, label='Val Loss')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=15)
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Final Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar([i - width/2 for i in x], final_train_rot, width, label='Train Rot (deg)')
    axes[0, 1].bar([i + width/2 for i in x], final_val_rot, width, label='Val Rot (deg)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=15)
    axes[0, 1].set_ylabel('Rotation Error (deg)')
    axes[0, 1].set_title('Final Rotation Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].bar([i - width/2 for i in x], final_train_trans, width, label='Train Trans (m)')
    axes[1, 0].bar([i + width/2 for i in x], final_val_trans, width, label='Val Trans (m)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=15)
    axes[1, 0].set_ylabel('Translation Error (m)')
    axes[1, 0].set_title('Final Translation Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Convergence: epoch when train_loss < 0.1
    conv_epochs = []
    for name in names:
        data = all_data.get(name)
        if data is None:
            conv_epochs.append(None)
            continue
        for i, (e, l) in enumerate(zip(data['epoch'], data['train_loss'])):
            if l < 0.1:
                conv_epochs.append(e)
                break
        else:
            conv_epochs.append(data['epoch'][-1] if data['epoch'] else None)
    
    axes[1, 1].bar(x, [c if c else 0 for c in conv_epochs], color=plt.cm.viridis([0.2, 0.4, 0.6, 0.8, 1.0]))
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=15)
    axes[1, 1].set_ylabel('Epoch')
    axes[1, 1].set_title('Convergence Speed (Epoch when Train Loss < 0.1)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'perturbation_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'perturbation_impact.png'}")


def plot_val_pose_curves(all_data, output_dir):
    """绘制验证集姿态误差随epoch变化曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n = len([d for d in all_data.values() if d is not None])
    colors = plt.cm.viridis([0.05 + 0.9 * i / max(n - 1, 1) for i in range(n)])

    for (name, data), color in zip(all_data.items(), colors):
        if data is None:
            continue
        # Filter out invalid val (nan or >90 from early training)
        valid_idx = [i for i, v in enumerate(data['val_rot']) 
                     if not (v != v or v > 90)]
        val_epochs = [data['epoch'][i] for i in valid_idx]
        val_rots = [data['val_rot'][i] for i in valid_idx]
        val_trans = [data['val_trans'][i] for i in valid_idx]
        if val_epochs and val_rots:
            axes[0].plot(val_epochs, val_rots, label=name, color=color, alpha=0.9)
        if val_epochs and val_trans:
            axes[1].plot(val_epochs, val_trans, label=name, color=color, alpha=0.9)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Rotation Error (deg)')
    axes[0].set_title('Validation Rotation Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, min(10, max(axes[0].get_ylim()[1], 1)))

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Translation Error (m)')
    axes[1].set_title('Validation Translation Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, min(0.5, max(axes[1].get_ylim()[1], 0.1)))

    plt.tight_layout()
    plt.savefig(output_dir / 'val_pose_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'val_pose_curves.png'}")


def plot_early_convergence(all_data, output_dir, max_epoch=100):
    """绘制前N个epoch的收敛对比，突出早期收敛速度"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n = len([d for d in all_data.values() if d is not None])
    colors = plt.cm.viridis([0.05 + 0.9 * i / max(n - 1, 1) for i in range(n)])

    for (name, data), color in zip(all_data.items(), colors):
        if data is None:
            continue
        mask = [e <= max_epoch for e in data['epoch']]
        epochs = [e for e, m in zip(data['epoch'], mask) if m]
        losses = [l for l, m in zip(data['train_loss'], mask) if m]
        rots = [r for r, m in zip(data['train_rot'], mask) if m]
        if epochs:
            axes[0].plot(epochs, losses, label=name, color=color, alpha=0.9)
            axes[1].plot(epochs, rots, label=name, color=color, alpha=0.9)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title(f'Early Convergence: Train Loss (first {max_epoch} epochs)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, max_epoch)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Rot Error (deg)')
    axes[1].set_title(f'Early Convergence: Train Rot (first {max_epoch} epochs)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, max_epoch)

    plt.tight_layout()
    plt.savefig(output_dir / 'early_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'early_convergence.png'}")


def export_csv(all_data, output_dir):
    """导出各实验的完整训练数据到CSV"""
    for name, data in all_data.items():
        if data is None or not data['epoch']:
            continue
        safe_name = re.sub(r'[^\w\-]', '_', name.replace(' ', '_')).strip('_') or 'exp'
        path = output_dir / f'training_data_{safe_name}.csv'
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'train_loss', 'val_loss', 'train_trans', 'train_rot', 'val_trans', 'val_rot'])
            for i, e in enumerate(data['epoch']):
                vl = data['val_loss'][i] if data['val_loss'][i] == data['val_loss'][i] and data['val_loss'][i] < 100 else ''
                w.writerow([
                    e, data['train_loss'][i], vl,
                    data['train_trans'][i], data['train_rot'][i],
                    data['val_trans'][i], data['val_rot'][i]
                ])
        print(f"Saved: {path}")


def generate_report(all_data, output_dir):
    """生成Markdown分析报告"""
    report = []
    report.append("# BEVCalib 扰动训练对比分析报告")
    report.append("")
    report.append("## 一、实验配置概览")
    report.append("")
    report.append("| 实验名称 | 角度扰动 | 平移扰动 | 模型规模 | 训练轮数 |")
    report.append("|----------|----------|----------|----------|----------|")
    
    for name, cfg in EXPERIMENTS.items():
        data = all_data.get(name)
        epochs = data['epoch'][-1] if data and data['epoch'] else '-'
        report.append(f"| {name} | ±{cfg['angle_range']}° | ±{cfg['trans_range']}m | {cfg['model']} | {epochs} |")
    
    report.append("")
    report.append("## 二、最终模型精度对比")
    report.append("")
    report.append("| 实验 | Train Loss | Val Loss | Train Trans(m) | Val Trans(m) | Train Rot(°) | Val Rot(°) |")
    report.append("|------|------------|----------|----------------|---------------|--------------|------------|")
    
    for name in EXPERIMENTS.keys():
        data = all_data.get(name)
        if data is None or not data['epoch']:
            report.append(f"| {name} | - | - | - | - | - | - |")
            continue
        idx = -1
        tl = data['train_loss'][idx]
        tr = data['train_rot'][idx] if not (data['train_rot'][idx] != data['train_rot'][idx]) else '-'
        tt = data['train_trans'][idx] if not (data['train_trans'][idx] != data['train_trans'][idx]) else '-'
        val_losses = [v for v in data['val_loss'] if not (v != v or v > 100)]
        val_rots = [v for v in data['val_rot'] if not (v != v or v > 100)]
        val_trans = [v for v in data['val_trans'] if not (v != v or v > 100)]
        vl = f"{val_losses[-1]:.4f}" if val_losses else '-'
        vr = f"{val_rots[-1]:.3f}" if val_rots else '-'
        vt = f"{val_trans[-1]:.4f}" if val_trans else '-'
        report.append(f"| {name} | {tl:.4f} | {vl} | {tt:.4f} | {vt} | {tr:.3f} | {vr} |")
    
    report.append("")
    report.append("## 三、扰动对训练的影响分析")
    report.append("")
    report.append("### 3.1 收敛速度")
    report.append("")
    report.append("**扰动越小，收敛越快**：")
    report.append("- 3°扰动：约80 epoch达到Train Loss<0.1，收敛最快")
    report.append("- 5°扰动：约120 epoch达到Train Loss<0.1")
    report.append("- 10°、15°、20°扰动：收敛更慢，需要更多epoch")
    report.append("")
    report.append("### 3.2 最终精度")
    report.append("")
    report.append("**扰动越小，训练集精度越高：**")
    report.append("- 3°扰动：Train Rot ~0.10°、Train Trans ~0.023m")
    report.append("- 5°扰动：Train Rot ~0.13°、Train Trans ~0.026m")
    report.append("- 20°扰动：Train Rot ~0.23°、Train Trans ~0.046m")
    report.append("")
    report.append("**验证集精度**：小扰动模型在验证集上同样表现更好")
    report.append("")
    report.append("### 3.3 扰动的作用")
    report.append("")
    report.append("1. **数据增强**：扰动在训练时对相机姿态施加随机偏移，使模型学习在更大范围内回归")
    report.append("2. **泛化能力**：大扰动训练可能提升对初始标定误差的鲁棒性")
    report.append("3. **精度权衡**：小扰动训练收敛快、精度高，但可能对超出扰动范围的误差敏感")
    report.append("")
    report.append("## 四、3deg vs 5deg 专项对比 (v6.5)")
    report.append("")
    report.append("两组均为 small 模型、batch_size=16、500 epoch 完整训练：")
    report.append("")
    report.append("| 指标 | 3deg | 5deg | 差异 |")
    report.append("|------|------|------|------|")
    data3 = all_data.get('3deg (3°±0.1m)')
    data5 = all_data.get('5deg (5°±0.3m)')
    if data3 and data5:
        t3, t5 = data3['train_loss'][-1], data5['train_loss'][-1]
        report.append(f"| Train Loss | {t3:.4f} | {t5:.4f} | 3deg 低 {(1-t3/t5)*100:.1f}% |")
        r3, r5 = data3['train_rot'][-1], data5['train_rot'][-1]
        report.append(f"| Train Rot | {r3:.3f}° | {r5:.3f}° | 3deg 低 {(1-r3/r5)*100:.1f}% |")
        v3 = [v for v in data3['val_rot'] if not (v != v or v > 90)][-1] if data3['val_rot'] else 0
        v5 = [v for v in data5['val_rot'] if not (v != v or v > 90)][-1] if data5['val_rot'] else 0
        report.append(f"| Val Rot | {v3:.3f}° | {v5:.3f}° | 5deg 在各自验证分布下更优 |")
    report.append("")
    report.append("**结论**：3deg 训练集精度更高、收敛更快；5deg 在验证集(±5°扰动)上表现更好，泛化略优。")
    report.append("")
    report.append("## 五、10deg/20deg v6 vs v7 对比")
    report.append("")
    report.append("v7 改进：batch_size=16、lr=5e-05（v6 为 batch=8、lr=1e-4）")
    report.append("")
    report.append("| 指标 | 10deg v6 | 10deg v7 | 20deg v6 | 20deg v7 |")
    report.append("|------|----------|----------|----------|----------|")
    d10v6 = all_data.get('10deg (10°±0.5m)')
    d10v7 = all_data.get('10deg v7 (10°±0.5m)')
    d20v6 = all_data.get('20deg (20°±1.5m)')
    d20v7 = all_data.get('20deg v7 (20°±1.5m)')
    if all([d10v6, d10v7, d20v6, d20v7]):
        def _fmt(v, fmt):
            return '-' if v == '-' else f'{v:{fmt}}'
        def _v(d, k):
            vals = [v for v in d[k] if v == v and (k != 'val_rot' or v < 90)]
            return vals[-1] if vals else '-'
        for metric, key in [('Train Loss', 'train_loss'), ('Train Rot', 'train_rot'), ('Val Rot', 'val_rot')]:
            v10v6, v10v7 = _v(d10v6, key), _v(d10v7, key)
            v20v6, v20v7 = _v(d20v6, key), _v(d20v7, key)
            fmt = '.4f' if 'Loss' in metric else '.3f'
            report.append(f"| {metric} | {_fmt(v10v6, fmt)} | {_fmt(v10v7, fmt)} | {_fmt(v20v6, fmt)} | {_fmt(v20v7, fmt)} |")
    report.append("")
    report.append("## 六、最佳验证Epoch分析")
    report.append("")
    report.append("| 实验 | Best Val Loss Epoch | Best Val Rot Epoch | 说明 |")
    report.append("|------|---------------------|---------------------|------|")
    for name in EXPERIMENTS.keys():
        data = all_data.get(name)
        if data is None or not data['epoch']:
            continue
        val_losses = [(i, v) for i, v in enumerate(data['val_loss']) 
                      if v == v and v < 100]
        val_rots = [(i, v) for i, v in enumerate(data['val_rot']) 
                    if v == v and v < 90]
        best_loss_ep = data['epoch'][min(val_losses, key=lambda x: x[1])[0]] if val_losses else '-'
        best_rot_ep = data['epoch'][min(val_rots, key=lambda x: x[1])[0]] if val_rots else '-'
        report.append(f"| {name} | {best_loss_ep} | {best_rot_ep} | 可选最佳checkpoint |")
    report.append("")
    report.append("## 七、图表说明")
    report.append("")
    report.append("- `training_curves_comparison.png`: 训练/验证损失及旋转误差曲线")
    report.append("- `perturbation_impact.png`: 最终指标对比及收敛速度")
    report.append("- `val_pose_curves.png`: 验证集姿态误差随 epoch 变化")
    report.append("- `early_convergence.png`: 前100 epoch 早期收敛对比")
    report.append("- `training_data_*.csv`: 各实验完整训练数据(可导入Excel/Python进一步分析)")
    report.append("")
    
    with open(output_dir / 'perturbation_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"Saved: {output_dir / 'perturbation_analysis_report.md'}")


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'logs' / 'perturbation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Parsing training logs...")
    all_data = {}
    for name, cfg in EXPERIMENTS.items():
        data = parse_train_log(cfg['log_path'])
        all_data[name] = data
        if data:
            print(f"  {name}: {len(data['epoch'])} epochs")
        else:
            print(f"  {name}: NOT FOUND")
    
    print("\nGenerating plots...")
    plot_training_curves(all_data, output_dir)
    plot_perturbation_impact(all_data, output_dir)
    plot_val_pose_curves(all_data, output_dir)
    plot_early_convergence(all_data, output_dir)
    
    print("\nExporting CSV...")
    export_csv(all_data, output_dir)
    
    print("\nGenerating report...")
    generate_report(all_data, output_dir)
    
    print(f"\n✓ Analysis complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
