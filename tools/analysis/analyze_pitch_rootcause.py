#!/usr/bin/env python3
"""
BEVCalib Pitch误差根因分析 — 基于数据驱动的证据
生成多张图表用于根因分析报告

分析维度:
1. Pitch vs Roll vs Yaw 误差分布对比 (所有模型)
2. 有符号误差分析: 系统性偏差 vs 随机散布
3. Per-sequence Pitch误差热力图: 场景依赖性
4. 训练 vs 泛化 误差放大倍数: 过拟合证据
5. V24 per-sequence 分析: Pitch在哪些场景最大
6. 误差相关性: Roll-Pitch-Yaw 是否耦合
7. Pitch误差CDF: 达到0.1度需要什么条件
"""
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

EVAL_BASE = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations"
OUTPUT_DIR = os.path.join(EVAL_BASE, "pitch_rootcause_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "V24-B": f"{EVAL_BASE}/generalization_eval_v25r_v2/V24-B-baseline",
    "v25r-A1-best": f"{EVAL_BASE}/generalization_eval_v25r_v2/v25r-A1-z10-best",
    "v25r-A1-ep400": f"{EVAL_BASE}/generalization_eval_v25r_v2/v25r-A1-z10-ep400",
    "v25r-A2-best": f"{EVAL_BASE}/generalization_eval_v25r_v2/v25r-A2-z15-best",
    "v25r-A2-ep400": f"{EVAL_BASE}/generalization_eval_v25r_v2/v25r-A2-z15-ep400",
}

TRAIN_ERRORS = {
    "V24-B":         {"rot": 0.57, "roll": 0.29, "pitch": 0.35, "yaw": 0.15},
    "v25r-A1-best":  {"rot": 0.52, "roll": 0.22, "pitch": 0.34, "yaw": 0.15},
    "v25r-A1-ep400": {"rot": 0.59, "roll": 0.24, "pitch": 0.39, "yaw": 0.14},
    "v25r-A2-best":  {"rot": 0.59, "roll": 0.26, "pitch": 0.39, "yaw": 0.15},
    "v25r-A2-ep400": {"rot": 0.66, "roll": 0.25, "pitch": 0.46, "yaw": 0.15},
}

VAL_ERRORS = {
    "V24-B":         {"rot": 0.07, "roll": 0.04, "pitch": 0.04, "yaw": 0.02},
    "v25r-A1-best":  {"rot": 0.27, "roll": 0.13, "pitch": 0.19, "yaw": 0.06},
    "v25r-A1-ep400": {"rot": 0.21, "roll": 0.09, "pitch": 0.15, "yaw": 0.03},
    "v25r-A2-best":  {"rot": 0.28, "roll": 0.12, "pitch": 0.20, "yaw": 0.08},
    "v25r-A2-ep400": {"rot": 0.19, "roll": 0.07, "pitch": 0.15, "yaw": 0.03},
}


def parse_errors(eval_dir):
    """Parse extrinsics_and_errors.txt with multi-line per-sample format."""
    fpath = os.path.join(eval_dir, "extrinsics_and_errors.txt")
    if not os.path.exists(fpath):
        return None
    
    samples = []
    cur_idx = cur_seq = None
    cur_rot = cur_roll = cur_pitch = cur_yaw = None
    
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            m_sample = re.match(r'Sample\s+(\d+)\s+\[Seq\s+(\d+)\]', line)
            if m_sample:
                if cur_idx is not None and cur_rot is not None:
                    samples.append({
                        'idx': cur_idx, 'seq': cur_seq,
                        'rot': cur_rot, 'roll': cur_roll,
                        'pitch': cur_pitch, 'yaw': cur_yaw,
                    })
                cur_idx = int(m_sample.group(1))
                cur_seq = m_sample.group(2)
                cur_rot = cur_roll = cur_pitch = cur_yaw = None
                continue
            
            m_total = re.match(r'Total:\s+([-\d.]+)\s+deg', line)
            if m_total:
                cur_rot = float(m_total.group(1))
                continue
            m_roll = re.match(r'Roll\s+\(LiDAR X\):\s+([-\d.]+)\s+deg', line)
            if m_roll:
                cur_roll = float(m_roll.group(1))
                continue
            m_pitch = re.match(r'Pitch\s+\(LiDAR Y\):\s+([-\d.]+)\s+deg', line)
            if m_pitch:
                cur_pitch = float(m_pitch.group(1))
                continue
            m_yaw = re.match(r'Yaw\s+\(LiDAR Z\):\s+([-\d.]+)\s+deg', line)
            if m_yaw:
                cur_yaw = float(m_yaw.group(1))
                continue
    
    if cur_idx is not None and cur_rot is not None:
        samples.append({
            'idx': cur_idx, 'seq': cur_seq,
            'rot': cur_rot, 'roll': cur_roll,
            'pitch': cur_pitch, 'yaw': cur_yaw,
        })
    return samples


def _R_to_euler_zyx(R):
    """ZYX Euler angles from rotation matrix (signed)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def parse_signed_errors(eval_dir):
    """Extract signed Euler angle errors from single GT + per-sample pred matrices."""
    fpath = os.path.join(eval_dir, "extrinsics_and_errors.txt")
    if not os.path.exists(fpath):
        return None
    
    signed = {'seq': [], 'roll': [], 'pitch': [], 'yaw': []}
    gt_R = None
    reading_what = None  # 'gt', 'pred', or None
    mat_lines = []
    cur_seq = None
    pred_Rs = []
    sample_seqs = []
    
    with open(fpath, 'r') as f:
        for line in f:
            stripped = line.strip()
            
            if 'Ground Truth Extrinsics' in stripped:
                reading_what = 'gt'
                mat_lines = []
                continue
            
            m_sample = re.match(r'Sample\s+\d+\s+\[Seq\s+(\d+)\]', stripped)
            if m_sample:
                cur_seq = m_sample.group(1)
                continue
            
            if 'Predicted Extrinsics' in stripped:
                reading_what = 'pred'
                mat_lines = []
                continue
            
            if 'Rotation Errors' in stripped:
                reading_what = None
                continue
            
            if reading_what:
                nums = re.findall(r'[-\d.]+', stripped)
                if len(nums) >= 4:
                    mat_lines.append([float(x) for x in nums[:4]])
                    if len(mat_lines) == 4:
                        mat = np.array(mat_lines)
                        if reading_what == 'gt':
                            gt_R = mat[:3, :3]
                        elif reading_what == 'pred' and cur_seq is not None:
                            pred_Rs.append(mat[:3, :3])
                            sample_seqs.append(cur_seq)
                        reading_what = None
                        mat_lines = []
    
    if gt_R is None or len(pred_Rs) == 0:
        return None
    
    for pred_R, seq in zip(pred_Rs, sample_seqs):
        R_err = pred_R @ gt_R.T
        roll, pitch, yaw = _R_to_euler_zyx(R_err)
        signed['seq'].append(seq)
        signed['roll'].append(np.degrees(roll))
        signed['pitch'].append(np.degrees(pitch))
        signed['yaw'].append(np.degrees(yaw))
    
    for k in ['roll', 'pitch', 'yaw']:
        signed[k] = np.array(signed[k])
    return signed if len(signed['seq']) > 0 else None


print("Loading evaluation data...")
all_data = {}
all_signed = {}
for name, path in MODELS.items():
    data = parse_errors(path)
    signed = parse_signed_errors(path)
    if data:
        all_data[name] = data
        print(f"  {name}: {len(data)} samples")
    if signed:
        all_signed[name] = signed


# ============================================================================
# Figure 1: Pitch vs Roll vs Yaw — 箱线图 + 比例分析
# ============================================================================
print("\n[Fig 1] Axis error distributions...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_names = list(all_data.keys())
for i, name in enumerate(model_names):
    data = all_data[name]
    rolls = [d['roll'] for d in data]
    pitches = [d['pitch'] for d in data]
    yaws = [d['yaw'] for d in data]
    
    pos = i * 4
    bp = axes[0].boxplot(
        [rolls, pitches, yaws],
        positions=[pos, pos+1, pos+2],
        widths=0.7,
        tick_labels=['R', 'P', 'Y'],
        patch_artist=True,
        showfliers=False,
    )
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

axes[0].set_xticks([i*4+1 for i in range(len(model_names))])
axes[0].set_xticklabels([n.replace('v25r-', '') for n in model_names], rotation=15, fontsize=9)
axes[0].set_ylabel('Error (degrees)')
axes[0].set_title('Per-Axis Error Distributions (no outliers)')
axes[0].legend(
    [plt.Rectangle((0,0),1,1, fc=c, alpha=0.7) for c in ['#3498db', '#e74c3c', '#2ecc71']],
    ['Roll', 'Pitch', 'Yaw'], loc='upper right'
)

pitch_ratios = []
for name in model_names:
    data = all_data[name]
    mean_roll = np.mean([abs(d['roll']) for d in data])
    mean_pitch = np.mean([abs(d['pitch']) for d in data])
    mean_yaw = np.mean([abs(d['yaw']) for d in data])
    total = mean_roll + mean_pitch + mean_yaw
    pitch_ratios.append({
        'name': name,
        'roll_pct': mean_roll / total * 100,
        'pitch_pct': mean_pitch / total * 100,
        'yaw_pct': mean_yaw / total * 100,
    })

x = np.arange(len(model_names))
roll_pcts = [r['roll_pct'] for r in pitch_ratios]
pitch_pcts = [r['pitch_pct'] for r in pitch_ratios]
yaw_pcts = [r['yaw_pct'] for r in pitch_ratios]

axes[1].bar(x, roll_pcts, label='Roll', color='#3498db', alpha=0.8)
axes[1].bar(x, pitch_pcts, bottom=roll_pcts, label='Pitch', color='#e74c3c', alpha=0.8)
axes[1].bar(x, yaw_pcts, bottom=[r+p for r,p in zip(roll_pcts, pitch_pcts)], label='Yaw', color='#2ecc71', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([n.replace('v25r-', '') for n in model_names], rotation=15, fontsize=9)
axes[1].set_ylabel('Percentage (%)')
axes[1].set_title('Pitch Contribution to Total Error')
axes[1].legend()

for i, pr in enumerate(pitch_ratios):
    axes[1].text(i, pr['roll_pct'] + pr['pitch_pct']/2, f"{pr['pitch_pct']:.0f}%",
                ha='center', va='center', fontweight='bold', color='white', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_axis_error_distributions.png"), bbox_inches='tight')
pcts = [f"{r['pitch_pct']:.0f}%" for r in pitch_ratios]
print(f"  Saved fig1. Pitch contributions: {pcts}")


# ============================================================================
# Figure 2: 有符号误差 — 系统性偏差 vs 随机散布
# ============================================================================
print("\n[Fig 2] Signed error analysis (bias vs variance)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col, axis_name in enumerate(['roll', 'pitch', 'yaw']):
    for name in model_names:
        s = all_signed[name]
        vals = s[axis_name]
        axes[0, col].hist(vals, bins=80, alpha=0.4, label=name.replace('v25r-', ''), density=True)
    axes[0, col].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, col].set_title(f'{axis_name.capitalize()} Signed Error Distribution')
    axes[0, col].set_xlabel('Error (degrees)')
    axes[0, col].legend(fontsize=7)

bias_data = {}
for name in model_names:
    s = all_signed[name]
    bias_data[name] = {}
    for axis in ['roll', 'pitch', 'yaw']:
        vals = s[axis]
        bias_data[name][axis] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'abs_mean': np.mean(np.abs(vals)),
            'bias_ratio': abs(np.mean(vals)) / np.std(vals) if np.std(vals) > 0 else 0,
        }

for col, axis_name in enumerate(['roll', 'pitch', 'yaw']):
    names_short = [n.replace('v25r-', '') for n in model_names]
    means = [bias_data[n][axis_name]['mean'] for n in model_names]
    stds = [bias_data[n][axis_name]['std'] for n in model_names]
    
    x = np.arange(len(model_names))
    axes[1, col].bar(x - 0.15, means, 0.3, label='Bias (mean)', color='#e74c3c', alpha=0.8)
    axes[1, col].bar(x + 0.15, stds, 0.3, label='Std (variance)', color='#3498db', alpha=0.8)
    axes[1, col].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[1, col].set_xticks(x)
    axes[1, col].set_xticklabels(names_short, rotation=15, fontsize=8)
    axes[1, col].set_ylabel('Degrees')
    axes[1, col].set_title(f'{axis_name.capitalize()}: Bias vs Variance')
    axes[1, col].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_signed_error_bias_variance.png"), bbox_inches='tight')

for name in model_names:
    b = bias_data[name]
    print(f"  {name}: "
          f"Roll(bias={b['roll']['mean']:+.3f} std={b['roll']['std']:.3f}) "
          f"Pitch(bias={b['pitch']['mean']:+.3f} std={b['pitch']['std']:.3f}) "
          f"Yaw(bias={b['yaw']['mean']:+.3f} std={b['yaw']['std']:.3f})")


# ============================================================================
# Figure 3: Per-Sequence Pitch误差热力图
# ============================================================================
print("\n[Fig 3] Per-sequence pitch error heatmap...")
seqs = sorted(set(d['seq'] for d in all_data[model_names[0]]))
heatmap_data = np.zeros((len(model_names), len(seqs)))

for i, name in enumerate(model_names):
    data = all_data[name]
    for j, seq in enumerate(seqs):
        seq_samples = [d for d in data if d['seq'] == seq]
        if seq_samples:
            heatmap_data[i, j] = np.mean([abs(d['pitch']) for d in seq_samples])

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(range(len(seqs)))
ax.set_xticklabels([f'Seq {s}' for s in seqs], rotation=45, fontsize=9)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels([n.replace('v25r-', '') for n in model_names], fontsize=9)
ax.set_title('Mean |Pitch| Error by Sequence (degrees)')
plt.colorbar(im, ax=ax, label='Mean |Pitch| Error (deg)')

for i in range(len(model_names)):
    for j in range(len(seqs)):
        val = heatmap_data[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_pitch_per_sequence_heatmap.png"), bbox_inches='tight')
print(f"  Saved fig3. Shape: {heatmap_data.shape}")


# ============================================================================
# Figure 4: 训练→验证→泛化 误差放大链
# ============================================================================
print("\n[Fig 4] Train→Val→Test error amplification chain...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, axis_name in enumerate(['pitch', 'rot']):
    x = np.arange(len(model_names))
    train_vals = [TRAIN_ERRORS[n][axis_name] for n in model_names]
    val_vals = [VAL_ERRORS[n][axis_name] for n in model_names]
    if axis_name == 'rot':
        test_vals = [np.mean([d['rot'] for d in all_data[n]]) for n in model_names]
    else:
        test_vals = [np.mean([abs(d[axis_name]) for d in all_data[n]]) for n in model_names]
    
    width = 0.25
    bars1 = axes[ax_idx].bar(x - width, train_vals, width, label='Train', color='#27ae60', alpha=0.8)
    bars2 = axes[ax_idx].bar(x, val_vals, width, label='Val', color='#3498db', alpha=0.8)
    bars3 = axes[ax_idx].bar(x + width, test_vals, width, label='Test (generalization)', color='#e74c3c', alpha=0.8)
    
    for i, (tv, gv) in enumerate(zip(val_vals, test_vals)):
        ratio = gv / tv if tv > 0 else 0
        axes[ax_idx].annotate(f'{ratio:.0f}x', xy=(i+width, gv), fontsize=9,
                             fontweight='bold', color='red', ha='center', va='bottom')
    
    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels([n.replace('v25r-', '') for n in model_names], rotation=15, fontsize=9)
    axes[ax_idx].set_ylabel('Error (degrees)')
    axes[ax_idx].set_title(f'{axis_name.capitalize()}: Train→Val→Test Amplification')
    axes[ax_idx].legend()
    axes[ax_idx].axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Target: 0.1°')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig4_train_val_test_amplification.png"), bbox_inches='tight')

for n in model_names:
    te = TRAIN_ERRORS[n]['pitch']
    ve = VAL_ERRORS[n]['pitch']
    ge = np.mean([abs(d['pitch']) for d in all_data[n]])
    print(f"  {n}: Pitch Train={te:.2f}→Val={ve:.2f}→Test={ge:.2f} "
          f"(Val→Test: {ge/ve:.1f}x, Train→Test: {ge/te:.1f}x)")


# ============================================================================
# Figure 5: Pitch误差随序列帧号的时间变化 — 抖动分析
# ============================================================================
print("\n[Fig 5] Pitch error temporal variation within sequences...")
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

name = "V24-B"
data = all_data[name]
signed = all_signed[name]

for j, seq in enumerate(seqs):
    if j >= 12:
        break
    ax = axes[j]
    mask = np.array(signed['seq']) == seq
    pitch_vals = signed['pitch'][mask]
    roll_vals = signed['roll'][mask]
    
    frames = np.arange(len(pitch_vals))
    ax.plot(frames, pitch_vals, 'r-', alpha=0.6, linewidth=0.8, label='Pitch')
    ax.plot(frames, roll_vals, 'b-', alpha=0.4, linewidth=0.8, label='Roll')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.fill_between(frames, -0.1, 0.1, alpha=0.1, color='green')
    ax.set_title(f'Seq {seq} (V24-B)', fontsize=9)
    ax.set_ylim(-2.5, 2.5)
    if j == 0:
        ax.legend(fontsize=7)
    
    pitch_std = np.std(pitch_vals)
    roll_std = np.std(roll_vals)
    ax.text(0.95, 0.95, f'P_std={pitch_std:.2f}\nR_std={roll_std:.2f}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("V24-B: Pitch vs Roll Signed Error Temporal Traces (per sequence)\n"
             "Green band = ±0.1° target", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig5_pitch_temporal_traces_v24b.png"), bbox_inches='tight')
print(f"  Saved fig5")


# ============================================================================
# Figure 6: V24-B vs v25r Pitch temporal comparison for key sequences
# ============================================================================
print("\n[Fig 6] V24-B vs v25r Pitch comparison for key sequences...")
key_seqs = ['04', '09', '07', '05']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, seq in enumerate(key_seqs):
    ax = axes[idx]
    for name in ["V24-B", "v25r-A1-ep400"]:
        s = all_signed[name]
        mask = np.array(s['seq']) == seq
        pitch_vals = s['pitch'][mask]
        frames = np.arange(len(pitch_vals))
        label = name.replace('v25r-', '')
        style = '-' if name == "V24-B" else '--'
        ax.plot(frames, pitch_vals, style, alpha=0.7, linewidth=1, label=label)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.fill_between(np.arange(400), -0.1, 0.1, alpha=0.1, color='green')
    
    v24_p = all_signed["V24-B"]['pitch'][np.array(all_signed["V24-B"]['seq']) == seq]
    v25_p = all_signed["v25r-A1-ep400"]['pitch'][np.array(all_signed["v25r-A1-ep400"]['seq']) == seq]
    
    ax.set_title(f'Seq {seq}: V24-B (std={np.std(v24_p):.2f}) vs v25r-A1 (std={np.std(v25_p):.2f})', fontsize=10)
    ax.set_ylabel('Pitch Error (deg)')
    ax.set_xlabel('Frame Index')
    ax.legend(fontsize=8)
    ax.set_ylim(-3, 3)

plt.suptitle("Pitch Error Comparison: V24-B vs v25r-A1-ep400\n"
             "Seq04=best V24, Seq09=best V24 worst v25r, Seq07=worst overall, Seq05=high variance",
             fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig6_pitch_comparison_key_sequences.png"), bbox_inches='tight')
print(f"  Saved fig6")


# ============================================================================
# Figure 7: Pitch CDF — 多少帧能达到0.1°
# ============================================================================
print("\n[Fig 7] Pitch error CDF analysis...")
fig, ax = plt.subplots(figsize=(12, 6))

thresholds = np.linspace(0, 2.0, 200)

for name in model_names:
    data = all_data[name]
    pitch_abs = sorted([abs(d['pitch']) for d in data])
    cdf = np.searchsorted(pitch_abs, thresholds) / len(pitch_abs)
    ax.plot(thresholds, cdf * 100, label=name.replace('v25r-', ''), linewidth=1.5)

ax.axvline(0.1, color='red', linestyle='--', alpha=0.8, label='Target: 0.1°')
ax.axvline(0.25, color='orange', linestyle='--', alpha=0.5, label='Intermediate: 0.25°')
ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='Current V24: ~0.5°')
ax.set_xlabel('|Pitch| Error Threshold (degrees)')
ax.set_ylabel('Percentage of frames below threshold (%)')
ax.set_title('Pitch Error CDF: How many frames achieve target accuracy?')
ax.legend()
ax.grid(True, alpha=0.3)

for name in model_names:
    data = all_data[name]
    pitch_abs = np.array([abs(d['pitch']) for d in data])
    pct_01 = np.mean(pitch_abs < 0.1) * 100
    pct_025 = np.mean(pitch_abs < 0.25) * 100
    pct_05 = np.mean(pitch_abs < 0.5) * 100
    print(f"  {name}: <0.1°={pct_01:.1f}% <0.25°={pct_025:.1f}% <0.5°={pct_05:.1f}%")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig7_pitch_cdf.png"), bbox_inches='tight')


# ============================================================================
# Figure 8: Pitch误差与Roll/Yaw的关联性 — 是否存在耦合
# ============================================================================
print("\n[Fig 8] Axis error correlation...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

name = "V24-B"
s = all_signed[name]
pairs = [('roll', 'pitch'), ('pitch', 'yaw'), ('roll', 'yaw')]
for idx, (a1, a2) in enumerate(pairs):
    v1, v2 = s[a1], s[a2]
    corr = np.corrcoef(v1, v2)[0, 1]
    axes[idx].scatter(v1, v2, alpha=0.15, s=5, c='steelblue')
    axes[idx].set_xlabel(f'{a1.capitalize()} Error (deg)')
    axes[idx].set_ylabel(f'{a2.capitalize()} Error (deg)')
    axes[idx].set_title(f'V24-B: {a1.capitalize()} vs {a2.capitalize()} (corr={corr:.3f})')
    axes[idx].axhline(0, color='gray', alpha=0.3)
    axes[idx].axvline(0, color='gray', alpha=0.3)
    z = np.polyfit(v1, v2, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(v1.min(), v1.max(), 100)
    axes[idx].plot(x_fit, p(x_fit), 'r-', linewidth=2, alpha=0.7)

plt.suptitle("Axis Error Correlations (V24-B): Is Pitch coupled with Roll/Yaw?", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig8_axis_correlation.png"), bbox_inches='tight')
print(f"  Saved fig8")


# ============================================================================
# Figure 9: Per-Sequence Pitch Std vs Mean — 识别高方差场景
# ============================================================================
print("\n[Fig 9] Per-sequence Pitch variance analysis...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, name in enumerate(["V24-B", "v25r-A1-ep400"]):
    ax = axes[ax_idx]
    s = all_signed[name]
    
    seq_stats = []
    for seq in seqs:
        mask = np.array(s['seq']) == seq
        pv = s['pitch'][mask]
        seq_stats.append({
            'seq': seq,
            'mean': np.mean(pv),
            'std': np.std(pv),
            'abs_mean': np.mean(np.abs(pv)),
            'n': len(pv),
        })
    
    for st in seq_stats:
        ax.scatter(st['std'], st['abs_mean'], s=100, alpha=0.8, zorder=5)
        ax.annotate(f"Seq{st['seq']}", (st['std'], st['abs_mean']),
                   fontsize=8, ha='center', va='bottom')
    
    ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Target: <0.1°')
    ax.set_xlabel('Pitch Std (within-sequence variability)')
    ax.set_ylabel('Mean |Pitch| Error')
    ax.set_title(f'{name}: Sequence Pitch Std vs Mean Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig9_pitch_std_vs_mean_per_sequence.png"), bbox_inches='tight')
print(f"  Saved fig9")


# ============================================================================
# Figure 10: Train→Val→Test 三阶段放大倍数 (关键证据)
# ============================================================================
print("\n[Fig 10] Amplification factors...")
fig, ax = plt.subplots(figsize=(14, 6))

axes_names = ['roll', 'pitch', 'yaw']
x = np.arange(len(model_names))
width = 0.25

for j, axis in enumerate(axes_names):
    amplifications = []
    for name in model_names:
        val_err = VAL_ERRORS[name][axis]
        if axis == 'rot':
            test_err = np.mean([d['rot'] for d in all_data[name]])
        else:
            test_err = np.mean([abs(d[axis]) for d in all_data[name]])
        amp = test_err / val_err if val_err > 0.001 else 0
        amplifications.append(amp)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    ax.bar(x + j*width - width, amplifications, width, 
           label=axis.capitalize(), color=colors[j], alpha=0.8)

ax.axhline(1.0, color='black', linestyle='--', alpha=0.3, label='No degradation (1x)')
ax.set_xticks(x)
ax.set_xticklabels([n.replace('v25r-', '') for n in model_names], rotation=15, fontsize=9)
ax.set_ylabel('Val→Test Amplification Factor (x)')
ax.set_title('Generalization Gap: Val→Test Error Amplification per Axis\n'
             '(Higher = worse generalization, indicates overfitting or domain gap)')
ax.legend()
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig10_amplification_factors.png"), bbox_inches='tight')

for name in model_names:
    amps = {}
    for axis in axes_names:
        ve = VAL_ERRORS[name][axis]
        ge = np.mean([abs(d[axis]) for d in all_data[name]])
        amps[axis] = ge / ve if ve > 0.001 else 0
    print(f"  {name}: Roll={amps['roll']:.1f}x Pitch={amps['pitch']:.1f}x Yaw={amps['yaw']:.1f}x")


# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS SUMMARY")
print("="*80)

print("\n--- Key Finding 1: Pitch dominance across ALL models ---")
for name in model_names:
    data = all_data[name]
    mr = np.mean([abs(d['roll']) for d in data])
    mp = np.mean([abs(d['pitch']) for d in data])
    my = np.mean([abs(d['yaw']) for d in data])
    total = mr + mp + my
    print(f"  {name}: Roll={mr:.3f}({mr/total*100:.0f}%) "
          f"Pitch={mp:.3f}({mp/total*100:.0f}%) Yaw={my:.3f}({my/total*100:.0f}%)")

print("\n--- Key Finding 2: Pitch Variance >> Bias ---")
for name in model_names:
    b = bias_data[name]
    print(f"  {name}: Pitch bias={b['pitch']['mean']:+.3f}° "
          f"std={b['pitch']['std']:.3f}° ratio={b['pitch']['bias_ratio']:.2f}")

print("\n--- Key Finding 3: Val→Test amplification ---")
for name in model_names:
    ve = VAL_ERRORS[name]['pitch']
    ge = np.mean([abs(d['pitch']) for d in all_data[name]])
    print(f"  {name}: Pitch Val={ve:.2f}→Test={ge:.2f} ({ge/ve:.0f}x amplification)")

print(f"\nAll figures saved to: {OUTPUT_DIR}/")
print("Done!")
