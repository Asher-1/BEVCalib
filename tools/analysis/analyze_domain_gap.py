#!/usr/bin/env python3
"""
分析训练数据和测试数据之间的域差异 (Domain Gap)
重点: 外参差异 + 内参差异 + 序列达标分析
"""
import os
import sys
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

TRAIN_DIR = "/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data"
TEST_DIR = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
EVAL_DIR = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations"
OUTPUT_DIR = os.path.join(EVAL_DIR, "pitch_rootcause_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def R_to_euler_zyx(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def parse_calib(calib_path):
    """Parse KITTI-style calib.txt."""
    data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            key, vals = line.split(':', 1)
            key = key.strip()
            try:
                nums = [float(x) for x in vals.strip().split()]
                data[key] = nums
            except ValueError:
                data[key] = vals.strip()
    
    Tr = np.eye(4)
    if 'Tr' in data and len(data['Tr']) == 12:
        Tr[:3, :] = np.array(data['Tr']).reshape(3, 4)
    
    P = None
    if 'P0' in data and len(data['P0']) == 12:
        P = np.array(data['P0']).reshape(3, 4)
    
    return {'Tr': Tr, 'P': P, 'raw': data}


# ============================================================================
# Part 1: 训练vs测试 外参差异分析
# ============================================================================
print("="*80)
print("Part 1: Training vs Test Extrinsic (Tr) Differences")
print("="*80)

train_seqs = sorted([d for d in os.listdir(os.path.join(TRAIN_DIR, 'sequences')) 
                      if d.isdigit()])
test_seqs = sorted([d for d in os.listdir(os.path.join(TEST_DIR, 'sequences')) 
                     if d.isdigit()])

train_calibs = {}
test_calibs = {}

print(f"\nTraining sequences: {len(train_seqs)} ({', '.join(train_seqs)})")
print(f"Test sequences: {len(test_seqs)} ({', '.join(test_seqs)})")

for seq in train_seqs:
    cpath = os.path.join(TRAIN_DIR, 'sequences', seq, 'calib.txt')
    if os.path.exists(cpath):
        train_calibs[seq] = parse_calib(cpath)

for seq in test_seqs:
    cpath = os.path.join(TEST_DIR, 'sequences', seq, 'calib.txt')
    if os.path.exists(cpath):
        test_calibs[seq] = parse_calib(cpath)

print(f"\nParsed calibrations: Train={len(train_calibs)}, Test={len(test_calibs)}")

# Extract Euler angles from Tr (LiDAR→Camera)
print("\n--- Training Set Extrinsics (Tr: LiDAR→Camera) ---")
print(f"{'Seq':>4s} | {'Roll':>8s} | {'Pitch':>8s} | {'Yaw':>8s} | {'tx':>8s} | {'ty':>8s} | {'tz':>8s}")
print("-" * 68)

train_eulers = []
for seq in sorted(train_calibs.keys()):
    Tr = train_calibs[seq]['Tr']
    R = Tr[:3, :3]
    t = Tr[:3, 3]
    roll, pitch, yaw = R_to_euler_zyx(R)
    train_eulers.append({'seq': seq, 'roll': roll, 'pitch': pitch, 'yaw': yaw, 't': t})
    print(f"{seq:>4s} | {roll:+8.3f}° | {pitch:+8.3f}° | {yaw:+8.3f}° | {t[0]:+8.4f} | {t[1]:+8.4f} | {t[2]:+8.4f}")

print("\n--- Test Set Extrinsics (Tr: LiDAR→Camera) ---")
print(f"{'Seq':>4s} | {'Roll':>8s} | {'Pitch':>8s} | {'Yaw':>8s} | {'tx':>8s} | {'ty':>8s} | {'tz':>8s}")
print("-" * 68)

test_eulers = []
for seq in sorted(test_calibs.keys()):
    Tr = test_calibs[seq]['Tr']
    R = Tr[:3, :3]
    t = Tr[:3, 3]
    roll, pitch, yaw = R_to_euler_zyx(R)
    test_eulers.append({'seq': seq, 'roll': roll, 'pitch': pitch, 'yaw': yaw, 't': t})
    print(f"{seq:>4s} | {roll:+8.3f}° | {pitch:+8.3f}° | {yaw:+8.3f}° | {t[0]:+8.4f} | {t[1]:+8.4f} | {t[2]:+8.4f}")

# Statistics
train_rolls = [e['roll'] for e in train_eulers]
train_pitches = [e['pitch'] for e in train_eulers]
train_yaws = [e['yaw'] for e in train_eulers]
test_rolls = [e['roll'] for e in test_eulers]
test_pitches = [e['pitch'] for e in test_eulers]
test_yaws = [e['yaw'] for e in test_eulers]

print("\n--- Summary Statistics ---")
print(f"{'':>10s} | {'Train Mean':>12s} | {'Train Std':>12s} | {'Test Mean':>12s} | {'Test Std':>12s} | {'Gap (Mean)':>12s}")
print("-" * 76)
for name, tv, gv in [('Roll', train_rolls, test_rolls),
                      ('Pitch', train_pitches, test_pitches),
                      ('Yaw', train_yaws, test_yaws)]:
    print(f"{name:>10s} | {np.mean(tv):+12.3f}° | {np.std(tv):12.3f}° | "
          f"{np.mean(gv):+12.3f}° | {np.std(gv):12.3f}° | {abs(np.mean(gv)-np.mean(tv)):12.3f}°")

# Intrinsics comparison
print("\n--- Intrinsic Parameters (P0) ---")
print("Training set focal lengths:")
for seq in sorted(train_calibs.keys()):
    P = train_calibs[seq]['P']
    if P is not None:
        fx, fy = P[0, 0], P[1, 1]
        cx, cy = P[0, 2], P[1, 2]
        print(f"  Seq {seq}: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

print("\nTest set focal lengths:")
for seq in sorted(test_calibs.keys()):
    P = test_calibs[seq]['P']
    if P is not None:
        fx, fy = P[0, 0], P[1, 1]
        cx, cy = P[0, 2], P[1, 2]
        print(f"  Seq {seq}: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")


# ============================================================================
# Figure 11: Train vs Test Extrinsic Distribution
# ============================================================================
print("\n[Fig 11] Train vs Test extrinsic comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for col, (name, tv, gv) in enumerate([
    ('Roll', train_rolls, test_rolls),
    ('Pitch', train_pitches, test_pitches),
    ('Yaw', train_yaws, test_yaws)
]):
    axes[col].hist(tv, bins=20, alpha=0.6, label=f'Train ({len(tv)} seqs)', color='#3498db')
    axes[col].hist(gv, bins=15, alpha=0.6, label=f'Test ({len(gv)} seqs)', color='#e74c3c')
    axes[col].axvline(np.mean(tv), color='blue', linestyle='--', alpha=0.8, label=f'Train mean={np.mean(tv):.2f}°')
    axes[col].axvline(np.mean(gv), color='red', linestyle='--', alpha=0.8, label=f'Test mean={np.mean(gv):.2f}°')
    axes[col].set_xlabel(f'{name} Angle (degrees)')
    axes[col].set_ylabel('Count')
    axes[col].set_title(f'{name}: Train vs Test GT Extrinsics')
    axes[col].legend(fontsize=8)

plt.suptitle("Domain Gap Evidence: GT Extrinsic (Tr) Distribution Shift\n"
             "Different sensor mounting → different calibration parameters", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig11_train_vs_test_extrinsics.png"), bbox_inches='tight')
print("  Saved fig11")


# ============================================================================
# Figure 12: Extrinsic scatter — per-sequence with train/test coloring
# ============================================================================
print("\n[Fig 12] Extrinsic scatter (Roll vs Pitch)...")
fig, ax = plt.subplots(figsize=(10, 8))

for e in train_eulers:
    ax.scatter(e['roll'], e['pitch'], c='#3498db', s=80, alpha=0.7, zorder=5,
               edgecolors='navy', linewidths=0.5)
    ax.annotate(f'Train-{e["seq"]}', (e['roll'], e['pitch']),
               fontsize=7, ha='left', va='bottom', color='blue')

for e in test_eulers:
    ax.scatter(e['roll'], e['pitch'], c='#e74c3c', s=100, alpha=0.8, zorder=6,
               edgecolors='darkred', linewidths=0.5, marker='s')
    ax.annotate(f'Test-{e["seq"]}', (e['roll'], e['pitch']),
               fontsize=7, ha='left', va='bottom', color='red')

ax.set_xlabel('Roll Angle (degrees)')
ax.set_ylabel('Pitch Angle (degrees)')
ax.set_title('GT Extrinsic Roll vs Pitch: Train (circles) vs Test (squares)\n'
             'Each point = one sequence\'s sensor mounting angle')
ax.legend([plt.Line2D([0],[0], marker='o', color='#3498db', ls='', markersize=8),
           plt.Line2D([0],[0], marker='s', color='#e74c3c', ls='', markersize=8)],
          [f'Training ({len(train_eulers)} seqs)', f'Test ({len(test_eulers)} seqs)'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig12_extrinsic_scatter.png"), bbox_inches='tight')
print("  Saved fig12")


# ============================================================================
# Part 2: Per-Sequence 达标分析
# ============================================================================
print("\n" + "="*80)
print("Part 2: Per-Sequence Achievement Analysis (which seqs reach <0.1°?)")
print("="*80)

# Load V24-B per-sample errors
v24b_dir = os.path.join(EVAL_DIR, "generalization_eval_v25r_v2/V24-B-baseline")
samples = []
with open(os.path.join(v24b_dir, "extrinsics_and_errors.txt"), 'r') as f:
    cur_idx = cur_seq = None
    cur = {}
    for line in f:
        line = line.strip()
        m = re.match(r'Sample\s+(\d+)\s+\[Seq\s+(\d+)\]', line)
        if m:
            if cur_idx is not None and cur:
                cur['idx'] = cur_idx
                cur['seq'] = cur_seq
                samples.append(cur)
            cur_idx = int(m.group(1))
            cur_seq = m.group(2)
            cur = {}
            continue
        m_t = re.match(r'Total:\s+([\d.]+)', line)
        if m_t: cur['rot'] = float(m_t.group(1))
        m_r = re.match(r'Roll\s+\(LiDAR X\):\s+([\d.]+)', line)
        if m_r: cur['roll'] = float(m_r.group(1))
        m_p = re.match(r'Pitch\s+\(LiDAR Y\):\s+([\d.]+)', line)
        if m_p: cur['pitch'] = float(m_p.group(1))
        m_y = re.match(r'Yaw\s+\(LiDAR Z\):\s+([\d.]+)', line)
        if m_y: cur['yaw'] = float(m_y.group(1))
    if cur_idx is not None and cur:
        cur['idx'] = cur_idx
        cur['seq'] = cur_seq
        samples.append(cur)

print(f"\nLoaded {len(samples)} V24-B samples")

seqs = sorted(set(s['seq'] for s in samples))
print(f"\n{'Seq':>4s} | {'N':>5s} | {'Mean Rot':>9s} | {'Mean Roll':>10s} | {'Mean Pitch':>11s} | {'Mean Yaw':>9s} | "
      f"{'P<0.1°':>7s} | {'P<0.25°':>8s} | {'Status':>10s}")
print("-" * 108)

seq_stats = []
for seq in seqs:
    ss = [s for s in samples if s['seq'] == seq]
    n = len(ss)
    mr = np.mean([s['rot'] for s in ss])
    mroll = np.mean([s['roll'] for s in ss])
    mpitch = np.mean([s['pitch'] for s in ss])
    myaw = np.mean([s['yaw'] for s in ss])
    p01 = np.mean([s['pitch'] < 0.1 for s in ss]) * 100
    p025 = np.mean([s['pitch'] < 0.25 for s in ss]) * 100
    
    status = "PASS" if mpitch < 0.1 else ("CLOSE" if mpitch < 0.2 else "FAIL")
    print(f"{seq:>4s} | {n:>5d} | {mr:>9.3f}° | {mroll:>10.3f}° | {mpitch:>11.3f}° | {myaw:>9.3f}° | "
          f"{p01:>6.1f}% | {p025:>7.1f}% | {status:>10s}")
    
    # Find nearest train extrinsic
    test_ext = None
    for e in test_eulers:
        if e['seq'] == seq:
            test_ext = e
            break
    
    seq_stats.append({
        'seq': seq, 'n': n,
        'rot': mr, 'roll': mroll, 'pitch': mpitch, 'yaw': myaw,
        'p01': p01, 'p025': p025,
        'test_ext': test_ext,
    })

# Find closest train sequence for each test sequence
print("\n--- Domain Gap per Test Sequence ---")
print(f"{'Test Seq':>8s} | {'Test Pitch':>11s} | {'Closest Train':>14s} | {'Train Pitch':>12s} | {'Pitch Gap':>10s} | {'Pred Error':>11s}")
print("-" * 80)

for st in seq_stats:
    te = st['test_ext']
    if te is None:
        continue
    min_dist = float('inf')
    closest = None
    for tr_e in train_eulers:
        d = np.sqrt((te['roll'] - tr_e['roll'])**2 + 
                    (te['pitch'] - tr_e['pitch'])**2 +
                    (te['yaw'] - tr_e['yaw'])**2)
        if d < min_dist:
            min_dist = d
            closest = tr_e
    
    if closest:
        pitch_gap = abs(te['pitch'] - closest['pitch'])
        print(f"Test {st['seq']:>3s} | {te['pitch']:>+10.3f}° | Train {closest['seq']:>3s}     | "
              f"{closest['pitch']:>+11.3f}° | {pitch_gap:>9.3f}° | {st['pitch']:>10.3f}°")


# ============================================================================
# Figure 13: Per-Sequence Pitch Error vs Domain Gap
# ============================================================================
print("\n[Fig 13] Pitch error vs domain gap...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: per-seq pitch error bar
seq_labels = [f"Seq {s['seq']}" for s in seq_stats]
pitch_vals = [s['pitch'] for s in seq_stats]
colors = ['#27ae60' if v < 0.1 else '#f39c12' if v < 0.25 else '#e74c3c' for v in pitch_vals]

bars = axes[0].bar(range(len(seq_stats)), pitch_vals, color=colors, alpha=0.8, edgecolor='gray')
axes[0].axhline(0.1, color='green', linestyle='--', linewidth=2, label='Target: 0.1°')
axes[0].axhline(0.25, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='0.25°')
axes[0].set_xticks(range(len(seq_stats)))
axes[0].set_xticklabels(seq_labels, rotation=45, fontsize=8)
axes[0].set_ylabel('Mean |Pitch| Error (V24-B)')
axes[0].set_title('V24-B Per-Sequence Pitch Error\n(Green=<0.1°, Orange=<0.25°, Red=>0.25°)')
axes[0].legend()

for i, v in enumerate(pitch_vals):
    axes[0].text(i, v + 0.01, f'{v:.2f}°', ha='center', fontsize=8)

# Right: Domain gap vs prediction error scatter
gaps = []
pred_errors = []
seq_labels_r = []
for st in seq_stats:
    te = st['test_ext']
    if te is None:
        continue
    min_dist = float('inf')
    for tr_e in train_eulers:
        d = np.sqrt((te['roll'] - tr_e['roll'])**2 + 
                    (te['pitch'] - tr_e['pitch'])**2 +
                    (te['yaw'] - tr_e['yaw'])**2)
        if d < min_dist:
            min_dist = d
    gaps.append(min_dist)
    pred_errors.append(st['pitch'])
    seq_labels_r.append(st['seq'])

axes[1].scatter(gaps, pred_errors, s=100, c='steelblue', alpha=0.8, zorder=5, edgecolors='navy')
for i, (g, p, s) in enumerate(zip(gaps, pred_errors, seq_labels_r)):
    axes[1].annotate(f'Seq{s}', (g, p), fontsize=8, ha='left', va='bottom')

axes[1].axhline(0.1, color='green', linestyle='--', alpha=0.7, label='Target: 0.1°')
axes[1].set_xlabel('Distance to Nearest Training Extrinsic (degrees)')
axes[1].set_ylabel('Mean |Pitch| Prediction Error (V24-B)')
axes[1].set_title('Pitch Error vs Domain Gap\n(closer to train → lower error expected)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

if len(gaps) > 1:
    corr = np.corrcoef(gaps, pred_errors)[0, 1]
    z = np.polyfit(gaps, pred_errors, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(min(gaps), max(gaps), 100)
    axes[1].plot(x_fit, p(x_fit), 'r-', linewidth=2, alpha=0.7, label=f'Trend (corr={corr:.2f})')
    axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig13_pitch_error_vs_domain_gap.png"), bbox_inches='tight')
print("  Saved fig13")


# ============================================================================
# Figure 14: 训练集外参多样性 vs 测试集
# ============================================================================
print("\n[Fig 14] Intrinsic comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_fx = [train_calibs[s]['P'][0, 0] for s in sorted(train_calibs.keys()) if train_calibs[s]['P'] is not None]
test_fx = [test_calibs[s]['P'][0, 0] for s in sorted(test_calibs.keys()) if test_calibs[s]['P'] is not None]

axes[0].hist(train_fx, bins=20, alpha=0.6, label=f'Train ({len(train_fx)} seqs)', color='#3498db')
axes[0].hist(test_fx, bins=12, alpha=0.6, label=f'Test ({len(test_fx)} seqs)', color='#e74c3c')
axes[0].set_xlabel('Focal Length fx')
axes[0].set_ylabel('Count')
axes[0].set_title('Intrinsic: Focal Length Distribution')
axes[0].legend()

train_cx = [train_calibs[s]['P'][0, 2] for s in sorted(train_calibs.keys()) if train_calibs[s]['P'] is not None]
test_cx = [test_calibs[s]['P'][0, 2] for s in sorted(test_calibs.keys()) if test_calibs[s]['P'] is not None]
train_cy = [train_calibs[s]['P'][1, 2] for s in sorted(train_calibs.keys()) if train_calibs[s]['P'] is not None]
test_cy = [test_calibs[s]['P'][1, 2] for s in sorted(test_calibs.keys()) if test_calibs[s]['P'] is not None]

axes[1].scatter(train_cx, train_cy, c='#3498db', s=80, alpha=0.7, label='Train', zorder=5)
axes[1].scatter(test_cx, test_cy, c='#e74c3c', s=100, alpha=0.8, label='Test', marker='s', zorder=6)
axes[1].set_xlabel('Principal Point cx')
axes[1].set_ylabel('Principal Point cy')
axes[1].set_title('Intrinsic: Principal Point Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Intrinsic Parameters: Train vs Test", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig14_intrinsic_comparison.png"), bbox_inches='tight')
print("  Saved fig14")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("DOMAIN GAP ANALYSIS SUMMARY")
print("="*80)

# Check if all train seqs share same extrinsic
train_pitch_range = max(train_pitches) - min(train_pitches)
test_pitch_range = max(test_pitches) - min(test_pitches)
print(f"\nTrain Pitch range: {min(train_pitches):.3f}° to {max(train_pitches):.3f}° (span={train_pitch_range:.3f}°)")
print(f"Test  Pitch range: {min(test_pitches):.3f}° to {max(test_pitches):.3f}° (span={test_pitch_range:.3f}°)")

n_unique_train_ext = len(set(f"{e['roll']:.2f}_{e['pitch']:.2f}_{e['yaw']:.2f}" for e in train_eulers))
n_unique_test_ext = len(set(f"{e['roll']:.2f}_{e['pitch']:.2f}_{e['yaw']:.2f}" for e in test_eulers))
print(f"\nUnique extrinsics: Train={n_unique_train_ext}, Test={n_unique_test_ext}")

pitch_gap = abs(np.mean(test_pitches) - np.mean(train_pitches))
print(f"\nTrain→Test Pitch Mean Gap: {pitch_gap:.3f}°")

n_pass = sum(1 for s in seq_stats if s['pitch'] < 0.1)
n_close = sum(1 for s in seq_stats if 0.1 <= s['pitch'] < 0.25)
n_fail = sum(1 for s in seq_stats if s['pitch'] >= 0.25)
print(f"\nSequences achieving <0.1°: {n_pass}/{len(seq_stats)}")
print(f"Sequences close (0.1-0.25°): {n_close}/{len(seq_stats)}")
print(f"Sequences failing (>0.25°): {n_fail}/{len(seq_stats)}")

print(f"\nAll figures saved to: {OUTPUT_DIR}/")
print("Done!")
