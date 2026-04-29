#!/usr/bin/env python3
"""Analyze vehicle trajectory to detect turns/curves in test sequences."""

import os
import numpy as np

DATA_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"

def load_poses(pose_file):
    """Load KITTI-format poses (3x4 flattened per line)."""
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


def compute_heading_rate(poses, fps=10.0):
    """Compute heading change rate (yaw rate) from poses."""
    headings = []
    for T in poses:
        yaw = np.arctan2(T[0, 2], T[0, 0])
        headings.append(yaw)
    
    headings = np.array(headings)
    yaw_rates = np.diff(headings) * fps * 180 / np.pi
    
    for i in range(1, len(yaw_rates)):
        if abs(yaw_rates[i]) > 90:
            yaw_rates[i] = yaw_rates[i-1]
    
    return headings, yaw_rates


def compute_curvature(poses):
    """Compute path curvature from trajectory."""
    positions = np.array([T[:3, 3] for T in poses])
    
    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    
    headings = np.arctan2(dy, dx)
    dheading = np.diff(headings)
    dheading = np.where(dheading > np.pi, dheading - 2*np.pi, dheading)
    dheading = np.where(dheading < -np.pi, dheading + 2*np.pi, dheading)
    
    ds_mid = (ds[:-1] + ds[1:]) / 2
    ds_mid = np.where(ds_mid < 0.001, 0.001, ds_mid)
    curvature = dheading / ds_mid
    
    return curvature, ds


def analyze_sequence(seq_str):
    """Full trajectory analysis for a sequence."""
    pose_file = os.path.join(DATA_ROOT, 'poses', f'{seq_str}.txt')
    if not os.path.exists(pose_file):
        return None
    
    poses = load_poses(pose_file)
    if len(poses) < 10:
        return None
    
    positions = np.array([T[:3, 3] for T in poses])
    headings, yaw_rates = compute_heading_rate(poses)
    curvature, ds = compute_curvature(poses)
    
    speeds = ds * 10.0
    
    return {
        'n_frames': len(poses),
        'positions': positions,
        'headings': headings,
        'yaw_rates': yaw_rates,
        'curvature': curvature,
        'speeds': speeds,
    }


def classify_frames(yaw_rates, curvature, speed_threshold=0.5):
    """Classify frames as straight/curve/turn based on yaw rate."""
    n = len(yaw_rates)
    labels = ['straight'] * n
    
    for i in range(n):
        abs_yr = abs(yaw_rates[i])
        if abs_yr > 5.0:
            labels[i] = 'sharp_turn'
        elif abs_yr > 2.0:
            labels[i] = 'turn'
        elif abs_yr > 0.5:
            labels[i] = 'curve'
    
    return labels


SEQ_INFO = {
    '00': "B26A1-1", '01': "C01-60", '02': "C01T-45", '03': "D037-3",
    '04': "DE061-5", '05': "DE07-5", '06': "DE08-8", '07': "EC15S-8",
    '08': "LPD19A-4", '09': "M81-31", '10': "P03-4", '11': "P789-22"
}


def main():
    print("=" * 90)
    print("TEST DATA TRAJECTORY ANALYSIS - Turn/Curve Detection")
    print("=" * 90)
    
    all_seq_stats = {}
    
    for seq in sorted(SEQ_INFO.keys()):
        result = analyze_sequence(seq)
        if result is None:
            continue
        
        yr = result['yaw_rates']
        curv = result['curvature']
        labels = classify_frames(yr, curv)
        
        n = len(labels)
        n_straight = labels.count('straight')
        n_curve = labels.count('curve')
        n_turn = labels.count('turn')
        n_sharp = labels.count('sharp_turn')
        
        print(f"\nSeq {seq} ({SEQ_INFO[seq]:>12}): {result['n_frames']} frames")
        print(f"  Speed: mean={np.mean(result['speeds']):.1f} max={np.max(result['speeds']):.1f} m/s")
        print(f"  Yaw rate: mean={np.mean(np.abs(yr)):.2f} max={np.max(np.abs(yr)):.2f} deg/s")
        print(f"  Classification: straight={n_straight}({100*n_straight/n:.0f}%) "
              f"curve={n_curve}({100*n_curve/n:.0f}%) "
              f"turn={n_turn}({100*n_turn/n:.0f}%) "
              f"sharp={n_sharp}({100*n_sharp/n:.0f}%)")
        
        all_seq_stats[seq] = {
            'n_frames': result['n_frames'],
            'n_straight': n_straight,
            'n_curve': n_curve,
            'n_turn': n_turn,
            'n_sharp': n_sharp,
            'pct_non_straight': 100 * (n - n_straight) / n,
        }
    
    # Focus on Seq 03 burst region (frames 855-877)
    print("\n" + "=" * 90)
    print("SEQ 03 BURST REGION DETAIL (frames 840-895)")
    print("=" * 90)
    
    result03 = analyze_sequence('03')
    if result03:
        yr03 = result03['yaw_rates']
        sp03 = result03['speeds']
        
        print(f"  {'Frame':>6} {'Speed':>8} {'YawRate':>10} {'AbsYR':>8} {'Label':>12}")
        for i in range(840, min(895, len(yr03))):
            abs_yr = abs(yr03[i])
            if abs_yr > 5.0:
                label = 'SHARP_TURN'
            elif abs_yr > 2.0:
                label = 'TURN'
            elif abs_yr > 0.5:
                label = 'CURVE'
            else:
                label = 'straight'
            
            marker = " ***" if 855 <= i <= 877 else ""
            print(f"  {i:>6} {sp03[i] if i < len(sp03) else 0:>8.1f} {yr03[i]:>+10.2f} {abs_yr:>8.2f} {label:>12}{marker}")
    
    # Summary table for report
    print("\n" + "=" * 90)
    print("SUMMARY: Non-Straight Frame Percentages")
    print("=" * 90)
    print(f"  {'Seq':>4} {'Vehicle':>12} {'Total':>6} {'Straight':>8} {'Curve':>6} {'Turn':>5} {'Sharp':>5} {'NonStr%':>8}")
    for seq in sorted(all_seq_stats.keys()):
        s = all_seq_stats[seq]
        print(f"  {seq:>4} {SEQ_INFO[seq]:>12} {s['n_frames']:>6} {s['n_straight']:>8} "
              f"{s['n_curve']:>6} {s['n_turn']:>5} {s['n_sharp']:>5} {s['pct_non_straight']:>7.1f}%")
    
    # Impact analysis: what if we filter non-straight?
    print("\n" + "=" * 90)
    print("IMPACT ANALYSIS: Filtering Non-Straight Samples")
    print("=" * 90)
    
    total_frames = sum(s['n_frames'] for s in all_seq_stats.values())
    total_straight = sum(s['n_straight'] for s in all_seq_stats.values())
    total_removed = total_frames - total_straight
    
    print(f"  Total test frames: {total_frames}")
    print(f"  Straight frames:   {total_straight} ({100*total_straight/total_frames:.1f}%)")
    print(f"  Would be removed:  {total_removed} ({100*total_removed/total_frames:.1f}%)")
    
    # Per-sequence sampling impact
    print(f"\n  Sampling impact (400 samples per seq):")
    for seq in sorted(all_seq_stats.keys()):
        s = all_seq_stats[seq]
        straight_ratio = s['n_straight'] / s['n_frames']
        sampled_straight = int(400 * straight_ratio)
        print(f"    Seq {seq} ({SEQ_INFO[seq]:>12}): {sampled_straight}/400 samples remain "
              f"({100*straight_ratio:.0f}%), {400-sampled_straight} removed")


if __name__ == '__main__':
    main()
