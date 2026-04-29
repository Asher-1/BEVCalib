#!/usr/bin/env python3
"""
Focused analysis: pitch-only slope detection from poses.
Roll estimation from trajectory is unreliable (singularity during turns).
"""

import os
import numpy as np

DATA_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
SEQ_INFO = {
    '00': "B26A1-1", '01': "C01-60", '02': "C01T-45", '03': "D037-3",
    '04': "DE061-5", '05': "DE07-5", '06': "DE08-8", '07': "EC15S-8",
    '08': "LPD19A-4", '09': "M81-31", '10': "P03-4", '11': "P789-22"
}


def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


def compute_pitch_slope(poses, window=10):
    """Compute pitch slope from height gradient along trajectory."""
    positions = np.array([T[:3, 3] for T in poses])
    n = len(positions)
    slopes = np.zeros(n)
    
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        pos = positions[lo:hi]
        
        if len(pos) < 3:
            continue
        
        dx = pos[-1, 0] - pos[0, 0]
        dy = pos[-1, 1] - pos[0, 1]
        dz = pos[-1, 2] - pos[0, 2]
        horiz = np.sqrt(dx**2 + dy**2)
        
        if horiz > 0.5:
            slopes[i] = np.degrees(np.arctan2(dz, horiz))
    
    return slopes


def main():
    print("=" * 100)
    print("PITCH-ONLY SLOPE ANALYSIS (window=10, roll ignored)")
    print("=" * 100)
    
    # Threshold sweep
    thresholds = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    all_slopes = {}
    for seq in sorted(SEQ_INFO.keys()):
        pose_file = os.path.join(DATA_ROOT, 'poses', f'{seq}.txt')
        if not os.path.exists(pose_file):
            continue
        poses = load_poses(pose_file)
        if len(poses) < 10:
            continue
        slopes = compute_pitch_slope(poses, window=10)
        all_slopes[seq] = slopes
    
    # Per-sequence stats
    print(f"\n{'Seq':>4} {'Vehicle':>12} {'Frames':>6} {'MeanAbs':>8} {'MaxAbs':>8} {'P95':>6} {'P99':>6}")
    for seq in sorted(all_slopes.keys()):
        s = all_slopes[seq]
        abs_s = np.abs(s)
        print(f"{seq:>4} {SEQ_INFO[seq]:>12} {len(s):>6} {np.mean(abs_s):>8.3f} {np.max(abs_s):>8.3f} "
              f"{np.percentile(abs_s, 95):>6.3f} {np.percentile(abs_s, 99):>6.3f}")
    
    # Threshold sweep table
    print(f"\n{'Threshold':>10}", end="")
    for seq in sorted(all_slopes.keys()):
        print(f" {seq:>6}", end="")
    print(f" {'TOTAL':>8} {'%Removed':>9}")
    
    for th in thresholds:
        print(f"  >{th:.1f}°    ", end="")
        total_removed = 0
        total_frames = 0
        for seq in sorted(all_slopes.keys()):
            s = all_slopes[seq]
            n_removed = np.sum(np.abs(s) > th)
            pct = 100 * n_removed / len(s)
            total_removed += n_removed
            total_frames += len(s)
            print(f" {pct:>5.0f}%", end="")
        
        total_pct = 100 * total_removed / total_frames
        print(f" {total_removed:>5}/{total_frames:<5} {total_pct:>7.1f}%")
    
    # Per-sequence @ 0.5° threshold (practical choice)
    th = 0.5
    print(f"\n{'='*100}")
    print(f"RECOMMENDED: |pitch_slope| > {th}° filter")
    print(f"{'='*100}")
    print(f"{'Seq':>4} {'Vehicle':>12} {'Total':>6} {'Flat':>5} {'Slope':>5} {'Removed%':>9} {'400->':>5}")
    
    total_f = 0
    total_flat = 0
    for seq in sorted(all_slopes.keys()):
        s = all_slopes[seq]
        n = len(s)
        n_flat = int(np.sum(np.abs(s) <= th))
        n_slope = n - n_flat
        remain = int(400 * n_flat / n)
        total_f += n
        total_flat += n_flat
        print(f"  {seq:>2} {SEQ_INFO[seq]:>12} {n:>6} {n_flat:>5} {n_slope:>5} "
              f"{100*n_slope/n:>8.1f}% {remain:>5}")
    
    print(f"  {'':>2} {'TOTAL':>12} {total_f:>6} {total_flat:>5} {total_f-total_flat:>5} "
          f"{100*(total_f-total_flat)/total_f:>8.1f}% {int(4800*total_flat/total_f):>5}")
    
    # Cross-reference with error data
    print(f"\n{'='*100}")
    print("CROSS-REFERENCE: Slope vs Error for Problem Sequences")
    print(f"{'='*100}")
    
    error_files = {
        'pitch-wt3': '/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/'
                     'generalization_eval_test_models2_v2/v20-v8recipe-pitch-wt3/extrinsics_and_errors.txt',
    }
    
    for model_name, err_file in error_files.items():
        if not os.path.exists(err_file):
            continue
        
        errors = parse_errors_quick(err_file)
        if not errors:
            continue
        
        print(f"\n  Model: {model_name}")
        print(f"  Sampled 400 frames per seq from total frames")
        
        for seq in ['03', '05', '07', '08']:
            if seq not in all_slopes:
                continue
            
            slopes = all_slopes[seq]
            n_total = len(slopes)
            
            seq_idx = int(seq)
            seq_errors = {k: v for k, v in errors.items() if k // 400 == seq_idx}
            
            if not seq_errors:
                continue
            
            flat_errors = []
            slope_errors = []
            
            for sample_id, err in seq_errors.items():
                frame_idx = (sample_id % 400) * n_total // 400
                frame_idx = min(frame_idx, n_total - 1)
                
                if abs(slopes[frame_idx]) <= th:
                    flat_errors.append(err)
                else:
                    slope_errors.append(err)
            
            flat_errors = np.array(flat_errors) if flat_errors else np.array([0])
            slope_errors = np.array(slope_errors) if slope_errors else np.array([0])
            
            print(f"\n  Seq {seq} ({SEQ_INFO[seq]}):")
            print(f"    Flat  frames: n={len(flat_errors):>3}, mean={np.mean(flat_errors):.3f}°, "
                  f"max={np.max(flat_errors):.3f}°, P95={np.percentile(flat_errors, 95):.3f}°")
            print(f"    Slope frames: n={len(slope_errors):>3}, mean={np.mean(slope_errors):.3f}°, "
                  f"max={np.max(slope_errors):.3f}°, P95={np.percentile(slope_errors, 95):.3f}°")
            
            if len(flat_errors) > 0 and len(slope_errors) > 0:
                diff = np.mean(slope_errors) - np.mean(flat_errors)
                print(f"    Slope penalty: {diff:+.3f}° ({diff/np.mean(flat_errors)*100:+.0f}%)")


def parse_errors_quick(filepath):
    """Quick parser for extrinsics_and_errors.txt - extract total rotation error per sample."""
    errors = {}
    current_sample = None
    
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('Sample '):
                parts = stripped.split()
                try:
                    current_sample = int(parts[1].rstrip(',').lstrip('0') or '0')
                except (IndexError, ValueError):
                    current_sample = None
            elif stripped.startswith('Total:') and current_sample is not None:
                parts = stripped.split()
                try:
                    val = float(parts[1])
                    errors[current_sample] = val
                except (IndexError, ValueError):
                    pass
    
    return errors


if __name__ == '__main__':
    main()
