#!/usr/bin/env python3
"""
Compare augmented (V16) vs non-augmented (V20) models on slope vs flat frames.
Goal: verify if augmentation reduces slope-related errors.
"""

import os
import numpy as np

DATA_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
EVAL_ROOT = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/generalization_eval_test_models2_v2"

SEQ_INFO = {
    '00': "B26A1-1", '01': "C01-60", '02': "C01T-45", '03': "D037-3",
    '04': "DE061-5", '05': "DE07-5", '06': "DE08-8", '07': "EC15S-8",
    '08': "LPD19A-4", '09': "M81-31", '10': "P03-4", '11': "P789-22"
}

MODELS = {
    'v20-pitch-wt3': {'dir': 'v20-v8recipe-pitch-wt3', 'aug': 'none', 'backend': 'spconv'},
    'v20-z10':       {'dir': 'v20-v8recipe-z10',       'aug': 'none', 'backend': 'spconv'},
    'v16-ultimate':  {'dir': 'v16-ultimate',            'aug': 'full', 'backend': 'drcv'},
    'v16-a30-ult':   {'dir': 'v16-a30-ultimate',        'aug': 'full', 'backend': 'drcv'},
    'v16-a30-sf':    {'dir': 'v16-a30-pitch3-signflip', 'aug': 'signflip', 'backend': 'drcv'},
    'v16-sf':        {'dir': 'v16-pitch3-signflip',     'aug': 'signflip', 'backend': 'drcv'},
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


def parse_errors_by_component(filepath):
    """Parse error file, return {sample_id: {'total': X, 'roll': X, 'pitch': X, 'yaw': X}}."""
    errors = {}
    current_sample = None

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('Sample '):
                parts = stripped.split()
                try:
                    current_sample = int(parts[1].rstrip(',').lstrip('0') or '0')
                    errors[current_sample] = {}
                except (IndexError, ValueError):
                    current_sample = None
            elif current_sample is not None:
                if stripped.startswith('Total:'):
                    try:
                        errors[current_sample]['total'] = float(stripped.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif 'Roll' in stripped and 'LiDAR' in stripped:
                    try:
                        errors[current_sample]['roll'] = float(stripped.split()[-2])
                    except (IndexError, ValueError):
                        pass
                elif 'Pitch' in stripped and 'LiDAR' in stripped:
                    try:
                        errors[current_sample]['pitch'] = float(stripped.split()[-2])
                    except (IndexError, ValueError):
                        pass
                elif 'Yaw' in stripped and 'LiDAR' in stripped:
                    try:
                        errors[current_sample]['yaw'] = float(stripped.split()[-2])
                    except (IndexError, ValueError):
                        pass

    return errors


def get_slope_labels():
    """Compute slope label for each of the 4800 test samples."""
    slope_labels = np.zeros(4800)
    seq_n_frames = {}
    seq_slopes = {}

    for seq_idx, seq_str in enumerate(sorted(SEQ_INFO.keys())):
        pose_file = os.path.join(DATA_ROOT, 'poses', f'{seq_str}.txt')
        if not os.path.exists(pose_file):
            continue
        poses = load_poses(pose_file)
        slopes = compute_pitch_slope(poses, window=10)
        n_total = len(slopes)
        seq_n_frames[seq_str] = n_total
        seq_slopes[seq_str] = slopes

        for s in range(400):
            sample_id = seq_idx * 400 + s
            frame_idx = min(s * n_total // 400, n_total - 1)
            slope_labels[sample_id] = slopes[frame_idx]

    return slope_labels, seq_n_frames, seq_slopes


def main():
    slope_th = 0.5

    print("=" * 110)
    print(f"AUGMENTATION vs SLOPE: Do augmented models handle slopes better?")
    print(f"Slope threshold: |pitch_slope| > {slope_th}°")
    print("=" * 110)

    slope_labels, _, _ = get_slope_labels()

    all_model_results = {}

    for model_name, model_info in MODELS.items():
        err_file = os.path.join(EVAL_ROOT, model_info['dir'], 'extrinsics_and_errors.txt')
        if not os.path.exists(err_file):
            continue

        errors = parse_errors_by_component(err_file)
        if not errors:
            continue

        flat_total = []
        slope_total = []
        flat_pitch = []
        slope_pitch = []

        for sid in range(4800):
            if sid not in errors or 'total' not in errors[sid]:
                continue
            e = errors[sid]
            if abs(slope_labels[sid]) <= slope_th:
                flat_total.append(e.get('total', 0))
                flat_pitch.append(e.get('pitch', 0))
            else:
                slope_total.append(e.get('total', 0))
                slope_pitch.append(e.get('pitch', 0))

        flat_total = np.array(flat_total) if flat_total else np.zeros(1)
        slope_total = np.array(slope_total) if slope_total else np.zeros(1)
        flat_pitch = np.array(flat_pitch) if flat_pitch else np.zeros(1)
        slope_pitch = np.array(slope_pitch) if slope_pitch else np.zeros(1)

        result = {
            'flat_n': len(flat_total),
            'slope_n': len(slope_total),
            'flat_total_mean': np.mean(flat_total),
            'slope_total_mean': np.mean(slope_total),
            'flat_pitch_mean': np.mean(flat_pitch),
            'slope_pitch_mean': np.mean(slope_pitch),
            'flat_total_p95': np.percentile(flat_total, 95),
            'slope_total_p95': np.percentile(slope_total, 95),
            'total_penalty': np.mean(slope_total) - np.mean(flat_total),
            'pitch_penalty': np.mean(slope_pitch) - np.mean(flat_pitch),
            'aug': model_info['aug'],
            'backend': model_info['backend'],
        }
        all_model_results[model_name] = result

    # Global comparison table
    print(f"\n{'Model':<22} {'Aug':>8} {'Back':>7} "
          f"{'Flat(n)':>7} {'FlatMean':>9} {'FlatP95':>8} "
          f"{'Slope(n)':>8} {'SlopeMean':>10} {'SlopeP95':>9} "
          f"{'Penalty':>8} {'Penalty%':>9}")
    print("-" * 110)

    for model_name in MODELS.keys():
        if model_name not in all_model_results:
            continue
        r = all_model_results[model_name]
        pct = r['total_penalty'] / r['flat_total_mean'] * 100 if r['flat_total_mean'] > 0 else 0
        print(f"{model_name:<22} {r['aug']:>8} {r['backend']:>7} "
              f"{r['flat_n']:>7} {r['flat_total_mean']:>9.3f} {r['flat_total_p95']:>8.3f} "
              f"{r['slope_n']:>8} {r['slope_total_mean']:>10.3f} {r['slope_total_p95']:>9.3f} "
              f"{r['total_penalty']:>+8.3f} {pct:>+8.1f}%")

    # Pitch-specific penalty
    print(f"\n{'Model':<22} {'Aug':>8} "
          f"{'FlatPitch':>10} {'SlopePitch':>11} {'PitchPenalty':>13} {'PitchPenalty%':>14}")
    print("-" * 80)

    for model_name in MODELS.keys():
        if model_name not in all_model_results:
            continue
        r = all_model_results[model_name]
        pct = r['pitch_penalty'] / r['flat_pitch_mean'] * 100 if r['flat_pitch_mean'] > 0 else 0
        print(f"{model_name:<22} {r['aug']:>8} "
              f"{r['flat_pitch_mean']:>10.3f} {r['slope_pitch_mean']:>11.3f} "
              f"{r['pitch_penalty']:>+13.3f} {pct:>+13.1f}%")

    # Per-sequence breakdown for key sequences
    print(f"\n{'='*110}")
    print("PER-SEQUENCE: Augmented vs Non-augmented on Problem Sequences")
    print(f"{'='*110}")

    focus_seqs = ['03', '05', '07', '08']
    focus_models = ['v20-pitch-wt3', 'v16-ultimate', 'v16-a30-ult', 'v16-a30-sf']

    for seq_str in focus_seqs:
        seq_idx = int(seq_str)
        print(f"\n  Seq {seq_str} ({SEQ_INFO[seq_str]}):")
        print(f"    {'Model':<22} {'Aug':>8} {'FlatN':>5} {'FlatMean':>9} {'SlopeN':>6} "
              f"{'SlopeMean':>10} {'Penalty':>8} {'%':>7}")

        for model_name in focus_models:
            if model_name not in all_model_results:
                continue
            model_info = MODELS[model_name]
            err_file = os.path.join(EVAL_ROOT, model_info['dir'], 'extrinsics_and_errors.txt')
            errors = parse_errors_by_component(err_file)

            flat_e = []
            slope_e = []
            for s in range(400):
                sid = seq_idx * 400 + s
                if sid not in errors or 'total' not in errors[sid]:
                    continue
                if abs(slope_labels[sid]) <= slope_th:
                    flat_e.append(errors[sid]['total'])
                else:
                    slope_e.append(errors[sid]['total'])

            flat_e = np.array(flat_e) if flat_e else np.zeros(1)
            slope_e = np.array(slope_e) if slope_e else np.zeros(1)
            penalty = np.mean(slope_e) - np.mean(flat_e)
            pct = penalty / np.mean(flat_e) * 100 if np.mean(flat_e) > 0 else 0
            print(f"    {model_name:<22} {model_info['aug']:>8} {len(flat_e):>5} "
                  f"{np.mean(flat_e):>9.3f} {len(slope_e):>6} {np.mean(slope_e):>10.3f} "
                  f"{penalty:>+8.3f} {pct:>+6.1f}%")

    # Key insight: does augmentation reduce slope penalty?
    print(f"\n{'='*110}")
    print("KEY QUESTION: Does augmentation reduce the slope penalty?")
    print(f"{'='*110}")

    if 'v20-pitch-wt3' in all_model_results:
        v20 = all_model_results['v20-pitch-wt3']
        print(f"\n  V20 (no aug):       slope penalty = {v20['total_penalty']:+.3f}° "
              f"({v20['total_penalty']/v20['flat_total_mean']*100:+.1f}%)")
        print(f"                      pitch penalty = {v20['pitch_penalty']:+.3f}° "
              f"({v20['pitch_penalty']/v20['flat_pitch_mean']*100:+.1f}%)")

    for mn in ['v16-ultimate', 'v16-a30-ult', 'v16-a30-sf']:
        if mn in all_model_results:
            r = all_model_results[mn]
            print(f"\n  {mn} ({r['aug']}): slope penalty = {r['total_penalty']:+.3f}° "
                  f"({r['total_penalty']/r['flat_total_mean']*100:+.1f}%)")
            print(f"  {'':>22}  pitch penalty = {r['pitch_penalty']:+.3f}° "
                  f"({r['pitch_penalty']/r['flat_pitch_mean']*100:+.1f}%)")

    print(f"\n  Conclusion:", end="")
    if 'v20-pitch-wt3' in all_model_results and 'v16-a30-ult' in all_model_results:
        v20_p = all_model_results['v20-pitch-wt3']['total_penalty']
        v16_p = all_model_results['v16-a30-ult']['total_penalty']
        if abs(v16_p) < abs(v20_p):
            reduction = (1 - abs(v16_p) / abs(v20_p)) * 100
            print(f" V16 augmented model reduces slope penalty by {reduction:.0f}%")
        else:
            print(f" Augmentation does NOT reduce slope penalty (V16={v16_p:+.3f} vs V20={v20_p:+.3f})")
    print()


if __name__ == '__main__':
    main()
