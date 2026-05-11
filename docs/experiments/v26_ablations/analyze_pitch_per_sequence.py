#!/usr/bin/env python3
"""Analyze per-sequence Pitch errors from evaluation data.

Parses extrinsics_and_errors.txt to find which sequences have the worst Pitch
errors, computing per-sequence statistics and identifying systematic patterns.
"""
import re
import sys
import os
from collections import defaultdict
from pathlib import Path

def parse_eval_file(filepath):
    """Parse extrinsics_and_errors.txt for per-sample RPY errors."""
    samples = []
    current_seq = None

    with open(filepath, 'r') as f:
        content = f.read()

    sample_pattern = re.compile(
        r'Sample (\d+) \[Seq (\d+)\].*?'
        r'Roll\s+\(LiDAR X\):\s+([\d.]+) deg.*?'
        r'Pitch \(LiDAR Y\):\s+([\d.]+) deg.*?'
        r'Yaw\s+\(LiDAR Z\):\s+([\d.]+) deg',
        re.DOTALL
    )

    for m in sample_pattern.finditer(content):
        samples.append({
            'sample_id': int(m.group(1)),
            'seq': int(m.group(2)),
            'roll': float(m.group(3)),
            'pitch': float(m.group(4)),
            'yaw': float(m.group(5)),
        })

    return samples


def analyze_sequences(samples):
    """Compute per-sequence statistics."""
    seq_data = defaultdict(lambda: {'roll': [], 'pitch': [], 'yaw': []})

    for s in samples:
        seq_data[s['seq']]['roll'].append(s['roll'])
        seq_data[s['seq']]['pitch'].append(s['pitch'])
        seq_data[s['seq']]['yaw'].append(s['yaw'])

    results = []
    for seq_id in sorted(seq_data.keys()):
        d = seq_data[seq_id]
        n = len(d['pitch'])
        pitch_vals = sorted(d['pitch'])
        roll_vals = sorted(d['roll'])
        yaw_vals = sorted(d['yaw'])

        results.append({
            'seq': seq_id,
            'n': n,
            'pitch_mean': sum(d['pitch']) / n,
            'pitch_median': pitch_vals[n // 2],
            'pitch_p95': pitch_vals[int(n * 0.95)],
            'pitch_max': pitch_vals[-1],
            'pitch_std': (sum((x - sum(d['pitch'])/n)**2 for x in d['pitch']) / n) ** 0.5,
            'roll_mean': sum(d['roll']) / n,
            'yaw_mean': sum(d['yaw']) / n,
            'pitch_pct_gt_0.5': sum(1 for x in d['pitch'] if x > 0.5) / n * 100,
            'pitch_pct_gt_1.0': sum(1 for x in d['pitch'] if x > 1.0) / n * 100,
        })

    return results


def main():
    eval_dirs = [
        'logs/evaluations/generalization_eval_v24/v24-B-diff-only',
        'logs/evaluations/generalization_eval_v24/v24-A-baseline',
    ]

    base_dir = Path(__file__).resolve().parent.parent.parent

    for eval_dir in eval_dirs:
        filepath = base_dir / eval_dir / 'extrinsics_and_errors.txt'
        if not filepath.exists():
            print(f"Skipping {eval_dir}: file not found")
            continue

        model_name = eval_dir.split('/')[-1]
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        samples = parse_eval_file(filepath)
        if not samples:
            print("  No samples parsed!")
            continue

        results = analyze_sequences(samples)

        # Sort by pitch_mean descending (worst first)
        results.sort(key=lambda x: -x['pitch_mean'])

        print(f"\n{'Seq':>4} | {'N':>5} | {'Pitch Mean':>10} | {'Pitch Med':>9} | "
              f"{'Pitch P95':>9} | {'Pitch Max':>9} | {'Pitch Std':>9} | "
              f"{'Roll Mean':>9} | {'Yaw Mean':>9} | {'>0.5°':>6} | {'>1.0°':>6}")
        print("-" * 120)

        for r in results:
            print(f"  {r['seq']:02d} | {r['n']:5d} | {r['pitch_mean']:10.4f} | "
                  f"{r['pitch_median']:9.4f} | {r['pitch_p95']:9.4f} | "
                  f"{r['pitch_max']:9.4f} | {r['pitch_std']:9.4f} | "
                  f"{r['roll_mean']:9.4f} | {r['yaw_mean']:9.4f} | "
                  f"{r['pitch_pct_gt_0.5']:5.1f}% | {r['pitch_pct_gt_1.0']:5.1f}%")

        # Overall statistics
        all_pitch = [s['pitch'] for s in samples]
        n_total = len(all_pitch)
        mean_pitch = sum(all_pitch) / n_total
        print(f"\nOverall: {n_total} samples, Mean Pitch={mean_pitch:.4f}°, "
              f">0.5°: {sum(1 for x in all_pitch if x > 0.5)/n_total*100:.1f}%, "
              f">1.0°: {sum(1 for x in all_pitch if x > 1.0)/n_total*100:.1f}%")

        # Identify worst sequences
        worst = results[0]
        best = results[-1]
        gap = worst['pitch_mean'] - best['pitch_mean']
        print(f"\nWorst seq: {worst['seq']:02d} (Pitch={worst['pitch_mean']:.4f}°)")
        print(f"Best  seq: {best['seq']:02d} (Pitch={best['pitch_mean']:.4f}°)")
        print(f"Gap: {gap:.4f}° ({gap/best['pitch_mean']*100:.1f}% relative)")


if __name__ == '__main__':
    main()
