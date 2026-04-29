#!/usr/bin/env python3
"""Analyze worst samples from BEVCalib evaluation results for seq 07 and seq 08."""

import re
import os
import json
from collections import defaultdict

EVAL_BASE = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/generalization_eval_test_models2_v2"
MODELS = ["v20-v8recipe-pitch-wt3", "v20-v8recipe-z10"]
TARGET_SEQS = [7, 8]

def parse_errors(filepath):
    """Parse extrinsics_and_errors.txt and return per-sample errors."""
    samples = {}
    current_sample = None
    
    with open(filepath) as f:
        for line in f:
            m = re.match(r'Sample (\d+) \[Seq (\d+)\]', line)
            if m:
                current_sample = int(m.group(1))
                current_seq = int(m.group(2))
                samples[current_sample] = {'seq': current_seq}
                continue
            
            if current_sample is not None:
                m = re.match(r'\s+Total:\s+([\d.]+) deg', line)
                if m:
                    samples[current_sample]['total'] = float(m.group(1))
                m = re.match(r'\s+Roll\s+\(LiDAR X\):\s+([\d.]+) deg', line)
                if m:
                    samples[current_sample]['roll'] = float(m.group(1))
                m = re.match(r'\s+Pitch\s+\(LiDAR Y\):\s+([\d.]+) deg', line)
                if m:
                    samples[current_sample]['pitch'] = float(m.group(1))
                m = re.match(r'\s+Yaw\s+\(LiDAR Z\):\s+([\d.]+) deg', line)
                if m:
                    samples[current_sample]['yaw'] = float(m.group(1))
    
    return samples


def analyze_model(model_name, samples):
    """Analyze errors for target sequences."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    for seq in TARGET_SEQS:
        seq_samples = {k: v for k, v in samples.items() if v.get('seq') == seq}
        if not seq_samples:
            print(f"\n  Seq {seq:02d}: No samples found!")
            continue
        
        totals = [(k, v['total']) for k, v in seq_samples.items() if 'total' in v]
        totals.sort(key=lambda x: x[1], reverse=True)
        
        avg_total = sum(t for _, t in totals) / len(totals)
        max_total = totals[0][1]
        min_total = totals[-1][1]
        
        rolls = [v['roll'] for v in seq_samples.values() if 'roll' in v]
        pitches = [v['pitch'] for v in seq_samples.values() if 'pitch' in v]
        yaws = [v['yaw'] for v in seq_samples.values() if 'yaw' in v]
        
        seq_info = {7: "EC15S-8", 8: "LPD19A-4"}
        
        print(f"\n--- Seq {seq:02d} ({seq_info.get(seq, '?')}) ---")
        print(f"  Samples: {len(seq_samples)}")
        print(f"  Total Error: mean={avg_total:.4f}° | max={max_total:.4f}° | min={min_total:.4f}°")
        print(f"  Roll  mean={sum(rolls)/len(rolls):.4f}° | max={max(rolls):.4f}°")
        print(f"  Pitch mean={sum(pitches)/len(pitches):.4f}° | max={max(pitches):.4f}°")
        print(f"  Yaw   mean={sum(yaws)/len(yaws):.4f}° | max={max(yaws):.4f}°")
        
        print(f"\n  Top 20 worst samples (by Total Error):")
        print(f"  {'Sample':>8} {'Total°':>8} {'Roll°':>8} {'Pitch°':>8} {'Yaw°':>8} {'Dominant':>10}")
        for sample_id, total in totals[:20]:
            s = seq_samples[sample_id]
            r, p, y = s.get('roll', 0), s.get('pitch', 0), s.get('yaw', 0)
            dominant = 'Roll' if r >= p and r >= y else ('Pitch' if p >= y else 'Yaw')
            print(f"  {sample_id:>8d} {total:>8.4f} {r:>8.4f} {p:>8.4f} {y:>8.4f} {dominant:>10}")
        
        p95_idx = int(0.95 * len(totals))
        p90_idx = int(0.90 * len(totals))
        print(f"\n  P90={totals[p90_idx][1]:.4f}° | P95={totals[p95_idx][1]:.4f}° | Max={max_total:.4f}°")
        
        above_1deg = [(k, t) for k, t in totals if t > 1.0]
        above_1_5deg = [(k, t) for k, t in totals if t > 1.5]
        above_2deg = [(k, t) for k, t in totals if t > 2.0]
        print(f"  Samples >1.0°: {len(above_1deg)} ({100*len(above_1deg)/len(totals):.1f}%)")
        print(f"  Samples >1.5°: {len(above_1_5deg)} ({100*len(above_1_5deg)/len(totals):.1f}%)")
        print(f"  Samples >2.0°: {len(above_2deg)} ({100*len(above_2deg)/len(totals):.1f}%)")
        
        return_data = {}
        return_data[seq] = {
            'worst_samples': [(sid, seq_samples[sid]) for sid, _ in totals[:20]],
            'above_1_5deg': [(sid, seq_samples[sid]) for sid, _ in above_1_5deg],
        }


def find_global_worst(all_model_samples):
    """Find worst samples across all sequences and both models."""
    print(f"\n{'='*80}")
    print("GLOBAL WORST SAMPLES (>1.5° in any model)")
    print(f"{'='*80}")
    
    worst_by_model = {}
    for model_name, samples in all_model_samples.items():
        for seq in TARGET_SEQS:
            seq_samples = {k: v for k, v in samples.items() if v.get('seq') == seq}
            for sid, s in seq_samples.items():
                if s.get('total', 0) > 1.5:
                    key = (seq, sid)
                    if key not in worst_by_model:
                        worst_by_model[key] = {}
                    worst_by_model[key][model_name] = s
    
    sorted_worst = sorted(worst_by_model.items(), 
                         key=lambda x: max(v.get('total', 0) for v in x[1].values()), 
                         reverse=True)
    
    seq_info = {7: "EC15S-8", 8: "LPD19A-4"}
    for (seq, sid), model_data in sorted_worst:
        print(f"\n  Sample {sid:04d} [Seq {seq:02d} - {seq_info.get(seq, '?')}]:")
        for model_name, s in model_data.items():
            print(f"    {model_name}: Total={s['total']:.4f}° Roll={s.get('roll',0):.4f}° Pitch={s.get('pitch',0):.4f}° Yaw={s.get('yaw',0):.4f}°")
    
    return sorted_worst


def find_all_worst_global(all_model_samples):
    """Find the global max-error samples across ALL sequences."""
    print(f"\n{'='*80}")
    print("GLOBAL MAX-ERROR SAMPLES (all sequences, top 30)")
    print(f"{'='*80}")
    
    for model_name, samples in all_model_samples.items():
        print(f"\n--- {model_name} ---")
        totals = [(k, v) for k, v in samples.items() if 'total' in v]
        totals.sort(key=lambda x: x[1]['total'], reverse=True)
        
        seq_info = {0: "B26A1-1", 1: "C01-60", 2: "C01T-45", 3: "D037-3",
                    4: "DE061-5", 5: "DE07-5", 6: "DE08-8", 7: "EC15S-8",
                    8: "LPD19A-4", 9: "M81-31", 10: "P03-4", 11: "P789-22"}
        
        print(f"  {'Sample':>8} {'Seq':>4} {'Vehicle':>12} {'Total°':>8} {'Roll°':>8} {'Pitch°':>8} {'Yaw°':>8}")
        for sid, s in totals[:30]:
            seq = s['seq']
            vehicle = seq_info.get(seq, '?')
            print(f"  {sid:>8d} {seq:>4d} {vehicle:>12} {s['total']:>8.4f} {s.get('roll',0):>8.4f} {s.get('pitch',0):>8.4f} {s.get('yaw',0):>8.4f}")


if __name__ == '__main__':
    all_model_samples = {}
    
    for model in MODELS:
        filepath = os.path.join(EVAL_BASE, model, "extrinsics_and_errors.txt")
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        samples = parse_errors(filepath)
        all_model_samples[model] = samples
        analyze_model(model, samples)
    
    find_global_worst(all_model_samples)
    find_all_worst_global(all_model_samples)
    
    # Also output sample-to-frame mapping info
    print(f"\n{'='*80}")
    print("SAMPLE-TO-FRAME MAPPING")
    print(f"{'='*80}")
    print("Seq 07 (EC15S-8): samples 2800-3199 → frames sampled from 2698 total frames")
    print("Seq 08 (LPD19A-4): samples 3200-3599 → frames sampled from 970 total frames")
    print("\nTo find the actual frame index in the dataset:")
    print("  sample_offset = sample_id - seq_start")
    print("  Seq 07: frame_idx = sample_id - 2800") 
    print("  Seq 08: frame_idx = sample_id - 3200")
