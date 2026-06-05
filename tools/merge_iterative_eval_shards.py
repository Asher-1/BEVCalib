#!/usr/bin/env python3
"""Merge sharded iterative eval JSON outputs into one summary."""

import argparse
import glob
import json
import os
import sys

import numpy as np


def _summarize_errors(errors_list):
    if not errors_list:
        return {}
    keys = ['rot_error', 'roll_error', 'pitch_error', 'yaw_error']
    out = {}
    for k in keys:
        vals = np.array([e[k] for e in errors_list], dtype=np.float64)
        out[k] = {
            'mean': float(vals.mean()),
            'median': float(np.median(vals)),
            'p90': float(np.percentile(vals, 90)),
            'p95': float(np.percentile(vals, 95)),
            'max': float(vals.max()),
        }
    rot = np.array([e['rot_error'] for e in errors_list])
    roll = np.array([e['roll_error'] for e in errors_list])
    pitch = np.array([e['pitch_error'] for e in errors_list])
    yaw = np.array([e['yaw_error'] for e in errors_list])
    out['pct_rot_lt_0p1'] = float((rot < 0.1).mean() * 100)
    out['pct_all_rpy_lt_0p1'] = float(((roll < 0.1) & (pitch < 0.1) & (yaw < 0.1)).mean() * 100)
    out['pct_rot_lt_0p5'] = float((rot < 0.5).mean() * 100)
    out['n_frames'] = len(errors_list)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_glob', required=True,
                        help='Glob pattern for shard JSON files')
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.shard_glob))
    if not shard_paths:
        print(f'[FATAL] No shard files match: {args.shard_glob}')
        sys.exit(1)

    merged_errors = {}
    meta = None
    for path in shard_paths:
        with open(path) as f:
            data = json.load(f)
        if meta is None:
            meta = {k: data.get(k) for k in (
                'checkpoint', 'epoch', 'dataset', 'perturbation_deg',
                'pointgpt_ckpt', 'pointgpt_config', 'num_shards')}
        raw = data.get('raw_errors_by_mode') or {}
        for mode, errs in raw.items():
            merged_errors.setdefault(mode, []).extend(errs)

    summary = dict(meta or {})
    summary['n_val_frames'] = len(next(iter(merged_errors.values()), []))
    summary['modes'] = {}
    summary['shard_files'] = shard_paths
    for mode, errs in merged_errors.items():
        summary['modes'][mode] = _summarize_errors(errs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Merged {len(shard_paths)} shards -> {args.output_json}')
    print(f'Frames: {summary["n_val_frames"]}')
    for mode, s in summary['modes'].items():
        print(f'  {mode}: rot_mean={s["rot_error"]["mean"]:.3f}°')


if __name__ == '__main__':
    main()
