#!/usr/bin/env python3
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    rows = []
    for arm in ['kitti_pointgpt', 'nusc_pointgpt']:
        path = os.path.join(args.out_dir, arm, 'merged.json')
        with open(path) as f:
            d = json.load(f)
        s = d['modes'].get('single_forward') or next(iter(d['modes'].values()))
        rows.append((arm, s['rot_error']['mean'], s['rot_error']['median'],
                     s['roll_error']['mean'], s['pitch_error']['mean'], s['yaw_error']['mean'],
                     s.get('pct_rot_lt_0p1', 0.0), d['n_val_frames']))
        if 'iterative_3' in d['modes']:
            s3 = d['modes']['iterative_3']
            rows.append((arm + '_iter3', s3['rot_error']['mean'], s3['rot_error']['median'],
                         s3['roll_error']['mean'], s3['pitch_error']['mean'], s3['yaw_error']['mean'],
                         s3.get('pct_rot_lt_0p1', 0.0), d['n_val_frames']))

    print('\n=== PointGPT A/B Summary ===')
    print(f"{'Arm':<22} {'Rot mean':>9} {'Rot med':>9} {'R/P/Y':>18} {'<0.1°':>8} {'N':>6}")
    print('-' * 78)
    for arm, rot_m, rot_med, r, p, y, p01, n in rows:
        print(f'{arm:<22} {rot_m:9.3f} {rot_med:9.3f} {r:.2f}/{p:.2f}/{y:.2f} {p01:7.1f}% {n:6d}')

    kitti = next((r for r in rows if r[0] == 'kitti_pointgpt'), None)
    nusc = next((r for r in rows if r[0] == 'nusc_pointgpt'), None)
    if kitti and nusc:
        delta = (nusc[1] - kitti[1]) / kitti[1] * 100
        print(f'\nnuScenes vs KITTI (single_forward rot mean): {delta:+.2f}%')

    summary = {'arms': {}}
    for arm, rot_m, rot_med, r, p, y, p01, n in rows:
        summary['arms'][arm] = {
            'rot_mean': rot_m, 'rot_median': rot_med,
            'roll_mean': r, 'pitch_mean': p, 'yaw_mean': y,
            'pct_rot_lt_0p1': p01, 'n_frames': n,
        }
    out_path = os.path.join(args.out_dir, 'ab_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
