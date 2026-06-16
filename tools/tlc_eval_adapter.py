#!/usr/bin/env python3
"""Export BEVCalib per-frame predictions to TLC-Calib rig format for metrics_pose.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kitti-bev-calib'))


def write_cams_to_lidar(path: str, rigs: list[np.ndarray], cam_ids: list[int] | None = None):
    """Write TLC-style cams_to_lidar.txt (one 4x4 per camera rig)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for i, rig in enumerate(rigs):
            cid = cam_ids[i] if cam_ids else i
            flat = rig.reshape(-1)
            f.write(f"# CAM {cid:02d}\n")
            for row in range(4):
                f.write(' '.join(f'{v:.8f}' for v in flat[row * 4:(row + 1) * 4]) + '\n')


def aggregate_predictions(pred_json: str, output_dir: str):
    """Convert evaluate_checkpoint JSON output to TLC rig file."""
    with open(pred_json) as f:
        data = json.load(f)
    rigs = []
    cam_ids = []
    for entry in data.get('frames', data.get('predictions', [])):
        T = np.array(entry.get('T_pred') or entry.get('c2l'), dtype=np.float64)
        if T.shape == (4, 4):
            rigs.append(T)
            cam_ids.append(int(entry.get('cam_id', 0)))
    if not rigs:
        raise ValueError(f"No frame predictions in {pred_json}")
    out = os.path.join(output_dir, 'cams_to_lidar.txt')
    write_cams_to_lidar(out, rigs, cam_ids)
    print(f"Wrote {len(rigs)} rig(s) to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description='BEVCalib → TLC-Calib rig export')
    parser.add_argument('--pred_json', required=True, help='BEVCalib eval JSON with T_pred per frame')
    parser.add_argument('--output_dir', required=True, help='Directory for cams_to_lidar.txt')
    args = parser.parse_args()
    aggregate_predictions(args.pred_json, args.output_dir)


if __name__ == '__main__':
    main()
