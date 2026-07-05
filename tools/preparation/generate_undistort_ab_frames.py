#!/usr/bin/env python3
"""Per-frame OpenCV vs C++ undistort projection comparison from ONE synced dataset.

Each frame uses the SAME raw fisheye image from bag (matched by times.txt),
then applies both undistort modes and projects with paired K.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = TOOLS_DIR.parent / 'validation'
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(VALIDATION_DIR))

from prepare_custom_dataset import ConfigParser, ProtobufUtils  # noqa: E402
from undistort_utils import (  # noqa: E402
    build_cpp_undistort_rectify_maps,
    build_opencv_fisheye_undistort_maps,
    get_cpp_focal_scale,
    scale_intrinsics,
    undistort_image,
)
from validation_utils import list_sequence_images  # noqa: E402


def decode_jpeg_from_proto(data: bytes) -> Optional[np.ndarray]:
    j = data.find(b'\xff\xd8')
    e = data.rfind(b'\xff\xd9')
    if j < 0 or e <= j:
        return None
    img = cv2.imdecode(np.frombuffer(data[j:e + 2], np.uint8), cv2.IMREAD_COLOR)
    return img


def extract_string_msg(rawdata: bytes) -> Optional[bytes]:
    import struct
    try:
        if len(rawdata) < 4:
            return None
        length = struct.unpack('<I', rawdata[:4])[0]
        if 0 < length < len(rawdata):
            return rawdata[4:4 + length]
        return rawdata
    except Exception:
        return rawdata


def load_calib(calib_file: Path) -> dict:
    calib = {}
    with open(calib_file) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            try:
                calib[k.strip()] = np.array([float(x) for x in v.split()])
            except ValueError:
                calib[k.strip()] = v.strip()
    return calib


def load_target_timestamps(seq_dir: Path, max_frames: int) -> List[float]:
    times_file = seq_dir / 'times.txt'
    if times_file.exists():
        ts = [float(line.strip()) for line in open(times_file) if line.strip()]
        return ts[:max_frames]
    # fallback: frame count only
    n = len(list_sequence_images(seq_dir / 'image_2'))
    return [float(i) for i in range(min(n, max_frames))]


def index_bag_images(
    bag_dir: Path,
    camera_name: str,
    ts_lo: float,
    ts_hi: float,
) -> List[Tuple[float, np.ndarray]]:
    from rosbags.rosbag1 import Reader

    topics = [
        f'/sensors/camera/{camera_name}_raw_data/compressed_proto',
        f'/sensors/camera/{camera_name}/compressed_proto',
    ]
    bags = sorted(bag_dir.rglob('*.bag'))
    if not bags:
        raise FileNotFoundError(f'no .bag under {bag_dir}')

    out: List[Tuple[float, np.ndarray]] = []
    margin = 0.15
    lo, hi = ts_lo - margin, ts_hi + margin

    for bag_path in bags:
        try:
            with Reader(str(bag_path)) as reader:
                active = [t for t in topics if t in reader.topics]
                if not active:
                    continue
                for conn, bag_ts, rawdata in reader.messages():
                    if conn.topic not in active:
                        continue
                    data = extract_string_msg(rawdata)
                    if not data:
                        continue
                    header_ts = ProtobufUtils.extract_header_timestamp(data)
                    ts_sec = header_ts if header_ts is not None else bag_ts / 1e9
                    if ts_sec < lo or ts_sec > hi:
                        continue
                    img = decode_jpeg_from_proto(data)
                    if img is not None:
                        out.append((ts_sec, img))
        except Exception as e:
            print(f'  warn: skip {bag_path.name}: {e}')
    out.sort(key=lambda x: x[0])
    print(f'  indexed {len(out)} raw images in [{ts_lo:.3f}, {ts_hi:.3f}] from {len(bags)} bags')
    return out


def match_image(
    target_ts: float,
    indexed: List[Tuple[float, np.ndarray]],
    max_diff: float = 0.055,
) -> Tuple[Optional[np.ndarray], float]:
    best_img, best_ts, best_dt = None, None, float('inf')
    for ts, img in indexed:
        dt = abs(ts - target_ts)
        if dt < best_dt:
            best_dt, best_img, best_ts = dt, img, ts
    if best_img is None or best_dt > max_diff:
        return None, best_dt
    return best_img, best_ts


def project_points(points: np.ndarray, Tr: np.ndarray, K: np.ndarray, img_shape):
    T = np.vstack([Tr.reshape(3, 4), [0, 0, 0, 1]])
    pc = (np.linalg.inv(T) @ np.hstack([points[:, :3], np.ones((len(points), 1))]).T).T[:, :3]
    pc = pc[pc[:, 2] > 0]
    P = np.hstack([K, np.zeros((3, 1))])
    uv = (P @ np.hstack([pc, np.ones((len(pc), 1))]).T).T
    uv = uv[:, :2] / uv[:, 2:3]
    depths = pc[:, 2]
    h, w = img_shape[:2]
    m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[m], depths[m], int(m.sum()), len(points)


def build_maps(config_dir: Path, camera_name: str, output_size: Tuple[int, int]):
    cameras = ConfigParser.parse_cameras_cfg(str(config_dir / 'cameras.cfg'))
    cam = cameras[camera_name]
    intr, dist = cam['intrinsic'], cam['distortion']
    w, h = int(intr['img_width']), int(intr['img_height'])
    K = np.array([[intr['f_x'], 0, intr['o_x']],
                  [0, intr['f_y'], intr['o_y']],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([dist['k1'], dist['k2'], dist['k3'], dist['k4']], dtype=np.float64)
    focal_scale = get_cpp_focal_scale(camera_name)

    map_o1, map_o2, K_o = build_opencv_fisheye_undistort_maps(K, D, w, h)
    map_c1, map_c2, K_c = build_cpp_undistort_rectify_maps(
        K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0], D[1], D[2], D[3], w, h, focal_scale)

    ow, oh = output_size
    K_o_out = scale_intrinsics(K_o, ow, oh, w, h)
    K_c_out = scale_intrinsics(K_c, ow, oh, w, h)
    return (map_o1, map_o2, K_o_out), (map_c1, map_c2, K_c_out), focal_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True,
                        help='已同步数据集（提供 times/velodyne/Tr，帧序以此为准）')
    parser.add_argument('--config_dir', required=True)
    parser.add_argument('--bag_dir', required=True)
    parser.add_argument('--camera_name', default='camera_1')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sequence', default='00')
    parser.add_argument('--max_frames', type=int, default=10)
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)
    parser.add_argument('--max_time_diff', type=float, default=0.055)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    seq_dir = dataset_root / 'sequences' / args.sequence
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_size = (args.output_width, args.output_height)

    calib = load_calib(seq_dir / 'calib.txt')
    Tr = calib['Tr']

    target_ts = load_target_timestamps(seq_dir, args.max_frames)
    if not target_ts:
        raise RuntimeError('no frames in dataset')

    print('Building undistort maps...')
    (map_o1, map_o2, K_o), (map_c1, map_c2, K_c), focal_scale = build_maps(
        Path(args.config_dir), args.camera_name, output_size)

    print('Indexing raw fisheye from bag...')
    indexed = index_bag_images(
        Path(args.bag_dir), args.camera_name, min(target_ts), max(target_ts))

    images = list_sequence_images(seq_dir / 'image_2')[:args.max_frames]
    summary = []

    for img_path in images:
        frame = int(img_path.stem)
        ts_target = target_ts[frame] if frame < len(target_ts) else None
        pc_path = seq_dir / 'velodyne' / f'{frame:06d}.bin'
        points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)

        raw, matched_ts = match_image(ts_target, indexed, args.max_time_diff)
        if raw is None:
            print(f'  skip frame {frame:06d}: no raw image for ts={ts_target:.6f}')
            continue

        img_o = undistort_image(raw, map_o1, map_o2, output_size)
        img_c = undistort_image(raw, map_c1, map_c2, output_size)

        fig, axes = plt.subplots(1, 2, figsize=(19.2, 5.4), dpi=100)
        stats = []
        for ax, img, K, label in [
            (axes[0], img_o, K_o, 'OpenCV (balance=0)'),
            (axes[1], img_c, K_c, f'C++ Equidistant (focal_scale={focal_scale})'),
        ]:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            uv, depths, nin, ntot = project_points(points, Tr, K, rgb.shape)
            ax.imshow(rgb)
            if len(uv):
                ax.scatter(uv[:, 0], uv[:, 1], c=depths, cmap='jet', s=0.4, alpha=0.55)
            bottom = int((uv[:, 1] > rgb.shape[0] * 2 / 3).sum()) if len(uv) else 0
            u_lo = f'{uv[:,0].min():.0f}' if len(uv) else '-'
            u_hi = f'{uv[:,0].max():.0f}' if len(uv) else '-'
            ax.set_title(
                f'{label}\nfx={K[0,0]:.0f} cx={K[0,2]:.0f}  '
                f'in={nin}/{ntot}  bottom={bottom}  u=[{u_lo},{u_hi}]',
                fontsize=9,
            )
            ax.axis('off')
            stats.append((nin, bottom))

        fig.suptitle(
            f'Frame {frame:06d} — same raw fisheye (bag ts={matched_ts:.6f})',
            fontsize=12,
        )
        plt.tight_layout()
        out_path = out_dir / f'frame_{frame:06d}_ab.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()

        summary.append((frame, matched_ts, stats[0][0], stats[1][0], stats[0][1], stats[1][1],
                        K_o[0, 0], K_c[0, 0]))
        print(f'  frame {frame:06d}: raw_ts={matched_ts:.6f} opencv_in={stats[0][0]} '
              f'cpp_in={stats[1][0]} -> {out_path.name}')

    report = out_dir / 'SUMMARY.md'
    with open(report, 'w') as f:
        f.write('# Undistort A/B per-frame (same raw fisheye from bag)\n\n')
        f.write('左/右均从**同一帧 raw 鱼眼**去畸变，各自 K 与图像配对。\n\n')
        f.write('| frame | bag image ts | opencv in | cpp in | opencv bottom3 | cpp bottom3 | opencv fx | cpp fx |\n')
        f.write('|-------|--------------|-----------|--------|----------------|-------------|-----------|--------|\n')
        for row in summary:
            f.write(
                f'| {row[0]:06d} | {row[1]:.6f} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | '
                f'{row[6]:.1f} | {row[7]:.1f} |\n'
            )
    print(f'  report: {report}')


if __name__ == '__main__':
    main()
