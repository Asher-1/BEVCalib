#!/usr/bin/env python3
"""Convert a TLC-Calib scene to KITTI-Odometry layout for BEVCalib CustomDataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit("open3d is required: pip install open3d") from exc


def _read_intrinsics(path: Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split()])
    k = np.array(rows[:3], dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics in {path}, got {k.shape}")
    return k


def _read_rig_line(path: Path, cam_id: int) -> np.ndarray:
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(float(parts[0]))
            if cid == cam_id:
                return np.array([float(x) for x in parts[1:17]], dtype=np.float64).reshape(4, 4)
    raise ValueError(f"Camera {cam_id} not found in {path}")


def _read_valid_frames(path: Path | None) -> set[str] | None:
    if path is None or not path.is_file():
        return None
    frames = set()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.isdigit():
                frames.add(f"{int(s):06d}")
    return frames or None


def _write_calib(out_calib: Path, k: np.ndarray, c2l: np.ndarray) -> None:
    p2 = np.zeros((3, 4), dtype=np.float64)
    p2[:3, :3] = k
    tr = c2l[:3, :4]
    p_flat = ' '.join(f'{v:.12e}' for v in p2.reshape(-1))
    with open(out_calib, 'w') as f:
        f.write(f'P0: {p_flat}\n')
        f.write(f'P1: {p_flat}\n')
        f.write(f'P2: {p_flat}\n')
        f.write(f'P3: {p_flat}\n')
        f.write('Tr: ' + ' '.join(f'{v:.12e}' for v in tr.reshape(-1)) + '\n')


def _pcd_to_bin(pcd_path: Path, bin_path: Path) -> None:
    pc = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pc.points, dtype=np.float32)
    if pts.size == 0:
        raise ValueError(f"Empty point cloud: {pcd_path}")
    if pts.shape[1] == 3:
        intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pts = np.hstack([pts, intensity])
    pts.astype(np.float32).tofile(bin_path)


def _default_cam_id(dataset_name: str) -> int:
    """Default camera for BEVCalib eval adapter.

    KITTI-360 cam2 (~90° rig mount) often fails on BEVCalib (trained on KITTI-Odometry
    image_2 geometry). cam0 aligns much better on TLC scenes including large_zigzag.
    FAST-LIVO2 uses cam0 as well.
    """
    return 0


def _default_image_dir(scene_dir: Path, cam_id: int) -> Path:
    for candidate in (
        scene_dir / 'images' / f'image_{cam_id:02d}',
        scene_dir / 'images' / f'image_{cam_id}',
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No image dir for cam {cam_id} under {scene_dir / 'images'}")


def prepare_scene(
    scene_dir: Path,
    output_root: Path,
    cam_id: int | None = None,
    seq_id: str = '00',
    use_valid_frames: bool = True,
    symlink_images: bool = True,
    force: bool = False,
) -> Path:
    scene_dir = scene_dir.resolve()
    params = scene_dir / 'params'
    if not params.is_dir():
        raise FileNotFoundError(f"Missing params/: {scene_dir}")

    dataset_name = scene_dir.parent.name
    cam_id = _default_cam_id(dataset_name) if cam_id is None else cam_id

    gt_path = params / 'cams_to_lidar_gt.txt'
    if not gt_path.is_file():
        gt_path = params / f'cam{cam_id}_to_lidar.txt'
    c2l = _read_rig_line(gt_path, cam_id)

    intr_path = params / 'intrinsics.txt'
    if not intr_path.is_file():
        intr_path = params / f'cam{cam_id}.txt'
    k = _read_intrinsics(intr_path)

    image_dir = _default_image_dir(scene_dir, cam_id)
    pcd_dir = scene_dir / 'pcds'
    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"Missing pcds/: {scene_dir}")

    valid_frames = _read_valid_frames(scene_dir / 'valid_frame.txt') if use_valid_frames else None

    out_seq = output_root / 'sequences' / seq_id
    out_img = out_seq / 'image_2'
    out_velo = out_seq / 'velodyne'
    out_calib = out_seq / 'calib.txt'

    if out_seq.exists() and force:
        shutil.rmtree(out_seq)
    out_img.mkdir(parents=True, exist_ok=True)
    out_velo.mkdir(parents=True, exist_ok=True)

    _write_calib(out_calib, k, c2l)

    n_frames = 0
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        frame_id = img_path.stem
        if valid_frames is not None and frame_id not in valid_frames:
            continue
        pcd_path = pcd_dir / f'{frame_id}.pcd'
        if not pcd_path.is_file():
            continue

        out_png = out_img / f'{frame_id}.png'
        if symlink_images:
            if out_png.exists() or out_png.is_symlink():
                out_png.unlink()
            os.symlink(img_path.resolve(), out_png)
        else:
            shutil.copy2(img_path, out_png)

        _pcd_to_bin(pcd_path, out_velo / f'{frame_id}.bin')
        n_frames += 1

    if n_frames == 0:
        raise ValueError(f"No valid frame pairs under {scene_dir} (cam={cam_id})")

    times_path = out_seq / 'times.txt'
    with open(times_path, 'w') as f:
        for i in range(n_frames):
            f.write(f'{i * 0.1:.6f}\n')

    meta = {
        'scene': scene_dir.name,
        'dataset': dataset_name,
        'cam_id': cam_id,
        'frames': n_frames,
        'seq_id': seq_id,
    }
    with open(output_root / 'tlc_scene_meta.json', 'w') as f:
        import json
        json.dump(meta, f, indent=2)

    return out_seq


def main():
    parser = argparse.ArgumentParser(description='Prepare TLC scene for BEVCalib eval')
    parser.add_argument('--scene_dir', required=True, help='TLC scene directory')
    parser.add_argument('--output_root', required=True, help='Output KITTI-like root')
    parser.add_argument('--cam_id', type=int, default=None, help='Camera id (default: auto)')
    parser.add_argument('--seq_id', default='00', help='Sequence id under sequences/')
    parser.add_argument('--no_valid_frames', action='store_true',
                        help='Ignore valid_frame.txt, use all image+pcd pairs')
    parser.add_argument('--copy_images', action='store_true', help='Copy instead of symlink images')
    parser.add_argument('--force', action='store_true', help='Remove existing output sequence')
    args = parser.parse_args()

    out_seq = prepare_scene(
        scene_dir=Path(args.scene_dir),
        output_root=Path(args.output_root),
        cam_id=args.cam_id,
        seq_id=args.seq_id,
        use_valid_frames=not args.no_valid_frames,
        symlink_images=not args.copy_images,
        force=args.force,
    )
    print(f"Prepared {out_seq}")


if __name__ == '__main__':
    main()
