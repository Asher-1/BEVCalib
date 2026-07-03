#!/usr/bin/env python3
"""Benchmark BEVCalib checkpoints on TLC-Calib scenes (KITTI-360 + FAST-LIVO2)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_tlc_for_bevcalib import prepare_scene, _read_rig_line  # noqa: E402


def discover_scenes(tlc_root: Path) -> list[tuple[str, Path]]:
    scenes: list[tuple[str, Path]] = []
    for dataset in ('FAST-LIVO2', 'KITTI-360'):
        ds_dir = tlc_root / dataset
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not (scene_dir / 'params').is_dir():
                continue
            if scene_dir.name.endswith('.zip'):
                continue
            scenes.append((dataset, scene_dir))
    return scenes


def rotation_error_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
    r_err = r_pred @ r_gt.T
    cos_theta = (np.trace(r_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error_m(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_pred - t_gt))


def read_l2c_gt(scene_dir: Path, cam_id: int) -> np.ndarray:
    params = scene_dir / 'params'
    gt_path = params / 'cams_to_lidar_gt.txt'
    if not gt_path.is_file():
        gt_path = params / f'cam{cam_id}_to_lidar.txt'
    c2l = _read_rig_line(gt_path, cam_id)
    return np.linalg.inv(c2l)


def parse_extrinsics_file(path: Path) -> list[np.ndarray]:
    """Parse evaluate_checkpoint extrinsics_and_errors.txt pred matrices (l2c)."""
    preds: list[np.ndarray] = []
    if not path.is_file():
        return preds
    with open(path) as f:
        text = f.read()

    for block in text.split('Sample ')[1:]:
        if 'Predicted Extrinsics' not in block:
            continue
        sub = block.split('Predicted Extrinsics', 1)[1]
        mat = []
        for ln in sub.splitlines()[1:]:
            ln = ln.strip()
            if not ln or ln.startswith('Rotation Errors') or ln.startswith('='):
                break
            if ln.replace('-', '').strip() == '':
                break
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                mat.append([float(x) for x in parts[:4]])
            except ValueError:
                break
            if len(mat) == 4:
                break
        if len(mat) == 4:
            preds.append(np.array(mat, dtype=np.float64))
    return preds


def aggregate_median_pose(mats: list[np.ndarray]) -> np.ndarray:
    rots = Rotation.from_matrix([m[:3, :3] for m in mats])
    quats = rots.as_quat()
    median_q = np.median(quats, axis=0)
    median_q /= np.linalg.norm(median_q) + 1e-12
    r_med = Rotation.from_quat(median_q).as_matrix()
    t_med = np.median([m[:3, 3] for m in mats], axis=0)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r_med
    out[:3, 3] = t_med
    return out


def run_eval(
    ckpt_path: Path,
    dataset_root: Path,
    output_dir: Path,
    angle_deg: float,
    trans_range: float,
    batch_size: int,
    max_frames: int | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / 'evaluate_checkpoint.py'),
        '--ckpt_path', str(ckpt_path),
        '--dataset_root', str(dataset_root),
        '--output_dir', str(output_dir),
        '--angle_range_deg', str(angle_deg),
        '--trans_range', str(trans_range),
        '--rotation_only', '1',
        '--use_full_dataset',
        '--batch_size', str(batch_size),
        '--max_batches', '0',
    ]
    if max_frames is not None:
        cmd.extend(['--eval_max_frames_per_seq', str(max_frames)])

    env = os.environ.copy()
    env.setdefault('USE_DRCV_BACKEND', '0')
    env.setdefault('HF_HUB_OFFLINE', '1')
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return output_dir / 'extrinsics_and_errors.txt'


def eval_scene(
    scene_dir: Path,
    dataset_name: str,
    ckpt_path: Path,
    work_root: Path,
    angle_deg: float,
    trans_range: float,
    batch_size: int,
    max_frames: int | None,
    cam_id: int | None,
    use_valid_frames: bool,
) -> dict:
    scene_key = f"{dataset_name}/{scene_dir.name}"
    prep_root = work_root / 'prepared' / scene_key.replace('/', '_')
    eval_dir = work_root / 'eval' / scene_key.replace('/', '_')

    prepare_scene(
        scene_dir=scene_dir,
        output_root=prep_root,
        cam_id=cam_id,
        use_valid_frames=use_valid_frames,
        force=True,
    )
    with open(prep_root / 'tlc_scene_meta.json') as f:
        meta = json.load(f)
    cam = int(meta['cam_id'])

    ext_file = run_eval(
        ckpt_path=ckpt_path,
        dataset_root=prep_root,
        output_dir=eval_dir,
        angle_deg=angle_deg,
        trans_range=trans_range,
        batch_size=batch_size,
        max_frames=max_frames,
    )

    preds = parse_extrinsics_file(ext_file)
    if not preds:
        raise RuntimeError(f"No predictions parsed from {ext_file}")

    gt_l2c = read_l2c_gt(scene_dir, cam)
    per_frame_rot = [rotation_error_deg(p[:3, :3], gt_l2c[:3, :3]) for p in preds]
    per_frame_trans = [translation_error_m(p[:3, 3], gt_l2c[:3, 3]) for p in preds]

    agg = aggregate_median_pose(preds)
    rig_rot = rotation_error_deg(agg[:3, :3], gt_l2c[:3, :3])
    rig_trans = translation_error_m(agg[:3, 3], gt_l2c[:3, 3])

    return {
        'dataset': dataset_name,
        'scene': scene_dir.name,
        'cam_id': cam,
        'n_frames': len(preds),
        'frame_rot_mean_deg': float(np.mean(per_frame_rot)),
        'frame_rot_median_deg': float(np.median(per_frame_rot)),
        'frame_trans_mean_m': float(np.mean(per_frame_trans)),
        'frame_trans_median_m': float(np.median(per_frame_trans)),
        'rig_rot_deg': rig_rot,
        'rig_trans_m': rig_trans,
        'eval_dir': str(eval_dir),
    }


def write_report(results: list[dict], out_path: Path, ckpt_path: Path, args: argparse.Namespace) -> None:
    lines = [
        '# TLC × BEVCalib Benchmark Report',
        '',
        f'- Generated: {datetime.now().isoformat(timespec="seconds")}',
        f'- Checkpoint: `{ckpt_path}`',
        f'- Perturbation: {args.angle_deg}° rot, {args.trans_range}m trans',
        f'- use_valid_frames: {args.use_valid_frames}',
        '',
        '## Per-Scene Results',
        '',
        '| Dataset | Scene | Cam | Frames | Frame Rot (med) | Frame Trans (med) | Rig Rot | Rig Trans |',
        '|---------|-------|-----|--------|-----------------|-------------------|---------|-----------|',
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['scene']} | {r['cam_id']} | {r['n_frames']} | "
            f"{r['frame_rot_median_deg']:.4f}° | {r['frame_trans_median_m']:.4f}m | "
            f"{r['rig_rot_deg']:.4f}° | {r['rig_trans_m']:.4f}m |"
        )

    if results:
        lines.extend([
            '',
            '## Aggregate',
            '',
            f"- Mean rig rotation error: **{np.mean([r['rig_rot_deg'] for r in results]):.4f}°**",
            f"- Mean rig translation error: **{np.mean([r['rig_trans_m'] for r in results]):.4f}m**",
            f"- Mean frame median rotation: **{np.mean([r['frame_rot_median_deg'] for r in results]):.4f}°**",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='TLC dataset benchmark for BEVCalib')
    parser.add_argument('--tlc_root', default=str(ROOT.parent / 'TLC-Calib' / 'data' / 'TLC-Calib'))
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--output_dir', default='logs/evaluations/tlc_benchmark')
    parser.add_argument('--datasets', default='FAST-LIVO2,KITTI-360',
                        help='Comma-separated dataset folders under tlc_root')
    parser.add_argument('--scenes', default='',
                        help='Optional comma-separated scene names filter')
    parser.add_argument('--angle_deg', type=float, default=5.0)
    parser.add_argument('--trans_range', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--cam_id', type=int, default=None)
    parser.add_argument('--use_valid_frames', action='store_true',
                        help='Use valid_frame.txt (default: all image+pcd pairs)')
    parser.add_argument('--work_dir', default='', help='Persistent work dir (default: temp under output_dir)')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    tlc_root = Path(args.tlc_root)
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    allowed_ds = {s.strip() for s in args.datasets.split(',') if s.strip()}
    scene_filter = {s.strip() for s in args.scenes.split(',') if s.strip()}

    scenes = [
        (ds, p) for ds, p in discover_scenes(tlc_root)
        if ds in allowed_ds and (not scene_filter or p.name in scene_filter)
    ]
    if not scenes:
        raise SystemExit(f"No scenes found under {tlc_root} for datasets={allowed_ds}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Would benchmark {len(scenes)} scenes:")
        for ds, p in scenes:
            print(f"  - {ds}/{p.name}")
        return

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix='tlc_bench_', dir=str(out_dir)))
        cleanup = True

    results: list[dict] = []
    try:
        for ds, scene_dir in scenes:
            print(f"\n=== {ds}/{scene_dir.name} ===")
            try:
                row = eval_scene(
                    scene_dir=scene_dir,
                    dataset_name=ds,
                    ckpt_path=ckpt_path,
                    work_root=work_root,
                    angle_deg=args.angle_deg,
                    trans_range=args.trans_range,
                    batch_size=args.batch_size,
                    max_frames=args.max_frames,
                    cam_id=args.cam_id,
                    use_valid_frames=args.use_valid_frames,
                )
                results.append(row)
                print(
                    f"  frames={row['n_frames']} "
                    f"rig_rot={row['rig_rot_deg']:.4f}° "
                    f"rig_trans={row['rig_trans_m']:.4f}m"
                )
            except Exception as exc:
                print(f"  [FAIL] {exc}")
                results.append({
                    'dataset': ds,
                    'scene': scene_dir.name,
                    'error': str(exc),
                })
    finally:
        if cleanup:
            pass  # keep work dir under output for debugging; user can delete manually

    json_path = out_dir / 'tlc_benchmark_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    report_path = out_dir / 'TLC_BENCHMARK_REPORT.md'
    ok_results = [r for r in results if 'rig_rot_deg' in r]
    write_report(ok_results, report_path, ckpt_path, args)

    print(f"\nResults: {json_path}")
    print(f"Report:  {report_path}")


if __name__ == '__main__':
    main()
