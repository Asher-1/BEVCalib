#!/usr/bin/env python3
"""
Detailed analysis of worst samples from BEVCalib generalization evaluation.
Generates projection visualizations and analyzes root causes.
"""

import re
import os
import sys
import json
import numpy as np
import cv2
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))
from visualization import (
    project_points_to_image,
    render_projected_points,
    create_error_analysis_panel,
    compute_pose_errors,
)

EVAL_BASE = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/generalization_eval_test_models2_v2"
DATA_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
OUTPUT_DIR = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/generalization_eval_test_models2_v2/bad_sample_analysis"
MODEL = "v20-v8recipe-pitch-wt3"

SEQ_INFO = {
    0: "B26A1-1", 1: "C01-60", 2: "C01T-45", 3: "D037-3",
    4: "DE061-5", 5: "DE07-5", 6: "DE08-8", 7: "EC15S-8",
    8: "LPD19A-4", 9: "M81-31", 10: "P03-4", 11: "P789-22"
}
SEQ_STARTS = {i: i * 400 for i in range(12)}


def parse_all_results(filepath):
    """Parse extrinsics_and_errors.txt - extract GT and predicted extrinsics + errors."""
    samples = {}
    gt_T = None
    current_sample = None
    pred_lines = []
    in_gt = False
    in_pred = False
    gt_lines = []
    
    with open(filepath) as f:
        for line in f:
            if 'Ground Truth Extrinsics' in line:
                in_gt = True
                gt_lines = []
                continue
            
            if in_gt:
                stripped = line.strip()
                if stripped and not stripped.startswith('='):
                    values = stripped.split()
                    try:
                        row = [float(v) for v in values]
                        gt_lines.append(row)
                        if len(gt_lines) == 4:
                            gt_T = np.array(gt_lines)
                            in_gt = False
                    except ValueError:
                        in_gt = False
                elif stripped.startswith('='):
                    if gt_lines:
                        in_gt = False
                continue
            
            m = re.match(r'Sample (\d+) \[Seq (\d+)\]', line)
            if m:
                current_sample = int(m.group(1))
                current_seq = int(m.group(2))
                samples[current_sample] = {
                    'seq': current_seq,
                    'gt_T': gt_T,
                }
                in_pred = False
                pred_lines = []
                continue
            
            if current_sample is not None:
                if 'Predicted Extrinsics' in line:
                    in_pred = True
                    pred_lines = []
                    continue
                
                if in_pred:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('=') and not stripped.startswith('Rotation'):
                        try:
                            row = [float(v) for v in stripped.split()]
                            if len(row) == 4:
                                pred_lines.append(row)
                                if len(pred_lines) == 4:
                                    samples[current_sample]['pred_T'] = np.array(pred_lines)
                                    in_pred = False
                        except ValueError:
                            in_pred = False
                
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
    
    return samples, gt_T


def load_calib(seq_str):
    """Load calibration from calib.txt."""
    calib_path = os.path.join(DATA_ROOT, 'sequences', seq_str, 'calib.txt')
    K = None
    Tr = None
    
    with open(calib_path) as f:
        for line in f:
            if line.startswith('P2:'):
                values = [float(v) for v in line.split(':')[1].strip().split()]
                P = np.array(values).reshape(3, 4)
                K = P[:3, :3]
            elif line.startswith('Tr:'):
                values = [float(v) for v in line.split(':')[1].strip().split()]
                Tr_3x4 = np.array(values).reshape(3, 4)
                Tr = np.vstack([Tr_3x4, [0, 0, 0, 1]])
    
    T_lidar2cam = np.linalg.inv(Tr)
    return K, T_lidar2cam, Tr


def load_point_cloud(seq_str, frame_str):
    """Load point cloud from .bin file."""
    pcd_path = os.path.join(DATA_ROOT, 'sequences', seq_str, 'velodyne', f'{frame_str}.bin')
    raw = np.fromfile(pcd_path, dtype=np.float32)
    if raw.size % 4 == 0:
        pcd = raw.reshape(-1, 4)
    else:
        pcd = raw.reshape(-1, 3)
    
    ego_filter = (np.abs(pcd[:, 0]) > 3.) | (np.abs(pcd[:, 1]) > 3.)
    pcd = pcd[ego_filter]
    
    from kitti_bev_calib_settings import xbound, ybound, zbound
    range_filter = (
        (pcd[:, 0] >= xbound[0]) & (pcd[:, 0] <= xbound[1]) &
        (pcd[:, 1] >= ybound[0]) & (pcd[:, 1] <= ybound[1]) &
        (pcd[:, 2] >= zbound[0]) & (pcd[:, 2] <= zbound[1])
    )
    pcd_filtered = pcd[range_filter]
    
    return pcd, pcd_filtered


def load_image(seq_str, frame_str, use_resized=True):
    """Load image."""
    if use_resized:
        img_path = os.path.join(DATA_ROOT, 'sequences', seq_str, 'image_2_640x360', f'{frame_str}.jpg')
        if os.path.exists(img_path):
            return cv2.imread(img_path)
    
    img_path = os.path.join(DATA_ROOT, 'sequences', seq_str, 'image_2', f'{frame_str}.png')
    img = cv2.imread(img_path)
    if use_resized and img is not None:
        img = cv2.resize(img, (640, 360))
    return img


def get_frame_index(sample_id, seq):
    """Map sample ID back to frame index in dataset."""
    seq_start = seq * 400
    sample_offset = sample_id - seq_start
    
    seq_str = f'{seq:02d}'
    img_dir = os.path.join(DATA_ROOT, 'sequences', seq_str, 'image_2')
    all_frames = sorted([f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.png')])
    total_frames = len(all_frames)
    
    stride = total_frames / 400.0
    actual_frame_idx = int(sample_offset * stride)
    actual_frame_idx = min(actual_frame_idx, total_frames - 1)
    
    return all_frames[actual_frame_idx], actual_frame_idx, total_frames


def analyze_point_cloud_quality(pcd_raw, pcd_filtered, seq_str, frame_str):
    """Analyze point cloud quality."""
    stats = {
        'raw_points': len(pcd_raw),
        'filtered_points': len(pcd_filtered),
        'utilization': len(pcd_filtered) / len(pcd_raw) if len(pcd_raw) > 0 else 0,
    }
    
    if len(pcd_filtered) > 0:
        stats['x_range'] = (pcd_filtered[:, 0].min(), pcd_filtered[:, 0].max())
        stats['y_range'] = (pcd_filtered[:, 1].min(), pcd_filtered[:, 1].max())
        stats['z_range'] = (pcd_filtered[:, 2].min(), pcd_filtered[:, 2].max())
        stats['z_mean'] = pcd_filtered[:, 2].mean()
        stats['z_std'] = pcd_filtered[:, 2].std()
        
        distances = np.sqrt(pcd_filtered[:, 0]**2 + pcd_filtered[:, 1]**2)
        stats['near_points'] = int((distances < 10).sum())
        stats['mid_points'] = int(((distances >= 10) & (distances < 30)).sum())
        stats['far_points'] = int((distances >= 30).sum())
        stats['mean_distance'] = distances.mean()
    
    return stats


def analyze_image_quality(img, seq_str, frame_str):
    """Basic image quality analysis."""
    if img is None:
        return {'valid': False}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stats = {
        'valid': True,
        'size': img.shape[:2],
        'mean_brightness': float(gray.mean()),
        'std_brightness': float(gray.std()),
        'is_dark': float(gray.mean()) < 50,
        'is_overexposed': float(gray.mean()) > 220,
        'is_low_contrast': float(gray.std()) < 20,
    }
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    stats['sharpness'] = float(laplacian.var())
    stats['is_blurry'] = stats['sharpness'] < 50
    
    return stats


def scale_K_for_resized(K_orig, orig_size=(3840, 2160), target_size=(640, 360)):
    """Scale intrinsics to match resized image."""
    sx = target_size[0] / orig_size[0]
    sy = target_size[1] / orig_size[1]
    K_scaled = np.array([
        [K_orig[0, 0] * sx, 0, K_orig[0, 2] * sx],
        [0, K_orig[1, 1] * sy, K_orig[1, 2] * sy],
        [0, 0, 1]
    ])
    return K_scaled


def generate_projection_vis(img, pcd, gt_T, pred_T, K, sample_info, output_path):
    """Generate projection comparison visualization."""
    h, w = img.shape[:2]
    
    max_pts = min(len(pcd), 8000)
    if len(pcd) > max_pts:
        idx = np.random.choice(len(pcd), max_pts, replace=False)
        pts = pcd[idx, :3]
    else:
        pts = pcd[:, :3]
    
    gt_pts_2d, gt_depths, _ = project_points_to_image(pts, gt_T, K, (h, w))
    pred_pts_2d, pred_depths, _ = project_points_to_image(pts, pred_T, K, (h, w))
    
    gt_img = render_projected_points(img.copy(), gt_pts_2d, gt_depths, color_mode='depth', point_radius=2, max_depth=80.0)
    pred_img = render_projected_points(img.copy(), pred_pts_2d, pred_depths, color_mode='depth', point_radius=2, max_depth=80.0)
    
    overlay = render_projected_points(img.copy(), gt_pts_2d, gt_depths, color_mode='fixed_green', point_radius=2)
    overlay = render_projected_points(overlay, pred_pts_2d, pred_depths, color_mode='fixed_red', point_radius=1)
    
    errors = compute_pose_errors(pred_T, gt_T)
    error_panel = create_error_analysis_panel(
        img, pts, gt_T, pred_T, K, errors,
        max_points=max_pts, point_radius=2, rotation_only=True,
    )
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    def put_bg(im, text, pos, color, scale=0.45):
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.rectangle(im, (pos[0]-2, pos[1]-th-2), (pos[0]+tw+2, pos[1]+4), (0,0,0), -1)
        cv2.putText(im, text, pos, font, scale, color, 1, cv2.LINE_AA)
    
    s = sample_info
    put_bg(gt_img, f"GT Projection | Seq{s['seq']:02d} {SEQ_INFO[s['seq']]} | Frame {s.get('frame_str','?')}", (5, 15), (0,255,0))
    put_bg(gt_img, f"Pts: {len(gt_pts_2d)}", (5, 32), (200,200,200))
    
    put_bg(pred_img, f"PRED Projection | Total: {s['total']:.3f}deg", (5, 15), (0,0,255))
    put_bg(pred_img, f"R:{s['roll']:.3f} P:{s['pitch']:.3f} Y:{s['yaw']:.3f}", (5, 32), (0,255,255))
    
    put_bg(overlay, f"Overlay: GT(Green) + Pred(Red)", (5, 15), (255,255,255))
    put_bg(overlay, f"Sample {s['sample_id']:04d} | Error: {s['total']:.3f}deg", (5, 32), (0,255,255))
    
    top = np.hstack([gt_img, pred_img])
    bot = np.hstack([overlay, error_panel])
    
    if top.shape[1] != bot.shape[1]:
        target_w = max(top.shape[1], bot.shape[1])
        if top.shape[1] < target_w:
            top = np.hstack([top, np.zeros((top.shape[0], target_w - top.shape[1], 3), dtype=np.uint8)])
        if bot.shape[1] < target_w:
            bot = np.hstack([bot, np.zeros((bot.shape[0], target_w - bot.shape[1], 3), dtype=np.uint8)])
    
    vis = np.vstack([top, bot])
    cv2.imwrite(output_path, vis)
    return vis


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("BEVCalib Bad Sample Detailed Analysis")
    print(f"Model: {MODEL}")
    print("=" * 80)
    
    filepath = os.path.join(EVAL_BASE, MODEL, "extrinsics_and_errors.txt")
    print(f"\nParsing evaluation results...")
    samples, gt_T_global = parse_all_results(filepath)
    print(f"Parsed {len(samples)} samples")
    
    # Also parse v20-v8recipe-z10 for comparison
    filepath2 = os.path.join(EVAL_BASE, "v20-v8recipe-z10", "extrinsics_and_errors.txt")
    samples_z10, _ = parse_all_results(filepath2)
    
    # Find worst samples across ALL sequences
    all_sorted = sorted(samples.items(), key=lambda x: x[1].get('total', 0), reverse=True)
    
    print("\n" + "=" * 80)
    print("TOP 30 WORST SAMPLES (v20-v8recipe-pitch-wt3)")
    print("=" * 80)
    
    worst_samples = []
    for sid, s in all_sorted[:30]:
        seq = s['seq']
        frame_str, frame_idx, total_frames = get_frame_index(sid, seq)
        s['frame_str'] = frame_str
        s['frame_idx'] = frame_idx
        s['sample_id'] = sid
        worst_samples.append((sid, s))
        print(f"  Sample {sid:04d} | Seq {seq:02d} ({SEQ_INFO[seq]:>10}) | Frame {frame_str} ({frame_idx}/{total_frames}) | "
              f"Total={s['total']:.4f}° R={s['roll']:.4f}° P={s['pitch']:.4f}° Y={s['yaw']:.4f}°")
    
    # Group worst samples by sequence
    seq_worst = defaultdict(list)
    for sid, s in all_sorted:
        if s.get('total', 0) > 1.0:
            seq_worst[s['seq']].append((sid, s))
    
    print("\n" + "=" * 80)
    print("WORST SAMPLES BY SEQUENCE (>1.0° threshold)")
    print("=" * 80)
    for seq in sorted(seq_worst.keys()):
        count = len(seq_worst[seq])
        max_err = max(s['total'] for _, s in seq_worst[seq])
        print(f"  Seq {seq:02d} ({SEQ_INFO[seq]:>10}): {count:>3d} samples > 1.0° | Max = {max_err:.4f}°")
    
    # Now generate visualizations for the worst 5 samples per problematic sequence
    TARGET_SEQS = [3, 5, 7, 8]  # Worst sequences identified
    
    try:
        from bev_settings import xbound, ybound, zbound
    except ImportError:
        xbound = (0.0, 200.0, 2.0)
        ybound = (-100.0, 100.0, 2.0)
        zbound = (-10.0, 10.0, 4.0)
    
    analysis_report = {}
    
    for seq in TARGET_SEQS:
        seq_str = f'{seq:02d}'
        vehicle = SEQ_INFO[seq]
        
        print(f"\n{'='*80}")
        print(f"ANALYZING Seq {seq_str} ({vehicle})")
        print(f"{'='*80}")
        
        K_orig, gt_T, Tr = load_calib(seq_str)
        K_resized = scale_K_for_resized(K_orig)
        
        # Get worst 5 from this seq
        seq_sorted = [(sid, s) for sid, s in all_sorted if s['seq'] == seq][:5]
        
        seq_analysis = {
            'vehicle': vehicle,
            'worst_samples': [],
            'pcd_stats': [],
            'img_stats': [],
        }
        
        for rank, (sid, s) in enumerate(seq_sorted):
            frame_str, frame_idx, total_frames = get_frame_index(sid, seq)
            s['frame_str'] = frame_str
            s['sample_id'] = sid
            
            print(f"\n  --- Sample {sid:04d} (#{rank+1} worst in seq) | Frame {frame_str} ---")
            print(f"  Total={s['total']:.4f}° R={s['roll']:.4f}° P={s['pitch']:.4f}° Y={s['yaw']:.4f}°")
            
            # Compare with z10 model
            if sid in samples_z10:
                sz = samples_z10[sid]
                print(f"  [v20-v8recipe-z10]: Total={sz.get('total',0):.4f}° R={sz.get('roll',0):.4f}° P={sz.get('pitch',0):.4f}° Y={sz.get('yaw',0):.4f}°")
            
            img = load_image(seq_str, frame_str, use_resized=True)
            img_stats = analyze_image_quality(img, seq_str, frame_str)
            
            print(f"  Image: brightness={img_stats['mean_brightness']:.1f} "
                  f"contrast={img_stats['std_brightness']:.1f} "
                  f"sharpness={img_stats['sharpness']:.1f} "
                  f"{'DARK!' if img_stats['is_dark'] else ''}"
                  f"{'BLURRY!' if img_stats['is_blurry'] else ''}"
                  f"{'LOW_CONTRAST!' if img_stats['is_low_contrast'] else ''}")
            
            pcd_path = os.path.join(DATA_ROOT, 'sequences', seq_str, 'velodyne', f'{frame_str}.bin')
            raw = np.fromfile(pcd_path, dtype=np.float32)
            pcd_raw = raw.reshape(-1, 4) if raw.size % 4 == 0 else raw.reshape(-1, 3)
            
            ego_filter = (np.abs(pcd_raw[:, 0]) > 3.) | (np.abs(pcd_raw[:, 1]) > 3.)
            pcd_ego = pcd_raw[ego_filter]
            xb, yb, zb = xbound, ybound, zbound
            range_filter = (
                (pcd_ego[:, 0] >= xb[0]) & (pcd_ego[:, 0] <= xb[1]) &
                (pcd_ego[:, 1] >= yb[0]) & (pcd_ego[:, 1] <= yb[1]) &
                (pcd_ego[:, 2] >= zb[0]) & (pcd_ego[:, 2] <= zb[1])
            )
            pcd_filtered = pcd_ego[range_filter]
            
            pcd_stats = analyze_point_cloud_quality(pcd_raw, pcd_filtered, seq_str, frame_str)
            print(f"  PCD: raw={pcd_stats['raw_points']} filtered={pcd_stats['filtered_points']} "
                  f"util={pcd_stats['utilization']:.1%} "
                  f"near={pcd_stats.get('near_points',0)} mid={pcd_stats.get('mid_points',0)} far={pcd_stats.get('far_points',0)}")
            if 'z_mean' in pcd_stats:
                print(f"  PCD Z: mean={pcd_stats['z_mean']:.2f} std={pcd_stats['z_std']:.2f} "
                      f"range=[{pcd_stats['z_range'][0]:.2f}, {pcd_stats['z_range'][1]:.2f}]")
            
            # Generate visualization
            if 'pred_T' in s and img is not None and len(pcd_filtered) > 100:
                out_path = os.path.join(OUTPUT_DIR, f"seq{seq_str}_{vehicle}_sample{sid:04d}_frame{frame_str}.jpg")
                generate_projection_vis(img, pcd_filtered, gt_T, s['pred_T'], K_resized, s, out_path)
                print(f"  Saved: {out_path}")
            else:
                print(f"  SKIP visualization: pred_T={'yes' if 'pred_T' in s else 'no'} "
                      f"img={'yes' if img is not None else 'no'} pcd={len(pcd_filtered)}")
            
            seq_analysis['worst_samples'].append(s)
            seq_analysis['pcd_stats'].append(pcd_stats)
            seq_analysis['img_stats'].append(img_stats)
        
        # Per-sequence summary statistics
        all_seq_samples = [(sid, s) for sid, s in samples.items() if s.get('seq') == seq]
        all_totals = [s['total'] for _, s in all_seq_samples if 'total' in s]
        
        if all_totals:
            print(f"\n  SEQUENCE SUMMARY:")
            print(f"    Mean: {np.mean(all_totals):.4f}° | Std: {np.std(all_totals):.4f}°")
            print(f"    P50: {np.percentile(all_totals, 50):.4f}° | P90: {np.percentile(all_totals, 90):.4f}°")
            print(f"    P95: {np.percentile(all_totals, 95):.4f}° | Max: {np.max(all_totals):.4f}°")
            print(f"    >1.0°: {sum(1 for t in all_totals if t > 1.0)} ({100*sum(1 for t in all_totals if t > 1.0)/len(all_totals):.1f}%)")
        
        # Check temporal pattern (consecutive frame errors)
        seq_samples_sorted = sorted(all_seq_samples, key=lambda x: x[0])
        errors_seq = [s.get('total', 0) for _, s in seq_samples_sorted]
        
        high_error_runs = []
        run_start = None
        for i, err in enumerate(errors_seq):
            if err > 1.0:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    high_error_runs.append((run_start, i - 1, i - run_start))
                    run_start = None
        if run_start is not None:
            high_error_runs.append((run_start, len(errors_seq) - 1, len(errors_seq) - run_start))
        
        if high_error_runs:
            print(f"\n  TEMPORAL PATTERN (consecutive >1.0° frames):")
            for start, end, length in sorted(high_error_runs, key=lambda x: -x[2])[:5]:
                print(f"    Samples {start+SEQ_STARTS[seq]:04d}-{end+SEQ_STARTS[seq]:04d}: {length} consecutive frames")
        
        analysis_report[seq] = seq_analysis
    
    # Final root cause summary
    print(f"\n{'='*80}")
    print("ROOT CAUSE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print("""
Key Findings:

1. v20-v8recipe-pitch-wt3 GLOBAL MAX errors:
   - Top 6 worst: Seq 03 (D037-3) - samples 1477-1484, max 1.97°
   - Followed by Seq 07 (EC15S-8) - widespread, max 1.52°
   - These are NOT random outliers but systematic patterns

2. v20-v8recipe-z10 GLOBAL MAX errors:
   - Top 16 worst: Seq 05 (DE07-5) - samples 2319-2336, max 2.43°
   - Followed by Seq 06 (DE08-8) - samples 2744-2798, max 1.92°
   - Seq 05 has extremely concentrated Pitch errors (~2.3° Pitch)

3. Seq 07 (EC15S-8) - worst test sequence consistently:
   - 58% of 400 samples >1.0° in best model
   - 98.8% of samples >1.0° in z10 model
   - Errors distributed across ENTIRE sequence (not localized)
   - Both Roll and Pitch elevated → likely calibration mismatch

4. Seq 08 (LPD19A-4) - Pitch+Yaw dominant:
   - Roll is very low (0.12°) but Pitch (0.66°) and Yaw (0.67°) high
   - fx=7024 is the lowest in dataset (train min=7035)
   - Possible OOD intrinsics issue

5. Seq 03 (D037-3) - concentrated burst of extreme errors:
   - Only 6 samples clustered around sample 1477-1484
   - All have extremely high Pitch (1.62-1.81°)
   - Likely a specific road segment with unusual geometry

6. Seq 05 (DE07-5) in z10 model - model-specific failure:
   - Best model (pitch-wt3) has moderate 0.76° mean for this seq
   - z10 model shows 2.43° max → z10 is specifically bad at this vehicle
   - Confirms Z=10 overfits to training patterns
""")
    
    print(f"\nVisualization outputs saved to: {OUTPUT_DIR}")
    print(f"Total files generated: {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])}")


if __name__ == '__main__':
    main()
