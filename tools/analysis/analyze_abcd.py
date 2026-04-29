#!/usr/bin/env python3
"""
Tasks A, B, C combined analysis:
A: Seq 03 burst analysis (frames 855-877, slope estimation)
B: Seq 07 good vs bad frame scene feature comparison
C: Seq 08 fx OOD + intrinsic aug coverage analysis
"""

import os
import sys
import re
import json
import numpy as np
import cv2
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

DATA_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
TRAIN_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data"
EVAL_BASE = "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/generalization_eval_test_models2_v2"
OUTPUT_DIR = os.path.join(EVAL_BASE, "bad_sample_analysis")

SEQ_INFO = {
    0: "B26A1-1", 1: "C01-60", 2: "C01T-45", 3: "D037-3",
    4: "DE061-5", 5: "DE07-5", 6: "DE08-8", 7: "EC15S-8",
    8: "LPD19A-4", 9: "M81-31", 10: "P03-4", 11: "P789-22"
}


def parse_errors_only(filepath):
    """Quick parse: only errors, no extrinsics."""
    samples = {}
    current_sample = None
    with open(filepath) as f:
        for line in f:
            m = re.match(r'Sample (\d+) \[Seq (\d+)\]', line)
            if m:
                current_sample = int(m.group(1))
                samples[current_sample] = {'seq': int(m.group(2))}
                continue
            if current_sample is not None:
                m = re.match(r'\s+Total:\s+([\d.]+) deg', line)
                if m: samples[current_sample]['total'] = float(m.group(1))
                m = re.match(r'\s+Roll\s+\(LiDAR X\):\s+([\d.]+) deg', line)
                if m: samples[current_sample]['roll'] = float(m.group(1))
                m = re.match(r'\s+Pitch\s+\(LiDAR Y\):\s+([\d.]+) deg', line)
                if m: samples[current_sample]['pitch'] = float(m.group(1))
                m = re.match(r'\s+Yaw\s+\(LiDAR Z\):\s+([\d.]+) deg', line)
                if m: samples[current_sample]['yaw'] = float(m.group(1))
    return samples


def load_calib(root, seq_str):
    K, Tr_4x4 = None, None
    calib_path = os.path.join(root, 'sequences', seq_str, 'calib.txt')
    with open(calib_path) as f:
        for line in f:
            if line.startswith('P2:'):
                vals = [float(v) for v in line.split(':')[1].strip().split()]
                K = np.array(vals).reshape(3, 4)[:3, :3]
            elif line.startswith('Tr:'):
                vals = [float(v) for v in line.split(':')[1].strip().split()]
                Tr_4x4 = np.vstack([np.array(vals).reshape(3, 4), [0, 0, 0, 1]])
    return K, Tr_4x4


def get_frame_for_sample(sample_id, seq, data_root):
    seq_start = seq * 400
    offset = sample_id - seq_start
    seq_str = f'{seq:02d}'
    img_dir = os.path.join(data_root, 'sequences', seq_str, 'image_2')
    frames = sorted([f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.png')])
    total = len(frames)
    stride = total / 400.0
    idx = min(int(offset * stride), total - 1)
    return frames[idx], idx, total


def estimate_ground_slope(pcd_filtered):
    """Estimate ground slope from point cloud using RANSAC-like ground fitting."""
    z_vals = pcd_filtered[:, 2]
    ground_mask = z_vals < np.percentile(z_vals, 20)
    ground_pts = pcd_filtered[ground_mask]
    
    if len(ground_pts) < 100:
        return None
    
    A = np.column_stack([ground_pts[:, 0], ground_pts[:, 1], np.ones(len(ground_pts))])
    z = ground_pts[:, 2]
    try:
        result = np.linalg.lstsq(A, z, rcond=None)
        coeffs = result[0]
        slope_x = np.arctan(coeffs[0]) * 180 / np.pi
        slope_y = np.arctan(coeffs[1]) * 180 / np.pi
        residuals = z - A @ coeffs
        fit_quality = np.std(residuals)
        return {
            'slope_fwd_deg': slope_x,
            'slope_lat_deg': slope_y,
            'fit_std': fit_quality,
            'n_ground_pts': len(ground_pts),
        }
    except Exception:
        return None


def analyze_image_features(img_path, resize=True):
    """Analyze image scene characteristics."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    if resize:
        img = cv2.resize(img, (640, 360))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    top_half = gray[:h//2, :]
    bot_half = gray[h//2:, :]
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    sky_region = gray[:h//4, :]
    road_region = gray[3*h//4:, :]
    
    return {
        'brightness': float(gray.mean()),
        'contrast': float(gray.std()),
        'top_brightness': float(top_half.mean()),
        'bot_brightness': float(bot_half.mean()),
        'sharpness': float(laplacian.var()),
        'sky_brightness': float(sky_region.mean()),
        'road_brightness': float(road_region.mean()),
        'saturation_mean': float(hsv[:,:,1].mean()),
        'value_std': float(hsv[:,:,2].std()),
    }


# =============================================================================
# TASK A: Seq 03 Burst Analysis
# =============================================================================
def task_a_seq03_burst():
    print("=" * 80)
    print("TASK A: Seq 03 (D037-3) Burst Analysis - Frames 855-877")
    print("=" * 80)
    
    seq_str = '03'
    burst_frames = list(range(855, 878))
    context_frames = list(range(840, 855)) + list(range(878, 895))
    
    from bev_settings import xbound, ybound, zbound
    
    print("\n--- Ground Slope Estimation ---")
    for frame_idx in burst_frames + context_frames:
        frame_str = f'{frame_idx:06d}'
        pcd_path = os.path.join(DATA_ROOT, 'sequences', seq_str, 'velodyne', f'{frame_str}.bin')
        if not os.path.exists(pcd_path):
            continue
        
        raw = np.fromfile(pcd_path, dtype=np.float32)
        pcd = raw.reshape(-1, 4) if raw.size % 4 == 0 else raw.reshape(-1, 3)
        ego_filter = (np.abs(pcd[:, 0]) > 3.) | (np.abs(pcd[:, 1]) > 3.)
        pcd = pcd[ego_filter]
        range_filter = (
            (pcd[:, 0] >= xbound[0]) & (pcd[:, 0] <= xbound[1]) &
            (pcd[:, 1] >= ybound[0]) & (pcd[:, 1] <= ybound[1]) &
            (pcd[:, 2] >= zbound[0]) & (pcd[:, 2] <= zbound[1])
        )
        pcd_f = pcd[range_filter]
        
        slope = estimate_ground_slope(pcd_f)
        is_burst = frame_idx in burst_frames
        marker = " *** BURST ***" if is_burst else ""
        
        img_path = os.path.join(DATA_ROOT, 'sequences', seq_str, 'image_2', f'{frame_str}.png')
        img_feat = analyze_image_features(img_path)
        
        if slope and img_feat:
            print(f"  Frame {frame_str}: slope_fwd={slope['slope_fwd_deg']:+.3f}° "
                  f"slope_lat={slope['slope_lat_deg']:+.3f}° "
                  f"fit_std={slope['fit_std']:.3f} "
                  f"brightness={img_feat['brightness']:.1f} "
                  f"sharp={img_feat['sharpness']:.0f} "
                  f"pts={len(pcd_f)}{marker}")
    
    # Generate burst vs context comparison images
    print("\n--- Burst vs Context Image Comparison ---")
    for label, frame_list in [("BURST", burst_frames[:3]), ("CONTEXT_BEFORE", list(range(845, 848))), ("CONTEXT_AFTER", list(range(880, 883)))]:
        for fi in frame_list:
            fs = f'{fi:06d}'
            ip = os.path.join(DATA_ROOT, 'sequences', seq_str, 'image_2_640x360', f'{fs}.jpg')
            if not os.path.exists(ip):
                ip = os.path.join(DATA_ROOT, 'sequences', seq_str, 'image_2', f'{fs}.png')
            if os.path.exists(ip):
                img = cv2.imread(ip)
                if img is not None:
                    if img.shape[1] > 640:
                        img = cv2.resize(img, (640, 360))
                    out = os.path.join(OUTPUT_DIR, f"taskA_seq03_{label}_frame{fs}.jpg")
                    cv2.imwrite(out, img)
                    print(f"  Saved {label} frame: {out}")


# =============================================================================
# TASK B: Seq 07 Good vs Bad Comparison
# =============================================================================
def task_b_seq07_good_vs_bad():
    print("\n" + "=" * 80)
    print("TASK B: Seq 07 (EC15S-8) Good vs Bad Frame Comparison")
    print("=" * 80)
    
    samples = parse_errors_only(os.path.join(EVAL_BASE, "v20-v8recipe-pitch-wt3", "extrinsics_and_errors.txt"))
    seq7_samples = {k: v for k, v in samples.items() if v.get('seq') == 7}
    sorted_s7 = sorted(seq7_samples.items(), key=lambda x: x[1].get('total', 0))
    
    best_5 = sorted_s7[:5]
    worst_5 = sorted_s7[-5:]
    
    from bev_settings import xbound, ybound, zbound
    
    print("\n--- Best 5 samples (lowest error) ---")
    best_stats = []
    for sid, s in best_5:
        fs, fi, total = get_frame_for_sample(sid, 7, DATA_ROOT)
        ip = os.path.join(DATA_ROOT, 'sequences', '07', 'image_2', f'{fs}.png')
        img_feat = analyze_image_features(ip)
        
        pcd_path = os.path.join(DATA_ROOT, 'sequences', '07', 'velodyne', f'{fs}.bin')
        raw = np.fromfile(pcd_path, dtype=np.float32)
        pcd = raw.reshape(-1, 4) if raw.size % 4 == 0 else raw.reshape(-1, 3)
        ego_filter = (np.abs(pcd[:, 0]) > 3.) | (np.abs(pcd[:, 1]) > 3.)
        pcd = pcd[ego_filter]
        rf = (pcd[:, 0] >= xbound[0]) & (pcd[:, 0] <= xbound[1]) & (pcd[:, 1] >= ybound[0]) & (pcd[:, 1] <= ybound[1]) & (pcd[:, 2] >= zbound[0]) & (pcd[:, 2] <= zbound[1])
        pcd_f = pcd[rf]
        
        slope = estimate_ground_slope(pcd_f)
        dist = np.sqrt(pcd_f[:, 0]**2 + pcd_f[:, 1]**2)
        
        stat = {
            'sample': sid, 'frame': fs, 'total': s['total'],
            'roll': s.get('roll', 0), 'pitch': s.get('pitch', 0), 'yaw': s.get('yaw', 0),
            'brightness': img_feat['brightness'] if img_feat else 0,
            'contrast': img_feat['contrast'] if img_feat else 0,
            'sharpness': img_feat['sharpness'] if img_feat else 0,
            'n_points': len(pcd_f),
            'near_pts': int((dist < 10).sum()),
            'far_pts': int((dist >= 30).sum()),
            'mean_dist': float(dist.mean()),
            'z_mean': float(pcd_f[:, 2].mean()),
            'z_std': float(pcd_f[:, 2].std()),
            'slope_fwd': slope['slope_fwd_deg'] if slope else None,
        }
        best_stats.append(stat)
        
        slope_str = f" slope={stat['slope_fwd']:+.3f}d" if stat['slope_fwd'] else ""
        print(f"  Sample {sid:04d} Frame {fs}: Total={s['total']:.4f}d | "
              f"bright={stat['brightness']:.1f} sharp={stat['sharpness']:.0f} "
              f"pts={stat['n_points']} near={stat['near_pts']} far={stat['far_pts']} "
              f"mean_d={stat['mean_dist']:.1f} z_mean={stat['z_mean']:.2f}{slope_str}")
        
        ip2 = os.path.join(DATA_ROOT, 'sequences', '07', 'image_2_640x360', f'{fs}.jpg')
        if not os.path.exists(ip2):
            ip2 = ip
        img = cv2.imread(ip2)
        if img is not None:
            if img.shape[1] > 640:
                img = cv2.resize(img, (640, 360))
            out = os.path.join(OUTPUT_DIR, f"taskB_seq07_GOOD_sample{sid:04d}_frame{fs}.jpg")
            cv2.imwrite(out, img)
    
    print("\n--- Worst 5 samples (highest error) ---")
    worst_stats = []
    for sid, s in worst_5:
        fs, fi, total = get_frame_for_sample(sid, 7, DATA_ROOT)
        ip = os.path.join(DATA_ROOT, 'sequences', '07', 'image_2', f'{fs}.png')
        img_feat = analyze_image_features(ip)
        
        pcd_path = os.path.join(DATA_ROOT, 'sequences', '07', 'velodyne', f'{fs}.bin')
        raw = np.fromfile(pcd_path, dtype=np.float32)
        pcd = raw.reshape(-1, 4) if raw.size % 4 == 0 else raw.reshape(-1, 3)
        ego_filter = (np.abs(pcd[:, 0]) > 3.) | (np.abs(pcd[:, 1]) > 3.)
        pcd = pcd[ego_filter]
        rf = (pcd[:, 0] >= xbound[0]) & (pcd[:, 0] <= xbound[1]) & (pcd[:, 1] >= ybound[0]) & (pcd[:, 1] <= ybound[1]) & (pcd[:, 2] >= zbound[0]) & (pcd[:, 2] <= zbound[1])
        pcd_f = pcd[rf]
        
        slope = estimate_ground_slope(pcd_f)
        dist = np.sqrt(pcd_f[:, 0]**2 + pcd_f[:, 1]**2)
        
        stat = {
            'sample': sid, 'frame': fs, 'total': s['total'],
            'roll': s.get('roll', 0), 'pitch': s.get('pitch', 0), 'yaw': s.get('yaw', 0),
            'brightness': img_feat['brightness'] if img_feat else 0,
            'contrast': img_feat['contrast'] if img_feat else 0,
            'sharpness': img_feat['sharpness'] if img_feat else 0,
            'n_points': len(pcd_f),
            'near_pts': int((dist < 10).sum()),
            'far_pts': int((dist >= 30).sum()),
            'mean_dist': float(dist.mean()),
            'z_mean': float(pcd_f[:, 2].mean()),
            'z_std': float(pcd_f[:, 2].std()),
            'slope_fwd': slope['slope_fwd_deg'] if slope else None,
        }
        worst_stats.append(stat)
        
        slope_str = f" slope={stat['slope_fwd']:+.3f}d" if stat['slope_fwd'] else ""
        print(f"  Sample {sid:04d} Frame {fs}: Total={s['total']:.4f}d | "
              f"bright={stat['brightness']:.1f} sharp={stat['sharpness']:.0f} "
              f"pts={stat['n_points']} near={stat['near_pts']} far={stat['far_pts']} "
              f"mean_d={stat['mean_dist']:.1f} z_mean={stat['z_mean']:.2f}{slope_str}")
        
        ip2 = os.path.join(DATA_ROOT, 'sequences', '07', 'image_2_640x360', f'{fs}.jpg')
        if not os.path.exists(ip2):
            ip2 = ip
        img = cv2.imread(ip2)
        if img is not None:
            if img.shape[1] > 640:
                img = cv2.resize(img, (640, 360))
            out = os.path.join(OUTPUT_DIR, f"taskB_seq07_BAD_sample{sid:04d}_frame{fs}.jpg")
            cv2.imwrite(out, img)
    
    # Statistical comparison
    print("\n--- Statistical Comparison: GOOD vs BAD ---")
    for metric in ['brightness', 'contrast', 'sharpness', 'n_points', 'near_pts', 'far_pts', 'mean_dist', 'z_mean', 'z_std']:
        good_vals = [s[metric] for s in best_stats if s[metric] is not None]
        bad_vals = [s[metric] for s in worst_stats if s[metric] is not None]
        if good_vals and bad_vals:
            print(f"  {metric:>12s}: GOOD={np.mean(good_vals):>10.2f} ± {np.std(good_vals):>6.2f} | "
                  f"BAD={np.mean(bad_vals):>10.2f} ± {np.std(bad_vals):>6.2f} | "
                  f"delta={np.mean(bad_vals)-np.mean(good_vals):+.2f}")


# =============================================================================
# TASK C: Seq 08 fx OOD + Intrinsic Aug Coverage
# =============================================================================
def task_c_seq08_ood():
    print("\n" + "=" * 80)
    print("TASK C: Seq 08 (LPD19A-4) fx OOD + Intrinsic Augmentation Coverage")
    print("=" * 80)
    
    # Parse all calibrations
    train_calibs = {}
    test_calibs = {}
    
    for root, calibs, label in [(TRAIN_ROOT, train_calibs, "Train"), (DATA_ROOT, test_calibs, "Test")]:
        seq_dir = os.path.join(root, 'sequences')
        if not os.path.exists(seq_dir):
            print(f"  Warning: {seq_dir} not found")
            continue
        for seq in sorted(os.listdir(seq_dir)):
            calib_path = os.path.join(seq_dir, seq, 'calib.txt')
            if not os.path.exists(calib_path):
                continue
            K, Tr = load_calib(root, seq)
            if K is not None:
                calibs[seq] = {
                    'fx': K[0, 0], 'fy': K[1, 1],
                    'cx': K[0, 2], 'cy': K[1, 2],
                    'K': K,
                }
                if Tr is not None:
                    from scipy.spatial.transform import Rotation
                    R = Tr[:3, :3]
                    r = Rotation.from_matrix(R)
                    euler = r.as_euler('xyz', degrees=True)
                    calibs[seq]['roll'] = euler[0]
                    calibs[seq]['pitch'] = euler[1]
                    calibs[seq]['yaw'] = euler[2]
    
    # Extract statistics
    train_fx = [c['fx'] for c in train_calibs.values()]
    train_fy = [c['fy'] for c in train_calibs.values()]
    train_cx = [c['cx'] for c in train_calibs.values()]
    train_cy = [c['cy'] for c in train_calibs.values()]
    
    test_fx = [c['fx'] for c in test_calibs.values()]
    test_fy = [c['fy'] for c in test_calibs.values()]
    test_cx = [c['cx'] for c in test_calibs.values()]
    test_cy = [c['cy'] for c in test_calibs.values()]
    
    print("\n--- Intrinsic Parameter Ranges ---")
    for param, tr_vals, te_vals in [
        ('fx', train_fx, test_fx), ('fy', train_fy, test_fy),
        ('cx', train_cx, test_cx), ('cy', train_cy, test_cy)
    ]:
        tr_min, tr_max, tr_mean = min(tr_vals), max(tr_vals), np.mean(tr_vals)
        te_min, te_max, te_mean = min(te_vals), max(te_vals), np.mean(te_vals)
        print(f"  {param}: Train=[{tr_min:.1f}, {tr_max:.1f}] mean={tr_mean:.1f} | "
              f"Test=[{te_min:.1f}, {te_max:.1f}] mean={te_mean:.1f}")
    
    # Per-test-sequence OOD analysis
    print("\n--- Per-Test-Sequence OOD Analysis ---")
    print(f"  {'Seq':>4} {'Vehicle':>12} {'fx':>10} {'OOD_fx':>8} {'fy':>10} {'OOD_fy':>8} {'cx':>10} {'cy':>10}")
    
    tr_fx_min, tr_fx_max = min(train_fx), max(train_fx)
    tr_fy_min, tr_fy_max = min(train_fy), max(train_fy)
    
    for seq in sorted(test_calibs.keys()):
        c = test_calibs[seq]
        ood_fx = ""
        ood_fy = ""
        if c['fx'] < tr_fx_min:
            ood_fx = f"-{tr_fx_min - c['fx']:.1f}"
        elif c['fx'] > tr_fx_max:
            ood_fx = f"+{c['fx'] - tr_fx_max:.1f}"
        if c['fy'] < tr_fy_min:
            ood_fy = f"-{tr_fy_min - c['fy']:.1f}"
        elif c['fy'] > tr_fy_max:
            ood_fy = f"+{c['fy'] - tr_fy_max:.1f}"
        
        seq_int = int(seq)
        vehicle = SEQ_INFO.get(seq_int, '?')
        print(f"  {seq:>4} {vehicle:>12} {c['fx']:>10.1f} {ood_fx:>8} {c['fy']:>10.1f} {ood_fy:>8} {c['cx']:>10.1f} {c['cy']:>10.1f}")
    
    # Intrinsic Augmentation coverage analysis
    print("\n--- V21 Intrinsic Augmentation Coverage ---")
    print("  Config: fx/fy ±3%, cx/cy ±3%")
    
    tr_fx_mean = np.mean(train_fx)
    tr_fy_mean = np.mean(train_fy)
    tr_cx_mean = np.mean(train_cx)
    tr_cy_mean = np.mean(train_cy)
    
    aug_pcts = [0.01, 0.02, 0.03, 0.05]
    
    print(f"\n  Augmentation coverage for Seq 08 (LPD19A-4, fx=7023.9):")
    for pct in aug_pcts:
        aug_fx_min = tr_fx_mean * (1 - pct)
        aug_fx_max = tr_fx_mean * (1 + pct)
        covered = "YES" if test_calibs['08']['fx'] >= aug_fx_min else "NO"
        print(f"    ±{pct*100:.0f}%: fx range=[{aug_fx_min:.1f}, {aug_fx_max:.1f}] → "
              f"Seq08 fx={test_calibs['08']['fx']:.1f} → {covered} "
              f"(gap={test_calibs['08']['fx'] - aug_fx_min:.1f})")
    
    for seq in ['06', '08']:
        if seq in test_calibs:
            c = test_calibs[seq]
            print(f"\n  Seq {seq} ({SEQ_INFO.get(int(seq), '?')}) coverage analysis:")
            for param, val, tr_mean in [('fx', c['fx'], tr_fx_mean), ('fy', c['fy'], tr_fy_mean),
                                         ('cx', c['cx'], tr_cx_mean), ('cy', c['cy'], tr_cy_mean)]:
                pct_diff = (val - tr_mean) / tr_mean * 100
                covered_3pct = abs(pct_diff) <= 3.0
                print(f"    {param}: value={val:.1f} train_mean={tr_mean:.1f} diff={pct_diff:+.2f}% → "
                      f"{'COVERED by ±3%' if covered_3pct else 'NOT covered by ±3%'}")
    
    # Correlation: intrinsic distance vs error
    print("\n--- Intrinsic Distance vs Error Correlation ---")
    samples = parse_errors_only(os.path.join(EVAL_BASE, "v20-v8recipe-pitch-wt3", "extrinsics_and_errors.txt"))
    
    seq_errors = {}
    for sid, s in samples.items():
        seq = s['seq']
        if seq not in seq_errors:
            seq_errors[seq] = []
        seq_errors[seq].append(s.get('total', 0))
    
    print(f"  {'Seq':>4} {'Vehicle':>12} {'fx':>8} {'fx_diff':>10} {'MeanErr':>8} {'MaxErr':>8}")
    for seq in sorted(test_calibs.keys()):
        seq_int = int(seq)
        c = test_calibs[seq]
        fx_diff = c['fx'] - tr_fx_mean
        errs = seq_errors.get(seq_int, [])
        mean_err = np.mean(errs) if errs else 0
        max_err = np.max(errs) if errs else 0
        print(f"  {seq:>4} {SEQ_INFO.get(seq_int,'?'):>12} {c['fx']:>8.1f} {fx_diff:>+10.1f} {mean_err:>8.4f} {max_err:>8.4f}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_a_seq03_burst()
    task_b_seq07_good_vs_bad()
    task_c_seq08_ood()
