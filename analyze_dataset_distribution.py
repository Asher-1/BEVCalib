#!/usr/bin/env python3
"""Comprehensive dataset distribution analysis: training vs test."""
import os, sys, glob, json, struct
import numpy as np
from pathlib import Path
from collections import defaultdict

TRAIN_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data"
TEST_ROOT  = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"

def parse_calib(calib_path):
    """Parse KITTI-style calib.txt -> dict of arrays."""
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            key = key.strip()
            vals = vals.strip()
            try:
                data[key] = np.array([float(x) for x in vals.split()])
            except ValueError:
                data[key] = vals
    return data

def rotation_to_euler(R):
    """3x3 rotation -> (roll, pitch, yaw) in degrees."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def analyze_calibration(root, name):
    """Extract intrinsics and extrinsics for all sequences."""
    seq_dir = os.path.join(root, "sequences")
    seqs = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])
    
    results = {
        'name': name, 'seqs': seqs,
        'fx': [], 'fy': [], 'cx': [], 'cy': [],
        'tr_roll': [], 'tr_pitch': [], 'tr_yaw': [],
        'tr_tx': [], 'tr_ty': [], 'tr_tz': [],
        'cam2s_roll': [], 'cam2s_pitch': [], 'cam2s_yaw': [],
        'cam2s_tx': [], 'cam2s_ty': [], 'cam2s_tz': [],
        'frame_counts': [],
    }
    
    for seq in seqs:
        calib_path = os.path.join(seq_dir, seq, "calib.txt")
        if not os.path.exists(calib_path):
            continue
        calib = parse_calib(calib_path)
        
        P2 = calib['P2'].reshape(3, 4)
        fx, fy = P2[0, 0], P2[1, 1]
        cx, cy = P2[0, 2], P2[1, 2]
        results['fx'].append(fx)
        results['fy'].append(fy)
        results['cx'].append(cx)
        results['cy'].append(cy)
        
        Tr = calib['Tr'].reshape(3, 4)
        R_tr = Tr[:, :3]
        t_tr = Tr[:, 3]
        roll, pitch, yaw = rotation_to_euler(R_tr)
        results['tr_roll'].append(roll)
        results['tr_pitch'].append(pitch)
        results['tr_yaw'].append(yaw)
        results['tr_tx'].append(t_tr[0])
        results['tr_ty'].append(t_tr[1])
        results['tr_tz'].append(t_tr[2])
        
        if 'T_cam2sensing' in calib:
            T_cs = calib['T_cam2sensing'].reshape(3, 4)
            R_cs = T_cs[:, :3]
            t_cs = T_cs[:, 3]
            r2, p2, y2 = rotation_to_euler(R_cs)
            results['cam2s_roll'].append(r2)
            results['cam2s_pitch'].append(p2)
            results['cam2s_yaw'].append(y2)
            results['cam2s_tx'].append(t_cs[0])
            results['cam2s_ty'].append(t_cs[1])
            results['cam2s_tz'].append(t_cs[2])
        
        vel_dir = os.path.join(seq_dir, seq, "velodyne")
        if os.path.exists(vel_dir):
            n_frames = len(os.listdir(vel_dir))
        else:
            n_frames = 0
        results['frame_counts'].append(n_frames)
    
    return results

def analyze_pointclouds_sample(root, n_samples=5):
    """Sample a few point clouds per sequence, compute stats."""
    seq_dir = os.path.join(root, "sequences")
    seqs = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])
    
    all_stats = {
        'n_points': [], 'x_range': [], 'y_range': [], 'z_range': [],
        'x_mean': [], 'y_mean': [], 'z_mean': [],
        'x_std': [], 'y_std': [], 'z_std': [],
        'intensity_mean': [], 'intensity_std': [],
    }
    
    for seq in seqs:
        vel_dir = os.path.join(seq_dir, seq, "velodyne")
        if not os.path.exists(vel_dir):
            continue
        files = sorted(glob.glob(os.path.join(vel_dir, "*.bin")))
        if not files:
            continue
        step = max(1, len(files) // n_samples)
        sampled = files[::step][:n_samples]
        
        for f in sampled:
            pc = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            xyz = pc[:, :3]
            intensity = pc[:, 3]
            
            all_stats['n_points'].append(len(pc))
            all_stats['x_range'].append(xyz[:, 0].max() - xyz[:, 0].min())
            all_stats['y_range'].append(xyz[:, 1].max() - xyz[:, 1].min())
            all_stats['z_range'].append(xyz[:, 2].max() - xyz[:, 2].min())
            all_stats['x_mean'].append(xyz[:, 0].mean())
            all_stats['y_mean'].append(xyz[:, 1].mean())
            all_stats['z_mean'].append(xyz[:, 2].mean())
            all_stats['x_std'].append(xyz[:, 0].std())
            all_stats['y_std'].append(xyz[:, 1].std())
            all_stats['z_std'].append(xyz[:, 2].std())
            all_stats['intensity_mean'].append(intensity.mean())
            all_stats['intensity_std'].append(intensity.std())
    
    return {k: np.array(v) for k, v in all_stats.items()}

def analyze_images_sample(root, n_samples=3):
    """Sample images to check resolution."""
    from PIL import Image
    seq_dir = os.path.join(root, "sequences")
    seqs = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])
    
    orig_sizes = []
    resized_sizes = []
    
    for seq in seqs:
        orig_dir = os.path.join(seq_dir, seq, "image_2")
        resized_dir = os.path.join(seq_dir, seq, "image_2_640x360")
        
        if os.path.exists(orig_dir):
            files = sorted(glob.glob(os.path.join(orig_dir, "*.png")) + glob.glob(os.path.join(orig_dir, "*.jpg")))
            if files:
                step = max(1, len(files) // n_samples)
                for f in files[::step][:n_samples]:
                    img = Image.open(f)
                    orig_sizes.append(img.size)
        
        if os.path.exists(resized_dir):
            files = sorted(glob.glob(os.path.join(resized_dir, "*.png")) + glob.glob(os.path.join(resized_dir, "*.jpg")))
            if files:
                step = max(1, len(files) // n_samples)
                for f in files[::step][:n_samples]:
                    img = Image.open(f)
                    resized_sizes.append(img.size)
    
    return orig_sizes, resized_sizes

def print_stat(name, arr):
    a = np.array(arr)
    return f"{name}: mean={a.mean():.4f}, std={a.std():.4f}, min={a.min():.4f}, max={a.max():.4f}, range={a.max()-a.min():.4f}"

def print_comparison(label, train_arr, test_arr):
    t = np.array(train_arr)
    s = np.array(test_arr)
    delta_mean = abs(t.mean() - s.mean())
    overlap = max(0, min(t.max(), s.max()) - max(t.min(), s.min()))
    total_range = max(t.max(), s.max()) - min(t.min(), s.min())
    overlap_pct = (overlap / total_range * 100) if total_range > 0 else 100
    
    print(f"\n  {label}:")
    print(f"    TRAIN: mean={t.mean():.6f}, std={t.std():.6f}, [{t.min():.6f}, {t.max():.6f}]")
    print(f"    TEST:  mean={s.mean():.6f}, std={s.std():.6f}, [{s.min():.6f}, {s.max():.6f}]")
    print(f"    Delta(mean): {delta_mean:.6f}, Range overlap: {overlap_pct:.1f}%")
    
    return delta_mean, overlap_pct

def main():
    print("=" * 80)
    print("DATASET DISTRIBUTION ANALYSIS")
    print(f"  Training: {TRAIN_ROOT} ")
    print(f"  Test:     {TEST_ROOT}")
    print("=" * 80)
    
    print("\n[1/5] Analyzing calibration parameters...")
    train_calib = analyze_calibration(TRAIN_ROOT, "TRAIN")
    test_calib = analyze_calibration(TEST_ROOT, "TEST")
    
    print(f"\n  Train: {len(train_calib['seqs'])} sequences, {sum(train_calib['frame_counts'])} total frames")
    print(f"  Test:  {len(test_calib['seqs'])} sequences, {sum(test_calib['frame_counts'])} total frames")
    
    print("\n" + "=" * 80)
    print("[2/5] INTRINSICS (from P2 matrix, original 4K resolution)")
    print("=" * 80)
    
    critical_diffs = {}
    
    for param in ['fx', 'fy', 'cx', 'cy']:
        d, o = print_comparison(param, train_calib[param], test_calib[param])
        critical_diffs[f'intrinsic_{param}'] = (d, o)
    
    train_fov_h = 2 * np.arctan(np.array(train_calib['cx']) / np.array(train_calib['fx']))
    test_fov_h = 2 * np.arctan(np.array(test_calib['cx']) / np.array(test_calib['fx']))
    train_fov_v = 2 * np.arctan(np.array(train_calib['cy']) / np.array(train_calib['fy']))
    test_fov_v = 2 * np.arctan(np.array(test_calib['cy']) / np.array(test_calib['fy']))
    
    print(f"\n  Derived FOV (horizontal):")
    print(f"    TRAIN: mean={np.degrees(train_fov_h.mean()):.2f} deg, range=[{np.degrees(train_fov_h.min()):.2f}, {np.degrees(train_fov_h.max()):.2f}]")
    print(f"    TEST:  mean={np.degrees(test_fov_h.mean()):.2f} deg, range=[{np.degrees(test_fov_h.min()):.2f}, {np.degrees(test_fov_h.max()):.2f}]")
    print(f"  Derived FOV (vertical):")
    print(f"    TRAIN: mean={np.degrees(train_fov_v.mean()):.2f} deg, range=[{np.degrees(train_fov_v.min()):.2f}, {np.degrees(train_fov_v.max()):.2f}]")
    print(f"    TEST:  mean={np.degrees(test_fov_v.mean()):.2f} deg, range=[{np.degrees(test_fov_v.min()):.2f}, {np.degrees(test_fov_v.max()):.2f}]")
    
    print("\n" + "=" * 80)
    print("[3/5] EXTRINSICS - Tr (LiDAR -> Camera) and T_cam2sensing")
    print("=" * 80)
    
    print("\n  --- Tr (LiDAR->Camera) Rotation (Euler degrees) ---")
    for param in ['tr_roll', 'tr_pitch', 'tr_yaw']:
        d, o = print_comparison(param, train_calib[param], test_calib[param])
        critical_diffs[f'extrinsic_{param}'] = (d, o)
    
    print("\n  --- Tr (LiDAR->Camera) Translation (meters) ---")
    for param in ['tr_tx', 'tr_ty', 'tr_tz']:
        d, o = print_comparison(param, train_calib[param], test_calib[param])
        critical_diffs[f'extrinsic_{param}'] = (d, o)
    
    print("\n  --- T_cam2sensing Rotation (Euler degrees) ---")
    for param in ['cam2s_roll', 'cam2s_pitch', 'cam2s_yaw']:
        d, o = print_comparison(param, train_calib[param], test_calib[param])
        critical_diffs[f'cam2sensing_{param}'] = (d, o)
    
    print("\n  --- T_cam2sensing Translation (meters) ---")
    for param in ['cam2s_tx', 'cam2s_ty', 'cam2s_tz']:
        d, o = print_comparison(param, train_calib[param], test_calib[param])
        critical_diffs[f'cam2sensing_{param}'] = (d, o)
    
    print("\n  --- Per-Sequence Tr Pitch comparison ---")
    print(f"  {'Seq':>4s} | {'TRAIN Pitch':>12s} | {'TEST Pitch':>12s} | {'Delta':>10s}")
    print(f"  {'-'*4} | {'-'*12} | {'-'*12} | {'-'*10}")
    n_common = min(len(train_calib['tr_pitch']), len(test_calib['tr_pitch']))
    for i in range(max(len(train_calib['tr_pitch']), len(test_calib['tr_pitch']))):
        t_val = f"{train_calib['tr_pitch'][i]:.6f}" if i < len(train_calib['tr_pitch']) else "N/A"
        s_val = f"{test_calib['tr_pitch'][i]:.6f}" if i < len(test_calib['tr_pitch']) else "N/A"
        t_seq = train_calib['seqs'][i] if i < len(train_calib['seqs']) else "?"
        s_seq = test_calib['seqs'][i] if i < len(test_calib['seqs']) else "?"
        delta = ""
        if i < len(train_calib['tr_pitch']) and i < len(test_calib['tr_pitch']):
            delta = f"{abs(train_calib['tr_pitch'][i] - test_calib['tr_pitch'][i]):.6f}"
        print(f"  {t_seq:>4s} | {t_val:>12s} | {s_val:>12s} | {delta:>10s}")
    
    print("\n" + "=" * 80)
    print("[4/5] POINT CLOUD STATISTICS (sampled)")
    print("=" * 80)
    
    print("\n  Sampling point clouds (5 per sequence)...")
    train_pc = analyze_pointclouds_sample(TRAIN_ROOT, n_samples=5)
    test_pc = analyze_pointclouds_sample(TEST_ROOT, n_samples=5)
    
    for param in ['n_points', 'x_range', 'y_range', 'z_range', 
                   'x_mean', 'y_mean', 'z_mean',
                   'x_std', 'y_std', 'z_std',
                   'intensity_mean', 'intensity_std']:
        d, o = print_comparison(f"PC {param}", train_pc[param], test_pc[param])
        critical_diffs[f'pc_{param}'] = (d, o)
    
    print("\n" + "=" * 80)
    print("[5/5] IMAGE ANALYSIS")
    print("=" * 80)
    
    print("\n  Sampling images...")
    train_orig, train_resized = analyze_images_sample(TRAIN_ROOT, n_samples=2)
    test_orig, test_resized = analyze_images_sample(TEST_ROOT, n_samples=2)
    
    if train_orig:
        train_orig_set = set(train_orig)
        print(f"\n  Train original sizes: {train_orig_set}")
    if test_orig:
        test_orig_set = set(test_orig)
        print(f"  Test  original sizes: {test_orig_set}")
    if train_resized:
        train_resized_set = set(train_resized)
        print(f"  Train resized sizes:  {train_resized_set}")
    if test_resized:
        test_resized_set = set(test_resized)
        print(f"  Test  resized sizes:  {test_resized_set}")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    print("\n  Top differences by delta(mean):")
    sorted_diffs = sorted(critical_diffs.items(), key=lambda x: x[1][0], reverse=True)
    for k, (d, o) in sorted_diffs[:15]:
        flag = " *** SIGNIFICANT ***" if (o < 70 or d > 0.1) else ""
        print(f"    {k:30s}: delta_mean={d:.6f}, overlap={o:.1f}%{flag}")
    
    print("\n  Model input analysis for inference:")
    print("  ─────────────────────────────────")
    
    train_fx_norm = np.array(train_calib['fx']) / 3840  # 4K width
    test_fx_norm = np.array(test_calib['fx']) / 3840
    print(f"\n  Normalized fx (fx/4K_width):")
    print(f"    TRAIN: mean={train_fx_norm.mean():.6f}, std={train_fx_norm.std():.6f}")
    print(f"    TEST:  mean={test_fx_norm.mean():.6f}, std={test_fx_norm.std():.6f}")
    
    train_scaled_fx = np.array(train_calib['fx']) * 640 / 3840
    test_scaled_fx = np.array(test_calib['fx']) * 640 / 3840
    train_scaled_fy = np.array(train_calib['fy']) * 360 / 2160
    test_scaled_fy = np.array(test_calib['fy']) * 360 / 2160
    train_scaled_cx = np.array(train_calib['cx']) * 640 / 3840
    test_scaled_cx = np.array(test_calib['cx']) * 640 / 3840
    train_scaled_cy = np.array(train_calib['cy']) * 360 / 2160
    test_scaled_cy = np.array(test_calib['cy']) * 360 / 2160
    
    print(f"\n  Scaled intrinsics (to 640x360 input):")
    print(f"    fx: TRAIN mean={train_scaled_fx.mean():.2f} | TEST mean={test_scaled_fx.mean():.2f} | delta={abs(train_scaled_fx.mean()-test_scaled_fx.mean()):.2f}")
    print(f"    fy: TRAIN mean={train_scaled_fy.mean():.2f} | TEST mean={test_scaled_fy.mean():.2f} | delta={abs(train_scaled_fy.mean()-test_scaled_fy.mean()):.2f}")
    print(f"    cx: TRAIN mean={train_scaled_cx.mean():.2f} | TEST mean={test_scaled_cx.mean():.2f} | delta={abs(train_scaled_cx.mean()-test_scaled_cx.mean()):.2f}")
    print(f"    cy: TRAIN mean={train_scaled_cy.mean():.2f} | TEST mean={test_scaled_cy.mean():.2f} | delta={abs(train_scaled_cy.mean()-test_scaled_cy.mean()):.2f}")
    
    print(f"\n  GT Extrinsic Euler angles (what model tries to predict):")
    train_gt_pitch = np.array(train_calib['tr_pitch'])
    test_gt_pitch = np.array(test_calib['tr_pitch'])
    print(f"    Tr Pitch: TRAIN mean={train_gt_pitch.mean():.4f}° range=[{train_gt_pitch.min():.4f}°, {train_gt_pitch.max():.4f}°]")
    print(f"              TEST  mean={test_gt_pitch.mean():.4f}° range=[{test_gt_pitch.min():.4f}°, {test_gt_pitch.max():.4f}°]")
    print(f"              Delta(mean)={abs(train_gt_pitch.mean()-test_gt_pitch.mean()):.4f}°")
    print(f"              TRAIN range width: {train_gt_pitch.max()-train_gt_pitch.min():.4f}°")
    print(f"              TEST  range width: {test_gt_pitch.max()-test_gt_pitch.min():.4f}°")
    
    print("\n  Scene diversity:")
    print(f"    TRAIN: {len(train_calib['seqs'])} sequences, {sum(train_calib['frame_counts'])} frames")
    print(f"    TEST:  {len(test_calib['seqs'])} sequences, {sum(test_calib['frame_counts'])} frames")
    
    train_unique_cars = len(set([
        (f"{train_calib['fx'][i]:.0f}", f"{train_calib['fy'][i]:.0f}")
        for i in range(len(train_calib['fx']))
    ]))
    test_unique_cars = len(set([
        (f"{test_calib['fx'][i]:.0f}", f"{test_calib['fy'][i]:.0f}")
        for i in range(len(test_calib['fx']))
    ]))
    print(f"    TRAIN unique (fx,fy) combos (≈ unique vehicles): {train_unique_cars}")
    print(f"    TEST  unique (fx,fy) combos (≈ unique vehicles): {test_unique_cars}")
    
    print("\n  T_cam2sensing (sensor mounting) analysis:")
    t_tz = np.array(train_calib['cam2s_tz'])
    s_tz = np.array(test_calib['cam2s_tz'])
    print(f"    cam2s_tz (height): TRAIN mean={t_tz.mean():.4f}m range=[{t_tz.min():.4f}, {t_tz.max():.4f}]")
    print(f"                       TEST  mean={s_tz.mean():.4f}m range=[{s_tz.min():.4f}, {s_tz.max():.4f}]")
    t_tx = np.array(train_calib['cam2s_tx'])
    s_tx = np.array(test_calib['cam2s_tx'])
    print(f"    cam2s_tx (forward): TRAIN mean={t_tx.mean():.4f}m range=[{t_tx.min():.4f}, {t_tx.max():.4f}]")
    print(f"                        TEST  mean={s_tx.mean():.4f}m range=[{s_tx.min():.4f}, {s_tx.max():.4f}]")

    print("\n" + "=" * 80)
    print("ROOT CAUSE DIAGNOSIS")
    print("=" * 80)
    
    intrinsic_similar = all(critical_diffs.get(f'intrinsic_{p}', (0,100))[1] > 80 for p in ['fx','fy','cx','cy'])
    extrinsic_pitch_delta = critical_diffs.get('extrinsic_tr_pitch', (0,100))[0]
    pc_similar = all(critical_diffs.get(f'pc_{p}', (0,100))[1] > 50 for p in ['n_points','x_range','y_range','z_range'])
    
    print(f"\n  Intrinsics consistent: {'YES' if intrinsic_similar else 'NO'}")
    print(f"  GT Pitch delta:        {extrinsic_pitch_delta:.4f}°")
    print(f"  Point cloud similar:   {'YES' if pc_similar else 'NO'}")
    
    tr_pitch_train_range = np.array(train_calib['tr_pitch'])
    tr_pitch_test_range = np.array(test_calib['tr_pitch'])
    test_in_train = sum(1 for p in tr_pitch_test_range 
                        if tr_pitch_train_range.min() <= p <= tr_pitch_train_range.max())
    print(f"\n  Test GT Pitch values within train range: {test_in_train}/{len(tr_pitch_test_range)}")
    
    cam2s_pitch_train = np.array(train_calib['cam2s_pitch'])
    cam2s_pitch_test = np.array(test_calib['cam2s_pitch'])
    print(f"\n  cam2sensing Pitch:")
    print(f"    TRAIN: mean={cam2s_pitch_train.mean():.4f}°, range=[{cam2s_pitch_train.min():.4f}°, {cam2s_pitch_train.max():.4f}°]")
    print(f"    TEST:  mean={cam2s_pitch_test.mean():.4f}°, range=[{cam2s_pitch_test.min():.4f}°, {cam2s_pitch_test.max():.4f}°]")
    cam2s_overlap = max(0, min(cam2s_pitch_train.max(), cam2s_pitch_test.max()) - max(cam2s_pitch_train.min(), cam2s_pitch_test.min()))
    cam2s_total = max(cam2s_pitch_train.max(), cam2s_pitch_test.max()) - min(cam2s_pitch_train.min(), cam2s_pitch_test.min())
    print(f"    Range overlap: {cam2s_overlap/cam2s_total*100:.1f}%")

if __name__ == "__main__":
    main()
