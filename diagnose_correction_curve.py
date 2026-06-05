"""
Diagnose calibration model's correction ability with progressive bias sweep.

Generates correction-rate curves across bias magnitudes (0.1° to 10°) for each axis,
revealing whether the model has genuine T_init-adaptive calibration or shortcut learning.

Usage:
    python diagnose_correction_curve.py \
        --ckpt_path <checkpoint> \
        --trip_dir <trip_data_dir> \
        --output_dir <results_dir>
"""

import argparse
import os
import sys
import json
import time

import numpy as np
import torch

_KITTI_PATH = os.path.join(os.path.dirname(__file__), "kitti-bev-calib")
if _KITTI_PATH not in sys.path:
    sys.path.insert(0, _KITTI_PATH)

from scipy.spatial.transform import Rotation as R_sp


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    return R_sp.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


def geodesic_rotation_deg(R1, R2):
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle_rad)


def apply_bias_to_T(T_4x4, roll_deg=0, pitch_deg=0, yaw_deg=0):
    bias_R = euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    T_biased = T_4x4.copy()
    T_biased[:3, :3] = bias_R @ T_4x4[:3, :3]
    return T_biased


def run_inference_batch(wrapper, frames, T_init_4x4, K_full, device, batch_size=8):
    """Run model inference with a given T_init, return predicted T matrices."""
    all_preds = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]
        B = len(batch_frames)

        imgs_np = np.stack([f["image"] for f in batch_frames])
        pcs_list = [f["pc"] for f in batch_frames]
        max_pts = max(p.shape[0] for p in pcs_list)
        pcs_padded = np.zeros((B, max_pts, 3), dtype=np.float32)
        for i, pc in enumerate(pcs_list):
            n = pc.shape[0]
            pcs_padded[i, :n, :3] = pc[:, :3]

        imgs_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float().to(device)
        pcs_t = torch.from_numpy(pcs_padded).float().to(device)
        init_T_t = torch.from_numpy(
            np.tile(T_init_4x4.astype(np.float32), (B, 1, 1))).to(device)
        post_T_t = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        K_t = torch.from_numpy(
            np.tile(K_full.astype(np.float32), (B, 1, 1))).to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    T_pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
            else:
                T_pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
        all_preds.append(T_pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Progressive Bias Correction Curve Diagnostic")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--trip_dir", type=str, required=True)
    parser.add_argument("--camera_name", type=str, default="traffic_2")
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label", type=str, default="model",
                        help="Label for this model in the output (e.g. v32.1, v33)")
    args = parser.parse_args()

    _proj_root = os.path.dirname(os.path.abspath(__file__))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    _prep_path = os.path.join(_proj_root, "kitti-bev-calib", "tools", "preparation")
    if _prep_path not in sys.path:
        sys.path.insert(0, _prep_path)

    from run_bag_calibration import (
        compute_T_lidar_to_cam, load_bevcalib_inference, _lazy_bev_bounds,
        find_trip_config_dir, find_trip_bag_paths, stage_bags_symlink,
        subsample_bags_for_calibration, parse_kitti_calib_txt,
        list_sequence_frames, load_raw_pointcloud, filter_pointcloud_for_bev,
    )
    from prepare_custom_dataset import ConfigParser, BEVCalibDatasetPreparer
    import cv2
    import tempfile

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config_dir = find_trip_config_dir(args.trip_dir)
    cameras = ConfigParser.parse_cameras_cfg(os.path.join(config_dir, "cameras.cfg"))
    lidars = ConfigParser.parse_lidars_cfg(os.path.join(config_dir, "lidars.cfg"))
    T_orig = compute_T_lidar_to_cam(cameras[args.camera_name], lidars)

    print("=" * 80)
    print(f"Progressive Bias Correction Curve [{args.label}]")
    print("=" * 80)

    extract_root = tempfile.mkdtemp(prefix="bevcalib_diag_")
    bag_paths = find_trip_bag_paths(args.trip_dir)
    bag_paths = subsample_bags_for_calibration(bag_paths, target_frames=args.max_frames * 3)
    bag_root = stage_bags_symlink(bag_paths, os.path.join(extract_root, "bag_staging"))

    print("1. Extracting data...")
    preparer = BEVCalibDatasetPreparer(
        bag_path=bag_root, config_dir=config_dir, output_dir=extract_root,
        camera_name=args.camera_name, target_fps=10.0, max_time_diff=0.055,
        batch_size=500, num_workers=8, max_frames=args.max_frames * 3,
        save_debug_samples=0, max_pose_gap=0.5, force_config=False, sequence_id="00",
    )
    preparer.extract_data_from_bag()
    preparer.sync_and_save(sequence_id="00")

    seq_dir = os.path.join(extract_root, "sequences", "00")
    calib_info = parse_kitti_calib_txt(os.path.join(seq_dir, "calib.txt"))
    K_full = calib_info["K"].copy()
    stems = list_sequence_frames(seq_dir)
    xbound, ybound, zbound = _lazy_bev_bounds()

    print("2. Loading frames...")
    frames = []
    for stem in stems[:args.max_frames]:
        ip = os.path.join(seq_dir, "image_2", stem + ".png")
        if not os.path.isfile(ip):
            ip = os.path.join(seq_dir, "image_2", stem + ".jpg")
        pp = os.path.join(seq_dir, "velodyne", stem + ".bin")
        im = cv2.imread(ip, cv2.IMREAD_COLOR)
        if im is None:
            continue
        pc_raw = load_raw_pointcloud(pp)
        pc_f = filter_pointcloud_for_bev(pc_raw, xbound, ybound, zbound)
        if pc_f.shape[0] < 10:
            continue
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im_rgb, (640, 360))
        frames.append({"image": im_resized, "pc": pc_f, "K": K_full, "stem": stem})
    print(f"   Loaded {len(frames)} frames")

    print("3. Loading model...")
    wrapper, epoch = load_bevcalib_inference(args.ckpt_path, device=str(device), img_shape=(360, 640))
    print(f"   Epoch: {epoch}")

    BIAS_MAGNITUDES = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    AXES = [
        ("Roll+", lambda d: (d, 0, 0)),
        ("Roll-", lambda d: (-d, 0, 0)),
        ("Pitch+", lambda d: (0, d, 0)),
        ("Pitch-", lambda d: (0, -d, 0)),
        ("Yaw+", lambda d: (0, 0, d)),
        ("Yaw-", lambda d: (0, 0, -d)),
        ("Combined+", lambda d: (d/np.sqrt(3), d/np.sqrt(3), d/np.sqrt(3))),
    ]

    print(f"\n4. Running {len(BIAS_MAGNITUDES)} x {len(AXES)} = {len(BIAS_MAGNITUDES)*len(AXES)} bias sweep tests...")
    print("-" * 100)

    results = {}
    for axis_name, bias_fn in AXES:
        axis_results = []
        for bias_mag in BIAS_MAGNITUDES:
            r, p, y = bias_fn(bias_mag)
            T_biased = apply_bias_to_T(T_orig, r, p, y)
            actual_bias = geodesic_rotation_deg(T_orig[:3, :3], T_biased[:3, :3])

            all_preds = run_inference_batch(
                wrapper, frames, T_biased, K_full, device, args.batch_size)

            corrections, residuals = [], []
            for i in range(all_preds.shape[0]):
                pred_R = all_preds[i, :3, :3]
                corrections.append(geodesic_rotation_deg(T_biased[:3, :3], pred_R))
                residuals.append(geodesic_rotation_deg(T_orig[:3, :3], pred_R))

            mean_corr = np.mean(corrections)
            mean_resid = np.mean(residuals)
            corr_pct = (mean_corr / max(actual_bias, 0.001)) * 100 if actual_bias > 0.01 else 0

            axis_results.append({
                "bias_target_deg": bias_mag,
                "bias_actual_deg": actual_bias,
                "mean_correction_deg": mean_corr,
                "mean_residual_deg": mean_resid,
                "correction_pct": corr_pct,
                "std_correction": float(np.std(corrections)),
                "std_residual": float(np.std(residuals)),
            })

            status = "GOOD" if corr_pct > 60 else ("PART" if corr_pct > 30 else "FAIL")
            print(f"  {axis_name:>10s} bias={actual_bias:6.2f}° → corr={mean_corr:6.3f}° resid={mean_resid:6.3f}° rate={corr_pct:6.1f}% [{status}]")

        results[axis_name] = axis_results

    os.makedirs(args.output_dir, exist_ok=True)

    report = {
        "model_label": args.label,
        "checkpoint": args.ckpt_path,
        "epoch": epoch,
        "trip": os.path.basename(args.trip_dir),
        "n_frames": len(frames),
        "bias_magnitudes": BIAS_MAGNITUDES,
        "axes": [a[0] for a in AXES],
        "results": results,
    }
    report_path = os.path.join(args.output_dir, f"correction_curve_{args.label}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY: Correction Rate vs Bias Magnitude")
    print("=" * 80)
    header = f"{'Bias':>8s}"
    for axis_name, _ in AXES:
        header += f" {axis_name:>10s}"
    print(header)
    print("-" * (8 + 11 * len(AXES)))

    for bi, bias_mag in enumerate(BIAS_MAGNITUDES):
        row = f"{bias_mag:>7.1f}°"
        for axis_name, _ in AXES:
            cpct = results[axis_name][bi]["correction_pct"]
            row += f" {cpct:>9.1f}%"
        print(row)

    ideal_behavior = any(
        results[ax][bi]["correction_pct"] > 50
        for ax in results for bi in range(len(BIAS_MAGNITUDES)) if BIAS_MAGNITUDES[bi] >= 2.0
    )

    print()
    if ideal_behavior:
        print("ASSESSMENT: Model shows ADAPTIVE correction at large biases")
    else:
        avg_high_bias = np.mean([
            results[ax][bi]["correction_pct"]
            for ax in results for bi in range(len(BIAS_MAGNITUDES)) if BIAS_MAGNITUDES[bi] >= 2.0
        ])
        print(f"ASSESSMENT: Model shows FIXED-OFFSET behavior (avg correction rate at >=2° bias: {avg_high_bias:.1f}%)")

    print(f"\nReport saved: {report_path}")

    import shutil
    shutil.rmtree(extract_root, ignore_errors=True)


if __name__ == "__main__":
    main()
