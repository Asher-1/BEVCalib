"""
Test v32.1 model's ability to correct large extrinsic biases.

Injects known rotation biases into T_init and checks if the model can correct them.
This validates that the model has genuine calibration ability (not just T_init passthrough).

Usage:
    python test_bias_correction.py \
        --ckpt_path <checkpoint> \
        --trip_dir <trip_data_dir> \
        --camera_name traffic_2
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_KITTI_PATH = os.path.join(os.path.dirname(__file__), "kitti-bev-calib")
if _KITTI_PATH not in sys.path:
    sys.path.insert(0, _KITTI_PATH)

from scipy.spatial.transform import Rotation as R_sp


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    return R_sp.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


def rotation_matrix_to_euler(R):
    return R_sp.from_matrix(R).as_euler("xyz", degrees=True)


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


def main():
    parser = argparse.ArgumentParser(description="Bias Correction Test for BEVCalib")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--trip_dir", type=str, required=True)
    parser.add_argument("--camera_name", type=str, default="traffic_2")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    _proj_root = os.path.dirname(os.path.abspath(__file__))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    _prep_path = os.path.join(_proj_root, "kitti-bev-calib", "tools", "preparation")
    if _prep_path not in sys.path:
        sys.path.insert(0, _prep_path)

    from run_bag_calibration import (
        compute_T_lidar_to_cam,
        load_bevcalib_inference,
        _lazy_bev_bounds,
        find_trip_config_dir,
        find_trip_bag_paths,
        stage_bags_symlink,
        subsample_bags_for_calibration,
        parse_kitti_calib_txt,
        list_sequence_frames,
        load_raw_pointcloud,
        filter_pointcloud_for_bev,
    )
    from prepare_custom_dataset import ConfigParser, BEVCalibDatasetPreparer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config_dir = find_trip_config_dir(args.trip_dir)
    cameras_cfg = os.path.join(config_dir, "cameras.cfg")
    lidars_cfg = os.path.join(config_dir, "lidars.cfg")
    cameras = ConfigParser.parse_cameras_cfg(cameras_cfg)
    lidars = ConfigParser.parse_lidars_cfg(lidars_cfg)
    cam_cfg = cameras[args.camera_name]
    T_orig = compute_T_lidar_to_cam(cam_cfg, lidars)

    print("=" * 80)
    print("BEVCalib Bias Correction Test")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Trip: {os.path.basename(args.trip_dir)}")
    print(f"Device: {device}")
    print(f"Original T_lidar_to_cam RPY: {rotation_matrix_to_euler(T_orig[:3, :3])}")
    print()

    import tempfile
    extract_root = tempfile.mkdtemp(prefix="bevcalib_bias_test_")
    bag_paths = find_trip_bag_paths(args.trip_dir)
    bag_paths = subsample_bags_for_calibration(bag_paths, target_frames=args.max_frames * 3)
    staging = os.path.join(extract_root, "bag_staging")
    bag_root = stage_bags_symlink(bag_paths, staging)

    print("1. Extracting data from bags...")
    preparer = BEVCalibDatasetPreparer(
        bag_path=bag_root, config_dir=config_dir, output_dir=extract_root,
        camera_name=args.camera_name, target_fps=10.0, max_time_diff=0.055,
        batch_size=500, num_workers=8, max_frames=args.max_frames * 3,
        save_debug_samples=0, max_pose_gap=0.5, force_config=False,
        sequence_id="00",
    )
    preparer.extract_data_from_bag()
    preparer.sync_and_save(sequence_id="00")

    seq_dir = os.path.join(extract_root, "sequences", "00")
    calib_info = parse_kitti_calib_txt(os.path.join(seq_dir, "calib.txt"))
    K_full = calib_info["K"].copy()
    stems = list_sequence_frames(seq_dir)
    print(f"   Extracted {len(stems)} frames")

    print("2. Loading model...")
    wrapper, epoch = load_bevcalib_inference(args.ckpt_path, device=str(device), img_shape=(360, 640))
    print(f"   Checkpoint epoch: {epoch}")

    import cv2
    xbound, ybound, zbound = _lazy_bev_bounds()
    Hm, Wm = 360, 640

    print("3. Loading frames...")
    frames = []
    for fi, stem in enumerate(stems[:args.max_frames]):
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
        im_resized = cv2.resize(im_rgb, (Wm, Hm))
        frames.append({
            "image": im_resized,
            "pc": pc_f,
            "K": K_full,
            "stem": stem,
        })
    print(f"   Loaded {len(frames)} valid frames")

    BIAS_SCENARIOS = [
        ("No bias (baseline)", 0, 0, 0),
        ("Roll +2°", 2, 0, 0),
        ("Roll -2°", -2, 0, 0),
        ("Pitch +2°", 0, 2, 0),
        ("Pitch -2°", 0, -2, 0),
        ("Yaw +2°", 0, 0, 2),
        ("Yaw -2°", 0, 0, -2),
        ("Roll +5°", 5, 0, 0),
        ("Pitch +5°", 0, 5, 0),
        ("Yaw +5°", 0, 0, 5),
        ("Combined +3°/+3°/+3°", 3, 3, 3),
        ("Combined -5°/+2°/-3°", -5, 2, -3),
    ]

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kitti-bev-calib"))
    from tools import generate_single_perturbation_from_T as gen_pert

    print("\n--- Training-style perturbation test (same function as training) ---")
    print(f"{'Label':<30} {'Pert Mag':>10} {'Correction':>10} {'Residual':>10} {'Corr%':>8}")
    print("-" * 70)
    for pert_deg in [1.0, 2.0, 3.0, 5.0]:
        T_gt_batch = np.tile(T_orig[np.newaxis], (len(frames), 1, 1))
        np.random.seed(42)
        T_init_batch, _, _ = gen_pert(
            T_gt_batch, angle_range_deg=pert_deg, rotation_only=True)

        actual_perts = [geodesic_rotation_deg(T_orig[:3,:3], T_init_batch[i,:3,:3])
                        for i in range(len(frames))]
        mean_pert = np.mean(actual_perts)

        all_preds_native = []
        for start in range(0, len(frames), args.batch_size):
            batch_frames = frames[start:start + args.batch_size]
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
                T_init_batch[start:start+B].astype(np.float32)).to(device)
            post_T_t = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
            K_t = torch.from_numpy(
                np.tile(K_full.astype(np.float32), (B, 1, 1))).to(device)

            with torch.no_grad():
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        T_pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
                else:
                    T_pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
            all_preds_native.append(T_pred.cpu().numpy())

        all_preds_native = np.concatenate(all_preds_native, axis=0)
        corrs, resids = [], []
        for i in range(all_preds_native.shape[0]):
            pred_R = all_preds_native[i, :3, :3]
            c = geodesic_rotation_deg(T_init_batch[i, :3, :3], pred_R)
            r = geodesic_rotation_deg(T_orig[:3, :3], pred_R)
            corrs.append(c)
            resids.append(r)
        mean_c = np.mean(corrs)
        mean_r = np.mean(resids)
        cpct = (mean_c / max(mean_pert, 0.01)) * 100
        print(f"±{pert_deg}° (mean={mean_pert:.2f}°) {' ':>7} {mean_c:>10.3f}° {mean_r:>10.3f}° {cpct:>7.1f}%")

    print()
    print("4. Running bias correction tests...")
    print("=" * 80)
    print(f"{'Scenario':<35} {'Bias Mag':>10} {'Pred Corr':>10} {'Resid':>10} {'Corr %':>8} {'Status':>8}")
    print("-" * 80)

    results = []

    for scenario_name, r_bias, p_bias, y_bias in BIAS_SCENARIOS:
        T_biased = apply_bias_to_T(T_orig, r_bias, p_bias, y_bias)
        bias_mag = geodesic_rotation_deg(T_orig[:3, :3], T_biased[:3, :3])

        all_preds = []
        for start in range(0, len(frames), args.batch_size):
            batch_frames = frames[start:start + args.batch_size]
            B = len(batch_frames)

            imgs_np = np.stack([f["image"] for f in batch_frames])
            pcs_list = [f["pc"] for f in batch_frames]
            max_pts = max(p.shape[0] for p in pcs_list)
            pcs_padded = np.zeros((B, max_pts, 3), dtype=np.float32)
            masks = np.zeros((B, max_pts), dtype=np.float32)
            for i, pc in enumerate(pcs_list):
                n = pc.shape[0]
                pcs_padded[i, :n, :3] = pc[:, :3]
                masks[i, :n] = 1.0

            imgs_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float().to(device)
            pcs_t = torch.from_numpy(pcs_padded).float().to(device)

            init_T_t = torch.from_numpy(
                np.tile(T_biased.astype(np.float32), (B, 1, 1))).to(device)
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

        all_preds = np.concatenate(all_preds, axis=0)

        pred_corrections = []
        residuals = []
        for i in range(all_preds.shape[0]):
            pred_R = all_preds[i, :3, :3]
            corr = geodesic_rotation_deg(T_biased[:3, :3], pred_R)
            resid = geodesic_rotation_deg(T_orig[:3, :3], pred_R)
            pred_corrections.append(corr)
            residuals.append(resid)

        mean_corr = np.mean(pred_corrections)
        mean_resid = np.mean(residuals)
        corr_pct = (mean_corr / max(bias_mag, 0.001)) * 100 if bias_mag > 0.01 else 0

        if bias_mag < 0.01:
            status = "BASE"
        elif corr_pct > 60:
            status = "GOOD"
        elif corr_pct > 30:
            status = "PARTIAL"
        else:
            status = "FAIL"

        print(f"{scenario_name:<35} {bias_mag:>10.3f}° {mean_corr:>10.3f}° {mean_resid:>10.3f}° {corr_pct:>7.1f}% {status:>8}")

        rpy_orig = rotation_matrix_to_euler(T_orig[:3, :3])
        rpy_biased = rotation_matrix_to_euler(T_biased[:3, :3])
        mean_pred_R = R_sp.from_matrix(all_preds[:, :3, :3]).mean().as_matrix()
        rpy_pred = rotation_matrix_to_euler(mean_pred_R)

        results.append({
            "scenario": scenario_name,
            "bias": (r_bias, p_bias, y_bias),
            "bias_mag": bias_mag,
            "mean_correction": mean_corr,
            "mean_residual": mean_resid,
            "correction_pct": corr_pct,
            "status": status,
            "rpy_orig": rpy_orig,
            "rpy_biased": rpy_biased,
            "rpy_pred_mean": rpy_pred,
            "std_correction": np.std(pred_corrections),
        })

    print("=" * 80)

    good_count = sum(1 for r in results if r["status"] in ("GOOD", "BASE"))
    partial_count = sum(1 for r in results if r["status"] == "PARTIAL")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    print()
    print("Summary:")
    print(f"  GOOD/BASE: {good_count}/{len(results)}")
    print(f"  PARTIAL:   {partial_count}/{len(results)}")
    print(f"  FAIL:      {fail_count}/{len(results)}")

    biased_results = [r for r in results if r["bias_mag"] > 0.01]
    if biased_results:
        avg_corr_pct = np.mean([r["correction_pct"] for r in biased_results])
        avg_resid = np.mean([r["mean_residual"] for r in biased_results])
        print(f"  Avg correction %: {avg_corr_pct:.1f}%")
        print(f"  Avg residual: {avg_resid:.3f}°")

        if avg_corr_pct > 50:
            print("\n  VERDICT: Model has GENUINE calibration ability")
        elif avg_corr_pct > 20:
            print("\n  VERDICT: Model has PARTIAL calibration ability (needs improvement)")
        else:
            print("\n  VERDICT: Model has WEAK calibration ability (possible shortcut learning)")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        import json
        report = {
            "checkpoint": args.ckpt_path,
            "trip": os.path.basename(args.trip_dir),
            "n_frames": len(frames),
            "device": str(device),
            "results": [{
                "scenario": r["scenario"],
                "bias_rpy": r["bias"],
                "bias_mag_deg": r["bias_mag"],
                "mean_correction_deg": r["mean_correction"],
                "mean_residual_deg": r["mean_residual"],
                "correction_pct": r["correction_pct"],
                "status": r["status"],
                "rpy_orig": r["rpy_orig"].tolist(),
                "rpy_biased": r["rpy_biased"].tolist(),
                "rpy_pred_mean": r["rpy_pred_mean"].tolist(),
                "std_correction": r["std_correction"],
            } for r in results],
        }
        report_path = os.path.join(args.output_dir, "bias_correction_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {report_path}")

    import shutil
    shutil.rmtree(extract_root, ignore_errors=True)


if __name__ == "__main__":
    main()
