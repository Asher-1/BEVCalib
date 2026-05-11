#!/usr/bin/env python3
"""
Regenerate temporal projection comparison images (2x2 grid) from saved evaluation data.

Uses parsed predictions from extrinsics_and_errors.txt and saved temporal_aggregated_T.npy,
combined with the dataset loaded using the same configuration as the evaluation.
No GPU required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))

import numpy as np
import cv2
import re
import argparse
from custom_dataset import CustomDataset
from visualization import project_points_to_image, render_projected_points, compute_pose_errors
from tools import generate_single_perturbation_from_T


def parse_predicted_extrinsics(filepath):
    """Parse all predicted 4x4 extrinsic matrices from extrinsics_and_errors.txt."""
    predictions = {}
    gt_T_global = None

    with open(filepath) as f:
        content = f.read()

    gt_match = re.search(
        r'Ground Truth Extrinsics \(LiDAR.*?\):\n'
        r'([\s\S]*?)(?=\n={2,})',
        content
    )
    if gt_match:
        lines = [l.strip() for l in gt_match.group(1).strip().split('\n') if l.strip()]
        gt_T_global = np.array([[float(x) for x in l.split()] for l in lines[:4]])

    pattern = (
        r'Sample (\d+) \[Seq (\d+)\]\n-+\n\n'
        r'Predicted Extrinsics \(LiDAR.*?\):\n'
        r'([\s\S]*?)(?=\nRotation Errors)'
    )
    for m in re.finditer(pattern, content):
        idx = int(m.group(1))
        matrix_text = m.group(3).strip()
        lines = [l.strip() for l in matrix_text.split('\n') if l.strip()]
        T = np.array([[float(x) for x in l.split()] for l in lines[:4]])
        predictions[idx] = T

    return gt_T_global, predictions


def render_panel(image, pts, T_mat, K, h, w, errs, label, color,
                 max_pts=80000, point_radius=1):
    """Render a single projection panel with labels."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, lh = 0.45, 1, 18

    pts_2d, depths, _ = project_points_to_image(
        pts, T_mat, K, (h, w), min_depth=0.1, max_depth=200.0)
    rendered = render_projected_points(
        image, pts_2d, depths, color_mode='depth',
        point_radius=point_radius, max_depth=100.0)

    y = 18
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    cv2.rectangle(rendered, (3, y - th - 2), (7 + tw, y + 4), (0, 0, 0), -1)
    cv2.putText(rendered, label, (5, y), font, fs, color, ft)
    y += lh

    pts_txt = f"Pts: {len(pts_2d)}"
    (tw, th), _ = cv2.getTextSize(pts_txt, font, fs, ft)
    cv2.rectangle(rendered, (3, y - th - 2), (7 + tw, y + 4), (0, 0, 0), -1)
    cv2.putText(rendered, pts_txt, (5, y), font, fs, (255, 255, 255), ft)
    y += lh

    if errs is not None:
        rot_txt = f"Rot: {errs['rot_error']:.3f}deg"
        (tw, th), _ = cv2.getTextSize(rot_txt, font, fs, ft)
        cv2.rectangle(rendered, (3, y - th - 2), (7 + tw, y + 4), (0, 0, 0), -1)
        ec = ((0, 255, 0) if errs['rot_error'] < 0.5 else
              (0, 255, 255) if errs['rot_error'] < 1.0 else (0, 0, 255))
        cv2.putText(rendered, rot_txt, (5, y), font, fs, ec, ft)
        y += lh
        rpy = f"R:{errs['roll_error']:.3f} P:{errs['pitch_error']:.3f} Y:{errs['yaw_error']:.3f}"
        (tw, th), _ = cv2.getTextSize(rpy, font, fs, ft)
        cv2.rectangle(rendered, (3, y - th - 2), (7 + tw, y + 4), (0, 0, 0), -1)
        cv2.putText(rendered, rpy, (5, y), font, fs, (0, 255, 255), ft)

    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', required=True,
                        help='Evaluation directory containing extrinsics_and_errors.txt')
    parser.add_argument('--data_dir', required=True, help='Test dataset directory')
    parser.add_argument('--max_frames_per_seq', type=int, default=400)
    parser.add_argument('--vis_interval', type=int, default=200)
    parser.add_argument('--max_pts', type=int, default=80000)
    parser.add_argument('--point_radius', type=int, default=1)
    parser.add_argument('--angle_range', type=float, default=5.0)
    parser.add_argument('--trans_range', type=float, default=0.15)
    parser.add_argument('--target_width', type=int, default=640)
    parser.add_argument('--target_height', type=int, default=360)
    args = parser.parse_args()

    eval_dir = args.eval_dir
    extrinsics_file = os.path.join(eval_dir, 'extrinsics_and_errors.txt')
    agg_T_file = os.path.join(eval_dir, 'temporal_aggregated_T.npy')

    if not os.path.exists(extrinsics_file):
        print(f"ERROR: {extrinsics_file} not found")
        return
    if not os.path.exists(agg_T_file):
        print(f"ERROR: {agg_T_file} not found")
        return

    print("1. Parsing predicted extrinsics...")
    gt_T_global, predictions = parse_predicted_extrinsics(extrinsics_file)
    print(f"   Parsed {len(predictions)} predictions")

    print("2. Loading aggregated T matrices...")
    T_agg_all = np.load(agg_T_file)
    print(f"   Shape: {T_agg_all.shape}")

    print("3. Loading dataset...")
    dataset = CustomDataset(
        data_folder=args.data_dir,
        auto_detect=True,
        max_frames_per_seq=args.max_frames_per_seq,
    )
    print(f"   Dataset size: {len(dataset)} samples")

    if len(dataset) != len(predictions):
        print(f"   WARNING: Dataset size ({len(dataset)}) != predictions ({len(predictions)})")
        print(f"   This may cause index mismatch!")

    proj_dir = os.path.join(eval_dir, 'temporal_projections')
    os.makedirs(proj_dir, exist_ok=True)

    sample_indices = list(range(0, len(dataset), args.vis_interval))
    print(f"4. Generating {len(sample_indices)} temporal projection images...")

    font = cv2.FONT_HERSHEY_SIMPLEX

    for count, idx in enumerate(sample_indices):
        if idx not in predictions:
            print(f"   Skip sample {idx}: no prediction found")
            continue
        if idx >= len(T_agg_all):
            print(f"   Skip sample {idx}: out of T_agg range")
            continue

        result = dataset[idx]
        img_raw = result[0]
        pcd_raw = result[1]
        gt_T = np.array(result[2]).copy()
        K_orig = np.array(result[3]).copy()

        image = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        h_orig, w_orig = image.shape[:2]
        tw, th = args.target_width, args.target_height
        image = cv2.resize(image, (tw, th))
        scale_x = tw / w_orig
        scale_y = th / h_orig
        K = np.array([
            [K_orig[0, 0] * scale_x, 0, K_orig[0, 2] * scale_x],
            [0, K_orig[1, 1] * scale_y, K_orig[1, 2] * scale_y],
            [0, 0, 1]
        ])

        points = np.array(pcd_raw).copy()

        points = points[:, :3]
        valid = np.all(np.abs(points) < 999998, axis=1)
        points = points[valid]
        dist = np.linalg.norm(points[:, :3], axis=1)
        points = points[dist > 1.0]
        if len(points) > args.max_pts:
            rng = np.random.RandomState(42 + idx)
            points = points[rng.choice(len(points), args.max_pts, replace=False)]

        pred_T = predictions[idx]
        T_agg = T_agg_all[idx]

        np.random.seed(12345 + idx)
        init_T, _, _ = generate_single_perturbation_from_T(
            gt_T[np.newaxis],
            angle_range_deg=args.angle_range,
            trans_range=args.trans_range,
            rotation_only=True,
        )
        init_T = init_T[0]

        h, w = image.shape[:2]
        init_err = compute_pose_errors(init_T, gt_T)
        pred_err = compute_pose_errors(pred_T, gt_T)
        agg_err = compute_pose_errors(T_agg, gt_T)

        p_gt = render_panel(image, points, gt_T, K, h, w, None,
                            "GT (Ground Truth)", (0, 255, 0),
                            args.max_pts, args.point_radius)
        p_init = render_panel(image, points, init_T, K, h, w, init_err,
                              "Init (Perturbed)", (100, 200, 255),
                              args.max_pts, args.point_radius)
        p_pred = render_panel(image, points, pred_T, K, h, w, pred_err,
                              "Per-frame Pred", (100, 100, 255),
                              args.max_pts, args.point_radius)
        p_agg = render_panel(image, points, T_agg, K, h, w, agg_err,
                             "Aggregated (400f)", (0, 255, 200),
                             args.max_pts, args.point_radius)

        sep_v = np.full((h, 2, 3), 180, dtype=np.uint8)
        top_row = np.hstack([p_gt, sep_v, p_init])
        bot_row = np.hstack([p_pred, sep_v, p_agg])
        sep_full = np.full((2, top_row.shape[1], 3), 180, dtype=np.uint8)
        grid = np.vstack([top_row, sep_full, bot_row])

        bar_h = 36
        bar = np.full((bar_h, grid.shape[1], 3), 30, dtype=np.uint8)
        seq_id = idx // args.max_frames_per_seq
        if pred_err['rot_error'] > 1e-6:
            improve = (1 - agg_err['rot_error'] / pred_err['rot_error']) * 100
        else:
            improve = 0.0
        title = (f"Sample {idx:04d} [Seq {seq_id:02d}]  |  "
                 f"Init: {init_err['rot_error']:.2f}deg  ->  "
                 f"Per-frame: {pred_err['rot_error']:.3f}deg  ->  "
                 f"Aggregated: {agg_err['rot_error']:.3f}deg  "
                 f"({improve:+.1f}%)")
        cv2.putText(bar, title, (10, 24), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        combined = np.vstack([bar, grid])

        out_path = os.path.join(proj_dir, f"temporal_compare_{idx:04d}.png")
        cv2.imwrite(out_path, combined)

        if (count + 1) % 5 == 0 or count == 0:
            print(f"   [{count+1}/{len(sample_indices)}] Sample {idx:04d} "
                  f"Pred: {pred_err['rot_error']:.3f}° -> Agg: {agg_err['rot_error']:.3f}°")

    print(f"\nDone! {len(sample_indices)} images saved to {proj_dir}/")


if __name__ == '__main__':
    main()
