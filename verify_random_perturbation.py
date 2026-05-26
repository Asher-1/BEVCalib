#!/usr/bin/env python3
"""Quick verification: random perturbation at inference vs fixed T_init.

Tests three conditions:
  A) Fixed WRONG T_init (current run_bag_calibration behavior)
  B) Random perturbation around GT (KITTI eval behavior)  
  C) Random perturbation around WRONG T_init (proposed fix)

Uses KITTI test_data_v2 to compare, since GT is known.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kitti-bev-calib"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from collections import deque

CKPT = "logs/all_training_data/v30/model_small_5deg_v30_G3_dinov2small_dann/all_training_data_scratch/checkpoint/ckpt_400.pth"
DATASET = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"
SEQ = "00"
N_FRAMES = 200
ANGLE_RANGE = 5.0
WRONG_OFFSET_DEG = 1.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_H, IMG_W = 360, 640


def load_calib(seq_dir):
    calib_path = os.path.join(seq_dir, "calib.txt")
    with open(calib_path) as f:
        for line in f:
            if line.startswith("Tr:") or line.startswith("Tr_velo_to_cam:"):
                vals = [float(x) for x in line.split(":")[1].strip().split()]
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                return T
    raise ValueError("No Tr found in " + calib_path)


def load_K(seq_dir):
    calib_path = os.path.join(seq_dir, "calib.txt")
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P2:"):
                vals = [float(x) for x in line.split(":")[1].strip().split()]
                P = np.array(vals).reshape(3, 4)
                K = P[:3, :3].copy()
                return K
    raise ValueError("No P2 found")


def perturb_T(T_gt, angle_range_deg):
    from tools import generate_single_perturbation_from_T
    T_np = T_gt[np.newaxis, :, :].astype(np.float32)
    init_T, _, _ = generate_single_perturbation_from_T(
        T_np, angle_range_deg=angle_range_deg, trans_range=0.0,
        rotation_only=True, distribution="truncated_normal")
    return init_T[0]


def make_wrong_T(T_gt, offset_deg):
    rv = np.array([offset_deg * np.pi / 180, 0.0, 0.0])
    dR = R.from_rotvec(rv).as_matrix()
    T_wrong = T_gt.copy()
    T_wrong[:3, :3] = dR @ T_gt[:3, :3]
    return T_wrong


def rotation_error_deg(A, B):
    R_diff = A[:3, :3] @ np.linalg.inv(B[:3, :3])
    r = R.from_matrix(R_diff)
    return np.linalg.norm(r.as_rotvec()) * 180 / np.pi


def aggregate_median(predictions):
    if not predictions:
        return None
    Rs = [p[:3, :3] for p in predictions]
    rvecs = [R.from_matrix(r).as_rotvec() for r in Rs]
    med = np.median(rvecs, axis=0)
    R_med = R.from_rotvec(med).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_med
    T[:3, 3] = np.median([p[:3, 3] for p in predictions], axis=0)
    return T


def main():
    import cv2
    from utils.bevcalib_inference import load_bevcalib_inference

    seq_dir = os.path.join(DATASET, "sequences", SEQ)
    T_gt = load_calib(seq_dir)
    K_full = load_K(seq_dir)
    T_wrong = make_wrong_T(T_gt, WRONG_OFFSET_DEG)

    print("=" * 70)
    print("GT extrinsic vs Wrong extrinsic:")
    print("  Offset: {:.4f} deg".format(rotation_error_deg(T_wrong, T_gt)))
    print("=" * 70)

    wrapper, epoch = load_bevcalib_inference(CKPT, device=DEVICE, img_shape=(IMG_H, IMG_W))
    print("Loaded model epoch={} device={}".format(epoch, DEVICE))

    img_dir = os.path.join(seq_dir, "image_2")
    pc_dir = os.path.join(seq_dir, "velodyne")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])[:N_FRAMES]

    from run_bag_calibration import collate_and_pad_batch

    preds_A, preds_B, preds_C = [], [], []

    for i, img_f in enumerate(img_files):
        base = os.path.splitext(img_f)[0]
        img = cv2.imread(os.path.join(img_dir, img_f))
        img = cv2.resize(img, (IMG_W, IMG_H))
        raw_pc = np.fromfile(os.path.join(pc_dir, base + ".bin"), dtype=np.float32)
        if raw_pc.size % 4 == 0:
            pc = raw_pc.reshape(-1, 4)[:, :3]
        else:
            continue

        K_scaled = K_full.copy()
        orig_h, orig_w = 376, 1241
        K_scaled[0, :] *= IMG_W / orig_w
        K_scaled[1, :] *= IMG_H / orig_h

        T_init_A = T_wrong.copy()
        T_init_B = perturb_T(T_gt, ANGLE_RANGE)
        T_init_C = perturb_T(T_wrong, ANGLE_RANGE)

        with torch.no_grad():
            for T_init, preds_list in [(T_init_A, preds_A), (T_init_B, preds_B), (T_init_C, preds_C)]:
                batch = collate_and_pad_batch([img], [pc], T_init, [K_scaled], DEVICE, torch)
                if batch is None:
                    continue
                imgs_t, pcs_t, init_T_t, post_T_t, K_t = batch
                if DEVICE.startswith("cuda"):
                    with torch.cuda.amp.autocast():
                        pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
                else:
                    pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
                preds_list.append(pred[0].cpu().numpy())

        if (i + 1) % 50 == 0:
            print("  Processed {}/{} frames".format(i + 1, len(img_files)))

    for label, preds in [("A: Fixed WRONG T_init", preds_A),
                          ("B: Random perturb GT", preds_B),
                          ("C: Random perturb WRONG", preds_C)]:
        if not preds:
            continue
        for n_agg in [50, 100, 200]:
            if len(preds) < n_agg:
                continue
            T_agg = aggregate_median(preds[:n_agg])
            err = rotation_error_deg(T_agg, T_gt)
            err_vs_wrong = rotation_error_deg(T_agg, T_wrong)
            print("{} | {:>3d} frames | Error vs GT: {:.4f} deg | vs Wrong: {:.4f} deg".format(
                label, n_agg, err, err_vs_wrong))

    print("\n=== Summary ===")
    print("A = current run_bag_calibration.py (all frames use same wrong T_init)")
    print("B = KITTI eval style (each frame perturbed around GT)")
    print("C = proposed fix (each frame perturbed around wrong T_init)")
    print("If C << A, then random perturbation helps even without knowing GT")


if __name__ == "__main__":
    main()
