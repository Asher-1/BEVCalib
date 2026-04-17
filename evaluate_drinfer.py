#!/usr/bin/env python3
"""
Evaluate a DrInfer-exported BEVCalib model on a test dataset.

Produces extrinsics_and_errors.txt in the same format as evaluate_checkpoint.py,
so results can be consumed by run_generalization_eval.py.

Usage:
    python evaluate_drinfer.py \
        --ckpt_path  logs/.../checkpoint/best.pth \
        --export_dir logs/.../drinfer \
        --dataset_root /path/to/test_data \
        --output_dir   results/drinfer_eval

For batch evaluation, run_generalization_eval.py calls this script as a subprocess.
"""

import os
import sys
import argparse
import time
import json
import re
import glob
import subprocess
import tempfile

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.abspath(__file__))
_KITTI_DIR = os.path.join(_ROOT, 'kitti-bev-calib')
_UTILS_DIR = os.path.join(_ROOT, 'utils')
for _d in [_KITTI_DIR, _UTILS_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


def _find_bin(export_dir, model_name="bevcalib_fusion_head"):
    """Check if a DrInfer .bin file already exists under export_dir."""
    if not os.path.isdir(export_dir):
        return None
    for pattern in [
        os.path.join(export_dir, "*.bin"),
        os.path.join(export_dir, model_name, "engine_graph", "*.bin"),
    ]:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return None


def _auto_export_if_needed(args, export_dir, rotation_only):
    """If no .bin exists in export_dir, run torch2drinfer.py automatically."""
    if _find_bin(export_dir, args.model_name):
        return
    print(f"\n[auto-export] No DrInfer model found in {export_dir}")
    print(f"[auto-export] Running torch2drinfer.py to export the model ...\n")

    cfg = {
        "ckpt_path": args.ckpt_path,
        "export_dir": export_dir,
        "img_height": args.target_height,
        "img_width": args.target_width,
        "max_num_points": args.max_num_points,
        "rotation_only": rotation_only,
        "bev_zbound_step": getattr(args, "_bev_zbound_step", "2.0"),
        "voxel_mode": args.voxel_mode or "scatter",
        "scatter_reduce": args.scatter_reduce or "sum",
        "to_bev_mode": args.to_bev_mode or "concat",
        "model_name": args.model_name,
        "model_version": args.model_version,
        "verify_export_graph": True,
        "export_strategy": "full",
    }

    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="drinfer_auto_", delete=False)
    try:
        yaml.safe_dump(cfg, tmp_cfg)
        tmp_cfg.close()

        convert_script = os.path.join(_UTILS_DIR, "torch2drinfer.py")
        cmd = [sys.executable, convert_script,
               "--config", tmp_cfg.name, "--layout", "flat"]
        print(f"[auto-export] {' '.join(cmd)}")
        env = {**os.environ}
        env["BEV_ZBOUND_STEP"] = cfg["bev_zbound_step"]
        if args.use_drcv:
            env["USE_DRCV_BACKEND"] = "1"
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            raise RuntimeError(
                f"Auto DrInfer export failed (exit code {ret.returncode})")

        if not _find_bin(export_dir, args.model_name):
            raise RuntimeError(
                f"Export completed but no .bin found in {export_dir}")
        print(f"\n[auto-export] Export succeeded.\n")
    finally:
        os.unlink(tmp_cfg.name)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DrInfer BEVCalib model (compatible with evaluate_checkpoint.py output)")
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="PyTorch checkpoint (.pth) used for export")
    p.add_argument("--export_dir", type=str, default=None,
                   help="DrInfer export directory (default: <ckpt_dir>/drinfer)")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--angle_range_deg", type=float, default=5.0)
    p.add_argument("--trans_range", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--rotation_only", type=int, default=1)
    p.add_argument("--vis_interval", type=int, default=200)
    p.add_argument("--vis_points", type=int, default=8000)
    p.add_argument("--vis_point_radius", type=int, default=2)
    p.add_argument("--use_full_dataset", action="store_true")
    p.add_argument("--use_drcv", action="store_true")
    p.add_argument("--voxel_mode", type=str, default=None)
    p.add_argument("--scatter_reduce", type=str, default=None)
    p.add_argument("--to_bev_mode", type=str, default=None)
    p.add_argument("--use_mlp_head", type=int, default=-1)
    p.add_argument("--bev_pool_factor", type=int, default=0)
    p.add_argument("--target_width", type=int, default=640)
    p.add_argument("--target_height", type=int, default=360)
    p.add_argument("--max_num_points", type=int, default=200000)
    p.add_argument("--model_name", type=str, default="bevcalib_fusion_head")
    p.add_argument("--model_version", type=str, default="v2")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--compare_pytorch", action="store_true",
                   help="Also run PyTorch inference and append comparison stats")
    return p.parse_args()


def _resolve_from_checkpoint(args):
    """Auto-detect settings from checkpoint."""
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    epoch = ckpt.get("epoch", -1)
    ckpt_args = ckpt.get("args", None)

    def _get(key, default=None):
        if ckpt_args is None:
            return default
        if isinstance(ckpt_args, dict):
            return ckpt_args.get(key, default)
        return getattr(ckpt_args, key, default)

    if ckpt_args is not None:
        if args.voxel_mode is None:
            args.voxel_mode = _get("voxel_mode", "scatter")
        if args.scatter_reduce is None:
            args.scatter_reduce = _get("scatter_reduce", "sum")
        if args.to_bev_mode is None:
            args.to_bev_mode = _get("to_bev_mode", "concat")
        if args.use_mlp_head == -1:
            args.use_mlp_head = 1 if _get("use_mlp_head", 0) else 0
        rot = _get("rotation_only", None)
        if rot is not None and args.rotation_only == 1:
            args.rotation_only = 1 if rot else 0

    bz = _get("bev_zbound_step", None)
    if bz is None:
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
        proj_key = "img_branch.proj_head.projection.weight"
        if proj_key in state:
            feat_dim = state[proj_key].shape[0]
            total_dim = state[proj_key].shape[1]
            z_voxels = total_dim // feat_dim
            bz = str(round(20.0 / z_voxels, 1))
            print(f"  [auto-detect] bev_zbound_step={bz} "
                  f"(z_voxels={z_voxels} from {proj_key} shape {list(state[proj_key].shape)})")
    args._bev_zbound_step = str(bz) if bz else "4.0"

    if args.voxel_mode is None:
        args.voxel_mode = "scatter"
    if args.scatter_reduce is None:
        args.scatter_reduce = "sum"
    if args.to_bev_mode is None:
        args.to_bev_mode = "concat"
    if args.use_mlp_head == -1:
        args.use_mlp_head = 0

    return epoch


def build_dataloader(args):
    from torch.utils.data import DataLoader, random_split
    from custom_dataset import CustomDataset
    from evaluate_checkpoint import make_collate_fn

    dataset = CustomDataset(data_folder=args.dataset_root, auto_detect=True)
    target_size = (args.target_width, args.target_height)
    collate_fn = make_collate_fn(target_size)

    if args.use_full_dataset:
        val_dataset = dataset
    else:
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        _, val_dataset = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(114514),
        )

    seq_boundaries = []
    try:
        files = dataset.all_files if hasattr(dataset, 'all_files') else []
        indices = range(len(val_dataset))
        if hasattr(val_dataset, 'indices'):
            indices = val_dataset.indices
        cur_seq, start_idx = None, 0
        for i, idx in enumerate(indices):
            if idx < len(files):
                seq = files[idx].split(os.sep)[0] if os.sep in files[idx] else "00"
            else:
                seq = "00"
            if cur_seq is None:
                cur_seq = seq
            if seq != cur_seq:
                seq_boundaries.append((cur_seq, start_idx, i - 1))
                cur_seq, start_idx = seq, i
        if cur_seq is not None:
            seq_boundaries.append((cur_seq, start_idx, len(list(indices)) - 1))
    except Exception:
        pass

    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader, seq_boundaries


def _write_stats_block(f, all_errors, sample_count, rotation_only, seq_boundaries,
                       sample_sequences):
    """Write the EVALUATION STATISTICS block (same format as evaluate_checkpoint.py)."""
    avg_errors = {k: np.mean(v) for k, v in all_errors.items()}
    std_errors = {k: np.std(v) for k, v in all_errors.items()}
    max_errors = {k: np.max(v) for k, v in all_errors.items()}
    min_errors = {k: np.min(v) for k, v in all_errors.items()}
    med_errors = {k: np.median(v) for k, v in all_errors.items()}
    p90_errors = {k: np.percentile(v, 90) for k, v in all_errors.items()}
    p95_errors = {k: np.percentile(v, 95) for k, v in all_errors.items()}
    p99_errors = {k: np.percentile(v, 99) for k, v in all_errors.items()}

    def _write_metric_block(f, label, keys, unit):
        header = (f"{'Metric':<14} {'Mean':>10} {'Std':>10} {'Min':>10} "
                  f"{'Median':>10} {'P90':>10} {'P95':>10} {'P99':>10} {'Max':>10}")
        f.write(f"{label} ({unit}):\n")
        f.write(f"  {header}\n")
        f.write(f"  {'-' * len(header)}\n")
        name_map = {
            'trans_error': 'Total', 'fwd_error': 'X (Fwd)',
            'lat_error': 'Y (Lat)', 'ht_error': 'Z (Ht)',
            'rot_error': 'Total', 'roll_error': 'Roll (LiDAR-X)',
            'pitch_error': 'Pitch (LiDAR-Y)', 'yaw_error': 'Yaw (LiDAR-Z)',
        }
        for k in keys:
            name = name_map.get(k, k)
            f.write(f"  {name:<14} {avg_errors[k]:>10.6f} {std_errors[k]:>10.6f} "
                    f"{min_errors[k]:>10.6f} {med_errors[k]:>10.6f} "
                    f"{p90_errors[k]:>10.6f} {p95_errors[k]:>10.6f} "
                    f"{p99_errors[k]:>10.6f} {max_errors[k]:>10.6f}\n")
        f.write("\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("EVALUATION STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total samples evaluated: {sample_count}\n\n")

    trans_keys = ['trans_error', 'fwd_error', 'lat_error', 'ht_error']
    rot_keys = ['rot_error', 'roll_error', 'pitch_error', 'yaw_error']
    if not rotation_only:
        _write_metric_block(f, "Translation Errors", trans_keys, "m")
    _write_metric_block(f, "Rotation Errors", rot_keys, "deg")

    f.write("=" * 80 + "\n")
    f.write("AVERAGE ERRORS ACROSS ALL SAMPLES\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total samples evaluated: {sample_count}\n\n")

    if not rotation_only:
        f.write("Average Translation Errors (in LiDAR coordinate system):\n")
        f.write(f"  Total:   {avg_errors['trans_error']:.6f} ± {std_errors['trans_error']:.6f} m\n")
        f.write(f"  X (Fwd): {avg_errors['fwd_error']:.6f} ± {std_errors['fwd_error']:.6f} m\n")
        f.write(f"  Y (Lat): {avg_errors['lat_error']:.6f} ± {std_errors['lat_error']:.6f} m\n")
        f.write(f"  Z (Ht):  {avg_errors['ht_error']:.6f} ± {std_errors['ht_error']:.6f} m\n")

    f.write("\nAverage Rotation Errors (axis-angle, LiDAR frame: X=Fwd, Y=Left, Z=Up):\n")
    f.write(f"  Total:            {avg_errors['rot_error']:.6f} ± {std_errors['rot_error']:.6f} deg\n")
    f.write(f"  Roll  (LiDAR X):  {avg_errors['roll_error']:.6f} ± {std_errors['roll_error']:.6f} deg\n")
    f.write(f"  Pitch (LiDAR Y):  {avg_errors['pitch_error']:.6f} ± {std_errors['pitch_error']:.6f} deg\n")
    f.write(f"  Yaw   (LiDAR Z):  {avg_errors['yaw_error']:.6f} ± {std_errors['yaw_error']:.6f} deg\n")

    if seq_boundaries and sample_sequences:
        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-SEQUENCE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        rot_arr = np.array(all_errors['rot_error'])
        seq_arr = np.array(sample_sequences)
        for seq_id, s_start, s_end in seq_boundaries:
            mask = seq_arr == seq_id
            if not mask.any():
                continue
            seq_rot = rot_arr[mask]
            f.write(f"  Sequence {seq_id} ({mask.sum()} samples):\n")
            f.write(f"    Rotation: mean={np.mean(seq_rot):.4f}° "
                    f"median={np.median(seq_rot):.4f}° "
                    f"p95={np.percentile(seq_rot, 95):.4f}° "
                    f"max={np.max(seq_rot):.4f}°\n\n")


def main():
    args = parse_args()

    if args.use_drcv:
        os.environ["USE_DRCV_BACKEND"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = _resolve_from_checkpoint(args)
    rotation_only = args.rotation_only > 0

    export_dir = args.export_dir
    if export_dir is None:
        export_dir = os.path.join(os.path.dirname(args.ckpt_path), "drinfer")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.ckpt_path), "drinfer_eval")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    print("=" * 80)
    print("BEVCalib DrInfer Model Evaluation")
    print(f"  checkpoint : {args.ckpt_path}")
    print(f"  export_dir : {export_dir}")
    print(f"  dataset    : {args.dataset_root}")
    print(f"  output_dir : {output_dir}")
    print(f"  angle/trans: {args.angle_range_deg}° / {args.trans_range}m")
    print(f"  voxel_mode : {args.voxel_mode}")
    print(f"  to_bev_mode: {args.to_bev_mode}")
    print(f"  scatter_red: {args.scatter_reduce}")
    print(f"  bev_zbound  : {args._bev_zbound_step}")
    print(f"  rotation_only: {rotation_only}")
    print("=" * 80)

    os.environ["BEV_ZBOUND_STEP"] = args._bev_zbound_step
    _auto_export_if_needed(args, export_dir, rotation_only)

    from bevcalib_inference import load_bevcalib_inference
    wrapper, _ = load_bevcalib_inference(
        ckpt_path=args.ckpt_path,
        device=device,
        img_shape=(args.target_height, args.target_width),
        rotation_only=rotation_only,
        use_mlp_head=bool(args.use_mlp_head),
        voxel_mode=args.voxel_mode,
        to_bev_mode=args.to_bev_mode,
        scatter_reduce=args.scatter_reduce,
    )

    from drinfer_infer import DrInferBackend
    cfg = {
        "img_height": args.target_height,
        "img_width": args.target_width,
        "max_num_points": args.max_num_points,
        "model_name": args.model_name,
        "model_version": args.model_version,
        "rotation_only": rotation_only,
    }
    dr_backend = DrInferBackend(wrapper, export_dir, cfg, device=device)

    pt_backend = None
    if args.compare_pytorch:
        from drinfer_infer import PyTorchInferenceBackend
        pt_backend = PyTorchInferenceBackend(wrapper, device=device)

    val_loader, seq_boundaries = build_dataloader(args)

    from tools import generate_single_perturbation_from_T
    from visualization import compute_pose_errors

    extrinsics_file = os.path.join(output_dir, "extrinsics_and_errors.txt")
    all_errors = {
        'trans_error': [], 'fwd_error': [], 'lat_error': [], 'ht_error': [],
        'rot_error': [], 'roll_error': [], 'pitch_error': [], 'yaw_error': []
    }
    sample_sequences = []
    timings_dr = []
    timings_pt = []
    per_sample_deltas = []
    gt_extrinsics_written = False
    sample_count = 0
    max_batches = args.max_batches if args.max_batches > 0 else len(val_loader)

    def _seq_for_sample(idx):
        for seq_id, s, e in seq_boundaries:
            if s <= idx <= e:
                return seq_id
        return "??"

    print(f"\nEvaluating DrInfer model ({max_batches} batches) ...\n")

    with torch.no_grad():
        for batch_idx, (imgs, pcs, masks, gt_T_list, intrinsics_list) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            gt_T_np = np.array(gt_T_list).astype(np.float32)
            init_T_np, _, _ = generate_single_perturbation_from_T(
                gt_T_np,
                angle_range_deg=args.angle_range_deg,
                trans_range=args.trans_range if not rotation_only else 0.0,
            )

            imgs_np = np.array(imgs)
            pcs_np = np.array(pcs)[:, :, :3]
            masks_np = np.array(masks)
            K_np = np.array(intrinsics_list).astype(np.float32)

            imgs_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float().to(device)
            pcs_t = torch.from_numpy(pcs_np).float().to(device)
            gt_T_t = torch.from_numpy(gt_T_np).float().to(device)
            init_T_t = torch.from_numpy(init_T_np).float().to(device)
            K_t = torch.from_numpy(K_np).float().to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_T_dr = dr_backend.predict(imgs_t, pcs_t, masks_np, init_T_t, gt_T_t, K_t)
            torch.cuda.synchronize()
            timings_dr.append((time.perf_counter() - t0) * 1000.0)

            pred_T_dr_np = pred_T_dr.detach().cpu().numpy()

            pred_T_pt_np = None
            if pt_backend is not None:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred_T_pt = pt_backend.predict(imgs_t, pcs_t, masks_np, init_T_t, gt_T_t, K_t)
                torch.cuda.synchronize()
                timings_pt.append((time.perf_counter() - t0) * 1000.0)
                pred_T_pt_np = pred_T_pt.detach().cpu().numpy()

            B = imgs_np.shape[0]
            for i in range(B):
                sample_idx = sample_count + i
                seq_id = _seq_for_sample(sample_idx)
                sample_sequences.append(seq_id)

                errors = compute_pose_errors(pred_T_dr_np[i], gt_T_np[i])
                for key in all_errors:
                    all_errors[key].append(errors[key])

                if pred_T_pt_np is not None:
                    err_pt = compute_pose_errors(pred_T_pt_np[i], gt_T_np[i])
                    per_sample_deltas.append({
                        "sample": sample_idx,
                        "dr_rot": errors["rot_error"],
                        "pt_rot": err_pt["rot_error"],
                        "delta_rot": errors["rot_error"] - err_pt["rot_error"],
                        "T_max_diff": float(np.abs(pred_T_dr_np[i] - pred_T_pt_np[i]).max()),
                    })

                with open(extrinsics_file, 'a') as f:
                    if not gt_extrinsics_written:
                        mode = "全量数据集泛化测试 (DrInfer)" if args.use_full_dataset else "验证集评估 (DrInfer)"
                        f.write(f"Checkpoint: {os.path.basename(args.ckpt_path)}\n")
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Backend: DrInfer ({args.model_name}_{args.model_version})\n")
                        f.write(f"Dataset: {args.dataset_root}\n")
                        f.write(f"Mode: {mode}\n")
                        f.write(f"Perturbation: {args.angle_range_deg}deg, {args.trans_range}m\n")
                        if seq_boundaries:
                            f.write(f"\nSequence Boundaries:\n")
                            for sb_seq, sb_s, sb_e in seq_boundaries:
                                f.write(f"  Seq {sb_seq}: samples {sb_s} - {sb_e} ({sb_e - sb_s + 1} frames)\n")
                        f.write("=" * 80 + "\n\n")
                        f.write("Ground Truth Extrinsics (LiDAR → Camera):\n")
                        for row in gt_T_np[i]:
                            f.write(f"  {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
                        f.write("\n" + "=" * 80 + "\n\n")
                        gt_extrinsics_written = True

                    f.write(f"Sample {sample_idx:04d} [Seq {seq_id}]\n")
                    f.write("-" * 80 + "\n")
                    f.write("\nPredicted Extrinsics (LiDAR → Camera):\n")
                    for row in pred_T_dr_np[i]:
                        f.write(f"  {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")

                    if not rotation_only:
                        f.write("\nTranslation Errors (in LiDAR coordinate system):\n")
                        f.write(f"  Total:   {errors['trans_error']:.6f} m\n")
                        f.write(f"  X (Fwd): {errors['fwd_error']:.6f} m\n")
                        f.write(f"  Y (Lat): {errors['lat_error']:.6f} m\n")
                        f.write(f"  Z (Ht):  {errors['ht_error']:.6f} m\n")

                    f.write("\nRotation Errors (axis-angle, LiDAR frame: X=Fwd, Y=Left, Z=Up):\n")
                    f.write(f"  Total:            {errors['rot_error']:.6f} deg\n")
                    f.write(f"  Roll  (LiDAR X):  {errors['roll_error']:.6f} deg\n")
                    f.write(f"  Pitch (LiDAR Y):  {errors['pitch_error']:.6f} deg\n")
                    f.write(f"  Yaw   (LiDAR Z):  {errors['yaw_error']:.6f} deg\n")
                    f.write("\n" + "=" * 80 + "\n\n")

                if sample_idx % 50 == 0:
                    print(f"   Sample {sample_idx} [Seq {seq_id}]: "
                          f"rot={errors['rot_error']:.2f}° "
                          f"batch_time={timings_dr[-1]:.1f}ms")

            sample_count += B

    with open(extrinsics_file, 'a') as f:
        _write_stats_block(f, all_errors, sample_count, rotation_only,
                           seq_boundaries, sample_sequences)

        f.write("\n" + "=" * 80 + "\n")
        f.write("DRINFER PERFORMANCE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        t_arr = np.array(timings_dr)
        f.write(f"Total batches: {len(timings_dr)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Latency (ms/batch): mean={t_arr.mean():.1f} "
                f"median={np.median(t_arr):.1f} "
                f"p95={np.percentile(t_arr, 95):.1f} "
                f"min={t_arr.min():.1f} max={t_arr.max():.1f}\n")
        per_sample_ms = t_arr.mean() / max(args.batch_size, 1)
        f.write(f"Latency (ms/sample): ~{per_sample_ms:.1f}\n")

        if per_sample_deltas:
            f.write("\n" + "=" * 80 + "\n")
            f.write("PYTORCH vs DRINFER COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            deltas = np.array([d["delta_rot"] for d in per_sample_deltas])
            t_diffs = np.array([d["T_max_diff"] for d in per_sample_deltas])
            f.write(f"Rotation error delta (DrInfer - PyTorch):\n")
            f.write(f"  mean={np.mean(deltas):.6f}° "
                    f"std={np.std(deltas):.6f}° "
                    f"max_abs={np.max(np.abs(deltas)):.6f}°\n")
            f.write(f"Transform matrix max absolute diff:\n")
            f.write(f"  mean={np.mean(t_diffs):.8f} "
                    f"max={np.max(t_diffs):.8f}\n")
            if timings_pt:
                pt_arr = np.array(timings_pt)
                speedup = pt_arr.mean() / max(t_arr.mean(), 1e-3)
                f.write(f"\nLatency comparison:\n")
                f.write(f"  PyTorch: {pt_arr.mean():.1f}ms/batch\n")
                f.write(f"  DrInfer: {t_arr.mean():.1f}ms/batch\n")
                f.write(f"  Speedup: {speedup:.2f}x\n")

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete: {sample_count} samples")
    rot_mean = np.mean(all_errors['rot_error'])
    rot_p95 = np.percentile(all_errors['rot_error'], 95)
    print(f"  Rotation error: mean={rot_mean:.4f}°  p95={rot_p95:.4f}°")
    print(f"  Latency: {np.mean(timings_dr):.1f}ms/batch ({per_sample_ms:.1f}ms/sample)")
    print(f"  Results: {extrinsics_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
