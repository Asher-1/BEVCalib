#!/usr/bin/env python3
"""
BEVCalib DrInfer / PyTorch inference benchmark and evaluation.

Supports two backends:
    1. pytorch  – runs BEVCalib.forward() directly (same path as training
                  and evaluate_checkpoint.py)
    2. drinfer  – loads exported fusion+head model from DrInfer engine

Usage:
    python utils/drinfer_infer.py --config utils/drinfer_config.yaml
    python utils/drinfer_infer.py --config utils/drinfer_config.yaml --backend pytorch
"""

import os
import sys
import argparse
import time
import json

import numpy as np
import torch
import yaml
import cv2

_KITTI_DIR = os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib')
if _KITTI_DIR not in sys.path:
    sys.path.insert(0, _KITTI_DIR)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def build_val_loader(cfg):
    """Create validation dataloader matching training split."""
    from torch.utils.data import DataLoader, random_split
    from custom_dataset import CustomDataset
    from train_kitti import PreprocessedDataset, collate_fn

    dataset = CustomDataset(
        data_folder=cfg["dataset_root"], suf=".png"
    )
    val_ratio = cfg.get("validate_sample_ratio", 0.2)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    _, val_raw = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(114514),
    )

    target_w = cfg.get("img_width", 640)
    target_h = cfg.get("img_height", 360)
    val_ds = PreprocessedDataset(val_raw, target_size=(target_w, target_h))

    return DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# PyTorch backend – uses BEVCalib.forward() (same path as training / evaluate_checkpoint.py)
# ---------------------------------------------------------------------------
class PyTorchBackend:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, imgs, pcs, masks_np, init_T, gt_T, intrinsics):
        B = imgs.shape[0]
        post_T = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1).contiguous()
        with torch.cuda.amp.autocast():
            T_pred, _, _ = self.model(
                imgs, pcs, gt_T, init_T, post_T, intrinsics,
                masks=masks_np, out_init_loss=False,
            )
        return T_pred


# ---------------------------------------------------------------------------
# DrInfer backend (full model via InferenceEngine with fixed shapes)
# ---------------------------------------------------------------------------
class DrInferBackend:
    """
    Full-model DrInfer inference. The entire BEVCalibInference graph
    (image branch, bev_pool, spconv PC branch, fusion, transformer, head)
    runs inside the DrInfer engine.
    """

    def __init__(self, wrapper, drinfer_model_dir, cfg, device="cuda"):
        import drinfer as dr
        from frontend_python.pytorch_parser.inference_utils import (
            InferenceEngine, EngineInOutConfig,
        )

        self.device = device
        self.rotation_only = cfg.get("rotation_only", True)
        H = cfg["img_height"]
        W = cfg["img_width"]
        N = cfg.get("max_num_points", 200000)

        model_name = cfg.get("model_name", "bevcalib_fusion_head")
        model_version = cfg.get("model_version", "v1")
        full_name = f"{model_name}_{model_version}"

        bin_path = os.path.abspath(os.path.join(drinfer_model_dir, f"{full_name}.bin"))
        if not os.path.isfile(bin_path):
            bin_path = os.path.abspath(os.path.join(drinfer_model_dir, f"{model_name}.bin"))
            if not os.path.isfile(bin_path):
                raise FileNotFoundError(
                    f"DrInfer model not found in {drinfer_model_dir}\n"
                    f"Run torch2drinfer.py first to export the model."
                )
        print(f"  Loading DrInfer graph: {bin_path}")

        input_names = ["image", "point_cloud", "init_T", "post_cam2ego_T", "intrinsic"]
        max_input_shapes = {
            "image": [1, 3, H, W],
            "point_cloud": [1, N, 3],
            "init_T": [1, 4, 4],
            "post_cam2ego_T": [1, 4, 4],
            "intrinsic": [1, 3, 3],
        }

        engine_config = EngineInOutConfig(
            name=model_name,
            max_input_shapes=max_input_shapes,
            output_tensors=[("pred_T", 0)],
            fixed_shape_configs={n: True for n in input_names},
        )

        optimization_flags = (
            dr.OptimizationFlag.OPTIMIZATION_CONV_RELU_MERGE
            | dr.OptimizationFlag.OPTIMIZATION_CONV_BN_MERGE
            | dr.OptimizationFlag.OPTIMIZATION_ARITHMATIC_MERGE
            | dr.OptimizationFlag.OPTIMIZATION_MERGE_CONCAT
            | dr.OptimizationFlag.OPTIMIZATION_MEMORY_REUSE
            | dr.OptimizationFlag.OPTIMIZATION_AUTOTUNE
            | dr.OptimizationFlag.OPTIMIZATION_TENSORFORMAT_TRANSFORM
            | dr.OptimizationFlag.OPTIMIZATION_CUDA_GRAPH
            | dr.OptimizationFlag.OPTIMIZATION_MULTIPLE_STREAM_SINGLE_BATCH
            | dr.OptimizationFlag.OPTIMIZATION_MEM_INTENSIVE
            | dr.OptimizationFlag.OPTIMIZATION_FC_BN_MERGE
            | dr.OptimizationFlag.OPTIMIZATION_FC_RELU_MERGE
        )

        inf_engine = InferenceEngine(
            engine_graph_path=bin_path,
            engine_in_out_config=engine_config,
            runtime_data_type=dr.MODEL_DATA_TYPE.MODEL_FLOAT,
            optimization_flags=optimization_flags,
            engine_math_mode=dr.EngineMathMode.Default,
        )
        inf_engine.build()

        self._inf_engine = inf_engine
        self.dr = dr
        self.H = H
        self.W = W
        self.N = N

    def predict(self, imgs, pcs, masks_np, init_T, gt_T, intrinsics):
        B = imgs.shape[0]
        engine = self._inf_engine._engine
        post_T = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1).contiguous()

        if pcs.shape[1] < self.N:
            pad = torch.zeros(B, self.N - pcs.shape[1], 3, device=pcs.device, dtype=pcs.dtype)
            pcs = torch.cat([pcs, pad], dim=1)
        elif pcs.shape[1] > self.N:
            pcs = pcs[:, :self.N, :]

        img_np = np.ascontiguousarray(imgs.cpu().numpy().astype(np.float32))
        pc_np = np.ascontiguousarray(pcs.cpu().numpy().astype(np.float32))
        init_np = np.ascontiguousarray(init_T.cpu().numpy().astype(np.float32))
        post_np = np.ascontiguousarray(post_T.cpu().numpy().astype(np.float32))
        K_np = np.ascontiguousarray(intrinsics.cpu().numpy().astype(np.float32))

        engine.register_infer_input_shape("image", list(img_np.shape))
        engine.register_infer_input_data("image", img_np)
        engine.register_infer_input_shape("point_cloud", list(pc_np.shape))
        engine.register_infer_input_data("point_cloud", pc_np)
        engine.register_infer_input_shape("init_T", list(init_np.shape))
        engine.register_infer_input_data("init_T", init_np)
        engine.register_infer_input_shape("post_cam2ego_T", list(post_np.shape))
        engine.register_infer_input_data("post_cam2ego_T", post_np)
        engine.register_infer_input_shape("intrinsic", list(K_np.shape))
        engine.register_infer_input_data("intrinsic", K_np)

        engine.inference(B)

        pred_T_np = np.zeros([B, 4, 4], dtype=np.float32)
        engine.copy_output_device_data_to_host_from_layer(
            "pred_T", 0, self.dr.MODEL_FLOAT, pred_T_np, True
        )

        return torch.from_numpy(pred_T_np).to(self.device)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(backend, val_loader, cfg, device="cuda", vis_dir=None, epoch=-1):
    """Run evaluation, collect per-sample errors and timing stats.
    If vis_dir is set, save point cloud projection images for each sample.
    Uses the same model call and visualization as evaluate_checkpoint.py."""
    from tools import generate_single_perturbation_from_T
    from visualization import (
        compute_pose_errors as _cpe,
        visualize_batch_projection,
    )

    angle_deg = cfg.get("angle_range_deg", 10.0)
    trans_range = cfg.get("trans_range", 0.3)
    max_batches = cfg.get("max_batches", 0)
    rotation_only = cfg.get("rotation_only", True)
    vis_points = cfg.get("vis_points", 8000)
    vis_point_radius = cfg.get("vis_point_radius", 2)

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  Saving visualizations to: {vis_dir}")

    all_errors = []
    timings = []
    total_samples = 0

    print(f"\nEvaluating (angle={angle_deg}deg, trans={trans_range}m, "
          f"max_batches={max_batches or 'all'}) ...\n")

    for batch_idx, batch_data in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if batch_data is None:
            continue

        imgs_list, pcs_list, masks_list, gt_T_list, intrinsics_list = batch_data
        imgs_np = np.array(imgs_list)
        imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float().to(device)
        pcs_np_full = np.array(pcs_list)
        pcs = torch.from_numpy(pcs_np_full[:, :, :3]).float().to(device)
        masks_np = np.array(masks_list)
        gt_T_np = np.array(gt_T_list).astype(np.float32)
        gt_T = torch.from_numpy(gt_T_np).float().to(device)
        K_np = np.array(intrinsics_list).astype(np.float32)
        intrinsics = torch.from_numpy(K_np).float().to(device)
        B = imgs.shape[0]
        perturbed_batch, _, _ = generate_single_perturbation_from_T(
            gt_T_np,
            angle_range_deg=angle_deg,
            trans_range=trans_range if not rotation_only else 0.0,
        )
        init_T = torch.from_numpy(perturbed_batch).float().to(device)
        init_T_np = perturbed_batch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_T = backend.predict(imgs, pcs, masks_np, init_T, gt_T, intrinsics)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(dt_ms)

        pred_np = pred_T.detach().cpu().numpy()
        for b in range(B):
            try:
                err = _cpe(pred_np[b], gt_T_np[b])
                all_errors.append(err)
            except Exception as e:
                print(f"  WARNING: batch {batch_idx} sample {b}: {e}")
                continue
            total_samples += 1

            if vis_dir:
                vis_image = visualize_batch_projection(
                    images=imgs_np[b:b+1],
                    points_batch=pcs_np_full[b:b+1, :, :3],
                    init_T_batch=init_T_np[b:b+1],
                    gt_T_batch=gt_T_np[b:b+1],
                    pred_T_batch=pred_np[b:b+1],
                    K_batch=K_np[b:b+1],
                    masks=masks_np[b:b+1],
                    num_samples=1,
                    max_points=vis_points,
                    point_radius=vis_point_radius,
                    rotation_only=rotation_only,
                    phase="Eval",
                    epoch=epoch,
                )
                fname = os.path.join(vis_dir, f"batch{batch_idx:03d}_sample{b:02d}.png")
                cv2.imwrite(fname, vis_image)

        if not all_errors:
            continue
        avg_rot = np.mean([e["rot_error"] for e in all_errors[-min(B, len(all_errors)):]])
        print(f"  batch {batch_idx:4d}  |  samples {total_samples:5d}  |  "
              f"batch_time {dt_ms:7.1f}ms  |  avg_rot {avg_rot:.4f}deg")

    return all_errors, timings


def print_report(all_errors, timings, output_path=None):
    """Print and optionally save an evaluation report."""
    if not all_errors:
        print("No samples evaluated.")
        return

    keys = ["rot_error", "roll_error", "pitch_error", "yaw_error",
            "trans_error", "fwd_error", "lat_error", "ht_error"]
    report = {}
    for k in keys:
        vals = [e[k] for e in all_errors]
        report[k] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
        }

    timing_arr = np.array(timings)
    report["timing_ms"] = {
        "mean": float(timing_arr.mean()),
        "median": float(np.median(timing_arr)),
        "p95": float(np.percentile(timing_arr, 95)),
        "min": float(timing_arr.min()),
        "max": float(timing_arr.max()),
    }
    report["num_samples"] = len(all_errors)
    report["num_batches"] = len(timings)

    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT  ({report['num_samples']} samples, "
          f"{report['num_batches']} batches)")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Mean':>10} {'Median':>10} {'P95':>10} {'Max':>10}")
    print("-" * 62)
    for k in keys:
        unit = "deg" if "rot" in k or "roll" in k or "pitch" in k or "yaw" in k else "m"
        s = report[k]
        print(f"{k:<20} {s['mean']:10.4f} {s['median']:10.4f} "
              f"{s['p95']:10.4f} {s['max']:10.4f}  {unit}")

    print(f"\n{'Timing (ms)':<20} {'Mean':>10} {'Median':>10} {'P95':>10} {'Min':>10} {'Max':>10}")
    print("-" * 72)
    t = report["timing_ms"]
    print(f"{'batch_time':<20} {t['mean']:10.1f} {t['median']:10.1f} "
          f"{t['p95']:10.1f} {t['min']:10.1f} {t['max']:10.1f}")
    per_sample = t["mean"] / max(1, report["num_samples"] / report["num_batches"])
    print(f"{'per_sample (est.)':<20} {per_sample:10.1f}")
    print("=" * 70)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BEVCalib inference & evaluation")
    parser.add_argument("--config", type=str, default="utils/drinfer_config.yaml")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["pytorch", "drinfer"],
                        help="Override inference backend from config")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON report path")
    parser.add_argument("--vis-dir", type=str, default=None,
                        help="Directory to save projection visualization images")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    backend_name = args.backend or cfg.get("inference_backend", "pytorch")
    device = cfg.get("device", "cuda")

    # Must set BEFORE importing any bevcalib modules (bev_settings reads at import)
    os.environ["BEV_ZBOUND_STEP"] = str(cfg.get("bev_zbound_step", "2.0"))

    print("=" * 60)
    print("BEVCalib Inference Evaluation")
    print(f"  backend    : {backend_name}")
    print(f"  checkpoint : {cfg['ckpt_path']}")
    print(f"  dataset    : {cfg['dataset_root']}")
    print(f"  device     : {device}")
    print("=" * 60)

    rotation_only = cfg.get("rotation_only", True)
    deformable = cfg.get("deformable", False)
    bev_encoder = cfg.get("bev_encoder", True)
    img_shape = (cfg["img_height"], cfg["img_width"])

    use_mlp_head = cfg.get("use_mlp_head", None)

    if backend_name == "pytorch":
        from bevcalib_inference import load_bevcalib_inference, BEVCalibInference
        wrapper, epoch = load_bevcalib_inference(
            ckpt_path=cfg["ckpt_path"],
            device=device,
            img_shape=img_shape,
            rotation_only=rotation_only,
            deformable=deformable,
            bev_encoder=bev_encoder,
            use_mlp_head=use_mlp_head,
        )
        model = wrapper.model
        print(f"  epoch      : {epoch}")
        backend = PyTorchBackend(model, device=device)
    elif backend_name == "drinfer":
        from bevcalib_inference import load_bevcalib_inference
        wrapper, epoch = load_bevcalib_inference(
            ckpt_path=cfg["ckpt_path"],
            device=device,
            img_shape=img_shape,
            rotation_only=rotation_only,
            deformable=deformable,
            bev_encoder=bev_encoder,
            use_mlp_head=cfg.get("use_mlp_head", None),
        )
        print(f"  epoch      : {epoch}")
        drinfer_dir = cfg.get("export_dir",
                              os.path.join(os.path.dirname(cfg["ckpt_path"]), "drinfer"))
        backend = DrInferBackend(wrapper, drinfer_dir, cfg, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    val_loader = build_val_loader(cfg)
    vis_dir = args.vis_dir or cfg.get("vis_dir", None)
    all_errors, timings = evaluate(backend, val_loader, cfg, device=device, vis_dir=vis_dir, epoch=epoch)

    output_path = args.output or cfg.get("report_output", None)
    if vis_dir and output_path is None:
        output_path = os.path.join(vis_dir, "eval_report.json")
    print_report(all_errors, timings, output_path=output_path)


if __name__ == "__main__":
    main()
