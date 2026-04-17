#!/usr/bin/env python3
"""
BEVCalib DrInfer / PyTorch inference benchmark and evaluation.

Supports three modes:
    eval    – run a single backend and report accuracy + timing
    compare – run both PyTorch and DrInfer on the same data, report
              per-sample accuracy delta, latency ratio, and GPU memory usage

Usage:
    python utils/drinfer_infer.py --config configs/drinfer_config_xxx.yaml
    python utils/drinfer_infer.py --config configs/drinfer_config_xxx.yaml --mode compare
    python utils/drinfer_infer.py --config configs/drinfer_config_xxx.yaml --backend pytorch
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
# GPU memory helpers
# ---------------------------------------------------------------------------
def reset_memory_stats(device="cuda"):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

def get_memory_mb(device="cuda"):
    torch.cuda.synchronize(device)
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


# ---------------------------------------------------------------------------
# PyTorch backend – BEVCalib.forward() (same path as training)
# ---------------------------------------------------------------------------
class PyTorchBackend:
    name = "pytorch"

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
# PyTorch Inference backend – BEVCalibInference.forward()
# (same computation graph as DrInfer export, for fair compare mode)
# ---------------------------------------------------------------------------
class PyTorchInferenceBackend:
    name = "pytorch_inference"

    def __init__(self, wrapper, device="cuda"):
        self.wrapper = wrapper
        self.device = device
        self.last_timing = {}

    @torch.no_grad()
    def predict(self, imgs, pcs, masks_np, init_T, gt_T, intrinsics):
        B = imgs.shape[0]
        post_T = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1).contiguous()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast():
            pred_T = self.wrapper(imgs, pcs, init_T, post_T, intrinsics)
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self.last_timing = {"infer_ms": infer_ms, "transfer_ms": 0.0}
        return pred_T


# ---------------------------------------------------------------------------
# DrInfer backend
# ---------------------------------------------------------------------------
class DrInferBackend:
    name = "drinfer"
    accepts_numpy = True

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

        candidates = [
            os.path.join(drinfer_model_dir, f"{full_name}.bin"),
            os.path.join(drinfer_model_dir, f"{model_name}.bin"),
            os.path.join(drinfer_model_dir, model_name, "engine_graph",
                         f"{model_name}_engine_graph.bin"),
        ]
        bin_path = None
        for c in candidates:
            if os.path.isfile(c):
                bin_path = os.path.abspath(c)
                break
        if bin_path is None:
            raise FileNotFoundError(
                f"DrInfer model not found in {drinfer_model_dir}\n"
                f"Searched: {[os.path.basename(c) for c in candidates]}\n"
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
            fixed_shape_configs={
                "image": True, "point_cloud": False,
                "init_T": True, "post_cam2ego_T": True, "intrinsic": True,
            },
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

    def predict(self, imgs, pcs, masks_np, init_T, gt_T, intrinsics,
                imgs_np=None, pcs_np=None, init_T_np=None, K_np=None):
        """Run DrInfer inference.  When numpy arrays are supplied via the
        ``*_np`` kwargs the engine is fed directly from CPU — skipping the
        GPU→CPU round-trip that otherwise dominates benchmark latency."""
        B = imgs.shape[0] if imgs is not None else imgs_np.shape[0]
        engine = self._inf_engine._engine

        use_numpy = imgs_np is not None
        if use_numpy:
            img_f = np.ascontiguousarray(
                np.transpose(imgs_np, (0, 3, 1, 2)).astype(np.float32))
            pc_f = np.ascontiguousarray(pcs_np[:, :, :3].astype(np.float32))
            N = pc_f.shape[1]
            if N < self.N:
                pc_f = np.concatenate(
                    [pc_f, np.zeros((B, self.N - N, 3), dtype=np.float32)], axis=1)
            elif N > self.N:
                pc_f = pc_f[:, :self.N, :]
            init_f = np.ascontiguousarray(init_T_np.astype(np.float32))
            post_f = np.ascontiguousarray(
                np.tile(np.eye(4, dtype=np.float32), (B, 1, 1)))
            K_f = np.ascontiguousarray(K_np.astype(np.float32))
        else:
            post_T = torch.eye(4, device=self.device).unsqueeze(0).expand(
                B, -1, -1).contiguous()
            if pcs.shape[1] < self.N:
                pad = torch.zeros(
                    B, self.N - pcs.shape[1], 3,
                    device=pcs.device, dtype=pcs.dtype)
                pcs = torch.cat([pcs, pad], dim=1)
            elif pcs.shape[1] > self.N:
                pcs = pcs[:, :self.N, :]
            torch.cuda.synchronize()
            img_f = np.ascontiguousarray(imgs.cpu().numpy().astype(np.float32))
            pc_f = np.ascontiguousarray(pcs.cpu().numpy().astype(np.float32))
            init_f = np.ascontiguousarray(init_T.cpu().numpy().astype(np.float32))
            post_f = np.ascontiguousarray(post_T.cpu().numpy().astype(np.float32))
            K_f = np.ascontiguousarray(intrinsics.cpu().numpy().astype(np.float32))

        t_xfer0 = time.perf_counter()
        engine.register_infer_input_shape("image", list(img_f.shape))
        engine.register_infer_input_data("image", img_f)
        engine.register_infer_input_shape("point_cloud", list(pc_f.shape))
        engine.register_infer_input_data("point_cloud", pc_f)
        engine.register_infer_input_shape("init_T", list(init_f.shape))
        engine.register_infer_input_data("init_T", init_f)
        engine.register_infer_input_shape("post_cam2ego_T", list(post_f.shape))
        engine.register_infer_input_data("post_cam2ego_T", post_f)
        engine.register_infer_input_shape("intrinsic", list(K_f.shape))
        engine.register_infer_input_data("intrinsic", K_f)
        t_xfer1 = time.perf_counter()

        engine.inference(B)
        t_infer1 = time.perf_counter()

        pred_T_np = np.zeros([B, 4, 4], dtype=np.float32)
        engine.copy_output_device_data_to_host_from_layer(
            "pred_T", 0, self.dr.MODEL_FLOAT, pred_T_np, True
        )
        result = torch.from_numpy(pred_T_np).to(self.device)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        transfer_ms = (t_xfer1 - t_xfer0) * 1000.0 + (t_end - t_infer1) * 1000.0
        infer_ms = (t_infer1 - t_xfer1) * 1000.0
        self.last_timing = {"infer_ms": infer_ms, "transfer_ms": transfer_ms}
        return result


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(backend, val_loader, cfg, device="cuda", vis_dir=None, epoch=-1,
             warmup_batches=3, track_memory=False):
    """Run evaluation, collect per-sample errors, timing, and optional memory stats."""
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
    infer_timings = []
    transfer_timings = []
    total_samples = 0

    if track_memory:
        reset_memory_stats(device)

    print(f"\nEvaluating [{backend.name}] (angle={angle_deg}deg, trans={trans_range}m, "
          f"max_batches={max_batches or 'all'}, warmup={warmup_batches}) ...\n")

    for batch_idx, batch_data in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches + warmup_batches:
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

        np_kwargs = {}
        if getattr(backend, "accepts_numpy", False):
            np_kwargs = dict(imgs_np=imgs_np, pcs_np=pcs_np_full,
                             init_T_np=init_T_np, K_np=K_np)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_T = backend.predict(imgs, pcs, masks_np, init_T, gt_T, intrinsics,
                                 **np_kwargs)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        bt = getattr(backend, "last_timing", {})

        if batch_idx < warmup_batches:
            if batch_idx == 0:
                print(f"  [warmup] batch {batch_idx}: {dt_ms:.1f}ms")
            continue

        timings.append(dt_ms)
        if bt:
            infer_timings.append(bt.get("infer_ms", dt_ms))
            transfer_timings.append(bt.get("transfer_ms", 0.0))

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
        infer_str = ""
        if bt:
            infer_str = f"  infer {bt.get('infer_ms', 0):.1f}ms  xfer {bt.get('transfer_ms', 0):.1f}ms"
        print(f"  batch {batch_idx:4d}  |  samples {total_samples:5d}  |  "
              f"total {dt_ms:7.1f}ms{infer_str}  |  avg_rot {avg_rot:.4f}deg")

    memory_stats = get_memory_mb(device) if track_memory else None
    timing_detail = {}
    if infer_timings:
        timing_detail["infer_timings"] = infer_timings
        timing_detail["transfer_timings"] = transfer_timings
    return all_errors, timings, memory_stats, timing_detail


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def print_report(all_errors, timings, output_path=None, memory_stats=None,
                 backend_name="", timing_detail=None):
    """Print and optionally save an evaluation report."""
    if not all_errors:
        print("No samples evaluated.")
        return {}

    keys = ["rot_error", "roll_error", "pitch_error", "yaw_error",
            "trans_error", "fwd_error", "lat_error", "ht_error"]
    report = {"backend": backend_name}
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

    td = timing_detail or {}
    if td.get("infer_timings"):
        infer_arr = np.array(td["infer_timings"])
        xfer_arr = np.array(td["transfer_timings"])
        report["infer_ms"] = {
            "mean": float(infer_arr.mean()),
            "median": float(np.median(infer_arr)),
            "p95": float(np.percentile(infer_arr, 95)),
            "min": float(infer_arr.min()),
            "max": float(infer_arr.max()),
        }
        report["transfer_ms"] = {
            "mean": float(xfer_arr.mean()),
            "median": float(np.median(xfer_arr)),
        }

    if memory_stats:
        report["memory"] = memory_stats

    label = f" [{backend_name}]" if backend_name else ""
    print(f"\n{'=' * 70}")
    print(f"EVALUATION REPORT{label}  ({report['num_samples']} samples, "
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
    print(f"{'total (e2e)':<20} {t['mean']:10.1f} {t['median']:10.1f} "
          f"{t['p95']:10.1f} {t['min']:10.1f} {t['max']:10.1f}")
    if report.get("infer_ms"):
        im = report["infer_ms"]
        xm = report["transfer_ms"]
        print(f"{'  pure_infer':<20} {im['mean']:10.1f} {im['median']:10.1f} "
              f"{im['p95']:10.1f} {im['min']:10.1f} {im['max']:10.1f}")
        print(f"{'  data_transfer':<20} {xm['mean']:10.1f} {xm['median']:10.1f} "
              f"{'':>10} {'':>10} {'':>10}")
    per_sample = t["mean"] / max(1, report["num_samples"] / report["num_batches"])
    print(f"{'per_sample (est.)':<20} {per_sample:10.1f}")

    if memory_stats:
        print(f"\n{'GPU Memory (MB)':<20} {'Allocated':>12} {'Reserved':>12} {'Peak':>12}")
        print("-" * 58)
        print(f"{'current':<20} {memory_stats['allocated_mb']:12.1f} "
              f"{memory_stats['reserved_mb']:12.1f} "
              f"{memory_stats['peak_allocated_mb']:12.1f}")

    print("=" * 70)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")

    return report


def print_compare_report(report_pt, report_dr, output_path=None):
    """Print side-by-side comparison of PyTorch vs DrInfer."""
    print(f"\n{'=' * 80}")
    print("COMPARISON REPORT: PyTorch (BEVCalibInference) vs DrInfer")
    print("=" * 80)

    keys = ["rot_error", "roll_error", "pitch_error", "yaw_error",
            "trans_error", "fwd_error", "lat_error", "ht_error"]

    print(f"\n{'Metric':<20} {'PyTorch':>10} {'DrInfer':>10} {'Delta':>10} {'RelDiff%':>10}")
    print("-" * 62)
    for k in keys:
        unit = "deg" if "rot" in k or "roll" in k or "pitch" in k or "yaw" in k else "m"
        pt_v = report_pt[k]["mean"]
        dr_v = report_dr[k]["mean"]
        delta = dr_v - pt_v
        rel = abs(delta) / max(pt_v, 1e-8) * 100
        print(f"{k:<20} {pt_v:10.4f} {dr_v:10.4f} {delta:+10.4f} {rel:10.2f}%  {unit}")

    pt_t = report_pt["timing_ms"]["mean"]
    dr_t = report_dr["timing_ms"]["mean"]
    speedup = pt_t / max(dr_t, 1e-3)
    print(f"\n{'Latency (ms)':<24} {'PyTorch':>10} {'DrInfer':>10} {'Speedup':>10}")
    print("-" * 56)
    print(f"{'total (e2e)':<24} {pt_t:10.1f} {dr_t:10.1f} {speedup:10.2f}x")

    pt_infer = report_pt.get("infer_ms", {}).get("mean")
    dr_infer = report_dr.get("infer_ms", {}).get("mean")
    if pt_infer is not None and dr_infer is not None:
        infer_speedup = pt_infer / max(dr_infer, 1e-3)
        print(f"{'pure_infer':<24} {pt_infer:10.1f} {dr_infer:10.1f} {infer_speedup:10.2f}x")
    dr_xfer = report_dr.get("transfer_ms", {}).get("mean")
    if dr_xfer is not None:
        print(f"{'drinfer_data_transfer':<24} {'--':>10} {dr_xfer:10.1f} {'':>10}")

    pt_ps = pt_t / max(1, report_pt["num_samples"] / report_pt["num_batches"])
    dr_ps = dr_t / max(1, report_dr["num_samples"] / report_dr["num_batches"])
    ps_speedup = pt_ps / max(dr_ps, 1e-3)
    print(f"{'per_sample (est.)':<24} {pt_ps:10.1f} {dr_ps:10.1f} {ps_speedup:10.2f}x")

    if report_pt.get("memory") and report_dr.get("memory"):
        pt_m = report_pt["memory"]["peak_allocated_mb"]
        dr_m = report_dr["memory"]["peak_allocated_mb"]
        ratio = dr_m / max(pt_m, 1e-3)
        print(f"\n{'GPU Peak Memory (MB)':<20} {'PyTorch':>10} {'DrInfer':>10} {'Ratio':>10}")
        print("-" * 52)
        print(f"{'peak_allocated':<20} {pt_m:10.1f} {dr_m:10.1f} {ratio:10.2f}x")

    print("=" * 80)

    comparison = {
        "rot_error_delta_deg": report_dr["rot_error"]["mean"] - report_pt["rot_error"]["mean"],
        "e2e_speedup": speedup,
        "per_sample_speedup": ps_speedup,
    }
    if pt_infer is not None and dr_infer is not None:
        comparison["pure_infer_speedup"] = pt_infer / max(dr_infer, 1e-3)
    combined = {
        "pytorch": report_pt,
        "drinfer": report_dr,
        "comparison": comparison,
    }
    if report_pt.get("memory") and report_dr.get("memory"):
        combined["comparison"]["memory_ratio"] = (
            report_dr["memory"]["peak_allocated_mb"]
            / max(report_pt["memory"]["peak_allocated_mb"], 1e-3)
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nComparison report saved to {output_path}")

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_backends(cfg, device, mode):
    """Load model and create backend(s) based on mode."""
    from bevcalib_inference import load_bevcalib_inference

    rotation_only = cfg.get("rotation_only", True)
    deformable = cfg.get("deformable", False)
    bev_encoder = cfg.get("bev_encoder", True)
    img_shape = (cfg["img_height"], cfg["img_width"])
    use_mlp_head = cfg.get("use_mlp_head", None)
    voxel_mode = cfg.get("voxel_mode", "scatter")
    to_bev_mode = cfg.get("to_bev_mode", "concat")
    scatter_reduce = cfg.get("scatter_reduce", "sum")

    wrapper, epoch = load_bevcalib_inference(
        ckpt_path=cfg["ckpt_path"],
        device=device,
        img_shape=img_shape,
        rotation_only=rotation_only,
        deformable=deformable,
        bev_encoder=bev_encoder,
        use_mlp_head=use_mlp_head,
        voxel_mode=voxel_mode,
        to_bev_mode=to_bev_mode,
        scatter_reduce=scatter_reduce,
    )
    print(f"  epoch: {epoch}")

    return wrapper, epoch


def main():
    parser = argparse.ArgumentParser(description="BEVCalib inference & evaluation")
    parser.add_argument("--config", type=str, default="utils/drinfer_config.yaml")
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["eval", "compare"],
                        help="eval: single backend; compare: PyTorch vs DrInfer side-by-side")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["pytorch", "drinfer"],
                        help="Override inference backend (for eval mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON report path")
    parser.add_argument("--vis-dir", type=str, default=None,
                        help="Directory to save projection visualization images")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    os.environ["BEV_ZBOUND_STEP"] = str(cfg.get("bev_zbound_step", "2.0"))

    if args.mode == "compare":
        _run_compare(cfg, device, args)
    else:
        _run_eval(cfg, device, args)


def _run_eval(cfg, device, args):
    backend_name = args.backend or cfg.get("inference_backend", "pytorch")

    print("=" * 60)
    print("BEVCalib Inference Evaluation")
    print(f"  backend    : {backend_name}")
    print(f"  checkpoint : {cfg['ckpt_path']}")
    print(f"  dataset    : {cfg['dataset_root']}")
    print(f"  device     : {device}")
    print("=" * 60)

    wrapper, epoch = _load_backends(cfg, device, "eval")

    if backend_name == "pytorch":
        backend = PyTorchBackend(wrapper.model, device=device)
    elif backend_name == "drinfer":
        drinfer_dir = cfg.get("export_dir",
                              os.path.join(os.path.dirname(cfg["ckpt_path"]), "drinfer"))
        backend = DrInferBackend(wrapper, drinfer_dir, cfg, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    val_loader = build_val_loader(cfg)
    vis_dir = args.vis_dir or cfg.get("vis_dir", None)
    all_errors, timings, memory, timing_detail = evaluate(
        backend, val_loader, cfg, device=device,
        vis_dir=vis_dir, epoch=epoch, track_memory=True,
    )

    output_path = args.output or cfg.get("report_output", None)
    if vis_dir and output_path is None:
        output_path = os.path.join(vis_dir, "eval_report.json")
    print_report(all_errors, timings, output_path=output_path,
                 memory_stats=memory, backend_name=backend_name,
                 timing_detail=timing_detail)


def _run_compare(cfg, device, args):
    print("=" * 60)
    print("BEVCalib PyTorch vs DrInfer Comparison")
    print(f"  checkpoint : {cfg['ckpt_path']}")
    print(f"  dataset    : {cfg['dataset_root']}")
    print(f"  device     : {device}")
    print("=" * 60)

    wrapper, epoch = _load_backends(cfg, device, "compare")

    drinfer_dir = cfg.get("export_dir",
                          os.path.join(os.path.dirname(cfg["ckpt_path"]), "drinfer"))

    pt_backend = PyTorchInferenceBackend(wrapper, device=device)
    dr_backend = DrInferBackend(wrapper, drinfer_dir, cfg, device=device)

    val_loader = build_val_loader(cfg)
    warmup = cfg.get("warmup_batches", 3)

    print("\n--- Phase 1: PyTorch (BEVCalibInference) ---")
    torch.cuda.empty_cache()
    reset_memory_stats(device)
    pt_errors, pt_timings, pt_mem, pt_detail = evaluate(
        pt_backend, val_loader, cfg, device=device,
        warmup_batches=warmup, track_memory=True,
    )

    print("\n--- Phase 2: DrInfer ---")
    torch.cuda.empty_cache()
    reset_memory_stats(device)
    dr_errors, dr_timings, dr_mem, dr_detail = evaluate(
        dr_backend, val_loader, cfg, device=device,
        warmup_batches=warmup, track_memory=True,
    )

    report_pt = print_report(pt_errors, pt_timings,
                             memory_stats=pt_mem, backend_name="pytorch_inference",
                             timing_detail=pt_detail)
    report_dr = print_report(dr_errors, dr_timings,
                             memory_stats=dr_mem, backend_name="drinfer",
                             timing_detail=dr_detail)

    output_path = args.output or cfg.get("report_output", None)
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(cfg["ckpt_path"]), "compare_report.json")
    print_compare_report(report_pt, report_dr, output_path=output_path)


if __name__ == "__main__":
    main()
