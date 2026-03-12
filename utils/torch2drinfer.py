#!/usr/bin/env python3
"""
BEVCalib model -> DrInfer conversion via model_parserv3.

Full-model export only -- no fallback strategies.
Patches dynamic ops (Swin padding, TransformerEncoder fused path) before export.

Usage (in bevcalib310 conda env):
    export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH
    python utils/torch2drinfer.py --config utils/drinfer_config.yaml
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import yaml

_KITTI_DIR = os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib')
if _KITTI_DIR not in sys.path:
    sys.path.insert(0, _KITTI_DIR)


def export_full_drinfer(wrapper, cfg, export_dir):
    """Export the full BEVCalib model using DrInfer's model_parserv3."""
    from frontend_python.pytorch_parser.engine_graph_exporter import model_parserv3
    import drinfer

    H = cfg["img_height"]
    W = cfg["img_width"]
    N = cfg.get("max_num_points", 200000)

    dummy_img = torch.randn(1, 3, H, W, device="cuda")
    dummy_pc = torch.zeros(1, N, 3, device="cuda")
    dummy_pc[0, :, 0] = torch.rand(N, device="cuda") * 200.0
    dummy_pc[0, :, 1] = torch.rand(N, device="cuda") * 200.0 - 100
    dummy_pc[0, :, 2] = torch.rand(N, device="cuda") * 20.0 - 10

    dummy_init_T = torch.eye(4, device="cuda").unsqueeze(0)
    dummy_post_T = torch.eye(4, device="cuda").unsqueeze(0)
    dummy_K = torch.eye(3, device="cuda").unsqueeze(0)
    dummy_K[0, 0, 0] = 1222.0
    dummy_K[0, 1, 1] = 1222.0
    dummy_K[0, 0, 2] = 320.0
    dummy_K[0, 1, 2] = 180.0

    dummy_input = (dummy_img, dummy_pc, dummy_init_T, dummy_post_T, dummy_K)

    optimization_flags = (
        drinfer.OptimizationFlag.OPTIMIZATION_CONV_RELU_MERGE
        | drinfer.OptimizationFlag.OPTIMIZATION_CONV_BN_MERGE
        | drinfer.OptimizationFlag.OPTIMIZATION_ARITHMATIC_MERGE
        | drinfer.OptimizationFlag.OPTIMIZATION_MERGE_CONCAT
        | drinfer.OptimizationFlag.OPTIMIZATION_MEMORY_REUSE
        | drinfer.OptimizationFlag.OPTIMIZATION_AUTOTUNE
        | drinfer.OptimizationFlag.OPTIMIZATION_TENSORFORMAT_TRANSFORM
        | drinfer.OptimizationFlag.OPTIMIZATION_CUDA_GRAPH
        | drinfer.OptimizationFlag.OPTIMIZATION_MULTIPLE_STREAM_SINGLE_BATCH
        | drinfer.OptimizationFlag.OPTIMIZATION_MEM_INTENSIVE
        | drinfer.OptimizationFlag.OPTIMIZATION_FC_BN_MERGE
        | drinfer.OptimizationFlag.OPTIMIZATION_FC_RELU_MERGE
    )

    model_name = cfg.get("model_name", "bevcalib_fusion_head")
    model_version = cfg.get("model_version", "v1")

    fixed_shapes = [True] * len(dummy_input)

    print(f"\n[export] Exporting full model via model_parserv3 ")
    engine_res, gt = model_parserv3(
        wrapper, model_name, model_version,
        dummy_input, export_dir, verbose=True,
        input_names=["image", "point_cloud", "init_T", "post_cam2ego_T", "intrinsic"],
        output_names=["pred_T"],
        runtime_type=drinfer.MODEL_FLOAT,
        optimization_flag=optimization_flags,
        jit_check_trace=False,
        fixed_shape_configs=fixed_shapes,
        empty_cache=True,
        verify_export_graph=True,
    )

    gt_np = gt.cpu().detach().numpy()
    dr_np = engine_res[0]

    if np.isnan(dr_np).any():
        print("[export] ERROR: DrInfer output contains NaN!")
        print(f"  GT (PyTorch):\n{gt_np}")
        print(f"  DrInfer:\n{dr_np}")
        return False
    else:
        print(f"GT (PyTorch):\n{gt_np}")
        print(f"DrInfer:\n{dr_np}")

    diff = np.abs(dr_np - gt_np)
    print(f"[export] max diff: {diff.max():.6f}, mean: {diff.mean():.6f}")
    np.testing.assert_allclose(dr_np, gt_np, rtol=5e-2, atol=5e-2)
    print("[export] Numerical verification PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="BEVCalib -> DrInfer converter")
    parser.add_argument("--config", type=str, default="utils/drinfer_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = cfg["ckpt_path"]
    export_dir = cfg.get("export_dir",
                         os.path.join(os.path.dirname(ckpt_path), "drinfer"))
    os.makedirs(export_dir, exist_ok=True)

    os.environ["BEV_ZBOUND_STEP"] = str(cfg.get("bev_zbound_step", "2.0"))

    from bevcalib_inference import load_bevcalib_inference, prepare_for_drinfer_export

    img_h, img_w = cfg["img_height"], cfg["img_width"]

    print("=" * 60)
    print("BEVCalib -> DrInfer Conversion")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  export_dir : {export_dir}")
    print(f"  img_shape  : ({img_h}, {img_w})")
    print("=" * 60)

    wrapper, epoch = load_bevcalib_inference(
        ckpt_path=ckpt_path,
        device="cuda",
        img_shape=(img_h, img_w),
        rotation_only=cfg.get("rotation_only", True),
        deformable=cfg.get("deformable", False),
        bev_encoder=cfg.get("bev_encoder", True),
    )
    print(f"  epoch  : {epoch}")
    print(f"  params : {sum(p.numel() for p in wrapper.parameters()):,}")

    print("\n--- Preparing model for DrInfer export ---")
    prepare_for_drinfer_export(wrapper, img_shape=(img_h, img_w))

    t0 = time.time()
    ok = export_full_drinfer(wrapper, cfg, export_dir)
    dt = time.time() - t0
    if ok:
        print(f"\nFull model export completed in {dt:.1f}s -> {export_dir}")
    else:
        print(f"\nExport FAILED in {dt:.1f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
