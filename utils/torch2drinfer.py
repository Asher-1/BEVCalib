#!/usr/bin/env python3
"""
BEVCalib model -> DrInfer conversion.

Supports two output layouts:
  --layout flat    (default) Exports .bin + .txt directly into export_dir.
                   Fast export with optional numpy-level verification.
  --layout pmodel  Produces the full pmodel deployment directory layout:
                     $dir/bevcalib_fusion_head/
                     ├── engine_graph/ (.bin, .txt, nn_param.cfg, trace.json)
                     ├── input_data/   (per-input .bin)
                     └── output_data/  (PyTorch reference .bin)
                   Compatible with `pmodel forward` / `pmodel calibrate`.

Usage (in bevcalib310 conda env):
    export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH

    # Flat layout (quick export + verify)
    python utils/torch2drinfer.py --config configs/drinfer_config_xxx.yaml

    # pmodel layout (full deployment artifacts)
    python utils/torch2drinfer.py --config configs/drinfer_config_xxx.yaml --layout pmodel

    # Verify pmodel artifacts:
    pmodel forward $trace_dir/bevcalib_fusion_head
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import yaml
import drinfer

_KITTI_DIR = os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib')
if _KITTI_DIR not in sys.path:
    sys.path.insert(0, _KITTI_DIR)


INPUT_NAMES = ["image", "point_cloud", "init_T", "post_cam2ego_T", "intrinsic"]
OUTPUT_NAMES = ["pred_T"]
FIXED_SHAPES = [True, False, True, True, True]


def _build_dummy_inputs(cfg, sparse=False):
    """Build trace inputs.  Prefers loading one real sample from the validation
    dataset so that bev_pool succeeds and cam_bev_mask is non-zero (critical for
    correct DrInfer tracing of the mask path).  Falls back to synthetic data when
    the dataset is unavailable."""
    H = cfg["img_height"]
    W = cfg["img_width"]
    N_target = cfg.get("trace_num_points", 30000 if sparse else 80000)

    real = _try_load_real_sample(cfg, N_target)
    if real is not None:
        return real

    print("[trace] WARNING: real data unavailable, using synthetic (bev_pool may fail)")
    N = N_target
    dummy_img = torch.randn(1, 3, H, W, device="cuda")
    dummy_pc = torch.zeros(1, N, 3, device="cuda")
    if sparse:
        n_anchors = min(N // 20, 500)
        anchors = torch.zeros(n_anchors, 3, device="cuda")
        anchors[:, 0] = torch.rand(n_anchors, device="cuda") * 60.0 + 10.0
        anchors[:, 1] = torch.rand(n_anchors, device="cuda") * 40.0 - 20.0
        anchors[:, 2] = torch.rand(n_anchors, device="cuda") * 4.0 - 2.0
        idx = torch.randint(0, n_anchors, (N,), device="cuda")
        dummy_pc[0] = anchors[idx] + torch.randn(N, 3, device="cuda") * 0.001
    else:
        dummy_pc[0, :, 0] = torch.rand(N, device="cuda") * 200.0
        dummy_pc[0, :, 1] = torch.rand(N, device="cuda") * 200.0 - 100
        dummy_pc[0, :, 2] = torch.rand(N, device="cuda") * 20.0 - 10

    dummy_init_T = torch.eye(4, device="cuda").unsqueeze(0)
    dummy_post_T = torch.eye(4, device="cuda").unsqueeze(0)
    dummy_K = torch.eye(3, device="cuda").unsqueeze(0)
    dummy_K[0, 0, 0] = 1222.0
    dummy_K[0, 1, 1] = 1222.0
    dummy_K[0, 0, 2] = float(W) / 2
    dummy_K[0, 1, 2] = float(H) / 2
    return (dummy_img, dummy_pc, dummy_init_T, dummy_post_T, dummy_K)


def _try_load_real_sample(cfg, N_target):
    """Load one real sample via the same pipeline as drinfer_infer.py.
    Returns (img, pc, init_T, post_T, K) on GPU, or None on failure."""
    try:
        from torch.utils.data import DataLoader, random_split
        from custom_dataset import CustomDataset
        from train_kitti import PreprocessedDataset, collate_fn

        ds = CustomDataset(data_folder=cfg["dataset_root"], suf=".png")
        val_ratio = cfg.get("validate_sample_ratio", 0.2)
        n_val = int(len(ds) * val_ratio)
        _, val_raw = random_split(
            ds, [len(ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(114514))
        target_w = cfg.get("img_width", 640)
        target_h = cfg.get("img_height", 360)
        val_ds = PreprocessedDataset(val_raw, target_size=(target_w, target_h))
        dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)
        batch = next(iter(dl))
        imgs_list, pcs_list, masks_list, gt_T_list, intrinsics_list = batch
        imgs = torch.from_numpy(np.array(imgs_list)).permute(0, 3, 1, 2).float().cuda()
        pcs_full = torch.from_numpy(np.array(pcs_list)[:, :, :3]).float().cuda()
        gt_T = torch.from_numpy(np.array(gt_T_list).astype(np.float32)).cuda()
        K = torch.from_numpy(np.array(intrinsics_list).astype(np.float32)).cuda()
        post_T = torch.eye(4, device="cuda").unsqueeze(0)

        N_real = pcs_full.shape[1]
        if N_real > N_target:
            pc3 = pcs_full[:, :N_target, :]
        elif N_real < N_target:
            pc3 = torch.cat([
                pcs_full,
                torch.zeros(1, N_target - N_real, 3, device="cuda")
            ], dim=1)
        else:
            pc3 = pcs_full
        print(f"[trace] Loaded real sample: img={list(imgs.shape)}, "
              f"pc={list(pc3.shape)} (from {N_real}), "
              f"init_T={list(gt_T.shape)}")
        return (imgs, pc3, gt_T, post_T, K)
    except Exception as e:
        import traceback
        print(f"[trace] Could not load real data: {e}")
        traceback.print_exc()
        return None


def _build_optimization_flags():
    return (
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


# ---------------------------------------------------------------------------
# Layout: flat
# ---------------------------------------------------------------------------
def _export_flat(wrapper, cfg, export_dir):
    """Export .bin + .txt directly into export_dir with optional numpy verify."""
    from frontend_python.pytorch_parser.engine_graph_exporter import model_parserv3
    import drinfer

    verify = cfg.get("verify_export_graph", True)
    model_name = cfg.get("model_name", "bevcalib_fusion_head")
    model_version = cfg.get("model_version", "v2")

    dummy_input = _build_dummy_inputs(cfg, sparse=True)

    print(f"\n[flat] Exporting model via model_parserv3 (verify={verify})")
    engine_res, gt = model_parserv3(
        wrapper, model_name, model_version,
        dummy_input, export_dir, verbose=True,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        runtime_type=drinfer.MODEL_FLOAT,
        optimization_flag=_build_optimization_flags(),
        fixed_shape_configs=FIXED_SHAPES,
        empty_cache=True,
        verify_export_graph=verify,
        export_pmodel=False,
    )

    if not verify:
        print("[flat] Graph exported (runtime verification skipped)")
        return True

    gt_np = gt.cpu().detach().numpy()
    dr_np = engine_res[0]

    if np.isnan(dr_np).any():
        print("[flat] ERROR: DrInfer output contains NaN!")
        print(f"  GT (PyTorch):\n{gt_np}")
        print(f"  DrInfer:\n{dr_np}")
        return False

    print(f"GT (PyTorch):\n{gt_np}")
    print(f"DrInfer:\n{dr_np}")

    diff = np.abs(dr_np - gt_np)
    print(f"[flat] max diff: {diff.max():.6f}, mean: {diff.mean():.6f}")
    np.testing.assert_allclose(dr_np, gt_np, rtol=5e-2, atol=5e-2)
    print("[flat] Numerical verification PASSED")
    return True


# ---------------------------------------------------------------------------
# Layout: pmodel
# ---------------------------------------------------------------------------
def _export_pmodel(wrapper, cfg, trace_dir):
    """Export with full pmodel deployment directory layout."""
    from pmodel.utils.import_utils import model_parserv3
    from pmodel.utils.release_utils import create_layout, save_trace_model2
    from pmodel.utils import InfoTrace
    from pmodel.utils.trace_utils import flatten
    import drinfer

    model_name = cfg.get("model_name", "bevcalib_fusion_head")
    model_version = cfg.get("model_version", "v2")

    trace_data = _build_dummy_inputs(cfg, sparse=True)

    create_layout(model_name, parent_dir=trace_dir)
    model_path = os.path.join(trace_dir, model_name)
    info = InfoTrace(model_path)

    print(f"\n[pmodel] Tracing model: {model_name} v{model_version}")
    print(f"  trace_dir : {trace_dir}")
    print(f"  inputs    : {[f'{n}={list(t.shape)}' for n, t in zip(INPUT_NAMES, trace_data)]}")

    verify = cfg.get("verify_export_graph", True)

    with torch.no_grad():
        engine_res, pytorch_output = model_parserv3(
            wrapper, model_name, model_version,
            trace_data,
            export_dir=info.engine_graph,
            verbose=False,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            runtime_type=drinfer.MODEL_FLOAT,
            optimization_flag=_build_optimization_flags(),
            fixed_shape_configs=FIXED_SHAPES,
            verify_export_graph=verify,
            export_pmodel=False,
            jit_check_trace=False,
            empty_cache=True,
            export_name=info.export_name,
            log_file=os.path.join(info.engine_graph, "infer_parserv3.log"),
        )

        if pytorch_output is None:
            pytorch_output = wrapper(*trace_data)
        elif verify and engine_res is not None:
            gt_np = pytorch_output.cpu().detach().numpy()
            dr_np = engine_res[0]
            diff = np.abs(dr_np - gt_np)
            print(f"[pmodel] verify: max_diff={diff.max():.6f}, mean={diff.mean():.6f}")
            if diff.max() < 0.05:
                print("[pmodel] Numerical verification PASSED")
            else:
                print("[pmodel] WARNING: large numerical diff (may be acceptable for sparse dummy data)")

    data_flat, _ = flatten(trace_data)
    if isinstance(pytorch_output, torch.Tensor):
        out_flat = [pytorch_output.cpu().detach()]
    else:
        out_flat, _ = flatten(pytorch_output)

    save_trace_model2(
        info,
        input_datas=data_flat,
        output_datas=out_flat,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        runtime_dtype="float",
        max_input_dim_status={
            "image": [True, True, True, True],
            "point_cloud": [True, False, True],
            "init_T": [True, True, True],
            "post_cam2ego_T": [True, True, True],
            "intrinsic": [True, True, True],
        },
    )

    print(f"\n[pmodel] Trace artifacts:")
    for f in [info.graph_bin, info.graph_cfg, info.graph_txt]:
        exists = os.path.isfile(f)
        size = os.path.getsize(f) / 1024**2 if exists else 0
        print(f"  {'OK' if exists else 'MISSING':>7} {os.path.basename(f)} ({size:.1f}MB)")

    if os.path.isdir(info.input):
        print(f"  {'OK':>7} input_data/  ({len(os.listdir(info.input))} files)")
    if os.path.isdir(info.output):
        print(f"  {'OK':>7} output_data/ ({len(os.listdir(info.output))} files)")

    return model_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BEVCalib -> DrInfer converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Layouts:\n"
               "  flat    .bin+.txt only (fast, with optional numpy verify)\n"
               "  pmodel  full deployment dir (nn_param.cfg + input/output data)")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file")
    parser.add_argument("--layout", type=str, default="flat",
                        choices=["flat", "pmodel"],
                        help="Output layout: flat (default) or pmodel")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = cfg["ckpt_path"]
    default_dir = cfg.get("export_dir",
                          os.path.join(os.path.dirname(ckpt_path), "drinfer"))
    output_dir = args.output_dir or cfg.get("trace_dir", default_dir)
    os.makedirs(output_dir, exist_ok=True)

    os.environ["BEV_ZBOUND_STEP"] = str(cfg.get("bev_zbound_step", "2.0"))

    from bevcalib_inference import load_bevcalib_inference, prepare_for_drinfer_export

    img_h, img_w = cfg["img_height"], cfg["img_width"]

    print("=" * 70)
    print(f"BEVCalib -> DrInfer Conversion  [layout: {args.layout}]")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  output_dir : {output_dir}")
    print(f"  img_shape  : ({img_h}, {img_w})")
    print(f"  voxel_mode : {cfg.get('voxel_mode', 'scatter')}")
    print(f"  to_bev     : {cfg.get('to_bev_mode', 'concat')}")
    print(f"  scatter_red: {cfg.get('scatter_reduce', 'sum')}")
    print("=" * 70)

    wrapper, epoch = load_bevcalib_inference(
        ckpt_path=ckpt_path,
        device="cuda",
        img_shape=(img_h, img_w),
        rotation_only=cfg.get("rotation_only", True),
        deformable=cfg.get("deformable", False),
        bev_encoder=cfg.get("bev_encoder", True),
        use_mlp_head=cfg.get("use_mlp_head", None),
        voxel_mode=cfg.get("voxel_mode", "scatter"),
        to_bev_mode=cfg.get("to_bev_mode", "concat"),
        scatter_reduce=cfg.get("scatter_reduce", "sum"),
    )
    print(f"  epoch  : {epoch}")
    print(f"  params : {sum(p.numel() for p in wrapper.parameters()):,}")

    print("\n--- Preparing model for DrInfer export ---")
    prepare_for_drinfer_export(wrapper, img_shape=(img_h, img_w))

    t0 = time.time()

    if args.layout == "pmodel":
        result = _export_pmodel(wrapper, cfg, output_dir)
        dt = time.time() - t0
        print(f"\n{'=' * 70}")
        print(f"Export completed in {dt:.1f}s  [pmodel layout]")
        print(f"  Model path: {result}")
        print(f"\nVerify with:")
        print(f"  pmodel forward {result}")
        print(f"  python utils/drinfer_infer.py --config {args.config} --mode compare")
        print(f"{'=' * 70}")
    else:
        ok = _export_flat(wrapper, cfg, output_dir)
        dt = time.time() - t0
        if ok:
            print(f"\nExport completed in {dt:.1f}s  [flat layout] -> {output_dir}")
        else:
            print(f"\nExport FAILED in {dt:.1f}s")
            sys.exit(1)


if __name__ == "__main__":
    main()
