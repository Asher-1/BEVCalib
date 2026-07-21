#!/usr/bin/env python3
"""Run gdiag only (skip main per-frame eval) for iterative recovery comparison."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kitti-bev-calib"))

from evaluate_checkpoint import (
    _build_eval_custom_dataset,
    _build_model_from_ckpt,
    _run_generalization_diagnostics,
    make_collate_fn,
)
from torch.utils.data import DataLoader
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_width", type=int, default=960)
    p.add_argument("--target_height", type=int, default=540)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--rotation_only", type=int, default=1)
    p.add_argument("--xyz_only", type=int, default=1)
    p.add_argument("--gdiag_inject_deg", type=float, default=2.0)
    p.add_argument("--cf_bev_r_iter_steps", type=int, default=2)
    p.add_argument("--gdiag_max_batches", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--cuda_device", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class A:
        pass

    eval_args = A()
    for k, v in vars(args).items():
        setattr(eval_args, k, v)
    eval_args.generalization_diag = True
    eval_args.perturb_distribution = "uniform"
    eval_args.per_axis_prob = 0.0
    eval_args.gdiag_configs_per_pass = 3

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model, _, _ = _build_model_from_ckpt(
        eval_args, checkpoint, device, bool(args.rotation_only), quiet=False)
    ds = _build_eval_custom_dataset(args.dataset_root, eval_args)
    from torch.utils.data import random_split
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    generator = torch.Generator().manual_seed(114514)
    _, eval_ds = random_split(ds, [train_size, val_size], generator=generator)
    print(f"   Val split: {len(eval_ds)} samples (80/20, seed=114514)")

    collate_fn = make_collate_fn((args.target_width, args.target_height))
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)

    seq_boundaries = {}
    _eval_idx_to_seq = {}
    if hasattr(eval_ds, 'indices'):
        indices = eval_ds.indices
    else:
        indices = list(range(len(eval_ds)))
    for loader_idx, global_idx in enumerate(indices):
        f = ds.all_files[global_idx]
        seq = f.split("/")[0]
        _eval_idx_to_seq[loader_idx] = seq
        if seq not in seq_boundaries:
            seq_boundaries[seq] = [loader_idx, loader_idx]
        else:
            seq_boundaries[seq][1] = loader_idx

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = _run_generalization_diagnostics(
        model=model,
        val_loader=loader,
        args=eval_args,
        device=device,
        eval_dir=args.output_dir,
        rotation_only=bool(args.rotation_only),
        _eval_idx_to_seq=_eval_idx_to_seq,
        seq_boundaries=seq_boundaries,
        use_dp=False,
    )

    comp = results.get("composite", {}).get("raw", {})
    fi = results.get("fixed_inject", {}).get("inject", {})
    print("\n=== GDIAG SUMMARY ===")
    print(f"  cf_bev_r_iter_steps: {args.cf_bev_r_iter_steps}")
    print(f"  Zero-Drift rot: {results.get('zero_drift', {}).get('rot_mean', -1):.4f}°")
    print(f"  Fixed-Inject recovery: {fi.get('mean_recovery_pct', -1):.1f}%")
    print(f"  Genuine recovery: {comp.get('genuine_recovery_pct', -1):.1f}%")
    print(f"  Pred Independence: {results.get('prediction_independence', {}).get('overall_independence', -1):.3f}")
    print(f"  GS_medw: {results.get('composite', {}).get('GS_medw', -1):.4f}")


if __name__ == "__main__":
    main()
