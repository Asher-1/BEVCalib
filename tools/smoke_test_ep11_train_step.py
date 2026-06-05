#!/usr/bin/env python3
"""Fast regression test for V41 ep11 train step (consistency + jacobian supervision).

Simulates epoch=10 (log Epoch [11/200]): main forward + consistency, main backward,
then jacobian probe + backward (same order as train_kitti.py), optimizer step,
second forward (DDP reducer sanity).

Usage:
  python tools/smoke_test_ep11_train_step.py              # all variants, 1 GPU
  python tools/smoke_test_ep11_train_step.py --variant a
  torchrun --standalone --nproc_per_node=2 tools/smoke_test_ep11_train_step.py --ddp
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
KITTI = ROOT / "kitti-bev-calib"
sys.path.insert(0, str(KITTI))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("USE_DRCV_BACKEND", "0")
os.environ.setdefault("PROJFUSION_ROOT", str(ROOT.parent / "ProjFusion"))

from hybrid_triple_calib import build_calib_model
from tools import generate_single_perturbation_from_T
from train_kitti import _compute_jacobian_supervision_loss

VARIANT_CFG = {
    "a": "configs/v41_gmp_a_v32_baseline.yaml",
    "b": "configs/v41_gmp_b_v32_jacloss.yaml",
    "c": "configs/v41_gmp_c_v32_jacloss_match.yaml",
}


def _load_args(variant: str) -> SimpleNamespace:
    cfg_path = ROOT / VARIANT_CFG[variant]
    with open(cfg_path) as f:
        doc = yaml.safe_load(f)
    params = dict(doc.get("defaults", {}).get("params", {}))
    params["fusion_backend"] = "geo_match_proj"
    params["axis_weights"] = params.get("axis_weights", "1.0,3.0,1.0")
    if isinstance(params.get("projfusion_image_hw"), list):
        params["projfusion_image_hw"] = tuple(params["projfusion_image_hw"])
    return SimpleNamespace(**params)


def _init_dist(use_ddp: bool):
    if not use_ddp:
        return False, 0, 0, 1
    if "RANK" not in os.environ:
        raise RuntimeError("Pass --ddp only under torchrun (RANK not set)")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, dist.get_world_size()


def _load_batch(args, device, batch_size=4):
    from custom_dataset import CustomDataset

    target_size = (640, 360)
    ds = CustomDataset(
        "/mnt/drtraining/user/dahailu/data/bevcalib/all_training_data",
        target_size=target_size,
        max_frames_per_seq=32,
        pose_aware_sampling=True,
    )

    def collate(batch):
        batch = [x for x in batch if x is not None]
        if not batch:
            return None
        imgs = [item[0] for item in batch]
        gt_T = [item[2] for item in batch]
        intrinsics = [item[3] for item in batch]
        max_pts = max(item[1].shape[0] for item in batch)
        pcs, masks = [], []
        for item in batch:
            pc = item[1][:, :3]
            n = pc.shape[0]
            pad = max_pts - n
            if pad > 0:
                pc = np.concatenate([pc, np.full((pad, 3), 999999, dtype=pc.dtype)])
            pcs.append(pc)
            masks.append(np.concatenate([np.ones(n), np.zeros(pad)]))
        return imgs, pcs, masks, gt_T, intrinsics

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    batch = next(iter(loader))
    imgs, pcs, masks, gt_T, intrinsics = batch
    resize_imgs = torch.from_numpy(np.stack([np.array(im) for im in imgs])).permute(0, 3, 1, 2).float().to(device)
    pcs_t = torch.from_numpy(np.array(pcs)).float().to(device)
    masks_t = torch.from_numpy(np.array(masks)).float().to(device)
    gt_np = np.array(gt_T, dtype=np.float32)
    gt_t = torch.from_numpy(gt_np).to(device)
    init_np, _, _ = generate_single_perturbation_from_T(
        gt_np, angle_range_deg=5.0, trans_range=0.15, rotation_only=True,
        distribution="truncated_normal", per_axis_prob=0.3)
    init_alt_np, _, _ = generate_single_perturbation_from_T(
        gt_np, angle_range_deg=5.0, trans_range=0.15, rotation_only=True,
        distribution="truncated_normal", per_axis_prob=0.3)
    init_t = torch.from_numpy(init_np.astype(np.float32)).to(device)
    init_alt_t = torch.from_numpy(init_alt_np.astype(np.float32)).to(device)
    post = torch.eye(4, device=device).unsqueeze(0).expand(gt_t.shape[0], -1, -1)
    K = torch.from_numpy(np.array(intrinsics, dtype=np.float32)).to(device)
    return resize_imgs, pcs_t, masks_t, gt_t, init_t, init_alt_t, post, K, init_np


def _ep11_train_step(model, raw_model, batch, args, epoch=10):
    device = batch[0].device
    resize_imgs, pcs_t, masks_t, gt_t, init_t, init_alt_t, post, K, init_np = batch
    raw_model.set_training_epoch(epoch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    scaler = GradScaler(enabled=False)
    use_amp = not bool(getattr(args, "no_amp", 0))
    amp_dtype = torch.float32

    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp, dtype=amp_dtype):
        B = resize_imgs.shape[0]
        if epoch >= getattr(args, "consistency_loss_start_epoch", 10):
            imgs_fwd = torch.cat([resize_imgs, resize_imgs], dim=0)
            pcs_fwd = torch.cat([pcs_t, pcs_t], dim=0)
            gt_fwd = torch.cat([gt_t, gt_t], dim=0)
            init_fwd = torch.cat([init_t, init_alt_t], dim=0)
            post_fwd = torch.cat([post, post], dim=0)
            K_fwd = torch.cat([K, K], dim=0)
            masks_fwd = torch.cat([masks_t, masks_t], dim=0) if masks_t is not None else None
            T_all, _, loss = model(
                imgs_fwd, pcs_fwd, gt_fwd, init_fwd, post_fwd, K_fwd,
                masks=masks_fwd, out_init_loss=False)
            T_pred = T_all[:B]
            T_alt = T_all[B:].detach()
        else:
            T_pred, _, loss = model(
                resize_imgs, pcs_t, gt_t, init_t, post, K,
                masks=masks_t, out_init_loss=False)
            T_alt = None
        total_loss = loss["total_loss"]

        if T_alt is not None:
            R_diff = torch.bmm(T_pred[:, :3, :3], T_alt[:, :3, :3].transpose(1, 2))
            trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
            cons = (1.0 - trace / 3.0).mean()
            w = float(getattr(args, "consistency_loss_weight", 0.5))
            total_loss = total_loss + w * cons

    scaler.scale(total_loss).backward()

    jac_out = None
    if epoch >= getattr(args, "jacobian_loss_start_epoch", 10):
        jac_out = _compute_jacobian_supervision_loss(
            model, resize_imgs, pcs_t, gt_t, init_np, post, K, masks_t,
            float(getattr(args, "jacobian_loss_probe_deg", 3.0)),
            use_amp, amp_dtype, T_pred_center=T_pred.detach())
        assert jac_out is not None, "jacobian supervision returned None"
        jac_sup, j_est = jac_out
        jac_term = float(args.jacobian_loss_weight) * jac_sup
        scaler.scale(jac_term).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
    scaler.step(optimizer)
    scaler.update()

    logged_total = float(total_loss.detach())
    if jac_out is not None:
        logged_total += float((float(args.jacobian_loss_weight) * jac_out[0]).detach())

    if jac_out is not None:
        j_est = jac_out[1]
    else:
        j_est = torch.tensor(0.0)

    # Second forward mimics ep11 batch 1 — DDP reducer fails here if ep11 step 0 broken.
    with autocast(enabled=use_amp, dtype=amp_dtype):
        _, _, loss2 = model(
            resize_imgs, pcs_t, gt_t, init_t, post, K,
            masks=masks_t, out_init_loss=False)
    return logged_total, float(loss2["total_loss"].detach()), float(j_est.detach() if torch.is_tensor(j_est) else j_est)


def run_variant(variant: str, use_ddp: bool, rank: int, local_rank: int) -> bool:
    args = _load_args(variant)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    model = build_calib_model(args, device, img_shape=(360, 640), rotation_only=True, is_main=is_main, tprint=print)
    model.train()
    if use_ddp:
        model = DDP(
            model, device_ids=[local_rank],
            find_unused_parameters=True,
        )
    raw_model = model.module if use_ddp else model

    batch = _load_batch(args, device, batch_size=4)
    try:
        total, total2, j_est = _ep11_train_step(model, raw_model, batch, args, epoch=10)
    except RuntimeError as e:
        if is_main:
            print(f"FAIL [{variant}] ep11 train step: {e}")
        return False

    if is_main:
        print(f"PASS [{variant}] ep11 step: loss={total:.4f} next_fwd={total2:.4f} j_est={j_est:.4f}"
              f"{' DDP×'+str(dist.get_world_size()) if use_ddp else ''}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--ddp", action="store_true", help="Requires torchrun")
    cli = parser.parse_args()

    use_ddp, rank, local_rank, world = _init_dist(cli.ddp)
    variants = ["a", "b", "c"] if cli.variant == "all" else [cli.variant]
    ok = True
    for v in variants:
        if not run_variant(v, use_ddp, rank, local_rank):
            ok = False
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0 and not ok:
        sys.exit(1)
    if rank == 0:
        print("All ep11 smoke tests passed.")


if __name__ == "__main__":
    main()
