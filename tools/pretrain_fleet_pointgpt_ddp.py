#!/usr/bin/env python3
"""Fleet-domain PointGPT pretrain with DDP (8-GPU version).

Optimized for fast domain adaptation on bevcalib/all_training_data.
Launch with: torchrun --nproc_per_node=8 pretrain_fleet_pointgpt_ddp.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pykitti
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJFUSION_ROOT = os.environ.get(
    'PROJFUSION_ROOT', '/mnt/drtraining/user/dahailu/code/ProjFusion')
sys.path.insert(0, PROJFUSION_ROOT)
os.chdir(PROJFUSION_ROOT)

from dataset import KITTIFilter, Resampler  # noqa: E402
from models.pointgpt import PointTransformer  # noqa: F401,E402
from models.pointgpt.config import get_config  # noqa: E402
from models.pointgpt.build import build_model_from_cfg  # noqa: E402


class FleetVeloOnlyDataset(Dataset):
    """LiDAR-only fleet dataset for PointGPT self-supervised pretrain."""

    def __init__(self, basedir, seqs, meta_json='data_len.json',
                 skip_frame=2, skip_point=1, min_dist=0.1, npoints=8192):
        with open(os.path.join(basedir, meta_json), 'r') as f:
            dict_len = json.load(f)
        self.kitti_datalist = []
        self.sep = []
        for seq in seqs:
            frames = list(range(0, dict_len[seq], skip_frame))
            self.kitti_datalist.append(pykitti.odometry(basedir, seq, frames=frames))
            self.sep.append(len(frames))
        self.sumsep = np.cumsum(self.sep)
        self.pcd_tran = KITTIFilter(None, min_dist=min_dist, skip_point=skip_point)
        self.resample = Resampler(npoints)
        self.tensor_tran = lambda x: torch.from_numpy(x).to(torch.float32)

    def __len__(self):
        return int(self.sumsep[-1])

    def __getitem__(self, index):
        group_id = np.digitize(index, self.sumsep, right=False).item()
        sub_idx = index - self.sumsep[group_id - 1] if group_id > 0 else index
        data = self.kitti_datalist[group_id]
        pcd = data.get_velo(sub_idx)[:, :3]
        pcd = self.pcd_tran(pcd)
        pcd = self.resample(pcd)
        return {'pcd': self.tensor_tran(pcd)}


def _build_fleet_dataset(cfg_path, split='train', skip_frame=2):
    cfg = get_config(cfg_path)
    ds_cfg = cfg['dataset'][split]
    base_cfg = ds_cfg['_base_']
    others = ds_cfg.get('others', {})
    seqs = others.get('SEQS', base_cfg.SEQS)
    max_depth = float(others.get('MAX_DEPTH', base_cfg.MAX_DEPTH))
    npoints = int(others.get('npoints', 8192))
    basedir = base_cfg.BASEDIR
    meta_json = base_cfg.get('META_JSON', 'data_len.json')
    skip_frame_override = int(others.get('SKIP_FRAME', skip_frame))
    dataset = FleetVeloOnlyDataset(
        basedir=basedir,
        seqs=seqs,
        meta_json=meta_json,
        skip_frame=skip_frame_override,
        skip_point=int(base_cfg.get('SKIP_POINT', 1)),
        min_dist=float(base_cfg.get('MIN_DIST', 0.1)),
        npoints=npoints,
    )
    return dataset, max_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cfg/pointgpt/pretrain_fleet_8gpu.yaml')
    parser.add_argument('--init_ckpt',
                        default='pretrained/kitti_pointgpt_tiny.pth')
    parser.add_argument('--output',
                        default='pretrained/fleet_pointgpt_tiny_8gpu.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Per-GPU batch size')
    parser.add_argument('--lr', type=float, default=2.8e-4)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # DDP init
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f'[DDP] world_size={world_size}, config={args.config}')
        print(f'[DDP] global_bs={args.batch_size * world_size}, lr={args.lr}')

    # Model
    cfg = get_config(args.config)
    model = build_model_from_cfg(cfg.model).to(device)
    
    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        model.load_model_from_ckpt(args.init_ckpt)
        if rank == 0:
            print(f'Init from {args.init_ckpt}')
    
    # GPT_extractor(pretrained=True) omits finetune heads
    td, cls_dim = model.trans_dim, model.cls_dim
    if not hasattr(model.blocks, 'cls_norm'):
        model.blocks.cls_norm = torch.nn.LayerNorm(td).to(device)
    if not hasattr(model.blocks, 'cls_head_finetune'):
        model.blocks.cls_head_finetune = torch.nn.Sequential(
            torch.nn.Linear(td * 2, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, cls_dim),
        ).to(device)

    model = DDP(model, device_ids=[args.local_rank], 
                output_device=args.local_rank,
                find_unused_parameters=True)

    # Dataset
    dataset, max_depth = _build_fleet_dataset(args.config, 'train', skip_frame=2)
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    def collate(batch):
        pcds = torch.stack([b['pcd'] for b in batch])
        return pcds

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, collate_fn=collate, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=4, collate_fn=collate, pin_memory=True,
    )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
        log_file = args.output.replace('.pth', '_train.log')
        log_f = open(log_file, 'w', buffering=1)
        print(f'[DDP] Logging to {log_file}')

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        
        train_loss = 0.0
        n_batches = 0
        
        for pcds in train_loader:
            pcds = pcds.to(device) / max_depth
            opt.zero_grad(set_to_none=True)
            _, loss = model(pcds)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        
        # Reduce train loss
        train_loss_tensor = torch.tensor(train_loss / max(n_batches, 1), device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        train_loss_avg = train_loss_tensor.item()

        # Val
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for pcds in val_loader:
                pcds = pcds.to(device) / max_depth
                _, loss = model(pcds)
                val_loss += loss.item()
                vn += 1
        
        # Reduce val loss
        val_loss_tensor = torch.tensor(val_loss / max(vn, 1), device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss_avg = val_loss_tensor.item()

        elapsed = time.time() - t0
        
        if rank == 0:
            msg = (f'Ep {epoch}/{args.epochs} train={train_loss_avg:.4f} '
                   f'val={val_loss_avg:.4f} lr={scheduler.get_last_lr()[0]:.2e} '
                   f'({elapsed:.0f}s)')
            print(msg)
            log_f.write(msg + '\n')

            if val_loss_avg < best_val:
                best_val = val_loss_avg
                ckpt = {
                    'base_model': model.module.state_dict(),
                    'epoch': epoch,
                    'metrics': {'val_loss': val_loss_avg},
                    'best_metrics': {'val_loss': val_loss_avg},
                }
                torch.save(ckpt, args.output)
                print(f'  saved {args.output} (val={val_loss_avg:.4f})')
                log_f.write(f'  saved {args.output} (val={val_loss_avg:.4f})\n')
        
        scheduler.step()

    if rank == 0:
        print(f'Done. best_val={best_val:.4f} -> {args.output}')
        log_f.close()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
