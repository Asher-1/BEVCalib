#!/usr/bin/env python3
"""Fleet-domain PointGPT (PointTransformer) finetune on all_training_data.

Uses ProjFusion PointTransformer + BaseKITTIDataset. Output ckpt is compatible
with BEVCalib native_cross_pointgpt_ckpt (base_model format).

Init from KITTI PointGPT by default; trains self-supervised chamfer reconstruction
on fleet forward-sparse LiDAR (8192 pts, max_depth=50m).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pykitti
import torch
from torch.utils.data import DataLoader, Dataset

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
    dataset = FleetVeloOnlyDataset(
        basedir=basedir,
        seqs=seqs,
        meta_json=meta_json,
        skip_frame=skip_frame,
        skip_point=int(base_cfg.get('SKIP_POINT', 1)),
        min_dist=float(base_cfg.get('MIN_DIST', 0.1)),
        npoints=npoints,
    )
    return dataset, max_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cfg/pointgpt/finetune_fleet_tiny.yaml')
    parser.add_argument('--init_ckpt',
                        default='pretrained/kitti_pointgpt_tiny.pth')
    parser.add_argument('--output',
                        default='pretrained/fleet_pointgpt_tiny.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='0 = full epoch')
    parser.add_argument('--skip_frame', type=int, default=2)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = get_config(args.config)
    model = build_model_from_cfg(cfg.model).to(device)
    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        model.load_model_from_ckpt(args.init_ckpt)
        print(f'Init from {args.init_ckpt}')
    # GPT_extractor(pretrained=True) omits finetune heads; forward() still calls them.
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
    model.train()

    dataset, max_depth = _build_fleet_dataset(args.config, 'train', args.skip_frame)
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    def collate(batch):
        pcds = torch.stack([b['pcd'] for b in batch])
        return pcds

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate, drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_val = float('inf')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches = 0
        max_b = args.max_batches if args.max_batches > 0 else len(train_loader)
        for i, pcds in enumerate(train_loader):
            if i >= max_b:
                break
            pcds = pcds.to(device) / max_depth
            opt.zero_grad(set_to_none=True)
            _, loss = model(pcds)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for pcds in val_loader:
                pcds = pcds.to(device) / max_depth
                _, loss = model(pcds)
                val_loss += loss.item()
                vn += 1
        val_loss /= max(vn, 1)

        print(f'Ep {epoch}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f} '
              f'({time.time()-t0:.0f}s)')

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                'base_model': model.state_dict(),
                'epoch': epoch,
                'metrics': {'val_loss': val_loss},
                'best_metrics': {'val_loss': val_loss},
            }
            torch.save(ckpt, args.output)
            print(f'  saved {args.output} (val={val_loss:.4f})')

    print(f'Done. best_val={best_val:.4f} -> {args.output}')


if __name__ == '__main__':
    main()
