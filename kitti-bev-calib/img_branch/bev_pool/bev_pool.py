import torch

try:
    from . import bev_pool_ext
except:
    import bev_pool_ext

__all__ = ["bev_pool"]


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCudaV2(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, geom_feats, B, D, H, W, interval_starts, interval_lengths):
        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None, None


def bev_pool_ext(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]
    coords = coords.int()

    kept = torch.cat([
        torch.ones(1, device=feats.device, dtype=torch.bool),
        ranks[1:] != ranks[:-1]
    ])
    interval_starts = torch.where(kept)[0].int()
    ends = torch.cat([interval_starts[1:], torch.tensor([feats.shape[0]], device=interval_starts.device, dtype=interval_starts.dtype)])
    interval_lengths = ends - interval_starts

    x = QuickCumsumCudaV2.apply(feats, coords, B, D, H, W, interval_starts, interval_lengths)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def bev_pool_drcv(feats, coords, B, D, H, W):
    """BEV pool via drcv bev_pool_v2.

    The drcv kernel computes:  out[ranks_bev[i]] += depth[ranks_depth[i]] * feat[ranks_feat[i]]
    We set depth=1 for every point to bypass depth weighting and directly scatter feats.

    coords layout: (N, 4) = [x_voxel, y_voxel, z_voxel, batch_idx]
    Output shape from bev_pool_v2_func: (B, C, D, H, W)  (permuted inside bev_pool_v2_func).
    ranks_bev = flat index into (B, D, H, W):
      batch*(D*H*W) + z*(H*W) + x*W + y
      = coords[:,3]*(D*H*W) + coords[:,2]*(H*W) + coords[:,0]*W + coords[:,1]
    """
    from drcv.ops.bev_pool_v2 import bev_pool_v2_func
    assert feats.shape[0] == coords.shape[0]
    N = feats.shape[0]
    C = feats.shape[1]
    D_int, H_int, W_int = int(D), int(H), int(W)
    if N == 0:
        return feats.new_zeros((B, C, D_int, H_int, W_int))

    ranks_bev = (
        coords[:, 3] * (D_int * H_int * W_int)
        + coords[:, 2] * (H_int * W_int)
        + coords[:, 0] * W_int
        + coords[:, 1]
    ).int()

    indices = ranks_bev.argsort()
    ranks_bev = ranks_bev[indices]
    feats_sorted = feats[indices]

    kept = torch.cat([
        torch.ones(1, device=feats.device, dtype=torch.bool),
        ranks_bev[1:] != ranks_bev[:-1]
    ])
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = N - interval_starts[-1]

    depth = torch.ones((1, 1, 1, N, 1), device=feats.device, dtype=feats.dtype)
    feat_5d = feats_sorted.view(1, 1, 1, N, C)
    ranks_depth = torch.zeros(N, device=feats.device, dtype=torch.int32)
    ranks_feat = torch.arange(N, device=feats.device, dtype=torch.int32)
    bev_feat_shape = (B, D_int, H_int, W_int, C)

    x = bev_pool_v2_func(
        depth, feat_5d, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths,
    )
    return x

def bev_pool(feats, coords, B, D, H, W):
    return bev_pool_drcv(feats, coords, B, D, H, W)

if __name__ == "__main__":
    torch.manual_seed(42)
    N, C = 200, 128
    B, D, H, W = 2, 5, 10, 10

    feats = torch.randn(N, C, device="cuda")
    coords = torch.stack([
        torch.randint(0, H, (N,)),   # x_voxel in [0, H)
        torch.randint(0, W, (N,)),   # y_voxel in [0, W)
        torch.randint(0, D, (N,)),   # z_voxel in [0, D)
        torch.randint(0, B, (N,)),   # batch_idx in [0, B)
    ], dim=1).to("cuda")

    print("=" * 60)
    print(f"Test: N={N}, C={C}, B={B}, D={D}, H={H}, W={W}")
    print("=" * 60)

    x_drcv = bev_pool_drcv(feats, coords, B, D, H, W)
    print(f"\n[drcv]  shape={x_drcv.shape}, non-zero={int((x_drcv != 0).sum())}, nan={torch.isnan(x_drcv).any().item()}")

    x_ext = bev_pool_ext(feats, coords, B, D, H, W)
    print(f"[ext]   shape={x_ext.shape}, non-zero={int((x_ext != 0).sum())}, nan={torch.isnan(x_ext).any().item()}")

    diff = (x_drcv - x_ext).abs()
    print(f"\n[compare] max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")
    print("MATCH" if diff.max().item() < 1e-5 else "MISMATCH")
