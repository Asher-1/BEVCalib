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


def bev_pool_custom(feats, coords, B, D, H, W):
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
    """BEV pool via drcv bev_pool_v2 (training backend, NOT JIT-traceable).

    Accepts unfiltered feats/coords; out-of-bounds are filtered internally.
    Uses ``QuickCumsumCuda`` (autograd.Function) for gradient support.
    For DrInfer export use ``bev_pool_scatter`` instead.

    coords layout: (N, 4) = [x_voxel, y_voxel, z_voxel, batch_idx]
    Output: (B, C, D, H, W)
    """
    from drcv.ops.bev_pool_v2 import bev_pool_v2_func
    assert feats.shape[0] == coords.shape[0]
    C = feats.shape[1]
    D_int, H_int, W_int = int(D), int(H), int(W)

    kept = (
        (coords[:, 0] >= 0) & (coords[:, 0] < H_int)
        & (coords[:, 1] >= 0) & (coords[:, 1] < W_int)
        & (coords[:, 2] >= 0) & (coords[:, 2] < D_int)
        & (coords[:, 3] >= 0) & (coords[:, 3] < B)
    )
    feats = feats[kept]
    coords = coords[kept]
    N = feats.shape[0]
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

def bev_pool_scatter(feats, coords, B, D, H, W):
    """JIT-traceable BEV pool using scatter_add (no custom CUDA extension).

    Accepts UNFILTERED feats/coords (the caller should NOT boolean-index
    with `kept`).  Invalid coordinates are handled via masking: their
    features are zeroed so scatter_add is a no-op for those positions.

    coords layout: (N, 4) = [x_voxel, y_voxel, z_voxel, batch_idx]  (long)
    Output: (B, C, D, H, W)
    """
    C = feats.shape[1]
    D_int, H_int, W_int = int(D), int(H), int(W)
    total = B * D_int * H_int * W_int

    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < H_int)
        & (coords[:, 1] >= 0) & (coords[:, 1] < W_int)
        & (coords[:, 2] >= 0) & (coords[:, 2] < D_int)
        & (coords[:, 3] >= 0) & (coords[:, 3] < B)
    ).unsqueeze(-1).float()

    feats_masked = feats * valid

    flat_idx = (
        coords[:, 3] * (D_int * H_int * W_int)
        + coords[:, 2] * (H_int * W_int)
        + coords[:, 0] * W_int
        + coords[:, 1]
    ).clamp(0, total - 1).long()

    out_flat = feats.new_zeros(total, C)
    out_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, C), feats_masked)

    return out_flat.view(B, D_int, H_int, W_int, C).permute(0, 4, 1, 2, 3).contiguous()


_USE_SCATTER = True

def set_bev_pool_backend(use_scatter: bool):
    """Switch bev_pool between scatter (traceable) and drcv (faster CUDA)."""
    global _USE_SCATTER
    _USE_SCATTER = use_scatter


def bev_pool(feats, coords, B, D, H, W):
    if _USE_SCATTER:
        return bev_pool_scatter(feats, coords, B, D, H, W)
    return bev_pool_drcv(feats, coords, B, D, H, W)

if __name__ == "__main__":
    torch.manual_seed(42)

    def _run_test(N, C, B, D, H, W, include_oob=False):
        print("=" * 60)
        tag = f"N={N}, C={C}, B={B}, D={D}, H={H}, W={W}"
        if include_oob:
            tag += " (with out-of-bounds coords)"
        print(f"Test: {tag}")
        print("=" * 60)

        feats = torch.randn(N, C, device="cuda")
        if include_oob:
            coords = torch.stack([
                torch.randint(-5, H + 5, (N,)),
                torch.randint(-5, W + 5, (N,)),
                torch.randint(-2, D + 2, (N,)),
                torch.randint(0, B, (N,)),
            ], dim=1).to("cuda")
        else:
            coords = torch.stack([
                torch.randint(0, H, (N,)),
                torch.randint(0, W, (N,)),
                torch.randint(0, D, (N,)),
                torch.randint(0, B, (N,)),
            ], dim=1).to("cuda")

        kept = (
            (coords[:, 0] >= 0) & (coords[:, 0] < H)
            & (coords[:, 1] >= 0) & (coords[:, 1] < W)
            & (coords[:, 2] >= 0) & (coords[:, 2] < D)
        )
        feats_f, coords_f = feats[kept], coords[kept]
        n_valid = int(kept.sum())

        x_drcv = bev_pool_drcv(feats_f, coords_f, B, D, H, W)
        print(f"[drcv]    shape={x_drcv.shape}, nonzero={int((x_drcv!=0).sum()):>8}, "
              f"nan={torch.isnan(x_drcv).any().item()}")

        x_custom = bev_pool_custom(feats_f, coords_f, B, D, H, W)
        print(f"[custom]  shape={x_custom.shape}, nonzero={int((x_custom!=0).sum()):>8}, "
              f"nan={torch.isnan(x_custom).any().item()}")

        x_scatter = bev_pool_scatter(feats, coords, B, D, H, W)
        print(f"[scatter] shape={x_scatter.shape}, nonzero={int((x_scatter!=0).sum()):>8}, "
              f"nan={torch.isnan(x_scatter).any().item()}  "
              f"(unfiltered, {n_valid}/{N} valid)")

        d1 = (x_drcv - x_custom).abs()
        d2 = (x_drcv - x_scatter).abs()
        print(f"\n[drcv vs custom]  max={d1.max().item():.6e}, mean={d1.mean().item():.6e}  "
              f"{'MATCH' if d1.max().item() < 1e-4 else 'MISMATCH'}")
        print(f"[drcv vs scatter] max={d2.max().item():.6e}, mean={d2.mean().item():.6e}  "
              f"{'MATCH' if d2.max().item() < 1e-4 else 'MISMATCH'}")
        print()

    _run_test(200, 128, 2, 5, 10, 10)
    _run_test(5000, 128, 1, 5, 100, 100)
    _run_test(50000, 128, 1, 5, 100, 100, include_oob=True)
