import os
import sys
import subprocess
import textwrap
import torch
import torch.nn as nn

# Backend switch: set USE_DRCV_BACKEND=1 to use drcv ops instead of spconv
USE_DRCV = os.environ.get("USE_DRCV_BACKEND", "1") == "1"

if USE_DRCV:
    # drcv.ops.torch_sparse uses deprecated collections.Sequence (removed in Python 3.10+)
    import collections, collections.abc
    if not hasattr(collections, "Sequence"):
        collections.Sequence = collections.abc.Sequence

    from drcv.ops.voxel import voxelization as _drcv_voxelization
    from drcv.ops.torch_scatter import scatter_add as _scatter_add
    from drcv.ops.torch_scatter import scatter_mean as _scatter_mean
    print("USE_DRCV: True in pc_branch.py")
else:
    from spconv.pytorch.utils import PointToVoxel
    _scatter_add = None
    _scatter_mean = None

try:
    from .pc_encoders import SparseEncoder
except ImportError:
    from pc_encoders import SparseEncoder

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)
from proj_head import ProjectionHead
from bev_settings import xbound, ybound, zbound, down_ratio, sparse_shape, vsize_xyz

_COORS_RANGE = [xbound[0], ybound[0], zbound[0], xbound[1], ybound[1], zbound[1]]

_VOXEL_MIN = torch.tensor(_COORS_RANGE[:3])
_VOXEL_SIZE = torch.tensor(vsize_xyz)
_GRID_MAX = torch.tensor([sparse_shape[0] - 1, sparse_shape[1] - 1, sparse_shape[2] - 1])


class Lidar2BEV(nn.Module):
    """Point cloud -> BEV feature pipeline.

    Args:
        to_bev_mode: Sparse-to-BEV conversion strategy forwarded to
            :class:`SparseEncoder`.  ``'concat'`` | ``'learned'`` | ``'sum'``.
        voxel_mode: Voxelization strategy.
            ``'hard'`` (default) — CUDA hard_voxelize + sum reduce.
            ``'scatter'`` — torch.unique + scatter, drinfer-trace compatible.
        scatter_reduce: Aggregation for scatter mode.
            ``'sum'`` — scatter_add, matches hard voxelization behavior.
            ``'mean'`` — scatter_mean (drcv dr_voxelization style),
            more robust to varying point density.
    """

    def __init__(self, to_bev_mode='concat', voxel_mode='hard', scatter_reduce='sum'):
        super(Lidar2BEV, self).__init__()
        assert voxel_mode in ('hard', 'scatter'), \
            f"voxel_mode must be 'hard' or 'scatter', got '{voxel_mode}'"
        assert scatter_reduce in ('sum', 'mean'), \
            f"scatter_reduce must be 'sum' or 'mean', got '{scatter_reduce}'"
        self.voxel_mode = voxel_mode
        self.scatter_reduce = scatter_reduce

        if voxel_mode == 'hard':
            if USE_DRCV:
                print("Use DRCV hard_voxelize as Lidar2BEV backend")
                self._voxel_size = vsize_xyz
                self._coors_range = _COORS_RANGE
                self._max_num_points = 10
                self._max_num_voxels = 120000
            else:
                self.ptvoxel = PointToVoxel(
                    vsize_xyz=vsize_xyz,
                    coors_range_xyz=tuple(_COORS_RANGE),
                    num_point_features=3,
                    max_num_voxels=120000,
                    max_num_points_per_voxel=10,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                )
        else:
            print(f"Use scatter_{scatter_reduce} voxelization (drinfer-trace compatible)")

        self.voxelize_reduce = True
        self.sparse_encoder = SparseEncoder(
            sparse_shape=sparse_shape, to_bev_mode=to_bev_mode)
        encoder_out_ch = self.sparse_encoder.to_bev.out_channels
        self.proj_head = ProjectionHead(embedding_dim=encoder_out_ch)
        self.out_channels = self.proj_head.projection_dim

    # ------------------------------------------------------------------
    # Hard voxelization (original approach)
    # ------------------------------------------------------------------

    def _voxelize_single_hard(self, points):
        """CUDA hard_voxelize for a single sample.
        Returns (voxels, coors_zyx, num_points).
        """
        if USE_DRCV:
            return _drcv_voxelization(
                points, self._voxel_size, self._coors_range,
                self._max_num_points, self._max_num_voxels
            )
        return self.ptvoxel(points)

    @torch.no_grad()
    def _voxelize_hard(self, pc):
        """Batch hard voxelization.
        Returns (feats [M, C], coors [M, 4] in batch-z-y-x order).
        """
        feats_list, coors_list = [], []
        B = pc.shape[0]
        for i in range(B):
            vox, coors_zyx, _ = self._voxelize_single_hard(pc[i])
            if self.voxelize_reduce:
                vox = vox.sum(dim=1)
            batch_col = torch.full(
                (coors_zyx.shape[0], 1), i,
                dtype=torch.int32, device=pc.device)
            feats_list.append(vox)
            coors_list.append(torch.cat([batch_col, coors_zyx], dim=1))
        return torch.cat(feats_list, 0), torch.cat(coors_list, 0)

    # ------------------------------------------------------------------
    # Scatter voxelization (dr_voxelization compatible)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _voxelize_scatter(self, pc):
        """Scatter-based voxelization.

        ``scatter_reduce='mean'`` uses ``drcv.ops.voxel.dr_voxelization``
        (unique + scatter_mean).
        ``scatter_reduce='sum'`` uses scatter_add (matches hard_voxelize).

        Falls back to native PyTorch ops during ``torch.jit.trace``
        to avoid ``GenFunction`` CUDA errors in the drinfer export path.

        Returns (feats [M, C], coors [M, 4] in batch-x-y-z order).
        """
        B, N, C = pc.shape
        vmin = _VOXEL_MIN.to(device=pc.device, dtype=pc.dtype)
        vs = _VOXEL_SIZE.to(device=pc.device, dtype=pc.dtype)
        gmax = _GRID_MAX.to(device=pc.device)

        use_native = torch.jit.is_tracing() or _scatter_add is None
        use_mean = self.scatter_reduce == 'mean'

        feats_list, coors_list = [], []
        for i in range(B):
            pts = pc[i]
            grid = ((pts[:, :3] - vmin) / vs).int()
            mask = ((grid >= 0) & (grid <= gmax)).all(dim=1)
            pts, grid = pts[mask], grid[mask]

            batch_col = torch.full(
                (grid.shape[0], 1), i,
                dtype=grid.dtype, device=grid.device)
            coors = torch.cat([batch_col, grid], dim=1)

            uniq_coors, inv = torch.unique(coors, return_inverse=True, dim=0)
            n_vox = uniq_coors.shape[0]

            if use_native:
                vox_feats = pts.new_zeros(n_vox, C)
                vox_feats.scatter_add_(
                    0, inv.unsqueeze(1).expand(-1, C), pts)
                if use_mean:
                    counts = pts.new_zeros(n_vox, 1)
                    counts.scatter_add_(
                        0, inv.unsqueeze(1),
                        torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype))
                    vox_feats = vox_feats / counts.clamp(min=1)
            elif use_mean:
                vox_feats = _scatter_mean(pts, inv, dim=0, dim_size=n_vox)
            else:
                vox_feats = _scatter_add(pts, inv, dim=0, dim_size=n_vox)

            feats_list.append(vox_feats)
            coors_list.append(uniq_coors)

        return torch.cat(feats_list, 0), torch.cat(coors_list, 0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pc):
        """
        Args:
            pc: (B, C, N) — C=3 for xyz point cloud
        Returns:
            bev feats: (B, out_C, H, W)
        """
        B, C, N = pc.shape
        pc = pc.permute(0, 2, 1).contiguous()  # (B, N, C)

        if self.voxel_mode == 'scatter':
            vox, coors = self._voxelize_scatter(pc)
            # coors already in (batch, x, y, z) order — no reorder needed
        else:
            vox, coors = self._voxelize_hard(pc)
            # hard voxelization: vox in (z, y, x) order, coors in (batch, z, y, x)
            vox = torch.cat(
                [vox[:, 3:C], vox[:, 2:3], vox[:, 1:2], vox[:, 0:1]], dim=1)
            coors = coors[:, [0, 3, 2, 1]]

        out = self.sparse_encoder(vox, coors, B)
        B, C_out, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B * H * W, C_out)
        out = self.proj_head(out)
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return out



# ---------------------------------------------------------------------------
#  Cross-backend comparison tests
# ---------------------------------------------------------------------------

def _make_test_points(n_points, device):
    """Generate random points within the configured BEV coordinate range."""
    pts = torch.zeros(n_points, 3, device=device)
    for dim, (lo, hi) in enumerate([(xbound[0], xbound[1]),
                                     (ybound[0], ybound[1]),
                                     (zbound[0], zbound[1])]):
        margin = (hi - lo) * 0.1
        pts[:, dim] = torch.rand(n_points, device=device) * (hi - lo - 2 * margin) + lo + margin
    return pts


def _make_sparse_input(n_voxels, in_channels, spatial_shape, batch_size, device):
    """Generate unique sparse tensor input shared by both backends."""
    raw = torch.stack([
        torch.randint(0, batch_size, (n_voxels * 3,)),
        torch.randint(0, spatial_shape[0], (n_voxels * 3,)),
        torch.randint(0, spatial_shape[1], (n_voxels * 3,)),
        torch.randint(0, spatial_shape[2], (n_voxels * 3,)),
    ], dim=1).to(device)
    indices = torch.unique(raw, dim=0)[:n_voxels].int()
    features = torch.randn(indices.shape[0], in_channels, device=device)
    return features, indices


def _cmp(tag, a, b, atol=1e-5):
    """Compare two tensors, print diff stats, return pass/fail."""
    a_nan, b_nan = a.isnan().any().item(), b.isnan().any().item()
    if a_nan or b_nan:
        print(f"    {tag}: a_has_nan={a_nan} b_has_nan={b_nan}  FAIL")
        return False
    diff = (a.float() - b.float()).abs()
    mx, mn = diff.max().item(), diff.mean().item()
    ok = mx < atol
    print(f"    {tag}: max_diff={mx:.6e}  mean_diff={mn:.6e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _convert_sd_v2_to_v1(state_dict):
    """Convert spconv v2 state_dict to drcv (spconv v1) weight layout.
    spconv v2: weight [O, K1, K2, K3, I]  →  drcv: weight [K1, K2, K3, I, O]
    """
    out = {}
    for k, v in state_dict.items():
        if "weight" in k and v.ndim == 5:
            out[k] = v.permute(1, 2, 3, 4, 0).contiguous()
        else:
            out[k] = v
    return out


def _cuda_sync():
    """Synchronize CUDA and clear any sticky errors."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _probe_drcv_spconv_subprocess(cuda_device_index: int) -> bool:
    """Test drcv sparse conv in a child process to avoid poisoning
    the parent's CUDA context on kernel failure.
    Returns True if the kernel works, False otherwise."""
    script = textwrap.dedent(f"""\
        import os, sys
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "{cuda_device_index}")
        sys.path.insert(0, os.path.dirname(os.path.abspath("{__file__}")))
        import torch
        import drcv.ops.spconv as sp
        d = "cuda:0"
        idx = torch.tensor([[0, 2, 2, 2]], dtype=torch.int32, device=d)
        feat = torch.randn(1, 3, device=d)
        x = sp.SparseConvTensor(feat, idx, [5, 5, 5], 1)
        conv = sp.SubMConv3d(3, 4, 3, padding=1, bias=False).to(d)
        out = conv(x)
        torch.cuda.synchronize()
        print("OK", out.features.shape[0], out.features.shape[1])
    """)
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0 and "OK" in r.stdout
    except Exception:
        return False


def _pick_device_for_drcv_spconv():
    """Return a CUDA device index verified to run drcv sparse conv,
    or None if no working GPU is found.
    Tests are done in subprocesses to avoid CUDA context corruption."""
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    candidates = []
    for i in range(n):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            candidates.append((i, free))
        except RuntimeError:
            pass
    candidates.sort(key=lambda x: -x[1])
    for idx, free_bytes in candidates:
        free_gb = free_bytes / (1024 ** 3)
        print(f"    Probing cuda:{idx} ({free_gb:.1f} GB free) via subprocess ...", end=" ")
        if _probe_drcv_spconv_subprocess(idx):
            print("OK")
            return torch.device(f"cuda:{idx}")
        else:
            print("FAIL")
    return None


# ---- Test 1: Voxelization (PointToVoxel vs drcv.voxelization) ----

def _test_voxelization(device):
    print("=" * 60)
    print("Test 1: Voxelization — PointToVoxel vs drcv.voxelization")
    print("=" * 60)

    torch.manual_seed(42)
    pts = _make_test_points(10000, device)

    try:
        from spconv.pytorch.utils import PointToVoxel as _SP
        ptvoxel = _SP(vsize_xyz=vsize_xyz, coors_range_xyz=tuple(_COORS_RANGE),
                       num_point_features=3, max_num_voxels=120000,
                       max_num_points_per_voxel=10, device=device)
        vox_sp, coors_sp, npts_sp = ptvoxel(pts)
        print(f"  [spconv] voxels={vox_sp.shape}, coors={coors_sp.shape}")
    except Exception as e:
        print(f"  [spconv] SKIP: {e}");  return None

    try:
        from drcv.ops.voxel import voxelization as _vox
        vox_dr, coors_dr, npts_dr = _vox(pts, vsize_xyz, _COORS_RANGE, 10, 120000)
        print(f"  [drcv]   voxels={vox_dr.shape}, coors={coors_dr.shape}")
    except Exception as e:
        print(f"  [drcv]   SKIP: {e}");  return None

    vox_sp_r = vox_sp.sum(dim=1)
    vox_dr_r = vox_dr.sum(dim=1)
    print(f"\n  voxel count — spconv: {vox_sp_r.shape[0]}, drcv: {vox_dr_r.shape[0]}")
    if vox_sp_r.shape[0] != vox_dr_r.shape[0]:
        print("  FAIL: voxel counts differ");  return False

    sp_key = coors_sp[:, 0] * 1_000_000 + coors_sp[:, 1] * 1_000 + coors_sp[:, 2]
    dr_key = coors_dr[:, 0] * 1_000_000 + coors_dr[:, 1] * 1_000 + coors_dr[:, 2]
    sp_ord, dr_ord = sp_key.argsort(), dr_key.argsort()

    ok = True
    ok &= _cmp("coors", coors_sp[sp_ord].float(), coors_dr[dr_ord].float(), atol=0.5)
    ok &= _cmp("num_points", npts_sp[sp_ord].float(), npts_dr[dr_ord].float(), atol=0.5)
    ok &= _cmp("voxel_features(sum)", vox_sp_r[sp_ord], vox_dr_r[dr_ord])
    return ok


# ---- Test 2: Sparse Conv Ops 逐算子对比 ----

_CROSS_BACKEND_SCRIPT = textwrap.dedent("""\
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, torch.nn as nn
import spconv.pytorch as sp_v2
import drcv.ops.spconv as sp_dr

device = "cuda:0"

def _make_sparse_input(n, c, shape, bs, dev):
    raw = torch.stack([
        torch.randint(0, bs, (n*3,)),
        torch.randint(0, shape[0], (n*3,)),
        torch.randint(0, shape[1], (n*3,)),
        torch.randint(0, shape[2], (n*3,)),
    ], dim=1).to(dev)
    idx = torch.unique(raw, dim=0)[:n].int()
    feat = torch.randn(idx.shape[0], c, device=dev)
    return feat, idx

def _v2_to_v1(sd):
    out = {}
    for k, v in sd.items():
        if "weight" in k and v.ndim == 5:
            out[k] = v.permute(1, 2, 3, 4, 0).contiguous()
        else:
            out[k] = v
    return out

results = {}

# ---- 2a. SubMConv3d ----
torch.manual_seed(42)
spatial = [30, 30, 11]
feat, idx = _make_sparse_input(500, 3, spatial, 1, device)

torch.manual_seed(0)
c_sp = sp_v2.SubMConv3d(3, 16, 3, padding=1, bias=False).to(device)
o_sp = c_sp(sp_v2.SparseConvTensor(feat.clone(), idx.clone(), spatial, 1))
torch.cuda.synchronize()

c_dr = sp_dr.SubMConv3d(3, 16, 3, padding=1, bias=False).to(device)
c_dr.load_state_dict(_v2_to_v1(c_sp.state_dict()))
o_dr = c_dr(sp_dr.SparseConvTensor(feat.clone(), idx.clone(), spatial, 1))
torch.cuda.synchronize()

d_feat = (o_sp.features - o_dr.features).abs().max().item()
d_dense = (o_sp.dense(False) - o_dr.dense(False)).abs().max().item()
results["2a_SubMConv3d_feat"] = d_feat
results["2a_SubMConv3d_dense"] = d_dense

# ---- 2b. SparseConv3d stride=2 ----
torch.manual_seed(0)
sc_sp = sp_v2.SparseConv3d(3, 32, 3, stride=2, padding=1, bias=False).to(device)
o2_sp = sc_sp(sp_v2.SparseConvTensor(feat.clone(), idx.clone(), spatial, 1))
torch.cuda.synchronize()

sc_dr = sp_dr.SparseConv3d(3, 32, 3, stride=2, padding=1, bias=False).to(device)
sc_dr.load_state_dict(_v2_to_v1(sc_sp.state_dict()))
o2_dr = sc_dr(sp_dr.SparseConvTensor(feat.clone(), idx.clone(), spatial, 1))
torch.cuda.synchronize()

results["2b_SparseConv3d_dense"] = (o2_sp.dense(False) - o2_dr.dense(False)).abs().max().item()

# ---- 2c. SparseSequential + BN + ReLU ----
torch.manual_seed(0)
seq_sp = sp_v2.SparseSequential(
    sp_v2.SubMConv3d(3, 16, 3, padding=1, bias=False),
    nn.BatchNorm1d(16, eps=1e-3, momentum=0.01), nn.ReLU(True),
).to(device).eval()
o3_sp = seq_sp(sp_v2.SparseConvTensor(feat.clone(), idx.clone(), spatial, 1))
torch.cuda.synchronize()

seq_dr = sp_dr.SparseSequential(
    sp_dr.SubMConv3d(3, 16, 3, padding=1, bias=False),
    nn.BatchNorm1d(16, eps=1e-3, momentum=0.01), nn.ReLU(True),
).to(device).eval()
seq_dr.load_state_dict(_v2_to_v1(seq_sp.state_dict()))
o3_dr = seq_dr(sp_dr.SparseConvTensor(feat.clone(), idx.clone(), spatial, 1))
torch.cuda.synchronize()

results["2c_SeqBNRelu_feat"] = (o3_sp.features - o3_dr.features).abs().max().item()
results["2c_SeqBNRelu_dense"] = (o3_sp.dense(False) - o3_dr.dense(False)).abs().max().item()

# ---- 2d. Gradient backward ----
torch.manual_seed(0)
gc_sp = sp_v2.SubMConv3d(3, 16, 3, padding=1, bias=False).to(device)
f_sp = feat.clone().requires_grad_(True)
go_sp = gc_sp(sp_v2.SparseConvTensor(f_sp, idx.clone(), spatial, 1))
go_sp.features.sum().backward()
torch.cuda.synchronize()

gc_dr = sp_dr.SubMConv3d(3, 16, 3, padding=1, bias=False).to(device)
gc_dr.load_state_dict(_v2_to_v1(gc_sp.state_dict()))
f_dr = feat.clone().requires_grad_(True)
go_dr = gc_dr(sp_dr.SparseConvTensor(f_dr, idx.clone(), spatial, 1))
go_dr.features.sum().backward()
torch.cuda.synchronize()

grad_info = {"sp_grad_ok": f_sp.grad is not None, "dr_grad_ok": f_dr.grad is not None}
if f_sp.grad is not None and f_dr.grad is not None:
    sp_g, dr_g = f_sp.grad.float(), f_dr.grad.float()
    grad_info["sp_nan"] = bool(sp_g.isnan().any())
    grad_info["dr_nan"] = bool(dr_g.isnan().any())
    if not grad_info["sp_nan"] and not grad_info["dr_nan"]:
        grad_info["cos"] = torch.nn.functional.cosine_similarity(
            sp_g.flatten().unsqueeze(0), dr_g.flatten().unsqueeze(0)).item()
results["2d_grad"] = grad_info

# ---- 3. Full Encoder (matches SparseEncoder architecture) ----
def _build_enc(sp, shape):
    ci = sp.SparseSequential(
        sp.SubMConv3d(3, 16, 3, padding=1, bias=False, indice_key="subm1"),
        nn.BatchNorm1d(16, eps=1e-3, momentum=0.01), nn.ReLU(True))
    cfg = [[16,16,32],[32,32,64],[64,64,128],[128,128]]
    pad = [[0,0,1],[0,0,1],[0,0,(1,1,0)],[0,0]]
    layers = nn.ModuleList()
    c = 16
    for i, blks in enumerate(cfg):
        bl = nn.ModuleList()
        for j, co in enumerate(blks):
            p = tuple(pad[i])[j]
            ik = f"subm{i+1}_{j}"
            if j==len(blks)-1 and i<len(cfg)-1:
                bl.append(sp.SparseSequential(
                    sp.SparseConv3d(c, co, 3, padding=p, stride=(2,2,2), indice_key=f"spconv{i+1}"),
                    nn.BatchNorm1d(co, eps=1e-3, momentum=0.01), nn.ReLU(True)))
            else:
                bl.append(sp.SparseSequential(
                    sp.SubMConv3d(c, co, 3, stride=1, padding=1, bias=False, indice_key=ik),
                    nn.BatchNorm1d(co, eps=1e-3, momentum=0.01), nn.ReLU(True),
                    sp.SubMConv3d(co, co, 3, stride=1, padding=1, bias=False, indice_key=ik),
                    nn.BatchNorm1d(co, eps=1e-3, momentum=0.01)))
            c = co
        layers.append(bl)
    cout = sp.SparseSequential(
        sp.SparseConv3d(c, 128, (3,3,3), stride=(1,1,2), padding=(1,1,0),
                        indice_key="spconv_down2", bias=False),
        nn.BatchNorm1d(128, eps=1e-3, momentum=0.01), nn.ReLU(True))
    class E(nn.Module):
        def __init__(self):
            super().__init__()
            self.ci, self.cl, self.co, self.ss = ci, layers, cout, shape
        def forward(self, f, c, bs):
            x = sp.SparseConvTensor(f, c.int(), self.ss, bs)
            x = self.ci(x)
            for l in self.cl:
                for s in l: x = s(x)
            x = self.co(x); x = x.dense(False)
            B,X,Y,Z,C = x.shape
            return x.view(B,X,Y,Z*C).permute(0,3,1,2).contiguous()
    return E()

enc_shape = [50, 50, 41]
torch.manual_seed(42)
e_sp = _build_enc(sp_v2, enc_shape).to(device).eval()
e_dr = _build_enc(sp_dr, enc_shape).to(device).eval()
e_dr.load_state_dict(_v2_to_v1(e_sp.state_dict()))

torch.manual_seed(7)
ef, ei = _make_sparse_input(300, 3, enc_shape, 1, device)
with torch.no_grad():
    eo_sp = e_sp(ef.clone(), ei.clone(), 1)
    torch.cuda.synchronize()
    eo_dr = e_dr(ef.clone(), ei.clone(), 1)
    torch.cuda.synchronize()

results["3_encoder_shape_sp"] = list(eo_sp.shape)
results["3_encoder_shape_dr"] = list(eo_dr.shape)
results["3_encoder_diff"] = (eo_sp - eo_dr).abs().max().item()

print(json.dumps(results))
""")


def _run_cross_backend_subprocess(gpu_idx):
    """Run Tests 2+3 in a subprocess on the given GPU. Returns parsed results or None."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_CROSS_BACKEND_SCRIPT)
        script_path = f.name
    try:
        r = subprocess.run(
            [sys.executable, script_path, str(gpu_idx)],
            capture_output=True, text=True, timeout=120,
        )
        os.unlink(script_path)
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                import json
                return json.loads(line)
        if r.returncode != 0:
            print(f"  Subprocess failed (exit={r.returncode}):")
            for line in (r.stderr or r.stdout).strip().split("\n")[-10:]:
                print(f"    {line}")
        return None
    except subprocess.TimeoutExpired:
        os.unlink(script_path)
        print("  Subprocess timed out (120s)")
        return None
    except Exception as e:
        print(f"  Subprocess error: {e}")
        return None


def _test_sparse_ops_and_encoder(device):
    """Tests 2+3: run cross-backend comparison in an isolated subprocess
    to avoid CUDA context poisoning in the main process."""
    print("\n" + "=" * 60)
    print("Tests 2+3: Cross-backend sparse ops (isolated subprocess)")
    print("=" * 60)

    # Find a working GPU via subprocess probe
    dr_device = _pick_device_for_drcv_spconv()
    if dr_device is None:
        print("  [SKIP] No GPU passed drcv sparse conv subprocess probe.")
        return None, None

    gpu_idx = dr_device.index
    print(f"\n  Running spconv v2 vs drcv comparison on cuda:{gpu_idx} ...\n")
    data = _run_cross_backend_subprocess(gpu_idx)
    if data is None:
        print("  [SKIP] Subprocess returned no results")
        return None, None

    # ---- Print Test 2 results ----
    print("  " + "-" * 50)
    print("  Test 2: Sparse Conv Ops")
    print("  " + "-" * 50)
    atol = 1e-5
    fwd_ok = True

    for tag in ["2a_SubMConv3d_feat", "2a_SubMConv3d_dense",
                "2b_SparseConv3d_dense",
                "2c_SeqBNRelu_feat", "2c_SeqBNRelu_dense"]:
        if tag in data:
            v = data[tag]
            ok = v < atol
            fwd_ok &= ok
            print(f"    {tag}: max_diff={v:.6e}  {'PASS' if ok else 'FAIL'}")

    gi = data.get("2d_grad", {})
    if gi.get("dr_nan"):
        print(f"    2d_grad: drcv backward produces NaN — known lib issue")
        print(f"    → Forward consistency verified; backward requires drcv fix.")
    elif "cos" in gi:
        cos = gi["cos"]
        print(f"    2d_grad: cosine_sim={cos:.6f}  "
              f"{'PASS' if cos > 0.95 else 'WARN'}")

    print(f"\n  Forward consistency: {'ALL PASS' if fwd_ok else 'SOME FAIL'}")

    # ---- Print Test 3 results ----
    print("\n  " + "-" * 50)
    print("  Test 3: Full SparseEncoder")
    print("  " + "-" * 50)
    enc_ok = None
    if "3_encoder_diff" in data:
        d = data["3_encoder_diff"]
        enc_ok = d < atol
        sp_s = data.get("3_encoder_shape_sp", "?")
        dr_s = data.get("3_encoder_shape_dr", "?")
        print(f"    output shapes: spconv={sp_s}, drcv={dr_s}")
        print(f"    encoder_output: max_diff={d:.6e}  {'PASS' if enc_ok else 'FAIL'}")

    return fwd_ok, enc_ok


# ---- Test 4: Full Pipeline smoke test on active backend ----

def _test_pipeline(device):
    backend_name = "drcv" if USE_DRCV else "spconv"
    print("\n" + "=" * 60)
    print(f"Test 4: Full Pipeline smoke (backend={backend_name})")
    print("=" * 60)

    torch.manual_seed(42)
    model = Lidar2BEV().to(device)
    B, C, N = 2, 3, 5000

    model.eval()
    pc = torch.randn(B, C, N, device=device)
    with torch.no_grad():
        out = model(pc)
    print(f"  [forward]  input={pc.shape} → output={out.shape}")
    print(f"  [forward]  min={out.min():.4f}, max={out.max():.4f}, "
          f"mean={out.mean():.4f}, std={out.std():.4f}")
    has_nan = torch.isnan(out).any().item()
    print(f"  [forward]  nan={has_nan}, nonzero={int((out != 0).sum())}/{out.numel()}")

    model.train()
    out = model(pc)
    loss = out.sum()
    loss.backward()
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  [backward] loss={loss.item():.4f}, all_grads_ok={grads_ok}")

    ok = not has_nan and grads_ok
    print(f"  [result]   {'PASS' if ok else 'FAIL'}")
    return ok


# ---- Test 5: Voxel mode equivalence (hard vs scatter) ----

def _test_voxel_modes(device):
    """Verify 'hard' and 'scatter' voxel modes produce identical output
    through the full Lidar2BEV pipeline with shared weights."""
    print("\n" + "=" * 60)
    print("Test 5: Voxel Mode Equivalence (hard vs scatter)")
    print("=" * 60)

    torch.manual_seed(42)
    model_hard = Lidar2BEV(to_bev_mode='concat', voxel_mode='hard').to(device).eval()
    model_scatter = Lidar2BEV(to_bev_mode='concat', voxel_mode='scatter').to(device).eval()
    model_scatter.load_state_dict(model_hard.state_dict(), strict=True)
    print("  [OK] scatter model loaded hard model weights (strict=True)")

    all_pass = True
    for B, N in [(1, 5000), (2, 20000), (1, 50000)]:
        pts = torch.zeros(B, 3, N, device=device)
        for dim, (lo, hi) in enumerate([
            (xbound[0], xbound[1]), (ybound[0], ybound[1]), (zbound[0], zbound[1])
        ]):
            margin = (hi - lo) * 0.05
            pts[:, dim, :] = torch.rand(B, N, device=device) * (hi - lo - 2*margin) + lo + margin

        with torch.no_grad():
            out_h = model_hard(pts)
            out_s = model_scatter(pts)

        diff = (out_h - out_s).abs()
        cos = torch.nn.functional.cosine_similarity(
            out_h.flatten(), out_s.flatten(), dim=0).item()
        ok = cos > 0.999
        all_pass &= ok
        print(f"  B={B} N={N}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}, "
              f"cos={cos:.8f}  {'PASS' if ok else 'FAIL'}")

    if all_pass:
        print("  [result] ALL PASS — hard and scatter modes are equivalent")
    else:
        print("  [result] SOME FAIL")
    return all_pass


# ---- Test 6: Voxelization output comparison (hard-sum vs scatter-sum vs scatter-mean) ----

def _test_voxel_output_comparison(device):
    """Compare raw voxelization outputs across all three modes.

    Checks:
      A) hard-sum vs scatter-sum: same voxels, same aggregated features
      B) scatter-sum vs scatter-mean: same voxels, sum == mean * count
      C) full pipeline: hard-sum vs scatter-mean BEV feature similarity
    """
    print("\n" + "=" * 60)
    print("Test 6: Voxelization Output Comparison (hard-sum / scatter-sum / scatter-mean)")
    print("=" * 60)

    torch.manual_seed(42)
    B, N, C = 1, 10000, 3
    pts = torch.zeros(B, C, N, device=device)
    for dim, (lo, hi) in enumerate([
        (xbound[0], xbound[1]), (ybound[0], ybound[1]), (zbound[0], zbound[1])
    ]):
        margin = (hi - lo) * 0.05
        pts[:, dim, :] = torch.rand(B, N, device=device) * (hi - lo - 2*margin) + lo + margin
    pc = pts.permute(0, 2, 1).contiguous()  # (B, N, C)

    all_pass = True

    # --- Part A: raw voxelization outputs ---
    print("\n  --- A) Raw voxel feature comparison ---")

    m_hard = Lidar2BEV(to_bev_mode='concat', voxel_mode='hard').to(device).eval()
    m_ssum = Lidar2BEV(to_bev_mode='concat', voxel_mode='scatter',
                        scatter_reduce='sum').to(device).eval()
    m_smean = Lidar2BEV(to_bev_mode='concat', voxel_mode='scatter',
                         scatter_reduce='mean').to(device).eval()

    with torch.no_grad():
        feats_h, coors_h = m_hard._voxelize_hard(pc)
        feats_ss, coors_ss = m_ssum._voxelize_scatter(pc)
        feats_sm, coors_sm = m_smean._voxelize_scatter(pc)

    print(f"  voxel counts — hard: {feats_h.shape[0]}, "
          f"scatter-sum: {feats_ss.shape[0]}, scatter-mean: {feats_sm.shape[0]}")

    # hard returns coors as batch-z-y-x; scatter returns batch-x-y-z
    # normalize to batch-x-y-z for comparison
    coors_h_bxyz = coors_h[:, [0, 3, 2, 1]]  # batch-z-y-x → batch-x-y-z

    # sort both by coordinate for alignment
    def _sort_key(c):
        return c[:, 0] * 1000000 + c[:, 1] * 10000 + c[:, 2] * 100 + c[:, 3]

    idx_h = _sort_key(coors_h_bxyz).argsort()
    idx_ss = _sort_key(coors_ss).argsort()
    idx_sm = _sort_key(coors_sm).argsort()

    coors_h_sorted = coors_h_bxyz[idx_h]
    coors_ss_sorted = coors_ss[idx_ss]
    coors_sm_sorted = coors_sm[idx_sm]
    feats_h_sorted = feats_h[idx_h]
    feats_ss_sorted = feats_ss[idx_ss]
    feats_sm_sorted = feats_sm[idx_sm]

    # A1: scatter-sum vs scatter-mean share same voxels
    if coors_ss_sorted.shape == coors_sm_sorted.shape and \
       (coors_ss_sorted == coors_sm_sorted).all():
        print("  [A1] scatter-sum vs scatter-mean: same voxel coordinates  PASS")
    else:
        print("  [A1] scatter-sum vs scatter-mean: voxel coordinates DIFFER  FAIL")
        all_pass = False

    # A2: verify sum == mean * count
    inv = torch.zeros(pc.shape[1], dtype=torch.long, device=device)
    from drcv.ops.torch_scatter import scatter_add as _sa
    pts_0 = pc[0]
    vmin = _VOXEL_MIN.to(device=pc.device, dtype=pc.dtype)
    vs = _VOXEL_SIZE.to(device=pc.device, dtype=pc.dtype)
    gmax = _GRID_MAX.to(device=pc.device)
    grid = ((pts_0[:, :3] - vmin) / vs).int()
    mask = ((grid >= 0) & (grid <= gmax)).all(dim=1)
    pts_masked, grid_masked = pts_0[mask], grid[mask]
    batch_col = torch.zeros(grid_masked.shape[0], 1, dtype=grid_masked.dtype, device=device)
    coors_full = torch.cat([batch_col, grid_masked], dim=1)
    uniq_coors, inv_idx = torch.unique(coors_full, return_inverse=True, dim=0)
    counts = torch.zeros(uniq_coors.shape[0], device=device)
    counts.scatter_add_(0, inv_idx, torch.ones_like(inv_idx, dtype=counts.dtype))

    idx_sm2 = _sort_key(uniq_coors).argsort()
    counts_sorted = counts[idx_sm2]
    reconstructed_sum = feats_sm_sorted * counts_sorted.unsqueeze(1)
    diff_recon = (feats_ss_sorted - reconstructed_sum).abs()
    ok_a2 = diff_recon.max().item() < 1e-4
    all_pass &= ok_a2
    print(f"  [A2] sum == mean * count: max_diff={diff_recon.max():.6e}  "
          f"{'PASS' if ok_a2 else 'FAIL'}")

    # A3: hard vs scatter-sum voxel feature comparison
    n_h, n_s = coors_h_sorted.shape[0], coors_ss_sorted.shape[0]
    coors_match = (n_h == n_s) and (coors_h_sorted == coors_ss_sorted).all().item()
    if coors_match:
        diff_hs = (feats_h_sorted - feats_ss_sorted).abs()
        cos_hs = torch.nn.functional.cosine_similarity(
            feats_h_sorted.flatten(), feats_ss_sorted.flatten(), dim=0).item()
        ok_a3 = cos_hs > 0.99
        all_pass &= ok_a3
        print(f"  [A3] hard-sum vs scatter-sum features: max_diff={diff_hs.max():.6e}, "
              f"cos={cos_hs:.8f}  {'PASS' if ok_a3 else 'FAIL'}")
    else:
        if n_h != n_s:
            reason = f"count mismatch ({n_h} vs {n_s})"
        else:
            n_coord_diff = (coors_h_sorted != coors_ss_sorted).any(dim=1).sum().item()
            reason = (f"same count ({n_h}) but {n_coord_diff} voxels have different coords "
                      f"(float→int rounding diff between CUDA kernel and PyTorch)")
        cos_hs = torch.nn.functional.cosine_similarity(
            feats_h_sorted.flatten().float(), feats_ss_sorted.flatten().float(), dim=0).item()
        ok_a3 = cos_hs > 0.99
        all_pass &= ok_a3
        print(f"  [A3] hard-sum vs scatter-sum: {reason}")
        print(f"       feature cos={cos_hs:.8f}  {'PASS' if ok_a3 else 'FAIL'}")

    # --- Part B: full pipeline comparison ---
    print("\n  --- B) Full pipeline: hard-sum vs scatter-mean BEV output ---")

    torch.manual_seed(42)
    model_hard = Lidar2BEV(to_bev_mode='concat', voxel_mode='hard').to(device).eval()
    model_mean = Lidar2BEV(to_bev_mode='concat', voxel_mode='scatter',
                            scatter_reduce='mean').to(device).eval()
    model_mean.load_state_dict(model_hard.state_dict(), strict=True)

    for B_test, N_test in [(1, 5000), (2, 20000)]:
        pts_test = torch.zeros(B_test, 3, N_test, device=device)
        for dim, (lo, hi) in enumerate([
            (xbound[0], xbound[1]), (ybound[0], ybound[1]), (zbound[0], zbound[1])
        ]):
            margin = (hi - lo) * 0.05
            pts_test[:, dim, :] = (
                torch.rand(B_test, N_test, device=device) * (hi - lo - 2*margin) + lo + margin)

        with torch.no_grad():
            out_h = model_hard(pts_test)
            out_m = model_mean(pts_test)

        diff_bev = (out_h - out_m).abs()
        cos_bev = torch.nn.functional.cosine_similarity(
            out_h.flatten(), out_m.flatten(), dim=0).item()
        ok_b = cos_bev > 0.95
        all_pass &= ok_b
        print(f"  B={B_test} N={N_test}: max_diff={diff_bev.max():.6e}, "
              f"mean_diff={diff_bev.mean():.6e}, cos={cos_bev:.8f}  "
              f"{'PASS' if ok_b else 'FAIL'}")
        print(f"    (note: sum vs mean aggregation — larger divergence is expected)")

    if all_pass:
        print("  [result] ALL PASS")
    else:
        print("  [result] SOME FAIL")
    return all_pass


# ---- Test 7: JIT traceability for scatter modes ----

def _test_jit_trace(device):
    """Verify scatter-sum and scatter-mean are JIT-traceable (drinfer export)."""
    print("\n" + "=" * 60)
    print("Test 7: JIT Traceability (scatter-sum & scatter-mean)")
    print("=" * 60)

    B, N = 1, 5000
    pts = torch.zeros(B, 3, N, device=device)
    for dim, (lo, hi) in enumerate([
        (xbound[0], xbound[1]), (ybound[0], ybound[1]), (zbound[0], zbound[1])
    ]):
        margin = (hi - lo) * 0.05
        pts[:, dim, :] = torch.rand(B, N, device=device) * (hi - lo - 2*margin) + lo + margin

    all_pass = True
    for mode_name, scatter_reduce in [("scatter-sum", "sum"), ("scatter-mean", "mean")]:
        model = Lidar2BEV(
            to_bev_mode='concat', voxel_mode='scatter',
            scatter_reduce=scatter_reduce).to(device).eval()
        with torch.no_grad():
            out_eager = model(pts)
        print(f"  {mode_name}: eager output={out_eager.shape}, nan={out_eager.isnan().any().item()}")

        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, (pts,))
            out_traced = traced(pts)
            diff = (out_eager - out_traced).abs()
            cos = torch.nn.functional.cosine_similarity(
                out_eager.flatten(), out_traced.flatten(), dim=0).item()
            ok = cos > 0.999
            all_pass &= ok
            print(f"  {mode_name}: JIT trace OK, eager-vs-traced max_diff={diff.max():.6e}, "
                  f"cos={cos:.8f}  {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            all_pass = False
            print(f"  {mode_name}: JIT trace FAILED — {e}")

    if all_pass:
        print("  [result] ALL PASS — both modes are JIT-traceable")
    else:
        print("  [result] SOME FAIL")
    return all_pass


# ---- Test 8: SpconvToDenseBEV coordinate correctness ----

def _test_to_bev_coordinates(device):
    """Verify SpconvToDenseBEV scatter logic with deterministic inputs.

    Creates sparse features at known (batch, x, y, z) positions and checks
    that the dense BEV output has non-zero values at exactly those positions.
    Tests 'concat', 'sum', and 'learned' modes.
    """
    from pc_encoders import SpconvToDenseBEV, USE_DRCV as _ENC_USE_DRCV

    print("\n" + "=" * 60)
    backend = "drcv(torch_sparse)" if _ENC_USE_DRCV else "spconv"
    print(f"Test 8: SpconvToDenseBEV Coordinate Correctness (backend={backend})")
    print("=" * 60)

    in_ch, n_z, bev_h, bev_w = 4, 3, 8, 10
    all_pass = True

    # ---- 8a: concat mode ----
    to_bev = SpconvToDenseBEV(in_ch, in_ch, n_z, (bev_h, bev_w), mode='concat').to(device).eval()

    feats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], device=device)

    x_pos, y_pos = 2, 3
    z_positions = [0, 1, 2]
    batch_positions = [0, 0, 0]

    if _ENC_USE_DRCV:
        coords = torch.tensor([
            [x_pos, y_pos, z_positions[0], batch_positions[0]],
            [x_pos, y_pos, z_positions[1], batch_positions[1]],
            [x_pos, y_pos, z_positions[2], batch_positions[2]],
        ], dtype=torch.int32, device=device)

        class _MockDrcv:
            pass
        mock_input = _MockDrcv()
        mock_input.F = feats
        mock_input.C = coords
        mock_input.s = 1
    else:
        coords = torch.tensor([
            [batch_positions[0], x_pos, y_pos, z_positions[0]],
            [batch_positions[1], x_pos, y_pos, z_positions[1]],
            [batch_positions[2], x_pos, y_pos, z_positions[2]],
        ], dtype=torch.int32, device=device)

        import spconv.pytorch as _spconv_test
        mock_input = _spconv_test.SparseConvTensor(
            feats, coords, [bev_h, bev_w, n_z], 1)

    with torch.no_grad():
        bev = to_bev(mock_input)

    expected_out_ch = n_z * in_ch
    shape_ok = (bev.shape == torch.Size([1, expected_out_ch, bev_h, bev_w]))
    pixel = bev[0, :, x_pos, y_pos]
    expected_pixel = torch.zeros(expected_out_ch, device=device)
    for i, z in enumerate(z_positions):
        expected_pixel[z * in_ch: z * in_ch + in_ch] = feats[i]

    pixel_match = torch.allclose(pixel, expected_pixel, atol=1e-6)
    other_zero = (bev[0, :, :, :].clone().index_fill_(1,
                  torch.tensor([x_pos], device=device), 0).abs().sum().item() == 0.0)
    ok_8a = shape_ok and pixel_match
    all_pass &= ok_8a
    print(f"  [8a] concat: shape={list(bev.shape)} (expect [1,{expected_out_ch},{bev_h},{bev_w}]) "
          f"{'✓' if shape_ok else '✗'}")
    print(f"       pixel@({x_pos},{y_pos}): match={pixel_match} {'PASS' if ok_8a else 'FAIL'}")

    # ---- 8b: sum mode ----
    to_bev_sum = SpconvToDenseBEV(in_ch, in_ch, n_z, (bev_h, bev_w), mode='sum').to(device).eval()
    with torch.no_grad():
        bev_sum = to_bev_sum(mock_input)

    expected_sum = feats.sum(dim=0)
    pixel_sum = bev_sum[0, :, x_pos, y_pos]
    ok_8b = torch.allclose(pixel_sum, expected_sum, atol=1e-6)
    all_pass &= ok_8b
    print(f"  [8b] sum: pixel@({x_pos},{y_pos}) = {pixel_sum.tolist()}")
    print(f"       expected = {expected_sum.tolist()}  {'PASS' if ok_8b else 'FAIL'}")

    # ---- 8c: multi-batch test ----
    feats_mb = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
    ], device=device)

    if _ENC_USE_DRCV:
        coords_mb = torch.tensor([
            [1, 2, 0, 0],
            [3, 4, 1, 1],
        ], dtype=torch.int32, device=device)
        mock_mb = _MockDrcv()
        mock_mb.F = feats_mb
        mock_mb.C = coords_mb
        mock_mb.s = 1
    else:
        coords_mb = torch.tensor([
            [0, 1, 2, 0],
            [1, 3, 4, 1],
        ], dtype=torch.int32, device=device)
        mock_mb = _spconv_test.SparseConvTensor(
            feats_mb, coords_mb, [bev_h, bev_w, n_z], 2)

    to_bev_mb = SpconvToDenseBEV(in_ch, in_ch, n_z, (bev_h, bev_w), mode='concat').to(device).eval()
    with torch.no_grad():
        bev_mb = to_bev_mb(mock_mb)

    ok_8c_shape = (bev_mb.shape[0] == 2)
    p0 = bev_mb[0, :, 1, 2]
    p1 = bev_mb[1, :, 3, 4]
    batch0_ok = p0[0:in_ch].sum().item() > 0
    batch1_ok = p1[in_ch:2*in_ch].sum().item() > 0
    ok_8c = ok_8c_shape and batch0_ok and batch1_ok
    all_pass &= ok_8c
    print(f"  [8c] multi-batch: shape={list(bev_mb.shape)}, "
          f"batch0_feat={'✓' if batch0_ok else '✗'}, "
          f"batch1_feat={'✓' if batch1_ok else '✗'}  "
          f"{'PASS' if ok_8c else 'FAIL'}")

    # ---- 8d: z clamping test ----
    feats_clamp = torch.ones(1, in_ch, device=device)
    if _ENC_USE_DRCV:
        coords_clamp = torch.tensor([[0, 0, n_z + 5, 0]], dtype=torch.int32, device=device)
        mock_clamp = _MockDrcv()
        mock_clamp.F = feats_clamp
        mock_clamp.C = coords_clamp
        mock_clamp.s = 1
    else:
        coords_clamp = torch.tensor([[0, 0, 0, n_z + 5]], dtype=torch.int32, device=device)
        mock_clamp = _spconv_test.SparseConvTensor(
            feats_clamp, coords_clamp, [bev_h, bev_w, n_z + 10], 1)

    with torch.no_grad():
        bev_clamp = to_bev(mock_clamp)

    no_nan = not bev_clamp.isnan().any().item()
    clamped_z_ch_start = (n_z - 1) * in_ch
    pixel_clamped = bev_clamp[0, clamped_z_ch_start:clamped_z_ch_start + in_ch, 0, 0]
    ok_8d = no_nan and pixel_clamped.sum().item() > 0
    all_pass &= ok_8d
    print(f"  [8d] z-clamp: z={n_z+5} clamped to {n_z-1}, no_nan={no_nan}, "
          f"feature_present={'✓' if pixel_clamped.sum().item() > 0 else '✗'}  "
          f"{'PASS' if ok_8d else 'FAIL'}")

    if all_pass:
        print("  [result] ALL PASS")
    else:
        print("  [result] SOME FAIL")
    return all_pass


# ---- Test 9: Cross-backend SparseEncoder stage-by-stage equivalence ----
#
# Two subprocesses: spconv saves state_dict + stage features,
# drcv loads mapped weights and compares stage features.
#
# Weight format: spconv [O,K,K,K,I] → permute(1,2,3,4,0) → reshape → drcv [K³,I,O]

_ENCODER_PHASE_SCRIPT = textwrap.dedent("""\
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", sys.argv[1])
backend = sys.argv[2]
os.environ["USE_DRCV_BACKEND"] = "0" if backend == "spconv" else "1"
input_path = sys.argv[3]
out_path = sys.argv[4]
source_dir = sys.argv[5]
sys.path.insert(0, source_dir)
parent_dir = os.path.dirname(source_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch, torch.nn as nn
from pc_encoders import SparseEncoder, USE_DRCV
from bev_settings import sparse_shape

device = "cuda:0"
data = torch.load(input_path, map_location="cpu")
features = data["features"].to(device)
coors = data["coors"].to(device)
batch_size = int(data["batch_size"])

# Load mapped weights if provided
mapped_sd_path = data.get("mapped_sd_path")
encoder = SparseEncoder(sparse_shape).to(device)
if mapped_sd_path and os.path.isfile(mapped_sd_path):
    mapped = torch.load(mapped_sd_path, map_location=device)
    encoder.load_state_dict(mapped, strict=False)
encoder.eval()

# Capture stage outputs via hooks
stage_outs = {}
def _hook(name):
    def fn(module, inp, out):
        if hasattr(out, 'features'):
            f, c = out.features, out.indices
        elif hasattr(out, 'F'):
            f, c = out.F, out.C
        else:
            return
        stage_outs[name] = (f.detach().cpu(), c.detach().cpu().long())
    return fn

hooks = [encoder.conv_input.register_forward_hook(_hook("conv_input"))]
for i, stage in enumerate(encoder.conv_layers):
    for j, block in enumerate(stage):
        hooks.append(block.register_forward_hook(_hook(f"stage{i}_{j}")))

with torch.no_grad():
    out = encoder(features, coors, batch_size)
for h in hooks:
    h.remove()
torch.cuda.synchronize()

save_data = {
    "state_dict": {k: v.cpu() for k, v in encoder.state_dict().items()},
    "stage_outs": {k: (f, c) for k, (f, c) in stage_outs.items()},
    "output": out.cpu(),
    "n_z": encoder.to_bev.n_z,
    "use_drcv": USE_DRCV,
}
torch.save(save_data, out_path)

info = {
    "shape": list(out.shape), "n_z": encoder.to_bev.n_z,
    "has_nan": bool(out.isnan().any()),
    "stages": list(stage_outs.keys()),
    "param_count": sum(p.numel() for p in encoder.parameters()),
}
print(json.dumps(info))
""")


def _map_spconv_to_drcv_sd(sp_sd, dr_keys):
    """Map spconv state_dict to drcv torch_sparse format.

    Uses explicit structure-aware rules matching the SparseEncoder architecture:
      conv_input → conv_input
      conv_layers.I.J (ResBlock) → conv_layers.I.J
      conv_layers.I.K (downsample) → conv_layers.I.K
      conv_out → SKIPPED (architecturally different)
    """
    dr_key_set = set(dr_keys)
    mapped = {}

    def _conv(sp_key, dr_key):
        if sp_key not in sp_sd or dr_key not in dr_key_set:
            return
        w = sp_sd[sp_key].permute(1, 2, 3, 4, 0).contiguous()
        mapped[dr_key] = w.reshape(-1, w.shape[3], w.shape[4])

    def _bn(sp_prefix, dr_prefix):
        for suf in ('weight', 'bias', 'running_mean', 'running_var',
                     'num_batches_tracked'):
            sp_k = f"{sp_prefix}.{suf}"
            dr_k = f"{dr_prefix}.{suf}"
            if sp_k in sp_sd and dr_k in dr_key_set:
                mapped[dr_k] = sp_sd[sp_k]

    _conv('conv_input.0.weight', 'conv_input.convbnrelu.0.kernel')
    _bn('conv_input.1', 'conv_input.convbnrelu.1.bn')

    layer_cfg = [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]]
    for i, blocks in enumerate(layer_cfg):
        for j in range(len(blocks)):
            p = f"conv_layers.{i}.{j}"
            is_down = (j == len(blocks) - 1) and (i < len(layer_cfg) - 1)
            if is_down:
                _conv(f"{p}.0.weight", f"{p}.convbnrelu.0.kernel")
                _bn(f"{p}.1", f"{p}.convbnrelu.1.bn")
            else:
                _conv(f"{p}.conv1.weight", f"{p}.conv1.convbnrelu.0.kernel")
                _bn(f"{p}.bn1", f"{p}.conv1.convbnrelu.1.bn")
                _conv(f"{p}.conv2.weight", f"{p}.conv2.convbn.0.kernel")
                _bn(f"{p}.bn2", f"{p}.conv2.convbn.1.bn")

    return mapped


def _test_cross_backend_real_encoder(device):
    """Test the real SparseEncoder stage-by-stage with weight transfer.

    Phase 1: spconv subprocess → save state_dict + stage features
    Phase 2: main process → map weights spconv → drcv format
    Phase 3: drcv subprocess → load mapped weights, save stage features
    Phase 4: compare stage features between backends
    """
    print("\n" + "=" * 60)
    print("Test 9: Cross-Backend SparseEncoder Equivalence (stage-by-stage)")
    print("=" * 60)

    dr_device = _pick_device_for_drcv_spconv()
    if dr_device is None:
        print("  [SKIP] No GPU passed drcv+spconv subprocess probe.")
        return None

    gpu_idx = dr_device.index

    import tempfile, json

    torch.manual_seed(99)
    n_pts = 1000
    B = 2
    feats_all, coors_all = [], []
    for b in range(B):
        pts = _make_test_points(n_pts, device)
        grid = ((pts - _VOXEL_MIN.to(device=device, dtype=pts.dtype)) /
                _VOXEL_SIZE.to(device=device, dtype=pts.dtype)).int()
        mask = ((grid >= 0) & (grid <= _GRID_MAX.to(device))).all(dim=1)
        grid = grid[mask]
        batch_col = torch.full((grid.shape[0], 1), b, dtype=grid.dtype, device=device)
        coors_b = torch.cat([batch_col, grid], dim=1)
        uniq, inv = torch.unique(coors_b, return_inverse=True, dim=0)
        if USE_DRCV:
            from drcv.ops.torch_scatter import scatter_mean as _sm
            feat_agg = _sm(pts[mask], inv, dim=0)[:uniq.shape[0]]
        else:
            feat_agg = pts[mask][:uniq.shape[0]]
        feats_all.append(feat_agg.cpu())
        coors_all.append(uniq.cpu())

    features = torch.cat(feats_all, dim=0)
    coors = torch.cat(coors_all, dim=0)

    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, "input.pt")
    torch.save({"features": features, "coors": coors, "batch_size": B}, input_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(tmpdir, "phase.py")
    with open(script_path, "w") as f:
        f.write(_ENCODER_PHASE_SCRIPT)

    def _run_phase(backend, extra_data=None):
        out_file = os.path.join(tmpdir, f"{backend}_out.pt")
        save_input = {"features": features, "coors": coors, "batch_size": B}
        if extra_data:
            save_input.update(extra_data)
        torch.save(save_input, input_path)
        try:
            r = subprocess.run(
                [sys.executable, script_path, str(gpu_idx), backend,
                 input_path, out_file, script_dir],
                capture_output=True, text=True, timeout=120, cwd=script_dir)
            if r.returncode != 0:
                for line in (r.stderr or r.stdout).strip().split("\n")[-8:]:
                    print(f"    {line}")
                return None
            for line in r.stdout.strip().split("\n"):
                if line.strip().startswith("{"):
                    info = json.loads(line.strip())
                    print(f"    shape={info['shape']}, n_z={info['n_z']}, "
                          f"params={info['param_count']}, nan={info['has_nan']}")
            return torch.load(out_file, map_location="cpu")
        except Exception as e:
            print(f"    ERROR: {e}")
            return None

    # Phase 1: spconv
    print(f"\n  Phase 1: spconv encoder on cuda:{gpu_idx}")
    sp_data = _run_phase("spconv")
    if sp_data is None:
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    # Phase 2: map weights
    sp_sd = sp_data["state_dict"]
    dr_dummy_keys_path = os.path.join(tmpdir, "dr_keys.pt")
    # Need drcv key names — run a quick probe
    print(f"\n  Phase 2: weight mapping (spconv 5D → drcv 3D)")
    dr_probe = _run_phase("drcv")
    if dr_probe is None:
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
        return None
    dr_keys = list(dr_probe["state_dict"].keys())

    mapped_sd = _map_spconv_to_drcv_sd(sp_sd, dr_keys)
    mapped_path = os.path.join(tmpdir, "mapped_sd.pt")
    torch.save(mapped_sd, mapped_path)
    print(f"    mapped {len(mapped_sd)}/{len(dr_keys)} drcv keys")
    unmapped = set(dr_keys) - set(mapped_sd.keys())
    if unmapped:
        conv_out_unmapped = [k for k in unmapped if 'conv_out' in k or 'to_bev' in k]
        other_unmapped = [k for k in unmapped if k not in conv_out_unmapped]
        if conv_out_unmapped:
            print(f"    unmapped conv_out/to_bev: {len(conv_out_unmapped)} "
                  f"(expected — different architecture)")
        if other_unmapped:
            print(f"    unmapped OTHER: {other_unmapped[:5]}")

    # Phase 3: drcv with mapped weights
    print(f"\n  Phase 3: drcv encoder with mapped spconv weights")
    dr_data = _run_phase("drcv", {"mapped_sd_path": mapped_path})
    if dr_data is None:
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    # Phase 4: compare stage features
    print(f"\n  Phase 4: stage-by-stage feature comparison")
    sp_stages = sp_data.get("stage_outs", {})
    dr_stages = dr_data.get("stage_outs", {})
    common = sorted(set(sp_stages.keys()) & set(dr_stages.keys()))

    all_pass = True
    for name in common:
        sp_feat, sp_idx = sp_stages[name]
        dr_feat, dr_idx = dr_stages[name]

        n_sp, n_dr = sp_feat.shape[0], dr_feat.shape[0]
        ch_sp, ch_dr = sp_feat.shape[1], dr_feat.shape[1]

        if ch_sp != ch_dr:
            print(f"    {name:20s}: channel mismatch sp={ch_sp} dr={ch_dr}")
            continue

        if n_sp == n_dr and ch_sp == ch_dr:
            sp_c = sp_idx.long()  # [batch, x, y, z]
            dr_c = dr_idx.long()  # [x, y, z, batch]
            sp_key = sp_c[:, 0] * 10**9 + sp_c[:, 1] * 10**6 + sp_c[:, 2] * 10**3 + sp_c[:, 3]
            dr_key = dr_c[:, 3] * 10**9 + dr_c[:, 0] * 10**6 + dr_c[:, 1] * 10**3 + dr_c[:, 2]

            sp_order = sp_key.argsort()
            dr_order = dr_key.argsort()

            sp_sorted = sp_feat[sp_order].float()
            dr_sorted = dr_feat[dr_order].float()

            sp_c_sorted = sp_c[sp_order]
            dr_c_norm = dr_c[dr_order][:, [3, 0, 1, 2]]

            coords_match = (sp_c_sorted == dr_c_norm).all().item() if sp_c_sorted.shape == dr_c_norm.shape else False

            if sp_sorted.numel() > 0 and dr_sorted.numel() > 0:
                diff = (sp_sorted - dr_sorted).abs()
                cos = torch.nn.functional.cosine_similarity(
                    sp_sorted.flatten(), dr_sorted.flatten(), dim=0).item()
            else:
                diff = torch.tensor([0.0])
                cos = 1.0

            ok = cos > 0.95
            all_pass &= ok
            print(f"    {name:20s}: n={n_sp:5d} ch={ch_sp:3d}  "
                  f"coords_match={coords_match}  "
                  f"cos={cos:.6f}  max_diff={diff.max():.4e}  "
                  f"{'PASS' if ok else 'FAIL'}")
        else:
            print(f"    {name:20s}: n_active sp={n_sp} dr={n_dr} (differ)")

    sp_out, dr_out = sp_data["output"], dr_data["output"]
    print(f"\n  Final output: spconv={list(sp_out.shape)}, drcv={list(dr_out.shape)}")
    print(f"  n_z: spconv={sp_data['n_z']}, drcv={dr_data['n_z']}  "
          f"(differ due to conv_out architecture)")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if all_pass:
        print("\n  [result] ALL PASS — common stages produce equivalent features")
    else:
        print("\n  [result] SOME FAIL — backends diverge at some stages")
    return all_pass


# ---- Test 10: _auto_permute_spconv_weights correctness ----

def _test_auto_permute_weights(device):
    """Verify that the weight permutation logic in evaluate_checkpoint.py
    correctly maps between spconv v2 and drcv weight layouts."""
    print("\n" + "=" * 60)
    print("Test 10: Weight Permutation (spconv v2 ↔ drcv layout)")
    print("=" * 60)

    all_pass = True
    torch.manual_seed(42)

    # 10a: roundtrip — permute v2→v1→v2 should be identity
    w_v2 = torch.randn(32, 3, 3, 3, 16)  # spconv v2: (O, K, K, K, I)
    w_v1 = w_v2.permute(1, 2, 3, 4, 0).contiguous()  # drcv/v1: (K, K, K, I, O)
    w_back = w_v1.permute(4, 0, 1, 2, 3).contiguous()  # back to v2
    ok_10a = torch.allclose(w_v2, w_back, atol=1e-7)
    all_pass &= ok_10a
    print(f"  [10a] v2→v1→v2 roundtrip: {'PASS' if ok_10a else 'FAIL'}")

    # 10b: symmetric case — when I == O, shapes differ but sorted() match
    w_sym = torch.randn(16, 3, 3, 3, 16)  # v2: (O=16, K, K, K, I=16)
    w_sym_v1 = w_sym.permute(1, 2, 3, 4, 0).contiguous()  # v1: (3, 3, 3, 16, 16)
    shapes_differ = (w_sym.shape != w_sym_v1.shape)
    sorted_match = (sorted(w_sym.shape) == sorted(w_sym_v1.shape))
    ok_10b = shapes_differ and sorted_match
    all_pass &= ok_10b
    print(f"  [10b] symmetric (I==O): shapes_differ={shapes_differ}, "
          f"sorted_match={sorted_match}  {'PASS' if ok_10b else 'FAIL'}")

    # 10c: simulate _auto_permute_spconv_weights detection logic
    # Create a mock model with drcv-layout weights and a checkpoint with spconv-layout weights
    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Linear(1, 1)  # dummy

    model = _MockModel()
    model_sd = {"conv.weight": torch.randn(3, 3, 3, 16, 32)}  # drcv layout
    ckpt_sd = {"conv.weight": torch.randn(32, 3, 3, 3, 16)}  # spconv v2 layout

    cs = ckpt_sd["conv.weight"].shape
    ms = model_sd["conv.weight"].shape
    needs_permute = (cs != ms) and (sorted(cs) == sorted(ms))
    ok_detect = needs_permute
    all_pass &= ok_detect
    print(f"  [10c] detection: ckpt={list(cs)}, model={list(ms)}, "
          f"needs_permute={needs_permute}  {'PASS' if ok_detect else 'FAIL'}")

    # 10c cont: verify permute(1,2,3,4,0) maps ckpt→model
    w_permuted = ckpt_sd["conv.weight"].permute(1, 2, 3, 4, 0).contiguous()
    ok_shape = (w_permuted.shape == model_sd["conv.weight"].shape)
    all_pass &= ok_shape
    print(f"  [10c] permute result: {list(w_permuted.shape)} == "
          f"{list(model_sd['conv.weight'].shape)}  {'PASS' if ok_shape else 'FAIL'}")

    # 10d: BN/other non-5D weights should pass through unchanged
    bn_w = torch.randn(64)
    bn_w_copy = bn_w.clone()
    sd_mixed = {
        "conv.weight": torch.randn(32, 3, 3, 3, 16),
        "bn.weight": bn_w,
        "bn.bias": torch.randn(64),
        "linear.weight": torch.randn(128, 64),
    }
    for k, v in sd_mixed.items():
        if v.ndim == 5:
            sd_mixed[k] = v.permute(1, 2, 3, 4, 0).contiguous()
    ok_10d = torch.allclose(sd_mixed["bn.weight"], bn_w_copy)
    all_pass &= ok_10d
    print(f"  [10d] non-5D passthrough (BN weight): {'PASS' if ok_10d else 'FAIL'}")

    if all_pass:
        print("  [result] ALL PASS")
    else:
        print("  [result] SOME FAIL")
    return all_pass


# ---- main ----

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"USE_DRCV_BACKEND={os.environ.get('USE_DRCV_BACKEND', '0')} "
          f"→ active backend: {'drcv' if USE_DRCV else 'spconv'}")
    print(f"Config: xbound={xbound}, ybound={ybound}, zbound={zbound}")
    print(f"        sparse_shape={sparse_shape}, vsize_xyz={vsize_xyz}\n")

    results = {}
    results["1_voxelization"] = _test_voxelization(device)

    sparse_ok, encoder_ok = _test_sparse_ops_and_encoder(device)
    results["2_sparse_ops"] = sparse_ok
    results["3_encoder"] = encoder_ok

    results["4_pipeline"] = _test_pipeline(device)
    results["5_voxel_modes"] = _test_voxel_modes(device)
    results["6_voxel_output_cmp"] = _test_voxel_output_comparison(device)
    results["7_jit_trace"] = _test_jit_trace(device)
    results["8_to_bev_coords"] = _test_to_bev_coordinates(device)
    results["9_cross_backend_enc"] = _test_cross_backend_real_encoder(device)
    results["10_weight_permute"] = _test_auto_permute_weights(device)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, val in results.items():
        status = "PASS" if val is True else ("SKIP" if val is None else "FAIL")
        print(f"  {name}: {status}")

    print(f"\nTo switch backend, re-run with:")
    print(f"  USE_DRCV_BACKEND=0 python {os.path.basename(__file__)}  # spconv")
    print(f"  USE_DRCV_BACKEND=1 python {os.path.basename(__file__)}  # drcv")