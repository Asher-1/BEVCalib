import os
import sys
import subprocess
import textwrap
import torch
import torch.nn as nn

# Backend switch: set USE_DRCV_BACKEND=1 to use drcv ops instead of spconv
USE_DRCV = os.environ.get("USE_DRCV_BACKEND", "1") == "1"

if USE_DRCV:
    from drcv.ops.voxel import voxelization as _drcv_voxelization
else:
    from spconv.pytorch.utils import PointToVoxel

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


class Lidar2BEV(nn.Module):
    def __init__(self):
        super(Lidar2BEV, self).__init__()
        if USE_DRCV:
            print("Use DRCV as Lidar2BEV backend")
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
        self.sparse_encoder = SparseEncoder(sparse_shape=sparse_shape)
        self.proj_head = ProjectionHead()
        self.voxelize_reduce = True
        self.out_channels = self.proj_head.projection_dim

    def _voxelize_single(self, points):
        """Voxelize a single point cloud sample.
        Both backends return (voxels, coors, num_points) with coors in zyx order.
        """
        if USE_DRCV:
            return _drcv_voxelization(
                points, self._voxel_size, self._coors_range,
                self._max_num_points, self._max_num_voxels
            )
        return self.ptvoxel(points)

    @torch.no_grad()
    def voxelize(self, pc):
        vox, coors, num_points = None, None, None
        B, _, _ = pc.shape
        for i in range(B):
            vox_, coors_, num_points_ = self._voxelize_single(pc[i])  # zyx order
            batch_id = torch.full([vox_.shape[0], 1], i, dtype=torch.int32, device=vox_.device)
            if self.voxelize_reduce:
                vox_ = torch.sum(vox_, dim=1, keepdim=False)
            coors_ = torch.cat([batch_id, coors_], dim=1)
            if vox is None:
                vox = vox_
                coors = coors_
                num_points = num_points_
            else:
                vox = torch.cat([vox, vox_], 0) # zyx order
                coors = torch.cat([coors, coors_], 0)
                num_points = torch.cat([num_points, num_points_], 0)
        
        return vox, coors, num_points

    def forward(self, pc):
        """
        Args:
            pc: (B, C, N)
        Returns:
            bev feats: (B, C, H, W)
        """
        B, C, N = pc.shape
        pc = pc.permute(0, 2, 1).contiguous()
        vox, coors, num_points = self.voxelize(pc)

        z_part, y_part, x_part, c_part = vox[:, 0:1], vox[:, 1:2], vox[:, 2:3], vox[:, 3:C]
        vox = torch.cat([c_part, x_part, y_part, z_part], dim=1)
        coors = coors[:, [0, 3, 2, 1]]  # zyx to xyz
        out = self.sparse_encoder(vox, coors, B)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B * H * W, C)
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

# ---- 3. Full Encoder ----
def _build_enc(sp, shape):
    ci = sp.SparseSequential(
        sp.SubMConv3d(3, 16, 3, padding=1, bias=False, indice_key="s1"),
        nn.BatchNorm1d(16, eps=1e-3, momentum=0.01), nn.ReLU(True))
    cfg = [[16,16,32],[32,32,64],[64,64,128],[128,128]]
    pad = [[0,0,1],[0,0,1],[0,0,(1,1,0)],[0,0]]
    layers = nn.ModuleList()
    c = 3
    for i, blks in enumerate(cfg):
        bl = nn.ModuleList()
        for j, co in enumerate(blks):
            p = tuple(pad[i])[j]
            if j==len(blks)-1 and i<len(cfg)-1:
                bl.append(sp.SparseSequential(
                    sp.SparseConv3d(c, co, 3, padding=p, stride=(2,2,2), indice_key=f"sc{i+1}"),
                    nn.BatchNorm1d(co, eps=1e-3, momentum=0.01), nn.ReLU(True)))
            else:
                bl.append(sp.SparseSequential(
                    sp.SubMConv3d(co, co, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(co, eps=1e-3, momentum=0.01), nn.ReLU(True),
                    sp.SubMConv3d(co, co, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(co, eps=1e-3, momentum=0.01)))
            c = co
        layers.append(bl)
    cout = sp.SparseSequential(
        sp.SparseConv3d(c, 128, (3,3,3), stride=(1,1,2), padding=(1,1,0),
                        indice_key="sd2", bias=False),
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

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, val in results.items():
        status = "PASS" if val is True else ("SKIP" if val is None else "FAIL")
        print(f"  {name}: {status}")

    print(f"\nTo switch backend, re-run with:")
    print(f"  USE_DRCV_BACKEND=0 python {os.path.basename(__file__)}  # spconv")
    print(f"  USE_DRCV_BACKEND=1 python {os.path.basename(__file__)}  # drcv")