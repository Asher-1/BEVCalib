"""
CLAIM-inspired Calibration Refiner for BEVCalib
================================================
Pure PyTorch implementation of CLAIM's core: coarse-to-fine search with
structure + texture alignment losses.

Based on: Zhang et al., "CLAIM: Camera-LiDAR Alignment with Intensity
and Monodepth", IROS 2025.

Two loss components:
  - Structure loss: Patched Pearson correlation between LiDAR depth projection
    and monodepth (or edge-based proxy when monodepth is unavailable)
  - Texture loss: NID (Normalized Information Distance) between LiDAR intensity
    projection and grayscale image

Search strategy: Coarse-to-fine grid/random search over rotation parameters.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class FrameData:
    """Container for a single frame's data on GPU."""
    image: np.ndarray          # (H, W, 3) BGR
    pointcloud: torch.Tensor   # (N, 3) float32 on GPU
    intensity: Optional[torch.Tensor]  # (N,) float32 on GPU, or None
    K: torch.Tensor            # (3, 3) float32 on GPU
    image_gray: torch.Tensor   # (H, W) float32 on GPU, normalized [0, 1]
    mono_depth: Optional[torch.Tensor] = None  # (H, W) float32 on GPU


def project_points_batch(
    points: torch.Tensor,
    T_l2c: torch.Tensor,
    K: torch.Tensor,
    H: int, W: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch project LiDAR points to image plane.
    
    Args:
        points: (N, 3) in LiDAR frame
        T_l2c: (B, 4, 4) transform candidates
        K: (3, 3) intrinsics
        H, W: image dimensions
    Returns:
        uvd: (B, N, 3) where [:,:,0]=u, [:,:,1]=v, [:,:,2]=depth
        mask: (B, N) valid projections
    """
    B = T_l2c.shape[0]
    N = points.shape[0]
    
    pts_cam = torch.matmul(points, T_l2c[:, :3, :3].transpose(2, 1)) + \
              T_l2c[:, :3, 3:4].transpose(2, 1)  # (B, N, 3)
    
    uvs = torch.matmul(pts_cam, K.T)  # (B, N, 3)
    depth = pts_cam[:, :, 2:3]
    safe_depth = depth.clamp(min=1e-6)
    uvs = uvs / safe_depth
    uvs[:, :, 2] = depth.squeeze(-1)
    
    mask_d = depth.squeeze(-1) > 0.1
    mask_u = (uvs[:, :, 0] >= 0) & (uvs[:, :, 0] < W)
    mask_v = (uvs[:, :, 1] >= 0) & (uvs[:, :, 1] < H)
    mask = mask_d & mask_u & mask_v
    
    return uvs, mask


def generate_lidar_depth_image(
    uvd: torch.Tensor,
    mask: torch.Tensor,
    H: int, W: int,
) -> torch.Tensor:
    """Scatter LiDAR depth values onto a 2D image (batch-optimized).
    
    Returns: (B, H, W) depth images. Zero where no points projected.
    """
    B, N = uvd.shape[0], uvd.shape[1]
    depth_imgs = torch.zeros(B, H * W, device=uvd.device, dtype=uvd.dtype)
    
    u_all = uvd[:, :, 0].long().clamp(0, W - 1)
    v_all = uvd[:, :, 1].long().clamp(0, H - 1)
    d_all = uvd[:, :, 2]
    idx_all = v_all * W + u_all  # (B, N)
    
    d_masked = d_all * mask.float()
    
    for b in range(B):
        depth_imgs[b].scatter_(0, idx_all[b], d_masked[b])
    
    return depth_imgs.view(B, H, W)


def generate_lidar_intensity_image(
    uvd: torch.Tensor,
    mask: torch.Tensor,
    intensity: torch.Tensor,
    H: int, W: int,
) -> torch.Tensor:
    """Scatter LiDAR intensity values onto a 2D image (batch-optimized).
    
    Returns: (B, H, W) intensity images.
    """
    B, N = uvd.shape[0], uvd.shape[1]
    int_imgs = torch.zeros(B, H * W, device=uvd.device, dtype=uvd.dtype)
    
    u_all = uvd[:, :, 0].long().clamp(0, W - 1)
    v_all = uvd[:, :, 1].long().clamp(0, H - 1)
    idx_all = v_all * W + u_all
    
    int_expanded = intensity.unsqueeze(0).expand(B, -1)
    int_masked = int_expanded * mask.float()
    
    for b in range(B):
        int_imgs[b].scatter_(0, idx_all[b], int_masked[b])
    
    return int_imgs.view(B, H, W)


def pearson_loss_patched(
    lidar_depth: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
    shift: int = 0,
) -> torch.Tensor:
    """Patched Pearson correlation loss (CLAIM structure loss), fully vectorized.
    
    Args:
        lidar_depth: (B, H, W) LiDAR depth projection
        target: (H, W) monodepth or edge map
        patch_size: patch size for block correlation
        shift: offset for overlapping patches
    Returns:
        loss: (B,) per-candidate loss (lower = better alignment)
    """
    B, H, W = lidar_depth.shape
    p = patch_size
    
    nh = (H - shift) // p
    nw = (W - shift) // p
    if nh <= 0 or nw <= 0:
        return torch.zeros(B, device=lidar_depth.device)
    
    K = nh * nw
    ld_crop = lidar_depth[:, shift:shift + nh * p, shift:shift + nw * p]
    tg_crop = target[shift:shift + nh * p, shift:shift + nw * p]
    
    ld_patches = ld_crop.unfold(1, p, p).unfold(2, p, p)
    ld_patches = ld_patches.reshape(B, K, p * p)
    
    tg_patches = tg_crop.unfold(0, p, p).unfold(1, p, p)
    tg_patches = tg_patches.reshape(K, p * p).unsqueeze(0).expand(B, -1, -1)
    
    hit_mask = ld_patches.abs() > 1e-6
    hit_count = hit_mask.sum(dim=2)
    min_hits = max(4, p * p // 10)
    valid = hit_count >= min_hits
    
    ld_masked = ld_patches * hit_mask.float()
    tg_masked = tg_patches * hit_mask.float()
    
    n = hit_count.float().clamp(min=1)
    mean_x = ld_masked.sum(dim=2) / n
    mean_y = tg_masked.sum(dim=2) / n
    
    dx = (ld_masked - mean_x.unsqueeze(2)) * hit_mask.float()
    dy = (tg_masked - mean_y.unsqueeze(2)) * hit_mask.float()
    
    cov_xy = (dx * dy).sum(dim=2)
    var_x = (dx * dx).sum(dim=2)
    var_y = (dy * dy).sum(dim=2)
    
    denom = torch.sqrt(var_x * var_y).clamp(min=1e-8)
    pearson_r = cov_xy / denom
    
    losses = 1.0 - pearson_r
    losses[~valid] = 1.0
    
    low_var = (var_x < 1e-12) | (var_y < 1e-12)
    losses[low_var & valid] = 0.5
    
    return losses.mean(dim=1)


def nid_loss(
    lidar_intensity: torch.Tensor,
    image_gray: torch.Tensor,
    n_bins: int = 16,
) -> torch.Tensor:
    """Normalized Information Distance between LiDAR intensity and grayscale image.
    
    NID = (H(X,Y) - MI(X,Y)) / H(X,Y), where MI = H(X) + H(Y) - H(X,Y).
    Uses a shared valid mask (from batch=0) to avoid per-sample recomputation
    when most candidates project to similar pixel sets.
    
    Args:
        lidar_intensity: (B, H, W) LiDAR intensity projection (0 = no data)
        image_gray: (H, W) grayscale image [0, 1]
        n_bins: histogram bins
    Returns:
        loss: (B,) NID values
    """
    B, H, W = lidar_intensity.shape
    device = lidar_intensity.device
    
    losses = torch.ones(B, device=device)
    img_flat = image_gray.view(-1)
    
    for b in range(B):
        li_flat = lidar_intensity[b].view(-1)
        valid = li_flat > 1e-6
        n_valid = valid.sum()
        if n_valid < 100:
            continue
        
        x = li_flat[valid]
        y = img_flat[valid]
        
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min
        if x_range < 1e-8:
            losses[b] = 1.0
            continue
        x_norm = (x - x_min) / x_range
        
        x_bin = (x_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
        y_bin = (y * (n_bins - 1)).long().clamp(0, n_bins - 1)
        
        N = n_valid.float()
        ones = torch.ones(n_valid, device=device)
        
        hist_x = torch.zeros(n_bins, device=device).scatter_add_(0, x_bin, ones)
        hist_y = torch.zeros(n_bins, device=device).scatter_add_(0, y_bin, ones)
        joint_idx = x_bin * n_bins + y_bin
        hist_xy_flat = torch.zeros(n_bins * n_bins, device=device).scatter_add_(0, joint_idx, ones)
        
        eps = 1e-8
        px = hist_x / N + eps
        py = hist_y / N + eps
        pxy = hist_xy_flat / N + eps
        
        Hx = -(px * px.log()).sum()
        Hy = -(py * py.log()).sum()
        Hxy = -(pxy * pxy.log()).sum()
        
        MI = Hx + Hy - Hxy
        losses[b] = (Hxy - MI) / (Hxy + eps)
    
    return losses


def euler_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles (radians) to rotation matrices.
    
    Args:
        angles: (..., 3) Euler angles
    Returns:
        R: (..., 3, 3) rotation matrices
    """
    shape = angles.shape[:-1]
    rx, ry, rz = angles[..., 0], angles[..., 1], angles[..., 2]
    
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    
    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)
    
    Rx = torch.stack([ones, zeros, zeros,
                      zeros, cx, -sx,
                      zeros, sx, cx], dim=-1).reshape(*shape, 3, 3)
    Ry = torch.stack([cy, zeros, sy,
                      zeros, ones, zeros,
                      -sy, zeros, cy], dim=-1).reshape(*shape, 3, 3)
    Rz = torch.stack([cz, -sz, zeros,
                      sz, cz, zeros,
                      zeros, zeros, ones], dim=-1).reshape(*shape, 3, 3)
    
    return Rz @ Ry @ Rx


def matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """Extract XYZ Euler angles from rotation matrix.
    
    Args:
        R: (..., 3, 3) rotation matrix
    Returns:
        angles: (..., 3) Euler angles in radians
    """
    sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = sy < 1e-6
    
    rx = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    ry = torch.atan2(-R[..., 2, 0], sy)
    rz = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    
    rx_s = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
    rz_s = torch.zeros_like(rz)
    
    rx = torch.where(singular, rx_s, rx)
    rz = torch.where(singular, rz_s, rz)
    
    return torch.stack([rx, ry, rz], dim=-1)


class CLAIMRefiner:
    """CLAIM-inspired calibration refiner with coarse-to-fine search.
    
    Operates as a post-processor on BEVCalib predictions:
      1. Take BEVCalib's predicted T_pred as initial guess
      2. Search nearby rotation space using structure + texture losses
      3. Return refined calibration
    """
    
    def __init__(
        self,
        mode: str = "finetune_rotation",
        search_mode: str = "random",
        rot_range_deg: float = 1.0,
        rot_resolution_deg: float = 0.05,
        n_iters: int = 300,
        patch_size: int = 96,
        intensity_equalize: bool = True,
        gray_equalize: bool = True,
        max_points: int = 50000,
        half_resolution: bool = True,
        device: str = "cuda",
        verbose: bool = False,
    ):
        self.mode = mode
        self.search_mode = search_mode
        self.rot_range_deg = rot_range_deg
        self.rot_resolution_deg = rot_resolution_deg
        self.n_iters = n_iters
        self.patch_size = patch_size
        self.intensity_equalize = intensity_equalize
        self.gray_equalize = gray_equalize
        self.max_points = max_points
        self.half_resolution = half_resolution
        self.device = device
        self.verbose = verbose
        self.mono_depth_model = None
    
    def load_mono_depth_model(self, model_path: str):
        """Load DepthAnything-V2 model (optional, enhances structure loss)."""
        raise NotImplementedError(
            "DepthAnything-V2 not available in current environment. "
            "Using edge-based proxy for structure loss."
        )
    
    def _prepare_frame(
        self,
        image: np.ndarray,
        point_cloud: np.ndarray,
        K: np.ndarray,
    ) -> FrameData:
        """Convert BEVCalib data to FrameData."""
        K_used = K.copy()
        if self.half_resolution:
            image = image[::2, ::2]
            K_used = K.copy()
            K_used[0, :] /= 2
            K_used[1, :] /= 2
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.gray_equalize:
            gray = cv2.equalizeHist(gray)
        
        pts = point_cloud[:, :3].copy()
        if len(pts) > self.max_points:
            step = max(1, len(pts) // self.max_points)
            idx = np.arange(0, len(pts), step)[:self.max_points]
            pts = pts[idx]
            point_cloud = point_cloud[idx]
        
        intensity = None
        if point_cloud.shape[1] >= 4:
            raw_int = point_cloud[:, 3].copy()
            if self.intensity_equalize:
                order = np.argsort(raw_int)
                equalized = np.zeros_like(raw_int)
                n = len(raw_int)
                for rank, orig_idx in enumerate(order):
                    equalized[orig_idx] = rank / n
                equalized = np.clip(equalized, 1e-3, 1.0)
                intensity = torch.from_numpy(equalized).float().to(self.device)
            else:
                i_min, i_max = raw_int.min(), raw_int.max()
                if i_max - i_min > 1e-6:
                    raw_int = (raw_int - i_min) / (i_max - i_min)
                raw_int = np.clip(raw_int, 1e-3, 1.0)
                intensity = torch.from_numpy(raw_int).float().to(self.device)
        
        mono_depth = None
        if self.mono_depth_model is not None:
            pass  # mono_depth = self.mono_depth_model(image)
        
        return FrameData(
            image=image,
            pointcloud=torch.from_numpy(pts).float().to(self.device),
            intensity=intensity,
            K=torch.from_numpy(K_used.astype(np.float64)).float().to(self.device),
            image_gray=torch.from_numpy(gray.astype(np.float32) / 255.0).to(self.device),
            mono_depth=mono_depth,
        )
    
    def _compute_edge_target(self, image: np.ndarray) -> torch.Tensor:
        """Compute edge-based proxy for structure loss (when monodepth unavailable).
        
        Uses Sobel gradient magnitude as a proxy for depth discontinuities.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        sx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        sy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        edge_mag = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
        v_max = edge_mag.max()
        if v_max > 1e-6:
            edge_mag /= v_max
        return torch.from_numpy(edge_mag).float().to(self.device)
    
    def _compute_loss(
        self,
        T_candidates: torch.Tensor,
        frame: FrameData,
        edge_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined structure + texture loss for all candidates.
        
        Args:
            T_candidates: (B, 4, 4) candidate transforms
            frame: FrameData
            edge_target: (H, W) edge magnitude or monodepth
        Returns:
            total_loss: (B,)
        """
        H, W = frame.image.shape[:2]
        B = T_candidates.shape[0]
        p = self.patch_size
        
        uvd, mask = project_points_batch(
            frame.pointcloud, T_candidates, frame.K, H, W
        )
        
        depth_imgs = generate_lidar_depth_image(uvd, mask, H, W)
        
        target = frame.mono_depth if frame.mono_depth is not None else edge_target
        loss_s1 = pearson_loss_patched(depth_imgs, target, p, shift=0)
        loss_s2 = pearson_loss_patched(depth_imgs, target, p, shift=p // 2)
        structure_loss = (loss_s1 + loss_s2) / 5.0
        
        texture_loss = torch.zeros(B, device=self.device)
        if frame.intensity is not None:
            int_imgs = generate_lidar_intensity_image(uvd, mask, frame.intensity, H, W)
            texture_loss = nid_loss(int_imgs, frame.image_gray, n_bins=16)
        
        return structure_loss + texture_loss
    
    def _grid_search(
        self,
        T_init: torch.Tensor,
        frames: List[FrameData],
        edge_targets: List[torch.Tensor],
    ) -> torch.Tensor:
        """Grid search over rotation space."""
        rng = self.rot_range_deg
        res = self.rot_resolution_deg
        
        n = int(2 * rng / res) + 1
        vals = torch.linspace(-rng, rng, n, device=self.device)
        vals_rad = torch.deg2rad(vals)
        
        gx, gy, gz = torch.meshgrid(vals_rad, vals_rad, vals_rad, indexing='ij')
        perturbations = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
        
        init_angles = matrix_to_euler(T_init[:3, :3])
        
        N = perturbations.shape[0]
        batch_size = min(256, N)
        
        best_T = T_init.clone()
        best_score = float('inf')
        
        n_batches = (N + batch_size - 1) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = min(N, start + batch_size)
            B = end - start
            
            angles = init_angles.unsqueeze(0) + perturbations[start:end]
            T_cands = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()
            T_cands[:, :3, :3] = euler_to_matrix(angles)
            T_cands[:, :3, 3] = T_init[:3, 3]
            
            total_loss = torch.zeros(B, device=self.device)
            for frame, edge_tgt in zip(frames, edge_targets):
                total_loss += self._compute_loss(T_cands, frame, edge_tgt)
            
            min_idx = torch.argmin(total_loss)
            if total_loss[min_idx] < best_score:
                best_score = total_loss[min_idx].item()
                best_T = T_cands[min_idx].clone()
            
            if self.verbose and (i % max(1, n_batches // 10) == 0):
                print(f"  Grid search: {i}/{n_batches}, best_score={best_score:.4f}")
        
        if self.verbose:
            print(f"  Grid search done: {N} candidates, best_score={best_score:.4f}")
        
        return best_T
    
    def _random_search(
        self,
        T_init: torch.Tensor,
        frames: List[FrameData],
        edge_targets: List[torch.Tensor],
        fine: bool = False,
    ) -> torch.Tensor:
        """Random search: fixed candidate set around current best (CLAIM-style)."""
        if fine:
            ang_degs = torch.tensor([-0.1, -0.04, -0.02, 0.02, 0.04, 0.1],
                                    device=self.device)
        else:
            ang_degs = torch.tensor([-0.5, -0.2, -0.1, 0.1, 0.2, 0.5],
                                    device=self.device)
        
        ang_rads = torch.deg2rad(ang_degs)
        gx, gy, gz = torch.meshgrid(ang_rads, ang_rads, ang_rads, indexing='ij')
        perturbations = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
        B = perturbations.shape[0]  # 6³ = 216
        
        best_T = T_init.clone()
        best_score = float('inf')
        
        for it in range(self.n_iters):
            cur_angles = matrix_to_euler(best_T[:3, :3])
            angles = cur_angles.unsqueeze(0) + perturbations  # (B, 3)
            
            T_cands = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()
            T_cands[:, :3, :3] = euler_to_matrix(angles)
            T_cands[:, :3, 3] = T_init[:3, 3]
            
            total_loss = torch.zeros(B, device=self.device)
            for frame, edge_tgt in zip(frames, edge_targets):
                total_loss += self._compute_loss(T_cands, frame, edge_tgt)
            
            min_idx = torch.argmin(total_loss)
            if total_loss[min_idx] < best_score:
                best_score = total_loss[min_idx].item()
                best_T = T_cands[min_idx].clone()
            
            if self.verbose and (it % max(1, self.n_iters // 10) == 0):
                print(f"  Random search {'(fine)' if fine else '(coarse)'}: "
                      f"iter {it}/{self.n_iters}, best_score={best_score:.4f}")
        
        if self.verbose:
            print(f"  Random search done: {self.n_iters} iterations, "
                  f"best_score={best_score:.4f}")
        
        return best_T
    
    def refine(
        self,
        point_cloud: np.ndarray,
        image: np.ndarray,
        K: np.ndarray,
        T_pred: np.ndarray,
    ) -> np.ndarray:
        """Refine BEVCalib prediction using CLAIM-style optimization.
        
        Args:
            point_cloud: (N, 4) with x,y,z,intensity
            image: (H, W, 3) BGR image
            K: (3, 3) camera intrinsics
            T_pred: (4, 4) BEVCalib predicted extrinsic (lidar-to-camera)
        Returns:
            T_refined: (4, 4) refined extrinsic
        """
        frame = self._prepare_frame(image, point_cloud, K)
        edge_target = self._compute_edge_target(image)
        frames = [frame]
        edge_targets = [edge_target]
        
        T_init = torch.from_numpy(T_pred.astype(np.float64)).float().to(self.device)
        
        if self.search_mode == "grid":
            T_refined = self._grid_search(T_init, frames, edge_targets)
        elif self.search_mode == "random":
            T_refined = self._random_search(T_init, frames, edge_targets, fine=False)
            T_refined = self._random_search(T_refined, frames, edge_targets, fine=True)
        elif self.search_mode == "hybrid":
            T_refined = self._grid_search(T_init, frames, edge_targets)
            T_refined = self._random_search(T_refined, frames, edge_targets, fine=True)
        else:
            raise ValueError(f"Unknown search mode: {self.search_mode}")
        
        return T_refined.cpu().numpy()
    
    def refine_multi_frame(
        self,
        point_clouds: List[np.ndarray],
        images: List[np.ndarray],
        K: np.ndarray,
        T_pred: np.ndarray,
    ) -> np.ndarray:
        """Multi-frame refinement for improved robustness.
        
        Uses multiple frames from the same sequence for joint optimization.
        """
        frames = []
        edge_targets = []
        for pc, img in zip(point_clouds, images):
            frames.append(self._prepare_frame(img, pc, K))
            edge_targets.append(self._compute_edge_target(img))
        
        T_init = torch.from_numpy(T_pred.astype(np.float64)).float().to(self.device)
        
        if self.search_mode == "grid":
            T_refined = self._grid_search(T_init, frames, edge_targets)
        else:
            T_refined = self._random_search(T_init, frames, edge_targets, fine=False)
            T_refined = self._random_search(T_refined, frames, edge_targets, fine=True)
        
        return T_refined.cpu().numpy()


def test_claim_refiner():
    """Quick smoke test with synthetic data."""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refiner = CLAIMRefiner(
        search_mode="random",
        rot_range_deg=1.0,
        n_iters=10,
        patch_size=48,
        device=device,
        verbose=True,
    )
    
    H, W = 480, 640
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    N = 10000
    pc = np.random.randn(N, 4).astype(np.float32)
    pc[:, 0] = np.abs(pc[:, 0]) * 10 + 5  # forward
    pc[:, 1] *= 5   # lateral
    pc[:, 2] *= 2   # vertical
    pc[:, 3] = np.abs(pc[:, 3])  # intensity
    
    K = np.array([[500, 0, W / 2],
                  [0, 500, H / 2],
                  [0, 0, 1]], dtype=np.float64)
    
    T_gt = np.eye(4)
    T_pred = T_gt.copy()
    R_pert = R_scipy.from_euler('xyz', [0.5, -0.3, 0.2], degrees=True).as_matrix()
    T_pred[:3, :3] = R_pert @ T_pred[:3, :3]
    
    t0 = time.time()
    T_refined = refiner.refine(pc, image, K, T_pred)
    elapsed = time.time() - t0
    
    print(f"\nTest completed in {elapsed:.1f}s")
    print(f"T_pred rotation:\n{T_pred[:3,:3]}")
    print(f"T_refined rotation:\n{T_refined[:3,:3]}")
    print("PASS: CLAIMRefiner runs without errors")


if __name__ == "__main__":
    test_claim_refiner()
