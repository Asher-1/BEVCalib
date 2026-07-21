"""
BEVCalib 部署 API

三阶段 Pipeline:
  Phase 1: TTA Warm-up (估计当前车辆的 Zero-Drift, 自动选择 Affine/LoRA)
  Phase 2: Iterative Calibration (迭代推理校正)
  Phase 3: Online ZD Tracking (持续更新 ZD 估计)

TTA 策略自动选择:
  - ZD 稳定 (CoV < 0.3): Affine TTA (6 params, <1s)
  - ZD 不稳定 (CoV >= 0.3): LoRA TTA (~2K params, ~30s)
  
Usage:
    from bevcalib_deploy import BEVCalibDeployment

    deployer = BEVCalibDeployment('path/to/checkpoint.pth')
    deployer.warmup(frames[:100])  # Phase 1 (auto-selects strategy)
    result = deployer.calibrate(frame)  # Phase 2
"""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))


@dataclass
class CalibResult:
    """Single frame calibration result."""
    T_calibrated: np.ndarray
    correction_euler_deg: np.ndarray
    confidence: float
    n_iterations: int
    residual_deg: float
    converged: bool


@dataclass
class WarmupResult:
    """TTA warmup result."""
    zd_estimate_deg: np.ndarray
    zd_std_deg: np.ndarray
    confidence: float
    n_frames_used: int
    is_stable: bool
    strategy: str = 'affine'  # 'affine' or 'lora'
    lora_weights_path: Optional[str] = None


class BEVCalibDeployment:
    """
    Production deployment wrapper for BEVCalib model.

    Implements the three-phase pipeline:
    1. TTA Warm-up: Estimate per-vehicle Zero-Drift
    2. Iterative Calibration: Multi-pass correction with ZD compensation
    3. Online ZD Tracking: Continuous drift monitoring
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda',
                 target_width: int = 960, target_height: int = 540,
                 max_iterations: int = 5, convergence_threshold_deg: float = 0.05,
                 zd_tracking_window: int = 100, zd_ema_alpha: float = 0.1,
                 tta_strategy: str = 'auto',
                 lora_rank: int = 4,
                 zd_cov_threshold: float = 0.3):
        """
        Args:
            tta_strategy: 'auto'|'affine'|'lora'|'none'. auto = 根据 ZD 稳定性自动选择
            lora_rank: LoRA rank when using LoRA TTA
            zd_cov_threshold: ZD CoV (coefficient of variation) 阈值,
                              CoV < threshold → affine, else → lora
        """
        self.device = torch.device(device)
        self.target_width = target_width
        self.target_height = target_height
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold_deg * np.pi / 180
        self.zd_tracking_window = zd_tracking_window
        self.zd_ema_alpha = zd_ema_alpha
        self.tta_strategy = tta_strategy
        self.lora_rank = lora_rank
        self.zd_cov_threshold = zd_cov_threshold

        self._zd_bias = torch.zeros(3, device=self.device)
        self._zd_scale = torch.ones(3, device=self.device)
        self._is_adapted = False
        self._active_strategy = 'none'
        self._lora_tta = None
        self._zd_history: List[torch.Tensor] = []
        self._cached_zd_path: Optional[str] = None

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        import argparse
        from train_kitti import build_model_from_args

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_args = ckpt.get('args', {})
        if isinstance(saved_args, dict):
            self._model_args = argparse.Namespace(**saved_args)
        else:
            self._model_args = saved_args

        self._model_args.target_width = self.target_width
        self._model_args.target_height = self.target_height

        self._model = build_model_from_args(self._model_args)
        self._model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self._model = self._model.to(self.device).eval()

        self._model_epoch = ckpt.get('epoch', -1)
        print(f"[BEVCalibDeploy] Model loaded from epoch {self._model_epoch}")
        print(f"[BEVCalibDeploy] Resolution: {self.target_width}x{self.target_height}")

    @torch.no_grad()
    def warmup(self, frames: list, n_frames: int = 100,
               cached_zd_path: Optional[str] = None,
               lora_weights_path: Optional[str] = None) -> WarmupResult:
        """
        Phase 1: TTA Warm-up with automatic strategy selection.

        Steps:
        1. Collect predictions on unperturbed frames
        2. Compute ZD statistics (mean, std, CoV)
        3. Auto-select strategy based on CoV:
           - CoV < threshold → Affine (fast, stable ZD)
           - CoV >= threshold → LoRA (powerful, variable ZD)
        4. Execute chosen TTA
        """
        if cached_zd_path and os.path.exists(cached_zd_path):
            cached = np.load(cached_zd_path, allow_pickle=True)
            self._zd_bias = torch.from_numpy(cached['bias']).float().to(self.device)
            self._zd_scale = torch.from_numpy(
                cached['scale'] if 'scale' in cached else np.ones(3)
            ).float().to(self.device)
            self._is_adapted = True
            self._active_strategy = str(cached.get('strategy', 'affine'))
            self._cached_zd_path = cached_zd_path

            if self._active_strategy == 'lora' and lora_weights_path and os.path.exists(lora_weights_path):
                self._apply_lora_from_file(lora_weights_path)

            return WarmupResult(
                zd_estimate_deg=cached['bias'] * 180 / np.pi,
                zd_std_deg=cached.get('std', np.zeros(3)) * 180 / np.pi,
                confidence=float(cached.get('confidence', 0.8)),
                n_frames_used=int(cached.get('n_frames', 0)),
                is_stable=True,
                strategy=self._active_strategy,
                lora_weights_path=lora_weights_path
            )

        predictions = []
        use_frames = min(n_frames, len(frames))

        for frame in frames[:use_frames]:
            euler = self._forward_single(frame)
            if euler is not None:
                predictions.append(euler)

        if len(predictions) < 5:
            print(f"[BEVCalibDeploy] WARNING: Only {len(predictions)} valid frames")
            self._is_adapted = False
            return WarmupResult(
                zd_estimate_deg=np.zeros(3), zd_std_deg=np.zeros(3),
                confidence=0.0, n_frames_used=len(predictions),
                is_stable=False, strategy='none'
            )

        pred_tensor = torch.stack(predictions)
        mean_pred = pred_tensor.mean(dim=0)
        std_pred = pred_tensor.std(dim=0)

        cov = self._compute_zd_cov(mean_pred, std_pred)
        chosen_strategy = self._select_strategy(cov)

        print(f"[BEVCalibDeploy] ZD CoV={cov:.3f} → strategy='{chosen_strategy}' "
              f"(threshold={self.zd_cov_threshold})")

        if chosen_strategy == 'lora':
            result = self._warmup_lora(frames[:use_frames], predictions,
                                       mean_pred, std_pred, cov,
                                       lora_weights_path)
        else:
            result = self._warmup_affine(predictions, mean_pred, std_pred, cov)

        if cached_zd_path:
            os.makedirs(os.path.dirname(cached_zd_path) or '.', exist_ok=True)
            np.savez(cached_zd_path,
                     bias=self._zd_bias.cpu().numpy(),
                     std=std_pred.cpu().numpy(),
                     scale=self._zd_scale.cpu().numpy(),
                     confidence=result.confidence,
                     n_frames=len(predictions),
                     strategy=self._active_strategy,
                     cov=cov)
            self._cached_zd_path = cached_zd_path

        return result

    def _compute_zd_cov(self, mean: torch.Tensor, std: torch.Tensor) -> float:
        """Compute Coefficient of Variation for ZD stability assessment."""
        mean_abs = mean.abs().mean().item()
        std_mean = std.mean().item()
        if mean_abs < 1e-5:
            return std_mean / 0.001
        return std_mean / mean_abs

    def _select_strategy(self, cov: float) -> str:
        """Auto-select TTA strategy based on ZD variability."""
        if self.tta_strategy != 'auto':
            return self.tta_strategy
        if cov < self.zd_cov_threshold:
            return 'affine'
        return 'lora'

    def _warmup_affine(self, predictions: list, mean_pred: torch.Tensor,
                       std_pred: torch.Tensor, cov: float) -> WarmupResult:
        """Affine TTA: simple bias subtraction (for stable ZD)."""
        self._zd_bias = mean_pred
        self._is_adapted = True
        self._active_strategy = 'affine'

        confidence = max(0.0, min(1.0, 1.0 - cov))

        result = WarmupResult(
            zd_estimate_deg=(mean_pred.cpu().numpy() * 180 / np.pi),
            zd_std_deg=(std_pred.cpu().numpy() * 180 / np.pi),
            confidence=confidence,
            n_frames_used=len(predictions),
            is_stable=True,
            strategy='affine'
        )
        print(f"[BEVCalibDeploy] Affine warmup: "
              f"ZD=[{result.zd_estimate_deg[0]:.3f}, {result.zd_estimate_deg[1]:.3f}, "
              f"{result.zd_estimate_deg[2]:.3f}]° (conf={confidence:.2f})")
        return result

    def _warmup_lora(self, frames: list, predictions: list,
                     mean_pred: torch.Tensor, std_pred: torch.Tensor,
                     cov: float, save_path: Optional[str] = None) -> WarmupResult:
        """LoRA TTA: adapt model for variable ZD patterns."""
        from lora_tta import LoRATTA

        self._lora_tta = LoRATTA(self._model, rank=self.lora_rank,
                                  target_patterns=['corr_head', 'head.fc', 'decoder'])
        history = self._lora_tta.adapt(frames, device=self.device,
                                        lr=1e-3, steps=50)
        self._lora_tta.merge()
        self._lora_tta = None

        re_predictions = []
        for frame in frames[:50]:
            euler = self._forward_single(frame)
            if euler is not None:
                re_predictions.append(euler)

        if re_predictions:
            new_mean = torch.stack(re_predictions).mean(dim=0)
            new_std = torch.stack(re_predictions).std(dim=0)
        else:
            new_mean = mean_pred
            new_std = std_pred

        self._zd_bias = new_mean
        self._is_adapted = True
        self._active_strategy = 'lora'

        final_loss = history['loss'][-1] if history['loss'] else 1.0
        confidence = max(0.0, min(1.0, 1.0 - final_loss))

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            # LoRA已merge, 保存整个模型的相关权重
            torch.save({
                'zd_bias': self._zd_bias.cpu(),
                'history': history,
                'cov': cov,
            }, save_path)

        result = WarmupResult(
            zd_estimate_deg=(new_mean.cpu().numpy() * 180 / np.pi),
            zd_std_deg=(new_std.cpu().numpy() * 180 / np.pi),
            confidence=confidence,
            n_frames_used=len(frames),
            is_stable=False,
            strategy='lora',
            lora_weights_path=save_path
        )
        print(f"[BEVCalibDeploy] LoRA warmup: "
              f"ZD=[{result.zd_estimate_deg[0]:.3f}, {result.zd_estimate_deg[1]:.3f}, "
              f"{result.zd_estimate_deg[2]:.3f}]° "
              f"(final_loss={final_loss:.5f}, conf={confidence:.2f})")
        return result

    def _apply_lora_from_file(self, path: str):
        """Load previously saved LoRA adaptation state."""
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
            if 'zd_bias' in state:
                self._zd_bias = state['zd_bias'].to(self.device)
            print(f"[BEVCalibDeploy] Loaded LoRA state from {path}")
        except Exception as e:
            print(f"[BEVCalibDeploy] Failed to load LoRA state: {e}")

    @torch.no_grad()
    def calibrate(self, frame, T_init: Optional[np.ndarray] = None) -> CalibResult:
        """
        Phase 2: Iterative Calibration

        Run multi-pass inference with ZD compensation until convergence.
        """
        if not self._is_adapted:
            print("[BEVCalibDeploy] WARNING: Not adapted yet! Call warmup() first.")

        T_current = T_init if T_init is not None else frame.get('init_T', np.eye(4))
        total_correction = np.zeros(3)
        converged = False

        for it in range(self.max_iterations):
            euler = self._forward_single(frame, T_override=T_current)
            if euler is None:
                break

            corrected = (euler * self._zd_scale - self._zd_bias)
            correction_rad = corrected.cpu().numpy()
            correction_deg = correction_rad * 180 / np.pi

            residual = np.sqrt(np.sum(correction_deg ** 2))
            if residual < self.convergence_threshold * 180 / np.pi:
                converged = True
                break

            T_current = self._apply_euler_correction(T_current, correction_rad)
            total_correction += correction_deg

        self._zd_history.append(euler.detach() if euler is not None else self._zd_bias)
        if len(self._zd_history) > self.zd_tracking_window * 2:
            self._zd_history = self._zd_history[-self.zd_tracking_window:]

        return CalibResult(
            T_calibrated=T_current,
            correction_euler_deg=total_correction,
            confidence=1.0 - residual / 3.0 if residual else 1.0,
            n_iterations=it + 1,
            residual_deg=residual,
            converged=converged
        )

    def update_zd_tracking(self):
        """
        Phase 3: Online ZD Tracking

        Update ZD estimate using exponential moving average.
        Call periodically (e.g., every 100 frames).
        """
        if len(self._zd_history) < 10:
            return

        recent = torch.stack(self._zd_history[-self.zd_tracking_window:])
        new_bias = recent.mean(dim=0)

        self._zd_bias = (1 - self.zd_ema_alpha) * self._zd_bias + self.zd_ema_alpha * new_bias

        if self._cached_zd_path:
            np.savez(self._cached_zd_path,
                     bias=self._zd_bias.cpu().numpy(),
                     std=recent.std(dim=0).cpu().numpy(),
                     scale=self._zd_scale.cpu().numpy(),
                     confidence=0.9,
                     n_frames=len(self._zd_history))

    def _forward_single(self, frame, T_override=None) -> Optional[torch.Tensor]:
        """Run single forward pass, return euler prediction in radians."""
        try:
            if isinstance(frame, dict):
                imgs = frame['image'].unsqueeze(0).to(self.device) if frame['image'].dim() == 3 else frame['image'].to(self.device)
                pcs = frame['pointcloud'].unsqueeze(0).to(self.device) if frame['pointcloud'].dim() == 2 else frame['pointcloud'].to(self.device)
                init_T = torch.from_numpy(T_override).float().unsqueeze(0).to(self.device) if T_override is not None else frame['init_T'].unsqueeze(0).to(self.device)
                intrinsic = frame['intrinsic'].unsqueeze(0).to(self.device) if frame['intrinsic'].dim() == 2 else frame['intrinsic'].to(self.device)
                masks = frame.get('mask', None)
                if masks is not None:
                    masks = masks.unsqueeze(0).to(self.device) if masks.dim() == 2 else masks.to(self.device)
            else:
                return None

            output = self._model(imgs, pcs, init_T, intrinsic, masks)
            if isinstance(output, dict):
                quat = output.get('quat', output.get('rotation'))
            else:
                quat = output[0] if isinstance(output, (list, tuple)) else output

            return self._quat_to_euler(quat.squeeze(0))

        except Exception as e:
            print(f"[BEVCalibDeploy] Forward error: {e}")
            return None

    @staticmethod
    def _quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
        """(4,) wxyz quaternion → (3,) euler rad."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = torch.clamp(2*(w*y - z*x), -1, 1)
        pitch = torch.asin(sinp)
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return torch.stack([roll, pitch, yaw])

    @staticmethod
    def _apply_euler_correction(T_current: np.ndarray, correction_rad: np.ndarray) -> np.ndarray:
        """Apply euler angle correction to transformation matrix."""
        from scipy.spatial.transform import Rotation as R
        correction_rot = R.from_euler('xyz', correction_rad).as_matrix()
        T_new = T_current.copy()
        T_new[:3, :3] = correction_rot @ T_current[:3, :3]
        return T_new

    @property
    def is_adapted(self) -> bool:
        return self._is_adapted

    @property
    def zd_estimate_deg(self) -> np.ndarray:
        return self._zd_bias.cpu().numpy() * 180 / np.pi

    def get_status(self) -> dict:
        return {
            'adapted': self._is_adapted,
            'strategy': self._active_strategy,
            'zd_deg': self.zd_estimate_deg.tolist(),
            'zd_history_len': len(self._zd_history),
            'model_epoch': self._model_epoch,
            'resolution': f"{self.target_width}x{self.target_height}",
            'max_iterations': self.max_iterations,
            'tta_config': {
                'mode': self.tta_strategy,
                'cov_threshold': self.zd_cov_threshold,
                'lora_rank': self.lora_rank,
            }
        }
