"""ZD (Zero-Drift) Online Compensator for BEVCalib deployment.

Pure deployment-side module that reduces systematic zero-drift without
retraining the model.

Key insight from V66 DINOv2 evaluation:
  - Per-frame Rot: 1.140°
  - Temporal aggregation (W=800): 0.197°
  - Remaining error is systematic bias (0.1°-0.9° per sequence)
  - Seq 03: bias=1.047° (worst), Seq 09: bias=0.113° (best)

Two operating modes:

Mode 1: Online bias tracking (no GT needed)
  - Track EMA of prediction in axis-angle space
  - The EMA converges to (true_calibration + bias)
  - If we assume the true calibration is constant per sequence,
    the EMA is our best estimate → use it directly
  - This is equivalent to temporal aggregation but in a causal filter

Mode 2: Bias-corrected tracking (GT available for calibration)
  - Use initial frames to estimate bias = EMA(pred) - GT
  - Subtract estimated bias from all future predictions
  - Requires GT for first N frames (calibration phase)

Usage:
    # Mode 1: Pure online (no GT)
    comp = ZDBiasCompensator(ema_alpha=0.05)
    T_comp = comp.update(T_pred, T_init)

    # Mode 2: With calibration
    comp = ZDBiasCompensator(ema_alpha=0.05, use_calibration=True)
    for T_pred, T_init, T_gt in stream:
        T_comp = comp.update(T_pred, T_init, T_gt)
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot


class ZDBiasCompensator:
    """Bias-aware ZD compensator using axis-angle EMA.

    Tracks the model's predictions in axis-angle space and estimates
    systematic bias. Can operate with or without ground truth.

    Args:
        ema_alpha: EMA smoothing factor (0, 1]. Lower = more stable.
            Default 0.05.
        max_correction_deg: Maximum bias correction in degrees.
            Safety clamp. Default 0.5°.
        use_calibration: If True, use GT frames to estimate bias.
            If False, use prediction EMA as best estimate.
            Default False.
        calibration_frames: Number of initial frames for bias calibration.
            Default 50.
        per_sequence: If True, maintain separate state per sequence.
            Default True.
        reset_on_large_jump: Reset EMA if prediction jumps by more than
            this many degrees. Default 5.0°.
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        max_correction_deg: float = 0.5,
        use_calibration: bool = False,
        calibration_frames: int = 50,
        per_sequence: bool = True,
        reset_on_large_jump: float = 5.0,
    ):
        self.ema_alpha = ema_alpha
        self.max_correction_deg = max_correction_deg
        self.use_calibration = use_calibration
        self.calibration_frames = calibration_frames
        self.per_sequence = per_sequence
        self.reset_on_large_jump = reset_on_large_jump

        # Per-sequence state
        self._states = {}  # seq_id → dict

    def _get_state(self, seq_id):
        if seq_id not in self._states:
            self._states[seq_id] = {
                'ema_aa': None,         # EMA in axis-angle (3,)
                'bias_aa': None,        # Estimated bias (3,)
                'frame_count': 0,
                'calib_pred_aa': [],    # Calibration predictions
                'calib_gt_aa': [],      # Calibration GT
                'initialized': False,
            }
        return self._states[seq_id]

    def reset(self, seq_id=None):
        """Reset state for a specific sequence or all."""
        if seq_id is None:
            self._states.clear()
        elif seq_id in self._states:
            del self._states[seq_id]

    def _update_ema(self, state, new_aa):
        """Update EMA with new axis-angle observation."""
        if state['ema_aa'] is None:
            state['ema_aa'] = new_aa.copy()
        else:
            # SLERP-like interpolation in axis-angle space
            # For small angles, linear interpolation is fine
            current = state['ema_aa']
            diff = new_aa - current

            # Handle angle wrapping (axis-angle can flip)
            # If the angle is close to pi, the representation is ambiguous
            angle_diff = np.linalg.norm(diff)
            if angle_diff > math.pi:
                # Large jump — likely scene change
                return False

            state['ema_aa'] = current + self.ema_alpha * diff

        return True

    def update(self, T_pred, T_init, T_gt=None, seq_id='default'):
        """Update compensator and return compensated prediction.

        Args:
            T_pred: (4, 4) or (1, 4, 4) model prediction
            T_init: (4, 4) or (1, 4, 4) input extrinsic
            T_gt: (4, 4) or (1, 4, 4) ground truth (optional, for calibration)
            seq_id: sequence identifier for per-sequence tracking

        Returns:
            T_compensated: (4, 4) compensated prediction
        """
        if T_pred.ndim == 3:
            T_pred = T_pred[0]
        if T_init.ndim == 3:
            T_init = T_init[0]
        if T_gt is not None and T_gt.ndim == 3:
            T_gt = T_gt[0]

        state = self._get_state(seq_id)
        state['frame_count'] += 1

        # Convert prediction to axis-angle
        pred_aa = ScipyRot.from_matrix(T_pred[:3, :3]).as_rotvec()

        # Check for large jump (scene change)
        if state['ema_aa'] is not None:
            jump = np.linalg.norm(pred_aa - state['ema_aa']) * 180 / math.pi
            if jump > self.reset_on_large_jump:
                state['ema_aa'] = pred_aa.copy()
                state['bias_aa'] = None
                state['calib_pred_aa'] = []
                state['calib_gt_aa'] = []
                return T_pred.copy()

        # Calibration phase
        if self.use_calibration and T_gt is not None:
            gt_aa = ScipyRot.from_matrix(T_gt[:3, :3]).as_rotvec()
            state['calib_pred_aa'].append(pred_aa)
            state['calib_gt_aa'].append(gt_aa)

            if len(state['calib_pred_aa']) >= self.calibration_frames and not state['initialized']:
                # Estimate bias
                mean_pred = np.mean(state['calib_pred_aa'], axis=0)
                mean_gt = np.mean(state['calib_gt_aa'], axis=0)
                state['bias_aa'] = mean_pred - mean_gt
                state['initialized'] = True
                bias_deg = np.linalg.norm(state['bias_aa']) * 180 / math.pi
                print(f"  [ZD] Seq {seq_id}: bias estimated = {bias_deg:.3f}° "
                      f"(from {self.calibration_frames} frames)")

        # Update EMA
        ok = self._update_ema(state, pred_aa)
        if not ok:
            return T_pred.copy()

        # Compute compensated prediction
        T_comp = T_pred.copy()

        if self.use_calibration and state['bias_aa'] is not None:
            # Mode 2: Subtract estimated bias
            corrected_aa = state['ema_aa'] - state['bias_aa']
            # Clamp correction magnitude
            correction = state['ema_aa'] - pred_aa
            corr_deg = np.linalg.norm(correction) * 180 / math.pi
            if corr_deg > self.max_correction_deg:
                correction = correction / corr_deg * self.max_correction_deg
            corrected_aa = pred_aa - state['bias_aa'] * (self.ema_alpha * state['frame_count'] / (1 + self.ema_alpha * state['frame_count']))
            # Clamp
            comp_deg = np.linalg.norm(corrected_aa - pred_aa) * 180 / math.pi
            if comp_deg > self.max_correction_deg:
                direction = (corrected_aa - pred_aa) / (comp_deg + 1e-8)
                corrected_aa = pred_aa + direction * self.max_correction_deg

            R_comp = ScipyRot.from_rotvec(corrected_aa).as_matrix()
            T_comp[:3, :3] = R_comp
            T_comp[:3, 3] = T_pred[:3, 3]

        elif not self.use_calibration:
            # Mode 1: Use EMA as best estimate (temporal filtering)
            # The EMA converges to the best estimate of true calibration
            # For early frames, blend between pred and EMA
            n = state['frame_count']
            blend = min(1.0, n * self.ema_alpha)  # Gradually trust EMA more
            use_aa = pred_aa + blend * (state['ema_aa'] - pred_aa)

            R_comp = ScipyRot.from_rotvec(use_aa).as_matrix()
            T_comp[:3, :3] = R_comp
            T_comp[:3, 3] = T_pred[:3, 3]

        return T_comp

    def get_stats(self, seq_id='default'):
        """Get compensator statistics."""
        state = self._get_state(seq_id)
        stats = {
            'frame_count': state['frame_count'],
            'ema_norm': np.linalg.norm(state['ema_aa']) * 180 / math.pi if state['ema_aa'] is not None else 0,
        }
        if state['bias_aa'] is not None:
            stats['bias_deg'] = np.linalg.norm(state['bias_aa']) * 180 / math.pi
        return stats


def evaluate_offline(input_npz, output_dir=None, params=None):
    """Offline evaluation of ZD compensation on pre-computed predictions.

    This function:
    1. Loads predictions and GT
    2. Applies ZD compensation per sequence
    3. Computes temporal aggregation with and without compensation
    4. Reports improvement

    Args:
        input_npz: path to all_T_pred_gt.npz
        output_dir: output directory (default: same as input)
        params: compensator parameters dict
    """
    import os

    if params is None:
        params = {
            'ema_alpha': 0.05,
            'max_correction_deg': 0.5,
            'use_calibration': True,
            'calibration_frames': 50,
        }

    data = np.load(input_npz)
    all_T_pred = data['all_T_pred']
    all_T_gt = data['all_T_gt']
    all_seqs = data['sample_sequences']

    unique_seqs = sorted(set(all_seqs))
    N = len(all_T_pred)

    print(f"Loaded {N} frames, {len(unique_seqs)} sequences")
    print(f"Parameters: {params}")

    # Apply compensation per sequence
    T_compensated = np.copy(all_T_pred)
    comp = ZDBiasCompensator(**params)

    for sid in unique_seqs:
        mask = all_seqs == sid
        indices = np.where(mask)[0]
        for idx in indices:
            T_comp = comp.update(
                all_T_pred[idx], all_T_pred[idx],  # T_init ≈ T_pred for offline
                T_gt=all_T_gt[idx],
                seq_id=sid,
            )
            T_compensated[idx] = T_comp

    # Compute per-frame errors
    def _compute_errors(T_pred, T_gt):
        R_err = T_pred[:, :3, :3] @ np.transpose(T_gt[:, :3, :3], (0, 2, 1))
        trace = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2]
        total = np.abs(np.arccos(np.clip((trace - 1) / 2, -1+1e-7, 1-1e-7))) * 180 / np.pi
        roll = np.abs(np.arctan2(R_err[:, 2, 1], R_err[:, 2, 2])) * 180 / np.pi
        pitch = np.abs(np.arctan2(-R_err[:, 2, 0],
                                  np.sqrt(R_err[:, 2, 1]**2 + R_err[:, 2, 2]**2))) * 180 / np.pi
        yaw = np.abs(np.arctan2(R_err[:, 1, 0], R_err[:, 0, 0])) * 180 / np.pi
        return total, roll, pitch, yaw

    base_total, base_r, base_p, base_y = _compute_errors(all_T_pred, all_T_gt)
    comp_total, comp_r, comp_p, comp_y = _compute_errors(T_compensated, all_T_gt)

    print(f"\nPer-frame errors:")
    print(f"  Baseline:   Rot={np.mean(base_total):.3f}°  R={np.mean(base_r):.3f}°  P={np.mean(base_p):.3f}°  Y={np.mean(base_y):.3f}°")
    print(f"  Compensated: Rot={np.mean(comp_total):.3f}°  R={np.mean(comp_r):.3f}°  P={np.mean(comp_p):.3f}°  Y={np.mean(comp_y):.3f}°")

    # Temporal aggregation comparison
    print(f"\nTemporal aggregation comparison:")
    print(f"{'Window':>8} | {'Baseline':>10} | {'Compensated':>12} | {'Improvement':>12}")
    print(f"{'-'*8} | {'-'*10} | {'-'*12} | {'-'*12}")

    for ws in [1, 5, 10, 20, 50, 100, 200, 400, 800]:
        base_errs = []
        comp_errs = []
        for sid in unique_seqs:
            mask = all_seqs == sid
            n = mask.sum()
            seq_base = all_T_pred[mask]
            seq_comp = T_compensated[mask]
            seq_gt = all_T_gt[mask]

            if ws >= n:
                # Aggregate entire sequence
                for T_src, errs_list in [(seq_base, base_errs), (seq_comp, comp_errs)]:
                    aa = ScipyRot.from_matrix(T_src[:, :3, :3]).as_rotvec()
                    R_avg = ScipyRot.from_rotvec(np.mean(aa, axis=0)).as_matrix()
                    R_gt = seq_gt[0, :3, :3]
                    R_e = R_avg @ R_gt.T
                    tr = np.clip((np.trace(R_e) - 1) / 2, -1+1e-7, 1-1e-7)
                    errs_list.append(np.abs(np.arccos(tr)) * 180 / np.pi)
            else:
                for start in range(0, n - ws + 1, max(1, ws // 2)):
                    end = start + ws
                    for T_src, errs_list in [(seq_base, base_errs), (seq_comp, comp_errs)]:
                        aa = ScipyRot.from_matrix(T_src[start:end, :3, :3]).as_rotvec()
                        R_avg = ScipyRot.from_rotvec(np.mean(aa, axis=0)).as_matrix()
                        gt_aa = ScipyRot.from_matrix(seq_gt[start:end, :3, :3]).as_rotvec()
                        R_gt = ScipyRot.from_rotvec(np.mean(gt_aa, axis=0)).as_matrix()
                        R_e = R_avg @ R_gt.T
                        tr = np.clip((np.trace(R_e) - 1) / 2, -1+1e-7, 1-1e-7)
                        errs_list.append(np.abs(np.arccos(tr)) * 180 / np.pi)

        base_mean = np.mean(base_errs) if base_errs else 0
        comp_mean = np.mean(comp_errs) if comp_errs else 0
        imp = base_mean - comp_mean
        pct = imp / base_mean * 100 if base_mean > 0 else 0
        marker = " ◄" if ws >= 100 else ""
        print(f"{ws:>8} | {base_mean:>9.3f}° | {comp_mean:>11.3f}° | {imp:>+10.3f}° ({pct:+.1f}%){marker}")

    # Per-sequence comparison
    print(f"\nPer-sequence analysis:")
    print(f"{'Seq':>6} | {'Base PF':>8} | {'Comp PF':>8} | {'Base Agg':>9} | {'Comp Agg':>9} | {'Δ Agg':>7}")
    print(f"{'-'*6} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*7}")

    for sid in unique_seqs:
        mask = all_seqs == sid
        seq_base = all_T_pred[mask]
        seq_comp = T_compensated[mask]
        seq_gt = all_T_gt[mask]

        # Per-frame
        base_pf = np.mean(base_total[mask])
        comp_pf = np.mean(comp_total[mask])

        # Aggregated (full sequence)
        base_aa = ScipyRot.from_matrix(seq_base[:, :3, :3]).as_rotvec()
        comp_aa = ScipyRot.from_matrix(seq_comp[:, :3, :3]).as_rotvec()
        gt_aa = ScipyRot.from_matrix(seq_gt[:, :3, :3]).as_rotvec()

        R_base = ScipyRot.from_rotvec(np.mean(base_aa, axis=0)).as_matrix()
        R_comp = ScipyRot.from_rotvec(np.mean(comp_aa, axis=0)).as_matrix()
        R_gt = ScipyRot.from_rotvec(np.mean(gt_aa, axis=0)).as_matrix()

        base_agg = np.abs(np.arccos(np.clip((np.trace(R_base @ R_gt.T) - 1) / 2, -1+1e-7, 1-1e-7))) * 180 / np.pi
        comp_agg = np.abs(np.arccos(np.clip((np.trace(R_comp @ R_gt.T) - 1) / 2, -1+1e-7, 1-1e-7))) * 180 / np.pi
        delta = comp_agg - base_agg

        marker = " ✓" if delta < -0.01 else (" ✗" if delta > 0.01 else "")
        print(f"{sid:>6} | {base_pf:>7.3f}° | {comp_pf:>7.3f}° | {base_agg:>8.3f}° | {comp_agg:>8.3f}° | {delta:>+6.3f}°{marker}")

    return T_compensated


if __name__ == '__main__':
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    if input_path is None:
        print("Usage: python zd_online_compensator.py <path_to_all_T_pred_gt.npz>")
        sys.exit(1)
    evaluate_offline(input_path)
