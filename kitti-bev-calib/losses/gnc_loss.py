"""Graduated Non-Convexity (GNC) robust loss — aligned with NPC / classical GNC-GM.

Reference: maijiayao1/NPC (ICLR 2026), Environment/GNC_CostFactor_PointCloudRegistration.py
  - target_f:       rho = c * r^2 / (r^2 + c)
  - GNC_GM_weight:  w = (mu * c / (r^2 + mu * c))^2
  - mu init:        mu = min(2 * max(r^2) / c, cap)
  - mu anneal:      mu /= gnc_factor each outer step, stop when mu < 1

Training maps NPC outer-loop to epoch schedule; init residual uses T_init vs T_gt
(same information as classical GNC before the corrector inner loop).
"""

import math
from typing import Optional, Sequence, Tuple

import torch


def _euler_from_delta_R(R_err: torch.Tensor) -> torch.Tensor:
    """Extract roll/pitch/yaw (rad) from delta rotation matrix."""
    sy = torch.sqrt(R_err[:, 0, 0] ** 2 + R_err[:, 1, 0] ** 2)
    singular = sy < 1e-6
    roll = torch.where(
        singular,
        torch.atan2(-R_err[:, 1, 2], R_err[:, 1, 1]),
        torch.atan2(R_err[:, 2, 1], R_err[:, 2, 2]),
    )
    pitch = torch.where(
        singular,
        torch.atan2(-R_err[:, 2, 0], sy),
        torch.atan2(-R_err[:, 2, 0], sy),
    )
    yaw = torch.where(
        singular,
        torch.zeros_like(sy),
        torch.atan2(R_err[:, 1, 0], R_err[:, 0, 0]),
    )
    return torch.stack([roll, pitch, yaw], dim=1)


def axis_error_deg(R_gt: torch.Tensor, R_pred: torch.Tensor) -> torch.Tensor:
    """Signed per-axis error in degrees (LiDAR frame)."""
    R_err = torch.bmm(R_gt.float().transpose(1, 2), R_pred.float())
    return _euler_from_delta_R(R_err) * (180.0 / math.pi)


def gnc_noise_bound_sq(args) -> float:
    """Inlier squared-error bound c (deg^2), NPC noise_bound analogue."""
    deg = float(getattr(args, "gnc_noise_bound_deg", 0.05))
    return max(deg * deg, 1e-12)


def gnc_mu_end_homotopy(args, noise_bound_sq: float) -> float:
    """NPC termination scale (dimensionless homotopy, default 1.0)."""
    mu_end = float(getattr(args, "gnc_mu_end", 0.0))
    if mu_end > 0:
        return mu_end
    d1 = float(getattr(args, "gnc_mu_end_deg", 0.05))
    return max((d1 * d1) / float(noise_bound_sq), 1.0)


def init_mu_homotopy_from_r_max(
    r_max_deg: float,
    noise_bound_sq: float,
    mu_cap: float = 1e6,
) -> float:
    """NPC mu initialization: min(2 * max(r^2) / c, cap)."""
    r2 = float(r_max_deg) * float(r_max_deg)
    c = float(noise_bound_sq)
    mu = 2.0 * r2 / c
    return min(max(mu, 1.0), float(mu_cap))


def mu_start_from_config(args, noise_bound_sq: float, mu_cap: float) -> float:
    """Fallback mu_start when adaptive init is disabled."""
    mu_start = float(getattr(args, "gnc_mu_start", 0.0))
    if mu_start <= 0:
        d0 = float(getattr(args, "gnc_mu_start_deg", 2.0))
        mu_start = max((d0 * d0) / float(noise_bound_sq), 1.0)
    return min(max(mu_start, 1.0), float(mu_cap))


def gnc_npc_surrogate(
    r: torch.Tensor,
    mu_homotopy: float,
    noise_bound_sq: float,
) -> torch.Tensor:
    """NPC target_f surrogate: mu_eff * r^2 / (r^2 + mu_eff)."""
    mu_eff = float(mu_homotopy) * float(noise_bound_sq)
    r2 = r * r
    if mu_eff <= 0:
        return r2
    return mu_eff * r2 / (r2 + mu_eff + 1e-12)


def gnc_npc_weight(
    r: torch.Tensor,
    mu_homotopy: float,
    noise_bound_sq: float,
) -> torch.Tensor:
    """NPC GNC_GM_weight_update: (mu_eff / (r^2 + mu_eff))^2."""
    mu_eff = float(mu_homotopy) * float(noise_bound_sq)
    r2 = r * r
    w = mu_eff / (r2 + mu_eff + 1e-12)
    return w * w


def weighted_gnc_axis_mean(
    axis_err_deg: torch.Tensor,
    mu_homotopy: float,
    axis_weights: Sequence[float] = (1.0, 1.0, 1.0),
    noise_bound_sq: float = 0.0025,
    use_irls_weight: bool = False,
) -> torch.Tensor:
    """Weighted mean GNC cost over roll/pitch/yaw axis errors (degrees)."""
    r = axis_err_deg.abs()
    if use_irls_weight:
        costs = gnc_npc_weight(r, mu_homotopy, noise_bound_sq) * r * r
    else:
        costs = gnc_npc_surrogate(r, mu_homotopy, noise_bound_sq)
    w = torch.tensor(axis_weights, dtype=axis_err_deg.dtype, device=axis_err_deg.device)
    w = w / w.sum().clamp(min=1e-6) * 3.0
    return (costs * w.unsqueeze(0)).mean()


def parse_gnc_axis_weights(args, default: str = "1.0,1.0,1.0") -> Tuple[float, float, float]:
    raw = getattr(args, "gnc_axis_weights", "") or getattr(args, "axis_weights", default)
    parts = tuple(float(x) for x in str(raw).split(","))
    if len(parts) != 3:
        raise ValueError(f"Expected 3 axis weights, got {raw!r}")
    return parts


def effective_gnc_mu_homotopy(
    args,
    epoch: int,
    init_r_max_deg: Optional[float] = None,
) -> Optional[float]:
    """Dimensionless homotopy mu with NPC factor or legacy geometric schedule."""
    if int(getattr(args, "enable_gnc_loss", 0)) <= 0:
        return None
    start_ep = int(getattr(args, "gnc_start_epoch", 1))
    if epoch < start_ep:
        return None

    noise_bound_sq = gnc_noise_bound_sq(args)
    mu_cap = float(getattr(args, "gnc_mu_cap", 1e6))
    mu_end = gnc_mu_end_homotopy(args, noise_bound_sq)
    schedule = str(getattr(args, "gnc_schedule", "geometric")).lower()

    if int(getattr(args, "gnc_mu_adaptive_init", 0)) > 0 and init_r_max_deg is not None:
        if init_r_max_deg == init_r_max_deg and init_r_max_deg > 0:
            mu_start = init_mu_homotopy_from_r_max(init_r_max_deg, noise_bound_sq, mu_cap)
        else:
            mu_start = mu_start_from_config(args, noise_bound_sq, mu_cap)
    else:
        mu_start = mu_start_from_config(args, noise_bound_sq, mu_cap)

    step = max(0, epoch - start_ep)
    if schedule == "npc_factor":
        factor = float(getattr(args, "gnc_factor", 1.4))
        if factor <= 1.0:
            factor = 1.4
        mu = mu_start / (factor ** step)
        return max(mu, mu_end)

    anneal = int(getattr(args, "gnc_anneal_epochs", 100))
    t = min(1.0, (step + 1) / max(1, anneal))
    log_start = math.log(max(mu_start, 1e-6))
    log_end = math.log(max(mu_end, 1e-6))
    return math.exp(log_start + t * (log_end - log_start))


def effective_gnc_mu_deg(
    args,
    epoch: int,
    init_r_max_deg: Optional[float] = None,
) -> Optional[float]:
    """Returns mu_homotopy (dimensionless); name kept for log compatibility."""
    return effective_gnc_mu_homotopy(args, epoch, init_r_max_deg=init_r_max_deg)


def init_r_max_deg_from_matrices(R_gt: torch.Tensor, R_init: torch.Tensor) -> float:
    """Max |RPY| init error in degrees (NPC residual scale before optimization)."""
    with torch.no_grad():
        err = axis_error_deg(R_gt, R_init).abs()
        if err.numel() == 0:
            return 0.0
        return float(err.max().item())
