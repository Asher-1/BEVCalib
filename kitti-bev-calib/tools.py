import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import truncnorm


def _sample_truncated_normal(low, high, size, sigma_scale=0.5):
    """Sample from truncated normal distribution clipped to [low, high]."""
    mu = 0.0
    sigma = (high - low) * sigma_scale
    if sigma < 1e-9:
        return np.zeros(size)
    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)


def generate_single_perturbation_from_T(T, angle_range_deg=20, trans_range=1.5,
                                         rotation_only=False,
                                         distribution='uniform',
                                         per_axis_prob=0.0,
                                         curriculum_scale=1.0,
                                         per_axis_weights=None):
    """
    Vectorized batch perturbation with configurable distribution and per-axis mode.

    Parameters:
        T: np.ndarray, shape (B, 4, 4)
        angle_range_deg: float, rotation perturbation range in degrees
        trans_range: float, translation perturbation range in meters
        rotation_only: bool, if True skip translation perturbation
        distribution: 'uniform' or 'truncated_normal'
        per_axis_prob: probability of perturbing a single axis only (0=disabled)
        curriculum_scale: scale factor for perturbation range (0..1 for curriculum)
        per_axis_weights: tuple of 3 floats for (roll, pitch, yaw) sampling weights
                          in per-axis mode. None = uniform. e.g. (0.5, 0.3, 0.2)
    """
    B = T.shape[0]
    effective_angle = angle_range_deg * curriculum_scale
    effective_trans = trans_range * curriculum_scale

    orig_rots = R.from_matrix(T[:, :3, :3])
    orig_trans = T[:, :3, 3]

    use_per_axis = (per_axis_prob > 0) and (np.random.rand() < per_axis_prob)

    if use_per_axis:
        rotvecs = np.zeros((B, 3))
        if per_axis_weights is not None:
            w = np.array(per_axis_weights, dtype=np.float64)
            w = w / w.sum()
            axis_idx = np.random.choice(3, size=B, p=w)
        else:
            axis_idx = np.random.randint(0, 3, size=B)
        if distribution == 'truncated_normal':
            angles = _sample_truncated_normal(-effective_angle, effective_angle, B)
        else:
            angles = np.random.uniform(-effective_angle, effective_angle, B)
        angles_rad = np.deg2rad(angles)
        for i in range(B):
            rotvecs[i, axis_idx[i]] = angles_rad[i]
        delta_rots = R.from_rotvec(rotvecs)
    else:
        rand_axes = np.random.randn(B, 3)
        rand_axes /= np.linalg.norm(rand_axes, axis=1, keepdims=True)
        if distribution == 'truncated_normal':
            rand_angles = np.deg2rad(
                _sample_truncated_normal(-effective_angle, effective_angle, B))
        else:
            rand_angles = np.deg2rad(
                np.random.uniform(-effective_angle, effective_angle, B))
        delta_rots = R.from_rotvec(rand_axes * rand_angles[:, None])

    new_rots = delta_rots * orig_rots

    if not rotation_only:
        rand_dirs = np.random.randn(B, 3)
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        rand_magnitudes = np.random.uniform(0, effective_trans, B)
        new_trans = orig_trans + rand_dirs * rand_magnitudes[:, None]
    else:
        rand_magnitudes = np.zeros(B)
        new_trans = orig_trans.copy()

    T_new = np.broadcast_to(np.eye(4), (B, 4, 4)).copy()
    T_new[:, :3, :3] = new_rots.as_matrix()
    T_new[:, :3, 3] = new_trans

    last_angle = np.rad2deg(delta_rots[-1:].magnitude()[0]) if B > 0 else 0.0
    return T_new, last_angle, rand_magnitudes[-1] if B > 0 else 0.0

def generate_intrinsic_matrix(fx, fy, cx, cy):
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic_matrix
