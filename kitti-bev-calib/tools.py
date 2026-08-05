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
                                         per_axis_weights=None,
                                         symmetric_perturb=False,
                                         rotation_definition='per_axis'):
    """
    Vectorized batch perturbation with configurable distribution and per-axis mode.

    Parameters:
        T: np.ndarray, shape (B, 4, 4)
        angle_range_deg: float, rotation perturbation range in degrees
        trans_range: float, translation perturbation range in meters
        rotation_only: bool, if True skip translation perturbation
        distribution: 'uniform', 'truncated_normal', or 'magnitude_balanced'
        per_axis_prob: probability of perturbing a single axis only (0=disabled)
        curriculum_scale: scale factor for perturbation range (0..1 for curriculum)
        per_axis_weights: tuple of 3 floats for (roll, pitch, yaw) sampling weights
                          in per-axis mode. None = uniform. e.g. (0.5, 0.3, 0.2)
        symmetric_perturb: if True, force half of batch positive and half negative
        rotation_definition: ``per_axis`` samples LiDAR roll/pitch/yaw independently
            in ``[-angle_range_deg, angle_range_deg]``; ``total`` samples one
            axis-angle whose total magnitude is bounded by ``angle_range_deg``.

    distribution='magnitude_balanced': sample |angle| uniformly from [0, max],
    then assign random sign. Ensures equal representation of all magnitudes
    including near-zero, improving zero-drift and magnitude awareness.
    """
    B = T.shape[0]
    effective_angle = angle_range_deg * curriculum_scale
    effective_trans = trans_range * curriculum_scale

    orig_rots = R.from_matrix(T[:, :3, :3])
    orig_trans = T[:, :3, 3]

    use_per_axis = (per_axis_prob > 0) and (np.random.rand() < per_axis_prob)

    def _apply_symmetric(angles_arr, n):
        """Force exact 50/50 positive/negative split within a batch."""
        if not symmetric_perturb or n < 2:
            return angles_arr
        half = n // 2
        magnitudes = np.abs(angles_arr)
        signs = np.ones(n)
        signs[:half] = -1.0
        np.random.shuffle(signs)
        return magnitudes * signs

    def _sample_angles(n):
        if distribution == 'truncated_normal':
            return _sample_truncated_normal(-effective_angle, effective_angle, n)
        elif distribution == 'magnitude_balanced':
            magnitudes = np.random.uniform(0, effective_angle, n)
            signs = np.random.choice([-1.0, 1.0], size=n)
            return magnitudes * signs
        else:
            return np.random.uniform(-effective_angle, effective_angle, n)

    if use_per_axis:
        rotvecs = np.zeros((B, 3))
        if per_axis_weights is not None:
            w = np.array(per_axis_weights, dtype=np.float64)
            w = w / w.sum()
            axis_idx = np.random.choice(3, size=B, p=w)
        else:
            axis_idx = np.random.randint(0, 3, size=B)
        angles = _apply_symmetric(_sample_angles(B), B)
        angles_rad = np.deg2rad(angles)
        for i in range(B):
            rotvecs[i, axis_idx[i]] = angles_rad[i]
        delta_rots = R.from_rotvec(rotvecs)
    elif rotation_definition == 'per_axis':
        euler_deg = np.stack([_apply_symmetric(_sample_angles(B), B)
                              for _ in range(3)], axis=1)
        delta_rots = R.from_euler('xyz', euler_deg, degrees=True)
    elif rotation_definition == 'total':
        rand_axes = np.random.randn(B, 3)
        rand_axes /= np.linalg.norm(rand_axes, axis=1, keepdims=True)
        rand_angles = np.deg2rad(_apply_symmetric(_sample_angles(B), B))
        delta_rots = R.from_rotvec(rand_axes * rand_angles[:, None])
    else:
        raise ValueError(f"Unknown rotation_definition={rotation_definition!r}")

    # Canonical LiDAR-frame convention: R_init = R_gt @ delta_R.
    new_rots = orig_rots * delta_rots

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

def augment_gt_pitch_flip(T, prob=0.5, max_deg=6.0, sign_flip_prob=0.0):
    """Balance pitch distribution via LiDAR Y-axis rotation of GT extrinsics.

    The camera-LiDAR pitch angle manifests as the sign of R[0,1] and R[2,2]
    in the LiDAR→Camera rotation matrix. Training data is often imbalanced
    (e.g. most sequences have positive pitch, while Seq02/06 have negative),
    causing systematic prediction bias on underrepresented mounting configs.

    Two complementary augmentation modes (applied independently per sample):

    1. **Pitch perturbation** (prob, max_deg): random Y-axis rotation in
       [-max_deg, +max_deg]. Adds geometric diversity around the existing
       pitch distribution.

    2. **Pitch sign flip** (sign_flip_prob): extracts the current pitch
       angle from R and applies a rotation of -2*pitch, precisely negating
       the pitch sign. This directly creates training samples that simulate
       reversed camera mounting (like Seq02/06 from Seq00-like data).

    Both are applied BEFORE perturbation generation, so the perturbed BEV
    and correction target stay consistent.

    Parameters:
        T: np.ndarray, shape (B, 4, 4) — GT LiDAR→Camera transforms
        prob: probability of applying random pitch perturbation per sample
        max_deg: maximum rotation angle in degrees for perturbation
        sign_flip_prob: probability of explicitly flipping pitch sign
    Returns:
        T_aug: np.ndarray, shape (B, 4, 4) — augmented GT transforms
    """
    if prob <= 0 and sign_flip_prob <= 0:
        return T
    B = T.shape[0]
    T_aug = T.copy()

    if sign_flip_prob > 0:
        flip_mask = np.random.rand(B) < sign_flip_prob
        for i in np.where(flip_mask)[0]:
            R = T_aug[i, :3, :3]
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            neg_pitch = -pitch
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(neg_pitch), np.sin(neg_pitch)
            cy, sy_ = np.cos(yaw), np.sin(yaw)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
            T_aug[i, :3, :3] = Rz @ Ry @ Rx

    if prob > 0 and max_deg > 0:
        perturb_mask = np.random.rand(B) < prob
        if perturb_mask.any():
            n_perturb = perturb_mask.sum()
            angles_rad = np.deg2rad(
                np.random.uniform(-max_deg, max_deg, n_perturb)
            )
            cos_a = np.cos(angles_rad)
            sin_a = np.sin(angles_rad)
            for idx, i in enumerate(np.where(perturb_mask)[0]):
                Ry = np.array([
                    [ cos_a[idx], 0, sin_a[idx]],
                    [ 0,          1, 0         ],
                    [-sin_a[idx], 0, cos_a[idx]],
                ])
                T_aug[i, :3, :3] = T_aug[i, :3, :3] @ Ry

    return T_aug


def augment_mount_jitter(T, prob=0.3, rotation_sigma_deg=0.5,
                         translation_sigma_m=0.01, return_point_transform=False):
    """Simulate camera mount installation diversity by jittering GT extrinsics.

    Unlike perturbation augmentation (which creates the init→GT correction target),
    mount jitter modifies the GT itself to represent a DIFFERENT camera installation.
    This forces the model to handle diverse mount configurations rather than
    memorizing the specific extrinsics of training vehicles.

    The key domain gap insight: test vehicles have different camera-LiDAR mounting
    (e.g. test Seq02 has ~164° roll difference from train Seq02). Small mount
    jitter during training builds robustness to such installation variations.

    Applied BEFORE perturbation generation, so both GT and init stay consistent.

    Parameters:
        T: (B, 4, 4) GT extrinsic matrices
        prob: per-sample probability of applying mount jitter
        rotation_sigma_deg: std of rotation jitter per axis (degrees)
        translation_sigma_m: std of translation jitter per axis (meters)
    Returns:
        T_aug: (B, 4, 4) jittered GT extrinsics
    """
    if prob <= 0:
        if return_point_transform:
            identity = np.broadcast_to(np.eye(4, dtype=np.float32), T.shape).copy()
            return T, identity
        return T
    B = T.shape[0]
    T_original = T.copy()
    T_aug = T.copy()
    jitter_mask = np.random.rand(B) < prob
    n_jitter = jitter_mask.sum()
    if n_jitter == 0:
        if return_point_transform:
            identity = np.broadcast_to(np.eye(4, dtype=np.float32), T.shape).copy()
            return T_aug, identity
        return T_aug

    rot_jitter_rad = np.deg2rad(
        np.random.normal(0, rotation_sigma_deg, (n_jitter, 3))
    )
    delta_rots = R.from_rotvec(rot_jitter_rad)
    trans_jitter = np.random.normal(0, translation_sigma_m, (n_jitter, 3)).astype(np.float32)

    for idx, i in enumerate(np.where(jitter_mask)[0]):
        T_aug[i, :3, :3] = T_aug[i, :3, :3] @ delta_rots[idx].as_matrix().astype(np.float32)
        T_aug[i, :3, 3] += trans_jitter[idx]

    if return_point_transform:
        # Preserve the observation while changing the virtual LiDAR mounting:
        # T_aug @ P_aug == T_original @ P_original.
        point_transform = np.linalg.inv(T_aug) @ T_original
        return T_aug, point_transform.astype(np.float32)
    return T_aug


def generate_intrinsic_matrix(fx, fy, cx, cy):
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic_matrix
