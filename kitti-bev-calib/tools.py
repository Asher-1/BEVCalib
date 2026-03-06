import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_single_perturbation_from_T(T, angle_range_deg=20, trans_range=1.5):
    """
    Vectorized batch perturbation: generates perturbed transformation matrices
    without Python loops over the batch dimension.

    Parameters:
        T: np.ndarray, shape (B, 4, 4)
        angle_range_deg: float, rotation perturbation range in degrees
        trans_range: float, translation perturbation range in meters
    """
    B = T.shape[0]

    orig_rots = R.from_matrix(T[:, :3, :3])
    orig_trans = T[:, :3, 3]

    rand_axes = np.random.randn(B, 3)
    rand_axes /= np.linalg.norm(rand_axes, axis=1, keepdims=True)
    rand_angles = np.deg2rad(np.random.uniform(-angle_range_deg, angle_range_deg, B))
    delta_rots = R.from_rotvec(rand_axes * rand_angles[:, None])
    new_rots = delta_rots * orig_rots

    rand_dirs = np.random.randn(B, 3)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_magnitudes = np.random.uniform(0, trans_range, B)
    new_trans = orig_trans + rand_dirs * rand_magnitudes[:, None]

    T_new = np.broadcast_to(np.eye(4), (B, 4, 4)).copy()
    T_new[:, :3, :3] = new_rots.as_matrix()
    T_new[:, :3, 3] = new_trans

    return T_new, np.rad2deg(rand_angles[-1]), rand_magnitudes[-1]

def generate_intrinsic_matrix(fx, fy, cx, cy):
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic_matrix
