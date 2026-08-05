import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "kitti-bev-calib"
sys.path.insert(0, str(MODULE_ROOT))
SPEC = importlib.util.spec_from_file_location("train_kitti", MODULE_ROOT / "train_kitti.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class _Dataset:
    def __init__(self):
        self.all_files = [
            f"{seq}/{frame:06d}" for seq in ("00", "01", "02", "03")
            for frame in range(5)
        ]

    def __len__(self):
        return len(self.all_files)


def _sequences(subset, dataset):
    return {dataset.all_files[i].split('/')[0] for i in subset.indices}


def test_group_holdout_has_no_sequence_leakage():
    dataset = _Dataset()
    train, val, _, train_seqs, val_seqs = TRAIN.group_holdout_split_by_sequence(
        dataset, val_sequences="01,03")
    assert set(train_seqs) == {"00", "02"}
    assert set(val_seqs) == {"01", "03"}
    assert _sequences(train, dataset).isdisjoint(_sequences(val, dataset))


def test_canonical_injection_is_right_multiplied_per_axis_two_degrees():
    gt = np.eye(4, dtype=np.float32)
    injected = TRAIN._gdiag_inject_T(gt, [2.0, 2.0, 2.0])
    trace = np.trace(injected[:3, :3])
    geodesic = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    assert abs(geodesic - 3.4437) < 0.01
    np.testing.assert_allclose(
        TRAIN._perturbation_euler_from_T_pair(gt, injected), [2.0, 2.0, 2.0],
        atol=1e-4)


def test_random_training_perturbation_uses_same_lidar_axis_convention():
    from tools import generate_single_perturbation_from_T

    gt = np.eye(4, dtype=np.float32)[None]
    np.random.seed(7)
    injected, _, _ = generate_single_perturbation_from_T(
        gt, angle_range_deg=2.0, rotation_only=True,
        rotation_definition='per_axis')
    signed = TRAIN._perturbation_euler_from_T_pair(gt[0], injected[0])
    assert np.max(np.abs(signed)) <= 2.0 + 1e-5


def test_ood_gate_requires_genuine_recovery_slope_and_zero_drift():
    good = {
        "genuine_recovery_pct": 96.0,
        "signed_correction_slope": 0.9,
        "signed_correction_slope_min_axis": 0.85,
        "zd_max_deg": 0.08,
    }
    assert TRAIN._recovery_gate_pass(None, 0, good, 95, 0.8, 0.1)
    for key, value in (
        ("genuine_recovery_pct", 94.9),
        ("signed_correction_slope_min_axis", 0.79),
        ("zd_max_deg", 0.11),
    ):
        bad = dict(good)
        bad[key] = value
        assert not TRAIN._recovery_gate_pass(None, 0, bad, 95, 0.8, 0.1)


def test_mount_randomization_preserves_camera_observation():
    from tools import augment_mount_jitter

    gt = np.eye(4, dtype=np.float32)[None]
    gt[0, :3, 3] = [0.2, -0.1, 1.4]
    np.random.seed(3)
    augmented, point_transform = augment_mount_jitter(
        gt, prob=1.0, rotation_sigma_deg=3.0,
        translation_sigma_m=0.1, return_point_transform=True)
    points = np.array([[2.0, 0.5, -0.3, 1.0], [8.0, -1.0, 0.2, 1.0]], dtype=np.float32).T
    np.testing.assert_allclose(
        augmented[0] @ point_transform[0] @ points,
        gt[0] @ points,
        atol=1e-5)


def test_position_encoding_does_not_reuse_other_rig_intrinsics():
    from modules.position_encoding_3d import PositionEncoding3D

    encoder = PositionEncoding3D(feat_dim=16, depth_bins=4, patch_size=(15.0, 16.0))
    encoder.eval()
    k1 = torch.tensor([[[700.0, 0.0, 480.0], [0.0, 710.0, 270.0], [0.0, 0.0, 1.0]]])
    k2 = k1.clone()
    k2[:, 0, 2] += 40.0
    pe1 = encoder(4, 6, k1)
    pe2 = encoder(4, 6, k2)
    assert not torch.allclose(pe1, pe2)


def test_paper_input_materializes_init_perturbation_and_preserves_observation():
    from paper_explicit_bev_calib import _materialize_misregistered_points

    gt = torch.eye(4).unsqueeze(0)
    gt[:, :3, 3] = torch.tensor([[0.2, -0.1, 1.4]])
    init_np = TRAIN._gdiag_inject_T(gt[0].numpy(), [2.0, -1.0, 3.0])
    init = torch.from_numpy(init_np).unsqueeze(0)
    points = torch.tensor([[[2.0, 0.5, -0.3], [8.0, -1.0, 0.2]]])

    points_mis, delta, target = _materialize_misregistered_points(points, gt, init)
    assert not torch.allclose(points_mis, points)
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_mis_h = torch.cat(
        [points_mis, torch.ones_like(points_mis[..., :1])], dim=-1)
    camera_original = torch.bmm(gt, points_h.transpose(1, 2))
    camera_recovered = torch.bmm(target, points_mis_h.transpose(1, 2))
    torch.testing.assert_close(camera_recovered, camera_original, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.bmm(target, delta), gt, atol=1e-5, rtol=1e-5)
