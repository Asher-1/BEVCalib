"""Shortcut-learning diagnostics for BEVCalib extrinsic calibration models."""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

_KITTI_PATH = os.path.join(os.path.dirname(__file__), "kitti-bev-calib")
if _KITTI_PATH not in sys.path:
    sys.path.insert(0, _KITTI_PATH)

from tools import generate_single_perturbation_from_T
from visualization import compute_pose_errors


FIXED_BIASES_DEG = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (2.0, 2.0, 2.0),
]


def euler_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Roll=X, Pitch=Y, Yaw=Z (LiDAR frame). Returns 3x3 rotation matrix."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def bias_to_T4x4(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    return T


def rotation_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = R_a.T @ R_b
    trace = np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def T_geodesic_rot_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    return rotation_geodesic_deg(T_a[:3, :3], T_b[:3, :3])


def residual_correction(T_pred: np.ndarray, T_init: np.ndarray) -> np.ndarray:
    """T_residual s.t. T_pred = inv(T_residual) @ T_init."""
    return T_init @ np.linalg.inv(T_pred)


def T_to_rpy_deg(T: np.ndarray) -> np.ndarray:
    sy = math.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2)
    if sy >= 1e-6:
        roll = math.atan2(T[2, 1], T[2, 2])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = math.atan2(T[1, 0], T[0, 0])
    else:
        roll = math.atan2(-T[1, 2], T[1, 1])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])


def pose_matrix_std_deg(T_list: List[np.ndarray]) -> float:
    if len(T_list) < 2:
        return 0.0
    rpys = np.array([T_to_rpy_deg(T) for T in T_list])
    return float(np.mean(np.std(rpys, axis=0)))


def activation_entropy(activations: np.ndarray) -> float:
    flat = np.abs(activations).astype(np.float64).flatten()
    total = flat.sum()
    if total < 1e-12:
        return 0.0
    p = flat / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _prepare_tensors(
    imgs,
    pcs,
    masks,
    gt_T,
    intrinsics,
    device: torch.device,
    xyz_only: bool = True,
    keep_cpu: bool = False,
) -> Dict[str, torch.Tensor]:
    _dev = torch.device("cpu") if keep_cpu else device
    gt_np = np.array(gt_T).astype(np.float32)
    pcs_np = np.array(pcs)[:, :, :3] if xyz_only else np.array(pcs)
    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(_dev)
    pcs_t = torch.from_numpy(pcs_np).float().to(_dev)
    gt_t = torch.from_numpy(gt_np).float().to(_dev)
    post_t = torch.eye(4, device=_dev).unsqueeze(0).repeat(gt_t.shape[0], 1, 1)
    K_t = torch.from_numpy(np.array(intrinsics)).float().to(_dev)
    return {
        "resize_imgs": resize_imgs,
        "pcs": pcs_t,
        "gt_T": gt_t,
        "gt_np": gt_np,
        "pcs_np": pcs_np,
        "post_T": post_t,
        "K": K_t,
        "masks": masks,
        "masks_np": np.array(masks),
        "imgs_np": np.array(imgs),
    }


@torch.no_grad()
def _infer_batch(
    model,
    batch: Dict[str, torch.Tensor],
    init_T_np: np.ndarray,
    device: torch.device,
    zero_image: bool = False,
    zero_pc: bool = False,
    identity_init: bool = False,
) -> np.ndarray:
    resize_imgs = batch["resize_imgs"]
    if zero_image:
        resize_imgs = torch.zeros_like(resize_imgs)
    pcs = batch["pcs"]
    masks = batch["masks"]
    if zero_pc:
        pcs = torch.zeros_like(pcs)
        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks = torch.zeros_like(masks)
            else:
                masks = np.zeros_like(np.array(masks))

    B = batch["gt_np"].shape[0]
    if identity_init:
        init_np = np.tile(np.eye(4, dtype=np.float32), (B, 1, 1))
    else:
        init_np = init_T_np.astype(np.float32)
    init_t = torch.from_numpy(init_np).float().to(device)

    T_pred, _, _ = model(
        resize_imgs,
        pcs,
        batch["gt_T"],
        init_t,
        batch["post_T"],
        batch["K"],
        masks=masks,
        out_init_loss=False,
    )
    return T_pred.detach().cpu().numpy()


def _perturb_init(
    gt_T_np: np.ndarray,
    args,
    rotation_only: bool,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is not None:
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
    _paw = None
    if getattr(args, "per_axis_weights", "") and args.per_axis_weights:
        _paw = tuple(float(x) for x in args.per_axis_weights.split(","))
    init_T, _, _ = generate_single_perturbation_from_T(
        gt_T_np,
        angle_range_deg=args.angle_range_deg,
        trans_range=args.trans_range,
        rotation_only=rotation_only,
        distribution=getattr(args, "perturb_distribution", "uniform"),
        per_axis_prob=getattr(args, "per_axis_prob", 0.0),
        per_axis_weights=_paw,
    )
    if rng is not None:
        np.random.set_state(state)
    return init_T


def _collect_samples(
    val_loader,
    max_samples: int,
    device: torch.device,
    args,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for batch in val_loader:
        imgs, pcs, masks, gt_T, intrinsics = batch
        tensors = _prepare_tensors(imgs, pcs, masks, gt_T, intrinsics, device, keep_cpu=True)
        B = tensors["gt_np"].shape[0]
        for i in range(B):
            samples.append({
                "global_idx": len(samples),
                "resize_imgs": tensors["resize_imgs"][i : i + 1],
                "pcs": tensors["pcs"][i : i + 1],
                "gt_np": tensors["gt_np"][i : i + 1],
                "pcs_np": tensors["pcs_np"][i : i + 1],
                "post_T": tensors["post_T"][i : i + 1],
                "K": tensors["K"][i : i + 1],
                "masks": tensors["masks"][i] if tensors["masks"] is not None else None,
                "masks_np": tensors["masks_np"][i : i + 1],
                "imgs_np": tensors["imgs_np"][i : i + 1],
                "intrinsics": np.array(intrinsics)[i : i + 1],
            })
            if len(samples) >= max_samples:
                return samples
    return samples


def _sample_to_batch(sample: Dict[str, Any], device: torch.device = None) -> Dict[str, torch.Tensor]:
    _dev = device or sample["resize_imgs"].device
    return {
        "resize_imgs": sample["resize_imgs"].to(_dev),
        "pcs": sample["pcs"].to(_dev),
        "gt_T": torch.from_numpy(sample["gt_np"]).float().to(_dev),
        "gt_np": sample["gt_np"],
        "pcs_np": sample["pcs_np"],
        "post_T": sample["post_T"].to(_dev),
        "K": sample["K"].to(_dev),
        "masks": sample["masks"],
        "masks_np": sample["masks_np"],
        "imgs_np": sample["imgs_np"],
    }


def _seq_groups(
    n_samples: int,
    seq_boundaries: Optional[List[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    if seq_boundaries:
        return [(sid, s, e) for sid, s, e in seq_boundaries]
    return [(0, 0, n_samples - 1)]


def test_fixed_bias_correction(
    model,
    val_loader,
    args,
    device: torch.device,
    rotation_only: bool,
    seq_boundaries: Optional[List[Tuple[int, int, int]]],
) -> Dict[str, Any]:
    samples = _collect_samples(val_loader, max_samples=200, device=device, args=args)
    groups = _seq_groups(len(samples), seq_boundaries)
    results = {"biases": [], "per_sequence": {}, "summary": {}}

    for bias in FIXED_BIASES_DEG:
        bias_T = bias_to_T4x4(*bias)
        bias_mag = T_geodesic_rot_deg(np.eye(4), bias_T)
        bias_key = f"r{bias[0]:g}_p{bias[1]:g}_y{bias[2]:g}"
        seq_errors: Dict[str, List[float]] = {}

        for sid, start, end in groups:
            rot_errors = []
            for idx in range(start, min(end + 1, len(samples))):
                s = samples[idx]
                gt = s["gt_np"][0]
                biased_init = (bias_T @ gt).astype(np.float32)[np.newaxis]
                batch = _sample_to_batch(s, device)
                pred = _infer_batch(model, batch, biased_init, device)[0]
                err = compute_pose_errors(pred, gt)
                rot_errors.append(err["rot_error"])

            if rot_errors:
                seq_errors[str(sid)] = rot_errors

        all_errs = [e for errs in seq_errors.values() for e in errs]
        mean_err = float(np.mean(all_errs)) if all_errs else 0.0
        correction_rate = 1.0 - (mean_err / bias_mag) if bias_mag > 1e-6 else 0.0
        correction_rate = float(np.clip(correction_rate, -1.0, 1.0))

        entry = {
            "bias_deg": list(bias),
            "bias_magnitude_deg": bias_mag,
            "mean_rot_error_deg": mean_err,
            "correction_rate": correction_rate,
            "n_frames": len(all_errs),
            "shortcut_indicator": mean_err >= 0.8 * bias_mag,
        }
        results["biases"].append(entry)
        results["per_sequence"][bias_key] = {
            sid: {"mean_rot_error": float(np.mean(errs)), "n": len(errs)}
            for sid, errs in seq_errors.items()
        }

    rates = [b["correction_rate"] for b in results["biases"]]
    results["summary"] = {
        "mean_correction_rate": float(np.mean(rates)) if rates else 0.0,
        "min_correction_rate": float(np.min(rates)) if rates else 0.0,
        "n_biases": len(FIXED_BIASES_DEG),
        "n_frames_total": sum(b["n_frames"] for b in results["biases"]) // max(len(FIXED_BIASES_DEG), 1),
        "interpretation": (
            "Low correction_rate (error ≈ bias) suggests T_init passthrough shortcut."
            if np.mean(rates) < 0.3
            else "Model shows partial or good bias correction."
        ),
    }
    return results


def test_t_init_invariance(
    model,
    val_loader,
    args,
    device: torch.device,
    rotation_only: bool,
    n_frames: int = 50,
    n_perturbations: int = 10,
) -> Dict[str, Any]:
    samples = _collect_samples(val_loader, max_samples=n_frames, device=device, args=args)
    rng = np.random.default_rng(42)
    per_frame = []

    for s in samples:
        gt = s["gt_np"]
        preds, inits = [], []
        for _ in range(n_perturbations):
            init_T = _perturb_init(gt, args, rotation_only, rng=rng)
            batch = _sample_to_batch(s, device)
            pred = _infer_batch(model, batch, init_T, device)[0]
            preds.append(pred[0])
            inits.append(init_T[0])

        std_pred = pose_matrix_std_deg(preds)
        std_init = pose_matrix_std_deg(inits)
        if std_init < 1e-6:
            score = 1.0 if std_pred < 1e-6 else 0.0
        else:
            score = float(np.clip(1.0 - std_pred / std_init, -1.0, 1.0))

        per_frame.append({
            "std_T_pred_deg": std_pred,
            "std_T_init_deg": std_init,
            "invariance_score": score,
        })

    scores = [f["invariance_score"] for f in per_frame]
    return {
        "n_frames": len(per_frame),
        "n_perturbations": n_perturbations,
        "per_frame": per_frame,
        "summary": {
            "mean_invariance_score": float(np.mean(scores)) if scores else 0.0,
            "median_invariance_score": float(np.median(scores)) if scores else 0.0,
            "mean_std_pred_deg": float(np.mean([f["std_T_pred_deg"] for f in per_frame])),
            "mean_std_init_deg": float(np.mean([f["std_T_init_deg"] for f in per_frame])),
            "interpretation": (
                "Score near 0: model passes through T_init (shortcut). "
                "Score near 1: model ignores T_init (good)."
            ),
        },
    }


def test_identity_output_detector(
    model,
    val_loader,
    args,
    device: torch.device,
    rotation_only: bool,
) -> Dict[str, Any]:
    samples = _collect_samples(val_loader, max_samples=200, device=device, args=args)
    delta_residual = []
    init_rpy = []
    pred_rpy = []
    init_pert_rpy = []

    for s in samples:
        gt = s["gt_np"]
        init_T = _perturb_init(gt, args, rotation_only)
        batch = _sample_to_batch(s, device)
        pred = _infer_batch(model, batch, init_T, device)[0]

        T_res = residual_correction(pred, init_T[0])
        delta_residual.append(T_geodesic_rot_deg(np.eye(3), T_res[:3, :3]))

        init_rpy.append(T_to_rpy_deg(init_T[0]))
        pred_rpy.append(T_to_rpy_deg(pred))
        T_pert = np.linalg.inv(gt[0]) @ init_T[0]
        init_pert_rpy.append(T_to_rpy_deg(T_pert))

    delta_residual = np.array(delta_residual)
    init_pert_rpy = np.array(init_pert_rpy)
    pred_rpy = np.array(pred_rpy)

    correlations = {}
    for i, axis in enumerate(["roll", "pitch", "yaw"]):
        if np.std(init_pert_rpy[:, i]) < 1e-8 or np.std(pred_rpy[:, i]) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(init_pert_rpy[:, i], pred_rpy[:, i])[0, 1])
        correlations[axis] = corr

    mean_corr = float(np.mean(list(correlations.values())))
    mean_delta = float(np.mean(delta_residual))
    frac_small_residual = float(np.mean(delta_residual < 0.5)) if len(delta_residual) else 0.0
    passthrough_prob = frac_small_residual if mean_corr > 0.9 else 0.0

    return {
        "n_frames": len(delta_residual),
        "mean_residual_to_identity_deg": mean_delta,
        "median_residual_to_identity_deg": float(np.median(delta_residual)),
        "correlations_init_pert_vs_pred": correlations,
        "mean_correlation": mean_corr,
        "passthrough_probability": passthrough_prob,
        "shortcut_confirmed": mean_delta < 0.5 and mean_corr > 0.9,
        "summary": {
            "interpretation": (
                "Residual ≈ identity and high init-pred correlation → T_init passthrough shortcut."
                if mean_delta < 0.5 and mean_corr > 0.9
                else "No strong identity-passthrough signature detected."
            ),
        },
    }


def test_input_ablation(
    model,
    val_loader,
    args,
    device: torch.device,
    rotation_only: bool,
    n_frames: int = 100,
) -> Dict[str, Any]:
    samples = _collect_samples(val_loader, max_samples=n_frames, device=device, args=args)
    conditions = {
        "A_normal": {"zero_image": False, "zero_pc": False, "identity_init": False},
        "B_zero_image": {"zero_image": True, "zero_pc": False, "identity_init": False},
        "C_zero_pointcloud": {"zero_image": False, "zero_pc": True, "identity_init": False},
        "D_identity_init": {"zero_image": False, "zero_pc": False, "identity_init": True},
    }
    errors_by_cond: Dict[str, List[float]] = {k: [] for k in conditions}

    for s in samples:
        gt = s["gt_np"][0]
        init_T = _perturb_init(s["gt_np"], args, rotation_only)
        batch = _sample_to_batch(s, device)
        for name, flags in conditions.items():
            pred = _infer_batch(
                model, batch, init_T, device,
                zero_image=flags["zero_image"],
                zero_pc=flags["zero_pc"],
                identity_init=flags["identity_init"],
            )[0]
            errors_by_cond[name].append(compute_pose_errors(pred, gt)["rot_error"])

    summary = {}
    baseline = float(np.mean(errors_by_cond["A_normal"])) if errors_by_cond["A_normal"] else 0.0
    for name, errs in errors_by_cond.items():
        m = float(np.mean(errs)) if errs else 0.0
        summary[name] = {
            "mean_rot_error_deg": m,
            "std_rot_error_deg": float(np.std(errs)) if errs else 0.0,
            "ratio_vs_baseline": m / baseline if baseline > 1e-6 else 0.0,
        }

    spread_bc = max(
        abs(summary["B_zero_image"]["mean_rot_error_deg"] - baseline),
        abs(summary["C_zero_pointcloud"]["mean_rot_error_deg"] - baseline),
    )
    visual_insensitive = spread_bc < 0.1 * max(baseline, 1e-3)
    init_over_reliance = summary["D_identity_init"]["mean_rot_error_deg"] > 2.0 * baseline

    return {
        "n_frames": len(samples),
        "conditions": summary,
        "baseline_rot_error_deg": baseline,
        "visual_modality_insensitive": visual_insensitive,
        "init_over_reliance": init_over_reliance,
        "summary": {
            "interpretation": (
                "A≈B≈C: model may not use visual/LiDAR features (shortcut). "
                if visual_insensitive
                else ""
            )
            + (
                "D>>A: model over-relies on T_init."
                if init_over_reliance
                else "Modalities and init show expected sensitivity."
            ),
        },
    }


def _find_last_conv2d(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def test_gradcam_activation(
    model,
    val_loader,
    args,
    device: torch.device,
    eval_dir: str,
    n_frames: int = 10,
) -> Dict[str, Any]:
    diag_dir = os.path.join(eval_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    samples = _collect_samples(val_loader, max_samples=n_frames, device=device, args=args)
    img_entropies, pc_entropies = []
    per_frame = []

    img_layer = _find_last_conv2d(getattr(model, "img_branch", None))
    pc_layer = getattr(model, "pc_branch", None)

    for fi, s in enumerate(samples):
        batch = _sample_to_batch(s, device)
        gt = s["gt_np"]
        init_T = _perturb_init(gt, args, getattr(args, "rotation_only", True))

        activations: Dict[str, torch.Tensor] = {}
        gradients: Dict[str, torch.Tensor] = {}
        hooks = []

        def _save_act(key):
            def hook(_m, _inp, out):
                activations[key] = out
            return hook

        def _save_grad(key):
            def hook(_m, _grad_in, grad_out):
                if grad_out[0] is not None:
                    gradients[key] = grad_out[0]
            return hook

        if img_layer is not None:
            hooks.append(img_layer.register_forward_hook(_save_act("img")))
            hooks.append(img_layer.register_full_backward_hook(_save_grad("img")))
        if pc_layer is not None:
            hooks.append(pc_layer.register_forward_hook(_save_act("pc")))
            hooks.append(pc_layer.register_full_backward_hook(_save_grad("pc")))

        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        try:
            model.zero_grad(set_to_none=True)
            init_t = torch.from_numpy(init_T.astype(np.float32)).float().to(device)
            T_pred, _, loss_dict = model(
                batch["resize_imgs"],
                batch["pcs"],
                batch["gt_T"],
                init_t,
                batch["post_T"],
                batch["K"],
                masks=batch["masks"],
                out_init_loss=False,
            )
            loss = loss_dict["rotation_loss"].sum()
            loss.backward()

            img_entropy = pc_entropy = 0.0
            if "img" in activations and "img" in gradients:
                act = activations["img"][0]
                grad = gradients["img"][0]
                weights = grad.mean(dim=(1, 2), keepdim=True)
                cam = torch.relu((weights * act).sum(dim=0))
                cam_np = cam.detach().cpu().numpy()
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
                img_entropy = activation_entropy(cam_np)

                img_bgr = cv2.cvtColor(s["imgs_np"][0], cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]
                cam_resized = cv2.resize(cam_np, (w, h))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
                out_path = os.path.join(diag_dir, f"gradcam_img_{fi:04d}.png")
                cv2.imwrite(out_path, overlay)

            if "pc" in activations and "pc" in gradients:
                act = activations["pc"]
                grad = gradients["pc"]
                if act.dim() == 4:
                    gnorm = (grad.abs() * act.abs()).mean(dim=1)[0].detach().cpu().numpy()
                else:
                    gnorm = (grad.abs() * act.abs()).mean().detach().cpu().numpy()
                pc_entropy = activation_entropy(gnorm)
                gnorm_vis = gnorm - gnorm.min()
                if gnorm_vis.max() > 1e-8:
                    gnorm_vis = gnorm_vis / gnorm_vis.max()
                gnorm_u8 = np.uint8(255 * gnorm_vis)
                if gnorm_u8.ndim == 2:
                    pc_color = cv2.applyColorMap(gnorm_u8, cv2.COLORMAP_VIRIDIS)
                    cv2.imwrite(os.path.join(diag_dir, f"gradcam_pc_{fi:04d}.png"), pc_color)

            img_entropies.append(img_entropy)
            pc_entropies.append(pc_entropy)
            per_frame.append({
                "frame_idx": fi,
                "img_entropy": img_entropy,
                "pc_entropy": pc_entropy,
                "loss_rotation_deg": float(loss_dict["rotation_loss"].mean().item()),
            })
        finally:
            for h in hooks:
                h.remove()
            for p in model.parameters():
                p.requires_grad_(False)
            if was_training:
                model.train()
            else:
                model.eval()

    max_ent = math.log(max(np.prod(s["imgs_np"][0].shape[:2]), 2))
    mean_img_ent = float(np.mean(img_entropies)) if img_entropies else 0.0
    mean_pc_ent = float(np.mean(pc_entropies)) if pc_entropies else 0.0

    return {
        "n_frames": len(per_frame),
        "output_dir": diag_dir,
        "per_frame": per_frame,
        "summary": {
            "mean_img_activation_entropy": mean_img_ent,
            "mean_pc_activation_entropy": mean_pc_ent,
            "normalized_img_entropy": mean_img_ent / max_ent if max_ent > 0 else 0.0,
            "interpretation": (
                "High entropy → diffuse/random activations; "
                "low entropy → focused geometric features."
            ),
        },
    }


def compute_shortcut_risk_score(diag_results: Dict[str, Any]) -> Dict[str, Any]:
    t1 = diag_results.get("test1_fixed_bias", {}).get("summary", {})
    t2 = diag_results.get("test2_t_init_invariance", {}).get("summary", {})
    t3 = diag_results.get("test3_identity_output", {})
    t4 = diag_results.get("test4_input_ablation", {})
    t5 = diag_results.get("test5_gradcam", {}).get("summary", {})

    t1_risk = float(np.clip(1.0 - t1.get("mean_correction_rate", 0.0), 0.0, 1.0))
    t2_risk = float(np.clip(1.0 - t2.get("mean_invariance_score", 0.0), 0.0, 1.0))
    t3_risk = float(t3.get("passthrough_probability", 0.0))

    baseline = t4.get("baseline_rot_error_deg", 1.0) or 1.0
    cond = t4.get("conditions", {})
    ratio_b = cond.get("B_zero_image", {}).get("ratio_vs_baseline", 1.0)
    ratio_c = cond.get("C_zero_pointcloud", {}).get("ratio_vs_baseline", 1.0)
    ratio_d = cond.get("D_identity_init", {}).get("ratio_vs_baseline", 1.0)
    modality_risk = 1.0 - min(ratio_b, ratio_c, 1.0)
    init_risk = float(np.clip((ratio_d - 1.0) / 2.0, 0.0, 1.0))
    t4_risk = float(np.clip(0.6 * modality_risk + 0.4 * init_risk, 0.0, 1.0))

    norm_ent = float(t5.get("normalized_img_entropy", 0.5))
    t5_risk = float(np.clip(norm_ent, 0.0, 1.0))

    components = {
        "fixed_bias": t1_risk,
        "t_init_invariance": t2_risk,
        "identity_output": t3_risk,
        "input_ablation": t4_risk,
        "gradcam_entropy": t5_risk,
    }
    weights = {
        "fixed_bias": 0.25,
        "t_init_invariance": 0.25,
        "identity_output": 0.20,
        "input_ablation": 0.20,
        "gradcam_entropy": 0.10,
    }
    score = 100.0 * sum(components[k] * weights[k] for k in weights)

    if score < 25:
        level = "LOW"
    elif score < 50:
        level = "MODERATE"
    elif score < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {
        "shortcut_risk_score": round(score, 1),
        "risk_level": level,
        "component_risks": {k: round(v * 100, 1) for k, v in components.items()},
        "weights": weights,
    }


def run_shortcut_diagnostics(
    model,
    val_loader,
    args,
    device: torch.device,
    eval_dir: str,
    rotation_only: bool,
    seq_boundaries: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict[str, Any]:
    """Run all 5 shortcut diagnostic tests and return results dict."""
    os.makedirs(os.path.join(eval_dir, "diagnostics"), exist_ok=True)
    model.eval()

    print("[ShortcutDiag] Test 1: Fixed-bias correction...")
    test1 = test_fixed_bias_correction(
        model, val_loader, args, device, rotation_only, seq_boundaries)

    print("[ShortcutDiag] Test 2: T_init invariance...")
    test2 = test_t_init_invariance(model, val_loader, args, device, rotation_only)

    print("[ShortcutDiag] Test 3: Identity output detector...")
    test3 = test_identity_output_detector(model, val_loader, args, device, rotation_only)

    print("[ShortcutDiag] Test 4: Input ablation...")
    test4 = test_input_ablation(model, val_loader, args, device, rotation_only)

    print("[ShortcutDiag] Test 5: GradCAM activation...")
    test5 = test_gradcam_activation(model, val_loader, args, device, eval_dir)

    diag_results = {
        "test1_fixed_bias": test1,
        "test2_t_init_invariance": test2,
        "test3_identity_output": test3,
        "test4_input_ablation": test4,
        "test5_gradcam": test5,
    }
    diag_results["shortcut_risk"] = compute_shortcut_risk_score(diag_results)

    json_path = os.path.join(eval_dir, "diagnostics", "shortcut_diagnostics.json")
    with open(json_path, "w") as f:
        json.dump(diag_results, f, indent=2, default=_json_default)
    print(f"[ShortcutDiag] Results saved to {json_path}")

    report_path = generate_diagnostic_report(diag_results, eval_dir)
    print(f"[ShortcutDiag] Report saved to {report_path}")

    return diag_results


def generate_diagnostic_report(diag_results: Dict[str, Any], eval_dir: str) -> str:
    """Write markdown diagnostic report; returns output path."""
    diag_dir = os.path.join(eval_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, "shortcut_diagnostics_report.md")

    risk = diag_results.get("shortcut_risk", {})
    t1 = diag_results.get("test1_fixed_bias", {})
    t2 = diag_results.get("test2_t_init_invariance", {})
    t3 = diag_results.get("test3_identity_output", {})
    t4 = diag_results.get("test4_input_ablation", {})
    t5 = diag_results.get("test5_gradcam", {})

    lines = [
        "# BEVCalib Shortcut Learning Diagnostic Report",
        "",
        "## Shortcut Risk Assessment",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Risk Score** | **{risk.get('shortcut_risk_score', 'N/A')} / 100** |",
        f"| Risk Level | {risk.get('risk_level', 'N/A')} |",
        "",
        "### Component Contributions",
        "",
        "| Component | Risk (0-100) | Weight |",
        "|-----------|--------------|--------|",
    ]
    for comp, w in risk.get("weights", {}).items():
        val = risk.get("component_risks", {}).get(comp, "N/A")
        lines.append(f"| {comp} | {val} | {w:.0%} |")

    lines.extend([
        "",
        "> Score 0 = no shortcut detected, 100 = pure T_init passthrough.",
        "",
        "---",
        "",
        "## Test 1: Fixed-Bias Correction",
        "",
        "Same fixed rotation bias applied to T_init for all frames per sequence.",
        "",
        "| Bias (roll, pitch, yaw) ° | Bias Mag ° | Mean Error ° | Correction Rate | Shortcut? |",
        "|---------------------------|------------|--------------|-----------------|-----------|",
    ])
    for b in t1.get("biases", []):
        flag = "YES" if b.get("shortcut_indicator") else "no"
        lines.append(
            f"| {b['bias_deg']} | {b['bias_magnitude_deg']:.2f} | "
            f"{b['mean_rot_error_deg']:.3f} | {b['correction_rate']:.3f} | {flag} |"
        )
    s1 = t1.get("summary", {})
    lines.extend([
        "",
        f"**Summary:** mean correction rate = {s1.get('mean_correction_rate', 0):.3f}. "
        f"{s1.get('interpretation', '')}",
        "",
        "---",
        "",
        "## Test 2: T_init Invariance Score",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Frames tested | {t2.get('n_frames', 0)} |",
        f"| Perturbations per frame | {t2.get('n_perturbations', 10)} |",
        f"| Mean invariance score | {t2.get('summary', {}).get('mean_invariance_score', 0):.3f} |",
        f"| Mean std(T_pred) ° | {t2.get('summary', {}).get('mean_std_pred_deg', 0):.4f} |",
        f"| Mean std(T_init) ° | {t2.get('summary', {}).get('mean_std_init_deg', 0):.4f} |",
        "",
        f"{t2.get('summary', {}).get('interpretation', '')}",
        "",
        "---",
        "",
        "## Test 3: Identity Output Detector",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Frames | {t3.get('n_frames', 0)} |",
        f"| Mean residual→identity ° | {t3.get('mean_residual_to_identity_deg', 0):.4f} |",
        f"| Mean init-pred correlation | {t3.get('mean_correlation', 0):.4f} |",
        f"| Passthrough probability | {t3.get('passthrough_probability', 0):.3f} |",
        f"| Shortcut confirmed | {t3.get('shortcut_confirmed', False)} |",
        "",
        "**Per-axis correlation (init perturbation vs pred):**",
        "",
        "| Axis | Correlation |",
        "|------|-------------|",
    ])
    for axis, corr in t3.get("correlations_init_pert_vs_pred", {}).items():
        lines.append(f"| {axis} | {corr:.4f} |")
    lines.extend([
        "",
        f"{t3.get('summary', {}).get('interpretation', '')}",
        "",
        "---",
        "",
        "## Test 4: Input Ablation",
        "",
        "| Condition | Mean Rot Error ° | Ratio vs A |",
        "|-----------|------------------|------------|",
    ])
    for name, stats in t4.get("conditions", {}).items():
        lines.append(
            f"| {name} | {stats.get('mean_rot_error_deg', 0):.4f} | "
            f"{stats.get('ratio_vs_baseline', 0):.3f} |"
        )
    lines.extend([
        "",
        f"- Visual modality insensitive (A≈B≈C): **{t4.get('visual_modality_insensitive', False)}**",
        f"- Init over-reliance (D>>A): **{t4.get('init_over_reliance', False)}**",
        "",
        f"{t4.get('summary', {}).get('interpretation', '')}",
        "",
        "---",
        "",
        "## Test 5: GradCAM Activation",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Frames | {t5.get('n_frames', 0)} |",
        f"| Mean image activation entropy | {t5.get('summary', {}).get('mean_img_activation_entropy', 0):.4f} |",
        f"| Mean PC activation entropy | {t5.get('summary', {}).get('mean_pc_activation_entropy', 0):.4f} |",
        f"| Output directory | `{t5.get('output_dir', '')}` |",
        "",
        f"{t5.get('summary', {}).get('interpretation', '')}",
        "",
        "Visualizations: `gradcam_img_*.png`, `gradcam_pc_*.png`",
        "",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
