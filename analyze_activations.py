#!/usr/bin/env python3
"""
BEVCalib 激活分析脚本 — 诊断 Zero-Drift / Recovery 的根本原因

分析内容:
1. GIN gate forward 动态值分布（是否 collapse 到 0 或 1）
2. T_init 对特征的影响（扰动 vs 无扰动的特征差异）
3. Rotation head 对不同轴的梯度敏感度
4. Consistency loss 对 init-dependent 特征的压制程度

用法:
  conda run -n bevcalib310 python3 analyze_activations.py \
    --ckpt logs/all_training_data/model_small_5deg_v48a_ms_only_S1_quick/all_training_data_scratch/checkpoint/ckpt_best_val.pth \
    --data_root /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --n_samples 50
"""

import argparse
import json
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kitti-bev-calib"))


def load_model(ckpt_path, device="cuda:0"):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", None)
    if args is None:
        raise ValueError("Checkpoint missing 'args'")

    from cf_bev_r_calib import CFBevRCalib

    sd = ckpt["model_state_dict"]

    gin_channels = 0
    use_gin = False
    if "feat_in.in_norm.weight" in sd:
        gin_channels = sd["feat_in.in_norm.weight"].shape[0]
        use_gin = True

    has_magnitude = any("magnitude" in k for k in sd)
    has_pitch = any("pitch_branch" in k for k in sd)

    pitch_bands = getattr(args, "pitch_vertical_bands", 3)
    if has_pitch:
        for k in sd:
            if "pitch_branch.img_encoder" in k and "weight" in k and sd[k].dim() == 4:
                in_ch = sd[k].shape[1]
                if in_ch != 256:
                    pitch_bands = max(2, in_ch // 256) if in_ch > 256 else 3
                break

    init_params = {
        "use_dla": getattr(args, "use_dla", False),
        "use_instance_norm": getattr(args, "use_instance_norm", False),
        "use_gated_instance_norm": use_gin,
        "gin_channels": gin_channels,
        "gin_init_gate": getattr(args, "gin_init_gate", 0.5),
        "gin_gate_reg_target": getattr(args, "gin_gate_reg_target", 0.0),
        "use_pitch_branch": has_pitch or getattr(args, "use_pitch_branch", False),
        "pitch_vertical_bands": pitch_bands,
        "use_pitch_fusion": getattr(args, "use_pitch_fusion", False),
        "use_magnitude_head": has_magnitude or getattr(args, "use_magnitude_head", False),
        "head_dropout": getattr(args, "head_dropout", 0.1),
    }

    param_map = {
        "cf_feat_dim": "feat_dim", "cf_n_groups": "n_groups", "cf_knn": "knn",
        "cf_corr_heads": "num_corr_heads", "cf_corr_radius": "default_corr_radius",
        "cf_num_queries": "num_queries", "cf_encoder_layers": "encoder_layers",
        "cf_decoder_layers": "decoder_layers",
    }
    for src, dst in param_map.items():
        if hasattr(args, src):
            init_params[dst] = getattr(args, src)

    model = CFBevRCalib(**init_params)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device).eval()

    return model, args


def analyze_gin_gate(model, device="cuda:0"):
    """Analyze GIN gate distribution from bias and with random inputs."""
    results = {}
    feat_in = getattr(model, "feat_in", None)
    if feat_in is None:
        results["gin_enabled"] = False
        return results

    results["gin_enabled"] = True
    results["gin_channels"] = feat_in.gin_channels
    results["bypass_channels"] = feat_in.bypass_channels

    bias = feat_in.gate_fc[-2].bias.detach()
    bias_gate = torch.sigmoid(bias)
    results["bias_gate_mean"] = bias_gate.mean().item()
    results["bias_gate_std"] = bias_gate.std().item()
    results["bias_gate_min"] = bias_gate.min().item()
    results["bias_gate_max"] = bias_gate.max().item()
    results["bias_gate_near_0_pct"] = (bias_gate < 0.1).float().mean().item() * 100
    results["bias_gate_near_1_pct"] = (bias_gate > 0.9).float().mean().item() * 100

    total_ch = feat_in.channels
    dummy_inputs = [
        torch.randn(4, total_ch, 45, 80).to(device),
        torch.randn(4, total_ch, 45, 80).to(device) * 2.0,
        torch.randn(4, total_ch, 45, 80).to(device) * 0.5,
    ]

    gate_values = []
    with torch.no_grad():
        for inp in dummy_inputs:
            _ = feat_in(inp)
            stats = getattr(feat_in, "_last_gate_stats", None)
            if stats:
                gate_values.append(stats)

    if gate_values:
        results["forward_gate_mean"] = np.mean([g["mean"] for g in gate_values])
        results["forward_gate_std"] = np.mean([g["std"] for g in gate_values])
        results["forward_gate_min"] = min(g["min"] for g in gate_values)
        results["forward_gate_max"] = max(g["max"] for g in gate_values)

    return results


def analyze_tinit_sensitivity(model, device="cuda:0"):
    """Analyze how sensitive the model output is to T_init perturbations.

    Measures the Jacobian: ∂output/∂T_init at small angular perturbations.
    """
    results = {}

    total_ch = getattr(model, "feat_in", None)
    if total_ch and hasattr(total_ch, "channels"):
        ch = total_ch.channels
    else:
        ch = 256

    dummy_img = torch.randn(1, 6, 3, 360, 640).to(device)
    dummy_pc = torch.randn(1, 16384, 3).to(device)

    identity = torch.eye(4).unsqueeze(0).to(device)

    perturbation_angles = [0.0, 0.5, 1.0, 2.0, 5.0]
    outputs = {}

    for angle in perturbation_angles:
        T_init = identity.clone()
        if angle > 0:
            rad = angle * np.pi / 180.0
            T_init[0, 0, 0] = np.cos(rad)
            T_init[0, 0, 1] = -np.sin(rad)
            T_init[0, 1, 0] = np.sin(rad)
            T_init[0, 1, 1] = np.cos(rad)

        try:
            with torch.no_grad():
                out = model(dummy_img, dummy_pc, T_init)
                if isinstance(out, tuple):
                    out = out[0]
                outputs[angle] = out.cpu().numpy()
        except Exception as e:
            results[f"tinit_{angle}deg_error"] = str(e)
            return results

    if 0.0 in outputs:
        baseline = outputs[0.0]
        for angle in [0.5, 1.0, 2.0, 5.0]:
            if angle in outputs:
                diff = np.abs(outputs[angle] - baseline)
                results[f"output_diff_{angle}deg_mean"] = float(diff.mean())
                results[f"output_diff_{angle}deg_max"] = float(diff.max())

                if angle > 0 and diff.mean() > 0:
                    sensitivity = diff.mean() / (angle * np.pi / 180)
                    results[f"sensitivity_{angle}deg"] = float(sensitivity)

    return results


def analyze_rotation_head_gradients(model, device="cuda:0"):
    """Analyze gradient magnitudes for Roll/Pitch/Yaw axes."""
    from modules.corr_transformer_decoder import PoseQueryDecoder

    head = None
    for module in model.modules():
        if isinstance(module, PoseQueryDecoder):
            head = module
            break

    results = {}
    if head is None:
        results["error"] = "CorrTransformerDecoder not found"
        return results

    quat_head = head.quat_head
    bias = quat_head[-1].bias.detach().cpu().numpy()
    weight = quat_head[-1].weight.detach().cpu().numpy()

    results["bias_values"] = bias.tolist()
    results["weight_norm_per_output"] = np.linalg.norm(weight, axis=1).tolist()
    results["weight_frobenius_norm"] = float(np.linalg.norm(weight))

    w_abs = np.abs(weight)
    results["weight_mean_magnitude_per_quat"] = w_abs.mean(axis=1).tolist()
    results["weight_std_per_quat"] = weight.std(axis=1).tolist()

    return results


def main():
    parser = argparse.ArgumentParser(description="BEVCalib Activation Analysis")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model from: {args.ckpt}")
    model, model_args = load_model(args.ckpt, args.device)

    print(f"\n{'='*60}")
    print("1. GIN Gate Analysis")
    print(f"{'='*60}")
    gin_results = analyze_gin_gate(model, args.device)
    for k, v in gin_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print("2. Rotation Head Gradient Analysis")
    print(f"{'='*60}")
    rot_results = analyze_rotation_head_gradients(model, args.device)
    for k, v in rot_results.items():
        if isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.4f}' for x in v)}]")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    all_results = {
        "checkpoint": args.ckpt,
        "gin_gate": gin_results,
        "rotation_head": rot_results,
    }

    out_path = os.path.splitext(args.ckpt)[0] + "_activation_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
