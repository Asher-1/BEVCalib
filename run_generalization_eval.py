#!/usr/bin/env python3
"""
多模型泛化性能评估 + 分析报告生成
对所有指定模型在test_data上运行评估，生成对比报告和可视化

用法:
  python run_generalization_eval.py --config configs/eval_generalization.yaml
  python run_generalization_eval.py  # 使用内置默认配置
"""
import os
import sys
import re
import subprocess
import json
import argparse
import time
import numpy as np
from datetime import datetime

import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _format_elapsed(seconds):
    """Format elapsed seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s}s"
    h, m = divmod(m, 60)
    return f"{h}h{m}m{s}s"


DEFAULT_MODELS = [
    {
        "label": "10deg-v2-z5",
        "dir_name": "model_small_10deg_v2_z5",
        "ckpt": "ckpt_240.pth",
        "bev_zbound_step": "4.0",
        "rotation_only": 0,
        "angle_deg": 10, "z_voxels": 5, "version": "v2",
        "mode_desc": "rotation+translation",
    },
    {
        "label": "10deg-v3-z10",
        "dir_name": "model_small_10deg_v3_z10",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "2.0",
        "rotation_only": 0,
        "angle_deg": 10, "z_voxels": 10, "version": "v3",
        "mode_desc": "rotation+translation",
    },
    {
        "label": "10deg-v4-z1-rot",
        "dir_name": "model_small_10deg_v4_z1_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "20.0",
        "rotation_only": 1,
        "angle_deg": 10, "z_voxels": 1, "version": "v4",
        "mode_desc": "rotation_only",
    },
    {
        "label": "10deg-v4-z5-rot",
        "dir_name": "model_small_10deg_v4_z5_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "4.0",
        "rotation_only": 1,
        "angle_deg": 10, "z_voxels": 5, "version": "v4",
        "mode_desc": "rotation_only",
    },
    {
        "label": "10deg-v4-z10-rot",
        "dir_name": "model_small_10deg_v4_z10_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "2.0",
        "rotation_only": 1,
        "angle_deg": 10, "z_voxels": 10, "version": "v4",
        "mode_desc": "rotation_only",
    },
    {
        "label": "5deg-v4-z1-rot",
        "dir_name": "model_small_5deg_v4_z1_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "20.0",
        "rotation_only": 1,
        "angle_deg": 5, "z_voxels": 1, "version": "v4",
        "mode_desc": "rotation_only",
    },
    {
        "label": "5deg-v4-z5-rot",
        "dir_name": "model_small_5deg_v4_z5_rotation",
        "ckpt": "ckpt_400.pth",
        "bev_zbound_step": "4.0",
        "rotation_only": 1,
        "angle_deg": 5, "z_voxels": 5, "version": "v4",
        "mode_desc": "rotation_only",
    },
]

DEFAULT_BEVCALIB_ROOT = "/mnt/drtraining/user/dahailu/code/BEVCalib"


def _validate_eval_models(models, config_path=None):
    """Reject YAML schemas that don't match run_generalization_eval.py expectations."""
    _WRONG_KEYS = {"model_dir", "checkpoint", "tag", "name", "train_config"}
    _REQUIRED = ("label", "dir_name", "ckpt")
    _CANONICAL = "configs/c1_retrain/eval_generalization_c1_v62.yaml"
    for i, m in enumerate(models or []):
        if not isinstance(m, dict):
            raise ValueError(f"models[{i}] must be a mapping")
        wrong = sorted(_WRONG_KEYS & set(m.keys()))
        if wrong:
            src = f" ({config_path})" if config_path else ""
            raise ValueError(
                f"models[{i}] uses unsupported keys {wrong}{src}. "
                f"run_generalization_eval.py requires {_REQUIRED}. "
                f"Copy schema from {_CANONICAL} — do NOT use v63/v64 model_dir/checkpoint format."
            )
        missing = [k for k in _REQUIRED if k not in m]
        if missing:
            src = f" ({config_path})" if config_path else ""
            raise ValueError(
                f"models[{i}] missing required keys {missing}{src}. "
                f"Use label/dir_name/ckpt per {_CANONICAL}."
            )


def load_config(config_path=None):
    """Load evaluation config from YAML file or use defaults."""
    if config_path and os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        root = cfg.get("bevcalib_root", DEFAULT_BEVCALIB_ROOT)
        models_dir_rel = cfg.get("models_dir", "logs/all_training_data")
        output_dir_rel = cfg.get("output_dir", "logs/multi_eval_generalization_5deg")
        eval_script_rel = cfg.get("eval_script", "evaluate_checkpoint.py")

        config = {
            "BEVCALIB_ROOT": root,
            "MODELS_DIR": os.path.join(root, models_dir_rel) if not os.path.isabs(models_dir_rel) else models_dir_rel,
            "TEST_DATA": cfg.get("test_data", "/mnt/drtraining/user/dahailu/data/bevcalib/test_data"),
            "OUTPUT_DIR": os.path.join(root, output_dir_rel) if not os.path.isabs(output_dir_rel) else output_dir_rel,
            "EVAL_SCRIPT": os.path.join(root, eval_script_rel) if not os.path.isabs(eval_script_rel) else eval_script_rel,
            "ANGLE_RANGE": cfg.get("eval_params", {}).get("angle_range", 5.0),
            "TRANS_RANGE": cfg.get("eval_params", {}).get("trans_range", 0.3),
            "BATCH_SIZE": cfg.get("eval_params", {}).get("batch_size", 8),
            "VIS_INTERVAL": cfg.get("eval_params", {}).get("vis_interval", 200),
            "TIMEOUT": cfg.get("eval_params", {}).get("timeout", 1800),
            "EVAL_SAMPLE_STEP": cfg.get("eval_params", {}).get("eval_sample_step", None),
            "EVAL_MAX_FRAMES_PER_SEQ": cfg.get("eval_params", {}).get("eval_max_frames_per_seq", None),
            "SHORTCUT_DIAG": cfg.get("eval_params", {}).get("shortcut_diag", False),
            "GENERALIZATION_DIAG": cfg.get("eval_params", {}).get("generalization_diag", False),
            "GDIAG_INJECT_DEG": cfg.get("eval_params", {}).get("gdiag_inject_deg", 2.0),
            "GDIAG_GENUINE_RECOVERY_MIN": cfg.get("eval_params", {}).get(
                "gdiag_genuine_recovery_min", 95.0),
            "GDIAG_SIGNED_SLOPE_MIN": cfg.get("eval_params", {}).get(
                "gdiag_signed_slope_min", 0.8),
            "GDIAG_ZD_MAX_DEG": cfg.get("eval_params", {}).get(
                "gdiag_zd_max_deg", 0.1),
            "REQUIRE_ACCEPTANCE_GATE": cfg.get("eval_params", {}).get(
                "require_acceptance_gate", False),
            "CF_BEV_R_ITER_STEPS": cfg.get("eval_params", {}).get("cf_bev_r_iter_steps", 0),
            "EXCLUDE_SEQS": cfg.get("eval_params", {}).get("exclude_seqs", None),
            "PROJFUSION_ROOT": cfg.get("projfusion_root",
                                       "/mnt/drtraining/user/dahailu/code/ProjFusion"),
            "MODELS": cfg.get("models", DEFAULT_MODELS),
            "BAG_EVAL_DIR": cfg.get("bag_eval_dir", None),
        }
        _validate_eval_models(config["MODELS"], config_path)
        if cfg.get("eval_settings") or cfg.get("eval_config"):
            print(f"[WARN] {config_path}: eval_settings/eval_config blocks are ignored by "
                  f"run_generalization_eval.py — use top-level test_data + eval_params instead.")
        print(f"[Config] Loaded from: {config_path}")
        print(f"  Models: {len(config['MODELS'])}, Angle: {config['ANGLE_RANGE']}°, Trans: {config['TRANS_RANGE']}m")
        return config

    root = DEFAULT_BEVCALIB_ROOT
    return {
        "BEVCALIB_ROOT": root,
        "MODELS_DIR": os.path.join(root, "logs/all_training_data"),
        "TEST_DATA": "/mnt/drtraining/user/dahailu/data/bevcalib/test_data",
        "OUTPUT_DIR": os.path.join(root, "logs/multi_eval_generalization_5deg"),
        "EVAL_SCRIPT": os.path.join(root, "evaluate_checkpoint.py"),
        "ANGLE_RANGE": 5.0,
        "TRANS_RANGE": 0.3,
        "BATCH_SIZE": 8,
        "VIS_INTERVAL": 200,
        "TIMEOUT": 1800,
        "EVAL_SAMPLE_STEP": None,
        "EVAL_MAX_FRAMES_PER_SEQ": None,
        "SHORTCUT_DIAG": False,
        "PROJFUSION_ROOT": "/mnt/drtraining/user/dahailu/code/ProjFusion",
        "MODELS": DEFAULT_MODELS,
        "BAG_EVAL_DIR": None,
    }


def _detect_gpu_count():
    """Detect available CUDA GPUs."""
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return _torch.cuda.device_count()
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except Exception:
        pass
    return 1


def parse_script_args():
    parser = argparse.ArgumentParser(description="Multi-model generalization evaluation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. configs/eval_generalization.yaml)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--angle_range", type=float, default=None,
                        help="Override eval angle range (degrees)")
    parser.add_argument("--trans_range", type=float, default=None,
                        help="Override eval translation range (meters)")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Multi-GPU parallel evaluation: 0=sequential (default), "
                             "N=use N GPUs in parallel, -1=auto-detect all GPUs")
    parser.add_argument("--eval_sample_step", type=int, default=None,
                        help="测试数据采样步长 (每隔N帧取1帧, None=使用全部帧)")
    parser.add_argument("--eval_max_frames_per_seq", type=int, default=None,
                        help="测试集每序列最多帧数 (与 eval_sample_step 互斥)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="强制重新评估所有模型 (忽略已有结果)")
    parser.add_argument("--shortcut_diag", action="store_true", default=False,
                        help="对每个模型运行 shortcut 诊断测试 (fixed-bias, invariance, ablation, GradCAM)")
    parser.add_argument("--generalization_diag", action="store_true", default=False,
                        help="对每个模型运行泛化诊断 (zero-drift, inject, shortcut-resistance)")
    parser.add_argument("--gdiag_inject_deg", type=float, default=None,
                        help="泛化诊断注入角度 (default: 2.0°)")
    parser.add_argument("--cf_bev_r_iter_steps", type=int, default=0,
                        help="CF-BEV-R gdiag 迭代推理步数 (0=单步, 2=部署对齐)")
    parser.add_argument("--exclude_seqs", type=str, default=None,
                        help="逗号分隔的序列ID列表，评估时跳过这些序列 (例如: seq07,seq12)")
    parser.add_argument("--bag_eval_dir", type=str, default=None,
                        help="BAG 泛化评估输出根目录 (run_bag_calibration.py 输出), "
                             "用于生成跨模型 BAG 泛化汇总排行报告")
    parser.add_argument("--report_only", action="store_true", default=False,
                        help="跳过评估，仅从已有结果重新生成报告和图表")
    parser.add_argument("--gdiag_only", action="store_true", default=False,
                        help="仅补跑泛化诊断 (需已有 extrinsics_and_errors.txt, 跳过主评估)")
    return parser.parse_args()


# Parse args and load config at module level for backward compatibility
_script_args = parse_script_args()
CFG = load_config(_script_args.config)
if _script_args.output_dir:
    CFG["OUTPUT_DIR"] = _script_args.output_dir
if _script_args.angle_range is not None:
    CFG["ANGLE_RANGE"] = _script_args.angle_range
if _script_args.trans_range is not None:
    CFG["TRANS_RANGE"] = _script_args.trans_range
if _script_args.bag_eval_dir:
    CFG["BAG_EVAL_DIR"] = _script_args.bag_eval_dir

_PARALLEL_GPUS = _script_args.parallel
if _PARALLEL_GPUS == -1:
    _PARALLEL_GPUS = _detect_gpu_count()
elif _PARALLEL_GPUS == 0:
    _PARALLEL_GPUS = 1

MODELS = CFG["MODELS"]
BEVCALIB_ROOT = CFG["BEVCALIB_ROOT"]
MODELS_DIR = CFG["MODELS_DIR"]
TEST_DATA = CFG["TEST_DATA"]
OUTPUT_DIR = CFG["OUTPUT_DIR"]
EVAL_SCRIPT = CFG["EVAL_SCRIPT"]
ANGLE_RANGE = CFG["ANGLE_RANGE"]
TRANS_RANGE = CFG["TRANS_RANGE"]
BATCH_SIZE = CFG["BATCH_SIZE"]
VIS_INTERVAL = CFG["VIS_INTERVAL"]
EVAL_TIMEOUT = CFG["TIMEOUT"]
EVAL_SAMPLE_STEP = _script_args.eval_sample_step if _script_args.eval_sample_step is not None else CFG.get("EVAL_SAMPLE_STEP")
_EVAL_MF_CLI = getattr(_script_args, "eval_max_frames_per_seq", None)
EVAL_MAX_FRAMES_PER_SEQ = _EVAL_MF_CLI if _EVAL_MF_CLI is not None else CFG.get("EVAL_MAX_FRAMES_PER_SEQ")
SHORTCUT_DIAG = getattr(_script_args, "shortcut_diag", False) or CFG.get("SHORTCUT_DIAG", False)
GENERALIZATION_DIAG = getattr(_script_args, "generalization_diag", False) or CFG.get("GENERALIZATION_DIAG", False)
GDIAG_INJECT_DEG = (_script_args.gdiag_inject_deg
                    if _script_args.gdiag_inject_deg is not None
                    else CFG.get("GDIAG_INJECT_DEG", 2.0))
GDIAG_GENUINE_RECOVERY_MIN = CFG.get("GDIAG_GENUINE_RECOVERY_MIN", 95.0)
GDIAG_SIGNED_SLOPE_MIN = CFG.get("GDIAG_SIGNED_SLOPE_MIN", 0.8)
GDIAG_ZD_MAX_DEG = CFG.get("GDIAG_ZD_MAX_DEG", 0.1)
REQUIRE_ACCEPTANCE_GATE = bool(CFG.get("REQUIRE_ACCEPTANCE_GATE", False))
CF_BEV_R_ITER_STEPS = int(getattr(_script_args, "cf_bev_r_iter_steps", 0) or CFG.get("CF_BEV_R_ITER_STEPS", 0) or 0)
GDIAG_ONLY = getattr(_script_args, "gdiag_only", False)
_EXCLUDE_SEQS_CLI = getattr(_script_args, "exclude_seqs", None)
EXCLUDE_SEQS = _EXCLUDE_SEQS_CLI if _EXCLUDE_SEQS_CLI else CFG.get("EXCLUDE_SEQS")
PROJFUSION_ROOT = CFG.get("PROJFUSION_ROOT", "/mnt/drtraining/user/dahailu/code/ProjFusion")
BAG_EVAL_DIR = CFG.get("BAG_EVAL_DIR")
GDIAG_MIN_SEQS = 10


def _is_gdiag_complete(gdiag_json):
    """Return True if generalization_diagnostics.json looks like a full run."""
    if not os.path.isfile(gdiag_json):
        return False
    try:
        with open(gdiag_json, 'r') as f:
            data = json.load(f)
        n_seqs = data.get('composite', {}).get('raw', {}).get('n_seqs_evaluated', 0)
        acceptance_ok = (not REQUIRE_ACCEPTANCE_GATE or 'acceptance_gate' in data)
        return int(n_seqs) >= GDIAG_MIN_SEQS and acceptance_ok
    except Exception:
        return False


def _resolve_ckpt_path(mcfg, model_base):
    """Resolve checkpoint path with optional fallbacks (e.g. dual → medw → latest)."""
    scratch_label = mcfg.get("scratch_label")
    if scratch_label:
        ckpt_dir = os.path.join(model_base, f"{scratch_label}/checkpoint")
    else:
        ckpt_dir = os.path.join(model_base,
                                os.path.basename(MODELS_DIR) + "_scratch/checkpoint")
    ckpt_names = [mcfg.get("ckpt", "ckpt_best_val.pth")]
    for alt in mcfg.get("ckpt_fallback", []) or []:
        if alt not in ckpt_names:
            ckpt_names.append(alt)
    for name in ckpt_names:
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            return path, ckpt_dir
    import glob as _glob
    for name in ckpt_names:
        hits = _glob.glob(os.path.join(model_base, "*_scratch/checkpoint", name))
        if hits:
            return hits[0], os.path.dirname(hits[0])
    return os.path.join(ckpt_dir, ckpt_names[0]), ckpt_dir


def parse_eval_stats(extrinsics_path):
    """Parse EVALUATION STATISTICS block from extrinsics_and_errors.txt."""
    if not os.path.isfile(extrinsics_path):
        return None
    with open(extrinsics_path, 'r') as f:
        text = f.read()
    if "EVALUATION STATISTICS" not in text:
        return None
    block = text[text.find("EVALUATION STATISTICS"):]

    result = {}
    m = re.search(r'Total samples evaluated:\s*(\d+)', block)
    result['samples'] = int(m.group(1)) if m else 0

    name_maps = {
        'rot': {'Total': 'rot_error', 'Roll (X)': 'roll_error', 'Roll (LiDAR-X)': 'roll_error',
                'Pitch (Y)': 'pitch_error', 'Pitch (LiDAR-Y)': 'pitch_error',
                'Yaw (Z)': 'yaw_error', 'Yaw (LiDAR-Z)': 'yaw_error'},
        'trans': {'Total': 'trans_error', 'X (Fwd)': 'fwd_error',
                  'Y (Lat)': 'lat_error', 'Z (Ht)': 'ht_error'},
    }

    in_section = None
    for line in block.split('\n'):
        s = line.strip()
        if 'Rotation Errors' in s and 'Average' not in s:
            in_section = 'rot'
            continue
        elif 'Translation Errors' in s and 'Average' not in s:
            in_section = 'trans'
            continue
        elif s.startswith('===') or s.startswith('AVERAGE'):
            in_section = None
            continue

        if in_section is None:
            continue
        active = name_maps[in_section]
        for display_name, key in active.items():
            if s.startswith(display_name):
                parts = s.split()
                offset = len(display_name.split()) if display_name != 'Total' else 1
                try:
                    vals = [float(x) for x in parts[offset:offset+8]]
                    result[f'{key}_mean'] = vals[0]
                    result[f'{key}_std'] = vals[1]
                    result[f'{key}_min'] = vals[2]
                    result[f'{key}_median'] = vals[3]
                    result[f'{key}_p90'] = vals[4]
                    result[f'{key}_p95'] = vals[5]
                    result[f'{key}_p99'] = vals[6]
                    result[f'{key}_max'] = vals[7]
                except (ValueError, IndexError):
                    pass

    # Parse PER-SEQUENCE STATISTICS block
    per_seq = {}
    seq_block_start = text.find("PER-SEQUENCE STATISTICS")
    if seq_block_start >= 0:
        seq_block = text[seq_block_start:]
        for line in seq_block.split('\n'):
            s = line.strip()
            if not s or s.startswith('=') or s.startswith('-') or s.startswith('Seq') or s.startswith('PER'):
                continue
            parts = s.split()
            if len(parts) >= 10:
                try:
                    seq_id = parts[0]
                    per_seq[seq_id] = {
                        'samples': int(parts[1]),
                        'rot_mean': float(parts[2]),
                        'rot_std': float(parts[3]),
                        'rot_median': float(parts[4]),
                        'rot_p95': float(parts[5]),
                        'rot_max': float(parts[6]),
                        'roll_mean': float(parts[7]),
                        'pitch_mean': float(parts[8]),
                        'yaw_mean': float(parts[9]),
                    }
                except (ValueError, IndexError):
                    pass
    result['per_sequence'] = per_seq

    # Parse Sequence Boundaries (scan until separator or blank line)
    seq_bounds = []
    bounds_start = text.find("Sequence Boundaries:")
    if bounds_start >= 0:
        for line in text[bounds_start:].split('\n')[1:]:
            m_b = re.match(r'\s*Seq\s+(\S+):\s*samples\s+(\d+)\s*-\s*(\d+)\s*\((\d+)\s*frames\)', line)
            if m_b:
                seq_bounds.append({
                    'seq': m_b.group(1), 'start': int(m_b.group(2)),
                    'end': int(m_b.group(3)), 'count': int(m_b.group(4)),
                })
            elif line.strip().startswith('=') or (line.strip() == '' and seq_bounds):
                break
    result['seq_boundaries'] = seq_bounds

    # Parse Macro-Averaged metrics
    macro_match = re.search(r'Macro \(sequence-level\):\s*([\d.]+)', text)
    if macro_match:
        result['macro_rot_mean'] = float(macro_match.group(1))
    primary_match = re.search(r'PRIMARY_METRIC:\s*(\w+)\s+([\d.]+)', text)
    if primary_match:
        result['primary_metric_type'] = primary_match.group(1)
        result['primary_metric_value'] = float(primary_match.group(2))

    return result if 'rot_error_mean' in result else None


def parse_train_log_final(log_path):
    """Parse the final epoch metrics from train.log."""
    if not os.path.isfile(log_path):
        return None

    last_train = None
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            if 'Train Pose Error' not in line:
                continue
            rot_m = re.search(
                r'Rot:\s*([\d.]+).*?(?:R|Roll):([\d.]+).*?(?:P|Pitch):([\d.]+).*?(?:Y|Yaw):([\d.]+)', line)
            trans_m = re.search(
                r'Trans:\s*([\d.]+)m\s*\((?:Fwd|Forward):([\d.]+).*?(?:Lat|Lateral):([\d.]+).*?(?:Ht|Height):([\d.]+)', line)
            if rot_m:
                last_train = {
                    'rot_error': float(rot_m.group(1)),
                    'roll': float(rot_m.group(2)),
                    'pitch': float(rot_m.group(3)),
                    'yaw': float(rot_m.group(4)),
                }
                if trans_m:
                    last_train['trans_error'] = float(trans_m.group(1))
                    last_train['fwd'] = float(trans_m.group(2))
                    last_train['lat'] = float(trans_m.group(3))
                    last_train['ht'] = float(trans_m.group(4))
    return last_train


def _detect_backbone_from_model(mcfg):
    """Detect backbone type from training log or dir_name.

    Priority: train.log > dir_name > default 'Swin'.
    """
    if mcfg.get('fusion_backend') == 'paper_explicit_bev':
        return 'ResNet-50 (paper)'
    model_base = _resolve_model_base(mcfg)
    train_log = os.path.join(model_base, "train.log")
    if os.path.isfile(train_log):
        with open(train_log, 'r', errors='ignore') as f:
            for line in f:
                if 'backbone=' in line:
                    if 'backbone=dinov2' in line:
                        return 'DINOv2'
                    elif 'backbone=swin' in line.lower():
                        return 'Swin'
                if '[DINOv2Encoder]' in line:
                    return 'DINOv2'
                if '[SwinTransformer]' in line:
                    return 'Swin'
    if 'dinov2' in mcfg.get('dir_name', '').lower():
        return 'DINOv2'
    return 'Swin'


def _resolve_model_base(mcfg):
    """Resolve model base directory, respecting per-model base_dir override."""
    per_model_base_dir = mcfg.get("base_dir")
    if per_model_base_dir:
        if not os.path.isabs(per_model_base_dir):
            per_model_base_dir = os.path.join(BEVCALIB_ROOT, per_model_base_dir)
        return os.path.join(per_model_base_dir, mcfg["dir_name"])
    return os.path.join(MODELS_DIR, mcfg["dir_name"])


def _build_eval_cmd_and_env(mcfg, per_model_dir):
    """Build subprocess command and env for a single model evaluation."""
    model_base = _resolve_model_base(mcfg)
    ckpt_path, _ckpt_dir = _resolve_ckpt_path(mcfg, model_base)
    if not os.path.isfile(ckpt_path):
        return None, None, ckpt_path

    env = os.environ.copy()
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("USE_DRCV_BACKEND", None)
    env["BEV_ZBOUND_STEP"] = mcfg["bev_zbound_step"]
    env["HF_HUB_OFFLINE"] = "1"
    env["PROJFUSION_ROOT"] = mcfg.get("projfusion_root", PROJFUSION_ROOT)
    if "use_drcv" in mcfg:
        env["USE_DRCV_BACKEND"] = "1" if mcfg["use_drcv"] else "0"
    elif mcfg.get("fusion_backend") in ("geo_match_proj", "paper_explicit_bev") or mcfg.get("gmp", False):
        env["USE_DRCV_BACKEND"] = "0"
    for env_key in ("BEV_XBOUND_MIN", "BEV_XBOUND_MAX",
                    "BEV_YBOUND_MIN", "BEV_YBOUND_MAX", "BEV_XY_STEP"):
        if env_key.lower() in mcfg:
            env[env_key] = str(mcfg[env_key.lower()])

    use_drinfer_backend = mcfg.get("backend", "pytorch") == "drinfer"

    if use_drinfer_backend:
        drinfer_eval_script = os.path.join(
            os.path.dirname(EVAL_SCRIPT), "evaluate_drinfer.py")
        export_dir = mcfg.get("export_dir", "")
        if not export_dir:
            model_base_dr = _resolve_model_base(mcfg)
            export_dir = os.path.join(model_base_dr, "drinfer")
        cmd = [
            sys.executable, drinfer_eval_script,
            "--ckpt_path", ckpt_path,
            "--export_dir", export_dir,
            "--dataset_root", TEST_DATA,
            "--output_dir", per_model_dir,
            "--angle_range_deg", str(ANGLE_RANGE),
            "--trans_range", str(TRANS_RANGE),
            "--use_full_dataset",
            "--max_batches", "0",
            "--vis_interval", str(VIS_INTERVAL),
            "--batch_size", str(mcfg.get("batch_size", BATCH_SIZE)),
        ]
        if "rotation_only" in mcfg:
            cmd.extend(["--rotation_only", str(mcfg["rotation_only"])])
        if mcfg.get("compare_pytorch"):
            cmd.append("--compare_pytorch")
        if mcfg.get("model_name"):
            cmd.extend(["--model_name", str(mcfg["model_name"])])
        if mcfg.get("model_version"):
            cmd.extend(["--model_version", str(mcfg["model_version"])])
        if mcfg.get("max_attn_tokens") is not None:
            cmd.extend(["--max_attn_tokens", str(mcfg["max_attn_tokens"])])
        if mcfg.get("use_drcv"):
            cmd.append("--use_drcv")
        if mcfg.get("use_deformable"):
            cmd.extend(["--deformable", "1"])
        if mcfg.get("max_attn_tokens") is not None:
            cmd.extend(["--max_attn_tokens", str(mcfg["max_attn_tokens"])])
    else:
        cmd = [
            sys.executable, EVAL_SCRIPT,
            "--ckpt_path", ckpt_path,
            "--dataset_root", TEST_DATA,
            "--output_dir", per_model_dir,
            "--angle_range_deg", str(ANGLE_RANGE),
            "--trans_range", str(TRANS_RANGE),
            "--use_full_dataset",
            "--max_batches", "0",
            "--vis_interval", str(VIS_INTERVAL),
            "--batch_size", str(mcfg.get("batch_size", BATCH_SIZE)),
        ]

    # Checkpoint-first: model architecture params are auto-detected from
    # checkpoint by evaluate_checkpoint.py.  Only pass when YAML explicitly
    # overrides the checkpoint value (e.g. testing rotation_only=1 on a
    # joint-trained model).
    _OPTIONAL_OVERRIDES = [
        "rotation_only", "deformable", "use_mlp_head", "bev_pool_factor",
        "voxel_mode", "scatter_reduce", "to_bev_mode", "fuser_type",
        "use_foundation_depth", "depth_model_type", "fd_mode",
        "intrinsic_input", "target_width", "target_height",
        "rotation_target_definition",
        "use_hard_route_eval",
        "pitch_vertical_bands",
    ]
    _BOOL_TO_INT = {"intrinsic_input", "rotation_only", "deformable",
                     "use_mlp_head", "use_foundation_depth"}
    for p in _OPTIONAL_OVERRIDES:
        if p in mcfg:
            val = mcfg[p]
            if p in _BOOL_TO_INT and isinstance(val, bool):
                val = int(val)
            cmd.extend([f"--{p}", str(val)])

    if "use_drcv" in mcfg:
        if mcfg["use_drcv"]:
            if not use_drinfer_backend:
                cmd.append("--use_drcv")
            env["USE_DRCV_BACKEND"] = "1"
        else:
            env["USE_DRCV_BACKEND"] = "0"
    if mcfg.get("use_foundation_depth"):
        env["HF_HUB_OFFLINE"] = "1"
    if EVAL_SAMPLE_STEP is not None:
        cmd.extend(["--eval_sample_step", str(EVAL_SAMPLE_STEP)])
    if EVAL_MAX_FRAMES_PER_SEQ is not None:
        cmd.extend(["--eval_max_frames_per_seq", str(EVAL_MAX_FRAMES_PER_SEQ)])
    if mcfg.get("data_balance"):
        cmd.extend(["--data_balance", str(mcfg["data_balance"])])
    if mcfg.get("zero_image"):
        cmd.append("--zero_image")
    if mcfg.get("shortcut_diag", False) or SHORTCUT_DIAG:
        cmd.append("--shortcut_diag")
    if mcfg.get("generalization_diag", False) or GENERALIZATION_DIAG:
        cmd.append("--generalization_diag")
        cmd.extend(["--gdiag_inject_deg", str(mcfg.get("gdiag_inject_deg", GDIAG_INJECT_DEG))])
        cmd.extend([
            "--gdiag_genuine_recovery_min",
            str(mcfg.get("gdiag_genuine_recovery_min", GDIAG_GENUINE_RECOVERY_MIN)),
            "--gdiag_signed_slope_min",
            str(mcfg.get("gdiag_signed_slope_min", GDIAG_SIGNED_SLOPE_MIN)),
            "--gdiag_zd_max_deg",
            str(mcfg.get("gdiag_zd_max_deg", GDIAG_ZD_MAX_DEG)),
        ])
        _iter_steps = int(mcfg.get("cf_bev_r_iter_steps", CF_BEV_R_ITER_STEPS) or 0)
        if _iter_steps > 0:
            cmd.extend(["--cf_bev_r_iter_steps", str(_iter_steps)])
    if GDIAG_ONLY:
        cmd.append("--gdiag_only")
    _ex_seqs = mcfg.get("exclude_seqs") or EXCLUDE_SEQS
    if _ex_seqs:
        cmd.extend(["--exclude_seqs", str(_ex_seqs)])

    return cmd, env, ckpt_path


def _copy_existing_results():
    """Check/copy existing test_data_eval results for models without outputs."""
    for mcfg in MODELS:
        label = mcfg["label"]
        per_model_dir = os.path.join(OUTPUT_DIR, label)
        extrinsics_path = os.path.join(per_model_dir, "extrinsics_and_errors.txt")
        if os.path.isfile(extrinsics_path):
            continue
        alt_path = os.path.join(_resolve_model_base(mcfg),
                                 "test_data_eval/extrinsics_and_errors.txt")
        if os.path.isfile(alt_path):
            with open(alt_path, 'r') as f:
                if "EVALUATION STATISTICS" in f.read():
                    import shutil
                    os.makedirs(per_model_dir, exist_ok=True)
                    shutil.copy2(alt_path, extrinsics_path)
                    src_dir = os.path.dirname(alt_path)
                    for fn in os.listdir(src_dir):
                        if fn.endswith('.png'):
                            shutil.copy2(os.path.join(src_dir, fn),
                                         os.path.join(per_model_dir, fn))
                    print(f"  Copied existing results for {label} from test_data_eval")


def _precheck_models():
    """Pre-check all models: classify into done/ready/missing.

    Returns (done, ready, missing) where each is a list of
    (idx, label, mcfg, per_model_dir, ckpt_path_or_None, reason).
    """
    done, ready, missing = [], [], []

    for idx, mcfg in enumerate(MODELS):
        label = mcfg["label"]
        per_model_dir = os.path.join(OUTPUT_DIR, label)
        extrinsics_path = os.path.join(per_model_dir, "extrinsics_and_errors.txt")
        _wants_gdiag = mcfg.get("generalization_diag", False) or GENERALIZATION_DIAG
        _gdiag_json = os.path.join(per_model_dir, "generalization_diagnostics.json")

        if GDIAG_ONLY:
            if not os.path.isfile(extrinsics_path):
                missing.append((idx, label, mcfg, per_model_dir, None,
                                "gdiag_only: 缺少 extrinsics_and_errors.txt"))
                continue
            if _wants_gdiag and _is_gdiag_complete(_gdiag_json) and not getattr(_script_args, 'force', False):
                done.append((idx, label, mcfg, per_model_dir, None, "gdiag complete"))
                continue
            if not (_wants_gdiag or GENERALIZATION_DIAG):
                done.append((idx, label, mcfg, per_model_dir, None, "gdiag not requested"))
                continue
            model_base = _resolve_model_base(mcfg)
            ckpt_path, _ckpt_dir = _resolve_ckpt_path(mcfg, model_base)
            if not os.path.isfile(ckpt_path):
                missing.append((idx, label, mcfg, per_model_dir, ckpt_path, "ckpt not found"))
                continue
            ready.append((idx, label, mcfg, per_model_dir, ckpt_path, "gdiag pending"))
            continue

        if os.path.isfile(extrinsics_path) and not getattr(_script_args, 'force', False):
            _is_complete = False
            with open(extrinsics_path, 'r') as f:
                _content = f.read()
                _is_complete = "EVALUATION STATISTICS" in _content
            if not _is_complete:
                _ta_path = os.path.join(per_model_dir, "temporal_aggregation.txt")
                _is_complete = os.path.isfile(_ta_path)
            if _wants_gdiag and not _is_gdiag_complete(_gdiag_json):
                _is_complete = False
            if _is_complete:
                done.append((idx, label, mcfg, per_model_dir, None, "eval complete"))
                continue

        model_base = _resolve_model_base(mcfg)
        ckpt_path, _ckpt_dir = _resolve_ckpt_path(mcfg, model_base)
        if not os.path.isfile(ckpt_path):
            train_log = os.path.join(model_base, "train.log")
            tried = mcfg.get("ckpt", "?")
            fallbacks = mcfg.get("ckpt_fallback", []) or []
            tried_all = [tried] + list(fallbacks)
            if os.path.isdir(model_base):
                reason = f"ckpt not found (tried {tried_all}), dir exists"
                if os.path.isfile(train_log):
                    reason += " (training may be in progress)"
            else:
                reason = f"model dir not found: {model_base}"
            missing.append((idx, label, mcfg, per_model_dir, ckpt_path, reason))
            continue

        ready.append((idx, label, mcfg, per_model_dir, ckpt_path, "ready"))

    return done, ready, missing


def run_evaluations():
    """Run evaluations for all models, optionally in multi-GPU parallel."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    done, ready, missing = _precheck_models()

    print(f"\n{'='*80}")
    print(f"Pre-check: {len(MODELS)} models in config")
    if GDIAG_ONLY:
        print("  模式: gdiag_only (仅补跑泛化诊断)")
    print(f"  ✓ Already evaluated: {len(done)}")
    print(f"  ⏳ Ready to evaluate: {len(ready)}")
    print(f"  ✗ Missing (no checkpoint): {len(missing)}")
    print(f"{'='*80}")

    if done:
        print(f"\nAlready done ({len(done)}):")
        for _, label, _, _, _, reason in done:
            print(f"  ✓ {label}")

    if missing:
        print(f"\nMissing ({len(missing)}):")
        for _, label, _, _, _, reason in missing:
            print(f"  ✗ {label}: {reason}")

    if ready:
        print(f"\nWill evaluate ({len(ready)}):")
        for _, label, mcfg, _, ckpt_path, _ in ready:
            print(f"  ⏳ {label} [{ckpt_path}]")

    if not ready:
        if missing:
            print(f"\nNo models ready. {len(missing)} model(s) missing checkpoints.")
        else:
            print("\nAll models already evaluated.")
        _copy_existing_results()
        return

    pending_tasks = []
    for idx, label, mcfg, per_model_dir, ckpt_path, _ in ready:
        extrinsics_path = os.path.join(per_model_dir, "extrinsics_and_errors.txt")
        if os.path.isfile(extrinsics_path) and not GDIAG_ONLY:
            import shutil
            shutil.rmtree(per_model_dir, ignore_errors=True)

        cmd, env, _ = _build_eval_cmd_and_env(mcfg, per_model_dir)
        if cmd is None:
            continue

        pending_tasks.append({
            "idx": idx, "label": label, "mcfg": mcfg,
            "per_model_dir": per_model_dir, "cmd": cmd, "env": env,
            "ckpt_path": ckpt_path,
        })

    if not pending_tasks:
        _copy_existing_results()
        return

    num_gpus = _PARALLEL_GPUS
    if num_gpus <= 1:
        _run_evaluations_sequential(pending_tasks)
    else:
        actual_gpus = min(num_gpus, len(pending_tasks))
        if actual_gpus < num_gpus:
            print(f"\nOnly {len(pending_tasks)} model(s) to evaluate, "
                  f"using {actual_gpus}/{num_gpus} GPUs")
        _run_evaluations_parallel(pending_tasks, actual_gpus)

    _copy_existing_results()


def _run_evaluations_sequential(tasks):
    """Original sequential evaluation."""
    for task in tasks:
        idx, label = task["idx"], task["label"]
        mcfg, cmd, env = task["mcfg"], task["cmd"], task["env"]
        per_model_dir = task["per_model_dir"]

        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(MODELS)}] Evaluating: {label}")
        print(f"  ckpt: {task['ckpt_path']}")
        rot_disp = mcfg.get("rotation_only", "auto (checkpoint)")
        print(f"  BEV_ZBOUND_STEP={mcfg['bev_zbound_step']}, rotation_only={rot_disp}")
        print(f"{'='*80}")

        log_path = os.path.join(per_model_dir, "eval_run.log")
        os.makedirs(per_model_dir, exist_ok=True)

        _model_t0 = time.time()
        try:
            with open(log_path, 'w') as log_f:
                proc = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in proc.stdout:
                    sys.stdout.write(f"  {line}")
                    sys.stdout.flush()
                    log_f.write(line)
                proc.wait(timeout=EVAL_TIMEOUT)

            _model_elapsed = time.time() - _model_t0
            if proc.returncode != 0:
                print(f"  [ERROR] {label} failed (exit {proc.returncode}, 耗时: {_format_elapsed(_model_elapsed)})")
                print(f"  Log saved: {log_path}")
                continue
            print(f"  [OK] {label} evaluation complete (耗时: {_format_elapsed(_model_elapsed)})")
            print(f"  Log saved: {log_path}")
        except subprocess.TimeoutExpired:
            proc.kill()
            _model_elapsed = time.time() - _model_t0
            print(f"  [ERROR] {label} timed out ({EVAL_TIMEOUT}s, 耗时: {_format_elapsed(_model_elapsed)})")
            print(f"  Partial log: {log_path}")
        except Exception as e:
            _model_elapsed = time.time() - _model_t0
            print(f"  [ERROR] {label}: {e} (耗时: {_format_elapsed(_model_elapsed)})")
            print(f"  Log: {log_path}")


def _parse_progress_from_log(log_path):
    """Extract latest progress info from an eval log file.

    Returns (last_sample_idx, last_rot_error, status_line) or None.
    """
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, 'r', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return None

    last_sample = -1
    last_rot = None
    total_samples = None

    for line in reversed(lines):
        s = line.strip()
        if last_sample < 0:
            m = re.search(r'处理样本\s*(\d+)(?:/(\d+))?', s)
            if m:
                last_sample = int(m.group(1))
                if m.group(2):
                    total_samples = int(m.group(2))
            m2 = re.search(r'Sample\s+(\d+)', s)
            if m2 and last_sample < 0:
                last_sample = int(m2.group(1))
        if last_rot is None:
            m_rot = re.search(r'[Rr]ot[=:]\s*([\d.]+)', s)
            if m_rot:
                last_rot = float(m_rot.group(1))
        if 'Evaluation complete' in s or '评估完成' in s:
            m_done = re.search(r'(\d+)\s*samples', s)
            if m_done:
                return (int(m_done.group(1)), last_rot, "DONE")
        if last_sample >= 0 and last_rot is not None:
            break

    if last_sample < 0:
        for line in lines:
            if '加载数据集' in line or 'Loading dataset' in line:
                return (0, None, "loading")
            if '初始化模型' in line or 'Initializing' in line:
                return (0, None, "init")
        return None

    pct = ""
    if total_samples and total_samples > 0:
        pct = f" ({100 * last_sample / total_samples:.0f}%)"
    rot_str = f" rot={last_rot:.2f}°" if last_rot is not None else ""
    return (last_sample, last_rot, f"sample {last_sample}{pct}{rot_str}")


def _print_wave_progress(running, wave_idx, num_waves, elapsed_sec):
    """Print a compact progress summary for all processes in a wave."""
    status_parts = []
    all_done = True
    for r in running:
        label = r["label"]
        proc = r["proc"]
        gpu_id = r["gpu_id"]
        is_alive = proc.poll() is None
        if is_alive:
            all_done = False

        progress = _parse_progress_from_log(r["log_path"])
        if progress is not None:
            _, _, status = progress
            mark = "⏳" if is_alive else ("✓" if proc.returncode == 0 else "✗")
            status_parts.append(f"  GPU{gpu_id} {mark} {label}: {status}")
        else:
            mark = "⏳" if is_alive else ("✓" if proc.returncode == 0 else "✗")
            status_parts.append(f"  GPU{gpu_id} {mark} {label}: starting...")

    elapsed_str = f"{int(elapsed_sec)}s"
    if elapsed_sec >= 60:
        elapsed_str = f"{int(elapsed_sec // 60)}m{int(elapsed_sec % 60)}s"
    header = f"[Wave {wave_idx+1}/{num_waves}] {elapsed_str} elapsed"
    print(f"\n{header}")
    for part in status_parts:
        print(part)
    return all_done


def _run_evaluations_parallel(tasks, num_gpus):
    """Run evaluations in parallel batches, each model on a separate GPU.

    Models are grouped into waves of `num_gpus`. Within each wave,
    each subprocess is pinned to a unique GPU via CUDA_VISIBLE_DEVICES.
    Progress is polled every 30 seconds.
    """
    import math
    import time as _time

    total = len(tasks)
    num_waves = math.ceil(total / num_gpus)
    print(f"\n{'='*80}")
    print(f"Parallel evaluation: {total} models on {num_gpus} GPUs ({num_waves} wave(s))")
    print(f"{'='*80}")

    POLL_INTERVAL = 30

    for wave_idx in range(num_waves):
        wave_start = wave_idx * num_gpus
        wave_end = min(wave_start + num_gpus, total)
        wave_tasks = tasks[wave_start:wave_end]

        print(f"\n--- Wave {wave_idx+1}/{num_waves}: "
              f"{', '.join(t['label'] + f' [GPU {gi}]' for gi, t in enumerate(wave_tasks))} ---")

        running = []
        wave_t0 = _time.monotonic()
        for gpu_id, task in enumerate(wave_tasks):
            label = task["label"]
            per_model_dir = task["per_model_dir"]
            env = task["env"].copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            log_path = os.path.join(per_model_dir, "eval_run.log")
            os.makedirs(per_model_dir, exist_ok=True)
            log_f = open(log_path, 'w')

            print(f"  [GPU {gpu_id}] Starting: {label}")
            proc = subprocess.Popen(
                task["cmd"], env=env,
                stdout=log_f, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            running.append({
                "proc": proc, "label": label, "gpu_id": gpu_id,
                "log_path": log_path, "log_f": log_f,
            })

        while True:
            alive = [r for r in running if r["proc"].poll() is None]
            if not alive:
                break
            elapsed = _time.monotonic() - wave_t0
            if elapsed > EVAL_TIMEOUT:
                for r in alive:
                    r["proc"].kill()
                    print(f"  [GPU {r['gpu_id']}] [TIMEOUT] {r['label']} killed "
                          f"after {EVAL_TIMEOUT}s")
                break
            _print_wave_progress(running, wave_idx, num_waves, elapsed)
            _time.sleep(POLL_INTERVAL)

        for r in running:
            r["proc"].wait()
            r["log_f"].close()

        elapsed = _time.monotonic() - wave_t0
        _print_wave_progress(running, wave_idx, num_waves, elapsed)

        for r in running:
            rc = r["proc"].returncode
            if rc == 0:
                print(f"  [GPU {r['gpu_id']}] [OK] {r['label']} complete")
            else:
                print(f"  [GPU {r['gpu_id']}] [ERROR] {r['label']} exit={rc} "
                      f"(log: {r['log_path']})")

        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        print(f"--- Wave {wave_idx+1}/{num_waves} finished ({elapsed_str}) ---")


def collect_all_stats():
    """Collect stats from all completed evaluations."""
    all_stats = []
    for mcfg in MODELS:
        label = mcfg["label"]
        per_model_dir = os.path.join(OUTPUT_DIR, label)
        extrinsics_path = os.path.join(per_model_dir, "extrinsics_and_errors.txt")
        stats = parse_eval_stats(extrinsics_path)
        if stats:
            stats['label'] = label
            stats['config'] = mcfg
            train_log = os.path.join(_resolve_model_base(mcfg), "train.log")
            train_metrics = parse_train_log_final(train_log)
            stats['train_metrics'] = train_metrics
            stats['backbone'] = _detect_backbone_from_model(mcfg)
            ta_path = os.path.join(per_model_dir, "temporal_aggregation.txt")
            stats['temporal'] = _parse_temporal_aggregation(ta_path)
            gdiag_path = os.path.join(per_model_dir, "generalization_diagnostics.json")
            stats['gdiag'] = _parse_generalization_diagnostics(gdiag_path)
            all_stats.append(stats)
            gdiag_info = ""
            if stats['gdiag']:
                gs = stats['gdiag'].get('composite', {}).get('GS_medw', -1)
                gdiag_info = f", GS_medw={gs:.4f}"
            print(f"  {label}: Mean Rot={stats['rot_error_mean']:.3f} deg, "
                  f"P95={stats['rot_error_p95']:.3f} deg ({stats['samples']} samples){gdiag_info}")
        else:
            print(f"  {label}: NO RESULTS")
    return all_stats


def _parse_generalization_diagnostics(path):
    """Parse generalization_diagnostics.json for zero-drift / inject / shortcut metrics.

    Dynamically recomputes S2(correction) and GS_medw to fix the geodesic/RPY
    dimension mismatch: residuals are geodesic but were previously divided by
    per-axis RPY inject values, causing S2 to be clamped to 1.0 for most models.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}")
        return None

    comp = data.get('composite', {})
    raw = comp.get('raw', {})
    sub = comp.get('sub_scores', {})
    mm = data.get('multi_magnitude', {})
    fi = data.get('fixed_inject', {}).get('inject', {})
    zd_signed = raw.get('zero_drift_signed_rpy', [0, 0, 0])
    inject_deg = raw.get('inject_deg', 2.0)

    if not sub or not mm:
        return data

    def _zd_deduction(raw_resid, zd_vals, injected_axes=None):
        n = len(zd_vals)
        if n <= 0:
            return raw_resid
        if injected_axes is None:
            injected_axes = [True] * n
        n_axes = sum(injected_axes)
        if n_axes <= 0:
            return raw_resid
        inj_hat = [(1.0 / (n_axes ** 0.5) if injected_axes[i] else 0.0) for i in range(n)]
        zd_proj = sum(z * h for z, h in zip(zd_vals, inj_hat))
        deduction = max(0, min(zd_proj, raw_resid))
        return max(0, raw_resid - deduction)

    small_genuine_resids = []
    for m in ['0.5', '1.0']:
        if m in mm:
            raw_resid = mm[m].get('residual', -1)
            recovery = mm[m].get('recovery_pct', -999)
            if raw_resid >= 0 and recovery > -100:
                actual_inject_geo = raw_resid / (1.0 - recovery / 100.0) if recovery < 99.9 else raw_resid * 10
                genuine_resid = _zd_deduction(raw_resid, zd_signed)
                small_genuine_resids.append(genuine_resid / max(actual_inject_geo, 1e-6))
    if small_genuine_resids:
        s2_corr = min(1.0, sum(small_genuine_resids) / len(small_genuine_resids))
    else:
        fi_injected = fi.get('mean_injected', inject_deg)
        raw_fi = fi.get('mean_residual', fi_injected)
        genuine_fi = _zd_deduction(raw_fi, zd_signed)
        s2_corr = min(1.0, genuine_fi / max(fi_injected, 1e-6))

    sub['correction'] = s2_corr
    w = [0.40, 0.30, 0.15, 0.15]
    gs_medw = (w[0] * sub.get('zero_drift', 0) + w[1] * s2_corr +
               w[2] * sub.get('shortcut', 0) + w[3] * sub.get('consistency', 0))
    comp['GS_medw'] = gs_medw

    fi_rot_mean = fi.get('mean_residual', inject_deg)
    fi_injected_geo = fi.get('mean_injected', inject_deg)
    genuine_fi_residual = _zd_deduction(fi_rot_mean, zd_signed)
    raw['genuine_recovery_pct'] = ((fi_injected_geo - genuine_fi_residual) / fi_injected_geo * 100) if fi_injected_geo > 1e-6 else 0

    return data


def _parse_temporal_aggregation(path):
    """Parse temporal_aggregation.txt for SVD/MED/BEST and optional per-sequence JSON."""
    import re
    result = {}
    if not os.path.isfile(path):
        return result
    try:
        with open(path, 'r') as f:
            content = f.read()
        pat = re.compile(
            r'Window=\s*(\d+)\s+\([^)]*\):\s+Rot=([\d.]+)°\s+\(R=([\d.]+)°\s+P=([\d.]+)°\s+Y=([\d.]+)°\)'
        )
        current_section = ""
        for line in content.split('\n'):
            if 'SVD-Mean' in line:
                current_section = 'svd'
            elif 'Robust Median' in line:
                current_section = 'med'
            elif 'Trimmed-Mean' in line:
                current_section = 'trm'
            elif 'ORACLE' in line and 'Bias Correction + Temporal' in line:
                current_section = 'oracle_combined'
            elif 'Bias Correction + Temporal' in line:
                current_section = 'combined'
            elif 'BEST RESULT' in line:
                best_text = line.split(':')[-1].strip()
                result['best_method'] = best_text
                continue
            m = pat.search(line)
            if m and current_section in ('svd', 'med', 'trm'):
                window = int(m.group(1))
                entry = {
                    'rot': float(m.group(2)), 'roll': float(m.group(3)),
                    'pitch': float(m.group(4)), 'yaw': float(m.group(5)),
                }
                result.setdefault(current_section, {})[window] = entry
            if current_section in ('combined', 'oracle_combined'):
                cm = re.search(
                    r'(bias\d+%\+\w+):\s+Rot=([\d.]+)°\s+\(R=([\d.]+)°\s+P=([\d.]+)°\s+Y=([\d.]+)°\)',
                    line
                )
                if cm:
                    key = cm.group(1).strip()
                    section_key = 'oracle' if current_section == 'oracle_combined' else 'combined'
                    result.setdefault(section_key, {})[key] = {
                        'rot': float(cm.group(2)), 'roll': float(cm.group(3)),
                        'pitch': float(cm.group(4)), 'yaw': float(cm.group(5)),
                    }
        best_key = result.get('best_method', '')
        all_pure = {}
        for sec in ('svd', 'med', 'trm'):
            for w, entry in result.get(sec, {}).items():
                all_pure[f"{sec.upper()}W{w}"] = entry
        if best_key and best_key in all_pure:
            result['best'] = all_pure[best_key]
            result['best']['method'] = best_key
        elif all_pure:
            bk = min(all_pure, key=lambda k: all_pure[k]['rot'])
            result['best'] = all_pure[bk]
            result['best']['method'] = bk
        oracle_candidates = result.get('oracle', result.get('combined', {}))
        if oracle_candidates:
            ok = min(oracle_candidates, key=lambda k: oracle_candidates[k]['rot'])
            result['oracle_best'] = oracle_candidates[ok]
            result['oracle_best']['method'] = ok
        mj = re.search(r'^\s*BEST_PER_SEQ_JSON:\s*(\S+)\s*$', content, re.MULTILINE)
        if mj:
            jp = os.path.join(os.path.dirname(os.path.abspath(path)), mj.group(1))
            if os.path.isfile(jp):
                with open(jp, 'r') as jf:
                    result['best_per_sequence'] = json.load(jf)
        deploy_pat = re.compile(
            r'MEDW\s*(\d+)\s+\(uniform\):\s+Rot=([\d.]+)°\s+\(R=([\d.]+)°\s+P=([\d.]+)°\s+Y=([\d.]+)°\)')
        deploy_sim = {}
        for dm in deploy_pat.finditer(content):
            deploy_sim[int(dm.group(1))] = {
                'rot': float(dm.group(2)), 'roll': float(dm.group(3)),
                'pitch': float(dm.group(4)), 'yaw': float(dm.group(5)),
            }
        if deploy_sim:
            result['deploy_sim'] = deploy_sim
        dsj = os.path.join(os.path.dirname(os.path.abspath(path)), "deploy_simulation.json")
        if os.path.isfile(dsj):
            with open(dsj, 'r') as djf:
                result['deploy_sim_detail'] = json.load(djf)
    except Exception:
        pass
    return result


def generate_charts(all_stats):
    """Generate comparison charts."""
    charts_dir = os.path.join(OUTPUT_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    labels = [s['label'] for s in all_stats]
    n = len(labels)

    z_color_map = {'z1': '#e74c3c', 'z5': '#e67e22', 'z10': '#2196F3'}
    def _color(label):
        for zk, c in z_color_map.items():
            if zk in label:
                return c
        return '#888'
    colors = [_color(l) for l in labels]

    def _save(fig, name):
        fig.savefig(os.path.join(charts_dir, name), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"   {name}")

    means = [s.get('rot_error_mean', 0) for s in all_stats]
    p95s = [s.get('rot_error_p95', 0) for s in all_stats]
    maxs = [s.get('rot_error_max', 0) for s in all_stats]
    medians = [s.get('rot_error_median', 0) for s in all_stats]
    stds = [s.get('rot_error_std', 0) for s in all_stats]

    # Chart 1: Mean / P95 / Max bar chart
    x = np.arange(n)
    w = 0.22
    fig, ax = plt.subplots(figsize=(max(14, n * 2), 7))
    b1 = ax.bar(x - w, means, w, label='Mean', color=colors, alpha=0.9, edgecolor='white', linewidth=1)
    b2 = ax.bar(x, p95s, w, label='P95', color=colors, alpha=0.6, edgecolor='white', linewidth=1)
    b3 = ax.bar(x + w, maxs, w, label='Max', color=colors, alpha=0.35, edgecolor='white', linewidth=1)
    for i, v in enumerate(means):
        ax.text(i - w, v + 0.15, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    for i, v in enumerate(p95s):
        ax.text(i, v + 0.15, f'{v:.2f}', ha='center', fontsize=8, color='#555')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Rotation Error (deg)', fontsize=12)
    ax.set_title('Generalization Test: Rotation Error (Mean / P95 / Max)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    _save(fig, 'rotation_error_bar.png')

    # Chart 2: Component breakdown (stacked)
    rolls = [s.get('roll_error_mean', 0) for s in all_stats]
    pitches = [s.get('pitch_error_mean', 0) for s in all_stats]
    yaws = [s.get('yaw_error_mean', 0) for s in all_stats]
    fig, ax = plt.subplots(figsize=(max(14, n * 2), 7))
    ax.bar(x, rolls, 0.55, label='Roll (LiDAR-X)', color='#1abc9c')
    ax.bar(x, pitches, 0.55, bottom=rolls, label='Pitch (LiDAR-Y)', color='#f39c12')
    bottoms = [r + p for r, p in zip(rolls, pitches)]
    ax.bar(x, yaws, 0.55, bottom=bottoms, label='Yaw (LiDAR-Z)', color='#8e44ad')
    for i, v in enumerate(means):
        ax.text(i, v + 0.15, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Mean Rotation Error (deg)', fontsize=12)
    ax.set_title('Rotation Error Component Breakdown (Roll + Pitch + Yaw)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    _save(fig, 'rotation_components_bar.png')

    # Chart 3: Model ranking (horizontal)
    sorted_idx = np.argsort(means)
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7)))
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_means = [means[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    bars = ax.barh(range(n), sorted_means, color=sorted_colors, edgecolor='white',
                   linewidth=1.5, height=0.6)
    for i, (bar, v) in enumerate(zip(bars, sorted_means)):
        ax.text(v + 0.15, i, f'{v:.2f} deg', va='center', fontsize=11, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_labels, fontsize=11)
    ax.set_xlabel('Mean Rotation Error (deg)', fontsize=12)
    ax.set_title('Model Ranking on test_data (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, 'model_ranking.png')

    # Chart 4: Train vs Test comparison
    train_rots = []
    test_rots = []
    valid_labels = []
    valid_colors = []
    degradations = []
    for s in all_stats:
        tm = s.get('train_metrics')
        if tm and tm.get('rot_error', 0) > 0:
            train_rots.append(tm['rot_error'])
            test_rots.append(s['rot_error_mean'])
            valid_labels.append(s['label'])
            valid_colors.append(_color(s['label']))
            degradations.append(s['rot_error_mean'] / tm['rot_error'])

    if valid_labels:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        xv = np.arange(len(valid_labels))
        w2 = 0.35
        b1 = ax1.bar(xv - w2/2, train_rots, w2, label='Train (final epoch)',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        b2 = ax1.bar(xv + w2/2, test_rots, w2, label='Test (test_data)',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                         f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.set_xticks(xv)
        ax1.set_xticklabels(valid_labels, rotation=25, ha='right', fontsize=10)
        ax1.set_ylabel('Rotation Error (deg)', fontsize=12)
        ax1.set_title('Training vs Test Error', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        deg_colors = ['#2ecc71' if d < 3 else '#f39c12' if d < 5 else '#e74c3c' for d in degradations]
        bars = ax2.bar(xv, degradations, color=deg_colors, alpha=0.8, edgecolor='black', linewidth=1)
        for bar, deg in zip(bars, degradations):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                     f'{deg:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax2.axhline(y=3, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='3x threshold')
        ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5x threshold')
        ax2.set_xticks(xv)
        ax2.set_xticklabels(valid_labels, rotation=25, ha='right', fontsize=10)
        ax2.set_ylabel('Degradation (Test / Train)', fontsize=12)
        ax2.set_title('Generalization Degradation', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        _save(fig, 'train_vs_test.png')

    # Chart 5: Translation errors for full-mode models
    trans_models = [s for s in all_stats if s.get('trans_error_mean') and s['trans_error_mean'] > 0]
    if trans_models:
        fig, ax = plt.subplots(figsize=(max(10, len(trans_models) * 2.5), 6))
        tlabels = [s['label'] for s in trans_models]
        xt = np.arange(len(tlabels))
        w3 = 0.2
        fwd_vals = [s.get('fwd_error_mean', 0) for s in trans_models]
        lat_vals = [s.get('lat_error_mean', 0) for s in trans_models]
        ht_vals = [s.get('ht_error_mean', 0) for s in trans_models]
        ax.bar(xt - w3, fwd_vals, w3, label='Forward (X)', color='#3498db')
        ax.bar(xt, lat_vals, w3, label='Lateral (Y)', color='#e74c3c')
        ax.bar(xt + w3, ht_vals, w3, label='Height (Z)', color='#2ecc71')
        for i in range(len(tlabels)):
            total = trans_models[i].get('trans_error_mean', 0)
            ax.text(i, max(fwd_vals[i], lat_vals[i], ht_vals[i]) + 0.005,
                    f'Total: {total:.4f}m', ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(xt)
        ax.set_xticklabels(tlabels, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('Translation Error (m)', fontsize=12)
        ax.set_title('Translation Error Components (Full-mode Models)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        _save(fig, 'translation_components.png')

    # Chart 6: Temporal aggregation convergence (multi-model)
    temporal_models = [s for s in all_stats if s.get('temporal', {}).get('svd')]
    if temporal_models:
        fig, ax = plt.subplots(figsize=(14, 8))
        cmap = plt.cm.get_cmap('tab10', len(temporal_models))
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'p', 'h']

        for idx, s in enumerate(temporal_models):
            svd_data = s['temporal'].get('svd', {})
            if not svd_data:
                continue
            ws = sorted(svd_data.keys())
            rots = [svd_data[w]['rot'] for w in ws]
            marker = markers[idx % len(markers)]
            ax.plot(ws, rots, color=cmap(idx), marker=marker, markersize=5,
                    linewidth=1.8, label=s['label'], alpha=0.85)

        ax.set_xscale('symlog', linthresh=2)
        ax.set_xlabel('Window Size (frames)', fontsize=12)
        ax.set_ylabel('Rotation Error (°)', fontsize=12)
        ax.set_title('Temporal Aggregation Convergence (SVD-Mean, All Models)',
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, 'temporal_convergence_all.png')

    # Chart 7: BEST aggregated ranking (horizontal bar)
    best_models = [s for s in all_stats
                   if s.get('temporal', {}).get('best', {}).get('rot') is not None]
    if best_models:
        sorted_by_best = sorted(best_models, key=lambda s: s['temporal']['best']['rot'])
        nb = len(sorted_by_best)
        fig, ax = plt.subplots(figsize=(12, max(4, nb * 0.7)))
        b_labels = [s['label'] for s in sorted_by_best]
        b_rots = [s['temporal']['best']['rot'] for s in sorted_by_best]
        b_methods = [s['temporal']['best'].get('method', '?') for s in sorted_by_best]
        bar_colors = ['#2ecc71' if r < 0.1 else '#f39c12' if r < 0.3 else '#e74c3c'
                      for r in b_rots]
        bars = ax.barh(range(nb), b_rots, color=bar_colors, edgecolor='white',
                       linewidth=1.5, height=0.6)
        for i, (bar, v, m) in enumerate(zip(bars, b_rots, b_methods)):
            ax.text(v + 0.005, i, f'{v:.3f}° ({m})', va='center', fontsize=10,
                    fontweight='bold')
        ax.set_yticks(range(nb))
        ax.set_yticklabels(b_labels, fontsize=11)
        ax.set_xlabel('BEST Aggregated Rotation Error (°)', fontsize=12)
        ax.set_title('Model Ranking by BEST Temporal Aggregation (lower is better)',
                      fontsize=14, fontweight='bold')
        ax.axvline(0.1, color='red', ls='--', lw=1.5, alpha=0.7, label='0.1° target')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis='x')
        ax.invert_yaxis()
        plt.tight_layout()
        _save(fig, 'best_aggregated_ranking.png')

    # Chart 8: BEST vs Per-frame scatter
    if best_models:
        fig, ax = plt.subplots(figsize=(10, 8))
        perframe_vals = [s.get('rot_error_mean', 0) for s in best_models]
        best_vals = [s['temporal']['best']['rot'] for s in best_models]
        scat_labels = [s['label'] for s in best_models]
        sc_colors = [colors[labels.index(s['label'])] if s['label'] in labels else '#888'
                     for s in best_models]
        ax.scatter(perframe_vals, best_vals, s=120, c=sc_colors, edgecolor='black',
                   linewidth=1, zorder=3, alpha=0.85)
        for i, lbl in enumerate(scat_labels):
            ax.annotate(lbl, (perframe_vals[i], best_vals[i]),
                        textcoords="offset points", xytext=(8, 5), fontsize=8)
        max_v = max(max(perframe_vals), max(best_vals)) * 1.15
        ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.3, label='y=x (no improvement)')
        ax.axhline(0.1, color='red', ls=':', lw=1.5, alpha=0.6, label='BEST 0.1° target')
        ax.set_xlabel('Per-frame Mean Rotation Error (°)', fontsize=12)
        ax.set_ylabel('BEST Aggregated Rotation Error (°)', fontsize=12)
        ax.set_title('Per-frame vs BEST Aggregated (lower-right = high gain)',
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, 'best_vs_perframe_scatter.png')

    # Chart 9: Best model per-sequence breakdown
    if sorted_by_best:
        top_model = sorted_by_best[0]
        per_seq_data = top_model.get('temporal', {}).get('best_per_sequence', {})
        per_seq_list = per_seq_data.get('per_sequence', []) if per_seq_data else []
        if per_seq_list:
            ns = len(per_seq_list)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, ns * 1.5), 10))
            seq_ids = [str(p['seq']) for p in per_seq_list]
            rots_ps = [p['rot'] for p in per_seq_list]
            rolls_ps = [p['roll'] for p in per_seq_list]
            pitches_ps = [p['pitch'] for p in per_seq_list]
            yaws_ps = [p['yaw'] for p in per_seq_list]
            xs = np.arange(ns)

            mean_rot_ps = np.mean(rots_ps)
            bar_cs = ['#e74c3c' if r > mean_rot_ps * 1.5 else
                       '#f39c12' if r > mean_rot_ps else '#2ecc71' for r in rots_ps]
            ax1.bar(xs, rots_ps, color=bar_cs, alpha=0.85, edgecolor='white')
            ax1.axhline(mean_rot_ps, color='blue', ls='--', lw=1.5,
                         label=f'Mean={mean_rot_ps:.4f}°')
            for i, v in enumerate(rots_ps):
                ax1.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8,
                         fontweight='bold')
            ax1.set_xticks(xs)
            ax1.set_xticklabels(seq_ids, fontsize=10)
            ax1.set_ylabel('Rotation Error (°)', fontsize=12)
            ax1.set_title(f'Best Model ({top_model["label"]}) Per-Sequence Rot Error',
                           fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.2, axis='y')

            w_bar = 0.25
            ax2.bar(xs - w_bar, rolls_ps, w_bar, label='Roll', color='#E45756', alpha=0.85)
            ax2.bar(xs, pitches_ps, w_bar, label='Pitch', color='#4C78A8', alpha=0.85)
            ax2.bar(xs + w_bar, yaws_ps, w_bar, label='Yaw', color='#72B7B2', alpha=0.85)
            ax2.set_xticks(xs)
            ax2.set_xticklabels(seq_ids, fontsize=10)
            ax2.set_ylabel('Component Error (°)', fontsize=12)
            ax2.set_title(f'Best Model ({top_model["label"]}) Per-Sequence RPY Breakdown',
                           fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.2, axis='y')

            plt.tight_layout()
            _save(fig, 'best_model_per_seq.png')

    print(f"   All charts saved to: {charts_dir}/")


def generate_projection_comparison(all_stats):
    """Collect projection images from each model for side-by-side comparison."""
    comparison_dir = os.path.join(OUTPUT_DIR, "projection_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    import cv2
    sample_indices = [0, 200, 400, 600, 800, 1000, 1200, 1400]

    for sample_idx in sample_indices:
        images = []
        model_labels = []
        for mcfg in MODELS:
            label = mcfg["label"]
            per_model_dir = os.path.join(OUTPUT_DIR, label)
            img_path = os.path.join(per_model_dir, "perframe_projections", f"sample_{sample_idx:04d}.png")
            if not os.path.isfile(img_path):
                img_path = os.path.join(per_model_dir, f"sample_{sample_idx:04d}_projection.png")
            if not os.path.isfile(img_path):
                alt_dir = os.path.join(_resolve_model_base(mcfg), "test_data_eval")
                img_path = os.path.join(alt_dir, f"sample_{sample_idx:04d}_projection.png")
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    model_labels.append(label)

        if len(images) < 2:
            continue

        max_w = max(img.shape[1] for img in images)
        padded = []
        for img, label in zip(images, model_labels):
            h, w = img.shape[:2]
            if w < max_w:
                img = np.hstack([img, np.zeros((h, max_w - w, 3), dtype=np.uint8)])

            bar_h = 40
            bar = np.full((bar_h, max_w, 3), (40, 40, 40), dtype=np.uint8)
            cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            padded.append(np.vstack([bar, img]))

        combined = np.vstack(padded)

        title_h = 50
        title_bar = np.full((title_h, max_w, 3), (60, 60, 60), dtype=np.uint8)
        cv2.putText(title_bar, f"Sample {sample_idx:04d} - Point Cloud Projection Comparison",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        combined = np.vstack([title_bar, combined])

        out_path = os.path.join(comparison_dir, f"comparison_sample_{sample_idx:04d}.png")
        cv2.imwrite(out_path, combined)
        print(f"   comparison_sample_{sample_idx:04d}.png ({len(model_labels)} models)")

    print(f"   Projection comparisons saved to: {comparison_dir}/")


def _collect_bag_eval_data(bag_eval_dir):
    """Read all model subdirs under bag_eval_dir, return structured data for report."""
    import io as _io
    model_data = {}
    for model_dir_name in sorted(os.listdir(bag_eval_dir)):
        model_path = os.path.join(bag_eval_dir, model_dir_name)
        if not os.path.isdir(model_path) or model_dir_name.startswith(("_", ".")):
            continue
        trips = {}
        for trip_name in sorted(os.listdir(model_path)):
            sidecar = os.path.join(model_path, trip_name, "result_sidecar.json")
            if not os.path.isfile(sidecar):
                continue
            try:
                with _io.open(sidecar, "r", encoding="utf-8") as f:
                    data = json.load(f)
                trips[trip_name] = data
            except Exception:
                continue
        if trips:
            model_data[model_dir_name] = trips
    return model_data


def _bag_report_lines(bag_eval_dir, model_data, section_num):
    """Generate report lines for BAG cross-model summary."""
    lines = []
    all_trips = sorted(set(t for trips in model_data.values() for t in trips))
    trip_short = {t: t.split("_")[0] for t in all_trips}

    def _best_medw(data):
        """Return the best available MEDW window dict, preferring 200 > 100 > 50."""
        mwe = data.get("multi_window_errors") or {}
        for k in ("200", "100", "50"):
            w = mwe.get(k)
            if isinstance(w, dict):
                return w
        return {}

    def _medw200(data):
        v = _best_medw(data).get("rot")
        return float(v) if v is not None else float("nan")

    def _medw200_rpy(data):
        w = _best_medw(data)
        def _s(k):
            v = w.get(k)
            return float(v) if v is not None else float("nan")
        return _s("roll"), _s("pitch"), _s("yaw")

    model_avg_medw = []
    for model, trips in model_data.items():
        medws = [_medw200(d) for d in trips.values()
                 if d.get("status") != "failed" and _medw200(d) == _medw200(d)]
        if medws:
            model_avg_medw.append((model, float(np.mean(medws)), float(np.std(medws)),
                                   float(np.min(medws)), float(np.max(medws)), len(medws)))
    model_avg_medw.sort(key=lambda x: x[1])

    _CN = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七',
           8: '八', 9: '九', 10: '十', 11: '十一', 12: '十二', 13: '十三', 14: '十四'}

    lines.append("=" * 80)
    lines.append(f"{_CN.get(section_num, str(section_num))}、BAG 泛化评估跨模型汇总 (真实行程)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"评估目录: `{bag_eval_dir}`")
    lines.append(f"模型数: {len(model_data)}  行程数: {len(all_trips)}")
    lines.append(f"行程: {', '.join(trip_short[t] for t in all_trips)}")
    lines.append("")

    lines.append("MEDW200 总排名 (跨行程均值, 越低越好):")
    lines.append("")
    lines.append("| 排名 | 模型 | Avg MEDW200 | Std | Min | Max | #Trips |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for rank, (model, avg, std, mn, mx, n) in enumerate(model_avg_medw[:20], 1):
        lines.append(f"| {rank} | {model} | {avg:.4f}° | {std:.4f} | {mn:.4f} | {mx:.4f} | {n} |")
    if len(model_avg_medw) > 20:
        lines.append(f"| ... | ({len(model_avg_medw) - 20} more) | | | | | |")
    lines.append("")

    lines.append("Per-Trip MEDW200 对比 (Top 15):")
    lines.append("")
    trip_headers = " | ".join(trip_short[t] for t in all_trips)
    lines.append(f"| 模型 | {trip_headers} | Avg |")
    lines.append(f"| --- | {' | '.join(['---:'] * len(all_trips))} | ---: |")
    for model, avg, _, _, _, _ in model_avg_medw[:15]:
        trips = model_data[model]
        cells = []
        for t in all_trips:
            if t in trips:
                m = _medw200(trips[t])
                gs = trips[t].get("gt_source", "?")
                flag = "*" if gs == "install_angle_error" else ""
                cells.append(f"{m:.4f}{flag}" if m == m else "N/A")
            else:
                cells.append("-")
        lines.append(f"| {model} | {' | '.join(cells)} | {avg:.4f} |")
    lines.append("")
    lines.append("> \\* = MEDW参考基准为init外参(非GT), 指标不可靠")
    lines.append("")

    lines.append("Per-Trip RPY 分量 (MEDW200, Top 10 per trip):")
    lines.append("")
    for t in all_trips:
        lines.append(f"**{trip_short[t]}** ({t}):")
        lines.append("")
        lines.append("| 模型 | MEDW200 | Roll | Pitch | Yaw | GT源 | Std |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
        trip_models = []
        for model, avg, _, _, _, _ in model_avg_medw:
            if t in model_data[model]:
                td = model_data[model][t]
                if td.get("status") == "failed":
                    continue
                trip_models.append((model, td))
        trip_models.sort(key=lambda x: _medw200(x[1]))
        for model, d in trip_models[:10]:
            m = _medw200(d)
            r, p, y = _medw200_rpy(d)
            gs = d.get("gt_source", "?")
            if gs == "install_angle_error":
                gs = "init(不可靠)"
            elif gs == "gt_lidars_cfg":
                gs = "GT"
            std = d.get("total_std") or 0.0
            lines.append(f"| {model} | {m:.4f}° | {r:.4f} | {p:.4f} | {y:.4f} | {gs} | {std:.4f} |")
        lines.append("")

    scenario_suffixes = [("_inject_small", "inject_small"),
                         ("_shortcut", "shortcut"),
                         ("_baseline", "baseline")]
    scenarios = {}
    for model in model_data:
        for suffix, scenario in scenario_suffixes:
            if model.endswith(suffix):
                base_model = model[:-len(suffix)]
                scenarios.setdefault(base_model, {})[scenario] = model
                break

    if scenarios:
        lines.append("场景对比 (baseline vs shortcut vs inject_small):")
        lines.append("")
        lines.append("| 模型 | Baseline | Shortcut | Inject Small | Recovery% | Risk |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for base_model in sorted(scenarios.keys()):
            sc = scenarios[base_model]
            bl_medw = inj_medw = sc_medw = float("nan")
            recovery = shortcut_risk = "N/A"
            for scenario, full_name in sc.items():
                trips = model_data[full_name]
                medws = [_medw200(d) for d in trips.values()
                         if d.get("status") != "failed" and _medw200(d) == _medw200(d)]
                avg = float(np.mean(medws)) if medws else float("nan")
                if scenario == "baseline":
                    bl_medw = avg
                elif scenario == "shortcut":
                    sc_medw = avg
                    risks = [d.get("shortcut_risk") for d in trips.values()
                             if d.get("shortcut_risk")]
                    shortcut_risk = "/".join(sorted(set(risks))) if risks else "N/A"
                elif scenario == "inject_small":
                    inj_medw = avg
                    recs = []
                    for d in trips.values():
                        cr = d.get("compensation_ratio_pct")
                        if cr is not None and isinstance(cr, (int, float)) and cr == cr:
                            recs.append(float(cr))
                    if recs:
                        recovery = f"{np.mean(recs):.1f}%"
            bl_s = f"{bl_medw:.4f}°" if bl_medw == bl_medw else "-"
            sc_s = f"{sc_medw:.4f}°" if sc_medw == sc_medw else "-"
            inj_s = f"{inj_medw:.4f}°" if inj_medw == inj_medw else "-"
            lines.append(f"| {base_model} | {bl_s} | {sc_s} | {inj_s} | {recovery} | {shortcut_risk} |")
        lines.append("")

    lines.append("跨车型一致性 (Top 10):")
    lines.append("")
    for model, avg, std, mn, mx, n in model_avg_medw[:10]:
        if n < 2:
            continue
        cv = std / avg * 100 if avg > 0 else 0
        verdict = "优秀" if cv < 10 else ("良好" if cv < 20 else "需改进")
        lines.append(f"- **{model}**: Avg={avg:.4f}° Std={std:.4f} CV={cv:.1f}% → {verdict}")
    lines.append("")

    return lines, model_avg_medw


def generate_report(all_stats):
    """Generate Feishu-compatible markdown report."""
    report_path = os.path.join(OUTPUT_DIR, "GENERALIZATION_REPORT.md")
    lines = []

    sorted_stats = sorted(all_stats, key=lambda x: x.get('rot_error_mean', 999))
    best = sorted_stats[0]
    worst = sorted_stats[-1]
    n_samples = all_stats[0].get('samples', '?')
    first_with_bounds = next((s for s in all_stats if s.get('seq_boundaries')), None)
    n_sequences = len(first_with_bounds['seq_boundaries']) if first_with_bounds else '?'
    frames_per_seq = first_with_bounds['seq_boundaries'][0]['count'] if first_with_bounds and first_with_bounds['seq_boundaries'] else '?'

    lines.append("BEVCalib 多模型泛化性能对比报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"评估日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"测试数据集: test_data ({n_samples} samples, {n_sequences} sequences, 每序列 {frames_per_seq} 帧)")
    lines.append(f"扰动范围: +/-{ANGLE_RANGE} deg, +/-{TRANS_RANGE} m")
    lines.append(f"评估模型数: {len(all_stats)}")
    lines.append("")

    # Section 1
    lines.append("=" * 80)
    lines.append("一、实验配置概况")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| 模型标签 | BEV模式 | Backbone | 描述 | Checkpoint |")
    lines.append("| --- | --- | --- | --- | --- |")
    for s in all_stats:
        c = s['config']
        desc = c.get('mode_desc', '')
        if not desc:
            parts = []
            if 'angle_deg' in c:
                parts.append(f"{c['angle_deg']}deg")
            if 'version' in c:
                parts.append(c['version'])
            desc = ', '.join(parts) if parts else s['label']
        _dir = c.get('dir_name', '').lower()
        _all_text = f"{s['label'].lower()} {desc.lower()} {_dir}"
        bev_mode = "Query-BEV" if "query" in _all_text else "LSS"
        backbone = s.get('backbone', 'DINOv2' if 'dinov2' in _all_text else 'Swin')
        if "frozen" in s['label'].lower() or "frozen" in desc.lower():
            backbone += " (frozen)"
        lines.append(f"| {s['label']} | {bev_mode} | {backbone} | {desc} | {c.get('ckpt', 'best_val')} |")
    lines.append("")

    # Section 2
    lines.append("=" * 80)
    lines.append("二、旋转误差总览 (Total Rotation Error, deg)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| 模型 | Mean | Std | Median | P90 | P95 | P99 | Max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in sorted_stats:
        lines.append(
            f"| {s['label']} "
            f"| {s.get('rot_error_mean', -1):.3f} "
            f"| {s.get('rot_error_std', -1):.3f} "
            f"| {s.get('rot_error_median', -1):.3f} "
            f"| {s.get('rot_error_p90', -1):.3f} "
            f"| {s.get('rot_error_p95', -1):.3f} "
            f"| {s.get('rot_error_p99', -1):.3f} "
            f"| {s.get('rot_error_max', -1):.3f} |"
        )
    lines.append("")
    lines.append("![Rotation Error Bar Chart](charts/rotation_error_bar.png)")
    lines.append("")

    # Macro vs Micro comparison (if macro data available)
    has_macro = any(s.get('macro_rot_mean') for s in all_stats)
    if has_macro:
        lines.append("Micro vs Macro 对比 (均衡评估):")
        lines.append("")
        lines.append("| 模型 | Micro(样本级) | Macro(序列级) | 差异 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for s in sorted_stats:
            micro = s.get('rot_error_mean', -1)
            macro = s.get('macro_rot_mean', -1)
            diff = macro - micro if macro > 0 and micro > 0 else 0
            sign = "+" if diff >= 0 else ""
            lines.append(f"| {s['label']} | {micro:.3f} | {macro:.3f} | {sign}{diff:.3f} |")
        lines.append("")

    # Section 3
    lines.append("=" * 80)
    lines.append("三、旋转分量分析 (Roll / Pitch / Yaw Mean, deg)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| 模型 | Roll(LiDAR-X) | Pitch(LiDAR-Y) | Yaw(LiDAR-Z) | Total |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for s in sorted_stats:
        lines.append(
            f"| {s['label']} "
            f"| {s.get('roll_error_mean', -1):.3f} "
            f"| {s.get('pitch_error_mean', -1):.3f} "
            f"| {s.get('yaw_error_mean', -1):.3f} "
            f"| {s.get('rot_error_mean', -1):.3f} |"
        )
    lines.append("")
    lines.append("![Rotation Components](charts/rotation_components_bar.png)")
    lines.append("")

    # Section: Temporal Aggregation + Bias Correction
    has_temporal = any(s.get('temporal') for s in all_stats)
    if has_temporal:
        lines.append("=" * 80)
        lines.append("四、时序聚合与偏差矫正 (多帧推理)")
        lines.append("=" * 80)
        lines.append("")
        lines.append("通过多帧时序聚合（SVD-Mean / Robust Median）和偏差矫正，"
                      "大幅降低单帧随机误差，模拟在线标定场景：")
        lines.append("")

        key_windows = [1, 50, 200, 400]
        for method, method_name in [('svd', 'SVD-Mean'), ('med', 'Robust Median')]:
            lines.append(f"{method_name} 聚合:")
            lines.append("")
            header = "| 模型 |" + " | ".join(
                f"{'Per-frame' if w == 1 else f'{w}-frame'}" for w in key_windows
            ) + " |"
            lines.append(header)
            lines.append("| --- |" + " | ".join("---:" for _ in key_windows) + " |")
            for s in sorted_stats:
                ta = s.get('temporal', {}).get(method, {})
                row = f"| {s['label']}"
                for w in key_windows:
                    if w in ta:
                        e = ta[w]
                        row += f" | {e['rot']:.3f}° (R:{e['roll']:.2f} P:{e['pitch']:.2f} Y:{e['yaw']:.2f})"
                    else:
                        row += " | -"
                row += " |"
                lines.append(row)
            lines.append("")

        lines.append("BEST (纯时序聚合, 可部署, 不依赖GT, 按BEST Rot排序):")
        lines.append("")
        lines.append("| 排名 | 模型 | 方法 | Rot | Roll | Pitch | Yaw | Per-frame Rot | 改善 |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        best_sorted = sorted(
            [s for s in sorted_stats if s.get('temporal', {}).get('best', {}).get('rot') is not None],
            key=lambda s: s['temporal']['best']['rot'])
        for rank, s in enumerate(best_sorted, 1):
            best_ta = s['temporal']['best']
            pf = s.get('rot_error_mean', 999)
            improve = (1 - best_ta['rot'] / pf) * 100 if pf > 0 else 0
            lines.append(
                f"| {rank} | {s['label']} | {best_ta.get('method', '?')} "
                f"| {best_ta['rot']:.3f}° "
                f"| {best_ta['roll']:.3f}° "
                f"| {best_ta['pitch']:.3f}° "
                f"| {best_ta['yaw']:.3f}° "
                f"| {pf:.3f}° "
                f"| {improve:.1f}% |"
            )
        no_temporal = [s for s in sorted_stats if s.get('temporal', {}).get('best', {}).get('rot') is None]
        for s in no_temporal:
            lines.append(f"| - | {s['label']} | - | - | - | - | - | {s.get('rot_error_mean', 0):.3f}° | - |")
        lines.append("")

        # Deployment Simulation table (uniform sampling)
        has_deploy = any(s.get('temporal', {}).get('deploy_sim') for s in all_stats)
        if has_deploy:
            lines.append("部署模拟 (Uniform Sampling, Median聚合):")
            lines.append("")
            lines.append("从每个sequence均匀采样N帧, 聚合后与GT比较, 模拟真实部署场景:")
            lines.append("")
            deploy_ws = [50, 100, 200, 400, 800]
            dh = "| 模型 |" + " | ".join(f"MEDW{w}" for w in deploy_ws) + " |"
            lines.append(dh)
            lines.append("| --- |" + " | ".join("---:" for _ in deploy_ws) + " |")
            for s in best_sorted:
                ds = s.get('temporal', {}).get('deploy_sim', {})
                row = f"| {s['label']}"
                for w in deploy_ws:
                    if w in ds:
                        e = ds[w]
                        row += f" | {e['rot']:.3f}° (R:{e['roll']:.2f} P:{e['pitch']:.2f} Y:{e['yaw']:.2f})"
                    else:
                        row += " | -"
                row += " |"
                lines.append(row)
            lines.append("")

        lines.append("聚合方法说明:")
        lines.append("- SVDW{N}: 对 N 帧预测旋转矩阵取 SVD-Mean (Euclidean 均值投影到 SO(3))")
        lines.append("- MEDW{N}: 对 N 帧预测旋转矩阵转 axis-angle 后取逐轴 Median, 再映射回 SO(3)")
        lines.append("- TRMW{N}: 对 N 帧预测旋转矩阵取 Trimmed-Mean (去掉 10% 极端值后取均值)")
        lines.append("- BEST: 遍历所有 (方法, 窗口) 组合, 选 Rot 最低者作为该模型的最优可部署方案")
        lines.append("- 部署模拟: 从全序列均匀采样N帧, 全部聚合为一个预测外参, 与GT比较. 比滑动窗口更真实")
        lines.append("- 计算流程: 每个 sequence 均匀采样N帧 → 独立聚合得到一个预测外参 → 与 GT 比较得 RPY 误差 → 所有 seq 取均值")
        lines.append("")

        has_per_seq_best = any(
            s.get('temporal', {}).get('best_per_sequence') for s in best_sorted)
        if has_per_seq_best:
            all_seq_ids_ta = set()
            for s in best_sorted:
                bps = s.get('temporal', {}).get('best_per_sequence', {})
                if bps and 'per_sequence' in bps:
                    for r in bps['per_sequence']:
                        all_seq_ids_ta.add(int(r['seq']))
            all_seq_ids_ta = sorted(all_seq_ids_ta)
            if all_seq_ids_ta:
                lines.append("BEST Per-Sequence Rot 总误差:")
                lines.append("")
                header = "| 模型 | 方法 |" + " | ".join(f"Seq{s:02d}" for s in all_seq_ids_ta) + " | Mean | Std |"
                sep = "| --- | --- |" + " | ".join("---:" for _ in all_seq_ids_ta) + " | ---: | ---: |"
                lines.append(header)
                lines.append(sep)
                import statistics as _st
                for s in best_sorted:
                    bps = s.get('temporal', {}).get('best_per_sequence', {})
                    method = s['temporal']['best'].get('method', '?')
                    if bps and 'per_sequence' in bps:
                        seq_map = {int(r['seq']): r for r in bps['per_sequence']}
                        vals = [seq_map.get(sid, {}).get('rot', -1) for sid in all_seq_ids_ta]
                        valid = [v for v in vals if v >= 0]
                        mean_v = sum(valid) / len(valid) if valid else 0
                        std_v = _st.stdev(valid) if len(valid) > 1 else 0
                        row = f"| {s['label']} | {method} |"
                        for sid in all_seq_ids_ta:
                            v = seq_map.get(sid, {}).get('rot', -1)
                            row += f" {v:.3f}° |" if v >= 0 else " - |"
                        row += f" {mean_v:.3f}° | {std_v:.3f}° |"
                        lines.append(row)
                    else:
                        row = f"| {s['label']} | {method} |"
                        row += " - |" * len(all_seq_ids_ta)
                        row += f" {s['temporal']['best']['rot']:.3f}° | - |"
                        lines.append(row)
                lines.append("")

                for axis, axis_key in [("Roll", "roll"), ("Pitch", "pitch"), ("Yaw", "yaw")]:
                    lines.append(f"BEST Per-Sequence {axis} 误差:")
                    lines.append("")
                    lines.append(header)
                    lines.append(sep)
                    for s in best_sorted:
                        bps = s.get('temporal', {}).get('best_per_sequence', {})
                        method = s['temporal']['best'].get('method', '?')
                        if bps and 'per_sequence' in bps:
                            seq_map = {int(r['seq']): r for r in bps['per_sequence']}
                            vals = [seq_map.get(sid, {}).get(axis_key, -1) for sid in all_seq_ids_ta]
                            valid = [v for v in vals if v >= 0]
                            mean_v = sum(valid) / len(valid) if valid else 0
                            std_v = _st.stdev(valid) if len(valid) > 1 else 0
                            row = f"| {s['label']} | {method} |"
                            for sid in all_seq_ids_ta:
                                v = seq_map.get(sid, {}).get(axis_key, -1)
                                row += f" {v:.3f}° |" if v >= 0 else " - |"
                            row += f" {mean_v:.3f}° | {std_v:.3f}° |"
                            lines.append(row)
                        else:
                            row = f"| {s['label']} | {method} |"
                            row += " - |" * len(all_seq_ids_ta)
                            row += f" {s['temporal']['best'].get(axis_key, -1):.3f}° | - |"
                            lines.append(row)
                    lines.append("")

                lines.append("BEST 方法选择详情 (各聚合方法在 400 帧窗口下的对比):")
                lines.append("")
                lines.append("| 模型 | SVDW400 Rot | MEDW400 Rot | TRMW400 Rot | 选中方法 | 选中Rot |")
                lines.append("| --- | ---: | ---: | ---: | --- | ---: |")
                for s in best_sorted:
                    ta = s.get('temporal', {})
                    svd400 = ta.get('svd', {}).get(400, {}).get('rot', -1)
                    med400 = ta.get('med', {}).get(400, {}).get('rot', -1)
                    trm400 = ta.get('trm', {}).get(400, {}).get('rot', -1)
                    best_m = ta.get('best', {}).get('method', '?')
                    best_r = ta.get('best', {}).get('rot', -1)
                    svd_s = f"{svd400:.3f}°" if svd400 >= 0 else "-"
                    med_s = f"{med400:.3f}°" if med400 >= 0 else "-"
                    trm_s = f"{trm400:.3f}°" if trm400 >= 0 else "-"
                    lines.append(
                        f"| {s['label']} | {svd_s} | {med_s} | {trm_s} "
                        f"| {best_m} | {best_r:.3f}° |"
                    )
                lines.append("")
                lines.append("注: SVD-Mean 和 Robust Median 并非冗余 — 不同模型的误差分布特性不同, "
                             "某些模型 SVD 更优 (如 v27-E10), 某些 MED 更优。"
                             "BEST 机制自动选择最优组合, 无需人工指定。"
                             "报告展示 SVD/MED 聚合趋势有助于理解误差随帧数的收敛行为。")
                lines.append("")

        _chart_dir = os.path.join(OUTPUT_DIR, "charts")
        for _cn, _cf in [("Temporal Convergence", "temporal_convergence_all.png"),
                          ("BEST Aggregated Ranking", "best_aggregated_ranking.png"),
                          ("BEST vs Per-frame", "best_vs_perframe_scatter.png"),
                          ("Best Model Per-Sequence", "best_model_per_seq.png")]:
            if os.path.isfile(os.path.join(_chart_dir, _cf)):
                lines.append(f"![{_cn}](charts/{_cf})")
                lines.append("")

    _sec = 5 if has_temporal else 4

    trans_stats = [s for s in all_stats if s.get('trans_error_mean') and s['trans_error_mean'] > 0]
    _CN = {5: '五', 6: '六', 7: '七', 8: '八', 9: '九', 10: '十', 11: '十一', 12: '十二'}
    if trans_stats:
        lines.append("=" * 80)
        lines.append(f"{_CN.get(_sec, str(_sec))}、平移误差对比 (仅 rotation+translation 模型, m)")
        _sec += 1
        lines.append("=" * 80)
        lines.append("")
        lines.append("| 模型 | Trans Mean | Fwd(X) | Lat(Y) | Ht(Z) | Trans P95 | Trans Max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in sorted(trans_stats, key=lambda x: x.get('trans_error_mean', 999)):
            lines.append(
                f"| {s['label']} "
                f"| {s.get('trans_error_mean', -1):.4f} "
                f"| {s.get('fwd_error_mean', -1):.4f} "
                f"| {s.get('lat_error_mean', -1):.4f} "
                f"| {s.get('ht_error_mean', -1):.4f} "
                f"| {s.get('trans_error_p95', -1):.4f} "
                f"| {s.get('trans_error_max', -1):.4f} |"
            )
        lines.append("")
        lines.append("![Translation Components](charts/translation_components.png)")
        lines.append("")

    # Section 5: Train vs Test
    has_train = [s for s in all_stats if s.get('train_metrics') and s['train_metrics'].get('rot_error', 0) > 0]
    if has_train:
        lines.append("=" * 80)
        lines.append(f"{_CN.get(_sec, str(_sec))}、训练精度 vs 泛化精度对比")
        _sec += 1
        lines.append("=" * 80)
        lines.append("")
        lines.append("| 模型 | 训练Rot(deg) | 测试Rot(deg) | 泛化衰退 | 评级 |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for s in sorted(has_train, key=lambda x: x.get('rot_error_mean', 999)):
            train_rot = s['train_metrics']['rot_error']
            test_rot = s['rot_error_mean']
            deg = test_rot / train_rot if train_rot > 0 else 0
            if deg < 1.2:
                rating = "优秀"
            elif deg < 1.5:
                rating = "良好"
            elif deg < 2.0:
                rating = "一般"
            elif deg < 3.0:
                rating = "较差"
            else:
                rating = "需改进"
            lines.append(f"| {s['label']} | {train_rot:.2f} | {test_rot:.3f} | {deg:.1f}x | {rating} |")
        lines.append("")
        lines.append("![Train vs Test](charts/train_vs_test.png)")
        lines.append("")
        under_1x = [s for s in has_train
                     if s['train_metrics']['rot_error'] > 0
                     and s['rot_error_mean'] / s['train_metrics']['rot_error'] < 1.0]
        if under_1x:
            lines.append("注: 泛化衰退 < 1.0x (测试优于训练) 的模型通常使用了强数据增强 "
                         "(mount jitter, DANN 等), 导致训练 loss 偏高, "
                         "但在实际泛化测试中反而表现更好。")
            lines.append("")

    # Section 6: Comprehensive ranking (per-frame + BEST)
    lines.append("=" * 80)
    lines.append(f"{_CN.get(_sec, str(_sec))}、模型综合排名")
    _sec += 1
    lines.append("=" * 80)
    lines.append("")
    lines.append("Per-frame 排名 (单帧推理精度):")
    lines.append("")
    lines.append("| 排名 | 模型 | Mean Rot | Median | P95 | Max |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for rank, s in enumerate(sorted_stats, 1):
        lines.append(
            f"| {rank} | {s['label']} "
            f"| {s.get('rot_error_mean', -1):.3f}° "
            f"| {s.get('rot_error_median', -1):.3f}° "
            f"| {s.get('rot_error_p95', -1):.3f}° "
            f"| {s.get('rot_error_max', -1):.3f}° |"
        )
    lines.append("")
    temporal_models = [s for s in all_stats
                       if s.get('temporal', {}).get('best', {}).get('rot') is not None]
    if temporal_models:
        sorted_by_best_rank = sorted(temporal_models,
                                      key=lambda s: s['temporal']['best']['rot'])
        lines.append("BEST 时序聚合排名 (可部署, 400帧聚合):")
        lines.append("")
        lines.append("| 排名 | 模型 | BEST Rot | 方法 | Per-frame Rot | 改善率 |")
        lines.append("| ---: | --- | ---: | --- | ---: | ---: |")
        for rank, s in enumerate(sorted_by_best_rank, 1):
            tb = s['temporal']['best']
            pf = s.get('rot_error_mean', 999)
            improve = (1 - tb['rot'] / pf) * 100 if pf > 0 else 0
            lines.append(
                f"| {rank} | {s['label']} "
                f"| {tb['rot']:.3f}° "
                f"| {tb.get('method', '?')} "
                f"| {pf:.3f}° "
                f"| {improve:.1f}% |"
            )
        lines.append("")
    lines.append("![Model Ranking](charts/model_ranking.png)")
    lines.append("")

    # Section 7: Per-sequence analysis
    has_seq_data = any(s.get('per_sequence') for s in all_stats)
    if has_seq_data:
        lines.append("=" * 80)
        lines.append(f"{_CN.get(_sec, str(_sec))}、Per-Sequence 误差分析")
        _sec += 1
        lines.append("=" * 80)
        lines.append("")
        lines.append("各模型在不同 sequence 上的 Mean Rotation Error (deg):")
        lines.append("")
        all_seq_ids = sorted(set(
            sid for s in all_stats for sid in s.get('per_sequence', {}).keys()
        ))
        if all_seq_ids:
            header = "| 模型 |" + " | ".join(f"Seq {sid}" for sid in all_seq_ids) + " |"
            sep = "| --- |" + " | ".join("---:" for _ in all_seq_ids) + " |"
            lines.append(header)
            lines.append(sep)
            for s in sorted_stats:
                ps = s.get('per_sequence', {})
                row = f"| {s['label']}"
                for sid in all_seq_ids:
                    if sid in ps:
                        row += f" | {ps[sid]['rot_mean']:.3f}"
                    else:
                        row += " | -"
                row += " |"
                lines.append(row)
            lines.append("")

            # Flag problematic sequences (>1.5x overall mean)
            anomalies = []
            for s in sorted_stats:
                ps = s.get('per_sequence', {})
                overall_mean = s.get('rot_error_mean', 0)
                for sid, sv in ps.items():
                    ratio = sv['rot_mean'] / max(overall_mean, 0.001)
                    if ratio > 1.5 and sv['samples'] >= 50:
                        anomalies.append((s['label'], sid, sv['rot_mean'], overall_mean, ratio))
            if anomalies:
                lines.append("异常 sequence 识别 (Mean Rot > 1.5x 整体 Mean):")
                lines.append("")
                lines.append("| 模型 | Sequence | Seq Mean | 整体 Mean | 比值 |")
                lines.append("| --- | --- | ---: | ---: | ---: |")
                for label, sid, seq_mean, overall, ratio in sorted(anomalies, key=lambda x: -x[4]):
                    lines.append(f"| {label} | Seq {sid} | {seq_mean:.3f}° | {overall:.3f}° | {ratio:.1f}x |")
            else:
                lines.append("异常 sequence 识别: 无显著异常 (所有序列 Mean Rot 均在 1.5x 整体 Mean 以内)")
            lines.append("")
            # Per-sequence stability analysis
            lines.append("跨序列稳定性分析 (序列间误差标准差):")
            lines.append("")
            lines.append("| 排名 | 模型 | 序列间Std | CV(变异系数) | Min Seq | Max Seq | Range |")
            lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
            stability = []
            for s in sorted_stats:
                ps = s.get('per_sequence', {})
                if ps:
                    vals = [v['rot_mean'] for v in ps.values()]
                    import statistics
                    mean_v = statistics.mean(vals) if vals else 0
                    std_v = statistics.stdev(vals) if len(vals) > 1 else 0
                    cv = std_v / mean_v if mean_v > 0 else 0
                    stability.append((s['label'], std_v, cv, min(vals), max(vals), max(vals) - min(vals)))
            for rank, (label, std_v, cv, mn, mx, rng) in enumerate(
                    sorted(stability, key=lambda x: x[1]), 1):
                lines.append(f"| {rank} | {label} | {std_v:.3f}° | {cv:.3f} | {mn:.3f}° | {mx:.3f}° | {rng:.3f}° |")
            lines.append("")

        # Per-sequence boundary info
        first_with_bounds = next((s for s in all_stats if s.get('seq_boundaries')), None)
        if first_with_bounds:
            lines.append("Sequence 到 Sample Index 映射:")
            lines.append("")
            lines.append("| Sequence | Sample范围 | 帧数 |")
            lines.append("| --- | --- | ---: |")
            for sb in first_with_bounds['seq_boundaries']:
                lines.append(f"| Seq {sb['seq']} | {sb['start']} - {sb['end']} | {sb['count']} |")
            lines.append("")

    # Section 8: Projection comparison
    lines.append("=" * 80)
    lines.append(f"{_CN.get(_sec, str(_sec))}、点云投影效果图对比")
    _sec += 1
    lines.append("=" * 80)
    lines.append("")
    proj_dir = os.path.join(OUTPUT_DIR, "projection_comparison")
    if os.path.isdir(proj_dir):
        for fn in sorted(os.listdir(proj_dir)):
            if fn.endswith('.png'):
                sample_num = fn.replace('comparison_sample_', '').replace('.png', '')
                lines.append(f"Sample {sample_num}:")
                lines.append(f"![{fn}](projection_comparison/{fn})")
                lines.append("")

    # Temporal aggregation projections (2x2 grid: GT|Init / Per-frame|Aggregated)
    # Show key models: top 3 by BEST + per-frame best + per-frame worst for contrast
    stats_with_temporal = [s for s in all_stats
                           if s.get('temporal', {}).get('best', {}).get('rot') is not None]
    key_labels = set()
    if stats_with_temporal:
        sorted_by_best_local = sorted(stats_with_temporal,
                                       key=lambda s: s['temporal']['best']['rot'])
        for s in sorted_by_best_local[:3]:
            key_labels.add(s['label'])
        if sorted_stats:
            key_labels.add(sorted_stats[0]['label'])
        if len(sorted_by_best_local) > 3:
            key_labels.add(sorted_by_best_local[-1]['label'])

    has_temporal_proj = False
    for s in all_stats:
        label = s['label']
        tp_dir = os.path.join(OUTPUT_DIR, label, "temporal_projections")
        if os.path.isdir(tp_dir) and os.listdir(tp_dir):
            if not has_temporal_proj:
                lines.append("=" * 80)
                lines.append(f"{_CN.get(_sec, str(_sec))}、时序聚合投影效果对比 (2x2: GT | Init / Per-frame | Aggregated)")
                _sec += 1
                lines.append("=" * 80)
                lines.append("")
                lines.append("田字格布局：左上=GT (真值) | 右上=Init (扰动输入) | 左下=Per-frame (单帧预测) | 右下=Aggregated (400帧聚合)")
                lines.append("")
                if key_labels:
                    lines.append(f"展示关键模型 ({len(key_labels)}个): "
                                 + ", ".join(sorted(key_labels))
                                 + " (完整投影图见各模型子目录)")
                    lines.append("")
                has_temporal_proj = True
            is_key = label in key_labels
            if not is_key:
                continue
            best_rot = s.get('temporal', {}).get('best', {}).get('rot', 999)
            pf_rot = s.get('rot_error_mean', 999)
            lines.append(f"{label} (Per-frame: {pf_rot:.3f}°, BEST: {best_rot:.3f}°):")
            lines.append("")
            tp_files = sorted(fn for fn in os.listdir(tp_dir) if fn.endswith('.png'))
            shown = 0
            for fn in tp_files:
                sample_num = fn.replace('temporal_compare_', '').replace('.png', '')
                seq_id = int(sample_num) // 400 if sample_num.isdigit() else '?'
                frame_in_seq = int(sample_num) % 400 if sample_num.isdigit() else 0
                if frame_in_seq == 0:
                    lines.append(f"Sample {sample_num} (Seq {seq_id:02d}):")
                    lines.append(f"![{fn}]({label}/temporal_projections/{fn})")
                    lines.append("")
                    shown += 1
            if shown == 0:
                for fn in tp_files[:4]:
                    sample_num = fn.replace('temporal_compare_', '').replace('.png', '')
                    seq_id = int(sample_num) // 400 if sample_num.isdigit() else '?'
                    lines.append(f"Sample {sample_num} (Seq {seq_id:02d}):")
                    lines.append(f"![{fn}]({label}/temporal_projections/{fn})")
                    lines.append("")

    # === Generalization Diagnostics Section ===
    has_gdiag = any(s.get('gdiag') for s in all_stats)
    if has_gdiag:
        lines.append("=" * 80)
        lines.append(f"{_CN.get(_sec, str(_sec))}、泛化诊断 (MEDW聚合: Zero-Drift / Inject / Shortcut / GS_medw)")
        _sec += 1
        lines.append("=" * 80)
        lines.append("")
        lines.append("通过五项测试综合评估模型泛化能力:")
        lines.append("- **Zero-Drift**: 无扰动输入, 度量模型固有偏差 (越低越好)")
        lines.append("- **Fixed-Inject**: 注入已知固定扰动(2°RPY), 度量矫正恢复能力 (Recovery%越高越好)")
        lines.append("- **Per-Axis Shortcut**: 单轴注入Roll/Pitch/Yaw, 检测模型是否独立校准各轴")
        lines.append("- **Asymmetry**: 正/负方向注入对比, 检测方向偏置")
        lines.append("- **Linearity**: 多幅度注入(0.5°/1.0°/2.0°)残差一致性, 检测矫正稳定性")
        lines.append("- **GS_medw**: 基于MEDW多帧聚合的综合泛化得分 (4项子指标, 越低越好)")
        lines.append("  > 注: 扰动在LiDAR坐标系RPY轴注入 (右乘), 确保轴定义与误差分解一致")
        lines.append("")

        gdiag_models = [s for s in all_stats
                        if s.get('gdiag') and s['gdiag'].get('composite', {}).get('valid', True) is not False]
        gdiag_sorted = sorted(gdiag_models,
                               key=lambda s: s['gdiag'].get('composite', {}).get('GS_medw', 999))

        lines.append("GS_medw 泛化诊断排名 (MEDW多帧聚合, lower=better):")
        lines.append("")
        lines.append("| 排名 | 模型 | GS_medw | S1:ZeroDrift | S2:Correction | S3:Shortcut | S4:Consistency | Risk | #Seqs |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
        for rank, s in enumerate(gdiag_sorted, 1):
            gd = s['gdiag']
            comp = gd.get('composite', {})
            sub = comp.get('sub_scores', {})
            sc_risk = gd.get('shortcut_risk', 'N/A')
            n_seqs = comp.get('raw', {}).get('n_seqs_evaluated', '-')
            lines.append(
                f"| {rank} | {s['label']} "
                f"| {comp.get('GS_medw', -1):.4f} "
                f"| {sub.get('zero_drift', -1):.4f} "
                f"| {sub.get('correction', -1):.4f} "
                f"| {sub.get('shortcut', -1):.4f} "
                f"| {sub.get('consistency', -1):.4f} "
                f"| {sc_risk} "
                f"| {n_seqs} |"
            )
        lines.append("")

        lines.append("Zero-Drift 详细 (init=GT, 无扰动, MEDW聚合后RPY分量):")
        lines.append("")
        lines.append("| 模型 | Rot Mean | Roll | Pitch | Yaw | max(R,P,Y) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for s in gdiag_sorted:
            zd = s['gdiag'].get('zero_drift', {})
            lines.append(
                f"| {s['label']} "
                f"| {zd.get('rot_mean', -1):.4f}° "
                f"| {zd.get('roll_mean', -1):.4f}° "
                f"| {zd.get('pitch_mean', -1):.4f}° "
                f"| {zd.get('yaw_mean', -1):.4f}° "
                f"| {zd.get('max_rpy', -1):.4f}° |"
            )
        lines.append("")

        lines.append("V70.2 OOD 联合验收 (唯一 checkpoint/部署结论门控):")
        lines.append("")
        lines.append("| 模型 | Verdict | Genuine Mean | Genuine Worst Rig | Signed Slope | Worst Rig/Axis | ZD max |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for s in gdiag_sorted:
            gate = s['gdiag'].get('acceptance_gate', {})
            if not gate:
                lines.append(f"| {s['label']} | N/A | - | - | - | - | - |")
                continue
            lines.append(
                f"| {s['label']} "
                f"| {'PASS' if gate.get('passed') else 'FAIL'} "
                f"| {gate.get('genuine_recovery_pct', -1):.1f}% "
                f"| {gate.get('genuine_recovery_min_rig_pct', -1):.1f}% "
                f"| {gate.get('signed_correction_slope', -1):.3f} "
                f"| {gate.get('signed_correction_slope_min_axis', -1):.3f} "
                f"| {gate.get('zd_max_deg', -1):.4f}° |")
        lines.append("")

        lines.append("Fixed-Inject 恢复能力 (辅助诊断，不替代上述联合门控):")
        lines.append("")
        lines.append("| 模型 | Injected | Raw Residual | ZD(rot) | ZD方向扣除 | Genuine Resid | Raw Rec% | Genuine Rec% |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in gdiag_sorted:
            fi = s['gdiag'].get('fixed_inject', {}).get('inject', {})
            comp_raw = s['gdiag'].get('composite', {}).get('raw', {})
            zd_val = s['gdiag'].get('zero_drift', {}).get('rot_mean', 0)
            raw_res = fi.get('mean_residual', -1)
            injected = fi.get('mean_injected', -1)
            raw_rec = fi.get('mean_recovery_pct', -1)
            zd_signed = comp_raw.get('zero_drift_signed_rpy', None)
            has_signed = zd_signed is not None and len(zd_signed) == 3
            if raw_res >= 0 and injected > 0:
                if has_signed:
                    n_ax = len(zd_signed)
                    inj_hat = [1.0 / np.sqrt(n_ax)] * n_ax
                    zd_proj = sum(z * h for z, h in zip(zd_signed, inj_hat))
                    zd_deduct = max(0, min(zd_proj, raw_res))
                    genuine_res = max(0, raw_res - zd_deduct)
                    genuine_rec = (injected - genuine_res) / injected * 100
                else:
                    zd_deduct_scalar = min(zd_val, raw_res)
                    genuine_res = max(0, raw_res - zd_deduct_scalar)
                    genuine_rec = (injected - genuine_res) / injected * 100
                    zd_deduct = zd_deduct_scalar
            else:
                zd_deduct = 0
                genuine_res = -1
                genuine_rec = -1
            lines.append(
                f"| {s['label']} "
                f"| {injected:.3f}° "
                f"| {raw_res:.4f}° "
                f"| {zd_val:.4f}° "
                f"| {zd_deduct:.4f}° "
                f"| {genuine_res:.4f}° "
                f"| {raw_rec:.1f}% "
                f"| {genuine_rec:.1f}% |"
            )
        lines.append("")

        lines.append("Per-Axis Shortcut 检测 (单轴注入, Raw=原始Recovery, Genuine=方向感知扣除ZeroDrift):")
        lines.append("")
        lines.append("| 模型 | R-Raw% | P-Raw% | Y-Raw% | R-Genuine% | P-Genuine% | Y-Genuine% | Risk | R/P/Y |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for s in gdiag_sorted:
            gd = s['gdiag']
            spa = gd.get('shortcut_per_axis', {})
            _cr = gd.get('composite', {}).get('raw', {})
            pag = _cr.get('per_axis_genuine_recovery', {})
            _has_signed = _cr.get('zero_drift_signed_rpy') is not None
            zd = gd.get('zero_drift', {})
            inject_d = _cr.get('inject_deg', GDIAG_INJECT_DEG)
            axis_data = {}
            for ai, ax in enumerate(['roll', 'pitch', 'yaw']):
                ax_sc = spa.get(ax, {})
                raw_rec = ax_sc.get('axis_specific_recovery', -1)
                ax_residual = ax_sc.get('axis_residual', inject_d)
                if ax_residual < 0:
                    axis_data[ax] = {'raw': -1, 'genuine': -1}
                    continue
                if _has_signed and ax in pag:
                    genuine_rec = pag[ax].get('genuine_recovery_pct', raw_rec)
                else:
                    ax_zd = zd.get(f'{ax}_mean', 0)
                    ax_genuine_residual = max(0, ax_residual - min(ax_zd, ax_residual))
                    genuine_rec = ((inject_d - ax_genuine_residual) / inject_d * 100) if inject_d > 0 else -1
                axis_data[ax] = {'raw': raw_rec, 'genuine': genuine_rec}
            sc_risk = gd.get('shortcut_risk', 'N/A')
            risk_d = gd.get('shortcut_risk_detail', {})
            def _pct_or_na(v):
                return f"{v:.1f}%" if v >= 0 else "N/A"
            lines.append(
                f"| {s['label']} "
                f"| {_pct_or_na(axis_data['roll']['raw'])} "
                f"| {_pct_or_na(axis_data['pitch']['raw'])} "
                f"| {_pct_or_na(axis_data['yaw']['raw'])} "
                f"| {_pct_or_na(axis_data['roll']['genuine'])} "
                f"| {_pct_or_na(axis_data['pitch']['genuine'])} "
                f"| {_pct_or_na(axis_data['yaw']['genuine'])} "
                f"| {sc_risk} "
                f"| {risk_d.get('roll', '?')}/{risk_d.get('pitch', '?')}/{risk_d.get('yaw', '?')} |"
            )
        lines.append("")

        lines.append("Cross-Axis Leakage (单轴注入时其他轴误差增量, 基线=Zero-Drift):")
        lines.append("")
        lines.append("| 模型 | Roll注入→其他轴 | Pitch注入→其他轴 | Yaw注入→其他轴 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for s in gdiag_sorted:
            cl = s['gdiag'].get('cross_axis_leakage', {})
            r_leak = cl.get('roll', {}).get('mean_leakage_deg', -1)
            p_leak = cl.get('pitch', {}).get('mean_leakage_deg', -1)
            y_leak = cl.get('yaw', {}).get('mean_leakage_deg', -1)
            lines.append(
                f"| {s['label']} "
                f"| {r_leak:.4f}° "
                f"| {p_leak:.4f}° "
                f"| {y_leak:.4f}° |"
            )
        lines.append("")

        lines.append("Fixed-Inject Per-Axis Residual (RPY分量残差):")
        lines.append("")
        lines.append("| 模型 | Roll残差 | Pitch残差 | Yaw残差 | 最大轴 |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for s in gdiag_sorted:
            fi = s['gdiag'].get('fixed_inject', {}).get('inject', {})
            r_res = fi.get('roll_residual', -1)
            p_res = fi.get('pitch_residual', -1)
            y_res = fi.get('yaw_residual', -1)
            worst = max([(r_res, 'R'), (p_res, 'P'), (y_res, 'Y')], key=lambda x: x[0])
            lines.append(
                f"| {s['label']} "
                f"| {r_res:.4f}° "
                f"| {p_res:.4f}° "
                f"| {y_res:.4f}° "
                f"| {worst[1]} |"
            )
        lines.append("")

        lines.append("Multi-Magnitude 矫正线性度 (0.5°→1.0°→2.0°):")
        lines.append("")
        lines.append("| 模型 | 0.5° Resid | 0.5° Recv% | 1.0° Resid | 1.0° Recv% | 2.0° Resid | 2.0° Recv% |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in gdiag_sorted:
            mm = s['gdiag'].get('multi_magnitude', {})
            cols = []
            for mag in ['0.5', '1.0', '2.0']:
                entry = mm.get(mag, {})
                cols.append(f"{entry.get('residual', -1):.4f}°")
                cols.append(f"{entry.get('recovery_pct', -1):.1f}%")
            lines.append(f"| {s['label']} | " + " | ".join(cols) + " |")
        lines.append("")

        lines.append("正/负注入对称性:")
        lines.append("")
        lines.append("| 模型 | +inject Resid | -inject Resid | |Δ| | 对称性 |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for s in gdiag_sorted:
            gd = s['gdiag']
            pos = gd.get('fixed_inject', {}).get('inject', {}).get('mean_residual', -1)
            neg = gd.get('neg_inject', {}).get('inject', {}).get('mean_residual', -1)
            delta = abs(pos - neg) if pos >= 0 and neg >= 0 else -1
            sym = "优" if delta < 0.1 else ("良" if delta < 0.3 else "差")
            lines.append(
                f"| {s['label']} "
                f"| {pos:.4f}° "
                f"| {neg:.4f}° "
                f"| {delta:.4f}° "
                f"| {sym} |"
            )
        lines.append("")

        lines.append("Prediction Independence (历史辅助诊断，不参与 V70.2 门控):")
        lines.append("")
        lines.append("| 模型 | Roll独立性 | Pitch独立性 | Yaw独立性 | 综合独立性 | 判定 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for s in gdiag_sorted:
            pi = s['gdiag'].get('prediction_independence', {})
            if not pi:
                lines.append(f"| {s['label']} | - | - | - | - | N/A |")
                continue
            per_axis = pi.get('per_axis', {})
            r_ind = per_axis.get('roll', {}).get('independence', -1)
            p_ind = per_axis.get('pitch', {}).get('independence', -1)
            y_ind = per_axis.get('yaw', {}).get('independence', -1)
            overall = pi.get('overall_independence', -1)
            verdict = pi.get('verdict', 'N/A')
            lines.append(
                f"| {s['label']} "
                f"| {r_ind:.3f} "
                f"| {p_ind:.3f} "
                f"| {y_ind:.3f} "
                f"| {overall:.3f} "
                f"| {verdict} |"
            )
        lines.append("")

        lines.append("GS_medw (MEDW-aggregated Generalization Score) 计算说明:")
        lines.append("- 所有指标基于 MEDW 多帧聚合: per-sequence robust median → 计算RPY误差 → 跨sequence取均值")
        lines.append("- 与实际部署完全一致: 部署时使用多帧聚合标定, 不依赖单帧精度")
        lines.append("- GS_medw = Σ(wi × Si), 范围 [0,1], 越低越好")
        lines.append("- S1 Zero-Drift (w=0.40): MEDW max(R,P,Y)/0.3° 聚合后的固有偏差")
        lines.append("- S2 Correction (w=0.30): (MEDW残差 - 方向感知ZD扣除) / 实际geodesic注入量, 取0.5°/1.0°均值")
        lines.append("- S3 Shortcut (w=0.15): 1-mean(genuine per-axis recovery)%, 方向感知扣除ZD同向分量")
        lines.append("- S4 Consistency (w=0.15): Asymmetry×0.6 + MagnitudeSensitivity×0.4 方向对称性(更重要)与量级敏感度")
        lines.append("- 注: 扰动使用LiDAR坐标系RPY轴 (右乘gt_R @ dR_lidar)")
        lines.append("- 点云投影可视化见各模型 gdiag_projections/ 目录")
        lines.append("")
        lines.append("指标联合解读指南:")
        lines.append("")
        lines.append("| | Shortcut低(<20%) | Shortcut中(20-50%) | Shortcut高(>50%) |")
        lines.append("| --- | --- | --- | --- |")
        lines.append("| Genuine高(>70%) | **理想模型**: 实打实的矫正 | 良好但有ZD辅助 | 以V70.2联合门控为准 |")
        lines.append("| Genuine中(30-70%) | 可用模型 | 一般 | 偏弱 |")
        lines.append("| Genuine低(<30%) | 较差模型 | 差 | **纯捷径**: 全靠ZD |")
        lines.append("")
        lines.append("- Genuine Recovery: 扣除ZD后的净残差, 反映模型输出离GT多近")
        lines.append("- Shortcut Proportion: 表观矫正中ZD贡献的比例, 反映矫正来源")
        lines.append("- Prediction Independence: 仅保留历史可比性，不进入checkpoint或部署判定")
        lines.append("- V70.2 只以 Genuine Recovery + 最差轴 signed slope + ZD 联合门控判定")
        lines.append("")

    # === ZD-Corrected 部署精度估算 ===
    stats_with_zd_and_temporal = [
        s for s in all_stats
        if (s.get('gdiag') or {}).get('zero_drift')
        and (s.get('temporal') or {}).get('best', {}).get('rot') is not None
    ]
    if stats_with_zd_and_temporal:
        lines.append("=" * 80)
        lines.append(f"{_CN.get(_sec, str(_sec))}、ZD-Corrected 部署精度估算 (减去固有偏置后)")
        _sec += 1
        lines.append("=" * 80)
        lines.append("")
        lines.append("假设部署时通过一次性标定消除 ZeroDrift 偏置 (Rbias), 估算矫正后的精度:")
        lines.append("  ZD-Corrected BEST ≈ sqrt((BEST_Roll - ZD_Roll)² + (BEST_Pitch - ZD_Pitch)² + (BEST_Yaw - ZD_Yaw)²)")
        lines.append("  其中 ZD_axis 取 signed mean (保留方向), BEST_axis 取 BEST 聚合的 per-seq 均值")
        lines.append("  注: 这是理想估计, 实际效果取决于 ZD 的时间稳定性")
        lines.append("")
        lines.append("| 排名 | 模型 | BEST Rot | ZD Rot | ZD-Corrected Rot | Roll | Pitch | Yaw | 改善 |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        zd_corrected_list = []
        for s in stats_with_zd_and_temporal:
            best_ta = s['temporal']['best']
            zd = s['gdiag']['zero_drift']
            zd_signed_rpy = [
                zd.get('roll_signed_mean', zd.get('roll_mean', 0)),
                zd.get('pitch_signed_mean', zd.get('pitch_mean', 0)),
                zd.get('yaw_signed_mean', zd.get('yaw_mean', 0)),
            ]
            best_roll = best_ta.get('roll', 0)
            best_pitch = best_ta.get('pitch', 0)
            best_yaw = best_ta.get('yaw', 0)
            corr_roll = max(0, abs(best_roll) - abs(zd_signed_rpy[0]))
            corr_pitch = max(0, abs(best_pitch) - abs(zd_signed_rpy[1]))
            corr_yaw = max(0, abs(best_yaw) - abs(zd_signed_rpy[2]))
            corr_rot = (corr_roll + corr_pitch + corr_yaw)
            zd_corrected_list.append({
                'label': s['label'],
                'best_rot': best_ta['rot'],
                'zd_rot': zd.get('rot_mean', 0),
                'corr_rot': corr_rot,
                'corr_roll': corr_roll,
                'corr_pitch': corr_pitch,
                'corr_yaw': corr_yaw,
            })

        zd_corrected_list.sort(key=lambda x: x['corr_rot'])
        for rank, item in enumerate(zd_corrected_list, 1):
            improve = (1 - item['corr_rot'] / item['best_rot']) * 100 if item['best_rot'] > 0 else 0
            lines.append(
                f"| {rank} | {item['label']} "
                f"| {item['best_rot']:.3f}° "
                f"| {item['zd_rot']:.3f}° "
                f"| {item['corr_rot']:.3f}° "
                f"| {item['corr_roll']:.3f}° "
                f"| {item['corr_pitch']:.3f}° "
                f"| {item['corr_yaw']:.3f}° "
                f"| {improve:.1f}% |"
            )
        lines.append("")
        lines.append("说明: ZD-Corrected 精度 = 减去模型固有偏置后的残差。部署时通过一次性零扰动标定获取 ZD bias,")
        lines.append("推理时 R_final = R_pred @ R_bias^(-1)。这消除系统误差, 仅保留随机噪声(可被聚合消除)。")
        lines.append("")

    # === BAG 泛化评估跨模型汇总 ===
    if BAG_EVAL_DIR and os.path.isdir(BAG_EVAL_DIR):
        bag_data = _collect_bag_eval_data(BAG_EVAL_DIR)
        if bag_data:
            bag_lines, bag_rankings = _bag_report_lines(BAG_EVAL_DIR, bag_data, _sec)
            _sec += 1
            lines.extend(bag_lines)

    lines.append("=" * 80)
    lines.append(f"{_CN.get(_sec, str(_sec))}、结论与建议")
    _sec += 1
    lines.append("=" * 80)
    lines.append("")

    # --- Per-frame ranking ---
    lines.append("--- 按 Per-frame 单帧误差排名 ---")
    lines.append("")
    lines.append("| 排名 | 模型 | Per-frame Rot | Roll | Pitch | Yaw |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for rank, s in enumerate(sorted_stats[:5], 1):
        lines.append(
            f"| {rank} | {s['label']} "
            f"| {s.get('rot_error_mean', -1):.3f}° "
            f"| {s.get('roll_error_mean', -1):.3f}° "
            f"| {s.get('pitch_error_mean', -1):.3f}° "
            f"| {s.get('yaw_error_mean', -1):.3f}° |"
        )
    lines.append("")

    # --- BEST temporal ranking ---
    stats_with_temporal = [s for s in all_stats
                          if s.get('temporal', {}).get('best', {}).get('rot') is not None]
    if stats_with_temporal:
        sorted_by_best = sorted(stats_with_temporal,
                                key=lambda s: s['temporal']['best']['rot'])
        best_ta = sorted_by_best[0]

        lines.append("--- 按 BEST 时序聚合排名 (可部署, 不依赖GT) ---")
        lines.append("")
        lines.append("| 排名 | 模型 | BEST Rot | 方法 | Roll | Pitch | Yaw | Per-frame→BEST改善 |")
        lines.append("| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |")
        for rank, s in enumerate(sorted_by_best, 1):
            tb = s['temporal']['best']
            pf = s.get('rot_error_mean', 999)
            improve = (1 - tb['rot'] / pf) * 100 if pf > 0 else 0
            lines.append(
                f"| {rank} | {s['label']} "
                f"| {tb['rot']:.3f}° "
                f"| {tb.get('method', '?')} "
                f"| {tb.get('roll', -1):.3f}° "
                f"| {tb.get('pitch', -1):.3f}° "
                f"| {tb.get('yaw', -1):.3f}° "
                f"| {improve:.1f}% |"
            )
        lines.append("")
    else:
        best_ta = None

    # --- Key findings ---
    lines.append("--- 关键发现 ---")
    lines.append("")
    finding_idx = 1
    if best_ta and best_ta['label'] != best['label']:
        lines.append(
            f"{finding_idx}. Per-frame 最佳 ({best['label']}, {best.get('rot_error_mean', -1):.3f}°) "
            f"≠ BEST 时序聚合最佳 ({best_ta['label']}, {best_ta['temporal']['best']['rot']:.3f}°)"
        )
        lines.append(f"   说明: 单帧精度高不等于聚合后精度高, "
                     f"关键在于误差是否为可聚合消除的随机噪声")
        finding_idx += 1
    lines.append(f"{finding_idx}. 最佳泛化模型 (per-frame): {best['label']} (Mean Rot: {best.get('rot_error_mean', -1):.3f}°)")
    finding_idx += 1
    if best_ta:
        lines.append(f"{finding_idx}. 最佳泛化模型 (BEST时序聚合): {best_ta['label']} "
                     f"(BEST: {best_ta['temporal']['best']['rot']:.3f}°, "
                     f"方法: {best_ta['temporal']['best'].get('method', '?')})")
        finding_idx += 1
    if has_gdiag and gdiag_sorted:
        best_gs = gdiag_sorted[0]
        gs_val = best_gs['gdiag'].get('composite', {}).get('GS_medw', -1)
        zd_rot = best_gs['gdiag'].get('zero_drift', {}).get('rot_mean', 0)
        raw_rec = best_gs['gdiag'].get('fixed_inject', {}).get('inject', {}).get('mean_recovery_pct', 0)
        _comp_raw = best_gs['gdiag'].get('composite', {}).get('raw', {})
        _has_signed = _comp_raw.get('zero_drift_signed_rpy') is not None
        genuine_rec = _comp_raw.get('genuine_recovery_pct', raw_rec) if _has_signed else raw_rec
        _accept = best_gs['gdiag'].get('acceptance_gate', {})
        _accept_text = (
            f", Acceptance={'PASS' if _accept.get('passed') else 'FAIL'}"
            f", SlopeWorst={_accept.get('signed_correction_slope_min_axis', -1):.3f}"
            if _accept else "")
        lines.append(f"{finding_idx}. 最佳综合泛化 (GS_medw): {best_gs['label']} "
                     f"(GS_medw={gs_val:.4f}, GenuineRecovery={genuine_rec:.1f}%, "
                     f"ZeroDrift={zd_rot:.4f}°, "
                     f"Shortcut={best_gs['gdiag'].get('shortcut_risk', 'N/A')}"
                     f"{_accept_text})")
        finding_idx += 1
    lines.append("")

    # --- 0.1° target analysis ---
    if best_ta:
        tb = best_ta['temporal']['best']
        lines.append("--- 距 0.1° 目标的评估 ---")
        lines.append("")
        lines.append(f"最优模型 {best_ta['label']} 的 BEST 指标:")
        lines.append("")
        lines.append("| 轴 | BEST | 距0.1° | 状态 |")
        lines.append("| --- | ---: | ---: | --- |")
        for axis, key in [("Roll", "roll"), ("Pitch", "pitch"), ("Yaw", "yaw")]:
            val = tb.get(key, 999)
            ratio = val / 0.1
            status = "已达标" if val <= 0.1 else ("接近" if val <= 0.15 else "需优化")
            lines.append(f"| {axis} | {val:.3f}° | {ratio:.1f}x | {status} |")
        lines.append(f"| Total | {tb['rot']:.3f}° | {tb['rot']/0.3:.1f}x | {'已达标' if tb['rot'] <= 0.3 else '需优化'} |")
        lines.append("")

    # --- Series summary ---
    lines.append("--- 各系列整体评价 ---")
    lines.append("")
    series_groups = {}
    import re as _re
    for s in sorted_stats:
        label = s['label']
        m = _re.match(r'(v\d+)', label)
        key = m.group(1).upper() if m else 'Other'
        if key == 'V25':
            key = 'V25r'
        series_groups.setdefault(key, []).append(s)
    for series, models in sorted(series_groups.items()):
        pf_best = min(m.get('rot_error_mean', 999) for m in models)
        pf_worst = max(m.get('rot_error_mean', 999) for m in models)
        best_models_ta = [m for m in models
                          if m.get('temporal', {}).get('best', {}).get('rot') is not None]
        if best_models_ta:
            ta_best = min(m['temporal']['best']['rot'] for m in best_models_ta)
            ta_label = min(best_models_ta, key=lambda m: m['temporal']['best']['rot'])['label']
            lines.append(f"- {series} ({len(models)}模型): "
                         f"Per-frame {pf_best:.3f}°~{pf_worst:.3f}°, "
                         f"BEST {ta_best:.3f}° ({ta_label})")
        else:
            lines.append(f"- {series} ({len(models)}模型): "
                         f"Per-frame {pf_best:.3f}°~{pf_worst:.3f}°")
    lines.append("")

    lines.append("--- 应用建议 ---")
    lines.append("")
    if best_ta:
        lines.append(f"- 生产环境推荐 (时序聚合): {best_ta['label']} (BEST {best_ta['temporal']['best']['rot']:.3f}°)")
        if len(sorted_by_best) > 1:
            second = sorted_by_best[1]
            lines.append(f"- 生产环境备选: {second['label']} "
                         f"(BEST {second['temporal']['best']['rot']:.3f}°)")
    lines.append(f"- 单帧最佳: {best['label']} (Per-frame {best.get('rot_error_mean', -1):.3f}°)")
    if len(sorted_stats) > 1:
        lines.append(f"- 单帧备选: {sorted_stats[1]['label']} (Per-frame {sorted_stats[1].get('rot_error_mean', -1):.3f}°)")
    lines.append("")

    report_text = "\n".join(lines) + "\n"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    _total_t0 = time.time()
    print("=" * 80)
    print("BEVCalib 多模型泛化性能评估")
    print(f"  模型数: {len(MODELS)}")
    print(f"  测试数据: {TEST_DATA}")
    print(f"  扰动: {ANGLE_RANGE} deg, {TRANS_RANGE} m")
    if EVAL_SAMPLE_STEP is not None and EVAL_MAX_FRAMES_PER_SEQ is not None:
        print("[FATAL] eval_sample_step 与 eval_max_frames_per_seq 互斥，不能同时配置。")
        return
    if EVAL_SAMPLE_STEP is not None:
        print(f"  采样: 每隔{EVAL_SAMPLE_STEP}帧取1帧")
    elif EVAL_MAX_FRAMES_PER_SEQ is not None:
        print(f"  采样: 每序列最多 {EVAL_MAX_FRAMES_PER_SEQ} 帧 (均匀下采样)")
    print(f"  输出: {OUTPUT_DIR}")
    if BAG_EVAL_DIR:
        print(f"  BAG评估: {BAG_EVAL_DIR}")
    if GDIAG_ONLY:
        print("  模式: gdiag_only")
    if _PARALLEL_GPUS > 1:
        print(f"  并行模式: {_PARALLEL_GPUS} GPUs")
    else:
        print(f"  并行模式: 关闭 (使用 --parallel -1 启用多卡并行)")
    print("=" * 80)

    if not _script_args.report_only:
        # Step 1: Run evaluations
        step_label = "Running gdiag-only evaluations..." if GDIAG_ONLY else "Running evaluations..."
        print(f"\n>>> Step 1: {step_label}")
        run_evaluations()
    else:
        print("\n>>> Step 1: SKIPPED (--report_only)")

    # Step 2: Collect stats
    print("\n>>> Step 2: Collecting results...")
    all_stats = collect_all_stats()
    if not all_stats:
        print("[ERROR] No evaluation results available!")
        return

    # Step 3: Generate charts
    print("\n>>> Step 3: Generating charts...")
    generate_charts(all_stats)

    # Step 4: Generate projection comparison
    print("\n>>> Step 4: Generating projection comparisons...")
    generate_projection_comparison(all_stats)

    # Step 5: Generate report
    print("\n>>> Step 5: Generating report...")
    report_path = generate_report(all_stats)

    _total_elapsed = time.time() - _total_t0
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"  Models evaluated: {len(all_stats)}/{len(MODELS)}")
    print(f"  Report: {report_path}")
    print(f"  Charts: {os.path.join(OUTPUT_DIR, 'charts')}/")
    print(f"  Projections: {os.path.join(OUTPUT_DIR, 'projection_comparison')}/")
    if BAG_EVAL_DIR and os.path.isdir(BAG_EVAL_DIR):
        print(f"  BAG泛化汇总: 已集成到报告 (来源: {BAG_EVAL_DIR})")
    print(f"  总耗时: {_format_elapsed(_total_elapsed)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
