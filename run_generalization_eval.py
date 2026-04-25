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
            "MODELS": cfg.get("models", DEFAULT_MODELS),
        }
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
        "MODELS": DEFAULT_MODELS,
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

    # Parse Sequence Boundaries
    seq_bounds = []
    bounds_start = text.find("Sequence Boundaries:")
    if bounds_start >= 0:
        for line in text[bounds_start:bounds_start+500].split('\n')[1:]:
            m_b = re.match(r'\s*Seq\s+(\S+):\s*samples\s+(\d+)\s*-\s*(\d+)\s*\((\d+)\s*frames\)', line)
            if m_b:
                seq_bounds.append({
                    'seq': m_b.group(1), 'start': int(m_b.group(2)),
                    'end': int(m_b.group(3)), 'count': int(m_b.group(4)),
                })
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
    ckpt_path = os.path.join(model_base,
                             os.path.basename(MODELS_DIR) +
                            "_scratch/checkpoint", mcfg["ckpt"])
    if not os.path.isfile(ckpt_path):
        import glob as _glob
        candidates = _glob.glob(os.path.join(model_base, "*_scratch/checkpoint", mcfg["ckpt"]))
        if candidates:
            ckpt_path = candidates[0]
        else:
            return None, None, ckpt_path

    env = os.environ.copy()
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("USE_DRCV_BACKEND", None)
    env["BEV_ZBOUND_STEP"] = mcfg["bev_zbound_step"]
    env["HF_HUB_OFFLINE"] = "1"
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
            "--rotation_only", str(mcfg["rotation_only"]),
            "--vis_interval", str(VIS_INTERVAL),
            "--batch_size", str(mcfg.get("batch_size", BATCH_SIZE)),
        ]
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
            "--rotation_only", str(mcfg["rotation_only"]),
            "--vis_interval", str(VIS_INTERVAL),
            "--batch_size", str(mcfg.get("batch_size", BATCH_SIZE)),
        ]
    if mcfg.get("use_deformable"):
        cmd.extend(["--deformable", "1"])
    if "use_mlp_head" in mcfg:
        cmd.extend(["--use_mlp_head", str(mcfg["use_mlp_head"])])
    if "bev_pool_factor" in mcfg:
        cmd.extend(["--bev_pool_factor", str(mcfg["bev_pool_factor"])])
    if "use_drcv" in mcfg:
        if mcfg["use_drcv"]:
            if not use_drinfer_backend:
                cmd.append("--use_drcv")
            env["USE_DRCV_BACKEND"] = "1"
        else:
            env["USE_DRCV_BACKEND"] = "0"
    if mcfg.get("use_foundation_depth"):
        cmd.extend(["--use_foundation_depth", str(mcfg["use_foundation_depth"])])
        env["HF_HUB_OFFLINE"] = "1"
    if mcfg.get("depth_model_type"):
        cmd.extend(["--depth_model_type", str(mcfg["depth_model_type"])])
    if mcfg.get("fd_mode"):
        cmd.extend(["--fd_mode", str(mcfg["fd_mode"])])
    if mcfg.get("voxel_mode"):
        cmd.extend(["--voxel_mode", str(mcfg["voxel_mode"])])
    if mcfg.get("scatter_reduce"):
        cmd.extend(["--scatter_reduce", str(mcfg["scatter_reduce"])])
    if mcfg.get("to_bev_mode"):
        cmd.extend(["--to_bev_mode", str(mcfg["to_bev_mode"])])
    if EVAL_SAMPLE_STEP is not None:
        cmd.extend(["--eval_sample_step", str(EVAL_SAMPLE_STEP)])
    if EVAL_MAX_FRAMES_PER_SEQ is not None:
        cmd.extend(["--eval_max_frames_per_seq", str(EVAL_MAX_FRAMES_PER_SEQ)])
    if mcfg.get("data_balance"):
        cmd.extend(["--data_balance", str(mcfg["data_balance"])])
    if mcfg.get("zero_image"):
        cmd.append("--zero_image")

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

        if os.path.isfile(extrinsics_path):
            with open(extrinsics_path, 'r') as f:
                if "EVALUATION STATISTICS" in f.read():
                    done.append((idx, label, mcfg, per_model_dir, None, "eval complete"))
                    continue

        model_base = _resolve_model_base(mcfg)
        ckpt_path = os.path.join(model_base,
                                 os.path.basename(MODELS_DIR) +
                                "_scratch/checkpoint", mcfg["ckpt"])
        if not os.path.isfile(ckpt_path):
            import glob as _glob
            candidates = _glob.glob(os.path.join(model_base, "*_scratch/checkpoint", mcfg["ckpt"]))
            if candidates:
                ckpt_path = candidates[0]
            else:
                train_log = os.path.join(model_base, "train.log")
                if os.path.isdir(model_base):
                    reason = f"ckpt not found ({mcfg['ckpt']}), dir exists"
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
        if os.path.isfile(extrinsics_path):
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
        print(f"  BEV_ZBOUND_STEP={mcfg['bev_zbound_step']}, rotation_only={mcfg['rotation_only']}")
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
            all_stats.append(stats)
            print(f"  {label}: Mean Rot={stats['rot_error_mean']:.3f} deg, "
                  f"P95={stats['rot_error_p95']:.3f} deg ({stats['samples']} samples)")
        else:
            print(f"  {label}: NO RESULTS")
    return all_stats


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


def generate_report(all_stats):
    """Generate Feishu-compatible markdown report."""
    report_path = os.path.join(OUTPUT_DIR, "GENERALIZATION_REPORT.md")
    lines = []

    sorted_stats = sorted(all_stats, key=lambda x: x.get('rot_error_mean', 999))
    best = sorted_stats[0]
    worst = sorted_stats[-1]
    n_samples = all_stats[0].get('samples', '?')

    lines.append("BEVCalib 多模型泛化性能对比报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"评估日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"测试数据集: test_data ({n_samples} samples, 3 sequences)")
    lines.append(f"扰动范围: +/-{ANGLE_RANGE} deg, +/-{TRANS_RANGE} m")
    lines.append(f"评估模型数: {len(all_stats)}")
    lines.append("")

    # Section 1
    lines.append("=" * 80)
    lines.append("一、实验配置概况")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| 模型标签 | 训练角度 | Z体素 | 架构版本 | 优化模式 | Checkpoint |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for s in all_stats:
        c = s['config']
        lines.append(f"| {s['label']} | {c['angle_deg']}deg | z={c['z_voxels']} | {c['version']} | {c['mode_desc']} | {c['ckpt']} |")
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
        lines.append("**Micro vs Macro 对比 (均衡评估)**:")
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

    # Section 4: Translation (full-mode models only)
    trans_stats = [s for s in all_stats if s.get('trans_error_mean') and s['trans_error_mean'] > 0]
    if trans_stats:
        lines.append("=" * 80)
        lines.append("四、平移误差对比 (仅 rotation+translation 模型, m)")
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
        sec_num = "五" if trans_stats else "四"
        lines.append("=" * 80)
        lines.append(f"{sec_num}、训练精度 vs 泛化精度对比")
        lines.append("=" * 80)
        lines.append("")
        lines.append("| 模型 | 训练Rot(deg) | 测试Rot(deg) | 泛化衰退 | 评级 |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for s in sorted(has_train, key=lambda x: x.get('rot_error_mean', 999)):
            train_rot = s['train_metrics']['rot_error']
            test_rot = s['rot_error_mean']
            deg = test_rot / train_rot if train_rot > 0 else 0
            if deg < 3:
                rating = "优秀"
            elif deg < 5:
                rating = "良好"
            elif deg < 7:
                rating = "一般"
            else:
                rating = "需改进"
            lines.append(f"| {s['label']} | {train_rot:.2f} | {test_rot:.3f} | {deg:.1f}x | {rating} |")
        lines.append("")
        lines.append("![Train vs Test](charts/train_vs_test.png)")
        lines.append("")

    # Section 6: Ranking
    sec_num_rank = "六" if trans_stats else "五"
    lines.append("=" * 80)
    lines.append(f"{sec_num_rank}、模型排名 (按 Mean Rotation Error)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| 排名 | 模型 | Mean Rot(deg) | Median | P95 | Max |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for rank, s in enumerate(sorted_stats, 1):
        lines.append(
            f"| {rank} | {s['label']} "
            f"| {s.get('rot_error_mean', -1):.3f} "
            f"| {s.get('rot_error_median', -1):.3f} "
            f"| {s.get('rot_error_p95', -1):.3f} "
            f"| {s.get('rot_error_max', -1):.3f} |"
        )
    lines.append("")
    lines.append("![Model Ranking](charts/model_ranking.png)")
    lines.append("")

    # Section 7: Per-sequence analysis
    sec_num_seq = "七" if trans_stats else "六"
    has_seq_data = any(s.get('per_sequence') for s in all_stats)
    if has_seq_data:
        lines.append("=" * 80)
        lines.append(f"{sec_num_seq}、Per-Sequence 误差分析")
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

            # Flag problematic sequences
            lines.append("异常 sequence 识别（任一模型 Mean Rot > 2x 整体 Mean）:")
            for s in sorted_stats:
                ps = s.get('per_sequence', {})
                overall_mean = s.get('rot_error_mean', 0)
                for sid, sv in ps.items():
                    if sv['rot_mean'] > overall_mean * 2 and sv['samples'] >= 50:
                        lines.append(
                            f"  - {s['label']}: Seq {sid} Mean={sv['rot_mean']:.3f}° "
                            f"(整体={overall_mean:.3f}°, 比值={sv['rot_mean']/max(overall_mean,0.001):.1f}x, "
                            f"{sv['samples']}样本)")
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
    sec_num_proj = "八" if has_seq_data else ("七" if trans_stats else "六")
    lines.append("=" * 80)
    lines.append(f"{sec_num_proj}、点云投影效果图对比")
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

    # Section 9: Conclusions
    sec_num_conc = "九" if has_seq_data else ("八" if trans_stats else "七")
    lines.append("=" * 80)
    lines.append(f"{sec_num_conc}、结论与建议")
    lines.append("=" * 80)
    lines.append("")

    ratio = worst.get('rot_error_mean', 1) / max(best.get('rot_error_mean', 1), 0.001)
    lines.append(f"1. 最佳泛化模型: {best['label']} (Mean Rot: {best.get('rot_error_mean', -1):.3f} deg)")
    lines.append(f"2. 最差泛化模型: {worst['label']} (Mean Rot: {worst.get('rot_error_mean', -1):.3f} deg)")
    lines.append(f"3. 最差/最佳比值: {ratio:.1f}x")
    lines.append("")

    z_groups = {}
    for s in all_stats:
        z = s['config']['z_voxels']
        z_groups.setdefault(z, []).append(s)

    lines.append("Z体素影响分析:")
    for z in sorted(z_groups.keys()):
        group = z_groups[z]
        avg_rot = np.mean([s['rot_error_mean'] for s in group])
        lines.append(f"  - z={z}: 平均Mean Rot = {avg_rot:.3f} deg ({len(group)}个模型)")
    lines.append("")

    angle_groups = {}
    for s in all_stats:
        a = s['config']['angle_deg']
        angle_groups.setdefault(a, []).append(s)

    lines.append("训练角度影响分析:")
    for a in sorted(angle_groups.keys()):
        group = angle_groups[a]
        avg_rot = np.mean([s['rot_error_mean'] for s in group])
        lines.append(f"  - {a}deg训练: 平均Mean Rot = {avg_rot:.3f} deg ({len(group)}个模型)")
    lines.append("")

    mode_groups = {}
    for s in all_stats:
        m = s['config']['mode_desc']
        mode_groups.setdefault(m, []).append(s)

    lines.append("优化模式影响分析:")
    for m, group in mode_groups.items():
        avg_rot = np.mean([s['rot_error_mean'] for s in group])
        lines.append(f"  - {m}: 平均Mean Rot = {avg_rot:.3f} deg ({len(group)}个模型)")
    lines.append("")

    lines.append("应用建议:")
    lines.append(f"  - 生产环境推荐: {best['label']} (最佳泛化能力)")
    if len(sorted_stats) > 1:
        lines.append(f"  - 备选方案: {sorted_stats[1]['label']} (第二名)")
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
    if _PARALLEL_GPUS > 1:
        print(f"  并行模式: {_PARALLEL_GPUS} GPUs")
    else:
        print(f"  并行模式: 关闭 (使用 --parallel -1 启用多卡并行)")
    print("=" * 80)

    # Step 1: Run evaluations
    print("\n>>> Step 1: Running evaluations...")
    run_evaluations()

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
    print(f"  总耗时: {_format_elapsed(_total_elapsed)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
