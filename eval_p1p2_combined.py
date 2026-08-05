#!/usr/bin/env python3
"""P1+P2 Combined Evaluation Pipeline.

This script automates the full evaluation pipeline:
  1. Wait for V81 training to complete (monitor train.log)
  2. Run generalization evaluation for V81 + V66 baseline
  3. Apply P2 ZD compensation on V81 predictions
  4. Generate combined comparison report

Usage:
    # Full pipeline (wait for training + eval + ZD compensation):
    python eval_p1p2_combined.py

    # Skip training wait (if training already done):
    python eval_p1p2_combined.py --skip_wait

    # Only run ZD compensation on existing eval results:
    python eval_p1p2_combined.py --zd_only

    # Custom ZD params (default: optimal from V66 grid search):
    python eval_p1p2_combined.py --zd_only --ema_alpha 0.15 --max_correction_deg 2.0
"""

import argparse
import os
import sys
import glob
import json
import time
import subprocess
import numpy as np

BEVCALIB_ROOT = "/mnt/drtraining/user/dahailu/code/BEVCalib"
V81_TRAIN_LOG = os.path.join(BEVCALIB_ROOT, "logs/train_v81.log")
V81_EVAL_CONFIG = os.path.join(BEVCALIB_ROOT, "configs/c1_retrain/eval_generalization_c1_v81.yaml")
V81_TRAIN_DIR = os.path.join(
    BEVCALIB_ROOT,
    "logs/all_training_data_c1/model_small_5deg_c1_v81_iterative_jacobian_S1"
)

# Default ZD params (optimal from V66 grid search)
DEFAULT_ZD_PARAMS = {
    'ema_alpha': 0.15,
    'max_correction_deg': 2.0,
    'calibration_frames': 100,
}


def tprint(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def wait_for_training(timeout_hours=48):
    """Wait for V81 training to complete."""
    tprint("等待 V81 训练完成...")
    start = time.time()
    last_epoch = -1
    stall_count = 0

    while True:
        elapsed_h = (time.time() - start) / 3600
        if elapsed_h > timeout_hours:
            tprint(f"  ⚠️ 超时 ({timeout_hours}h)，继续执行评估...")
            break

        if not os.path.isfile(V81_TRAIN_LOG):
            time.sleep(60)
            continue

        with open(V81_TRAIN_LOG, 'r', errors='ignore') as f:
            content = f.read()

        # Check for training completion
        if "Training complete" in content or "best model saved" in content.lower():
            tprint("  ✅ 训练完成!")
            break

        # Check for early stopping
        if "Early stopping" in content:
            tprint("  ✅ 训练早停!")
            break

        # Parse current epoch
        import re
        epochs = re.findall(r'Epoch \[(\d+)/200\] completed', content)
        if epochs:
            cur_epoch = int(epochs[-1])
            if cur_epoch > last_epoch:
                last_epoch = cur_epoch
                stall_count = 0
                tprint(f"  📊 Epoch {cur_epoch}/200 完成...")
            elif cur_epoch == 199:
                tprint("  ✅ 达到最大 epoch (200)!")
                break
        else:
            # Check process alive
            if not _is_training_alive():
                tprint("  ⚠️ 训练进程已退出")
                break

        time.sleep(120)  # Check every 2 minutes


def _is_training_alive():
    """Check if V81 training process is still running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "v81_iterative_jacobian"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def run_generalization_eval():
    """Run the generalization evaluation."""
    tprint("启动泛化评估...")
    cmd = [
        sys.executable,
        os.path.join(BEVCALIB_ROOT, "run_generalization_eval.py"),
        "--config", V81_EVAL_CONFIG,
        "--parallel", "-1",
        "--eval_max_frames_per_seq", "500",
    ]
    tprint(f"  命令: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, cwd=BEVCALIB_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  [EVAL] {line}")
    proc.wait()
    tprint(f"  评估完成 (exit code: {proc.returncode})")
    return proc.returncode == 0


def find_eval_npz(label):
    """Find all_T_pred_gt.npz for a given model label."""
    eval_dir = os.path.join(BEVCALIB_ROOT, "logs/evaluations/generalization_c1_v81", label)
    npz_path = os.path.join(eval_dir, "all_T_pred_gt.npz")
    if os.path.isfile(npz_path):
        return npz_path
    # Also check test_data_eval subdirectory
    npz_path2 = os.path.join(eval_dir, "test_data_eval", "all_T_pred_gt.npz")
    if os.path.isfile(npz_path2):
        return npz_path2
    return None


def compute_medw_from_npz(npz_path, window=200):
    """Compute MEDW metric from prediction npz."""
    from scipy.spatial.transform import Rotation as ScipyRot

    data = np.load(npz_path)
    T_pred = data['T_pred']
    T_gt = data['T_gt']
    seq_ids = data.get('seq_ids', None)
    if seq_ids is None:
        seq_ids = data.get('seq_id', None)

    N = T_pred.shape[0]
    R_pred = T_pred[:, :3, :3]
    R_gt = T_gt[:, :3, :3]
    R_err_mat = np.matmul(R_pred, np.transpose(R_gt, (0, 2, 1)))

    errors_deg = np.zeros(N)
    rpy_errors = np.zeros((N, 3))
    for i in range(N):
        try:
            r = ScipyRot.from_matrix(R_err_mat[i])
            rv = r.as_rotvec()
            errors_deg[i] = np.linalg.norm(rv) * 180 / np.pi
            rpy_errors[i] = r.as_euler('xyz', degrees=True)
        except Exception:
            errors_deg[i] = 999.0

    # Per-sequence MEDW
    results = {}
    if seq_ids is not None:
        unique_seqs = np.unique(seq_ids)
        for sid in unique_seqs:
            mask = seq_ids == sid
            errs = errors_deg[mask]
            rpy = rpy_errors[mask]
            n = len(errs)
            if n < window:
                w = n
            else:
                w = window
            # Sliding window
            medw_vals = []
            for start in range(0, n - w + 1, max(1, w // 4)):
                medw_vals.append(np.median(errs[start:start + w]))
            if medw_vals:
                results[str(sid)] = {
                    'medw': float(np.mean(medw_vals)),
                    'medw_r': float(np.median(np.abs(rpy[:, 0]))),
                    'medw_p': float(np.median(np.abs(rpy[:, 1]))),
                    'medw_y': float(np.median(np.abs(rpy[:, 2]))),
                    'n_frames': n,
                }

    # Overall
    all_medw = []
    for sid, r in results.items():
        all_medw.append(r['medw'])
    overall = float(np.mean(all_medw)) if all_medw else float(np.mean(errors_deg))

    return {
        'overall_medw': overall,
        'per_seq': results,
        'mean_error': float(np.mean(errors_deg[errors_deg < 900])),
    }


def apply_zd_compensation(npz_path, params=None):
    """Apply ZD compensation and return improved metrics."""
    sys.path.insert(0, os.path.join(BEVCALIB_ROOT, 'utils'))
    from zd_online_compensator import evaluate_offline

    if params is None:
        params = DEFAULT_ZD_PARAMS

    output_dir = npz_path.replace('all_T_pred_gt.npz', 'zd_compensated')
    os.makedirs(output_dir, exist_ok=True)

    result = evaluate_offline(npz_path, output_dir=output_dir, params=params)
    return result


def generate_report(v81_results, v66_results=None):
    """Generate comparison report."""
    tprint("\n" + "=" * 70)
    tprint("P1+P2 联合评估报告")
    tprint("=" * 70)

    tprint("\n📊 V81 (DINOv2 + Iterative + Jacobian + Pitch强化):")
    if v81_results:
        for label, metrics in v81_results.items():
            tprint(f"  {label}:")
            tprint(f"    MEDW overall: {metrics.get('overall_medw', 'N/A'):.3f}°")
            if 'zd_compensated' in metrics:
                zd = metrics['zd_compensated']
                tprint(f"    + ZD补偿:     {zd.get('best_medw', 'N/A'):.3f}°")
                if 'best_medw' in zd and metrics.get('overall_medw'):
                    imp = (1 - zd['best_medw'] / metrics['overall_medw']) * 100
                    tprint(f"    改善:         {imp:.1f}%")

    if v66_results:
        tprint("\n📊 V66 (DINOv2 baseline):")
        for label, metrics in v66_results.items():
            tprint(f"  {label}:")
            tprint(f"    MEDW overall: {metrics.get('overall_medw', 'N/A'):.3f}°")

    tprint("\n" + "=" * 70)
    tprint("目标: Gen BEST < 0.1°")
    tprint("=" * 70)


def run_zd_only(args):
    """Run only ZD compensation on existing eval results."""
    tprint("仅运行 ZD 补偿模式")
    params = {
        'ema_alpha': args.ema_alpha,
        'max_correction_deg': args.max_correction_deg,
        'calibration_frames': args.calibration_frames,
    }

    v81_labels = ["c1-v81-best-dual", "c1-v81-best-medw", "c1-v81-best-val"]
    v81_results = {}

    for label in v81_labels:
        npz_path = find_eval_npz(label)
        if npz_path is None:
            tprint(f"  ⚠️ {label}: all_T_pred_gt.npz 未找到，跳过")
            continue

        tprint(f"  📁 {label}: {npz_path}")

        # Baseline metrics
        baseline = compute_medw_from_npz(npz_path)
        tprint(f"    基线 MEDW: {baseline['overall_medw']:.3f}°")

        # Apply ZD compensation
        try:
            zd_result = apply_zd_compensation(npz_path, params)
            v81_results[label] = {
                'overall_medw': baseline['overall_medw'],
                'zd_compensated': zd_result if isinstance(zd_result, dict) else {'best_medw': 0},
            }
        except Exception as e:
            tprint(f"    ⚠️ ZD 补偿失败: {e}")
            v81_results[label] = {'overall_medw': baseline['overall_medw']}

    generate_report(v81_results)


def main():
    parser = argparse.ArgumentParser(description="P1+P2 Combined Evaluation")
    parser.add_argument("--skip_wait", action="store_true",
                        help="跳过等待训练完成")
    parser.add_argument("--zd_only", action="store_true",
                        help="仅运行 ZD 补偿 (需要已有评估结果)")
    parser.add_argument("--ema_alpha", type=float, default=DEFAULT_ZD_PARAMS['ema_alpha'])
    parser.add_argument("--max_correction_deg", type=float, default=DEFAULT_ZD_PARAMS['max_correction_deg'])
    parser.add_argument("--calibration_frames", type=int, default=DEFAULT_ZD_PARAMS['calibration_frames'])
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过泛化评估 (已有结果时)")
    args = parser.parse_args()

    os.chdir(BEVCALIB_ROOT)

    if args.zd_only:
        run_zd_only(args)
        return

    # Step 1: Wait for training
    if not args.skip_wait:
        wait_for_training()
    else:
        tprint("跳过等待训练完成")

    # Step 2: Run generalization evaluation
    if not args.skip_eval:
        success = run_generalization_eval()
        if not success:
            tprint("⚠️ 泛化评估可能有问题，继续检查已有结果...")
    else:
        tprint("跳过泛化评估")

    # Step 3: Apply ZD compensation on V81 results
    tprint("\n应用 P2 ZD 在线补偿...")
    params = {
        'ema_alpha': args.ema_alpha,
        'max_correction_deg': args.max_correction_deg,
        'calibration_frames': args.calibration_frames,
    }

    v81_labels = ["c1-v81-best-dual", "c1-v81-best-medw", "c1-v81-best-val"]
    v81_results = {}

    for label in v81_labels:
        npz_path = find_eval_npz(label)
        if npz_path is None:
            tprint(f"  ⚠️ {label}: all_T_pred_gt.npz 未找到，跳过")
            continue

        tprint(f"  📁 {label}")
        baseline = compute_medw_from_npz(npz_path)
        tprint(f"    基线 MEDW: {baseline['overall_medw']:.3f}°")

        try:
            zd_result = apply_zd_compensation(npz_path, params)
            v81_results[label] = {
                'overall_medw': baseline['overall_medw'],
                'zd_compensated': zd_result if isinstance(zd_result, dict) else {'best_medw': 0},
            }
        except Exception as e:
            tprint(f"    ⚠️ ZD 补偿失败: {e}")
            v81_results[label] = {'overall_medw': baseline['overall_medw']}

    # Step 4: V66 baseline for comparison
    v66_results = {}
    v66_eval_dir = os.path.join(BEVCALIB_ROOT, "logs/evaluations/generalization_c1_v66")
    for label in ["c1-v66-best-dual-baseline", "c1-v66-best-medw-baseline"]:
        npz_path = find_eval_npz(label)
        if npz_path is None:
            # Try V66 eval directory
            alt = os.path.join(v66_eval_dir, label.replace("-baseline", ""), "all_T_pred_gt.npz")
            if os.path.isfile(alt):
                npz_path = alt
        if npz_path and os.path.isfile(npz_path):
            baseline = compute_medw_from_npz(npz_path)
            v66_results[label] = {'overall_medw': baseline['overall_medw']}

    # Step 5: Report
    generate_report(v81_results, v66_results if v66_results else None)


if __name__ == "__main__":
    main()
