#!/usr/bin/env python3
"""
多帧融合评估脚本 —— 从已有单帧评估结果分析多帧平均对精度的影响

核心思路:
  在真实部署中，车辆静止时多帧预测取平均可显著降低随机误差。
  本脚本无需重新推理，直接解析已有 extrinsics_and_errors.txt，
  在同一 sequence 内做滑动窗口平均，统计误差随窗口大小的衰减。

用法:
  python evaluate_multiframe.py \
      --eval_dirs logs/evaluations/generalization_eval_v24/v24-B-diff-only \
                  logs/evaluations/generalization_eval_v24/v24-A-baseline \
      --window_sizes 1,5,10,20,50,100 \
      --output_dir logs/evaluations/multiframe_analysis
"""

import os
import re
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))
from evaluate_extrinsics import evaluate_sensor_extrinsic


# ─────────────────────────────────────────────────────────────────────────────
# Rotation averaging utilities
# ─────────────────────────────────────────────────────────────────────────────

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def average_quaternions(quats: np.ndarray) -> np.ndarray:
    """Average multiple quaternions using the eigenvector method (Markley et al.).
    
    quats: (N, 4) array of unit quaternions [w, x, y, z]
    Returns: (4,) average quaternion
    """
    N = quats.shape[0]
    # Ensure consistent hemisphere (flip if dot < 0 relative to first)
    for i in range(1, N):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    M = quats.T @ quats  # (4, 4)
    eigvals, eigvecs = np.linalg.eigh(M)
    avg_q = eigvecs[:, -1]  # largest eigenvalue
    if avg_q[0] < 0:
        avg_q = -avg_q
    return avg_q / np.linalg.norm(avg_q)


def average_T_matrices(T_list: List[np.ndarray]) -> np.ndarray:
    """Average multiple 4x4 transformation matrices.
    
    Rotation: quaternion eigenvector averaging
    Translation: arithmetic mean
    """
    N = len(T_list)
    if N == 1:
        return T_list[0].copy()
    
    quats = np.array([rotmat_to_quat(T[:3, :3]) for T in T_list])
    avg_q = average_quaternions(quats)
    avg_R = quat_to_rotmat(avg_q)
    avg_t = np.mean([T[:3, 3] for T in T_list], axis=0)
    
    avg_T = np.eye(4)
    avg_T[:3, :3] = avg_R
    avg_T[:3, 3] = avg_t
    return avg_T


# ─────────────────────────────────────────────────────────────────────────────
# Error computation — delegates to the same function used by evaluate_checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def compute_rotation_error(pred_T: np.ndarray, gt_T: np.ndarray) -> Dict[str, float]:
    """Compute errors using the canonical evaluate_sensor_extrinsic."""
    angle_error, axis_angle_error, pos_error, axis_pos_error = \
        evaluate_sensor_extrinsic(pred_T, gt_T)
    return {
        'rot_error': angle_error,
        'roll_error': abs(axis_angle_error[0]),
        'pitch_error': abs(axis_angle_error[1]),
        'yaw_error': abs(axis_angle_error[2]),
        'trans_error': pos_error / 100.0,
        'fwd_error': abs(axis_pos_error[0]) / 100.0,
        'lat_error': abs(axis_pos_error[1]) / 100.0,
        'ht_error': abs(axis_pos_error[2]) / 100.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parse existing evaluation results
# ─────────────────────────────────────────────────────────────────────────────

def parse_extrinsics_file(filepath: str) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    """Parse extrinsics_and_errors.txt to extract per-sample predictions.
    
    Returns:
        seq_gt_map: {seq_id: gt_T (4x4)} loaded from dataset calib files
        samples: list of dicts with 'idx', 'seq', 'pred_T' (4x4)
    """
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Detect dataset path for loading per-sequence GT
    ds_match = re.search(r'Dataset:\s*(.+)', text)
    dataset_root = ds_match.group(1).strip() if ds_match else None
    
    # Parse per-sample predictions
    samples = []
    sample_pattern = re.compile(
        r'Sample (\d+) \[Seq (\d+)\].*?'
        r'Predicted Extrinsics.*?\n((?:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n){4})',
        re.DOTALL
    )
    for m in sample_pattern.finditer(text):
        idx = int(m.group(1))
        seq = int(m.group(2))
        pred_lines = m.group(3).strip().split('\n')
        pred_T = np.array([[float(x) for x in line.split()] for line in pred_lines])
        samples.append({'idx': idx, 'seq': seq, 'pred_T': pred_T})
    
    # Load per-sequence GT from dataset calib files
    seq_ids = sorted(set(s['seq'] for s in samples))
    seq_gt_map = load_per_sequence_gt(dataset_root, seq_ids)
    
    return seq_gt_map, samples


def load_per_sequence_gt(
    dataset_root: Optional[str],
    seq_ids: List[int],
) -> Dict[int, np.ndarray]:
    """Load per-sequence GT extrinsics (LiDAR→Camera) from calib.txt files.
    
    The `Tr:` line in calib.txt is Camera→LiDAR (3x4). We invert to get LiDAR→Camera.
    """
    seq_gt = {}
    if not dataset_root or not os.path.isdir(dataset_root):
        return seq_gt
    
    for seq_id in seq_ids:
        calib_path = os.path.join(dataset_root, "sequences", f"{seq_id:02d}", "calib.txt")
        if not os.path.isfile(calib_path):
            continue
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('Tr:'):
                    vals = [float(x) for x in line.strip().split()[1:]]
                    T_cam2lidar = np.eye(4)
                    T_cam2lidar[:3, :] = np.array(vals).reshape(3, 4)
                    T_lidar2cam = np.linalg.inv(T_cam2lidar)
                    seq_gt[seq_id] = T_lidar2cam
                    break
    return seq_gt


def parse_sequence_boundaries(filepath: str) -> List[Tuple[int, int, int]]:
    """Parse sequence boundaries from extrinsics_and_errors.txt.
    
    Returns: list of (seq_id, start_idx, end_idx)
    """
    with open(filepath, 'r') as f:
        text = f.read()
    
    boundaries = []
    for m in re.finditer(r'Seq (\d+): samples (\d+) - (\d+)', text):
        boundaries.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Multi-frame averaging analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_multiframe(
    seq_gt_map: Dict[int, np.ndarray],
    samples: List[Dict],
    window_sizes: List[int],
    seq_boundaries: List[Tuple[int, int, int]],
) -> Dict[int, Dict[str, np.ndarray]]:
    """Perform sliding-window averaging within each sequence.
    
    For each window size N:
      - Within each sequence, slide a window of size N
      - Average the N predicted T matrices
      - Compute error of the averaged prediction vs sequence-specific GT
    
    Returns: {window_size: {metric_name: array_of_errors}}
    """
    seq_samples = {}
    for s in samples:
        seq_samples.setdefault(s['seq'], []).append(s)
    for seq in seq_samples:
        seq_samples[seq].sort(key=lambda x: x['idx'])
    
    results = {}
    for N in window_sizes:
        all_errors = {
            'rot_error': [], 'roll_error': [], 'pitch_error': [],
            'yaw_error': [], 'trans_error': [],
        }
        
        for seq_id, seq_list in seq_samples.items():
            if seq_id not in seq_gt_map:
                continue
            gt_T = seq_gt_map[seq_id]
            
            if len(seq_list) < N:
                continue
            
            # Sliding window with step = N//2 (overlapping for more data points)
            step = max(1, N // 2)
            for start in range(0, len(seq_list) - N + 1, step):
                window = seq_list[start:start + N]
                T_list = [s['pred_T'] for s in window]
                avg_T = average_T_matrices(T_list)
                errors = compute_rotation_error(avg_T, gt_T)
                for key in all_errors:
                    all_errors[key].append(errors[key])
        
        results[N] = {k: np.array(v) for k, v in all_errors.items()}
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    all_model_results: Dict[str, Dict],
    window_sizes: List[int],
    output_dir: str,
):
    """Generate multi-frame analysis report with charts."""
    os.makedirs(output_dir, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    lines = []
    lines.append("BEVCalib 多帧融合评估报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"窗口大小: {window_sizes}")
    lines.append(f"评估模型数: {len(all_model_results)}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("一、核心结论: 误差 vs 窗口大小")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| 模型 | " + " | ".join(f"N={N}" for N in window_sizes) + " | 理论√N衰减 |")
    lines.append("| --- | " + " | ".join("---:" for _ in window_sizes) + " | --- |")
    
    for label, model_data in all_model_results.items():
        mf_results = model_data['multiframe_results']
        row_vals = []
        base_err = np.mean(mf_results[1]['rot_error']) if 1 in mf_results else 0
        for N in window_sizes:
            if N in mf_results and len(mf_results[N]['rot_error']) > 0:
                mean_err = np.mean(mf_results[N]['rot_error'])
                row_vals.append(f"{mean_err:.3f}°")
            else:
                row_vals.append("-")
        
        # Theoretical √N decay from single-frame
        if base_err > 0 and len(window_sizes) > 1:
            theory = f"{base_err:.3f}/√N"
        else:
            theory = "-"
        lines.append(f"| {label} | " + " | ".join(row_vals) + f" | {theory} |")
    
    lines.append("")
    
    # Detailed per-model tables
    lines.append("=" * 80)
    lines.append("二、各模型详细分析")
    lines.append("=" * 80)
    
    for label, model_data in all_model_results.items():
        mf_results = model_data['multiframe_results']
        lines.append("")
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 窗口大小 | Mean Rot | Median Rot | P95 Rot | Roll | Pitch | Yaw | 相对改善 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        
        base_mean = np.mean(mf_results[1]['rot_error']) if 1 in mf_results else 1
        
        for N in window_sizes:
            if N not in mf_results or len(mf_results[N]['rot_error']) == 0:
                continue
            errs = mf_results[N]
            mean_rot = np.mean(errs['rot_error'])
            median_rot = np.median(errs['rot_error'])
            p95_rot = np.percentile(errs['rot_error'], 95)
            mean_roll = np.mean(errs['roll_error'])
            mean_pitch = np.mean(errs['pitch_error'])
            mean_yaw = np.mean(errs['yaw_error'])
            improvement = base_mean / max(mean_rot, 0.001) if N > 1 else 1.0
            theory_ratio = np.sqrt(N)
            
            lines.append(
                f"| {N} | {mean_rot:.4f}° | {median_rot:.4f}° | {p95_rot:.4f}° "
                f"| {mean_roll:.4f}° | {mean_pitch:.4f}° | {mean_yaw:.4f}° "
                f"| {improvement:.2f}x (理论{theory_ratio:.1f}x) |"
            )
        lines.append("")
    
    # Per-axis improvement table
    lines.append("=" * 80)
    lines.append("三、Pitch 轴改善分析（关键瓶颈）")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| 模型 | 单帧 Pitch | N=10 Pitch | N=50 Pitch | N=100 Pitch | 改善倍数(N=50) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    
    for label, model_data in all_model_results.items():
        mf_results = model_data['multiframe_results']
        single = np.mean(mf_results[1]['pitch_error']) if 1 in mf_results else -1
        n10 = np.mean(mf_results[10]['pitch_error']) if 10 in mf_results and len(mf_results[10]['pitch_error']) > 0 else -1
        n50 = np.mean(mf_results[50]['pitch_error']) if 50 in mf_results and len(mf_results[50]['pitch_error']) > 0 else -1
        n100 = np.mean(mf_results[100]['pitch_error']) if 100 in mf_results and len(mf_results[100]['pitch_error']) > 0 else -1
        ratio = f"{single / n50:.2f}x" if n50 > 0 and single > 0 else "-"
        
        def _fmt(v):
            return f"{v:.4f}°" if v > 0 else "-"
        
        lines.append(f"| {label} | {_fmt(single)} | {_fmt(n10)} | {_fmt(n50)} | {_fmt(n100)} | {ratio} |")
    
    lines.append("")
    
    # Section 4: Can we reach 0.1°?
    lines.append("=" * 80)
    lines.append("四、能否达到 0.1° 目标？")
    lines.append("=" * 80)
    lines.append("")
    
    for label, model_data in all_model_results.items():
        mf_results = model_data['multiframe_results']
        lines.append(f"{label}:")
        found_target = False
        for N in sorted(mf_results.keys()):
            if len(mf_results[N]['rot_error']) == 0:
                continue
            mean_rot = np.mean(mf_results[N]['rot_error'])
            if mean_rot <= 0.1 and not found_target:
                lines.append(f"  - N={N} 帧平均即可达到 Mean Rot = {mean_rot:.4f}° < 0.1° ✓")
                found_target = True
            elif N == max(window_sizes):
                if not found_target:
                    lines.append(f"  - N={N} 帧平均: Mean Rot = {mean_rot:.4f}°")
                    needed_N = int(np.ceil((np.mean(mf_results[1]['rot_error']) / 0.1) ** 2))
                    lines.append(f"  - 理论需要约 N={needed_N} 帧才能达到 0.1° (假设 √N 衰减)")
        lines.append("")
    
    # Generate charts
    _generate_charts(all_model_results, window_sizes, charts_dir, lines)
    
    lines.append("=" * 80)
    lines.append("五、部署建议")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. 静止窗口检测: IMU/轮速计判断车辆静止，触发标定更新")
    lines.append("2. 多帧累积: 在静止窗口内累积 30-100 帧预测，取四元数平均")
    lines.append("3. 异常值剔除: 丢弃与中位数偏差 > 2σ 的预测帧")
    lines.append("4. EMA 滤波: 对历史标定结果做指数移动平均，防止突变")
    lines.append("5. 置信度: 可根据预测方差估算置信区间，仅在置信度足够时更新标定")
    lines.append("")
    
    report_path = os.path.join(output_dir, "MULTIFRAME_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[OK] Report: {report_path}")
    return report_path


def _generate_charts(
    all_model_results: Dict[str, Dict],
    window_sizes: List[int],
    charts_dir: str,
    report_lines: List[str],
):
    """Generate multi-frame analysis charts."""
    
    colors = ['#2196F3', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e']
    
    # Chart 1: Error vs Window Size (all models)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    for idx, (label, model_data) in enumerate(all_model_results.items()):
        mf = model_data['multiframe_results']
        Ns = sorted(N for N in mf if len(mf[N]['rot_error']) > 0)
        means = [np.mean(mf[N]['rot_error']) for N in Ns]
        color = colors[idx % len(colors)]
        ax.plot(Ns, means, 'o-', color=color, label=label, linewidth=2, markersize=6)
    
    # Add theoretical √N decay line
    if all_model_results:
        first_label = list(all_model_results.keys())[0]
        first_mf = all_model_results[first_label]['multiframe_results']
        if 1 in first_mf and len(first_mf[1]['rot_error']) > 0:
            base = np.mean(first_mf[1]['rot_error'])
            theory_Ns = np.array(window_sizes)
            theory_vals = base / np.sqrt(theory_Ns)
            ax.plot(theory_Ns, theory_vals, '--', color='gray', alpha=0.5,
                    label=f'理论 {base:.2f}°/√N', linewidth=1.5)
    
    ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.7, linewidth=2, label='目标 0.1°')
    ax.set_xlabel('窗口大小 N (帧数)', fontsize=12)
    ax.set_ylabel('Mean Rotation Error (°)', fontsize=12)
    ax.set_title('多帧平均: 误差 vs 窗口大小', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Chart 2: Per-axis error vs window size (best model only)
    ax = axes[1]
    if all_model_results:
        best_label = min(all_model_results.keys(),
                        key=lambda l: np.mean(all_model_results[l]['multiframe_results'].get(1, {}).get('rot_error', [999])))
        best_mf = all_model_results[best_label]['multiframe_results']
        Ns = sorted(N for N in best_mf if len(best_mf[N]['rot_error']) > 0)
        
        for axis_name, axis_key, color in [
            ('Roll (X)', 'roll_error', '#1abc9c'),
            ('Pitch (Y)', 'pitch_error', '#f39c12'),
            ('Yaw (Z)', 'yaw_error', '#8e44ad'),
            ('Total', 'rot_error', '#2196F3'),
        ]:
            vals = [np.mean(best_mf[N][axis_key]) for N in Ns]
            ax.plot(Ns, vals, 'o-', color=color, label=axis_name, linewidth=2, markersize=6)
        
        ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.7, linewidth=2, label='目标 0.1°')
        ax.set_xlabel('窗口大小 N (帧数)', fontsize=12)
        ax.set_ylabel('Mean Error (°)', fontsize=12)
        ax.set_title(f'分轴误差 vs 窗口大小 ({best_label})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    plt.tight_layout()
    path = os.path.join(charts_dir, 'error_vs_window_size.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Chart] {path}")
    
    report_lines.append("")
    report_lines.append("![Error vs Window Size](charts/error_vs_window_size.png)")
    report_lines.append("")
    
    # Chart 3: Improvement ratio vs window size
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (label, model_data) in enumerate(all_model_results.items()):
        mf = model_data['multiframe_results']
        if 1 not in mf or len(mf[1]['rot_error']) == 0:
            continue
        base = np.mean(mf[1]['rot_error'])
        Ns = sorted(N for N in mf if N > 1 and len(mf[N]['rot_error']) > 0)
        ratios = [base / max(np.mean(mf[N]['rot_error']), 0.001) for N in Ns]
        color = colors[idx % len(colors)]
        ax.plot(Ns, ratios, 'o-', color=color, label=label, linewidth=2, markersize=6)
    
    # Theoretical √N line
    theory_Ns = np.array([N for N in window_sizes if N > 1])
    ax.plot(theory_Ns, np.sqrt(theory_Ns), '--', color='gray', alpha=0.5,
            label='理论 √N', linewidth=2)
    
    ax.set_xlabel('窗口大小 N (帧数)', fontsize=12)
    ax.set_ylabel('改善倍数 (单帧误差 / N帧误差)', fontsize=12)
    ax.set_title('多帧融合改善倍数 vs 窗口大小', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    path = os.path.join(charts_dir, 'improvement_ratio.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Chart] {path}")
    
    report_lines.append("![Improvement Ratio](charts/improvement_ratio.png)")
    report_lines.append("")
    
    # Chart 4: Box plot for different window sizes (best model)
    if all_model_results:
        best_label = min(all_model_results.keys(),
                        key=lambda l: np.mean(all_model_results[l]['multiframe_results'].get(1, {}).get('rot_error', [999])))
        best_mf = all_model_results[best_label]['multiframe_results']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        box_data = []
        box_labels = []
        for N in window_sizes:
            if N in best_mf and len(best_mf[N]['rot_error']) > 0:
                box_data.append(best_mf[N]['rot_error'])
                box_labels.append(f"N={N}")
        
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.6)
            ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.7, linewidth=2, label='目标 0.1°')
            ax.set_ylabel('Rotation Error (°)', fontsize=12)
            ax.set_title(f'误差分布 vs 窗口大小 ({best_label})', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            path = os.path.join(charts_dir, 'error_distribution_boxplot.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  [Chart] {path}")
            
            report_lines.append("![Error Distribution](charts/error_distribution_boxplot.png)")
            report_lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-frame fusion evaluation")
    parser.add_argument("--eval_dirs", nargs='+', required=True,
                       help="Directories containing extrinsics_and_errors.txt")
    parser.add_argument("--window_sizes", type=str, default="1,5,10,20,50,100,200",
                       help="Comma-separated window sizes to test")
    parser.add_argument("--output_dir", type=str,
                       default="logs/evaluations/multiframe_analysis",
                       help="Output directory for report")
    args = parser.parse_args()
    
    window_sizes = [int(x) for x in args.window_sizes.split(',')]
    
    print("=" * 80)
    print("BEVCalib 多帧融合评估")
    print(f"  窗口大小: {window_sizes}")
    print(f"  模型数: {len(args.eval_dirs)}")
    print("=" * 80)
    
    all_model_results = {}
    
    for eval_dir in args.eval_dirs:
        label = os.path.basename(eval_dir)
        ext_path = os.path.join(eval_dir, "extrinsics_and_errors.txt")
        
        if not os.path.isfile(ext_path):
            print(f"\n[SKIP] {label}: extrinsics_and_errors.txt not found")
            continue
        
        print(f"\n>>> Processing: {label}")
        print(f"    File: {ext_path}")
        
        seq_gt_map, samples = parse_extrinsics_file(ext_path)
        seq_bounds = parse_sequence_boundaries(ext_path)
        
        print(f"    {len(seq_gt_map)} sequence GTs loaded, {len(samples)} samples, {len(seq_bounds)} sequences")
        
        # Validate: single-frame errors should match original report
        if seq_gt_map:
            errs_check = [compute_rotation_error(s['pred_T'], seq_gt_map[s['seq']])['rot_error']
                         for s in samples[:10] if s['seq'] in seq_gt_map]
            if errs_check:
                print(f"    Validation (first 10): {[f'{e:.3f}' for e in errs_check]}")
        
        mf_results = analyze_multiframe(seq_gt_map, samples, window_sizes, seq_bounds)
        
        for N in sorted(mf_results.keys()):
            errs = mf_results[N]
            if len(errs['rot_error']) > 0:
                print(f"    N={N:>4d}: Mean Rot = {np.mean(errs['rot_error']):.4f}° "
                      f"(P={np.mean(errs['pitch_error']):.4f}°) "
                      f"[{len(errs['rot_error'])} windows]")
        
        all_model_results[label] = {
            'seq_gt_map': seq_gt_map,
            'samples': samples,
            'seq_boundaries': seq_bounds,
            'multiframe_results': mf_results,
        }
    
    if not all_model_results:
        print("\n[ERROR] No valid evaluation results found!")
        return
    
    print(f"\n>>> Generating report...")
    report_path = generate_report(all_model_results, window_sizes, args.output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"分析完成!")
    print(f"  Report: {report_path}")
    print(f"  Charts: {os.path.join(args.output_dir, 'charts')}/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
