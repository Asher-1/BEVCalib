#!/usr/bin/env python3
"""Diagnose signed residual bias from saved generalization eval outputs.

The script reads evaluate_checkpoint.py outputs (`all_T_pred_gt.npz`) and
summarizes whether remaining MEDW400 errors are global bias, hard-sequence
bias, or mostly frame noise. It is intentionally offline-only: no model
loading and no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_extrinsics import evaluate_sensor_extrinsic  # noqa: E402


AXES = ("roll", "pitch", "yaw")
DEFAULT_MODELS = [
    (
        "v60-ckpt15",
        "logs/evaluations/generalization_v60_seq_balance_pitch_guard_only/"
        "v60-ckpt15/all_T_pred_gt.npz",
    ),
    (
        "v62b-ckpt2",
        "logs/evaluations/generalization_v62b_stable_bias_lite_only/"
        "v62b-ckpt2/all_T_pred_gt.npz",
    ),
    (
        "v62b-ckpt6",
        "logs/evaluations/generalization_v62b_stable_bias_lite_only/"
        "v62b-ckpt6/all_T_pred_gt.npz",
    ),
    (
        "v62b-best-jacobian",
        "logs/evaluations/generalization_v62b_stable_bias_lite_only/"
        "v62b-best-jacobian/all_T_pred_gt.npz",
    ),
    (
        "v63-soup-v62b2-a05",
        "logs/evaluations/generalization_v63_soup_v60_v62b_only/"
        "v63-soup-v62b2-a05/all_T_pred_gt.npz",
    ),
    (
        "v63-soup-v62b2-a10",
        "logs/evaluations/generalization_v63_soup_v60_v62b_only/"
        "v63-soup-v62b2-a10/all_T_pred_gt.npz",
    ),
    (
        "v63-soup-v62bjac-a10",
        "logs/evaluations/generalization_v63_soup_v60_v62b_only/"
        "v63-soup-v62bjac-a10/all_T_pred_gt.npz",
    ),
]


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["all_T_pred"], data["all_T_gt"], data["sample_sequences"].astype(str)


def _pose_error_signed(T_pred: np.ndarray, T_gt: np.ndarray) -> Dict[str, float]:
    angle, axis_deg, pos_cm, axis_pos_cm = evaluate_sensor_extrinsic(T_pred, T_gt)
    return {
        "rot": float(angle),
        "roll": float(abs(axis_deg[0])),
        "pitch": float(abs(axis_deg[1])),
        "yaw": float(abs(axis_deg[2])),
        "roll_signed": float(axis_deg[0]),
        "pitch_signed": float(axis_deg[1]),
        "yaw_signed": float(axis_deg[2]),
        "trans_m": float(pos_cm / 100.0),
        "fwd_m": float(abs(axis_pos_cm[0]) / 100.0),
        "lat_m": float(abs(axis_pos_cm[1]) / 100.0),
        "ht_m": float(abs(axis_pos_cm[2]) / 100.0),
    }


def _median_rotation(Rs: np.ndarray) -> np.ndarray:
    aa = ScipyRot.from_matrix(Rs).as_rotvec()
    aa_med = np.median(aa, axis=0)
    return ScipyRot.from_rotvec(aa_med).as_matrix()


def _aggregate_uniform_median(
    T_pred: np.ndarray,
    T_gt: np.ndarray,
    seqs: np.ndarray,
    seq_id: str,
    window: int,
) -> Dict[str, float]:
    mask = seqs == seq_id
    seq_pred = T_pred[mask]
    seq_gt = T_gt[mask]
    n = len(seq_pred)
    ns = min(window, n)
    idx = np.linspace(0, n - 1, num=ns, dtype=int)

    R_avg = _median_rotation(seq_pred[idx, :3, :3])
    t_avg = np.mean(seq_pred[idx, :3, 3], axis=0)

    T_agg = np.eye(4, dtype=np.float64)
    T_agg[:3, :3] = R_avg
    T_agg[:3, 3] = t_avg

    err = _pose_error_signed(T_agg, seq_gt[0])
    err["n_frames"] = int(n)
    err["n_sampled"] = int(ns)
    return err


def _perframe_stats(T_pred: np.ndarray, T_gt: np.ndarray, seqs: np.ndarray, seq_id: str) -> Dict[str, object]:
    mask = seqs == seq_id
    signed = []
    rots = []
    for pred, gt in zip(T_pred[mask], T_gt[mask]):
        err = _pose_error_signed(pred, gt)
        signed.append([err["roll_signed"], err["pitch_signed"], err["yaw_signed"]])
        rots.append(err["rot"])
    arr = np.asarray(signed, dtype=np.float64)
    rot_arr = np.asarray(rots, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        "rot_mean": float(np.mean(rot_arr)),
        "rot_median": float(np.median(rot_arr)),
        "signed_mean": {ax: float(np.mean(arr[:, i])) for i, ax in enumerate(AXES)},
        "signed_median": {ax: float(np.median(arr[:, i])) for i, ax in enumerate(AXES)},
        "signed_std": {ax: float(np.std(arr[:, i])) for i, ax in enumerate(AXES)},
        "signed_p10": {ax: float(np.percentile(arr[:, i], 10)) for i, ax in enumerate(AXES)},
        "signed_p90": {ax: float(np.percentile(arr[:, i], 90)) for i, ax in enumerate(AXES)},
        "sign_positive_ratio": {ax: float(np.mean(arr[:, i] > 0)) for i, ax in enumerate(AXES)},
    }


def _axis_vec(row: Dict[str, float]) -> np.ndarray:
    return np.asarray([row[f"{ax}_signed"] for ax in AXES], dtype=np.float64)


def _macro_from_vectors(seq_vectors: Dict[str, np.ndarray], correction: Dict[str, np.ndarray]) -> Dict[str, float]:
    corrected = []
    for seq, vec in seq_vectors.items():
        corrected.append(vec - correction.get(seq, np.zeros(3)))
    mat = np.asarray(corrected, dtype=np.float64)
    return {
        "rot": float(np.mean(np.linalg.norm(mat, axis=1))),
        "roll": float(np.mean(np.abs(mat[:, 0]))),
        "pitch": float(np.mean(np.abs(mat[:, 1]))),
        "yaw": float(np.mean(np.abs(mat[:, 2]))),
    }


def _bias_probe(seq_vectors: Dict[str, np.ndarray], hard_seqs: set[str]) -> Dict[str, object]:
    seq_ids = sorted(seq_vectors)
    mat = np.asarray([seq_vectors[s] for s in seq_ids], dtype=np.float64)
    zero = {s: np.zeros(3) for s in seq_ids}

    normal_ids = [s for s in seq_ids if s not in hard_seqs]
    hard_ids = [s for s in seq_ids if s in hard_seqs]
    global_bias = np.median(mat, axis=0)
    normal_bias = np.median([seq_vectors[s] for s in normal_ids], axis=0) if normal_ids else global_bias
    hard_bias = np.median([seq_vectors[s] for s in hard_ids], axis=0) if hard_ids else global_bias

    global_corr = {s: global_bias for s in seq_ids}
    normal_corr = {s: normal_bias for s in seq_ids}
    domain_corr = {s: (hard_bias if s in hard_seqs else normal_bias) for s in seq_ids}
    hard_only_oracle = {s: (seq_vectors[s] if s in hard_seqs else np.zeros(3)) for s in seq_ids}

    loo_corr = {}
    for s in seq_ids:
        others = [seq_vectors[o] for o in seq_ids if o != s]
        loo_corr[s] = np.median(others, axis=0) if others else np.zeros(3)

    return {
        "bias_deg": {
            "global_median": {ax: float(global_bias[i]) for i, ax in enumerate(AXES)},
            "normal_median": {ax: float(normal_bias[i]) for i, ax in enumerate(AXES)},
            "hard_median": {ax: float(hard_bias[i]) for i, ax in enumerate(AXES)},
        },
        "macro": {
            "raw": _macro_from_vectors(seq_vectors, zero),
            "global_constant": _macro_from_vectors(seq_vectors, global_corr),
            "normal_constant": _macro_from_vectors(seq_vectors, normal_corr),
            "leave_one_seq_out_constant": _macro_from_vectors(seq_vectors, loo_corr),
            "domain_oracle_hard_vs_normal": _macro_from_vectors(seq_vectors, domain_corr),
            "hard_seq_exact_oracle": _macro_from_vectors(seq_vectors, hard_only_oracle),
        },
    }


def _sign(v: float) -> str:
    if abs(v) < 1e-4:
        return "0"
    return "+" if v > 0 else "-"


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _fmt_vec(vec: Dict[str, float] | np.ndarray) -> str:
    if isinstance(vec, dict):
        vals = [vec[ax] for ax in AXES]
    else:
        vals = list(vec)
    return "/".join(f"{v:+.3f}" for v in vals)


def _parse_model_specs(specs: Iterable[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--model must be label=path, got: {spec}")
        label, path = spec.split("=", 1)
        parsed.append((label.strip(), _repo_path(path.strip())))
    return parsed


def _discover_eval_dirs(eval_dirs: Iterable[str]) -> List[Tuple[str, Path]]:
    models = []
    for eval_dir in eval_dirs:
        root = _repo_path(eval_dir)
        for npz_path in sorted(root.glob("*/all_T_pred_gt.npz")):
            models.append((npz_path.parent.name, npz_path))
    return models


def analyze_model(label: str, npz_path: Path, window: int, hard_seqs: set[str]) -> Dict[str, object]:
    T_pred, T_gt, seqs = _load_npz(npz_path)
    seq_ids = sorted(set(seqs.tolist()))

    per_seq = {}
    seq_vectors = {}
    perframe = {}
    for seq_id in seq_ids:
        agg = _aggregate_uniform_median(T_pred, T_gt, seqs, seq_id, window)
        per_seq[seq_id] = agg
        seq_vectors[seq_id] = _axis_vec(agg)
        perframe[seq_id] = _perframe_stats(T_pred, T_gt, seqs, seq_id)

    agg_mat = np.asarray([seq_vectors[s] for s in seq_ids], dtype=np.float64)
    mean_abs_axes = np.mean(np.abs(agg_mat), axis=0)
    mean_rot = float(np.mean(np.linalg.norm(agg_mat, axis=1)))

    hard_rows = {s: per_seq[s] for s in seq_ids if s in hard_seqs}
    dominant = sorted(
        ((s, per_seq[s]["rot"], max(AXES, key=lambda ax: abs(per_seq[s][f"{ax}_signed"]))) for s in seq_ids),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "label": label,
        "npz_path": str(npz_path),
        "window": window,
        "seq_ids": seq_ids,
        "summary": {
            "mean_medw_rot": mean_rot,
            "mean_roll": float(mean_abs_axes[0]),
            "mean_pitch": float(mean_abs_axes[1]),
            "mean_yaw": float(mean_abs_axes[2]),
            "num_sequences": len(seq_ids),
            "num_frames": int(len(seqs)),
        },
        "per_seq": per_seq,
        "perframe": perframe,
        "hard_seq": hard_rows,
        "dominant_sequences": [
            {"seq": s, "rot": float(rot), "dominant_axis": axis} for s, rot, axis in dominant[:5]
        ],
        "bias_probe": _bias_probe(seq_vectors, hard_seqs),
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_markdown(path: Path, results: List[Dict[str, object]], hard_seqs: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# V64 Residual Bias Diagnostic")
    lines.append("")
    lines.append("This is an offline diagnostic from saved `all_T_pred_gt.npz` files. Signed axes are LiDAR-frame roll/pitch/yaw in degrees.")
    lines.append("")

    lines.append("## Macro Ranking")
    lines.append("")
    lines.append("| Candidate | MEDW Rot | Roll | Pitch | Yaw | Frames |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in sorted(results, key=lambda x: x["summary"]["mean_medw_rot"]):
        s = r["summary"]
        lines.append(
            f"| {r['label']} | {_fmt(s['mean_medw_rot'])} | {_fmt(s['mean_roll'])} | "
            f"{_fmt(s['mean_pitch'])} | {_fmt(s['mean_yaw'])} | {s['num_frames']} |"
        )
    lines.append("")

    lines.append("## Hard-Sequence Signed MEDW400")
    lines.append("")
    lines.append("| Candidate | Seq | Rot | Signed R/P/Y | Sign | Dominant |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for r in results:
        for seq in hard_seqs:
            row = r["per_seq"].get(seq)
            if not row:
                continue
            vec = _axis_vec(row)
            dominant = AXES[int(np.argmax(np.abs(vec)))]
            sign = "".join(_sign(v) for v in vec)
            lines.append(
                f"| {r['label']} | {seq} | {_fmt(row['rot'])} | {_fmt_vec(vec)} | {sign} | {dominant} |"
            )
    lines.append("")

    lines.append("## Per-Frame Bias Check On Hard Sequences")
    lines.append("")
    lines.append("| Candidate | Seq | Per-frame Rot Mean | Median Signed R/P/Y | Std R/P/Y | PosRatio R/P/Y |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for r in results:
        for seq in hard_seqs:
            pf = r["perframe"].get(seq)
            if not pf:
                continue
            lines.append(
                f"| {r['label']} | {seq} | {_fmt(pf['rot_mean'])} | "
                f"{_fmt_vec(pf['signed_median'])} | {_fmt_vec(pf['signed_std'])} | "
                f"{pf['sign_positive_ratio']['roll']:.2f}/"
                f"{pf['sign_positive_ratio']['pitch']:.2f}/"
                f"{pf['sign_positive_ratio']['yaw']:.2f} |"
            )
    lines.append("")

    lines.append("## Bias-Correction Probe")
    lines.append("")
    lines.append("These rows subtract signed residual vectors in small-angle space as an oracle diagnostic. They are not directly deployable unless the bias source can be estimated without GT.")
    lines.append("")
    lines.append("| Candidate | Raw | Global Constant | Normal Constant | Leave-One-Seq-Out | Hard/Normal Oracle | Hard-Exact Oracle |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        macro = r["bias_probe"]["macro"]
        lines.append(
            f"| {r['label']} | {_fmt(macro['raw']['rot'])} | "
            f"{_fmt(macro['global_constant']['rot'])} | "
            f"{_fmt(macro['normal_constant']['rot'])} | "
            f"{_fmt(macro['leave_one_seq_out_constant']['rot'])} | "
            f"{_fmt(macro['domain_oracle_hard_vs_normal']['rot'])} | "
            f"{_fmt(macro['hard_seq_exact_oracle']['rot'])} |"
        )
    lines.append("")

    lines.append("## Bias Vectors")
    lines.append("")
    lines.append("| Candidate | Global Median R/P/Y | Normal Median R/P/Y | Hard Median R/P/Y |")
    lines.append("| --- | --- | --- | --- |")
    for r in results:
        b = r["bias_probe"]["bias_deg"]
        lines.append(
            f"| {r['label']} | {_fmt_vec(b['global_median'])} | "
            f"{_fmt_vec(b['normal_median'])} | {_fmt_vec(b['hard_median'])} |"
        )
    lines.append("")

    lines.append("## Readout")
    lines.append("")
    best = min(results, key=lambda x: x["summary"]["mean_medw_rot"])
    lines.append(f"- Best among diagnosed candidates remains `{best['label']}` with MEDW{best['window']} Rot={best['summary']['mean_medw_rot']:.4f}°.")
    lines.append("- If `Global Constant` or `Leave-One-Seq-Out` is not better than raw, the residual is not a simple global bias.")
    lines.append("- If `Hard/Normal Oracle` helps while global correction does not, the next useful path is domain-aware residual estimation rather than another low-LR continuation.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help="Model spec: label=path/to/all_T_pred_gt.npz")
    parser.add_argument("--eval-dir", action="append", default=[], help="Discover */all_T_pred_gt.npz under this eval dir")
    parser.add_argument("--output-dir", default="logs/analysis/v64_residual_bias", help="Directory for JSON/Markdown outputs")
    parser.add_argument("--window", type=int, default=400, help="Uniform MEDW window size")
    parser.add_argument("--hard-seqs", default="02,03,10", help="Comma-separated hard sequence IDs")
    args = parser.parse_args()

    hard_seqs = {s.strip() for s in args.hard_seqs.split(",") if s.strip()}
    models: List[Tuple[str, Path]] = []
    if args.model:
        models.extend(_parse_model_specs(args.model))
    if args.eval_dir:
        models.extend(_discover_eval_dirs(args.eval_dir))
    if not models:
        models = [(label, _repo_path(path)) for label, path in DEFAULT_MODELS]

    seen = set()
    filtered = []
    for label, path in models:
        if (label, str(path)) in seen:
            continue
        seen.add((label, str(path)))
        if not path.exists():
            print(f"[WARN] skip missing: {label} -> {path}")
            continue
        filtered.append((label, path))

    if not filtered:
        raise SystemExit("No valid all_T_pred_gt.npz files found.")

    results = []
    for label, path in filtered:
        print(f"[Analyze] {label}: {path}")
        results.append(analyze_model(label, path, args.window, hard_seqs))

    out_dir = _repo_path(args.output_dir)
    payload = {
        "window": args.window,
        "hard_seqs": sorted(hard_seqs),
        "models": results,
    }
    _write_json(out_dir / "residual_bias_diagnostic.json", payload)
    _write_markdown(out_dir / "RESIDUAL_BIAS_DIAGNOSTIC.md", results, sorted(hard_seqs))
    print(f"[Done] {out_dir / 'RESIDUAL_BIAS_DIAGNOSTIC.md'}")
    print(f"[Done] {out_dir / 'residual_bias_diagnostic.json'}")


if __name__ == "__main__":
    main()
