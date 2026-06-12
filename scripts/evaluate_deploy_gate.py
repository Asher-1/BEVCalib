#!/usr/bin/env python3
"""双 ckpt 门控泛化评估入口。

加载 MEDW(primary) + Recovery(secondary) 两个 checkpoint，在同一次 eval 中：
  - Zero-Drift → primary (dual)
  - Fixed-Inject / Multi-Mag / Shortcut → recovery (val)

MEDW400 部署模拟仍使用 primary ckpt 的主 eval 流程。

Usage:
  python scripts/evaluate_deploy_gate.py --config configs/eval_generalization_v52_deploy.yaml
  python scripts/evaluate_deploy_gate.py --config ... --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_ckpt(root: str, models_dir: str, entry: dict) -> str:
    base = os.path.join(root, models_dir, entry["dir_name"],
                        "all_training_data_scratch/checkpoint")
    path = os.path.join(base, entry["ckpt"])
    if not os.path.isfile(path):
        for fb in entry.get("ckpt_fallback") or []:
            alt = os.path.join(base, fb)
            if os.path.isfile(alt):
                path = alt
                break
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_generalization_v52_deploy.yaml")
    ap.add_argument("--output-label", default="v52a-S1-v2-deploy-gate",
                    help="Subdir under output_dir for gate eval results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT, cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    bev_root = cfg["bevcalib_root"]
    models_dir = cfg["models_dir"]
    test_data = cfg["test_data"]
    out_base = cfg["output_dir"]
    if not os.path.isabs(out_base):
        out_base = os.path.join(bev_root, out_base)
    eval_dir = os.path.join(out_base, args.output_label)

    policy = cfg.get("deploy_policy") or {}
    medw = policy.get("ckpt_medw") or {}
    rec = policy.get("ckpt_recovery") or {}
    if not medw or not rec:
        print("[ERROR] deploy_policy.ckpt_medw / ckpt_recovery required", file=sys.stderr)
        sys.exit(1)

    ckpt_primary = _resolve_ckpt(bev_root, models_dir, medw)
    ckpt_recovery = _resolve_ckpt(bev_root, models_dir, rec)
    gate_deg = float(policy.get("gate_deg", 1.5))

    for label, path in [("primary", ckpt_primary), ("recovery", ckpt_recovery)]:
        if not os.path.isfile(path):
            print(f"[ERROR] {label} ckpt not found: {path}", file=sys.stderr)
            sys.exit(1)

    ep = cfg.get("eval_params") or {}
    m_medw = next((m for m in cfg["models"] if m["label"] == medw.get("label")), medw)

    env = os.environ.copy()
    env.pop("USE_DRCV_BACKEND", None)
    env["BEV_ZBOUND_STEP"] = str(m_medw.get("bev_zbound_step", "4.0"))
    env["HF_HUB_OFFLINE"] = "1"
    if m_medw.get("use_drcv"):
        env["USE_DRCV_BACKEND"] = "1"
    else:
        env["USE_DRCV_BACKEND"] = "0"

    cmd = [
        sys.executable, "evaluate_checkpoint.py",
        "--ckpt_path", ckpt_primary,
        "--ckpt_path_recovery", ckpt_recovery,
        "--deploy_gate",
        "--deploy_gate_deg", str(gate_deg),
        "--dataset_root", test_data,
        "--output_dir", eval_dir,
        "--angle_range_deg", str(ep.get("angle_range", 5.0)),
        "--trans_range", str(ep.get("trans_range", 0.0)),
        "--batch_size", str(ep.get("batch_size", 16)),
        "--use_full_dataset",
        "--max_batches", "0",
        "--generalization_diag",
    ]
    if "rotation_only" in m_medw:
        cmd.extend(["--rotation_only", str(m_medw["rotation_only"])])
    if m_medw.get("use_drcv"):
        cmd.append("--use_drcv")
    if ep.get("exclude_seqs"):
        cmd.extend(["--exclude_seqs", ep["exclude_seqs"]])
    if ep.get("eval_max_frames_per_seq"):
        cmd.extend(["--eval_max_frames_per_seq", str(ep["eval_max_frames_per_seq"])])
    if m_medw.get("pitch_vertical_bands") is not None:
        cmd.extend(["--pitch_vertical_bands", str(m_medw["pitch_vertical_bands"])])

    print("=" * 72)
    print("V52 Deploy Gate Evaluation")
    print(f"  Primary (MEDW/ZD):  {ckpt_primary}")
    print(f"  Recovery (Inject):  {ckpt_recovery}")
    print(f"  gate_deg={gate_deg}°  output={eval_dir}")
    print("=" * 72)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    os.makedirs(eval_dir, exist_ok=True)
    os.chdir(bev_root)
    rc = subprocess.call(cmd, cwd=bev_root, env=env)
    if rc != 0:
        sys.exit(rc)

    print(f"\nGate eval complete → {eval_dir}/generalization_diagnostics_gate.json")


if __name__ == "__main__":
    main()
