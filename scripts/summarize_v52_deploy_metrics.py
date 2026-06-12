#!/usr/bin/env python3
"""从各 ckpt 的 generalization_diagnostics.json 汇总部署验收表。

双 ckpt 门控在 eval 主流程中未仿真；本脚本按 deploy_policy 映射：
  - MEDW / ZD(小扰动) → ckpt_medw (dual)
  - Fixed-Inject Recovery → ckpt_recovery (val)
并对比 acceptance 硬门槛。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime

import yaml


def _load_gdiag(eval_root: str, label: str, aliases: list | None = None) -> dict | None:
    candidates = [label] + list(aliases or [])
    for name in candidates:
        for fname in ("generalization_diagnostics_gate.json", "generalization_diagnostics.json"):
            pattern = os.path.join(eval_root, name, fname)
            if os.path.isfile(pattern):
                with open(pattern, "r") as f:
                    return json.load(f)
    return None


def _parse_report_medw(report_path: str, labels: list[str]) -> float | None:
    if not os.path.isfile(report_path):
        return None
    with open(report_path, "r") as f:
        text = f.read()
    for label in labels:
        pat = re.compile(
            rf"\|\s*{re.escape(label)}\s*\|\s*MEDW400\s*\|\s*([\d.]+)°"
        )
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def _genuine_rec(gdiag: dict) -> float | None:
    comp = gdiag.get("composite") or {}
    raw = comp.get("raw") or {}
    if "genuine_recovery_pct" in raw:
        return float(raw["genuine_recovery_pct"])
    fi = gdiag.get("fixed_inject") or {}
    if "genuine_recovery_pct" in fi:
        return float(fi["genuine_recovery_pct"])
    return None


def _zd_max(gdiag: dict) -> float | None:
    comp = gdiag.get("composite") or {}
    raw = comp.get("raw") or {}
    if "zero_drift_max_rpy" in raw:
        return float(raw["zero_drift_max_rpy"])
    zd = gdiag.get("zero_drift") or {}
    if "max_rpy" in zd:
        return float(zd["max_rpy"])
    r, p, y = zd.get("roll_mean"), zd.get("pitch_mean"), zd.get("yaw_mean")
    if r is not None and p is not None and y is not None:
        return max(float(r), float(p), float(y))
    return zd.get("rot_mean")


def _merge_gate_composite(dual_g: dict, val_g: dict) -> dict:
    """Scenario-routed composite: ZD from dual, inject metrics from val."""
    out = {}
    if dual_g:
        out["zero_drift"] = dual_g.get("zero_drift")
    if val_g:
        out["fixed_inject"] = val_g.get("fixed_inject")
        out["multi_magnitude"] = val_g.get("multi_magnitude")
        out["shortcut_per_axis"] = val_g.get("shortcut_per_axis")
    comp = {}
    if dual_g and val_g:
        zd = dual_g.get("zero_drift") or {}
        fi = val_g.get("fixed_inject") or {}
        inj = fi.get("inject") or {}
        comp["raw"] = {
            "zero_drift_max_rpy": zd.get("max_rpy"),
            "genuine_recovery_pct": inj.get("mean_recovery_pct"),
        }
    out["composite"] = comp
    return out


def _write_gate_acceptance(out_dir: str, policy: dict, acc: dict,
                           gate_gdiag: dict | None, medw_m: dict,
                           gate_label: str) -> str:
    medw_thresh = acc.get("medw400_max_deg", 0.18)
    zd_thresh = acc.get("zd_max_rpy_max_deg", 0.10)
    rec_thresh = acc.get("genuine_recovery_2deg_min_pct", 90.0)
    recv05_thresh = acc.get("multi_mag_0p5_recv_min_pct", 0.0)
    recv10_thresh = acc.get("multi_mag_1p0_recv_min_pct", 0.0)

    medw = medw_m.get("medw400")
    zd = _zd_max(gate_gdiag) if gate_gdiag else None
    rec = _genuine_rec(gate_gdiag) if gate_gdiag else None
    r05, rec05 = _multi_mag(gate_gdiag, "0.5") if gate_gdiag else (None, None)
    r10, rec10 = _multi_mag(gate_gdiag, "1.0") if gate_gdiag else (None, None)

    def pass_deg(v, thresh):
        return v is not None and v <= thresh

    def pass_pct(v, thresh):
        return v is not None and v >= thresh

    checks = [
        ("MEDW400", medw, medw_thresh, "°", pass_deg),
        ("ZD max(R,P,Y)", zd, zd_thresh, "°", pass_deg),
        ("GenuineRec@2°", rec, rec_thresh, "%", pass_pct),
        ("0.5° Recovery", rec05, recv05_thresh, "%", pass_pct),
        ("1.0° Recovery", rec10, recv10_thresh, "%", pass_pct),
    ]

    def fmt(v, suffix=""):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}{suffix}"
        return str(v)

    all_pass = all(fn(v, t) for _, v, t, _, fn in checks if v is not None and t is not None)
    any_pending = any(v is None for _, v, _, _, _ in checks)

    lines = [
        "# V52 双 ckpt 门控部署验收",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 门控策略",
        "",
        f"- init_rot ≤ {policy.get('gate_deg', 1.5)}° 或 n_frames≥{policy.get('medw_n_frames', 400)} → **primary (dual)**",
        f"- init_rot > {policy.get('gate_deg', 1.5)}° 或 inject 场景 → **recovery (val)**",
        f"- gdiag 路由: Zero-Drift → dual；Fixed-Inject / Multi-Mag → val",
        "",
        f"门控 eval 输出: `{gate_label}/generalization_diagnostics_gate.json`",
        "",
        "## 组合指标（门控后）",
        "",
        "| 指标 | 门槛 | 门控值 | 判定 |",
        "|------|------|-------:|------|",
    ]

    for name, v, thresh, unit, fn in checks:
        if v is None or thresh is None:
            status = "待评估"
        elif fn(v, thresh):
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        op = "≤" if unit == "°" else "≥"
        lines.append(f"| {name} | {op} {thresh}{unit} | {fmt(v, unit)} | {status} |")

    lines.extend([
        "",
        "## 总判定",
        "",
    ])
    if any_pending:
        lines.append("**待评估** — 门控 eval 尚未完成或缺少 MEDW 报告行。")
    elif all_pass:
        lines.append("**✅ 全部泛化指标达标** — 双 ckpt 门控方案可上车。")
    else:
        failed = [n for n, v, t, _, fn in checks if v is not None and t is not None and not fn(v, t)]
        lines.append(f"**❌ 未达标** — 未通过: {', '.join(failed)}。")
        lines.append("")
        lines.append("建议: 继续 v52d MGDA 单 ckpt 训练，或调整 recovery ckpt / 门控阈值。")

    lines.append("")
    out_path = os.path.join(out_dir, "DEPLOY_GATE_ACCEPTANCE.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


def _multi_mag(gdiag: dict, mag: str) -> tuple[float | None, float | None]:
    mm = gdiag.get("multi_magnitude") or {}
    entry = mm.get(mag) or mm.get(f"{mag}_deg")
    if not entry:
        return None, None
    return entry.get("residual"), entry.get("recovery_pct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_generalization_v52_deploy.yaml")
    ap.add_argument("--report-dir", default="logs/evaluations/generalization_v52_deploy")
    ap.add_argument("--gate-label", default="v52a-S1-v2-deploy-gate",
                    help="Subdir with gate eval gdiag JSON")
    ap.add_argument("--fallback-all-dir", default="logs/evaluations/generalization_v52_all",
                    help="Use dual+val gdiag from Phase3 all-eval if gate eval missing")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root = cfg["bevcalib_root"]
    out_dir = args.report_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir)

    policy = cfg.get("deploy_policy") or {}
    acc = policy.get("acceptance") or {}
    medw_label = (policy.get("ckpt_medw") or {}).get("label", "v52a-S1-v2-deploy-dual")
    rec_label = (policy.get("ckpt_recovery") or {}).get("label", "v52a-S1-v2-deploy-val")
    medw_aliases = (policy.get("ckpt_medw") or {}).get("eval_aliases") or []
    rec_aliases = (policy.get("ckpt_recovery") or {}).get("eval_aliases") or []

    report_path = os.path.join(out_dir, "GENERALIZATION_REPORT.md")

    def metrics_for(label: str, aliases: list) -> dict:
        g = _load_gdiag(out_dir, label, aliases)
        medw = _parse_report_medw(report_path, [label] + aliases)
        row = {
            "label": label,
            "medw400": medw,
            "zd_max_rpy": _zd_max(g) if g else None,
            "genuine_rec_2deg": _genuine_rec(g) if g else None,
        }
        if g:
            r05, rec05 = _multi_mag(g, "0.5")
            r10, rec10 = _multi_mag(g, "1.0")
            row["recv_0p5"] = rec05
            row["recv_1p0"] = rec10
        return row

    medw_m = metrics_for(medw_label, medw_aliases)
    rec_m = metrics_for(rec_label, rec_aliases)

    lines = [
        "# V52 部署验收摘要",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 评估管线说明",
        "",
        "| 能力 | 当前 eval 是否覆盖 | 说明 |",
        "|------|-------------------|------|",
        "| 单 ckpt MEDW 部署模拟 | ✅ | `evaluate_checkpoint.py` Section 4 uniform sample + median |",
        "| 单 ckpt gdiag ZD/Inject | ✅ | init=GT 无扰动 / 2° Fixed-Inject |",
        "| **双 ckpt 场景门控 gdiag** | ✅ | `evaluate_deploy_gate.py` ZD→dual, Inject→val |",
        "",
        "完整 per-frame init_err 门控 MEDW 主 eval 仍使用 primary ckpt；见 `run_v52_deploy_gate_eval.sh`。",
        "",
        "## 分路径指标（各 ckpt 单独评估，非门控合并）",
        "",
        "| 部署路径 | ckpt | MEDW400 | ZD max(R,P,Y) | GenuineRec@2° | 0.5°Recv | 1.0°Recv |",
        "|---------|------|--------:|--------------:|--------------:|---------:|---------:|",
    ]

    def fmt(v, suffix=""):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}{suffix}"
        return str(v)

    for m, path in [(medw_m, "MEDW/小扰动"), (rec_m, "Recovery/大扰动")]:
        lines.append(
            f"| {path} | `{m['label']}` | {fmt(m.get('medw400'), '°')} | "
            f"{fmt(m.get('zd_max_rpy'), '°')} | {fmt(m.get('genuine_rec_2deg'), '%')} | "
            f"{fmt(m.get('recv_0p5'), '%')} | {fmt(m.get('recv_1p0'), '%')} |"
        )

    lines.extend([
        "",
        "## 上车硬门槛对比",
        "",
        "| 指标 | 门槛 | MEDW路径(dual) | Recovery路径(val) | 判定 |",
        "|------|------|----------------|-------------------|------|",
    ])

    checks = [
        ("MEDW400", acc.get("medw400_max_deg", 0.18), medw_m.get("medw400"), rec_m.get("medw400"), "°", "dual"),
        ("ZD max(R,P,Y)", acc.get("zd_max_rpy_max_deg", 0.10), medw_m.get("zd_max_rpy"), rec_m.get("zd_max_rpy"), "°", "dual"),
        ("GenuineRec@2°", acc.get("genuine_recovery_2deg_min_pct", 90), medw_m.get("genuine_rec_2deg"), rec_m.get("genuine_rec_2deg"), "%", "val"),
    ]

    for name, thresh, v_dual, v_val, unit, primary in checks:
        v = v_dual if primary == "dual" else v_val
        if v is None or thresh is None:
            status = "待评估"
        elif unit == "°":
            status = "✅" if v <= thresh else "❌"
        else:
            status = "✅" if v >= thresh else "❌"
        lines.append(
            f"| {name} | ≤/≥ {thresh}{unit} | {fmt(v_dual, unit)} | {fmt(v_val, unit)} | {status} |"
        )

    lines.extend([
        "",
        "## 结论",
        "",
        "- **MEDW400 ~0.17°** 衡量的是「多帧聚合后相对 GT 的误差」，与 **ZD**（无扰动固有偏差）和 **GenuineRec**（2° 注入后矫正比例）是不同物理量。",
        "- 当前 V52 最佳：MEDW 路径可达 ~0.17°，但 GenuineRec 仅 ~59%（val），dual 仅 ~29%；**均未达 90% Recovery / 0.1° ZD 硬门槛**。",
        "- 双 ckpt 门控 gdiag 已实现；组合指标见 `DEPLOY_GATE_ACCEPTANCE.md`。",
        "",
    ])

    out_path = os.path.join(out_dir, "DEPLOY_ACCEPTANCE.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")

    # Gate acceptance: prefer live gate eval, else merge from v52_all dual+val
    gate_g = _load_gdiag(out_dir, args.gate_label)
    if gate_g is None:
        all_dir = args.fallback_all_dir
        if not os.path.isabs(all_dir):
            all_dir = os.path.join(root, all_dir)
        dual_g = _load_gdiag(all_dir, medw_label, medw_aliases)
        val_g = _load_gdiag(all_dir, rec_label, rec_aliases)
        if dual_g and val_g:
            gate_g = _merge_gate_composite(dual_g, val_g)
            if medw_m.get("medw400") is None:
                all_report = os.path.join(all_dir, "GENERALIZATION_REPORT.md")
                medw_m["medw400"] = _parse_report_medw(
                    all_report, [medw_label] + medw_aliases)

    gate_medw = dict(medw_m)
    if gate_medw.get("medw400") is None and gate_g is not None:
        gate_dir = os.path.join(out_dir, args.gate_label)
        ta_path = os.path.join(gate_dir, "deploy_simulation.json")
        if os.path.isfile(ta_path):
            with open(ta_path) as f:
                ds = json.load(f)
            entry = ds.get("400") or ds.get(400)
            if entry:
                gate_medw["medw400"] = entry.get("mean_rot") or entry.get("rot_mean")

    gate_path = _write_gate_acceptance(
        out_dir, policy, acc, gate_g, gate_medw, args.gate_label)
    print(f"Wrote {gate_path}")


if __name__ == "__main__":
    main()
