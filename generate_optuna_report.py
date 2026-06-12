#!/usr/bin/env python3
"""Generate Optuna HPO analysis report with charts for BEVCalib v49 search."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna

BEVCALIB_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = (
    BEVCALIB_ROOT
    / "logs/all_training_data/optuna_trials/optuna_search/bevcalib_parallel_v49/optuna_study.db"
)
DEFAULT_OUT = (
    BEVCALIB_ROOT
    / "logs/all_training_data/optuna_trials/optuna_search/bevcalib_parallel_v49/reports"
)
FULL_TRAIN_LOG = (
    BEVCALIB_ROOT
    / "logs/all_training_data/model_small_5deg_v49_optuna_best_t32_S1_full/train.log"
)


def _load_study(db_path: str, study_name: str) -> optuna.Study:
    storage = f"sqlite:///{db_path}"
    return optuna.load_study(study_name=study_name, storage=storage)


def _trial_rows(study: optuna.Study) -> list[dict]:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        if not np.isfinite(t.value):
            continue
        row = {
            "number": t.number,
            "value": t.value,
            "roll": t.user_attrs.get("best_roll"),
            "pitch": t.user_attrs.get("best_pitch"),
            "yaw": t.user_attrs.get("best_yaw"),
            "epoch": t.user_attrs.get("best_epoch"),
            **t.params,
        }
        rows.append(row)
    return rows


def _save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path.name


def _chart_score_histogram(rows, out_dir: Path) -> str:
    scores = [r["value"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    best = min(scores)
    ax.axvline(best, color="#C44E52", linestyle="--", linewidth=2, label=f"Best={best:.4f}°")
    ax.axvline(0.1, color="#55A868", linestyle=":", linewidth=1.5, label="Target 0.1°")
    ax.set_xlabel("MEDW max(R,P,Y) (deg)")
    ax.set_ylabel("Trial count")
    ax.set_title("Optuna Trial Score Distribution (42 completed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, out_dir, "trial_score_hist.png")


def _chart_param_importance(study, out_dir: Path) -> str:
    imp = optuna.importance.get_param_importances(study)
    params = list(imp.keys())
    values = [imp[p] for p in params]
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(params))
    ax.barh(y_pos, values, color="#4C72B0", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (fANOVA)")
    ax.set_title("Hyperparameter Importance")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    return _save_fig(fig, out_dir, "param_importance.png")


def _chart_top_trials(rows, out_dir: Path, top_n: int = 10) -> str:
    top = sorted(rows, key=lambda r: r["value"])[:top_n]
    labels = [f"#{r['number']}" for r in top]
    scores = [r["value"] for r in top]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#C44E52" if r["number"] == 32 else "#4C72B0" for r in top]
    ax.bar(labels, scores, color=colors, alpha=0.85)
    ax.axhline(0.1, color="#55A868", linestyle=":", label="Target 0.1°")
    ax.set_ylabel("MEDW max(R,P,Y) (deg)")
    ax.set_title(f"Top-{top_n} Trials (Trial#32 highlighted)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (s, r) in enumerate(zip(scores, top)):
        ax.text(i, s + 0.002, f"{s:.4f}", ha="center", fontsize=8)
    return _save_fig(fig, out_dir, "top10_trials.png")


def _chart_scatter(rows, param: str, out_dir: Path) -> str | None:
    xs, ys = [], []
    for r in rows:
        if param not in r:
            continue
        xs.append(r[param])
        ys.append(r["value"])
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, alpha=0.7, c="#4C72B0", edgecolors="white", s=60)
    best_idx = int(np.argmin(ys))
    ax.scatter([xs[best_idx]], [ys[best_idx]], c="#C44E52", s=120, zorder=5,
               label=f"Best Trial#{rows[best_idx]['number']}")
    ax.set_xlabel(param)
    ax.set_ylabel("MEDW max(R,P,Y) (deg)")
    ax.set_title(f"Score vs {param}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    safe_name = param.replace("/", "_")
    return _save_fig(fig, out_dir, f"score_vs_{safe_name}.png")


def _chart_rpy_breakdown(rows, out_dir: Path) -> str:
    top5 = sorted(rows, key=lambda r: r["value"])[:5]
    labels = [f"#{r['number']}" for r in top5]
    rolls = [r.get("roll", 0) or 0 for r in top5]
    pitches = [r.get("pitch", 0) or 0 for r in top5]
    yaws = [r.get("yaw", 0) or 0 for r in top5]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, rolls, w, label="Roll", color="#4C72B0")
    ax.bar(x, pitches, w, label="Pitch", color="#55A868")
    ax.bar(x + w, yaws, w, label="Yaw", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (deg)")
    ax.set_title("R/P/Y Breakdown — Top-5 Trials")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save_fig(fig, out_dir, "rpy_breakdown_top5.png")


def _chart_gin_ablation(rows, out_dir: Path) -> str:
    gin0 = [r["value"] for r in rows if r.get("gin_channels", 0) == 0]
    gin_pos = [r["value"] for r in rows if r.get("gin_channels", 0) > 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [gin0, gin_pos]
    bp = ax.boxplot(data, labels=["GIN=0", "GIN>0"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#55A868")
    bp["boxes"][1].set_facecolor("#C44E52")
    ax.set_ylabel("MEDW max(R,P,Y) (deg)")
    ax.set_title(f"GIN Ablation (GIN=0: n={len(gin0)}, GIN>0: n={len(gin_pos)})")
    ax.grid(True, axis="y", alpha=0.3)
    med0 = np.median(gin0) if gin0 else 0
    med1 = np.median(gin_pos) if gin_pos else 0
    ax.text(1, max(gin0 + gin_pos) * 0.95, f"median: {med0:.4f} vs {med1:.4f}", ha="center")
    return _save_fig(fig, out_dir, "gin_ablation.png")


def _chart_hpo_vs_full(out_dir: Path) -> str:
    stages = ["HPO Trial#32\n(40ep, 200fr)", "Full Train\n(150ep, 500fr)"]
    hpo = 0.0522
    full = 0.0334
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(stages, [hpo, full], color=["#4C72B0", "#C44E52"], alpha=0.85)
    ax.set_ylabel("MEDW max(R,P,Y) (deg)")
    ax.set_title("HPO Quick Train vs Full Training (Trial#32)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, [hpo, full]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.4f}°", ha="center", fontsize=11)
    improve = (hpo - full) / hpo * 100
    ax.text(0.5, max(hpo, full) * 0.7, f"↓{improve:.1f}%", ha="center",
            fontsize=14, color="#55A868", transform=ax.transAxes)
    return _save_fig(fig, out_dir, "hpo_vs_full_training.png")


def _chart_consistency_ablation(rows, out_dir: Path) -> str:
    buckets = {}
    for r in rows:
        w = r.get("consistency_loss_weight", 0)
        key = f"{w:.1f}"
        buckets.setdefault(key, []).append(r["value"])
    keys = sorted(buckets.keys(), key=float)
    meds = [np.median(buckets[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(keys, meds, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("consistency_loss_weight")
    ax.set_ylabel("Median MEDW max(R,P,Y) (deg)")
    ax.set_title("Consistency Loss Weight vs Trial Performance")
    ax.grid(True, axis="y", alpha=0.3)
    return _save_fig(fig, out_dir, "consistency_ablation.png")


def _load_full_train_medw_curve() -> list[tuple[int, float]] | None:
    if not FULL_TRAIN_LOG.exists():
        return None
    curve = []
    import re
    pat = re.compile(
        r"Epoch \[(\d+)/\d+\].*MEDW.*max\(R,P,Y\):\s*([\d.]+)"
    )
    with open(FULL_TRAIN_LOG) as f:
        for line in f:
            m = pat.search(line)
            if m:
                curve.append((int(m.group(1)), float(m.group(2))))
    return curve if curve else None


def _chart_full_train_curve(out_dir: Path) -> str | None:
    curve = _load_full_train_medw_curve()
    if not curve:
        return None
    epochs, vals = zip(*curve)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, vals, "o-", color="#4C72B0", markersize=4, alpha=0.8)
    best_idx = int(np.argmin(vals))
    ax.scatter([epochs[best_idx]], [vals[best_idx]], c="#C44E52", s=100, zorder=5,
               label=f"Best ep{epochs[best_idx]}={vals[best_idx]:.4f}°")
    ax.axvline(46, color="#DD8452", linestyle=":", alpha=0.7, label="LR restart (T0=40+6)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MEDW max(R,P,Y) (deg)")
    ax.set_title("Full Training MEDW Curve (Trial#32, 150 epoch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, out_dir, "full_train_medw_curve.png")


def _write_report(study, rows, charts: dict[str, str], out_dir: Path):
    best = study.best_trial
    top5 = sorted(rows, key=lambda r: r["value"])[:5]
    gin0_scores = [r["value"] for r in rows if r.get("gin_channels", 0) == 0]
    gin_pos_scores = [r["value"] for r in rows if r.get("gin_channels", 0) > 0]
    imp = optuna.importance.get_param_importances(study)
    target_met = sum(1 for r in rows if r["value"] < 0.1)
    n_finite = len(rows)
    n_complete = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    )

    lines = [
        "BEVCalib V49 Optuna 超参搜索分析报告",
        "=" * 72,
        "",
        "一、搜索概况",
        "",
        f"Study: {study.study_name}",
        f"总 Trial 数: {len(study.trials)} (完成 {n_complete}, 有效分数 {n_finite})",
        f"目标: 最小化 val MEDW max(R,P,Y)",
        f"满足 <0.1° 目标的 Trial: {target_met}/{n_finite}",
        "",
        "二、最佳 Trial #32 选择依据",
        "",
        f"HPO 阶段最佳 MEDW max(R,P,Y) = {study.best_value:.4f}°",
        f"  Roll  = {best.user_attrs.get('best_roll', 'N/A'):.4f}°",
        f"  Pitch = {best.user_attrs.get('best_pitch', 'N/A'):.4f}°",
        f"  Yaw   = {best.user_attrs.get('best_yaw', 'N/A'):.4f}°",
        f"  Best epoch (HPO): {best.user_attrs.get('best_epoch', 'N/A')}",
        "",
        "选择 Trial#32 的核心理由:",
        "1. 在 42 个完成 Trial 中排名第一，显著优于第二名 Trial#31 (0.0596°)",
        "2. 三轴均衡: Pitch 仅 0.034° (最低), Roll/Yaw ~0.052°, 无单轴瓶颈",
        "3. gin_channels=0 确认 GIN 无益 (GIN=0 中位数 {:.4f}° vs GIN>0 中位数 {:.4f}°)".format(
            np.median(gin0_scores), np.median(gin_pos_scores)
        ),
        "4. cf_n_groups=128 在 GIN=0 子集中最优 (256 组未带来泛化收益)",
        "5. pitch_vertical_bands=5 提供更强 Pitch 分辨率 (重要性排名第3)",
        "",
        "三、关键超参 (Trial#32)",
        "",
        "| 参数 | 值 | 解读 |",
        "| --- | --- | --- |",
        f"| cf_n_groups | {best.params.get('cf_n_groups')} | 点云 FPS 分组数, 128 兼顾速度与精度 |",
        f"| pitch_vertical_bands | {best.params.get('pitch_vertical_bands')} | Pitch 分支垂直条带, 5 条带增强俯仰感知 |",
        f"| lr | {best.params.get('lr', 0):.2e} | 较低学习率, 配合 cosine restart 稳定收敛 |",
        f"| head_dropout | {best.params.get('head_dropout')} | 低 dropout 保留回归头表达能力 |",
        f"| axis_weights | 1.5,{best.params.get('pitch_weight', 5.5)},1.5 | Pitch 权重 5.5 强化俯仰校准 |",
        f"| cosine_T0 | {best.params.get('cosine_T0')} | LR 重启周期, 重要性最高 (36%) |",
        f"| warmup_epochs | {best.params.get('warmup_epochs')} | 6 epoch 线性预热 |",
        f"| consistency_loss_weight | {best.params.get('consistency_loss_weight')} | 一致性损失权重 0.6 |",
        f"| multi_scale_perturb | 0.5:{best.params.get('ms_small_ratio')},1.0:{best.params.get('ms_mid_ratio')},2.0:{best.params.get('ms_large_ratio')} | 小扰动比例更高 (20%) |",
        f"| gin_channels | {best.params.get('gin_channels')} | 关闭 GIN (与 v48a 结论一致) |",
        "",
        "四、参数重要性排名",
        "",
        "| 排名 | 参数 | 重要性 |",
        "| ---: | --- | ---: |",
    ]
    for i, (p, v) in enumerate(imp.items(), 1):
        lines.append(f"| {i} | {p} | {v:.3f} |")

    lines += [
        "",
        f"![参数重要性]({charts.get('param_importance', '')})",
        "",
        "五、Top-5 Trial 对比",
        "",
        "| 排名 | Trial | MEDW max | Roll | Pitch | Yaw | gin | lr | bands |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(top5, 1):
        lines.append(
            f"| {i} | #{r['number']} | {r['value']:.4f} | "
            f"{r.get('roll', 0):.4f} | {r.get('pitch', 0):.4f} | {r.get('yaw', 0):.4f} | "
            f"{r.get('gin_channels', 0)} | {r.get('lr', 0):.2e} | {r.get('pitch_vertical_bands', '?')} |"
        )

    lines += [
        "",
        f"![Top-10 Trials]({charts.get('top10_trials', '')})",
        f"![Score Distribution]({charts.get('trial_score_hist', '')})",
        "",
        "六、完整训练验证 (150 epoch)",
        "",
        "使用 Trial#32 参数进行 150 epoch 全量数据训练:",
        "",
        "| 阶段 | Epoch | MEDW max(R,P,Y) | 说明 |",
        "| --- | ---: | ---: | --- |",
        "| HPO 快速训练 | 31 | 0.0522° | 40 epoch, max_frames=200 |",
        "| 完整训练 best-dual | 121 | 0.0334° | 150 epoch, max_frames=500 |",
        "| 完整训练 best-val | 41 | val_rot=0.14° | 泛化 checkpoint |",
        "",
        "完整训练相比 HPO 提升 36.0%, 证明 Optuna 搜索空间有效。",
        "cosine_T0=40 在 epoch 46 触发 LR 重启, epoch 51 出现短暂回退 (0.147°),",
        "Cycle2 后期在 epoch 121 达到最优。",
        "",
        f"![HPO vs Full]({charts.get('hpo_vs_full', '')})",
    ]
    if charts.get("full_train_medw_curve"):
        lines.append(f"![Full Train MEDW Curve]({charts['full_train_medw_curve']})")

    lines += [
        "",
        "七、消融分析",
        "",
        f"![GIN Ablation]({charts.get('gin_ablation', '')})",
        f"![Consistency Ablation]({charts.get('consistency_ablation', '')})",
        f"![RPY Breakdown]({charts.get('rpy_breakdown', '')})",
        "",
        "GIN 消融: gin_channels=0 的 Trial 中位数明显优于 gin>0, 与 v48 系列结论一致。",
        "consistency_loss_weight: 0.6 (Trial#32) 在 HPO 中最优, 但 v49b 消融显示",
        "关闭 consistency 后 MEDW400 部署性能更优 (0.196° vs 0.303°), 需在 v50 中验证。",
        "",
        "八、泛化评估交叉验证 (test_data_v2)",
        "",
        "| 模型 | 单帧 Mean Rot | MEDW400 | GS_medw |",
        "| --- | ---: | ---: | ---: |",
        "| v45c-best-val (基线) | 0.752° | 0.394° | — |",
        "| v49-optuna-t32-best-val | 1.111° (#3) | 0.335° | 0.5709 |",
        "| v49-optuna-t32-best-dual | 1.229° (#5) | 0.330° | 0.6548 |",
        "| v49b-no-cons-best-dual | 1.399° | 0.272° | — |",
        "",
        "结论: Optuna 参数在 V49 系列中单帧泛化最优, 但 MEDW400 部署不如 v49b。",
        "训练 MEDW 极低 (0.033°) 未完全转化为 test 泛化, 存在 val→test 域偏移。",
        "",
        "九、v50 配置建议",
        "",
        "基于本报告 + 泛化评估, v50 应融合:",
        "1. Optuna T32: lr=4e-5, cosine_T0=40, head_dropout=0.05, pitch_bands=5, axis_weights=1.5,5.5,1.5",
        "2. v49b 优势: consistency_loss_weight=0 (MEDW400 最优)",
        "3. v45c 优势: cf_n_groups=256 作为消融对比 (单帧泛化 0.752°)",
        "",
        "推荐实验:",
        "- v50a: T32 超参 + 关闭 consistency (主候选)",
        "- v50b: T32 超参 + cf_n_groups=256 + 关闭 consistency",
        "- v50c: T32 超参 + 条件一致性 threshold=3.0 (折中)",
        "",
        "配置文件: configs/v50_cf_bev_r.yaml",
        "",
        "十、图表索引",
        "",
    ]
    for name, fname in sorted(charts.items()):
        lines.append(f"- {name}: {fname}")

    report_path = out_dir / "OPTUNA_T32_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate Optuna analysis report")
    parser.add_argument("--study_name", default="bevcalib_parallel_v49")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    study = _load_study(args.db, args.study_name)
    rows = _trial_rows(study)
    print(f"Loaded {len(rows)} completed trials")

    charts = {}
    charts["trial_score_hist"] = _chart_score_histogram(rows, out_dir)
    charts["param_importance"] = _chart_param_importance(study, out_dir)
    charts["top10_trials"] = _chart_top_trials(rows, out_dir)
    charts["rpy_breakdown"] = _chart_rpy_breakdown(rows, out_dir)
    charts["gin_ablation"] = _chart_gin_ablation(rows, out_dir)
    charts["consistency_ablation"] = _chart_consistency_ablation(rows, out_dir)
    charts["hpo_vs_full"] = _chart_hpo_vs_full(out_dir)

    for param in ["lr", "cosine_T0", "pitch_vertical_bands", "cf_n_groups",
                  "head_dropout", "consistency_loss_weight"]:
        fname = _chart_scatter(rows, param, out_dir)
        if fname:
            charts[f"scatter_{param}"] = fname

    fname = _chart_full_train_curve(out_dir)
    if fname:
        charts["full_train_medw_curve"] = fname

    _write_report(study, rows, charts, out_dir)
    print(f"Charts saved to: {out_dir}")


if __name__ == "__main__":
    main()
