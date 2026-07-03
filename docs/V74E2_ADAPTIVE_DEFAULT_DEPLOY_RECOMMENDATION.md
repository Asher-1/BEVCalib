# V74e2 Adaptive-default Deploy Recommendation

Date: 2026-07-02

## Recommendation

- 当前推荐把 V74e2 + adaptive-default 作为部署侧正式评估基线。
- 推理策略：默认单轮；只有当序列 probe 中位修正幅度 >= 0.35 deg 时才触发第二轮。
- 默认门控：mode=seq_median_anchor, probe_frames=12, med_thr=0.35, lo_thr=0.0, std_thr=0.20.

## Evidence

- Random full, test_data_v2, exclude Seq07: per-frame 0.9009 deg, sequence-median 0.3465 deg, RPY all < 0.3 deg = 21.8 percent, trigger 0/11 seqs.
- Fixed 2/2/2 full adaptive formal report: residual 0.9551 deg, mean recovery 72.3 percent, median recovery 73.6 percent, trigger 7/11 seqs.
- Formal artifacts are archived under `logs/evaluations/adaptive_default_formal/v74e2_adaptive_default/`.

## Decision

- 先把 adaptive-default 固化为正式脚本入口，再用同一协议归档 random/full 和 fixed/full。
- 现阶段还没有足够证据要求立刻重构网络或切更强 DINOv2 backbone。
- 2026-07-03 已完成新脚本 full 正式归档，random/fixed 两条线均已落盘。

## Command

- bash run_adaptive_default_formal_eval.sh full
- bash run_adaptive_default_formal_eval.sh smoke

## Escalation

- 如果 random sequence-median 不能继续稳定在约 0.35 deg，或 fixed recovery 跌破 72 percent，再重新打开架构评审。
- 在升级 backbone 之前，先复查 gate 阈值、触发比例和序列分布漂移。
