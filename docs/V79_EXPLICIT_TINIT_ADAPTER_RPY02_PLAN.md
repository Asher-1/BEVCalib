# V79 Explicit-Tinit Adapter RPY02 Plan

Date: 2026-07-03

## Why V79

V78 proved that the V29/V78 DINOv2-query prior can satisfy strict random sequence-level RPY02:

- Random sequence median: `0.1431 deg`
- R/P/Y: `0.1080 / 0.0733 / 0.0360 deg`

But fixed 2/2/2 remains unsolved:

- Fixed residual: `3.4126 deg`
- Mean recovery: `0.9%`
- `RPY全<0.2 deg`: `0.0%`

So this is no longer a backbone-capacity issue. The model needs a direct `T_init` sensitivity path.

## Design

V79 is a minimal structural adapter:

- Start from V78 `ckpt_best_medw.pth`.
- Enable `explicit_tinit=1`.
- Freeze backbone, BEV, fusion, and transformer.
- Train only `tinit_encoder` and `rotation_pred`.
- Use moderate inject/Jacobian pressure to learn controlled fixed-bias correction.
- Use strong zero-drift and small-perturb sampling to protect random RPY02.

This intentionally avoids the V66 failure mode: high LR, broad unfrozen modules, and overly strong recovery loss that destroyed MEDW.

## Files

- Config: `configs/v79_explicit_tinit_adapter_rpy02.yaml`
- Eval script: `run_v79_rpy02_eval.sh`

## Smoke Gate

Promote only if both hold:

- Random full sequence-median all R/P/Y `< 0.2 deg`.
- Fixed 2/2/2 mean recovery materially improves over V78, with a target path back toward `72%+`.

If V79 still cannot lift fixed recovery, the next change should be V80 native/cross-attention or another architecture where `T_init` affects the matching process before pooling, not just the final head.
