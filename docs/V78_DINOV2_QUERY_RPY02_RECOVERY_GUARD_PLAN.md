# V78 DINOv2-Query RPY02 Recovery Guard Plan

Date: 2026-07-03

## Conclusion From V77

The KPI is now strict `R/P/Y all < 0.2 deg`. Under that threshold, V77 is not promotable:

- Random full sequence-median: `0.3716 deg`, with Roll `0.2041 deg` and Pitch `0.2166 deg`.
- Random full per-frame `RPY全<0.2 deg`: `12.2%`.
- Fixed 2/2/2 adaptive mean recovery: `71.0%`, below the previous `72%+` floor.
- Fixed 2/2/2 adaptive `RPY全<0.2 deg`: `9.5%`.

The internal dual gate overestimated deploy transfer. V77 optimized the CF-BEV-R train-val tail but did not preserve the low-bias sequence aggregation behavior required by the stricter RPY02 metric.

## V78 Direction

V78 returns to the strongest low-bias prior: V29-G3 Query-BEV + DINOv2-small. The plan is deliberately conservative:

- Start from `v29_G3_partial_unfreeze_dinov2_quick/ckpt_best_val.pth`.
- Keep the DINOv2 backbone effectively frozen with `backbone_lr_scale=0.0`.
- Avoid `explicit_tinit` and direct correction-quaternion supervision because V66/V68 showed MEDW collapse.
- Add only gentle fixed-inject and Jacobian pressure.
- Increase zero-drift and small-residual sampling so random sequence-median R/P/Y stays below `0.2 deg`.

## Training

Config:

- `configs/v78_dinov2_query_rpy02_recovery_guard.yaml`

Smoke:

- 4-GPU DDP, default `CUDA_VISIBLE_DEVICES=0,1,2,5`
- 8 epochs
- 220 frames per sequence
- Pretrain: V29-G3 best-val

Full:

- Launch only if smoke external evaluation passes the RPY02 random sequence-median gate and does not further damage fixed recovery.
- 28 epochs
- 500 frames per sequence
- Pretrain: V78 smoke `ckpt_best_medw.pth`

## External Evaluation

Script:

- `run_v78_rpy02_eval.sh`

Default strict threshold:

- `RPY_THRESHOLD_DEG=0.2`

Primary smoke decision:

- Random full sequence-median all axes `< 0.2 deg`.
- Random full aggregate median should stay close to the V29-G3 prior, ideally `<= 0.12 deg`.
- Fixed 2/2/2 mean recovery must improve over V29-G3 and avoid dropping below the V74/V77 recovery floor.

If V78 cannot satisfy both random RPY02 and fixed recovery, the next step should not be another loss-weight sweep. The right next branch is a structural V79: native/cross-attention or residual adapter that makes `T_init` sensitivity explicit while preserving V29-style low-bias temporal noise.
