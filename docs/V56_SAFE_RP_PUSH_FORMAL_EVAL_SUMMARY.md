# V56 Safe RP Push Formal Eval Summary

**Date**: 2026-06-29  
**Object**: `model_small_5deg_v56_safe_rp_push`

## 1. Training outcome

From
`logs/all_training_data/model_small_5deg_v56_safe_rp_push/all_training_data_scratch/training_summary.md`:

- Training completed in `25` epochs
- Best train: `Epoch 21`, `Rot 0.34°`
- Best MEDW200 proxy: `Epoch 21`, `max(R,P,Y)=0.0247°`
- Best dual gate: `Epoch 21`, `Jacobian=0.993`
- Best scheduled checkpoint by internal eval table: `ckpt_10.pth`

## 2. Formal test_data_v2 result

Two representative checkpoints were formally evaluated:

### Candidate A: `ckpt_best_dual.pth` (Epoch 21)

From
`logs/evaluations/generalization_v56_safe_rp_push_only/v56-safe-rp-push-best-dual/temporal_aggregation.txt`:

- Per-frame: `Rot 1.3893°`
- Deploy `MEDW200 = 0.2557°`
- Deploy `MEDW400 = 0.2125°`
- MEDW400 axes:
  - `Roll 0.1353°`
  - `Pitch 0.1006°`
  - `Yaw 0.0886°`

### Candidate B: `ckpt_10.pth`

From
`logs/evaluations/generalization_v56_safe_rp_push_only/v56-safe-rp-push-ckpt10/temporal_aggregation.txt`:

- Per-frame: `Rot 1.3732°`
- Deploy `MEDW200 = 0.2546°`
- Deploy `MEDW400 = 0.2141°`
- MEDW400 axes:
  - `Roll 0.1427°`
  - `Pitch 0.1033°`
  - `Yaw 0.0826°`

## 3. Comparison against previous safe trunk

Reference points:

- `V55-safe full / best-dual`: `MEDW400 = 0.255°`
- `V55-safe refine / ckpt_15`: `MEDW400 = 0.2351°`
- `V56 / best-dual`: `MEDW400 = 0.2125°`

So V56 delivered a real improvement:

- `0.255° -> 0.2351° -> 0.2125°`
- relative to `safe refine / ckpt_15`, V56 improved by about `0.0226°`

## 4. What improved

### Positive

- No V55/V55b-style catastrophic `68°~80°` failures
- Yaw improved further below target:
  - `0.0886°`
- Pitch is almost at target:
  - `0.1006°`
- Roll also improved:
  - from `0.1526°` in `safe refine / ckpt_15`
  - to `0.1353°` in `V56 / best-dual`

## 5. What is still missing

Deployment target is still not met:

- target: `MEDW400 <= 0.18°`
- current best: `0.2125°`
- remaining gap: about `0.0325°`

Main bottleneck is now clearly `roll`:

- `Roll 0.1353°` is the largest remaining axis
- `Pitch 0.1006°` is already very close
- `Yaw 0.0886°` is already under the `0.10°` axis target

## 6. Recommendation

Current deployment ordering:

1. `V56 / ckpt_best_dual.pth`
2. `V56 / ckpt_10.pth`
3. `V55-safe refine / ckpt_15.pth`

Recommended next move:

1. Keep iterating on the `safe trunk` family
2. Bias the next variant even more toward roll reduction
3. Avoid re-introducing aggressive DP-head routing
4. Use `V56 / best-dual` as the new pretrain anchor for the next run
