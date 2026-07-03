# V58 Safe Balance Push Formal Eval Summary

**Date**: 2026-06-30  
**Object**: `model_small_5deg_v58_safe_balance_push`

## 1. Training outcome

From
`logs/all_training_data/model_small_5deg_v58_safe_balance_push/all_training_data_scratch/training_summary.md`:

- Training completed in `30` epochs
- Best val checkpoint id: `ckpt_best_val.pth` from `Epoch 26`
- Best dual-gate checkpoint stayed at `Epoch 1`
- Best MEDW200 proxy also stayed at `Epoch 1`

This is an important signal:

- extending training did improve the internal val metric
- but deploy-oriented proxy did not keep improving after the earliest phase

## 2. Formal test_data_v2 result

### Candidate A: `ckpt_best_dual.pth`

- BEST deployable: `MEDW200 = 0.2404°`
- `MEDW400 = 0.2491°`
- axes at BEST:
  - `Roll 0.1204°`
  - `Pitch 0.1247°`
  - `Yaw 0.1162°`

### Candidate B: `ckpt_best_val.pth`

- BEST deployable: `MEDW200 = 0.2536°`
- `MEDW400 = 0.2620°`
- axes at BEST:
  - `Roll 0.1203°`
  - `Pitch 0.1565°`
  - `Yaw 0.1206°`

### Candidate C: `ckpt_20.pth`

- BEST deployable: `MEDW200 = 0.2536°`
- `MEDW400 = 0.2662°`
- axes at BEST:
  - `Roll 0.1233°`
  - `Pitch 0.1465°`
  - `Yaw 0.1227°`

## 3. Comparison against V56

Reference best formal candidate:

- `V56 / ckpt_best_dual`: `BEST = 0.2125°`

So V58 is still a regression:

- best V58 candidate: `0.2404°`
- regression vs V56: about `+0.0279°`

## 4. Interpretation

### Positive

- no catastrophic `68°~80°` failures
- roll remained controlled near `0.12°`

### Negative

- pitch and yaw did not stay as low as in `V56`
- longer training alone did not improve formal deploy metric
- the best formal result still comes from the early dual-gate candidate, not the later val-best checkpoint

## 5. Recommendation

Recommended action:

1. Keep `V56 / best-dual` as the current strongest deploy anchor
2. Do not continue `V58` directly
3. Start a low-risk long continuation from `V56 / best-dual` instead of from `V58`
