# V59 Safe Long Continue Formal Eval Summary

**Date**: 2026-06-30  
**Object**: `model_small_5deg_v59_safe_long_continue`

## 1. Training outcome

From
`logs/all_training_data/model_small_5deg_v59_safe_long_continue/all_training_data_scratch/training_summary.md`:

- Training completed in `40` epochs
- Best val: `Epoch 36`, `Rot 0.13°`
- Best MEDW200 proxy: `Epoch 31`, `max(R,P,Y)=0.0253°`
- Best dual gate: `Epoch 31`, `Jacobian=0.998`
- Best scheduled eval checkpoint: `ckpt_40.pth`, validation `Rot 0.10°`

This answered the epoch question directly: longer training did improve the
internal validation view, but it did not translate into a better formal deploy
metric.

## 2. Formal test_data_v2 result

Four representative checkpoints were evaluated on the same formal protocol used
for V56/V58: `test_data_v2`, `exclude_seqs=07`, `400` frames per sequence,
`MEDW400` deployment aggregation.

### Candidate A: `ckpt_best_dual.pth`

- Per-frame: `Rot 1.2592°`
- Deploy `MEDW200 = 0.2498°`
- Deploy `MEDW400 = 0.2178°`
- MEDW400 axes:
  - `Roll 0.1231°`
  - `Pitch 0.1248°`
  - `Yaw 0.0881°`

### Candidate B: `ckpt_best_medw.pth`

`ckpt_best_medw.pth` resolves to the same epoch as `ckpt_best_dual.pth`, so it
matches Candidate A:

- Deploy `MEDW400 = 0.2178°`
- Axes: `R 0.1231° / P 0.1248° / Y 0.0881°`

### Candidate C: `ckpt_best_val.pth`

- Per-frame: `Rot 1.2525°`
- Deploy `MEDW200 = 0.2470°`
- Deploy `MEDW400 = 0.2179°`
- MEDW400 axes:
  - `Roll 0.1228°`
  - `Pitch 0.1284°`
  - `Yaw 0.0826°`

### Candidate D: `ckpt_40.pth`

- Per-frame: `Rot 1.2557°`
- Deploy `MEDW200 = 0.2499°`
- Deploy `MEDW400 = 0.2198°`
- MEDW400 axes:
  - `Roll 0.1228°`
  - `Pitch 0.1283°`
  - `Yaw 0.0868°`

## 3. Comparison against current best

Current strongest formal reference:

- `V56 / ckpt_best_dual`: `MEDW400 = 0.2125°`
- `V59 / ckpt_best_dual`: `MEDW400 = 0.2178°`

So V59 is a small regression:

- regression vs V56: about `+0.0053°`
- target remains `MEDW400 <= 0.18°`
- remaining gap from V59: about `0.0378°`

## 4. Interpretation

Positive:

- No V55/V55b-style catastrophic failures
- Yaw remains stable and below `0.10°`
- Roll is improved relative to V56, from `0.1353°` to about `0.123°`

Negative:

- Pitch drifted upward from `0.1006°` in V56 to about `0.125°`
- Longer continuation did not improve formal deploy performance
- The last checkpoint did not beat the dual-gate checkpoint

The key failure mode is not lack of epochs. It is roll/pitch tradeoff drift:
continuing from V56 can reduce roll, but the formal deployment metric loses more
through pitch than it gains through roll.

## 5. Recommendation

Recommended next move:

1. Keep `V56 / ckpt_best_dual.pth` as the deployment anchor.
2. Do not continue from V59.
3. Start V60 from `V56 / ckpt_best_dual.pth` with a shorter, lower-risk polish:
   sequence-balanced sampling, lower backbone LR, less large perturbation,
   stronger pitch guard, and reduced inject pressure.
