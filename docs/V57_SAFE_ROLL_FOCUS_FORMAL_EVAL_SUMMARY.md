# V57 Safe Roll Focus Formal Eval Summary

**Date**: 2026-06-29  
**Object**: `model_small_5deg_v57_safe_roll_focus`

## 1. Training outcome

From
`logs/all_training_data/model_small_5deg_v57_safe_roll_focus/all_training_data_scratch/training_summary.md`:

- Training completed in `22` epochs
- Best train: `Epoch 2`, `Rot 0.27°`
- Best val proxy: `Epoch 11`, `Rot 0.13°`
- Best MEDW200 proxy: `Epoch 21`, `max(R,P,Y)=0.0261°`
- Best dual gate: `Epoch 21`, `Jacobian=0.993`

This means:

- proxy deploy indicators kept improving near the end
- but the proxy improvement still needed formal test verification

## 2. Formal test_data_v2 result

Two representative checkpoints were evaluated:

### Candidate A: `ckpt_best_dual.pth`

From
`logs/evaluations/generalization_v57_safe_roll_focus_only/v57-best-dual/temporal_aggregation.txt`:

- Per-frame: `Rot 1.3062°`
- BEST deployable: `MEDW200 = 0.2415°`
- `MEDW400 = 0.2457°`
- BEST axes at MEDW200:
  - `Roll 0.1131°`
  - `Pitch 0.1482°`
  - `Yaw 0.1116°`

### Candidate B: `ckpt_20.pth`

From
`logs/evaluations/generalization_v57_safe_roll_focus_only/v57-ckpt20/temporal_aggregation.txt`:

- Per-frame: `Rot 1.3124°`
- BEST deployable: `MEDW200 = 0.2469°`
- `MEDW400 = 0.2515°`
- BEST axes at MEDW200:
  - `Roll 0.1118°`
  - `Pitch 0.1473°`
  - `Yaw 0.1240°`

## 3. Comparison against V56

Reference:

- `V56 / best-dual`: `MEDW400 = 0.2125°`
- `V57 / best-dual`: `MEDW400 = 0.2457°`

So V57 is a regression on formal deployment metric.

## 4. What changed

### Improvement

- Roll was reduced:
  - `V56 roll 0.1353°`
  - `V57 roll 0.1131°`

### Regression

- Pitch worsened:
  - `V56 pitch 0.1006°`
  - `V57 pitch 0.1482°`
- Yaw also worsened relative to V56
- Overall deploy metric became worse

Inference:

- V57 over-optimized roll
- the roll gain was not worth the pitch / yaw tradeoff

## 5. Recommendation

Recommended action:

1. Do not continue training V57 directly
2. Roll back to the `V56` family as the better anchor
3. Start a more balanced successor instead of further increasing roll pressure
