# V55-safe Refine Formal Eval Summary

**Date**: 2026-06-29  
**Object**: `model_small_5deg_v55_safe_refine`

## 1. Training outcome

From
`logs/all_training_data/model_small_5deg_v55_safe_refine/all_training_data_scratch/training_summary.md`:

- Training completed in `20` epochs
- Best train: `Epoch 2`, `Rot 0.50°`
- `ckpt_best_val`, `ckpt_best_medw`, `ckpt_best_dual` were all selected from `Epoch 1`
- Checkpoint-eval table on saved checkpoints shows the best scheduled checkpoint is `ckpt_15.pth`
  - `Epoch 15 checkpoint eval Rot = 0.11°`

Important implication:

- The deployment candidate chosen by early dual-gate proxy (`Epoch 1`) is not necessarily the best real candidate
- `ckpt_15.pth` must be treated as an additional formal evaluation candidate

## 2. Formal test_data_v2 result

### Candidate A: `ckpt_best_dual.pth` (Epoch 1)

From
`logs/evaluations/generalization_v55_safe_refine_only/v55-safe-refine-best-dual/temporal_aggregation.txt`:

- Per-frame: `Rot 1.3628°`
- Deploy `MEDW200 = 0.2684°`
- Deploy `MEDW400 = 0.2518°`
- MEDW400 axes:
  - `Roll 0.1517°`
  - `Pitch 0.1318°`
  - `Yaw 0.1023°`

### Candidate B: `ckpt_15.pth` (Epoch 15)

From
`logs/evaluations/generalization_v55_safe_refine_only/v55-safe-refine-ckpt15/temporal_aggregation.txt`:

- Per-frame: `Rot 1.4364°`
- Deploy `MEDW200 = 0.2577°`
- Deploy `MEDW400 = 0.2351°`
- MEDW400 axes:
  - `Roll 0.1526°`
  - `Pitch 0.1141°`
  - `Yaw 0.0978°`

## 3. Comparison

Against `V55-safe full`:

- `V55-safe full MEDW400 = 0.255°`
- `safe refine / best_dual = 0.2518°`
- `safe refine / ckpt15 = 0.2351°`

So refine did help, but the gain is moderate:

- `ckpt_best_dual` improves only slightly over `safe full`
- `ckpt_15` is the real winner for deploy metric
- even the best refine candidate still misses target `MEDW400 <= 0.18°`

## 4. Key observations

### Positive

- No V55/V55b-style catastrophic `68°~80°` failures were observed
- Formal per-frame errors stayed in the low-single-degree regime
- Deploy metric improved from `0.255°` to `0.2351°`
- Yaw reached `0.0978°`, already below the `0.10°` axis target

### Limitation

- Roll and pitch are still the bottleneck
  - `Roll 0.1526°`
  - `Pitch 0.1141°`
- Training still showed intermittent `NaN GUARD`, though far less destructive than V55/V55b
- Early dual-gate checkpoint selection again failed to identify the best deploy candidate

## 5. Recommendation

Recommended current deployment ordering:

1. `V55-safe refine / ckpt_15.pth`
2. `V55-safe refine / ckpt_best_dual.pth`
3. `V55-safe full / ckpt_best_dual.pth`

Recommended next move:

1. Promote `ckpt_15.pth` as the current best `safe-trunk` deploy candidate
2. Update future eval configs so `safe refine` is not represented only by `ckpt_best_dual`
3. Continue from the `safe-trunk` family, focusing on roll/pitch reduction rather than re-introducing aggressive DP-head routing
