# V55b Full Formal Eval Summary

**Date**: 2026-06-26  
**Object**: `model_small_5deg_v55b_route_tight`

## 1. Training outcome

From `logs/all_training_data/model_small_5deg_v55b_route_tight/all_training_data_scratch/training_summary.md`:

- Best train: `Epoch 46`, `Rot 0.68°`
- Best val (full-val internal): `Epoch 1`, `Rot 15.16°`
- Best MEDW200 proxy: `Epoch 1`, `max(R,P,Y)=0.1654°`
- Best dual gate: `Epoch 1`, `Jacobian=0.985`

Important detail:

- `ckpt_best_val.pth`, `ckpt_best_medw.pth`, `ckpt_best_dual.pth` are all selected from `Epoch 1`
- So V55b deployment candidates are effectively the same early-training checkpoint

## 2. Formal test_data_v2 result

Formal deploy metric from
`logs/evaluations/generalization_v55/v55b-best-dual/temporal_aggregation.txt`
and `v55b-best-medw/temporal_aggregation.txt`:

- Per-frame: `Rot 20.9605°`
- Deploy `MEDW200 = 0.5053°`
- Deploy `MEDW400 = 0.5035°`
- MEDW400 axes:
  - `Roll 0.2925°`
  - `Pitch 0.2521°`
  - `Yaw 0.2214°`

Observed failure pattern:

- Large numbers of catastrophic samples in formal eval
- Typical bad cases are around `68° ~ 70°`
- This is the same failure family seen in the failed V55 main line

## 3. Comparison against V55-safe

Current stable baseline (`V55-safe`) formal deploy metric:

- `MEDW400 = 0.255°`
- axes = `0.152 / 0.133 / 0.103°`

So V55b is:

- much worse than `V55-safe`
- slightly better than failed `V55-main` on deploy MEDW
- still far above target `MEDW400 <= 0.18°`

## 4. Root-cause update

V55b did improve some training-side symptoms:

- later-epoch train rot reached `0.68°`
- train `route_w_mean` dropped from early `0.9827` toward roughly `0.95~0.96`

But it did **not** fix the deployment failure:

- cross-domain catastrophic route outputs still exist
- early proxy dual-gate success at `Epoch 1` was not predictive of real generalization
- frequent `NaN GUARD` events remained throughout training

Inference:

- simply tightening `route / perturb / GIN` on the V55 DP-head branch is not enough
- the current DP-head routing family is still unsafe as the primary deployment line

## 5. Recommendation

Recommended next move:

1. Keep `V55-safe` as the stable deployment baseline
2. Do not promote `V55b` to deployment candidate
3. If continuing to iterate, start from the `V55-safe` trunk rather than from the V55/V55b DP-head trunk

Suggested direction for next variant:

- retain `Partial GIN 128ch` stable trunk
- add only limited recovery capability
- avoid letting early `dual gate` proxy decide the deployment candidate by itself
- further reduce catastrophic-route exposure before increasing recovery aggressiveness
