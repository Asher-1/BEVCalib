# V64b Full and V29-G3 Formal Eval Summary

Date: 2026-07-01

## Status

V64b full completed and was evaluated on the current formal deployment protocol:

- Test data: `test_data_v2`
- Excluded sequence: `07`
- Sampling: 400 frames per sequence, 11 sequences, 4400 frames total
- Perturbation: `angle_range=5.0`, `trans_range=0.0`
- Metric: deployable temporal aggregation `MEDW400`

The same protocol was then used to re-evaluate the historical V29-G3 DINOv2 Query-BEV candidate.

## V64b Full Result

V64b full is stable but does not improve deployment generalization.

| Candidate | MEDW400 Rot | Roll | Pitch | Yaw | Seq02 | Seq03 | Seq10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v64b-ckpt2 | 0.213° | 0.124° | 0.127° | 0.086° | 0.413° | 0.692° | 0.216° |
| v64b-ckpt4 | 0.218° | 0.126° | 0.135° | 0.086° | 0.418° | 0.695° | 0.236° |
| v64b-best-dual | 0.224° | 0.130° | 0.138° | 0.088° | 0.429° | 0.704° | 0.221° |

Reference:

| Candidate | MEDW400 Rot | Roll | Pitch | Yaw | Seq02 | Seq03 | Seq10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V60 ckpt15 | 0.205° | 0.121° | 0.120° | 0.082° | 0.406° | 0.690° | 0.225° |

Decision:

- Do not promote V64b full.
- Do not continue low-LR V60/V64 family training as the main path.
- V64b confirms the same hard-sequence residual pattern: Seq02 pitch bias, Seq03 roll/pitch/yaw positive bias, Seq10 smaller roll residual.

Root cause:

- The full run improves internal proxy metrics but shifts low-frequency aggregation bias.
- Early V64b `ckpt2` is best, while later/best-dual checkpoints regress MEDW400.
- This matches the previous V62b/V63 finding: continuation can reduce single-frame noise while worsening deploy aggregation bias.

## V29-G3 Current-Protocol Re-Eval

Historical candidate:

- Model: `v29-G3-partial-unfreeze-dinov2`
- Architecture: Query-BEV + DINOv2-small
- Checkpoint: `ckpt_best_val.pth`
- Config: `configs/eval_generalization_v29_g3_current_protocol.yaml`
- Report: `logs/evaluations/generalization_v29_g3_current_protocol/GENERALIZATION_REPORT.md`

Current-protocol result:

| Candidate | MEDW100 | MEDW200 | MEDW400 | Roll | Pitch | Yaw | Seq02 | Seq03 | Seq10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v29-g3-best-val | 0.176° | 0.137° | 0.090° | 0.048° | 0.049° | 0.033° | 0.098° | 0.102° | 0.111° |
| v29-g3-ckpt400 | 0.189° | 0.139° | 0.091° | 0.037° | 0.064° | 0.034° | 0.134° | 0.132° | 0.106° |

This is a decisive improvement over the V60/V64 line:

- V29-G3 best-val MEDW400: 0.090°
- V60 ckpt15 MEDW400: 0.205°
- V64b best full candidate MEDW400: 0.213°

Interpretation:

- V29-G3 has high single-frame noise (`Mean Rot ~= 2.30°`), but the error is much more temporally unbiased.
- Robust median aggregation cancels the noise very effectively.
- The hard-sequence ceiling seen in CF-BEV-R is not present: Seq03 drops from about 0.69° to about 0.10°.

## V29-G3 Acceptance Diagnostic

A fast acceptance diagnostic was run with `gdiag_max_batches=64`:

- Output: `logs/evaluations/generalization_v29_g3_acceptance_fast/v29-g3-best-val/generalization_diagnostics.json`
- Zero-drift: `rot_mean=0.0259°`, `max_rpy=0.0228°`
- Fixed 2° RPY inject residual: `3.4610°`
- Fixed 2° recovery: `-0.5%`
- Multi-magnitude recovery:
  - `0.5°`: `-2.0%`
  - `1.0°`: `-1.0%`
  - `2.0°`: `-0.5%`
- Shortcut risk: `HIGH`

This changes the interpretation materially:

- V29-G3 is excellent when the evaluation perturbations are independent and symmetric around GT.
- It does not meaningfully recover a fixed injected calibration error.
- Therefore the strong MEDW400 result is mostly a temporally unbiased random-perturbation effect, not proof that the model can correct a fixed wrong initial extrinsic on vehicle.

## Current Recommendation

Do not directly promote `v29-g3-best-val` as the final car-side calibration model.

Use it as an important diagnostic / architecture clue:

- Query-BEV + DINOv2-small produces much lower low-frequency hard-sequence bias than CF-BEV-R/Swin.
- The architecture is worth reviving, but the training objective must force real fixed-bias recovery.
- Any deployment claim must pass both MEDW aggregation and fixed-inject recovery diagnostics.

Keep V60 `ckpt15` only as the best CF-BEV-R/Swin reference, not as the overall deployment candidate.

## Remaining Todos

1. Treat V29-G3 as the new architecture direction, not as a deploy-ready model.
2. Check runtime and deployment compatibility of Query-BEV + DINOv2-small versus CF-BEV-R/Swin.
3. Start `V65 Query-BEV/DINOv2 recovery rebaseline`: preserve V29's low hard-sequence bias while adding fixed-inject / zero-drift / recovery supervision.
4. Keep V60 `ckpt15` as the safest existing recovery-aware CF-BEV-R reference until V65 passes both MEDW and recovery diagnostics.
