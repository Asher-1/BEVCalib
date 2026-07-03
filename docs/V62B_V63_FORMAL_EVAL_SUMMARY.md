# V62b / V63 Formal Eval Summary

Date: 2026-06-30

## Current Best

V60 `ckpt_15.pth` remains the best deploy-generalization candidate.

| Candidate | MEDW400 Rot | Roll | Pitch | Yaw | Per-frame Mean Rot |
| --- | ---: | ---: | ---: | ---: | ---: |
| V60 ckpt15 | 0.205° | 0.121° | 0.120° | 0.082° | 1.266° |

## V62b Result

V62b was a safer low-LR refine from V60 ckpt15. It improved internal validation and per-frame formal error, but did not improve the deployment MEDW400 metric.

| Candidate | MEDW400 Rot | Roll | Pitch | Yaw | Per-frame Mean Rot |
| --- | ---: | ---: | ---: | ---: | ---: |
| v62b-ckpt2 | 0.213° | 0.123° | 0.127° | 0.090° | 1.249° |
| v62b-ckpt6 | 0.216° | 0.127° | 0.132° | 0.085° | 1.229° |
| v62b-best-jacobian | 0.216° | 0.123° | 0.130° | 0.089° | 1.231° |
| v62b-best-dual | 0.220° | 0.130° | 0.130° | 0.092° | 1.228° |

Interpretation:
- Per-frame error improved from V60's 1.266° to about 1.227-1.249°.
- MEDW400 worsened from 0.205° to 0.213-0.220°.
- Seq03 and Seq02 remained the dominant blockers.
- This indicates the continuation reduced random single-frame noise but shifted the low-frequency aggregation bias.

## V63 Soup Result

V63 averaged V60 ckpt15 with V62b checkpoints to test whether we could keep V60's aggregation bias while borrowing V62b's lower per-frame noise.

| Candidate | MEDW400 Rot | Roll | Pitch | Yaw | Per-frame Mean Rot |
| --- | ---: | ---: | ---: | ---: | ---: |
| V60 ckpt15 baseline | 0.205° | 0.121° | 0.120° | 0.082° | 1.266° |
| soup V62b2 alpha 0.05 | 0.206° | 0.121° | 0.121° | 0.082° | 1.266° |
| soup V62b2 alpha 0.10 | 0.206° | 0.122° | 0.121° | 0.082° | 1.265° |
| soup V62bjac alpha 0.10 | 0.207° | 0.122° | 0.122° | 0.082° | 1.263° |
| soup V62b6 alpha 0.10 | 0.207° | 0.123° | 0.122° | 0.081° | 1.263° |
| soup V62b2 alpha 0.20 | 0.208° | 0.123° | 0.122° | 0.083° | 1.264° |
| soup V62b2 alpha 0.35 | 0.209° | 0.124° | 0.123° | 0.083° | 1.262° |

Interpretation:
- Soup improves per-frame mean slightly, but every soup regresses MEDW400.
- V60's aggregation bias is already close to the best achievable point among these related weights.
- More continuation or simple model averaging is unlikely to reach 0.18°.

## Recommendation

Stop the V60-continuation family as the main path. Keep V60 ckpt15 as the current deploy candidate.

Next useful work should target the remaining hard-sequence bias directly:
- Run a Seq03/Seq02 residual-sign diagnostic on V60 ckpt15 to determine whether the error is systematic roll/pitch/yaw bias or scene-dependent spread.
- Compare train/test metadata and calibration GT for Seq03/Seq02; the same hard cases persisted across V60/V61/V62b/V63.
- If more training is attempted, use a new signal rather than another low-LR continuation: sequence/domain-aware residual correction, better calibration-output supervision, or a deployment-time bias estimator trained on train-side held-out sequences.

Guardrail:
- Do not replace V60 ckpt15 with V61/V62b/V63 for deployment; all are worse on MEDW400.
