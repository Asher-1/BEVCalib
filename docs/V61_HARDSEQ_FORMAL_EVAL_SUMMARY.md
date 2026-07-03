# V61 Hard-Seq Refine Formal Eval Summary

Date: 2026-06-30

## Result

V61 full completed normally and all 7 formal candidates were evaluated on `test_data_v2`.

| Candidate | MEDW400 Rot | Roll | Pitch | Yaw | Per-frame Mean Rot |
| --- | ---: | ---: | ---: | ---: | ---: |
| v61-best-jacobian | 0.210° | 0.125° | 0.137° | 0.069° | 1.253° |
| v61-ckpt12 | 0.210° | 0.122° | 0.138° | 0.073° | 1.247° |
| v61-best-dual | 0.217° | 0.122° | 0.139° | 0.080° | 1.249° |
| v61-best-medw | 0.217° | 0.122° | 0.139° | 0.080° | 1.249° |
| v61-best-val | 0.217° | 0.122° | 0.139° | 0.080° | 1.249° |
| v61-ckpt8 | 0.217° | 0.121° | 0.136° | 0.083° | 1.252° |
| v61-ckpt4 | 0.218° | 0.123° | 0.135° | 0.084° | 1.247° |

Reference:
- V60 `ckpt15`: MEDW400 = 0.205° (R 0.121 / P 0.120 / Y 0.082).
- V56 `ckpt_best_dual`: MEDW400 = 0.2125° (R 0.1353 / P 0.1006 / Y 0.0886).

## Interpretation

V61 did not beat V60. The strong hard-sequence sampler improved neither the formal total nor the hard sequence bottleneck.

Main observations:
- Seq03 remains dominant: best V61 Seq03 MEDW400 Rot is still about 0.65-0.67°.
- Seq02 remains second hardest at about 0.35-0.36°.
- Yaw improved on the best V61 candidates, but pitch regressed from V60's 0.120° to about 0.137-0.139°.
- V61 used `data_balance=1` plus `03:2.00`, which likely over-pressured the hard sequences and shifted the V60 sweet spot.
- Training was stable enough to finish, but still had several `NaN GUARD` skipped batches.

## Next Step

Use V60 `ckpt_15.pth` as the anchor again. Do not continue from V61.

V62 should be a smaller bias-lite polish:
- Return to `data_balance=2` sqrt balancing.
- Add only mild sequence overrides for Seq02/Seq03/Seq10.
- Disable MGDA for this short refine so zero-drift does not dominate as a single task.
- Lower LR, backbone LR, jacobian pressure, inject pressure, and mount/large-perturb augmentation.
- Keep stronger pitch guard, because V61's main formal regression was pitch.

Target:
- Primary: recover and beat V60, MEDW400 <= 0.200°.
- Stretch: MEDW400 <= 0.195°.
- Guardrails: pitch <= 0.125°, yaw <= 0.085°, per-frame mean <= 1.28°.
