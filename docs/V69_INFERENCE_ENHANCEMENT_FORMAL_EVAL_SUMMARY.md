# V69 Inference Enhancement Formal Eval Summary

Date: 2026-07-01

## Protocol

- Dataset: `test_data_v2`, excluding Seq07.
- Samples: 11 sequences x 400 frames = 4400 frames.
- Tool: `tools/analysis/evaluate_p0_refinement.py`.
- Script: `run_v69_inference_enhancement_eval.sh`.

## Random Perturbation Sequence Median

| Candidate | Per-frame Rot | Seq-median Rot | Roll | Pitch | Yaw |
| --- | ---: | ---: | ---: | ---: | ---: |
| V29-G3 best-val | 2.4958 | 0.0594 | 0.0266 | 0.0364 | 0.0282 |
| V60 ckpt15 | 1.2718 | 0.1986 | 0.1258 | 0.1089 | 0.0776 |

Interpretation:

- V29-G3 is excellent for temporally unbiased random perturbations.
- V60 remains the stronger single-frame model, but its sequence-median random perturbation result is weaker than V29.

## Fixed 2/2/2 Degree Inject Sequence Median

| Candidate | Injected Rot | Residual Rot | Mean Recovery | Median Recovery |
| --- | ---: | ---: | ---: | ---: |
| V29-G3 best-val | 3.4437 | 3.4598 | -0.5% | -0.5% |
| V60 ckpt15 | 3.4437 | 1.6510 | 52.1% | 47.0% |

V60 fixed-inject hard sequences:

| Seq | Residual Rot | Recovery |
| --- | ---: | ---: |
| 00 | 2.3823 | 30.8% |
| 06 | 2.4155 | 29.9% |
| 10 | 2.5104 | 27.1% |
| 11 | 2.1379 | 37.9% |

## Decision

Do not promote V29-G3 as a deployment calibrator despite its strong random MEDW: it fails fixed-bias recovery completely.

Keep V60 ckpt15 as the best recovery-aware reference, but it still misses the target fixed-inject recovery gate (`>=70%`). V70 should target fixed recovery, not random MEDW:

- Anchor from V60 ckpt15.
- Disable MGDA to avoid zero-drift-only domination.
- Reduce zero-drift dedicated ratio and increase fixed-inject/Jacobian pressure.
- Upweight fixed-inject hard sequences Seq00/06/10/11 while keeping Seq02/03 monitored for random MEDW.

## V70 Smoke Follow-Up

V70 smoke was trained from V60 ckpt15 with stronger fixed-inject pressure and hard-sequence weighting.

Training health:

- 4-GPU smoke completed 8 epochs.
- Only 2 NaN-guarded batches.
- Internal validation remained stable: best MEDW200 max R/P/Y `0.0267°`, best controlled Jacobian about `1.004`.

External V69b fixed-inject sequence-median result:

| Candidate | Residual Rot | Mean Recovery | Median Recovery |
| --- | ---: | ---: | ---: |
| V60 ckpt15 | 1.6510 | 52.1% | 47.0% |
| V70 smoke best-dual | 1.6221 | 52.9% | 48.2% |
| V70 smoke best-jacobian | 1.5750 | 54.3% | 49.5% |

Interpretation:

- V70 gives only a small fixed-recovery gain (`+2.2pp` mean recovery at best).
- Hard fixed-inject sequences Seq00/06/10 remain around only `28-32%` recovery.
- Do not launch V70 full. Loss/sampler pressure alone is not enough.

Recommended next direction:

- Keep V60/V70 as recovery-aware references, not final deploy candidates.
- Move to a structural recovery mechanism: explicit large-error route, residual correction head with direct `T_init` conditioning, or a classical/self-supervised projection refiner after the learned model.
- Any next candidate must be selected by V69b fixed-inject sequence recovery first, then random MEDW second.
