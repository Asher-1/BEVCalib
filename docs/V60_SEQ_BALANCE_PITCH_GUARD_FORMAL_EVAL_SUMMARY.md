# V60 Seq-Balance Pitch-Guard Formal Eval Summary

Date: 2026-06-30

## Result

V60 full completed normally and improved the current deploy-generalization best, but did not reach the stretch target.

| Candidate | MEDW400 Rot | Roll | Pitch | Yaw | Per-frame Mean Rot |
| --- | ---: | ---: | ---: | ---: | ---: |
| v60-ckpt15 | 0.205° | 0.121° | 0.120° | 0.082° | 1.266° |
| v60-ckpt18 | 0.207° | 0.124° | 0.119° | 0.082° | 1.267° |
| v60-best-dual | 0.209° | 0.122° | 0.120° | 0.088° | 1.266° |
| v60-best-jacobian | 0.216° | 0.133° | 0.121° | 0.085° | 1.292° |
| v60-ckpt10 | 0.217° | 0.128° | 0.127° | 0.087° | 1.265° |

Previous best reference: V56 `ckpt_best_dual`, MEDW400 = 0.2125° (R 0.1353 / P 0.1006 / Y 0.0886).

## Interpretation

V60 is a small win on total MEDW400: 0.2125° -> 0.205°.

The improvement comes mainly from roll/yaw:
- Roll: 0.1353° -> 0.121°.
- Yaw: 0.0886° -> 0.082°.

The remaining blocker is pitch:
- V56 pitch was 0.1006°.
- V60 best pitch is 0.120°.

Per-sequence analysis shows the pitch issue is concentrated rather than global:
- Seq03 remains the dominant outlier: Rot 0.690°, R/P/Y 0.438/0.448/0.291.
- Seq02 is the second hard case: Rot 0.406°, R/P/Y 0.233/0.330/0.032.
- Seq10 is a smaller residual hard case: Rot 0.225°.

## Recommendation

Keep V60 `ckpt15` as the current best experimental candidate, but do not call the 0.18° target solved.

Next step should be V61 hard-sequence refine:
- Anchor from V60 `ckpt_15.pth`, not from V56.
- Add sampler-level sequence overrides for Seq02/Seq03, with mild support for Seq10/Seq06/Seq08.
- Lower LR and inject pressure to reduce the recurring NaN-guard skips seen in V60.
- Keep candidate selection broad: evaluate best-dual, best-medw, best-val, ckpt10, ckpt12/15 if produced.

Target for V61:
- Primary: MEDW400 <= 0.195°.
- Stretch: MEDW400 <= 0.18°.
- Guardrail: do not regress yaw above 0.09° or per-frame mean above 1.30°.
