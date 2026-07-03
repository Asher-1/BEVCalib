# V80 Tinit-BEV-FiLM RPY02 Plan

## Target

The deployment/generalization target is strict: all R/P/Y axes should be below `0.2 deg`, not the previous `0.3 deg` acceptance line.

## Current Evidence

V78 is the best random-generalization prior so far:

- Random full sequence-median aggregate: `0.1431 deg`.
- Random full R/P/Y: `0.1080 / 0.0733 / 0.0360 deg`.
- Fixed 2/2/2 recovery: only `0.9%`, so the model has almost no deploy-usable recovery path.

V79 showed that injecting `T_init` only at the final head is too late:

- Best internal MEDW max R/P/Y stayed above `0.2 deg`.
- Jacobian was weak and axis-imbalanced.
- Random RPY02 stability degraded after the first epochs.

## Design

V80 keeps the V78 DINOv2-query low-bias prior and adds a zero-initialized `T_init` FiLM adapter before BEV pooling:

1. Encode `T_init` RPY using the existing Fourier `ExplicitTInitEncoder`.
2. Predict channel-wise BEV `gamma/beta`.
3. Apply FiLM on the fused BEV feature map before pose embedding, transformer, and pooling.
4. Freeze the visual/BEV prior and train only:
   - `tinit_bev_encoder`
   - `tinit_bev_film`
   - `rotation_pred`

This moves the fixed-offset signal earlier than V79 while keeping the initial model function identical to V78 because the final FiLM projection is zero-initialized.

## Promotion Gate

V80 smoke should only be promoted to full if external evaluation shows both:

- Random full aggregate R/P/Y all `< 0.2 deg`.
- Fixed 2/2/2 recovery clearly improves over V78 and trends back toward the `72%+` recovery floor.

If V80 still cannot form fixed recovery, the next step should be a deeper architectural branch: Tinit-conditioned query offsets or native/cross-attention matching where the current pose changes point-image association directly, not only BEV channel statistics.

## Artifacts

- Code: `kitti-bev-calib/bev_calib.py`
- Config: `configs/v80_tinit_bev_film_rpy02.yaml`
- Eval script: `run_v80_rpy02_eval.sh`
