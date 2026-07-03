# V65 Query-DINOv2 Recovery Plan

Date: 2026-07-01

## Motivation

V64b full did not beat the V60 CF-BEV-R reference. V29-G3 then exposed a much better low-frequency aggregation pattern with Query-BEV + DINOv2-small, but its acceptance diagnostic showed near-zero fixed-inject recovery.

Therefore V65 is not a plain continuation. It is a recovery rebaseline:

- Keep V29-G3 architecture and checkpoint as the low-bias prior.
- Add explicit fixed-inject, zero-drift, and Jacobian supervision from the V51+ training stack.
- Use direct weighted-sum recovery losses; MGDA is disabled because the current Query-BEV/DINOv2 path has no MGDA bottleneck parameter group.
- Keep the V29 regression head (`use_mlp_head: 0`); using the newer MLP head randomizes the head and causes 120 deg+ initial pose error.
- Promote only if the model passes both MEDW and fixed-bias recovery diagnostics.

## Training Lines

Smoke:

- Config: `configs/v65_query_dinov2_recovery.yaml`
- Experiment: `v65_query_dinov2_recovery_smoke`
- GPUs: 4
- Epochs: 20
- Data: 200 frames per seq
- Purpose: verify that Query-BEV + DINOv2 accepts recovery supervision without NaNs or collapsing the V29 low-bias prior.

Full:

- Experiment: `v65_query_dinov2_recovery`
- GPUs: 8
- Epochs: 90
- Data: 500 frames per seq
- Purpose: recover fixed 2 deg injected errors while preserving MEDW400 competitiveness.

## Acceptance

V65 should not be selected by MEDW alone.

Minimum deploy gates:

- MEDW400 <= 0.18 deg on `test_data_v2`, excluding Seq07.
- Zero-drift max RPY <= 0.10 deg.
- Fixed 2 deg inject recovery >= 70%.
- No `HIGH` shortcut-risk diagnostic.

Reference baselines:

- V29-G3: MEDW400 ~= 0.09 deg, but fixed-inject recovery ~= 0%.
- V60 ckpt15: best current CF-BEV-R recovery-aware reference, MEDW400 ~= 0.205 deg.
- V64b ckpt2: stable but regressed, MEDW400 ~= 0.213 deg.

## Next Actions

1. Run V65 smoke.
2. Evaluate smoke with `configs/eval_generalization_v65_query_dinov2_recovery_only.yaml`.
3. If smoke improves fixed-inject recovery without MEDW collapse, launch full.
4. If recovery remains near zero, increase inject-dedicated ratio and lower mount jitter before full.

## V65b Recovery Push

Early V65 smoke is stable but still has near-zero Jacobian response, so V65b is prepared as the recovery-push branch:

- Config: `configs/v65b_query_dinov2_recovery_push.yaml`
- Eval config: `configs/eval_generalization_v65b_query_dinov2_recovery_push_only.yaml`
- Main changes: higher head LR, stronger inject/Jacobian supervision, lower zero-drift and overcorrection pressure, lower mount jitter.
- Run V65b smoke in parallel on GPUs 4-7 while V65 smoke continues on GPUs 0-3.

## Smoke Outcomes

V65 smoke:

- Loaded V29-G3 cleanly with `use_mlp_head: 0` (`752/752` keys).
- Stayed numerically stable and kept low random-perturbation MEDW.
- Failed the fixed-bias recovery goal: controlled Jacobian remained near zero and triggered early stop.
- Decision: do not launch V65 full.

V65b smoke:

- Stronger inject/Jacobian pressure reduced inject loss, but controlled Jacobian still stayed near zero.
- Run hit a DDP rendezvous failure after epoch 4 and was stopped after enough diagnostic evidence.
- Decision: do not launch V65b full.

Root cause from V65/V65b:

- Query-BEV + DINOv2 preserves V29's low random-perturbation bias, but the V29-compatible head has no direct high-bandwidth `T_init` input.
- Fixed-inject final-pose loss alone can reduce local training loss without forcing output sensitivity to controlled initial-bias changes.
- The next line must change structure, not just increase recovery loss weights.

## V66 Explicit Tinit Result

V66 enabled `explicit_tinit` and patched pretrain loading so the V29 linear head is expanded from 256 to 320 inputs while preserving old weights.

- Config: `configs/v66_explicit_tinit_recovery.yaml`
- Smoke checkpoint source: V29-G3 best-val.
- Load result: `752/758` keys, `rotation_pred.weight` expanded with new T_init columns zero-initialized.
- Best early MEDW: epoch 1 max R/P/Y `0.2339°`.
- Best Jacobian: epoch 9 overall `0.145` with R/P/Y `0.080/0.354/0.003`.
- Final status: early-stopped at epoch 9, dual gate not passed.

Interpretation:

- `explicit_tinit` opens a real recovery-gain path: Pitch Jacobian rose from near zero to `0.354`.
- The gain is axis-imbalanced, with Yaw still effectively zero.
- MEDW degraded to about `0.55°`, so V66 is not promotable and should not run full.

## V67 Axis-Balanced Follow-Up

V67 keeps the useful V66 structural change but reduces the noisy sensitivity pressure and targets the weak axes directly.

- Config: `configs/v67_explicit_tinit_axis_balance.yaml`
- Code addition: `jacobian_loss_axis_weights`, default empty and backward compatible.
- Smoke weights: Jacobian train-axis sampling R/P/Y = `1.3/1.0/2.7`.
- Tinit sensitivity is reduced from `0.08` to `0.02`.
- Inject dedicated ratio is reduced from `0.55` to `0.32` to protect MEDW.
- Yaw is emphasized in `axis_weights` and `per_axis_weights`.

Promotion rule:

- Continue to full only if smoke improves all-axis Jacobian, especially Yaw, without the V66 MEDW collapse.
- If V67 still reaches only single-axis recovery, the next step should be a deeper residual/iterative correction objective rather than another weight-only sweep.

## V67/V68 Outcome And V69 Pivot

V67 smoke result:

- Axis-weighted Jacobian sampling did not fix the yaw recovery failure.
- Epoch 5 MEDW200 regressed to `0.5962°`; controlled Jacobian was only `0.072` overall with R/P/Y `0.028/0.191/-0.002`.
- Decision: stop V67; do not run full.

V68 smoke result:

- Direct raw correction quaternion supervision loaded the V29 head cleanly (`752/752`) and preserved low epoch-1 MEDW200 (`0.2100°`).
- By epoch 5 MEDW200 collapsed to `0.8885°`, while controlled Jacobian remained `0.000`.
- Decision: stop V68; do not run full.

Root cause update:

- The train-time Jacobian loss is backpropagated before the optimizer step, so the failure is not a missing-gradient bug.
- Query-BEV + DINOv2 is still valuable as a low-bias temporal regressor, but current training objectives do not make it a genuine fixed-bias recovery model.
- Repeating loss-weight sweeps is low expected value. The next deploy-facing line should test inference-side sequence aggregation/adaptation against the stable V60 recovery-aware baseline.

V69 pivot:

- Script: `run_v69_inference_enhancement_eval.sh`
- Primary check: P0 sequence-median aggregation on `test_data_v2`, excluding Seq07, max 400 frames/seq.
- Candidates: V29-G3 best-val as the low-MEDW query prior, and V60 ckpt15 as the stable CF-BEV-R reference.
- TTA-BN smoke was neutral/slightly negative; keep it diagnostic-only unless a stronger self-supervised signal is added.
