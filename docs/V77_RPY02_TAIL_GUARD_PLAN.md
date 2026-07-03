# V77 RPY02 Tail Guard Plan

Date: 2026-07-03

## Goal

The target KPI is strict deployment generalization with all R/P/Y axes below
`0.2 deg`, not `0.3 deg`.

V74e2 adaptive-default remains the current deployment baseline, but it is only
sequence-level acceptable:

- random sequence-median: `0.3465 deg`
- fixed `2/2/2` recovery: `72.3%`
- random `RPY all <0.3 deg`: `21.8%`
- fixed adaptive `RPY all <0.3 deg`: `22.9%`

Because the `0.2 deg` all-axis target is much stricter, V77 must be treated as a
new precision-tail optimization track, not as a minor report threshold change.

## Design

V77 starts from V74e2 `ckpt_2.pth`, the best random sequence-median anchor.

The first run intentionally avoids a full backbone or DINOv2 upgrade:

- Trainable modules: `corr_head`, `pose_query_init`, `adir_refiner`, `pitch_branch`, `pitch_conf_net`.
- Frozen modules: image/point/BEV backbone and fusion trunk.
- Loss shift: stronger balanced axis loss, more yaw/pitch single-axis exposure, lower recovery pressure than V75/V76.
- Guardrails: keep zero-drift and fixed-inject losses active so adaptive-default gates do not regress.
- Validation: checkpoint selection remains MEDW/Jacobian aware, but final acceptance is external adaptive-default eval with `rpy_threshold_deg=0.2`.

## Training

Smoke:

```bash
bash batch_train.sh --skip-pattern full configs/v77_rpy02_tail_guard_cf_bev_r.yaml
```

Full, only if smoke keeps the deployment gates healthy:

```bash
bash batch_train.sh --skip-pattern smoke configs/v77_rpy02_tail_guard_cf_bev_r.yaml
```

## Evaluation

Smoke checkpoints:

```bash
GPU=0 LABEL=v77_smoke_best_dual bash run_v77_rpy02_eval.sh both_full \
  logs/all_training_data/model_small_5deg_v77_rpy02_tail_guard_smoke/all_training_data_scratch/checkpoint/ckpt_best_dual.pth
```

Fallback checkpoints if `ckpt_best_dual.pth` is absent:

- `ckpt_best_jacobian.pth`
- `ckpt_best_medw.pth`
- final epoch checkpoint, e.g. `ckpt_4.pth`

## Acceptance

V77 is a candidate only if it improves strict RPY02 while preserving the already
passed deployment gates:

- `random_full`: `RPY all <0.2 deg` must improve over the V74e2 baseline re-run.
- `random_full`: sequence-median should remain around `0.35 deg`.
- `fixed222_full`: mean recovery must stay `>=72%`.
- Adaptive random trigger should stay near `0/11`.

If V77 cannot move strict `RPY all <0.2 deg`, the next step is not another
micro-polish. Reopen architecture review with either:

- a residual confidence/gating head trained explicitly on per-frame axis tails,
  or
- a stronger DINOv2 backbone experiment with the same adaptive-default
  acceptance protocol.
