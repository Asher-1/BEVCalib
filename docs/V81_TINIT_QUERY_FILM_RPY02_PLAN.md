# V81 Tinit-Query-FiLM RPY02 Plan

## Why V81

V80 confirmed the BEV-level adapter is not enough:

- Best internal MEDW200: `max(R,P,Y)=0.2036 deg`, just above the strict `0.2 deg` target.
- Best Jacobian: `overall=0.072`, with yaw near zero.
- Dual gate never passed.

The fixed-offset recovery signal still reaches the model too late and too globally.

## Design

V81 moves T_init conditioning into `Cam2BEVQuery` before self/cross-attention:

1. Encode `T_init` RPY with a small Fourier MLP.
2. Predict query-token channel FiLM (`gamma/beta`).
3. Apply it to BEV query embeddings before they attend to image tokens.
4. Start from V78 best MEDW, preserving the best random RPY02 prior.
5. Train only:
   - `img_branch.tinit_query_encoder`
   - `img_branch.tinit_query_film`
   - `rotation_pred`

This lets the current pose alter query/image association rather than only the final pooled BEV representation.

## Gate

Smoke promotion requires:

- Internal MEDW R/P/Y all `< 0.2 deg`.
- Jacobian materially above V80, especially yaw.
- External random full all-axis RPY02 and fixed 2/2/2 recovery trend toward `72%+`.

If V81 fails, the next branch should stop doing adapter-only training and move to either:

- DINOv2-base query backbone with longer full training, or
- Native/cross-attention correspondence architecture where T_init directly changes point-image matching.

## Artifacts

- Code: `kitti-bev-calib/img_branch/cam2bev_query.py`, `kitti-bev-calib/bev_calib.py`
- Config: `configs/v81_tinit_query_film_rpy02.yaml`
- Eval script: `run_v81_rpy02_eval.sh`
