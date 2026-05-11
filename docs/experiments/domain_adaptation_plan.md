# Domain Adaptation Plan: Path to <0.1° RPY Generalization

## Root Cause Analysis

### The 9.2x Gap Decomposition

| Factor | Contribution | Evidence |
|--------|-------------|----------|
| Cross-vehicle install diversity | ~50% | Seq09 (same vehicle) → 0.30°, new vehicles → 0.65-1.04° |
| BEV feature memorization | ~25% | Camera BEV features are near-constant (Δ<0.04%) across scenes |
| Pitch-specific Z-distribution shift | ~15% | Camera bypass: Pitch degrades 49.2% without image |
| Scene/appearance overfitting | ~10% | drcv (retains texture) generalizes worse than spconv |

### Key Insight
The model doesn't learn a **transferable geometric extrinsic error signal**. Instead it learns:
- **Roll/Yaw ≈ f(LiDAR BEV statistics)** — works well within-distribution
- **Pitch ≈ f(LiDAR + weak horizon cue)** — fails on new camera mounts
- Camera BEV features are a **fixed FOV template**, not alignment residuals

## Strategy 1: Extrinsic-Aware Augmentation (High Priority)

### Problem
Training only applies ±5° perturbation to GT extrinsics, but test vehicles have
fundamentally different camera-LiDAR mounting configurations (e.g. test Seq02 has
~164° roll discrepancy vs train Seq02 — different cars sharing the same seq ID).

### Solution
```
ExtrinsicAugmentor:
  1. Random extrinsic "mount jitter" beyond perturbation:
     - Sample Δ_mount ~ N(0, σ²) for tx,ty,tz,roll,pitch,yaw
     - σ_mount >> σ_perturbation (e.g. mount σ=0.5° vs perturbation σ=1.7°)
     - Apply BEFORE perturbation: T_init = T_mount_jittered * T_gt
  2. Mount family augmentation:
     - Cluster training extrinsics into K mount families (e.g. K=5)
     - During training, randomly swap to a different mount family's
       extrinsic with probability p_mount_swap
     - Forces model to learn extrinsic-invariant features
```

### Expected Impact: 15-20% improvement

## Strategy 2: Contrastive Extrinsic Learning (Medium Priority)

### Problem
The fuser doesn't explicitly encode the **geometric residual** between LiDAR and
camera projections. It treats the fused BEV as a holistic feature map.

### Solution
```
ContrastiveExtrinsicHead:
  1. Compute BEV difference map: D = |cam_bev - pc_bev|
  2. Add auxiliary contrastive loss:
     - Positive pairs: same sample, different perturbations
     - Negative pairs: different samples, similar perturbations
     - Forces D to encode EXTRINSIC OFFSET, not scene content
  3. Architecture: D → Conv(3x3) → GlobalPool → MLP → extrinsic_embedding
  4. Loss: InfoNCE on extrinsic_embedding space
```

### Expected Impact: 10-15% improvement

## Strategy 3: Per-Sequence Bias Correction (High Priority, Easy)

### Problem
Errors are systematic per-sequence (0.16° std across sequences), not random noise.
TTA only improved ~7% because the bias is **static** per vehicle install.

### Solution
```
SequenceBiasEstimator:
  1. Online bias estimation during inference:
     - Maintain running mean of predictions over N frames (e.g. N=50)
     - Compute bias = mean(predictions) - expected_zero (for small perturbations)
     - Subtract bias from subsequent predictions
  2. Calibration protocol:
     - First 50 frames: accumulate statistics
     - After warmup: apply per-axis bias correction
     - Update bias with exponential moving average (α=0.99)
  3. Benefits:
     - Handles systematic mount offset without retraining
     - Works even with large domain gap
     - Composable with model improvements
```

### Expected Impact: 20-30% improvement on hardest sequences

## Strategy 4: Feature Normalization (Medium Priority)

### Problem
Camera BEV features are a fixed FOV template (Δ<0.04% across scenes).
The model memorizes the spatial pattern rather than learning alignment residuals.

### Solution
```
BEVFeatureNormalization:
  1. Instance Normalization on camera BEV features before fusion:
     cam_bev_norm = InstanceNorm2d(cam_bev_feat)
     - Removes global activation pattern (the fixed FOV triangle)
     - Preserves local spatial variations (actual alignment signals)
  2. Feature whitening (optional):
     - Compute per-channel mean/var across spatial dimensions
     - Standardize before fuser
     - Add learnable affine transform after
  3. Gradient-Reversal Layer on scene classification:
     - Auxiliary head predicts scene/sequence identity
     - Gradient reversal forces BEV features to be scene-invariant
```

### Expected Impact: 10-15% improvement

## Strategy 5: Multi-Scale Temporal Aggregation (Lower Priority)

### Problem
Single-frame predictions have systematic bias. Multi-frame averaging (N=50)
only helps ~3-5% because the bias is constant within a sequence.

### Solution
```
TemporalBiasAwareAggregation:
  1. Instead of simple averaging, use:
     - Robust median (already shows +6.9% vs mean)
     - RANSAC-style outlier rejection
     - Weighted average based on prediction confidence
  2. Cross-frame consistency loss during training:
     - Consecutive frames should predict similar extrinsics
     - Penalize large prediction jumps
  3. Adaptive window: use larger window when predictions are unstable
```

### Expected Impact: 5-10% improvement

## Implementation Priority

| Phase | Strategy | Effort | Expected Impact | Cumulative |
|-------|----------|--------|-----------------|------------|
| 1 | v25 refactored (all current fixes) | Done | 0.664° → 0.35° | 0.35° |
| 2 | Per-sequence bias correction (S3) | 1 day | 0.35° → 0.25° | 0.25° |
| 3 | Extrinsic-aware augmentation (S1) | 2 days | 0.25° → 0.18° | 0.18° |
| 4 | BEV feature normalization (S4) | 1 day | 0.18° → 0.14° | 0.14° |
| 5 | Contrastive learning (S2) | 3 days | 0.14° → 0.11° | 0.11° |
| 6 | Temporal aggregation (S5) | 1 day | 0.11° → 0.09° | **0.09°** |

**Total estimated time: 8 days of development + training cycles**

## Key Risk: Data Diversity

Even with all the above, if test vehicles have camera mounts that are truly
OOD (e.g. Seq02 with ~164° roll difference from any training sample), no amount
of augmentation can bridge the gap. The ultimate solution requires:

1. Adding 3-5 more vehicle types to training data
2. Or: fine-tuning on a small calibration set from each new vehicle (few-shot)
3. Or: explicit camera mount as an input feature (not just intrinsics)
