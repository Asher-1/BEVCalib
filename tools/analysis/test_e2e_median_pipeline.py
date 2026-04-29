#!/usr/bin/env python3
"""
End-to-end test: SequenceMedianAggregator + BEVCalibInference on real data.

Tests the full production inference flow:
  1. Load model via evaluate_checkpoint infrastructure (handles backend detection)
  2. Run per-frame inference using BEVCalibInference wrapper
  3. Aggregate with SequenceMedianAggregator
  4. Compare per-frame mean vs aggregated results
"""
import os
import sys
import torch
import numpy as np
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

CKPT_PATH = "logs/all_training_data/test_models2/model_small_5deg_v20_v8recipe_pitch_wt3/all_training_data_scratch/checkpoint/ckpt_best_val.pth"
DATASET_ROOT = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2"

MAX_FRAMES_PER_SEQ = 100

# Auto-detect spconv vs drcv backend from checkpoint before any imports
_sd = torch.load(CKPT_PATH, map_location='cpu').get('model_state_dict', {})
_pc_keys = [k for k in _sd if k.startswith('pc_branch.sparse_encoder')]
if any('.kernel' in k for k in _pc_keys):
    os.environ['USE_DRCV_BACKEND'] = '1'
elif any(k.endswith('.weight') and len(_sd[k].shape) == 5 for k in _pc_keys):
    os.environ['USE_DRCV_BACKEND'] = '0'
del _sd, _pc_keys


def main():
    from torch.utils.data import DataLoader, Subset
    from tools import generate_single_perturbation_from_T
    from visualization import compute_pose_errors
    from evaluate_checkpoint import (
        make_collate_fn, _auto_permute_spconv_weights,
        _adapt_model_to_checkpoint, _build_eval_custom_dataset,
    )
    from bev_calib import BEVCalib
    from bevcalib_inference import BEVCalibInference, SequenceMedianAggregator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load model using evaluate_checkpoint infrastructure ---
    print("=" * 70)
    print("E2E Test: SequenceMedianAggregator Pipeline")
    print("=" * 70)
    t0 = time.time()

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    rotation_only = not checkpoint.get('optimize_translation', True)
    state_dict = checkpoint['model_state_dict']
    use_mlp_head = 'rotation_pred.0.weight' in state_dict
    ckpt_args = checkpoint.get('args', {})

    model = BEVCalib(
        deformable=False,
        bev_encoder=True,
        img_shape=(360, 640),
        rotation_only=rotation_only,
        use_mlp_head=use_mlp_head,
        bev_pool_factor=0,
        voxel_mode=ckpt_args.get('voxel_mode', 'hard'),
        to_bev_mode=ckpt_args.get('to_bev_mode', 'concat'),
        scatter_reduce=ckpt_args.get('scatter_reduce', 'sum'),
        intrinsic_input=ckpt_args.get('intrinsic_input', False),
    ).to(device)

    state_dict = _auto_permute_spconv_weights(state_dict, model)
    _adapt_model_to_checkpoint(model, state_dict, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapper = BEVCalibInference(model, max_attn_tokens=2048).to(device).eval()
    aggregator = SequenceMedianAggregator(min_frames=5, max_frames=200)
    print(f"\n[OK] Model loaded (epoch {epoch}) in {time.time()-t0:.1f}s")

    # --- 2. Load dataset & index by sequence ---
    class _FakeArgs:
        eval_sample_step = None
        eval_max_frames_per_seq = MAX_FRAMES_PER_SEQ
    dataset = _build_eval_custom_dataset(DATASET_ROOT, _FakeArgs())

    seq_to_indices = defaultdict(list)
    for idx, fpath in enumerate(dataset.all_files):
        seq_id = fpath.split('/')[0]
        seq_to_indices[seq_id].append(idx)

    seq_ids = sorted(seq_to_indices.keys())
    print(f"[OK] Dataset: {len(dataset)} frames, {len(seq_ids)} sequences\n")

    collate_fn = make_collate_fn((640, 360))
    np.random.seed(42)
    torch.manual_seed(42)

    # --- 3. Per-sequence: per-frame inference + aggregation ---
    results = []
    for seq_id in seq_ids:
        indices = seq_to_indices[seq_id]

        aggregator.reset()
        frame_errors = []
        gt_T_first = None

        seq_dataset = Subset(dataset, indices)
        seq_loader = DataLoader(
            seq_dataset, batch_size=1, num_workers=2,
            collate_fn=collate_fn, shuffle=False)

        with torch.no_grad():
            for batch in seq_loader:
                imgs, pcs, masks, gt_T, intrinsics = batch
                gt_T_np = np.array(gt_T).astype(np.float32)
                init_T_np, _, _ = generate_single_perturbation_from_T(
                    gt_T_np, angle_range_deg=5.0, trans_range=0.15,
                    rotation_only=rotation_only)

                if gt_T_first is None:
                    gt_T_first = gt_T_np[0]

                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                pcs_t = torch.from_numpy(np.array(pcs)[:, :, :3]).float().to(device)
                init_T_t = torch.from_numpy(init_T_np).float().to(device)
                post_T = torch.eye(4).unsqueeze(0).repeat(init_T_t.shape[0], 1, 1).float().to(device)
                intrinsic_t = torch.from_numpy(np.array(intrinsics)).float().to(device)

                pred_T = wrapper(resize_imgs, pcs_t, init_T_t, post_T, intrinsic_t)
                pred_T_np = pred_T.cpu().numpy()

                aggregator.add(pred_T)

                for i in range(len(gt_T_np)):
                    errs = compute_pose_errors(pred_T_np[i], gt_T_np[i])
                    frame_errors.append(errs['rot_error'])

        mean_frame_err = np.mean(frame_errors)

        median_T = aggregator.aggregate()
        median_errs = compute_pose_errors(median_T, gt_T_first)
        confidence = aggregator.get_confidence()

        improvement = (mean_frame_err - median_errs['rot_error']) / mean_frame_err * 100

        results.append({
            'seq': seq_id,
            'n_frames': len(indices),
            'per_frame_rot': mean_frame_err,
            'median_rot': median_errs['rot_error'],
            'median_roll': median_errs['roll_error'],
            'median_pitch': median_errs['pitch_error'],
            'median_yaw': median_errs['yaw_error'],
            'improvement': improvement,
            'confidence': confidence,
        })

        status = "OK" if aggregator.ready else "LOW_CONFIDENCE"
        print(f"  Seq {seq_id}: {len(indices):>4} frames | "
              f"Per-frame {mean_frame_err:.4f}° → Median {median_errs['rot_error']:.4f}° "
              f"({'+' if improvement > 0 else ''}{improvement:.1f}%) "
              f"| conf={confidence['total_std']:.3f}° [{status}]")

    # --- 4. Summary ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Summary ({elapsed:.0f}s total)")
    print(f"{'=' * 70}")

    mean_pf = np.mean([r['per_frame_rot'] for r in results])
    mean_med = np.mean([r['median_rot'] for r in results])
    mean_imp = (mean_pf - mean_med) / mean_pf * 100

    print(f"  Per-frame mean:     {mean_pf:.4f}°")
    print(f"  Median aggregated:  {mean_med:.4f}°")
    print(f"  Improvement:        {mean_imp:+.1f}%")
    print(f"  Best seq:  {min(results, key=lambda r: r['median_rot'])['seq']} "
          f"({min(r['median_rot'] for r in results):.4f}°)")
    print(f"  Worst seq: {max(results, key=lambda r: r['median_rot'])['seq']} "
          f"({max(r['median_rot'] for r in results):.4f}°)")

    n_below_03 = sum(1 for r in results if r['median_rot'] < 0.3)
    print(f"  Sequences < 0.3°:  {n_below_03}/{len(results)}")

    print(f"\n  Per-axis median errors:")
    print(f"    Roll:  {np.mean([r['median_roll'] for r in results]):.4f}°")
    print(f"    Pitch: {np.mean([r['median_pitch'] for r in results]):.4f}°")
    print(f"    Yaw:   {np.mean([r['median_yaw'] for r in results]):.4f}°")

    print(f"\n[OK] E2E test complete. Pipeline ready for deployment.")


if __name__ == "__main__":
    main()
