"""
BEVCalib inference wrapper -- clean forward path without loss computation.
Used for model export (drinfer/ONNX) and standalone inference benchmarking.

Includes SequenceMedianAggregator for production deployment: accumulates
per-frame predictions across a sequence, then returns a robust aggregated
calibration via rotation matrix median (SVD-projected) + translation median.

IMPORTANT: set BEV_ZBOUND_STEP env var *before* importing this module so that
bev_settings.py picks up the correct Z resolution.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import deque

_KITTI_DIR = os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib')
if _KITTI_DIR not in sys.path:
    sys.path.insert(0, _KITTI_DIR)


class BEVCalibInference(nn.Module):
    """
    Inference-only wrapper for BEVCalib.
    Strips loss computation and returns the predicted LiDAR->Camera transform.
    """

    def __init__(self, model, max_attn_tokens=0):
        super().__init__()
        self.model = model
        self.rotation_only = model.rotation_only
        self.intrinsic_input = getattr(model, 'intrinsic_input', False)
        self.max_attn_tokens = max_attn_tokens

    @torch.no_grad()
    def forward(self, img, pc, init_T_to_camera, post_cam2ego_T, cam_intrinsic):
        """
        Args:
            img:               (B, 3, H, W)   RGB image
            pc:                (B, N, 3)       point cloud (XYZ)
            init_T_to_camera:  (B, 4, 4)       initial LiDAR->Camera transform
            post_cam2ego_T:    (B, 4, 4)       post-augmentation (identity at inference)
            cam_intrinsic:     (B, 3, 3)       camera intrinsic matrix

        Returns:
            pred_T:  (B, 4, 4)  predicted LiDAR->Camera transform
        """
        m = self.model
        B = img.shape[0]

        img_ = img.unsqueeze(1)
        init_ = init_T_to_camera.unsqueeze(1)
        post_ = post_cam2ego_T.unsqueeze(1)
        K_ = cam_intrinsic.unsqueeze(1)
        cam2ego_T = torch.linalg.inv(init_)

        cam_bev_feats, cam_bev_mask = m.img_branch(
            cam2ego_T=cam2ego_T, cam_intrins=K_,
            post_cam2ego_T=post_, imgs=img_,
        )

        pc_perm = pc.permute(0, 2, 1).contiguous()
        pc_bev_feats = m.pc_branch(pc_perm)

        x = m.conv_fuser(cam_bev_feats, pc_bev_feats)
        if m.bev_encoder_use:
            x = m.bev_encoder(x)
        x = x + m.pose_embed

        if m.deformable:
            x = m.deformable_transformer(x)
            _B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(_B, H * W, C)
            bev_mask = cam_bev_mask.reshape(_B, H * W).float().unsqueeze(-1)
            x = (x * bev_mask).sum(dim=1) / bev_mask.sum(dim=1).clamp(min=1)
        else:
            _B, C, H, W = x.shape
            if hasattr(m, 'bev_pool_factor') and m.bev_pool_factor > 1:
                pf = m.bev_pool_factor
                x = nn.functional.avg_pool2d(x, pf)
                cam_bev_mask = nn.functional.max_pool2d(
                    cam_bev_mask.reshape(_B, 1, H, W).float(), pf).squeeze(1)
                _, _, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(_B, H * W, C)
            bev_mask_f = cam_bev_mask.reshape(_B, H * W).float()

            seq_len = H * W
            max_tok = self.max_attn_tokens
            if 0 < max_tok < seq_len:
                _, pack_idx = torch.topk(bev_mask_f, k=max_tok, dim=1, sorted=False)
                x = torch.gather(x, 1, pack_idx.unsqueeze(-1).expand(-1, -1, C))
                bev_mask_f = torch.gather(bev_mask_f, 1, pack_idx)

            padding_mask = (1.0 - bev_mask_f) * (-1e4)
            x = m.transformer(x, src_key_padding_mask=padding_mask)
            valid_mask = bev_mask_f.unsqueeze(-1)
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        if self.intrinsic_input:
            K = cam_intrinsic  # (B, 3, 3)
            intr_vec = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], dim=-1)
            intr_normed = (intr_vec - m.intr_mean) / m.intr_std
            intr_feat = m.intrinsic_proj(intr_normed)
            x = torch.cat([x, intr_feat], dim=-1)

        if not self.rotation_only:
            translation = m.translation_pred(x)
        else:
            translation = torch.zeros(B, 3, device=x.device)
        rotation = m.rotation_pred(x)

        from losses.quat_tools import batch_quat2mat, batch_tvector2mat
        T_pred = batch_tvector2mat(translation)
        R_pred = batch_quat2mat(rotation)
        T_pred = torch.bmm(T_pred, R_pred)

        with torch.cuda.amp.autocast(enabled=False):
            T_gt_expected = torch.matmul(
                torch.linalg.inv(T_pred.float()), init_T_to_camera.float()
            )

        if self.rotation_only:
            T_gt_expected = T_gt_expected.clone()
            T_gt_expected[:, :3, 3] = init_T_to_camera[:, :3, 3]

        return T_gt_expected


class SequenceMedianAggregator:
    """
    Production-ready sequence-level median aggregator for BEVCalib predictions.

    Accumulates per-frame 4x4 transform predictions and produces a robust
    aggregate via element-wise rotation matrix median (SVD-projected to SO(3))
    and element-wise translation median.

    Usage (streaming mode -- accumulate then query):
        agg = SequenceMedianAggregator(min_frames=5, max_frames=100)
        for frame in sequence:
            pred_T = model(img, pc, init_T, post_T, K)
            agg.add(pred_T)              # (B, 4, 4) or (4, 4) numpy/torch
            if agg.ready:
                calib = agg.aggregate()  # (4, 4) numpy
        agg.reset()                      # start new sequence

    Usage (batch mode -- all frames at once):
        Ts = [model(img_i, ...).cpu().numpy() for img_i in seq]
        calib = SequenceMedianAggregator.aggregate_batch(Ts)
    """

    def __init__(self, min_frames=5, max_frames=200):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._buffer = deque(maxlen=max_frames)

    def reset(self):
        self._buffer.clear()

    def add(self, pred_T):
        """Add one or more predictions. Accepts (4,4), (B,4,4), numpy or torch."""
        if isinstance(pred_T, torch.Tensor):
            pred_T = pred_T.detach().cpu().numpy()
        if pred_T.ndim == 2:
            self._buffer.append(pred_T.copy())
        elif pred_T.ndim == 3:
            for i in range(pred_T.shape[0]):
                self._buffer.append(pred_T[i].copy())

    @property
    def ready(self):
        return len(self._buffer) >= self.min_frames

    @property
    def count(self):
        return len(self._buffer)

    def aggregate(self):
        """Return the SVD-projected rotation median + translation median as (4,4)."""
        if not self._buffer:
            raise RuntimeError("No predictions to aggregate. Call add() first.")
        return self.aggregate_batch(list(self._buffer))

    @staticmethod
    def _rotation_matrix_median_svd(Rs):
        """Element-wise median of rotation matrices + SVD projection to SO(3)."""
        R_med_raw = np.median(Rs, axis=0)
        U, _, Vt = np.linalg.svd(R_med_raw)
        R_median = U @ Vt
        if np.linalg.det(R_median) < 0:
            U[:, -1] *= -1
            R_median = U @ Vt
        return R_median

    @staticmethod
    def aggregate_batch(pred_Ts):
        """
        Compute median calibration from a list of (4,4) transform arrays.

        Args:
            pred_Ts: list of numpy (4,4) LiDAR->Camera transforms

        Returns:
            median_T: (4,4) numpy, the aggregated calibration
        """
        Ts = np.array(pred_Ts)
        Rs = Ts[:, :3, :3]
        ts = Ts[:, :3, 3]

        R_median = SequenceMedianAggregator._rotation_matrix_median_svd(Rs)
        t_median = np.median(ts, axis=0)

        median_T = np.eye(4, dtype=np.float64)
        median_T[:3, :3] = R_median
        median_T[:3, 3] = t_median
        return median_T

    def get_confidence(self):
        """
        Return a per-axis consistency score (lower = more confident).
        Uses the angular spread of predictions in axis-angle representation.
        """
        if len(self._buffer) < 2:
            return None
        from scipy.spatial.transform import Rotation as R
        median_T = self.aggregate()
        R_med = median_T[:3, :3]
        angles = []
        for T in self._buffer:
            R_delta = T[:3, :3] @ R_med.T
            rotvec = R.from_matrix(R_delta).as_rotvec()
            angles.append(np.degrees(rotvec))
        angles = np.array(angles)
        return {
            'roll_std': float(np.std(angles[:, 0])),
            'pitch_std': float(np.std(angles[:, 1])),
            'yaw_std': float(np.std(angles[:, 2])),
            'total_std': float(np.std(np.linalg.norm(angles, axis=1))),
            'n_frames': len(self._buffer),
        }


def _detect_use_mlp_head(state_dict):
    """Auto-detect whether checkpoint uses MLP or Linear head."""
    has_mlp = any(k.startswith('rotation_pred.0.') for k in state_dict)
    has_linear = 'rotation_pred.weight' in state_dict
    if has_mlp:
        return True
    if has_linear:
        return False
    return True


def prepare_for_drinfer_export(wrapper, img_shape=(360, 640)):
    import types
    from transformers.models.swin.modeling_swin import (
        SwinLayer, SwinPatchMerging, SwinPatchEmbeddings,
        SwinModel,
    )

    model = wrapper.model
    img_h, img_w = img_shape

    # --- 1. Patch Swin dynamic padding to static constants --------------------
    swin_backbone = None
    for module in model.modules():
        if isinstance(module, SwinModel):
            swin_backbone = module
            break

    if swin_backbone is None:
        print("[patch] WARNING: no SwinModel found, skipping Swin patches")
    else:
        config = swin_backbone.config
        patch_size = config.patch_size
        window_size = config.window_size

        fH, fW = img_h // patch_size, img_w // patch_size
        patched_layers = 0
        patched_merges = 0
        patched_embeds = 0

        for module in swin_backbone.modules():
            if isinstance(module, SwinPatchEmbeddings):
                def _noop_maybe_pad(self, pixel_values, height, width):
                    return pixel_values
                module.maybe_pad = types.MethodType(_noop_maybe_pad, module)
                patched_embeds += 1

        encoder = swin_backbone.encoder
        cur_h, cur_w = fH, fW

        for stage_idx, stage in enumerate(encoder.layers):
            if isinstance(stage, nn.Identity):
                print(f"  Stage {stage_idx}: skipped (replaced with Identity)")
                continue

            pad_right = (window_size - cur_w % window_size) % window_size
            pad_bottom = (window_size - cur_h % window_size) % window_size
            const_pad = (0, 0, 0, pad_right, 0, pad_bottom)
            needs_layer_pad = pad_right > 0 or pad_bottom > 0

            for layer in stage.blocks:
                if isinstance(layer, SwinLayer):
                    def _make_const_maybe_pad(pv, do_pad):
                        def _const_maybe_pad(self, hidden_states, height, width):
                            if do_pad:
                                hidden_states = nn.functional.pad(
                                    hidden_states, pv)
                            return hidden_states, pv
                        return _const_maybe_pad

                    layer.maybe_pad = types.MethodType(
                        _make_const_maybe_pad(const_pad, needs_layer_pad),
                        layer,
                    )
                    patched_layers += 1

            if stage.downsample is not None and isinstance(
                    stage.downsample, SwinPatchMerging):
                merge_pad_h = cur_h % 2
                merge_pad_w = cur_w % 2
                merge_const = (0, 0, 0, merge_pad_w, 0, merge_pad_h)
                needs_merge_pad = merge_pad_h > 0 or merge_pad_w > 0

                def _make_const_merge_pad(pv, do_pad):
                    def _const_merge_pad(self, input_feature, height, width):
                        if do_pad:
                            input_feature = nn.functional.pad(
                                input_feature, pv)
                        return input_feature
                    return _const_merge_pad

                stage.downsample.maybe_pad = types.MethodType(
                    _make_const_merge_pad(merge_const, needs_merge_pad),
                    stage.downsample,
                )
                patched_merges += 1
                cur_h = (cur_h + merge_pad_h) // 2
                cur_w = (cur_w + merge_pad_w) // 2
            else:
                cur_h = cur_h // 2
                cur_w = cur_w // 2

            print(f"  Stage {stage_idx}: {cur_h}x{cur_w}, "
                  f"layer_pad={const_pad}, needs_pad={needs_layer_pad}")

        print(f"[patch] Swin: {patched_layers} SwinLayers, "
              f"{patched_merges} PatchMerging, "
              f"{patched_embeds} PatchEmbeddings patched")

    # --- 2. Replace nn.MultiheadAttention with drinfer-native DRMHA ----------
    #
    # PyTorch 2.x TransformerEncoderLayer uses a fused C++ forward
    # (aten::_transformer_encoder_layer_fwd) in eval mode, bypassing
    # module-level self_attn calls.  We must also patch layer.forward to
    # force the Python path through _sa_block → self.self_attn → DRMHA.
    if hasattr(model, 'transformer') and not model.deformable:
        import types
        from frontend_python.pytorch_parser.parse_utils.multi_head_attention import (
            MultiheadAttention as DRMHA,
        )

        encoder = model.transformer
        patched_te = 0
        for i, layer in enumerate(encoder.layers):
            if not isinstance(layer, nn.TransformerEncoderLayer):
                continue

            orig = layer.self_attn
            dr_mha = DRMHA(
                embed_dim=orig.embed_dim,
                num_heads=orig.num_heads,
                dropout=0.0,
                bias=orig.in_proj_bias is not None,
                batch_first=True,
            ).to(next(orig.parameters()).device)

            dr_mha.in_proj_weight.data.copy_(orig.in_proj_weight.data)
            if orig.in_proj_bias is not None:
                dr_mha.in_proj_bias.data.copy_(orig.in_proj_bias.data)
            dr_mha.out_proj.weight.data.copy_(orig.out_proj.weight.data)
            if orig.out_proj.bias is not None:
                dr_mha.out_proj.bias.data.copy_(orig.out_proj.bias.data)

            layer.self_attn = dr_mha

            def _make_python_forward(lyr):
                """Bypass fused C++ _transformer_encoder_layer_fwd."""
                def _forward(src, src_mask=None, src_key_padding_mask=None,
                             is_causal=False):
                    x = src
                    if lyr.norm_first:
                        x = x + lyr._sa_block(lyr.norm1(x), src_mask,
                                              src_key_padding_mask, is_causal)
                        x = x + lyr._ff_block(lyr.norm2(x))
                    else:
                        x = lyr.norm1(x + lyr._sa_block(x, src_mask,
                                                        src_key_padding_mask, is_causal))
                        x = lyr.norm2(x + lyr._ff_block(x))
                    return x
                return _forward

            layer.forward = _make_python_forward(layer)
            patched_te += 1

        print(f"[patch] TransformerEncoder: {patched_te} layers patched "
              f"(nn.MHA → DRMHA with dr_sdpa, Python forward)")
    else:
        print("[patch] No standard transformer to patch")

    print("[patch] Model prepared for DrInfer export")
    return wrapper


def _adapt_proj_heads_to_checkpoint(model, state_dict, device):
    """Adapt ProjectionHead and SpconvToDenseBEV dimensions to match checkpoint.

    See evaluate_checkpoint._adapt_model_to_checkpoint for full explanation.
    """
    from proj_head import ProjectionHead

    for branch_name in ('img_branch', 'pc_branch'):
        proj_key = f'{branch_name}.proj_head.projection.weight'
        if proj_key not in state_dict:
            continue
        ckpt_shape = state_dict[proj_key].shape
        model_params = dict(model.named_parameters())
        if proj_key not in model_params or ckpt_shape == model_params[proj_key].shape:
            continue

        ckpt_embed = ckpt_shape[1]
        branch = getattr(model, branch_name)

        if branch_name == 'pc_branch' and hasattr(branch, 'sparse_encoder'):
            to_bev = branch.sparse_encoder.to_bev
            if to_bev.mode == 'concat':
                new_n_z = ckpt_embed // to_bev.in_channels
                if new_n_z * to_bev.in_channels == ckpt_embed:
                    to_bev.n_z = new_n_z
                    to_bev.out_channels = ckpt_embed
                    print(f"[load] Adapted {branch_name}.to_bev n_z → {new_n_z}")

        branch.proj_head = ProjectionHead(
            embedding_dim=ckpt_embed,
            projection_dim=ckpt_shape[0],
        ).to(device)
        print(f"[load] Adapted {branch_name}.proj_head embedding_dim → {ckpt_embed}")


def load_bevcalib_inference(
    ckpt_path: str,
    device: str = "cuda",
    img_shape=(360, 640),
    rotation_only=None,
    deformable=False,
    bev_encoder=True,
    use_mlp_head=None,
    voxel_mode="scatter",
    to_bev_mode="concat",
    scatter_reduce="sum",
    bev_pool_factor=0,
    max_attn_tokens=0,
):
    """
    Load a BEVCalib checkpoint and return an inference wrapper.

    Args:
        ckpt_path:     path to .pth checkpoint
        device:        'cuda' or 'cpu'
        img_shape:     (H, W) image dimensions
        rotation_only: None=auto-detect from checkpoint, True=rotation-only, False=joint
        deformable:    whether the model uses deformable attention
        bev_encoder:   whether the model uses BEV encoder
        use_mlp_head:  None=auto-detect from checkpoint, True=MLP head, False=Linear head
        voxel_mode:    'hard' or 'scatter' (default 'scatter' for drinfer trace)
        to_bev_mode:   'concat', 'learned', or 'sum' (must match training config)
        scatter_reduce: 'sum' or 'mean' (scatter voxelization reduce mode)
        bev_pool_factor: BEV avg-pool factor before transformer (must match training)
        max_attn_tokens: max tokens fed to transformer (0 = all H*W; >0 packs
                         valid tokens via topk+gather to cut O(S^2) attention cost)

    Returns:
        wrapper: BEVCalibInference on the specified device
        epoch:   training epoch of the checkpoint
    """
    from bev_calib import BEVCalib

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    if use_mlp_head is None:
        use_mlp_head = _detect_use_mlp_head(state)
        print(f"[load] Auto-detected use_mlp_head={use_mlp_head}")

    if rotation_only is None:
        if 'rotation_only' in ckpt:
            rotation_only = bool(ckpt['rotation_only'])
        elif 'optimize_translation' in ckpt:
            rotation_only = not ckpt['optimize_translation']
        else:
            ckpt_args_ro = ckpt.get('args', {})
            if 'rotation_only' in ckpt_args_ro:
                rotation_only = bool(ckpt_args_ro['rotation_only'])
            else:
                has_trans = any('translation_pred' in k for k in state.keys())
                rotation_only = not has_trans
        print(f"[load] Auto-detected rotation_only={rotation_only}")

    ckpt_args = ckpt.get('args', {})
    _intrinsic_input = ckpt_args.get('intrinsic_input', False)
    print(f"[load] voxel_mode={voxel_mode}, to_bev_mode={to_bev_mode}, scatter_reduce={scatter_reduce}"
          f", rotation_only={rotation_only}{', intrinsic_input=True' if _intrinsic_input else ''}")
    model = BEVCalib(
        deformable=deformable,
        bev_encoder=bev_encoder,
        img_shape=(img_shape[0], img_shape[1]),
        rotation_only=rotation_only,
        use_mlp_head=use_mlp_head,
        voxel_mode=voxel_mode,
        to_bev_mode=to_bev_mode,
        scatter_reduce=scatter_reduce,
        bev_pool_factor=bev_pool_factor,
        intrinsic_input=_intrinsic_input,
    )

    _adapt_proj_heads_to_checkpoint(model, state, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        loss_keys = [k for k in unexpected if 'loss_fn' in k]
        non_loss = [k for k in unexpected if 'loss_fn' not in k]
        if loss_keys:
            print(f"[load] Skipped {len(loss_keys)} loss-only keys (not needed for inference)")
        if non_loss:
            print(f"[load] WARNING: unexpected non-loss keys: {non_loss}")
    if missing:
        print(f"[load] WARNING: missing keys: {missing}")
    epoch = ckpt.get("epoch", -1)

    model.to(device).eval()
    wrapper = BEVCalibInference(model, max_attn_tokens=max_attn_tokens).to(device).eval()
    if max_attn_tokens > 0:
        bev_h = model.bev_shape[0]
        bev_w = model.bev_shape[1]
        print(f"[load] Token packing enabled: max_attn_tokens={max_attn_tokens} "
              f"(BEV {bev_h}x{bev_w}={bev_h * bev_w})")
    return wrapper, epoch


def load_bevcalib_with_aggregation(
    ckpt_path: str,
    min_frames=5,
    max_frames=200,
    **kwargs,
):
    """
    Load BEVCalib + create a SequenceMedianAggregator for production deployment.

    Returns:
        wrapper:    BEVCalibInference model
        aggregator: SequenceMedianAggregator (call .add() per frame, .aggregate() for result)
        epoch:      training epoch

    Example:
        model, agg, epoch = load_bevcalib_with_aggregation("ckpt_best_val.pth")
        for img, pc, init_T, post_T, K in sequence_frames:
            pred_T = model(img, pc, init_T, post_T, K)
            agg.add(pred_T)
        calibration = agg.aggregate()       # robust (4,4) result
        confidence = agg.get_confidence()   # per-axis std in degrees
        agg.reset()                         # ready for next sequence
    """
    wrapper, epoch = load_bevcalib_inference(ckpt_path, **kwargs)
    aggregator = SequenceMedianAggregator(
        min_frames=min_frames, max_frames=max_frames)
    return wrapper, aggregator, epoch
