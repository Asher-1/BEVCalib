"""
BEVCalib inference wrapper -- clean forward path without loss computation.
Used for model export (drinfer/ONNX) and standalone inference benchmarking.

IMPORTANT: set BEV_ZBOUND_STEP env var *before* importing this module so that
bev_settings.py picks up the correct Z resolution.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

_KITTI_DIR = os.path.join(os.path.dirname(__file__), '..', 'kitti-bev-calib')
if _KITTI_DIR not in sys.path:
    sys.path.insert(0, _KITTI_DIR)


class BEVCalibInference(nn.Module):
    """
    Inference-only wrapper for BEVCalib.
    Strips loss computation and returns the predicted LiDAR->Camera transform.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rotation_only = model.rotation_only

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
            bev_mask = cam_bev_mask.reshape(_B, H * W).unsqueeze(-1)
            x = x * bev_mask
            x = x.mean(dim=1)
        else:
            _B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(_B, H * W, C)
            bev_mask_f = cam_bev_mask.reshape(_B, H * W).float()
            x = x * bev_mask_f.unsqueeze(-1)
            padding_mask = 1.0 - bev_mask_f
            x = m.transformer(x, src_key_padding_mask=padding_mask)
            x = x.mean(dim=1)

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
    """
    Patch the model graph to remove all operations that DrInfer's
    model_parserv3 cannot trace:
      1. Swin's dynamic maybe_pad  -> hardcoded constant padding
      2. nn.TransformerEncoderLayer -> decomposed forward (bypass fused path)
      3. Register aten::zero / aten::alias symbolics for spconv
      4. Switch bev_pool to scatter_add (DrInfer-exportable)
    Must be called *before* model_parserv3.
    """
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

    # --- 2. Patch TransformerEncoderLayer: bypass fused path ------------------
    if hasattr(model, 'transformer') and not model.deformable:
        try:
            import frontend_python.pytorch_parser.parse_utils.multi_head_attention as DR_MHA

            encoder = model.transformer
            patched_te = 0
            for i, layer in enumerate(encoder.layers):
                if not isinstance(layer, nn.TransformerEncoderLayer):
                    continue

                d_model = layer.self_attn.embed_dim
                nhead = layer.self_attn.num_heads

                dr_mha = DR_MHA.MultiheadAttention(
                    d_model, nhead, dropout=0.0, bias=True, batch_first=True,
                )
                dr_mha.in_proj_weight = layer.self_attn.in_proj_weight
                dr_mha.in_proj_bias = layer.self_attn.in_proj_bias
                dr_mha.out_proj.weight = layer.self_attn.out_proj.weight
                dr_mha.out_proj.bias = layer.self_attn.out_proj.bias

                layer.self_attn = dr_mha

                def _make_patched_forward(lyr):
                    def _patched_forward(src, src_mask=None, src_key_padding_mask=None,
                                         is_causal=False):
                        import torch.nn.functional as _F
                        src_key_padding_mask = _F._canonical_mask(
                            mask=src_key_padding_mask,
                            mask_name="src_key_padding_mask",
                            other_type=_F._none_or_dtype(src_mask),
                            other_name="src_mask",
                            target_type=src.dtype,
                        )
                        src_mask = _F._canonical_mask(
                            mask=src_mask, mask_name="src_mask",
                            other_type=None, other_name="",
                            target_type=src.dtype, check_other=False,
                        )
                        x = src
                        if lyr.norm_first:
                            x = x + lyr._sa_block(lyr.norm1(x), src_mask,
                                                   src_key_padding_mask,
                                                   is_causal=is_causal)
                            x = x + lyr._ff_block(lyr.norm2(x))
                        else:
                            x = lyr.norm1(x + lyr._sa_block(x, src_mask,
                                                             src_key_padding_mask,
                                                             is_causal=is_causal))
                            x = lyr.norm2(x + lyr._ff_block(x))
                        return x
                    return _patched_forward

                layer.forward = _make_patched_forward(layer)
                patched_te += 1

            print(f"[patch] TransformerEncoder: {patched_te} layers patched "
                  f"(fused path bypassed, DrInfer MHA installed)")
        except ImportError as e:
            print(f"[patch] WARNING: could not patch TransformerEncoder: {e}")

    # --- 3. Register custom op symbolics for drcv voxel/spconv/bev_pool ------
    try:
        from frontend_python.pytorch_parser.registry import OperatorRegister
        from frontend_python.pytorch_parser.dr_symbolic.jit_utils import create_dr_op
        from frontend_python.pytorch_parser.dr_symbolic.helper import CUSTOM_DOMAIN

        @OperatorRegister.register_version("aten", 1)
        def zero(g, self):
            return create_dr_op(g, "dr::ZerosLike", self)

        @OperatorRegister.register_version("aten", 1)
        def alias(g, self):
            return self

        @OperatorRegister.register_version(CUSTOM_DOMAIN, 1)
        def _Voxelization(g, points, *args):
            return create_dr_op(g, "dr::HardVoxelize", points, outputs=3)

        from frontend_python.pytorch_parser.dr_symbolic import helper as dr_helper
        from frontend_python.pytorch_parser.dr_symbolic.helper import parse_dr_args

        @OperatorRegister.register_version(CUSTOM_DOMAIN, 1)
        @parse_dr_args('v', 'v', 'v', 'v', 'i')
        def SparseConvFunction(g, features, filters, indice_pairs,
                               indice_pair_num, num_activate_out):
            return create_dr_op(g, "dr::SparseConv", features, filters,
                                indice_pairs, indice_pair_num,
                                transpose_t=dr_helper._get_bool_tensor(False),
                                subm_t=dr_helper._get_bool_tensor(False),
                                num_act_out_i=num_activate_out)

        @OperatorRegister.register_version(CUSTOM_DOMAIN, 1)
        @parse_dr_args('v', 'v', 'v', 'v', 'i')
        def SubMConvFunction(g, features, filters, indice_pairs,
                             indice_pair_num, num_activate_out):
            return create_dr_op(g, "dr::SparseConv", features, filters,
                                indice_pairs, indice_pair_num,
                                transpose_t=dr_helper._get_bool_tensor(False),
                                subm_t=dr_helper._get_bool_tensor(True),
                                num_act_out_i=num_activate_out)

        print("[patch] Registered drcv op symbolics: "
              "_Voxelization, SparseConvFunction, SubMConvFunction, "
              "aten::zero, aten::alias")
    except Exception as e:
        print(f"[patch] WARNING: could not register op symbolics: {e}")

    print("[patch] Model prepared for DrInfer export")
    return wrapper


def load_bevcalib_inference(
    ckpt_path: str,
    device: str = "cuda",
    img_shape=(360, 640),
    rotation_only=True,
    deformable=False,
    bev_encoder=True,
    use_mlp_head=None,
):
    """
    Load a BEVCalib checkpoint and return an inference wrapper.

    Args:
        ckpt_path:     path to .pth checkpoint
        device:        'cuda' or 'cpu'
        img_shape:     (H, W) image dimensions
        rotation_only: whether the model was trained in rotation-only mode
        deformable:    whether the model uses deformable attention
        bev_encoder:   whether the model uses BEV encoder
        use_mlp_head:  None=auto-detect from checkpoint, True=MLP head, False=Linear head

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

    model = BEVCalib(
        deformable=deformable,
        bev_encoder=bev_encoder,
        img_shape=(img_shape[0], img_shape[1]),
        rotation_only=rotation_only,
        use_mlp_head=use_mlp_head,
    )

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
    wrapper = BEVCalibInference(model).to(device).eval()
    return wrapper, epoch
