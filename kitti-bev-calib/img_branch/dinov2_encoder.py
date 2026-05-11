"""
DINOv2 backbone encoder for BEVCalib image branch.

DINOv2's self-supervised pre-training (DINO loss) learns features that are
naturally invariant to viewpoint, illumination, and domain — making them
more robust to the camera-installation domain gap that degrades BEVCalib
generalization.

Drop-in replacement for SwinT_tiny_Encoder: same forward signature, same output shape.

Usage in img_branch.py or cam2bev_query.py:
    from .dinov2_encoder import DINOv2Encoder
    self.CamEncode = DINOv2Encoder(featureShape=featureShape, ...)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal ViT that exactly matches DINOv2's state_dict key naming and
# supports ARBITRARY input sizes (not limited to 518x518).
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                          # (B, D, H//ps, W//ps)
        return x.flatten(2).transpose(1, 2)        # (B, num_patches, D)


class _Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _LayerScale(nn.Module):
    def __init__(self, dim, init_val=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_val * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    """Transformer block matching DINOv2's naming: norm1, attn, ls1, norm2, mlp, ls2."""
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _DINOv2ViT(nn.Module):
    """Minimal ViT matching DINOv2's state_dict key naming exactly.

    Supports arbitrary input sizes via positional embedding interpolation.
    Key mapping: patch_embed.proj, cls_token, pos_embed, blocks.{i}.*, norm.*
    """
    def __init__(self, patch_size=14, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + (518 // patch_size) ** 2, embed_dim)
        )
        self.blocks = nn.ModuleList([
            _Block(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and h == w:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        sqrt_N = int(math.sqrt(N))

        patch_pos = patch_pos.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(), size=(h0, w0), mode="bicubic", align_corners=False
        ).to(patch_pos.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h0 * w0, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Simple FPN
# ---------------------------------------------------------------------------

class _SimpleFPN(nn.Module):
    """Lightweight FPN to create multi-scale features from DINOv2's single-scale output."""

    def __init__(self, in_channels, out_channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.scale_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()

        for i in range(num_scales):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
            elif i == 1:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
            self.scale_convs.append(conv)

        for i in range(num_scales - 1):
            self.merge_convs.append(nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ))

    def forward(self, x):
        multi_scale = [conv(x) for conv in self.scale_convs]
        out = multi_scale[-1]
        for i in range(self.num_scales - 2, -1, -1):
            target_h, target_w = multi_scale[i].shape[2:]
            up = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
            out = self.merge_convs[i](torch.cat([multi_scale[i], up], dim=1))
        return out


# ---------------------------------------------------------------------------
# DINOv2Encoder: main encoder class
# ---------------------------------------------------------------------------

class DINOv2Encoder(nn.Module):
    """DINOv2 ViT backbone with FPN, as a drop-in for SwinT_tiny_Encoder.

    Uses a custom ViT implementation that:
    - Matches DINOv2's state_dict key naming exactly
    - Supports arbitrary input sizes via positional embedding interpolation
    - No dependency on timm or specific torchvision versions
    """

    VARIANTS = {
        "dinov2-small": {"hub_name": "dinov2_vits14", "embed_dim": 384, "num_heads": 6},
        "dinov2-base":  {"hub_name": "dinov2_vitb14", "embed_dim": 768, "num_heads": 12},
    }

    WEIGHT_URLS = {
        "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    }

    PROJECT_CKPT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "ckpt", "checkpoints",
    )

    def __init__(
        self,
        featureShape,
        out_channels=256,
        variant="dinov2-small",
        freeze_backbone=False,
        freeze_layers=None,
        use_fpn=True,
        weights_path=None,
    ):
        super().__init__()
        assert variant in self.VARIANTS, f"Unknown variant: {variant}. Choose from {list(self.VARIANTS)}"
        vinfo = self.VARIANTS[variant]

        _, self.fH, self.fW = featureShape
        self.out_channels = out_channels
        self.embed_dim = vinfo["embed_dim"]
        self.variant = variant
        self._weights_path = weights_path
        self.patch_size = 14

        self.backbone = self._load_backbone(vinfo)

        if freeze_backbone:
            if freeze_layers is not None:
                self._partial_freeze(freeze_layers)
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                print(f"[DINOv2Encoder] Backbone fully frozen ({variant})")

        if use_fpn:
            self.fpn = _SimpleFPN(self.embed_dim, out_channels)
        else:
            self.fpn = None
            self.channel_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        param_count = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[DINOv2Encoder] {variant} (embed={self.embed_dim}), "
              f"output: {out_channels}x{self.fH}x{self.fW}, "
              f"params: {param_count:.1f}M, fpn={use_fpn}")

    def _partial_freeze(self, freeze_layers):
        """Freeze a subset of backbone transformer blocks.

        Args:
            freeze_layers: str like "0:-2" meaning freeze blocks[0:-2], unfreeze last 2.
        """
        blocks = list(self.backbone.blocks) if hasattr(self.backbone, 'blocks') else []
        if not blocks:
            print(f"[DINOv2Encoder] No blocks found, falling back to full freeze")
            for p in self.backbone.parameters():
                p.requires_grad = False
            return

        for p in self.backbone.parameters():
            p.requires_grad = False

        parts = freeze_layers.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(blocks)
        if end < 0:
            end = len(blocks) + end

        unfrozen_count = 0
        for i in range(end, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True
            unfrozen_count += 1

        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        frozen_count = len(blocks) - unfrozen_count
        print(f"[DINOv2Encoder] Partial freeze: {frozen_count}/{len(blocks)} blocks frozen, "
              f"last {unfrozen_count} blocks trainable ({self.variant})")

    def _load_backbone(self, vinfo):
        """Load DINOv2 with our custom ViT. Raises RuntimeError if weights unavailable."""
        hub_name = vinfo["hub_name"]
        weight_filename = f"{hub_name}_pretrain.pth"
        hub_cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")

        search_paths = []
        if self._weights_path:
            search_paths.append(self._weights_path)
        search_paths.extend([
            os.path.join(self.PROJECT_CKPT_DIR, weight_filename),
            os.path.join(hub_cache_dir, weight_filename),
        ])

        local_path = None
        for p in search_paths:
            if os.path.isfile(p):
                local_path = p
                break

        if local_path is None:
            download_url = self.WEIGHT_URLS.get(hub_name, "unknown")
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"[DINOv2Encoder] FATAL: Cannot load pretrained weights for {hub_name}.\n"
                f"Training with random init is NOT allowed.\n\n"
                f"Download:\n  wget {download_url}\n\n"
                f"Copy to:\n  mkdir -p {self.PROJECT_CKPT_DIR}\n"
                f"  cp {weight_filename} {self.PROJECT_CKPT_DIR}/\n\n"
                f"Searched:\n" + "\n".join(f"  - {p}" for p in search_paths)
                + f"\n{'='*70}"
            )

        model = _DINOv2ViT(
            patch_size=14,
            embed_dim=vinfo["embed_dim"],
            depth=12,
            num_heads=vinfo["num_heads"],
        )

        state_dict = torch.load(local_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(unexpected)
        print(f"[DINOv2Encoder] Loaded {local_path}")
        print(f"[DINOv2Encoder]   {loaded}/{len(state_dict)} keys loaded, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            for k in missing[:5]:
                print(f"[DINOv2Encoder]   missing: {k}")

        return model

    def forward(self, x):
        """
        Args:
            x: (B, N, C, H, W), N is number of cameras
        Returns:
            out: (B, N, out_channels, fH, fW), matches SwinT_tiny_Encoder output
        """
        B, N, C, H, W = x.shape
        imgs = x.view(B * N, C, H, W)

        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode="reflect")

        H_pad, W_pad = imgs.shape[2], imgs.shape[3]

        out = self.backbone(imgs)       # (BN, 1 + num_patches, embed_dim)
        tokens = out[:, 1:]             # drop CLS token

        pH = H_pad // self.patch_size
        pW = W_pad // self.patch_size
        feat_map = tokens.permute(0, 2, 1).view(B * N, self.embed_dim, pH, pW)

        if feat_map.shape[2] != self.fH or feat_map.shape[3] != self.fW:
            feat_map = F.interpolate(
                feat_map, size=(self.fH, self.fW), mode="bilinear", align_corners=False
            )

        if self.fpn is not None:
            out = self.fpn(feat_map)
        else:
            out = self.channel_proj(feat_map)

        return out.view(B, N, self.out_channels, self.fH, self.fW)
