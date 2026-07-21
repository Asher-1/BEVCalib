"""
LoRA TTA (Test-Time Adaptation) for BEVCalib

升级版 TTA: 在模型的最后 2 层添加 LoRA (Low-Rank Adaptation),
通过自监督信号在部署时快速适应新域。

相比 Affine TTA 的优势:
- 可学习非线性校正模式
- 适配完成后 merge 到权重, 零额外推理延迟
- 仅 ~2K 参数, 50-100 帧即可收敛

Usage:
    from lora_tta import LoRATTA
    
    tta = LoRATTA(model, rank=4, target_layers=['corr_head', 'decoder.layers.3'])
    tta.adapt(frames[:100], lr=1e-3, steps=50)
    tta.merge()  # 合并到模型权重
    # 现在模型已适配, 零额外开销推理
"""
import os
import sys
import math
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitti-bev-calib'))


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    
    Wraps a linear layer: output = original(x) + alpha/rank * B(A(x))
    where A: (in, rank), B: (rank, out)
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out

    def merge(self):
        """Merge LoRA weights into original layer (permanent, irreversible)."""
        with torch.no_grad():
            self.original.weight += (self.lora_B @ self.lora_A * self.scaling)
        return self.original

    @property
    def n_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


class LoRATTA:
    """
    LoRA-based Test-Time Adaptation for BEVCalib.
    
    Adds LoRA layers to specified model components and trains them
    using self-supervised signals (temporal consistency + fixpoint).
    """

    def __init__(self, model: nn.Module, rank: int = 4, alpha: float = 1.0,
                 target_patterns: Optional[List[str]] = None):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.lora_layers: Dict[str, LoRALayer] = {}
        self._original_layers: Dict[str, nn.Linear] = {}

        if target_patterns is None:
            target_patterns = ['corr_head', 'head.fc', 'decoder']

        self._inject_lora(target_patterns)
        total_params = sum(l.n_params for l in self.lora_layers.values())
        print(f"[LoRA TTA] Injected {len(self.lora_layers)} LoRA layers "
              f"(rank={rank}, total params={total_params})")

    def _inject_lora(self, patterns: List[str]):
        """Find and wrap Linear layers matching patterns with LoRA."""
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(p in name for p in patterns):
                continue
            if module.in_features < 32 or module.out_features < 32:
                continue

            lora = LoRALayer(module, rank=self.rank, alpha=self.alpha)
            self.lora_layers[name] = lora
            self._original_layers[name] = module

            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(self.model.named_modules())[parent_name]
                setattr(parent, attr_name, lora)
            else:
                setattr(self.model, name, lora)

    def get_lora_params(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimizer."""
        params = []
        for lora in self.lora_layers.values():
            params.extend([lora.lora_A, lora.lora_B])
        return params

    @torch.no_grad()
    def _collect_predictions(self, frames: list, device: torch.device,
                             max_frames: int = 100) -> torch.Tensor:
        """Run model on frames and collect euler predictions."""
        self.model.eval()
        predictions = []

        for frame in frames[:max_frames]:
            euler = self._forward_frame(frame, device)
            if euler is not None:
                predictions.append(euler)

        if predictions:
            return torch.stack(predictions)
        return torch.zeros(1, 3, device=device)

    def _forward_frame(self, frame, device) -> Optional[torch.Tensor]:
        """Single frame forward, return euler (3,) in radians."""
        try:
            if isinstance(frame, dict):
                imgs = frame['image'].unsqueeze(0).to(device)
                pcs = frame['pointcloud'].unsqueeze(0).to(device)
                init_T = frame['init_T'].unsqueeze(0).to(device)
                intrinsic = frame['intrinsic'].unsqueeze(0).to(device)
                masks = frame.get('mask', None)
                if masks is not None:
                    masks = masks.unsqueeze(0).to(device)
            else:
                return None

            output = self.model(imgs, pcs, init_T, intrinsic, masks)
            if isinstance(output, dict):
                quat = output.get('quat', output.get('rotation'))
            else:
                quat = output[0] if isinstance(output, (list, tuple)) else output

            return _quat_to_euler(quat.squeeze(0))
        except Exception:
            return None

    def adapt(self, frames: list, device: torch.device = None,
              lr: float = 1e-3, steps: int = 50,
              lambda_temporal: float = 1.0,
              lambda_fixpoint: float = 0.5,
              lambda_smooth: float = 0.2,
              verbose: bool = True) -> dict:
        """
        Run LoRA adaptation on given frames.
        
        Self-supervised losses:
        1. Temporal consistency: predictions should have low variance
        2. Fixpoint: mean prediction should be zero (no correction needed)
        3. Smoothness: adjacent frames should have similar predictions
        """
        if device is None:
            device = next(self.model.parameters()).device

        optimizer = optim.Adam(self.get_lora_params(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        self.model.train()
        for name, param in self.model.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False

        history = {'loss': [], 'temporal': [], 'fixpoint': [], 'smooth': []}

        for step in range(steps):
            optimizer.zero_grad()

            predictions = []
            for frame in frames:
                euler = self._forward_frame(frame, device)
                if euler is not None:
                    predictions.append(euler)
                if len(predictions) >= 50:
                    break

            if len(predictions) < 5:
                break

            pred_tensor = torch.stack(predictions)

            mean_pred = pred_tensor.mean(dim=0)
            loss_temporal = pred_tensor.var(dim=0).sum() * lambda_temporal
            loss_fixpoint = mean_pred.pow(2).sum() * lambda_fixpoint

            if len(predictions) > 1:
                diffs = pred_tensor[1:] - pred_tensor[:-1]
                loss_smooth = diffs.pow(2).mean() * lambda_smooth
            else:
                loss_smooth = torch.tensor(0.0, device=device)

            total_loss = loss_temporal + loss_fixpoint + loss_smooth
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.get_lora_params(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history['loss'].append(total_loss.item())
            history['temporal'].append(loss_temporal.item())
            history['fixpoint'].append(loss_fixpoint.item())
            history['smooth'].append(loss_smooth.item())

            if verbose and (step % 10 == 0 or step == steps - 1):
                bias_deg = mean_pred.detach().cpu().numpy() * 180 / 3.14159
                print(f"  [LoRA TTA] Step {step:3d}: loss={total_loss.item():.5f} "
                      f"(temp={loss_temporal.item():.5f} fp={loss_fixpoint.item():.5f}) "
                      f"mean_pred=[{bias_deg[0]:.3f},{bias_deg[1]:.3f},{bias_deg[2]:.3f}]°")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

        return history

    def merge(self):
        """
        Merge all LoRA weights into original layers.
        After this, the model has zero additional inference cost.
        LoRA layers are replaced with their merged linear counterparts.
        """
        for name, lora in self.lora_layers.items():
            merged = lora.merge()
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(self.model.named_modules())[parent_name]
                setattr(parent, attr_name, merged)
            else:
                setattr(self.model, name, merged)

        n_merged = len(self.lora_layers)
        self.lora_layers.clear()
        print(f"[LoRA TTA] Merged {n_merged} LoRA layers into model weights")
        return self.model

    def save_lora_weights(self, path: str):
        """Save only LoRA weights (tiny file, ~8KB for rank=4)."""
        state = {}
        for name, lora in self.lora_layers.items():
            state[f"{name}.lora_A"] = lora.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = lora.lora_B.detach().cpu()
        state['_meta'] = {'rank': self.rank, 'alpha': self.alpha}
        torch.save(state, path)
        print(f"[LoRA TTA] Saved LoRA weights to {path} "
              f"({os.path.getsize(path) / 1024:.1f} KB)")

    def load_lora_weights(self, path: str):
        """Load previously saved LoRA weights."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        for name, lora in self.lora_layers.items():
            if f"{name}.lora_A" in state:
                lora.lora_A.data.copy_(state[f"{name}.lora_A"])
                lora.lora_B.data.copy_(state[f"{name}.lora_B"])
        print(f"[LoRA TTA] Loaded LoRA weights from {path}")


def _quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """(4,) wxyz → (3,) euler rad."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = torch.clamp(2*(w*y - z*x), -1, 1)
    pitch = torch.asin(sinp)
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return torch.stack([roll, pitch, yaw])
