"""V54 Rig Consistency Loss: TLC use_rig-inspired sequence-level extrinsic consistency."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RigConsistencyLoss(nn.Module):
    """Penalize rotation prediction variance within the same sequence (zero-perturb batches)."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        rotations: torch.Tensor,
        seq_ids: torch.Tensor,
        is_zero_pert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rotations: (B, 4) normalized quaternions (w, x, y, z)
            seq_ids: (B,) integer sequence / domain ids
            is_zero_pert: (B,) bool or float mask for zero-perturbation samples
        """
        if rotations is None or seq_ids is None:
            return rotations.new_tensor(0.0) if rotations is not None else torch.tensor(0.0)

        device = rotations.device
        mask = is_zero_pert.float().view(-1) > 0.5
        if mask.sum() < 2:
            return rotations.new_tensor(0.0)

        q = F.normalize(rotations.float(), dim=-1, eps=1e-6)
        seq = seq_ids.view(-1).long()
        losses = []
        unique = torch.unique(seq[mask])
        for sid in unique:
            m = mask & (seq == sid)
            if m.sum() < 2:
                continue
            qg = q[m]
            q_ref = qg[0:1]
            dots = (qg * q_ref).sum(dim=-1).abs().clamp(-1.0, 1.0)
            ang = 2.0 * torch.acos(dots)
            losses.append(ang.var())

        if not losses:
            return rotations.new_tensor(0.0)
        return self.weight * torch.stack(losses).mean()
